//! Support for `rustnx.rx`, the rustworkx-compatible API.
//!
//! rustworkx graphs are petgraph `StableGraph`s, and their algorithms see
//! neighbors in petgraph's order: for each node, its outgoing edges newest
//! first, then (undirected graphs) its incoming edges newest first, skipping
//! self-loops the second time. `build_rx` lays rows out in that order, and
//! the traversals below are ports of petgraph's, so order-dependent results
//! (strong components, topological order) match rustworkx exactly.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::OnceLock;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::algorithms::traversal::HeapKey;
use crate::graph::{CoreGraph, Csr, Weights};

/// Edge-cost attribute name used for rustworkx `weight_fn` / `edge_cost_fn`.
pub const RX_WEIGHT: &str = "__rx_cost__";

fn fill_rows(n: usize, counts: &[usize], parts: &[Vec<(u32, u32, u32)>]) -> (Csr, Vec<u32>) {
    let mut offsets = vec![0usize; n + 1];
    for v in 0..n {
        offsets[v + 1] = offsets[v] + counts[v];
    }
    let mut next = offsets.clone();
    let m = offsets[n];
    let mut targets = vec![0u32; m];
    let mut edge_of = vec![0u32; m];
    // Parts are written one after another, so row order is part 0, part 1...
    for part in parts {
        for &(row, target, e) in part {
            let slot = next[row as usize];
            next[row as usize] += 1;
            targets[slot] = target;
            edge_of[slot] = e;
        }
    }
    (Csr { offsets, targets }, edge_of)
}

/// Build from compact node positions. `src`/`dst` are edges in insertion
/// order; `costs`, if given, has one value per edge.
#[pyfunction]
#[pyo3(signature = (n, directed, src, dst, costs=None))]
pub fn build_rx(
    n: usize,
    directed: bool,
    src: Vec<u32>,
    dst: Vec<u32>,
    costs: Option<Vec<f64>>,
) -> PyResult<CoreGraph> {
    let m = src.len();
    if dst.len() != m || costs.as_ref().is_some_and(|c| c.len() != m) {
        return Err(PyValueError::new_err("edge arrays differ in length"));
    }
    if src.iter().chain(&dst).any(|&v| v as usize >= n) {
        return Err(PyValueError::new_err("edge endpoint out of range"));
    }
    // Newest first: walk edges in reverse insertion order.
    let out: Vec<(u32, u32, u32)> = (0..m).rev().map(|e| (src[e], dst[e], e as u32)).collect();
    let inc: Vec<(u32, u32, u32)> = (0..m)
        .rev()
        .filter(|&e| directed || src[e] != dst[e])
        .map(|e| (dst[e], src[e], e as u32))
        .collect();

    let mut out_count = vec![0usize; n];
    for &(r, _, _) in &out {
        out_count[r as usize] += 1;
    }
    let mut in_count = vec![0usize; n];
    for &(r, _, _) in &inc {
        in_count[r as usize] += 1;
    }

    let (succ, succ_edge, pred, pred_edge) = if directed {
        let (succ, se) = fill_rows(n, &out_count, std::slice::from_ref(&out));
        let (pred, pe) = fill_rows(n, &in_count, std::slice::from_ref(&inc));
        (succ, se, Some(pred), Some(pe))
    } else {
        let counts: Vec<usize> = (0..n).map(|v| out_count[v] + in_count[v]).collect();
        let (rows, edge_of) = fill_rows(n, &counts, &[out, inc]);
        (rows, edge_of, None, None)
    };

    let mut weights = HashMap::new();
    if let Some(costs) = costs {
        let gather = |edge_of: &[u32]| edge_of.iter().map(|&e| costs[e as usize]).collect();
        weights.insert(
            RX_WEIGHT.to_string(),
            Weights {
                succ: gather(&succ_edge),
                pred: pred_edge.as_deref().map(gather),
                all_int: false,
                has_hidden: false,
                any_int: false,
            },
        );
    }
    Ok(CoreGraph {
        n,
        directed,
        succ,
        pred,
        exact_pred: OnceLock::new(),
        pred_is_exact: true,
        weights,
        native: None,
        ones: OnceLock::new(),
    })
}

/// petgraph's `Dfs::next`: pop until an undiscovered node, push its
/// undiscovered neighbors, return it.
fn dfs_next(adj: &Csr, stack: &mut Vec<u32>, discovered: &mut [bool]) -> Option<u32> {
    while let Some(node) = stack.pop() {
        if !discovered[node as usize] {
            discovered[node as usize] = true;
            for &succ in adj.neighbors(node as usize) {
                if !discovered[succ as usize] {
                    stack.push(succ);
                }
            }
            return Some(node);
        }
    }
    None
}

/// petgraph's `kosaraju_scc`, as used by `rustworkx.strongly_connected_components`.
pub fn kosaraju_scc(succ: &Csr, pred: &Csr, n: usize) -> Vec<Vec<u32>> {
    // Phase 1: DfsPostOrder over the reversed graph, recording finish order.
    let mut discovered = vec![false; n];
    let mut finished = vec![false; n];
    let mut stack: Vec<u32> = Vec::new();
    let mut finish_order = Vec::with_capacity(n);
    for i in 0..n {
        if discovered[i] {
            continue;
        }
        stack.clear();
        stack.push(i as u32);
        while let Some(&nx) = stack.last() {
            if !discovered[nx as usize] {
                discovered[nx as usize] = true;
                for &s in pred.neighbors(nx as usize) {
                    if !discovered[s as usize] {
                        stack.push(s);
                    }
                }
            } else {
                stack.pop();
                if !finished[nx as usize] {
                    finished[nx as usize] = true;
                    finish_order.push(nx);
                }
            }
        }
    }
    // Phase 2: Dfs over the graph, leaders in decreasing finish time.
    let mut discovered = vec![false; n];
    let mut sccs = Vec::new();
    for &i in finish_order.iter().rev() {
        if discovered[i as usize] {
            continue;
        }
        stack.clear();
        stack.push(i);
        let mut scc = Vec::new();
        while let Some(nx) = dfs_next(succ, &mut stack, &mut discovered) {
            scc.push(nx);
        }
        sccs.push(scc);
    }
    sccs
}

/// petgraph's `toposort`, or `None` if the graph has a cycle.
pub fn toposort(succ: &Csr, pred: &Csr, n: usize) -> Option<Vec<u32>> {
    let mut discovered = vec![false; n];
    let mut finished = vec![false; n];
    let mut stack: Vec<u32> = Vec::new();
    let mut finish_stack = Vec::with_capacity(n);
    for i in 0..n {
        if discovered[i] {
            continue;
        }
        stack.push(i as u32);
        while let Some(&nx) = stack.last() {
            if !discovered[nx as usize] {
                discovered[nx as usize] = true;
                for &s in succ.neighbors(nx as usize) {
                    if s == nx {
                        return None; // self-loop
                    }
                    if !discovered[s as usize] {
                        stack.push(s);
                    }
                }
            } else {
                stack.pop();
                if !finished[nx as usize] {
                    finished[nx as usize] = true;
                    finish_stack.push(nx);
                }
            }
        }
    }
    finish_stack.reverse();
    // Cycle check: walking the reversed graph from each node in this order
    // must reach no node not already seen.
    let mut discovered = vec![false; n];
    for &i in &finish_stack {
        stack.clear();
        stack.push(i);
        let mut seen_new = false;
        while dfs_next(pred, &mut stack, &mut discovered).is_some() {
            if seen_new {
                return None;
            }
            seen_new = true;
        }
    }
    Some(finish_stack)
}

pub enum CostError {
    NaN,
    Negative,
}

/// rustworkx's Dijkstra lengths: costs are validated as each edge is
/// relaxed (so a bad cost only matters if Dijkstra reaches it), and the
/// search stops once `goal` is popped. Returns distances (NaN = unreached).
pub fn rx_dijkstra(
    adj: &Csr,
    n: usize,
    costs: &[f64],
    source: usize,
    goal: Option<usize>,
) -> Result<Vec<f64>, CostError> {
    let mut dist = vec![f64::NAN; n];
    let mut done = vec![false; n];
    let mut best = vec![f64::INFINITY; n];
    let mut heap: BinaryHeap<Reverse<(HeapKey, u32)>> = BinaryHeap::new();
    let mut counter = 0u64;
    best[source] = 0.0;
    heap.push(Reverse((HeapKey(0.0, counter), source as u32)));
    while let Some(Reverse((HeapKey(d, _), v))) = heap.pop() {
        let v = v as usize;
        if done[v] {
            continue;
        }
        done[v] = true;
        dist[v] = d;
        if goal == Some(v) {
            break;
        }
        for e in adj.range(v) {
            let c = costs[e];
            if c.is_nan() {
                return Err(CostError::NaN);
            }
            if c < 0.0 {
                return Err(CostError::Negative);
            }
            let w = adj.targets[e] as usize;
            let nd = d + c;
            if !done[w] && nd < best[w] {
                best[w] = nd;
                counter += 1;
                heap.push(Reverse((HeapKey(nd, counter), w as u32)));
            }
        }
    }
    Ok(dist)
}
