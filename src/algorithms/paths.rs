//! Shortest paths that return the paths themselves, ported from NetworkX so
//! that ties are broken the same way (and so the same path comes back).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::traversal::{HeapKey, NegativeCycle};
use crate::graph::Csr;

pub const NO_PARENT: u32 = u32::MAX;

/// BFS from `source` as `_single_shortest_path`: nodes in discovery order,
/// each with the node that discovered it (the source's parent is
/// `NO_PARENT`).
pub fn bfs_tree(adj: &Csr, n: usize, source: usize, cutoff: f64) -> (Vec<u32>, Vec<u32>) {
    let mut seen = vec![false; n];
    let mut order = vec![source as u32];
    let mut parent = vec![NO_PARENT];
    seen[source] = true;
    let mut start = 0;
    let mut level = 0u32;
    while start < order.len() && cutoff > level as f64 {
        let end = order.len();
        for i in start..end {
            let v = order[i];
            for &w in adj.neighbors(v as usize) {
                if !seen[w as usize] {
                    seen[w as usize] = true;
                    order.push(w);
                    parent.push(v);
                }
            }
        }
        start = end;
        level += 1;
    }
    (order, parent)
}

pub struct DijkstraTree {
    /// Finalized nodes in pop order, with their distances.
    pub order: Vec<u32>,
    pub dist: Vec<f64>,
    /// For every node index: the last node that improved its tentative
    /// distance (NetworkX's `pred_dict[u][0]`), or `NO_PARENT`.
    pub parent: Vec<u32>,
    /// Nodes in order of first push (older NetworkX orders its paths dict
    /// this way); the source first.
    pub seen_order: Vec<u32>,
}

/// `_dijkstra_multisource` with one source, recording parents. Stops once
/// `target` is popped. NaN weights mark hidden edges.
pub fn dijkstra_tree(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    source: usize,
    cutoff: Option<f64>,
    target: Option<usize>,
) -> Result<DijkstraTree, NegativeCycle> {
    let mut done = vec![false; n];
    let mut final_dist = vec![0.0; n];
    let mut seen = vec![f64::INFINITY; n];
    let mut has_seen = vec![false; n];
    let mut parent = vec![NO_PARENT; n];
    let mut heap: BinaryHeap<Reverse<(HeapKey, u32)>> = BinaryHeap::new();
    let mut counter = 0u64;
    let mut order = Vec::new();
    let mut dist = Vec::new();
    let mut seen_order = vec![source as u32];
    seen[source] = 0.0;
    has_seen[source] = true;
    heap.push(Reverse((HeapKey(0.0, counter), source as u32)));
    while let Some(Reverse((HeapKey(d, _), v))) = heap.pop() {
        let v = v as usize;
        if done[v] {
            continue;
        }
        done[v] = true;
        final_dist[v] = d;
        order.push(v as u32);
        dist.push(d);
        if target == Some(v) {
            break;
        }
        for e in adj.range(v) {
            let u = adj.targets[e] as usize;
            let cost = weights.map_or(1.0, |w| w[e]);
            if cost.is_nan() {
                continue;
            }
            let vu = d + cost;
            if cutoff.is_some_and(|c| vu > c) {
                continue;
            }
            if done[u] {
                if vu < final_dist[u] {
                    return Err(NegativeCycle);
                }
            } else if !has_seen[u] || vu < seen[u] {
                if !has_seen[u] {
                    has_seen[u] = true;
                    seen_order.push(u as u32);
                }
                seen[u] = vu;
                counter += 1;
                heap.push(Reverse((HeapKey(vu, counter), u as u32)));
                parent[u] = v as u32;
            }
        }
    }
    Ok(DijkstraTree {
        order,
        dist,
        parent,
        seen_order,
    })
}

/// `_bidirectional_pred_succ` + path assembly from
/// `bidirectional_shortest_path`. `succ`/`pred` are the forward and reverse
/// adjacency (the same rows for undirected graphs).
pub fn bidirectional_bfs(succ: &Csr, pred: &Csr, n: usize, s: usize, t: usize) -> Option<Vec<u32>> {
    const UNSET: i64 = -2;
    const NONE: i64 = -1;
    if s == t {
        return Some(vec![s as u32]);
    }
    let mut pred_of = vec![UNSET; n]; // forward tree: node -> predecessor
    let mut succ_of = vec![UNSET; n]; // reverse tree: node -> successor
    pred_of[s] = NONE;
    succ_of[t] = NONE;
    let mut forward = vec![s as u32];
    let mut reverse = vec![t as u32];
    let mut meet = None;
    'search: while !forward.is_empty() && !reverse.is_empty() {
        if forward.len() <= reverse.len() {
            let this_level = std::mem::take(&mut forward);
            for &v in &this_level {
                for &w in succ.neighbors(v as usize) {
                    let wi = w as usize;
                    if pred_of[wi] == UNSET {
                        forward.push(w);
                        pred_of[wi] = v as i64;
                    }
                    if succ_of[wi] != UNSET {
                        meet = Some(wi);
                        break 'search;
                    }
                }
            }
        } else {
            let this_level = std::mem::take(&mut reverse);
            for &v in &this_level {
                for &w in pred.neighbors(v as usize) {
                    let wi = w as usize;
                    if succ_of[wi] == UNSET {
                        succ_of[wi] = v as i64;
                        reverse.push(w);
                    }
                    if pred_of[wi] != UNSET {
                        meet = Some(wi);
                        break 'search;
                    }
                }
            }
        }
    }
    let w = meet?;
    let mut path = Vec::new();
    let mut cur = w as i64;
    while cur != NONE {
        path.push(cur as u32);
        cur = pred_of[cur as usize];
    }
    path.reverse();
    let mut cur = succ_of[*path.last().expect("non-empty") as usize];
    while cur != NONE {
        path.push(cur as u32);
        cur = succ_of[cur as usize];
    }
    Some(path)
}
