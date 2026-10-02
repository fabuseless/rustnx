//! Betweenness (Brandes) and closeness centrality.
//!
//! Each per-source pass is a literal port of the NetworkX helpers in
//! `networkx/algorithms/centrality/betweenness.py`, including how path counts
//! and ties are handled, so results agree with NetworkX up to floating-point
//! summation order across sources.

use std::cmp::Reverse;

use rayon::prelude::*;

use super::distance::bfs_stats;
use super::traversal::{DijkstraState, HeapKey, MinHeap, NegativeCycle};
use crate::graph::Csr;

const UNSET: u32 = u32::MAX;

/// Cap on memory for per-block accumulators (bytes).
const BLOCK_MEMORY: usize = 512 << 20;
const MAX_BLOCKS: usize = 64;

struct BcState {
    sigma: Vec<f64>,
    delta: Vec<f64>,
    /// Nodes in the order they were finalized (NetworkX's `S`).
    order: Vec<u32>,
    // Unweighted
    level: Vec<u32>,
    // Weighted
    dist: Vec<f64>,
    done: Vec<bool>,
    seen: Vec<f64>,
    has_seen: Vec<bool>,
    preds: Vec<Vec<u32>>,
    touched: Vec<u32>,
    heap: MinHeap<(u32, u32)>,
}

impl BcState {
    fn new(n: usize, weighted: bool) -> Self {
        let wn = if weighted { n } else { 0 };
        BcState {
            sigma: vec![0.0; n],
            delta: vec![0.0; n],
            order: Vec::new(),
            level: vec![UNSET; if weighted { 0 } else { n }],
            dist: vec![0.0; wn],
            done: vec![false; wn],
            seen: vec![0.0; wn],
            has_seen: vec![false; wn],
            preds: vec![Vec::new(); wn],
            touched: Vec::new(),
            heap: MinHeap::new(),
        }
    }

    /// `_single_source_shortest_path_basic`
    fn bfs(&mut self, adj: &Csr, s: usize) {
        for &v in &self.order {
            self.level[v as usize] = UNSET;
            self.sigma[v as usize] = 0.0;
        }
        self.order.clear();
        self.sigma[s] = 1.0;
        self.level[s] = 0;
        self.order.push(s as u32);
        let mut head = 0;
        while head < self.order.len() {
            let v = self.order[head] as usize;
            head += 1;
            let dv = self.level[v];
            let sigmav = self.sigma[v];
            for &w in adj.neighbors(v) {
                let w = w as usize;
                if self.level[w] == UNSET {
                    self.order.push(w as u32);
                    self.level[w] = dv + 1;
                }
                if self.level[w] == dv + 1 {
                    self.sigma[w] += sigmav;
                }
            }
        }
    }

    /// `_single_source_dijkstra_path_basic`
    fn dijkstra(&mut self, adj: &Csr, weights: &[f64], s: usize) {
        for &v in &self.touched {
            let v = v as usize;
            self.done[v] = false;
            self.has_seen[v] = false;
            self.sigma[v] = 0.0;
            self.preds[v].clear();
        }
        self.touched.clear();
        self.order.clear();
        self.heap.clear();

        let mut counter = 0u64;
        self.sigma[s] = 1.0;
        self.seen[s] = 0.0;
        self.has_seen[s] = true;
        self.touched.push(s as u32);
        self.heap
            .push(Reverse((HeapKey(0.0, counter), (s as u32, s as u32))));
        while let Some(Reverse((HeapKey(dist, _), (pred, v)))) = self.heap.pop() {
            let v = v as usize;
            if self.done[v] {
                continue;
            }
            self.sigma[v] += self.sigma[pred as usize];
            self.order.push(v as u32);
            self.dist[v] = dist;
            self.done[v] = true;
            for e in adj.range(v) {
                let w = adj.targets[e] as usize;
                let vw_dist = dist + weights[e];
                if !self.done[w] && (!self.has_seen[w] || vw_dist < self.seen[w]) {
                    if !self.has_seen[w] {
                        self.has_seen[w] = true;
                        self.touched.push(w as u32);
                    }
                    self.seen[w] = vw_dist;
                    counter += 1;
                    self.heap
                        .push(Reverse((HeapKey(vw_dist, counter), (v as u32, w as u32))));
                    self.sigma[w] = 0.0;
                    self.preds[w].clear();
                    self.preds[w].push(v as u32);
                } else if self.has_seen[w] && vw_dist == self.seen[w] {
                    self.sigma[w] += self.sigma[v];
                    self.preds[w].push(v as u32);
                }
            }
        }
    }

    /// `_accumulate_basic` / `_accumulate_endpoints`
    fn accumulate(
        &mut self,
        in_adj: &Csr,
        weighted: bool,
        s: usize,
        endpoints: bool,
        bc: &mut [f64],
    ) {
        if endpoints {
            bc[s] += (self.order.len() - 1) as f64;
        }
        for &v in &self.order {
            self.delta[v as usize] = 0.0;
        }
        let extra = if endpoints { 1.0 } else { 0.0 };
        for &w in self.order.iter().rev() {
            let w = w as usize;
            let coeff = (1.0 + self.delta[w]) / self.sigma[w];
            if weighted {
                for &v in &self.preds[w] {
                    let v = v as usize;
                    self.delta[v] += self.sigma[v] * coeff;
                }
            } else {
                // Predecessors on shortest paths are exactly the in-neighbors
                // one level closer to the source.
                let lw = self.level[w];
                for &v in in_adj.neighbors(w) {
                    let v = v as usize;
                    if self.level[v] != UNSET && self.level[v] + 1 == lw {
                        self.delta[v] += self.sigma[v] * coeff;
                    }
                }
            }
            if w != s {
                bc[w] += self.delta[w] + extra;
            }
        }
    }
}

/// Unscaled betweenness summed over `sources`.
///
/// Sources are split into a fixed number of contiguous blocks that depends
/// only on the input size (not on the thread count), so results are
/// reproducible on any machine.
pub fn betweenness(
    adj: &Csr,
    in_adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    endpoints: bool,
    sources: &[u32],
) -> Vec<f64> {
    if n == 0 || sources.is_empty() {
        return vec![0.0; n];
    }
    let max_by_memory = (BLOCK_MEMORY / (8 * n)).max(1);
    let nblocks = MAX_BLOCKS.min(max_by_memory).min(sources.len());
    let block_len = sources.len().div_ceil(nblocks);

    let partials: Vec<Vec<f64>> = sources
        .par_chunks(block_len)
        .map(|block| {
            let mut state = BcState::new(n, weights.is_some());
            let mut bc = vec![0.0; n];
            for &s in block {
                let s = s as usize;
                match weights {
                    Some(w) => state.dijkstra(adj, w, s),
                    None => state.bfs(adj, s),
                }
                state.accumulate(in_adj, weights.is_some(), s, endpoints, &mut bc);
            }
            bc
        })
        .collect();

    let mut total = vec![0.0; n];
    for part in &partials {
        for (t, p) in total.iter_mut().zip(part) {
            *t += p;
        }
    }
    total
}

/// `nx.closeness_centrality` for each node in `sources`, on `adj` (which the
/// caller has already reversed for directed graphs, as NetworkX does).
pub fn closeness(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    wf_improved: bool,
    sources: &[u32],
) -> Result<Vec<f64>, NegativeCycle> {
    let score = |reached: usize, totsp: f64| -> f64 {
        let mut c = 0.0;
        if totsp > 0.0 && n > 1 {
            c = (reached as f64 - 1.0) / totsp;
            if wf_improved {
                c *= (reached as f64 - 1.0) / (n as f64 - 1.0);
            }
        }
        c
    };
    match weights {
        None => Ok(bfs_stats(adj, n, sources)
            .into_iter()
            .map(|st| score(st.reached, st.total as f64))
            .collect()),
        Some(w) => sources
            .par_iter()
            .map_init(
                || DijkstraState::new(n),
                |state, &s| {
                    state.run(adj, Some(w), s as usize, None)?;
                    // Same left-to-right order as `sum(sp.values())`.
                    let totsp = state
                        .order
                        .iter()
                        .fold(0.0, |acc, &v| acc + state.dist[v as usize]);
                    Ok(score(state.order.len(), totsp))
                },
            )
            .collect(),
    }
}
