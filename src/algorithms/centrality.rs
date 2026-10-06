//! Betweenness (Brandes) and closeness centrality.
//!
//! Each per-source pass is a literal port of the NetworkX helpers in
//! `networkx/algorithms/centrality/betweenness.py`, including how path counts
//! and ties are handled. With `ordered`, contributions are summed in
//! NetworkX's source order, so results match NetworkX bit for bit; without,
//! sources are split into blocks summed separately (faster, but the last
//! bits of a float can differ).

use std::cmp::Reverse;

use rayon::prelude::*;

use super::distance::bfs_stats;
use super::traversal::{DijkstraState, HeapKey, MinHeap, NegativeCycle};
use crate::graph::Csr;

const UNSET: u32 = u32::MAX;

/// Cap on memory for per-source contributions or per-block partial sums
/// held at once (bytes).
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
    /// The arc (CSR position) each entry of `preds` arrived by.
    pred_arcs: Vec<Vec<u32>>,
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
            pred_arcs: vec![Vec::new(); wn],
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
            self.pred_arcs[v].clear();
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
                    self.pred_arcs[w].clear();
                    self.pred_arcs[w].push(e as u32);
                } else if self.has_seen[w] && vw_dist == self.seen[w] {
                    self.sigma[w] += self.sigma[v];
                    self.preds[w].push(v as u32);
                    self.pred_arcs[w].push(e as u32);
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
    /// `_accumulate_edges`: adds each shortest-path DAG arc's share to
    /// `edge_bc[edge_id[arc]]` (`in_edge_id` numbers the arcs of `in_adj`).
    fn accumulate_edges(
        &mut self,
        in_adj: &Csr,
        weighted: bool,
        edge_id: &[u32],
        in_edge_id: &[u32],
        edge_bc: &mut [f64],
    ) {
        for &v in &self.order {
            self.delta[v as usize] = 0.0;
        }
        if weighted {
            for &w in self.order.iter().rev() {
                let w = w as usize;
                let coeff = (1.0 + self.delta[w]) / self.sigma[w];
                for (&v, &arc) in self.preds[w].iter().zip(&self.pred_arcs[w]) {
                    let c = self.sigma[v as usize] * coeff;
                    edge_bc[edge_id[arc as usize] as usize] += c;
                    self.delta[v as usize] += c;
                }
            }
        } else {
            // Push from each `w` in pop order, as NetworkX does, so that
            // `delta[v]` takes its shares in the same order.
            for &w in self.order.iter().rev() {
                let w = w as usize;
                let coeff = (1.0 + self.delta[w]) / self.sigma[w];
                let lw = self.level[w];
                for e in in_adj.range(w) {
                    let v = in_adj.targets[e] as usize;
                    if self.level[v] != UNSET && self.level[v] + 1 == lw {
                        let c = self.sigma[v] * coeff;
                        edge_bc[in_edge_id[e] as usize] += c;
                        self.delta[v] += c;
                    }
                }
            }
        }
    }
}

/// Sums each source's contribution (`len` values that `add(state, s, out)`
/// adds into a zeroed `out`) in source order, as NetworkX's loop over
/// sources does, so the float sums match it exactly. Sources run in
/// parallel in chunks; within a chunk each index's total takes the chunk's
/// contributions in source order. Indices a source does not reach get
/// `+ 0.0`, which leaves a non-negative total unchanged.
fn sum_in_source_order(
    len: usize,
    n: usize,
    weighted: bool,
    sources: &[u32],
    add: &(dyn Fn(&mut BcState, usize, &mut [f64]) + Sync),
) -> Vec<f64> {
    let mut total = vec![0.0; len];
    if len == 0 || sources.is_empty() {
        return total;
    }
    let by_memory = (BLOCK_MEMORY / (8 * len)).max(1);
    let chunk_len = (rayon::current_num_threads() * 16)
        .min(by_memory)
        .min(sources.len())
        .max(1);
    let mut parts: Vec<Vec<f64>> = vec![vec![0.0; len]; chunk_len];
    for chunk in sources.chunks(chunk_len) {
        parts[..chunk.len()]
            .par_iter_mut()
            .zip(chunk)
            .for_each_init(
                || BcState::new(n, weighted),
                |state, (out, &s)| add(state, s as usize, out),
            );
        // Adding also clears each contribution for the next chunk.
        let mut columns: Vec<std::slice::ChunksMut<'_, f64>> = parts[..chunk.len()]
            .iter_mut()
            .map(|part| part.chunks_mut(4096))
            .collect();
        let blocks: Vec<(&mut [f64], Vec<&mut [f64]>)> = total
            .chunks_mut(4096)
            .map(|block| {
                let row = columns.iter_mut().map(|c| c.next().unwrap()).collect();
                (block, row)
            })
            .collect();
        blocks.into_par_iter().for_each(|(block, row)| {
            for part in row {
                for (t, p) in block.iter_mut().zip(part.iter_mut()) {
                    *t += *p;
                    *p = 0.0;
                }
            }
        });
    }
    total
}

/// Sums each source's contribution in a fixed number of contiguous blocks of
/// sources (one partial sum per block, run in parallel), then adds the
/// partials. The blocks depend only on the input size, not on the thread
/// count, so results are the same on any machine, but floats can differ
/// from NetworkX's source-order sums in the last bits.
fn sum_in_blocks(
    len: usize,
    n: usize,
    weighted: bool,
    sources: &[u32],
    add: &(dyn Fn(&mut BcState, usize, &mut [f64]) + Sync),
) -> Vec<f64> {
    let mut total = vec![0.0; len];
    if len == 0 || sources.is_empty() {
        return total;
    }
    let max_by_memory = (BLOCK_MEMORY / (8 * len.max(n))).max(1);
    let nblocks = MAX_BLOCKS.min(max_by_memory).min(sources.len());
    let block_len = sources.len().div_ceil(nblocks);
    let partials: Vec<Vec<f64>> = sources
        .par_chunks(block_len)
        .map(|block| {
            let mut state = BcState::new(n, weighted);
            let mut part = vec![0.0; len];
            for &s in block {
                add(&mut state, s as usize, &mut part);
            }
            part
        })
        .collect();
    for part in &partials {
        for (t, p) in total.iter_mut().zip(part) {
            *t += p;
        }
    }
    total
}

/// Unscaled edge betweenness summed over `sources`, indexed by edge id
/// (`edge_id` and `in_edge_id` map each arc of `adj` and `in_adj` to its
/// edge; `num_edges` ids). Matches NetworkX bit for bit if `ordered`.
#[allow(clippy::too_many_arguments)]
pub fn edge_betweenness(
    adj: &Csr,
    in_adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    edge_id: &[u32],
    in_edge_id: &[u32],
    num_edges: usize,
    sources: &[u32],
    ordered: bool,
) -> Vec<f64> {
    if n == 0 {
        return vec![0.0; num_edges];
    }
    let sum = if ordered {
        sum_in_source_order
    } else {
        sum_in_blocks
    };
    sum(
        num_edges,
        n,
        weights.is_some(),
        sources,
        &|state: &mut BcState, s: usize, out: &mut [f64]| {
            match weights {
                Some(w) => state.dijkstra(adj, w, s),
                None => state.bfs(adj, s),
            }
            state.accumulate_edges(in_adj, weights.is_some(), edge_id, in_edge_id, out);
        },
    )
}

/// Unscaled betweenness summed over `sources`. Matches NetworkX bit for bit
/// if `ordered`.
pub fn betweenness(
    adj: &Csr,
    in_adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    endpoints: bool,
    sources: &[u32],
    ordered: bool,
) -> Vec<f64> {
    let sum = if ordered {
        sum_in_source_order
    } else {
        sum_in_blocks
    };
    sum(
        n,
        n,
        weights.is_some(),
        sources,
        &|state: &mut BcState, s: usize, out: &mut [f64]| {
            match weights {
                Some(w) => state.dijkstra(adj, w, s),
                None => state.bfs(adj, s),
            }
            state.accumulate(in_adj, weights.is_some(), s, endpoints, out);
        },
    )
}

/// `nx.closeness_centrality` for each node in `sources`, on `adj` (which the
/// caller has already reversed for directed graphs, as NetworkX does).
pub fn closeness(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    wf_improved: bool,
    sources: &[u32],
    compensated: bool,
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
                    // `sum(sp.values())`: same order, and compensated
                    // from Python 3.12.
                    let totsp = crate::algorithms::spectral::py_sum(
                        state.order.iter().map(|&v| state.dist[v as usize]),
                        compensated,
                    );
                    Ok(score(state.order.len(), totsp))
                },
            )
            .collect(),
    }
}

/// `nx.harmonic_centrality` without transposition: for each node, the sum of
/// `1 / d(v, u)` over `sources` in the given order, skipping `d == 0`, and
/// only for nodes with `in_nbunch`. Distances are computed in parallel in
/// blocks, but added sequentially in source order, so float sums match
/// NetworkX exactly. Returns the sums and whether each node received any
/// term (NetworkX keeps the int 0 otherwise).
pub fn harmonic(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    sources: &[u32],
    in_nbunch: &[bool],
) -> Result<(Vec<f64>, Vec<bool>), NegativeCycle> {
    let mut total = vec![0.0f64; n];
    let mut touched = vec![false; n];
    let block = (rayon::current_num_threads() * 4).max(1);
    for chunk in sources.chunks(block) {
        let results: Vec<Result<Vec<(u32, f64)>, NegativeCycle>> = chunk
            .par_iter()
            .map_init(
                || DijkstraState::new(n),
                |state, &s| match weights {
                    None => {
                        let (order, levels) =
                            super::traversal::bfs_lengths(adj, n, s as usize, f64::INFINITY);
                        Ok(order
                            .into_iter()
                            .zip(levels)
                            .map(|(v, d)| (v, d as f64))
                            .collect())
                    }
                    Some(w) => {
                        state.run(adj, Some(w), s as usize, None)?;
                        Ok(state
                            .order
                            .iter()
                            .map(|&v| (v, state.dist[v as usize]))
                            .collect())
                    }
                },
            )
            .collect();
        for result in results {
            for (u, d) in result? {
                let u = u as usize;
                if d != 0.0 && in_nbunch[u] {
                    total[u] += 1.0 / d;
                    touched[u] = true;
                }
            }
        }
    }
    Ok((total, touched))
}
