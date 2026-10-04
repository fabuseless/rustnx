//! More shortest-path centralities: subset betweenness, load, percolation,
//! reaching, group degree and closeness, VoteRank and dispersion.
//!
//! Each per-source pass is a literal port of the NetworkX helper it replaces.
//! Where NetworkX adds per-source contributions into one running total,
//! sources run in parallel but their contributions are added in source order
//! (see `in_source_order`), so float sums match NetworkX bit for bit.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use rayon::prelude::*;

use super::traversal::{HeapKey, MinHeap, NegativeCycle};
use crate::graph::Csr;

const UNSET: u32 = u32::MAX;

/// Runs `work` for every source, in parallel blocks, and passes each result
/// to `apply` in source order. `item_bytes` estimates the size of one result,
/// to bound memory.
pub(crate) fn in_source_order<S, T, I, W, A>(
    sources: &[u32],
    item_bytes: usize,
    init: I,
    work: W,
    mut apply: A,
) -> Result<(), NegativeCycle>
where
    I: Fn() -> S + Sync + Send,
    W: Fn(&mut S, usize) -> Result<T, NegativeCycle> + Sync + Send,
    T: Send,
    A: FnMut(T),
{
    let threads = rayon::current_num_threads().max(1);
    let by_memory = ((256usize << 20) / item_bytes.max(1)).max(threads);
    let block = (threads * 16).min(by_memory).max(1);
    for chunk in sources.chunks(block) {
        let results: Vec<Result<T, NegativeCycle>> = chunk
            .par_iter()
            .map_init(&init, |state, &s| work(state, s as usize))
            .collect();
        for result in results {
            apply(result?);
        }
    }
    Ok(())
}

/// CPython's `sum()` of floats as a running state (see `spectral::py_sum`),
/// so that sums along tree paths can be extended one edge at a time.
#[derive(Clone, Copy)]
struct PySum {
    s: f64,
    c: f64,
}

impl PySum {
    const ZERO: PySum = PySum { s: 0.0, c: 0.0 };

    #[inline]
    fn add(self, x: f64, compensated: bool) -> PySum {
        if !compensated {
            return PySum {
                s: self.s + x,
                c: 0.0,
            };
        }
        let t = self.s + x;
        let c = if self.s.abs() >= x.abs() {
            self.c + ((self.s - t) + x)
        } else {
            self.c + ((x - t) + self.s)
        };
        PySum { s: t, c }
    }

    #[inline]
    fn value(self) -> f64 {
        if self.c != 0.0 && self.c.is_finite() {
            self.s + self.c
        } else {
            self.s
        }
    }
}

/// Shortest-path DAG from one source, as NetworkX's
/// `_single_source_shortest_path_basic` / `_single_source_dijkstra_path_basic`
/// build it: `S` (finalization order), `sigma` and the predecessor lists `P`
/// in NetworkX's order, with the arc each predecessor arrived by.
pub struct Spt {
    pub(crate) order: Vec<u32>,
    pub(crate) sigma: Vec<f64>,
    pub(crate) delta: Vec<f64>,
    level: Vec<u32>,
    done: Vec<bool>,
    seen: Vec<f64>,
    has_seen: Vec<bool>,
    pub(crate) preds: Vec<Vec<u32>>,
    pub(crate) pred_arcs: Vec<Vec<u32>>,
    touched: Vec<u32>,
    heap: MinHeap<(u32, u32)>,
    /// Position of each reached node in `order` (`UNSET` otherwise); only
    /// filled by `run_with_positions`.
    pos: Vec<u32>,
}

impl Spt {
    pub fn new(n: usize) -> Self {
        Spt {
            order: Vec::new(),
            sigma: vec![0.0; n],
            delta: vec![0.0; n],
            level: vec![UNSET; n],
            done: vec![false; n],
            seen: vec![0.0; n],
            has_seen: vec![false; n],
            preds: vec![Vec::new(); n],
            pred_arcs: vec![Vec::new(); n],
            touched: Vec::new(),
            heap: MinHeap::new(),
            pos: vec![UNSET; n],
        }
    }

    fn reset(&mut self) {
        for &v in &self.touched {
            let v = v as usize;
            self.pos[v] = UNSET;
            self.level[v] = UNSET;
            self.done[v] = false;
            self.has_seen[v] = false;
            self.sigma[v] = 0.0;
            self.preds[v].clear();
            self.pred_arcs[v].clear();
        }
        self.touched.clear();
        self.order.clear();
        self.heap.clear();
    }

    pub(crate) fn run(&mut self, adj: &Csr, weights: Option<&[f64]>, s: usize) {
        self.reset();
        match weights {
            None => self.bfs(adj, s),
            Some(w) => self.dijkstra(adj, w, s),
        }
        for &v in &self.order {
            self.delta[v as usize] = 0.0;
        }
    }

    fn run_with_positions(&mut self, adj: &Csr, weights: Option<&[f64]>, s: usize) {
        self.run(adj, weights, s);
        for (i, &v) in self.order.iter().enumerate() {
            self.pos[v as usize] = i as u32;
        }
    }

    /// NetworkX's `D[v]` for a reached node.
    fn dist(&self, v: usize, weighted: bool) -> f64 {
        if weighted {
            self.seen[v]
        } else {
            self.level[v] as f64
        }
    }

    fn bfs(&mut self, adj: &Csr, s: usize) {
        self.sigma[s] = 1.0;
        self.level[s] = 0;
        self.order.push(s as u32);
        self.touched.push(s as u32);
        let mut head = 0;
        while head < self.order.len() {
            let v = self.order[head] as usize;
            head += 1;
            let dv = self.level[v];
            let sigmav = self.sigma[v];
            for e in adj.range(v) {
                let w = adj.targets[e] as usize;
                if self.level[w] == UNSET {
                    self.order.push(w as u32);
                    self.touched.push(w as u32);
                    self.level[w] = dv + 1;
                }
                if self.level[w] == dv + 1 {
                    self.sigma[w] += sigmav;
                    self.preds[w].push(v as u32);
                    self.pred_arcs[w].push(e as u32);
                }
            }
        }
    }

    fn dijkstra(&mut self, adj: &Csr, weights: &[f64], s: usize) {
        let mut counter = 0u64;
        self.sigma[s] = 1.0;
        self.seen[s] = 0.0;
        self.has_seen[s] = true;
        self.touched.push(s as u32);
        self.heap
            .push(Reverse((HeapKey(0.0, counter), (s as u32, s as u32))));
        while let Some(Reverse((HeapKey(d, _), (pred, v)))) = self.heap.pop() {
            let v = v as usize;
            if self.done[v] {
                continue;
            }
            self.sigma[v] += self.sigma[pred as usize];
            self.order.push(v as u32);
            self.done[v] = true;
            for e in adj.range(v) {
                let w = adj.targets[e] as usize;
                let vw_dist = d + weights[e];
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
                    // NetworkX's `elif` also runs for finalized nodes.
                    self.sigma[w] += self.sigma[v];
                    self.preds[w].push(v as u32);
                    self.pred_arcs[w].push(e as u32);
                }
            }
        }
    }
}

/// `betweenness_centrality_subset` before rescaling.
pub fn betweenness_subset(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    sources: &[u32],
    is_target: &[bool],
) -> Vec<f64> {
    let mut b = vec![0.0; n];
    let _ = in_source_order(
        sources,
        16 * n,
        || Spt::new(n),
        |st, s| {
            st.run(adj, weights, s);
            let mut log = Vec::with_capacity(st.order.len());
            // `_accumulate_subset` (its target set excludes `s`)
            for i in (0..st.order.len()).rev() {
                let w = st.order[i] as usize;
                let coeff = if is_target[w] && w != s {
                    (st.delta[w] + 1.0) / st.sigma[w]
                } else {
                    st.delta[w] / st.sigma[w]
                };
                for &v in &st.preds[w] {
                    let v = v as usize;
                    st.delta[v] += st.sigma[v] * coeff;
                }
                if w != s {
                    log.push((w as u32, st.delta[w]));
                }
            }
            Ok(log)
        },
        |log| {
            for (w, d) in log {
                b[w as usize] += d;
            }
        },
    );
    b
}

/// `edge_betweenness_centrality_subset` before rescaling, by edge id.
pub fn edge_betweenness_subset(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    sources: &[u32],
    is_target: &[bool],
    edge_id: &[u32],
    num_edges: usize,
) -> Vec<f64> {
    let mut b = vec![0.0; num_edges];
    let _ = in_source_order(
        sources,
        16 * n,
        || Spt::new(n),
        |st, s| {
            st.run(adj, weights, s);
            let mut log = Vec::with_capacity(st.order.len());
            // `_accumulate_edges_subset` (its target set includes `s`)
            for i in (0..st.order.len()).rev() {
                let w = st.order[i] as usize;
                let npreds = st.preds[w].len() as f64;
                for (&v, &arc) in st.preds[w].iter().zip(&st.pred_arcs[w]) {
                    let v = v as usize;
                    let c = if is_target[w] {
                        (st.sigma[v] / st.sigma[w]) * (1.0 + st.delta[w])
                    } else {
                        st.delta[w] / npreds
                    };
                    log.push((edge_id[arc as usize], c));
                    st.delta[v] += c;
                }
            }
            Ok(log)
        },
        |log| {
            for (e, c) in log {
                b[e as usize] += c;
            }
        },
    );
    b
}

/// `percolation_centrality` before its final `1 / (n - 2)` scaling.
pub fn percolation(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    states: &[f64],
    total: f64,
) -> Vec<f64> {
    let mut p = vec![0.0; n];
    let sources: Vec<u32> = (0..n as u32).collect();
    let _ = in_source_order(
        &sources,
        16 * n,
        || Spt::new(n),
        |st, s| {
            st.run(adj, weights, s);
            let mut log = Vec::with_capacity(st.order.len());
            // `_accumulate_percolation`
            for i in (0..st.order.len()).rev() {
                let w = st.order[i] as usize;
                let coeff = (1.0 + st.delta[w]) / st.sigma[w];
                for &v in &st.preds[w] {
                    let v = v as usize;
                    st.delta[v] += st.sigma[v] * coeff;
                }
                if w != s {
                    let pw = states[s] / (total - states[w]);
                    log.push((w as u32, st.delta[w] * pw));
                }
            }
            Ok(log)
        },
        |log| {
            for (w, x) in log {
                p[w as usize] += x;
            }
        },
    );
    p
}

/// Predecessor lists as `nx.predecessor` (BFS, `bfs = true`) or
/// `nx.dijkstra_predecessor_and_distance` build them, with distances and the
/// arcs predecessors arrived by.
struct PredState {
    dist: Vec<f64>,
    reached: Vec<u32>,
    is_reached: Vec<bool>,
    preds: Vec<Vec<u32>>,
    arcs: Vec<Vec<u32>>,
    seen: Vec<f64>,
    has_seen: Vec<bool>,
    touched: Vec<u32>,
    between: Vec<f64>,
}

impl PredState {
    fn new(n: usize) -> Self {
        PredState {
            dist: vec![0.0; n],
            reached: Vec::new(),
            is_reached: vec![false; n],
            preds: vec![Vec::new(); n],
            arcs: vec![Vec::new(); n],
            seen: vec![0.0; n],
            has_seen: vec![false; n],
            touched: Vec::new(),
            between: vec![0.0; n],
        }
    }

    fn reset(&mut self) {
        for &v in &self.touched {
            let v = v as usize;
            self.is_reached[v] = false;
            self.has_seen[v] = false;
            self.preds[v].clear();
            self.arcs[v].clear();
        }
        self.touched.clear();
        self.reached.clear();
    }

    /// `nx.predecessor(G, s, cutoff=cutoff)`; `cutoff` is `None` when
    /// NetworkX's `if cutoff and cutoff <= level` can never fire.
    fn bfs(&mut self, adj: &Csr, s: usize, cutoff: Option<f64>) {
        self.reset();
        self.is_reached[s] = true;
        self.dist[s] = 0.0;
        self.reached.push(s as u32);
        self.touched.push(s as u32);
        let mut begin = 0;
        let mut level = 0u32;
        while begin < self.reached.len() {
            level += 1;
            let end = self.reached.len();
            for i in begin..end {
                let v = self.reached[i] as usize;
                for e in adj.range(v) {
                    let w = adj.targets[e] as usize;
                    if !self.is_reached[w] {
                        self.is_reached[w] = true;
                        self.dist[w] = level as f64;
                        self.reached.push(w as u32);
                        self.touched.push(w as u32);
                        self.preds[w].push(v as u32);
                        self.arcs[w].push(e as u32);
                    } else if self.dist[w] == level as f64 {
                        self.preds[w].push(v as u32);
                        self.arcs[w].push(e as u32);
                    }
                }
            }
            begin = end;
            if cutoff.is_some_and(|c| c <= level as f64) {
                break;
            }
        }
    }

    /// `nx.dijkstra_predecessor_and_distance(G, s, cutoff)`. NaN weights
    /// hide edges.
    fn dijkstra(
        &mut self,
        adj: &Csr,
        weights: &[f64],
        s: usize,
        cutoff: Option<f64>,
    ) -> Result<(), NegativeCycle> {
        self.reset();
        let mut heap: BinaryHeap<Reverse<(HeapKey, u32)>> = BinaryHeap::new();
        let mut counter = 0u64;
        self.seen[s] = 0.0;
        self.has_seen[s] = true;
        self.touched.push(s as u32);
        heap.push(Reverse((HeapKey(0.0, counter), s as u32)));
        while let Some(Reverse((HeapKey(d, _), v))) = heap.pop() {
            let v = v as usize;
            if self.is_reached[v] {
                continue;
            }
            self.is_reached[v] = true;
            self.dist[v] = d;
            self.reached.push(v as u32);
            for e in adj.range(v) {
                let u = adj.targets[e] as usize;
                let cost = weights[e];
                if cost.is_nan() {
                    continue;
                }
                let vu = d + cost;
                if cutoff.is_some_and(|c| vu > c) {
                    continue;
                }
                if self.is_reached[u] {
                    if vu < self.dist[u] {
                        return Err(NegativeCycle);
                    } else if vu == self.dist[u] {
                        self.preds[u].push(v as u32);
                        self.arcs[u].push(e as u32);
                    }
                } else if !self.has_seen[u] || vu < self.seen[u] {
                    if !self.has_seen[u] {
                        self.has_seen[u] = true;
                        self.touched.push(u as u32);
                    }
                    self.seen[u] = vu;
                    counter += 1;
                    heap.push(Reverse((HeapKey(vu, counter), u as u32)));
                    self.preds[u].clear();
                    self.preds[u].push(v as u32);
                    self.arcs[u].clear();
                    self.arcs[u].push(e as u32);
                } else if vu == self.seen[u] {
                    self.preds[u].push(v as u32);
                    self.arcs[u].push(e as u32);
                }
            }
        }
        Ok(())
    }
}

/// `newman_betweenness_centrality` (load centrality) before normalization.
/// `rank` orders nodes as Python compares them (NetworkX sorts
/// `(distance, node)` tuples, so ties at one distance go by node).
pub fn load(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    cutoff: Option<f64>,
    rank: &[u32],
) -> Result<Vec<f64>, NegativeCycle> {
    let mut b = vec![0.0; n];
    let sources: Vec<u32> = (0..n as u32).collect();
    in_source_order(
        &sources,
        16 * n,
        || PredState::new(n),
        |st, s| {
            match weights {
                None => st.bfs(adj, s, cutoff),
                Some(w) => st.dijkstra(adj, w, s, cutoff)?,
            }
            // `_node_betweenness`
            let mut onodes: Vec<u32> = st
                .reached
                .iter()
                .copied()
                .filter(|&v| st.dist[v as usize] > 0.0)
                .collect();
            onodes.sort_unstable_by(|&a, &b| {
                let (a, b) = (a as usize, b as usize);
                st.dist[a]
                    .total_cmp(&st.dist[b])
                    .then(rank[a].cmp(&rank[b]))
            });
            for &v in &st.reached {
                st.between[v as usize] = 1.0;
            }
            for &v in onodes.iter().rev() {
                let v = v as usize;
                let num_paths = st.preds[v].len() as f64;
                for i in 0..st.preds[v].len() {
                    let x = st.preds[v][i] as usize;
                    if x == s {
                        // NetworkX stops at the source, skipping any
                        // predecessors listed after it.
                        break;
                    }
                    st.between[x] += st.between[v] / num_paths;
                }
            }
            Ok(st
                .reached
                .iter()
                .map(|&v| (v, st.between[v as usize] - 1.0))
                .collect::<Vec<_>>())
        },
        |log| {
            for (v, x) in log {
                b[v as usize] += x;
            }
        },
    )?;
    Ok(b)
}

/// `edge_load_centrality`: keys `(u, v)` and `(v, u)` for each edge of
/// `edges` (in that order, without repeats), and their values.
pub fn edge_load(
    adj: &Csr,
    n: usize,
    edges: (&[u32], &[u32]),
    cutoff: Option<f64>,
) -> (Vec<(u32, u32)>, Vec<f64>) {
    let mut ids: HashMap<(u32, u32), u32> = HashMap::new();
    let mut keys = Vec::new();
    for (&u, &v) in edges.0.iter().zip(edges.1) {
        for key in [(u, v), (v, u)] {
            ids.entry(key).or_insert_with(|| {
                keys.push(key);
                (keys.len() - 1) as u32
            });
        }
    }
    let nkeys = keys.len();
    // For each arc a -> b: the ids of (a, b) and (b, a).
    let mut fwd = vec![0u32; adj.targets.len()];
    let mut rev = vec![0u32; adj.targets.len()];
    for a in 0..n {
        for e in adj.range(a) {
            let b = adj.targets[e];
            fwd[e] = ids[&(a as u32, b)];
            rev[e] = ids[&(b, a as u32)];
        }
    }
    drop(ids);
    let mut total = vec![0.0; nkeys];
    let sources: Vec<u32> = (0..n as u32).collect();
    let _ = in_source_order(
        &sources,
        8 * nkeys + 16 * n,
        || PredState::new(n),
        |st, s| {
            st.bfs(adj, s, cutoff);
            // `_edge_betweenness`: nodes by distance (BFS order already is).
            let mut between = vec![1.0; nkeys];
            for &v in st.reached.iter().rev() {
                let v = v as usize;
                for (&w, &arc_wv) in st.preds[v].iter().zip(&st.arcs[v]) {
                    let w = w as usize;
                    let arc_wv = arc_wv as usize;
                    let num_paths = st.preds[w].len() as f64;
                    for &arc_xw in &st.arcs[w] {
                        let arc_xw = arc_xw as usize;
                        // between[(w, x)] += between[(v, w)] / num_paths
                        between[rev[arc_xw] as usize] += between[rev[arc_wv] as usize] / num_paths;
                        // between[(x, w)] += between[(w, v)] / num_paths
                        between[fwd[arc_xw] as usize] += between[fwd[arc_wv] as usize] / num_paths;
                    }
                }
            }
            Ok(between)
        },
        |between| {
            for (t, x) in total.iter_mut().zip(between) {
                *t += x;
            }
        },
    );
    (keys, total)
}

/// `voterank`: the elected nodes. `edges` is `G.edges()`; `avg_degree` is
/// NetworkX's average (out-)degree.
pub fn voterank(
    adj: &Csr,
    n: usize,
    directed: bool,
    edges: (&[u32], &[u32]),
    number: usize,
    avg_degree: f64,
) -> Vec<u32> {
    let mut ability = vec![1.0f64; n];
    let mut score = vec![0.0f64; n];
    let mut elected: Vec<u32> = Vec::new();
    for _ in 0..number {
        score.fill(0.0);
        for (&u, &v) in edges.0.iter().zip(edges.1) {
            score[u as usize] += ability[v as usize];
            if !directed {
                score[v as usize] += ability[u as usize];
            }
        }
        for &v in &elected {
            score[v as usize] = 0.0;
        }
        // `max(G.nodes, key=...)`: the first node with the top score.
        let mut best = 0;
        for v in 1..n {
            if score[v] > score[best] {
                best = v;
            }
        }
        if score[best] == 0.0 {
            break;
        }
        elected.push(best as u32);
        ability[best] = 0.0;
        for &w in adj.neighbors(best) {
            let w = w as usize;
            ability[w] -= 1.0 / avg_degree;
            if ability[w] < 0.0 {
                ability[w] = 0.0;
            }
        }
    }
    elected
}

/// Per-thread buffers for `dispersion_pair`.
pub struct DispersionState {
    in_u: Vec<u32>,
    in_a: Vec<u32>,
    epoch: u32,
    common: Vec<u32>,
}

impl DispersionState {
    pub fn new(n: usize) -> Self {
        DispersionState {
            in_u: vec![0; n],
            in_a: vec![0; n],
            epoch: 0,
            common: Vec::new(),
        }
    }
}

/// `_dispersion(G, u, v)` of an undirected graph without self-loops:
/// `(total, embeddedness)`. There NetworkX's test is symmetric in `s` and
/// `t`, so the order of its set of common neighbors doesn't matter.
pub fn dispersion_pair(adj: &Csr, st: &mut DispersionState, u: usize, v: usize) -> (u64, u64) {
    st.epoch += 1;
    let mark_u = st.epoch;
    for &x in adj.neighbors(u) {
        st.in_u[x as usize] = mark_u;
    }
    st.common.clear();
    for &x in adj.neighbors(v) {
        if st.in_u[x as usize] == mark_u {
            st.common.push(x);
        }
    }
    let mut total = 0u64;
    for i in 0..st.common.len() {
        let s = st.common[i] as usize;
        // nbrs_s = u_nbrs & G[s] - {u, v}
        st.epoch += 1;
        let mark_a = st.epoch;
        for &y in adj.neighbors(s) {
            let y = y as usize;
            if st.in_u[y] == mark_u && y != u && y != v {
                st.in_a[y] = mark_a;
            }
        }
        for j in i + 1..st.common.len() {
            let t = st.common[j] as usize;
            if st.in_a[t] == mark_a {
                continue;
            }
            if adj
                .neighbors(t)
                .iter()
                .all(|&y| st.in_a[y as usize] != mark_a)
            {
                total += 1;
            }
        }
    }
    (total, st.common.len() as u64)
}

/// `(total, embeddedness)` for each `(u, v)` pair.
pub fn dispersion(adj: &Csr, n: usize, us: &[u32], vs: &[u32]) -> Vec<(u64, u64)> {
    us.par_iter()
        .zip(vs)
        .map_init(
            || DispersionState::new(n),
            |st, (&u, &v)| dispersion_pair(adj, st, u as usize, v as usize),
        )
        .collect()
}

/// Number of nodes outside `group` adjacent from it (`group_degree_centrality`'s
/// numerator).
pub fn group_degree(adj: &Csr, n: usize, group: &[u32]) -> usize {
    let mut in_group = vec![false; n];
    for &v in group {
        in_group[v as usize] = true;
    }
    let mut counted = vec![false; n];
    let mut count = 0;
    for &v in group {
        for &w in adj.neighbors(v as usize) {
            let w = w as usize;
            if !in_group[w] && !counted[w] {
                counted[w] = true;
                count += 1;
            }
        }
    }
    count
}

/// Multi-source shortest-path distances (`None` = unreached). With
/// non-negative weights they don't depend on visiting order. NaN weights
/// hide edges.
pub fn multi_source_distances(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    sources: &[u32],
) -> Vec<Option<f64>> {
    let mut dist: Vec<Option<f64>> = vec![None; n];
    let mut seen = vec![f64::INFINITY; n];
    let mut heap: BinaryHeap<Reverse<(HeapKey, u32)>> = BinaryHeap::new();
    let mut counter = 0u64;
    for &s in sources {
        seen[s as usize] = 0.0;
        counter += 1;
        heap.push(Reverse((HeapKey(0.0, counter), s)));
    }
    while let Some(Reverse((HeapKey(d, _), v))) = heap.pop() {
        let v = v as usize;
        if dist[v].is_some() {
            continue;
        }
        dist[v] = Some(d);
        for e in adj.range(v) {
            let u = adj.targets[e] as usize;
            let cost = weights.map_or(1.0, |w| w[e]);
            if cost.is_nan() || dist[u].is_some() {
                continue;
            }
            let du = d + cost;
            if du < seen[u] {
                seen[u] = du;
                counter += 1;
                heap.push(Reverse((HeapKey(du, counter), u as u32)));
            }
        }
    }
    dist
}

/// Unweighted local reaching centrality parts for each source: the number of
/// nodes reached and `sum(1 / d)` over them in BFS order (CPython's `sum`).
pub fn reaching_unweighted(
    adj: &Csr,
    n: usize,
    sources: &[u32],
    compensated: bool,
) -> Vec<(usize, f64)> {
    sources
        .par_iter()
        .map(|&s| {
            let (order, levels) = super::traversal::bfs_lengths(adj, n, s as usize, f64::INFINITY);
            let mut sum = PySum::ZERO;
            for &d in &levels[1..] {
                sum = sum.add(1.0 / d as f64, compensated);
            }
            (order.len(), sum.value())
        })
        .collect()
}

/// Weighted local reaching centrality: for each source, NetworkX's
/// `sum(_average_weight(G, path, weight) for path in paths.values())`, with
/// paths from Dijkstra on lengths `total / weight`. `pop_order` picks the
/// order of the paths dict (pop order, or else first-push order).
pub fn reaching_weighted(
    adj: &Csr,
    n: usize,
    weights: &[f64],
    total: f64,
    sources: &[u32],
    pop_order: bool,
    compensated: bool,
) -> Vec<f64> {
    struct State {
        done: Vec<bool>,
        seen: Vec<f64>,
        has_seen: Vec<bool>,
        parent_arc: Vec<u32>,
        parent: Vec<u32>,
        depth: Vec<u32>,
        path_sum: Vec<PySum>,
        pushed: Vec<u32>,
        popped: Vec<u32>,
    }
    sources
        .par_iter()
        .map_init(
            || State {
                done: vec![false; n],
                seen: vec![0.0; n],
                has_seen: vec![false; n],
                parent_arc: vec![UNSET; n],
                parent: vec![UNSET; n],
                depth: vec![0; n],
                path_sum: vec![PySum::ZERO; n],
                pushed: Vec::new(),
                popped: Vec::new(),
            },
            |st, &s| {
                for &v in &st.pushed {
                    st.done[v as usize] = false;
                    st.has_seen[v as usize] = false;
                }
                st.pushed.clear();
                st.popped.clear();
                let s = s as usize;
                let mut heap: BinaryHeap<Reverse<(HeapKey, u32)>> = BinaryHeap::new();
                let mut counter = 0u64;
                st.seen[s] = 0.0;
                st.has_seen[s] = true;
                st.parent_arc[s] = UNSET;
                st.pushed.push(s as u32);
                heap.push(Reverse((HeapKey(0.0, counter), s as u32)));
                // `_dijkstra_multisource` with `paths`: a node's path is its
                // parent's path plus itself, the parent being the last node
                // that improved its distance.
                while let Some(Reverse((HeapKey(d, _), v))) = heap.pop() {
                    let v = v as usize;
                    if st.done[v] {
                        continue;
                    }
                    st.done[v] = true;
                    st.popped.push(v as u32);
                    for e in adj.range(v) {
                        let u = adj.targets[e] as usize;
                        let vu = d + total / weights[e];
                        if st.done[u] {
                            continue;
                        }
                        if !st.has_seen[u] || vu < st.seen[u] {
                            if !st.has_seen[u] {
                                st.has_seen[u] = true;
                                st.pushed.push(u as u32);
                            }
                            st.seen[u] = vu;
                            st.parent_arc[u] = e as u32;
                            st.parent[u] = v as u32;
                            counter += 1;
                            heap.push(Reverse((HeapKey(vu, counter), u as u32)));
                        }
                    }
                }
                // Parents are popped before their children.
                for &v in &st.popped {
                    let v = v as usize;
                    if v == s {
                        st.depth[v] = 0;
                        st.path_sum[v] = PySum::ZERO;
                        continue;
                    }
                    let e = st.parent_arc[v] as usize;
                    let p = st.parent[v] as usize;
                    st.depth[v] = st.depth[p] + 1;
                    st.path_sum[v] = st.path_sum[p].add(weights[e], compensated);
                }
                let order = if pop_order { &st.popped } else { &st.pushed };
                let mut sum = PySum::ZERO;
                for &v in order {
                    let v = v as usize;
                    if v != s {
                        sum = sum.add(st.path_sum[v].value() / st.depth[v] as f64, compensated);
                    }
                }
                sum.value()
            },
        )
        .collect()
}

/// `_group_preprocessing` restricted to the group nodes `set_v` (all
/// `K x K` row-major matrices, indexed by position in `set_v`).
pub struct GroupData {
    /// `sigma[x][y]`, halved for weighted graphs as NetworkX does.
    pub sigma: Vec<f64>,
    /// `D[x][y]` where `reached`.
    pub dist: Vec<f64>,
    pub reached: Vec<bool>,
    /// Position of `y` in `D[x]`'s key order.
    pub pos: Vec<u32>,
    /// `len(D[x])`.
    pub reach_len: Vec<u32>,
    /// `PB[x][y]`.
    pub pb: Vec<f64>,
}

pub fn group_preprocessing(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    set_v: &[u32],
) -> GroupData {
    let k = set_v.len();
    let weighted = weights.is_some();
    let half = |x: f64| if weighted { x / 2.0 } else { x };
    // One row of each matrix per group node, and `len(D[x])`.
    type Row = (Vec<f64>, Vec<f64>, Vec<bool>, Vec<u32>, u32);
    let rows: Vec<Row> = set_v
        .par_iter()
        .map_init(
            || Spt::new(n),
            |st, &s| {
                st.run_with_positions(adj, weights, s as usize);
                let mut sigma = vec![0.0; k];
                let mut dist = vec![0.0; k];
                let mut reached = vec![false; k];
                let mut pos = vec![UNSET; k];
                for (j, &t) in set_v.iter().enumerate() {
                    let t = t as usize;
                    if st.pos[t] != UNSET {
                        sigma[j] = half(st.sigma[t]);
                        dist[j] = st.dist(t, weighted);
                        reached[j] = true;
                        pos[j] = st.pos[t];
                    }
                }
                (sigma, dist, reached, pos, st.order.len() as u32)
            },
        )
        .collect();
    let mut data = GroupData {
        sigma: Vec::with_capacity(k * k),
        dist: Vec::with_capacity(k * k),
        reached: Vec::with_capacity(k * k),
        pos: Vec::with_capacity(k * k),
        reach_len: Vec::with_capacity(k),
        pb: vec![0.0; k * k],
    };
    for (sigma, dist, reached, pos, len) in rows {
        data.sigma.extend(sigma);
        data.dist.extend(dist);
        data.reached.extend(reached);
        data.pos.extend(pos);
        data.reach_len.push(len);
    }
    // `PB[g1][g2]` sums over every node in `G` order.
    let sources: Vec<u32> = (0..n as u32).collect();
    let (sigma1, dist1, reached1) = (&data.sigma, &data.dist, &data.reached);
    let pb = &mut data.pb;
    let _ = in_source_order(
        &sources,
        16 * n + 24 * k * k,
        || Spt::new(n),
        |st, s| {
            st.run_with_positions(adj, weights, s);
            // `_accumulate_endpoints`, then `delta[s][i] += 1` for `i != s`.
            for i in (0..st.order.len()).rev() {
                let w = st.order[i] as usize;
                let coeff = (1.0 + st.delta[w]) / st.sigma[w];
                for &v in &st.preds[w] {
                    let v = v as usize;
                    st.delta[v] += st.sigma[v] * coeff;
                }
            }
            let mut log = Vec::new();
            for (a, &g1) in set_v.iter().enumerate() {
                let g1 = g1 as usize;
                if st.pos[g1] == UNSET {
                    continue;
                }
                let d_g1 = st.dist(g1, weighted);
                let sigma_g1 = half(st.sigma[g1]);
                for (b, &g2) in set_v.iter().enumerate() {
                    let g2 = g2 as usize;
                    if !reached1[a * k + b] || st.pos[g2] == UNSET {
                        continue;
                    }
                    if st.dist(g2, weighted) == d_g1 + dist1[a * k + b] {
                        let delta = st.delta[g2] + if g2 != s { 1.0 } else { 0.0 };
                        let x = delta * sigma_g1 * sigma1[a * k + b] / half(st.sigma[g2]);
                        log.push(((a * k + b) as u32, x));
                    }
                }
            }
            Ok(log)
        },
        |log| {
            for (i, x) in log {
                pb[i as usize] += x;
            }
        },
    );
    data
}

/// A `D[a][b]` lookup NetworkX would make that raises `KeyError`.
pub struct MissingDistance;

/// The main loop of `group_betweenness_centrality` for one group (positions
/// in `set_v`, in the group set's order): `PB_m[v][v]` for each `v`, which
/// NetworkX adds up. `y_orders` (NetworkX 3.7+, which updates the matrices
/// in place) gives, for each `x`, the order of `group & D[x].keys()`;
/// without it this is the earlier version, which iterates `group` twice and
/// builds new matrices from the old ones.
pub fn group_main(
    data: &GroupData,
    k: usize,
    group: &[u32],
    y_orders: Option<&[Vec<u32>]>,
) -> Result<Vec<f64>, MissingDistance> {
    let at = |a: u32, b: u32| a as usize * k + b as usize;
    let d = &data.dist;
    let reached = &data.reached;
    let sigma = &data.sigma;
    let mut sm = sigma.clone();
    let mut pm = data.pb.clone();
    let mut out = Vec::with_capacity(group.len());
    match y_orders {
        None => {
            let mut sv = sm.clone();
            let mut pv = pm.clone();
            for &v in group {
                out.push(pm[at(v, v)]);
                for &x in group {
                    for &y in group {
                        let (mut dxvy, mut dxyv, mut dvxy) = (0.0, 0.0, 0.0);
                        if !(sm[at(x, y)] == 0.0 || sm[at(x, v)] == 0.0 || sm[at(v, y)] == 0.0) {
                            if !reached[at(y, v)] {
                                return Err(MissingDistance);
                            }
                            if d[at(x, v)] == d[at(x, y)] + d[at(y, v)] {
                                dxyv = sm[at(x, y)] * sm[at(y, v)] / sm[at(x, v)];
                            }
                            if d[at(x, y)] == d[at(x, v)] + d[at(v, y)] {
                                dxvy = sm[at(x, v)] * sm[at(v, y)] / sm[at(x, y)];
                            }
                            if !reached[at(v, x)] {
                                return Err(MissingDistance);
                            }
                            if d[at(v, y)] == d[at(v, x)] + d[at(x, y)] {
                                dvxy = sm[at(v, x)] * sigma[at(x, y)] / sigma[at(v, y)];
                            }
                        }
                        sv[at(x, y)] = sm[at(x, y)] * (1.0 - dxvy);
                        pv[at(x, y)] = pm[at(x, y)] - pm[at(x, y)] * dxvy;
                        if y != v {
                            pv[at(x, y)] -= pm[at(x, v)] * dxyv;
                        }
                        if x != v {
                            pv[at(x, y)] -= pm[at(v, y)] * dvxy;
                        }
                    }
                }
                std::mem::swap(&mut sm, &mut sv);
                std::mem::swap(&mut pm, &mut pv);
            }
        }
        Some(y_orders) => {
            for &v in group {
                out.push(pm[at(v, v)]);
                for (&x, ys) in group.iter().zip(y_orders) {
                    let mut sig_xv = sm[at(x, v)];
                    let sig_vx = sm[at(v, x)];
                    let x_in_dv = reached[at(v, x)];
                    let v_in_dx = reached[at(x, v)];
                    for &y in ys {
                        let sig_xy = sm[at(x, y)];
                        let sig_vy = sm[at(v, y)];
                        let v_in_dy = reached[at(y, v)];
                        let y_in_dv = reached[at(v, y)];
                        // Order x-y-v
                        if v_in_dy
                            && d[at(x, v)] == d[at(x, y)] + d[at(y, v)]
                            && sig_xv != 0.0
                            && y != v
                        {
                            pm[at(x, y)] -= pm[at(x, v)] * sig_xy * sm[at(y, v)] / sig_xv;
                        }
                        // Order v-x-y
                        if x_in_dv
                            && d[at(v, y)] == d[at(v, x)] + d[at(x, y)]
                            && sig_vy != 0.0
                            && x != v
                        {
                            pm[at(x, y)] -= pm[at(v, y)] * sig_vx * sig_xy / sig_vy;
                        }
                        // Order x-v-y
                        if v_in_dx
                            && y_in_dv
                            && d[at(x, y)] == d[at(x, v)] + d[at(v, y)]
                            && sig_xy != 0.0
                        {
                            let sig_xvy = sig_xv * sig_vy;
                            pm[at(x, y)] *= 1.0 - sig_xvy / sig_xy;
                            sm[at(x, y)] -= sig_xvy;
                            if y == v {
                                sig_xv -= sig_xvy;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Number of nodes reaching each of `targets` (themselves included).
pub fn reverse_reach_counts(pred: &Csr, n: usize, targets: &[u32]) -> Vec<u32> {
    targets
        .par_iter()
        .map(|&t| {
            let (order, _) = super::traversal::bfs_lengths(pred, n, t as usize, f64::INFINITY);
            order.len() as u32
        })
        .collect()
}

/// `prominent_group`'s candidate-list sort and heuristic.
type ClSort<'a> = dyn Fn(&[f64], &[u32]) -> Result<Vec<u32>, MissingDistance> + 'a;
type Heuristic<'a> = dyn Fn(&[f64], &[u32], usize) -> Result<f64, MissingDistance> + 'a;

/// One node of `prominent_group`'s search tree (`DF_tree`). `sigma` and
/// `betweenness` (`B[x][y]`, NetworkX's DataFrame column `x`, row `y`) are
/// shared with the parent when unchanged.
struct SearchNode {
    mats: std::rc::Rc<(Vec<f64>, Vec<f64>)>,
    cl: Vec<u32>,
    cont: std::rc::Rc<Vec<f64>>,
    gm: Vec<u32>,
    gbc: f64,
    heu: f64,
}

/// `prominent_group(G, k)` with `C=None`, where the group nodes are all of
/// `G` in order: `(max_GBC, max_group)` before endpoint and normalization
/// adjustments, `max_group` being `None` while NetworkX keeps its initial
/// `0, []`. `rank` orders nodes as Python compares them (NetworkX sorts
/// `(score, node)` tuples). `Err` where NetworkX raises (`KeyError` or
/// `IndexError`) or would sort NaN.
pub fn prominent_group(
    data: &GroupData,
    k: usize,
    greedy: bool,
    rank: &[u32],
) -> Result<(f64, Option<Vec<u32>>), MissingDistance> {
    let n = data.reach_len.len();
    let diag = |b: &[f64]| -> Vec<f64> { (0..n).map(|i| b[i * n + i]).collect() };
    let sorted_cl = |cont: &[f64], gm: &[u32]| -> Result<Vec<u32>, MissingDistance> {
        if cont.iter().any(|x| x.is_nan()) {
            return Err(MissingDistance);
        }
        let mut cl: Vec<u32> = (0..n as u32).filter(|v| !gm.contains(v)).collect();
        // sorted(zip(diag, nodes), reverse=True)
        cl.sort_unstable_by(|&a, &b| {
            let (a, b) = (a as usize, b as usize);
            cont[b].total_cmp(&cont[a]).then(rank[b].cmp(&rank[a]))
        });
        Ok(cl)
    };
    let heuristic = |cont: &[f64], cl: &[u32], need: usize| -> Result<f64, MissingDistance> {
        if need > cl.len() {
            return Err(MissingDistance); // IndexError
        }
        Ok(cl[..need].iter().fold(0.0, |h, &v| h + cont[v as usize]))
    };
    let pb = data.pb.clone();
    let cont = diag(&pb);
    let cl = sorted_cl(&cont, &[])?;
    let heu = heuristic(&cont, &cl, k)?;
    let root = SearchNode {
        mats: std::rc::Rc::new((data.sigma.clone(), pb)),
        cl,
        cont: std::rc::Rc::new(cont),
        gm: Vec::new(),
        gbc: 0.0,
        heu,
    };

    struct Search<'a> {
        data: &'a GroupData,
        n: usize,
        k: usize,
        greedy: bool,
        max_gbc: f64,
        max_group: Option<Vec<u32>>,
    }

    impl Search<'_> {
        /// `_heuristic`: the plus child (and the minus child unless greedy).
        fn children(
            &self,
            root: &SearchNode,
            sorted_cl: &ClSort,
            heuristic: &Heuristic,
        ) -> Result<(SearchNode, Option<SearchNode>), MissingDistance> {
            let n = self.n;
            let (d, reached) = (&self.data.dist, &self.data.reached);
            let (s, b) = (&root.mats.0, &root.mats.1);
            let a = root.cl[0] as usize;
            let mut gm = root.gm.clone();
            gm.push(a as u32);
            let gbc = root.gbc + root.cont[a];
            let mut ps = vec![0.0; n * n];
            let mut pbm = vec![0.0; n * n];
            for x in 0..n {
                for y in 0..n {
                    let (mut dxvy, mut dxyv, mut dvxy) = (0.0, 0.0, 0.0);
                    if !(s[x * n + y] == 0.0 || s[x * n + a] == 0.0 || s[a * n + y] == 0.0) {
                        if !reached[y * n + a] {
                            return Err(MissingDistance);
                        }
                        if d[x * n + a] == d[x * n + y] + d[y * n + a] {
                            dxyv = s[x * n + y] * s[y * n + a] / s[x * n + a];
                        }
                        if d[x * n + y] == d[x * n + a] + d[a * n + y] {
                            dxvy = s[x * n + a] * s[a * n + y] / s[x * n + y];
                        }
                        if !reached[a * n + x] {
                            return Err(MissingDistance);
                        }
                        if d[a * n + y] == d[a * n + x] + d[x * n + y] {
                            dvxy = s[a * n + x] * s[x * n + y] / s[a * n + y];
                        }
                    }
                    ps[x * n + y] = s[x * n + y] * (1.0 - dxvy);
                    let mut v = b[x * n + y] - b[x * n + y] * dxvy;
                    if y != a {
                        v -= b[x * n + a] * dxyv;
                    }
                    if x != a {
                        v -= b[a * n + y] * dvxy;
                    }
                    pbm[x * n + y] = v;
                }
            }
            let cont: Vec<f64> = (0..n).map(|i| pbm[i * n + i]).collect();
            let cl = sorted_cl(&cont, &gm)?;
            let heu = heuristic(&cont, &cl, self.k - gm.len().min(self.k))?;
            let plus = SearchNode {
                mats: std::rc::Rc::new((ps, pbm)),
                cl,
                cont: std::rc::Rc::new(cont),
                gm,
                gbc,
                heu,
            };
            let minus = if self.greedy {
                None
            } else {
                let cl = root.cl[1..].to_vec();
                let heu = heuristic(&root.cont, &cl, self.k - root.gm.len())?;
                Some(SearchNode {
                    mats: root.mats.clone(),
                    cl,
                    cont: root.cont.clone(),
                    gm: root.gm.clone(),
                    gbc: root.gbc,
                    heu,
                })
            };
            Ok((plus, minus))
        }

        /// `_dfbnb`
        fn dfbnb(
            &mut self,
            root: &SearchNode,
            sorted_cl: &ClSort,
            heuristic: &Heuristic,
        ) -> Result<(), MissingDistance> {
            let k = self.k;
            if root.gm.len() == k && root.gbc > self.max_gbc {
                self.max_gbc = root.gbc;
                self.max_group = Some(root.gm.clone());
                return Ok(());
            }
            if root.gm.len() == k
                || root.cl.len() <= k.saturating_sub(root.gm.len())
                || root.gbc + root.heu <= self.max_gbc
            {
                return Ok(());
            }
            let (plus, minus) = self.children(root, sorted_cl, heuristic)?;
            match minus {
                None => self.dfbnb(&plus, sorted_cl, heuristic)?,
                Some(minus) => {
                    if plus.gbc + plus.heu > minus.gbc + minus.heu {
                        self.dfbnb(&plus, sorted_cl, heuristic)?;
                        self.dfbnb(&minus, sorted_cl, heuristic)?;
                    } else {
                        self.dfbnb(&minus, sorted_cl, heuristic)?;
                        self.dfbnb(&plus, sorted_cl, heuristic)?;
                    }
                }
            }
            Ok(())
        }
    }

    let mut search = Search {
        data,
        n,
        k,
        greedy,
        max_gbc: 0.0,
        max_group: None,
    };
    search.dfbnb(&root, &sorted_cl, &heuristic)?;
    Ok((search.max_gbc, search.max_group))
}
