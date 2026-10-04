//! Batch 15: communities, efficiency and structural measures.
//!
//! Each function replays the NetworkX code it replaces step by step: the
//! order in which floats are added (Python's `sum()` is compensated from
//! 3.12, see `spectral::py_sum`), how ties are broken, and the random draws
//! of the label propagation functions (`Mt19937` is CPython's generator), so
//! results match bit for bit.

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet};

use rayon::prelude::*;

use super::centrality_more::{in_source_order, Spt};
use super::directed::strongly_connected_components;
use super::spectral::py_sum;
use super::traversal::DijkstraState;
use crate::graph::Csr;

const UNSET: u32 = u32::MAX;
/// 2^53: integers up to this size are exact in an f64.
const MAX_EXACT: f64 = 9_007_199_254_740_992.0;

/// CPython's `sum()` of floats as a running state, so one sum can be fed
/// from several parallel blocks in order.
#[derive(Clone, Copy)]
pub struct RunningSum {
    s: f64,
    c: f64,
    compensated: bool,
}

impl RunningSum {
    pub fn new(compensated: bool) -> Self {
        RunningSum {
            s: 0.0,
            c: 0.0,
            compensated,
        }
    }

    #[inline]
    pub fn add(&mut self, x: f64) {
        let t = self.s + x;
        if self.compensated {
            if self.s.abs() >= x.abs() {
                self.c += (self.s - t) + x;
            } else {
                self.c += (x - t) + self.s;
            }
        }
        self.s = t;
    }

    pub fn value(&self) -> f64 {
        if self.c != 0.0 && self.c.is_finite() {
            self.s + self.c
        } else {
            self.s
        }
    }
}

/// Runs `work` for every item, in parallel blocks, and hands each result to
/// `apply` in item order.
fn ordered_blocks<S, T, I, W, A>(items: &[u32], init: I, work: W, mut apply: A)
where
    I: Fn() -> S + Sync + Send,
    W: Fn(&mut S, u32) -> T + Sync + Send,
    T: Send,
    A: FnMut(T),
{
    let block = (rayon::current_num_threads().max(1) * 16).max(1);
    for chunk in items.chunks(block) {
        let results: Vec<T> = chunk
            .par_iter()
            .map_init(&init, |state, &s| work(state, s))
            .collect();
        for r in results {
            apply(r);
        }
    }
}

/// BFS buffers, reset by stamping instead of clearing.
struct Bfs {
    seen: Vec<u32>,
    stamp: u32,
    queue: Vec<u32>,
}

impl Bfs {
    fn new(n: usize) -> Self {
        Bfs {
            seen: vec![0; n],
            stamp: 0,
            queue: Vec::new(),
        }
    }

    fn next_stamp(&mut self) -> u32 {
        if self.stamp == u32::MAX {
            self.seen.fill(0);
            self.stamp = 0;
        }
        self.stamp += 1;
        self.stamp
    }

    /// Number of nodes at each distance 1, 2, ... from `s`, moving only to
    /// nodes `allowed` accepts and never through `skip`.
    fn level_counts(
        &mut self,
        adj: &Csr,
        s: usize,
        allowed: impl Fn(usize) -> bool,
        skip: usize,
    ) -> Vec<u64> {
        let stamp = self.next_stamp();
        self.queue.clear();
        self.queue.push(s as u32);
        self.seen[s] = stamp;
        let mut counts = Vec::new();
        let mut start = 0;
        while start < self.queue.len() {
            let end = self.queue.len();
            for i in start..end {
                let v = self.queue[i] as usize;
                for &w in adj.neighbors(v) {
                    let w = w as usize;
                    if self.seen[w] != stamp && w != skip && allowed(w) {
                        self.seen[w] = stamp;
                        self.queue.push(w as u32);
                    }
                }
            }
            if self.queue.len() > end {
                counts.push((self.queue.len() - end) as u64);
            }
            start = end;
        }
        counts
    }
}

/// Adds `1 / d` once per node at distance `d`, in BFS order (`counts[i]`
/// nodes at distance `i + 1`), as `global_efficiency`'s `g_eff += 1 / d`.
fn add_inverse_distances(g: &mut f64, counts: &[u64]) {
    for (i, &k) in counts.iter().enumerate() {
        let x = 1.0 / (i as f64 + 1.0);
        for _ in 0..k {
            *g += x;
        }
    }
}

/// `global_efficiency`'s running total before the division.
pub fn global_efficiency(adj: &Csr, n: usize) -> f64 {
    let sources: Vec<u32> = (0..n as u32).collect();
    let mut g = 0.0;
    ordered_blocks(
        &sources,
        || Bfs::new(n),
        |b, s| b.level_counts(adj, s as usize, |_| true, usize::MAX),
        |counts| add_inverse_distances(&mut g, &counts),
    );
    g
}

/// `local_efficiency`'s per-node efficiencies: `None` where NetworkX's
/// `global_efficiency` returns the int 0 (fewer than two neighbors).
/// `orders[v]` is the iteration order of `G.subgraph(G[v])` when it is the
/// order of a Python set; `None` means G's order.
pub fn local_efficiency(adj: &Csr, n: usize, orders: &[Option<Vec<u32>>]) -> Vec<Option<f64>> {
    (0..n)
        .into_par_iter()
        .map_init(
            || (Bfs::new(n), vec![0u32; n]),
            |(bfs, member), v| {
                let members: Vec<u32> = match &orders[v] {
                    Some(order) => order.clone(),
                    None => {
                        let mut m: Vec<u32> = adj.neighbors(v).to_vec();
                        m.sort_unstable();
                        m.dedup();
                        m
                    }
                };
                let k = members.len();
                if k < 2 {
                    return None;
                }
                let tag = v as u32 + 1;
                for &w in &members {
                    member[w as usize] = tag;
                }
                let mut g = 0.0;
                for &s in &members {
                    let counts =
                        bfs.level_counts(adj, s as usize, |w| member[w] == tag, usize::MAX);
                    add_inverse_distances(&mut g, &counts);
                }
                for &w in &members {
                    member[w as usize] = 0;
                }
                Some(g / (k * (k - 1)) as f64)
            },
        )
        .collect()
}

/// Weighted degrees as `G.degree(weight)` computes them for an undirected
/// graph: the adjacency's weights summed in order, plus a self-loop's
/// weight again. Ints are exact; `None` if an int sum gets too large.
pub fn degrees_int(adj: &Csr, n: usize, weights: Option<&[f64]>) -> Option<Vec<i128>> {
    (0..n)
        .map(|v| {
            let mut d = 0i128;
            for e in adj.range(v) {
                let w = weights.map_or(1.0, |w| w[e]);
                d += w as i128;
                if adj.targets[e] as usize == v {
                    d += w as i128;
                }
            }
            ((d.unsigned_abs() as f64) < MAX_EXACT).then_some(d)
        })
        .collect()
}

/// Float version of `degrees_int`: Python's `sum()` over the adjacency's
/// weights, then the self-loop's weight added.
pub fn degrees_float(adj: &Csr, n: usize, weights: &[f64], compensated: bool) -> Vec<f64> {
    (0..n)
        .map(|v| {
            let r = adj.range(v);
            let mut d = py_sum(weights[r.clone()].iter().copied(), compensated);
            for e in r {
                if adj.targets[e] as usize == v {
                    d += weights[e];
                }
            }
            d
        })
        .collect()
}

/// Per-row sums of `adj`'s weights (out- or in-degrees of a directed graph).
pub fn row_sums_int(adj: &Csr, n: usize, weights: Option<&[f64]>) -> Vec<i128> {
    (0..n)
        .map(|v| {
            adj.range(v)
                .map(|e| weights.map_or(1.0, |w| w[e]) as i128)
                .sum()
        })
        .collect()
}

pub fn row_sums_float(adj: &Csr, n: usize, weights: &[f64], compensated: bool) -> Vec<f64> {
    (0..n)
        .map(|v| py_sum(weights[adj.range(v)].iter().copied(), compensated))
        .collect()
}

/// Which distance index (`nx.gutman_index` and friends) to compute.
#[derive(Clone, Copy, PartialEq)]
pub enum IndexKind {
    Gutman,
    Schultz,
    HyperWiener,
}

/// Distances from `s` in NetworkX's order (BFS levels, or Dijkstra's pop
/// order), as f64.
fn distances_from(
    adj: &Csr,
    weights: Option<&[f64]>,
    s: usize,
    bfs: &mut Bfs,
    dij: &mut DijkstraState,
    skip: usize,
) -> Option<Vec<(u32, f64)>> {
    match weights {
        None => {
            let stamp = bfs.next_stamp();
            bfs.queue.clear();
            bfs.queue.push(s as u32);
            bfs.seen[s] = stamp;
            let mut out = vec![(s as u32, 0.0)];
            let mut start = 0;
            let mut level = 0.0;
            while start < bfs.queue.len() {
                let end = bfs.queue.len();
                level += 1.0;
                for i in start..end {
                    let v = bfs.queue[i] as usize;
                    for &w in adj.neighbors(v) {
                        let wu = w as usize;
                        if bfs.seen[wu] != stamp && wu != skip {
                            bfs.seen[wu] = stamp;
                            bfs.queue.push(w);
                            out.push((w, level));
                        }
                    }
                }
                start = end;
            }
            Some(out)
        }
        Some(w) => {
            dij.run(adj, Some(w), s, None).ok()?;
            Some(
                dij.order
                    .iter()
                    .map(|&v| (v, dij.dist[v as usize]))
                    .collect(),
            )
        }
    }
}

/// Integer `gutman_index` / `schultz_index` / `hyper_wiener_index` sums
/// (before the final halving), or `None` on overflow.
pub fn distance_index_int(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    deg: &[i128],
    kind: IndexKind,
) -> Option<i128> {
    let parts: Vec<Option<i128>> = (0..n)
        .into_par_iter()
        .map_init(
            || (Bfs::new(n), DijkstraState::new(n)),
            |(bfs, dij), s| {
                let mut total = 0i128;
                for (v, d) in distances_from(adj, weights, s, bfs, dij, usize::MAX)? {
                    if d >= MAX_EXACT {
                        return None;
                    }
                    let d = d as i128;
                    let term = match kind {
                        IndexKind::Gutman => d.checked_mul(deg[s])?.checked_mul(deg[v as usize])?,
                        IndexKind::Schultz => d.checked_mul(deg[s] + deg[v as usize])?,
                        IndexKind::HyperWiener => d.checked_add(d.checked_mul(d)?)?,
                    };
                    total = total.checked_add(term)?;
                }
                Some(total)
            },
        )
        .collect();
    parts
        .into_iter()
        .try_fold(0i128, |acc, p| acc.checked_add(p?))
}

/// Float `gutman_index` / `schultz_index` sums: every term added in
/// NetworkX's order (sources in order, each in Dijkstra's pop order).
pub fn distance_index_float(
    adj: &Csr,
    n: usize,
    weights: &[f64],
    deg: &[f64],
    kind: IndexKind,
    compensated: bool,
) -> Option<f64> {
    let sources: Vec<u32> = (0..n as u32).collect();
    let mut sum = RunningSum::new(compensated);
    let mut failed = false;
    ordered_blocks(
        &sources,
        || (Bfs::new(0), DijkstraState::new(n)),
        |(bfs, dij), s| {
            let s = s as usize;
            distances_from(adj, Some(weights), s, bfs, dij, usize::MAX).map(|dists| {
                dists
                    .into_iter()
                    .map(|(v, d)| match kind {
                        IndexKind::Schultz => d * (deg[s] + deg[v as usize]),
                        _ => d * deg[s] * deg[v as usize],
                    })
                    .collect::<Vec<f64>>()
            })
        },
        |terms| match terms {
            Some(terms) => {
                for t in terms {
                    sum.add(t);
                }
            }
            None => failed = true,
        },
    );
    (!failed).then(|| sum.value())
}

/// Arc weights with the arcs into `skip` hidden (NaN), so Dijkstra never
/// reaches it: G with `skip` removed, in the same neighbor order.
fn without_node(adj: &Csr, weights: Option<&[f64]>, skip: usize) -> Option<Vec<f64>> {
    let w = weights?;
    if skip == usize::MAX {
        return Some(w.to_vec());
    }
    Some(
        w.iter()
            .zip(&adj.targets)
            .map(|(&x, &t)| if t as usize == skip { f64::NAN } else { x })
            .collect(),
    )
}

/// One source's distances in G without `skip`, or `None` if it doesn't
/// reach every other node (a negative cycle can't happen: callers decline
/// negative weights).
fn wiener_row(
    adj: &Csr,
    weights: Option<&[f64]>,
    s: usize,
    size: usize,
    skip: usize,
    bfs: &mut Bfs,
    dij: &mut DijkstraState,
) -> Option<Vec<f64>> {
    let dists = distances_from(adj, weights, s, bfs, dij, skip)?;
    (dists.len() == size).then(|| dists.into_iter().map(|(_, d)| d).collect())
}

/// Adds one source's distances to a Wiener index total.
fn add_row(int_total: &mut i128, sum: &mut RunningSum, float: bool, row: Vec<f64>) {
    for d in row {
        if float {
            sum.add(d);
        } else {
            *int_total += d as i128;
        }
    }
}

/// The Wiener index total of G with `skip` removed (`usize::MAX`: none),
/// or `None` if that graph isn't (strongly) connected. NetworkX sums the
/// distances source by source, each in its search order.
fn wiener_total(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    (float, compensated): (bool, bool),
    skip: usize,
    parallel: bool,
) -> Option<Num> {
    let hidden = without_node(adj, weights, skip);
    let weights = hidden.as_deref();
    let size = if skip < n { n - 1 } else { n };
    let mut int_total = 0i128;
    let mut sum = RunningSum::new(compensated);
    let sources: Vec<u32> = (0..n as u32).filter(|&s| s as usize != skip).collect();
    let mut connected = true;
    if parallel {
        ordered_blocks(
            &sources,
            || (Bfs::new(n), DijkstraState::new(n)),
            |(bfs, dij), s| wiener_row(adj, weights, s as usize, size, skip, bfs, dij),
            |row| match row {
                Some(row) => add_row(&mut int_total, &mut sum, float, row),
                None => connected = false,
            },
        );
    } else {
        let (mut bfs, mut dij) = (Bfs::new(n), DijkstraState::new(n));
        for &s in &sources {
            let row = wiener_row(adj, weights, s as usize, size, skip, &mut bfs, &mut dij)?;
            add_row(&mut int_total, &mut sum, float, row);
        }
    }
    connected.then_some(if float {
        Num::Float(sum.value())
    } else {
        Num::Int(int_total)
    })
}

/// `closeness_vitality`'s Wiener index totals: G's (if `whole`), and G's
/// without each node of `removals`.
pub fn vitality_totals(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    float: bool,
    compensated: bool,
    removals: &[u32],
    whole: bool,
) -> (Option<Num>, Vec<Option<Num>>) {
    let mode = (float, compensated);
    let total = whole
        .then(|| wiener_total(adj, n, weights, mode, usize::MAX, true))
        .flatten();
    let without = if removals.len() > 1 {
        removals
            .par_iter()
            .map(|&r| wiener_total(adj, n, weights, mode, r as usize, false))
            .collect()
    } else {
        removals
            .iter()
            .map(|&r| wiener_total(adj, n, weights, mode, r as usize, true))
            .collect()
    };
    (total, without)
}

/// `flow_hierarchy`'s sums: total weight of the arcs inside strongly
/// connected components, and of all arcs.
pub fn scc_arc_weights(succ: &Csr, n: usize, weights: Option<&[f64]>) -> (i128, i128) {
    let mut comp = vec![0u32; n];
    for (i, c) in strongly_connected_components(succ, n, false)
        .iter()
        .enumerate()
    {
        for &v in c {
            comp[v as usize] = i as u32;
        }
    }
    let (mut inside, mut total) = (0i128, 0i128);
    for u in 0..n {
        for e in succ.range(u) {
            let w = weights.map_or(1.0, |w| w[e]) as i128;
            total += w;
            if comp[succ.targets[e] as usize] == comp[u] {
                inside += w;
            }
        }
    }
    (inside, total)
}

/// Edges (each undirected edge once, self-loops included) whose ends are
/// in the same block and in different blocks; ends in no block (`-1`) are
/// skipped. `None` if `strict` and some edge has such an end.
pub fn block_edge_counts(
    adj: &Csr,
    n: usize,
    directed: bool,
    block: &[i64],
    strict: bool,
) -> Option<(u64, u64)> {
    let (mut same, mut different) = (0u64, 0u64);
    for u in 0..n {
        for &v in adj.neighbors(u) {
            let v = v as usize;
            if !directed && v < u {
                continue;
            }
            if block[u] < 0 || block[v] < 0 {
                if strict {
                    return None;
                }
                continue;
            }
            if block[u] == block[v] {
                same += 1;
            } else {
                different += 1;
            }
        }
    }
    Some((same, different))
}

/// Edges inside each block, summed (blocks may overlap).
pub fn edges_inside_blocks(
    adj: &Csr,
    n: usize,
    directed: bool,
    flat: &[u32],
    ends: &[usize],
) -> u64 {
    let mut member = vec![false; n];
    let mut total = 0u64;
    let mut begin = 0;
    for &end in ends {
        let block = &flat[begin..end];
        for &v in block {
            member[v as usize] = true;
        }
        for &u in block {
            let u = u as usize;
            for &v in adj.neighbors(u) {
                let v = v as usize;
                if member[v] && (directed || v >= u) {
                    total += 1;
                }
            }
        }
        for &v in block {
            member[v as usize] = false;
        }
        begin = end;
    }
    total
}

/// A Python number produced by a sum: an exact int, or a float.
#[derive(Clone, Copy)]
pub enum Num {
    Int(i128),
    Float(f64),
}

/// `modularity`'s sums: the degree total (`m` for directed graphs, the
/// degree sum otherwise) and, per community, `L_c`, the out-degree sum and
/// the in-degree sum, each added in NetworkX's order. Communities are node
/// positions in the iteration order of `set(community)`.
#[allow(clippy::too_many_arguments)]
pub fn modularity_stats(
    succ: &Csr,
    pred: Option<(&Csr, Option<&[f64]>)>,
    n: usize,
    weights: Option<&[f64]>,
    float: bool,
    compensated: bool,
    flat: &[u32],
    ends: &[usize],
) -> (Num, Vec<(Num, Num, Num)>) {
    let directed = pred.is_some();
    let sum = |values: &mut dyn Iterator<Item = Num>| -> Num {
        if float {
            Num::Float(py_sum(
                values.map(|x| match x {
                    Num::Float(f) => f,
                    Num::Int(i) => i as f64,
                }),
                compensated,
            ))
        } else {
            Num::Int(
                values
                    .map(|x| match x {
                        Num::Int(i) => i,
                        Num::Float(f) => f as i128,
                    })
                    .sum(),
            )
        }
    };
    let out_deg: Vec<Num> = if float {
        let w = weights.expect("float weights");
        if directed {
            row_sums_float(succ, n, w, compensated)
        } else {
            degrees_float(succ, n, w, compensated)
        }
        .into_iter()
        .map(Num::Float)
        .collect()
    } else if directed {
        row_sums_int(succ, n, weights)
            .into_iter()
            .map(Num::Int)
            .collect()
    } else {
        degrees_int(succ, n, weights)
            .expect("checked weights")
            .into_iter()
            .map(Num::Int)
            .collect()
    };
    let in_deg: Option<Vec<Num>> = pred.map(|(p, pw)| {
        if float {
            row_sums_float(p, n, pw.expect("float weights"), compensated)
                .into_iter()
                .map(Num::Float)
                .collect()
        } else {
            row_sums_int(p, n, pw).into_iter().map(Num::Int).collect()
        }
    });
    let total = sum(&mut out_deg.iter().copied());
    let mut in_comm = vec![false; n];
    let mut done = vec![false; n];
    let mut per = Vec::with_capacity(ends.len());
    let mut begin = 0;
    for &end in ends {
        let comm = &flat[begin..end];
        for &v in comm {
            in_comm[v as usize] = true;
        }
        // `G.edges(comm)` reports an undirected edge from its first end in
        // `comm`'s order.
        let mut wts = Vec::new();
        for &u in comm {
            let u = u as usize;
            for e in succ.range(u) {
                let v = succ.targets[e] as usize;
                if in_comm[v] && (directed || !done[v]) {
                    wts.push(weights.map_or(1.0, |w| w[e]));
                }
            }
            done[u] = true;
        }
        let l_c = sum(&mut wts.iter().map(|&w| {
            if float {
                Num::Float(w)
            } else {
                Num::Int(w as i128)
            }
        }));
        let out_sum = sum(&mut comm.iter().map(|&u| out_deg[u as usize]));
        let in_sum = match &in_deg {
            Some(d) => sum(&mut comm.iter().map(|&u| d[u as usize])),
            None => out_sum,
        };
        per.push((l_c, out_sum, in_sum));
        for &v in comm {
            in_comm[v as usize] = false;
            done[v as usize] = false;
        }
        begin = end;
    }
    (total, per)
}

// --- greedy_modularity_communities ------------------------------------------------

/// An entry of a `MappedQueue`: priority `-dq`, then the element `(row,
/// col)` compared as Python compares the node tuples (by node rank).
#[derive(Clone, Copy)]
struct Entry {
    p: f64,
    row_rank: u32,
    col_rank: u32,
    row: u32,
    col: u32,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Entry {}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        // `_HeapElement.__lt__`: priorities first (never NaN here), then
        // the elements.
        if self.p == other.p {
            (self.row_rank, self.col_rank).cmp(&(other.row_rank, other.col_rank))
        } else if self.p < other.p {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }
}

/// NetworkX's `MappedQueue` as far as `greedy_modularity_communities` can
/// tell: its minimum is unique (priorities never tie with equal elements),
/// so the heap's layout doesn't matter, only which entries it holds.
#[derive(Default)]
struct Queue {
    set: BTreeSet<Entry>,
    prio: HashMap<(u32, u32), f64>,
}

/// A queue operation NetworkX would fail on (a `KeyError`) or that
/// `Queue` can't represent; rustnx then hands the call back.
struct Bail;

impl Queue {
    fn len(&self) -> usize {
        self.prio.len()
    }

    fn entry(&self, rank: &[u32], row: u32, col: u32, p: f64) -> Entry {
        Entry {
            p,
            row_rank: rank[row as usize],
            col_rank: rank[col as usize],
            row,
            col,
        }
    }

    fn min(&self) -> Option<Entry> {
        self.set.first().copied()
    }

    fn push(&mut self, rank: &[u32], row: u32, col: u32, p: f64) {
        if self.prio.contains_key(&(row, col)) {
            return; // already queued: `push` does nothing
        }
        self.prio.insert((row, col), p);
        self.set.insert(self.entry(rank, row, col, p));
    }

    fn pop(&mut self) -> Option<Entry> {
        let e = self.set.pop_first()?;
        self.prio.remove(&(e.row, e.col));
        Some(e)
    }

    fn remove(&mut self, rank: &[u32], row: u32, col: u32) -> Result<(), Bail> {
        let p = self.prio.remove(&(row, col)).ok_or(Bail)?;
        self.set.remove(&self.entry(rank, row, col, p));
        Ok(())
    }

    /// `update(old, new)`: `new` takes `old`'s place.
    fn update(&mut self, rank: &[u32], old: (u32, u32), new: Entry) -> Result<(), Bail> {
        if (new.row, new.col) != old && self.prio.contains_key(&(new.row, new.col)) {
            return Err(Bail);
        }
        self.remove(rank, old.0, old.1)?;
        self.prio.insert((new.row, new.col), new.p);
        self.set.insert(new);
        Ok(())
    }
}

/// Input of `greedy_modularity`.
pub struct GreedyInput<'a> {
    pub n: usize,
    pub directed: bool,
    /// Edges in `G.edges` order with their weights.
    pub edges: Vec<(u32, u32, f64)>,
    /// `a` and `b` (out- and in-degree fractions; `b` unused if undirected).
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub q0: f64,
    pub resolution: f64,
    pub rank: &'a [u32],
    pub cutoff: f64,
    pub best_n: f64,
}

/// `greedy_modularity_communities` driving
/// `_greedy_modularity_communities_generator`: the merges `(u, v)` (u
/// merged into v) in order, and whether the generator ran out (NetworkX
/// then merges the largest communities itself). `None` where NetworkX would
/// fail or rustnx can't follow.
pub fn greedy_modularity(input: GreedyInput) -> Option<(Vec<(u32, u32)>, bool)> {
    greedy_inner(input).ok()
}

fn greedy_inner(input: GreedyInput) -> Result<(Vec<(u32, u32)>, bool), Bail> {
    let GreedyInput {
        n,
        directed,
        edges,
        mut a,
        mut b,
        q0,
        resolution,
        rank,
        cutoff,
        best_n,
    } = input;
    if !directed {
        b = Vec::new();
    }
    macro_rules! bv {
        ($x:expr) => {
            if directed {
                b[$x as usize]
            } else {
                a[$x as usize]
            }
        };
    }
    let mut dq: Vec<HashMap<u32, f64>> = vec![HashMap::new(); n];
    for &(u, v, wt) in &edges {
        if u == v {
            continue;
        }
        *dq[u as usize].entry(v).or_insert(0.0) += wt;
        *dq[v as usize].entry(u).or_insert(0.0) += wt;
    }
    for u in 0..n {
        let keys: Vec<u32> = dq[u].keys().copied().collect();
        for v in keys {
            let wt = dq[u][&v];
            let value = q0 * wt - resolution * (a[u] * bv!(v) + bv!(u) * a[v as usize]);
            if !value.is_finite() {
                return Err(Bail);
            }
            dq[u].insert(v, value);
        }
    }
    let mut rows: Vec<Queue> = (0..n).map(|_| Queue::default()).collect();
    for (u, row) in rows.iter_mut().enumerate() {
        for (&v, &d) in &dq[u] {
            row.push(rank, u as u32, v, -d);
        }
    }
    let mut h = Queue::default();
    for row in &rows {
        if let Some(e) = row.min() {
            h.push(rank, e.row, e.col, e.p);
        }
    }

    let mut count = n;
    let mut merges = Vec::new();
    loop {
        if (count as f64) <= cutoff {
            return Ok((merges, false));
        }
        if h.len() <= 1 {
            return Ok((merges, true));
        }
        let top = h.pop().ok_or(Bail)?;
        let (u, v) = (top.row, top.col);
        let dq_uv = -top.p;
        if dq_uv < 0.0 && (count as f64) <= best_n {
            return Ok((merges, false));
        }
        // The rest of the generator's loop body, run by the next `next()`.
        let (ui, vi) = (u as usize, v as usize);
        rows[ui].pop();
        if let Some(e) = rows[ui].min() {
            h.push(rank, e.row, e.col, e.p);
        }
        let v_root = rows[vi].min().ok_or(Bail)?;
        if v_root.col == u {
            h.remove(rank, v, u)?;
            rows[vi].remove(rank, v, u)?;
            if let Some(e) = rows[vi].min() {
                h.push(rank, e.row, e.col, e.p);
            }
        } else {
            rows[vi].remove(rank, v, u)?;
        }
        merges.push((u, v));
        count -= 1;

        let u_nbrs: HashSet<u32> = dq[ui].keys().copied().collect();
        let v_nbrs: HashSet<u32> = dq[vi].keys().copied().collect();
        let mut all_nbrs: Vec<u32> = u_nbrs
            .union(&v_nbrs)
            .copied()
            .filter(|&w| w != u && w != v)
            .collect();
        all_nbrs.sort_unstable();
        for w in all_nbrs {
            let wi = w as usize;
            let in_u = u_nbrs.contains(&w);
            let in_v = v_nbrs.contains(&w);
            let dq_vw = if in_u && in_v {
                dq[vi][&w] + dq[ui][&w]
            } else if in_v {
                dq[vi][&w] - resolution * (a[ui] * bv!(w) + a[wi] * bv!(u))
            } else {
                dq[ui][&w] - resolution * (a[vi] * bv!(w) + a[wi] * bv!(v))
            };
            if !dq_vw.is_finite() {
                return Err(Bail);
            }
            for (row, col) in [(v, w), (w, v)] {
                let ri = row as usize;
                dq[ri].insert(col, dq_vw);
                let old_max = rows[ri].min();
                let negdq = -dq_vw;
                if in_v {
                    // `update(d, d, priority)`: the element must be there.
                    rows[ri].remove(rank, row, col)?;
                    rows[ri].push(rank, row, col, negdq);
                } else {
                    rows[ri].push(rank, row, col, negdq);
                }
                match old_max {
                    None => h.push(rank, row, col, negdq),
                    Some(old) => {
                        let new = rows[ri].min().ok_or(Bail)?;
                        if old.col != new.col || old.p != new.p {
                            h.update(rank, (old.row, old.col), new)?;
                        }
                    }
                }
            }
        }

        let u_keys: Vec<u32> = dq[ui].keys().copied().collect();
        for w in u_keys {
            let wi = w as usize;
            dq[wi].remove(&u).ok_or(Bail)?;
            if w != v {
                for (row, col) in [(w, u), (u, w)] {
                    let ri = row as usize;
                    let root = rows[ri].min().ok_or(Bail)?;
                    if root.col == col {
                        rows[ri].remove(rank, row, col)?;
                        h.remove(rank, row, col)?;
                        if let Some(e) = rows[ri].min() {
                            h.push(rank, e.row, e.col, e.p);
                        }
                    } else {
                        rows[ri].remove(rank, row, col)?;
                    }
                }
            }
        }
        dq[ui].clear();
        rows[ui] = Queue::default();
        a[vi] += a[ui];
        a[ui] = 0.0;
        if directed {
            b[vi] += b[ui];
            b[ui] = 0.0;
        }
    }
}

// --- naive_greedy_modularity_communities -------------------------------------------

/// `resolution` as Python multiplies it: an int (exact products) or a float.
#[derive(Clone, Copy)]
pub enum Resolution {
    Int(i128),
    Float(f64),
}

/// One community's term of `modularity` for an undirected graph with
/// integer weights: `L_c / m - resolution * D_c * D_c * norm`.
fn contribution(l: i128, d: i128, m: f64, norm: f64, res: Resolution) -> Option<f64> {
    let expected = match res {
        Resolution::Int(r) => {
            let p = r.checked_mul(d)?.checked_mul(d)?;
            (p as f64) * norm
        }
        Resolution::Float(r) => r * d as f64 * d as f64 * norm,
    };
    Some(l as f64 / m - expected)
}

/// `naive_greedy_modularity_communities` with integer weights (`None`:
/// unit weights): the merges `(i, j)` (community `i` into `j`) in order.
pub fn naive_greedy_modularity(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    m: f64,
    norm: f64,
    res: Resolution,
    compensated: bool,
) -> Option<Vec<(u32, u32)>> {
    let deg = degrees_int(adj, n, weights)?;
    let mut l = vec![0i128; n];
    let mut between: Vec<HashMap<u32, i128>> = vec![HashMap::new(); n];
    for u in 0..n {
        for e in adj.range(u) {
            let v = adj.targets[e] as usize;
            let w = weights.map_or(1.0, |w| w[e]) as i128;
            if v == u {
                l[u] += w;
            } else {
                *between[u].entry(v as u32).or_insert(0) += w;
            }
        }
    }
    let mut d = deg;
    let mut alive: Vec<u32> = (0..n as u32).collect();
    let modularity = |terms: &mut dyn Iterator<Item = f64>| py_sum(terms, compensated);
    let mut contrib: Vec<f64> = (0..n)
        .map(|c| contribution(l[c], d[c], m, norm, res))
        .collect::<Option<_>>()?;
    let mut new_q = modularity(&mut alive.iter().map(|&c| contrib[c as usize]));
    let mut old_q: Option<f64> = None;
    let mut merges = Vec::new();
    while old_q.is_none_or(|old| new_q > old) {
        old_q = Some(new_q);
        let mut to_merge: Option<(u32, u32)> = None;
        for (ai, &i) in alive.iter().enumerate() {
            for &j in &alive[ai + 1..] {
                let lij = l[i as usize]
                    + l[j as usize]
                    + between[i as usize].get(&j).copied().unwrap_or(0);
                let dij = d[i as usize] + d[j as usize];
                let merged = contribution(lij, dij, m, norm, res)?;
                let trial = modularity(&mut alive.iter().filter(|&&c| c != i).map(|&c| {
                    if c == j {
                        merged
                    } else {
                        contrib[c as usize]
                    }
                }));
                if trial >= new_q {
                    if trial > new_q {
                        new_q = trial;
                        to_merge = Some((i, j));
                    } else if let Some((ti, tj)) = to_merge {
                        if i < ti.min(tj) {
                            new_q = trial;
                            to_merge = Some((i, j));
                        }
                    }
                }
            }
        }
        if let Some((i, j)) = to_merge {
            merges.push((i, j));
            let (iu, ju) = (i as usize, j as usize);
            l[ju] += l[iu] + between[iu].get(&j).copied().unwrap_or(0);
            d[ju] += d[iu];
            let moved: Vec<(u32, i128)> = between[iu].drain().collect();
            for (c, w) in moved {
                between[c as usize].remove(&i);
                if c != j {
                    *between[ju].entry(c).or_insert(0) += w;
                    *between[c as usize].entry(j).or_insert(0) += w;
                }
            }
            between[ju].remove(&i);
            l[iu] = 0;
            d[iu] = 0;
            alive.retain(|&c| c != i);
            contrib[ju] = contribution(l[ju], d[ju], m, norm, res)?;
        }
    }
    Some(merges)
}

// --- girvan_newman and edge_betweenness_partition --------------------------------

/// A graph NetworkX edits edge by edge: adjacency lists in dict order.
pub struct EditableGraph {
    n: usize,
    adj: Vec<Vec<(u32, f64)>>,
}

impl EditableGraph {
    /// The undirected graph `G.copy()` (or `G.copy().to_undirected()`)
    /// builds: `add_edges_from` over G's adjacency in order, so a node's
    /// neighbors come in the order their edges were first added.
    pub fn rebuilt(succ: &Csr, n: usize, weights: Option<&[f64]>) -> Self {
        let mut adj: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        let mut slot: HashMap<(u32, u32), (usize, usize)> = HashMap::new();
        for u in 0..n {
            for e in succ.range(u) {
                let v = succ.targets[e];
                let w = weights.map_or(1.0, |w| w[e]);
                let key = ((u as u32).min(v), (u as u32).max(v));
                if let Some(&(i, j)) = slot.get(&key) {
                    // An existing edge's data is updated in place.
                    adj[key.0 as usize][i].1 = w;
                    adj[key.1 as usize][j].1 = w;
                    continue;
                }
                adj[u].push((v, w));
                if v as usize != u {
                    adj[v as usize].push((u as u32, w));
                }
                let (lo, hi) = (key.0 as usize, key.1 as usize);
                slot.insert(key, (adj[lo].len() - 1, adj[hi].len() - 1));
            }
        }
        EditableGraph { n, adj }
    }

    pub fn remove_self_loops(&mut self) {
        for (u, row) in self.adj.iter_mut().enumerate() {
            row.retain(|&(v, _)| v as usize != u);
        }
    }

    fn remove_edge(&mut self, u: u32, v: u32) {
        let (ui, vi) = (u as usize, v as usize);
        if let Some(p) = self.adj[ui].iter().position(|&(x, _)| x == v) {
            self.adj[ui].remove(p);
        }
        if ui != vi {
            if let Some(p) = self.adj[vi].iter().position(|&(x, _)| x == u) {
                self.adj[vi].remove(p);
            }
        }
    }

    pub fn number_of_edges(&self) -> usize {
        self.adj
            .iter()
            .enumerate()
            .map(|(u, row)| row.iter().filter(|&&(v, _)| v as usize >= u).count())
            .sum()
    }

    /// CSR arrays, weights, and each arc's edge in `G.edges()` order.
    #[allow(clippy::type_complexity)]
    fn snapshot(&self) -> (Csr, Vec<f64>, Vec<u32>, Vec<(u32, u32)>) {
        let mut offsets = vec![0usize];
        let mut targets = Vec::new();
        let mut weights = Vec::new();
        let mut edge_of = Vec::new();
        let mut edges = Vec::new();
        let mut ids: HashMap<(u32, u32), u32> = HashMap::new();
        for (u, row) in self.adj.iter().enumerate() {
            for &(v, w) in row {
                targets.push(v);
                weights.push(w);
                let key = ((u as u32).min(v), (u as u32).max(v));
                let id = if v as usize >= u {
                    let id = edges.len() as u32;
                    edges.push((u as u32, v));
                    ids.insert(key, id);
                    id
                } else {
                    UNSET // filled below: the edge comes later in G.edges()
                };
                edge_of.push(id);
            }
            offsets.push(targets.len());
        }
        for u in 0..self.n {
            for e in offsets[u]..offsets[u + 1] {
                if edge_of[e] == UNSET {
                    let v = targets[e];
                    edge_of[e] = ids[&((u as u32).min(v), (u as u32).max(v))];
                }
            }
        }
        (Csr { offsets, targets }, weights, edge_of, edges)
    }

    /// `nx.connected_components`: BFS from each unseen node in order.
    pub fn components(&self) -> Vec<Vec<u32>> {
        let mut seen = vec![false; self.n];
        let mut comps = Vec::new();
        for s in 0..self.n {
            if seen[s] {
                continue;
            }
            seen[s] = true;
            let mut comp = vec![s as u32];
            let mut head = 0;
            while head < comp.len() {
                let v = comp[head] as usize;
                head += 1;
                for &(w, _) in &self.adj[v] {
                    if !seen[w as usize] {
                        seen[w as usize] = true;
                        comp.push(w);
                    }
                }
            }
            comps.push(comp);
        }
        comps
    }

    fn number_of_components(&self) -> usize {
        self.components().len()
    }

    /// The edge `max(betweenness, key=betweenness.get)` picks from
    /// `edge_betweenness_centrality(G, weight=...)`, accumulated exactly as
    /// NetworkX does (sources in order) and rescaled by `scale`.
    fn most_central_edge(&self, weighted: bool, scale: Option<f64>) -> (u32, u32) {
        let (csr, weights, edge_of, edges) = self.snapshot();
        let n = self.n;
        let w = weighted.then_some(weights.as_slice());
        let mut b = vec![0.0f64; edges.len()];
        let sources: Vec<u32> = (0..n as u32).collect();
        let _ = in_source_order(
            &sources,
            16 * n,
            || Spt::new(n),
            |st, s| {
                st.run(&csr, w, s);
                // `_accumulate_edges`
                let mut log = Vec::with_capacity(st.order.len());
                for i in (0..st.order.len()).rev() {
                    let x = st.order[i] as usize;
                    let coeff = (1.0 + st.delta[x]) / st.sigma[x];
                    for (&v, &arc) in st.preds[x].iter().zip(&st.pred_arcs[x]) {
                        let v = v as usize;
                        let c = st.sigma[v] * coeff;
                        log.push((edge_of[arc as usize], c));
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
        if let Some(scale) = scale {
            for x in &mut b {
                *x *= scale;
            }
        }
        let mut best = 0;
        for (i, &x) in b.iter().enumerate() {
            if x > b[best] {
                best = i;
            }
        }
        edges[best]
    }

    /// `_without_most_central_edges` with the default edge betweenness.
    pub fn girvan_newman_step(&mut self, scale: Option<f64>) -> Vec<Vec<u32>> {
        let original = self.number_of_components();
        loop {
            let (u, v) = self.most_central_edge(false, scale);
            self.remove_edge(u, v);
            let comps = self.components();
            if comps.len() > original {
                return comps;
            }
        }
    }

    /// `edge_betweenness_partition`'s loop.
    pub fn betweenness_partition(
        &mut self,
        number_of_sets: usize,
        weighted: bool,
        scale: Option<f64>,
    ) -> Vec<Vec<u32>> {
        let mut partition = self.components();
        while partition.len() < number_of_sets {
            let (u, v) = self.most_central_edge(weighted, scale);
            self.remove_edge(u, v);
            partition = self.components();
        }
        partition
    }
}

// --- Label propagation with CPython's random numbers --------------------------------

/// CPython's Mersenne Twister (`random.Random`), from its `getstate()`.
pub struct Mt19937 {
    mt: [u32; 624],
    index: usize,
}

impl Mt19937 {
    /// From the 625 ints of `getstate()[1]`.
    pub fn from_state(state: &[u32]) -> Option<Self> {
        if state.len() != 625 || state[624] > 624 {
            return None;
        }
        let mut mt = [0u32; 624];
        mt.copy_from_slice(&state[..624]);
        Some(Mt19937 {
            mt,
            index: state[624] as usize,
        })
    }

    pub fn state(&self) -> Vec<u32> {
        let mut s = self.mt.to_vec();
        s.push(self.index as u32);
        s
    }

    fn genrand_uint32(&mut self) -> u32 {
        const N: usize = 624;
        const M: usize = 397;
        const MATRIX_A: u32 = 0x9908_b0df;
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        let mag01 = [0u32, MATRIX_A];
        if self.index >= N {
            let mt = &mut self.mt;
            for kk in 0..N - M {
                let y = (mt[kk] & UPPER) | (mt[kk + 1] & LOWER);
                mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            for kk in N - M..N - 1 {
                let y = (mt[kk] & UPPER) | (mt[kk + 1] & LOWER);
                mt[kk] = mt[kk + M - N] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            let y = (mt[N - 1] & UPPER) | (mt[0] & LOWER);
            mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `Random._randbelow_with_getrandbits(n)` for `1 <= n < 2^32`.
    fn randbelow(&mut self, n: usize) -> usize {
        let k = usize::BITS - n.leading_zeros();
        loop {
            let r = (self.genrand_uint32() >> (32 - k)) as usize;
            if r < n {
                return r;
            }
        }
    }

    /// `Random.shuffle`.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.randbelow(i + 1);
            x.swap(i, j);
        }
    }

    /// `Random.choice` (non-empty `seq`).
    fn choice<T: Copy>(&mut self, seq: &[T]) -> T {
        seq[self.randbelow(seq.len())]
    }
}

/// Label frequencies in first-seen order (`Counter` / `defaultdict` order).
struct Freq {
    slot: Vec<u32>,
    labels: Vec<u32>,
    counts: Vec<f64>,
}

impl Freq {
    fn new(n: usize) -> Self {
        Freq {
            slot: vec![UNSET; n],
            labels: Vec::new(),
            counts: Vec::new(),
        }
    }

    fn add(&mut self, label: u32, x: f64) {
        let s = self.slot[label as usize];
        if s == UNSET {
            self.slot[label as usize] = self.labels.len() as u32;
            self.labels.push(label);
            self.counts.push(0.0 + x);
        } else {
            self.counts[s as usize] += x;
        }
    }

    /// The labels whose frequency equals the maximum, in order.
    fn best(&self) -> Vec<u32> {
        let max = self
            .counts
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        self.labels
            .iter()
            .zip(&self.counts)
            .filter(|&(_, &c)| c == max)
            .map(|(&l, _)| l)
            .collect()
    }

    fn clear(&mut self) {
        for &l in &self.labels {
            self.slot[l as usize] = UNSET;
        }
        self.labels.clear();
        self.counts.clear();
    }
}

/// `asyn_lpa_communities`: each node's final label. Unweighted counts are
/// small integers, so f64 counts compare exactly like Python's ints.
pub fn asyn_lpa(succ: &Csr, n: usize, weights: Option<&[f64]>, rng: &mut Mt19937) -> Vec<u32> {
    let mut labels: Vec<u32> = (0..n as u32).collect();
    let mut freq = Freq::new(n);
    loop {
        let mut cont = false;
        let mut nodes: Vec<u32> = (0..n as u32).collect();
        rng.shuffle(&mut nodes);
        for &node in &nodes {
            let node = node as usize;
            if succ.range(node).is_empty() {
                continue;
            }
            for e in succ.range(node) {
                let v = succ.targets[e] as usize;
                freq.add(labels[v], weights.map_or(1.0, |w| w[e]));
            }
            let best = freq.best();
            freq.clear();
            if !best.contains(&labels[node]) {
                labels[node] = rng.choice(&best);
                cont = true;
            }
        }
        if !cont {
            return labels;
        }
    }
}

/// `fast_label_propagation_communities`: each node's final label.
/// `pred` (directed graphs) is in NetworkX's `G._pred` order:
/// `all_neighbors` lists predecessors first.
pub fn fast_label_propagation(
    succ: &Csr,
    pred: Option<(&Csr, Option<&[f64]>)>,
    n: usize,
    weights: Option<&[f64]>,
    rng: &mut Mt19937,
) -> Vec<u32> {
    let mut queue: Vec<u32> = (0..n as u32).collect();
    rng.shuffle(&mut queue);
    let mut queue: std::collections::VecDeque<u32> = queue.into();
    let mut in_queue = vec![true; n];
    let mut comms: Vec<u32> = (0..n as u32).collect();
    let mut freq = Freq::new(n);
    let mut nbrs: Vec<u32> = Vec::new();
    while let Some(node) = queue.pop_front() {
        let node = node as usize;
        in_queue[node] = false;
        // `all_neighbors`: predecessors, then successors.
        nbrs.clear();
        if let Some((p, _)) = pred {
            nbrs.extend_from_slice(p.neighbors(node));
        }
        nbrs.extend_from_slice(succ.neighbors(node));
        if nbrs.is_empty() {
            continue; // isolated: `G.degree(node) == 0`
        }
        match weights {
            None => {
                for &v in &nbrs {
                    freq.add(comms[v as usize], 1.0);
                }
            }
            Some(w) => {
                // `G.edges(node)`, then `G.in_edges(node)` if directed.
                for e in succ.range(node) {
                    freq.add(comms[succ.targets[e] as usize], w[e]);
                }
                if let Some((p, pw)) = pred {
                    let pw = pw.expect("weights given");
                    for e in p.range(node) {
                        freq.add(comms[p.targets[e] as usize], pw[e]);
                    }
                }
            }
        }
        let best = freq.best();
        freq.clear();
        let comm = rng.choice(&best);
        if comms[node] != comm {
            comms[node] = comm;
            for &v in &nbrs {
                let vi = v as usize;
                if comms[vi] != comm && !in_queue[vi] {
                    queue.push_back(v);
                    in_queue[vi] = true;
                }
            }
        }
    }
    comms
}

/// `asyn_fluidc`: each node's community (`UNSET`: none) and the order in
/// which nodes entered NetworkX's `communities` dict. `legacy`: NetworkX
/// before 3.6, which checks `iter_count > max_iter` after each round
/// instead of looping while `iter_count < max_iter`. `None` if a community
/// would empty out (NetworkX would divide by zero).
pub fn asyn_fluidc(
    adj: &Csr,
    n: usize,
    k: usize,
    max_iter: i64,
    legacy: bool,
    rng: &mut Mt19937,
) -> Option<(Vec<u32>, Vec<u32>)> {
    let mut vertices: Vec<u32> = (0..n as u32).collect();
    rng.shuffle(&mut vertices);
    let mut com = vec![UNSET; n];
    let mut order: Vec<u32> = Vec::with_capacity(n);
    let mut density = vec![1.0f64; k];
    let mut count = vec![1i64; k];
    for (i, &v) in vertices[..k].iter().enumerate() {
        com[v as usize] = i as u32;
        order.push(v);
    }
    let mut freq = Freq::new(k);
    let mut iter_count = 0i64;
    let mut cont = true;
    while cont && (legacy || iter_count < max_iter) {
        cont = false;
        iter_count += 1;
        let mut vertices: Vec<u32> = (0..n as u32).collect();
        rng.shuffle(&mut vertices);
        for &vertex in &vertices {
            let vertex = vertex as usize;
            let own = com[vertex];
            if own != UNSET {
                freq.add(own, density[own as usize]);
            }
            for &v in adj.neighbors(vertex) {
                let c = com[v as usize];
                if c != UNSET {
                    freq.add(c, density[c as usize]);
                }
            }
            if freq.labels.is_empty() {
                continue;
            }
            let max = freq
                .counts
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let best: Vec<u32> = freq
                .labels
                .iter()
                .zip(&freq.counts)
                .filter(|&(_, &f)| (max - f) < 0.0001)
                .map(|(&c, _)| c)
                .collect();
            freq.clear();
            if own != UNSET && best.contains(&own) {
                continue;
            }
            cont = true;
            let new = rng.choice(&best);
            if own != UNSET {
                count[own as usize] -= 1;
                if count[own as usize] == 0 {
                    return None;
                }
                density[own as usize] = 1.0 / count[own as usize] as f64;
            } else {
                order.push(vertex as u32);
            }
            com[vertex] = new;
            count[new as usize] += 1;
            density[new as usize] = 1.0 / count[new as usize] as f64;
        }
        if legacy && iter_count > max_iter {
            break;
        }
    }
    Some((com, order))
}

/// `overlapping_modularity`'s sums: the degree total and, per community
/// (positions in `set(community)` order), the overlap-discounted edge
/// weight and degree sums. `membership[v]`: how many communities hold v.
#[allow(clippy::too_many_arguments)]
pub fn overlap_stats(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    float: bool,
    compensated: bool,
    membership: &[u64],
    flat: &[u32],
    ends: &[usize],
) -> (Num, Vec<(f64, f64)>) {
    let deg: Vec<f64> = if float {
        degrees_float(adj, n, weights.expect("float weights"), compensated)
    } else {
        degrees_int(adj, n, weights)
            .expect("checked weights")
            .into_iter()
            .map(|d| d as f64)
            .collect()
    };
    let total = if float {
        Num::Float(py_sum(deg.iter().copied(), compensated))
    } else {
        Num::Int(deg.iter().map(|&d| d as i128).sum())
    };
    let mut in_comm = vec![false; n];
    let mut done = vec![false; n];
    let mut per = Vec::with_capacity(ends.len());
    let mut begin = 0;
    for &end in ends {
        let comm = &flat[begin..end];
        for &v in comm {
            in_comm[v as usize] = true;
        }
        let mut terms = Vec::new();
        for &u in comm {
            let u = u as usize;
            for e in adj.range(u) {
                let v = adj.targets[e] as usize;
                if in_comm[v] && !done[v] {
                    let wt = weights.map_or(1.0, |w| w[e]);
                    terms.push(wt / (membership[u] * membership[v]) as f64);
                }
            }
            done[u] = true;
        }
        let l = py_sum(terms.into_iter(), compensated);
        let k = py_sum(
            comm.iter()
                .map(|&u| deg[u as usize] / membership[u as usize] as f64),
            compensated,
        );
        per.push((l, k));
        for &v in comm {
            in_comm[v as usize] = false;
            done[v as usize] = false;
        }
        begin = end;
    }
    (total, per)
}
