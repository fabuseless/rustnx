//! Batch 7: the shortest path, DAG and simple path leftovers (Floyd-Warshall,
//! Johnson, Goldberg-Radzik, antichains, simple paths, Yen's k shortest
//! paths, minimum cycle basis), ported from NetworkX step by step so that
//! values, ties and dict orders match.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use rayon::prelude::*;

use super::paths::{bidirectional_dijkstra, BidirectionalError, NO_PARENT};
use super::shortest_paths_more::dijkstra_forest;
use super::spectral::py_sum;
use super::traversal::NegativeCycle;
use crate::graph::Csr;

const NONE: u32 = u32::MAX;
/// 2^53: ints of at most this size add exactly as f64.
const MAX_EXACT: f64 = 9_007_199_254_740_992.0;

fn copy_csr(adj: &Csr) -> Csr {
    Csr {
        offsets: adj.offsets.clone(),
        targets: adj.targets.clone(),
    }
}

/// Whether integer path sums stay exact in f64: the absolute weights add up
/// to less than 2^53 (NaN, hidden, weights are skipped).
pub fn int_sums_exact(weights: Option<&[f64]>) -> bool {
    let Some(w) = weights else { return true };
    let total: f64 = w.iter().filter(|x| !x.is_nan()).map(|x| x.abs()).sum();
    total < MAX_EXACT
}

/// Whether every (non-hidden) weight is finite.
pub fn weights_finite(weights: Option<&[f64]>) -> bool {
    weights.is_none_or(|w| w.iter().all(|x| x.is_nan() || x.is_finite()))
}

/// Whether some weight is -0.0 (NumPy's `minimum` may pick either zero).
pub fn has_negative_zero(weights: Option<&[f64]>) -> bool {
    weights.is_some_and(|w| w.iter().any(|&x| x == 0.0 && x.is_sign_negative()))
}

// --- Floyd-Warshall ---------------------------------------------------------------

pub enum FwError {
    /// NetworkX raises `NetworkXUnbounded("Negative cycle detected.")`.
    Negative,
    /// rustnx can't reproduce this case (huge ints, a `KeyError`, ...).
    Unsupported,
}

/// NetworkX's `dist` (defaultdict of defaultdicts) and `pred` (dict of
/// dicts) as `n x n` matrices plus each dict's key order.
pub struct FloydWarshall {
    pub n: usize,
    pub dist: Vec<f64>,
    pub is_int: Vec<bool>,
    present: Vec<bool>,
    pub dist_order: Vec<Vec<u32>>,
    pub pred: Vec<u32>,
    pub pred_order: Vec<Vec<u32>>,
    /// Keys of the outer `pred` dict, in order.
    pub pred_rows: Vec<u32>,
    pred_row_exists: Vec<bool>,
}

impl FloydWarshall {
    fn new(n: usize) -> Self {
        FloydWarshall {
            n,
            dist: vec![f64::INFINITY; n * n],
            is_int: vec![false; n * n],
            present: vec![false; n * n],
            dist_order: vec![Vec::new(); n],
            pred: vec![NONE; n * n],
            pred_order: vec![Vec::new(); n],
            pred_rows: Vec::new(),
            pred_row_exists: vec![false; n],
        }
    }

    /// `dist[u][v]` read through the defaultdict: adds the key (inf).
    fn touch(&mut self, u: usize, v: usize) -> (f64, bool) {
        let k = u * self.n + v;
        if !self.present[k] {
            self.present[k] = true;
            self.dist_order[u].push(v as u32);
        }
        (self.dist[k], self.is_int[k])
    }

    fn set_dist(&mut self, u: usize, v: usize, x: f64, int: bool) {
        self.touch(u, v);
        let k = u * self.n + v;
        self.dist[k] = x;
        self.is_int[k] = int;
    }

    fn set_pred(&mut self, u: usize, v: usize, p: u32) {
        if !self.pred_row_exists[u] {
            self.pred_row_exists[u] = true;
            self.pred_rows.push(u as u32);
        }
        let k = u * self.n + v;
        if self.pred[k] == NONE {
            self.pred_order[u].push(v as u32);
        }
        self.pred[k] = p;
    }

    /// `dist[u][v] = min(x, dist[u][v])` (Python's `min` keeps the first of
    /// equal values).
    fn set_min(&mut self, u: usize, v: usize, x: f64, int: bool) {
        let (cur, cur_int) = self.touch(u, v);
        if cur < x {
            self.set_dist(u, v, cur, cur_int);
        } else {
            self.set_dist(u, v, x, int);
        }
    }

    pub fn value(&self, u: usize, v: usize) -> (f64, bool) {
        let k = u * self.n + v;
        (self.dist[k], self.is_int[k])
    }
}

/// `dist[u][w] + dist[w][v]` with its type; `None` if ints could overflow
/// f64's exact range.
#[inline]
fn add(a: f64, a_int: bool, b: f64, b_int: bool) -> Option<(f64, bool)> {
    let int = a_int && b_int;
    let d = a + b;
    if int && d.abs() >= MAX_EXACT {
        return None;
    }
    Some((d, int))
}

/// The initial `pred` and `dist`. `old`: NetworkX 3.4/3.5's inline version
/// (`min(e_weight, dist[u][v])` over `G.edges(data=True)`); otherwise 3.6+'s
/// `_init_pred_dist` (rows of `G._adj`, hidden edges skipped, a negative
/// self-loop raises). Every weight has type int when `int_weights`.
fn fw_init(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    int_weights: bool,
    directed: bool,
    old: bool,
) -> Result<FloydWarshall, FwError> {
    let mut fw = FloydWarshall::new(n);
    if old {
        for u in 0..n {
            fw.set_dist(u, u, 0.0, true);
        }
        for u in 0..n {
            for e in adj.range(u) {
                let v = adj.targets[e] as usize;
                if !directed && v < u {
                    continue; // `G.edges` lists each undirected edge once
                }
                let x = weights.map_or(1.0, |w| w[e]);
                fw.set_min(u, v, x, int_weights);
                fw.set_pred(u, v, u as u32);
                if !directed {
                    fw.set_min(v, u, x, int_weights);
                    fw.set_pred(v, u, v as u32);
                }
            }
        }
        return Ok(fw);
    }
    for u in 0..n {
        for e in adj.range(u) {
            let v = adj.targets[e] as usize;
            let x = weights.map_or(1.0, |w| w[e]);
            if x.is_nan() {
                continue;
            }
            fw.set_dist(u, v, x, int_weights);
            fw.set_pred(u, v, u as u32);
        }
        if fw.touch(u, u).0 < 0.0 {
            return Err(FwError::Negative);
        }
        fw.set_dist(u, u, 0.0, true);
    }
    Ok(fw)
}

/// `floyd_warshall_predecessor_and_distance`. `check_negative`: NetworkX
/// 3.6+ raises if a diagonal entry ends up negative.
pub fn floyd_warshall(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    int_weights: bool,
    directed: bool,
    old: bool,
) -> Result<FloydWarshall, FwError> {
    let mut fw = fw_init(adj, n, weights, int_weights, directed, old)?;
    // The first pass (w = first node) reads every `dist[u][v]` through the
    // defaultdict, so each row gets its missing keys in node order.
    for u in 0..n {
        for v in 0..n {
            fw.touch(u, v);
        }
    }
    for w in 0..n {
        let (dww, _) = fw.value(w, w);
        if dww < 0.0 {
            fw_pass_sequential(&mut fw, w)?;
        } else {
            fw_pass_parallel(&mut fw, w)?;
        }
    }
    if !old && (0..n).any(|u| fw.value(u, u).0 < 0.0) {
        return Err(FwError::Negative);
    }
    Ok(fw)
}

/// One pass of the triple loop, with every value read live (row `w`
/// changes during the pass when `dist[w][w] < 0`).
fn fw_pass_sequential(fw: &mut FloydWarshall, w: usize) -> Result<(), FwError> {
    let n = fw.n;
    for u in 0..n {
        for v in 0..n {
            let (a, a_int) = fw.value(u, w);
            let (b, b_int) = fw.value(w, v);
            let (d, d_int) = add(a, a_int, b, b_int).ok_or(FwError::Unsupported)?;
            if fw.value(u, v).0 > d {
                let k = u * n + v;
                fw.dist[k] = d;
                fw.is_int[k] = d_int;
                let p = fw.pred[w * n + v];
                if p == NONE || !fw.pred_row_exists[u] {
                    return Err(FwError::Unsupported);
                }
                fw.set_pred(u, v, p);
            }
        }
    }
    Ok(())
}

/// One pass when row `w` can't change (`dist[w][w] >= 0`): rows are
/// independent, so they run in parallel.
fn fw_pass_parallel(fw: &mut FloydWarshall, w: usize) -> Result<(), FwError> {
    let n = fw.n;
    let row_w: Vec<f64> = fw.dist[w * n..(w + 1) * n].to_vec();
    let int_w: Vec<bool> = fw.is_int[w * n..(w + 1) * n].to_vec();
    let pred_w: Vec<u32> = fw.pred[w * n..(w + 1) * n].to_vec();
    let exists = &fw.pred_row_exists;
    fw.dist
        .par_chunks_mut(n)
        .zip(fw.is_int.par_chunks_mut(n))
        .zip(fw.pred.par_chunks_mut(n))
        .zip(fw.pred_order.par_iter_mut())
        .enumerate()
        .try_for_each(|(u, (((dist, is_int), pred), order))| {
            let (a, a_int) = (dist[w], is_int[w]);
            if a == f64::INFINITY {
                return Ok(()); // inf + x is never below anything
            }
            for v in 0..n {
                let (d, d_int) = add(a, a_int, row_w[v], int_w[v]).ok_or(FwError::Unsupported)?;
                if dist[v] > d {
                    dist[v] = d;
                    is_int[v] = d_int;
                    let p = pred_w[v];
                    if p == NONE || !exists[u] {
                        return Err(FwError::Unsupported);
                    }
                    if pred[v] == NONE {
                        order.push(v as u32);
                    }
                    pred[v] = p;
                }
            }
            Ok(())
        })
}

/// NetworkX 3.7's `floyd_warshall_tree`: for each `w`, walk the shortest
/// path tree of `w` in DFS preorder, skipping subtrees that don't improve.
pub fn floyd_warshall_tree(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    int_weights: bool,
) -> Result<FloydWarshall, FwError> {
    let mut fw = fw_init(adj, n, weights, int_weights, true, false)?;
    let mut children: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut dfs_next = vec![NONE; n];
    let mut skip = vec![NONE; n];
    let rows = fw.pred_rows.clone();
    for &w in &rows {
        let w = w as usize;
        for c in children.iter_mut() {
            c.clear();
        }
        for &v in &fw.pred_order[w] {
            if v as usize == w {
                continue;
            }
            children[fw.pred[w * n + v as usize] as usize].push(v);
        }
        dfs_next.fill(NONE);
        skip.fill(NONE);
        // `dfs_dict[None]` is `w`; `node` holds the previous popped node.
        let mut stack = vec![w as u32];
        let mut node = NONE;
        let mut pops = 0usize;
        while let Some(next) = stack.pop() {
            pops += 1;
            if pops > n + 1 {
                // A cycle among the parents: NetworkX would never finish.
                return Err(FwError::Unsupported);
            }
            if node != NONE {
                dfs_next[node as usize] = next;
            }
            if let Some(&top) = stack.last() {
                skip[next as usize] = top;
            }
            stack.extend_from_slice(&children[next as usize]);
            node = next;
        }
        for u in 0..n {
            if u == w {
                continue;
            }
            let (duw, duw_int) = fw.touch(u, w);
            if duw == f64::INFINITY {
                continue;
            }
            let mut v = dfs_next[w];
            while v != NONE {
                let vi = v as usize;
                let (dwv, dwv_int) = fw.touch(w, vi);
                let (d, d_int) = add(duw, duw_int, dwv, dwv_int).ok_or(FwError::Unsupported)?;
                if fw.touch(u, vi).0 > d {
                    fw.set_dist(u, vi, d, d_int);
                    let p = fw.pred[w * n + vi];
                    if p == NONE || !fw.pred_row_exists[u] {
                        return Err(FwError::Unsupported);
                    }
                    fw.set_pred(u, vi, p);
                    v = dfs_next[vi];
                } else {
                    v = skip[vi];
                }
            }
        }
    }
    if (0..n).any(|u| fw.value(u, u).0 < 0.0) {
        return Err(FwError::Negative);
    }
    Ok(fw)
}

/// `floyd_warshall_numpy`'s loop on a row-major matrix:
/// `A = np.minimum(A, A[i, :][None, :] + A[:, i][:, None])` for each `i`.
/// `check_negative` (NetworkX 3.6+) checks the diagonal before and after.
/// `Err(())` where NetworkX raises `NetworkXUnbounded`.
pub fn floyd_warshall_dense(a: &mut [f64], n: usize, check_negative: bool) -> Result<(), ()> {
    let negative_diag = |a: &[f64]| (0..n).any(|i| a[i * n + i] < 0.0);
    if check_negative && negative_diag(a) {
        return Err(());
    }
    for i in 0..n {
        a[i * n + i] = 0.0;
    }
    for i in 0..n {
        let row: Vec<f64> = a[i * n..(i + 1) * n].to_vec();
        let col: Vec<f64> = (0..n).map(|u| a[u * n + i]).collect();
        let row_has_neg_inf = row.contains(&f64::NEG_INFINITY);
        a.par_chunks_mut(n).enumerate().for_each(|(u, out)| {
            let c = col[u];
            if c == f64::INFINITY && !row_has_neg_inf {
                return; // every sum is inf: nothing changes
            }
            for v in 0..n {
                let b = row[v] + c;
                let x = out[v];
                // NumPy's minimum: NaN propagates, else the smaller value.
                out[v] = if x <= b || x.is_nan() { x } else { b };
            }
        });
    }
    if check_negative && negative_diag(a) {
        return Err(());
    }
    Ok(())
}

// --- Johnson -------------------------------------------------------------------------

/// `_inner_bellman_ford` from every node at once (`johnson`'s call, with
/// `dist` all 0 and `pred` all empty, heuristic on): the distances, or
/// `None` on a negative cycle. No weight may be hidden.
pub fn bellman_ford_all(adj: &Csr, n: usize, weights: Option<&[f64]>) -> Option<Vec<f64>> {
    let mut dist = vec![0.0f64; n];
    let mut pred: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut pred_edge = vec![NONE; n];
    let mut recent = vec![(NONE, NONE); n];
    let mut count = vec![0usize; n];
    let mut in_q = vec![true; n];
    let mut q: VecDeque<u32> = (0..n as u32).collect();
    while let Some(u) = q.pop_front() {
        let u = u as usize;
        in_q[u] = false;
        if pred[u].iter().any(|&p| in_q[p as usize]) {
            continue;
        }
        let dist_u = dist[u];
        for e in adj.range(u) {
            let v = adj.targets[e] as usize;
            let dist_v = dist_u + weights.map_or(1.0, |w| w[e]);
            if dist_v < dist[v] {
                let (a, b) = recent[u];
                if a == v as u32 || b == v as u32 {
                    return None;
                }
                recent[v] = if pred_edge[v] == u as u32 {
                    recent[u]
                } else {
                    (u as u32, v as u32)
                };
                if !in_q[v] {
                    q.push_back(v as u32);
                    in_q[v] = true;
                    count[v] += 1;
                    if count[v] == n {
                        return None;
                    }
                }
                dist[v] = dist_v;
                pred[v].clear();
                pred[v].push(u as u32);
                pred_edge[v] = u as u32;
            } else if dist_v == dist[v] {
                pred[v].push(u as u32);
            }
        }
    }
    Some(dist)
}

/// `johnson`'s reweighted edges: `weight(u, v, d) + h[u] - h[v]`.
pub fn reweight(adj: &Csr, n: usize, weights: Option<&[f64]>, h: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(adj.targets.len());
    for u in 0..n {
        for e in adj.range(u) {
            let v = adj.targets[e] as usize;
            out.push((weights.map_or(1.0, |w| w[e]) + h[u]) - h[v]);
        }
    }
    out
}

// --- Goldberg-Radzik -----------------------------------------------------------------

pub enum GrError {
    Negative,
    Unsupported,
}

/// `goldberg_radzik`'s state between rounds; Python drives the rounds
/// because each one iterates a Python set (`relabeled`) in hash order.
pub struct GoldbergRadzik {
    adj: Csr,
    weights: Option<Vec<f64>>,
    int_weights: bool,
    pub d: Vec<f64>,
    pub d_int: Vec<bool>,
    /// `pred[v]` (`NONE` = not a key, `source` maps to `NO_PARENT` = None).
    pub pred: Vec<u32>,
    pub pred_order: Vec<u32>,
    to_scan: Vec<u32>,
    neg_count: Vec<i64>,
    stamp: Vec<u64>,
    round: u64,
    in_stack: Vec<bool>,
}

impl GoldbergRadzik {
    pub fn new(
        adj: &Csr,
        n: usize,
        weights: Option<&[f64]>,
        int_weights: bool,
        source: usize,
    ) -> Self {
        let mut d = vec![f64::INFINITY; n];
        let mut d_int = vec![false; n];
        d[source] = 0.0;
        d_int[source] = true;
        let mut pred = vec![NONE; n];
        pred[source] = NO_PARENT;
        GoldbergRadzik {
            adj: copy_csr(adj),
            weights: weights.map(|w| w.to_vec()),
            int_weights,
            d,
            d_int,
            pred,
            pred_order: vec![source as u32],
            to_scan: Vec::new(),
            neg_count: vec![0; n],
            stamp: vec![0; n],
            round: 0,
            in_stack: vec![false; n],
        }
    }

    fn weight(&self, e: usize) -> f64 {
        self.weights.as_ref().map_or(1.0, |w| w[e])
    }

    fn set(&mut self, v: usize, t: f64, u: usize) -> Result<(), GrError> {
        if self.int_weights && t.abs() >= MAX_EXACT {
            return Err(GrError::Unsupported);
        }
        self.d[v] = t;
        self.d_int[v] = self.int_weights;
        if self.pred[v] == NONE {
            self.pred_order.push(v as u32);
        }
        self.pred[v] = u as u32;
        Ok(())
    }

    /// `topo_sort(relabeled)` with `relabeled` iterated in `order`.
    /// `skip_counted`: NetworkX 3.4/3.5 skip nodes already counted; 3.6+
    /// iterate a copy of `relabeled` made before any were.
    pub fn topo_sort(&mut self, order: &[u32], skip_counted: bool) -> Result<(), GrError> {
        self.round += 1;
        let round = self.round;
        self.to_scan.clear();
        for &u0 in order {
            let u0 = u0 as usize;
            if skip_counted && self.stamp[u0] == round {
                continue;
            }
            let d_u = self.d[u0];
            let all_ok = self.adj.range(u0).all(|e| {
                let v = self.adj.targets[e] as usize;
                d_u + self.weight(e) >= self.d[v]
            });
            if all_ok {
                continue;
            }
            let mut stack: Vec<(u32, usize)> = vec![(u0 as u32, self.adj.offsets[u0])];
            self.in_stack[u0] = true;
            self.stamp[u0] = round;
            self.neg_count[u0] = 0;
            while let Some(&(u, pos)) = stack.last() {
                let u = u as usize;
                if pos == self.adj.offsets[u + 1] {
                    self.to_scan.push(u as u32);
                    stack.pop();
                    self.in_stack[u] = false;
                    continue;
                }
                stack.last_mut().expect("non-empty").1 += 1;
                let v = self.adj.targets[pos] as usize;
                let t = self.d[u] + self.weight(pos);
                if t < self.d[v] {
                    self.set(v, t, u)?;
                    if self.stamp[v] != round {
                        self.stamp[v] = round;
                        self.neg_count[v] = self.neg_count[u] + 1;
                        stack.push((v as u32, self.adj.offsets[v]));
                        self.in_stack[v] = true;
                    } else if self.in_stack[v] && self.neg_count[u] + 1 > self.neg_count[v] {
                        for &(x, _) in &stack {
                            self.in_stack[x as usize] = false;
                        }
                        return Err(GrError::Negative);
                    }
                }
            }
        }
        self.to_scan.reverse();
        Ok(())
    }

    /// `relax(to_scan)`: the nodes added to `relabeled`, in order (repeats
    /// included).
    pub fn relax(&mut self) -> Result<Vec<u32>, GrError> {
        let mut added = Vec::new();
        let to_scan = std::mem::take(&mut self.to_scan);
        for &u in &to_scan {
            let u = u as usize;
            let d_u = self.d[u];
            for e in self.adj.range(u) {
                let v = self.adj.targets[e] as usize;
                let t = d_u + self.weight(e);
                if t < self.d[v] {
                    self.set(v, t, u)?;
                    added.push(v as u32);
                }
            }
        }
        self.to_scan = to_scan;
        Ok(added)
    }
}

// --- Antichains -------------------------------------------------------------------

/// Reachability as one bitset row per node.
pub struct Reach {
    words: usize,
    bits: Vec<u64>,
}

impl Reach {
    /// From a topological order of a DAG.
    pub fn new(adj: &Csr, n: usize, topo: &[u32]) -> Self {
        let words = n.div_ceil(64);
        let mut bits = vec![0u64; n * words];
        for &v in topo.iter().rev() {
            let v = v as usize;
            for &w in adj.neighbors(v) {
                let w = w as usize;
                if w == v {
                    continue;
                }
                let (lo, hi) = if v < w {
                    let (a, b) = bits.split_at_mut(w * words);
                    (&mut a[v * words..(v + 1) * words], &b[..words])
                } else {
                    let (a, b) = bits.split_at_mut(v * words);
                    (&mut b[..words], &a[w * words..(w + 1) * words])
                };
                for (x, y) in lo.iter_mut().zip(hi) {
                    *x |= *y;
                }
                bits[v * words + w / 64] |= 1 << (w % 64);
            }
        }
        Reach { words, bits }
    }

    #[inline]
    pub fn reaches(&self, u: usize, v: usize) -> bool {
        self.bits[u * self.words + v / 64] >> (v % 64) & 1 == 1
    }

    fn row(&self, u: usize) -> impl Iterator<Item = usize> + '_ {
        self.bits[u * self.words..(u + 1) * self.words]
            .iter()
            .enumerate()
            .flat_map(|(i, &word)| {
                let mut word = word;
                std::iter::from_fn(move || {
                    if word == 0 {
                        return None;
                    }
                    let b = word.trailing_zeros() as usize;
                    word &= word - 1;
                    Some(i * 64 + b)
                })
            })
    }
}

/// `nx.antichains`: the stack of `(antichain, stack)` frames, expanded one
/// popped frame at a time, as NetworkX's generator does after each yield.
pub struct Antichains {
    reach: Reach,
    frames: Vec<(Vec<u32>, Vec<u32>)>,
    pending: Option<(Vec<u32>, Vec<u32>)>,
}

impl Antichains {
    pub fn new(reach: Reach, topo: &[u32]) -> Self {
        Antichains {
            reach,
            frames: vec![(Vec::new(), topo.iter().rev().copied().collect())],
            pending: None,
        }
    }

    pub fn next_antichain(&mut self) -> Option<Vec<u32>> {
        if let Some((antichain, mut stack)) = self.pending.take() {
            while let Some(x) = stack.pop() {
                let mut new_antichain = antichain.clone();
                new_antichain.push(x);
                let xi = x as usize;
                let new_stack = stack
                    .iter()
                    .copied()
                    .filter(|&t| {
                        let t = t as usize;
                        !(self.reach.reaches(xi, t) || self.reach.reaches(t, xi))
                    })
                    .collect();
                self.frames.push((new_antichain, new_stack));
            }
        }
        let (antichain, stack) = self.frames.pop()?;
        self.pending = Some((antichain.clone(), stack));
        Some(antichain)
    }
}

/// Size of a maximum matching between left copies and right copies of the
/// nodes, with `u -> v` whenever `u` reaches `v` (Hopcroft-Karp). The
/// width of the DAG (`antichain_width`) is `n` minus this.
pub fn reach_matching_size(reach: &Reach, n: usize) -> usize {
    const FREE: u32 = u32::MAX;
    let mut mate_l = vec![FREE; n];
    let mut mate_r = vec![FREE; n];
    let mut layer = vec![u32::MAX; n];
    let mut size = 0;
    loop {
        // BFS from free left nodes, layering the alternating graph.
        let mut queue = VecDeque::new();
        for u in 0..n {
            if mate_l[u] == FREE {
                layer[u] = 0;
                queue.push_back(u);
            } else {
                layer[u] = u32::MAX;
            }
        }
        let mut found = false;
        while let Some(u) = queue.pop_front() {
            for v in reach.row(u) {
                let m = mate_r[v];
                if m == FREE {
                    found = true;
                } else if layer[m as usize] == u32::MAX {
                    layer[m as usize] = layer[u] + 1;
                    queue.push_back(m as usize);
                }
            }
        }
        if !found {
            return size;
        }
        // Iterative DFS for vertex-disjoint shortest augmenting paths.
        for u in 0..n {
            if mate_l[u] != FREE {
                continue;
            }
            // stack of (left node, candidates still to try)
            let mut stack: Vec<(usize, Vec<usize>)> = vec![(u, reach.row(u).collect())];
            let mut path: Vec<usize> = Vec::new(); // right nodes chosen
            let mut done = false;
            while let Some((x, cands)) = stack.last_mut() {
                let x = *x;
                if let Some(v) = cands.pop() {
                    let m = mate_r[v];
                    if m == FREE {
                        path.push(v);
                        done = true;
                        break;
                    }
                    let m = m as usize;
                    if layer[m] == layer[x] + 1 {
                        path.push(v);
                        let c = reach.row(m).collect();
                        stack.push((m, c));
                    }
                } else {
                    layer[x] = u32::MAX; // dead end
                    stack.pop();
                    path.pop();
                }
            }
            if done {
                for (i, &(x, _)) in stack.iter().enumerate() {
                    let v = path[i];
                    mate_l[x] = v as u32;
                    mate_r[v] = x as u32;
                }
                for &(x, _) in &stack {
                    layer[x] = u32::MAX;
                }
                size += 1;
            }
        }
    }
}

// --- Simple paths -------------------------------------------------------------------

/// `_all_simple_edge_paths` as a resumable search. Yields each path as its
/// node positions (source first).
pub struct SimplePaths {
    adj: Csr,
    source: u32,
    is_target: Vec<bool>,
    targets: usize,
    /// Targets that aren't nodes of G (never on the path, so always left).
    extra_targets: bool,
    /// `len(current_path) - 1 < cutoff` is `path.len() < limit`.
    limit: usize,
    on_path: Vec<bool>,
    targets_on_path: usize,
    path: Vec<u32>,
    /// Next edge position per node on the path.
    next: Vec<usize>,
    /// The root iterator (`iter([(None, source)])`) is still on the stack
    /// and not yet consumed.
    root: u8,
    /// Yielded a path to this node; extend through it on resume.
    resume: Option<u32>,
}

impl SimplePaths {
    pub fn new(
        adj: &Csr,
        n: usize,
        source: u32,
        targets: &[u32],
        extra_targets: bool,
        limit: usize,
    ) -> Self {
        let mut is_target = vec![false; n];
        let mut count = 0;
        for &t in targets {
            if !is_target[t as usize] {
                is_target[t as usize] = true;
                count += 1;
            }
        }
        SimplePaths {
            adj: copy_csr(adj),
            source,
            is_target,
            targets: count,
            extra_targets,
            limit,
            on_path: vec![false; n],
            targets_on_path: 0,
            path: Vec::new(),
            next: Vec::new(),
            root: 2,
            resume: None,
        }
    }

    fn extend(&mut self, v: u32) {
        let left = self.targets - self.targets_on_path - usize::from(self.is_target[v as usize]);
        if self.path.len() < self.limit && (self.extra_targets || left > 0) {
            self.on_path[v as usize] = true;
            if self.is_target[v as usize] {
                self.targets_on_path += 1;
            }
            self.path.push(v);
            self.next.push(self.adj.offsets[v as usize]);
        }
    }

    pub fn next_path(&mut self) -> Option<Vec<u32>> {
        if let Some(v) = self.resume.take() {
            self.extend(v);
        }
        loop {
            let candidate = if let Some(&u) = self.path.last() {
                let u = u as usize;
                let i = self.next.last_mut().expect("parallel to path");
                let end = self.adj.offsets[u + 1];
                let mut found = None;
                while *i < end {
                    let v = self.adj.targets[*i];
                    *i += 1;
                    if !self.on_path[v as usize] {
                        found = Some(v);
                        break;
                    }
                }
                if found.is_none() {
                    self.on_path[u] = false;
                    if self.is_target[u] {
                        self.targets_on_path -= 1;
                    }
                    self.path.pop();
                    self.next.pop();
                    continue;
                }
                found
            } else if self.root == 2 {
                self.root = 1;
                Some(self.source)
            } else {
                self.root = 0;
                None
            };
            let v = candidate?;
            if self.is_target[v as usize] {
                self.resume = Some(v);
                let mut p = self.path.clone();
                p.push(v);
                return Some(p);
            }
            self.extend(v);
        }
    }
}

// --- Yen's k shortest simple paths -----------------------------------------------

/// A heap key ordered like Python's `(cost, counter)` tuples (costs are
/// never NaN; `0.0 == -0.0`).
#[derive(Clone, Copy, PartialEq)]
struct CostKey(f64, u64);

impl Eq for CostKey {}

impl PartialOrd for CostKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CostKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(Ordering::Equal)
            .then(self.1.cmp(&other.1))
    }
}

pub enum YenError {
    NoPath,
    Contradictory,
}

/// `shortest_simple_paths` state: `listA`, `PathBuffer` and the previous
/// path. Without weights it runs `_bidirectional_shortest_path`, with
/// weights `_bidirectional_dijkstra` (both with ignored nodes and edges).
pub struct SimpleShortestPaths {
    n: usize,
    directed: bool,
    succ: Csr,
    pred: Csr,
    w_succ: Option<Vec<f64>>,
    w_pred: Option<Vec<f64>>,
    compensated: bool,
    source: u32,
    target: u32,
    list_a: Vec<Vec<u32>>,
    heap: BinaryHeap<Reverse<(CostKey, usize)>>,
    stored: Vec<Option<Vec<u32>>>,
    in_buffer: HashSet<Vec<u32>>,
    counter: u64,
    prev: Option<Vec<u32>>,
}

impl SimpleShortestPaths {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        succ: &Csr,
        pred: &Csr,
        n: usize,
        directed: bool,
        weights: Option<(&[f64], &[f64])>,
        compensated: bool,
        source: u32,
        target: u32,
    ) -> Self {
        SimpleShortestPaths {
            n,
            directed,
            succ: copy_csr(succ),
            pred: copy_csr(pred),
            w_succ: weights.map(|w| w.0.to_vec()),
            w_pred: weights.map(|w| w.1.to_vec()),
            compensated,
            source,
            target,
            list_a: Vec::new(),
            heap: BinaryHeap::new(),
            stored: Vec::new(),
            in_buffer: HashSet::new(),
            counter: 0,
            prev: None,
        }
    }

    fn push(&mut self, cost: f64, path: Vec<u32>) {
        if self.in_buffer.contains(&path) {
            return;
        }
        self.in_buffer.insert(path.clone());
        self.heap
            .push(Reverse((CostKey(cost, self.counter), self.stored.len())));
        self.counter += 1;
        self.stored.push(Some(path));
    }

    /// The weight of edge `u -> v` (as `G.get_edge_data(u, v)` gives it).
    fn edge_weight(&self, u: u32, v: u32) -> f64 {
        let w = self.w_succ.as_ref().expect("weighted");
        let u = u as usize;
        self.succ
            .range(u)
            .find(|&e| self.succ.targets[e] == v)
            .map_or(f64::NAN, |e| w[e])
    }

    /// `length_func(root)`: `len` without weights, else Python's `sum()`.
    fn length(&self, root: &[u32]) -> f64 {
        if self.w_succ.is_none() {
            return root.len() as f64;
        }
        py_sum(
            root.windows(2).map(|p| self.edge_weight(p[0], p[1])),
            self.compensated,
        )
    }

    fn shortest(
        &self,
        s: u32,
        ignore_nodes: &[bool],
        ignore_edges: &HashSet<(u32, u32)>,
    ) -> Result<(f64, Vec<u32>), YenError> {
        let filter = Filter {
            directed: self.directed,
            ignore_nodes,
            ignore_edges,
        };
        if self.w_succ.is_none() {
            bidirectional_bfs_filtered(&self.succ, &self.pred, self.n, s, self.target, &filter)
                .map(|p| (p.len() as f64, p))
        } else {
            bidirectional_dijkstra_filtered(
                [
                    (&self.succ, self.w_succ.as_deref().expect("weighted")),
                    (&self.pred, self.w_pred.as_deref().expect("weighted")),
                ],
                self.n,
                s,
                self.target,
                &filter,
            )
        }
    }

    /// The next path, `None` when they are all out.
    pub fn next_path(&mut self) -> Result<Option<Vec<u32>>, YenError> {
        match self.prev.take() {
            None => {
                let none = vec![false; self.n];
                let (length, path) = self.shortest(self.source, &none, &HashSet::new())?;
                self.push(length, path);
            }
            Some(prev) => {
                let mut ignore_nodes = vec![false; self.n];
                let mut ignore_edges = HashSet::new();
                // listA paths still sharing prev_path's first i nodes.
                let mut matching: Vec<usize> = (0..self.list_a.len()).collect();
                for i in 1..prev.len() {
                    let root = &prev[..i];
                    let root_length = self.length(root);
                    matching.retain(|&k| {
                        let p = &self.list_a[k];
                        p.len() >= i && p[i - 1] == root[i - 1]
                    });
                    for &k in &matching {
                        let p = &self.list_a[k];
                        if p.len() > i {
                            ignore_edges.insert((p[i - 1], p[i]));
                        }
                    }
                    let spur_node = root[i - 1];
                    let ignored = ignore_nodes.iter().any(|&x| x);
                    let found = if ignored
                        && (ignore_nodes[spur_node as usize] || ignore_nodes[self.target as usize])
                    {
                        Err(YenError::NoPath)
                    } else {
                        self.shortest(spur_node, &ignore_nodes, &ignore_edges)
                    };
                    match found {
                        Ok((length, spur)) => {
                            let mut path = root[..i - 1].to_vec();
                            path.extend_from_slice(&spur);
                            self.push(root_length + length, path);
                        }
                        Err(YenError::NoPath) => {}
                        Err(e) => return Err(e),
                    }
                    ignore_nodes[spur_node as usize] = true;
                }
            }
        }
        let Some(Reverse((_, k))) = self.heap.pop() else {
            return Ok(None);
        };
        let path = self.stored[k].take().expect("stored once");
        self.in_buffer.remove(&path);
        self.list_a.push(path.clone());
        self.prev = Some(path.clone());
        Ok(Some(path))
    }
}

struct Filter<'a> {
    directed: bool,
    ignore_nodes: &'a [bool],
    ignore_edges: &'a HashSet<(u32, u32)>,
}

impl Filter<'_> {
    /// Whether neighbor `w` of `v` is kept; `forward`: `v -> w` (successors),
    /// else `w -> v` (predecessors).
    #[inline]
    fn keep(&self, v: u32, w: u32, forward: bool) -> bool {
        if self.ignore_nodes[w as usize] {
            return false;
        }
        if self.ignore_edges.is_empty() {
            return true;
        }
        if !self.directed {
            return !self.ignore_edges.contains(&(v, w)) && !self.ignore_edges.contains(&(w, v));
        }
        let edge = if forward { (v, w) } else { (w, v) };
        !self.ignore_edges.contains(&edge)
    }
}

/// `_bidirectional_pred_succ` + `_bidirectional_shortest_path` (after the
/// ignored-endpoint check).
fn bidirectional_bfs_filtered(
    succ: &Csr,
    pred: &Csr,
    n: usize,
    s: u32,
    t: u32,
    filter: &Filter,
) -> Result<Vec<u32>, YenError> {
    if s == t {
        return Ok(vec![s]);
    }
    const UNSET: u32 = u32::MAX - 1;
    let mut pred_of = vec![UNSET; n];
    let mut succ_of = vec![UNSET; n];
    pred_of[s as usize] = NONE;
    succ_of[t as usize] = NONE;
    let mut forward = vec![s];
    let mut reverse = vec![t];
    let mut meet = None;
    'search: while !forward.is_empty() && !reverse.is_empty() {
        if forward.len() <= reverse.len() {
            let level = std::mem::take(&mut forward);
            for &v in &level {
                for &w in succ.neighbors(v as usize) {
                    if !filter.keep(v, w, true) {
                        continue;
                    }
                    if pred_of[w as usize] == UNSET {
                        forward.push(w);
                        pred_of[w as usize] = v;
                    }
                    if succ_of[w as usize] != UNSET {
                        meet = Some(w);
                        break 'search;
                    }
                }
            }
        } else {
            let level = std::mem::take(&mut reverse);
            for &v in &level {
                for &w in pred.neighbors(v as usize) {
                    if !filter.keep(v, w, false) {
                        continue;
                    }
                    if succ_of[w as usize] == UNSET {
                        succ_of[w as usize] = v;
                        reverse.push(w);
                    }
                    if pred_of[w as usize] != UNSET {
                        meet = Some(w);
                        break 'search;
                    }
                }
            }
        }
    }
    let w = meet.ok_or(YenError::NoPath)?;
    let mut path = Vec::new();
    let mut cur = pred_of[w as usize];
    while cur != NONE {
        path.push(cur);
        cur = pred_of[cur as usize];
    }
    path.reverse();
    let mut cur = w;
    while cur != NONE {
        path.push(cur);
        cur = succ_of[cur as usize];
    }
    Ok(path)
}

/// `simple_paths._bidirectional_dijkstra` (after the ignored-endpoint
/// check). `adj[0]` is the forward adjacency, `adj[1]` the reverse one in
/// `G.pred` order, each with its weights (NaN = hidden). `finalpath` is
/// built when it is set, as NetworkX copies it then.
fn bidirectional_dijkstra_filtered(
    adj: [(&Csr, &[f64]); 2],
    n: usize,
    source: u32,
    target: u32,
    filter: &Filter,
) -> Result<(f64, Vec<u32>), YenError> {
    if source == target {
        return Ok((0.0, vec![source]));
    }
    let mut dist = [vec![f64::NAN; n], vec![f64::NAN; n]];
    let mut done = [vec![false; n], vec![false; n]];
    let mut seen = [vec![f64::INFINITY; n], vec![f64::INFINITY; n]];
    let mut has_seen = [vec![false; n], vec![false; n]];
    let mut parent = [vec![NONE; n], vec![NONE; n]];
    let mut fringe: [BinaryHeap<Reverse<(CostKey, u32)>>; 2] =
        [BinaryHeap::new(), BinaryHeap::new()];
    let mut counter = 0u64;
    for (d, s) in [(0, source), (1, target)] {
        seen[d][s as usize] = 0.0;
        has_seen[d][s as usize] = true;
        fringe[d].push(Reverse((CostKey(0.0, counter), s)));
        counter += 1;
    }
    let mut finaldist = 0.0;
    let mut finalpath: Vec<u32> = Vec::new();
    let mut dir = 1;
    while !fringe[0].is_empty() && !fringe[1].is_empty() {
        dir = 1 - dir;
        let Reverse((CostKey(d, _), v)) = fringe[dir].pop().expect("non-empty");
        let vi = v as usize;
        if done[dir][vi] {
            continue;
        }
        done[dir][vi] = true;
        dist[dir][vi] = d;
        if done[1 - dir][vi] {
            return Ok((finaldist, finalpath));
        }
        let (rows, weights) = adj[dir];
        for e in rows.range(vi) {
            let w = rows.targets[e];
            if !filter.keep(v, w, dir == 0) {
                continue;
            }
            let cost = weights[e];
            if cost.is_nan() {
                continue;
            }
            let wi = w as usize;
            let vw = dist[dir][vi] + cost;
            if done[dir][wi] {
                if vw < dist[dir][wi] {
                    return Err(YenError::Contradictory);
                }
            } else if !has_seen[dir][wi] || vw < seen[dir][wi] {
                seen[dir][wi] = vw;
                has_seen[dir][wi] = true;
                fringe[dir].push(Reverse((CostKey(vw, counter), w)));
                counter += 1;
                parent[dir][wi] = v;
                if has_seen[0][wi] && has_seen[1][wi] {
                    let total = seen[0][wi] + seen[1][wi];
                    if finalpath.is_empty() || finaldist > total {
                        finaldist = total;
                        let mut path = Vec::new();
                        let mut cur = w;
                        while cur != NONE {
                            path.push(cur);
                            cur = parent[0][cur as usize];
                        }
                        path.reverse();
                        let mut cur = parent[1][wi];
                        while cur != NONE {
                            path.push(cur);
                            cur = parent[1][cur as usize];
                        }
                        finalpath = path;
                    }
                }
            }
        }
    }
    Err(YenError::NoPath)
}

// --- Minimum cycle basis ------------------------------------------------------------

pub enum McbError {
    /// `_dijkstra`'s `ValueError` (two arguments).
    Contradictory,
    /// `bidirectional_dijkstra`'s `ValueError` (one argument).
    ContradictoryBidirectional,
    Unsupported,
}

/// `_min_cycle_basis` of one connected component. `nodes` are the
/// subgraph's nodes in its iteration order, `edges` its edges in
/// `G.edges` order with their weights, `chords` the chord set in
/// iteration order. Returns each basis cycle as node positions.
pub fn min_cycle_basis(
    n: usize,
    nodes: &[u32],
    edges: &[(u32, u32, f64)],
    chords: &[(u32, u32)],
) -> Result<Vec<Vec<u32>>, McbError> {
    let ns = nodes.len();
    let mut local = vec![NONE; n];
    for (k, &v) in nodes.iter().enumerate() {
        local[v as usize] = k as u32;
    }
    let canon = |u: u32, v: u32| if u <= v { (u, v) } else { (v, u) };
    let mut edge_id: HashMap<(u32, u32), u32> = HashMap::with_capacity(edges.len());
    for (i, &(u, v, _)) in edges.iter().enumerate() {
        edge_id.entry(canon(u, v)).or_insert(i as u32);
    }
    let id_of = |u: u32, v: u32| -> Result<u32, McbError> {
        edge_id
            .get(&canon(u, v))
            .copied()
            .ok_or(McbError::Unsupported)
    };
    let mut set_orth: Vec<HashSet<u32>> = Vec::with_capacity(chords.len());
    for &(u, v) in chords {
        set_orth.push(HashSet::from([id_of(u, v)?]));
    }
    let mut cb = Vec::new();
    while let Some(base) = set_orth.pop() {
        let cycle = min_cycle(ns, &local, edges, &base)?;
        cb.push(cycle.iter().map(|&(_, v)| v).collect());
        let ids: Vec<u32> = cycle
            .iter()
            .map(|&(u, v)| id_of(u, v))
            .collect::<Result<_, _>>()?;
        for orth in set_orth.iter_mut() {
            let odd = ids.iter().filter(|e| orth.contains(e)).count() % 2 == 1;
            if odd {
                let mut next: HashSet<u32> =
                    orth.iter().copied().filter(|e| !base.contains(e)).collect();
                next.extend(base.iter().copied().filter(|e| !orth.contains(e)));
                *orth = next;
            }
        }
    }
    Ok(cb)
}

/// `_min_cycle(G, orth, weight)`: the oriented edges of the cycle found in
/// the lifted graph `Gi` (real nodes `0..ns`, lifted copies `ns..2 ns`).
fn min_cycle(
    ns: usize,
    local: &[u32],
    edges: &[(u32, u32, f64)],
    orth: &HashSet<u32>,
) -> Result<Vec<(u32, u32)>, McbError> {
    let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); 2 * ns];
    let mut add_edge = |a: u32, b: u32, wt: f64| {
        if let Some(entry) = rows[a as usize].iter_mut().find(|(x, _)| *x == b) {
            entry.1 = wt;
            if let Some(back) = rows[b as usize].iter_mut().find(|(x, _)| *x == a) {
                back.1 = wt;
            }
            return;
        }
        rows[a as usize].push((b, wt));
        if a != b {
            rows[b as usize].push((a, wt));
        }
    };
    let lift = |x: u32| x + ns as u32;
    for (i, &(u, v, wt)) in edges.iter().enumerate() {
        let (lu, lv) = (local[u as usize], local[v as usize]);
        if orth.contains(&(i as u32)) {
            add_edge(lu, lift(lv), wt);
            add_edge(lift(lu), lv, wt);
        } else {
            add_edge(lu, lv, wt);
            add_edge(lift(lu), lift(lv), wt);
        }
    }
    let mut offsets = vec![0usize];
    let mut targets = Vec::new();
    let mut weights = Vec::new();
    for row in &rows {
        for &(b, wt) in row {
            targets.push(b);
            weights.push(wt);
        }
        offsets.push(targets.len());
    }
    let gi = Csr { offsets, targets };
    let lifts: Vec<Option<f64>> = (0..ns)
        .into_par_iter()
        .map(|k| {
            let t = lift(k as u32) as usize;
            let tree = dijkstra_forest(&gi, 2 * ns, Some(&weights), &[k as u32], None, Some(t))?;
            Ok(if tree.order.last() == Some(&(t as u32)) {
                tree.dist.last().copied()
            } else {
                None
            })
        })
        .collect::<Result<_, NegativeCycle>>()
        .map_err(|_| McbError::Contradictory)?;
    // `min(lift, key=lift.get)`: the first smallest.
    let mut start = 0;
    let mut best = f64::NAN;
    for (k, l) in lifts.iter().enumerate() {
        let l = l.ok_or(McbError::Unsupported)?;
        if k == 0 || l < best {
            start = k;
            best = l;
        }
    }
    // `nx.shortest_path(Gi, start, (start, 1), weight)` runs
    // `nx.bidirectional_dijkstra`.
    let end = lift(start as u32) as usize;
    let path_i = match bidirectional_dijkstra(
        [(&gi, Some(&weights)), (&gi, Some(&weights))],
        2 * ns,
        start,
        end,
    ) {
        Ok((_, path)) => path,
        Err(BidirectionalError::Contradictory) => return Err(McbError::ContradictoryBidirectional),
        Err(BidirectionalError::NoPath) => return Err(McbError::Unsupported),
    };
    // Back to positions in G (lifted copies map to their node).
    let real: Vec<u32> = local.iter().enumerate().filter(|(_, &k)| k != NONE).fold(
        vec![0u32; ns],
        |mut acc, (v, &k)| {
            acc[k as usize] = v as u32;
            acc
        },
    );
    let min_path: Vec<u32> = path_i.iter().map(|&x| real[(x as usize) % ns]).collect();
    let edgelist: Vec<(u32, u32)> = min_path.windows(2).map(|p| (p[0], p[1])).collect();
    let mut edgeset: HashSet<(u32, u32)> = HashSet::new();
    for &(a, b) in &edgelist {
        if !edgeset.remove(&(a, b)) && !edgeset.remove(&(b, a)) {
            edgeset.insert((a, b));
        }
    }
    let mut min_edgelist = Vec::new();
    for &(a, b) in &edgelist {
        if edgeset.remove(&(a, b)) {
            min_edgelist.push((a, b));
        } else if edgeset.remove(&(b, a)) {
            min_edgelist.push((b, a));
        }
    }
    Ok(min_edgelist)
}

/// Whether `topo` lists every node once with every edge pointing forward.
pub fn is_topological_order(adj: &Csr, n: usize, topo: &[u32]) -> bool {
    if topo.len() != n {
        return false;
    }
    let mut rank = vec![NONE; n];
    for (i, &v) in topo.iter().enumerate() {
        if rank[v as usize] != NONE {
            return false;
        }
        rank[v as usize] = i as u32;
    }
    (0..n).all(|u| adj.neighbors(u).iter().all(|&v| rank[v as usize] > rank[u]))
}
