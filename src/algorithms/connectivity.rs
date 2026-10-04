//! Batch 13: connectivity, disjoint paths and augmentation.
//!
//! NetworkX computes node and edge connectivity, minimum cuts and disjoint
//! paths with maximum flows (Edmonds-Karp by default) on auxiliary digraphs
//! and their residual networks. `Residual` replays NetworkX's residual
//! `DiGraph`: arcs sit in its `_succ` and `_pred` insertion order, so the
//! bidirectional BFS finds the same augmenting paths and the flows (and so
//! the disjoint paths and cut orders) come out the same.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::hash::BuildHasherDefault;

use rayon::prelude::*;

use super::structure_more::PairHasher;
use crate::graph::Csr;

const NONE: u32 = u32::MAX;

type PairMap = HashMap<u64, u32, BuildHasherDefault<PairHasher>>;

#[inline]
fn pair(u: u32, v: u32) -> u64 {
    ((u as u64) << 32) | v as u64
}

/// An auxiliary digraph `H` as NetworkX builds it: each node's successors
/// and predecessors in insertion order. Every arc has capacity 1.
pub struct Aux {
    pub succ: Vec<Vec<u32>>,
    pub pred: Vec<Vec<u32>>,
}

impl Aux {
    fn new(n: usize) -> Self {
        Aux {
            succ: vec![Vec::new(); n],
            pred: vec![Vec::new(); n],
        }
    }

    fn add_arc(&mut self, u: u32, v: u32) {
        self.succ[u as usize].push(v);
        self.pred[v as usize].push(u);
    }

    pub fn len(&self) -> usize {
        self.succ.len()
    }
}

/// `G.edges()` of a simple graph, in NetworkX order.
fn graph_edges(adj: &Csr, n: usize, directed: bool) -> impl Iterator<Item = (u32, u32)> + '_ {
    (0..n).flat_map(move |u| {
        adj.neighbors(u)
            .iter()
            .filter(move |&&v| directed || v as usize >= u)
            .map(move |&v| (u as u32, v))
    })
}

/// `build_auxiliary_edge_connectivity`: the graph's nodes, and each edge
/// as an arc (both directions for undirected graphs).
pub fn edge_aux(adj: &Csr, n: usize, directed: bool) -> Aux {
    let mut h = Aux::new(n);
    for (u, v) in graph_edges(adj, n, directed) {
        h.add_arc(u, v);
        if !directed && u != v {
            h.add_arc(v, u);
        }
    }
    h
}

/// `build_auxiliary_node_connectivity`: node `i` becomes `iA` (`2i`) and
/// `iB` (`2i + 1`) joined by an arc, and each edge `u-v` an arc `uB -> vA`
/// (plus `vB -> uA` for undirected graphs).
pub fn node_aux(adj: &Csr, n: usize, directed: bool) -> Aux {
    let mut h = Aux::new(2 * n);
    for i in 0..n as u32 {
        h.add_arc(2 * i, 2 * i + 1);
    }
    for (u, v) in graph_edges(adj, n, directed) {
        h.add_arc(2 * u + 1, 2 * v);
        // An undirected self-loop adds the same arc twice.
        if !directed && u != v {
            h.add_arc(2 * v + 1, 2 * u);
        }
    }
    h
}

/// NetworkX's residual network (`build_residual_network`) with flows.
#[derive(Clone)]
pub struct Residual {
    head: Vec<u32>,
    tail: Vec<u32>,
    cap: Vec<i64>,
    flow: Vec<i64>,
    rev: Vec<u32>,
    /// Arc ids out of each node, in `R._succ` order.
    succ: Vec<Vec<u32>>,
    /// Arc ids into each node, in `R._pred` order.
    pred: Vec<Vec<u32>>,
    inf: i64,
    // Bidirectional BFS scratch space.
    stamp: u32,
    in_pred: Vec<u32>,
    in_succ: Vec<u32>,
    pred_arc: Vec<u32>,
    succ_arc: Vec<u32>,
}

/// An augmenting path with infinite capacity (can't happen with the unit
/// capacities used here; reported so the caller can decline).
pub struct Unbounded;

impl Residual {
    pub fn build(h: &Aux) -> Self {
        let n = h.len();
        let mut r = Residual {
            head: Vec::new(),
            tail: Vec::new(),
            cap: Vec::new(),
            flow: Vec::new(),
            rev: Vec::new(),
            succ: vec![Vec::new(); n],
            pred: vec![Vec::new(); n],
            inf: 1,
            stamp: 0,
            in_pred: vec![0; n],
            in_succ: vec![0; n],
            pred_arc: vec![NONE; n],
            succ_arc: vec![NONE; n],
        };
        let mut index = PairMap::default();
        let mut total = 0i64;
        for u in 0..n as u32 {
            for &v in &h.succ[u as usize] {
                if u == v {
                    continue;
                }
                total += 1;
                if let Some(&a) = index.get(&pair(u, v)) {
                    // The reverse arc of an earlier edge: R[u][v]['capacity'] = r.
                    r.cap[a as usize] = 1;
                    continue;
                }
                let a = r.head.len() as u32;
                r.head.extend([v, u]);
                r.tail.extend([u, v]);
                r.cap.extend([1, 0]);
                r.flow.extend([0, 0]);
                r.rev.extend([a + 1, a]);
                index.insert(pair(u, v), a);
                index.insert(pair(v, u), a + 1);
                r.succ[u as usize].push(a);
                r.pred[v as usize].push(a);
                r.succ[v as usize].push(a + 1);
                r.pred[u as usize].push(a + 1);
            }
        }
        if total > 0 {
            r.inf = 3 * total;
        }
        r
    }

    /// `edmonds_karp_core`: flows from zero, augmenting until `cutoff`.
    pub fn edmonds_karp(&mut self, s: u32, t: u32, cutoff: f64) -> Result<i64, Unbounded> {
        self.flow.iter_mut().for_each(|f| *f = 0);
        let mut value = 0i64;
        let mut path = Vec::new();
        while (value as f64) < cutoff {
            let Some(meet) = self.bidirectional_bfs(s, t) else {
                break;
            };
            path.clear();
            let mut u = meet;
            while u != s {
                let a = self.pred_arc[u as usize];
                path.push(a);
                u = self.tail[a as usize];
            }
            path.reverse();
            let mut u = meet;
            while u != t {
                let a = self.succ_arc[u as usize];
                path.push(a);
                u = self.head[a as usize];
            }
            let mut f = self.inf;
            for &a in &path {
                f = f.min(self.cap[a as usize] - self.flow[a as usize]);
            }
            if f * 2 > self.inf {
                return Err(Unbounded);
            }
            for &a in &path {
                self.flow[a as usize] += f;
                self.flow[self.rev[a as usize] as usize] -= f;
            }
            value += f;
        }
        Ok(value)
    }

    fn bidirectional_bfs(&mut self, s: u32, t: u32) -> Option<u32> {
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.in_pred.iter_mut().for_each(|x| *x = 0);
            self.in_succ.iter_mut().for_each(|x| *x = 0);
            self.stamp = 1;
        }
        let stamp = self.stamp;
        self.in_pred[s as usize] = stamp;
        self.in_succ[t as usize] = stamp;
        let mut q_s = vec![s];
        let mut q_t = vec![t];
        loop {
            let mut q = Vec::new();
            if q_s.len() <= q_t.len() {
                for &u in &q_s {
                    for &a in &self.succ[u as usize] {
                        let v = self.head[a as usize];
                        if self.in_pred[v as usize] != stamp
                            && self.flow[a as usize] < self.cap[a as usize]
                        {
                            self.in_pred[v as usize] = stamp;
                            self.pred_arc[v as usize] = a;
                            if self.in_succ[v as usize] == stamp {
                                return Some(v);
                            }
                            q.push(v);
                        }
                    }
                }
                if q.is_empty() {
                    return None;
                }
                q_s = q;
            } else {
                for &u in &q_t {
                    for &a in &self.pred[u as usize] {
                        let v = self.tail[a as usize];
                        if self.in_succ[v as usize] != stamp
                            && self.flow[a as usize] < self.cap[a as usize]
                        {
                            self.in_succ[v as usize] = stamp;
                            self.succ_arc[v as usize] = a;
                            if self.in_pred[v as usize] == stamp {
                                return Some(v);
                            }
                            q.push(v);
                        }
                    }
                }
                if q.is_empty() {
                    return None;
                }
                q_t = q;
            }
        }
    }

    /// `minimum_cut`'s sink side: nodes that reach `t` through unsaturated
    /// arcs, in the breadth-first order NetworkX adds them (a BFS over
    /// `R._pred`; 3.4 to 3.6 remove the saturated arcs first, which keeps
    /// the order of the rest).
    pub fn sink_side(&self, t: u32) -> (Vec<u32>, Vec<bool>) {
        let n = self.succ.len();
        let mut seen = vec![false; n];
        seen[t as usize] = true;
        let mut order = vec![t];
        let mut i = 0;
        while i < order.len() {
            let v = order[i];
            for &a in &self.pred[v as usize] {
                let u = self.tail[a as usize];
                if !seen[u as usize] && self.flow[a as usize] < self.cap[a as usize] {
                    seen[u as usize] = true;
                    order.push(u);
                }
            }
            i += 1;
        }
        (order, seen)
    }

    /// NetworkX 3.4 to 3.6's `minimum_cut` removes the saturated arcs and
    /// adds them back, which moves them to the end of `R._succ` and
    /// `R._pred`. A residual network shared between calls keeps that order.
    pub fn requeue_saturated(&mut self) {
        let mut saturated = Vec::new();
        let mut is_sat = vec![false; self.head.len()];
        for row in &self.succ {
            for &a in row {
                if self.flow[a as usize] == self.cap[a as usize] {
                    saturated.push(a);
                    is_sat[a as usize] = true;
                }
            }
        }
        if saturated.is_empty() {
            return;
        }
        for row in self.succ.iter_mut() {
            let (keep, moved): (Vec<u32>, Vec<u32>) =
                row.iter().partition(|&&a| !is_sat[a as usize]);
            row.clear();
            row.extend(keep);
            row.extend(moved);
        }
        for row in self.pred.iter_mut() {
            row.retain(|&a| !is_sat[a as usize]);
        }
        for &a in &saturated {
            self.pred[self.head[a as usize] as usize].push(a);
        }
    }

    /// `edge_disjoint_paths`: follow the saturated arcs the way NetworkX
    /// does (`flow_dict[u].popitem()` takes the last one added).
    pub fn disjoint_paths(&self, s: u32, t: u32, cutoff: f64) -> Vec<Vec<u32>> {
        let n = self.succ.len();
        let mut out_arcs: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (u, row) in self.succ.iter().enumerate() {
            for &a in row {
                let (c, f) = (self.cap[a as usize], self.flow[a as usize]);
                if c == f && f > 0 {
                    out_arcs[u].push(self.head[a as usize]);
                }
            }
        }
        let starts = out_arcs[s as usize].clone();
        let mut paths = Vec::new();
        let mut found = 0i64;
        for v in starts {
            if found as f64 >= cutoff {
                break;
            }
            let mut path = vec![s];
            if v == t {
                path.push(v);
                paths.push(path);
                continue;
            }
            let mut u = v;
            loop {
                if u == t {
                    path.push(t);
                    paths.push(path);
                    found += 1;
                    break;
                }
                path.push(u);
                match out_arcs[u as usize].pop() {
                    Some(x) => u = x,
                    None => break,
                }
            }
        }
        paths
    }
}

/// Arcs `u -> v` of `rows` from the source side to the sink side, grouped
/// by `u` (in node order) as `(us, offsets, vs)`.
pub fn cut_arcs<'a, F>(n: usize, rows: F, sink: &[bool]) -> (Vec<u32>, Vec<u32>, Vec<u32>)
where
    F: Fn(usize) -> &'a [u32],
{
    let (mut us, mut offsets, mut vs) = (Vec::new(), vec![0u32], Vec::new());
    for u in 0..n {
        if sink[u] {
            continue;
        }
        let before = vs.len();
        vs.extend(rows(u).iter().filter(|&&v| sink[v as usize]));
        if vs.len() > before {
            us.push(u as u32);
            offsets.push(vs.len() as u32);
        }
    }
    (us, offsets, vs)
}

/// One s-t minimum cut as Python rebuilds it: the flow value, the sink
/// side in insertion order, and the cut arcs (see `cut_arcs`).
pub struct StCut {
    pub value: i64,
    pub sink_order: Vec<u32>,
    pub us: Vec<u32>,
    pub offsets: Vec<u32>,
    pub vs: Vec<u32>,
}

/// Which rows `minimum_st_edge_cut` reads the cut edges from.
pub enum CutRows<'a> {
    Graph(&'a Csr),
    Aux,
}

pub fn st_cut(
    h: &Aux,
    r: &mut Residual,
    s: u32,
    t: u32,
    rows: &CutRows,
    requeue: bool,
) -> Result<StCut, Unbounded> {
    let value = r.edmonds_karp(s, t, f64::INFINITY)?;
    let (sink_order, sink) = r.sink_side(t);
    if requeue {
        r.requeue_saturated();
    }
    let (us, offsets, vs) = match rows {
        CutRows::Graph(adj) => cut_arcs(h.len(), |u| adj.neighbors(u), &sink),
        CutRows::Aux => cut_arcs(h.len(), |u| h.succ[u].as_slice(), &sink),
    };
    Ok(StCut {
        value,
        sink_order,
        us,
        offsets,
        vs,
    })
}

/// `len(node_cut - {x, y})` for a node-split cut: distinct original nodes
/// among the cut arcs' endpoints, other than `x` and `y`.
fn node_cut_len(cut: &StCut, x: u32, y: u32, mark: &mut [bool]) -> usize {
    let mut nodes = Vec::new();
    let mut add = |a: u32, nodes: &mut Vec<u32>| {
        let i = a / 2;
        if i != x && i != y && !mark[i as usize] {
            mark[i as usize] = true;
            nodes.push(i);
        }
    };
    for (k, &u) in cut.us.iter().enumerate() {
        add(u, &mut nodes);
        for &v in &cut.vs[cut.offsets[k] as usize..cut.offsets[k + 1] as usize] {
            add(v, &mut nodes);
        }
    }
    for &i in &nodes {
        mark[i as usize] = false;
    }
    nodes.len()
}

/// One step of `minimum_node_cut` / `minimum_edge_cut`'s search.
pub struct CutPair {
    pub s: u32,
    pub t: u32,
    /// `if y in G[x]: continue` (the neighbor pairs of `minimum_node_cut`).
    pub skip_if_edge: bool,
}

/// The cut `minimum_node_cut` / `minimum_edge_cut` returns: `None` keeps
/// the initial cut; `Some((i, None))` is pair `i`'s empty answer for
/// adjacent nodes; `Some((i, Some(cut)))` pair `i`'s minimum cut.
pub type ChosenCut = Option<(usize, Option<StCut>)>;

/// Runs the cuts in order on one shared residual network (as NetworkX
/// does), keeping the last one no larger than the best so far.
#[allow(clippy::too_many_arguments)]
pub fn search_cuts(
    adj: &Csr,
    h: &Aux,
    node_split: bool,
    pairs: &[CutPair],
    initial_len: usize,
    adjacent_both: bool,
    requeue: bool,
) -> Result<ChosenCut, Unbounded> {
    let mut r = Residual::build(h);
    let has_edge = |u: u32, v: u32| adj.neighbors(u as usize).contains(&v);
    let mut best = initial_len;
    let mut chosen = None;
    let mut mark = vec![false; adj.offsets.len().saturating_sub(1)];
    for (i, p) in pairs.iter().enumerate() {
        if p.skip_if_edge && has_edge(p.s, p.t) {
            continue;
        }
        if node_split {
            if has_edge(p.s, p.t) || (adjacent_both && has_edge(p.t, p.s)) {
                // minimum_st_node_cut returns an empty cut for adjacent nodes.
                best = 0;
                chosen = Some((i, None));
                continue;
            }
            let cut = st_cut(h, &mut r, 2 * p.s + 1, 2 * p.t, &CutRows::Aux, requeue)?;
            let len = node_cut_len(&cut, p.s, p.t, &mut mark);
            if best >= len {
                best = len;
                chosen = Some((i, Some(cut)));
            }
        } else {
            let cut = st_cut(h, &mut r, p.s, p.t, &CutRows::Aux, requeue)?;
            let len = cut.vs.len();
            if best >= len {
                best = len;
                chosen = Some((i, Some(cut)));
            }
        }
    }
    Ok(chosen)
}

/// Flow values for many pairs, each from scratch (the shared residual
/// network is reset by every Edmonds-Karp run, so pairs are independent).
pub fn pair_flows(h: &Aux, pairs: &[(u32, u32)], cutoff: f64) -> Result<Vec<i64>, Unbounded> {
    let base = Residual::build(h);
    pairs
        .par_iter()
        .map_init(|| base.clone(), |r, &(s, t)| r.edmonds_karp(s, t, cutoff))
        .collect()
}

/// The first node with the smallest isolating cut, as NetworkX 3.7's
/// `node_connectivity` and `minimum_node_cut` pick it: `(node, cut size,
/// whether the cut is its predecessors)`. The cut is the node's distinct
/// neighbors other than itself (directed: the smaller of predecessors and
/// successors, predecessors on ties).
pub fn isolating_cut(adj: &Csr, pred: &Csr, n: usize, directed: bool) -> (usize, usize, bool) {
    let mut mark = vec![NONE; n];
    let mut distinct_without = |row: &[u32], v: usize, stamp: u32| {
        let mut count = 0;
        for &w in row {
            if w as usize != v && mark[w as usize] != stamp {
                mark[w as usize] = stamp;
                count += 1;
            }
        }
        count
    };
    let mut best = (0, usize::MAX, false);
    for v in 0..n {
        let succ = distinct_without(adj.neighbors(v), v, 2 * v as u32);
        let (len, use_pred) = if directed {
            let p = distinct_without(pred.neighbors(v), v, 2 * v as u32 + 1);
            if p <= succ {
                (p, true)
            } else {
                (succ, false)
            }
        } else {
            (succ, false)
        };
        if len < best.1 {
            best = (v, len, use_pred);
        }
    }
    best
}

/// `node_connectivity(G)`: the starting bound `k` at node `v`, lowered by
/// the local connectivity of each pair NetworkX tries. The minimum doesn't
/// depend on the order NetworkX tries them in (each run's cutoff only caps
/// values that are already no smaller than the minimum so far).
/// `isolating`: NetworkX 3.7's version, which leaves `v` out of its
/// neighbors and tries both directions from `v` in directed graphs.
#[allow(clippy::too_many_arguments)]
pub fn node_connectivity(
    adj: &Csr,
    pred: &Csr,
    n: usize,
    directed: bool,
    v: usize,
    k: i64,
    isolating: bool,
) -> Result<i64, Unbounded> {
    let mut nbrs: Vec<u32> = Vec::new();
    if directed {
        nbrs.extend_from_slice(pred.neighbors(v));
    }
    nbrs.extend_from_slice(adj.neighbors(v));
    let mut is_nbr = vec![false; n];
    if isolating {
        // set(neighbors(v)) - {v}
        nbrs.retain(|&w| {
            let fresh = w as usize != v && !is_nbr[w as usize];
            is_nbr[w as usize] = true;
            fresh
        });
    }
    for &w in &nbrs {
        is_nbr[w as usize] = true;
    }
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for w in 0..n as u32 {
        if w as usize != v && !is_nbr[w as usize] {
            pairs.push((v as u32, w));
            if isolating && directed {
                pairs.push((w, v as u32));
            }
        }
    }
    let has_edge = |x: u32, y: u32| adj.neighbors(x as usize).contains(&y);
    for i in 0..nbrs.len() {
        let others: Box<dyn Iterator<Item = usize>> = if directed {
            Box::new((0..nbrs.len()).filter(move |&j| j != i))
        } else {
            Box::new(i + 1..nbrs.len())
        };
        for j in others {
            let (x, y) = (nbrs[i], nbrs[j]);
            if !has_edge(x, y) {
                pairs.push((x, y));
            }
        }
    }
    if pairs.is_empty() {
        return Ok(k);
    }
    let pairs: Vec<(u32, u32)> = pairs.into_iter().map(|(x, y)| (2 * x + 1, 2 * y)).collect();
    let h = node_aux(adj, n, directed);
    // With cutoff k a run returns its value or something >= k; the minimum
    // is the same.
    Ok(pair_flows(&h, &pairs, k as f64)?
        .into_iter()
        .fold(k, i64::min))
}

/// `edge_connectivity(G)`'s minimum over local edge connectivities, capped
/// by `cutoff`; `None` if NetworkX tries no pair (a complete graph). For
/// undirected graphs any dominating set works (Esfahanian and Hakimi),
/// since NetworkX's own pick depends on set order; directed graphs use
/// NetworkX's cycle through the nodes.
pub fn edge_connectivity(
    adj: &Csr,
    n: usize,
    directed: bool,
    cutoff: f64,
) -> Result<Option<i64>, Unbounded> {
    let pairs: Vec<(u32, u32)> = if directed {
        (0..n as u32).map(|i| (i, (i + 1) % n as u32)).collect()
    } else {
        let full = |u: usize| {
            adj.neighbors(u)
                .iter()
                .filter(|&&w| w as usize != u)
                .count()
                == n - 1
        };
        let Some(start) = (0..n).find(|&u| !full(u)) else {
            return Ok(None);
        };
        let mut dominated = vec![false; n];
        let mut dom = vec![start as u32];
        dominated[start] = true;
        for &w in adj.neighbors(start) {
            dominated[w as usize] = true;
        }
        for u in 0..n {
            if !dominated[u] {
                dom.push(u as u32);
                dominated[u] = true;
                for &w in adj.neighbors(u) {
                    dominated[w as usize] = true;
                }
            }
        }
        dom[1..].iter().map(|&w| (start as u32, w)).collect()
    };
    let h = edge_aux(adj, n, directed);
    Ok(pair_flows(&h, &pairs, cutoff)?.into_iter().min())
}

/// `stoer_wagner`'s result: the cut value, the nodes in the order of the
/// rebuilt graph, and the reachable side in breadth-first order.
pub struct StoerWagner {
    pub cut_value: f64,
    pub node_order: Vec<u32>,
    pub reachable: Vec<u32>,
}

#[derive(PartialEq)]
struct HeapItem(f64, u64, u32);

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    // A min-heap on (value, insertion count), like BinaryHeap's tuples.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .partial_cmp(&self.0)
            .unwrap_or(Ordering::Equal)
            .then(other.1.cmp(&self.1))
    }
}

/// `networkx.utils.BinaryHeap`: lazy deletion, ties by insertion order.
struct LazyHeap {
    heap: BinaryHeap<HeapItem>,
    value: Vec<f64>,
    present: Vec<bool>,
    keys: Vec<u32>,
    count: u64,
}

impl LazyHeap {
    fn new(n: usize) -> Self {
        LazyHeap {
            heap: BinaryHeap::new(),
            value: vec![0.0; n],
            present: vec![false; n],
            keys: Vec::new(),
            count: 0,
        }
    }

    fn clear(&mut self) {
        self.heap.clear();
        for &k in &self.keys {
            self.present[k as usize] = false;
        }
        self.keys.clear();
    }

    fn get(&self, key: u32) -> Option<f64> {
        self.present[key as usize].then(|| self.value[key as usize])
    }

    fn insert(&mut self, key: u32, value: f64) {
        let k = key as usize;
        if self.present[k] {
            if value < self.value[k] {
                self.value[k] = value;
            } else {
                return;
            }
        } else {
            self.present[k] = true;
            self.value[k] = value;
            self.keys.push(key);
        }
        self.heap.push(HeapItem(value, self.count, key));
        self.count += 1;
    }

    fn valid(&self, item: &HeapItem) -> bool {
        self.present[item.2 as usize] && item.0 == self.value[item.2 as usize]
    }

    fn min(&mut self) -> (u32, f64) {
        while let Some(top) = self.heap.peek() {
            if self.valid(top) {
                return (top.2, top.0);
            }
            self.heap.pop();
        }
        unreachable!("stoer_wagner's heap is never empty here")
    }

    fn pop(&mut self) -> (u32, f64) {
        while let Some(top) = self.heap.pop() {
            if self.valid(&top) {
                self.present[top.2 as usize] = false;
                return (top.2, top.0);
            }
        }
        unreachable!("stoer_wagner's heap is never empty here")
    }
}

/// `nx.stoer_wagner` on a connected graph with at least two nodes;
/// `weights` holds each `succ` entry's weight (NetworkX's arithmetic is
/// replayed in f64, which is exact for the ints the caller allows).
pub fn stoer_wagner(adj: &Csr, n: usize, weights: Option<&[f64]>) -> StoerWagner {
    // nx.Graph(edges without self-loops): nodes in order of appearance.
    let mut pos = vec![NONE; n];
    let mut node_order: Vec<u32> = Vec::new();
    let mut rows: Vec<Vec<(u32, u32)>> = Vec::new();
    let mut w: Vec<f64> = Vec::new();
    let mut index = PairMap::default();
    let mut add_node = |v: u32, order: &mut Vec<u32>, rows: &mut Vec<Vec<(u32, u32)>>| {
        if pos[v as usize] == NONE {
            pos[v as usize] = order.len() as u32;
            order.push(v);
            rows.push(Vec::new());
        }
        pos[v as usize]
    };
    for u in 0..n {
        for e in adj.range(u) {
            let v = adj.targets[e];
            if (v as usize) <= u {
                continue;
            }
            let (a, b) = (
                add_node(u as u32, &mut node_order, &mut rows),
                add_node(v, &mut node_order, &mut rows),
            );
            let id = w.len() as u32;
            w.push(weights.map_or(1.0, |ws| ws[e]));
            rows[a as usize].push((b, id));
            rows[b as usize].push((a, id));
            index.insert(pair(a, b), id);
            index.insert(pair(b, a), id);
        }
    }
    let m = node_order.len();
    let mut alive = vec![true; m];
    let mut first = 0usize;
    let mut in_a = vec![false; m];
    let mut heap = LazyHeap::new(m);
    let mut cut_value = f64::INFINITY;
    let mut best_phase = 0usize;
    let mut contractions: Vec<(u32, u32)> = Vec::new();
    for i in 0..m.saturating_sub(1) {
        while !alive[first] {
            first += 1;
        }
        let mut u = first as u32;
        let mut a_list = vec![u];
        in_a[u as usize] = true;
        heap.clear();
        for &(v, e) in &rows[u as usize] {
            if alive[v as usize] {
                heap.insert(v, -w[e as usize]);
            }
        }
        for _ in 0..(m - i - 2) {
            u = heap.pop().0;
            in_a[u as usize] = true;
            a_list.push(u);
            for &(v, e) in &rows[u as usize] {
                if alive[v as usize] && !in_a[v as usize] {
                    let cur = heap.get(v).unwrap_or(0.0);
                    heap.insert(v, cur - w[e as usize]);
                }
            }
        }
        let (v, value) = heap.min();
        let value = -value;
        if value < cut_value {
            cut_value = value;
            best_phase = i;
        }
        for &x in &a_list {
            in_a[x as usize] = false;
        }
        contractions.push((u, v));
        // Merge v into u: G[u][x] += G[v][x], or a new edge u-x.
        let v_row = rows[v as usize].clone();
        for (x, e) in v_row {
            if !alive[x as usize] || x == u {
                continue;
            }
            match index.get(&pair(u, x)) {
                Some(&id) => w[id as usize] += w[e as usize],
                None => {
                    let id = w.len() as u32;
                    w.push(w[e as usize]);
                    rows[u as usize].push((x, id));
                    rows[x as usize].push((u, id));
                    index.insert(pair(u, x), id);
                    index.insert(pair(x, u), id);
                }
            }
        }
        alive[v as usize] = false;
    }
    // nx.Graph(contractions[:best_phase]) plus v, searched from v. Each
    // contracted node appears once as a second element, so no edge repeats.
    let v = contractions[best_phase].1;
    let mut cidx: HashMap<u32, usize> = HashMap::new();
    let mut names: Vec<u32> = Vec::new();
    let mut crows: Vec<Vec<u32>> = Vec::new();
    let mut node = |x: u32, names: &mut Vec<u32>, crows: &mut Vec<Vec<u32>>| -> usize {
        *cidx.entry(x).or_insert_with(|| {
            names.push(x);
            crows.push(Vec::new());
            names.len() - 1
        })
    };
    for &(a, b) in &contractions[..best_phase] {
        let ia = node(a, &mut names, &mut crows);
        let ib = node(b, &mut names, &mut crows);
        crows[ia].push(ib as u32);
        crows[ib].push(ia as u32);
    }
    let iv = node(v, &mut names, &mut crows);
    let mut seen = vec![false; crows.len()];
    seen[iv] = true;
    let mut order = vec![iv as u32];
    let mut k = 0;
    while k < order.len() {
        let x = order[k] as usize;
        for &y in &crows[x] {
            if !seen[y as usize] {
                seen[y as usize] = true;
                order.push(y);
            }
        }
        k += 1;
    }
    let reachable = order
        .into_iter()
        .map(|x| node_order[names[x as usize] as usize])
        .collect();
    StoerWagner {
        cut_value,
        node_order,
        reachable,
    }
}

/// `bridge_components`: components of `G.copy()` without its bridges.
/// Copying an undirected graph reorders each adjacency row: neighbors
/// earlier in node order come first (added while copying those nodes).
pub fn bridge_components(adj: &Csr, n: usize, bridges: &[(u32, u32)]) -> Vec<Vec<u32>> {
    let mut is_bridge = std::collections::HashSet::<u64, BuildHasherDefault<PairHasher>>::default();
    for &(u, v) in bridges {
        is_bridge.insert(pair(u, v));
        is_bridge.insert(pair(v, u));
    }
    let mut rows: Vec<Vec<u32>> = vec![Vec::new(); n];
    for u in 0..n {
        for &v in adj.neighbors(u) {
            if (v as usize) < u || is_bridge.contains(&pair(u as u32, v)) {
                continue;
            }
            rows[u].push(v);
            if v as usize != u {
                rows[v as usize].push(u as u32);
            }
        }
    }
    let mut seen = vec![false; n];
    let mut comps = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut comp = vec![start as u32];
        let mut i = 0;
        while i < comp.len() {
            let v = comp[i] as usize;
            for &w in &rows[v] {
                if !seen[w as usize] {
                    seen[w as usize] = true;
                    comp.push(w);
                }
            }
            i += 1;
        }
        comps.push(comp);
    }
    comps
}

/// `complement_edges` for nodes `start..end` (as `u`): flat `(u, v)`
/// position pairs in yield order.
/// `pred` gives each node's predecessors (any order; directed only).
pub fn complement_edges(
    adj: &Csr,
    pred: &Csr,
    n: usize,
    directed: bool,
    start: usize,
    end: usize,
) -> Vec<u32> {
    // succ_mark[v]: u -> v exists; pred_mark[v]: v -> u exists.
    let mut succ_mark = vec![false; n];
    let mut pred_mark = vec![false; n];
    let mut out = Vec::new();
    for u in start..end {
        for &w in adj.neighbors(u) {
            succ_mark[w as usize] = true;
        }
        if directed {
            for &w in pred.neighbors(u) {
                pred_mark[w as usize] = true;
            }
        }
        for v in u + 1..n {
            if !succ_mark[v] {
                out.extend([u as u32, v as u32]);
            }
            if directed && !pred_mark[v] {
                out.extend([v as u32, u as u32]);
            }
        }
        for &w in adj.neighbors(u) {
            succ_mark[w as usize] = false;
        }
        if directed {
            for &w in pred.neighbors(u) {
                pred_mark[w as usize] = false;
            }
        }
    }
    out
}
