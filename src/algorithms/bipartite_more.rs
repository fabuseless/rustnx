//! Batch 26: multigraphs and bipartite measures.
//!
//! Multigraph snapshots (`graph.rs`) collapse parallel edges to one CSR
//! entry with the smallest weight, which is how NetworkX's path code sees
//! them. Functions that count parallel edges (degrees), add up their
//! weights (`pagerank`'s sparse matrix) or pick one of them (spanning
//! trees with keys) read the per-entry data kept here instead, aligned
//! with the snapshot's CSR rows.

use std::collections::HashMap;

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::pyset::PySet;
use crate::graph::{parse_value, CoreGraph, Csr, ValueKind};

fn changed() -> PyErr {
    PyNotImplementedError::new_err("the graph changed since it was converted")
}

// --- Multigraph edge data -----------------------------------------------------------

/// Per CSR entry of a multigraph snapshot: how many parallel edges it
/// stands for and, for one edge attribute, the sum, the first minimum and
/// the first maximum over them (with the position of that edge's key in
/// the key dict).
#[pyclass(frozen, module = "rustnx._core")]
pub struct MultiEdges {
    pub mult: Vec<u32>,
    pub sum: Vec<f64>,
    pub min_w: Vec<f64>,
    pub min_key: Vec<u32>,
    pub max_w: Vec<f64>,
    pub max_key: Vec<u32>,
    pub weighted: bool,
}

/// Read `adj` (a multigraph's `G._adj`, rows in `nodes` order) into
/// `MultiEdges` aligned with `g.succ`. `attr` is `(name, default)`; values
/// must be numbers (`None` weights are left to NetworkX).
pub fn read_multi<'py>(
    g: &CoreGraph,
    nodes: &Bound<'py, pyo3::types::PyList>,
    adj: &Bound<'py, PyAny>,
    attr: Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
) -> PyResult<MultiEdges> {
    if nodes.len() != g.n {
        return Err(changed());
    }
    let m = g.succ.targets.len();
    let weighted = attr.is_some();
    let cap = if weighted { m } else { 0 };
    let mut out = MultiEdges {
        mult: Vec::with_capacity(m),
        sum: Vec::with_capacity(cap),
        min_w: Vec::with_capacity(cap),
        min_key: Vec::with_capacity(cap),
        max_w: Vec::with_capacity(cap),
        max_key: Vec::with_capacity(cap),
        weighted,
    };
    let mut entry = |keydict: &Bound<'py, PyAny>| -> PyResult<()> {
        let kd = keydict.cast::<PyDict>().map_err(|_| changed())?;
        out.mult.push(kd.len() as u32);
        let Some((name, default)) = &attr else {
            return Ok(());
        };
        let (mut s, mut lo, mut hi) = (0.0f64, f64::NAN, f64::NAN);
        let (mut lo_k, mut hi_k) = (0u32, 0u32);
        for (k, data) in kd.values().iter().enumerate() {
            let value = match data.cast::<PyDict>() {
                Ok(d) => d.get_item(name)?.unwrap_or_else(|| default.clone()),
                Err(_) => data.call_method1("get", (name, default))?,
            };
            let (x, kind) = parse_value(&value)?;
            if kind == ValueKind::Hidden {
                return Err(PyNotImplementedError::new_err(
                    "rustnx does not support None edge weights here",
                ));
            }
            s += x;
            if k == 0 || x < lo {
                lo = x;
                lo_k = k as u32;
            }
            if k == 0 || x > hi {
                hi = x;
                hi_k = k as u32;
            }
        }
        if kd.is_empty() {
            return Err(PyNotImplementedError::new_err("edge without keys"));
        }
        out.sum.push(s);
        out.min_w.push(lo);
        out.min_key.push(lo_k);
        out.max_w.push(hi);
        out.max_key.push(hi_k);
        Ok(())
    };
    for (v, node) in nodes.iter().enumerate() {
        let row = adj.get_item(&node)?;
        if row.len()? != g.succ.neighbors(v).len() {
            return Err(changed());
        }
        if let Ok(d) = row.cast::<PyDict>() {
            for keydict in d.values().iter() {
                entry(&keydict)?;
            }
        } else {
            for keydict in row.call_method0("values")?.try_iter()? {
                entry(&keydict?)?;
            }
        }
    }
    if out.mult.len() != m {
        return Err(changed());
    }
    Ok(out)
}

/// For each `pred` slot of a directed snapshot, the `succ` entry it
/// mirrors (`graph::transpose` orders in-edges by source position).
pub fn pred_entries(g: &CoreGraph) -> PyResult<Vec<usize>> {
    let pred = g.adj(true);
    let mut next = pred.offsets.clone();
    let mut order = vec![0usize; g.succ.targets.len()];
    for u in 0..g.n {
        for e in g.succ.range(u) {
            let t = g.succ.targets[e] as usize;
            let slot = next[t];
            if slot >= pred.offsets[t + 1] || pred.targets[slot] as usize != u {
                return Err(PyNotImplementedError::new_err(
                    "in-edges are not in source order",
                ));
            }
            next[t] += 1;
            order[slot] = e;
        }
    }
    Ok(order)
}

/// NetworkX degrees of a multigraph, counting parallel edges: `(degree,
/// in-degree, out-degree)`; a self-loop counts twice in an undirected
/// graph's degree.
#[allow(clippy::needless_range_loop)]
pub fn multi_degrees(g: &CoreGraph, me: &MultiEdges) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let n = g.n;
    let mut out = vec![0u64; n];
    let mut inc = vec![0u64; n];
    for u in 0..n {
        for e in g.succ.range(u) {
            let k = me.mult[e] as u64;
            out[u] += k;
            let v = g.succ.targets[e] as usize;
            if g.directed {
                inc[v] += k;
            } else if v == u {
                out[u] += k;
            }
        }
    }
    let deg = if g.directed {
        out.iter().zip(&inc).map(|(a, b)| a + b).collect()
    } else {
        out.clone()
    };
    (deg, inc, out)
}

/// The undirected edges of `G.edges` order (each once, from the earlier
/// endpoint) with the entry each comes from.
fn undirected_entries(succ: &Csr, n: usize) -> Vec<(u32, u32, usize)> {
    let mut edges = Vec::new();
    for u in 0..n {
        for e in succ.range(u) {
            let v = succ.targets[e];
            if v as usize >= u {
                edges.push((u as u32, v, e));
            }
        }
    }
    edges
}

/// `kruskal_mst_edges` on an undirected multigraph. NetworkX sorts every
/// parallel edge (stably, by weight) in `G.edges(keys=True)` order; the
/// first one of a node pair in sorted order is its first minimum (first
/// maximum, for maximum trees), and the rest join nodes already joined.
/// So it runs on the collapsed graph. Returns `(us, vs, key positions)`.
pub fn multi_kruskal(
    g: &CoreGraph,
    me: &MultiEdges,
    maximum: bool,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let entries = undirected_entries(&g.succ, g.n);
    let (w, keys) = if maximum {
        (&me.max_w, &me.max_key)
    } else {
        (&me.min_w, &me.min_key)
    };
    let edges: Vec<(u32, u32, f64)> = entries.iter().map(|&(u, v, e)| (u, v, w[e])).collect();
    let kept = super::structure::kruskal(g.n, &edges, maximum);
    let mut out = (Vec::new(), Vec::new(), Vec::new());
    for i in kept {
        let (u, v, e) = entries[i as usize];
        out.0.push(u);
        out.1.push(v);
        out.2.push(keys[e]);
    }
    out
}

/// `prim_mst_edges` on an undirected multigraph: NetworkX pushes each
/// parallel edge with consecutive counters, so the first one popped for a
/// node pair is again the first minimum (or maximum), as on the collapsed
/// graph.
pub fn multi_prim(
    g: &CoreGraph,
    me: &MultiEdges,
    starts: &[u32],
    minimum: bool,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let (w, keys, sign) = if minimum {
        (&me.min_w, &me.min_key, 1.0)
    } else {
        (&me.max_w, &me.max_key, -1.0)
    };
    let tree = super::trees_more::prim(&g.succ, Some(w), starts, sign);
    let mut entry: HashMap<(u32, u32), usize> = HashMap::with_capacity(tree.len());
    let wanted: std::collections::HashSet<(u32, u32)> = tree.iter().copied().collect();
    for u in 0..g.n {
        for e in g.succ.range(u) {
            let key = (u as u32, g.succ.targets[e]);
            if wanted.contains(&key) {
                entry.entry(key).or_insert(e);
            }
        }
    }
    let mut out = (Vec::new(), Vec::new(), Vec::new());
    for (u, v) in tree {
        out.0.push(u);
        out.1.push(v);
        out.2.push(keys[entry[&(u, v)]]);
    }
    out
}

// --- Bipartite clustering -------------------------------------------------------------

/// `{u for nbr in G[v] for u in G[nbr]} - {v}` as CPython builds it, with
/// each node's `hash()`. `set - {v}` copies the set and discards `v` when
/// the set has at least 8 elements, and otherwise adds every other element
/// to a new set.
pub fn two_hop_set(succ: &Csr, v: usize, hashes: &[i64], first_hop: Option<&PySet>) -> PySet {
    let mut s = PySet::default();
    match first_hop {
        Some(hop) => {
            for w in hop.iter() {
                for &x in succ.neighbors(w as usize) {
                    s.add(x, hashes);
                }
            }
        }
        None => {
            for &w in succ.neighbors(v) {
                for &x in succ.neighbors(w as usize) {
                    s.add(x, hashes);
                }
            }
        }
    }
    minus_one(s, v as u32, hashes)
}

/// `s - {v}` (CPython's `set_difference` with a one-element set).
pub fn minus_one(s: PySet, v: u32, hashes: &[i64]) -> PySet {
    if (s.len() >> 2) > 1 {
        let mut copy = PySet::default();
        copy.merge(&s, hashes);
        copy.discard(v, hashes);
        copy.after_difference_update(hashes);
        copy
    } else {
        let mut fresh = PySet::default();
        for x in s.iter() {
            if x != v {
                fresh.add(x, hashes);
            }
        }
        fresh
    }
}

/// Bipartite `latapy_clustering` for the nodes `sources`: per node, the
/// sum over its second neighbors (in set order) of `cc(N(u), N(v))`,
/// divided by their number. `mode` 0 dot, 1 min, 2 max.
pub fn latapy(succ: &Csr, n: usize, sources: &[u32], mode: u8, hashes: &[i64]) -> Vec<f64> {
    let mut mark = vec![false; n];
    let mut out = Vec::with_capacity(sources.len());
    for &v in sources {
        let v = v as usize;
        let nv = succ.neighbors(v);
        // `set(G[v])` has each neighbor once (a multigraph row already does).
        for &w in nv {
            mark[w as usize] = true;
        }
        let lv = nv.len();
        let nbrs2 = two_hop_set(succ, v, hashes, None);
        let mut cc = 0.0f64;
        for u in nbrs2.iter() {
            let nu = succ.neighbors(u as usize);
            let common = nu.iter().filter(|&&w| mark[w as usize]).count();
            let lu = nu.len();
            let denom = match mode {
                0 => lu + lv - common,
                1 => lu.min(lv),
                _ => lu.max(lv),
            };
            cc += common as f64 / denom as f64;
        }
        for &w in nv {
            mark[w as usize] = false;
        }
        if cc > 0.0 {
            cc /= nbrs2.len() as f64;
        }
        out.push(cc);
    }
    out
}

/// `(4 times the number of 4-cycles, twice the number of 3-paths)` for
/// `robins_alexander_clustering`, as NetworkX counts them (`_four_cycles`
/// or `butterflies`, and `_threepaths`) on an undirected graph without
/// self-loops.
pub fn cycles_and_paths(succ: &Csr, n: usize) -> (u64, u64) {
    let mut mark = vec![0u32; n];
    let mut stamp = 0u32;
    let mut paths = 0u64;
    // `_threepaths`: for v, u in G[v], w in set(G[u]) - {v}: the size of
    // set(G[w]) - {v, u}, where u is always in G[w].
    for v in 0..n {
        stamp += 1;
        for &u in succ.neighbors(v) {
            mark[u as usize] = stamp;
        }
        for &u in succ.neighbors(v) {
            for &w in succ.neighbors(u as usize) {
                if w as usize == v {
                    continue;
                }
                let k = succ.neighbors(w as usize).len() as u64 - 1;
                paths += if mark[w as usize] == stamp { k - 1 } else { k };
            }
        }
    }
    // Ordered pairs of common neighbors of each node pair, counted once
    // per unordered pair: each 4-cycle twice per diagonal, four in all.
    let mut count = vec![0u64; n];
    let mut cycles2 = 0u64;
    for v in 0..n {
        stamp += 1;
        let mut touched = Vec::new();
        for &u in succ.neighbors(v) {
            for &x in succ.neighbors(u as usize) {
                let x = x as usize;
                if x <= v {
                    continue;
                }
                if mark[x] != stamp {
                    mark[x] = stamp;
                    count[x] = 0;
                    touched.push(x);
                }
                count[x] += 1;
            }
        }
        for x in touched {
            let p = count[x];
            cycles2 += p * (p - 1);
        }
    }
    (cycles2, paths)
}

// --- Eppstein's matching -------------------------------------------------------------

/// An insertion-ordered dict from node to node (a Python dict: assigning
/// an existing key keeps its place).
struct OrderedMap {
    keys: Vec<u32>,
    value: Vec<u32>,
    present: Vec<bool>,
}

impl OrderedMap {
    fn new(n: usize) -> Self {
        OrderedMap {
            keys: Vec::new(),
            value: vec![0; n],
            present: vec![false; n],
        }
    }

    fn set(&mut self, k: u32, v: u32) {
        if !self.present[k as usize] {
            self.present[k as usize] = true;
            self.keys.push(k);
        }
        self.value[k as usize] = v;
    }

    fn get(&self, k: u32) -> Option<u32> {
        self.present[k as usize].then(|| self.value[k as usize])
    }
}

const NO_PRED: u32 = u32::MAX;
const UNMATCHED: u32 = u32::MAX - 1;

/// State of `eppstein_matching`'s `recurse`.
struct Augment<'a> {
    preds: &'a mut Vec<Option<Vec<u32>>>,
    pred: &'a mut [u32],
    matching: &'a mut OrderedMap,
    max_depth: usize,
}

impl Augment<'_> {
    fn recurse(&mut self, v: u32, depth: usize) -> Result<bool, ()> {
        if depth > self.max_depth {
            return Err(());
        }
        if let Some(list) = self.preds[v as usize].take() {
            for u in list {
                let pu = self.pred[u as usize];
                if pu != NO_PRED {
                    self.pred[u as usize] = NO_PRED;
                    if pu == UNMATCHED || self.recurse(pu, depth + 1)? {
                        self.matching.set(v, u);
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }
}

/// `eppstein_matching` on `nx.DiGraph(edges)` (`edges` is `G.edges(left)`
/// as node positions): the matching dict's items in order, or `None` if
/// `recurse` would nest deeper than `max_depth` (left to NetworkX, which
/// may hit Python's recursion limit).
pub fn eppstein(n: usize, edges: &[(u32, u32)], max_depth: usize) -> Option<Vec<(u32, u32)>> {
    // `nx.DiGraph(edges)`: nodes in order of first appearance, rows in
    // insertion order (a repeated edge stays where it was).
    let mut order = Vec::new();
    let mut in_h = vec![false; n];
    let mut rows: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut seen_edge = std::collections::HashSet::new();
    for &(u, v) in edges {
        for x in [u, v] {
            if !in_h[x as usize] {
                in_h[x as usize] = true;
                order.push(x);
            }
        }
        if seen_edge.insert((u, v)) {
            rows[u as usize].push(v);
        }
    }
    let mut matching = OrderedMap::new(n);
    for &u in &order {
        for &v in &rows[u as usize] {
            if matching.get(v).is_none() {
                matching.set(v, u);
                break;
            }
        }
    }
    let mut pred = vec![NO_PRED; n];
    let mut preds: Vec<Option<Vec<u32>>> = vec![None; n];
    let mut in_layer = vec![usize::MAX; n];
    let mut slot_of = vec![0usize; n];
    loop {
        for p in preds.iter_mut() {
            *p = None;
        }
        let mut unmatched: Vec<u32> = Vec::new();
        for &u in &order {
            pred[u as usize] = UNMATCHED;
        }
        for &v in &matching.keys {
            pred[matching.value[v as usize] as usize] = NO_PRED;
        }
        let mut layer: Vec<u32> = order
            .iter()
            .copied()
            .filter(|&u| pred[u as usize] != NO_PRED)
            .collect();
        let mut round = 0usize;
        while !layer.is_empty() && unmatched.is_empty() {
            // `newLayer`: an insertion-ordered dict of lists.
            let mut new_keys: Vec<u32> = Vec::new();
            let mut new_lists: Vec<Vec<u32>> = Vec::new();
            for &u in &layer {
                for &v in &rows[u as usize] {
                    if preds[v as usize].is_none() {
                        if in_layer[v as usize] == round {
                            new_lists[slot_of[v as usize]].push(u);
                        } else {
                            in_layer[v as usize] = round;
                            slot_of[v as usize] = new_keys.len();
                            new_keys.push(v);
                            new_lists.push(vec![u]);
                        }
                    }
                }
            }
            layer = Vec::new();
            for (v, list) in new_keys.into_iter().zip(new_lists) {
                preds[v as usize] = Some(list);
                match matching.get(v) {
                    Some(m) => {
                        layer.push(m);
                        pred[m as usize] = v;
                    }
                    None => unmatched.push(v),
                }
            }
            round += 1;
        }
        for x in in_layer.iter_mut() {
            *x = usize::MAX;
        }
        if unmatched.is_empty() {
            // `for key in matching.copy(): matching[matching[key]] = key`
            // reads the live dict.
            for k in matching.keys.clone() {
                let v = matching.value[k as usize];
                matching.set(v, k);
            }
            return Some(
                matching
                    .keys
                    .iter()
                    .map(|&k| (k, matching.value[k as usize]))
                    .collect(),
            );
        }
        let mut aug = Augment {
            preds: &mut preds,
            pred: &mut pred,
            matching: &mut matching,
            max_depth,
        };
        for &v in &unmatched {
            aug.recurse(v, 1).ok()?;
        }
    }
}

// --- Maximal extendability ---------------------------------------------------------------

/// Why `maximal_extendability` stops before computing anything.
pub enum Extendability {
    NotStronglyConnected,
    Value(i64),
}

/// `maximal_extendability` after its checks: orient each edge (matched
/// edges from `V` to `U`, the rest from `U` to `V`, by NetworkX's rule),
/// check the residual digraph is strongly connected, and take the fewest
/// node-disjoint paths from a node of `U` to a node of `V` (the maximum
/// flow from `uB` to `vA` in the node-split digraph, which is how many
/// paths `node_disjoint_paths` yields). `in_u` marks `U`; `in_v` marks `V`.
pub fn extendability(
    succ: &Csr,
    n: usize,
    in_u: &[bool],
    in_v: &[bool],
    mate: &[u32],
) -> Extendability {
    let mut arcs: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (x, y, _) in undirected_entries(succ, n) {
        let in_pm = |a: u32, b: u32| in_v[a as usize] && mate[a as usize] == b;
        if (in_v[x as usize] && in_pm(x, y)) || (in_u[x as usize] && !in_pm(y, x)) {
            arcs[x as usize].push(y);
        } else {
            arcs[y as usize].push(x);
        }
    }
    let mut offsets = Vec::with_capacity(n + 1);
    offsets.push(0);
    let mut targets = Vec::new();
    for row in &arcs {
        targets.extend_from_slice(row);
        offsets.push(targets.len());
    }
    let residual = Csr { offsets, targets };
    let sccs = super::directed::strongly_connected_components(&residual, n, true);
    if n == 0 || sccs.len() != 1 || sccs[0].len() != n {
        return Extendability::NotStronglyConnected;
    }
    let aux = super::connectivity::node_aux(&residual, n, true);
    let us: Vec<u32> = (0..n as u32).filter(|&u| in_u[u as usize]).collect();
    let vs: Vec<u32> = (0..n as u32).filter(|&v| in_v[v as usize]).collect();
    let mut best = i64::MAX;
    for &u in &us {
        let pairs: Vec<(u32, u32)> = vs.iter().map(|&v| (2 * u + 1, 2 * v)).collect();
        let flows = super::connectivity::pair_flows(&aux, &pairs, f64::INFINITY)
            .unwrap_or_else(|_| unreachable!("unit capacities"));
        if let Some(&m) = flows.iter().min() {
            best = best.min(m);
        }
    }
    Extendability::Value(best)
}

// --- Weighted projections -----------------------------------------------------------------

/// Which weighted projection to build.
#[derive(Clone, Copy)]
pub enum Projection {
    /// `overlap_weighted_projected_graph`: Jaccard (`true`) or min overlap.
    Overlap(bool),
    /// `collaboration_weighted_projected_graph`.
    Collaboration,
}

/// The edges `G.add_edge(u, v, weight=w)` of a weighted projection, in the
/// order NetworkX adds them: for each of `members` (positions of `nodes`),
/// its second neighbors in the iteration order of the set NetworkX builds,
/// each with its weight. `pred` is `B.pred` (`B.adj` when undirected) in
/// NetworkX's row order; `compensated` is Python 3.12+'s float `sum()`.
pub fn projection_edges(
    succ: &Csr,
    pred: &Csr,
    n: usize,
    members: &[u32],
    kind: Projection,
    hashes: &[i64],
    compensated: bool,
) -> PyResult<Vec<(u32, u32, super::flow::Val)>> {
    use super::flow::Val;
    let mut count = vec![0u64; n];
    let mut pred_sets: Vec<Option<PySet>> = vec![None; n];
    let mut out = Vec::new();
    for &u in members {
        let u = u as usize;
        // `unbrs = set(B[u])`, added one neighbor at a time.
        let mut unbrs = PySet::default();
        for &w in succ.neighbors(u) {
            unbrs.add(w, hashes);
        }
        let mut second = PySet::default();
        for w in unbrs.iter() {
            for &x in succ.neighbors(w as usize) {
                count[x as usize] += 1;
                if matches!(kind, Projection::Overlap(_)) || x as usize != u {
                    second.add(x, hashes);
                }
            }
        }
        if matches!(kind, Projection::Overlap(_)) {
            second = minus_one(second, u as u32, hashes);
        }
        let lu = unbrs.len() as u64;
        for v in second.iter() {
            let vi = v as usize;
            let common = count[vi];
            let w = match kind {
                Projection::Overlap(jaccard) => {
                    let lv = pred.neighbors(vi).len() as u64;
                    let denom = if jaccard {
                        lu + lv - common
                    } else {
                        lu.min(lv)
                    };
                    Val::F(common as f64 / denom as f64)
                }
                Projection::Collaboration => {
                    // `vnbrs = set(pred[v])`; `unbrs & vnbrs` walks the
                    // smaller set (`vnbrs` on a tie) in its order.
                    let vnbrs = pred_sets[vi].get_or_insert_with(|| {
                        let mut s = PySet::default();
                        for &x in pred.neighbors(vi) {
                            s.add(x, hashes);
                        }
                        s
                    });
                    let (walk, other) = if unbrs.len() > vnbrs.len() {
                        (&*vnbrs, &unbrs)
                    } else if vnbrs.len() > unbrs.len() {
                        (&unbrs, &*vnbrs)
                    } else {
                        (&*vnbrs, &unbrs)
                    };
                    let mut both = PySet::default();
                    for x in walk.iter() {
                        if other.contains(x, hashes) {
                            both.add(x, hashes);
                        }
                    }
                    let terms = both.iter().filter_map(|x| {
                        let deg = succ.neighbors(x as usize).len();
                        (deg > 1).then(|| Val::F(1.0 / (deg - 1) as f64))
                    });
                    super::flow::py_sum(terms, compensated)
                        .map_err(|_| PyNotImplementedError::new_err("sum not reproduced"))?
                }
            };
            out.push((u as u32, v, w));
        }
        for w in unbrs.iter() {
            for &x in succ.neighbors(w as usize) {
                count[x as usize] = 0;
            }
        }
    }
    Ok(out)
}
