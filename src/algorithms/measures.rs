//! Batch 14: assortativity, link prediction and reciprocity.
//!
//! Link prediction scores that add floats over `nx.common_neighbors(G, u, v)`
//! (a Python set) depend on the set's iteration order, so this module
//! replays CPython's set table (`Objects/setobject.c`) on node positions,
//! with each node's Python hash. The Python side checks once that the
//! replay agrees with the running interpreter before relying on it.

use std::collections::HashMap;

use rayon::prelude::*;

use super::spectral::py_sum;
use crate::graph::Csr;

const EMPTY: u32 = u32::MAX;
const DUMMY: u32 = u32::MAX - 1;
const LINEAR_PROBES: usize = 9;
const PERTURB_SHIFT: u32 = 5;
const SET_MINSIZE: usize = 8;

/// A CPython set of node positions: the same table, probe sequence, resize
/// rule and dummy entries, so it iterates in the same order.
struct PySet<'a> {
    hashes: &'a [i64],
    table: Vec<u32>,
    mask: usize,
    fill: usize,
    used: usize,
}

impl<'a> PySet<'a> {
    fn new(hashes: &'a [i64]) -> Self {
        PySet {
            hashes,
            table: vec![EMPTY; SET_MINSIZE],
            mask: SET_MINSIZE - 1,
            fill: 0,
            used: 0,
        }
    }

    #[inline]
    fn hash(&self, key: u32) -> u64 {
        // `(size_t)hash`
        self.hashes[key as usize] as u64
    }

    /// `set_add_entry` for a key no other element equals.
    fn add(&mut self, key: u32) {
        let hash = self.hash(key);
        let mask = self.mask;
        let mut i = hash as usize & mask;
        let mut perturb = hash;
        let mut freeslot = None;
        loop {
            let probes = if i + LINEAR_PROBES <= mask {
                LINEAR_PROBES
            } else {
                0
            };
            for j in i..=i + probes {
                let k = self.table[j];
                if k == EMPTY {
                    if let Some(slot) = freeslot {
                        self.used += 1;
                        self.table[slot] = key;
                        return;
                    }
                    self.fill += 1;
                    self.used += 1;
                    self.table[j] = key;
                    if self.fill * 5 >= mask * 3 {
                        let minused = if self.used > 50000 {
                            self.used * 2
                        } else {
                            self.used * 4
                        };
                        self.resize(minused);
                    }
                    return;
                }
                if k == key {
                    return;
                }
                if k == DUMMY {
                    freeslot = Some(j);
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i
                .wrapping_mul(5)
                .wrapping_add(1)
                .wrapping_add(perturb as usize))
                & mask;
        }
    }

    /// `set_insert_clean`: into a table with no dummies and no equal key.
    fn insert_clean(table: &mut [u32], mask: usize, key: u32, hash: u64) {
        let mut i = hash as usize & mask;
        let mut perturb = hash;
        loop {
            if table[i] == EMPTY {
                table[i] = key;
                return;
            }
            if i + LINEAR_PROBES <= mask {
                if let Some(slot) = table[i + 1..=i + LINEAR_PROBES]
                    .iter_mut()
                    .find(|slot| **slot == EMPTY)
                {
                    *slot = key;
                    return;
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i
                .wrapping_mul(5)
                .wrapping_add(1)
                .wrapping_add(perturb as usize))
                & mask;
        }
    }

    /// `set_table_resize`.
    fn resize(&mut self, minused: usize) {
        let mut newsize = SET_MINSIZE;
        while newsize <= minused {
            newsize <<= 1;
        }
        if newsize == SET_MINSIZE && self.mask == SET_MINSIZE - 1 && self.fill == self.used {
            return; // the small table, without dummies: nothing to do
        }
        let old = std::mem::replace(&mut self.table, vec![EMPTY; newsize]);
        self.mask = newsize - 1;
        self.fill = self.used;
        for key in old {
            if key != EMPTY && key != DUMMY {
                let hash = self.hash(key);
                Self::insert_clean(&mut self.table, self.mask, key, hash);
            }
        }
    }

    /// `set(d)` for a dict `d` with these keys, in order (presized).
    fn from_dict(hashes: &'a [i64], keys: &[u32]) -> Self {
        let mut s = PySet::new(hashes);
        if (s.fill + keys.len()) * 5 >= s.mask * 3 {
            s.resize((s.used + keys.len()) * 2);
        }
        for &k in keys {
            s.add(k);
        }
        s
    }

    /// `set_discard_entry`.
    fn discard(&mut self, key: u32) {
        let hash = self.hash(key);
        let mask = self.mask;
        let mut i = hash as usize & mask;
        let mut perturb = hash;
        loop {
            let probes = if i + LINEAR_PROBES <= mask {
                LINEAR_PROBES
            } else {
                0
            };
            for j in i..=i + probes {
                let k = self.table[j];
                if k == EMPTY {
                    return;
                }
                if k == key {
                    self.table[j] = DUMMY;
                    self.used -= 1;
                    return;
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i
                .wrapping_mul(5)
                .wrapping_add(1)
                .wrapping_add(perturb as usize))
                & mask;
        }
    }

    /// The end of `set_difference_update_internal`: purge many dummies.
    fn after_difference_update(&mut self) {
        if self.fill - self.used > self.mask / 4 {
            let minused = if self.used > 50000 {
                self.used * 2
            } else {
                self.used * 4
            };
            self.resize(minused);
        }
    }

    fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.table
            .iter()
            .copied()
            .filter(|&k| k != EMPTY && k != DUMMY)
    }
}

/// The iteration order of `G._adj[u].keys() & G._adj[v].keys() - {u, v}`
/// (NetworkX's `common_neighbors`), given its elements in `adj[u]` order
/// and a membership test for `adj[u]`.
///
/// `keys_v - {u, v}` is `set(G._adj[v])` minus `u` and `v`. Then
/// `keys_u & S` intersects through `S.intersection(keys_u)` (walking `keys_u`
/// in dict order) when `len(keys_u) <= len(S)`, and otherwise walks `S` in
/// table order; either way the result is a new set filled in that order.
fn set_order(
    adj: &Csr,
    hashes: &[i64],
    u: usize,
    v: usize,
    cn: &[u32],
    in_u: impl Fn(u32) -> bool,
) -> Vec<u32> {
    let nu = adj.neighbors(u);
    let nv = adj.neighbors(v);
    let has = |row: &[u32], x: usize| row.contains(&(x as u32));
    let mut len_s = nv.len() - has(nv, u) as usize;
    if u != v {
        len_s -= has(nv, v) as usize;
    }
    let mut result = PySet::new(hashes);
    if nu.len() <= len_s {
        for &w in cn {
            result.add(w);
        }
    } else {
        let mut s = PySet::from_dict(hashes, nv);
        s.discard(u as u32);
        if u != v {
            s.discard(v as u32);
        }
        s.after_difference_update();
        for w in s.iter() {
            if in_u(w) {
                result.add(w);
            }
        }
    }
    result.iter().collect()
}

/// Outcome codes of `link_scores`, per pair.
pub mod code {
    /// `f` holds the float score (or `a`, `b` the counts).
    pub const OK: u8 = 0;
    /// The score is the int 0 (an empty `sum()`, or different communities).
    pub const INT_ZERO: u8 = 1;
    /// `u` has no community.
    pub const NO_COMMUNITY_U: u8 = 2;
    /// `v` has no community.
    pub const NO_COMMUNITY_V: u8 = 3;
    /// Common neighbor `w` (in `w`) is the first without a community.
    pub const NO_COMMUNITY_W: u8 = 4;
    /// A term divides by zero (`1 / log(1)` in the Adamic-Adar index).
    pub const ZERO_DIVISION: u8 = 5;
}

/// What `link_scores` computes per pair.
#[derive(Clone, Copy, PartialEq)]
pub enum LinkMode {
    /// `a` = common neighbors, `b` = size of the neighborhoods' union.
    Counts,
    /// Communities: `a` = common neighbors, `b` = those in u's community,
    /// `f` = 1.0 if u and v share a community (else 0.0).
    Community,
    /// `f` = `sum(table[degree[w]] for w in common neighbors)` in set
    /// order (only those in u's community when classes are given).
    Sum,
}

pub struct LinkInput<'a> {
    pub adj: &'a Csr,
    pub n: usize,
    pub hashes: &'a [i64],
    pub degree: &'a [usize],
    pub table: &'a [f64],
    /// Community class per node, negative for none.
    pub classes: Option<&'a [i64]>,
    pub compensated: bool,
}

#[derive(Default)]
pub struct LinkScores {
    pub codes: Vec<u8>,
    pub a: Vec<i64>,
    pub b: Vec<i64>,
    pub f: Vec<f64>,
    pub w: Vec<u32>,
}

struct Marks {
    v: Vec<u32>,
    u: Vec<u32>,
    stamp: u32,
}

impl Marks {
    fn new(n: usize) -> Self {
        Marks {
            v: vec![0; n],
            u: vec![0; n],
            stamp: 0,
        }
    }

    fn next(&mut self) -> u32 {
        if self.stamp == u32::MAX {
            self.v.fill(0);
            self.u.fill(0);
            self.stamp = 0;
        }
        self.stamp += 1;
        self.stamp
    }
}

type PairScore = (u8, i64, i64, f64, u32);

/// `cn` (common neighbors in `adj[u]` order, `marks.v` set for `adj[v]`
/// with stamp `s`) in NetworkX's set order.
fn in_set_order(
    inp: &LinkInput,
    marks: &mut Marks,
    s: u32,
    u: usize,
    v: usize,
    cn: &[u32],
) -> Vec<u32> {
    for &x in inp.adj.neighbors(u) {
        marks.u[x as usize] = s;
    }
    let mu = &marks.u;
    set_order(inp.adj, inp.hashes, u, v, cn, |w| mu[w as usize] == s)
}

fn score_pair(inp: &LinkInput, mode: LinkMode, marks: &mut Marks, u: usize, v: usize) -> PairScore {
    let adj = inp.adj;
    let s = marks.next();
    for &x in adj.neighbors(v) {
        marks.v[x as usize] = s;
    }
    let nu = adj.neighbors(u);
    let mut inter = 0usize;
    let mut cn = Vec::new();
    for &w in nu {
        if marks.v[w as usize] == s {
            inter += 1;
            if w as usize != u && w as usize != v {
                cn.push(w);
            }
        }
    }
    if mode == LinkMode::Counts {
        let union = nu.len() + adj.neighbors(v).len() - inter;
        return (code::OK, cn.len() as i64, union as i64, 0.0, 0);
    }
    let mut community = None;
    if let Some(classes) = inp.classes {
        let (cu, cv) = (classes[u], classes[v]);
        if cu < 0 {
            return (code::NO_COMMUNITY_U, 0, 0, 0.0, 0);
        }
        if cv < 0 {
            return (code::NO_COMMUNITY_V, 0, 0, 0.0, 0);
        }
        if cu != cv {
            return match mode {
                LinkMode::Community => (code::OK, cn.len() as i64, 0, 0.0, 0),
                _ => (code::INT_ZERO, 0, 0, 0.0, 0),
            };
        }
        let missing = cn.iter().filter(|&&w| classes[w as usize] < 0).count();
        if missing > 0 {
            // NetworkX names the first one it meets, in set order.
            let first = if missing == 1 {
                *cn.iter().find(|&&w| classes[w as usize] < 0).unwrap()
            } else {
                let ordered = in_set_order(inp, marks, s, u, v, &cn);
                ordered
                    .into_iter()
                    .find(|&w| classes[w as usize] < 0)
                    .unwrap()
            };
            return (code::NO_COMMUNITY_W, 0, 0, 0.0, first);
        }
        let same = cn.iter().filter(|&&w| classes[w as usize] == cu).count();
        if mode == LinkMode::Community {
            return (code::OK, cn.len() as i64, same as i64, 1.0, 0);
        }
        community = Some((classes, cu));
    }
    // LinkMode::Sum
    let keep = |w: u32| community.is_none_or(|(classes, cu)| classes[w as usize] == cu);
    let value = |w: u32| inp.table[inp.degree[w as usize]];
    let terms: Vec<u32> = cn.iter().copied().filter(|&w| keep(w)).collect();
    if terms.is_empty() {
        return (code::INT_ZERO, 0, 0, 0.0, 0);
    }
    if terms.iter().any(|&w| value(w).is_nan()) {
        return (code::ZERO_DIVISION, 0, 0, 0.0, 0);
    }
    let first = value(terms[0]);
    // Sums of up to two terms, or of equal terms, don't depend on the order.
    let total = if terms.len() <= 2 || terms.iter().all(|&w| value(w) == first) {
        py_sum(terms.iter().map(|&w| value(w)), inp.compensated)
    } else {
        let ordered = in_set_order(inp, marks, s, u, v, &cn);
        py_sum(
            ordered.into_iter().filter(|&w| keep(w)).map(value),
            inp.compensated,
        )
    };
    (code::OK, terms.len() as i64, 0, total, 0)
}

/// Scores of the pairs `(us[i], vs[i])`; see `LinkMode` and `code`.
pub fn link_scores(inp: &LinkInput, mode: LinkMode, us: &[u32], vs: &[u32]) -> LinkScores {
    let rows: Vec<PairScore> = us
        .par_iter()
        .zip(vs.par_iter())
        .with_min_len(256)
        .map_init(
            || Marks::new(inp.n),
            |marks, (&u, &v)| score_pair(inp, mode, marks, u as usize, v as usize),
        )
        .collect();
    let mut out = LinkScores::default();
    for (c, a, b, f, w) in rows {
        out.codes.push(c);
        out.a.push(a);
        out.b.push(b);
        out.f.push(f);
        out.w.push(w);
    }
    out
}

/// `list(G._adj[u].keys() & G._adj[v].keys() - {u, v})` per pair, from the
/// set replay (for the Python-side check that it matches the interpreter).
pub fn common_neighbors_in_set_order(
    adj: &Csr,
    n: usize,
    hashes: &[i64],
    us: &[u32],
    vs: &[u32],
) -> Vec<Vec<u32>> {
    let mut marks = Marks::new(n);
    us.iter()
        .zip(vs)
        .map(|(&u, &v)| {
            let (u, v) = (u as usize, v as usize);
            let s = marks.next();
            for &x in adj.neighbors(v) {
                marks.v[x as usize] = s;
            }
            for &x in adj.neighbors(u) {
                marks.u[x as usize] = s;
            }
            let cn: Vec<u32> = adj
                .neighbors(u)
                .iter()
                .copied()
                .filter(|&w| marks.v[w as usize] == s && w as usize != u && w as usize != v)
                .collect();
            let mu = &marks.u;
            set_order(adj, hashes, u, v, &cn, |w| mu[w as usize] == s)
        })
        .collect()
}

/// Unweighted distance from `us[i]` to `vs[i]`, or -1 if unreachable.
/// Pairs are grouped by source; each search stops once its targets are found.
pub fn pair_distances(adj: &Csr, n: usize, us: &[u32], vs: &[u32]) -> Vec<i64> {
    let mut groups: HashMap<u32, Vec<usize>> = HashMap::new();
    let mut sources = Vec::new();
    for (i, &u) in us.iter().enumerate() {
        groups
            .entry(u)
            .or_insert_with(|| {
                sources.push(u);
                Vec::new()
            })
            .push(i);
    }
    let found: Vec<Vec<(usize, i64)>> = sources
        .par_iter()
        .map_init(
            || (vec![u32::MAX; n], Vec::new(), vec![0u32; n], 0u32),
            |(dist, touched, want, stamp), &s| {
                *stamp += 1;
                let idx = &groups[&s];
                let mut left = 0usize;
                for &i in idx {
                    let t = vs[i] as usize;
                    if want[t] != *stamp {
                        want[t] = *stamp;
                        left += 1;
                    }
                }
                let mut queue = std::collections::VecDeque::new();
                dist[s as usize] = 0;
                touched.push(s);
                queue.push_back(s);
                if want[s as usize] == *stamp {
                    left -= 1;
                }
                while left > 0 {
                    let Some(x) = queue.pop_front() else { break };
                    let d = dist[x as usize] + 1;
                    for &y in adj.neighbors(x as usize) {
                        if dist[y as usize] == u32::MAX {
                            dist[y as usize] = d;
                            touched.push(y);
                            queue.push_back(y);
                            if want[y as usize] == *stamp {
                                left -= 1;
                            }
                        }
                    }
                }
                let out = idx
                    .iter()
                    .map(|&i| {
                        let d = dist[vs[i] as usize];
                        (i, if d == u32::MAX { -1 } else { d as i64 })
                    })
                    .collect();
                for &x in touched.iter() {
                    dist[x as usize] = u32::MAX;
                }
                touched.clear();
                out
            },
        )
        .collect();
    let mut result = vec![-1i64; us.len()];
    for group in found {
        for (i, d) in group {
            result[i] = d;
        }
    }
    result
}

/// The `(x, y)` pairs of `node_degree_xy` / `node_attribute_xy` as node
/// positions: for each `u` in `order`, its neighbors in `adj` order, kept
/// if `nbr_mask` (when given) allows them.
pub fn xy_pairs(adj: &Csr, order: &[u32], nbr_mask: Option<&[bool]>) -> (Vec<u32>, Vec<u32>) {
    let mut us = Vec::new();
    let mut vs = Vec::new();
    for &u in order {
        for &v in adj.neighbors(u as usize) {
            if nbr_mask.is_none_or(|m| m[v as usize]) {
                us.push(u);
                vs.push(v);
            }
        }
    }
    (us, vs)
}

/// One outer key of a mixing dict: its class, the node whose value became
/// the key, and its inner `(class, node, count)` entries in insertion order.
pub type MixingRow = (i64, u32, Vec<(i64, u32, u64)>);

/// NetworkX's `mixing_dict` over the `xy_pairs` stream, with each node's x
/// and y value given as a class: the number of pairs, and the outer keys in
/// insertion order (`x` then `y` of each pair, when new).
pub fn mixing(
    adj: &Csr,
    order: &[u32],
    nbr_mask: Option<&[bool]>,
    xcls: &[i64],
    ycls: &[i64],
) -> (u64, Vec<MixingRow>) {
    let mut outer: HashMap<i64, usize> = HashMap::new();
    let mut rows: Vec<MixingRow> = Vec::new();
    let mut inner: Vec<HashMap<i64, usize>> = Vec::new();
    let mut total = 0u64;
    let mut key =
        |c: i64, node: u32, rows: &mut Vec<MixingRow>, inner: &mut Vec<HashMap<i64, usize>>| {
            *outer.entry(c).or_insert_with(|| {
                rows.push((c, node, Vec::new()));
                inner.push(HashMap::new());
                rows.len() - 1
            })
        };
    for &u in order {
        for &v in adj.neighbors(u as usize) {
            if !nbr_mask.is_none_or(|m| m[v as usize]) {
                continue;
            }
            let (x, y) = (xcls[u as usize], ycls[v as usize]);
            let ix = key(x, u, &mut rows, &mut inner);
            key(y, v, &mut rows, &mut inner);
            let entries = &mut rows[ix].2;
            let slot = *inner[ix].entry(y).or_insert_with(|| {
                entries.push((y, v, 0));
                entries.len() - 1
            });
            entries[slot].2 += 1;
            total += 1;
        }
    }
    (total, rows)
}

/// Per-node sum of edge weights (or counts) over `adj` rows.
pub fn row_sums(adj: &Csr, n: usize, weights: Option<&[f64]>) -> Vec<i64> {
    (0..n)
        .map(|v| match weights {
            None => adj.range(v).len() as i64,
            Some(w) => adj.range(v).map(|e| w[e] as i64).sum(),
        })
        .collect()
}

/// `average_neighbor_degree` terms for `nodes`: the numerator (the sum over
/// `succ` rows, then `pred` rows, of weight times target degree) and the
/// source degree. `None` on integer overflow.
pub fn neighbor_degree_terms(
    rows: &[(&Csr, Option<&[f64]>)],
    target_degree: &[i64],
    source_degree: &[i64],
    nodes: &[u32],
) -> Option<Vec<(i64, i64)>> {
    nodes
        .iter()
        .map(|&v| {
            let v = v as usize;
            let mut total = 0i64;
            for (adj, w) in rows {
                for e in adj.range(v) {
                    let t = target_degree[adj.targets[e] as usize];
                    let term = match w {
                        None => t,
                        Some(w) => (w[e] as i64).checked_mul(t)?,
                    };
                    total = total.checked_add(term)?;
                }
            }
            Some((total, source_degree[v]))
        })
        .collect()
}

/// `average_degree_connectivity` sums: per source degree `k` (first-seen
/// order over `nodes`, repeats included), the sum over neighbors (`adj`
/// rows) of weight times target degree, and the sum of weighted source
/// degrees. `None` on integer overflow.
#[allow(clippy::type_complexity)]
pub fn degree_connectivity(
    adj: &Csr,
    weights: Option<&[f64]>,
    k_degree: &[i64],
    target_degree: &[i64],
    weighted_source: &[i64],
    nodes: &[u32],
) -> Option<(Vec<i64>, Vec<i64>, Vec<i64>)> {
    let mut slot: HashMap<i64, usize> = HashMap::new();
    let (mut keys, mut sums, mut norms): (Vec<i64>, Vec<i64>, Vec<i64>) =
        (Vec::new(), Vec::new(), Vec::new());
    for &v in nodes {
        let v = v as usize;
        let mut s = 0i64;
        for e in adj.range(v) {
            let t = target_degree[adj.targets[e] as usize];
            let term = match weights {
                None => t,
                Some(w) => (w[e] as i64).checked_mul(t)?,
            };
            s = s.checked_add(term)?;
        }
        let k = k_degree[v];
        let i = *slot.entry(k).or_insert_with(|| {
            keys.push(k);
            sums.push(0);
            norms.push(0);
            keys.len() - 1
        });
        norms[i] = norms[i].checked_add(weighted_source[v])?;
        sums[i] = sums[i].checked_add(s)?;
    }
    Some((keys, sums, norms))
}

/// `(len(pred & succ), len(pred) + len(succ))` per node.
pub fn reciprocity_counts(succ: &Csr, pred: &Csr, n: usize, nodes: &[u32]) -> Vec<(u64, u64)> {
    let mut mark = vec![u32::MAX; n];
    nodes
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let v = v as usize;
            for &x in succ.neighbors(v) {
                mark[x as usize] = i as u32;
            }
            let overlap = pred
                .neighbors(v)
                .iter()
                .filter(|&&x| mark[x as usize] == i as u32)
                .count();
            // Reset so a repeated node can't see stale marks of its own index.
            for &x in succ.neighbors(v) {
                mark[x as usize] = u32::MAX;
            }
            (
                overlap as u64,
                (succ.neighbors(v).len() + pred.neighbors(v).len()) as u64,
            )
        })
        .collect()
}

/// Directed edges `u -> v` (u != v) whose reverse edge exists.
pub fn reciprocated_edges(succ: &Csr, n: usize) -> u64 {
    (0..n)
        .into_par_iter()
        .map(|u| {
            succ.neighbors(u)
                .iter()
                .filter(|&&v| v as usize != u && succ.neighbors(v as usize).contains(&(u as u32)))
                .count() as u64
        })
        .sum()
}

/// NetworkX's `_compute_rc` as `(nk, ek)` per degree `d = 0, 1, ...`: the
/// nodes of degree above `d` (while more than one) and the edges whose
/// endpoints both have degree above `d`.
pub fn rich_club_counts(adj: &Csr, n: usize, degree: &[usize]) -> Vec<(u64, u64)> {
    let max_deg = degree.iter().copied().max().unwrap_or(0);
    let mut hist = vec![0u64; max_deg + 1];
    for &d in degree {
        hist[d] += 1;
    }
    // Edges by the smaller endpoint degree.
    let mut by_min = vec![0u64; max_deg + 1];
    let mut m = 0u64;
    for u in 0..n {
        for &v in adj.neighbors(u) {
            if v as usize >= u {
                by_min[degree[u].min(degree[v as usize])] += 1;
                m += 1;
            }
        }
    }
    let total = n as u64;
    let mut out = Vec::new();
    let (mut cs, mut removed) = (0u64, 0u64);
    for d in 0..=max_deg {
        cs += hist[d];
        let nk = total - cs;
        if nk <= 1 {
            break;
        }
        removed += by_min[d];
        out.push((nk, m - removed));
    }
    out
}

/// `sum(G.degree(u) * G.degree(v) for u, v in G.edges())`.
pub fn s_metric(adj: &Csr, n: usize, directed: bool, degree: &[usize]) -> u128 {
    (0..n)
        .into_par_iter()
        .map(|u| {
            adj.neighbors(u)
                .iter()
                .filter(|&&v| directed || v as usize >= u)
                .map(|&v| degree[u] as u128 * degree[v as usize] as u128)
                .sum::<u128>()
        })
        .sum()
}

/// Rows of the `k`-th power of the adjacency matrix (`A[u][v] = 1` per edge
/// `u -> v`), in int64 arithmetic that wraps as NumPy and SciPy's does.
pub fn walk_counts(adj: &Csr, n: usize, k: u64) -> Vec<Vec<i64>> {
    (0..n)
        .into_par_iter()
        .map_init(
            || vec![0i64; n],
            |next, u| {
                let mut x = vec![0i64; n];
                x[u] = 1;
                for _ in 0..k {
                    next.fill(0);
                    for (w, &c) in x.iter().enumerate() {
                        if c != 0 {
                            for &v in adj.neighbors(w) {
                                next[v as usize] = next[v as usize].wrapping_add(c);
                            }
                        }
                    }
                    std::mem::swap(&mut x, next);
                }
                x
            },
        )
        .collect()
}
