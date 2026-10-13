//! NetworkX's Louvain community detection (`louvain_partitions`), level by
//! level, with Python's own arithmetic: ints and floats as Python mixes
//! them, and `sum()` as CPython adds (compensated for floats from 3.12, and
//! for ints after the first float from 3.14).
//! Two variants: NetworkX 3.4 to 3.6, and 3.7 (rescaled gains, a separate
//! `gamma`). The Python side replays the set operations (so the yielded
//! sets are NetworkX's own, with the same history) and hands back the
//! iteration order of the set copies `modularity` sums over.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::pyrandom::Mt19937;

/// A Python int or float.
#[derive(Clone, Copy, Debug)]
pub enum Num {
    I(i128),
    F(f64),
}

const EXACT: i128 = 1 << 53;
const INT_LIMIT: i128 = 1 << 62;

impl Num {
    #[inline]
    fn float(self) -> f64 {
        match self {
            // Through i64 when it fits: a single conversion instruction
            // (the i128 one is a library call); the same rounding.
            Num::I(i) => match i64::try_from(i) {
                Ok(x) => x as f64,
                Err(_) => i as f64,
            },
            Num::F(f) => f,
        }
    }

    fn int_ok(i: i128) -> Option<Num> {
        (i.abs() < INT_LIMIT).then_some(Num::I(i))
    }

    // With the bounds checked when a run starts (int weights, degrees and
    // resolution below 2^31 in total), no int expression here can overflow
    // an i128, so add/sub/mul are Python's exactly.
    #[inline]
    pub fn add(self, o: Num) -> Num {
        match (self, o) {
            (Num::F(a), Num::F(b)) => Num::F(a + b),
            (Num::I(a), Num::I(b)) => Num::I(a + b),
            _ => Num::F(self.float() + o.float()),
        }
    }

    #[inline]
    pub fn sub(self, o: Num) -> Num {
        match (self, o) {
            (Num::F(a), Num::F(b)) => Num::F(a - b),
            (Num::I(a), Num::I(b)) => Num::I(a - b),
            _ => Num::F(self.float() - o.float()),
        }
    }

    #[inline]
    pub fn mul(self, o: Num) -> Num {
        match (self, o) {
            (Num::F(a), Num::F(b)) => Num::F(a * b),
            (Num::I(a), Num::I(b)) => Num::I(a * b),
            _ => Num::F(self.float() * o.float()),
        }
    }

    /// `/` (true division); `None` for a zero divisor (Python raises) or
    /// ints too large to divide as floats exactly.
    #[inline]
    pub fn div(self, o: Num) -> Option<Num> {
        match (self, o) {
            (_, Num::I(0)) => None,
            (_, Num::F(0.0)) => None,
            (Num::I(a), Num::I(b)) => {
                (a.abs() <= EXACT && b.abs() <= EXACT).then(|| Num::F(a as f64 / b as f64))
            }
            _ => Some(Num::F(self.float() / o.float())),
        }
    }

    /// `x // 2` for a non-negative int.
    fn floor_half(self) -> Option<Num> {
        match self {
            Num::I(a) => Some(Num::I(a.div_euclid(2))),
            Num::F(_) => None,
        }
    }

    pub fn neg(self) -> Num {
        match self {
            Num::I(a) => Num::I(-a),
            Num::F(f) => Num::F(-f),
        }
    }

    /// `x ** 2`: for floats, C's `pow(x, 2.0)` as CPython calls it (which
    /// can differ from `x * x` in the last bit); the exponent is opaque so
    /// the compiler keeps the libm call.
    pub fn sq(self) -> Option<Num> {
        match self {
            Num::I(a) => Some(Num::I(a * a)),
            Num::F(f) => Some(Num::F(f.powf(std::hint::black_box(2.0)))),
        }
    }

    /// Python's exact `<` between ints and floats (`None` for an int too
    /// large to compare as a float exactly).
    #[inline]
    fn lt(self, o: Num) -> Option<bool> {
        match (self, o) {
            (Num::I(a), Num::I(b)) => Some(a < b),
            (Num::F(a), Num::F(b)) => Some(a < b),
            (Num::I(a), Num::F(b)) => (a.abs() <= EXACT).then_some((a as f64) < b),
            (Num::F(a), Num::I(b)) => (b.abs() <= EXACT).then_some(a < b as f64),
        }
    }

    #[inline]
    pub fn gt(self, o: Num) -> Option<bool> {
        o.lt(self)
    }

    pub fn le(self, o: Num) -> Option<bool> {
        match (self, o) {
            (Num::F(a), _) if a.is_nan() => Some(false),
            (_, Num::F(b)) if b.is_nan() => Some(false),
            _ => o.lt(self).map(|x| !x),
        }
    }

    fn is_zero(self) -> bool {
        match self {
            Num::I(a) => a == 0,
            Num::F(f) => f == 0.0,
        }
    }
}

/// CPython's builtin `sum()` over ints and floats, from the start value 0:
/// ints add exactly; the first float switches to float adding, compensated
/// (Neumaier) from Python 3.12 (`compensated`). Ints added in float mode
/// are converted and added plainly, or compensated too from Python 3.14
/// (`flow::SUM_INTS_COMPENSATED`). Partial int sums are kept within a C
/// long on every platform (2^31), else the caller declines.
pub struct PySum {
    compensated: bool,
    ints_compensated: bool,
    int: Option<i128>,
    f: f64,
    c: f64,
    ok: bool,
}

impl PySum {
    pub fn new(compensated: bool) -> Self {
        PySum {
            compensated,
            ints_compensated: compensated
                && super::flow::SUM_INTS_COMPENSATED.load(std::sync::atomic::Ordering::Relaxed),
            int: Some(0),
            f: 0.0,
            c: 0.0,
            ok: true,
        }
    }

    pub fn push(&mut self, x: Num) {
        match (self.int, x) {
            (Some(i), Num::I(v)) => {
                let r = i + v;
                if r.abs() >= (1 << 31) || v.abs() >= (1 << 31) {
                    self.ok = false;
                }
                self.int = Some(r);
            }
            (Some(i), Num::F(v)) => {
                self.int = None;
                self.f = i as f64 + v;
                self.c = 0.0;
            }
            (None, Num::F(x)) => self.add_float(x, self.compensated),
            (None, Num::I(v)) => {
                if v.abs() >= (1 << 31) {
                    self.ok = false;
                }
                self.add_float(v as f64, self.ints_compensated);
            }
        }
    }

    fn add_float(&mut self, x: f64, compensated: bool) {
        if compensated {
            let t = self.f + x;
            if self.f.abs() >= x.abs() {
                self.c += (self.f - t) + x;
            } else {
                self.c += (x - t) + self.f;
            }
            self.f = t;
        } else {
            self.f += x;
        }
    }

    pub fn finish(self) -> Option<Num> {
        if !self.ok {
            return None;
        }
        Some(match self.int {
            Some(i) => Num::I(i),
            None => {
                if self.compensated && self.c != 0.0 && self.c.is_finite() {
                    Num::F(self.f + self.c)
                } else {
                    Num::F(self.f)
                }
            }
        })
    }
}

fn py_sum(values: impl IntoIterator<Item = Num>, compensated: bool) -> Option<Num> {
    let mut s = PySum::new(compensated);
    for v in values {
        s.push(v);
    }
    s.finish()
}

/// A simple graph's rows in NetworkX's dict order, with each edge's weight.
#[derive(Clone)]
struct Level {
    directed: bool,
    succ: Vec<Vec<(u32, Num)>>,
    pred: Vec<Vec<(u32, Num)>>,
}

impl Level {
    fn n(&self) -> usize {
        self.succ.len()
    }

    /// `G.add_edge(u, v, weight=w)`: the weight replaced in place, or the
    /// edge appended to the rows.
    fn add_edge(&mut self, u: u32, v: u32, w: Num) {
        if let Some(e) = self.succ[u as usize].iter_mut().find(|e| e.0 == v) {
            e.1 = w;
            if self.directed {
                if let Some(p) = self.pred[v as usize].iter_mut().find(|e| e.0 == u) {
                    p.1 = w;
                }
            } else if u != v {
                if let Some(p) = self.succ[v as usize].iter_mut().find(|e| e.0 == u) {
                    p.1 = w;
                }
            }
            return;
        }
        self.succ[u as usize].push((v, w));
        if self.directed {
            self.pred[v as usize].push((u, w));
        } else if u != v {
            self.succ[v as usize].push((u, w));
        }
    }

    fn weight(&self, u: u32, v: u32) -> Option<Num> {
        self.succ[u as usize].iter().find(|e| e.0 == v).map(|e| e.1)
    }

    /// `G.edges(data=...)`: each edge once, as NetworkX's views list them.
    fn edges(&self) -> Vec<(u32, u32, Num)> {
        let mut out = Vec::new();
        let mut seen = vec![false; self.n()];
        for u in 0..self.n() {
            for &(v, w) in &self.succ[u] {
                if self.directed || !seen[v as usize] {
                    out.push((u as u32, v, w));
                }
            }
            seen[u] = true;
        }
        out
    }
}

/// Degrees as NetworkX's degree views compute them.
struct Degrees {
    /// Undirected: `G.degree(weight)`; directed: out-degrees.
    out: Vec<Num>,
    /// Directed: in-degrees.
    inn: Vec<Num>,
}

/// `weighted` false: `weight=None` (counts).
fn degrees(g: &Level, weighted: bool, comp: bool) -> Option<Degrees> {
    let n = g.n();
    let mut out = Vec::with_capacity(n);
    let mut inn = Vec::new();
    for u in 0..n {
        if g.directed {
            if weighted {
                out.push(py_sum(g.succ[u].iter().map(|e| e.1), comp)?);
                inn.push(py_sum(g.pred[u].iter().map(|e| e.1), comp)?);
            } else {
                out.push(Num::I(g.succ[u].len() as i128));
                inn.push(Num::I(g.pred[u].len() as i128));
            }
        } else {
            let self_w = g.succ[u].iter().find(|e| e.0 as usize == u).map(|e| e.1);
            if weighted {
                // sum(...) + (n in nbrs and nbrs[n].get(weight, 1))
                let s = py_sum(g.succ[u].iter().map(|e| e.1), comp)?;
                out.push(s.add(self_w.unwrap_or(Num::I(0))));
            } else {
                out.push(Num::I(
                    g.succ[u].len() as i128 + i128::from(self_w.is_some()),
                ));
            }
        }
    }
    Some(Degrees { out, inn })
}

/// `G.size(weight)` (`weighted` false: `weight=None`).
fn size(g: &Level, weighted: bool, comp: bool) -> Option<Num> {
    let d = degrees(g, weighted, comp)?;
    let per_node: Vec<Num> = if g.directed {
        // DiDegreeView: sum(succ weights) + sum(pred weights)
        d.out.iter().zip(&d.inn).map(|(o, i)| o.add(*i)).collect()
    } else {
        d.out
    };
    let s = py_sum(per_node, comp)?;
    if weighted {
        s.div(Num::I(2))
    } else {
        s.floor_half()
    }
}

/// `modularity(G, communities, weight, resolution)`, the communities given
/// as the iteration orders of `set(community)`. `None` where NetworkX
/// raises (or returns early differently: `m == 0` is left to it).
fn modularity(
    g: &Level,
    comms: &[Vec<u32>],
    weighted: bool,
    resolution: Num,
    comp: bool,
) -> Option<Num> {
    let d = degrees(g, weighted, comp)?;
    let (m, norm) = if g.directed {
        let m = py_sum(d.out.iter().copied(), comp)?;
        if m.is_zero() {
            return None;
        }
        (m, Num::I(1).div(m.sq()?)?)
    } else {
        let deg_sum = py_sum(d.out.iter().copied(), comp)?;
        let m = deg_sum.div(Num::I(2))?;
        if m.is_zero() {
            return None;
        }
        (m, Num::I(1).div(deg_sum.sq()?)?)
    };
    let n = g.n();
    let mut member = vec![false; n];
    let mut seen = vec![false; n];
    let mut contributions = Vec::with_capacity(comms.len());
    for comm in comms {
        for &u in comm {
            member[u as usize] = true;
        }
        let mut lc = PySum::new(comp);
        for &u in comm {
            for &(v, w) in &g.succ[u as usize] {
                if (g.directed || !seen[v as usize]) && member[v as usize] {
                    lc.push(w);
                }
            }
            seen[u as usize] = true;
        }
        let lc = lc.finish()?;
        let out_sum = py_sum(comm.iter().map(|&u| d.out[u as usize]), comp)?;
        let in_sum = if g.directed {
            py_sum(comm.iter().map(|&u| d.inn[u as usize]), comp)?
        } else {
            out_sum
        };
        // L_c / m - resolution * out_degree_sum * in_degree_sum * norm
        let c = lc
            .div(m)?
            .sub(resolution.mul(out_sum).mul(in_sum).mul(norm));
        contributions.push(c);
        for &u in comm {
            member[u as usize] = false;
            seen[u as usize] = false;
        }
    }
    py_sum(contributions, comp)
}

/// An insertion-ordered map from community to an accumulated weight, as a
/// `defaultdict(float)`.
struct ComWeights {
    keys: Vec<u32>,
    vals: Vec<Num>,
    pos: Vec<u32>,
    stamp: Vec<u32>,
    now: u32,
}

impl ComWeights {
    fn new(n: usize) -> Self {
        ComWeights {
            keys: Vec::new(),
            vals: Vec::new(),
            pos: vec![0; n],
            stamp: vec![0; n],
            now: 0,
        }
    }

    fn clear(&mut self) {
        self.keys.clear();
        self.vals.clear();
        self.now += 1;
    }

    /// `d[c]`, inserting `0.0` if missing.
    fn get(&mut self, c: u32) -> usize {
        if self.stamp[c as usize] != self.now {
            self.stamp[c as usize] = self.now;
            self.pos[c as usize] = self.keys.len() as u32;
            self.keys.push(c);
            self.vals.push(Num::F(0.0));
        }
        self.pos[c as usize] as usize
    }

    fn add(&mut self, c: u32, w: Num) -> Option<()> {
        let i = self.get(c);
        self.vals[i] = self.vals[i].add(w);
        Some(())
    }
}

/// One Louvain run.
#[pyclass(module = "rustnx._core")]
pub struct Louvain {
    g: Level,
    resolution: Num,
    m: Num,
    /// NetworkX 3.7's variant.
    v37: bool,
    comp: bool,
    modularity: Num,
    /// Each node's community after the last level, compacted.
    com: Vec<u32>,
    ncom: usize,
}

fn num_of(x: &Bound<'_, PyAny>) -> Option<Num> {
    // bool is an int subclass: True + 1 == 2.
    if let Ok(b) = x.cast::<pyo3::types::PyBool>() {
        return Some(Num::I(i128::from(b.is_true())));
    }
    if x.is_exact_instance_of::<pyo3::types::PyInt>() {
        let i: i64 = x.extract().ok()?;
        return Num::int_ok(i as i128);
    }
    if x.is_exact_instance_of::<pyo3::types::PyFloat>() {
        return Some(Num::F(x.extract().ok()?));
    }
    None
}

#[pymethods]
impl Louvain {
    /// From a Graph's or DiGraph's `_adj` and `_pred` (None when
    /// undirected); `weight` the attribute (None: weights 1 and degrees as
    /// counts). Fails (ValueError) where rustnx leaves the run to NetworkX.
    #[new]
    #[pyo3(signature = (adj, pred, weight, resolution, v37, compensated))]
    fn new(
        py: Python<'_>,
        adj: &Bound<'_, PyDict>,
        pred: Option<&Bound<'_, PyDict>>,
        weight: &Bound<'_, PyAny>,
        resolution: &Bound<'_, PyAny>,
        v37: bool,
        compensated: bool,
    ) -> PyResult<Self> {
        let bad = || {
            pyo3::exceptions::PyValueError::new_err("rustnx leaves this Louvain run to NetworkX")
        };
        let directed = pred.is_some();
        let index = PyDict::new(py);
        for (i, u) in adj.keys().iter().enumerate() {
            index.set_item(u, i)?;
        }
        let n = adj.len();
        let one = Num::I(1);
        let read = |rows: &Bound<'_, PyDict>| -> PyResult<Vec<Vec<(u32, Num)>>> {
            let mut out = Vec::with_capacity(n);
            for (_, nbrs) in rows.iter() {
                let nbrs = nbrs.cast_into::<PyDict>()?;
                let mut row = Vec::with_capacity(nbrs.len());
                for (v, dd) in nbrs.iter() {
                    let j: u32 = index.get_item(&v)?.ok_or_else(bad)?.extract()?;
                    // dd.get(weight, 1)
                    let dd = dd.cast_into::<PyDict>().map_err(|_| bad())?;
                    let w = match dd.get_item(weight)? {
                        Some(x) => num_of(&x).ok_or_else(bad)?,
                        None => one,
                    };
                    row.push((j, w));
                }
                out.push(row);
            }
            Ok(out)
        };
        let succ = read(adj)?;
        let pred_rows = match pred {
            Some(p) => read(p)?,
            None => Vec::new(),
        };
        let first = Level {
            directed,
            succ,
            pred: pred_rows,
        };
        // graph.add_weighted_edges_from(G.edges(data=weight, default=1))
        let mut g = Level {
            directed,
            succ: vec![Vec::new(); n],
            pred: if directed {
                vec![Vec::new(); n]
            } else {
                Vec::new()
            },
        };
        for (u, v, w) in first.edges() {
            g.add_edge(u, v, w);
        }
        let resolution = num_of(resolution).ok_or_else(bad)?;
        // Bounds that keep every int expression of the run inside an i128
        // (and every partial int sum inside a C long): the int weights'
        // magnitudes total below 2^30, an int resolution below 2^31.
        let mut total: i128 = 0;
        for row in first.succ.iter().chain(first.pred.iter()) {
            for &(_, w) in row {
                if let Num::I(i) = w {
                    total += i.abs();
                }
            }
        }
        if total >= 1 << 30 || matches!(resolution, Num::I(r) if r.abs() >= 1 << 31) {
            return Err(bad());
        }
        let first_weighted = !weight.is_none();
        // NetworkX 3.4 to 3.6: graph.size(weight="weight"); 3.7:
        // orig_G.size(weight=weight).
        let m = if v37 {
            size(&first, first_weighted, compensated)
        } else {
            size(&g, true, compensated)
        }
        .ok_or_else(bad)?;
        // The partition of singletons, on the input graph.
        let singletons: Vec<Vec<u32>> = (0..n as u32).map(|u| vec![u]).collect();
        let modularity = modularity(&first, &singletons, first_weighted, resolution, compensated)
            .ok_or_else(bad)?;
        Ok(Louvain {
            g,
            resolution,
            m,
            v37,
            comp: compensated,
            modularity,
            com: (0..n as u32).collect(),
            ncom: n,
        })
    }

    /// `_one_level` on the current graph: the moves `(node, from, to)` in
    /// order, whether any node moved, and the generator's new state.
    /// `None` where NetworkX's arithmetic would go where rustnx's doesn't.
    #[allow(clippy::type_complexity)]
    fn one_level(&mut self, state: Vec<u32>) -> Option<(Vec<(u32, u32, u32)>, bool, Vec<u32>)> {
        let mut rng = Mt19937::from_state(&state)?;
        let g = &self.g;
        let n = g.n();
        let comp = self.comp;
        let mut node2com: Vec<u32> = (0..n as u32).collect();
        let (deg, deg_in) = if g.directed {
            let d = degrees(g, true, comp)?;
            (d.out, d.inn)
        } else {
            (degrees(g, true, comp)?.out, Vec::new())
        };
        // Undirected: Stot = degrees; directed: Stot_in, Stot_out.
        let mut stot = deg.clone();
        let mut stot_in = deg_in.clone();
        // nbrs: the neighbors other than the node itself, with weights.
        let nbrs: Vec<Vec<(u32, Num)>> = if g.directed {
            let mut out = Vec::with_capacity(n);
            let mut acc = ComWeights::new(n);
            for u in 0..n {
                acc.clear();
                for &(v, w) in &g.succ[u] {
                    if v as usize != u {
                        acc.add(v, w);
                    }
                }
                for &(v, w) in &g.pred[u] {
                    if v as usize != u {
                        acc.add(v, w);
                    }
                }
                out.push(
                    acc.keys
                        .iter()
                        .copied()
                        .zip(acc.vals.iter().copied())
                        .collect(),
                );
            }
            out
        } else {
            g.succ
                .iter()
                .enumerate()
                .map(|(u, row)| row.iter().copied().filter(|e| e.0 as usize != u).collect())
                .collect()
        };
        let mut order: Vec<u32> = (0..n as u32).collect();
        rng.shuffle(&mut order);
        let m = self.m;
        let res = self.resolution;
        let two_m2 = Num::I(2).mul(m.sq()?);
        let m2 = m.sq()?;
        let gamma = if g.directed { res } else { res.div(Num::I(2))? };
        let mut w2c = ComWeights::new(n);
        let mut moves = Vec::new();
        let mut improvement = false;
        let mut nb_moves = 1;
        while nb_moves > 0 {
            nb_moves = 0;
            for &u in &order {
                let ui = u as usize;
                let u_com = node2com[ui];
                w2c.clear();
                for &(v, w) in &nbrs[ui] {
                    w2c.add(node2com[v as usize], w);
                }
                let mut best_com = u_com;
                if g.directed {
                    let (d_in, d_out) = (deg_in[ui], deg[ui]);
                    let c = u_com as usize;
                    stot_in[c] = stot_in[c].sub(d_in);
                    stot[c] = stot[c].sub(d_out);
                    if self.v37 {
                        let x = d_out.mul(stot_in[c]).add(d_in.mul(stot[c]));
                        let k = w2c.get(u_com);
                        let mut best = w2c.vals[k].mul(m).sub(gamma.mul(x));
                        for i in 0..w2c.keys.len() {
                            let nc = w2c.keys[i] as usize;
                            let x = d_out.mul(stot_in[nc]).add(d_in.mul(stot[nc]));
                            let gain = w2c.vals[i].mul(m).sub(gamma.mul(x));
                            if gain.gt(best)? {
                                best = gain;
                                best_com = nc as u32;
                            }
                        }
                    } else {
                        let k = w2c.get(u_com);
                        let x = d_out.mul(stot_in[c]).add(d_in.mul(stot[c]));
                        let remove = w2c.vals[k].neg().div(m)?.add(res.mul(x).div(m2)?);
                        let mut best = Num::I(0);
                        for i in 0..w2c.keys.len() {
                            let nc = w2c.keys[i] as usize;
                            let x = d_out.mul(stot_in[nc]).add(d_in.mul(stot[nc]));
                            let gain = remove.add(w2c.vals[i].div(m)?).sub(res.mul(x).div(m2)?);
                            if gain.gt(best)? {
                                best = gain;
                                best_com = nc as u32;
                            }
                        }
                    }
                    let b = best_com as usize;
                    stot_in[b] = stot_in[b].add(d_in);
                    stot[b] = stot[b].add(d_out);
                } else {
                    let d = deg[ui];
                    let c = u_com as usize;
                    stot[c] = stot[c].sub(d);
                    if self.v37 {
                        let x = stot[c].mul(d);
                        let k = w2c.get(u_com);
                        let mut best = w2c.vals[k].mul(m).sub(gamma.mul(x));
                        for i in 0..w2c.keys.len() {
                            let nc = w2c.keys[i] as usize;
                            let gain = w2c.vals[i].mul(m).sub(gamma.mul(stot[nc].mul(d)));
                            if gain.gt(best)? {
                                best = gain;
                                best_com = nc as u32;
                            }
                        }
                    } else {
                        let k = w2c.get(u_com);
                        let remove = w2c.vals[k]
                            .neg()
                            .div(m)?
                            .add(res.mul(stot[c].mul(d)).div(two_m2)?);
                        let mut best = Num::I(0);
                        for i in 0..w2c.keys.len() {
                            let nc = w2c.keys[i] as usize;
                            let gain = remove
                                .add(w2c.vals[i].div(m)?)
                                .sub(res.mul(stot[nc].mul(d)).div(two_m2)?);
                            if gain.gt(best)? {
                                best = gain;
                                best_com = nc as u32;
                            }
                        }
                    }
                    let b = best_com as usize;
                    stot[b] = stot[b].add(d);
                }
                if best_com != u_com {
                    moves.push((u, u_com, best_com));
                    improvement = true;
                    nb_moves += 1;
                    node2com[ui] = best_com;
                }
            }
        }
        // The non-empty communities, renumbered in order.
        let mut size = vec![0usize; n];
        for &c in &node2com {
            size[c as usize] += 1;
        }
        let mut new_id = vec![u32::MAX; n];
        let mut k = 0;
        for c in 0..n {
            if size[c] > 0 {
                new_id[c] = k;
                k += 1;
            }
        }
        self.com = node2com.iter().map(|&c| new_id[c as usize]).collect();
        self.ncom = k as usize;
        Some((moves, improvement, rng.state()))
    }

    /// `new_mod = modularity(G, inner_partition, ...)` with the partition
    /// as the iteration orders of its set copies, then
    /// `new_mod - mod <= threshold`: True to stop; otherwise the
    /// modularity is kept for the next level. `None` where NetworkX raises.
    fn stop(&mut self, orders: Vec<Vec<u32>>, threshold: &Bound<'_, PyAny>) -> Option<bool> {
        let threshold = num_of(threshold)?;
        let new = modularity(&self.g, &orders, true, self.resolution, self.comp)?;
        let stop = new.sub(self.modularity).le(threshold)?;
        if !stop {
            self.modularity = new;
        }
        Some(stop)
    }

    /// `_gen_graph` / `_aggregate_graph`: the graph of the communities.
    fn aggregate(&mut self) -> Option<()> {
        let k = self.ncom;
        let mut h = Level {
            directed: self.g.directed,
            succ: vec![Vec::new(); k],
            pred: if self.g.directed {
                vec![Vec::new(); k]
            } else {
                Vec::new()
            },
        };
        for (u, v, w) in self.g.edges() {
            let (a, b) = (self.com[u as usize], self.com[v as usize]);
            let w = match h.weight(a, b) {
                // 3.4-3.6: wt + H.get_edge_data(..., {"weight": 0})["weight"];
                // 3.7: wt + H._adj[com1][com2]["weight"]
                Some(old) => w.add(old),
                None if self.v37 => w,
                None => w.add(Num::I(0)),
            };
            h.add_edge(a, b, w);
        }
        self.g = h;
        Some(())
    }

    /// The current graph's node count.
    fn len(&self) -> usize {
        self.g.n()
    }
}

/// Python's `sum()` as rustnx replays it, for the self-test.
pub fn check_sum(values: &[Bound<'_, PyAny>], compensated: bool) -> Option<Num> {
    py_sum(
        values.iter().map(num_of).collect::<Option<Vec<_>>>()?,
        compensated,
    )
}
