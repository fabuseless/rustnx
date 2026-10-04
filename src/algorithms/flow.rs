//! Maximum flow, minimum cost flow and Gomory-Hu trees (batch 12), ported
//! from NetworkX step by step so that residual networks, flow dicts, cut
//! partitions and every int or float value come out exactly as NetworkX's.
//!
//! Values keep Python's int/float distinction (`Val`): NetworkX's flows are
//! ints or floats depending on the capacities, and float results depend on
//! the order of every addition. Integer overflow (Python ints never
//! overflow) and states where NetworkX would raise something unexpected end
//! the computation with `Fail::Unsupported`, which falls back to NetworkX.

use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::hash::BuildHasherDefault;

use crate::algorithms::pyset::{merge_sets, PySet};
use crate::algorithms::structure_more::PairHasher;

type PairMap = HashMap<u64, u32, BuildHasherDefault<PairHasher>>;

#[inline]
fn pair(u: u32, v: u32) -> u64 {
    ((u as u64) << 32) | v as u64
}

/// Why a computation stopped without a result.
#[derive(Debug, Clone, PartialEq)]
pub enum Fail {
    /// Leave it to NetworkX: an int overflowed, or NetworkX would raise an
    /// error rustnx doesn't reproduce.
    Unsupported,
    /// `NetworkXUnbounded` with this message.
    Unbounded(&'static str),
}

pub type Res<T> = Result<T, Fail>;

pub const UNBOUNDED_PATH: &str = "Infinite capacity path, flow unbounded above.";

// --- Python numbers ----------------------------------------------------------------

/// A Python int (within `i64`) or float, with Python's arithmetic: mixing
/// the two converts the int to float, and comparisons are exact.
#[derive(Clone, Copy, Debug)]
pub enum Val {
    I(i64),
    F(f64),
}

use Val::{F, I};

pub const FLOAT_INF: Val = F(f64::INFINITY);

impl Val {
    #[inline]
    pub fn to_f64(self) -> f64 {
        match self {
            I(i) => i as f64, // round to nearest, like float(int)
            F(x) => x,
        }
    }

    #[inline]
    pub fn add(self, o: Val) -> Res<Val> {
        match (self, o) {
            (I(a), I(b)) => a.checked_add(b).map(I).ok_or(Fail::Unsupported),
            _ => Ok(F(self.to_f64() + o.to_f64())),
        }
    }

    #[inline]
    pub fn sub(self, o: Val) -> Res<Val> {
        match (self, o) {
            (I(a), I(b)) => a.checked_sub(b).map(I).ok_or(Fail::Unsupported),
            _ => Ok(F(self.to_f64() - o.to_f64())),
        }
    }

    #[inline]
    pub fn mul(self, o: Val) -> Res<Val> {
        match (self, o) {
            (I(a), I(b)) => a.checked_mul(b).map(I).ok_or(Fail::Unsupported),
            _ => Ok(F(self.to_f64() * o.to_f64())),
        }
    }

    #[inline]
    pub fn neg(self) -> Res<Val> {
        match self {
            I(a) => a.checked_neg().map(I).ok_or(Fail::Unsupported),
            F(x) => Ok(F(-x)),
        }
    }

    pub fn abs(self) -> Res<Val> {
        match self {
            I(a) => a.checked_abs().map(I).ok_or(Fail::Unsupported),
            F(x) => Ok(F(x.abs())),
        }
    }

    #[inline]
    pub fn cmp(self, o: Val) -> Option<Ordering> {
        match (self, o) {
            (I(a), I(b)) => Some(a.cmp(&b)),
            (F(a), F(b)) => a.partial_cmp(&b),
            (I(a), F(b)) => cmp_int_float(a, b),
            (F(a), I(b)) => cmp_int_float(b, a).map(Ordering::reverse),
        }
    }

    #[inline]
    pub fn lt(self, o: Val) -> bool {
        self.cmp(o) == Some(Ordering::Less)
    }

    #[inline]
    pub fn le(self, o: Val) -> bool {
        matches!(self.cmp(o), Some(Ordering::Less | Ordering::Equal))
    }

    #[inline]
    pub fn gt(self, o: Val) -> bool {
        self.cmp(o) == Some(Ordering::Greater)
    }

    #[inline]
    pub fn ge(self, o: Val) -> bool {
        matches!(self.cmp(o), Some(Ordering::Greater | Ordering::Equal))
    }

    #[inline]
    pub fn eq(self, o: Val) -> bool {
        self.cmp(o) == Some(Ordering::Equal)
    }

    /// Python's `min(self, o)`: `o` only if it is strictly smaller.
    #[inline]
    pub fn min(self, o: Val) -> Val {
        if o.lt(self) {
            o
        } else {
            self
        }
    }

    /// Python's `max(self, o)`: `o` only if it is strictly larger.
    #[inline]
    pub fn max(self, o: Val) -> Val {
        if o.gt(self) {
            o
        } else {
            self
        }
    }

    #[inline]
    pub fn truthy(self) -> bool {
        match self {
            I(a) => a != 0,
            F(x) => x != 0.0,
        }
    }
}

/// Python's exact comparison of an int with a float.
fn cmp_int_float(i: i64, f: f64) -> Option<Ordering> {
    const TWO_63: f64 = 9_223_372_036_854_775_808.0;
    if f.is_nan() {
        return None;
    }
    if f >= TWO_63 {
        return Some(Ordering::Less);
    }
    if f < -TWO_63 {
        return Some(Ordering::Greater);
    }
    let t = f.trunc();
    match i.cmp(&(t as i64)) {
        Ordering::Equal => t.partial_cmp(&f),
        ord => Some(ord),
    }
}

/// Whether the running Python compensates ints met after the first float in
/// `sum()` (3.14+; 3.12 and 3.13 add them plainly). Set once at import from
/// a behavioural check (`_set_sum_ints_compensated`).
pub static SUM_INTS_COMPENSATED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// CPython's `sum()` (start 0) of ints and floats: ints add exactly until
/// the first float, then floats add in a C double (with Neumaier
/// compensation from Python 3.12, which ints after that skip until 3.14).
pub fn py_sum(items: impl IntoIterator<Item = Val>, compensated: bool) -> Res<Val> {
    let mut it = items.into_iter();
    let mut acc: i64 = 0;
    let mut f = loop {
        match it.next() {
            None => return Ok(I(acc)),
            Some(I(b)) => acc = acc.checked_add(b).ok_or(Fail::Unsupported)?,
            Some(F(x)) => break acc as f64 + x,
        }
    };
    let mut c = 0.0f64;
    let ints_compensated =
        compensated && SUM_INTS_COMPENSATED.load(std::sync::atomic::Ordering::Relaxed);
    for item in it {
        let item = match item {
            I(v) if ints_compensated => F(v as f64),
            other => other,
        };
        match item {
            F(x) => {
                if compensated {
                    let t = f + x;
                    if f.abs() >= x.abs() {
                        c += (f - t) + x;
                    } else {
                        c += (x - t) + f;
                    }
                    f = t;
                } else {
                    f += x;
                }
            }
            I(v) => f += v as f64,
        }
    }
    if compensated && c != 0.0 && c.is_finite() {
        f += c;
    }
    Ok(F(f))
}

// --- Python dicts ----------------------------------------------------------------------

/// A dict keyed by node position, remembering insertion order like a
/// Python dict (deleting a key and adding it again moves it to the end).
#[derive(Clone)]
pub struct NodeDict<T: Copy> {
    slot: Vec<u32>,
    vals: Vec<T>,
    order: Vec<u32>,
}

const NO_SLOT: u32 = u32::MAX;

impl<T: Copy> NodeDict<T> {
    pub fn new(n: usize, fill: T) -> Self {
        NodeDict {
            slot: vec![NO_SLOT; n],
            vals: vec![fill; n],
            order: Vec::new(),
        }
    }

    #[inline]
    pub fn contains(&self, k: u32) -> bool {
        self.slot[k as usize] != NO_SLOT
    }

    #[inline]
    pub fn get(&self, k: u32) -> Option<T> {
        self.contains(k).then(|| self.vals[k as usize])
    }

    pub fn insert(&mut self, k: u32, v: T) {
        if !self.contains(k) {
            self.slot[k as usize] = self.order.len() as u32;
            self.order.push(k);
        }
        self.vals[k as usize] = v;
    }

    pub fn remove(&mut self, k: u32) -> bool {
        let s = self.slot[k as usize];
        if s == NO_SLOT {
            return false;
        }
        self.order[s as usize] = NO_SLOT;
        self.slot[k as usize] = NO_SLOT;
        true
    }

    /// Keys in insertion order.
    pub fn keys(&self) -> impl Iterator<Item = u32> + '_ {
        self.order.iter().copied().filter(|&k| k != NO_SLOT)
    }

    pub fn items(&self) -> impl Iterator<Item = (u32, T)> + '_ {
        self.keys().map(|k| (k, self.vals[k as usize]))
    }
}

// --- Residual networks -----------------------------------------------------------------

/// NetworkX's residual network `R`: each edge with its capacity and flow,
/// and every node's `R._succ` and `R._pred` rows in dict order. There is
/// at most one edge per ordered pair, and `rev[e]` is the opposite edge.
#[derive(Clone)]
pub struct Residual {
    pub n: usize,
    pub tail: Vec<u32>,
    pub head: Vec<u32>,
    pub cap: Vec<Val>,
    pub flow: Vec<Val>,
    pub rev: Vec<u32>,
    pub succ: Vec<Vec<u32>>,
    pub pred: Vec<Vec<u32>>,
    /// `R.graph["inf"]`.
    pub inf: Val,
}

impl Residual {
    fn add_edge(&mut self, u: u32, v: u32, cap: Val) -> u32 {
        let e = self.tail.len() as u32;
        self.tail.push(u);
        self.head.push(v);
        self.cap.push(cap);
        self.flow.push(I(0));
        self.rev.push(u32::MAX);
        self.succ[u as usize].push(e);
        self.pred[v as usize].push(e);
        e
    }

    /// `build_residual_network` from `G.edges(data=True)` as `(u, v,
    /// capacity)`, missing capacities being `float("inf")`.
    pub fn build(
        n: usize,
        directed: bool,
        edges: &[(u32, u32, Val)],
        compensated: bool,
    ) -> Res<Residual> {
        let list: Vec<(u32, u32, Val)> = edges
            .iter()
            .copied()
            .filter(|&(u, v, c)| u != v && c.gt(I(0)))
            .collect();
        let total = py_sum(
            list.iter().filter(|e| !e.2.eq(FLOAT_INF)).map(|e| e.2),
            compensated,
        )?;
        let mut inf = I(3).mul(total)?;
        if !inf.truthy() {
            inf = I(1);
        }
        let mut r = Residual {
            n,
            tail: Vec::new(),
            head: Vec::new(),
            cap: Vec::new(),
            flow: Vec::new(),
            rev: Vec::new(),
            succ: vec![Vec::new(); n],
            pred: vec![Vec::new(); n],
            inf,
        };
        if directed {
            let mut index = PairMap::default();
            for &(u, v, c) in &list {
                let cap = c.min(inf);
                match index.get(&pair(u, v)) {
                    Some(&e) => r.cap[e as usize] = cap,
                    None => {
                        let e = r.add_edge(u, v, cap);
                        let b = r.add_edge(v, u, I(0));
                        r.rev[e as usize] = b;
                        r.rev[b as usize] = e;
                        index.insert(pair(u, v), e);
                        index.insert(pair(v, u), b);
                    }
                }
            }
        } else {
            for &(u, v, c) in &list {
                let cap = c.min(inf);
                let e = r.add_edge(u, v, cap);
                let b = r.add_edge(v, u, cap);
                r.rev[e as usize] = b;
                r.rev[b as usize] = e;
            }
        }
        Ok(r)
    }

    pub fn reset_flows(&mut self) {
        for f in self.flow.iter_mut() {
            *f = I(0);
        }
    }

    #[inline]
    fn residual_cap(&self, e: usize) -> Res<Val> {
        self.cap[e].sub(self.flow[e])
    }

    /// `attr["flow"] < attr["capacity"]`.
    #[inline]
    fn open(&self, e: usize) -> bool {
        self.flow[e].lt(self.cap[e])
    }

    #[inline]
    fn push_flow(&mut self, e: usize, f: Val) -> Res<()> {
        self.flow[e] = self.flow[e].add(f)?;
        let b = self.rev[e] as usize;
        self.flow[b] = self.flow[b].sub(f)?;
        Ok(())
    }

    /// `minimum_cut`'s reverse breadth-first search from `t`: the nodes
    /// that reach `t` over edges with spare capacity, in discovery order.
    /// NetworkX 3.7 follows edges with `flow < capacity`; earlier versions
    /// delete the edges with `flow == capacity` (`strict` false).
    pub fn cut_order(&self, t: u32, strict: bool) -> Vec<u32> {
        let mut seen = vec![false; self.n];
        seen[t as usize] = true;
        let mut order = vec![t];
        let mut queue = vec![t];
        while !queue.is_empty() {
            let mut next = Vec::new();
            for &v in &queue {
                for &e in &self.pred[v as usize] {
                    let e = e as usize;
                    let u = self.tail[e];
                    let open = if strict {
                        self.open(e)
                    } else {
                        !self.flow[e].eq(self.cap[e])
                    };
                    if !seen[u as usize] && open {
                        seen[u as usize] = true;
                        order.push(u);
                        next.push(u);
                    }
                }
            }
            queue = next;
        }
        order
    }

    /// NetworkX before 3.7 removes the saturated edges from `R` to find the
    /// cut and adds them back, which moves them to the end of their rows.
    /// That changes later flow computations on the same `R` (Gomory-Hu).
    pub fn move_saturated_last(&mut self) {
        let mut cutset = Vec::new();
        for u in 0..self.n {
            for &e in &self.succ[u] {
                if self.flow[e as usize].eq(self.cap[e as usize]) {
                    cutset.push(e);
                }
            }
        }
        if cutset.is_empty() {
            return;
        }
        let mut cut = vec![false; self.tail.len()];
        for &e in &cutset {
            cut[e as usize] = true;
        }
        for row in self.succ.iter_mut().chain(self.pred.iter_mut()) {
            row.retain(|&e| !cut[e as usize]);
        }
        for &e in &cutset {
            self.succ[self.tail[e as usize] as usize].push(e);
            self.pred[self.head[e as usize] as usize].push(e);
        }
    }

    /// `detect_unboundedness`: a path of infinite-capacity edges from s to t.
    fn detect_unboundedness(&self, s: u32, t: u32) -> Res<()> {
        let mut seen = vec![false; self.n];
        seen[s as usize] = true;
        let mut q = VecDeque::from([s]);
        while let Some(u) = q.pop_front() {
            for &e in &self.succ[u as usize] {
                let v = self.head[e as usize];
                if self.cap[e as usize].eq(self.inf) && !seen[v as usize] {
                    if v == t {
                        return Err(Fail::Unbounded(UNBOUNDED_PATH));
                    }
                    seen[v as usize] = true;
                    q.push_back(v);
                }
            }
        }
        Ok(())
    }

    /// Reverse breadth-first search over edges with spare capacity, as
    /// `preflow_push` and `shortest_augmenting_path` compute heights: the
    /// `heights` dict as (node, height) in insertion order.
    fn reverse_bfs(&self, src: u32) -> (Vec<(u32, u32)>, Vec<u32>) {
        const NONE: u32 = u32::MAX;
        let mut height = vec![NONE; self.n];
        height[src as usize] = 0;
        let mut order = vec![(src, 0)];
        let mut q = VecDeque::from([(src, 0u32)]);
        while let Some((u, h)) = q.pop_front() {
            let h = h + 1;
            for &e in &self.pred[u as usize] {
                let v = self.tail[e as usize];
                if height[v as usize] == NONE && self.open(e as usize) {
                    height[v as usize] = h;
                    order.push((v, h));
                    q.push_back((v, h));
                }
            }
        }
        (order, height)
    }

    /// Edge ids of a path given as consecutive edges: the bottleneck as
    /// `augment` computes it, raising `NetworkXUnbounded` like NetworkX.
    fn augment_path(&mut self, path: &[u32]) -> Res<Val> {
        let mut flow = self.inf;
        for &e in path {
            flow = flow.min(self.residual_cap(e as usize)?);
        }
        if flow.mul(I(2))?.gt(self.inf) {
            return Err(Fail::Unbounded(UNBOUNDED_PATH));
        }
        for &e in path {
            self.push_flow(e as usize, flow)?;
        }
        Ok(flow)
    }
}

// --- Edmonds-Karp --------------------------------------------------------------------

/// The bidirectional search's `pred`/`succ` dicts as per-search stamps,
/// kept between runs on residual networks with the same nodes.
#[derive(Clone)]
pub struct EkScratch {
    pred_stamp: Vec<u32>,
    succ_stamp: Vec<u32>,
    pred_edge: Vec<u32>,
    succ_edge: Vec<u32>,
    stamp: u32,
}

impl EkScratch {
    pub fn new(n: usize) -> Self {
        EkScratch {
            pred_stamp: vec![0; n],
            succ_stamp: vec![0; n],
            pred_edge: vec![u32::MAX; n],
            succ_edge: vec![u32::MAX; n],
            stamp: 0,
        }
    }
}

/// `edmonds_karp_core`: the flow value added (flows start as they are).
pub fn edmonds_karp_core(r: &mut Residual, s: u32, t: u32, cutoff: Val) -> Res<Val> {
    edmonds_karp_with(r, s, t, cutoff, &mut EkScratch::new(r.n))
}

/// `edmonds_karp_core` with scratch space from `EkScratch::new(r.n)`.
pub fn edmonds_karp_with(
    r: &mut Residual,
    s: u32,
    t: u32,
    cutoff: Val,
    sc: &mut EkScratch,
) -> Res<Val> {
    let EkScratch {
        pred_stamp,
        succ_stamp,
        pred_edge,
        succ_edge,
        stamp,
    } = sc;
    let mut flow_value = I(0);
    let mut path = Vec::new();
    while flow_value.lt(cutoff) {
        *stamp = stamp.wrapping_add(1);
        if *stamp == 0 {
            pred_stamp.iter_mut().for_each(|x| *x = 0);
            succ_stamp.iter_mut().for_each(|x| *x = 0);
            *stamp = 1;
        }
        let stamp = *stamp;
        pred_stamp[s as usize] = stamp;
        pred_edge[s as usize] = u32::MAX;
        succ_stamp[t as usize] = stamp;
        succ_edge[t as usize] = u32::MAX;
        let mut q_s = vec![s];
        let mut q_t = vec![t];
        let meet = 'search: loop {
            let mut q = Vec::new();
            if q_s.len() <= q_t.len() {
                for &u in &q_s {
                    for &e in &r.succ[u as usize] {
                        let v = r.head[e as usize];
                        if pred_stamp[v as usize] != stamp && r.open(e as usize) {
                            pred_stamp[v as usize] = stamp;
                            pred_edge[v as usize] = e;
                            if succ_stamp[v as usize] == stamp {
                                break 'search Some(v);
                            }
                            q.push(v);
                        }
                    }
                }
                if q.is_empty() {
                    break None;
                }
                q_s = q;
            } else {
                for &u in &q_t {
                    for &e in &r.pred[u as usize] {
                        let v = r.tail[e as usize];
                        if succ_stamp[v as usize] != stamp && r.open(e as usize) {
                            succ_stamp[v as usize] = stamp;
                            succ_edge[v as usize] = e;
                            if pred_stamp[v as usize] == stamp {
                                break 'search Some(v);
                            }
                            q.push(v);
                        }
                    }
                }
                if q.is_empty() {
                    break None;
                }
                q_t = q;
            }
        };
        let Some(v) = meet else { break };
        path.clear();
        let mut u = v;
        while u != s {
            let e = pred_edge[u as usize];
            path.push(e);
            u = r.tail[e as usize];
        }
        path.reverse();
        let mut u = v;
        while u != t {
            let e = succ_edge[u as usize];
            path.push(e);
            u = r.head[e as usize];
        }
        let f = r.augment_path(&path)?;
        flow_value = flow_value.add(f)?;
    }
    Ok(flow_value)
}

// --- Shortest augmenting path -----------------------------------------------------------

/// Node attributes `shortest_augmenting_path` and `preflow_push` leave on
/// `R`: heights and each `CurrentEdge`'s position in its `R._succ` row.
pub struct NodeState {
    pub height: Vec<u32>,
    pub curr: Vec<u32>,
}

/// `shortest_augmenting_path_impl` after the residual network is reset.
/// `d` is NetworkX's phase-one height limit, computed by the caller.
pub fn shortest_augmenting_path(
    r: &mut Residual,
    s: u32,
    t: u32,
    two_phase: bool,
    d: i64,
    cutoff: Val,
) -> Res<(Val, Option<NodeState>)> {
    r.reset_flows();
    let n = r.n;
    let (_, heights) = r.reverse_bfs(t);
    if heights[s as usize] == u32::MAX {
        return Ok((I(0), None));
    }
    let mut height: Vec<u32> = heights
        .iter()
        .map(|&h| if h == u32::MAX { n as u32 } else { h })
        .collect();
    let mut curr = vec![0u32; n];
    let mut counts = vec![0i64; 2 * n - 1];
    for &h in &height {
        counts[h as usize] += 1;
    }
    let relabel = |r: &Residual, height: &[u32], u: u32| -> u32 {
        let mut h = n as u32 - 1;
        for &e in &r.succ[u as usize] {
            if r.open(e as usize) {
                h = h.min(height[r.head[e as usize] as usize]);
            }
        }
        h + 1
    };
    let mut flow_value = I(0);
    let mut path = vec![s];
    let mut path_edges: Vec<u32> = Vec::new();
    let mut u = s;
    let mut done = height[s as usize] as i64 >= d;
    let state = |height: Vec<u32>, curr: Vec<u32>| Some(NodeState { height, curr });
    while !done {
        let mut h = height[u as usize];
        loop {
            let row = &r.succ[u as usize];
            if row.is_empty() {
                return Err(Fail::Unsupported); // CurrentEdge.get() would fail
            }
            let e = row[curr[u as usize] as usize];
            let v = r.head[e as usize];
            if h == height[v as usize] + 1 && r.open(e as usize) {
                path.push(v);
                path_edges.push(e);
                u = v;
                break;
            }
            if (curr[u as usize] as usize) + 1 < row.len() {
                curr[u as usize] += 1;
                continue;
            }
            curr[u as usize] = 0;
            counts[h as usize] -= 1;
            if counts[h as usize] == 0 {
                return Ok((flow_value, state(height, curr)));
            }
            h = relabel(r, &height, u);
            if u == s && h as i64 >= d {
                if !two_phase {
                    return Ok((flow_value, state(height, curr)));
                }
                done = true;
                break;
            }
            counts[h as usize] += 1;
            height[u as usize] = h;
            if u != s {
                path.pop();
                path_edges.pop();
                u = *path.last().unwrap();
                break;
            }
        }
        if u == t {
            let f = r.augment_path(&path_edges)?;
            flow_value = flow_value.add(f)?;
            if flow_value.ge(cutoff) {
                return Ok((flow_value, state(height, curr)));
            }
            path.clear();
            path.push(s);
            path_edges.clear();
            u = s;
        }
    }
    let rest = edmonds_karp_core(r, s, t, cutoff.sub(flow_value)?)?;
    flow_value = flow_value.add(rest)?;
    Ok((flow_value, state(height, curr)))
}

// --- Dinitz ------------------------------------------------------------------------------

/// `dinitz_impl` after the residual network is reset.
pub fn dinitz(r: &mut Residual, s: u32, t: u32, cutoff: Option<Val>) -> Res<Val> {
    r.reset_flows();
    let n = r.n;
    let inf = r.inf;
    let cutoff = cutoff.unwrap_or(inf);
    // `parents[v]`: a deque of (parent, edge parent -> v), as a vector
    // with a moving front; `dist` is `vertex_dist`.
    let mut parents: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
    let mut front = vec![0usize; n];
    let mut has_parents = vec![false; n];
    let mut dist = vec![0i64; n];
    let mut touched: Vec<u32> = Vec::new();
    let mut flow_value = I(0);
    while flow_value.lt(cutoff) {
        for &v in &touched {
            parents[v as usize].clear();
            front[v as usize] = 0;
            has_parents[v as usize] = false;
        }
        touched.clear();
        dist[s as usize] = 0;
        let mut queue = VecDeque::from([(s, 0i64)]);
        while let Some(&(u, du)) = queue.front() {
            if has_parents[t as usize] {
                break;
            }
            queue.pop_front();
            for &e in &r.succ[u as usize] {
                let v = r.head[e as usize];
                if r.residual_cap(e as usize)?.gt(I(0)) {
                    if has_parents[v as usize] {
                        if dist[v as usize] == du + 1 {
                            parents[v as usize].push((u, e));
                        }
                    } else {
                        has_parents[v as usize] = true;
                        touched.push(v);
                        parents[v as usize].push((u, e));
                        dist[v as usize] = du + 1;
                        queue.push_back((v, du + 1));
                    }
                }
            }
        }
        if !has_parents[t as usize] {
            break;
        }
        // depth_first_search: `path` holds nodes from t, `path_e[i]` the
        // edge from path[i + 1] to path[i].
        let mut total = I(0);
        let mut u = t;
        let mut path = vec![u];
        let mut path_e: Vec<u32> = Vec::new();
        loop {
            if !has_parents[u as usize] {
                return Err(Fail::Unsupported); // KeyError in NetworkX
            }
            let v;
            if front[u as usize] < parents[u as usize].len() {
                let (p, e) = parents[u as usize][front[u as usize]];
                v = p;
                path.push(v);
                path_e.push(e);
            } else {
                path.pop();
                path_e.pop();
                let Some(&last) = path.last() else { break };
                v = last;
                if !has_parents[v as usize] || front[v as usize] >= parents[v as usize].len() {
                    return Err(Fail::Unsupported);
                }
                front[v as usize] += 1;
            }
            let mut v = v;
            if v == s {
                let mut flow = inf;
                for &e in &path_e {
                    flow = flow.min(r.residual_cap(e as usize)?);
                }
                // pairwise(reversed(path)): from the s end towards t.
                let mut i = path_e.len();
                while i > 0 {
                    i -= 1;
                    if i >= path_e.len() {
                        continue;
                    }
                    let e = path_e[i] as usize;
                    r.push_flow(e, flow)?;
                    if r.residual_cap(e)?.eq(I(0)) {
                        let head = path[i];
                        if !has_parents[head as usize]
                            || front[head as usize] >= parents[head as usize].len()
                        {
                            return Err(Fail::Unsupported);
                        }
                        front[head as usize] += 1;
                        while *path.last().unwrap() != head {
                            path.pop();
                            path_e.pop();
                        }
                    }
                }
                total = total.add(flow)?;
                v = *path.last().unwrap();
            }
            u = v;
        }
        if total.mul(I(2))?.gt(inf) {
            return Err(Fail::Unbounded(UNBOUNDED_PATH));
        }
        flow_value = flow_value.add(total)?;
    }
    Ok(flow_value)
}

// --- Boykov-Kolmogorov -------------------------------------------------------------------

/// A Python deque of nodes where `x in active` and `active.remove(x)`
/// (first occurrence) must be fast: occurrences carry ids, and each node
/// keeps its live occurrence ids in order.
struct ActiveDeque {
    items: VecDeque<(u32, u64)>,
    dead: Vec<bool>,
    occ: Vec<VecDeque<u64>>,
    len: usize,
    next_id: u64,
}

impl ActiveDeque {
    fn new(n: usize) -> Self {
        ActiveDeque {
            items: VecDeque::new(),
            dead: Vec::new(),
            occ: vec![VecDeque::new(); n],
            len: 0,
            next_id: 0,
        }
    }

    fn append(&mut self, v: u32) {
        let id = self.next_id;
        self.next_id += 1;
        self.items.push_back((v, id));
        self.dead.push(false);
        self.occ[v as usize].push_back(id);
        self.len += 1;
    }

    fn contains(&self, v: u32) -> bool {
        !self.occ[v as usize].is_empty()
    }

    fn skip_dead(&mut self) {
        while let Some(&(_, id)) = self.items.front() {
            if self.dead[id as usize] {
                self.items.pop_front();
            } else {
                break;
            }
        }
    }

    fn front(&mut self) -> Option<u32> {
        self.skip_dead();
        self.items.front().map(|&(v, _)| v)
    }

    fn popleft(&mut self) {
        self.skip_dead();
        if let Some((v, _)) = self.items.pop_front() {
            self.occ[v as usize].pop_front();
            self.len -= 1;
        }
    }

    /// `active.remove(v)` for a `v` known to be present.
    fn remove(&mut self, v: u32) {
        if let Some(id) = self.occ[v as usize].pop_front() {
            self.dead[id as usize] = true;
            self.len -= 1;
        }
    }
}

/// `R.graph["trees"]`: each tree as (node, parent) in dict order.
pub type Trees = (Vec<(u32, Option<u32>)>, Vec<(u32, Option<u32>)>);

const NO_NODE: u32 = u32::MAX;

/// `boykov_kolmogorov_impl` after the residual network is reset.
pub fn boykov_kolmogorov(
    r: &mut Residual,
    s: u32,
    t: u32,
    cutoff: Option<Val>,
) -> Res<(Val, Trees)> {
    r.reset_flows();
    let n = r.n;
    let inf = r.inf;
    let cutoff = cutoff.unwrap_or(inf);
    // Trees map node -> (parent, edge between them); NO_NODE for None.
    let mut trees = [
        NodeDict::new(n, (NO_NODE, NO_NODE)),
        NodeDict::new(n, (NO_NODE, NO_NODE)),
    ];
    trees[0].insert(s, (NO_NODE, NO_NODE));
    trees[1].insert(t, (NO_NODE, NO_NODE));
    let mut active = ActiveDeque::new(n);
    active.append(s);
    active.append(t);
    let mut orphans: VecDeque<u32> = VecDeque::new();
    let mut flow_value = I(0);
    let mut time: i64 = 1;
    let mut timestamp: Vec<Option<i64>> = vec![None; n];
    let mut dist: Vec<Option<i64>> = vec![None; n];
    timestamp[s as usize] = Some(1);
    timestamp[t as usize] = Some(1);
    dist[s as usize] = Some(0);
    dist[t as usize] = Some(0);
    let get = |x: &[Option<i64>], v: u32| x[v as usize].ok_or(Fail::Unsupported);

    while flow_value.lt(cutoff) {
        // grow(): the connecting edge, oriented from the source tree.
        let mut found = None;
        'grow: while active.len > 0 {
            let u = active.front().ok_or(Fail::Unsupported)?;
            let (this, other) = if trees[0].contains(u) { (0, 1) } else { (1, 0) };
            let row = if this == 0 {
                &r.succ[u as usize]
            } else {
                &r.pred[u as usize]
            };
            for &e in row {
                let v = if this == 0 {
                    r.head[e as usize]
                } else {
                    r.tail[e as usize]
                };
                if r.residual_cap(e as usize)?.gt(I(0)) {
                    if !trees[this].contains(v) {
                        if trees[other].contains(v) {
                            found = Some(e);
                            break 'grow;
                        }
                        trees[this].insert(v, (u, e));
                        dist[v as usize] = Some(get(&dist, u)? + 1);
                        timestamp[v as usize] = timestamp[u as usize];
                        active.append(v);
                    } else {
                        // _is_closer(u, v)
                        let (tu, tv) = (get(&timestamp, u)?, get(&timestamp, v)?);
                        let (du, dv) = (get(&dist, u)?, get(&dist, v)?);
                        if tv <= tu && dv > du + 1 {
                            trees[this].insert(v, (u, e));
                            dist[v as usize] = Some(du + 1);
                            timestamp[v as usize] = Some(tu);
                        }
                    }
                }
            }
            active.popleft();
        }
        let Some(link) = found else { break };
        time += 1;

        // augment(u, v)
        let (u, v) = (r.tail[link as usize], r.head[link as usize]);
        let mut flow = inf.min(r.residual_cap(link as usize)?);
        let mut path_nodes = vec![u];
        let mut path_edges = Vec::new();
        let mut w = u;
        while w != s {
            let (p, e) = trees[0].get(w).ok_or(Fail::Unsupported)?;
            if p == NO_NODE {
                return Err(Fail::Unsupported);
            }
            flow = flow.min(r.residual_cap(e as usize)?);
            path_nodes.push(p);
            path_edges.push(e);
            w = p;
        }
        path_nodes.reverse();
        path_edges.reverse();
        path_nodes.push(v);
        path_edges.push(link);
        let mut w = v;
        while w != t {
            let (p, e) = trees[1].get(w).ok_or(Fail::Unsupported)?;
            if p == NO_NODE {
                return Err(Fail::Unsupported);
            }
            flow = flow.min(r.residual_cap(e as usize)?);
            path_nodes.push(p);
            path_edges.push(e);
            w = p;
        }
        let mut these_orphans = Vec::new();
        for (i, &e) in path_edges.iter().enumerate() {
            let (a, b) = (path_nodes[i], path_nodes[i + 1]);
            r.push_flow(e as usize, flow)?;
            if r.flow[e as usize].eq(r.cap[e as usize]) {
                if trees[0].contains(b) {
                    trees[0].insert(b, (NO_NODE, NO_NODE));
                    these_orphans.push(b);
                }
                if trees[1].contains(a) {
                    trees[1].insert(a, (NO_NODE, NO_NODE));
                    these_orphans.push(a);
                }
            }
        }
        let mut keyed = Vec::with_capacity(these_orphans.len());
        for &o in &these_orphans {
            keyed.push((get(&dist, o)?, o));
        }
        keyed.sort_by_key(|&(d, _)| d); // stable, like sorted(key=dist.get)
        orphans.extend(keyed.into_iter().map(|(_, o)| o));
        flow_value = flow_value.add(flow)?;

        // adopt()
        while let Some(u) = orphans.pop_front() {
            let tr = if trees[0].contains(u) { 0 } else { 1 };
            // Neighbors in the tree, sorted by dist (stable): for the source
            // tree, in-edges (parent -> u); for the target tree, out-edges.
            let nbrs =
                |trees: &[NodeDict<(u32, u32)>; 2], dist: &[Option<i64>]| -> Res<Vec<(u32, u32)>> {
                    let row = if tr == 0 {
                        &r.pred[u as usize]
                    } else {
                        &r.succ[u as usize]
                    };
                    let mut out = Vec::new();
                    for &e in row {
                        let x = if tr == 0 {
                            r.tail[e as usize]
                        } else {
                            r.head[e as usize]
                        };
                        if trees[tr].contains(x) {
                            out.push((dist[x as usize].ok_or(Fail::Unsupported)?, x, e));
                        }
                    }
                    out.sort_by_key(|&(d, _, _)| d);
                    Ok(out.into_iter().map(|(_, x, e)| (x, e)).collect())
                };
            let mut adopted = false;
            for (v, e) in nbrs(&trees, &dist)? {
                if r.residual_cap(e as usize)?.gt(I(0)) {
                    // _has_valid_root(v, tree)
                    let mut path = Vec::new();
                    let mut x = v;
                    let base = loop {
                        if x == NO_NODE {
                            break None;
                        }
                        path.push(x);
                        if x == s || x == t {
                            break Some(0);
                        }
                        if get(&timestamp, x)? == time {
                            break Some(get(&dist, x)?);
                        }
                        x = trees[tr].get(x).ok_or(Fail::Unsupported)?.0;
                    };
                    if let Some(base) = base {
                        let length = path.len() as i64;
                        for (i, &y) in path.iter().enumerate() {
                            dist[y as usize] = Some(base + length - (i as i64 + 1));
                            timestamp[y as usize] = Some(time);
                        }
                        trees[tr].insert(u, (v, e));
                        dist[u as usize] = Some(get(&dist, v)? + 1);
                        timestamp[u as usize] = Some(time);
                        adopted = true;
                        break;
                    }
                }
            }
            if !adopted {
                for (v, e) in nbrs(&trees, &dist)? {
                    if r.residual_cap(e as usize)?.gt(I(0)) && !active.contains(v) {
                        active.append(v);
                    }
                    let (p, _) = trees[tr].get(v).ok_or(Fail::Unsupported)?;
                    if p == u {
                        trees[tr].insert(v, (NO_NODE, NO_NODE));
                        orphans.push_front(v);
                    }
                }
                if active.contains(u) {
                    active.remove(u);
                }
                if !trees[tr].remove(u) {
                    return Err(Fail::Unsupported); // KeyError in NetworkX
                }
            }
        }
    }
    if flow_value.mul(I(2))?.gt(inf) {
        return Err(Fail::Unbounded(UNBOUNDED_PATH));
    }
    let export = |d: &NodeDict<(u32, u32)>| {
        d.items()
            .map(|(k, (p, _))| (k, (p != NO_NODE).then_some(p)))
            .collect::<Vec<_>>()
    };
    Ok((flow_value, (export(&trees[0]), export(&trees[1]))))
}

// --- Preflow-push --------------------------------------------------------------------------

/// What `preflow_push` leaves on `R`: the flow value and node attributes
/// (`excess` always; `height` and `curr_edge` once the run gets that far).
pub struct Preflow {
    pub value: Val,
    pub excess: Vec<Val>,
    pub state: Option<NodeState>,
}

struct Levels {
    active: Vec<PySet>,
    inactive: Vec<PySet>,
}

/// `preflow_push_impl` from `detect_unboundedness` on. `threshold` is the
/// `GlobalRelabelThreshold` (`(n + m) / freq`, or infinity), and `hashes`
/// the nodes' `hash()` values, for the order of the `Level` sets.
pub fn preflow_push(
    r: &mut Residual,
    s: u32,
    t: u32,
    threshold: f64,
    value_only: bool,
    hashes: &[i64],
) -> Res<Preflow> {
    r.detect_unboundedness(s, t)?;
    let n = r.n;
    let mut excess = vec![I(0); n];
    r.reset_flows();
    let (bfs, heights) = r.reverse_bfs(t);
    if heights[s as usize] == u32::MAX {
        return Ok(Preflow {
            value: I(0),
            excess,
            state: None,
        });
    }
    let mut max_height = bfs
        .iter()
        .filter(|&&(u, _)| u != s)
        .map(|&(_, h)| h as i64)
        .max()
        .ok_or(Fail::Unsupported)?;
    let nn = n as i64;
    let mut height: Vec<i64> = heights
        .iter()
        .map(|&h| if h == u32::MAX { nn + 1 } else { h as i64 })
        .collect();
    height[s as usize] = nn;
    let mut curr = vec![0u32; n];
    let mut work: u64 = 0;
    let reached = |work: u64| work as f64 >= threshold;

    let push = |r: &mut Residual, excess: &mut [Val], e: usize, f: Val| -> Res<()> {
        r.push_flow(e, f)?;
        let (u, v) = (r.tail[e] as usize, r.head[e] as usize);
        excess[u] = excess[u].sub(f)?;
        excess[v] = excess[v].add(f)?;
        Ok(())
    };
    for i in 0..r.succ[s as usize].len() {
        let e = r.succ[s as usize][i] as usize;
        let f = r.cap[e];
        if f.gt(I(0)) {
            push(r, &mut excess, e, f)?;
        }
    }
    let nlevels = 2 * n;
    let mut lv = Levels {
        active: vec![PySet::default(); nlevels],
        inactive: vec![PySet::default(); nlevels],
    };
    let level_index = |h: i64| -> Res<usize> {
        if h < 0 || h as usize >= nlevels {
            Err(Fail::Unsupported) // IndexError (or a negative index) in NetworkX
        } else {
            Ok(h as usize)
        }
    };
    for u in 0..n as u32 {
        if u != s && u != t {
            let l = level_index(height[u as usize])?;
            if excess[u as usize].gt(I(0)) {
                lv.active[l].add(u, hashes);
            } else {
                lv.inactive[l].add(u, hashes);
            }
        }
    }

    // discharge(u, is_phase1): the next height to look at.
    let discharge = |r: &mut Residual,
                     lv: &mut Levels,
                     excess: &mut [Val],
                     height: &mut [i64],
                     curr: &mut [u32],
                     work: &mut u64,
                     u: u32,
                     phase1: bool|
     -> Res<i64> {
        let mut h = height[u as usize];
        let mut next_height = h;
        let l = level_index(h)?;
        if !lv.active[l].discard(u, hashes) {
            return Err(Fail::Unsupported);
        }
        loop {
            let row_len = r.succ[u as usize].len();
            if row_len == 0 {
                return Err(Fail::Unsupported);
            }
            let e = r.succ[u as usize][curr[u as usize] as usize] as usize;
            let v = r.head[e];
            if h == height[v as usize] + 1 && r.open(e) {
                let f = excess[u as usize].min(r.residual_cap(e)?);
                push(r, excess, e, f)?;
                // activate(v)
                if v != s && v != t {
                    let lvl = level_index(height[v as usize])?;
                    if lv.inactive[lvl].contains(v, hashes) {
                        lv.inactive[lvl].discard(v, hashes);
                        lv.active[lvl].add(v, hashes);
                    }
                }
                if excess[u as usize].eq(I(0)) {
                    lv.inactive[level_index(h)?].add(u, hashes);
                    break;
                }
            }
            if (curr[u as usize] as usize) + 1 < row_len {
                curr[u as usize] += 1;
                continue;
            }
            curr[u as usize] = 0;
            // relabel(u)
            *work += row_len as u64;
            let mut best: Option<i64> = None;
            for &e in &r.succ[u as usize] {
                if r.open(e as usize) {
                    let hv = height[r.head[e as usize] as usize];
                    best = Some(best.map_or(hv, |b| b.min(hv)));
                }
            }
            h = best.ok_or(Fail::Unsupported)? + 1;
            if phase1 && h >= nn - 1 {
                lv.active[level_index(h)?].add(u, hashes);
                break;
            }
            next_height = h;
        }
        height[u as usize] = h;
        Ok(next_height)
    };

    // global_relabel(from_sink): the new max height.
    let global_relabel =
        |r: &Residual, lv: &mut Levels, height: &mut [i64], from_sink: bool| -> Res<i64> {
            let src = if from_sink { t } else { s };
            let (order, _) = r.reverse_bfs(src);
            let mut heights = NodeDict::new(n, 0i64);
            for &(u, h) in &order {
                heights.insert(u, h as i64);
            }
            if !from_sink && !heights.remove(t) {
                return Err(Fail::Unsupported);
            }
            let mut max_h = heights
                .items()
                .map(|(_, h)| h)
                .max()
                .ok_or(Fail::Unsupported)?;
            if from_sink {
                for u in 0..n as u32 {
                    if !heights.contains(u) && height[u as usize] < nn {
                        heights.insert(u, nn + 1);
                    }
                }
            } else {
                let keys: Vec<u32> = heights.keys().collect();
                for u in keys {
                    let h = heights.get(u).unwrap();
                    heights.insert(u, h + nn);
                }
                max_h += nn;
            }
            heights.remove(src);
            for (u, new_h) in heights.items() {
                let old_h = height[u as usize];
                if new_h != old_h {
                    let (lo, ln) = (level_index(old_h)?, level_index(new_h)?);
                    if lv.active[lo].contains(u, hashes) {
                        lv.active[lo].discard(u, hashes);
                        lv.active[ln].add(u, hashes);
                    } else {
                        if !lv.inactive[lo].discard(u, hashes) {
                            return Err(Fail::Unsupported);
                        }
                        lv.inactive[ln].add(u, hashes);
                    }
                    height[u as usize] = new_h;
                }
            }
            Ok(max_h)
        };

    // Phase 1: a maximum preflow.
    let mut h = max_height;
    while h > 0 {
        loop {
            let l = level_index(h)?;
            let Some(u) = lv.active[l].first() else {
                h -= 1;
                break;
            };
            let old_h = h;
            h = discharge(
                r,
                &mut lv,
                &mut excess,
                &mut height,
                &mut curr,
                &mut work,
                u,
                true,
            )?;
            if reached(work) {
                h = global_relabel(r, &mut lv, &mut height, true)?;
                max_height = h;
                work = 0;
            } else if lv.active[l].is_empty() && lv.inactive[l].is_empty() {
                // gap_heuristic(old_height)
                let start = (old_h + 1).max(0) as usize;
                let end = ((max_height + 1).max(0) as usize).min(nlevels);
                let top = level_index(nn + 1)?;
                for i in start..end.max(start) {
                    for u in lv.active[i].iter().chain(lv.inactive[i].iter()) {
                        height[u as usize] = nn + 1;
                    }
                    merge_sets(&mut lv.active, top, i, hashes);
                    lv.active[i].clear();
                    merge_sets(&mut lv.inactive, top, i, hashes);
                    lv.inactive[i].clear();
                }
                h = old_h - 1;
                max_height = h;
            } else {
                max_height = max_height.max(h);
            }
        }
    }
    let state = |height: Vec<i64>, curr: Vec<u32>| {
        Some(NodeState {
            height: height.into_iter().map(|h| h as u32).collect(),
            curr,
        })
    };
    if value_only {
        return Ok(Preflow {
            value: excess[t as usize],
            excess,
            state: state(height, curr),
        });
    }
    // Phase 2: send the excess back to the source.
    let mut h = global_relabel(r, &mut lv, &mut height, false)?;
    work = 0;
    while h > nn {
        loop {
            let l = level_index(h)?;
            let Some(u) = lv.active[l].first() else {
                h -= 1;
                break;
            };
            h = discharge(
                r,
                &mut lv,
                &mut excess,
                &mut height,
                &mut curr,
                &mut work,
                u,
                false,
            )?;
            if reached(work) {
                h = global_relabel(r, &mut lv, &mut height, false)?;
                work = 0;
            }
        }
    }
    Ok(Preflow {
        value: excess[t as usize],
        excess,
        state: state(height, curr),
    })
}

// --- Running a flow algorithm ----------------------------------------------------------------

/// NetworkX's maximum flow functions, with the parameters rustnx needs
/// computed by the caller exactly as NetworkX computes them.
#[derive(Clone, Copy)]
pub enum Algo {
    EdmondsKarp,
    /// `two_phase` and the phase-one height limit `d`.
    ShortestAugmentingPath(bool, i64),
    Dinitz,
    BoykovKolmogorov,
    /// The global relabel threshold.
    PreflowPush(f64),
}

/// The flow value and what else the algorithm leaves on `R`.
pub struct Outcome {
    pub value: Val,
    pub state: Option<NodeState>,
    pub excess: Option<Vec<Val>>,
    pub trees: Option<Trees>,
}

/// One maximum flow computation on `r` (resetting its flows first), as
/// NetworkX's `flow_func(G, s, t, residual=R, cutoff=..., value_only=...)`.
pub fn run_flow(
    r: &mut Residual,
    algo: Algo,
    s: u32,
    t: u32,
    cutoff: Option<Val>,
    value_only: bool,
    hashes: &[i64],
) -> Res<Outcome> {
    let mut out = Outcome {
        value: I(0),
        state: None,
        excess: None,
        trees: None,
    };
    match algo {
        Algo::EdmondsKarp => {
            r.reset_flows();
            out.value = edmonds_karp_core(r, s, t, cutoff.unwrap_or(FLOAT_INF))?;
        }
        Algo::ShortestAugmentingPath(two_phase, d) => {
            let (value, state) =
                shortest_augmenting_path(r, s, t, two_phase, d, cutoff.unwrap_or(FLOAT_INF))?;
            out.value = value;
            out.state = state;
        }
        Algo::Dinitz => out.value = dinitz(r, s, t, cutoff)?,
        Algo::BoykovKolmogorov => {
            let (value, trees) = boykov_kolmogorov(r, s, t, cutoff)?;
            out.value = value;
            out.trees = Some(trees);
        }
        Algo::PreflowPush(threshold) => {
            if hashes.len() != r.n {
                return Err(Fail::Unsupported);
            }
            let p = preflow_push(r, s, t, threshold, value_only, hashes)?;
            out.value = p.value;
            out.excess = Some(p.excess);
            out.state = p.state;
        }
    }
    Ok(out)
}

// --- Gomory-Hu trees ----------------------------------------------------------------------------

/// `gomory_hu_tree` on `r` (built from G): each non-root node's tree
/// parent and edge weight, in node order. `strict_cut` as in `cut_order`
/// (NetworkX 3.7); otherwise the pre-3.7 `minimum_cut` also reorders `R`.
pub fn gomory_hu(
    r: &mut Residual,
    algo: Algo,
    hashes: &[i64],
    strict_cut: bool,
) -> Res<Vec<(u32, Val)>> {
    let n = r.n;
    let root = 0u32;
    let mut tree = vec![root; n];
    let mut labels: PairMapVal = PairMapVal::default();
    let mut non_reachable = vec![false; n];
    for source in 1..n as u32 {
        let target = tree[source as usize];
        let cut_value = run_flow(r, algo, source, target, None, true, hashes)?.value;
        let order = r.cut_order(target, strict_cut);
        if !strict_cut {
            r.move_saturated_last();
        }
        non_reachable.iter_mut().for_each(|x| *x = false);
        for &v in &order {
            non_reachable[v as usize] = true;
        }
        labels.insert(pair(source, target), cut_value);
        for node in 0..n as u32 {
            if !non_reachable[node as usize]
                && node != source
                && node != root
                && tree[node as usize] == target
            {
                tree[node as usize] = source;
                let value = labels
                    .get(&pair(node, target))
                    .copied()
                    .unwrap_or(cut_value);
                labels.insert(pair(node, source), value);
            }
        }
        let tt = tree[target as usize];
        if target != root && !non_reachable[tt as usize] {
            let value = *labels.get(&pair(target, tt)).ok_or(Fail::Unsupported)?;
            labels.insert(pair(source, tt), value);
            labels.insert(pair(target, source), cut_value);
            tree[source as usize] = tt;
            tree[target as usize] = source;
        }
    }
    (1..n as u32)
        .map(|u| {
            let p = tree[u as usize];
            labels
                .get(&pair(u, p))
                .map(|&w| (p, w))
                .ok_or(Fail::Unsupported)
        })
        .collect()
}

type PairMapVal = HashMap<u64, Val, BuildHasherDefault<PairHasher>>;

// --- Network simplex -------------------------------------------------------------------------------

/// Where `network_simplex` stops with an error (indices into its input).
#[derive(Debug, PartialEq)]
pub enum SimplexError {
    Fail(Fail),
    InfiniteDemand(usize),
    InfiniteWeight(usize),
    SelfLoopInfiniteWeight(usize),
    DemandNotZero,
    NegativeCapacity(usize),
    SelfLoopNegativeCapacity(usize),
    NoFeasibleFlow,
    Unbounded,
}

impl From<Fail> for SimplexError {
    fn from(f: Fail) -> Self {
        SimplexError::Fail(f)
    }
}

/// `network_simplex` on a DiGraph: `demands` per node, the edges that are
/// not self-loops and don't have zero capacity (`src`, `dst`, `cap`,
/// `weight`), and the self-loops' (capacity, weight). Missing capacities
/// are `float("inf")`, missing weights and demands `0`.
pub struct SimplexInput<'a> {
    pub demands: &'a [Val],
    pub src: &'a [u32],
    pub dst: &'a [u32],
    pub cap: &'a [Val],
    pub weight: &'a [Val],
    pub loops: &'a [(Val, Val)],
}

/// The cost and each edge's flow; for each self-loop, its flow (`None`
/// for a zero entry: capacity times weight is added for negative weights).
pub struct SimplexResult {
    pub cost: Val,
    pub flows: Vec<Val>,
    pub loop_flows: Vec<Option<Val>>,
}

/// NetworkX before 3.6 takes the largest single demand for the faux
/// infinity (`max_single_demand`) where later versions sum the demands.
pub fn network_simplex(
    inp: &SimplexInput,
    max_single_demand: bool,
    compensated: bool,
) -> Result<SimplexResult, SimplexError> {
    let n = inp.demands.len();
    let e_count = inp.src.len();
    for (i, d) in inp.demands.iter().enumerate() {
        if d.abs()?.eq(FLOAT_INF) {
            return Err(SimplexError::InfiniteDemand(i));
        }
    }
    for (i, w) in inp.weight.iter().enumerate() {
        if w.abs()?.eq(FLOAT_INF) {
            return Err(SimplexError::InfiniteWeight(i));
        }
    }
    for (i, &(_, w)) in inp.loops.iter().enumerate() {
        if w.abs()?.eq(FLOAT_INF) {
            return Err(SimplexError::SelfLoopInfiniteWeight(i));
        }
    }
    if py_sum(inp.demands.iter().copied(), compensated)?.truthy_ne_zero() {
        return Err(SimplexError::DemandNotZero);
    }
    for (i, c) in inp.cap.iter().enumerate() {
        if c.lt(I(0)) {
            return Err(SimplexError::NegativeCapacity(i));
        }
    }
    for (i, &(c, _)) in inp.loops.iter().enumerate() {
        if c.lt(I(0)) {
            return Err(SimplexError::SelfLoopNegativeCapacity(i));
        }
    }
    // Node n is the artificial root (-1 in NetworkX, which indexes lists
    // from the end: `parent_edge` and `node_potentials` have only n
    // entries, so the root's are node n - 1's there).
    let root = n;
    let mut src: Vec<usize> = inp.src.iter().map(|&u| u as usize).collect();
    let mut dst: Vec<usize> = inp.dst.iter().map(|&v| v as usize).collect();
    for (i, d) in inp.demands.iter().enumerate() {
        if d.gt(I(0)) {
            src.push(root);
            dst.push(i);
        } else {
            src.push(i);
            dst.push(root);
        }
    }
    let cap_sum = py_sum(
        inp.cap.iter().copied().filter(|c| c.lt(FLOAT_INF)),
        compensated,
    )?;
    let mut abs_w = Vec::with_capacity(e_count);
    for w in inp.weight {
        abs_w.push(w.abs()?);
    }
    let w_sum = py_sum(abs_w, compensated)?;
    let mut abs_d = Vec::with_capacity(n);
    for d in inp.demands {
        abs_d.push(d.abs()?);
    }
    let biggest = if max_single_demand {
        abs_d.iter().fold(cap_sum.max(w_sum), |m, &d| m.max(d))
    } else {
        cap_sum
            .max(w_sum)
            .max(py_sum(abs_d.iter().copied(), compensated)?)
    };
    let mut faux_inf = I(3).mul(biggest)?;
    if !faux_inf.truthy() {
        faux_inf = I(1);
    }
    let mut weight: Vec<Val> = inp.weight.to_vec();
    weight.extend(std::iter::repeat_n(faux_inf, n));
    let mut cap: Vec<Val> = inp.cap.to_vec();
    cap.extend(std::iter::repeat_n(faux_inf, n));

    // initialize_spanning_tree
    const NONE: usize = usize::MAX;
    let mut flow: Vec<Val> = std::iter::repeat_n(I(0), e_count).collect();
    flow.extend(abs_d.iter().copied());
    let mut potential: Vec<Val> = Vec::with_capacity(n);
    for d in inp.demands {
        potential.push(if d.le(I(0)) {
            faux_inf
        } else {
            faux_inf.neg()?
        });
    }
    let mut parent: Vec<usize> = vec![root; n];
    parent.push(NONE);
    let mut parent_edge: Vec<usize> = (e_count..e_count + n).collect();
    let mut subtree_size: Vec<usize> = vec![1; n];
    subtree_size.push(n + 1);
    let mut next_dft: Vec<usize> = (1..n).collect();
    next_dft.push(root);
    next_dft.push(0);
    let mut prev_dft: Vec<usize> = vec![root];
    prev_dft.extend(0..n);
    let mut last_dft: Vec<usize> = (0..n).collect();
    last_dft.push(n - 1);
    let pidx = |x: usize| if x == root { n - 1 } else { x };

    let reduced_cost = |i: usize, potential: &[Val], flow: &[Val]| -> Res<Val> {
        let c = weight[i]
            .sub(potential[pidx(src[i])])?
            .add(potential[pidx(dst[i])])?;
        if flow[i].eq(I(0)) {
            Ok(c)
        } else {
            c.neg()
        }
    };
    let residual_capacity = |i: usize, p: usize, flow: &[Val]| -> Res<Val> {
        if src[i] == p {
            cap[i].sub(flow[i])
        } else {
            Ok(flow[i])
        }
    };

    if e_count > 0 {
        let b = (e_count as f64).sqrt().ceil() as usize;
        let blocks = e_count.div_ceil(b);
        let mut m = 0usize;
        let mut f = 0usize;
        while m < blocks {
            // find_entering_edges: the first edge of smallest reduced cost
            // in the next block of b edges (wrapping around).
            let mut l = f + b;
            let (r1, r2) = if l <= e_count {
                (f..l, 0..0)
            } else {
                l -= e_count;
                (f..e_count, 0..l)
            };
            f = l;
            let mut best: Option<(usize, Val)> = None;
            for i in r1.chain(r2) {
                let c = reduced_cost(i, &potential, &flow)?;
                if best.is_none_or(|(_, bc)| c.lt(bc)) {
                    best = Some((i, c));
                }
            }
            let (i, _) = best.ok_or(SimplexError::Fail(Fail::Unsupported))?;
            let c = reduced_cost(i, &potential, &flow)?;
            if c.ge(I(0)) {
                m += 1;
                continue;
            }
            let (mut p, mut q) = if flow[i].eq(I(0)) {
                (src[i], dst[i])
            } else {
                (dst[i], src[i])
            };
            m = 0;

            // find_cycle(i, p, q)
            // A parent of None would make NetworkX fail with a TypeError.
            let up = |x: usize| -> Res<usize> {
                match parent[x] {
                    NONE => Err(Fail::Unsupported),
                    p => Ok(p),
                }
            };
            let apex = {
                let (mut a, mut bq) = (p, q);
                let mut size_p = subtree_size[a];
                let mut size_q = subtree_size[bq];
                loop {
                    while size_p < size_q {
                        a = up(a)?;
                        size_p = subtree_size[a];
                    }
                    while size_p > size_q {
                        bq = up(bq)?;
                        size_q = subtree_size[bq];
                    }
                    if size_p == size_q {
                        if a != bq {
                            a = up(a)?;
                            size_p = subtree_size[a];
                            bq = up(bq)?;
                            size_q = subtree_size[bq];
                        } else {
                            break a;
                        }
                    }
                }
            };
            let trace = |mut x: usize| -> Res<(Vec<usize>, Vec<usize>)> {
                let mut wn = vec![x];
                let mut we = Vec::new();
                while x != apex {
                    match parent_edge[pidx(x)] {
                        NONE => return Err(Fail::Unsupported),
                        e => we.push(e),
                    }
                    x = up(x)?;
                    wn.push(x);
                }
                Ok((wn, we))
            };
            let (mut wn, mut we) = trace(p)?;
            wn.reverse();
            we.reverse();
            if we != [i] {
                we.push(i);
            }
            let (mut wnr, wer) = trace(q)?;
            wnr.pop();
            wn.extend(wnr);
            we.extend(wer);

            // find_leaving_edge(Wn, We)
            let mut leave: Option<(usize, usize, Val)> = None;
            for k in 0..we.len().min(wn.len()) {
                let (j, sj) = (we[we.len() - 1 - k], wn[wn.len() - 1 - k]);
                let rc = residual_capacity(j, sj, &flow)?;
                if leave.is_none_or(|(_, _, best)| rc.lt(best)) {
                    leave = Some((j, sj, rc));
                }
            }
            let (j, mut s, _) = leave.ok_or(SimplexError::Fail(Fail::Unsupported))?;
            let mut t = if src[j] == s { dst[j] } else { src[j] };
            // augment_flow(Wn, We, residual_capacity(j, s))
            let delta = residual_capacity(j, s, &flow)?;
            for (&ei, &pn) in we.iter().zip(wn.iter()) {
                flow[ei] = if src[ei] == pn {
                    flow[ei].add(delta)?
                } else {
                    flow[ei].sub(delta)?
                };
            }
            if i == j {
                continue;
            }
            if parent[t] != s {
                std::mem::swap(&mut s, &mut t);
            }
            let pos = |x: usize| we.iter().position(|&y| y == x);
            if pos(i) > pos(j) {
                std::mem::swap(&mut p, &mut q);
            }
            // remove_edge(s, t)
            {
                let size_t = subtree_size[t];
                let prev_t = prev_dft[t];
                let last_t = last_dft[t];
                let next_last_t = next_dft[last_t];
                parent[t] = NONE;
                parent_edge[pidx(t)] = NONE;
                next_dft[prev_t] = next_last_t;
                prev_dft[next_last_t] = prev_t;
                next_dft[last_t] = t;
                prev_dft[t] = last_t;
                let mut x = s;
                while x != NONE {
                    subtree_size[x] = subtree_size[x]
                        .checked_sub(size_t)
                        .ok_or(Fail::Unsupported)?;
                    if last_dft[x] == last_t {
                        last_dft[x] = prev_t;
                    }
                    x = parent[x];
                }
            }
            // make_root(q)
            {
                let mut ancestors = Vec::new();
                let mut x = q;
                while x != NONE {
                    ancestors.push(x);
                    x = parent[x];
                }
                ancestors.reverse();
                for w in ancestors.windows(2) {
                    let (p, q) = (w[0], w[1]);
                    let size_p = subtree_size[p];
                    let mut last_p = last_dft[p];
                    let prev_q = prev_dft[q];
                    let last_q = last_dft[q];
                    let next_last_q = next_dft[last_q];
                    parent[p] = q;
                    parent[q] = NONE;
                    parent_edge[pidx(p)] = parent_edge[pidx(q)];
                    parent_edge[pidx(q)] = NONE;
                    subtree_size[p] = size_p
                        .checked_sub(subtree_size[q])
                        .ok_or(Fail::Unsupported)?;
                    subtree_size[q] = size_p;
                    next_dft[prev_q] = next_last_q;
                    prev_dft[next_last_q] = prev_q;
                    next_dft[last_q] = q;
                    prev_dft[q] = last_q;
                    if last_p == last_q {
                        last_dft[p] = prev_q;
                        last_p = prev_q;
                    }
                    prev_dft[p] = last_q;
                    next_dft[last_q] = p;
                    next_dft[last_p] = q;
                    prev_dft[q] = last_p;
                    last_dft[q] = last_p;
                }
            }
            // add_edge(i, p, q)
            {
                let last_p = last_dft[p];
                let next_last_p = next_dft[last_p];
                let size_q = subtree_size[q];
                let last_q = last_dft[q];
                parent[q] = p;
                parent_edge[pidx(q)] = i;
                next_dft[last_p] = q;
                prev_dft[q] = last_p;
                prev_dft[next_last_p] = last_q;
                next_dft[last_q] = next_last_p;
                let mut x = p;
                while x != NONE {
                    subtree_size[x] += size_q;
                    if last_dft[x] == last_p {
                        last_dft[x] = last_q;
                    }
                    x = parent[x];
                }
            }
            // update_potentials(i, p, q)
            {
                let d = if q == dst[i] {
                    potential[pidx(p)].sub(weight[i])?.sub(potential[pidx(q)])?
                } else {
                    potential[pidx(p)].add(weight[i])?.sub(potential[pidx(q)])?
                };
                let mut x = q;
                let last = last_dft[q];
                loop {
                    potential[pidx(x)] = potential[pidx(x)].add(d)?;
                    if x == last {
                        break;
                    }
                    x = next_dft[x];
                }
            }
        }
    }
    if flow[e_count..].iter().any(|&f| !f.eq(I(0))) {
        return Err(SimplexError::NoFeasibleFlow);
    }
    for f in &flow[..e_count] {
        if f.mul(I(2))?.ge(faux_inf) {
            return Err(SimplexError::Unbounded);
        }
    }
    if inp
        .loops
        .iter()
        .any(|&(c, w)| c.eq(FLOAT_INF) && w.lt(I(0)))
    {
        return Err(SimplexError::Unbounded);
    }
    flow.truncate(e_count);
    let mut products = Vec::with_capacity(e_count);
    for (w, x) in inp.weight.iter().zip(&flow) {
        products.push(w.mul(*x)?);
    }
    let mut cost = py_sum(products, compensated)?;
    let mut loop_flows = Vec::with_capacity(inp.loops.len());
    for &(c, w) in inp.loops {
        if w.ge(I(0)) {
            loop_flows.push(None);
        } else {
            cost = cost.add(w.mul(c)?)?;
            loop_flows.push(Some(c));
        }
    }
    Ok(SimplexResult {
        cost,
        flows: flow,
        loop_flows,
    })
}

impl Val {
    /// `x != 0`.
    fn truthy_ne_zero(self) -> bool {
        !self.eq(I(0))
    }
}

// --- Cut measures ---------------------------------------------------------------------------------

/// Edge weights as Python numbers, per arc of the adjacency.
pub enum Weights<'a> {
    /// No weight: each edge counts 1.
    Unit,
    /// The converted values, all ints (`true`) or all floats.
    Stored(&'a [f64], bool),
    /// Read from the NetworkX graph, ints and floats mixed.
    Exact(Vec<Val>),
}

impl Weights<'_> {
    #[inline]
    fn at(&self, e: usize) -> Val {
        match self {
            Weights::Unit => I(1),
            Weights::Stored(w, true) => I(w[e] as i64),
            Weights::Stored(w, false) => F(w[e]),
            Weights::Exact(w) => w[e],
        }
    }
}

/// `nx.edge_boundary(G, nbunch1, nbunch2)` as positions of `adj` arcs:
/// `G.edges(nset1)` with `nset1` in iteration order `order`, filtered.
pub fn boundary_arcs(
    adj: &crate::graph::Csr,
    n: usize,
    directed: bool,
    order: &[u32],
    nset2: Option<&[u32]>,
) -> Vec<usize> {
    let mut in1 = vec![false; n];
    for &v in order {
        in1[v as usize] = true;
    }
    let in2 = nset2.map(|s| {
        let mut m = vec![false; n];
        for &v in s {
            m[v as usize] = true;
        }
        m
    });
    // `EdgeDataView` skips neighbors already reported as sources.
    let mut seen = vec![false; n];
    let mut out = Vec::new();
    for &u in order {
        for e in adj.range(u as usize) {
            let v = adj.targets[e] as usize;
            if !directed && seen[v] {
                continue;
            }
            let keep = match &in2 {
                None => !in1[v],
                Some(in2) => in2[v] || (in1[v] && in2[u as usize]),
            };
            if keep {
                out.push(e);
            }
        }
        seen[u as usize] = true;
    }
    out
}

/// `cut_size`: the `sum()` of the weights of the boundary edges of each
/// `(nset1 order, nset2)` part in turn (two parts for directed graphs).
pub fn cut_size(
    adj: &crate::graph::Csr,
    n: usize,
    directed: bool,
    w: &Weights,
    parts: &[(Vec<u32>, Option<Vec<u32>>)],
    compensated: bool,
) -> Res<Val> {
    let mut arcs = Vec::new();
    for (order, nset2) in parts {
        arcs.extend(boundary_arcs(adj, n, directed, order, nset2.as_deref()));
    }
    py_sum(arcs.into_iter().map(|e| w.at(e)), compensated)
}

/// `volume`: the `sum()` of `G.degree` (undirected) or `G.out_degree`
/// values of `nodes` (duplicates count again), each degree being the
/// `sum()` of its edge weights (plus a self-loop's weight again for
/// undirected graphs).
pub fn volume(
    adj: &crate::graph::Csr,
    directed: bool,
    w: &Weights,
    nodes: &[u32],
    compensated: bool,
) -> Res<Val> {
    let mut degrees = Vec::with_capacity(nodes.len());
    for &u in nodes {
        let range = adj.range(u as usize);
        let self_loop = (!directed)
            .then(|| range.clone().find(|&e| adj.targets[e] == u))
            .flatten();
        let mut d = match w {
            Weights::Unit => I(range.len() as i64),
            _ => py_sum(range.map(|e| w.at(e)), compensated)?,
        };
        if let Some(e) = self_loop {
            d = d.add(w.at(e))?;
        }
        degrees.push(d);
    }
    py_sum(degrees, compensated)
}
