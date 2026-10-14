//! NetworkX's `lukes_partitioning` replayed: the same tree `t_G` (a DFS tree
//! from the chosen root, or the arborescence itself), the same choice of the
//! next parent of leaves (the first in CPython's order of
//! `set(t_G.nodes) - set(leaves)`), its children in the order of the set
//! `nx.descendants` builds, and the same partitions and values compared in
//! the same order. Values follow Python's `sum()` over ints and floats
//! (including 3.12+'s compensated float sums), so ties and float rounding
//! go the same way.
//!
//! Clusters are kept as `{v}` leaves and `ccx.union(cci)` pairs; the caller
//! rebuilds the result's sets with the same calls, so they iterate the same.

use std::collections::HashMap;
use std::rc::Rc;

use super::pyset::PySet;

/// A Python int (bounded by the caller) or float.
#[derive(Clone, Copy, Debug)]
pub enum Val {
    I(i64),
    F(f64),
}

impl Val {
    fn as_f64(self) -> f64 {
        match self {
            Val::I(v) => v as f64,
            Val::F(v) => v,
        }
    }

    /// `self < other`, exact while ints stay below 2**53 (the caller's bound).
    fn lt(self, other: Val) -> bool {
        match (self, other) {
            (Val::I(a), Val::I(b)) => a < b,
            _ => self.as_f64() < other.as_f64(),
        }
    }

    fn le(self, other: Val) -> bool {
        match (self, other) {
            (Val::I(a), Val::I(b)) => a <= b,
            _ => self.as_f64() <= other.as_f64(),
        }
    }
}

/// How this interpreter's `sum()` adds floats: plainly (3.10, 3.11), with
/// Neumaier compensation for floats only (3.12), or for ints in the float
/// loop as well (3.14).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SumMode {
    Plain,
    Floats,
    All,
}

/// `sum(items)` from the int 0.
struct PySum {
    mode: SumMode,
    int: i64,
    float: bool,
    hi: f64,
    lo: f64,
}

impl PySum {
    fn new(mode: SumMode) -> Self {
        PySum {
            mode,
            int: 0,
            float: false,
            hi: 0.0,
            lo: 0.0,
        }
    }

    fn compensated(&mut self, x: f64) {
        let t = self.hi + x;
        if self.hi.abs() >= x.abs() {
            self.lo += (self.hi - t) + x;
        } else {
            self.lo += (x - t) + self.hi;
        }
        self.hi = t;
    }

    fn add(&mut self, v: Val) {
        match (self.float, v) {
            (false, Val::I(x)) => self.int += x,
            // The int loop's result plus the first float, then the float loop.
            (false, Val::F(x)) => {
                self.float = true;
                self.hi = self.int as f64 + x;
            }
            (true, Val::F(x)) => {
                if self.mode == SumMode::Plain {
                    self.hi += x;
                } else {
                    self.compensated(x);
                }
            }
            (true, Val::I(x)) => {
                if self.mode == SumMode::All {
                    self.compensated(x as f64);
                } else {
                    self.hi += x as f64;
                }
            }
        }
    }

    fn finish(self) -> Val {
        if !self.float {
            Val::I(self.int)
        } else if self.mode != SumMode::Plain && self.lo != 0.0 && self.lo.is_finite() {
            Val::F(self.hi + self.lo)
        } else {
            Val::F(self.hi)
        }
    }
}

/// A cluster: `{node}` (`b == NONE`) or `clusters[a].union(clusters[b])`.
struct Cluster {
    a: u32,
    b: u32,
    weight: i64,
    /// Indices (in `safe_G.edges` order) of the edges inside the cluster.
    edges: Rc<[u32]>,
    value: Val,
}

const NONE: u32 = u32::MAX;

/// A partition (a list of clusters) and the position of the cluster
/// holding the node whose `PKEY` dict it sits in.
#[derive(Clone)]
struct Part {
    sets: Rc<[u32]>,
    pos: u32,
    value: Val,
}

/// An option `_concatenate_or_merge` returned, built when kept.
#[derive(Clone)]
struct Opt {
    p1: Part,
    p2: Part,
    merged: Option<u32>,
    value: Val,
}

/// The input tree and weights.
pub struct Input {
    /// `t_G`'s nodes in its order, and each node's children in `t_G.succ`
    /// order (indices of `G`'s nodes).
    pub order: Vec<u32>,
    pub children: Vec<Vec<u32>>,
    /// For each node, the index of the edge to its parent (`safe_G.edges`
    /// order); unused for the root.
    pub parent_edge: Vec<u32>,
    pub node_weight: Vec<i64>,
    pub edge_weight: Vec<Val>,
    pub hashes: Vec<i64>,
    pub root: u32,
    pub max_size: i64,
    pub mode: SumMode,
}

/// The result: `None` where NetworkX returns None, or the clusters to
/// build (`(node, NONE)` for `{node}`, `(i, j)` for `built[i].union(built[j])`)
/// and the partition (indices into them). `Err(())` where NetworkX raises
/// `TypeError` (a `None` partition reached `_pivot`).
pub type Output = Option<(Vec<(u32, u32)>, Vec<u32>)>;

struct State<'a> {
    inp: &'a Input,
    clusters: Vec<Cluster>,
    unions: HashMap<(u32, u32), u32>,
    ints: bool,
}

impl State<'_> {
    fn leaf(&mut self, v: u32) -> u32 {
        self.clusters.push(Cluster {
            a: v,
            b: NONE,
            weight: self.inp.node_weight[v as usize],
            edges: Rc::from(Vec::new()),
            value: Val::I(0),
        });
        (self.clusters.len() - 1) as u32
    }

    fn union(&mut self, a: u32, b: u32, edge: u32) -> u32 {
        if let Some(&id) = self.unions.get(&(a, b)) {
            return id;
        }
        let (ea, eb) = (
            &self.clusters[a as usize].edges,
            &self.clusters[b as usize].edges,
        );
        let mut edges = Vec::with_capacity(ea.len() + eb.len() + 1);
        let (mut i, mut j) = (0, 0);
        let mut extra = Some(edge);
        loop {
            let mut next = u32::MAX;
            let mut which = 3;
            if i < ea.len() && ea[i] < next {
                next = ea[i];
                which = 0;
            }
            if j < eb.len() && eb[j] < next {
                next = eb[j];
                which = 1;
            }
            if let Some(e) = extra {
                if e < next {
                    next = e;
                    which = 2;
                }
            }
            match which {
                0 => i += 1,
                1 => j += 1,
                2 => extra = None,
                _ => break,
            }
            edges.push(next);
        }
        let mut sum = PySum::new(self.inp.mode);
        for &e in &edges {
            sum.add(self.inp.edge_weight[e as usize]);
        }
        let weight = self.clusters[a as usize].weight + self.clusters[b as usize].weight;
        self.clusters.push(Cluster {
            a,
            b,
            weight,
            edges: Rc::from(edges),
            value: sum.finish(),
        });
        let id = (self.clusters.len() - 1) as u32;
        self.unions.insert((a, b), id);
        id
    }

    fn value_of(&self, sets: impl Iterator<Item = u32>) -> Val {
        let mut sum = PySum::new(self.inp.mode);
        for c in sets {
            sum.add(self.clusters[c as usize].value);
        }
        sum.finish()
    }

    /// `_concatenate_or_merge(p1, p2, x, i, j)`.
    fn concatenate_or_merge(&mut self, p1: &Part, p2: &Part, i: u32, j: i64) -> Opt {
        let ccx = p1.sets[p1.pos as usize];
        let cci = p2.sets[p2.pos as usize];
        let weight = self.clusters[ccx as usize].weight + self.clusters[cci as usize].weight;
        if weight <= j {
            let merged = self.union(ccx, cci, self.inp.parent_edge[i as usize]);
            let value = if self.ints {
                // Int sums are exact, so the order doesn't matter.
                let get = |c: u32| match self.clusters[c as usize].value {
                    Val::I(v) => v,
                    Val::F(_) => unreachable!(),
                };
                match (p1.value, p2.value) {
                    (Val::I(a), Val::I(b)) => Val::I(get(merged) + a - get(ccx) + b - get(cci)),
                    _ => unreachable!(),
                }
            } else {
                let (s1, s2) = (p1.pos as usize, p2.pos as usize);
                self.value_of(
                    std::iter::once(merged)
                        .chain(
                            p1.sets
                                .iter()
                                .enumerate()
                                .filter(|&(k, _)| k != s1)
                                .map(|(_, &c)| c),
                        )
                        .chain(
                            p2.sets
                                .iter()
                                .enumerate()
                                .filter(|&(k, _)| k != s2)
                                .map(|(_, &c)| c),
                        ),
                )
            };
            Opt {
                p1: p1.clone(),
                p2: p2.clone(),
                merged: Some(merged),
                value,
            }
        } else {
            let value = match (self.ints, p1.value, p2.value) {
                (true, Val::I(a), Val::I(b)) => Val::I(a + b),
                _ => self.value_of(p1.sets.iter().chain(p2.sets.iter()).copied()),
            };
            Opt {
                p1: p1.clone(),
                p2: p2.clone(),
                merged: None,
                value,
            }
        }
    }
}

impl Opt {
    fn build(&self) -> Part {
        match self.merged {
            Some(m) => {
                let (s1, s2) = (self.p1.pos as usize, self.p2.pos as usize);
                let sets: Vec<u32> = std::iter::once(m)
                    .chain(
                        self.p1
                            .sets
                            .iter()
                            .enumerate()
                            .filter(|&(k, _)| k != s1)
                            .map(|(_, &c)| c),
                    )
                    .chain(
                        self.p2
                            .sets
                            .iter()
                            .enumerate()
                            .filter(|&(k, _)| k != s2)
                            .map(|(_, &c)| c),
                    )
                    .collect();
                Part {
                    sets: Rc::from(sets),
                    pos: 0,
                    value: self.value,
                }
            }
            None => {
                let sets: Vec<u32> = self
                    .p1
                    .sets
                    .iter()
                    .chain(self.p2.sets.iter())
                    .copied()
                    .collect();
                Part {
                    sets: Rc::from(sets),
                    pos: self.p1.pos,
                    value: self.value,
                }
            }
        }
    }
}

/// Runs the algorithm.
pub fn run(inp: &Input) -> Result<Output, ()> {
    let n = inp.children.len();
    let ints = inp.edge_weight.iter().all(|w| matches!(w, Val::I(_)));
    let mut st = State {
        inp,
        clusters: Vec::new(),
        unions: HashMap::new(),
        ints,
    };
    let mut children = inp.children.clone();
    let mut alive = vec![true; n];
    let mut order = inp.order.clone();
    // PKEY: per node, weight -> partition (None after a node found none).
    let mut pkey: Vec<HashMap<i64, Option<Part>>> = vec![HashMap::new(); n];
    for &v in &order {
        let c = st.leaf(v);
        let slot = inp.node_weight[v as usize];
        let part = Part {
            sets: Rc::from(vec![c]),
            pos: 0,
            value: Val::I(0),
        };
        pkey[v as usize].insert(slot, Some(part.clone()));
        if children[v as usize].is_empty() {
            pkey[v as usize].insert(0, Some(part));
        }
    }
    let hashes = &inp.hashes;
    loop {
        // _a_parent_of_leaves_only(t_G)
        let leaves = PySet::from_iter(
            order
                .iter()
                .copied()
                .filter(|&v| children[v as usize].is_empty()),
            hashes,
        );
        let all = PySet::from_iter(order.iter().copied(), hashes);
        let rest = all.difference(&leaves, hashes);
        let Some(x) = rest.iter().find(|&v| {
            children[v as usize]
                .iter()
                .all(|&c| children[c as usize].is_empty())
        }) else {
            // Only a lone root is left; the caller declines that case.
            return Ok(None);
        };
        let xu = x as usize;
        let weight_of_x = inp.node_weight[xu];
        let mut best_value = Val::I(0);
        let mut best: Option<Opt> = None;
        let desc = PySet::from_iter(children[xu].iter().copied(), hashes);
        for i in desc.iter() {
            let iu = i as usize;
            // bp_buffer, by j - weight_of_x (its order doesn't matter: it
            // only fills PKEY's slots).
            let span = if inp.max_size >= weight_of_x {
                (inp.max_size - weight_of_x + 1) as usize
            } else {
                0
            };
            let mut buffer: Vec<Option<Opt>> = vec![None; span];
            for (slot, j) in (weight_of_x..=inp.max_size).enumerate() {
                for a in weight_of_x..=j {
                    let b = j - a;
                    let (Some(p1), Some(p2)) = (pkey[xu].get(&a), pkey[iu].get(&b)) else {
                        continue;
                    };
                    let (Some(p1), Some(p2)) = (p1, p2) else {
                        return Err(());
                    };
                    let (p1, p2) = (p1.clone(), p2.clone());
                    let opt = st.concatenate_or_merge(&p1, &p2, i, j);
                    match &buffer[slot] {
                        Some(kept) if !kept.value.lt(opt.value) => {}
                        _ => buffer[slot] = Some(opt.clone()),
                    }
                    if best_value.le(opt.value) {
                        best_value = opt.value;
                        best = Some(opt);
                    }
                }
            }
            for (slot, opt) in buffer.into_iter().enumerate() {
                if let Some(opt) = opt {
                    pkey[xu].insert(weight_of_x + slot as i64, Some(opt.build()));
                }
            }
        }
        pkey[xu].insert(0, best.map(|o| o.build()));
        for c in std::mem::take(&mut children[xu]) {
            alive[c as usize] = false;
        }
        order.retain(|&v| alive[v as usize]);
        if x == inp.root {
            let Some(part) = pkey[xu].get(&0).cloned().flatten() else {
                return Ok(None);
            };
            return Ok(Some(st.recipe(&part)));
        }
    }
}

impl State<'_> {
    /// The clusters `part` needs, children before parents, and its sets.
    fn recipe(&self, part: &Part) -> (Vec<(u32, u32)>, Vec<u32>) {
        let mut index: HashMap<u32, u32> = HashMap::new();
        let mut out: Vec<(u32, u32)> = Vec::new();
        let mut sets = Vec::with_capacity(part.sets.len());
        for &top in part.sets.iter() {
            let mut stack = vec![(top, false)];
            while let Some((c, expanded)) = stack.pop() {
                if index.contains_key(&c) {
                    continue;
                }
                let cl = &self.clusters[c as usize];
                if cl.b == NONE {
                    index.insert(c, out.len() as u32);
                    out.push((cl.a, NONE));
                } else if expanded {
                    index.insert(c, out.len() as u32);
                    out.push((index[&cl.a], index[&cl.b]));
                } else {
                    stack.push((c, true));
                    stack.push((cl.b, false));
                    stack.push((cl.a, false));
                }
            }
            sets.push(index[&top]);
        }
        (out, sets)
    }
}
