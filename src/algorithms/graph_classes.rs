//! Batch 9: planarity, chordal graphs and graph classes. The planarity code
//! ports NetworkX's left-right planarity test (`LRPlanarity`) step for step,
//! so that the embedding it builds is NetworkX's, half-edge for half-edge.

use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;

use crate::algorithms::structure_more::PairHasher;
use crate::graph::Csr;

const NONE: u32 = u32::MAX;

type PairMap = HashMap<u64, u32, BuildHasherDefault<PairHasher>>;
type PairSet = HashSet<u64, BuildHasherDefault<PairHasher>>;

#[inline]
fn key(u: u32, v: u32) -> u64 {
    let (a, b) = if u < v { (u, v) } else { (v, u) };
    ((a as u64) << 32) | b as u64
}

/// NetworkX raises (or would misbehave) here; rustnx lets it run instead.
#[derive(Debug)]
pub struct Bail;

/// The simple undirected graph `LRPlanarity` (and `nx.Graph(G)`) builds:
/// G's edges in `G.edges` order, self-loops dropped, reciprocal arcs once.
pub fn planarity_graph(succ: &Csr, n: usize, directed: bool) -> Vec<Vec<u32>> {
    let mut adj = vec![Vec::new(); n];
    let mut seen = PairSet::default();
    for u in 0..n {
        for &v in succ.neighbors(u) {
            let vi = v as usize;
            if vi == u || (!directed && vi < u) || (directed && !seen.insert(key(u as u32, v))) {
                continue;
            }
            adj[u].push(v);
            adj[vi].push(u as u32);
        }
    }
    adj
}

#[derive(Clone, Copy)]
struct Interval {
    low: u32,
    high: u32,
    /// The default `Interval()` that every `ConflictPair()` shares (a
    /// mutable default argument in NetworkX). Changing it would change
    /// later calls too, so rustnx bails out if NetworkX would.
    shared: bool,
}

impl Interval {
    const SHARED: Interval = Interval {
        low: NONE,
        high: NONE,
        shared: true,
    };

    fn new(low: u32, high: u32) -> Self {
        Interval {
            low,
            high,
            shared: false,
        }
    }

    fn empty(&self) -> bool {
        self.low == NONE && self.high == NONE
    }

    fn copy(&self) -> Self {
        Interval::new(self.low, self.high)
    }
}

struct ConflictPair {
    left: Interval,
    right: Interval,
    id: u32,
}

impl ConflictPair {
    fn swap(&mut self) {
        std::mem::swap(&mut self.left, &mut self.right);
    }
}

/// One `add_half_edge` call of the final embedding phase.
pub enum HalfEdge {
    /// `add_half_edge_first(a, b)`
    First(u32, u32),
    /// `add_half_edge(a, b, ccw=r)`
    Ccw(u32, u32, u32),
    /// `add_half_edge(a, b, cw=r)`
    Cw(u32, u32, u32),
}

/// What `lr_planarity` built: per node, the half-edges added first (in
/// order, each `ccw` of the previous), then the depth-first calls.
pub struct Embedding {
    pub ordered: Vec<Vec<u32>>,
    pub calls: Vec<HalfEdge>,
}

pub struct Planarity {
    pub planar: bool,
    pub embedding: Option<Embedding>,
    /// The deepest recursion the recursive variant would need.
    pub depth: usize,
}

struct Lr<'a> {
    adj: &'a [Vec<u32>],
    ids: Vec<Vec<u32>>,
    n: usize,
    src: Vec<u32>,
    dst: Vec<u32>,
    oriented: Vec<bool>,
    height: Vec<u32>,
    lowpt: Vec<u32>,
    lowpt2: Vec<u32>,
    nesting: Vec<i64>,
    parent_edge: Vec<u32>,
    out: Vec<Vec<u32>>,
    ordered: Vec<Vec<u32>>,
    refs: Vec<u32>,
    side: Vec<i64>,
    stack: Vec<ConflictPair>,
    next_id: u32,
    stack_bottom: Vec<u32>,
    lowpt_edge: Vec<u32>,
    roots: Vec<u32>,
    depth: usize,
}

impl<'a> Lr<'a> {
    fn new(adj: &'a [Vec<u32>]) -> Self {
        let n = adj.len();
        let mut map = PairMap::default();
        let mut m = 0u32;
        let ids: Vec<Vec<u32>> = adj
            .iter()
            .enumerate()
            .map(|(u, row)| {
                row.iter()
                    .map(|&v| {
                        *map.entry(key(u as u32, v)).or_insert_with(|| {
                            m += 1;
                            m - 1
                        })
                    })
                    .collect()
            })
            .collect();
        let m = m as usize;
        Lr {
            adj,
            ids,
            n,
            src: vec![NONE; m],
            dst: vec![NONE; m],
            oriented: vec![false; m],
            height: vec![NONE; n],
            lowpt: vec![0; m],
            lowpt2: vec![0; m],
            nesting: vec![0; m],
            parent_edge: vec![NONE; n],
            out: vec![Vec::new(); n],
            ordered: Vec::new(),
            refs: vec![NONE; m],
            side: vec![1; m],
            stack: Vec::new(),
            next_id: 0,
            stack_bottom: vec![NONE; m],
            lowpt_edge: vec![NONE; m],
            roots: Vec::new(),
            depth: 0,
        }
    }

    fn lowpt_of(&self, e: u32) -> Result<u32, Bail> {
        // NetworkX's `lowpt` is a plain dict: `lowpt[None]` raises.
        if e == NONE {
            Err(Bail)
        } else {
            Ok(self.lowpt[e as usize])
        }
    }

    fn set_ref(&mut self, e: u32, value: u32) {
        // `ref[None] = ...` lands in a key NetworkX never reads.
        if e != NONE {
            self.refs[e as usize] = value;
        }
    }

    fn side_of(&self, e: u32) -> i64 {
        if e == NONE {
            1
        } else {
            self.side[e as usize]
        }
    }

    fn top_id(&self) -> u32 {
        self.stack.last().map_or(NONE, |p| p.id)
    }

    fn new_pair(&mut self, left: Interval, right: Interval) -> ConflictPair {
        self.next_id += 1;
        ConflictPair {
            left,
            right,
            id: self.next_id,
        }
    }

    fn conflicting(&self, i: &Interval, b: u32) -> Result<bool, Bail> {
        Ok(!i.empty() && self.lowpt_of(i.high)? > self.lowpt_of(b)?)
    }

    fn lowest(&self, p: &ConflictPair) -> Result<u32, Bail> {
        if p.left.empty() {
            return self.lowpt_of(p.right.low);
        }
        if p.right.empty() {
            return self.lowpt_of(p.left.low);
        }
        Ok(self.lowpt_of(p.left.low)?.min(self.lowpt_of(p.right.low)?))
    }

    /// `lr_planarity` up to the embedding: `Ok(false)` if not planar.
    fn test(&mut self) -> Result<bool, Bail> {
        let n = self.n;
        let m = self.src.len();
        if n > 2 && m > 3 * n - 6 {
            return Ok(false);
        }
        for v in 0..n {
            if self.height[v] == NONE {
                self.height[v] = 0;
                self.roots.push(v as u32);
                self.orientation(v as u32);
            }
        }
        self.ordered = self.sorted_out();
        for i in 0..self.roots.len() {
            if !self.testing(self.roots[i])? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Out-edges of each node in `DG` order, stably sorted by nesting depth.
    fn sorted_out(&self) -> Vec<Vec<u32>> {
        self.out
            .iter()
            .map(|row| {
                let mut row = row.clone();
                row.sort_by_key(|&e| self.nesting[e as usize]);
                row
            })
            .collect()
    }

    fn orientation(&mut self, root: u32) {
        let adj = self.adj;
        let mut dfs = vec![root];
        let mut ind = vec![0usize; self.n];
        // `skip_init` is keyed by the directed pair (v, w): a tree edge's
        // flag only applies when scanning from its tail.
        let mut skip_init = vec![false; self.src.len()];
        while let Some(v) = dfs.pop() {
            let vi = v as usize;
            let e = self.parent_edge[vi];
            let row = &adj[vi];
            let start = ind[vi];
            for i in start..row.len() {
                let w = row[i];
                let id = self.ids[vi][i] as usize;
                if !(skip_init[id] && self.src[id] == v) {
                    if self.oriented[id] {
                        ind[vi] += 1;
                        continue;
                    }
                    self.oriented[id] = true;
                    self.src[id] = v;
                    self.dst[id] = w;
                    self.out[vi].push(id as u32);
                    self.lowpt[id] = self.height[vi];
                    self.lowpt2[id] = self.height[vi];
                    let wi = w as usize;
                    if self.height[wi] == NONE {
                        self.parent_edge[wi] = id as u32;
                        self.height[wi] = self.height[vi] + 1;
                        self.depth = self.depth.max(self.height[wi] as usize);
                        dfs.push(v);
                        dfs.push(w);
                        skip_init[id] = true;
                        break;
                    }
                    self.lowpt[id] = self.height[wi];
                }
                self.nesting[id] = 2 * self.lowpt[id] as i64;
                if self.lowpt2[id] < self.height[vi] {
                    self.nesting[id] += 1;
                }
                if e != NONE {
                    let e = e as usize;
                    if self.lowpt[id] < self.lowpt[e] {
                        self.lowpt2[e] = self.lowpt[e].min(self.lowpt2[id]);
                        self.lowpt[e] = self.lowpt[id];
                    } else if self.lowpt[id] > self.lowpt[e] {
                        self.lowpt2[e] = self.lowpt2[e].min(self.lowpt[id]);
                    } else {
                        self.lowpt2[e] = self.lowpt2[e].min(self.lowpt2[id]);
                    }
                }
                ind[vi] += 1;
            }
        }
    }

    fn testing(&mut self, root: u32) -> Result<bool, Bail> {
        let mut dfs = vec![root];
        let mut ind = vec![0usize; self.n];
        let mut skip_init = vec![false; self.src.len()];
        while let Some(v) = dfs.pop() {
            let vi = v as usize;
            let e = self.parent_edge[vi];
            let mut skip_final = false;
            let start = ind[vi];
            for i in start..self.ordered[vi].len() {
                let ei = self.ordered[vi][i];
                let id = ei as usize;
                let w = self.dst[id];
                if !skip_init[id] {
                    self.stack_bottom[id] = self.top_id();
                    if ei == self.parent_edge[w as usize] {
                        dfs.push(v);
                        dfs.push(w);
                        skip_init[id] = true;
                        skip_final = true;
                        break;
                    }
                    self.lowpt_edge[id] = ei;
                    let pair = self.new_pair(Interval::SHARED, Interval::new(ei, ei));
                    self.stack.push(pair);
                }
                if self.lowpt[id] < self.height[vi] {
                    if i == 0 {
                        if e == NONE {
                            return Err(Bail);
                        }
                        self.lowpt_edge[e as usize] = self.lowpt_edge[id];
                    } else if !self.add_constraints(ei, e)? {
                        return Ok(false);
                    }
                }
                ind[vi] += 1;
            }
            if !skip_final && e != NONE {
                self.remove_back_edges(e)?;
            }
        }
        Ok(true)
    }

    fn add_constraints(&mut self, ei: u32, e: u32) -> Result<bool, Bail> {
        let mut p = self.new_pair(Interval::SHARED, Interval::SHARED);
        loop {
            let mut q = self.stack.pop().ok_or(Bail)?;
            if !q.left.empty() {
                q.swap();
            }
            if !q.left.empty() {
                return Ok(false);
            }
            if self.lowpt_of(q.right.low)? > self.lowpt_of(e)? {
                if p.right.empty() {
                    p.right = q.right.copy();
                } else {
                    self.set_ref(p.right.low, q.right.high);
                }
                if p.right.shared {
                    return Err(Bail);
                }
                p.right.low = q.right.low;
            } else {
                let target = self.lowpt_edge[e as usize];
                if target == NONE {
                    return Err(Bail);
                }
                self.set_ref(q.right.low, target);
            }
            if self.top_id() == self.stack_bottom[ei as usize] {
                break;
            }
        }
        loop {
            let top = self.stack.last().ok_or(Bail)?;
            if !(self.conflicting(&top.left, ei)? || self.conflicting(&top.right, ei)?) {
                break;
            }
            let mut q = self.stack.pop().ok_or(Bail)?;
            if self.conflicting(&q.right, ei)? {
                q.swap();
            }
            if self.conflicting(&q.right, ei)? {
                return Ok(false);
            }
            self.set_ref(p.right.low, q.right.high);
            if q.right.low != NONE {
                if p.right.shared {
                    return Err(Bail);
                }
                p.right.low = q.right.low;
            }
            if p.left.empty() {
                p.left = q.left.copy();
            } else {
                self.set_ref(p.left.low, q.left.high);
            }
            if p.left.shared {
                return Err(Bail);
            }
            p.left.low = q.left.low;
        }
        if !(p.left.empty() && p.right.empty()) {
            self.stack.push(p);
        }
        Ok(true)
    }

    fn remove_back_edges(&mut self, e: u32) -> Result<(), Bail> {
        let u = self.src[e as usize];
        let hu = self.height[u as usize];
        while let Some(top) = self.stack.last() {
            if self.lowest(top)? != hu {
                break;
            }
            let p = self.stack.pop().unwrap();
            if p.left.low != NONE {
                self.side[p.left.low as usize] = -1;
            }
        }
        if let Some(mut p) = self.stack.pop() {
            while p.left.high != NONE && self.dst[p.left.high as usize] == u {
                p.left.high = self.refs[p.left.high as usize];
            }
            if p.left.high == NONE && p.left.low != NONE {
                self.set_ref(p.left.low, p.right.low);
                self.side[p.left.low as usize] = -1;
                p.left.low = NONE;
            }
            while p.right.high != NONE && self.dst[p.right.high as usize] == u {
                p.right.high = self.refs[p.right.high as usize];
            }
            if p.right.high == NONE && p.right.low != NONE {
                self.set_ref(p.right.low, p.left.low);
                self.side[p.right.low as usize] = -1;
                p.right.low = NONE;
            }
            self.stack.push(p);
        }
        if self.lowpt[e as usize] < hu {
            let top = self.stack.last().ok_or(Bail)?;
            let (hl, hr) = (top.left.high, top.right.high);
            let value = if hl != NONE && (hr == NONE || self.lowpt[hl as usize] > self.lowpt[hr as usize]) {
                hl
            } else {
                hr
            };
            self.refs[e as usize] = value;
        }
        Ok(())
    }

    /// NetworkX's iterative `sign`. `old_ref` is local to each call.
    fn sign(&mut self, e: u32, old_ref: &mut [u32], touched: &mut Vec<u32>) -> i64 {
        let mut dfs = vec![e];
        let mut last = e;
        while let Some(x) = dfs.pop() {
            last = x;
            let xi = x as usize;
            let r = self.refs[xi];
            if r != NONE {
                dfs.push(x);
                dfs.push(r);
                self.depth = self.depth.max(dfs.len() / 2 + 1);
                old_ref[xi] = r;
                touched.push(x);
                self.refs[xi] = NONE;
            } else {
                self.side[xi] *= self.side_of(old_ref[xi]);
            }
        }
        for x in touched.drain(..) {
            old_ref[x as usize] = NONE;
        }
        self.side[last as usize]
    }

    /// The rest of `lr_planarity` for a planar graph: signs, the final
    /// nesting order and the half-edge calls.
    fn embed(&mut self) -> Result<Embedding, Bail> {
        let mut old_ref = vec![NONE; self.src.len()];
        let mut touched = Vec::new();
        for v in 0..self.n {
            for k in 0..self.out[v].len() {
                let e = self.out[v][k];
                let s = self.sign(e, &mut old_ref, &mut touched);
                self.nesting[e as usize] *= s;
            }
        }
        self.ordered = self.sorted_out();
        let ordered: Vec<Vec<u32>> = self
            .ordered
            .iter()
            .map(|row| row.iter().map(|&e| self.dst[e as usize]).collect())
            .collect();
        let mut calls = Vec::with_capacity(self.src.len());
        let mut left_ref = vec![NONE; self.n];
        let mut right_ref = vec![NONE; self.n];
        let mut ind = vec![0usize; self.n];
        for r in 0..self.roots.len() {
            let mut dfs = vec![self.roots[r]];
            while let Some(v) = dfs.pop() {
                let vi = v as usize;
                while ind[vi] < self.ordered[vi].len() {
                    let ei = self.ordered[vi][ind[vi]];
                    ind[vi] += 1;
                    let w = self.dst[ei as usize];
                    let wi = w as usize;
                    if ei == self.parent_edge[wi] {
                        calls.push(HalfEdge::First(w, v));
                        left_ref[vi] = w;
                        right_ref[vi] = w;
                        dfs.push(v);
                        dfs.push(w);
                        break;
                    }
                    if self.side[ei as usize] == 1 {
                        // `right_ref` and `left_ref` are plain dicts.
                        if right_ref[wi] == NONE {
                            return Err(Bail);
                        }
                        calls.push(HalfEdge::Ccw(w, v, right_ref[wi]));
                    } else {
                        if left_ref[wi] == NONE {
                            return Err(Bail);
                        }
                        calls.push(HalfEdge::Cw(w, v, left_ref[wi]));
                        left_ref[wi] = v;
                    }
                }
            }
        }
        Ok(Embedding { ordered, calls })
    }
}

/// The `PlanarEmbedding` the half-edge calls build, as its dicts: for each
/// node, its out-half-edges in `_succ` order as `(target, cw, ccw,
/// ccw_first)` (whether the data dict's first key is "ccw"), and for each
/// node its in-half-edges in `_pred` order, as indices into the flattened
/// out-half-edges.
pub struct Layout {
    pub offsets: Vec<usize>,
    pub target: Vec<u32>,
    pub cw: Vec<u32>,
    pub ccw: Vec<u32>,
    pub ccw_first: Vec<bool>,
    pub pred_offsets: Vec<usize>,
    pub pred: Vec<u32>,
}

struct HalfEdgeState {
    start: u32,
    end: u32,
    cw: u32,
    ccw: u32,
    ccw_first: bool,
    stamp: u64,
}

/// Replays `PlanarEmbedding.add_half_edge` (NetworkX 3.4 to 3.7): a node's
/// last `_succ` key is its leftmost neighbor, kept there by moving it back
/// to the end after each insertion. Calls NetworkX would reject bail out.
struct EmbeddingBuilder {
    edges: Vec<HalfEdgeState>,
    index: PairMapDirected,
    last: Vec<u32>,
    pred: Vec<Vec<u32>>,
    clock: u64,
}

type PairMapDirected = HashMap<u64, u32, BuildHasherDefault<PairHasher>>;

#[inline]
fn directed_key(u: u32, v: u32) -> u64 {
    ((u as u64) << 32) | v as u64
}

impl EmbeddingBuilder {
    fn new(n: usize) -> Self {
        EmbeddingBuilder {
            edges: Vec::new(),
            index: PairMapDirected::default(),
            last: vec![NONE; n],
            pred: vec![Vec::new(); n],
            clock: 0,
        }
    }

    fn find(&self, start: u32, end: u32) -> Result<usize, Bail> {
        self.index.get(&directed_key(start, end)).map(|&h| h as usize).ok_or(Bail)
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn create(&mut self, start: u32, end: u32, cw: u32, ccw: u32, ccw_first: bool) -> Result<(), Bail> {
        let h = self.edges.len() as u32;
        if self.index.insert(directed_key(start, end), h).is_some() {
            return Err(Bail); // NetworkX would update the existing half-edge
        }
        let stamp = self.tick();
        self.edges.push(HalfEdgeState {
            start,
            end,
            cw,
            ccw,
            ccw_first,
            stamp,
        });
        self.pred[end as usize].push(h);
        self.last[start as usize] = end;
        Ok(())
    }

    fn add(&mut self, start: u32, end: u32, cw: u32, ccw: u32) -> Result<(), Bail> {
        let leftmost = self.last[start as usize];
        if leftmost == NONE {
            if cw != NONE || ccw != NONE {
                return Err(Bail);
            }
            return self.create(start, end, end, end, true);
        }
        let move_leftmost = if cw != NONE {
            if ccw != NONE {
                return Err(Bail);
            }
            let reference = self.find(start, cw)?;
            let ref_ccw = self.edges[reference].ccw;
            self.create(start, end, cw, ref_ccw, false)?;
            let other = self.find(start, ref_ccw)?;
            self.edges[other].cw = end;
            self.edges[reference].ccw = end;
            cw != leftmost
        } else if ccw != NONE {
            let reference = self.find(start, ccw)?;
            let ref_cw = self.edges[reference].cw;
            self.create(start, end, ref_cw, ccw, false)?;
            let other = self.find(start, ref_cw)?;
            self.edges[other].ccw = end;
            self.edges[reference].cw = end;
            true
        } else {
            return Err(Bail);
        };
        if move_leftmost {
            let h = self.find(start, leftmost)?;
            self.edges[h].stamp = self.tick();
            self.last[start as usize] = leftmost;
        }
        Ok(())
    }

    fn add_first(&mut self, start: u32, end: u32) -> Result<(), Bail> {
        let leftmost = self.last[start as usize];
        self.add(start, end, leftmost, NONE)
    }

    fn finish(self, n: usize) -> Layout {
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (h, e) in self.edges.iter().enumerate() {
            rows[e.start as usize].push(h as u32);
        }
        let mut position = vec![0u32; self.edges.len()];
        let mut layout = Layout {
            offsets: vec![0],
            target: Vec::with_capacity(self.edges.len()),
            cw: Vec::with_capacity(self.edges.len()),
            ccw: Vec::with_capacity(self.edges.len()),
            ccw_first: Vec::with_capacity(self.edges.len()),
            pred_offsets: vec![0],
            pred: Vec::with_capacity(self.edges.len()),
        };
        for row in rows.iter_mut() {
            row.sort_unstable_by_key(|&h| self.edges[h as usize].stamp);
            for &h in row.iter() {
                let e = &self.edges[h as usize];
                position[h as usize] = layout.target.len() as u32;
                layout.target.push(e.end);
                layout.cw.push(e.cw);
                layout.ccw.push(e.ccw);
                layout.ccw_first.push(e.ccw_first);
            }
            layout.offsets.push(layout.target.len());
        }
        for row in &self.pred {
            layout.pred.extend(row.iter().map(|&h| position[h as usize]));
            layout.pred_offsets.push(layout.pred.len());
        }
        layout
    }
}

/// The dicts of the `PlanarEmbedding` that `embedding`'s calls build.
pub fn embedding_layout(n: usize, embedding: &Embedding) -> Result<Layout, Bail> {
    let mut builder = EmbeddingBuilder::new(n);
    for (v, row) in embedding.ordered.iter().enumerate() {
        let mut previous = NONE;
        for &w in row {
            builder.add(v as u32, w, NONE, previous)?;
            previous = w;
        }
    }
    for call in &embedding.calls {
        match *call {
            HalfEdge::First(a, b) => builder.add_first(a, b)?,
            HalfEdge::Ccw(a, b, r) => builder.add(a, b, NONE, r)?,
            HalfEdge::Cw(a, b, r) => builder.add(a, b, r, NONE)?,
        }
    }
    Ok(builder.finish(n))
}

/// NetworkX's left-right planarity test on a simple undirected adjacency
/// (`planarity_graph`), with the embedding if asked for and planar.
pub fn lr_planarity(adj: &[Vec<u32>], embed: bool) -> Result<Planarity, Bail> {
    let mut lr = Lr::new(adj);
    let planar = lr.test()?;
    let embedding = if planar && embed {
        Some(lr.embed()?)
    } else {
        None
    };
    Ok(Planarity {
        planar,
        embedding,
        depth: lr.depth,
    })
}

/// NetworkX's `get_counterexample`: on a copy of the simple graph `adj`
/// (`nx.Graph(G)` order), try removing each edge, putting back those
/// without which the graph is planar. Returns those edges in order (with
/// repeats, as NetworkX re-adds them), `None` if the graph is planar, and
/// the deepest recursion any test would need.
#[allow(clippy::type_complexity)]
pub fn counterexample(adj: &[Vec<u32>]) -> Result<(Option<Vec<(u32, u32)>>, usize), Bail> {
    let first = lr_planarity(adj, false)?;
    let mut depth = first.depth;
    if first.planar {
        return Ok((None, depth));
    }
    let mut g: Vec<Vec<u32>> = adj.to_vec();
    let mut kept = Vec::new();
    for u in 0..g.len() {
        let nbrs = g[u].clone();
        for v in nbrs {
            let vi = v as usize;
            remove(&mut g[u], v);
            remove(&mut g[vi], u as u32);
            // The removed edge's graph is tested in full, as NetworkX does
            // (check_planarity builds the embedding when it is planar).
            let result = lr_planarity(&g, true)?;
            depth = depth.max(result.depth);
            if result.planar {
                g[u].push(v);
                g[vi].push(u as u32);
                kept.push((u as u32, v));
            }
        }
    }
    Ok((Some(kept), depth))
}

fn remove(row: &mut Vec<u32>, v: u32) {
    if let Some(i) = row.iter().position(|&x| x == v) {
        row.remove(i);
    }
}

// --- Chordal graphs ---------------------------------------------------------

/// Maximum cardinality search: nodes in visiting order, and how many
/// visited neighbors each had when visited. Ties are broken arbitrarily:
/// callers only use what is the same for every MCS order.
fn mcs(adj: &Csr, n: usize) -> (Vec<u32>, Vec<u32>) {
    let mut weight = vec![0u32; n];
    let mut visited = vec![false; n];
    let mut buckets: Vec<Vec<u32>> = vec![(0..n as u32).rev().collect()];
    let mut order = Vec::with_capacity(n);
    let mut at_visit = vec![0u32; n];
    let mut top = 0usize;
    while order.len() < n {
        let v = loop {
            match buckets[top].pop() {
                Some(v) if !visited[v as usize] && weight[v as usize] as usize == top => break v,
                Some(_) => {}
                None => top -= 1,
            }
        };
        let vi = v as usize;
        visited[vi] = true;
        at_visit[vi] = weight[vi];
        order.push(v);
        for &w in adj.neighbors(vi) {
            let wi = w as usize;
            if !visited[wi] {
                weight[wi] += 1;
                let k = weight[wi] as usize;
                if k == buckets.len() {
                    buckets.push(Vec::new());
                }
                buckets[k].push(w);
                top = top.max(k);
            }
        }
    }
    (order, at_visit)
}

/// Whether a graph without self-loops is chordal, and (if so) its largest
/// clique size minus one, via MCS and Tarjan and Yannakakis's zero fill-in
/// test. NetworkX's MCS breaks ties by set order, but every MCS order of a
/// chordal graph is a perfect elimination order, and none is otherwise.
pub fn chordal(adj: &Csr, n: usize) -> (bool, u32) {
    let (order, at_visit) = mcs(adj, n);
    // Elimination order: the reverse of the visiting order.
    let mut pos = vec![0usize; n];
    for (i, &v) in order.iter().rev().enumerate() {
        pos[v as usize] = i;
    }
    let mut follower = vec![0u32; n];
    let mut index = vec![0usize; n];
    for (i, &w) in order.iter().rev().enumerate() {
        let wi = w as usize;
        follower[wi] = w;
        index[wi] = i;
        for &v in adj.neighbors(wi) {
            let vi = v as usize;
            if pos[vi] < i {
                index[vi] = i;
                if follower[vi] == v {
                    follower[vi] = w;
                }
            }
        }
        for &v in adj.neighbors(wi) {
            let vi = v as usize;
            if pos[vi] < i && index[follower[vi] as usize] < i {
                return (false, 0);
            }
        }
    }
    (true, at_visit.iter().copied().max().unwrap_or(0))
}

/// `complete_to_chordal_graph` on a non-chordal graph without self-loops:
/// each node's `alpha`, and the chords `(z, y)` in the order NetworkX adds
/// them to its `chords` set. NetworkX tests each candidate with a path
/// search through lighter unnumbered nodes; one bottleneck search from z
/// per step answers all of them: y gets a chord when some path from z
/// reaches y through unnumbered nodes all lighter than y.
pub fn complete_to_chordal(adj: &Csr, n: usize) -> (Vec<u32>, Vec<(u32, u32)>) {
    let mut alpha = vec![0u32; n];
    let mut weight = vec![0u32; n];
    let mut numbered = vec![false; n];
    let mut chords = Vec::new();
    let mut adjacent = vec![NONE; n];
    // Smallest possible "heaviest interior node" over paths from z (-1
    // for z's own neighbors), stored plus one so -1 becomes 0.
    let mut best = vec![u32::MAX; n];
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); n + 2];
    let mut updates = Vec::new();
    for i in (1..=n as u32).rev() {
        let mut z = NONE;
        for v in 0..n {
            if !numbered[v] && (z == NONE || weight[v] > weight[z as usize]) {
                z = v as u32;
            }
        }
        let zi = z as usize;
        numbered[zi] = true;
        alpha[zi] = i;
        for &y in adj.neighbors(zi) {
            adjacent[y as usize] = z;
        }
        // Bottleneck search from z over unnumbered nodes.
        best.fill(u32::MAX);
        for &y in adj.neighbors(zi) {
            let yi = y as usize;
            if !numbered[yi] && best[yi] != 0 {
                best[yi] = 0;
                buckets[0].push(y);
            }
        }
        for level in 0..buckets.len() {
            while let Some(x) = buckets[level].pop() {
                let xi = x as usize;
                if best[xi] as usize != level {
                    continue;
                }
                let through = best[xi].max(weight[xi] + 1);
                for &y in adj.neighbors(xi) {
                    let yi = y as usize;
                    if !numbered[yi] && through < best[yi] {
                        best[yi] = through;
                        buckets[through as usize].push(y);
                    }
                }
            }
        }
        updates.clear();
        for y in 0..n {
            if numbered[y] {
                continue;
            }
            if adjacent[y] == z {
                updates.push(y);
            } else if best[y] != u32::MAX && best[y] <= weight[y] {
                // best - 1 < weight
                updates.push(y);
                chords.push((z, y as u32));
            }
        }
        for &y in &updates {
            weight[y] += 1;
        }
    }
    (alpha, chords)
}

// --- AT-free graphs -----------------------------------------------------------

/// Whether a graph has no asteroidal triple, as `find_asteroidal_triple`
/// decides it: labels of the components of `G - N[v]` for every v, then
/// every triple of pairwise non-adjacent nodes. `None` if the label table
/// doesn't fit in memory.
pub fn is_at_free(adj: &Csr, n: usize) -> Option<bool> {
    if n < 6 {
        return Some(true);
    }
    let mut label: Vec<u32> = Vec::new();
    label.try_reserve_exact(n.checked_mul(n)?).ok()?;
    label.resize(n * n, 0);
    let mut queue = Vec::with_capacity(n);
    for v in 0..n {
        let row = &mut label[v * n..(v + 1) * n];
        // 0 marks N[v]; labels start at 1.
        row.fill(NONE);
        row[v] = 0;
        for &w in adj.neighbors(v) {
            row[w as usize] = 0;
        }
        let mut next = 0u32;
        for s in 0..n {
            if row[s] != NONE {
                continue;
            }
            next += 1;
            row[s] = next;
            queue.clear();
            queue.push(s as u32);
            while let Some(x) = queue.pop() {
                for &y in adj.neighbors(x as usize) {
                    if row[y as usize] == NONE {
                        row[y as usize] = next;
                        queue.push(y);
                    }
                }
            }
        }
    }
    for u in 0..n {
        let ru = &label[u * n..(u + 1) * n];
        for v in u + 1..n {
            let cuv = ru[v];
            if cuv == 0 {
                continue; // adjacent
            }
            let rv = &label[v * n..(v + 1) * n];
            let cvu = rv[u];
            for w in 0..n {
                if ru[w] == cuv && rv[w] == cvu {
                    let rw = &label[w * n..(w + 1) * n];
                    if rw[u] == rw[v] && rw[u] != 0 {
                        return Some(false);
                    }
                }
            }
        }
    }
    Some(true)
}

// --- Tournaments ------------------------------------------------------------------

/// Marks the tournament module's 2-neighborhood of v, `{v} | succ(v) |
/// succ(succ(v))`, with `tag`; returns its size.
fn two_neighborhood(succ: &Csr, v: usize, mark: &mut [u32], tag: u32) -> usize {
    let mut size = 0;
    let mut add = |x: u32| {
        if mark[x as usize] != tag {
            mark[x as usize] = tag;
            size += 1;
        }
    };
    add(v as u32);
    for &z in succ.neighbors(v) {
        add(z);
    }
    for &z in succ.neighbors(v) {
        for &x in succ.neighbors(z as usize) {
            add(x);
        }
    }
    size
}

/// Whether every node outside the marked set has an arc to all of it.
fn is_closed(succ: &Csr, n: usize, mark: &[u32], tag: u32, size: usize) -> bool {
    (0..n).all(|u| {
        if mark[u] == tag {
            return true;
        }
        let row = succ.neighbors(u);
        row.len() >= size && row.iter().filter(|&&x| mark[x as usize] == tag).count() == size
    })
}

/// `nx.tournament.is_reachable(G, s, t)` for positions (`NONE` for a node
/// not in G): no closed 2-neighborhood holds s but not t.
pub fn tournament_reachable(succ: &Csr, n: usize, s: u32, t: u32) -> bool {
    let mut mark = vec![NONE; n];
    !(0..n).any(|v| {
        let tag = v as u32;
        let size = two_neighborhood(succ, v, &mut mark, tag);
        let has = |x: u32| x != NONE && mark[x as usize] == tag;
        has(s) && !has(t) && is_closed(succ, n, &mark, tag, size)
    })
}

/// `nx.tournament.is_strongly_connected`: `is_reachable` for every pair,
/// that is, no closed 2-neighborhood misses a node.
pub fn tournament_strongly_connected(succ: &Csr, n: usize) -> bool {
    let mut mark = vec![NONE; n];
    !(0..n).any(|v| {
        let tag = v as u32;
        let size = two_neighborhood(succ, v, &mut mark, tag);
        size < n && is_closed(succ, n, &mark, tag, size)
    })
}

// --- Perfect graphs ------------------------------------------------------------

/// Adjacency for the chordless cycle search: G without its self-loop nodes,
/// or G's complement.
trait Adjacency {
    fn n(&self) -> usize;
    /// Neighbors of v among the nodes `alive` marks.
    fn neighbors(&self, v: u32, alive: &[bool], out: &mut Vec<u32>);
    fn adjacent(&self, u: u32, v: u32) -> bool;
}

/// G's simple graph (`chordless_cycles` drops nodes with self-loops),
/// with sorted rows for membership tests.
struct Sparse {
    rows: Vec<Vec<u32>>,
}

impl Adjacency for Sparse {
    fn n(&self) -> usize {
        self.rows.len()
    }

    fn neighbors(&self, v: u32, alive: &[bool], out: &mut Vec<u32>) {
        out.clear();
        out.extend(self.rows[v as usize].iter().copied().filter(|&w| alive[w as usize]));
    }

    fn adjacent(&self, u: u32, v: u32) -> bool {
        self.rows[u as usize].binary_search(&v).is_ok()
    }
}

/// G's complement as bit rows.
struct Dense {
    n: usize,
    words: usize,
    bits: Vec<u64>,
}

impl Adjacency for Dense {
    fn n(&self) -> usize {
        self.n
    }

    fn neighbors(&self, v: u32, alive: &[bool], out: &mut Vec<u32>) {
        out.clear();
        let row = &self.bits[v as usize * self.words..(v as usize + 1) * self.words];
        for (k, &word) in row.iter().enumerate() {
            let mut word = word;
            while word != 0 {
                let w = (k * 64 + word.trailing_zeros() as usize) as u32;
                word &= word - 1;
                if alive[w as usize] {
                    out.push(w);
                }
            }
        }
    }

    fn adjacent(&self, u: u32, v: u32) -> bool {
        let v = v as usize;
        self.bits[u as usize * self.words + v / 64] >> (v % 64) & 1 == 1
    }
}

/// Node sets of the biconnected components (more than 2 nodes) of the
/// subgraph on `alive`, by Tarjan's algorithm.
fn biconnected_sets<A: Adjacency>(g: &A, alive: &[bool], nodes: &[u32]) -> Vec<Vec<u32>> {
    let n = g.n();
    let mut disc = vec![0u32; n];
    let mut low = vec![0u32; n];
    let mut clock = 0u32;
    let mut sets = Vec::new();
    let mut edge_stack: Vec<(u32, u32)> = Vec::new();
    let mut in_set = vec![false; n];
    // DFS frames: (node, parent, neighbors, next index)
    let mut frames: Vec<(u32, u32, Vec<u32>, usize)> = Vec::new();
    for &root in nodes {
        if disc[root as usize] != 0 {
            continue;
        }
        clock += 1;
        disc[root as usize] = clock;
        low[root as usize] = clock;
        let mut nbrs = Vec::new();
        g.neighbors(root, alive, &mut nbrs);
        frames.push((root, NONE, nbrs, 0));
        while let Some(frame) = frames.last_mut() {
            let (v, parent) = (frame.0, frame.1);
            if frame.3 < frame.2.len() {
                let w = frame.2[frame.3];
                frame.3 += 1;
                let wi = w as usize;
                if disc[wi] == 0 {
                    edge_stack.push((v, w));
                    clock += 1;
                    disc[wi] = clock;
                    low[wi] = clock;
                    let mut nbrs = Vec::new();
                    g.neighbors(w, alive, &mut nbrs);
                    frames.push((w, v, nbrs, 0));
                } else if w != parent && disc[wi] < disc[v as usize] {
                    edge_stack.push((v, w));
                    low[v as usize] = low[v as usize].min(disc[wi]);
                }
                continue;
            }
            frames.pop();
            if parent == NONE {
                continue;
            }
            let (p, vi) = (parent as usize, v as usize);
            low[p] = low[p].min(low[vi]);
            if low[vi] >= disc[p] {
                // Pop the component's edges.
                let mut set = Vec::new();
                while let Some((a, b)) = edge_stack.pop() {
                    for x in [a, b] {
                        if !in_set[x as usize] {
                            in_set[x as usize] = true;
                            set.push(x);
                        }
                    }
                    if (a, b) == (parent, v) {
                        break;
                    }
                }
                for &x in &set {
                    in_set[x as usize] = false;
                }
                if set.len() > 2 {
                    sets.push(set);
                }
            }
        }
    }
    sets
}

/// Whether `chordless_cycles` would yield a cycle of odd length 5 or more
/// (a hole), searching as it does: in each biconnected component, from
/// every stem (u, v, w) at one node v, then without v.
fn has_odd_hole<A: Adjacency>(g: &A, present: &[bool]) -> bool {
    let n = g.n();
    let nodes: Vec<u32> = (0..n as u32).filter(|&v| present[v as usize]).collect();
    let mut alive = present.to_vec();
    let mut components = biconnected_sets(g, &alive, &nodes);
    alive.fill(false);
    let mut blocked = vec![0i32; n];
    let mut nv = Vec::new();
    while let Some(c) = components.pop() {
        for &x in &c {
            alive[x as usize] = true;
        }
        let v = c[0];
        g.neighbors(v, &alive, &mut nv);
        for i in 0..nv.len() {
            for j in i + 1..nv.len() {
                let (u, w) = (nv[i], nv[j]);
                if !g.adjacent(w, u) && odd_hole_from(g, &alive, &mut blocked, [u, v, w]) {
                    return true;
                }
            }
        }
        alive[v as usize] = false;
        let rest: Vec<u32> = c[1..].to_vec();
        components.extend(biconnected_sets(g, &alive, &rest));
        for &x in &c {
            alive[x as usize] = false;
        }
    }
    false
}

/// `_chordless_cycle_search` from a stem, stopping at an odd hole.
fn odd_hole_from<A: Adjacency>(g: &A, alive: &[bool], blocked: &mut [i32], stem: [u32; 3]) -> bool {
    let [target, v, w0] = stem;
    let mut nbrs = Vec::new();
    let mut lists: Vec<Vec<u32>> = Vec::new();
    let bump = |x: u32, delta: i32, blocked: &mut [i32], nbrs: &mut Vec<u32>| {
        g.neighbors(x, alive, nbrs);
        for &y in nbrs.iter() {
            blocked[y as usize] += delta;
        }
    };
    blocked[v as usize] = 1;
    bump(v, 1, blocked, &mut nbrs);
    bump(w0, 1, blocked, &mut nbrs);
    let mut path = vec![target, v, w0];
    let mut first = Vec::new();
    g.neighbors(w0, alive, &mut first);
    lists.push(first);
    let mut next = vec![0usize];
    let mut found = false;
    'outer: while let Some(top) = lists.last() {
        let k = lists.len() - 1;
        while next[k] < top.len() {
            let x = top[next[k]];
            next[k] += 1;
            if blocked[x as usize] != 1 {
                continue;
            }
            if g.adjacent(x, target) {
                let len = path.len() + 1;
                if len >= 5 && len % 2 == 1 {
                    found = true;
                    break 'outer;
                }
                continue;
            }
            bump(x, 1, blocked, &mut nbrs);
            path.push(x);
            let mut row = Vec::new();
            g.neighbors(x, alive, &mut row);
            lists.push(row);
            next.push(0);
            continue 'outer;
        }
        lists.pop();
        next.pop();
        let x = path.pop().unwrap();
        bump(x, -1, blocked, &mut nbrs);
    }
    // Undo what is still counted, leaving `blocked` all zero.
    while path.len() > 2 {
        let x = path.pop().unwrap();
        bump(x, -1, blocked, &mut nbrs);
    }
    bump(v, -1, blocked, &mut nbrs);
    blocked[v as usize] -= 1;
    found
}

/// `nx.is_perfect_graph` on a simple undirected adjacency (self-loops
/// allowed): no odd hole in G (nodes with self-loops left out, as
/// `chordless_cycles` does) and none in its complement. `None` if the
/// complement doesn't fit in memory.
pub fn is_perfect(adj: &Csr, n: usize) -> Option<bool> {
    let looped: Vec<bool> = (0..n).map(|v| adj.neighbors(v).contains(&(v as u32))).collect();
    let rows: Vec<Vec<u32>> = (0..n)
        .map(|v| {
            let mut row: Vec<u32> = if looped[v] {
                Vec::new()
            } else {
                adj.neighbors(v).iter().copied().filter(|&w| !looped[w as usize]).collect()
            };
            row.sort_unstable();
            row.dedup();
            row
        })
        .collect();
    let present: Vec<bool> = looped.iter().map(|&l| !l).collect();
    if has_odd_hole(&Sparse { rows }, &present) {
        return Some(false);
    }
    let words = n.div_ceil(64);
    let mut bits: Vec<u64> = Vec::new();
    bits.try_reserve_exact(n.checked_mul(words)?).ok()?;
    bits.resize(n * words, 0);
    for v in 0..n {
        let row = &mut bits[v * words..(v + 1) * words];
        for (k, word) in row.iter_mut().enumerate() {
            *word = if (k + 1) * 64 <= n { !0 } else { (1u64 << (n - k * 64)) - 1 };
        }
        row[v / 64] &= !(1u64 << (v % 64));
        for &w in adj.neighbors(v) {
            let w = w as usize;
            row[w / 64] &= !(1u64 << (w % 64));
        }
    }
    Some(!has_odd_hole(&Dense { n, words, bits }, &vec![true; n]))
}
