//! Batch 8: branchings and arborescences (Edmonds' algorithm exactly as
//! NetworkX implements it), greedy branchings, Prim's and partitioned
//! Kruskal's spanning edges, Prüfer and nested-tuple trees and lowest
//! common ancestors.

use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap, HashMap};

use crate::graph::Csr;

const NONE: u32 = u32::MAX;

/// A Python int or float, with Python's arithmetic and comparisons (an int
/// compares exactly with a float; int arithmetic that leaves `i64` is
/// reported as `None`, so the caller can decline).
#[derive(Clone, Copy, Debug)]
pub enum Num {
    Int(i64),
    Float(f64),
}

impl Num {
    fn as_f64(self) -> f64 {
        match self {
            Num::Int(i) => i as f64, // round to nearest, like float(int)
            Num::Float(f) => f,
        }
    }

    pub fn add(self, other: Num) -> Option<Num> {
        match (self, other) {
            (Num::Int(a), Num::Int(b)) => a.checked_add(b).map(Num::Int),
            _ => Some(Num::Float(self.as_f64() + other.as_f64())),
        }
    }

    pub fn sub(self, other: Num) -> Option<Num> {
        match (self, other) {
            (Num::Int(a), Num::Int(b)) => a.checked_sub(b).map(Num::Int),
            _ => Some(Num::Float(self.as_f64() - other.as_f64())),
        }
    }

    pub fn cmp(self, other: Num) -> Option<Ordering> {
        match (self, other) {
            (Num::Int(a), Num::Int(b)) => Some(a.cmp(&b)),
            (Num::Float(a), Num::Float(b)) => a.partial_cmp(&b),
            (Num::Int(a), Num::Float(b)) => cmp_int_float(a, b),
            (Num::Float(a), Num::Int(b)) => cmp_int_float(b, a).map(Ordering::reverse),
        }
    }

    fn gt(self, other: Num) -> bool {
        self.cmp(other) == Some(Ordering::Greater)
    }

    fn lt(self, other: Num) -> bool {
        self.cmp(other) == Some(Ordering::Less)
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
        Ordering::Equal => (t).partial_cmp(&f),
        ord => Some(ord),
    }
}

// --- Edmonds' algorithm ------------------------------------------------------------

/// The parts of `maximum_branching`'s run that decide its output: the keys
/// of the final branching `B` in `B_edge_index` order, and for each
/// contraction from the last to the first, the circuit's keys and the key
/// removed when expanding it. Python replays these on a real `set` so that
/// the result's edge order is the set's iteration order, as in NetworkX.
pub struct Branching {
    pub initial: Vec<u32>,
    pub log: Vec<(Vec<u32>, u32)>,
}

struct Contraction {
    circuit: Vec<u32>,
    /// Each circuit edge's head.
    targets: Vec<u32>,
    minedge: u32,
    /// The merged node's in-edges: key and the circuit node it entered.
    in_edges: Vec<(u32, u32)>,
}

/// NetworkX's `MultiDiGraph` as `maximum_branching` uses it: adjacency rows
/// in dict insertion order, with each row's keys in insertion order.
struct Multi {
    succ: Vec<Vec<(u32, Vec<u32>)>>,
    pred: Vec<Vec<(u32, Vec<u32>)>>,
    alive: Vec<bool>,
    /// Position in node order (new nodes go last).
    stamp: Vec<usize>,
    order: Vec<u32>,
}

impl Multi {
    fn new(n: usize) -> Self {
        Multi {
            succ: vec![Vec::new(); n],
            pred: vec![Vec::new(); n],
            alive: vec![false; n],
            stamp: vec![usize::MAX; n],
            order: Vec::new(),
        }
    }

    /// `G.add_node(x)`: a node not seen before joins the node order.
    fn add_node(&mut self, x: u32) {
        let x = x as usize;
        if self.stamp[x] == usize::MAX {
            self.alive[x] = true;
            self.stamp[x] = self.order.len();
            self.order.push(x as u32);
        }
    }

    fn push_node(&mut self) -> u32 {
        let x = self.alive.len() as u32;
        self.succ.push(Vec::new());
        self.pred.push(Vec::new());
        self.alive.push(false);
        self.stamp.push(usize::MAX);
        self.add_node(x);
        x
    }

    /// Removes the nodes marked in `in_q`, as `G.remove_node` does for each.
    fn remove_nodes(&mut self, nodes: &[u32], in_q: &[bool]) {
        let mut touched = Vec::new();
        for &q in nodes {
            let q = q as usize;
            for (v, _) in std::mem::take(&mut self.succ[q]) {
                if !in_q[v as usize] {
                    touched.push((v, true));
                }
            }
            for (u, _) in std::mem::take(&mut self.pred[q]) {
                if !in_q[u as usize] {
                    touched.push((u, false));
                }
            }
            self.alive[q] = false;
        }
        touched.sort_unstable();
        touched.dedup();
        for (x, is_pred_row) in touched {
            let row = if is_pred_row {
                &mut self.pred[x as usize]
            } else {
                &mut self.succ[x as usize]
            };
            row.retain(|&(y, _)| !in_q[y as usize]);
        }
    }
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let p = parent[x as usize];
        parent[x as usize] = parent[p as usize];
        x = p;
    }
    x
}

/// NetworkX's `maximum_branching` on edges `us[k] -> vs[k]` (key `k`, in
/// `G.edges` order) with weights `w` and partition states `part` (0 open,
/// 1 included, 2 excluded). `None` where NetworkX's arithmetic leaves
/// `i64` or its run would raise.
pub fn edmonds(n: usize, us: &[u32], vs: &[u32], w: &[Num], part: &[u8]) -> Option<Branching> {
    let m = us.len();
    let mut g = Multi::new(n);
    // `edmonds_add_edge` for each edge: nodes join in order of appearance.
    let mut eu = us.to_vec();
    let mut ev = vs.to_vec();
    let mut ew = w.to_vec();
    let mut cand = vec![false; m];
    for k in 0..m {
        let (u, v) = (eu[k], ev[k]);
        g.add_node(u);
        g.add_node(v);
        // G is a simple graph, so each (u, v) is new.
        g.succ[u as usize].push((v, vec![k as u32]));
        g.pred[v as usize].push((u, vec![k as u32]));
    }
    let total = g.alive.len();
    let mut uf: Vec<u32> = (0..total as u32).collect();
    let mut selected = vec![false; total];
    let mut bpred = vec![NONE; total];
    let mut b_order: Vec<u32> = Vec::new();
    let mut b_pos = vec![usize::MAX; m];
    let mut levels: Vec<Contraction> = Vec::new();
    let mut in_q = vec![false; total];
    let mut qin: Vec<Option<Num>> = vec![None; total];

    let mut cursor = 0;
    loop {
        while cursor < g.order.len() {
            let x = g.order[cursor] as usize;
            if g.alive[x] && !selected[x] {
                break;
            }
            cursor += 1;
        }
        if cursor == g.order.len() {
            break;
        }
        let v = g.order[cursor];
        cursor += 1;
        selected[v as usize] = true;
        // edmonds_find_desired_edge
        let mut edge = NONE;
        let mut max_weight = Num::Float(f64::NEG_INFINITY);
        'scan: for (_, keys) in &g.pred[v as usize] {
            for &k in keys {
                let p = part[k as usize];
                if p == 2 {
                    continue;
                }
                let new_weight = ew[k as usize];
                if p == 1 {
                    max_weight = new_weight;
                    edge = k;
                    break 'scan;
                }
                if new_weight.gt(max_weight) {
                    max_weight = new_weight;
                    edge = k;
                }
            }
        }
        if edge == NONE || !max_weight.gt(Num::Int(0)) {
            continue;
        }
        let u = eu[edge as usize];
        let circuit = find(&mut uf, u) == find(&mut uf, v);
        b_pos[edge as usize] = b_order.len();
        b_order.push(edge);
        bpred[v as usize] = edge;
        cand[edge as usize] = true;
        let (ru, rv) = (find(&mut uf, u), find(&mut uf, v));
        if ru != rv {
            uf[ru as usize] = rv;
        }
        if !circuit {
            continue;
        }
        // edmonds_step_I2: the circuit is B's path from v to u, then u -> v.
        let mut q_nodes = vec![u];
        let mut x = u;
        while x != v {
            x = eu[bpred[x as usize] as usize];
            q_nodes.push(x);
        }
        q_nodes.reverse();
        let mut q_edges: Vec<u32> = q_nodes[1..].iter().map(|&y| bpred[y as usize]).collect();
        q_edges.push(edge);
        let mut minweight = Num::Float(f64::INFINITY);
        let mut minedge = NONE;
        let mut targets = Vec::with_capacity(q_edges.len());
        for &k in &q_edges {
            let wk = ew[k as usize];
            targets.push(ev[k as usize]);
            qin[ev[k as usize] as usize] = Some(wk);
            if part[k as usize] == 1 {
                continue;
            }
            if wk.lt(minweight) {
                minweight = wk;
                minedge = k;
            }
        }
        let new_node = g.push_node();
        uf.push(new_node);
        selected.push(false);
        bpred.push(NONE);
        in_q.push(false);
        qin.push(None);
        for &q in &q_nodes {
            in_q[q as usize] = true;
        }
        // New edges in `G.edges` order: rows of the circuit's nodes and of
        // their predecessors, by node order.
        let mut rows: Vec<u32> = q_nodes.clone();
        for &q in &q_nodes {
            rows.extend(g.pred[q as usize].iter().map(|&(p, _)| p));
        }
        rows.sort_unstable_by_key(|&r| g.stamp[r as usize]);
        rows.dedup();
        let mut new_edges: Vec<(u32, u32, u32)> = Vec::new();
        let mut in_edges = Vec::new();
        for &r in &rows {
            for (y, keys) in &g.succ[r as usize] {
                let y = *y;
                if in_q[r as usize] {
                    if in_q[y as usize] {
                        continue;
                    }
                    for &k in keys {
                        new_edges.push((new_node, y, k));
                    }
                } else if in_q[y as usize] {
                    let shift = minweight.sub(qin[y as usize]?)?;
                    for &k in keys {
                        ew[k as usize] = ew[k as usize].add(shift)?;
                        new_edges.push((r, new_node, k));
                        in_edges.push((k, y));
                    }
                }
            }
        }
        // Remove the circuit from G and B (B loses the edges at its nodes).
        for &q in &q_nodes {
            let bk = bpred[q as usize];
            if bk != NONE {
                b_pos[bk as usize] = usize::MAX;
            }
            for (_, keys) in &g.succ[q as usize] {
                for &k in keys {
                    if cand[k as usize] && b_pos[k as usize] != usize::MAX {
                        b_pos[k as usize] = usize::MAX;
                        bpred[ev[k as usize] as usize] = NONE;
                    }
                }
            }
            bpred[q as usize] = NONE;
            selected[q as usize] = false;
        }
        g.remove_nodes(&q_nodes, &in_q);
        // Add the new edges (and to B those that were candidates).
        let mut out_row: HashMap<u32, usize> = HashMap::new();
        for &(a, b, k) in &new_edges {
            eu[k as usize] = a;
            ev[k as usize] = b;
            if a == new_node {
                match out_row.get(&b) {
                    Some(&i) => {
                        g.succ[a as usize][i].1.push(k);
                        g.pred[b as usize].last_mut().unwrap().1.push(k);
                    }
                    None => {
                        out_row.insert(b, g.succ[a as usize].len());
                        g.succ[a as usize].push((b, vec![k]));
                        g.pred[b as usize].push((a, vec![k]));
                    }
                }
            } else {
                // Edges from one row are consecutive, and the merged node
                // is newest, so an existing entry is the last one.
                match g.succ[a as usize].last_mut() {
                    Some(last) if last.0 == b => {
                        last.1.push(k);
                        g.pred[b as usize].last_mut().unwrap().1.push(k);
                    }
                    _ => {
                        g.succ[a as usize].push((b, vec![k]));
                        g.pred[b as usize].push((a, vec![k]));
                    }
                }
            }
            if cand[k as usize] {
                b_pos[k as usize] = b_order.len();
                b_order.push(k);
                bpred[b as usize] = k;
                let (ra, rb) = (find(&mut uf, a), find(&mut uf, b));
                if ra != rb {
                    uf[ra as usize] = rb;
                }
            }
        }
        for &q in &q_nodes {
            in_q[q as usize] = false;
            qin[q as usize] = None;
        }
        levels.push(Contraction {
            circuit: q_edges,
            targets,
            minedge,
            in_edges,
        });
    }

    let initial: Vec<u32> = b_order
        .iter()
        .enumerate()
        .filter(|&(i, &k)| b_pos[k as usize] == i)
        .map(|(_, &k)| k)
        .collect();
    let mut in_set = vec![false; m];
    for &k in &initial {
        in_set[k as usize] = true;
    }
    let mut log = Vec::with_capacity(levels.len());
    for level in levels.into_iter().rev() {
        // is_root: an in-edge of the merged node already in the branching.
        let mut entering = level.in_edges.iter().filter(|&&(k, _)| in_set[k as usize]);
        let found = entering.next().copied();
        if entering.next().is_some() {
            return None; // more than one: never happens for a branching
        }
        for &k in &level.circuit {
            in_set[k as usize] = true;
        }
        let removed = match found {
            None => {
                if level.minedge == NONE {
                    return None; // NetworkX raises a bare `Exception`
                }
                level.minedge
            }
            Some((_, target)) => {
                let i = level.targets.iter().position(|&t| t == target)?;
                level.circuit[i]
            }
        };
        in_set[removed as usize] = false;
        log.push((level.circuit, removed));
    }
    Some(Branching { initial, log })
}

/// `greedy_branching`: edge indices in the order they join the branching.
/// Edges sort by `(weight, rank[u], rank[v])`, descending for `maximum`.
pub fn greedy_branching(
    n: usize,
    us: &[u32],
    vs: &[u32],
    w: &[Num],
    rank: &[u32],
    maximum: bool,
) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..us.len() as u32).collect();
    let key = |i: u32| {
        let i = i as usize;
        (w[i], rank[us[i] as usize], rank[vs[i] as usize])
    };
    idx.sort_by(|&a, &b| {
        let (wa, ua, va) = key(a);
        let (wb, ub, vb) = key(b);
        let ord = wa
            .cmp(wb)
            .unwrap_or(Ordering::Equal)
            .then(ua.cmp(&ub))
            .then(va.cmp(&vb));
        if maximum {
            ord.reverse()
        } else {
            ord
        }
    });
    let mut uf: Vec<u32> = (0..n as u32).collect();
    let mut has_in = vec![false; n];
    let mut kept = Vec::new();
    for i in idx {
        let (u, v) = (us[i as usize], vs[i as usize]);
        let (ru, rv) = (find(&mut uf, u), find(&mut uf, v));
        if ru == rv || has_in[v as usize] {
            continue;
        }
        has_in[v as usize] = true;
        // `UnionFind.union` merges the sets; which root wins doesn't matter.
        uf[ru as usize] = rv;
        kept.push(i);
    }
    kept
}

/// Kruskal's algorithm with a partition (`kruskal_mst_edges(partition=...)`):
/// included edges first in edge order, then open edges by weight (stable),
/// excluded edges never. States: 0 open, 1 included, 2 excluded.
pub fn kruskal_partition(
    n: usize,
    edges: &[(u32, u32, f64)],
    state: &[u8],
    maximum: bool,
) -> Vec<u32> {
    let mut order: Vec<u32> = (0..edges.len() as u32)
        .filter(|&i| state[i as usize] == 1)
        .collect();
    let mut open: Vec<u32> = (0..edges.len() as u32)
        .filter(|&i| state[i as usize] == 0)
        .collect();
    open.sort_by(|&a, &b| {
        let ord = edges[a as usize]
            .2
            .partial_cmp(&edges[b as usize].2)
            .expect("no NaN weights");
        if maximum {
            ord.reverse()
        } else {
            ord
        }
    });
    order.extend(open);
    let mut uf: Vec<u32> = (0..n as u32).collect();
    let mut kept = Vec::new();
    for i in order {
        if kept.len() + 1 >= n {
            break;
        }
        let (u, v, _) = edges[i as usize];
        let (ru, rv) = (find(&mut uf, u), find(&mut uf, v));
        if ru != rv {
            uf[ru as usize] = rv;
            kept.push(i);
        }
    }
    kept
}

// --- Prim --------------------------------------------------------------------------

struct Frontier {
    weight: f64,
    count: u64,
    u: u32,
    v: u32,
}

impl PartialEq for Frontier {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Frontier {}
impl PartialOrd for Frontier {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Frontier {
    // Reversed: `BinaryHeap` is a max-heap and `heapq` a min-heap. Python
    // compares the tuples `(weight, count, ...)`; 0.0 equals -0.0.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .weight
            .partial_cmp(&self.weight)
            .expect("no NaN weights")
            .then(other.count.cmp(&self.count))
    }
}

/// `prim_mst_edges` on an undirected graph, growing a tree from each of
/// `starts` in turn (the nodes NetworkX pops from its node set).
pub fn prim(adj: &Csr, w: Option<&[f64]>, starts: &[u32], sign: f64) -> Vec<(u32, u32)> {
    let n = adj.offsets.len() - 1;
    let mut visited = vec![false; n];
    let mut count = 0u64;
    let mut heap = BinaryHeap::new();
    let mut tree = Vec::new();
    let mut push = |heap: &mut BinaryHeap<Frontier>, visited: &[bool], u: u32| {
        for e in adj.range(u as usize) {
            let v = adj.targets[e];
            if visited[v as usize] {
                continue;
            }
            heap.push(Frontier {
                weight: w.map_or(1.0, |w| w[e]) * sign,
                count,
                u,
                v,
            });
            count += 1;
        }
    };
    for &s in starts {
        visited[s as usize] = true;
        heap.clear();
        push(&mut heap, &visited, s);
        while let Some(Frontier { u, v, .. }) = heap.pop() {
            if visited[v as usize] {
                continue;
            }
            tree.push((u, v));
            visited[v as usize] = true;
            push(&mut heap, &visited, v);
        }
    }
    tree
}

// --- Prüfer sequences and nested tuples ---------------------------------------------

/// `from_prufer_sequence`'s edges, in the order NetworkX adds them (before
/// the final edge joining the two orphans), or `Err(i)` if entry `i` is out
/// of range (`-1` marks those), or `Err(usize::MAX)` if NetworkX's search
/// for the next leaf would fail.
pub fn prufer_edges(seq: &[i64]) -> Result<Vec<(u32, u32)>, usize> {
    let n = seq.len() + 2;
    let mut degree = vec![1u32; n];
    for &v in seq {
        if v >= 0 {
            degree[v as usize] += 1;
        }
    }
    let next_leaf = |degree: &[u32], from: usize| (from..n).find(|&k| degree[k] == 1);
    let mut index = next_leaf(&degree, 0).ok_or(usize::MAX)?;
    let mut u = index;
    let mut edges = Vec::with_capacity(seq.len());
    for (i, &v) in seq.iter().enumerate() {
        if v < 0 {
            return Err(i);
        }
        let v = v as usize;
        edges.push((u as u32, v as u32));
        degree[v] -= 1;
        if v < index && degree[v] == 1 {
            u = v;
        } else {
            index = next_leaf(&degree, index + 1).ok_or(usize::MAX)?;
            u = index;
        }
    }
    Ok(edges)
}

/// A nested tuple's shape: each node's children, node 0 the root.
pub struct Shape {
    pub children: Vec<Vec<u32>>,
}

/// A NetworkX `Graph` built by `from_nested_tuple`: node order, and each
/// node's adjacency (nodes are labelled `0..n-1`).
struct Tree {
    order: Vec<u32>,
    adj: Vec<Vec<u32>>,
}

impl Tree {
    /// `G.edges()` order of an undirected graph.
    fn edges(&self) -> Vec<(u32, u32)> {
        let mut seen = vec![false; self.adj.len()];
        let mut out = Vec::new();
        for &a in &self.order {
            for &b in &self.adj[a as usize] {
                if !seen[b as usize] {
                    out.push((a, b));
                }
            }
            seen[a as usize] = true;
        }
        out
    }
}

/// `join_trees([(T, 0) for T in trees])` for trees made by `_make_tree`,
/// returning the joined tree and the edges in the order `R` received them.
fn join_trees(trees: Vec<Tree>) -> (Tree, Vec<(u32, u32)>) {
    let total: usize = 1 + trees.iter().map(|t| t.order.len()).sum::<usize>();
    let mut joined = Tree {
        order: Vec::with_capacity(total),
        adj: vec![Vec::new(); total],
    };
    let mut added = Vec::with_capacity(total - 1);
    let mut roots = Vec::with_capacity(trees.len());
    let mut first = 1u32;
    for tree in &trees {
        // convert_node_labels_to_integers: labels follow node order, so the
        // copy's nodes are `first + 0..size` in order. relabel_nodes adds the
        // copy's edges in `tree.edges()` order; R.update then adds the
        // copy's nodes, then its edges in the copy's `edges()` order.
        let size = tree.order.len();
        let mut local = vec![0u32; size];
        for (i, &x) in tree.order.iter().enumerate() {
            local[x as usize] = i as u32;
        }
        roots.push(first + local[0]);
        let mut copy = Tree {
            order: (0..size as u32).collect(),
            adj: vec![Vec::new(); size],
        };
        for (a, b) in tree.edges() {
            let (a, b) = (local[a as usize], local[b as usize]);
            copy.adj[a as usize].push(b);
            copy.adj[b as usize].push(a);
        }
        joined.order.extend(copy.order.iter().map(|&x| x + first));
        for (a, b) in copy.edges() {
            let (a, b) = (a + first, b + first);
            joined.adj[a as usize].push(b);
            joined.adj[b as usize].push(a);
            added.push((a, b));
        }
        first += size as u32;
    }
    joined.order.push(0);
    for r in roots {
        joined.adj[0].push(r);
        joined.adj[r as usize].push(0);
        added.push((0, r));
    }
    (joined, added)
}

/// `from_nested_tuple`'s graph: node order and edges in insertion order.
pub fn nested_tuple_tree(shape: &Shape, sensible: bool) -> (Vec<u32>, Vec<(u32, u32)>) {
    // Build subtrees bottom-up (children have larger ids than parents).
    let n = shape.children.len();
    let mut built: Vec<Option<Tree>> = (0..n).map(|_| None).collect();
    let mut added = Vec::new(); // the last join's edges: the root's
    for x in (0..n).rev() {
        let kids = &shape.children[x];
        let tree = if kids.is_empty() {
            added = Vec::new();
            Tree {
                order: vec![0],
                adj: vec![Vec::new()],
            }
        } else {
            let trees = kids
                .iter()
                .map(|&c| built[c as usize].take().expect("child built"))
                .collect();
            let (tree, edges) = join_trees(trees);
            added = edges;
            tree
        };
        built[x] = Some(tree);
    }
    let tree = built[0].take().expect("root built");
    if !sensible {
        return (tree.order, added);
    }
    // bfs_edges(T, 0), then relabel_nodes(T, labels): a copy that adds the
    // nodes in order, then the edges in `T.edges()` order.
    let size = tree.order.len();
    let mut label = vec![NONE; size];
    label[0] = 0;
    let mut queue = vec![0u32];
    let mut head = 0;
    let mut next = 1u32;
    while head < queue.len() {
        let a = queue[head];
        head += 1;
        for &b in &tree.adj[a as usize] {
            if label[b as usize] == NONE {
                label[b as usize] = next;
                next += 1;
                queue.push(b);
            }
        }
    }
    let order = tree.order.iter().map(|&x| label[x as usize]).collect();
    let edges = tree
        .edges()
        .into_iter()
        .map(|(a, b)| (label[a as usize], label[b as usize]))
        .collect();
    (order, edges)
}

/// `to_nested_tuple(T, root, canonical_form=True)`: each node's children in
/// the order `sorted()` puts their tuples, and a postorder of the nodes.
/// Python orders nested tuples lexicographically, `()` first.
pub fn canonical_children(adj: &Csr, root: u32) -> (Vec<u32>, Vec<Vec<u32>>) {
    let n = adj.offsets.len() - 1;
    let mut parent = vec![NONE; n];
    let mut children: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut preorder = vec![root];
    let mut seen = vec![false; n];
    seen[root as usize] = true;
    let mut i = 0;
    while i < preorder.len() {
        let a = preorder[i];
        i += 1;
        for &b in adj.neighbors(a as usize) {
            if !seen[b as usize] {
                seen[b as usize] = true;
                parent[b as usize] = a;
                children[a as usize].push(b);
                preorder.push(b);
            }
        }
    }
    // Children before parents: reverse BFS order.
    let post: Vec<u32> = preorder.iter().rev().copied().collect();
    let mut shape = vec![NONE; n];
    let mut ids: HashMap<Vec<u32>, u32> = HashMap::new();
    for &x in &post {
        let mut kids = std::mem::take(&mut children[x as usize]);
        kids.sort_by(|&a, &b| compare_shapes(a, b, &shape, &children));
        let key: Vec<u32> = kids.iter().map(|&k| shape[k as usize]).collect();
        let next = ids.len() as u32;
        shape[x as usize] = *ids.entry(key).or_insert(next);
        children[x as usize] = kids;
    }
    (post, children)
}

/// Compares two subtrees' nested tuples (children already sorted). Equal
/// shapes have equal ids; otherwise the first differing child decides.
fn compare_shapes(mut a: u32, mut b: u32, shape: &[u32], children: &[Vec<u32>]) -> Ordering {
    loop {
        if shape[a as usize] == shape[b as usize] {
            return Ordering::Equal;
        }
        let (ca, cb) = (&children[a as usize], &children[b as usize]);
        match ca
            .iter()
            .zip(cb)
            .find(|&(&x, &y)| shape[x as usize] != shape[y as usize])
        {
            Some((&x, &y)) => {
                a = x;
                b = y;
            }
            None => return ca.len().cmp(&cb.len()),
        }
    }
}

// --- Lowest common ancestors -------------------------------------------------------

/// Ancestor sets (as bitsets, each including the node itself) for
/// `all_pairs_lowest_common_ancestor`, cached per node as NetworkX does.
pub struct DagLca {
    succ: Csr,
    pred: Csr,
    words: usize,
    cache: HashMap<u32, Vec<u64>>,
}

/// Result of one LCA query: no common ancestor, or several lowest ones (then
/// NetworkX's answer depends on set order, and Python works it out).
pub const LCA_NONE: i64 = -1;
pub const LCA_AMBIGUOUS: i64 = -2;

impl DagLca {
    pub fn new(succ: Csr, pred: Csr) -> Self {
        let n = succ.offsets.len() - 1;
        DagLca {
            succ,
            pred,
            words: n.div_ceil(64),
            cache: HashMap::new(),
        }
    }

    pub fn node_count(&self) -> usize {
        self.succ.offsets.len() - 1
    }

    fn ancestors(&mut self, v: u32) -> &Vec<u64> {
        let (pred, words) = (&self.pred, self.words);
        self.cache.entry(v).or_insert_with(|| {
            let mut bits = vec![0u64; words];
            bits[v as usize / 64] |= 1 << (v % 64);
            let mut stack = vec![v];
            while let Some(x) = stack.pop() {
                for &p in pred.neighbors(x as usize) {
                    let (i, b) = (p as usize / 64, 1u64 << (p % 64));
                    if bits[i] & b == 0 {
                        bits[i] |= b;
                        stack.push(p);
                    }
                }
            }
            bits
        })
    }

    /// The lowest common ancestor of each pair, if it is the only sink of
    /// the common ancestors (NetworkX walks down to some sink from an
    /// arbitrary common ancestor, so a unique sink is its answer).
    pub fn query(&mut self, us: &[u32], vs: &[u32]) -> Vec<i64> {
        let mut out = Vec::with_capacity(us.len());
        let mut common = vec![0u64; self.words];
        for (&u, &v) in us.iter().zip(vs) {
            self.ancestors(u);
            self.ancestors(v);
            let (a, b) = (&self.cache[&u], &self.cache[&v]);
            for i in 0..self.words {
                common[i] = a[i] & b[i];
            }
            let has = |x: u32| common[x as usize / 64] >> (x % 64) & 1 == 1;
            let mut sink = LCA_NONE;
            'nodes: for (i, &word) in common.iter().enumerate() {
                let mut word = word;
                while word != 0 {
                    let x = (i * 64 + word.trailing_zeros() as usize) as u32;
                    word &= word - 1;
                    if self.succ.neighbors(x as usize).iter().any(|&y| has(y)) {
                        continue;
                    }
                    if sink != LCA_NONE {
                        sink = LCA_AMBIGUOUS;
                        break 'nodes;
                    }
                    sink = x as i64;
                }
            }
            out.push(sink);
        }
        out
    }
}

/// `tree_all_pairs_lowest_common_ancestor` without `pairs`: Tarjan's
/// offline algorithm as NetworkX runs it, yielding `(v, node, ancestor)` for
/// each node in DFS postorder and each already-colored `v` in node order.
pub struct TreeLca {
    post: Vec<u32>,
    step: usize,
    parent: Vec<u32>,
    root: u32,
    uf: Vec<u32>,
    value: Vec<u32>,
    colored: BTreeSet<u32>,
}

impl TreeLca {
    /// `parent[v]` is the first node in `G.pred[v]` (or `NONE`).
    pub fn new(succ: &Csr, root: u32, parent: Vec<u32>) -> Self {
        let n = parent.len();
        TreeLca {
            post: dfs_postorder(succ, root),
            step: 0,
            parent,
            root,
            uf: (0..n as u32).collect(),
            value: (0..n as u32).collect(),
            colored: BTreeSet::new(),
        }
    }

    /// Results for the next nodes, until at least `limit` are produced.
    pub fn next_batch(&mut self, limit: usize) -> Vec<(u32, u32, u32)> {
        let mut out = Vec::new();
        while self.step < self.post.len() && out.len() < limit {
            let node = self.post[self.step];
            self.step += 1;
            self.colored.insert(node);
            for &v in &self.colored {
                let r = find(&mut self.uf, v);
                out.push((v, node, self.value[r as usize]));
            }
            if node != self.root {
                let p = self.parent[node as usize];
                let (rp, rn) = (find(&mut self.uf, p), find(&mut self.uf, node));
                if rp != rn {
                    self.uf[rn as usize] = rp;
                }
                let r = find(&mut self.uf, p);
                self.value[r as usize] = p;
            }
        }
        out
    }
}

/// `dfs_postorder_nodes(G, source)`.
fn dfs_postorder(succ: &Csr, source: u32) -> Vec<u32> {
    let n = succ.offsets.len() - 1;
    let mut seen = vec![false; n];
    let mut post = Vec::new();
    let mut stack = vec![(source, succ.offsets[source as usize])];
    seen[source as usize] = true;
    while let Some(top) = stack.last_mut() {
        let x = top.0;
        let end = succ.offsets[x as usize + 1];
        let mut pushed = None;
        while top.1 < end {
            let y = succ.targets[top.1];
            top.1 += 1;
            if !seen[y as usize] {
                seen[y as usize] = true;
                pushed = Some(y);
                break;
            }
        }
        match pushed {
            Some(y) => stack.push((y, succ.offsets[y as usize])),
            None => {
                post.push(x);
                stack.pop();
            }
        }
    }
    post
}
