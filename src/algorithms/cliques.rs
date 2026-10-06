//! Batch 25: cliques, structure and approximation algorithms whose results
//! follow the iteration order of Python sets. Each replays NetworkX's set
//! operations on `pyset::PySet` (node positions, with each node's `hash()`)
//! so the order matches; the Python side checks the replica once per
//! interpreter.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};

use super::operators::tuple_hash;
use super::pyrandom::Mt19937;
use super::pyset::PySet;
use crate::graph::Csr;

/// `{v for v in G[u] if v != u}` for every node (`find_cliques`' `adj`).
fn loopless_sets(adj: &Csr, n: usize, hashes: &[i64]) -> Vec<PySet> {
    (0..n)
        .map(|u| {
            PySet::from_iter(
                adj.neighbors(u)
                    .iter()
                    .copied()
                    .filter(|&v| v as usize != u),
                hashes,
            )
        })
        .collect()
}

/// `len(a & b)`, without building the set.
fn common_count(a: &PySet, b: &PySet, hashes: &[i64]) -> usize {
    let (small, big) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    small.iter().filter(|&k| big.contains(k, hashes)).count()
}

/// The state of `find_cliques` (and `find_cliques_recursive`, which visits
/// the same cliques in the same order) after its setup: Bron-Kerbosch with
/// Tomita pivots, replaying NetworkX's set operations so pivots (the first
/// best in `subg`'s order) and `ext_u` orders match.
pub struct CliqueSearch {
    hashes: Vec<i64>,
    adj: Vec<PySet>,
    /// The clique being grown (without the caller's `nodes`).
    q: Vec<u32>,
    subg: PySet,
    cand: PySet,
    ext: Vec<u32>,
    at: usize,
    stack: Vec<(PySet, PySet, Vec<u32>, usize)>,
    done: bool,
}

/// What `find_cliques`' setup found for the caller's `nodes`.
pub enum CliqueStart {
    /// `nodes[i]` isn't a node of G or isn't adjacent to all before it.
    NotAClique,
    /// No node extends `nodes`: the only clique is `nodes` itself.
    Only,
    Search(Box<CliqueSearch>),
}

impl CliqueSearch {
    /// `prefix`: the caller's `nodes` as positions (`None` for an object
    /// that isn't a node of G).
    pub fn start(adj: &Csr, n: usize, hashes: Vec<i64>, prefix: &[Option<u32>]) -> CliqueStart {
        let adj_sets = loopless_sets(adj, n, &hashes);
        // cand = set(G); for node in Q: cand &= adj[node]
        let mut cand = PySet::from_iter(0..n as u32, &hashes);
        for &node in prefix {
            match node {
                Some(v) if cand.contains(v, &hashes) => {
                    cand.intersection_update(&adj_sets[v as usize], &hashes)
                }
                _ => return CliqueStart::NotAClique,
            }
        }
        if cand.is_empty() {
            return CliqueStart::Only;
        }
        let subg = cand.copy(&hashes);
        let mut search = CliqueSearch {
            hashes,
            adj: adj_sets,
            q: vec![u32::MAX],
            subg,
            cand,
            ext: Vec::new(),
            at: 0,
            stack: Vec::new(),
            done: false,
        };
        search.ext = search.pivot_ext();
        CliqueStart::Search(Box::new(search))
    }

    /// `u = max(subg, key=lambda u: len(cand & adj[u]))`, then the
    /// iteration order of `cand - adj[u]` (`ext_u.pop()` takes elements in
    /// that order, as nothing is added to it).
    fn pivot_ext(&self) -> Vec<u32> {
        let h = &self.hashes;
        let mut best: Option<(u32, usize)> = None;
        for u in self.subg.iter() {
            let c = common_count(&self.cand, &self.adj[u as usize], h);
            if best.is_none_or(|(_, b)| c > b) {
                best = Some((u, c));
            }
        }
        let u = best.expect("subg is not empty").0;
        self.cand
            .difference(&self.adj[u as usize], h)
            .iter()
            .collect()
    }

    /// Up to `limit` more cliques (each without the caller's `nodes`).
    pub fn next_batch(&mut self, limit: usize) -> Vec<Vec<u32>> {
        let mut out = Vec::new();
        while !self.done && out.len() < limit {
            if self.at < self.ext.len() {
                let q = self.ext[self.at];
                self.at += 1;
                let h = &self.hashes;
                self.cand.discard(q, h);
                *self.q.last_mut().expect("Q is not empty") = q;
                let adj_q = &self.adj[q as usize];
                let subg_q = self.subg.intersection(adj_q, h);
                if subg_q.is_empty() {
                    out.push(self.q.clone());
                } else {
                    let cand_q = self.cand.intersection(adj_q, h);
                    if !cand_q.is_empty() {
                        let subg = std::mem::replace(&mut self.subg, subg_q);
                        let cand = std::mem::replace(&mut self.cand, cand_q);
                        let ext = std::mem::take(&mut self.ext);
                        self.stack.push((subg, cand, ext, self.at));
                        self.q.push(u32::MAX);
                        self.ext = self.pivot_ext();
                        self.at = 0;
                    }
                }
            } else {
                self.q.pop();
                match self.stack.pop() {
                    Some((subg, cand, ext, at)) => {
                        self.subg = subg;
                        self.cand = cand;
                        self.ext = ext;
                        self.at = at;
                    }
                    None => self.done = true,
                }
            }
        }
        out
    }
}

/// All maximal cliques in `find_cliques`' order (no `nodes`).
pub fn all_cliques(adj: &Csr, n: usize, hashes: Vec<i64>) -> Vec<Vec<u32>> {
    if n == 0 {
        return Vec::new();
    }
    match CliqueSearch::start(adj, n, hashes, &[]) {
        CliqueStart::Search(mut s) => s.next_batch(usize::MAX),
        _ => unreachable!("set(G) is not empty"),
    }
}

/// `make_max_clique_graph`'s edges: `(i, j)` for `i < j` in order where
/// cliques `i` and `j` share a node.
pub fn clique_overlaps(cliques: &[Vec<u32>], n: usize) -> Vec<(u32, u32)> {
    let mut member: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (i, c) in cliques.iter().enumerate() {
        for &v in c {
            member[v as usize].push(i as u32);
        }
    }
    let mut mark = vec![u32::MAX; cliques.len()];
    let mut out = Vec::new();
    let mut row = Vec::new();
    for (i, c) in cliques.iter().enumerate() {
        row.clear();
        for &v in c {
            for &j in &member[v as usize] {
                if j as usize > i && mark[j as usize] != i as u32 {
                    mark[j as usize] = i as u32;
                    row.push(j);
                }
            }
        }
        row.sort_unstable();
        out.extend(row.iter().map(|&j| (i as u32, j)));
    }
    out
}

/// `nx.dominating_set(G, start)` (`None`: `arbitrary_element(set(G))`) on
/// a non-empty graph: the nodes in the order they join the set (`{start}`,
/// then one `add` per round).
pub fn dominating_set(adj: &Csr, n: usize, hashes: &[i64], start: Option<u32>) -> Vec<u32> {
    let all_nodes = PySet::from_iter(0..n as u32, hashes);
    let start = start.unwrap_or_else(|| all_nodes.first().expect("G is not empty"));
    let mut dominating = PySet::from_iter([start], hashes);
    let dominated = PySet::from_iter(adj.neighbors(start as usize).iter().copied(), hashes);
    let mut remaining = all_nodes
        .difference(&dominated, hashes)
        .difference(&dominating, hashes);
    let mut order = vec![start];
    while let Some(v) = remaining.pop() {
        let undominated = PySet::from_iter(adj.neighbors(v as usize).iter().copied(), hashes)
            .difference(&dominating, hashes);
        dominating.add(v, hashes);
        order.push(v);
        remaining.difference_update(&undominated, hashes);
    }
    order
}

/// The draws of `maximal_independent_set` after its checks: `nodes` (the
/// starting set, positions) and `rng` at that point. Returns the nodes
/// added, in order.
pub fn maximal_independent_set(
    adj: &Csr,
    n: usize,
    hashes: &[i64],
    nodes: &[u32],
    rng: &mut Mt19937,
) -> Vec<u32> {
    // available = set(G.nodes()).difference(neighbors.union(nodes)): only
    // membership in the right operand matters.
    let mut excluded = vec![false; n];
    for &v in nodes {
        excluded[v as usize] = true;
        for &w in adj.neighbors(v as usize) {
            excluded[w as usize] = true;
        }
    }
    let excluded_set = PySet::from_iter((0..n as u32).filter(|&v| excluded[v as usize]), hashes);
    let mut available = PySet::from_iter(0..n as u32, hashes).difference(&excluded_set, hashes);
    // seed.choice(list(available)): the k-th element in table order, found
    // with a Fenwick tree over the table's slots.
    let mut slots = SlotCounts::of(&available);
    let mut added = Vec::new();
    while !available.is_empty() {
        let k = rng.below(available.len());
        let node = available.table()[slots.kth(k)];
        added.push(node);
        let row = adj.neighbors(node as usize).iter().copied();
        for v in row.chain([node]) {
            if let Some(slot) = available.discard_at(v, hashes) {
                slots.remove(slot);
            }
        }
        if available.after_discards(hashes) {
            slots = SlotCounts::of(&available);
        }
    }
    added
}

/// A Fenwick tree counting a set's live slots.
struct SlotCounts {
    tree: Vec<usize>,
}

impl SlotCounts {
    fn of(set: &PySet) -> Self {
        let table = set.table();
        let size = table.len();
        let mut tree = vec![0usize; size + 1];
        for (i, &k) in table.iter().enumerate() {
            if PySet::live(k) {
                tree[i + 1] += 1;
            }
        }
        for i in 1..=size {
            let j = i + (i & i.wrapping_neg());
            if j <= size {
                tree[j] += tree[i];
            }
        }
        SlotCounts { tree }
    }

    fn remove(&mut self, slot: usize) {
        let mut i = slot + 1;
        while i < self.tree.len() {
            self.tree[i] -= 1;
            i += i & i.wrapping_neg();
        }
    }

    /// The slot of the `k`-th live element (from 0).
    fn kth(&self, mut k: usize) -> usize {
        let size = self.tree.len() - 1;
        let mut pos = 0;
        let mut step = size.next_power_of_two();
        while step > 0 {
            let next = pos + step;
            if next <= size && self.tree[next] <= k {
                pos = next;
                k -= self.tree[next];
            }
            step >>= 1;
        }
        pos
    }
}

/// `nx.approximation.large_clique_size`. `degree[v]` is `G.degree[v]`.
pub fn large_clique_size(adj: &Csr, n: usize, hashes: &[i64], degree: &[usize]) -> usize {
    let mut best = 0;
    let candidates = |u: usize, best: usize| {
        PySet::from_iter(
            adj.neighbors(u)
                .iter()
                .copied()
                .filter(|&v| degree[v as usize] >= best),
            hashes,
        )
    };
    for u in 0..n {
        if degree[u] < best {
            continue;
        }
        let mut set = candidates(u, best);
        let mut size = 1;
        loop {
            if set.is_empty() {
                best = best.max(size);
                break;
            }
            // u = max(U, key=degrees): the first of the largest degree
            let mut pick: Option<u32> = None;
            for v in set.iter() {
                if pick.is_none_or(|p| degree[v as usize] > degree[p as usize]) {
                    pick = Some(v);
                }
            }
            let v = pick.expect("set is not empty");
            set.discard(v, hashes);
            let nbrs = candidates(v as usize, best);
            set = set.intersection(&nbrs, hashes);
            size += 1;
        }
    }
    best
}

/// A NetworkX `Graph` as rustnx's replay builds it: node order (original
/// positions) and each node's row (aligned with `order`), in the order
/// `G.copy()` and `G.subgraph(...).copy()` leave them.
#[derive(Clone, Default)]
pub struct SimGraph {
    pub order: Vec<u32>,
    pub rows: Vec<Vec<u32>>,
}

/// Position lookups for one `SimGraph` at a time (stamped, so no clearing).
struct Lookup {
    at: Vec<u32>,
    stamp: Vec<u32>,
    now: u32,
}

impl Lookup {
    fn new(n: usize) -> Self {
        Lookup {
            at: vec![0; n],
            stamp: vec![0; n],
            now: 0,
        }
    }

    fn load(&mut self, order: &[u32]) {
        self.now += 1;
        for (i, &v) in order.iter().enumerate() {
            self.at[v as usize] = i as u32;
            self.stamp[v as usize] = self.now;
        }
    }

    #[inline]
    fn get(&self, v: u32) -> Option<usize> {
        (self.stamp[v as usize] == self.now).then(|| self.at[v as usize] as usize)
    }
}

impl SimGraph {
    /// The NetworkX graph itself (rows as the CSR holds them).
    pub fn of(adj: &Csr, n: usize) -> Self {
        SimGraph {
            order: (0..n as u32).collect(),
            rows: (0..n).map(|v| adj.neighbors(v).to_vec()).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// `self.subgraph(nodes).copy()` for an undirected simple graph, where
    /// `nodes` is `show_nodes`' set (every element a node of `self`). The
    /// view iterates `nodes` itself when it is under half the graph, else
    /// `self` filtered; `copy` then adds the edges row by row.
    fn subgraph_copy(
        &self,
        nodes: &PySet,
        hashes: &[i64],
        mine: &mut Lookup,
        sub: &mut Lookup,
    ) -> SimGraph {
        let order: Vec<u32> = if 2 * nodes.len() < self.len() {
            nodes.iter().collect()
        } else {
            self.order
                .iter()
                .copied()
                .filter(|&v| nodes.contains(v, hashes))
                .collect()
        };
        self.copy_onto(order, mine, sub)
    }

    /// `copy()` of the induced subgraph on `order` (in that node order).
    fn copy_onto(&self, order: Vec<u32>, mine: &mut Lookup, sub: &mut Lookup) -> SimGraph {
        mine.load(&self.order);
        sub.load(&order);
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); order.len()];
        for (i, &u) in order.iter().enumerate() {
            let p = mine.get(u).expect("subgraph nodes are nodes of the graph");
            for &v in &self.rows[p] {
                let Some(j) = sub.get(v) else { continue };
                if j == i {
                    rows[i].push(u);
                } else if j > i {
                    // add_edge(u, v): new in both rows. (j < i: the edge
                    // came in from v's row already.)
                    rows[i].push(v);
                    rows[j].push(u);
                }
            }
        }
        SimGraph { order, rows }
    }

    /// `remove_nodes_from` (the order of removal doesn't matter).
    fn remove(&mut self, gone: &[u32], n: usize) {
        let mut dead = vec![false; n];
        for &v in gone {
            dead[v as usize] = true;
        }
        let mut order = Vec::with_capacity(self.order.len());
        let mut rows = Vec::with_capacity(self.order.len());
        for (i, &v) in self.order.iter().enumerate() {
            if !dead[v as usize] {
                order.push(v);
                rows.push(
                    self.rows[i]
                        .iter()
                        .copied()
                        .filter(|&w| !dead[w as usize])
                        .collect(),
                );
            }
        }
        self.order = order;
        self.rows = rows;
    }

    /// `nx.complement(G)`: each row holds the node's non-neighbors in node
    /// order (the earlier ones from their own `add_edge`, then its own).
    pub fn complement(adj: &Csr, n: usize) -> Self {
        let mut mark = vec![usize::MAX; n];
        let rows = (0..n)
            .map(|u| {
                for &v in adj.neighbors(u) {
                    mark[v as usize] = u;
                }
                (0..n as u32)
                    .filter(|&v| v as usize != u && mark[v as usize] != u)
                    .collect()
            })
            .collect();
        SimGraph {
            order: (0..n as u32).collect(),
            rows,
        }
    }
}

/// Replays `ramsey_R2` (undirected simple graphs): its clique and
/// independent set, each as its nodes in the order they were added.
struct Ramsey<'a> {
    hashes: &'a [i64],
    mine: Lookup,
    sub: Lookup,
}

enum RamseyTask {
    Call(SimGraph),
    Second(SimGraph, u32),
    Combine(u32),
}

type CliqueAndSet = (Vec<u32>, Vec<u32>);

impl<'a> Ramsey<'a> {
    fn new(n: usize, hashes: &'a [i64]) -> Self {
        Ramsey {
            hashes,
            mine: Lookup::new(n),
            sub: Lookup::new(n),
        }
    }

    /// The recursion on an explicit stack: `ramsey_R2(G.subgraph(nbrs).copy())`
    /// then `ramsey_R2(G.subgraph(nnbrs).copy())`, then the larger of each.
    fn run(&mut self, g: SimGraph) -> CliqueAndSet {
        let h = self.hashes;
        let mut tasks = vec![RamseyTask::Call(g)];
        let mut results: Vec<CliqueAndSet> = Vec::new();
        while let Some(task) = tasks.pop() {
            match task {
                RamseyTask::Call(g) => {
                    if g.len() == 0 {
                        results.push((Vec::new(), Vec::new()));
                        continue;
                    }
                    let node = g.order[0];
                    // set(nbunch_iter(nbr for nbr in G[node] if nbr != node))
                    let nbrs =
                        PySet::from_iter(g.rows[0].iter().copied().filter(|&v| v != node), h);
                    let c1 = g.subgraph_copy(&nbrs, h, &mut self.mine, &mut self.sub);
                    tasks.push(RamseyTask::Second(g, node));
                    tasks.push(RamseyTask::Call(c1));
                }
                RamseyTask::Second(g, node) => {
                    // non_neighbors: G._adj.keys() - G._adj[node].keys() - {node}
                    let mut nn = PySet::from_dict(&g.order, h);
                    nn.discard_all(g.rows[0].iter().copied(), h);
                    let nn = nn.difference(&PySet::from_iter([node], h), h);
                    let nodes = PySet::from_iter(nn.iter(), h);
                    let c2 = g.subgraph_copy(&nodes, h, &mut self.mine, &mut self.sub);
                    drop(g);
                    tasks.push(RamseyTask::Combine(node));
                    tasks.push(RamseyTask::Call(c2));
                }
                RamseyTask::Combine(node) => {
                    let (c2, mut i2) = results.pop().expect("second result");
                    let (mut c1, i1) = results.pop().expect("first result");
                    c1.push(node);
                    i2.push(node);
                    let c = if c1.len() >= c2.len() { c1 } else { c2 };
                    let i = if i1.len() >= i2.len() { i1 } else { i2 };
                    results.push((c, i));
                }
            }
        }
        results.pop().expect("one result")
    }
}

/// `nx.approximation.ramsey_R2(G)`.
pub fn ramsey_r2(adj: &Csr, n: usize, hashes: &[i64]) -> CliqueAndSet {
    Ramsey::new(n, hashes).run(SimGraph::of(adj, n))
}

/// `clique_removal` on `g` (`G` itself, or `max_clique`'s complement):
/// `(index of the largest independent set, independent sets, cliques)`.
pub fn clique_removal(
    g: SimGraph,
    n: usize,
    hashes: &[i64],
) -> (usize, Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let mut ramsey = Ramsey::new(n, hashes);
    let all = g.order.clone();
    let mut graph = g.copy_onto(all, &mut ramsey.mine, &mut ramsey.sub);
    let (mut c, i) = ramsey.run(graph.clone());
    let mut cliques = vec![c.clone()];
    let mut isets = vec![i];
    while graph.len() > 0 {
        graph.remove(&c, n);
        let (c_i, i_i) = ramsey.run(graph.clone());
        if !c_i.is_empty() {
            cliques.push(c_i.clone());
        }
        if !i_i.is_empty() {
            isets.push(i_i);
        }
        c = c_i;
    }
    let mut best = 0;
    for (k, s) in isets.iter().enumerate() {
        if s.len() > isets[best].len() {
            best = k;
        }
    }
    (best, isets, cliques)
}

/// `nx.tournament.hamiltonian_path` (any digraph): `v = arbitrary_element(G)`
/// is the first node of each nested subgraph view, whose order follows
/// Python's set order once the view's node set is under half the graph.
/// `succ_sorted` holds each node's successors, sorted.
pub fn hamiltonian_path(succ_sorted: &[Vec<u32>], hashes: &[i64]) -> Vec<u32> {
    let n = succ_sorted.len();
    if n == 0 {
        return Vec::new();
    }
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut pivots = Vec::new();
    let mut inside = vec![false; n];
    while order.len() > 1 {
        let v = order[0];
        pivots.push(v);
        // show_nodes(nbunch_iter(set(G) - {v})).nodes
        let all = PySet::from_iter(order.iter().copied(), hashes);
        let rest = all.difference(&PySet::from_iter([v], hashes), hashes);
        let nodes = PySet::from_iter(rest.iter(), hashes);
        order = if 2 * nodes.len() < n {
            nodes.iter().collect()
        } else {
            for &u in &order {
                inside[u as usize] = false;
            }
            for u in nodes.iter() {
                inside[u as usize] = true;
            }
            (0..n as u32).filter(|&u| inside[u as usize]).collect()
        };
    }
    let mut path = vec![order[0]];
    for &v in pivots.iter().rev() {
        let index = path
            .iter()
            .position(|&u| succ_sorted[u as usize].binary_search(&v).is_err())
            .unwrap_or(path.len());
        path.insert(index, v);
    }
    path
}

/// Where `chordal_graph_cliques` raises.
pub enum ChordalError {
    NotChordal,
    SelfLoop,
}

/// `chordal_graph_cliques` (undirected simple graphs): the cliques in
/// order, each as the order its set was built in (`frozenset(set(c))`
/// rebuilds it), and the error that stops the generator, if any.
pub fn chordal_graph_cliques(
    adj: &Csr,
    n: usize,
    hashes: &[i64],
) -> (Vec<Vec<u32>>, Option<ChordalError>) {
    let g = SimGraph::of(adj, n);
    let mut mine = Lookup::new(n);
    let mut sub = Lookup::new(n);
    let mut out = Vec::new();
    let mut seen = vec![false; n];
    let has_loop = |v: u32| adj.neighbors(v as usize).contains(&v);
    let mut in_numbered = vec![false; n];
    let mut count = vec![0usize; n];
    let mut in_cwb = vec![false; n];
    for start in 0..n {
        if seen[start] {
            continue;
        }
        // _plain_bfs: the component set, built in discovery order
        let mut comp = vec![start as u32];
        seen[start] = true;
        let mut k = 0;
        while k < comp.len() {
            let v = comp[k] as usize;
            k += 1;
            for &w in adj.neighbors(v) {
                if !seen[w as usize] {
                    seen[w as usize] = true;
                    comp.push(w);
                }
            }
        }
        let c = PySet::from_iter(comp.iter().copied(), hashes);
        let nodes = PySet::from_iter(c.iter(), hashes);
        let cg = g.subgraph_copy(&nodes, hashes, &mut mine, &mut sub);
        if cg.len() == 1 {
            if has_loop(cg.order[0]) {
                return (out, Some(ChordalError::NotChordal));
            }
            out.push(vec![cg.order[0]]);
            continue;
        }
        sub.load(&cg.order);
        let mut unnumbered = PySet::from_iter(cg.order.iter().copied(), hashes);
        let first = cg.order[0];
        unnumbered.discard(first, hashes);
        // Only removals follow, so `unnumbered` keeps its iteration order:
        // `max` takes the most-connected node of lowest rank.
        let ranked: Vec<u32> = unnumbered.iter().collect();
        let mut rank = vec![0u32; 0];
        rank.resize(cg.len(), 0);
        let mut buckets: Vec<BTreeSet<u32>> = vec![BTreeSet::new()];
        for (r, &v) in ranked.iter().enumerate() {
            rank[sub.get(v).expect("component node")] = r as u32;
            buckets[0].insert(r as u32);
        }
        let mut top = 0usize;
        let mut numbered = PySet::from_iter([first], hashes);
        let mut cwb = vec![first];
        let number = |v: u32,
                      in_numbered: &mut Vec<bool>,
                      count: &mut Vec<usize>,
                      buckets: &mut Vec<BTreeSet<u32>>,
                      top: &mut usize,
                      unnumbered: &PySet| {
            in_numbered[v as usize] = true;
            for &w in adj.neighbors(v as usize) {
                if w != v && !in_numbered[w as usize] && unnumbered.contains(w, hashes) {
                    let r = rank[sub.get(w).expect("component node")];
                    let c = count[w as usize];
                    buckets[c].remove(&r);
                    if buckets.len() <= c + 1 {
                        buckets.push(BTreeSet::new());
                    }
                    buckets[c + 1].insert(r);
                    count[w as usize] = c + 1;
                    *top = (*top).max(c + 1);
                }
            }
        };
        number(
            first,
            &mut in_numbered,
            &mut count,
            &mut buckets,
            &mut top,
            &unnumbered,
        );
        let mut error = None;
        while !unnumbered.is_empty() {
            while buckets[top].is_empty() {
                top -= 1;
            }
            let r = *buckets[top].iter().next().expect("bucket is not empty");
            buckets[top].remove(&r);
            let v = ranked[r as usize];
            unnumbered.discard(v, hashes);
            numbered.add(v, hashes);
            number(
                v,
                &mut in_numbered,
                &mut count,
                &mut buckets,
                &mut top,
                &unnumbered,
            );
            // new_clique_wanna_be = set(C.neighbors(v)) & numbered
            let row = PySet::from_iter(
                cg.rows[sub.get(v).expect("component node")].iter().copied(),
                hashes,
            );
            let (small, big) = if numbered.len() > row.len() {
                (&row, &numbered)
            } else {
                (&numbered, &row)
            };
            let mut new_cwb: Vec<u32> = small.iter().filter(|&x| big.contains(x, hashes)).collect();
            // _is_complete_graph(C.subgraph(clique_wanna_be))
            for &x in &cwb {
                in_cwb[x as usize] = true;
            }
            let mut loops = false;
            let mut degree_sum = 0usize;
            for &x in &cwb {
                for &y in adj.neighbors(x as usize) {
                    if y == x {
                        loops = true;
                    } else if in_cwb[y as usize] {
                        degree_sum += 1;
                    }
                }
            }
            let size = cwb.len();
            let complete = size < 2 || degree_sum == size * (size - 1);
            if loops {
                error = Some(ChordalError::SelfLoop);
            } else if !complete {
                error = Some(ChordalError::NotChordal);
            }
            if error.is_some() {
                for &x in &cwb {
                    in_cwb[x as usize] = false;
                }
                break;
            }
            new_cwb.push(v);
            let mut covered = 0;
            for &x in &new_cwb {
                if in_cwb[x as usize] {
                    covered += 1;
                }
            }
            for &x in &cwb {
                in_cwb[x as usize] = false;
            }
            if covered < cwb.len() {
                out.push(std::mem::take(&mut cwb));
            }
            cwb = new_cwb;
        }
        for &v in &cg.order {
            in_numbered[v as usize] = false;
            count[v as usize] = 0;
        }
        if error.is_some() {
            return (out, error);
        }
        out.push(cwb);
    }
    (out, None)
}

/// `treewidth_min_degree`'s eliminations: the nodes `MinDegreeHeuristic`
/// picks, in order, and for each bag `treewidth_decomp` adds (last
/// elimination first) the index of the bag it joins (0 the first bag, `k`
/// the `k`-th bag added). The heuristic pushes each eliminated node's
/// neighbors in its set's iteration order, so ties follow set order.
pub fn treewidth_min_degree(adj: &Csr, n: usize, hashes: &[i64]) -> (Vec<u32>, Vec<u32>) {
    let mut sets: Vec<PySet> = (0..n)
        .map(|v| {
            PySet::from_iter(adj.neighbors(v).iter().copied(), hashes)
                .difference(&PySet::from_iter([v as u32], hashes), hashes)
        })
        .collect();
    let mut alive = vec![true; n];
    let mut left = n;
    let mut count = 0u64;
    let mut heap: BinaryHeap<Reverse<(usize, u64, u32)>> = BinaryHeap::new();
    // MinDegreeHeuristic(G) reads `len(G[n])`, which counts a self-loop:
    // that node's entry is then stale (its set has no loop) until a
    // neighbor's elimination pushes it again.
    for v in 0..n {
        heap.push(Reverse((adj.neighbors(v).len(), count, v as u32)));
        count += 1;
    }
    let mut update: Vec<u32> = Vec::new();
    let mut order = Vec::new();
    let mut stack: Vec<(u32, Vec<u32>)> = Vec::new();
    loop {
        for &v in &update {
            heap.push(Reverse((sets[v as usize].len(), count, v)));
            count += 1;
        }
        let mut pick = None;
        while let Some(Reverse((d, _, v))) = heap.pop() {
            if !alive[v as usize] || sets[v as usize].len() != d {
                continue;
            }
            if d == left - 1 {
                break;
            }
            pick = Some(v);
            break;
        }
        let Some(e) = pick else { break };
        let nbrs = std::mem::take(&mut sets[e as usize]);
        let pool: Vec<u32> = nbrs.iter().collect();
        for &u in &pool {
            for &v in &pool {
                if u != v && !sets[u as usize].contains(v, hashes) {
                    sets[u as usize].add(v, hashes);
                }
            }
        }
        for &u in &pool {
            sets[u as usize].discard(e, hashes);
        }
        alive[e as usize] = false;
        left -= 1;
        order.push(e);
        update = pool.clone();
        stack.push((e, pool));
    }
    // Rebuild the decomposition's bag choices: the first bag (in insertion
    // order) holding all of `nbrs`.
    let mut bags_of: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut bag_sets: Vec<Vec<u32>> = Vec::new();
    let first: Vec<u32> = (0..n as u32).filter(|&v| alive[v as usize]).collect();
    for &v in &first {
        bags_of[v as usize].push(0);
    }
    bag_sets.push(first);
    let mut mark = vec![u32::MAX; n];
    let mut parents = Vec::with_capacity(stack.len());
    while let Some((curr, nbrs)) = stack.pop() {
        let mut parent = 0u32;
        if let Some(&x) = nbrs.iter().min_by_key(|&&x| bags_of[x as usize].len()) {
            let tag = bag_sets.len() as u32;
            for &y in &nbrs {
                mark[y as usize] = tag;
            }
            parent = u32::MAX;
            for &b in &bags_of[x as usize] {
                let held = bag_sets[b as usize]
                    .iter()
                    .filter(|&&y| mark[y as usize] == tag)
                    .count();
                if held == nbrs.len() {
                    parent = b;
                    break;
                }
            }
            if parent == u32::MAX {
                parent = 0;
            }
        }
        parents.push(parent);
        let id = bag_sets.len() as u32;
        let mut bag = nbrs;
        bag.push(curr);
        for &v in &bag {
            bags_of[v as usize].push(id);
        }
        bag_sets.push(bag);
    }
    (order, parents)
}

/// Scratch space for the approximate node connectivity searches.
pub struct ApproxPaths {
    pred: Vec<u32>,
    succ: Vec<u32>,
    in_pred: Vec<u32>,
    in_succ: Vec<u32>,
    excluded: Vec<u32>,
    search: u32,
    pair: u32,
}

const NONE: u32 = u32::MAX;

impl ApproxPaths {
    pub fn new(n: usize) -> Self {
        ApproxPaths {
            pred: vec![NONE; n],
            succ: vec![NONE; n],
            in_pred: vec![0; n],
            in_succ: vec![0; n],
            excluded: vec![0; n],
            search: 0,
            pair: 0,
        }
    }

    /// `_bidirectional_pred_succ` then the path (`None`: no path).
    fn path(&mut self, out: &Csr, back: &Csr, s: u32, t: u32, path: &mut Vec<u32>) -> bool {
        self.search += 1;
        let gen = self.search;
        let pair = self.pair;
        self.in_pred[s as usize] = gen;
        self.pred[s as usize] = NONE;
        self.in_succ[t as usize] = gen;
        self.succ[t as usize] = NONE;
        let mut forward = vec![s];
        let mut reverse = vec![t];
        let mut level = 0u64;
        let mut meet = None;
        'search: while !forward.is_empty() && !reverse.is_empty() {
            level += 1;
            if level % 2 != 0 {
                let this = std::mem::take(&mut forward);
                for v in this {
                    for &w in out.neighbors(v as usize) {
                        if self.excluded[w as usize] == pair {
                            continue;
                        }
                        if self.in_pred[w as usize] != gen {
                            forward.push(w);
                            self.in_pred[w as usize] = gen;
                            self.pred[w as usize] = v;
                        }
                        if self.in_succ[w as usize] == gen {
                            meet = Some(w);
                            break 'search;
                        }
                    }
                }
            } else {
                let this = std::mem::take(&mut reverse);
                for v in this {
                    for &w in back.neighbors(v as usize) {
                        if self.excluded[w as usize] == pair {
                            continue;
                        }
                        if self.in_succ[w as usize] != gen {
                            self.in_succ[w as usize] = gen;
                            self.succ[w as usize] = v;
                            reverse.push(w);
                        }
                        if self.in_pred[w as usize] == gen {
                            meet = Some(w);
                            break 'search;
                        }
                    }
                }
            }
        }
        let Some(w) = meet else { return false };
        path.clear();
        let mut x = w;
        while x != NONE {
            path.push(x);
            x = self.pred[x as usize];
        }
        path.reverse();
        let mut x = self.succ[w as usize];
        while x != NONE {
            path.push(x);
            x = self.succ[x as usize];
        }
        true
    }

    /// `approximation.local_node_connectivity(G, s, t, cutoff)` for `s !=
    /// t`. NetworkX caps the paths at the smaller degree too, but every
    /// path takes its own neighbor of `s` (and of `t`), so the cap never
    /// binds.
    pub fn local(&mut self, out: &Csr, back: &Csr, s: u32, t: u32, cutoff: usize) -> usize {
        self.pair += 1;
        let mut k = 0;
        let mut path = Vec::new();
        while k < cutoff {
            if !self.path(out, back, s, t, &mut path) {
                break;
            }
            for &v in &path {
                self.excluded[v as usize] = self.pair;
            }
            k += 1;
        }
        k
    }
}

/// `approximation.local_node_connectivity` for each `(s, t)` pair, with
/// `cutoff` (`usize::MAX` for none), in parallel.
pub fn approx_pairs(
    out: &Csr,
    back: &Csr,
    n: usize,
    pairs: &[(u32, u32)],
    cutoff: usize,
) -> Vec<usize> {
    use rayon::prelude::*;
    pairs
        .par_iter()
        .map_init(
            || ApproxPaths::new(n),
            |scratch, &(s, t)| scratch.local(out, back, s, t, cutoff),
        )
        .collect()
}

/// `approximation.node_connectivity(G)` on a connected graph, from `v`, the
/// first node of the smallest degree `min_degree`. The result is the
/// minimum over the same local searches as NetworkX's; each search caps
/// its count at the current minimum, which leaves the minimum unchanged,
/// so they can run in any order.
pub fn approx_node_connectivity(
    out: &Csr,
    back: &Csr,
    n: usize,
    directed: bool,
    v: u32,
    min_degree: usize,
) -> usize {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let mut is_nbr = vec![false; n];
    let mut nbrs: Vec<u32> = Vec::new();
    let both: Vec<u32> = if directed {
        back.neighbors(v as usize)
            .iter()
            .chain(out.neighbors(v as usize))
            .copied()
            .collect()
    } else {
        out.neighbors(v as usize).to_vec()
    };
    for &w in &both {
        if !is_nbr[w as usize] {
            is_nbr[w as usize] = true;
            nbrs.push(w);
        }
    }
    let mut pairs: Vec<(u32, u32)> = (0..n as u32)
        .filter(|&w| w != v && !is_nbr[w as usize])
        .map(|w| (v, w))
        .collect();
    let mut row_mark = vec![u32::MAX; n];
    for (i, &x) in nbrs.iter().enumerate() {
        for &y in out.neighbors(x as usize) {
            row_mark[y as usize] = x;
        }
        let ys: &[u32] = if directed { &nbrs } else { &nbrs[i + 1..] };
        for &y in ys {
            if y != x && row_mark[y as usize] != x {
                pairs.push((x, y));
            }
        }
    }
    let k = AtomicUsize::new(min_degree);
    pairs.par_iter().for_each_init(
        || ApproxPaths::new(n),
        |scratch, &(s, t)| {
            let cutoff = k.load(Ordering::Relaxed);
            if cutoff > 0 {
                let found = scratch.local(out, back, s, t, cutoff);
                k.fetch_min(found, Ordering::Relaxed);
            }
        },
    );
    k.into_inner()
}

/// The edges `build_auxiliary_edge_connectivity` adds, first occurrences
/// only (repeats find the edge already there): `G.edges()` in order, and
/// for undirected graphs `(u, v)` then `(v, u)` for each.
pub fn auxiliary_edge_pairs(succ: &Csr, n: usize, directed: bool) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for u in 0..n {
        for &v in succ.neighbors(u) {
            if directed {
                out.push((u as u32, v));
            } else if v as usize >= u {
                out.push((u as u32, v));
                if v as usize != u {
                    out.push((v, u as u32));
                }
            }
        }
    }
    out
}

/// `find_asteroidal_triple` on an undirected simple graph of 6 or more
/// nodes, visiting the non-edges as `nx.non_edges` does, or (`tuple_set`,
/// NetworkX 3.4) in the order of a set of edge tuples.
pub fn asteroidal_triple(
    adj: &Csr,
    n: usize,
    hashes: &[i64],
    tuple_set: bool,
) -> Option<(u32, u32, u32)> {
    let pairs = if tuple_set {
        // 3.4: `set(nx.complement(G).edges)`, each edge `(u, v)` with `v`
        // after `u`, iterated in the set's order (tuple hashes).
        let mut mark = vec![usize::MAX; n];
        let mut edges = Vec::new();
        for u in 0..n {
            for &v in adj.neighbors(u) {
                mark[v as usize] = u;
            }
            for v in u + 1..n {
                if mark[v] != u {
                    edges.push((u as u32, v as u32));
                }
            }
        }
        let tuple_hashes: Vec<i64> = edges
            .iter()
            .map(|&(u, v)| tuple_hash(&[hashes[u as usize], hashes[v as usize]]))
            .collect();
        let set = PySet::from_iter(0..edges.len() as u32, &tuple_hashes);
        set.iter().map(|e| edges[e as usize]).collect()
    } else {
        // non_edges: u = nodes.pop(), then nodes - set(G[u])
        let mut out = Vec::new();
        let mut nodes = PySet::from_iter(0..n as u32, hashes);
        while let Some(u) = nodes.pop() {
            let row = PySet::from_iter(adj.neighbors(u as usize).iter().copied(), hashes);
            out.extend(nodes.difference(&row, hashes).iter().map(|v| (u, v)));
        }
        out
    };
    let all = PySet::from_iter(0..n as u32, hashes);
    // component_structure[x]: components of G minus x's closed
    // neighborhood; only equality between labels matters.
    let mut labels: Vec<Option<Vec<u32>>> = vec![None; n];
    let mut queue = Vec::new();
    let mut label_of = |x: u32, labels: &mut Vec<Option<Vec<u32>>>| {
        if labels[x as usize].is_none() {
            let mut lab = vec![u32::MAX; n];
            lab[x as usize] = 0;
            for &y in adj.neighbors(x as usize) {
                lab[y as usize] = 0;
            }
            let mut next = 1;
            for s in 0..n {
                if lab[s] != u32::MAX {
                    continue;
                }
                lab[s] = next;
                queue.clear();
                queue.push(s as u32);
                while let Some(v) = queue.pop() {
                    for &w in adj.neighbors(v as usize) {
                        if lab[w as usize] == u32::MAX {
                            lab[w as usize] = next;
                            queue.push(w);
                        }
                    }
                }
                next += 1;
            }
            labels[x as usize] = Some(lab);
        }
    };
    for (u, v) in pairs {
        let closed = |x: u32| {
            let mut s =
                PySet::from_iter(adj.neighbors(x as usize).iter().copied(), hashes).copy(hashes);
            s.add(x, hashes);
            s
        };
        let union = closed(u).union(&closed(v), hashes);
        for w in all.difference(&union, hashes).iter() {
            for x in [u, v, w] {
                label_of(x, &mut labels);
            }
            let lab = |x: u32, y: u32| labels[x as usize].as_ref().expect("computed")[y as usize];
            if lab(u, v) == lab(u, w) && lab(v, u) == lab(v, w) && lab(w, u) == lab(w, v) {
                return Some((u, v, w));
            }
        }
    }
    None
}

/// `k_clique_communities` from `cliques` (the maximal cliques of size `k`
/// or more, in `find_cliques`' order) and each clique's `frozenset` hash:
/// for each community, the cliques it joins in the iteration order of
/// `connected_components`' set (`frozenset.union` takes them in that order).
pub fn k_clique_communities(
    cliques: &[Vec<u32>],
    clique_hashes: &[i64],
    hashes: &[i64],
    n: usize,
    k: usize,
) -> Vec<Vec<u32>> {
    let count = cliques.len();
    let mut member: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (i, c) in cliques.iter().enumerate() {
        for &v in c {
            member[v as usize].push(i as u32);
        }
    }
    let mut in_clique = vec![u32::MAX; n];
    let mut rows: Vec<Vec<u32>> = vec![Vec::new(); count];
    let mut edges = std::collections::HashSet::new();
    for (i, c) in cliques.iter().enumerate() {
        // _get_adjacent_cliques: walk frozenset(c) in its order
        let fs = PySet::from_iter(c.iter().copied(), hashes);
        let mut adjacent = PySet::default();
        for v in fs.iter() {
            for &j in &member[v as usize] {
                if j as usize != i {
                    adjacent.add(j, clique_hashes);
                }
            }
        }
        for &v in c {
            in_clique[v as usize] = i as u32;
        }
        for j in adjacent.iter() {
            let common = cliques[j as usize]
                .iter()
                .filter(|&&v| in_clique[v as usize] == i as u32)
                .count();
            if common + 1 >= k {
                let key = (i.min(j as usize), i.max(j as usize));
                if edges.insert(key) {
                    rows[i].push(j);
                    rows[j as usize].push(i as u32);
                }
            }
        }
    }
    // connected_components(perc_graph): sets built in BFS order
    let mut seen = vec![false; count];
    let mut out = Vec::new();
    for s in 0..count {
        if seen[s] {
            continue;
        }
        seen[s] = true;
        let mut comp = vec![s as u32];
        let mut at = 0;
        while at < comp.len() {
            let v = comp[at] as usize;
            at += 1;
            for &w in &rows[v] {
                if !seen[w as usize] {
                    seen[w as usize] = true;
                    comp.push(w);
                }
            }
        }
        out.push(PySet::from_iter(comp, clique_hashes).iter().collect());
    }
    out
}
