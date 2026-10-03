//! Batch 3: directed acyclic graphs, labeled traversals and components.
//!
//! Each function ports one NetworkX function (or its inner loop) and keeps
//! its visit order, so results come out in the same order as NetworkX's.

use std::collections::{BinaryHeap, HashSet, VecDeque};

use rayon::prelude::*;

use crate::graph::Csr;

/// Largest integer an f64 represents exactly (2^53).
const MAX_EXACT: f64 = 9_007_199_254_740_992.0;

/// `nx.dag_longest_path`'s main loop over `topo`. `pred` holds `G.pred` in
/// NetworkX's order (ties go to the first predecessor), with `w` aligned to
/// it (`None`: every edge weighs `constant`). Returns `None` if a distance
/// leaves the range where f64 sums of ints are exact.
pub fn longest_path(pred: &Csr, w: Option<&[f64]>, constant: f64, topo: &[u32]) -> Option<Vec<u32>> {
    let n = pred.offsets.len() - 1;
    let mut dist = vec![0.0f64; n];
    let mut from = vec![0u32; n];
    for &v in topo {
        let v = v as usize;
        let mut best: Option<(f64, u32)> = None;
        for e in pred.range(v) {
            let u = pred.targets[e];
            let d = dist[u as usize] + w.map_or(constant, |w| w[e]);
            if d.abs() >= MAX_EXACT {
                return None;
            }
            // Python's max() keeps the first maximal item.
            if best.is_none_or(|(b, _)| d > b) {
                best = Some((d, u));
            }
        }
        let (d, u) = best.unwrap_or((0.0, v as u32));
        if d >= 0.0 {
            dist[v] = d;
            from[v] = u;
        } else {
            dist[v] = 0.0;
            from[v] = v as u32;
        }
    }
    let mut path = Vec::new();
    let Some(&first) = topo.first() else {
        return Some(path);
    };
    let mut v = first;
    for &x in topo {
        if dist[x as usize] > dist[v as usize] {
            v = x;
        }
    }
    let mut u = u32::MAX;
    while u != v {
        path.push(v);
        u = v;
        v = from[v as usize];
    }
    path.reverse();
    Some(path)
}

/// `nx.lexicographical_topological_sort` with nodes ranked by their sort
/// key: the order nodes come out in, and whether a cycle stopped it.
pub fn lexicographical_topological_sort(succ: &Csr, pred: &Csr, rank: &[u32]) -> (Vec<u32>, bool) {
    let n = rank.len();
    let mut indegree: Vec<usize> = (0..n).map(|v| pred.neighbors(v).len()).collect();
    let mut heap: BinaryHeap<std::cmp::Reverse<(u32, u32)>> = (0..n)
        .filter(|&v| indegree[v] == 0)
        .map(|v| std::cmp::Reverse((rank[v], v as u32)))
        .collect();
    let mut order = Vec::with_capacity(n);
    while let Some(std::cmp::Reverse((_, v))) = heap.pop() {
        for &child in succ.neighbors(v as usize) {
            let d = &mut indegree[child as usize];
            *d -= 1;
            if *d == 0 {
                heap.push(std::cmp::Reverse((rank[child as usize], child)));
            }
        }
        order.push(v);
    }
    let cycle = order.len() < n;
    (order, cycle)
}

/// State of `nx.all_topological_sorts`, advanced one sort at a time.
pub struct AllTopologicalSorts {
    succ: Vec<Vec<u32>>,
    count: Vec<i64>,
    d: VecDeque<u32>,
    bases: Vec<u32>,
    current: Vec<u32>,
    /// A sort was just yielded: backtrack before searching again.
    pending: bool,
    done: bool,
}

pub struct HasCycle;

impl AllTopologicalSorts {
    pub fn new(succ: &Csr, pred: &Csr, n: usize) -> Self {
        let count: Vec<i64> = (0..n).map(|v| pred.neighbors(v).len() as i64).collect();
        let d = (0..n as u32).filter(|&v| count[v as usize] == 0).collect();
        AllTopologicalSorts {
            succ: (0..n).map(|v| succ.neighbors(v).to_vec()).collect(),
            count,
            d,
            bases: Vec::new(),
            current: Vec::new(),
            pending: false,
            done: false,
        }
    }

    pub fn next_sort(&mut self) -> Result<Option<Vec<u32>>, HasCycle> {
        if self.done {
            return Ok(None);
        }
        let n = self.count.len();
        if self.pending {
            self.pending = false;
            while let Some(q) = self.current.pop() {
                for &j in &self.succ[q as usize] {
                    self.count[j as usize] += 1;
                }
                while self.d.back().is_some_and(|&x| self.count[x as usize] > 0) {
                    self.d.pop_back();
                }
                self.d.push_front(q);
                if self.d.back() == self.bases.last() {
                    self.bases.pop();
                } else {
                    break;
                }
            }
            if self.bases.is_empty() {
                self.done = true;
                return Ok(None);
            }
        }
        loop {
            if self.current.len() == n {
                self.pending = true;
                return Ok(Some(self.current.clone()));
            }
            let Some(q) = self.d.pop_back() else {
                self.done = true;
                return Err(HasCycle);
            };
            for &j in &self.succ[q as usize] {
                let c = &mut self.count[j as usize];
                *c -= 1;
                if *c == 0 {
                    self.d.push_back(j);
                }
            }
            self.current.push(q);
            if self.bases.len() < self.current.len() {
                self.bases.push(q);
            }
            if self.bases.is_empty() {
                self.done = true;
                return Ok(None);
            }
        }
    }
}

/// Marks an arc `transitive_reduction` keeps.
pub const KEPT: u32 = u32::MAX;

/// `nx.transitive_reduction` on a DAG with topological positions `pos`.
/// For each arc `u -> w` of `succ`: `KEPT`, or the position in `u`'s row of
/// the successor `v` whose descendants removed `w`. NetworkX walks `G[u]`
/// and, for each `v` still in its candidate set, removes `v`'s descendants;
/// a set may be resized after each removal, so Python must repeat the same
/// removals in the same groups.
pub fn transitive_reduction(succ: &Csr, n: usize, pos: &[u32]) -> Vec<u32> {
    let rows: Vec<Vec<u32>> = (0..n)
        .into_par_iter()
        .map_init(
            || (vec![0u32; n], vec![u32::MAX; n], 0u32, Vec::new()),
            |(stamp, slot, mark, stack): &mut (Vec<u32>, Vec<u32>, u32, Vec<u32>), u| {
                let nbrs = succ.neighbors(u);
                let mut group = vec![KEPT; nbrs.len()];
                if nbrs.len() <= 1 {
                    return group;
                }
                for (k, &v) in nbrs.iter().enumerate() {
                    slot[v as usize] = k as u32;
                }
                // Nothing past the last successor (in topological order)
                // can be one of them.
                let limit = nbrs.iter().map(|&v| pos[v as usize]).max().unwrap_or(0);
                for (k, &v) in nbrs.iter().enumerate() {
                    if group[k] != KEPT {
                        continue; // already removed: NetworkX skips it
                    }
                    *mark += 1;
                    let m = *mark;
                    stack.clear();
                    stack.push(v);
                    while let Some(x) = stack.pop() {
                        for &y in succ.neighbors(x as usize) {
                            let yi = y as usize;
                            if stamp[yi] != m && pos[yi] <= limit {
                                stamp[yi] = m;
                                stack.push(y);
                                let j = slot[yi];
                                if j != u32::MAX && group[j as usize] == KEPT {
                                    group[j as usize] = k as u32;
                                }
                            }
                        }
                    }
                }
                for &v in nbrs {
                    slot[v as usize] = u32::MAX;
                }
                group
            },
        )
        .collect();
    rows.into_iter().flatten().collect()
}

/// For `nx.transitive_closure`: the heads `e[1]` of `nx.edge_bfs(G, v)` in
/// yield order, each once (`edge_bfs`), or the BFS discovery order from `v`
/// without `v` itself (`nx.descendants`' insertion order).
pub fn closure_heads(succ: &Csr, n: usize, directed: bool, v: usize, edge_bfs: bool) -> Vec<u32> {
    let mut out = Vec::new();
    let mut queued = vec![false; n];
    let mut queue = vec![v as u32];
    queued[v] = true;
    if !edge_bfs {
        let mut i = 0;
        while i < queue.len() {
            for &w in succ.neighbors(queue[i] as usize) {
                if !queued[w as usize] {
                    queued[w as usize] = true;
                    queue.push(w);
                    out.push(w);
                }
            }
            i += 1;
        }
        return out;
    }
    let mut head = vec![false; n];
    // Undirected: an edge is yielded from whichever end is popped first.
    let mut popped = vec![false; n];
    let mut i = 0;
    while i < queue.len() {
        let u = queue[i] as usize;
        for &w in succ.neighbors(u) {
            if !queued[w as usize] {
                queued[w as usize] = true;
                queue.push(w);
            }
            if !directed && popped[w as usize] {
                continue; // edge already yielded from w
            }
            if !head[w as usize] {
                head[w as usize] = true;
                out.push(w);
            }
        }
        popped[u] = true;
        i += 1;
    }
    out
}

/// `nx.transitive_closure_dag`'s closure built so far.
pub struct ClosureDag {
    pub rows: Vec<Vec<u32>>,
    stamp: Vec<u32>,
    mark: u32,
}

impl ClosureDag {
    pub fn new(succ: &Csr, n: usize) -> Self {
        ClosureDag {
            rows: (0..n).map(|v| succ.neighbors(v).to_vec()).collect(),
            stamp: vec![0; n],
            mark: 0,
        }
    }

    /// The third layer of `nx.bfs_layers(TC, v)` (distance 2), in order.
    pub fn layer2(&mut self, v: usize) -> Vec<u32> {
        self.mark += 1;
        let mark = self.mark;
        self.stamp[v] = mark;
        for &w in &self.rows[v] {
            self.stamp[w as usize] = mark;
        }
        let mut out = Vec::new();
        for &w in &self.rows[v] {
            for &x in &self.rows[w as usize] {
                if self.stamp[x as usize] != mark {
                    self.stamp[x as usize] = mark;
                    out.push(x);
                }
            }
        }
        out
    }
}

/// `nx.dag.root_to_leaf_paths`: `nx.all_simple_paths(G, root, leaf)` for
/// each (root, leaf) pair in order, one path at a time. Branches that can't
/// reach the leaf are skipped; NetworkX explores them but yields nothing.
pub struct RootLeafPaths {
    succ: Vec<Vec<u32>>,
    pred: Vec<Vec<u32>>,
    roots: Vec<u32>,
    leaves: Vec<u32>,
    ri: usize,
    li: usize,
    from_root: Vec<bool>,
    to_leaf: Vec<bool>,
    on_path: Vec<bool>,
    path: Vec<u32>,
    /// Next neighbor position for each node on `path`.
    pos: Vec<usize>,
    /// A DFS for pair (ri, li) is running.
    active: bool,
}

impl RootLeafPaths {
    pub fn new(succ: &Csr, pred: &Csr, n: usize) -> Self {
        let roots = (0..n as u32).filter(|&v| pred.neighbors(v as usize).is_empty()).collect();
        let leaves = (0..n as u32).filter(|&v| succ.neighbors(v as usize).is_empty()).collect();
        RootLeafPaths {
            succ: (0..n).map(|v| succ.neighbors(v).to_vec()).collect(),
            pred: (0..n).map(|v| pred.neighbors(v).to_vec()).collect(),
            roots,
            leaves,
            ri: 0,
            li: 0,
            from_root: Vec::new(),
            to_leaf: vec![false; n],
            on_path: vec![false; n],
            path: Vec::new(),
            pos: Vec::new(),
            active: false,
        }
    }

    fn csr_reach(rows: &[Vec<u32>], s: usize, within: Option<&[bool]>, seen: &mut [bool]) {
        seen.iter_mut().for_each(|x| *x = false);
        seen[s] = true;
        let mut stack = vec![s as u32];
        while let Some(v) = stack.pop() {
            for &w in &rows[v as usize] {
                let w = w as usize;
                if !seen[w] && within.is_none_or(|ok| ok[w]) {
                    seen[w] = true;
                    stack.push(w as u32);
                }
            }
        }
    }

    pub fn next_path(&mut self) -> Option<Vec<u32>> {
        let n = self.succ.len();
        loop {
            if self.ri >= self.roots.len() || self.leaves.is_empty() {
                return None;
            }
            let root = self.roots[self.ri] as usize;
            let leaf = self.leaves[self.li] as usize;
            if !self.active {
                if self.li == 0 {
                    let mut seen = vec![false; n];
                    Self::csr_reach(&self.succ, root, None, &mut seen);
                    self.from_root = seen;
                }
                if root == leaf {
                    self.advance_pair();
                    return Some(vec![root as u32]);
                }
                if !self.from_root[leaf] {
                    self.advance_pair();
                    continue;
                }
                let mut to_leaf = std::mem::take(&mut self.to_leaf);
                Self::csr_reach(&self.pred, leaf, Some(&self.from_root), &mut to_leaf);
                self.to_leaf = to_leaf;
                self.path.clear();
                self.pos.clear();
                self.path.push(root as u32);
                self.pos.push(0);
                self.on_path[root] = true;
                self.active = true;
            }
            if let Some(found) = self.step(leaf) {
                return Some(found);
            }
            self.active = false;
            self.advance_pair();
        }
    }

    fn advance_pair(&mut self) {
        self.li += 1;
        if self.li == self.leaves.len() {
            self.li = 0;
            self.ri += 1;
        }
    }

    /// Continue the DFS to the next path ending at `leaf`; `None` when done.
    fn step(&mut self, leaf: usize) -> Option<Vec<u32>> {
        while let Some(&top) = self.path.last() {
            let depth = self.path.len() - 1;
            let row = &self.succ[top as usize];
            let mut next = None;
            while self.pos[depth] < row.len() {
                let w = row[self.pos[depth]] as usize;
                self.pos[depth] += 1;
                if !self.on_path[w] {
                    next = Some(w);
                    break;
                }
            }
            match next {
                None => {
                    self.on_path[top as usize] = false;
                    self.path.pop();
                    self.pos.pop();
                }
                Some(w) if w == leaf => {
                    let mut found = self.path.clone();
                    found.push(w as u32);
                    return Some(found);
                }
                Some(w) => {
                    if self.to_leaf[w] {
                        self.on_path[w] = true;
                        self.path.push(w as u32);
                        self.pos.push(0);
                    }
                }
            }
        }
        None
    }
}

/// `nx.dag_to_branching`: the prefix tree of all root-to-leaf paths, as
/// each tree node's original node and parent (`u32::MAX` for none), in
/// `nx.prefix_tree`'s numbering (depth-first, children in first-seen order).
pub fn branching(paths: &mut RootLeafPaths) -> (Vec<u32>, Vec<u32>) {
    // Trie nodes: original node and children (in insertion order).
    let mut source: Vec<u32> = Vec::new();
    let mut children: Vec<Vec<u32>> = vec![Vec::new()]; // 0 = the root
    source.push(u32::MAX);
    while let Some(path) = paths.next_path() {
        let mut at = 0usize;
        for v in path {
            let found = children[at].iter().copied().find(|&c| source[c as usize] == v);
            at = match found {
                Some(c) => c as usize,
                None => {
                    let id = source.len();
                    source.push(v);
                    children.push(Vec::new());
                    children[at].push(id as u32);
                    id
                }
            };
        }
    }
    let mut out_source = Vec::with_capacity(source.len() - 1);
    let mut out_parent = Vec::with_capacity(source.len() - 1);
    // (trie node, its preorder number or MAX for the root, next child)
    let mut stack: Vec<(usize, u32, usize)> = vec![(0, u32::MAX, 0)];
    while let Some(top) = stack.last_mut() {
        let (node, number, next) = *top;
        if next == children[node].len() {
            stack.pop();
            continue;
        }
        top.2 += 1;
        let child = children[node][next] as usize;
        let id = out_source.len() as u32;
        out_source.push(source[child]);
        out_parent.push(number);
        stack.push((child, id, 0));
    }
    (out_source, out_parent)
}

/// `nx.dag.colliders` (or `v_structures`) from node `start` on, stopping
/// after the node where the output reaches `limit` triples: the triples
/// (flattened) and the next node to start from.
pub fn colliders(
    succ: &Csr,
    pred: &Csr,
    n: usize,
    start: usize,
    limit: usize,
    v_structures: bool,
) -> (Vec<u32>, usize) {
    let mut out = Vec::new();
    let mut stamp = vec![usize::MAX; if v_structures { n } else { 0 }];
    let mut node = start;
    while node < n && out.len() / 3 < limit {
        let preds = pred.neighbors(node);
        for (i, &p1) in preds.iter().enumerate() {
            if v_structures {
                let p = p1 as usize;
                for &x in succ.neighbors(p).iter().chain(pred.neighbors(p)) {
                    stamp[x as usize] = p;
                }
            }
            for &p2 in &preds[i + 1..] {
                if v_structures && stamp[p2 as usize] == p1 as usize {
                    continue;
                }
                out.extend_from_slice(&[p1, node as u32, p2]);
            }
        }
        node += 1;
    }
    (out, node)
}

/// NetworkX 3.4 to 3.6 `nx.is_aperiodic`'s BFS from `s`: how many nodes it
/// reached, and the gcd of `level[u] - level[v] + 1` over non-tree edges.
pub fn aperiodic_bfs(succ: &Csr, n: usize, s: usize) -> (usize, u64) {
    let mut level = vec![-1i64; n];
    level[s] = 0;
    let mut this_level = vec![s as u32];
    let mut g: u64 = 0;
    let mut reached = 1;
    let mut lev = 1;
    while !this_level.is_empty() {
        let mut next = Vec::new();
        for &u in &this_level {
            for &v in succ.neighbors(u as usize) {
                if level[v as usize] >= 0 {
                    let diff = level[u as usize] - level[v as usize] + 1;
                    g = gcd(g, diff.unsigned_abs());
                } else {
                    next.push(v);
                    level[v as usize] = lev;
                    reached += 1;
                }
            }
        }
        this_level = next;
        lev += 1;
    }
    (reached, g)
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

pub mod label {
    pub const FORWARD: u8 = 0;
    pub const NONTREE: u8 = 1;
    pub const REVERSE: u8 = 2;
    pub const REVERSE_DEPTH_LIMIT: u8 = 3;
    // bfs_labeled_edges
    pub const TREE: u8 = 4;
    pub const LEVEL: u8 = 5;
}

/// `nx.dfs_labeled_edges` from each of `starts` in turn (visited ones are
/// skipped): `(u, v, label)` in yield order.
pub fn dfs_labeled(adj: &Csr, n: usize, starts: &[u32], depth_limit: i64) -> Vec<(u32, u32, u8)> {
    let mut visited = vec![false; n];
    let mut out = Vec::new();
    let mut stack: Vec<(u32, usize)> = Vec::new();
    for &start in starts {
        if visited[start as usize] {
            continue;
        }
        out.push((start, start, label::FORWARD));
        visited[start as usize] = true;
        stack.push((start, adj.range(start as usize).start));
        let mut depth_now = 1i64;
        while let Some(&mut (parent, ref mut pos)) = stack.last_mut() {
            let end = adj.range(parent as usize).end;
            let mut descended = None;
            while *pos < end {
                let child = adj.targets[*pos];
                *pos += 1;
                if visited[child as usize] {
                    out.push((parent, child, label::NONTREE));
                } else {
                    out.push((parent, child, label::FORWARD));
                    visited[child as usize] = true;
                    if depth_now < depth_limit {
                        descended = Some(child);
                        break;
                    }
                    out.push((parent, child, label::REVERSE_DEPTH_LIMIT));
                }
            }
            match descended {
                Some(child) => {
                    stack.push((child, adj.range(child as usize).start));
                    depth_now += 1;
                }
                None => {
                    stack.pop();
                    depth_now -= 1;
                    if let Some(&(grand, _)) = stack.last() {
                        out.push((grand, parent, label::REVERSE));
                    }
                }
            }
        }
        out.push((start, start, label::REVERSE));
    }
    out
}

/// Marks a source missing from the graph in `bfs_labeled`.
pub const MISSING: u32 = u32::MAX;

/// `nx.bfs_labeled_edges` from `sources` (distinct; `MISSING` for nodes
/// not in the graph, where NetworkX raises once it pops them): the labeled
/// edges, and the position in `sources` of the missing source reached.
pub fn bfs_labeled(adj: &Csr, n: usize, directed: bool, sources: &[u32]) -> (Vec<(u32, u32, u8)>, Option<usize>) {
    let mut depth = vec![-1i64; n];
    let mut queue: VecDeque<(u32, i64)> = VecDeque::new();
    for &s in sources {
        if s != MISSING {
            depth[s as usize] = 0;
        }
        queue.push_back((s, 0));
    }
    let mut visited = vec![false; n];
    let mut out = Vec::new();
    let mut popped = 0usize;
    while let Some((u, du)) = queue.pop_front() {
        if u == MISSING {
            return (out, Some(popped));
        }
        popped += 1;
        for &v in adj.neighbors(u as usize) {
            let dv = depth[v as usize];
            if dv < 0 {
                depth[v as usize] = du + 1;
                queue.push_back((v, du + 1));
                out.push((u, v, label::TREE));
            } else if du == dv {
                if !visited[v as usize] {
                    out.push((u, v, label::LEVEL));
                }
            } else if du < dv {
                out.push((u, v, label::FORWARD));
            } else if directed {
                out.push((u, v, label::REVERSE));
            }
        }
        if !directed {
            visited[u as usize] = true;
        }
    }
    (out, None)
}

/// For each arc of `pred` (row `v`, entry `u`), the position of arc
/// `u -> v` in `succ`.
pub fn pred_arc_ids(succ: &Csr, pred: &Csr, n: usize) -> Vec<u32> {
    let mut by_head: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
    for u in 0..n {
        for e in succ.range(u) {
            by_head[succ.targets[e] as usize].push((u as u32, e as u32));
        }
    }
    // Rows were filled in increasing `u`, so each is sorted.
    let mut ids = vec![0u32; pred.targets.len()];
    for v in 0..n {
        let row = &by_head[v];
        for e in pred.range(v) {
            let u = pred.targets[e];
            let k = row.binary_search_by_key(&u, |&(t, _)| t).expect("pred matches succ");
            ids[e] = row[k].1;
        }
    }
    ids
}

/// Arcs `edge_bfs`/`edge_dfs` take from each node: out-arcs, in-arcs or both.
pub struct EdgeSource<'a> {
    pub succ: &'a Csr,
    pub succ_id: &'a [u32],
    /// In-arcs in `G.pred` order with their edge ids (directed graphs).
    pub pred: Option<(&'a Csr, &'a [u32])>,
    pub out: bool,
    pub inward: bool,
    pub edges: usize,
}

impl EdgeSource<'_> {
    fn degree(&self, v: usize) -> usize {
        let mut d = 0;
        if self.out {
            d += self.succ.neighbors(v).len();
        }
        if self.inward {
            d += self.pred.map_or(0, |(p, _)| p.neighbors(v).len());
        }
        d
    }

    /// Arc `k` of node `v`: `(u, w, label, edge id, child)`.
    fn arc(&self, v: usize, k: usize) -> (u32, u32, u8, u32, u32) {
        let outs = if self.out { self.succ.neighbors(v).len() } else { 0 };
        if k < outs {
            let e = self.succ.range(v).start + k;
            let w = self.succ.targets[e];
            (v as u32, w, label::FORWARD, self.succ_id[e], w)
        } else {
            let (pred, ids) = self.pred.expect("in-arcs");
            let e = pred.range(v).start + (k - outs);
            let u = pred.targets[e];
            (u, v as u32, label::REVERSE, ids[e], u)
        }
    }
}

/// `nx.edge_bfs` from `starts`: `(u, v, label)` in yield order.
pub fn edge_bfs(src: &EdgeSource, n: usize, starts: &[u32]) -> Vec<(u32, u32, u8)> {
    let mut visited_nodes = vec![false; n];
    for &s in starts {
        visited_nodes[s as usize] = true;
    }
    let mut visited_edges = vec![false; src.edges];
    let mut queue: VecDeque<u32> = starts.iter().copied().collect();
    let mut out = Vec::new();
    while let Some(parent) = queue.pop_front() {
        let parent = parent as usize;
        for k in 0..src.degree(parent) {
            let (u, w, lab, id, child) = src.arc(parent, k);
            if !visited_nodes[child as usize] {
                visited_nodes[child as usize] = true;
                queue.push_back(child);
            }
            if !visited_edges[id as usize] {
                visited_edges[id as usize] = true;
                out.push((u, w, lab));
            }
        }
    }
    out
}

/// `nx.edge_dfs` from `starts`: `(u, v, label)` in yield order.
pub fn edge_dfs(src: &EdgeSource, n: usize, starts: &[u32]) -> Vec<(u32, u32, u8)> {
    let mut visited_edges = vec![false; src.edges];
    // Each node's arc iterator, created on its first visit, persists.
    let mut pos = vec![0usize; n];
    let mut out = Vec::new();
    for &start in starts {
        let mut stack = vec![start];
        while let Some(&current) = stack.last() {
            let cur = current as usize;
            if pos[cur] >= src.degree(cur) {
                stack.pop();
                continue;
            }
            let (u, w, lab, id, child) = src.arc(cur, pos[cur]);
            pos[cur] += 1;
            if !visited_edges[id as usize] {
                visited_edges[id as usize] = true;
                stack.push(child);
                out.push((u, w, lab));
            }
        }
    }
    out
}

/// `nx.kosaraju_strongly_connected_components`. `post` is the DFS postorder
/// of the reversed graph. `stack_order` selects NetworkX 3.7's version
/// (explicit stack, stopping once `post.len()` nodes are seen); otherwise
/// 3.4 to 3.6's (DFS preorder over nodes not yet seen).
pub fn kosaraju(succ: &Csr, n: usize, mut post: Vec<u32>, stack_order: bool) -> Vec<Vec<u32>> {
    let mut seen = vec![false; n];
    let mut nseen = 0usize;
    let limit = post.len();
    let mut comps = Vec::new();
    while let Some(r) = post.pop() {
        if stack_order && nseen >= limit {
            break;
        }
        if seen[r as usize] {
            continue;
        }
        let mut comp = vec![r];
        seen[r as usize] = true;
        nseen += 1;
        if stack_order {
            let mut stack = vec![r];
            while nseen < limit {
                let Some(v) = stack.pop() else { break };
                for &w in succ.neighbors(v as usize) {
                    if !seen[w as usize] {
                        seen[w as usize] = true;
                        nseen += 1;
                        comp.push(w);
                        stack.push(w);
                    }
                }
            }
        } else {
            // Preorder; nodes seen earlier are dead ends (nothing unseen is
            // reachable through them).
            let mut stack: Vec<(u32, usize)> = vec![(r, succ.range(r as usize).start)];
            while let Some(&mut (v, ref mut p)) = stack.last_mut() {
                let end = succ.range(v as usize).end;
                let mut next = None;
                while *p < end {
                    let w = succ.targets[*p];
                    *p += 1;
                    if !seen[w as usize] {
                        next = Some(w);
                        break;
                    }
                }
                match next {
                    Some(w) => {
                        seen[w as usize] = true;
                        nseen += 1;
                        comp.push(w);
                        stack.push((w, succ.range(w as usize).start));
                    }
                    None => {
                        stack.pop();
                    }
                }
            }
        }
        comps.push(comp);
    }
    comps
}

/// Component of each node, given components.
pub fn component_of(comps: &[Vec<u32>], n: usize) -> Vec<u32> {
    let mut comp = vec![0u32; n];
    for (i, c) in comps.iter().enumerate() {
        for &v in c {
            comp[v as usize] = i as u32;
        }
    }
    comp
}

/// `nx.condensation`'s edges: `(comp[u], comp[v])` over `G.edges()` in
/// order, skipping loops within a component, each pair once.
pub fn condensation_edges(succ: &Csr, n: usize, comp: &[u32]) -> Vec<(u32, u32)> {
    let mut seen: HashSet<u64> = HashSet::new();
    let mut out = Vec::new();
    for u in 0..n {
        let cu = comp[u];
        for &v in succ.neighbors(u) {
            let cv = comp[v as usize];
            if cu != cv && seen.insert(((cu as u64) << 32) | cv as u64) {
                out.push((cu, cv));
            }
        }
    }
    out
}

/// `nx.is_semiconnected` for a weakly connected graph: the condensation has
/// a Hamiltonian path. Tarjan's components come in reverse topological
/// order, so consecutive components must be joined by an edge.
pub fn is_semiconnected(succ: &Csr, n: usize, comps: &[Vec<u32>]) -> bool {
    let comp = component_of(comps, n);
    let k = comps.len();
    let mut linked = vec![false; k];
    for u in 0..n {
        let cu = comp[u] as usize;
        for &v in succ.neighbors(u) {
            let cv = comp[v as usize] as usize;
            if cu == cv + 1 {
                linked[cv] = true;
            }
        }
    }
    linked[..k.saturating_sub(1)].iter().all(|&x| x)
}
