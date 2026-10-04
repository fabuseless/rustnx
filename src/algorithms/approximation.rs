//! Batch 16: approximation algorithms and graph operations.
//!
//! Each function replays one NetworkX algorithm step by step on node
//! positions: the same visiting order, the same tie-breaking and (for
//! numbers) the same Python arithmetic, so the result matches exactly.

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashSet};

use crate::algorithms::spectral::py_sum;
use crate::algorithms::trees_more::Num;
use crate::graph::Csr;

/// Edges in `G.edges()` order: undirected edges once, from the endpoint
/// that comes first in node order.
fn edge_list(adj: &Csr, n: usize, directed: bool) -> Vec<(u32, u32)> {
    let mut edges = Vec::new();
    for u in 0..n {
        for &v in adj.neighbors(u) {
            if directed || v as usize >= u {
                edges.push((u as u32, v));
            }
        }
    }
    edges
}

/// `min_weighted_vertex_cover`: the local-ratio cover in insertion order,
/// or `None` if int costs leave `i64`. `cost` follows `G.nodes`.
pub fn local_ratio_cover(
    adj: &Csr,
    n: usize,
    directed: bool,
    mut cost: Vec<Num>,
) -> Option<Vec<u32>> {
    let mut in_cover = vec![false; n];
    let mut cover = Vec::new();
    for (u, v) in edge_list(adj, n, directed) {
        let (u, v) = (u as usize, v as usize);
        if in_cover[u] || in_cover[v] {
            continue;
        }
        // `cost[u] <= cost[v]` (False when either is NaN).
        if matches!(cost[u].cmp(cost[v]), Some(Ordering::Less | Ordering::Equal)) {
            in_cover[u] = true;
            cover.push(u as u32);
            cost[v] = cost[v].sub(cost[u])?;
        } else {
            in_cover[v] = true;
            cover.push(v as u32);
            cost[u] = cost[u].sub(cost[v])?;
        }
    }
    Some(cover)
}

/// Segment tree giving the leftmost smallest key, as Python's `min()` picks
/// over a dict in order. Removed entries (NaN) never win.
struct LeftmostMin {
    size: usize,
    key: Vec<f64>,
    best: Vec<u32>,
}

impl LeftmostMin {
    fn new(keys: &[f64]) -> Self {
        let size = keys.len().next_power_of_two().max(1);
        let mut key = vec![f64::NAN; size];
        key[..keys.len()].copy_from_slice(keys);
        let mut best = vec![0u32; 2 * size];
        for i in 0..size {
            best[size + i] = i as u32;
        }
        let mut tree = LeftmostMin { size, key, best };
        for i in (1..size).rev() {
            tree.best[i] = tree.pick(tree.best[2 * i], tree.best[2 * i + 1]);
        }
        tree
    }

    fn pick(&self, a: u32, b: u32) -> u32 {
        let (ka, kb) = (self.key[a as usize], self.key[b as usize]);
        // `b` wins only if strictly smaller; removed entries lose.
        if !kb.is_nan() && (ka.is_nan() || kb < ka) {
            b
        } else {
            a
        }
    }

    fn set(&mut self, i: usize, value: f64) {
        self.key[i] = value;
        let mut p = (self.size + i) / 2;
        while p >= 1 {
            self.best[p] = self.pick(self.best[2 * p], self.best[2 * p + 1]);
            p /= 2;
        }
    }

    fn top(&self) -> usize {
        if self.size == 1 {
            0
        } else {
            self.best[1] as usize
        }
    }
}

/// `min_weighted_dominating_set`: nodes in the order NetworkX adds them.
/// `weights` are the node weights (`None`: all 1). With `uncovered_rule`
/// (NetworkX 3.6+) a node's cost counts its uncovered closed neighbors;
/// before, it counted the closed neighbors outside the dominating set.
pub fn min_weighted_dominating(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    uncovered_rule: bool,
) -> Vec<u32> {
    // Closed neighborhoods, each node once.
    let closed: Vec<Vec<u32>> = (0..n)
        .map(|v| {
            let mut row = vec![v as u32];
            row.extend(adj.neighbors(v).iter().filter(|&&w| w as usize != v));
            row
        })
        .collect();
    let weight = |v: usize| weights.map_or(1.0, |w| w[v]);
    let key = |v: usize, count: usize| {
        if count == 0 {
            f64::INFINITY
        } else {
            weight(v) / count as f64
        }
    };
    let mut count: Vec<usize> = closed.iter().map(Vec::len).collect();
    let keys: Vec<f64> = (0..n).map(|v| key(v, count[v])).collect();
    let mut tree = LeftmostMin::new(&keys);
    let mut alive = vec![true; n];
    let mut uncovered = vec![true; n];
    let mut left = n;
    let mut chosen = Vec::new();
    while left > 0 {
        let d = tree.top();
        chosen.push(d as u32);
        alive[d] = false;
        tree.set(d, f64::NAN);
        let mut newly = Vec::new();
        for &x in &closed[d] {
            if uncovered[x as usize] {
                uncovered[x as usize] = false;
                left -= 1;
                newly.push(x);
            }
        }
        if uncovered_rule {
            for x in newly {
                for &y in &closed[x as usize] {
                    let y = y as usize;
                    if alive[y] {
                        count[y] -= 1;
                        tree.set(y, key(y, count[y]));
                    }
                }
            }
        } else {
            for &y in &closed[d][1..] {
                let y = y as usize;
                if alive[y] {
                    count[y] -= 1;
                    tree.set(y, key(y, count[y]));
                }
            }
        }
    }
    chosen
}

pub enum TspError {
    NotComplete,
    Tie,
}

/// Whether every node has all other nodes as neighbors (self-loops aside).
pub fn is_complete(adj: &Csr, n: usize) -> bool {
    (0..n).all(|v| {
        adj.neighbors(v)
            .iter()
            .filter(|&&w| w as usize != v)
            .count()
            + 1
            == n
    })
}

/// `greedy_tsp` from `source`. NetworkX picks each next node with `min()`
/// over a set, so equal weights would make the result depend on set order:
/// those report `Tie`.
pub fn greedy_tsp(adj: &Csr, w: &[f64], n: usize, source: usize) -> Result<Vec<u32>, TspError> {
    if !is_complete(adj, n) {
        return Err(TspError::NotComplete);
    }
    let mut visited = vec![false; n];
    let mut row = vec![f64::NAN; n];
    visited[source] = true;
    let mut cycle = vec![source as u32];
    let mut current = source;
    for _ in 1..n {
        for e in adj.range(current) {
            row[adj.targets[e] as usize] = w[e];
        }
        let mut best: Option<(usize, f64)> = None;
        let mut tie = false;
        for (v, &x) in row.iter().enumerate() {
            if visited[v] {
                continue;
            }
            match best {
                None => best = Some((v, x)),
                Some((_, b)) if x < b => {
                    best = Some((v, x));
                    tie = false;
                }
                Some((_, b)) if x == b => tie = true,
                _ => {}
            }
        }
        if tie {
            return Err(TspError::Tie);
        }
        let (next, _) = best.expect("unvisited nodes remain");
        visited[next] = true;
        cycle.push(next as u32);
        current = next;
    }
    cycle.push(source as u32);
    Ok(cycle)
}

/// A tour's cost as Python's `sum()` returns it.
pub enum Cost {
    Int(i128),
    Float(f64),
}

/// The cycle `simulated_annealing_tsp` and `threshold_accepting_tsp` keep
/// mutating: entries index the original `init_cycle` list.
pub struct Tour {
    n: usize,
    dist: Vec<f64>,
    node: Vec<u32>,
    tour: Vec<u32>,
    best: Vec<u32>,
    ints: bool,
    compensated: bool,
}

impl Tour {
    /// `node[k]`: the node position of `init_cycle[k]`.
    pub fn new(
        adj: &Csr,
        w: &[f64],
        n: usize,
        node: Vec<u32>,
        ints: bool,
        compensated: bool,
    ) -> Self {
        let mut dist = vec![f64::NAN; n * n];
        for u in 0..n {
            for e in adj.range(u) {
                dist[u * n + adj.targets[e] as usize] = w[e];
            }
        }
        let tour: Vec<u32> = (0..node.len() as u32).collect();
        Tour {
            n,
            dist,
            node,
            best: tour.clone(),
            tour,
            ints,
            compensated,
        }
    }

    /// `sum(G[u][v].get(weight, 1) for u, v in pairwise(cycle))`.
    pub fn cost(&self) -> Cost {
        let n = self.n;
        let steps = self.tour.windows(2).map(|p| {
            let (a, b) = (
                self.node[p[0] as usize] as usize,
                self.node[p[1] as usize] as usize,
            );
            self.dist[a * n + b]
        });
        if self.ints {
            Cost::Int(steps.map(|x| x as i128).sum())
        } else {
            Cost::Float(py_sum(steps, self.compensated))
        }
    }

    /// Apply `swap_two_nodes` (kind 0) or `move_one_node` (kind 1) for the
    /// sampled indices `a`, `b`.
    pub fn apply(&mut self, kind: u8, a: usize, b: usize) -> bool {
        let len = self.tour.len();
        if a >= len || b >= len {
            return false;
        }
        if kind == 0 {
            self.tour.swap(a, b);
        } else {
            let x = self.tour.remove(a);
            self.tour.insert(b.min(self.tour.len()), x);
        }
        true
    }

    pub fn save_best(&mut self) {
        self.best.clone_from(&self.tour);
    }

    pub fn best(&self) -> Vec<u32> {
        self.best.clone()
    }
}

/// The nodes `treewidth_decomp` eliminates with `min_fill_in_heuristic`, in
/// order. Each round NetworkX sorts the remaining nodes by degree (stably,
/// so ties keep node order) and takes the first with the smallest fill-in,
/// stopping at once on a fill-in of 0, and stops when the remaining graph
/// is complete.
pub fn min_fill_in_order(adj: &Csr, n: usize) -> Vec<u32> {
    let mut nbrs: Vec<Vec<u32>> = (0..n)
        .map(|v| {
            let mut row: Vec<u32> = adj
                .neighbors(v)
                .iter()
                .copied()
                .filter(|&w| w as usize != v)
                .collect();
            row.sort_unstable();
            row.dedup();
            row
        })
        .collect();
    let mut buckets: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); n.max(1)];
    for (v, row) in nbrs.iter().enumerate() {
        buckets[row.len()].insert(v as u32);
    }
    let mut mark = vec![0u32; n];
    let mut stamp = 0u32;
    let mut left = n;
    let mut order = Vec::new();
    while left > 0 {
        let min_degree = (0..n)
            .find(|&d| !buckets[d].is_empty())
            .expect("nodes remain");
        if min_degree == left - 1 {
            break;
        }
        // Fill-in counted twice (num_fill_in before NetworkX halves it).
        let mut best: Option<(u32, i64)> = None;
        'scan: for bucket in &buckets[min_degree..] {
            for &x in bucket {
                let row = &nbrs[x as usize];
                stamp += 1;
                for &y in row {
                    mark[y as usize] = stamp;
                }
                let bound = best.map_or(i64::MAX, |b| b.1);
                let mut fill = 0i64;
                for &y in row {
                    let common = nbrs[y as usize]
                        .iter()
                        .filter(|&&z| mark[z as usize] == stamp)
                        .count();
                    fill += (row.len() - common) as i64 - 1;
                    if fill >= bound {
                        break;
                    }
                }
                if fill < bound {
                    best = Some((x, fill));
                    if fill == 0 {
                        break 'scan;
                    }
                }
            }
        }
        let (x, _) = best.expect("a node was scanned");
        let x = x as usize;
        let row = std::mem::take(&mut nbrs[x]);
        // Connect the neighbors with each other.
        for &u in &row {
            stamp += 1;
            for &z in &nbrs[u as usize] {
                mark[z as usize] = stamp;
            }
            let old = nbrs[u as usize].len();
            for &v in &row {
                if v != u && mark[v as usize] != stamp {
                    nbrs[u as usize].push(v);
                }
            }
            let new = nbrs[u as usize].len();
            if new != old {
                buckets[old].remove(&u);
                buckets[new].insert(u);
            }
        }
        // Remove x.
        for &u in &row {
            let list = &mut nbrs[u as usize];
            let old = list.len();
            if let Some(i) = list.iter().position(|&z| z as usize == x) {
                list.swap_remove(i);
            }
            buckets[old].remove(&u);
            buckets[old - 1].insert(u);
        }
        buckets[row.len()].remove(&(x as u32));
        left -= 1;
        order.push(x as u32);
    }
    order
}

/// Reusable bidirectional BFS (NetworkX's `_bidirectional_pred_succ`) on a
/// graph with some edges removed.
pub struct BiBfs {
    pred_of: Vec<i64>,
    succ_of: Vec<i64>,
    touched: Vec<u32>,
}

const UNSET: i64 = -2;
const NONE: i64 = -1;

impl BiBfs {
    pub fn new(n: usize) -> Self {
        BiBfs {
            pred_of: vec![UNSET; n],
            succ_of: vec![UNSET; n],
            touched: Vec::new(),
        }
    }

    fn reset(&mut self) {
        for &v in &self.touched {
            self.pred_of[v as usize] = UNSET;
            self.succ_of[v as usize] = UNSET;
        }
        self.touched.clear();
    }

    /// Shortest path `s -> t` avoiding arcs for which `removed(a, b)` holds.
    pub fn path(
        &mut self,
        succ: &Csr,
        pred: &Csr,
        s: usize,
        t: usize,
        removed: impl Fn(u32, u32) -> bool,
    ) -> Option<Vec<u32>> {
        if s == t {
            return Some(vec![s as u32]);
        }
        self.reset();
        self.pred_of[s] = NONE;
        self.succ_of[t] = NONE;
        self.touched.push(s as u32);
        self.touched.push(t as u32);
        let mut forward = vec![s as u32];
        let mut reverse = vec![t as u32];
        let mut meet = None;
        'search: while !forward.is_empty() && !reverse.is_empty() {
            if forward.len() <= reverse.len() {
                let this_level = std::mem::take(&mut forward);
                for &v in &this_level {
                    for &w in succ.neighbors(v as usize) {
                        if removed(v, w) {
                            continue;
                        }
                        let wi = w as usize;
                        if self.pred_of[wi] == UNSET {
                            forward.push(w);
                            self.pred_of[wi] = v as i64;
                            self.touched.push(w);
                        }
                        if self.succ_of[wi] != UNSET {
                            meet = Some(wi);
                            break 'search;
                        }
                    }
                }
            } else {
                let this_level = std::mem::take(&mut reverse);
                for &v in &this_level {
                    for &w in pred.neighbors(v as usize) {
                        if removed(w, v) {
                            continue;
                        }
                        let wi = w as usize;
                        if self.succ_of[wi] == UNSET {
                            self.succ_of[wi] = v as i64;
                            reverse.push(w);
                            self.touched.push(w);
                        }
                        if self.pred_of[wi] != UNSET {
                            meet = Some(wi);
                            break 'search;
                        }
                    }
                }
            }
        }
        let w = meet?;
        let mut path = Vec::new();
        let mut cur = w as i64;
        while cur != NONE {
            path.push(cur as u32);
            cur = self.pred_of[cur as usize];
        }
        path.reverse();
        let mut cur = self.succ_of[*path.last().expect("non-empty") as usize];
        while cur != NONE {
            path.push(cur as u32);
            cur = self.succ_of[cur as usize];
        }
        Some(path)
    }
}

/// The edges `is_kl_connected` / `kl_connected_subgraph` reject, in
/// `G.edges()` order: for each edge, NetworkX removes one shortest `u-v`
/// path after another from a fresh copy of G, and accepts the edge once it
/// has counted `limit` paths (the edge itself first).
pub fn kl_rejected(
    succ: &Csr,
    pred: &Csr,
    n: usize,
    directed: bool,
    limit: u64,
    first_only: bool,
) -> Vec<(u32, u32)> {
    let mut bfs = BiBfs::new(n);
    let mut removed: HashSet<(u32, u32)> = HashSet::new();
    let key = |a: u32, b: u32| {
        if directed || a <= b {
            (a, b)
        } else {
            (b, a)
        }
    };
    let mut rejected = Vec::new();
    for (u, v) in edge_list(succ, n, directed) {
        if u == v {
            // The path [u] comes back forever, so the count reaches `limit`.
            continue;
        }
        removed.clear();
        let mut path = vec![u, v];
        let mut count = 0u64;
        let accepted = loop {
            count += 1;
            if count >= limit {
                break true;
            }
            for p in path.windows(2) {
                removed.insert(key(p[0], p[1]));
            }
            match bfs.path(succ, pred, u as usize, v as usize, |a, b| {
                removed.contains(&key(a, b))
            }) {
                Some(p) => path = p,
                None => break false,
            }
        };
        if !accepted {
            rejected.push((u, v));
            if first_only {
                break;
            }
        }
    }
    rejected
}

/// `complement`'s new edges, in the order NetworkX adds them (undirected
/// edges once, as adding the reverse pair changes nothing).
pub fn complement_edges(adj: &Csr, n: usize, directed: bool) -> (Vec<u32>, Vec<u32>) {
    let mut mark = vec![u32::MAX; n];
    let (mut us, mut vs) = (Vec::new(), Vec::new());
    for u in 0..n {
        for &v in adj.neighbors(u) {
            mark[v as usize] = u as u32;
        }
        let start = if directed { 0 } else { u + 1 };
        for (v, &m) in mark.iter().enumerate().skip(start) {
            if v != u && m != u as u32 {
                us.push(u as u32);
                vs.push(v as u32);
            }
        }
    }
    (us, vs)
}

/// `power(G, k)`'s edges in the order NetworkX adds them: for each node,
/// the nodes its level-by-level search reaches within `k` hops (in the
/// order of NetworkX's merged `nextlevel` dicts), skipping those already
/// joined from an earlier node.
pub fn power_edges(adj: &Csr, n: usize, k: usize) -> (Vec<u32>, Vec<u32>) {
    let mut seen = vec![usize::MAX; n];
    let mut queued = vec![(usize::MAX, 0usize); n];
    let (mut us, mut vs) = (Vec::new(), Vec::new());
    for s in 0..n {
        let mut this_level: Vec<u32> = adj.neighbors(s).to_vec();
        let mut level = 1usize;
        let mut reached = Vec::new();
        while !this_level.is_empty() {
            let mut next = Vec::new();
            for &v in &this_level {
                let vi = v as usize;
                if vi == s || seen[vi] == s {
                    continue;
                }
                seen[vi] = s;
                reached.push(v);
                for &w in adj.neighbors(vi) {
                    if queued[w as usize] != (s, level) {
                        queued[w as usize] = (s, level);
                        next.push(w);
                    }
                }
            }
            if k <= level {
                break;
            }
            level += 1;
            this_level = next;
        }
        for v in reached {
            if v as usize > s {
                us.push(s as u32);
                vs.push(v);
            }
        }
    }
    (us, vs)
}

/// Edges of `adj` (in `G.edges()` order) whose image under `map` is not an
/// edge of `other`, as `difference` adds them.
pub fn edges_missing_from(
    adj: &Csr,
    n: usize,
    directed: bool,
    other: &Csr,
    map: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let mut mark = vec![u32::MAX; other.offsets.len().saturating_sub(1)];
    let (mut us, mut vs) = (Vec::new(), Vec::new());
    for u in 0..n {
        let hu = map[u] as usize;
        for &w in other.neighbors(hu) {
            mark[w as usize] = u as u32;
        }
        for &v in adj.neighbors(u) {
            if (directed || v as usize >= u) && mark[map[v as usize] as usize] != u as u32 {
                us.push(u as u32);
                vs.push(v);
            }
        }
    }
    (us, vs)
}

/// One-exchange max-cut state: which side each node is on, and how much
/// the cut grows if it switches sides.
pub struct MaxCut {
    offsets: Vec<usize>,
    targets: Vec<u32>,
    w: Vec<i64>,
    side: Vec<bool>,
    gain: Vec<i128>,
    cut: i128,
}

impl MaxCut {
    pub fn new(adj: &Csr, w: Vec<i64>, side: Vec<bool>) -> Self {
        let n = side.len();
        let mut gain = vec![0i128; n];
        let mut cut = 0i128;
        for u in 0..n {
            for e in adj.range(u) {
                let v = adj.targets[e] as usize;
                if v == u {
                    continue;
                }
                let x = w[e] as i128;
                if side[u] == side[v] {
                    gain[u] += x;
                } else {
                    gain[u] -= x;
                    if v > u {
                        cut += x;
                    }
                }
            }
        }
        MaxCut {
            offsets: adj.offsets.clone(),
            targets: adj.targets.clone(),
            w,
            side,
            gain,
            cut,
        }
    }

    pub fn cut(&self) -> i128 {
        self.cut
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.side.len()
    }

    /// The first node in `order` whose switch gives the largest cut, and
    /// that cut.
    pub fn best(&self, order: &[u32]) -> Option<(u32, i128)> {
        let mut best: Option<(u32, i128)> = None;
        for &v in order {
            let value = self.cut + *self.gain.get(v as usize)?;
            if best.is_none_or(|(_, b)| value > b) {
                best = Some((v, value));
            }
        }
        best
    }

    pub fn switch(&mut self, v: usize) {
        let old = self.side[v];
        for e in self.offsets[v]..self.offsets[v + 1] {
            let u = self.targets[e] as usize;
            if u == v {
                continue;
            }
            let x = 2 * self.w[e] as i128;
            if self.side[u] == old {
                self.gain[u] -= x;
            } else {
                self.gain[u] += x;
            }
        }
        self.cut += self.gain[v];
        self.gain[v] = -self.gain[v];
        self.side[v] = !old;
    }
}

/// Total weight of the edges between the two sides (int weights).
pub fn cut_value(adj: &Csr, w: Option<&[f64]>, side: &[bool]) -> i128 {
    let mut total = 0i128;
    for u in 0..side.len() {
        for e in adj.range(u) {
            let v = adj.targets[e] as usize;
            if v > u && side[u] != side[v] {
                total += w.map_or(1, |w| w[e] as i64) as i128;
            }
        }
    }
    total
}

/// `approximate_diameter`'s two sweeps from `source`: `Err(())` if the
/// graph isn't (strongly) connected.
pub fn two_sweep(succ: &Csr, pred: Option<&Csr>, n: usize, source: usize) -> Result<u32, ()> {
    use crate::algorithms::traversal::bfs_lengths;
    let ecc = |adj: &Csr, v: usize| *bfs_lengths(adj, n, v, f64::INFINITY).1.last().unwrap_or(&0);
    let (forward, _) = bfs_lengths(succ, n, source, f64::INFINITY);
    match pred {
        None => {
            if forward.len() != n {
                return Err(());
            }
            Ok(ecc(succ, *forward.last().expect("source") as usize))
        }
        Some(pred) => {
            let (backward, _) = bfs_lengths(pred, n, source, f64::INFINITY);
            if forward.len() != n || backward.len() != n {
                return Err(());
            }
            let a1 = *forward.last().expect("source") as usize;
            let a2 = *backward.last().expect("source") as usize;
            Ok(ecc(pred, a1).max(ecc(succ, a2)))
        }
    }
}
