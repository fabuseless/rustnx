//! Batch 5: cores, clustering, distance and coloring.
//!
//! Each function follows the NetworkX code it stands for closely enough
//! that results (and the order they come in) are the same; the comments
//! say where an ordering or a counting rule comes from.

use std::cmp::Reverse;
use std::collections::BTreeSet;

use rayon::prelude::*;

use super::spectral::py_sum;
use super::traversal::{bfs_edges, bfs_lengths, dfs_forward, DijkstraState, NegativeCycle};
use crate::graph::Csr;

const NONE: u32 = u32::MAX;

/// `nx.onion_layers` of an undirected graph without self-loops, as
/// `(node, layer)` in NetworkX's dict order.
///
/// NetworkX re-sorts the remaining nodes by degree (a stable sort, so ties
/// keep node order) for every layer and takes the prefix with degree at most
/// the current core. A layer is therefore every remaining node with degree
/// at most the core, ordered by (degree, position); buckets find it without
/// re-sorting the whole graph.
pub fn onion_layers(adj: &Csr, n: usize) -> Vec<(u32, u32)> {
    let mut deg: Vec<usize> = (0..n).map(|v| adj.neighbors(v).len()).collect();
    let mut removed = vec![false; n];
    let mut out = Vec::with_capacity(n);
    let mut layer = 1u32;
    for v in 0..n {
        if deg[v] == 0 {
            removed[v] = true;
            out.push((v as u32, layer));
        }
    }
    if !out.is_empty() {
        layer = 2;
    }
    let mut remaining = n - out.len();
    let mut core = 1usize;
    let max_deg = deg.iter().copied().max().unwrap_or(0);
    // Nodes with degree above the core, by degree (stale entries are skipped).
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); max_deg + 2];
    // Remaining nodes with degree at most the core.
    let mut low: Vec<u32> = Vec::new();
    let mut in_low = vec![false; n];
    for v in 0..n {
        if removed[v] {
            continue;
        }
        if deg[v] <= core {
            low.push(v as u32);
            in_low[v] = true;
        } else {
            buckets[deg[v]].push(v as u32);
        }
    }
    let mut ptr = core + 1;
    while remaining > 0 {
        let mut this: Vec<u32> = if low.is_empty() {
            // Every remaining degree exceeds the core: the core rises to
            // the smallest of them.
            loop {
                let bucket = std::mem::take(&mut buckets[ptr]);
                let valid: Vec<u32> = bucket
                    .into_iter()
                    .filter(|&v| !removed[v as usize] && deg[v as usize] == ptr)
                    .collect();
                if !valid.is_empty() {
                    core = ptr;
                    break valid;
                }
                ptr += 1;
            }
        } else {
            std::mem::take(&mut low)
        };
        this.sort_unstable_by_key(|&v| (deg[v as usize], v));
        for &v in &this {
            in_low[v as usize] = false;
            removed[v as usize] = true;
            out.push((v, layer));
        }
        for &v in &this {
            for &w in adj.neighbors(v as usize) {
                let w = w as usize;
                if removed[w] {
                    continue;
                }
                deg[w] -= 1;
                if deg[w] <= core {
                    if !in_low[w] {
                        in_low[w] = true;
                        low.push(w as u32);
                    }
                } else {
                    buckets[deg[w]].push(w as u32);
                }
            }
        }
        remaining -= this.len();
        layer += 1;
    }
    out
}

/// Each node's neighbors sorted by position, with the undirected edge id of
/// each, and the edges as `(u, v)` with `u < v`.
#[allow(clippy::type_complexity)]
fn sorted_edges(adj: &Csr, n: usize) -> (Vec<Vec<(u32, u32)>>, Vec<(u32, u32)>) {
    let mut rows: Vec<Vec<(u32, u32)>> = (0..n)
        .map(|v| {
            let mut row: Vec<(u32, u32)> = adj
                .neighbors(v)
                .iter()
                .filter(|&&w| w as usize != v)
                .map(|&w| (w, NONE))
                .collect();
            row.sort_unstable();
            row
        })
        .collect();
    let mut edges = Vec::new();
    for u in 0..n {
        for k in 0..rows[u].len() {
            let v = rows[u][k].0;
            if (v as usize) > u {
                let id = edges.len() as u32;
                edges.push((u as u32, v));
                rows[u][k].1 = id;
                let j = rows[v as usize]
                    .binary_search_by_key(&(u as u32), |&(w, _)| w)
                    .expect("undirected adjacency is symmetric");
                rows[v as usize][j].1 = id;
            }
        }
    }
    (rows, edges)
}

/// Common neighbors of two sorted rows, as `(edge id to a, edge id to b)`.
fn common(a: &[(u32, u32)], b: &[(u32, u32)], mut visit: impl FnMut(u32, u32)) {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                visit(a[i].1, b[j].1);
                i += 1;
                j += 1;
            }
        }
    }
}

/// `nx.k_truss` of an undirected graph without self-loops: the edges to
/// drop (as `(u, v)`, `u < v`) and whether each node keeps an edge.
///
/// NetworkX drops, round after round, every edge in fewer than `need`
/// triangles of the current graph. That reaches the same fixed point as
/// peeling edges one at a time (an edge's support only goes down), so the
/// usual queue-based peeling gives the same graph.
pub fn k_truss(adj: &Csr, n: usize, need: u64) -> (Vec<(u32, u32)>, Vec<bool>) {
    let (rows, edges) = sorted_edges(adj, n);
    let mut support: Vec<u64> = edges
        .par_iter()
        .map(|&(u, v)| {
            let mut s = 0;
            common(&rows[u as usize], &rows[v as usize], |_, _| s += 1);
            s
        })
        .collect();
    let mut alive = vec![true; edges.len()];
    let mut queue: Vec<u32> = (0..edges.len() as u32)
        .filter(|&e| support[e as usize] < need)
        .collect();
    while let Some(e) = queue.pop() {
        let e = e as usize;
        if !alive[e] {
            continue;
        }
        alive[e] = false;
        let (u, v) = edges[e];
        common(&rows[u as usize], &rows[v as usize], |eu, ev| {
            let (eu, ev) = (eu as usize, ev as usize);
            if alive[eu] && alive[ev] {
                for f in [eu, ev] {
                    support[f] -= 1;
                    if support[f] + 1 == need {
                        queue.push(f as u32);
                    }
                }
            }
        });
    }
    let mut keep = vec![false; n];
    let mut dropped = Vec::new();
    for (e, &(u, v)) in edges.iter().enumerate() {
        if alive[e] {
            keep[u as usize] = true;
            keep[v as usize] = true;
        } else {
            dropped.push((u, v));
        }
    }
    (dropped, keep)
}

/// Nodes of `nx.k_corona`: core number `k`, and exactly `k` neighbors
/// (`G[v]`, successors for directed graphs) with core number at least `k`.
pub fn k_corona(adj: &Csr, core: &[u32], k: u32) -> Vec<u32> {
    (0..core.len())
        .filter(|&v| {
            core[v] == k
                && adj
                    .neighbors(v)
                    .iter()
                    .filter(|&&w| core[w as usize] >= k)
                    .count()
                    == k as usize
        })
        .map(|v| v as u32)
        .collect()
}

/// `square_clustering` numerators and denominators per node, as NetworkX
/// 3.4 counts them (`old`) or as 3.5+ does. All counts are integers; the
/// caller divides.
pub fn square_clustering(adj: &Csr, n: usize, nodes: &[u32], old: bool) -> Vec<(i64, i64)> {
    nodes
        .par_iter()
        .map_init(
            || (vec![0u32; n], 0u32, Vec::<u32>::new()),
            |(mark, stamp, list), &v| {
                if old {
                    square_old(adj, v as usize, mark, stamp)
                } else {
                    square_new(adj, v as usize, mark, stamp, list)
                }
            },
        )
        .collect()
}

fn next_stamp(mark: &mut [u32], stamp: &mut u32) -> u32 {
    if *stamp == u32::MAX {
        mark.fill(0);
        *stamp = 0;
    }
    *stamp += 1;
    *stamp
}

/// NetworkX 3.4: for each pair `u, w` of `combinations(G[v], 2)`,
/// `squares = len((set(G[u]) & set(G[w])) - {v})` and the potential terms.
fn square_old(adj: &Csr, v: usize, mark: &mut [u32], stamp: &mut u32) -> (i64, i64) {
    let nbrs = adj.neighbors(v);
    let (mut total, mut potential) = (0i64, 0i64);
    for (i, &u) in nbrs.iter().enumerate() {
        let s = next_stamp(mark, stamp);
        for &x in adj.neighbors(u as usize) {
            mark[x as usize] = s;
        }
        let deg_u = adj.neighbors(u as usize).len() as i64;
        for &w in &nbrs[i + 1..] {
            let squares = adj
                .neighbors(w as usize)
                .iter()
                .filter(|&&x| x as usize != v && mark[x as usize] == s)
                .count() as i64;
            total += squares;
            let mut degm = squares + 1;
            if mark[w as usize] == s {
                degm += 1;
            }
            let deg_w = adj.neighbors(w as usize).len() as i64;
            potential += (deg_u - degm) + (deg_w - degm) + squares;
        }
    }
    (total, potential)
}

/// NetworkX 3.5+: neighbor sets without self-loops, squares counted
/// through the neighbors and the two-hop neighbors of `v`.
fn square_new(
    adj: &Csr,
    v: usize,
    mark: &mut [u32],
    stamp: &mut u32,
    two_hop: &mut Vec<u32>,
) -> (i64, i64) {
    let nv: Vec<u32> = adj
        .neighbors(v)
        .iter()
        .copied()
        .filter(|&w| w as usize != v)
        .collect();
    let dv = nv.len() as i64;
    if dv - 1 <= 0 {
        return (0, 0);
    }
    // `in_nv` marks v's neighbors; `seen` marks two-hop nodes already listed.
    let in_nv = next_stamp(mark, stamp);
    let seen = next_stamp(mark, stamp);
    for &u in &nv {
        mark[u as usize] = in_nv;
    }
    let p2_of = |x: usize, mark: &[u32]| -> i64 {
        adj.neighbors(x)
            .iter()
            .filter(|&&y| y as usize != x && mark[y as usize] == in_nv)
            .count() as i64
    };
    let (mut uw_degrees, mut triangles, mut squares) = (0i64, 0i64, 0i64);
    for &u in &nv {
        let u = u as usize;
        let du = adj
            .neighbors(u)
            .iter()
            .filter(|&&x| x as usize != u)
            .count() as i64;
        uw_degrees += du * (dv - 1);
        let p2 = p2_of(u, mark);
        triangles += p2;
        squares += p2 * (p2 - 1);
    }
    two_hop.clear();
    for &u in &nv {
        for &x in adj.neighbors(u as usize) {
            let xi = x as usize;
            if x == u || xi == v || mark[xi] == in_nv || mark[xi] == seen {
                continue;
            }
            mark[xi] = seen;
            two_hop.push(x);
        }
    }
    for &x in two_hop.iter() {
        let x = x as usize;
        let p2 = adj
            .neighbors(x)
            .iter()
            .filter(|&&y| y as usize != x && mark[y as usize] == in_nv)
            .count() as i64;
        squares += p2 * (p2 - 1);
    }
    squares /= 2;
    let uw_count = dv * (dv - 1);
    (squares, uw_degrees - uw_count - triangles - squares)
}

/// `generalized_degree` counts: for each node `vs[i]`, whose neighbor set
/// (without itself) is `ws[ends[i-1]..ends[i]]` in Python's set order, the
/// number of those neighbors each one is adjacent to.
pub fn generalized_degree(adj: &Csr, n: usize, ws: &[u32], ends: &[u32]) -> Vec<u32> {
    let starts: Vec<usize> = std::iter::once(0)
        .chain(ends.iter().map(|&e| e as usize))
        .collect();
    let per_node: Vec<Vec<u32>> = (0..ends.len())
        .into_par_iter()
        .map_init(
            || (vec![0u32; n], 0u32),
            |(mark, stamp), i| {
                let group = &ws[starts[i]..starts[i + 1]];
                let s = next_stamp(mark, stamp);
                for &w in group {
                    mark[w as usize] = s;
                }
                group
                    .iter()
                    .map(|&w| {
                        adj.neighbors(w as usize)
                            .iter()
                            .filter(|&&x| x != w && mark[x as usize] == s)
                            .count() as u32
                    })
                    .collect()
            },
        )
        .collect();
    per_node.into_iter().flatten().collect()
}

/// `nx.all_triangles` (NetworkX 3.7) work, in yield order. For each pair
/// `u, v` with triangles, `(u, v, start, end)` indexes `ws` and `qualify`:
/// `ws` holds `v_nbrs & u_nbrs` in the order CPython builds that set (it
/// iterates the smaller keys view, `u`'s on a tie), and `qualify` says
/// which of them NetworkX yields. With more than one, the caller rebuilds
/// the set to get Python's iteration order.
#[allow(clippy::type_complexity)]
pub fn all_triangles(
    adj: &Csr,
    n: usize,
    nbunch: &[u32],
    ids: &[i64],
) -> (Vec<(u32, u32, u32, u32)>, Vec<u32>, Vec<bool>) {
    let mut sorted: Vec<Vec<u32>> = Vec::with_capacity(n);
    for v in 0..n {
        let mut row = adj.neighbors(v).to_vec();
        row.sort_unstable();
        sorted.push(row);
    }
    let per_u: Vec<(Vec<(u32, u32, u32, u32)>, Vec<u32>, Vec<bool>)> = nbunch
        .par_iter()
        .map(|&u| {
            let mut groups = Vec::new();
            let mut ws = Vec::new();
            let mut qualify = Vec::new();
            let u_id = ids[u as usize];
            for &v in adj.neighbors(u as usize) {
                let v_id = ids[v as usize];
                if v_id <= u_id {
                    continue;
                }
                let (iter_over, other) =
                    if adj.neighbors(u as usize).len() > adj.neighbors(v as usize).len() {
                        (v, u)
                    } else {
                        (u, v)
                    };
                let start = ws.len();
                let mut any = false;
                for &w in adj.neighbors(iter_over as usize) {
                    if sorted[other as usize].binary_search(&w).is_ok() {
                        let q = ids[w as usize] > v_id;
                        any |= q;
                        ws.push(w);
                        qualify.push(q);
                    }
                }
                if any {
                    groups.push((u, v, start as u32, ws.len() as u32));
                } else {
                    ws.truncate(start);
                    qualify.truncate(start);
                }
            }
            (groups, ws, qualify)
        })
        .collect();
    let mut groups = Vec::new();
    let mut ws = Vec::new();
    let mut qualify = Vec::new();
    for (g, w, q) in per_u {
        let base = ws.len() as u32;
        groups.extend(g.into_iter().map(|(u, v, s, e)| (u, v, s + base, e + base)));
        ws.extend(w);
        qualify.extend(q);
    }
    (groups, ws, qualify)
}

/// Node ids NetworkX's `all_triangles` gives a `nbunch`: its nodes in
/// order, then their other neighbors (a repeated neighbor keeps the id of
/// its last appearance, as in a dict comprehension). -1 for other nodes.
pub fn triangle_ids(adj: &Csr, n: usize, nbunch: &[u32]) -> Vec<i64> {
    let mut ids = vec![-1i64; n];
    let mut in_bunch = vec![false; n];
    for (i, &u) in nbunch.iter().enumerate() {
        ids[u as usize] = i as i64;
        in_bunch[u as usize] = true;
    }
    let mut next = nbunch.len() as i64;
    for &u in nbunch {
        for &w in adj.neighbors(u as usize) {
            if !in_bunch[w as usize] {
                ids[w as usize] = next;
                next += 1;
            }
        }
    }
    ids
}

/// Result of `intersection_array`: the arrays, or which error NetworkX
/// raises ("Graph is not distance regular." or the same without the period).
pub enum Intersection {
    Ok(Vec<u32>, Vec<u32>),
    NotRegular,
    Inconsistent,
}

/// `nx.intersection_array` of a non-empty undirected graph. `pairs_once`
/// selects NetworkX 3.7's version: it checks connectivity first, visits only
/// pairs `u <= v` and gives up once the diameter passes `bound`; older
/// releases visit every ordered pair.
pub fn intersection_array(
    adj: &Csr,
    n: usize,
    degree: &[usize],
    pairs_once: bool,
    bound: f64,
) -> Intersection {
    if degree.iter().any(|&d| d != degree[0]) {
        return Intersection::NotRegular;
    }
    if pairs_once && bfs_lengths(adj, n, 0, f64::INFINITY).0.len() != n {
        return Intersection::NotRegular;
    }
    let mut bint: Vec<Option<u32>> = vec![None; n + 1];
    let mut cint: Vec<Option<u32>> = vec![None; n + 1];
    let mut diam = 0u32;
    let chunk = 4 * rayon::current_num_threads().max(1);
    let sources: Vec<u32> = (0..n as u32).collect();
    for batch in sources.chunks(chunk) {
        let dists: Vec<Vec<u32>> = batch
            .par_iter()
            .map(|&u| {
                let mut dist = vec![NONE; n];
                let (order, levels) = bfs_lengths(adj, n, u as usize, f64::INFINITY);
                for (v, l) in order.into_iter().zip(levels) {
                    dist[v as usize] = l;
                }
                dist
            })
            .collect();
        for (&u, dist) in batch.iter().zip(&dists) {
            let first = if pairs_once { u as usize } else { 0 };
            for v in first..n {
                let i = dist[v];
                if i == NONE {
                    return Intersection::NotRegular; // KeyError in NetworkX 3.4
                }
                diam = diam.max(i);
                if pairs_once && diam as f64 > bound {
                    return Intersection::NotRegular;
                }
                let (mut c, mut b) = (0u32, 0u32);
                for &w in adj.neighbors(v) {
                    let d = dist[w as usize] as i64;
                    if d == i as i64 - 1 {
                        c += 1;
                    } else if d == i as i64 + 1 {
                        b += 1;
                    }
                }
                let i = i as usize;
                if cint[i].is_some_and(|x| x != c) || bint[i].is_some_and(|x| x != b) {
                    return Intersection::Inconsistent;
                }
                bint[i] = Some(b);
                cint[i] = Some(c);
            }
        }
    }
    let diam = diam as usize;
    Intersection::Ok(
        (0..diam).map(|j| bint[j].unwrap_or(0)).collect(),
        (0..diam).map(|j| cint[j + 1].unwrap_or(0)).collect(),
    )
}

/// `nx.tree.centroid` of a tree (NetworkX 3.7's `centroid` shortcut):
/// subtree sizes rooted at the first node, then walk to the heaviest child
/// (the first one, in neighbor order) while it holds more than half.
pub fn tree_centroid(adj: &Csr, n: usize) -> Vec<u32> {
    let (order, _) = bfs_lengths(adj, n, 0, f64::INFINITY);
    let mut parent = vec![NONE; n];
    let mut seen = vec![false; n];
    seen[0] = true;
    for &v in &order {
        for &w in adj.neighbors(v as usize) {
            if !seen[w as usize] {
                seen[w as usize] = true;
                parent[w as usize] = v;
            }
        }
    }
    let mut sizes = vec![1usize; n];
    for &v in order.iter().rev() {
        let p = parent[v as usize];
        if p != NONE {
            sizes[p as usize] += sizes[v as usize];
        }
    }
    let total = n;
    let heaviest = |prev: u32, root: u32| -> Option<u32> {
        let mut best: Option<u32> = None;
        for &x in adj.neighbors(root as usize) {
            if x != prev && best.is_none_or(|b| sizes[x as usize] > sizes[b as usize]) {
                best = Some(x);
            }
        }
        best
    };
    let (mut prev, mut root) = (NONE, 0u32);
    let mut hc = heaviest(prev, root);
    while 2 * (total - sizes[root as usize]).max(hc.map_or(0, |h| sizes[h as usize])) > total {
        prev = root;
        root = hc.expect("a heavier side exists");
        hc = heaviest(prev, root);
    }
    let mut out = vec![root];
    out.extend(
        adj.neighbors(root as usize)
            .iter()
            .copied()
            .filter(|&x| x != prev && 2 * sizes[x as usize] == total),
    );
    out
}

/// Distances from each source in NetworkX's dict order (BFS or Dijkstra
/// pop order), a batch of sources at a time, handed to `visit` in source
/// order. Stops at the first source that hits a negative cycle.
fn distances_in_order(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    mut visit: impl FnMut(usize, &[f64]),
) -> Result<(), NegativeCycle> {
    let chunk = 16 * rayon::current_num_threads().max(1);
    let sources: Vec<u32> = (0..n as u32).collect();
    for batch in sources.chunks(chunk) {
        let results: Vec<Result<Vec<f64>, NegativeCycle>> = batch
            .par_iter()
            .map_init(
                || DijkstraState::new(n),
                |state, &s| match weights {
                    None => {
                        let (_, levels) = bfs_lengths(adj, n, s as usize, f64::INFINITY);
                        Ok(levels.into_iter().map(f64::from).collect())
                    }
                    Some(w) => {
                        state.run(adj, Some(w), s as usize, None)?;
                        Ok(state
                            .order
                            .iter()
                            .map(|&v| state.dist[v as usize])
                            .collect())
                    }
                },
            )
            .collect();
        for (&s, r) in batch.iter().zip(results) {
            visit(s as usize, &r?);
        }
    }
    Ok(())
}

/// `harmonic_diameter`'s running sum of `1 / d` over every nonzero
/// distance, in NetworkX's order, and whether any term was added.
pub fn harmonic_sum(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
) -> Result<(f64, bool), NegativeCycle> {
    let mut sum = 0.0f64;
    let mut any = false;
    distances_in_order(adj, n, weights, |_, dists| {
        for &d in dists {
            if d != 0.0 {
                sum += 1.0 / d;
                any = true;
            }
        }
    })?;
    Ok((sum, any))
}

/// Per source, `(nodes reached, sum(dists.values()))` with Python's float
/// `sum()` (compensated from Python 3.12), for `centroid`.
pub fn distance_sums(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    compensated: bool,
) -> Result<Vec<(usize, f64)>, NegativeCycle> {
    let mut out = Vec::with_capacity(n);
    distances_in_order(adj, n, weights, |_, dists| {
        out.push((dists.len(), py_sum(dists.iter().copied(), compensated)));
    })?;
    Ok(out)
}

/// Colors of `greedy_color`'s loop: each node in `order` takes the smallest
/// color not used by an already colored node in `G[u]`.
pub fn greedy_with_order(succ: &Csr, n: usize, order: &[u32]) -> Vec<u32> {
    let mut color = vec![NONE; n];
    let mut used: Vec<u32> = Vec::new();
    for (stamp, &u) in order.iter().enumerate() {
        color[u as usize] = smallest_free(succ, u, &color, &mut used, stamp as u32 + 1);
    }
    color
}

fn smallest_free(succ: &Csr, u: u32, color: &[u32], used: &mut Vec<u32>, stamp: u32) -> u32 {
    for &v in succ.neighbors(u as usize) {
        let c = color[v as usize];
        if c != NONE {
            let c = c as usize;
            if c >= used.len() {
                used.resize(c + 1, 0);
            }
            used[c] = stamp;
        }
    }
    let mut c = 0;
    while c < used.len() && used[c] == stamp {
        c += 1;
    }
    c as u32
}

/// `greedy_color(strategy="saturation_largest_first")`: the next node is
/// the uncolored one with the most distinct colors among the colored nodes
/// pointing at it, then the largest degree, then the first in node order
/// (`max` keeps the first maximum). Returns the order and the colors.
pub fn dsatur(succ: &Csr, n: usize, degree: &[usize]) -> (Vec<u32>, Vec<u32>) {
    let mut color = vec![NONE; n];
    let mut seen_colors: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut queue: BTreeSet<(usize, usize, Reverse<u32>)> =
        (0..n).map(|v| (0, degree[v], Reverse(v as u32))).collect();
    let mut order = Vec::with_capacity(n);
    let mut used = Vec::new();
    while let Some((_, _, Reverse(u))) = queue.pop_last() {
        let c = smallest_free(succ, u, &color, &mut used, order.len() as u32 + 1);
        color[u as usize] = c;
        order.push(u);
        for &v in succ.neighbors(u as usize) {
            let vi = v as usize;
            if color[vi] != NONE {
                continue;
            }
            if let Err(pos) = seen_colors[vi].binary_search(&c) {
                let old = (seen_colors[vi].len(), degree[vi], Reverse(v));
                queue.remove(&old);
                seen_colors[vi].insert(pos, c);
                queue.insert((seen_colors[vi].len(), degree[vi], Reverse(v)));
            }
        }
    }
    (order, color)
}

/// `strategy_connected_sequential`: each component's source, then the ends
/// of `bfs_edges` or `dfs_edges` from it.
pub fn connected_sequential(adj: &Csr, n: usize, sources: &[u32], dfs: bool) -> Vec<u32> {
    let mut order = Vec::with_capacity(n);
    for &s in sources {
        order.push(s);
        if dfs {
            let edges = dfs_forward(adj, n, &[s], n as i64);
            order.extend(edges.into_iter().filter(|&(a, b)| a != b).map(|(_, b)| b));
        } else {
            order.extend(
                bfs_edges(adj, n, s as usize, n as i64)
                    .into_iter()
                    .map(|(_, b)| b),
            );
        }
    }
    order
}

/// `is_coloring` over the edges in `G.edges` order: `Ok(true)` if no edge
/// joins two nodes of the same color, `Ok(false)` at the first that does,
/// or `Err(node)` for the first node NetworkX would look up and not find
/// (`ids` is negative for nodes missing from the coloring).
pub fn is_coloring(succ: &Csr, n: usize, directed: bool, ids: &[i64]) -> Result<bool, u32> {
    for u in 0..n {
        for &v in succ.neighbors(u) {
            if !directed && (v as usize) < u {
                continue;
            }
            let (a, b) = (ids[u], ids[v as usize]);
            if a < 0 {
                return Err(u as u32);
            }
            if b < 0 {
                return Err(v);
            }
            if a == b {
                return Ok(false);
            }
        }
    }
    Ok(true)
}
