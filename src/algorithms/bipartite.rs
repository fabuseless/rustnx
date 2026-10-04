//! Bipartite graph algorithms, ported step for step from NetworkX's
//! `networkx.algorithms.bipartite` so that dict orders, matchings and
//! counts come out exactly as NetworkX computes them.

use rayon::prelude::*;

use crate::graph::Csr;

const NONE: u32 = u32::MAX;
const INF: u64 = u64::MAX;

/// The graph has an edge inside one color class.
pub struct NotBipartite;

/// `nx.bipartite.color`: nodes in dict order with their colors. Starts a
/// search (a stack) at each uncolored node with successors, in node order;
/// directed graphs visit predecessors (`pred`, in NetworkX's order) and
/// then successors. Isolated nodes (`degree` 0) come last, with color 0.
pub fn color(
    succ: &Csr,
    pred: Option<&Csr>,
    n: usize,
    degree: &[usize],
) -> Result<(Vec<u32>, Vec<u8>), NotBipartite> {
    let mut colors = vec![u8::MAX; n];
    let mut order = Vec::with_capacity(n);
    let mut stack = Vec::new();
    for start in 0..n {
        if colors[start] != u8::MAX || succ.neighbors(start).is_empty() {
            continue;
        }
        colors[start] = 1;
        order.push(start as u32);
        stack.push(start);
        while let Some(v) = stack.pop() {
            let c = 1 - colors[v];
            let before = pred.map_or(&[][..], |p| p.neighbors(v));
            for &w in before.iter().chain(succ.neighbors(v)) {
                let w = w as usize;
                if colors[w] != u8::MAX {
                    if colors[w] == colors[v] {
                        return Err(NotBipartite);
                    }
                } else {
                    colors[w] = c;
                    order.push(w as u32);
                    stack.push(w);
                }
            }
        }
    }
    for v in 0..n {
        if degree[v] == 0 && colors[v] == u8::MAX {
            colors[v] = 0;
            order.push(v as u32);
        }
    }
    let values = order.iter().map(|&v| colors[v as usize]).collect();
    Ok((order, values))
}

/// `is_bipartite_node_set` after its duplicate check: for each component
/// (in `components` order), `Err` if it isn't bipartite, `Ok(false)` if
/// `in_set` doesn't select exactly one side of it.
pub fn is_node_set(
    adj: &Csr,
    n: usize,
    components: &[Vec<u32>],
    in_set: &[bool],
) -> Result<bool, NotBipartite> {
    let mut side = vec![u8::MAX; n];
    for comp in components {
        let start = comp[0] as usize;
        side[start] = 0;
        let mut stack = vec![start];
        while let Some(v) = stack.pop() {
            for &w in adj.neighbors(v) {
                let w = w as usize;
                if side[w] == u8::MAX {
                    side[w] = 1 - side[v];
                    stack.push(w);
                } else if side[w] == side[v] {
                    return Err(NotBipartite);
                }
            }
        }
        // Side 0 inside the set and side 1 outside, or the other way round.
        let first = comp
            .iter()
            .all(|&v| in_set[v as usize] == (side[v as usize] == 0));
        let second = comp
            .iter()
            .all(|&v| in_set[v as usize] == (side[v as usize] == 1));
        if !(first || second) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// NetworkX's `hopcroft_karp_matching` on positions: `left` in the
/// iteration order of NetworkX's `left` set. Returns each node's match
/// (`NONE` if unmatched) and the deepest recursion its
/// `depth_first_search` would reach. Every neighbor of a left node must
/// be a right node.
pub fn hopcroft_karp(adj: &Csr, n: usize, left: &[u32]) -> (Vec<u32>, usize) {
    let mut mate = vec![NONE; n];
    let mut dist = vec![INF; n];
    let mut max_depth = 0usize;
    // Distances of matched right nodes' partners; `None` (a free right
    // node) has its own distance, as NetworkX's `distances[None]`.
    loop {
        // breadth_first_search
        let mut queue = std::collections::VecDeque::new();
        for &v in left {
            let v = v as usize;
            if mate[v] == NONE {
                dist[v] = 0;
                queue.push_back(v as u32);
            } else {
                dist[v] = INF;
            }
        }
        let mut dist_none = INF;
        while let Some(v) = queue.pop_front() {
            if v == NONE {
                continue; // `distances[None] < distances[None]` is false
            }
            let v = v as usize;
            if dist[v] < dist_none {
                for &u in adj.neighbors(v) {
                    let r = mate[u as usize];
                    let d = if r == NONE {
                        dist_none
                    } else {
                        dist[r as usize]
                    };
                    if d == INF {
                        if r == NONE {
                            dist_none = dist[v] + 1;
                        } else {
                            dist[r as usize] = dist[v] + 1;
                        }
                        queue.push_back(r);
                    }
                }
            }
        }
        if dist_none == INF {
            break;
        }
        for &root in left {
            if mate[root as usize] != NONE {
                continue;
            }
            // depth_first_search(root), iteratively: frames are (node, next
            // neighbor index); `result` is a finished child call's answer.
            let mut stack: Vec<(usize, usize)> = vec![(root as usize, adj.offsets[root as usize])];
            let mut result: Option<bool> = None;
            while let Some(&mut (v, ref mut i)) = stack.last_mut() {
                if let Some(found) = result.take() {
                    if found {
                        let u = adj.targets[*i - 1] as usize;
                        mate[u] = v as u32;
                        mate[v] = u as u32;
                        stack.pop();
                        result = Some(true);
                        continue;
                    }
                }
                let end = adj.offsets[v + 1];
                let mut descend = None;
                while *i < end {
                    let u = adj.targets[*i] as usize;
                    *i += 1;
                    let r = mate[u];
                    let d = if r == NONE {
                        dist_none
                    } else {
                        dist[r as usize]
                    };
                    if d == dist[v].saturating_add(1) {
                        descend = Some(r);
                        break;
                    }
                }
                match descend {
                    Some(r) => {
                        // A call for `r` (or for `None`, which returns True).
                        max_depth = max_depth.max(stack.len() + 1);
                        if r == NONE {
                            result = Some(true);
                        } else {
                            stack.push((r as usize, adj.offsets[r as usize]));
                        }
                    }
                    None => {
                        dist[v] = INF;
                        stack.pop();
                        result = Some(false);
                    }
                }
            }
        }
    }
    (mate, max_depth)
}

/// For `to_vertex_cover`: whether NetworkX's `_is_connected_by_alternating_path`
/// finds a target from `v`, for each node not in `targets`. `matched(e)`
/// says whether `succ` entry `e` is a matched edge; `unmatched(e)` whether
/// it is an unmatched one (a self-loop can be neither).
pub fn alternating_reach(
    adj: &Csr,
    n: usize,
    targets: &[bool],
    matched: &[bool],
    unmatched: &[bool],
) -> Vec<bool> {
    (0..n)
        .into_par_iter()
        .map_init(
            || vec![0u32; n],
            |seen, v| {
                if targets[v] {
                    return true;
                }
                let mut stamp = 2 * v as u32;
                [true, false].iter().any(|&along_matched| {
                    stamp += 1;
                    alternating_dfs(
                        adj,
                        v,
                        along_matched,
                        targets,
                        matched,
                        unmatched,
                        seen,
                        stamp,
                    )
                })
            },
        )
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn alternating_dfs(
    adj: &Csr,
    start: usize,
    along_matched: bool,
    targets: &[bool],
    matched: &[bool],
    unmatched: &[bool],
    seen: &mut [u32],
    stamp: u32,
) -> bool {
    // `visited` starts empty (the start node isn't in it); `seen[x] ==
    // stamp` marks membership for this call.
    let mut stack: Vec<(usize, usize, usize)> =
        vec![(start, adj.offsets[start], if along_matched { 0 } else { 1 })];
    while let Some(&mut (parent, ref mut i, depth)) = stack.last_mut() {
        if *i == adj.offsets[parent + 1] {
            stack.pop();
            continue;
        }
        let e = *i;
        *i += 1;
        let child = adj.targets[e] as usize;
        if seen[child] == stamp {
            continue;
        }
        let valid = if depth % 2 == 1 {
            matched[e]
        } else {
            unmatched[e]
        };
        if valid {
            if targets[child] {
                return true;
            }
            seen[child] = stamp;
            stack.push((child, adj.offsets[child], depth + 1));
        }
    }
    false
}

/// `_node_redundancy` for each of `nodes`: its number of neighbors, and
/// how many pairs of them share a neighbor other than the node.
pub fn redundancy_overlaps(adj: &Csr, n: usize, nodes: &[u32]) -> Vec<(u64, u64)> {
    nodes
        .par_iter()
        .map_init(
            || vec![u32::MAX; n],
            |mark, &v| {
                let nbrs = adj.neighbors(v as usize);
                let mut overlap = 0u64;
                for (i, &u) in nbrs.iter().enumerate() {
                    for &x in adj.neighbors(u as usize) {
                        if x != v {
                            mark[x as usize] = i as u32;
                        }
                    }
                    for &w in &nbrs[i + 1..] {
                        if adj
                            .neighbors(w as usize)
                            .iter()
                            .any(|&x| x != v && mark[x as usize] == i as u32)
                        {
                            overlap += 1;
                        }
                    }
                    for &x in adj.neighbors(u as usize) {
                        mark[x as usize] = u32::MAX;
                    }
                }
                (nbrs.len() as u64, overlap)
            },
        )
        .collect()
}

/// NetworkX 3.7's `butterflies` (undirected): per node, the butterflies
/// it is part of, with NetworkX's wedge counting by `(degree, index)`
/// priority.
pub fn butterflies(adj: &Csr, n: usize, degree: &[usize]) -> Vec<u64> {
    let priority = |v: usize| (degree[v], v);
    let sorted: Vec<Vec<u32>> = (0..n)
        .map(|v| {
            let mut row = adj.neighbors(v).to_vec();
            row.sort_by_key(|&w| priority(w as usize));
            row
        })
        .collect();
    let mut bt = vec![0u64; n];
    let mut count = vec![0u64; n];
    let mut mids: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut touched: Vec<u32> = Vec::new();
    for u in 0..n {
        let pu = priority(u);
        for &v in &sorted[u] {
            if priority(v as usize) >= pu {
                break;
            }
            for &w in &sorted[v as usize] {
                if priority(w as usize) >= pu {
                    break;
                }
                if count[w as usize] == 0 {
                    touched.push(w);
                }
                count[w as usize] += 1;
                mids[w as usize].push(v);
            }
        }
        for &w in &touched {
            let w = w as usize;
            let k = count[w];
            if k >= 2 {
                let bf = k * (k - 1) / 2;
                bt[u] += bf;
                bt[w] += bf;
                for &v in &mids[w] {
                    bt[v as usize] += k - 1;
                }
            }
            count[w] = 0;
            mids[w].clear();
        }
        touched.clear();
    }
    bt
}
