//! Batch 6: trees and structural tests (bridges, chains, Euler tours,
//! cycles, dominators, Prüfer sequences). Each function ports NetworkX's
//! code closely enough to visit nodes and edges in the same order.

use std::collections::{HashSet, VecDeque};
use std::hash::{BuildHasherDefault, Hasher};

use crate::graph::Csr;

const NONE: u32 = u32::MAX;

/// FxHash-style hasher for `u64` keys (pairs of node positions).
#[derive(Default)]
pub struct PairHasher(u64);

impl Hasher for PairHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
        }
    }

    fn write_u64(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

type PairSet = HashSet<u64, BuildHasherDefault<PairHasher>>;

#[inline]
fn pair(u: u32, v: u32) -> u64 {
    ((u as u64) << 32) | v as u64
}

/// Transposed adjacency with rows ordered by source position: the
/// successor order of NetworkX's `G.reverse()`, which adds the reversed
/// edges in `G.edges` order.
pub fn transpose(succ: &Csr, n: usize) -> Csr {
    let mut offsets = vec![0usize; n + 1];
    for &v in &succ.targets {
        offsets[v as usize + 1] += 1;
    }
    for i in 0..n {
        offsets[i + 1] += offsets[i];
    }
    let mut next = offsets.clone();
    let mut targets = vec![0u32; succ.targets.len()];
    for u in 0..n {
        for &v in succ.neighbors(u) {
            targets[next[v as usize]] = u as u32;
            next[v as usize] += 1;
        }
    }
    Csr { offsets, targets }
}

/// `nx.chain_decomposition`: the chains (as edge lists) in yield order,
/// and the nodes the depth-first search reached.
pub fn chain_decomposition(
    adj: &Csr,
    n: usize,
    root: Option<u32>,
) -> (Vec<Vec<(u32, u32)>>, Vec<bool>) {
    // `_dfs_cycle_forest`: DFS order, parents, and for each node the
    // nontree edges leaving it in the DFS cycle graph, in insertion order.
    let mut parent = vec![NONE; n];
    let mut seen = vec![false; n];
    let mut preorder = Vec::new();
    let mut nontree_out: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut added = PairSet::default();
    let starts: Vec<u32> = match root {
        Some(r) => vec![r],
        None => (0..n as u32).collect(),
    };
    let mut stack: Vec<(u32, usize)> = Vec::new();
    for s in starts {
        if seen[s as usize] {
            continue;
        }
        seen[s as usize] = true;
        preorder.push(s);
        stack.push((s, adj.range(s as usize).start));
        while let Some(&mut (u, ref mut pos)) = stack.last_mut() {
            if *pos == adj.range(u as usize).end {
                stack.pop();
                continue;
            }
            let v = adj.targets[*pos];
            *pos += 1;
            if seen[v as usize] {
                // A nontree (u, v): the tree edge to the parent is already
                // in H as u -> v, and each other edge is added (as v -> u)
                // only the first time it is met.
                if v != parent[u as usize] && added.insert(pair(u.min(v), u.max(v))) {
                    nontree_out[v as usize].push(u);
                }
            } else {
                seen[v as usize] = true;
                parent[v as usize] = u;
                preorder.push(v);
                stack.push((v, adj.range(v as usize).start));
            }
        }
    }
    let mut visited = vec![false; n];
    let mut chains = Vec::new();
    for &u in &preorder {
        visited[u as usize] = true;
        for &v in &nontree_out[u as usize] {
            let mut chain = Vec::new();
            let (mut a, mut b) = (u, v);
            while !visited[b as usize] {
                chain.push((a, b));
                visited[b as usize] = true;
                a = b;
                b = parent[b as usize];
            }
            chain.push((a, b));
            chains.push(chain);
        }
    }
    (chains, seen)
}

/// Edges `(u, v)` in `G.edges` order (undirected: from the earlier endpoint).
pub fn undirected_edges(adj: &Csr, n: usize) -> Vec<(u32, u32)> {
    let mut edges = Vec::new();
    for u in 0..n {
        for &v in adj.neighbors(u) {
            if v as usize >= u {
                edges.push((u as u32, v));
            }
        }
    }
    edges
}

/// `nx.bridges` (no root): edges in `G.edges` order that lie on no chain.
/// With `first_only`, stops at the first one (`has_bridges`); with a
/// root, only edges in its component count, for `has_bridges(G, root)`.
pub fn bridges(adj: &Csr, n: usize, root: Option<u32>, first_only: bool) -> Vec<(u32, u32)> {
    let (chains, reached) = chain_decomposition(adj, n, root);
    let mut on_chain = PairSet::default();
    for chain in &chains {
        for &(u, v) in chain {
            on_chain.insert(pair(u.min(v), u.max(v)));
        }
    }
    let mut out = Vec::new();
    for (u, v) in undirected_edges(adj, n) {
        if reached[u as usize] && !on_chain.contains(&pair(u.min(v), u.max(v))) {
            out.push((u, v));
            if first_only {
                break;
            }
        }
    }
    out
}

/// `nx.local_bridges` without spans: edges in `G.edges` order whose
/// endpoints share no neighbor (`set(G[u]) & set(G[v])` is empty; a
/// self-loop makes a node its own neighbor).
pub fn local_bridges(adj: &Csr, n: usize) -> Vec<(u32, u32)> {
    // `mark[w] == u`: w is a neighbor of u (only u's row sets that value).
    let mut mark = vec![NONE; n];
    let mut marked = NONE;
    let mut out = Vec::new();
    for (u, v) in undirected_edges(adj, n) {
        if marked != u {
            for &w in adj.neighbors(u as usize) {
                mark[w as usize] = u;
            }
            marked = u;
        }
        if !adj
            .neighbors(v as usize)
            .iter()
            .any(|&w| mark[w as usize] == u)
        {
            out.push((u, v));
        }
    }
    out
}

/// Shortest `u`-`v` distance with the edge `u`-`v` hidden (the span of a
/// local bridge), or `None` if `v` can't be reached. `weights` holds int
/// values (NaN: hidden edge); `None` means unit weights.
pub fn hidden_edge_distance(
    adj: &Csr,
    n: usize,
    u: u32,
    v: u32,
    weights: Option<&[f64]>,
) -> Option<f64> {
    let blocked = |a: u32, b: u32| (a == u || a == v) && (b == u || b == v);
    match weights {
        None => {
            let mut dist = vec![NONE; n];
            let mut queue = VecDeque::new();
            dist[u as usize] = 0;
            queue.push_back(u);
            while let Some(x) = queue.pop_front() {
                if x == v {
                    return Some(dist[x as usize] as f64);
                }
                for &y in adj.neighbors(x as usize) {
                    if dist[y as usize] == NONE && !blocked(x, y) {
                        dist[y as usize] = dist[x as usize] + 1;
                        queue.push_back(y);
                    }
                }
            }
            None
        }
        Some(w) => {
            // Non-negative integer weights: any exact Dijkstra gives
            // NetworkX's (integer) distance.
            use std::cmp::Reverse;
            use std::collections::BinaryHeap;
            let mut dist = vec![i64::MAX; n];
            let mut heap = BinaryHeap::new();
            dist[u as usize] = 0;
            heap.push(Reverse((0i64, u)));
            while let Some(Reverse((d, x))) = heap.pop() {
                if d > dist[x as usize] {
                    continue;
                }
                if x == v {
                    return Some(d as f64);
                }
                for e in adj.range(x as usize) {
                    let y = adj.targets[e];
                    if w[e].is_nan() || blocked(x, y) {
                        continue;
                    }
                    let nd = d + w[e] as i64;
                    if nd < dist[y as usize] {
                        dist[y as usize] = nd;
                        heap.push(Reverse((nd, y)));
                    }
                }
            }
            None
        }
    }
}

/// Hierholzer's walk as `_simplegraph_eulerian_circuit` does it on `adj`
/// (the copied or reversed graph): at each node take its first remaining
/// edge. `edge_id` maps each arc to its edge (both arcs of an undirected
/// edge share one). Returns the yielded `(last, current)` pairs.
pub fn euler_walk(adj: &Csr, edge_id: &[u32], m: usize, source: u32) -> Vec<(u32, u32)> {
    let mut used = vec![false; m];
    let mut cursor: Vec<usize> = (0..adj.offsets.len() - 1).map(|v| adj.offsets[v]).collect();
    let mut stack = vec![source];
    let mut last = NONE;
    let mut out = Vec::new();
    while let Some(&cur) = stack.last() {
        let c = cur as usize;
        let end = adj.offsets[c + 1];
        while cursor[c] < end && used[edge_id[cursor[c]] as usize] {
            cursor[c] += 1;
        }
        if cursor[c] == end {
            if last != NONE {
                out.push((last, cur));
            }
            last = cur;
            stack.pop();
        } else {
            let e = cursor[c];
            used[edge_id[e] as usize] = true;
            stack.push(adj.targets[e]);
        }
    }
    out
}

/// The adjacency of NetworkX's `G.copy()` for an undirected graph, with an
/// edge id per arc. The copy adds edges row by row, so each row lists the
/// neighbors before it (by position) first, then the rest in `G` order.
pub fn undirected_copy(adj: &Csr, n: usize) -> (Csr, Vec<u32>, usize) {
    let mut offsets = vec![0usize; n + 1];
    for v in 0..n {
        offsets[v + 1] = offsets[v] + adj.neighbors(v).len();
    }
    let mut targets = vec![0u32; adj.targets.len()];
    let mut ids = vec![0u32; adj.targets.len()];
    let mut fill: Vec<usize> = offsets[..n].to_vec();
    let mut m = 0u32;
    for u in 0..n {
        for &v in adj.neighbors(u) {
            if (v as usize) < u {
                continue; // already added from row v
            }
            targets[fill[u]] = v;
            ids[fill[u]] = m;
            fill[u] += 1;
            if v as usize != u {
                let w = v as usize;
                targets[fill[w]] = u as u32;
                ids[fill[w]] = m;
                fill[w] += 1;
            }
            m += 1;
        }
    }
    // Each row fills exactly (a self-loop takes one slot, as in `adj`).
    (Csr { offsets, targets }, ids, m as usize)
}

/// `nx.cycle_basis` (port of its stack-based spanning tree walk).
pub fn cycle_basis(adj: &Csr, n: usize, root: Option<u32>) -> Vec<Vec<u32>> {
    let mut removed = vec![false; n]; // popped from `gnodes`
    let mut last = n; // `gnodes.popitem()` pops the last remaining node
    let mut pred = vec![NONE; n];
    let mut used = PairSet::default(); // (owner, member) of each `used` set
    let mut cycles = Vec::new();
    let mut root = root;
    let mut remaining = n;
    let mut comp = Vec::new();
    while remaining > 0 {
        let r = match root.take() {
            Some(r) => r,
            None => {
                while removed[last - 1] {
                    last -= 1;
                }
                last -= 1;
                removed[last] = true;
                remaining -= 1;
                last as u32
            }
        };
        comp.clear();
        let mut stack = vec![r];
        pred[r as usize] = r;
        comp.push(r);
        while let Some(z) = stack.pop() {
            for &nbr in adj.neighbors(z as usize) {
                if pred[nbr as usize] == NONE {
                    pred[nbr as usize] = z;
                    comp.push(nbr);
                    stack.push(nbr);
                    used.insert(pair(nbr, z));
                } else if nbr == z {
                    cycles.push(vec![z]);
                } else if !used.contains(&pair(z, nbr)) {
                    let mut cycle = vec![nbr, z];
                    let mut p = pred[z as usize];
                    while !used.contains(&pair(nbr, p)) {
                        cycle.push(p);
                        p = pred[p as usize];
                    }
                    cycle.push(p);
                    cycles.push(cycle);
                    used.insert(pair(nbr, z));
                }
            }
        }
        for &v in &comp {
            if !removed[v as usize] {
                removed[v as usize] = true;
                remaining -= 1;
            }
        }
    }
    cycles
}

/// `nx.girth`: a BFS from every node as `bfs_labeled_edges` labels the
/// edges, cut off at the current depth limit. `None` means infinite.
pub fn girth(adj: &Csr, n: usize) -> Option<i64> {
    const INF: i64 = i64::MAX;
    let mut girth = INF;
    let mut depth_limit = INF;
    let mut depth = vec![u32::MAX; n];
    let mut processed = vec![false; n];
    let mut touched = Vec::new();
    let mut queue = VecDeque::new();
    for s in 0..n {
        for &v in &touched {
            depth[v as usize] = u32::MAX;
            processed[v as usize] = false;
        }
        touched.clear();
        queue.clear();
        depth[s] = 0;
        touched.push(s as u32);
        queue.push_back(s as u32);
        'bfs: while let Some(u) = queue.pop_front() {
            let du = depth[u as usize] as i64;
            for &v in adj.neighbors(u as usize) {
                // Every yielded edge first checks the depth limit.
                let dv = depth[v as usize];
                let label = if dv == u32::MAX {
                    0 // tree
                } else if du == dv as i64 {
                    if processed[v as usize] {
                        continue;
                    }
                    1 // level
                } else if du < dv as i64 {
                    2 // forward
                } else {
                    continue; // reverse edges aren't yielded (undirected)
                };
                if du > depth_limit {
                    break 'bfs;
                }
                if label == 0 {
                    depth[v as usize] = du as u32 + 1;
                    touched.push(v);
                    queue.push_back(v);
                } else {
                    let delta = (label == 1) as i64;
                    let length = du + du + 2 - delta;
                    if length < girth {
                        girth = length;
                        depth_limit = du - delta;
                    }
                }
            }
            processed[u as usize] = true;
        }
    }
    (girth != INF).then_some(girth)
}

/// Orientation handling of `nx.find_cycle` / `nx.edge_dfs`.
#[derive(Clone, Copy, PartialEq)]
pub enum Orientation {
    /// `orientation=None`: plain `(u, v)` edges.
    Plain,
    /// Undirected graphs with any orientation, or "original": out-edges
    /// labeled "forward".
    Forward,
    /// "reverse" (directed): in-edges labeled "reverse".
    Reverse,
    /// "ignore" (directed): out-edges, then in-edges.
    Ignore,
}

/// `nx.find_cycle` from `starts`: the cycle's edges as `(u, v, reverse)`
/// in edge_dfs form (`reverse` marks an in-edge yielded as
/// `(pred, node, "reverse")`), or `None` (no cycle). `pred` is the exact
/// `G._pred` order (directed graphs).
pub fn find_cycle(
    succ: &Csr,
    pred: Option<&Csr>,
    n: usize,
    directed: bool,
    orientation: Orientation,
    starts: &[u32],
) -> Option<Vec<(u32, u32, bool)>> {
    let use_out = orientation != Orientation::Reverse || !directed;
    let use_in = directed && matches!(orientation, Orientation::Reverse | Orientation::Ignore);
    let empty = Csr {
        offsets: vec![0; n + 1],
        targets: Vec::new(),
    };
    let pred = pred.unwrap_or(&empty);
    // Each node's edge iterator: position in its out-edges, then in-edges.
    let mut out_pos = vec![0usize; n];
    let mut in_pos = vec![0usize; n];
    let mut visited_edges = PairSet::default();
    let mut explored = vec![false; n];
    let mut active = vec![false; n];
    let mut seen_list: Vec<u32> = Vec::new();

    // The next edge from `node`'s iterator: (u, v, is_reverse).
    let next_edge =
        |node: u32, out_pos: &mut [usize], in_pos: &mut [usize]| -> Option<(u32, u32, bool)> {
            let x = node as usize;
            if use_out && out_pos[x] < succ.neighbors(x).len() {
                let v = succ.neighbors(x)[out_pos[x]];
                out_pos[x] += 1;
                return Some((node, v, false));
            }
            if use_in && in_pos[x] < pred.neighbors(x).len() {
                let u = pred.neighbors(x)[in_pos[x]];
                in_pos[x] += 1;
                return Some((u, node, true));
            }
            None
        };
    let tailhead = |e: (u32, u32, bool)| -> (u32, u32) {
        if directed
            && (orientation == Orientation::Reverse || (orientation == Orientation::Ignore && e.2))
        {
            (e.1, e.0)
        } else {
            (e.0, e.1)
        }
    };

    for &start in starts {
        if explored[start as usize] {
            continue;
        }
        let mut edges: Vec<(u32, u32, bool)> = Vec::new();
        seen_list.clear();
        seen_list.push(start);
        let mut base = start; // the node in `active_nodes` below the path
        active[start as usize] = true;
        let mut previous_head = NONE;
        let mut stack = vec![start];
        let mut found = None;
        'dfs: while let Some(&cur) = stack.last() {
            let Some(edge) = next_edge(cur, &mut out_pos, &mut in_pos) else {
                stack.pop();
                continue;
            };
            let key = if directed {
                pair(edge.0, edge.1)
            } else {
                pair(edge.0.min(edge.1), edge.0.max(edge.1))
            };
            if !visited_edges.insert(key) {
                continue;
            }
            let (tail, head) = tailhead(edge);
            if explored[head as usize] {
                // NetworkX would walk on through explored nodes, but
                // everything reachable from them is explored too, so the
                // edges it finds there are all skipped like this one.
                continue;
            }
            stack.push(head);
            if previous_head != NONE && tail != previous_head {
                loop {
                    match edges.pop() {
                        None => {
                            active[base as usize] = false;
                            active[tail as usize] = true;
                            base = tail;
                            break;
                        }
                        Some(popped) => {
                            active[tailhead(popped).1 as usize] = false;
                        }
                    }
                    if let Some(&last) = edges.last() {
                        if tail == tailhead(last).1 {
                            break;
                        }
                    }
                }
            }
            edges.push(edge);
            if active[head as usize] {
                found = Some(head);
                break 'dfs;
            }
            seen_list.push(head);
            active[head as usize] = true;
            previous_head = head;
        }
        if let Some(final_node) = found {
            let i = edges
                .iter()
                .position(|&e| tailhead(e).0 == final_node)
                .unwrap_or(edges.len() - 1);
            return Some(edges.split_off(i));
        }
        // No cycle: clear the path's marks and mark what was seen.
        for &v in &seen_list {
            active[v as usize] = false;
            explored[v as usize] = true;
        }
        active[base as usize] = false;
    }
    None
}

/// `immediate_dominators` from `start`: nodes in NetworkX's dict order
/// (`start`, then reverse postorder) and the immediate dominator of each
/// (`start` for itself).
pub fn immediate_dominators(
    pred: &Csr,
    n: usize,
    start: u32,
    postorder: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let mut dfn = vec![NONE; n];
    for (i, &u) in postorder.iter().enumerate() {
        dfn[u as usize] = i as u32;
    }
    let mut idom = vec![NONE; n];
    idom[start as usize] = start;
    let mut order: Vec<u32> = postorder[..postorder.len() - 1].to_vec();
    order.reverse();
    let intersect = |idom: &[u32], mut u: u32, mut v: u32| {
        while u != v {
            while dfn[u as usize] < dfn[v as usize] {
                u = idom[u as usize];
            }
            while dfn[u as usize] > dfn[v as usize] {
                v = idom[v as usize];
            }
        }
        u
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &u in &order {
            let mut new_idom = NONE;
            for &v in pred.neighbors(u as usize) {
                if idom[v as usize] == NONE {
                    continue;
                }
                new_idom = if new_idom == NONE {
                    v
                } else {
                    intersect(&idom, new_idom, v)
                };
            }
            if idom[u as usize] != new_idom {
                idom[u as usize] = new_idom;
                changed = true;
            }
        }
    }
    let mut nodes = vec![start];
    nodes.extend_from_slice(&order);
    let values = nodes.iter().map(|&u| idom[u as usize]).collect();
    (nodes, values)
}

/// `dominance_frontiers`: the `df[v].add(u)` calls in NetworkX's order, as
/// `(v, u)`, given the dominator dict order and values from
/// `immediate_dominators`. `new_style` is NetworkX 3.7's version (start's
/// dominator is `None`, start goes last and is always processed).
pub fn dominance_frontier_adds(
    pred: &Csr,
    n: usize,
    nodes: &[u32],
    idom_of: &[u32],
    new_style: bool,
) -> Vec<(u32, u32)> {
    let mut idom = vec![NONE; n];
    for (&u, &d) in nodes.iter().zip(idom_of) {
        idom[u as usize] = d;
    }
    let start = nodes[0];
    let mut order: Vec<u32> = nodes.to_vec();
    if new_style {
        order.remove(0);
        order.push(start);
        idom[start as usize] = NONE - 1; // stands for None
    }
    let in_idom = |v: u32, idom: &[u32]| idom[v as usize] != NONE;
    let mut adds = Vec::new();
    for &u in &order {
        let preds = pred.neighbors(u as usize);
        if !((new_style && u == start) || preds.len() >= 2) {
            continue;
        }
        for &v0 in preds {
            if !in_idom(v0, &idom) {
                continue;
            }
            let target = idom[u as usize];
            let mut v = v0;
            while v != target {
                adds.push((v, u));
                v = idom[v as usize];
                if v == NONE - 1 {
                    break; // walked past start (None)
                }
            }
        }
    }
    adds
}

/// `nx.to_prufer_sequence` on a tree whose node labelled `k` sits at
/// position `pos[k]`; returns positions of the sequence's nodes.
pub fn prufer(adj: &Csr, n: usize, pos: &[u32], degree: &[usize]) -> Vec<u32> {
    // Work by label: `degree[label]`, neighbors of the node with a label.
    let mut label = vec![0u32; n];
    for (k, &p) in pos.iter().enumerate() {
        label[p as usize] = k as u32;
    }
    let mut deg: Vec<usize> = (0..n).map(|k| degree[pos[k] as usize]).collect();
    let parent = |u: usize, deg: &[usize]| -> usize {
        let p = adj
            .neighbors(pos[u] as usize)
            .iter()
            .find(|&&w| deg[label[w as usize] as usize] > 1)
            .expect("a tree has a parent");
        label[*p as usize] as usize
    };
    let mut index = (0..n).find(|&k| deg[k] == 1).expect("a tree has leaves");
    let mut u = index;
    let mut result = Vec::with_capacity(n.saturating_sub(2));
    for _ in 0..n.saturating_sub(2) {
        let v = parent(u, &deg);
        result.push(pos[v]);
        deg[v] -= 1;
        if v < index && deg[v] == 1 {
            u = v;
        } else {
            index = (index + 1..n)
                .find(|&k| deg[k] == 1)
                .expect("a leaf remains");
            u = index;
        }
    }
    result
}
