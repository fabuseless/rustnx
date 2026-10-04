//! BFS, Dijkstra and connected components, mirroring NetworkX's visit order.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::graph::Csr;

/// Min-heap key matching NetworkX's `(distance, counter)` heap tuples.
#[derive(Clone, Copy, PartialEq)]
pub struct HeapKey(pub f64, pub u64);

impl Eq for HeapKey {}

impl PartialOrd for HeapKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapKey {
    fn cmp(&self, other: &Self) -> Ordering {
        // Distances are never NaN: NaN weights are rejected at conversion.
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

pub type MinHeap<T> = BinaryHeap<Reverse<(HeapKey, T)>>;

pub struct NegativeCycle;

/// `nx.single_source_shortest_path_length`: nodes in BFS order with levels.
pub fn bfs_lengths(adj: &Csr, n: usize, source: usize, cutoff: f64) -> (Vec<u32>, Vec<u32>) {
    let mut seen = vec![false; n];
    let mut order = vec![source as u32];
    let mut levels = vec![0u32];
    seen[source] = true;
    let mut nseen = 1;
    let mut this_start = 0;
    let mut level = 0u32;
    'outer: while this_start < order.len() && cutoff > level as f64 {
        level += 1;
        let this_end = order.len();
        for i in this_start..this_end {
            let v = order[i] as usize;
            for &w in adj.neighbors(v) {
                if !seen[w as usize] {
                    seen[w as usize] = true;
                    nseen += 1;
                    order.push(w);
                    levels.push(level);
                }
            }
            if nseen == n {
                break 'outer;
            }
        }
        this_start = this_end;
    }
    (order, levels)
}

/// Reusable buffers for single-source Dijkstra.
pub struct DijkstraState {
    pub dist: Vec<f64>,
    pub done: Vec<bool>,
    seen: Vec<f64>,
    has_seen: Vec<bool>,
    /// Finalized nodes in pop order (the order of NetworkX's result dict).
    pub order: Vec<u32>,
    /// Every node whose `has_seen` flag was set, for cheap resets.
    touched: Vec<u32>,
    heap: MinHeap<u32>,
}

impl DijkstraState {
    pub fn new(n: usize) -> Self {
        DijkstraState {
            dist: vec![0.0; n],
            done: vec![false; n],
            seen: vec![0.0; n],
            has_seen: vec![false; n],
            order: Vec::new(),
            touched: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    fn reset(&mut self) {
        for &v in &self.touched {
            self.done[v as usize] = false;
            self.has_seen[v as usize] = false;
        }
        self.touched.clear();
        self.order.clear();
        self.heap.clear();
    }

    fn mark_seen(&mut self, u: usize, dist: f64) {
        if !self.has_seen[u] {
            self.has_seen[u] = true;
            self.touched.push(u as u32);
        }
        self.seen[u] = dist;
    }

    /// Mirrors `networkx...weighted._dijkstra_multisource` with one source.
    /// `weights = None` means every edge has weight 1. NaN marks hidden edges.
    pub fn run(
        &mut self,
        adj: &Csr,
        weights: Option<&[f64]>,
        source: usize,
        cutoff: Option<f64>,
    ) -> Result<(), NegativeCycle> {
        self.reset();
        let mut counter = 0u64;
        self.mark_seen(source, 0.0);
        self.heap
            .push(Reverse((HeapKey(0.0, counter), source as u32)));
        while let Some(Reverse((HeapKey(dist_v, _), v))) = self.heap.pop() {
            let v = v as usize;
            if self.done[v] {
                continue;
            }
            self.done[v] = true;
            self.dist[v] = dist_v;
            self.order.push(v as u32);
            for e in adj.range(v) {
                let u = adj.targets[e] as usize;
                let cost = weights.map_or(1.0, |w| w[e]);
                if cost.is_nan() {
                    continue; // hidden edge
                }
                let vu_dist = dist_v + cost;
                if cutoff.is_some_and(|c| vu_dist > c) {
                    continue;
                }
                if self.done[u] {
                    if vu_dist < self.dist[u] {
                        return Err(NegativeCycle);
                    }
                } else if !self.has_seen[u] || vu_dist < self.seen[u] {
                    self.mark_seen(u, vu_dist);
                    counter += 1;
                    self.heap
                        .push(Reverse((HeapKey(vu_dist, counter), u as u32)));
                }
            }
        }
        Ok(())
    }
}

/// Connected components in NetworkX order (by first node in `G`).
pub fn connected_components(adj: &Csr, n: usize) -> Vec<Vec<u32>> {
    let mut seen = vec![false; n];
    let mut comps = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut comp = vec![start as u32];
        let mut i = 0;
        while i < comp.len() {
            let v = comp[i] as usize;
            for &w in adj.neighbors(v) {
                if !seen[w as usize] {
                    seen[w as usize] = true;
                    comp.push(w);
                }
            }
            i += 1;
        }
        comps.push(comp);
    }
    comps
}

/// `nx.generic_bfs_edges` from `source`: tree edges `(parent, child)` in
/// yield order, stopping at `depth_limit` levels.
pub fn bfs_edges(adj: &Csr, n: usize, source: usize, depth_limit: i64) -> Vec<(u32, u32)> {
    let mut seen = vec![false; n];
    seen[source] = true;
    let mut count = 1;
    let mut edges = Vec::new();
    let mut level = vec![source as u32];
    let mut depth = 0i64;
    'outer: while !level.is_empty() && depth < depth_limit {
        let mut next = Vec::new();
        for &parent in &level {
            for &child in adj.neighbors(parent as usize) {
                if !seen[child as usize] {
                    seen[child as usize] = true;
                    count += 1;
                    next.push(child);
                    edges.push((parent, child));
                }
            }
            if count == n {
                break 'outer;
            }
        }
        level = next;
        depth += 1;
    }
    edges
}

/// `nx.dfs_labeled_edges`' forward edges from each of `starts` (skipping
/// visited ones): `(start, start)` marks a new start, then tree edges
/// `(parent, child)` in yield order.
pub fn dfs_forward(adj: &Csr, n: usize, starts: &[u32], depth_limit: i64) -> Vec<(u32, u32)> {
    let mut visited = vec![false; n];
    let mut out = Vec::new();
    // Stack of (node, next neighbor position).
    let mut stack: Vec<(u32, usize)> = Vec::new();
    for &start in starts {
        if visited[start as usize] {
            continue;
        }
        out.push((start, start));
        visited[start as usize] = true;
        stack.push((start, adj.range(start as usize).start));
        let mut depth_now = 1i64;
        while let Some(&mut (parent, ref mut pos)) = stack.last_mut() {
            let end = adj.range(parent as usize).end;
            let mut descended = None;
            while *pos < end {
                let child = adj.targets[*pos];
                *pos += 1;
                if !visited[child as usize] {
                    out.push((parent, child));
                    visited[child as usize] = true;
                    if depth_now < depth_limit {
                        descended = Some(child);
                        break;
                    }
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
                }
            }
        }
    }
    out
}

/// `nx.bfs_layers` after the first layer (`starts`, which may repeat
/// nodes): every later node in yield order, and where each layer ends.
/// `visited` is NetworkX's initial visited set, usually `starts`.
pub fn bfs_layers(adj: &Csr, n: usize, starts: &[u32], visited: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut seen = vec![false; n];
    for &s in visited {
        seen[s as usize] = true;
    }
    let mut visited = seen;
    let mut order = Vec::new();
    let mut ends = Vec::new();
    let mut current = starts.to_vec();
    loop {
        let begin = order.len();
        for &node in &current {
            for &child in adj.neighbors(node as usize) {
                if !visited[child as usize] {
                    visited[child as usize] = true;
                    order.push(child);
                }
            }
        }
        if order.len() == begin {
            break;
        }
        ends.push(order.len() as u32);
        current = order[begin..].to_vec();
    }
    (order, ends)
}

/// `nx.dfs_postorder_nodes`: the nodes `nx.dfs_labeled_edges` reports with
/// a "reverse" edge, in order. Nodes reached at the depth limit are never
/// descended into, so they get "reverse-depth_limit" and are left out.
pub fn dfs_postorder(adj: &Csr, n: usize, starts: &[u32], depth_limit: i64) -> Vec<u32> {
    let mut visited = vec![false; n];
    let mut out = Vec::new();
    let mut stack: Vec<(u32, usize)> = Vec::new();
    for &start in starts {
        if visited[start as usize] {
            continue;
        }
        visited[start as usize] = true;
        stack.push((start, adj.range(start as usize).start));
        let mut depth_now = 1i64;
        while let Some(&mut (parent, ref mut pos)) = stack.last_mut() {
            let end = adj.range(parent as usize).end;
            let mut descended = None;
            while *pos < end {
                let child = adj.targets[*pos];
                *pos += 1;
                if !visited[child as usize] {
                    visited[child as usize] = true;
                    if depth_now < depth_limit {
                        descended = Some(child);
                        break;
                    }
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
                    out.push(parent);
                }
            }
        }
    }
    out
}
