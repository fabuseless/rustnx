//! More shortest paths (batch 2): multi-source Dijkstra, predecessor lists,
//! Bellman-Ford, negative cycles and A*, ported from NetworkX step by step
//! so that ties, dict orders and error points match.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use super::paths::{DijkstraTree, NO_PARENT};
use super::traversal::{HeapKey, NegativeCycle};
use crate::graph::Csr;

/// `_dijkstra_multisource` recording parents, from every node of `sources`
/// (pushed in order, repeats included, as NetworkX does). Stops once
/// `target` is popped. NaN weights mark hidden edges. `seen_order` starts
/// with the distinct sources.
pub fn dijkstra_forest(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    sources: &[u32],
    cutoff: Option<f64>,
    target: Option<usize>,
) -> Result<DijkstraTree, NegativeCycle> {
    let mut done = vec![false; n];
    let mut final_dist = vec![0.0; n];
    let mut seen = vec![f64::INFINITY; n];
    let mut has_seen = vec![false; n];
    let mut parent = vec![NO_PARENT; n];
    let mut heap: BinaryHeap<Reverse<(HeapKey, u32)>> = BinaryHeap::new();
    let mut counter = 0u64;
    let mut order = Vec::new();
    let mut dist = Vec::new();
    let mut seen_order = Vec::new();
    for &s in sources {
        let s = s as usize;
        if !has_seen[s] {
            has_seen[s] = true;
            seen_order.push(s as u32);
        }
        seen[s] = 0.0;
        heap.push(Reverse((HeapKey(0.0, counter), s as u32)));
        counter += 1;
    }
    while let Some(Reverse((HeapKey(d, _), v))) = heap.pop() {
        let v = v as usize;
        if done[v] {
            continue;
        }
        done[v] = true;
        final_dist[v] = d;
        order.push(v as u32);
        dist.push(d);
        if target == Some(v) {
            break;
        }
        for e in adj.range(v) {
            let u = adj.targets[e] as usize;
            let cost = weights.map_or(1.0, |w| w[e]);
            if cost.is_nan() {
                continue;
            }
            let vu = d + cost;
            if cutoff.is_some_and(|c| vu > c) {
                continue;
            }
            if done[u] {
                if vu < final_dist[u] {
                    return Err(NegativeCycle);
                }
            } else if !has_seen[u] || vu < seen[u] {
                if !has_seen[u] {
                    has_seen[u] = true;
                    seen_order.push(u as u32);
                }
                seen[u] = vu;
                heap.push(Reverse((HeapKey(vu, counter), u as u32)));
                counter += 1;
                parent[u] = v as u32;
            }
        }
    }
    Ok(DijkstraTree {
        order,
        dist,
        parent,
        seen_order,
    })
}

/// Predecessor lists as NetworkX builds its `pred` dict: `order` is the
/// dict's key order, `lists[v]` the list for node `v` (empty if unset).
pub struct PredLists {
    pub order: Vec<u32>,
    pub lists: Vec<Vec<u32>>,
    pub reached: Vec<bool>,
}

impl PredLists {
    pub fn new(n: usize, source: usize) -> Self {
        let mut reached = vec![false; n];
        reached[source] = true;
        PredLists {
            order: vec![source as u32],
            lists: vec![Vec::new(); n],
            reached,
        }
    }

    /// The lists in key order, flattened, with where each one ends.
    pub fn flatten(&self) -> (Vec<u32>, Vec<u32>) {
        let mut flat = Vec::new();
        let mut ends = Vec::with_capacity(self.order.len());
        for &v in &self.order {
            flat.extend_from_slice(&self.lists[v as usize]);
            ends.push(flat.len() as u32);
        }
        (flat, ends)
    }
}

/// `nx.dijkstra_predecessor_and_distance`: pop order with distances, and
/// the predecessor lists (keys in first-push order).
pub fn dijkstra_pred_dist(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    source: usize,
    cutoff: Option<f64>,
) -> Result<(Vec<u32>, Vec<f64>, PredLists), NegativeCycle> {
    let mut done = vec![false; n];
    let mut dist = vec![0.0; n];
    let mut seen = vec![f64::INFINITY; n];
    let mut pred = PredLists::new(n, source);
    let mut heap: BinaryHeap<Reverse<(HeapKey, u32)>> = BinaryHeap::new();
    let mut counter = 0u64;
    let mut order = Vec::new();
    let mut dists = Vec::new();
    seen[source] = 0.0;
    heap.push(Reverse((HeapKey(0.0, counter), source as u32)));
    while let Some(Reverse((HeapKey(d, _), v))) = heap.pop() {
        let v = v as usize;
        if done[v] {
            continue;
        }
        done[v] = true;
        dist[v] = d;
        order.push(v as u32);
        dists.push(d);
        for e in adj.range(v) {
            let u = adj.targets[e] as usize;
            let cost = weights.map_or(1.0, |w| w[e]);
            if cost.is_nan() {
                continue;
            }
            let vu = d + cost;
            if cutoff.is_some_and(|c| vu > c) {
                continue;
            }
            if done[u] {
                if vu < dist[u] {
                    return Err(NegativeCycle);
                } else if vu == dist[u] {
                    pred.lists[u].push(v as u32);
                }
            } else if !pred.reached[u] || vu < seen[u] {
                if !pred.reached[u] {
                    pred.reached[u] = true;
                    pred.order.push(u as u32);
                }
                seen[u] = vu;
                counter += 1;
                heap.push(Reverse((HeapKey(vu, counter), u as u32)));
                pred.lists[u].clear();
                pred.lists[u].push(v as u32);
            } else if vu == seen[u] {
                pred.lists[u].push(v as u32);
            }
        }
    }
    Ok((order, dists, pred))
}

/// `nx.predecessor`: BFS levels and predecessor lists in discovery order,
/// stopping after level `max_level` (NetworkX's `cutoff <= level` check).
pub fn bfs_pred(
    adj: &Csr,
    n: usize,
    source: usize,
    max_level: Option<u32>,
) -> (Vec<u32>, PredLists) {
    let mut level = vec![u32::MAX; n];
    let mut pred = PredLists::new(n, source);
    let mut levels = vec![0u32];
    level[source] = 0;
    let mut this_level = vec![source as u32];
    let mut depth = 0u32;
    while !this_level.is_empty() {
        depth += 1;
        let mut next = Vec::new();
        for &v in &this_level {
            for &w in adj.neighbors(v as usize) {
                let wi = w as usize;
                if level[wi] == u32::MAX {
                    level[wi] = depth;
                    pred.reached[wi] = true;
                    pred.order.push(w);
                    pred.lists[wi].push(v);
                    levels.push(depth);
                    next.push(w);
                } else if level[wi] == depth {
                    pred.lists[wi].push(v);
                }
            }
        }
        this_level = next;
        if max_level.is_some_and(|m| m <= depth) {
            break;
        }
    }
    (levels, pred)
}

/// `_build_paths_from_predecessors({source}, target, pred)`, appending up to
/// `limit` paths (source first) to `out`, each followed by its end in
/// `ends`. `on_path` must be all false; it is left all false. With
/// `skip_ends_pass` (NetworkX before 3.7) skipping a predecessor already on
/// the path ends the pass, as in `paths::PathsFromPreds`.
#[allow(clippy::too_many_arguments)]
pub fn paths_from_preds(
    pred: &[Vec<u32>],
    source: u32,
    target: u32,
    skip_ends_pass: bool,
    limit: usize,
    on_path: &mut [bool],
    out: &mut Vec<u32>,
    ends: &mut Vec<u32>,
) -> usize {
    let mut path = vec![target];
    let mut next_pred = vec![0usize];
    on_path[target as usize] = true;
    let mut yielded = false;
    let mut found = 0;
    while let Some(&node) = path.last() {
        if !yielded && node == source {
            yielded = true;
            out.extend(path.iter().rev());
            ends.push(out.len() as u32);
            found += 1;
            if found == limit {
                break;
            }
            continue;
        }
        yielded = false;
        let preds = &pred[node as usize];
        let i = next_pred.last_mut().expect("parallel to path");
        let mut pushed = None;
        let mut skipped = false;
        while *i < preds.len() {
            let p = preds[*i];
            *i += 1;
            if !on_path[p as usize] {
                pushed = Some(p);
                break;
            }
            if skip_ends_pass {
                skipped = true;
                break;
            }
        }
        if skipped {
            continue;
        }
        match pushed {
            Some(p) => {
                path.push(p);
                next_pred.push(0);
                on_path[p as usize] = true;
            }
            None => {
                on_path[node as usize] = false;
                path.pop();
                next_pred.pop();
            }
        }
    }
    for &v in &path {
        on_path[v as usize] = false;
    }
    found
}

/// State of `_inner_bellman_ford` when it returns.
pub struct BellmanFord {
    /// Keys of NetworkX's `dist` (and `pred`) dicts, in insertion order.
    pub order: Vec<u32>,
    pub dist: Vec<f64>,
    pub pred: PredLists,
    /// The node `_inner_bellman_ford` returned: a negative cycle was found.
    pub cycle: Option<u32>,
}

/// `_inner_bellman_ford` from one source: NetworkX's queue-based
/// Bellman-Ford, with its deque order, predecessor lists, the "skip if a
/// predecessor is queued" rule and both negative cycle checks (`heuristic`,
/// and a node entering the queue `n` times). No weight may be hidden.
pub fn bellman_ford(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    source: usize,
    heuristic: bool,
) -> BellmanFord {
    const NONE: u32 = u32::MAX;
    let mut dist = vec![0.0f64; n];
    let mut pred = PredLists::new(n, source);
    // `pred_edge` and `recent_update` have keys for the source (value None).
    let mut pred_edge = vec![NONE; n];
    let mut recent = vec![(NONE, NONE); n];
    let mut count = vec![0usize; n];
    let mut in_q = vec![false; n];
    let mut q = VecDeque::new();
    q.push_back(source as u32);
    in_q[source] = true;
    let mut cycle = None;
    'outer: while let Some(u) = q.pop_front() {
        let u = u as usize;
        in_q[u] = false;
        if pred.lists[u].iter().any(|&p| in_q[p as usize]) {
            continue;
        }
        let dist_u = dist[u];
        for e in adj.range(u) {
            let v = adj.targets[e] as usize;
            let dist_v = dist_u + weights.map_or(1.0, |w| w[e]);
            // `dist.get(v, inf)`
            let current = if pred.reached[v] {
                dist[v]
            } else {
                f64::INFINITY
            };
            if dist_v < current {
                if heuristic {
                    let (a, b) = recent[u];
                    if a == v as u32 || b == v as u32 {
                        pred.lists[v].push(u as u32);
                        cycle = Some(v as u32);
                        break 'outer;
                    }
                    recent[v] = if pred_edge[v] == u as u32 {
                        recent[u]
                    } else {
                        (u as u32, v as u32)
                    };
                }
                if !in_q[v] {
                    q.push_back(v as u32);
                    in_q[v] = true;
                    count[v] += 1;
                    if count[v] == n {
                        cycle = Some(v as u32);
                        break 'outer;
                    }
                }
                if !pred.reached[v] {
                    pred.reached[v] = true;
                    pred.order.push(v as u32);
                }
                dist[v] = dist_v;
                pred.lists[v].clear();
                pred.lists[v].push(u as u32);
                pred_edge[v] = u as u32;
            } else if pred.reached[v] && dist_v == dist[v] {
                pred.lists[v].push(u as u32);
            }
        }
    }
    let order = pred.order.clone();
    let dists = order.iter().map(|&v| dist[v as usize]).collect();
    BellmanFord {
        order,
        dist: dists,
        pred,
        cycle,
    }
}

/// `find_negative_cycle`'s search through the predecessors once
/// `_inner_bellman_ford` returned `v`. `None` where NetworkX reaches its
/// "should not reach here" branches.
pub fn negative_cycle_from(pred: &[Vec<u32>], n: usize, v: u32) -> Option<Vec<u32>> {
    let mut neg_cycle: Vec<u32> = Vec::new();
    let mut stack: Vec<(u32, Vec<u32>)> = vec![(v, pred[v as usize].clone())];
    let mut seen = vec![false; n];
    seen[v as usize] = true;
    while let Some(top) = stack.len().checked_sub(1) {
        let node = stack[top].0;
        if stack[top].1.contains(&v) {
            neg_cycle.push(node);
            neg_cycle.push(v);
            neg_cycle.reverse();
            return Some(neg_cycle);
        }
        match stack[top].1.pop() {
            Some(nbr) => {
                if !seen[nbr as usize] {
                    stack.push((nbr, pred[nbr as usize].clone()));
                    neg_cycle.push(node);
                    seen[nbr as usize] = true;
                }
            }
            None => {
                stack.pop();
                neg_cycle.pop()?;
            }
        }
    }
    None
}

/// Whether any self-loop has a negative weight.
pub fn negative_selfloop(adj: &Csr, n: usize, weights: Option<&[f64]>) -> bool {
    let Some(w) = weights else { return false };
    (0..n).any(|u| {
        adj.range(u)
            .any(|e| adj.targets[e] as usize == u && w[e] < 0.0)
    })
}

/// `nx.negative_edge_cycle` after its empty-graph check: Bellman-Ford from a
/// new node joined to every node by weight-1 edges (appended to each row of
/// an undirected graph, as `G.add_edges_from` does).
pub fn negative_edge_cycle(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    directed: bool,
    heuristic: bool,
) -> bool {
    if negative_selfloop(adj, n, weights) {
        return true;
    }
    let m = adj.targets.len();
    let extra = if directed { n } else { 2 * n };
    let mut offsets = Vec::with_capacity(n + 2);
    let mut targets = Vec::with_capacity(m + extra);
    let mut w = Vec::with_capacity(m + extra);
    offsets.push(0);
    for u in 0..n {
        for e in adj.range(u) {
            targets.push(adj.targets[e]);
            w.push(weights.map_or(1.0, |ws| ws[e]));
        }
        if !directed {
            targets.push(n as u32);
            w.push(1.0);
        }
        offsets.push(targets.len());
    }
    for u in 0..n {
        targets.push(u as u32);
        w.push(1.0);
    }
    offsets.push(targets.len());
    let aug = Csr { offsets, targets };
    bellman_ford(&aug, n + 1, Some(&w), n, heuristic)
        .cycle
        .is_some()
}

/// `nx.astar_path` with the default (zero) heuristic: the path, and the
/// weight of each of its edges. `cutoff` is only given when truthy.
pub fn astar(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    source: usize,
    target: usize,
    cutoff: Option<f64>,
) -> Option<(Vec<u32>, Vec<f64>)> {
    const NONE: u32 = u32::MAX;
    // Heap entries: (priority, counter) -> (node, dist, parent, edge cost).
    let mut queue: BinaryHeap<Reverse<(HeapKey, u32)>> = BinaryHeap::new();
    let mut items: Vec<(u32, f64, u32, f64)> = Vec::new();
    let mut enqueued = vec![f64::NAN; n];
    let mut has_enqueued = vec![false; n];
    let mut explored = vec![NONE; n];
    let mut explored_cost = vec![0.0; n];
    let mut has_explored = vec![false; n];
    queue.push(Reverse((HeapKey(0.0, 0), 0)));
    items.push((source as u32, 0.0, NONE, 0.0));
    while let Some(Reverse((_, item))) = queue.pop() {
        let (cur, dist, parent, cost_in) = items[item as usize];
        let c = cur as usize;
        if c == target {
            let mut path = vec![cur];
            let mut costs = Vec::new();
            let mut node = parent;
            let mut cost = cost_in;
            while node != NONE {
                path.push(node);
                costs.push(cost);
                cost = explored_cost[node as usize];
                node = explored[node as usize];
            }
            path.reverse();
            costs.reverse();
            return Some((path, costs));
        }
        if has_explored[c] {
            if explored[c] == NONE {
                continue;
            }
            if enqueued[c] < dist {
                continue;
            }
        }
        has_explored[c] = true;
        explored[c] = parent;
        explored_cost[c] = cost_in;
        for e in adj.range(c) {
            let nb = adj.targets[e] as usize;
            let cost = weights.map_or(1.0, |w| w[e]);
            if cost.is_nan() {
                continue;
            }
            let ncost = dist + cost;
            if has_enqueued[nb] && enqueued[nb] <= ncost {
                continue;
            }
            if cutoff.is_some_and(|cut| ncost > cut) {
                continue;
            }
            has_enqueued[nb] = true;
            enqueued[nb] = ncost;
            let counter = items.len() as u64;
            items.push((nb as u32, ncost, cur, cost));
            queue.push(Reverse((HeapKey(ncost, counter), counter as u32)));
        }
    }
    None
}
