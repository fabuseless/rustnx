//! Shortest paths that return the paths themselves, ported from NetworkX so
//! that ties are broken the same way (and so the same path comes back).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::traversal::{HeapKey, NegativeCycle};
use crate::graph::Csr;

pub const NO_PARENT: u32 = u32::MAX;

/// BFS from `source` as `_single_shortest_path`: nodes in discovery order,
/// each with the node that discovered it (the source's parent is
/// `NO_PARENT`).
pub fn bfs_tree(adj: &Csr, n: usize, source: usize, cutoff: f64) -> (Vec<u32>, Vec<u32>) {
    let mut seen = vec![false; n];
    let mut order = vec![source as u32];
    let mut parent = vec![NO_PARENT];
    seen[source] = true;
    let mut start = 0;
    let mut level = 0u32;
    while start < order.len() && cutoff > level as f64 {
        let end = order.len();
        for i in start..end {
            let v = order[i];
            for &w in adj.neighbors(v as usize) {
                if !seen[w as usize] {
                    seen[w as usize] = true;
                    order.push(w);
                    parent.push(v);
                }
            }
        }
        start = end;
        level += 1;
    }
    (order, parent)
}

pub struct DijkstraTree {
    /// Finalized nodes in pop order, with their distances.
    pub order: Vec<u32>,
    pub dist: Vec<f64>,
    /// For every node index: the last node that improved its tentative
    /// distance (NetworkX's `pred_dict[u][0]`), or `NO_PARENT`.
    pub parent: Vec<u32>,
    /// Nodes in order of first push (older NetworkX orders its paths dict
    /// this way); the source first.
    pub seen_order: Vec<u32>,
}

/// `_dijkstra_multisource` with one source, recording parents. Stops once
/// `target` is popped. NaN weights mark hidden edges.
pub fn dijkstra_tree(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    source: usize,
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
    let mut seen_order = vec![source as u32];
    seen[source] = 0.0;
    has_seen[source] = true;
    heap.push(Reverse((HeapKey(0.0, counter), source as u32)));
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
                counter += 1;
                heap.push(Reverse((HeapKey(vu, counter), u as u32)));
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

/// `_bidirectional_pred_succ` + path assembly from
/// `bidirectional_shortest_path`. `succ`/`pred` are the forward and reverse
/// adjacency (the same rows for undirected graphs).
pub fn bidirectional_bfs(succ: &Csr, pred: &Csr, n: usize, s: usize, t: usize) -> Option<Vec<u32>> {
    const UNSET: i64 = -2;
    const NONE: i64 = -1;
    if s == t {
        return Some(vec![s as u32]);
    }
    let mut pred_of = vec![UNSET; n]; // forward tree: node -> predecessor
    let mut succ_of = vec![UNSET; n]; // reverse tree: node -> successor
    pred_of[s] = NONE;
    succ_of[t] = NONE;
    let mut forward = vec![s as u32];
    let mut reverse = vec![t as u32];
    let mut meet = None;
    'search: while !forward.is_empty() && !reverse.is_empty() {
        if forward.len() <= reverse.len() {
            let this_level = std::mem::take(&mut forward);
            for &v in &this_level {
                for &w in succ.neighbors(v as usize) {
                    let wi = w as usize;
                    if pred_of[wi] == UNSET {
                        forward.push(w);
                        pred_of[wi] = v as i64;
                    }
                    if succ_of[wi] != UNSET {
                        meet = Some(wi);
                        break 'search;
                    }
                }
            }
        } else {
            let this_level = std::mem::take(&mut reverse);
            for &v in &this_level {
                for &w in pred.neighbors(v as usize) {
                    let wi = w as usize;
                    if succ_of[wi] == UNSET {
                        succ_of[wi] = v as i64;
                        reverse.push(w);
                    }
                    if pred_of[wi] != UNSET {
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
        cur = pred_of[cur as usize];
    }
    path.reverse();
    let mut cur = succ_of[*path.last().expect("non-empty") as usize];
    while cur != NONE {
        path.push(cur as u32);
        cur = succ_of[cur as usize];
    }
    Some(path)
}

pub enum BidirectionalError {
    NoPath,
    Contradictory,
}

/// `nx.bidirectional_dijkstra` for `source != target`: the distance and
/// path. `succ`/`pred` are the forward adjacency and the reverse adjacency in
/// NetworkX's exact order (the same rows for undirected graphs), with
/// weights aligned to each (`None` = all 1). NaN weights hide edges. One
/// counter breaks heap ties across both directions, as in NetworkX.
pub fn bidirectional_dijkstra(
    adj: [(&Csr, Option<&[f64]>); 2],
    n: usize,
    source: usize,
    target: usize,
) -> Result<(f64, Vec<u32>), BidirectionalError> {
    let mut dist = [vec![f64::NAN; n], vec![f64::NAN; n]];
    let mut done = [vec![false; n], vec![false; n]];
    let mut seen = [vec![f64::INFINITY; n], vec![f64::INFINITY; n]];
    let mut has_seen = [vec![false; n], vec![false; n]];
    let mut pred = [vec![NO_PARENT; n], vec![NO_PARENT; n]];
    let mut fringe: [BinaryHeap<Reverse<(HeapKey, u32)>>; 2] =
        [BinaryHeap::new(), BinaryHeap::new()];
    let mut counter = 0u64;
    for (d, s) in [(0, source), (1, target)] {
        seen[d][s] = 0.0;
        has_seen[d][s] = true;
        fringe[d].push(Reverse((HeapKey(0.0, counter), s as u32)));
        counter += 1;
    }
    let mut finaldist: Option<f64> = None;
    let mut meet = NO_PARENT;
    let mut dir = 1;
    while !fringe[0].is_empty() && !fringe[1].is_empty() {
        dir = 1 - dir;
        let Reverse((HeapKey(d, _), v)) = fringe[dir].pop().expect("non-empty");
        let v = v as usize;
        if done[dir][v] {
            continue;
        }
        done[dir][v] = true;
        dist[dir][v] = d;
        if done[1 - dir][v] {
            let meet = meet as usize;
            let mut path = Vec::new();
            let mut cur = meet as u32;
            while cur != NO_PARENT {
                path.push(cur);
                cur = pred[0][cur as usize];
            }
            path.reverse();
            let mut cur = pred[1][meet];
            while cur != NO_PARENT {
                path.push(cur);
                cur = pred[1][cur as usize];
            }
            return Ok((finaldist.expect("set when both searches met"), path));
        }
        let (rows, weights) = adj[dir];
        for e in rows.range(v) {
            let w = rows.targets[e] as usize;
            let cost = weights.map_or(1.0, |ws| ws[e]);
            if cost.is_nan() {
                continue;
            }
            let vw = d + cost;
            if done[dir][w] {
                if vw < dist[dir][w] {
                    return Err(BidirectionalError::Contradictory);
                }
            } else if !has_seen[dir][w] || vw < seen[dir][w] {
                seen[dir][w] = vw;
                has_seen[dir][w] = true;
                fringe[dir].push(Reverse((HeapKey(vw, counter), w as u32)));
                counter += 1;
                pred[dir][w] = v as u32;
                if has_seen[1 - dir][w] {
                    let total = vw + seen[1 - dir][w];
                    if finaldist.is_none_or(|f| f > total) {
                        finaldist = Some(total);
                        meet = w as u32;
                    }
                }
            }
        }
    }
    Err(BidirectionalError::NoPath)
}

/// `nx.predecessor(G, source)`: every node reached by BFS with all its
/// predecessors one level closer, in discovery order (`None` = unreached).
pub fn bfs_predecessors(adj: &Csr, n: usize, source: usize) -> Vec<Option<Vec<u32>>> {
    let mut level = vec![u32::MAX; n];
    let mut pred: Vec<Option<Vec<u32>>> = vec![None; n];
    level[source] = 0;
    pred[source] = Some(Vec::new());
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
                    pred[wi] = Some(vec![v]);
                    next.push(w);
                } else if level[wi] == depth {
                    pred[wi].as_mut().expect("seen").push(v);
                }
            }
        }
        this_level = next;
    }
    pred
}

/// `nx.dijkstra_predecessor_and_distance(G, source)`'s predecessor lists,
/// including equal-length predecessors found after a node is finalized.
/// NaN weights hide edges.
pub fn dijkstra_predecessors(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    source: usize,
) -> Result<Vec<Option<Vec<u32>>>, NegativeCycle> {
    let mut done = vec![false; n];
    let mut dist = vec![0.0; n];
    let mut seen = vec![f64::INFINITY; n];
    let mut has_seen = vec![false; n];
    let mut pred: Vec<Option<Vec<u32>>> = vec![None; n];
    let mut heap: BinaryHeap<Reverse<(HeapKey, u32)>> = BinaryHeap::new();
    let mut counter = 0u64;
    seen[source] = 0.0;
    has_seen[source] = true;
    pred[source] = Some(Vec::new());
    heap.push(Reverse((HeapKey(0.0, counter), source as u32)));
    while let Some(Reverse((HeapKey(d, _), v))) = heap.pop() {
        let v = v as usize;
        if done[v] {
            continue;
        }
        done[v] = true;
        dist[v] = d;
        for e in adj.range(v) {
            let u = adj.targets[e] as usize;
            let cost = weights.map_or(1.0, |w| w[e]);
            if cost.is_nan() {
                continue;
            }
            let vu = d + cost;
            if done[u] {
                if vu < dist[u] {
                    return Err(NegativeCycle);
                } else if vu == dist[u] {
                    pred[u].as_mut().expect("seen").push(v as u32);
                }
            } else if !has_seen[u] || vu < seen[u] {
                seen[u] = vu;
                has_seen[u] = true;
                counter += 1;
                heap.push(Reverse((HeapKey(vu, counter), u as u32)));
                pred[u] = Some(vec![v as u32]);
            } else if vu == seen[u] {
                pred[u].as_mut().expect("seen").push(v as u32);
            }
        }
    }
    Ok(pred)
}

/// `_build_paths_from_predecessors({source}, target, pred)` as a resumable
/// search: each `next_path` call returns the next path (source first).
pub struct PathsFromPreds {
    pred: Vec<Option<Vec<u32>>>,
    source: u32,
    path: Vec<u32>,
    next_pred: Vec<usize>,
    on_path: Vec<bool>,
    /// The current pass already yielded; resume with its step.
    yielded: bool,
    /// NetworkX before 3.7: skipping a predecessor already on the path ends
    /// the pass, so the next pass re-checks (and re-yields at) the source.
    skip_ends_pass: bool,
}

impl PathsFromPreds {
    /// `None` if `target` wasn't reached.
    pub fn new(
        pred: Vec<Option<Vec<u32>>>,
        source: u32,
        target: u32,
        skip_ends_pass: bool,
    ) -> Option<Self> {
        pred[target as usize].as_ref()?;
        let mut on_path = vec![false; pred.len()];
        on_path[target as usize] = true;
        Some(PathsFromPreds {
            pred,
            source,
            path: vec![target],
            next_pred: vec![0],
            on_path,
            yielded: false,
            skip_ends_pass,
        })
    }

    pub fn next_path(&mut self) -> Option<Vec<u32>> {
        loop {
            let node = *self.path.last()?;
            // NetworkX checks at the top of every pass, including when it
            // returns to a node after exploring below it.
            if !self.yielded && node == self.source {
                self.yielded = true;
                return Some(self.path.iter().rev().copied().collect());
            }
            self.yielded = false;
            let preds = self.pred[node as usize].as_deref().unwrap_or(&[]);
            let i = self.next_pred.last_mut().expect("parallel to path");
            let mut pushed = None;
            let mut skipped = false;
            while *i < preds.len() {
                let p = preds[*i];
                *i += 1;
                if !self.on_path[p as usize] {
                    pushed = Some(p);
                    break;
                }
                if self.skip_ends_pass {
                    skipped = true;
                    break;
                }
            }
            if skipped {
                continue;
            }
            match pushed {
                Some(p) => {
                    self.path.push(p);
                    self.next_pred.push(0);
                    self.on_path[p as usize] = true;
                }
                None => {
                    self.on_path[node as usize] = false;
                    self.path.pop();
                    self.next_pred.pop();
                }
            }
        }
    }
}
