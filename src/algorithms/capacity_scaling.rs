//! The phases of NetworkX's `capacity_scaling` on its own residual network
//! (built by NetworkX), with Python's int arithmetic (`i128`; the caller
//! bounds the inputs), `BinaryHeap`'s order (value, then insertion count)
//! and `arbitrary_element(S)` from the set `S` in CPython's order.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::pyset::PySet;

/// The residual network as NetworkX iterates it: per node, its neighbor
/// groups `(v, first edge, end)` in `R.succ[u]` order, each group's edges
/// in key order; per edge its capacity, weight and reverse edge.
pub struct Residual {
    pub node_groups: Vec<std::ops::Range<usize>>,
    pub groups: Vec<(u32, usize, usize)>,
    pub cap: Vec<i128>,
    pub weight: Vec<i128>,
    pub partner: Vec<u32>,
}

/// Runs every phase from `delta`; returns each edge's flow and whether all
/// excesses ended at zero.
pub fn run(
    r: &Residual,
    mut excess: Vec<i128>,
    mut delta: i128,
    hashes: &[i64],
) -> (Vec<i128>, bool) {
    let n = excess.len();
    let mut flow = vec![0i128; r.cap.len()];
    let mut potential = vec![0i128; n];
    let mut in_t = vec![false; n];
    let mut dist: Vec<Option<i128>> = vec![None; n];
    let mut pred: Vec<Option<(u32, u32)>> = vec![None; n];
    let mut heap_val: Vec<Option<i128>> = vec![None; n];
    while delta >= 1 {
        for u in 0..n {
            let p_u = potential[u];
            for &(v, a, b) in &r.groups[r.node_groups[u].clone()] {
                for e in a..b {
                    if r.weight[e] - p_u + potential[v as usize] < 0 {
                        let f = r.cap[e] - flow[e];
                        if f >= delta {
                            flow[e] += f;
                            flow[r.partner[e] as usize] -= f;
                            excess[u] -= f;
                            excess[v as usize] += f;
                        }
                    }
                }
            }
        }
        let mut s_set = PySet::default();
        let mut t_count = 0usize;
        for u in 0..n {
            in_t[u] = false;
            if excess[u] >= delta {
                s_set.add(u as u32, hashes);
            } else if excess[u] <= -delta {
                in_t[u] = true;
                t_count += 1;
            }
        }
        while !s_set.is_empty() && t_count > 0 {
            let s = s_set.first().expect("S is not empty") as usize;
            // Dijkstra with NetworkX's BinaryHeap.
            let mut visited: Vec<usize> = Vec::new();
            let mut touched: Vec<usize> = vec![s];
            let mut heap = BinaryHeap::new();
            let mut count: u64 = 0;
            heap_val[s] = Some(0);
            heap.push(Reverse((0i128, count, s as u32)));
            count += 1;
            let mut live = 1usize;
            let mut t = None;
            let mut u = s;
            while live > 0 {
                let (d_u, x) = loop {
                    let Reverse((val, _, key)) = heap.pop().expect("a live entry");
                    if heap_val[key as usize] == Some(val) {
                        break (val, key as usize);
                    }
                };
                heap_val[x] = None;
                live -= 1;
                u = x;
                dist[u] = Some(d_u);
                visited.push(u);
                if in_t[u] {
                    t = Some(u);
                    break;
                }
                let p_u = potential[u];
                for &(v, a, b) in &r.groups[r.node_groups[u].clone()] {
                    let v = v as usize;
                    if dist[v].is_some() {
                        continue;
                    }
                    let mut best: Option<(i128, usize)> = None;
                    for e in a..b {
                        if r.cap[e] - flow[e] >= delta {
                            let w = r.weight[e];
                            if best.is_none_or(|(bw, _)| w < bw) {
                                best = Some((w, e));
                            }
                        }
                    }
                    let Some((wmin, emin)) = best else { continue };
                    let d_v = d_u + wmin - p_u + potential[v];
                    let inserted = match heap_val[v] {
                        None => {
                            touched.push(v);
                            live += 1;
                            true
                        }
                        Some(old) => d_v < old,
                    };
                    if inserted {
                        heap_val[v] = Some(d_v);
                        heap.push(Reverse((d_v, count, v as u32)));
                        count += 1;
                        pred[v] = Some((u as u32, emin as u32));
                    }
                }
            }
            if let Some(t) = t {
                let mut x = u;
                while x != s {
                    let (pu, e) = pred[x].expect("a path back to s");
                    flow[e as usize] += delta;
                    flow[r.partner[e as usize] as usize] -= delta;
                    x = pu as usize;
                }
                excess[s] -= delta;
                excess[t] += delta;
                if excess[s] < delta {
                    s_set.discard(s as u32, hashes);
                }
                if excess[t] > -delta && in_t[t] {
                    in_t[t] = false;
                    t_count -= 1;
                }
                let d_t = dist[t].expect("t was reached");
                for &x in &visited {
                    potential[x] -= dist[x].expect("visited") - d_t;
                }
            } else {
                s_set.discard(s as u32, hashes);
            }
            for &x in &visited {
                dist[x] = None;
            }
            for &x in &touched {
                heap_val[x] = None;
                pred[x] = None;
            }
        }
        delta /= 2;
    }
    let feasible = excess.iter().all(|&x| x == 0);
    (flow, feasible)
}
