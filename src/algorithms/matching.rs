//! Batch 10: triads, d-separation, degree sequences, boundaries, matching,
//! dominating sets and cliques. Each function ports NetworkX's code closely
//! enough to visit nodes and edges (and break ties) in the same order.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::hash::BuildHasherDefault;

use crate::algorithms::structure_more::PairHasher;
use crate::graph::Csr;

type PairSet = HashSet<u64, BuildHasherDefault<PairHasher>>;

#[inline]
fn pair(u: usize, v: usize) -> u64 {
    ((u as u64) << 32) | v as u64
}

// --- Triadic census -----------------------------------------------------------

/// `TRICODES` from `networkx.algorithms.triads`: triad code -> type number.
const TRICODES: [u8; 64] = [
    1, 2, 2, 3, 2, 4, 6, 8, 2, 6, 5, 7, 3, 8, 7, 11, 2, 6, 4, 8, 5, 9, 9, 13, 6, 10, 9, 14, 7, 14,
    12, 15, 2, 5, 6, 7, 6, 9, 10, 14, 4, 9, 9, 12, 8, 13, 14, 15, 3, 7, 8, 11, 7, 12, 14, 15, 8,
    14, 13, 15, 11, 15, 15, 16,
];
const T012: usize = 1;
const T102: usize = 2;

/// Each node's neighbors (predecessors or successors, each once, the node
/// itself for a self-loop) with flags: 1 = successor, 2 = predecessor.
fn flagged_neighbors(succ: &Csr, pred: &Csr, n: usize) -> Vec<Vec<(u32, u8)>> {
    let mut slot = vec![u32::MAX; n];
    (0..n)
        .map(|v| {
            let mut row: Vec<(u32, u8)> = Vec::new();
            for &w in succ.neighbors(v) {
                slot[w as usize] = row.len() as u32;
                row.push((w, 1));
            }
            for &w in pred.neighbors(v) {
                let s = slot[w as usize];
                if s != u32::MAX && (s as usize) < row.len() && row[s as usize].0 == w {
                    row[s as usize].1 |= 2;
                } else {
                    row.push((w, 2));
                }
            }
            for &(w, _) in &row {
                slot[w as usize] = u32::MAX;
            }
            row
        })
        .collect()
}

/// `nx.triadic_census` counts, in `TRIAD_NAMES` order. `nodeset` lists the
/// counted nodes (all of them without a `nodelist`). NetworkX numbers nodes
/// in set order; its counts don't depend on that order.
pub fn triadic_census(succ: &Csr, pred: &Csr, n: usize, nodeset: &[u32]) -> [i128; 16] {
    let nb = flagged_neighbors(succ, pred, n);
    let big_n = n as i128;
    let mut m = vec![0usize; n];
    let mut in_set = vec![false; n];
    for (i, &v) in nodeset.iter().enumerate() {
        m[v as usize] = i;
        in_set[v as usize] = true;
    }
    let nnot = n - nodeset.len();
    let mut k = 0;
    for v in 0..n {
        if !in_set[v] {
            m[v] = k + n;
            k += 1;
        }
    }
    let (mut sgl_edges_outside, mut dbl_edges_outside) = (0i128, 0i128);
    if nnot > 0 {
        let (mut sgl, mut dbl) = (0i128, 0i128);
        for v in (0..n).filter(|&v| !in_set[v]) {
            for &(w, f) in &nb[v] {
                if !in_set[w as usize] {
                    if f == 3 {
                        dbl += 1;
                    } else {
                        sgl += 1;
                    }
                }
            }
        }
        sgl_edges_outside = sgl / 2;
        dbl_edges_outside = dbl / 2;
    }
    let mut census = [0i128; 16];
    let mut vmark = vec![u32::MAX; n];
    let mut vflag = vec![0u8; n];
    let mut umark = vec![u32::MAX; n];
    let mut ustamp = 0u32;
    let mut uflag = vec![0u8; n];
    for &v in nodeset {
        let v = v as usize;
        for &(w, f) in &nb[v] {
            vmark[w as usize] = v as u32;
            vflag[w as usize] = f;
        }
        let in_v = |w: usize, vmark: &[u32]| vmark[w] == v as u32;
        let (mut sgl_bdy, mut sgl_out, mut dbl_bdy, mut dbl_out) = (0i128, 0i128, 0i128, 0i128);
        for &(u, fu) in &nb[v] {
            let u = u as usize;
            if m[u] <= m[v] {
                continue;
            }
            ustamp = ustamp.wrapping_add(1);
            for &(w, f) in &nb[u] {
                umark[w as usize] = ustamp;
                uflag[w as usize] = f;
            }
            let mut len = 0i128;
            let mut visit = |w: usize, census: &mut [i128; 16]| {
                len += 1;
                let in_vn = in_v(w, &vmark);
                if m[u] < m[w] || (m[v] < m[w] && m[w] < m[u] && !in_vn) {
                    let (fv, fu_w) = (
                        if in_vn { vflag[w] } else { 0 },
                        if umark[w] == ustamp { uflag[w] } else { 0 },
                    );
                    let code = (fu & 1) as usize
                        | (((fu >> 1) & 1) as usize) << 1
                        | ((fv & 1) as usize) << 2
                        | (((fv >> 1) & 1) as usize) << 3
                        | ((fu_w & 1) as usize) << 4
                        | (((fu_w >> 1) & 1) as usize) << 5;
                    census[TRICODES[code] as usize - 1] += 1;
                }
            };
            for &(w, _) in &nb[u] {
                let w = w as usize;
                if w != u && w != v {
                    visit(w, &mut census);
                }
            }
            for &(w, _) in &nb[v] {
                let w = w as usize;
                if w != u && w != v && umark[w] != ustamp {
                    visit(w, &mut census);
                }
            }
            let rest = big_n - len - 2;
            if fu == 3 {
                census[T102] += rest;
            } else {
                census[T012] += rest;
            }
            if nnot > 0 && !in_set[u] {
                for &(w, f) in &nb[u] {
                    let w = w as usize;
                    if in_set[w] {
                        continue;
                    }
                    let bdy = in_v(w, &vmark);
                    match (f == 3, bdy) {
                        (false, true) => sgl_bdy += 1,
                        (false, false) => sgl_out += 1,
                        (true, true) => dbl_bdy += 1,
                        (true, false) => dbl_out += 1,
                    }
                }
            }
        }
        if nnot > 0 {
            census[T012] += sgl_edges_outside - (sgl_out + sgl_bdy.div_euclid(2));
            census[T102] += dbl_edges_outside - (dbl_out + dbl_bdy.div_euclid(2));
        }
    }
    let nn = nnot as i128;
    let total = big_n * (big_n - 1) * (big_n - 2) / 6 - nn * (nn - 1) * (nn - 2) / 6;
    let counted: i128 = census.iter().sum();
    census[0] = total - counted;
    census
}

// --- d-separation -------------------------------------------------------------

/// Flags for the nodes in `nodes` and everything they reach along `adj`
/// (`nodes` themselves included).
pub fn closure_mask(adj: &Csr, n: usize, nodes: &[u32]) -> Vec<bool> {
    let mut mask = vec![false; n];
    let mut stack: Vec<u32> = Vec::new();
    for &v in nodes {
        if !mask[v as usize] {
            mask[v as usize] = true;
            stack.push(v);
        }
    }
    while let Some(v) = stack.pop() {
        for &w in adj.neighbors(v as usize) {
            if !mask[w as usize] {
                mask[w as usize] = true;
                stack.push(w);
            }
        }
    }
    mask
}

fn mask_of(n: usize, nodes: &[u32]) -> Vec<bool> {
    let mut mask = vec![false; n];
    for &v in nodes {
        mask[v as usize] = true;
    }
    mask
}

/// `nx.is_d_separator` on a DAG (after NetworkX's input checks).
pub fn is_d_separator(succ: &Csr, pred: &Csr, n: usize, x: &[u32], y: &[u32], z: &[u32]) -> bool {
    let in_y = mask_of(n, y);
    let in_z = mask_of(n, z);
    let mut anc_or_z = closure_mask(pred, n, x);
    for &v in z {
        anc_or_z[v as usize] = true;
    }
    let mut forward: VecDeque<u32> = VecDeque::new();
    let mut backward: VecDeque<u32> = x.iter().copied().collect();
    let mut fvis = vec![false; n];
    let mut bvis = vec![false; n];
    while !forward.is_empty() || !backward.is_empty() {
        if let Some(node) = backward.pop_front() {
            let v = node as usize;
            // Processing a node twice changes nothing; NetworkX does, but
            // reaches the same nodes.
            let again = bvis[v];
            bvis[v] = true;
            if in_y[v] {
                return false;
            }
            if in_z[v] {
                continue;
            }
            if !again {
                backward.extend(pred.neighbors(v).iter().filter(|&&p| !bvis[p as usize]));
                forward.extend(succ.neighbors(v).iter().filter(|&&s| !fvis[s as usize]));
            }
        }
        if let Some(node) = forward.pop_front() {
            let v = node as usize;
            let again = fvis[v];
            fvis[v] = true;
            if in_y[v] {
                return false;
            }
            if !again {
                if anc_or_z[v] {
                    backward.extend(pred.neighbors(v).iter().filter(|&&p| !bvis[p as usize]));
                }
                if !in_z[v] {
                    forward.extend(succ.neighbors(v).iter().filter(|&&s| !fvis[s as usize]));
                }
            }
        }
    }
    true
}

/// `nx.ancestors(G, v)` in insertion order: BFS discovery order along
/// `pred` (NetworkX's own in-edge order), without `v`.
pub fn ancestors_in_order(
    pred: &Csr,
    n: usize,
    v: u32,
    seen: &mut Vec<u32>,
    stamp: u32,
) -> Vec<u32> {
    if seen.len() < n {
        seen.resize(n, u32::MAX);
    }
    seen[v as usize] = stamp;
    let mut order = vec![v];
    let mut i = 0;
    while i < order.len() {
        let u = order[i] as usize;
        i += 1;
        for &w in pred.neighbors(u) {
            if seen[w as usize] != stamp {
                seen[w as usize] = stamp;
                order.push(w);
            }
        }
    }
    order.remove(0);
    order
}

/// `_reachable(G, x, a, z)` from `networkx.algorithms.d_separation`: the
/// second element of each entry of `processed`, in order (repeats kept).
pub fn d_reachable(
    succ: &Csr,
    pred: &Csr,
    n: usize,
    x: &[u32],
    a: &[bool],
    z: &[bool],
) -> Vec<u32> {
    // Entry `(f, v)` has index 2 * v + f.
    let mut processed = vec![false; 2 * n];
    let mut order: Vec<(bool, u32)> = Vec::new();
    for &v in x {
        if !pred.neighbors(v as usize).is_empty() {
            order.push((true, v));
        }
        if !succ.neighbors(v as usize).is_empty() {
            order.push((false, v));
        }
    }
    // `processed = queue.copy()`, repeats and all.
    for &(f, v) in &order {
        processed[2 * v as usize + f as usize] = true;
    }
    let mut head = 0;
    while head < order.len() {
        let (e, v) = order[head];
        head += 1;
        let preds = pred.neighbors(v as usize).iter().map(|&w| (false, w));
        let succs = succ.neighbors(v as usize).iter().map(|&w| (true, w));
        for (f, w) in preds.chain(succs) {
            let key = 2 * w as usize + f as usize;
            if !processed[key] && a[w as usize] && (!z[v as usize] || (e && !f)) {
                processed[key] = true;
                order.push((f, w));
            }
        }
    }
    order.into_iter().map(|(_, w)| w).collect()
}

/// `nx.is_minimal_d_separator` on a DAG after its input checks.
pub fn is_minimal_d_separator(
    succ: &Csr,
    pred: &Csr,
    n: usize,
    x: &[u32],
    y: &[u32],
    z: &[u32],
    included: &[u32],
) -> bool {
    let nodeset: Vec<u32> = x.iter().chain(y).chain(included).copied().collect();
    let anc = closure_mask(pred, n, &nodeset);
    let in_z = mask_of(n, z);
    let x_closure = mask_of(n, &d_reachable(succ, pred, n, x, &anc, &in_z));
    if y.iter().any(|&v| x_closure[v as usize]) {
        return false;
    }
    if z.iter().any(|&v| !anc[v as usize]) {
        return false;
    }
    let y_closure = mask_of(n, &d_reachable(succ, pred, n, y, &anc, &in_z));
    let in_included = mask_of(n, included);
    z.iter()
        .all(|&v| in_included[v as usize] || (x_closure[v as usize] && y_closure[v as usize]))
}

// --- Degree sequences ---------------------------------------------------------

/// `_basic_graphical_tests`: `(dmax, dmin, n, num_degs)`, or `None` where
/// NetworkX raises `NetworkXUnfeasible`.
fn basic_graphical_tests(seq: &[i64]) -> Option<(i64, i64, i64, Vec<i64>)> {
    let p = seq.len() as i64;
    let mut num_degs = vec![0i64; seq.len()];
    let (mut dmax, mut dmin, mut dsum, mut n) = (0i64, p, 0i64, 0i64);
    for &d in seq {
        if d < 0 || d >= p {
            return None;
        }
        if d > 0 {
            dmax = dmax.max(d);
            dmin = dmin.min(d);
            dsum += d;
            n += 1;
            num_degs[d as usize] += 1;
        }
    }
    if dsum % 2 != 0 || dsum > n * (n - 1) {
        return None;
    }
    Some((dmax, dmin, n, num_degs))
}

/// The Zverovich-Zverovich shortcut both tests take first.
fn zz_condition(n: i64, dmin: i64, dmax: i64) -> bool {
    n == 0 || 4 * dmin * n >= (dmax + dmin + 1) * (dmax + dmin + 1)
}

pub fn is_valid_degree_sequence_havel_hakimi(seq: &[i64]) -> bool {
    let Some((mut dmax, dmin, mut n, mut num_degs)) = basic_graphical_tests(seq) else {
        return false;
    };
    if zz_condition(n, dmin, dmax) {
        return true;
    }
    let mut modstubs = vec![0i64; dmax as usize + 1];
    while n > 0 {
        while num_degs[dmax as usize] == 0 {
            dmax -= 1;
        }
        if dmax > n - 1 {
            return false;
        }
        num_degs[dmax as usize] -= 1;
        n -= 1;
        let mut mslen = 0;
        let mut k = dmax;
        for _ in 0..dmax {
            while num_degs[k as usize] == 0 {
                k -= 1;
            }
            num_degs[k as usize] -= 1;
            n -= 1;
            if k > 1 {
                modstubs[mslen] = k - 1;
                mslen += 1;
            }
        }
        for &stub in &modstubs[..mslen] {
            num_degs[stub as usize] += 1;
            n += 1;
        }
    }
    true
}

pub fn is_valid_degree_sequence_erdos_gallai(seq: &[i64]) -> bool {
    let Some((dmax, dmin, n, num_degs)) = basic_graphical_tests(seq) else {
        return false;
    };
    if zz_condition(n, dmin, dmax) {
        return true;
    }
    let (mut k, mut sum_deg, mut sum_nj, mut sum_jnj) = (0i64, 0i64, 0i64, 0i64);
    let mut dk = dmax;
    while dk >= dmin {
        if dk < k + 1 {
            return true;
        }
        if num_degs[dk as usize] > 0 {
            let mut run_size = num_degs[dk as usize];
            if dk < k + run_size {
                run_size = dk - k;
            }
            sum_deg += run_size * dk;
            for v in 0..run_size {
                sum_nj += num_degs[(k + v) as usize];
                sum_jnj += (k + v) * num_degs[(k + v) as usize];
            }
            k += run_size;
            if sum_deg > k * (n - 1) - k * sum_nj + sum_jnj {
                return false;
            }
        }
        dk -= 1;
    }
    true
}

pub fn is_multigraphical(seq: &[i64]) -> bool {
    let (mut dsum, mut dmax) = (0i128, 0i128);
    for &d in seq {
        if d < 0 {
            return false;
        }
        dsum += d as i128;
        dmax = dmax.max(d as i128);
    }
    !(dsum % 2 != 0 || dsum < 2 * dmax)
}

/// `nx.is_digraphical`, or `None` where NetworkX would allocate a list of
/// `max(in_sequence) + 1` items too large to be worth matching.
pub fn is_digraphical(ins: &[i64], outs: &[i64]) -> Option<bool> {
    let (nin, nout) = (ins.len(), outs.len());
    let maxn = nin.max(nout);
    if maxn == 0 {
        return Some(true);
    }
    let (mut sumin, mut sumout, mut maxin) = (0i128, 0i128, 0i64);
    let mut stubheap: BinaryHeap<Reverse<(i64, i64)>> = BinaryHeap::new();
    let mut zeroheap: BinaryHeap<Reverse<i64>> = BinaryHeap::new();
    for i in 0..maxn {
        let out_deg = if i < nout { outs[i] } else { 0 };
        let in_deg = if i < nin { ins[i] } else { 0 };
        if in_deg < 0 || out_deg < 0 {
            return Some(false);
        }
        sumin += in_deg as i128;
        sumout += out_deg as i128;
        maxin = maxin.max(in_deg);
        if in_deg > 0 {
            stubheap.push(Reverse((-out_deg, -in_deg)));
        } else if out_deg > 0 {
            zeroheap.push(Reverse(-out_deg));
        }
    }
    if sumin != sumout {
        return Some(false);
    }
    if maxin > 1 << 24 {
        return None;
    }
    let mut modstubs = vec![(0i64, 0i64); maxin as usize + 1];
    while let Some(Reverse((freeout, freein))) = stubheap.pop() {
        let freein = -freein;
        if freein > (stubheap.len() + zeroheap.len()) as i64 {
            return Some(false);
        }
        let mut mslen = 0;
        for _ in 0..freein {
            let (stubout, stubin) = match (zeroheap.peek(), stubheap.peek()) {
                (Some(&Reverse(z)), s) if s.is_none_or(|&Reverse((o, _))| o > z) => {
                    zeroheap.pop();
                    (z, 0)
                }
                _ => {
                    let Reverse(item) = stubheap.pop().expect("checked above");
                    item
                }
            };
            if stubout == 0 {
                return Some(false);
            }
            if stubout + 1 < 0 || stubin < 0 {
                modstubs[mslen] = (stubout + 1, stubin);
                mslen += 1;
            }
        }
        for &stub in &modstubs[..mslen] {
            if stub.1 < 0 {
                stubheap.push(Reverse(stub));
            } else {
                zeroheap.push(Reverse(stub.0));
            }
        }
        if freeout < 0 {
            zeroheap.push(Reverse(freeout));
        }
    }
    Some(true)
}

// --- Boundaries ---------------------------------------------------------------

/// Neighbors of `nodes` (in that order), each once, first occurrence first:
/// the insertion order of `set(chain.from_iterable(G[v] for v in nodes))`.
pub fn neighbor_union(adj: &Csr, n: usize, nodes: &[u32]) -> Vec<u32> {
    let mut seen = vec![false; n];
    let mut out = Vec::new();
    for &v in nodes {
        for &w in adj.neighbors(v as usize) {
            if !seen[w as usize] {
                seen[w as usize] = true;
                out.push(w);
            }
        }
    }
    out
}

/// `nx.edge_boundary(G, nbunch1, nbunch2)` without data: `G.edges(nset1)`
/// with `nset1` in iteration order `order`, filtered. Returns pairs
/// `(index into order, neighbor)`.
pub fn edge_boundary(
    adj: &Csr,
    n: usize,
    directed: bool,
    order: &[u32],
    nset2: Option<&[u32]>,
) -> Vec<(u32, u32)> {
    let in1 = mask_of(n, order);
    let in2 = nset2.map(|s| mask_of(n, s));
    // `EdgeDataView` skips neighbors already reported as sources.
    let mut seen = vec![false; n];
    let mut out = Vec::new();
    for (i, &u) in order.iter().enumerate() {
        for &v in adj.neighbors(u as usize) {
            if !directed && seen[v as usize] {
                continue;
            }
            let keep = match &in2 {
                None => !in1[v as usize],
                Some(in2) => in2[v as usize] || (in1[v as usize] && in2[u as usize]),
            };
            if keep {
                out.push((i as u32, v));
            }
        }
        seen[u as usize] = true;
    }
    out
}

// --- Matching -------------------------------------------------------------------

/// Index of the first pair at which `is_matching` returns False: a
/// self-loop, a non-edge, or a node matched twice.
pub fn first_matching_failure(adj: &Csr, n: usize, us: &[u32], vs: &[u32]) -> Option<usize> {
    let mut used = vec![false; n];
    for (i, (&u, &v)) in us.iter().zip(vs).enumerate() {
        let (u, v) = (u as usize, v as usize);
        if u == v || !adj.neighbors(u).contains(&(v as u32)) || used[u] || used[v] {
            return Some(i);
        }
        used[u] = true;
        used[v] = true;
    }
    None
}

/// Whether some edge (not a self-loop) has neither end in `matched`.
pub fn has_unmatched_edge(adj: &Csr, n: usize, matched: &[u32]) -> bool {
    let used = mask_of(n, matched);
    (0..n).any(|u| {
        !used[u]
            && adj
                .neighbors(u)
                .iter()
                .any(|&v| v as usize != u && !used[v as usize])
    })
}

/// `nx.maximal_matching`: the chosen edges in `G.edges()` order.
pub fn maximal_matching(adj: &Csr, n: usize) -> Vec<(u32, u32)> {
    let mut used = vec![false; n];
    let mut out = Vec::new();
    for u in 0..n {
        for &v in adj.neighbors(u) {
            let vi = v as usize;
            if vi < u {
                continue; // reported from v already
            }
            if !used[u] && !used[vi] && vi != u {
                out.push((u as u32, v));
                used[u] = true;
                used[vi] = true;
            }
        }
    }
    out
}

/// An undirected graph with per-entry weights, rows in neighbor order.
pub struct WeightedAdj {
    pub offsets: Vec<usize>,
    pub targets: Vec<u32>,
    pub weights: Vec<f64>,
}

impl WeightedAdj {
    pub fn from_csr(adj: &Csr, weights: Option<&[f64]>) -> WeightedAdj {
        WeightedAdj {
            offsets: adj.offsets.clone(),
            targets: adj.targets.clone(),
            weights: weights.map_or_else(|| vec![1.0; adj.targets.len()], |w| w.to_vec()),
        }
    }

    fn row(&self, v: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let r = self.offsets[v]..self.offsets[v + 1];
        self.targets[r.clone()]
            .iter()
            .zip(&self.weights[r])
            .map(|(&w, &x)| (w as usize, x))
    }
}

/// `min_weight_matching`'s `InvG`: nodes in order of first appearance in
/// `G.edges`, weights `1 + max(w) - w`. Returns it and each node's position
/// in G.
pub fn inverted_graph(adj: &Csr, n: usize, weights: Option<&[f64]>) -> (WeightedAdj, Vec<u32>) {
    let mut edges = Vec::new();
    for u in 0..n {
        for e in adj.range(u) {
            let v = adj.targets[e] as usize;
            if v >= u {
                edges.push((u, v, weights.map_or(1.0, |w| w[e])));
            }
        }
    }
    // Python's `max`: replace only on a strictly greater value.
    let mut max_w = edges.first().map_or(f64::NAN, |e| e.2);
    for &(_, _, w) in &edges {
        if w > max_w {
            max_w = w;
        }
    }
    let max_weight = 1.0 + max_w;
    let mut pos = vec![u32::MAX; n];
    let mut nodes: Vec<u32> = Vec::new();
    let mut rows: Vec<Vec<(u32, f64)>> = Vec::new();
    let mut index = |v: usize, nodes: &mut Vec<u32>, rows: &mut Vec<Vec<(u32, f64)>>| {
        if pos[v] == u32::MAX {
            pos[v] = nodes.len() as u32;
            nodes.push(v as u32);
            rows.push(Vec::new());
        }
        pos[v] as usize
    };
    for &(u, v, w) in &edges {
        let a = index(u, &mut nodes, &mut rows);
        let b = index(v, &mut nodes, &mut rows);
        let x = max_weight - w;
        rows[a].push((b as u32, x));
        if a != b {
            rows[b].push((a as u32, x));
        }
    }
    let mut offsets = vec![0];
    let (mut targets, mut ws) = (Vec::new(), Vec::new());
    for row in rows {
        for (t, x) in row {
            targets.push(t);
            ws.push(x);
        }
        offsets.push(targets.len());
    }
    (
        WeightedAdj {
            offsets,
            targets,
            weights: ws,
        },
        nodes,
    )
}

const NONE: usize = usize::MAX;

type Edge = (usize, usize, f64);

/// `nx.max_weight_matching`, ported line by line (Galil's O(n^3) blossom
/// algorithm). Vertices are `0..n`; blossoms take ids `n..2n`, reused once
/// expanded, with `live` keeping NetworkX's dict order (creation order).
struct Blossoms<'a> {
    g: &'a WeightedAdj,
    n: usize,
    maxcardinality: bool,
    mate: Vec<usize>,
    mate_order: Vec<usize>,
    label: Vec<u8>,
    labeledge: Vec<Option<(usize, usize)>>,
    inblossom: Vec<usize>,
    parent: Vec<usize>,
    base: Vec<usize>,
    bestedge: Vec<Option<Edge>>,
    dualvar: Vec<f64>,
    bdual: Vec<f64>,
    allowedge: PairSet,
    queue: Vec<usize>,
    childs: Vec<Vec<usize>>,
    edges: Vec<Vec<(usize, usize)>>,
    mybestedges: Vec<Option<Vec<Edge>>>,
    free: Vec<usize>,
    live: Vec<usize>,
    is_live: Vec<bool>,
}

/// Python list indexing (negative indices count from the end).
#[inline]
fn at<T: Copy>(list: &[T], j: isize) -> T {
    if j < 0 {
        list[(list.len() as isize + j) as usize]
    } else {
        list[j as usize]
    }
}

impl<'a> Blossoms<'a> {
    fn new(g: &'a WeightedAdj, maxcardinality: bool) -> Self {
        let n = g.offsets.len() - 1;
        let mut maxweight = 0.0f64;
        for v in 0..n {
            for (w, x) in g.row(v) {
                if w != v && x > maxweight {
                    maxweight = x;
                }
            }
        }
        let ids = 2 * n + 1;
        Blossoms {
            g,
            n,
            maxcardinality,
            mate: vec![NONE; n],
            mate_order: Vec::new(),
            label: vec![0; ids],
            labeledge: vec![None; ids],
            inblossom: (0..n).collect(),
            parent: vec![NONE; ids],
            base: (0..ids).map(|i| if i < n { i } else { NONE }).collect(),
            bestedge: vec![None; ids],
            dualvar: vec![maxweight; n],
            bdual: vec![0.0; ids],
            allowedge: PairSet::default(),
            queue: Vec::new(),
            childs: vec![Vec::new(); ids],
            edges: vec![Vec::new(); ids],
            mybestedges: vec![None; ids],
            free: (n..ids).rev().collect(),
            live: Vec::new(),
            is_live: vec![false; ids],
        }
    }

    #[inline]
    fn is_blossom(&self, b: usize) -> bool {
        b >= self.n
    }

    #[inline]
    fn slack(&self, e: Edge) -> f64 {
        self.dualvar[e.0] + self.dualvar[e.1] - 2.0 * e.2
    }

    fn set_mate(&mut self, v: usize, w: usize) {
        if self.mate[v] == NONE {
            self.mate_order.push(v);
        }
        self.mate[v] = w;
    }

    fn leaves(&self, b: usize) -> Vec<usize> {
        let mut stack = self.childs[b].clone();
        let mut out = Vec::new();
        while let Some(t) = stack.pop() {
            if self.is_blossom(t) {
                stack.extend_from_slice(&self.childs[t]);
            } else {
                out.push(t);
            }
        }
        out
    }

    fn assign_label(&mut self, mut w: usize, mut t: u8, mut v: usize) {
        loop {
            let b = self.inblossom[w];
            self.label[w] = t;
            self.label[b] = t;
            let le = (v != NONE).then_some((v, w));
            self.labeledge[w] = le;
            self.labeledge[b] = le;
            self.bestedge[w] = None;
            self.bestedge[b] = None;
            if t == 1 {
                if self.is_blossom(b) {
                    let leaves = self.leaves(b);
                    self.queue.extend(leaves);
                } else {
                    self.queue.push(b);
                }
                return;
            }
            let base = self.base[b];
            w = self.mate[base];
            t = 1;
            v = base;
        }
    }

    fn scan_blossom(&mut self, mut v: usize, mut w: usize) -> usize {
        let mut path = Vec::new();
        let mut base = NONE;
        while v != NONE {
            let mut b = self.inblossom[v];
            if self.label[b] & 4 != 0 {
                base = self.base[b];
                break;
            }
            path.push(b);
            self.label[b] = 5;
            match self.labeledge[b] {
                None => v = NONE,
                Some((p, _)) => {
                    v = p;
                    b = self.inblossom[v];
                    v = self.labeledge[b].expect("T-blossom has a label edge").0;
                }
            }
            if w != NONE {
                std::mem::swap(&mut v, &mut w);
            }
        }
        for b in path {
            self.label[b] = 1;
        }
        base
    }

    fn add_blossom(&mut self, base: usize, v: usize, w: usize) {
        let bb = self.inblossom[base];
        let mut bv = self.inblossom[v];
        let mut bw = self.inblossom[w];
        let b = self.free.pop().expect("at most n / 2 blossoms are live");
        self.live.push(b);
        self.is_live[b] = true;
        self.base[b] = base;
        self.parent[b] = NONE;
        self.parent[bb] = b;
        let mut path = Vec::new();
        let mut edgs = vec![(v, w)];
        while bv != bb {
            self.parent[bv] = b;
            path.push(bv);
            let le = self.labeledge[bv].expect("labeled");
            edgs.push(le);
            bv = self.inblossom[le.0];
        }
        path.push(bb);
        path.reverse();
        edgs.reverse();
        while bw != bb {
            self.parent[bw] = b;
            path.push(bw);
            let le = self.labeledge[bw].expect("labeled");
            edgs.push((le.1, le.0));
            bw = self.inblossom[le.0];
        }
        self.label[b] = 1;
        self.labeledge[b] = self.labeledge[bb];
        self.bdual[b] = 0.0;
        self.childs[b] = path.clone();
        self.edges[b] = edgs;
        self.mybestedges[b] = None;
        for v in self.leaves(b) {
            if self.label[self.inblossom[v]] == 2 {
                self.queue.push(v);
            }
            self.inblossom[v] = b;
        }
        // `bestedgeto`, a dict: keys in first-insertion order.
        let mut keys: Vec<usize> = Vec::new();
        let mut best: HashMap<usize, Edge> = HashMap::new();
        for &bv in &path {
            let nblist: Vec<Edge> = if self.is_blossom(bv) {
                if let Some(list) = self.mybestedges[bv].take() {
                    list
                } else {
                    let mut list = Vec::new();
                    for v in self.leaves(bv) {
                        for (w, x) in self.g.row(v) {
                            if v != w {
                                list.push((v, w, x));
                            }
                        }
                    }
                    list
                }
            } else {
                self.g
                    .row(bv)
                    .filter(|&(w, _)| w != bv)
                    .map(|(w, x)| (bv, w, x))
                    .collect()
            };
            for k in nblist {
                let (mut i, mut j) = (k.0, k.1);
                if self.inblossom[j] == b {
                    std::mem::swap(&mut i, &mut j);
                }
                let bj = self.inblossom[j];
                if bj != b && self.label[bj] == 1 {
                    let better = match best.get(&bj) {
                        None => true,
                        Some(&old) => self.slack((i, j, k.2)) < self.slack(old),
                    };
                    if better {
                        if !best.contains_key(&bj) {
                            keys.push(bj);
                        }
                        best.insert(bj, k);
                    }
                }
            }
            self.bestedge[bv] = None;
        }
        let list: Vec<Edge> = keys.iter().map(|k| best[k]).collect();
        let mut mybestedge = None;
        let mut mybestslack = 0.0;
        for &k in &list {
            let kslack = self.slack(k);
            if mybestedge.is_none() || kslack < mybestslack {
                mybestedge = Some(k);
                mybestslack = kslack;
            }
        }
        self.mybestedges[b] = Some(list);
        self.bestedge[b] = mybestedge;
    }

    /// Drop blossom `b` from the bookkeeping (`label.pop(b)`, ...).
    fn remove_blossom(&mut self, b: usize) {
        self.label[b] = 0;
        self.labeledge[b] = None;
        self.bestedge[b] = None;
        self.parent[b] = NONE;
        self.base[b] = NONE;
        self.bdual[b] = 0.0;
        self.childs[b] = Vec::new();
        self.edges[b] = Vec::new();
        self.mybestedges[b] = None;
        self.is_live[b] = false;
        let i = self.live.iter().position(|&x| x == b).expect("live");
        self.live.remove(i);
        self.free.push(b);
    }

    /// Turn the children of `b` into top-level blossoms (the loop at the
    /// start of `_recurse`), recursing at end of stage into those with
    /// zero dual, depth first as NetworkX's trampoline does.
    fn expand_children(&mut self, b0: usize, endstage: bool) {
        let mut stack: Vec<(usize, usize)> = vec![(b0, 0)];
        while let Some(&mut (b, ref mut i)) = stack.last_mut() {
            if *i == self.childs[b].len() {
                stack.pop();
                if b != b0 {
                    self.remove_blossom(b);
                }
                continue;
            }
            let s = self.childs[b][*i];
            *i += 1;
            self.parent[s] = NONE;
            if self.is_blossom(s) {
                if endstage && self.bdual[s] == 0.0 {
                    stack.push((s, 0));
                } else {
                    for v in self.leaves(s) {
                        self.inblossom[v] = s;
                    }
                }
            } else {
                self.inblossom[s] = s;
            }
        }
    }

    fn expand_blossom(&mut self, b: usize, endstage: bool) {
        self.expand_children(b, endstage);
        if !endstage && self.label[b] == 2 {
            let childs = self.childs[b].clone();
            let edges = self.edges[b].clone();
            let len = childs.len() as isize;
            let (lv, lw) = self.labeledge[b].expect("T-blossom has a label edge");
            let entrychild = self.inblossom[lw];
            let mut j = childs.iter().position(|&c| c == entrychild).expect("child") as isize;
            let jstep: isize = if j & 1 != 0 {
                j -= len;
                1
            } else {
                -1
            };
            let (mut v, mut w) = (lv, lw);
            while j != 0 {
                let (p, q) = if jstep == 1 {
                    at(&edges, j)
                } else {
                    let (q, p) = at(&edges, j - 1);
                    (p, q)
                };
                self.label[w] = 0;
                self.label[q] = 0;
                self.assign_label(w, 2, v);
                self.allowedge.insert(pair(p, q));
                self.allowedge.insert(pair(q, p));
                j += jstep;
                if jstep == 1 {
                    (v, w) = at(&edges, j);
                } else {
                    (w, v) = at(&edges, j - 1);
                }
                self.allowedge.insert(pair(v, w));
                self.allowedge.insert(pair(w, v));
                j += jstep;
            }
            let bw = at(&childs, j);
            self.label[w] = 2;
            self.label[bw] = 2;
            self.labeledge[w] = Some((v, w));
            self.labeledge[bw] = Some((v, w));
            self.bestedge[bw] = None;
            j += jstep;
            while at(&childs, j) != entrychild {
                let bv = at(&childs, j);
                if self.label[bv] == 1 {
                    j += jstep;
                    continue;
                }
                let v = if self.is_blossom(bv) {
                    let leaves = self.leaves(bv);
                    // The first labeled leaf, else the last leaf (unlabeled).
                    leaves
                        .iter()
                        .copied()
                        .find(|&x| self.label[x] != 0)
                        .unwrap_or_else(|| *leaves.last().expect("blossoms have leaves"))
                } else {
                    bv
                };
                if self.label[v] != 0 {
                    self.label[v] = 0;
                    let m = self.mate[self.base[bv]];
                    self.label[m] = 0;
                    let from = self.labeledge[v].expect("reached vertex").0;
                    self.assign_label(v, 2, from);
                }
                j += jstep;
            }
        }
        self.remove_blossom(b);
    }

    fn augment_blossom(&mut self, b0: usize, v0: usize) {
        struct Frame {
            b: usize,
            v: usize,
            state: u8,
            t: usize,
            i: isize,
            j: isize,
            jstep: isize,
            w: usize,
            x: usize,
        }
        let frame = |b, v| Frame {
            b,
            v,
            state: 0,
            t: NONE,
            i: 0,
            j: 0,
            jstep: 0,
            w: NONE,
            x: NONE,
        };
        let mut stack = vec![frame(b0, v0)];
        while let Some(f) = stack.last_mut() {
            let b = f.b;
            match f.state {
                0 => {
                    let mut t = f.v;
                    while self.parent[t] != b {
                        t = self.parent[t];
                    }
                    f.t = t;
                    f.state = 1;
                    if t >= self.n {
                        let v = f.v;
                        stack.push(frame(t, v));
                    }
                }
                1 => {
                    let i = self.childs[b]
                        .iter()
                        .position(|&c| c == f.t)
                        .expect("child") as isize;
                    f.i = i;
                    f.j = i;
                    if i & 1 != 0 {
                        f.j -= self.childs[b].len() as isize;
                        f.jstep = 1;
                    } else {
                        f.jstep = -1;
                    }
                    f.state = 2;
                }
                2 => {
                    if f.j == 0 {
                        let i = f.i as usize;
                        self.childs[b].rotate_left(i);
                        self.edges[b].rotate_left(i);
                        self.base[b] = self.base[self.childs[b][0]];
                        stack.pop();
                        continue;
                    }
                    f.j += f.jstep;
                    let t = at(&self.childs[b], f.j);
                    if f.jstep == 1 {
                        (f.w, f.x) = at(&self.edges[b], f.j);
                    } else {
                        (f.x, f.w) = at(&self.edges[b], f.j - 1);
                    }
                    f.state = 3;
                    if t >= self.n {
                        let w = f.w;
                        stack.push(frame(t, w));
                    }
                }
                3 => {
                    f.j += f.jstep;
                    let t = at(&self.childs[b], f.j);
                    f.state = 4;
                    if t >= self.n {
                        let x = f.x;
                        stack.push(frame(t, x));
                    }
                }
                _ => {
                    let (w, x) = (f.w, f.x);
                    f.state = 2;
                    self.set_mate(w, x);
                    self.set_mate(x, w);
                }
            }
        }
    }

    fn augment_matching(&mut self, v: usize, w: usize) {
        for (mut s, mut j) in [(v, w), (w, v)] {
            loop {
                let bs = self.inblossom[s];
                if self.is_blossom(bs) {
                    self.augment_blossom(bs, s);
                }
                self.set_mate(s, j);
                let Some((t, _)) = self.labeledge[bs] else {
                    break;
                };
                let bt = self.inblossom[t];
                (s, j) = self.labeledge[bt].expect("T-blossom has a label edge");
                if self.is_blossom(bt) {
                    self.augment_blossom(bt, j);
                }
                self.set_mate(j, s);
            }
        }
    }

    fn min_dual(&self) -> f64 {
        // Python's `min`: replace only on a strictly smaller value.
        let mut m = self.dualvar[0];
        for &d in &self.dualvar {
            if d < m {
                m = d;
            }
        }
        m
    }

    fn run(&mut self) {
        let n = self.n;
        loop {
            for v in 0..n {
                self.label[v] = 0;
                self.labeledge[v] = None;
                self.bestedge[v] = None;
            }
            for i in 0..self.live.len() {
                let b = self.live[i];
                self.label[b] = 0;
                self.labeledge[b] = None;
                self.bestedge[b] = None;
                self.mybestedges[b] = None;
            }
            self.allowedge.clear();
            self.queue.clear();
            for v in 0..n {
                if self.mate[v] == NONE && self.label[self.inblossom[v]] == 0 {
                    self.assign_label(v, 1, NONE);
                }
            }
            let mut augmented = false;
            loop {
                while !augmented {
                    let Some(v) = self.queue.pop() else { break };
                    let g = self.g;
                    for (w, x) in g.row(v) {
                        if w == v {
                            continue;
                        }
                        let bv = self.inblossom[v];
                        let bw = self.inblossom[w];
                        if bv == bw {
                            continue;
                        }
                        let mut kslack = 0.0;
                        if !self.allowedge.contains(&pair(v, w)) {
                            kslack = self.slack((v, w, x));
                            if kslack <= 0.0 {
                                self.allowedge.insert(pair(v, w));
                                self.allowedge.insert(pair(w, v));
                            }
                        }
                        if self.allowedge.contains(&pair(v, w)) {
                            if self.label[bw] == 0 {
                                self.assign_label(w, 2, v);
                            } else if self.label[bw] == 1 {
                                let base = self.scan_blossom(v, w);
                                if base != NONE {
                                    self.add_blossom(base, v, w);
                                } else {
                                    self.augment_matching(v, w);
                                    augmented = true;
                                    break;
                                }
                            } else if self.label[w] == 0 {
                                self.label[w] = 2;
                                self.labeledge[w] = Some((v, w));
                            }
                        } else if self.label[bw] == 1 {
                            if self.bestedge[bv].is_none_or(|e| kslack < self.slack(e)) {
                                self.bestedge[bv] = Some((v, w, x));
                            }
                        } else if self.label[w] == 0
                            && self.bestedge[w].is_none_or(|e| kslack < self.slack(e))
                        {
                            self.bestedge[w] = Some((v, w, x));
                        }
                    }
                }
                if augmented {
                    break;
                }
                let mut deltatype = -1;
                let mut delta = 0.0;
                let mut deltaedge: Option<Edge> = None;
                let mut deltablossom = NONE;
                if !self.maxcardinality {
                    deltatype = 1;
                    delta = self.min_dual();
                }
                for v in 0..n {
                    if self.label[self.inblossom[v]] == 0 {
                        if let Some(e) = self.bestedge[v] {
                            let d = self.slack(e);
                            if deltatype == -1 || d < delta {
                                delta = d;
                                deltatype = 2;
                                deltaedge = Some(e);
                            }
                        }
                    }
                }
                // `blossomparent` order: vertices, then live blossoms.
                for b in (0..n).chain(self.live.iter().copied()) {
                    if self.parent[b] == NONE && self.label[b] == 1 {
                        if let Some(e) = self.bestedge[b] {
                            let d = self.slack(e) / 2.0;
                            if deltatype == -1 || d < delta {
                                delta = d;
                                deltatype = 3;
                                deltaedge = Some(e);
                            }
                        }
                    }
                }
                for &b in &self.live {
                    if self.parent[b] == NONE
                        && self.label[b] == 2
                        && (deltatype == -1 || self.bdual[b] < delta)
                    {
                        delta = self.bdual[b];
                        deltatype = 4;
                        deltablossom = b;
                    }
                }
                if deltatype == -1 {
                    deltatype = 1;
                    let m = self.min_dual();
                    delta = if m > 0.0 { m } else { 0.0 };
                }
                for v in 0..n {
                    match self.label[self.inblossom[v]] {
                        1 => self.dualvar[v] -= delta,
                        2 => self.dualvar[v] += delta,
                        _ => {}
                    }
                }
                for i in 0..self.live.len() {
                    let b = self.live[i];
                    if self.parent[b] == NONE {
                        match self.label[b] {
                            1 => self.bdual[b] += delta,
                            2 => self.bdual[b] -= delta,
                            _ => {}
                        }
                    }
                }
                match deltatype {
                    1 => break,
                    2 | 3 => {
                        let (v, w, _) = deltaedge.expect("set with the delta");
                        self.allowedge.insert(pair(v, w));
                        self.allowedge.insert(pair(w, v));
                        self.queue.push(v);
                    }
                    _ => self.expand_blossom(deltablossom, false),
                }
            }
            if !augmented {
                break;
            }
            for b in self.live.clone() {
                if self.is_live[b]
                    && self.parent[b] == NONE
                    && self.label[b] == 1
                    && self.bdual[b] == 0.0
                {
                    self.expand_blossom(b, true);
                }
            }
        }
    }
}

/// `nx.max_weight_matching` as `matching_dict_to_set(mate)` adds them:
/// pairs in order.
pub fn max_weight_matching(g: &WeightedAdj, maxcardinality: bool) -> Vec<(u32, u32)> {
    if g.offsets.len() <= 1 {
        return Vec::new();
    }
    let mut state = Blossoms::new(g, maxcardinality);
    state.run();
    let mut done = vec![false; state.n];
    let mut out = Vec::new();
    for &u in &state.mate_order {
        let v = state.mate[u];
        done[u] = true;
        if done[v] {
            continue; // (v, u) is already in
        }
        out.push((u as u32, v as u32));
    }
    out
}

// --- Dominating sets ------------------------------------------------------------

/// Whether every node is in `nodes` or adjacent (successor) to one.
pub fn is_dominating(adj: &Csr, n: usize, nodes: &[u32]) -> bool {
    let mut covered = mask_of(n, nodes);
    for &v in nodes {
        for &w in adj.neighbors(v as usize) {
            covered[w as usize] = true;
        }
    }
    covered.iter().all(|&c| c)
}

/// Whether the subgraph induced by `nodes` (non-empty) is connected.
pub fn induced_connected(adj: &Csr, n: usize, nodes: &[u32]) -> bool {
    let inside = mask_of(n, nodes);
    let mut seen = vec![false; n];
    let start = nodes[0] as usize;
    seen[start] = true;
    let mut stack = vec![start];
    let mut count = 1;
    while let Some(v) = stack.pop() {
        for &w in adj.neighbors(v) {
            let w = w as usize;
            if inside[w] && !seen[w] {
                seen[w] = true;
                count += 1;
                stack.push(w);
            }
        }
    }
    count == inside.iter().filter(|&&x| x).count()
}

/// `nx.connected_dominating_set` on a connected graph with two or more
/// nodes: nodes in the order they join the set.
pub fn connected_dominating_set(adj: &Csr, n: usize, degree: &[usize]) -> Option<Vec<u32>> {
    let mut unseen_degree: Vec<i64> = degree.iter().map(|&d| d as i64).collect();
    let mut max_node = 0;
    for v in 1..n {
        if unseen_degree[v] > unseen_degree[max_node] {
            max_node = v;
        }
    }
    let max_deg = unseen_degree[max_node];
    for &w in adj.neighbors(max_node) {
        unseen_degree[w as usize] -= 1;
    }
    let mut unseen = vec![true; n];
    unseen[max_node] = false;
    let mut left = n - 1;
    let mut counter = 0u64;
    let mut heap: BinaryHeap<Reverse<(i64, u64, usize)>> = BinaryHeap::new();
    heap.push(Reverse((-max_deg, counter, max_node)));
    counter += 1;
    let mut out = Vec::new();
    while left > 0 {
        let Reverse((neg_deg, cnt, u)) = heap.pop()?;
        if -neg_deg > unseen_degree[u] {
            heap.push(Reverse((-unseen_degree[u], cnt, u)));
            continue;
        }
        for &v in adj.neighbors(u) {
            let v = v as usize;
            if unseen[v] {
                unseen[v] = false;
                left -= 1;
                for &w in adj.neighbors(v) {
                    unseen_degree[w as usize] -= 1;
                }
                heap.push(Reverse((-unseen_degree[v], counter, v)));
                counter += 1;
            }
        }
        out.push(u as u32);
    }
    Some(out)
}

// --- Cliques --------------------------------------------------------------------

/// `nx.enumerate_all_cliques` as a resumable queue.
pub struct AllCliques {
    /// Neighbors after each node in G's order, sorted by position.
    later: Vec<Vec<u32>>,
    queue: VecDeque<(Vec<u32>, Vec<u32>)>,
}

impl AllCliques {
    pub fn new(adj: &Csr, n: usize) -> Self {
        let later: Vec<Vec<u32>> = (0..n)
            .map(|u| {
                let mut row: Vec<u32> = adj
                    .neighbors(u)
                    .iter()
                    .copied()
                    .filter(|&v| v as usize > u)
                    .collect();
                row.sort_unstable();
                row.dedup();
                row
            })
            .collect();
        let queue = (0..n).map(|u| (vec![u as u32], later[u].clone())).collect();
        AllCliques { later, queue }
    }

    /// Up to `limit` more cliques (fewer only at the end).
    pub fn next_batch(&mut self, limit: usize) -> Vec<Vec<u32>> {
        let mut out = Vec::new();
        while out.len() < limit {
            let Some((base, cnbrs)) = self.queue.pop_front() else {
                break;
            };
            for (i, &u) in cnbrs.iter().enumerate() {
                let mut next = base.clone();
                next.push(u);
                let nb = &self.later[u as usize];
                let rest = cnbrs[i + 1..]
                    .iter()
                    .copied()
                    .filter(|v| nb.binary_search(v).is_ok())
                    .collect();
                self.queue.push_back((next, rest));
            }
            out.push(base);
        }
        out
    }
}

/// Sorted neighbor rows without self-loops (for edge lookups).
fn sorted_rows(adj: &Csr, n: usize) -> Vec<Vec<u32>> {
    (0..n)
        .map(|u| {
            let mut row: Vec<u32> = adj
                .neighbors(u)
                .iter()
                .copied()
                .filter(|&v| v as usize != u)
                .collect();
            row.sort_unstable();
            row.dedup();
            row
        })
        .collect()
}

fn sorted_intersection(a: &[u32], b: &[u32]) -> Vec<u32> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// Size of a largest clique among `cand` (branch and bound with a greedy
/// coloring bound, Tomita's MCQ).
fn max_clique_size(rows: &[Vec<u32>], cand: Vec<u32>, size: usize, best: &mut usize) {
    if cand.is_empty() {
        *best = (*best).max(size);
        return;
    }
    // Greedy coloring of `cand`, listed by color.
    let mut colored: Vec<(u32, usize)> = Vec::with_capacity(cand.len());
    let mut classes: Vec<Vec<u32>> = Vec::new();
    for &v in &cand {
        let row = &rows[v as usize];
        let k = classes
            .iter()
            .position(|class| class.iter().all(|w| row.binary_search(w).is_err()))
            .unwrap_or(classes.len());
        if k == classes.len() {
            classes.push(Vec::new());
        }
        classes[k].push(v);
    }
    for (k, class) in classes.iter().enumerate() {
        for &v in class {
            colored.push((v, k + 1));
        }
    }
    // `cand` is sorted (rows and their intersections are).
    let mut cand: Vec<u32> = cand;
    while let Some((v, color)) = colored.pop() {
        if size + color <= *best {
            return;
        }
        let next = sorted_intersection(&cand, &rows[v as usize]);
        max_clique_size(rows, next, size + 1, best);
        if let Ok(i) = cand.binary_search(&v) {
            cand.remove(i);
        }
    }
}

/// `node_clique_number` for each of `nodes`: one plus the clique number of
/// the node's neighborhood (every maximal clique of the ego graph contains
/// the node).
pub fn node_clique_numbers(adj: &Csr, n: usize, nodes: &[u32]) -> Vec<u32> {
    let rows = sorted_rows(adj, n);
    let mut memo: HashMap<u32, u32> = HashMap::new();
    nodes
        .iter()
        .map(|&v| {
            *memo.entry(v).or_insert_with(|| {
                let mut best = 0;
                max_clique_size(&rows, rows[v as usize].clone(), 0, &mut best);
                best as u32 + 1
            })
        })
        .collect()
}

/// NetworkX's `MaxWeightClique` search: the incumbent clique and weight.
pub struct MaxWeightClique<'a> {
    rows: Vec<Vec<u32>>,
    weights: &'a [i64],
    incumbent: Vec<u32>,
    incumbent_weight: i64,
}

impl<'a> MaxWeightClique<'a> {
    pub fn run(adj: &Csr, n: usize, weights: &'a [i64], degree: &[usize]) -> (Vec<u32>, i64) {
        let mut rows: Vec<Vec<u32>> = (0..n)
            .map(|u| {
                let mut row = adj.neighbors(u).to_vec();
                row.sort_unstable();
                row.dedup();
                row
            })
            .collect();
        rows.shrink_to_fit();
        let mut nodes: Vec<u32> = (0..n as u32).collect();
        // `sorted(..., reverse=True)` is stable: ties keep G's order.
        nodes.sort_by_key(|&v| Reverse(degree[v as usize]));
        nodes.retain(|&v| weights[v as usize] > 0);
        let mut state = MaxWeightClique {
            rows,
            weights,
            incumbent: Vec::new(),
            incumbent_weight: 0,
        };
        state.expand(&[], 0, nodes);
        (state.incumbent, state.incumbent_weight)
    }

    #[inline]
    fn has_edge(&self, v: u32, w: u32) -> bool {
        self.rows[v as usize].binary_search(&w).is_ok()
    }

    fn greedily_find_independent_set(&self, p: &[u32]) -> Vec<u32> {
        let mut independent = Vec::new();
        let mut p = p.to_vec();
        while let Some(&v) = p.first() {
            independent.push(v);
            p.retain(|&w| v != w && !self.has_edge(v, w));
        }
        independent
    }

    fn find_branching_nodes(&self, p: &[u32], target: i64) -> Vec<u32> {
        let mut residual: HashMap<u32, i64> =
            p.iter().map(|&v| (v, self.weights[v as usize])).collect();
        let mut total = 0i64;
        let mut p = p.to_vec();
        while !p.is_empty() {
            let independent = self.greedily_find_independent_set(&p);
            let min_wt = independent
                .iter()
                .map(|v| residual[v])
                .min()
                .expect("non-empty");
            total += min_wt;
            if total > target {
                break;
            }
            for v in &independent {
                *residual.get_mut(v).expect("in P") -= min_wt;
            }
            p.retain(|v| residual[v] != 0);
        }
        p
    }

    fn expand(&mut self, c: &[u32], c_weight: i64, mut p: Vec<u32>) {
        if c_weight > self.incumbent_weight {
            self.incumbent = c.to_vec();
            self.incumbent_weight = c_weight;
        }
        let mut branching = self.find_branching_nodes(&p, self.incumbent_weight - c_weight);
        while let Some(v) = branching.pop() {
            if let Some(i) = p.iter().position(|&x| x == v) {
                p.remove(i);
            }
            let mut new_c = c.to_vec();
            new_c.push(v);
            let new_weight = c_weight + self.weights[v as usize];
            let new_p: Vec<u32> = p.iter().copied().filter(|&w| self.has_edge(v, w)).collect();
            self.expand(&new_c, new_weight, new_p);
        }
    }
}
