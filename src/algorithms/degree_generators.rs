//! Batch 22: degree-sequence and tree generators.
//!
//! Each function replays NetworkX's code (seeded ones draw for draw on
//! `pyrandom::Mt19937`) and records the result as `random_generators::Built`
//! rows: every row lists neighbors in the order NetworkX's `add_edge` calls
//! leave them, a parallel edge repeating its neighbor (`Sim::from_rows`
//! collapses repeats for simple graphs, keeping the first, which is where
//! `add_edge` leaves an existing edge). `lib.rs` fills the NetworkX graph.
//!
//! `None` means "let NetworkX run": cases where NetworkX raises after
//! drawing (a zero `log`, an overflowing `pow`) or that rustnx doesn't
//! model.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};

use super::pyrandom::Mt19937;
use super::random_generators::Built;

/// `_to_stublist`: node `v` repeated `degree[v]` times.
fn stublist(degree: &[u32]) -> Vec<u32> {
    let total: usize = degree.iter().map(|&d| d as usize).sum();
    let mut out = Vec::with_capacity(total);
    for (v, &d) in degree.iter().enumerate() {
        out.extend(std::iter::repeat_n(v as u32, d as usize));
    }
    out
}

/// `configuration_model`: shuffle the stub list, pair its two halves.
pub fn configuration_model(degree: &[u32], rng: &mut Mt19937) -> Built {
    let mut b = Built::new(degree.len(), false);
    if degree.is_empty() {
        return b;
    }
    let mut stubs = stublist(degree);
    let half = stubs.len() / 2;
    rng.shuffle(&mut stubs);
    let (out, inn) = stubs.split_at(half);
    for (&u, &v) in out.iter().zip(inn) {
        b.push_edge(u, v);
    }
    b
}

/// `directed_configuration_model`: the graph starts with
/// `len(out_degree)` nodes; stubs of the longer in-degree sequence add
/// the rest as edges reach them. `directed` is the class's direction
/// (NetworkX doesn't check it).
pub fn directed_configuration_model(
    out_degree: &[u32],
    in_degree: &[u32],
    directed: bool,
    rng: &mut Mt19937,
) -> Built {
    let n0 = out_degree.len();
    if n0 == 0 {
        return Built::new(0, directed);
    }
    let n = n0.max(in_degree.len());
    let pad = |s: &[u32]| {
        let mut v = s.to_vec();
        v.resize(n, 0);
        v
    };
    let mut out_stubs = stublist(&pad(out_degree));
    let mut in_stubs = stublist(&pad(in_degree));
    rng.shuffle(&mut out_stubs);
    rng.shuffle(&mut in_stubs);
    let mut b = Built::new(n, directed);
    let mut present = vec![false; n];
    let mut order: Vec<u32> = (0..n0 as u32).collect();
    present[..n0].fill(true);
    for (&u, &v) in out_stubs.iter().zip(&in_stubs) {
        for x in [u, v] {
            if !present[x as usize] {
                present[x as usize] = true;
                order.push(x);
            }
        }
        b.push_edge(u, v);
    }
    // Nodes past `len(out_degree)` that no edge reaches are never added.
    if order.len() < n || order.iter().enumerate().any(|(i, &x)| i as u32 != x) {
        b.order = Some(order);
    }
    b
}

/// Bipartite `configuration_model`: shuffled stubs of the two sides,
/// paired in order (nodes `0..lena` and `lena..lena+lenb`).
pub fn bipartite_configuration_model(aseq: &[u32], bseq: &[u32], rng: &mut Mt19937) -> Built {
    let lena = aseq.len();
    let mut b = Built::new(lena + bseq.len(), false);
    let mut astubs = stublist(aseq);
    let mut bstubs: Vec<u32> = stublist(bseq)
        .into_iter()
        .map(|v| v + lena as u32)
        .collect();
    rng.shuffle(&mut astubs);
    rng.shuffle(&mut bstubs);
    for (&u, &v) in astubs.iter().zip(&bstubs) {
        b.push_edge(u, v);
    }
    b
}

/// `random_clustered_graph`: independent edges from shuffled stubs, then
/// triangles, popping from the end of each list.
pub fn random_clustered(single: &[u32], triangle: &[u32], rng: &mut Mt19937) -> Built {
    let mut b = Built::new(single.len(), false);
    let mut ilist = stublist(single);
    let mut tlist = stublist(triangle);
    rng.shuffle(&mut ilist);
    rng.shuffle(&mut tlist);
    while ilist.len() >= 2 {
        let u = ilist.pop().unwrap();
        let v = ilist.pop().unwrap();
        b.push_edge(u, v);
    }
    while tlist.len() >= 3 {
        let n1 = tlist.pop().unwrap();
        let n2 = tlist.pop().unwrap();
        let n3 = tlist.pop().unwrap();
        b.push_edge(n1, n2);
        b.push_edge(n1, n3);
        b.push_edge(n2, n3);
    }
    b
}

/// `expected_degree_graph` for weights `w` (as floats) and `rho = 1 /
/// sum(w)`, worked out by the caller with Python's `sum`. `None` where
/// NetworkX's `math.log(r, 1 - p)` raises (`r == 0`, or `1 - p == 1`).
pub fn expected_degree(w: &[f64], rho: f64, selfloops: bool, rng: &mut Mt19937) -> Option<Built> {
    let n = w.len();
    let mut b = Built::new(n, false);
    // `sorted(enumerate(w), key=itemgetter(1), reverse=True)` is stable:
    // equal weights keep their original order.
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by(|&a, &c| w[c as usize].partial_cmp(&w[a as usize]).unwrap());
    let seq: Vec<f64> = order.iter().map(|&u| w[u as usize]).collect();
    let skip = usize::from(!selfloops);
    let last = n - skip;
    // Python's `min(x, 1)`.
    let cap = |x: f64| if 1.0 < x { 1.0 } else { x };
    for u in 0..last {
        let mut v = u + skip;
        let factor = seq[u] * rho;
        let mut p = cap(seq[v] * factor);
        while v < n && p > 0.0 {
            if p != 1.0 {
                let r = rng.random();
                let base = 1.0 - p;
                if r == 0.0 || base.ln() == 0.0 {
                    return None;
                }
                let jump = (r.ln() / base.ln()).floor();
                if !jump.is_finite() {
                    return None; // math.floor raises
                }
                if jump >= (n - v) as f64 {
                    v = n;
                } else {
                    v += jump as usize;
                }
            }
            if v < n {
                let q = cap(seq[v] * factor);
                if rng.random() < q / p {
                    b.push_edge(order[u], order[v]);
                }
                v += 1;
                p = q;
            }
        }
    }
    Some(b)
}

/// `havel_hakimi_graph` for a graphical sequence (checked by the caller).
pub fn havel_hakimi(degree: &[u32]) -> Built {
    let p = degree.len();
    let mut b = Built::new(p, false);
    let mut num_degs: Vec<Vec<u32>> = vec![Vec::new(); p];
    let (mut dmax, mut n) = (0usize, 0usize);
    for &d in degree {
        if d > 0 {
            num_degs[d as usize].push(n as u32);
            dmax = dmax.max(d as usize);
            n += 1;
        }
    }
    // NetworkX numbers the nonzero entries 0, 1, ... in turn: node `n`
    // above is the n-th nonzero entry, not its position in the sequence.
    if n == 0 {
        return b;
    }
    let mut modstubs: Vec<(usize, u32)> = Vec::with_capacity(dmax + 1);
    while n > 0 {
        while num_degs[dmax].is_empty() {
            dmax -= 1;
        }
        let source = num_degs[dmax].pop().unwrap();
        n -= 1;
        modstubs.clear();
        let mut k = dmax;
        for _ in 0..dmax {
            while num_degs[k].is_empty() {
                k -= 1;
            }
            let target = num_degs[k].pop().unwrap();
            b.push_edge(source, target);
            n -= 1;
            if k > 1 {
                modstubs.push((k - 1, target));
            }
        }
        for &(stubval, stubtarget) in &modstubs {
            num_degs[stubval].push(stubtarget);
            n += 1;
        }
    }
    b
}

/// `directed_havel_hakimi_graph` for non-negative sequences with equal
/// sums, padded to the same length; `Err` for "Non-digraphical integer
/// sequence". Heap entries are distinct (they hold the node), so any
/// priority queue pops them in `heapq`'s order.
pub fn directed_havel_hakimi(in_deg: &[u32], out_deg: &[u32], directed: bool) -> Result<Built, ()> {
    let n = in_deg.len();
    let mut b = Built::new(n, directed);
    let mut stubheap: BinaryHeap<Reverse<(i64, i64, u32)>> = BinaryHeap::new();
    let mut zeroheap: BinaryHeap<Reverse<(i64, u32)>> = BinaryHeap::new();
    for v in 0..n {
        let (i, o) = (in_deg[v] as i64, out_deg[v] as i64);
        if i > 0 {
            stubheap.push(Reverse((-o, -i, v as u32)));
        } else if o > 0 {
            zeroheap.push(Reverse((-o, v as u32)));
        }
    }
    let mut modstubs: Vec<(i64, i64, u32)> = Vec::new();
    while let Some(Reverse((freeout, freein, target))) = stubheap.pop() {
        let freein = -freein;
        if freein as usize > stubheap.len() + zeroheap.len() {
            return Err(());
        }
        modstubs.clear();
        for _ in 0..freein {
            let take_zero = match (zeroheap.peek(), stubheap.peek()) {
                (Some(_), None) => true,
                (Some(Reverse(z)), Some(Reverse(s))) => s.0 > z.0,
                _ => false,
            };
            let (stubout, stubin, stubsource) = if take_zero {
                let Reverse((o, v)) = zeroheap.pop().unwrap();
                (o, 0, v)
            } else {
                let Reverse(t) = stubheap.pop().ok_or(())?;
                t
            };
            if stubout == 0 {
                return Err(());
            }
            b.push_edge(stubsource, target);
            if stubout + 1 < 0 || stubin < 0 {
                modstubs.push((stubout + 1, stubin, stubsource));
            }
        }
        for &stub in &modstubs {
            if stub.1 < 0 {
                stubheap.push(Reverse(stub));
            } else {
                zeroheap.push(Reverse((stub.0, stub.2)));
            }
        }
        if freeout < 0 {
            zeroheap.push(Reverse((freeout, target)));
        }
    }
    Ok(b)
}

/// `degree_sequence_tree` for a valid tree sequence (checked by the
/// caller): a path backbone over the degrees above 1, then the leaves.
/// `legacy` (NetworkX 3.4 and 3.5) removes node 0 when the backbone made
/// one node too many.
pub fn degree_sequence_tree(degree: &[i64], legacy: bool) -> Built {
    let mut deg: Vec<i64> = degree.iter().copied().filter(|&s| s > 1).collect();
    deg.sort_unstable_by(|a, c| c.cmp(a));
    let n = deg.len() + 2;
    let leaves: i64 = deg.iter().map(|&d| d - 2).sum();
    let total = n + leaves.max(0) as usize;
    let mut b = Built::new(total, false);
    for u in 1..n as u32 {
        b.push_edge(u - 1, u);
    }
    let mut last = n as u32;
    for source in 1..n.saturating_sub(1) as u32 {
        let nedges = deg.pop().unwrap() - 2;
        for target in last..last + nedges as u32 {
            b.push_edge(source, target);
        }
        last += nedges as u32;
    }
    if legacy && total > degree.len() {
        // `G.remove_node(0)`: node 0's one edge goes to node 1.
        b.succ[0].clear();
        b.succ[1].retain(|&x| x != 0);
        b.order = Some((1..total as u32).collect());
    }
    b
}

/// The bipartite Havel-Hakimi variants (`kind` 0: `havel_hakimi_graph`,
/// 1: `reverse_havel_hakimi_graph`, 2: `alternating_havel_hakimi_graph`)
/// for non-negative sequences with equal sums. NetworkX's stub lists are
/// `[degree, node]` lists sorted by degree, then node; entries are distinct,
/// so ordered sets give the same picks as its sorts. A `b` stub leaves the
/// list when its degree drops to exactly 0 (stubs that start at 0 stay and
/// go negative if picked).
pub fn bipartite_havel_hakimi(aseq: &[u32], bseq: &[u32], kind: u8) -> Built {
    let lena = aseq.len();
    let n = lena + bseq.len();
    let mut b = Built::new(n, false);
    let mut adeg: Vec<(i64, u32)> = aseq
        .iter()
        .enumerate()
        .map(|(v, &d)| (d as i64, v as u32))
        .collect();
    adeg.sort_unstable();
    let mut bdeg: Vec<i64> = vec![0; n];
    for (i, &d) in bseq.iter().enumerate() {
        bdeg[lena + i] = d as i64;
    }
    let bnodes = lena as u32..n as u32;
    if kind == 1 {
        // Sorted once; then the first `degree` stubs still in the list.
        let mut order: Vec<u32> = bnodes.collect();
        order.sort_unstable_by_key(|&v| (bdeg[v as usize], v));
        // A linked list over the sorted stubs; `end` marks its end.
        let end = order.len();
        let mut next: Vec<usize> = (1..=end).collect();
        let mut prev: Vec<usize> = (0..end).map(|i| i.wrapping_sub(1)).collect();
        let mut head = 0usize;
        while let Some((degree, u)) = adeg.pop() {
            if degree == 0 {
                break;
            }
            let mut picks = Vec::with_capacity(degree as usize);
            let mut i = head;
            while i != end && picks.len() < degree as usize {
                picks.push(i);
                i = next[i];
            }
            for i in picks {
                let v = order[i];
                b.push_edge(u, v);
                bdeg[v as usize] -= 1;
                if bdeg[v as usize] == 0 {
                    // Unlink slot `i`.
                    let (p, nx) = (prev[i], next[i]);
                    if i == head {
                        head = nx;
                    } else {
                        next[p] = nx;
                    }
                    if nx != end {
                        prev[nx] = p;
                    }
                }
            }
        }
        return b;
    }
    let mut stubs: BTreeSet<(i64, u32)> = bnodes.map(|v| (bdeg[v as usize], v)).collect();
    let mut removed = vec![false; n];
    while let Some((degree, u)) = adeg.pop() {
        if degree == 0 {
            break;
        }
        let d = degree as usize;
        let picks: Vec<u32> = if kind == 0 {
            // `bstubs[-degree:]`, in ascending order.
            let mut top: Vec<u32> = stubs.iter().rev().take(d).map(|&(_, v)| v).collect();
            top.reverse();
            top
        } else {
            // `small = bstubs[:d // 2]`, `large = bstubs[-(d - d // 2):]`,
            // interleaved large first; one more of `large` (its last) if
            // the two lengths differ.
            let small: Vec<u32> = stubs.iter().take(d / 2).map(|&(_, v)| v).collect();
            let mut large: Vec<u32> = stubs
                .iter()
                .rev()
                .take(d - d / 2)
                .map(|&(_, v)| v)
                .collect();
            large.reverse();
            let mut out = Vec::with_capacity(d);
            for (&l, &s) in large.iter().zip(&small) {
                out.push(l);
                out.push(s);
            }
            if out.len() < small.len() + large.len() {
                out.push(*large.last().unwrap());
            }
            out
        };
        for &v in &picks {
            stubs.remove(&(bdeg[v as usize], v));
        }
        for &v in &picks {
            b.push_edge(u, v);
            bdeg[v as usize] -= 1;
            if bdeg[v as usize] == 0 {
                removed[v as usize] = true;
            }
        }
        for &v in &picks {
            if !removed[v as usize] {
                stubs.insert((bdeg[v as usize], v));
            }
        }
    }
    b
}

/// `random_powerlaw_tree_sequence(n, gamma, tries)` with `alpha = gamma -
/// 1`: `Some(Ok(seq))`, `Some(Err(()))` when NetworkX gives up
/// after `tries`, `None` where its `pow` or `round` raises. `legacy` (3.4
/// and 3.5) accepts any sequence summing to `2n - 2`; later versions also
/// need positive degrees (unless the sequence is `[0]`).
pub fn powerlaw_tree_sequence(
    n: usize,
    alpha: f64,
    tries: usize,
    legacy: bool,
    rng: &mut Mt19937,
) -> Option<Result<Vec<u64>, ()>> {
    let draw = |rng: &mut Mt19937| -> Option<u64> {
        let s = rng.paretovariate(alpha);
        if !s.is_finite() {
            return None; // `pow` overflows (or `round` gets NaN): Python raises
        }
        let r = s.round_ties_even();
        Some(if r >= n as f64 {
            n as u64
        } else if r <= 0.0 {
            0
        } else {
            r as u64
        })
    };
    let mut zseq = Vec::with_capacity(n);
    for _ in 0..n {
        zseq.push(draw(rng)?);
    }
    let mut swap = Vec::with_capacity(tries);
    for _ in 0..tries {
        swap.push(draw(rng)?);
    }
    let target = 2 * n as u64;
    let mut total: u64 = zseq.iter().sum();
    let mut zeros = zseq.iter().filter(|&&d| d == 0).count();
    // `for _ in swap:` while popping from `swap`: a list iterator stops once
    // its index reaches the shrinking length.
    let mut i = 0;
    while i < swap.len() {
        i += 1;
        let valid = total + 2 == target && (legacy || zeros == 0 || (n == 1 && zseq[0] == 0));
        if valid {
            return Some(Ok(zseq));
        }
        if n == 0 {
            return None; // randint(0, -1) raises
        }
        let index = rng.randint(0, n as i64 - 1)? as usize;
        let new = swap.pop().unwrap();
        let old = std::mem::replace(&mut zseq[index], new);
        total = total - old + new;
        zeros = zeros + usize::from(new == 0) - usize::from(old == 0);
    }
    // The last check before the loop ends never happens: NetworkX raises.
    Some(Err(()))
}

/// `random_labeled_tree(n)` for `n >= 2`: a random Prüfer sequence turned
/// into a tree as `from_prufer_sequence` does.
pub fn random_labeled_tree(n: usize, rng: &mut Mt19937) -> Built {
    let seq: Vec<i64> = (0..n - 2).map(|_| rng.below(n) as i64).collect();
    let edges = super::trees_more::prufer_edges(&seq).expect("entries are in range");
    let mut b = Built::new(n, false);
    let mut not_orphaned = vec![false; n];
    for &(u, v) in &edges {
        b.push_edge(u, v);
        not_orphaned[u as usize] = true;
    }
    // The two orphans' edge: the same rows whichever end comes first.
    let mut orphans = (0..n as u32).filter(|&v| !not_orphaned[v as usize]);
    let u = orphans.next().unwrap();
    let v = orphans.next().unwrap();
    b.push_edge(u, v);
    b
}

/// `G.copy()`, `relabel_nodes` and the unions all rebuild a graph from
/// `G.edges()`: each row ends up with its earlier neighbors (in node
/// order) first, then its later ones in their current order.
fn rebuild(rows: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let mut new: Vec<Vec<u32>> = rows.iter().map(|r| Vec::with_capacity(r.len())).collect();
    for (u, row) in rows.iter().enumerate() {
        for &v in row {
            if v as usize > u {
                new[u].push(v);
                new[v as usize].push(u as u32);
            } else if v as usize == u {
                new[u].push(v);
            }
        }
    }
    new
}

/// `random_cograph(n)`: `2^n` nodes from `n` full joins or disjoint
/// unions of the graph with a relabelled copy of itself.
pub fn random_cograph(n: u32, rng: &mut Mt19937) -> Built {
    let mut rows: Vec<Vec<u32>> = vec![Vec::new()];
    for _ in 0..n {
        let m = rows.len();
        let base = rebuild(&rows);
        let join = rng.randint(0, 1) == Some(0);
        let mut next = base.clone();
        next.extend(
            base.iter()
                .map(|row| row.iter().map(|&v| v + m as u32).collect::<Vec<u32>>()),
        );
        if join {
            for row in next.iter_mut().take(m) {
                row.extend(m as u32..2 * m as u32);
            }
            for row in next.iter_mut().skip(m) {
                row.extend(0..m as u32);
            }
        }
        rows = next;
    }
    Built {
        succ: rows,
        pred: None,
        order: None,
    }
}

/// A Fenwick tree of non-negative counts over positions `0..n`.
struct Fenwick {
    tree: Vec<u64>,
}

impl Fenwick {
    fn new(values: &[u64]) -> Self {
        let n = values.len();
        let mut tree = vec![0u64; n + 1];
        for (i, &v) in values.iter().enumerate() {
            tree[i + 1] += v;
            let j = (i + 1) + (1 << (i + 1).trailing_zeros());
            if j <= n {
                tree[j] += tree[i + 1];
            }
        }
        Fenwick { tree }
    }

    fn add(&mut self, i: usize, delta: i64) {
        let mut i = i + 1;
        while i < self.tree.len() {
            self.tree[i] = self.tree[i].wrapping_add(delta as u64);
            i += 1 << i.trailing_zeros();
        }
    }

    /// The first position whose prefix sum (inclusive) reaches `target`
    /// (`target >= 1` and at most the total).
    fn lower_bound(&self, mut target: u64) -> usize {
        let n = self.tree.len() - 1;
        let mut pos = 0;
        let mut step = n.checked_next_power_of_two().unwrap_or(0).max(1);
        while step > 0 {
            if pos + step <= n && self.tree[pos + step] < target {
                pos += step;
                target -= self.tree[pos];
            }
            step >>= 1;
        }
        pos
    }
}

#[inline]
fn pair_key(u: u32, v: u32) -> u64 {
    let (a, b) = if u <= v { (u, v) } else { (v, u) };
    ((a as u64) << 32) | b as u64
}

/// One `DegreeSequenceRandomGraph.generate()` run.
struct DegreeSequenceRun<'a> {
    degree: &'a [u32],
    /// `4.0 * self.m`.
    four_m: f64,
    dmax: u64,
    remaining: Vec<u32>,
    alive: usize,
    /// Nodes per remaining degree, for `max(remaining_degree.values())`.
    histogram: Vec<usize>,
    max_remaining: usize,
    edges: std::collections::HashSet<u64>,
    built: Built,
}

impl DegreeSequenceRun<'_> {
    fn p(&self, u: u32, v: u32) -> f64 {
        let num = self.degree[u as usize] as u64 * self.degree[v as usize] as u64;
        1.0 - num as f64 / self.four_m
    }

    fn q(&mut self, u: u32, v: u32) -> f64 {
        while self.histogram[self.max_remaining] == 0 {
            self.max_remaining -= 1;
        }
        let m = self.max_remaining as u64;
        let num = self.remaining[u as usize] as u64 * self.remaining[v as usize] as u64;
        num as f64 / (m * m) as f64
    }

    fn has_edge(&self, u: u32, v: u32) -> bool {
        self.edges.contains(&pair_key(u, v))
    }

    fn add_edge(&mut self, u: u32, v: u32) {
        self.edges.insert(pair_key(u, v));
        self.built.push_edge(u, v);
    }

    /// `update_remaining` for one end; whether it left the dict.
    fn decrement(&mut self, u: u32) -> bool {
        let d = self.remaining[u as usize] as usize;
        self.histogram[d] -= 1;
        self.remaining[u as usize] -= 1;
        if d == 1 {
            self.alive -= 1;
            true
        } else {
            self.histogram[d - 1] += 1;
            false
        }
    }
}

/// `random_degree_sequence_graph(sequence, tries)` for a graphical
/// sequence whose sum and squared maximum are below 2^53: `Some(Some(g))`,
/// `Some(None)` when every try fails, `None` where NetworkX would raise
/// something else (an exhausted roulette wheel).
pub fn random_degree_sequence(
    degree: &[u32],
    tries: usize,
    rng: &mut Mt19937,
) -> Option<Option<Built>> {
    for _ in 0..tries {
        if let Some(b) = degree_sequence_try(degree, rng)? {
            return Some(Some(b));
        }
    }
    Some(None)
}

/// One `generate()`: `Some(None)` for `NetworkXUnfeasible`.
fn degree_sequence_try(degree: &[u32], rng: &mut Mt19937) -> Option<Option<Built>> {
    let n = degree.len();
    let total: u64 = degree.iter().map(|&d| d as u64).sum();
    let dmax = degree.iter().copied().max().unwrap_or(0) as u64;
    let mut histogram = vec![0usize; dmax as usize + 1];
    for &d in degree {
        histogram[d as usize] += 1;
    }
    histogram[0] = 0;
    let mut run = DegreeSequenceRun {
        degree,
        four_m: 4.0 * (total as f64 / 2.0),
        dmax,
        remaining: degree.to_vec(),
        alive: degree.iter().filter(|&&d| d > 0).count(),
        histogram,
        max_remaining: dmax as usize,
        edges: std::collections::HashSet::new(),
        built: Built::new(n, false),
    };
    if run.alive == 0 {
        return Some(Some(run.built));
    }
    // Phase 1: pairs from the degree-weighted roulette wheel. Weights are
    // ints and the wheel's running float stays exact while non-negative,
    // so the first prefix sum above the draw picks the same key.
    let mut weights = Fenwick::new(&run.remaining.iter().map(|&d| d as u64).collect::<Vec<_>>());
    let mut rem_sum = total;
    while rem_sum >= 2 * run.dmax * run.dmax {
        let mut pick = || -> Option<u32> {
            let rnd = rng.random() * rem_sum as f64;
            let target = rnd.floor() as u64 + 1;
            (target <= rem_sum).then(|| weights.lower_bound(target) as u32)
        };
        let a = pick()?;
        let mut b = pick()?;
        while b == a {
            b = pick()?;
        }
        let (u, v) = if a < b { (a, b) } else { (b, a) };
        if run.has_edge(u, v) {
            continue;
        }
        if rng.random() < run.p(u, v) {
            run.add_edge(u, v);
            for x in [u, v] {
                run.decrement(x);
                weights.add(x as usize, -1);
            }
            rem_sum -= 2;
        }
    }
    // Phase 2: uniform pairs from `list(remaining_deg.keys())` (in node
    // order), with rejection.
    let mut present = Fenwick::new(
        &run.remaining
            .iter()
            .map(|&d| u64::from(d > 0))
            .collect::<Vec<_>>(),
    );
    while run.alive as u64 >= 2 * run.dmax {
        let (u, v) = loop {
            let picks = rng.sample_range(run.alive, 2);
            let a = present.lower_bound(picks[0] as u64 + 1) as u32;
            let b = present.lower_bound(picks[1] as u64 + 1) as u32;
            let (u, v) = if a < b { (a, b) } else { (b, a) };
            if run.has_edge(u, v) {
                continue;
            }
            if rng.random() < run.q(u, v) {
                break (u, v);
            }
        };
        if rng.random() < run.p(u, v) {
            run.add_edge(u, v);
            for x in [u, v] {
                if run.decrement(x) {
                    present.add(x as usize, -1);
                }
            }
        }
    }
    // Phase 3: the remaining candidate pairs as an auxiliary graph `H`.
    let mut rest: Vec<u32> = (0..n as u32)
        .filter(|&x| run.remaining[x as usize] > 0)
        .collect();
    let mut hpos = vec![u32::MAX; n];
    let mut hnodes: Vec<u32> = Vec::new();
    let mut hrows: Vec<Vec<(u32, u32)>> = Vec::new();
    let mut ends: Vec<(u32, u32)> = Vec::new();
    for (i, &a) in rest.iter().enumerate() {
        for &b in &rest[i + 1..] {
            if run.has_edge(a, b) {
                continue;
            }
            for x in [a, b] {
                if hpos[x as usize] == u32::MAX {
                    hpos[x as usize] = hnodes.len() as u32;
                    hnodes.push(x);
                    hrows.push(Vec::new());
                }
            }
            let id = ends.len() as u32;
            ends.push((a, b));
            hrows[hpos[a as usize] as usize].push((b, id));
            hrows[hpos[b as usize] as usize].push((a, id));
        }
    }
    // `H.edges()` order, which removals only thin out.
    let mut slot_of = vec![0u32; ends.len()];
    let mut listed: Vec<(u32, u32)> = Vec::with_capacity(ends.len());
    {
        let mut seen = vec![false; hnodes.len()];
        for (hi, &x) in hnodes.iter().enumerate() {
            for &(y, id) in &hrows[hi] {
                if !seen[hpos[y as usize] as usize] {
                    slot_of[id as usize] = listed.len() as u32;
                    listed.push((x, y));
                }
            }
            seen[hi] = true;
        }
    }
    let mut live = Fenwick::new(&vec![1u64; listed.len()]);
    let mut dead = vec![false; ends.len()];
    let mut live_count = listed.len();
    let mut kill = |id: u32, live: &mut Fenwick, count: &mut usize| {
        if !dead[id as usize] {
            dead[id as usize] = true;
            live.add(slot_of[id as usize] as usize, -1);
            *count -= 1;
        }
    };
    while run.alive > 0 {
        rest.retain(|&x| run.remaining[x as usize] > 0);
        // `suitable_edge`: the first remaining node still has a non-neighbor.
        let first = rest[0];
        if !rest[1..].iter().any(|&v| !run.has_edge(first, v)) {
            return Some(None);
        }
        let (u, v) = loop {
            if live_count == 0 {
                return None; // choice() of an empty list raises
            }
            let i = rng.below(live_count);
            let (a, b) = listed[live.lower_bound(i as u64 + 1)];
            let (u, v) = if a < b { (a, b) } else { (b, a) };
            if rng.random() < run.q(u, v) {
                break (u, v);
            }
        };
        if rng.random() < run.p(u, v) {
            run.add_edge(u, v);
            // `aux_graph.remove_edge(u, v)`, then nodes that are done.
            let hu = hpos[u as usize] as usize;
            let id = hrows[hu]
                .iter()
                .find(|&&(y, _)| y == v)
                .map(|&(_, id)| id)?;
            kill(id, &mut live, &mut live_count);
            for x in [u, v] {
                if run.decrement(x) {
                    let hx = hpos[x as usize] as usize;
                    for &(_, id) in &hrows[hx] {
                        kill(id, &mut live, &mut live_count);
                    }
                }
            }
        }
    }
    Some(Some(run.built))
}

/// `joint_degree_graph` for a valid joint degree dict, given as the node
/// count of each degree in NetworkX's order (`classes`: `(degree, count)`)
/// and the entries to realise (`(k index, l index, count)` with `k >= l`,
/// in dict order). `None` where NetworkX's neighbor switch fails.
pub fn joint_degree(
    classes: &[(u32, u32)],
    entries: &[(u32, u32, u64)],
    rng: &mut Mt19937,
) -> Option<Built> {
    use super::pyset::PySet;
    let n: usize = classes.iter().map(|&(_, c)| c as usize).sum();
    let hashes: Vec<i64> = (0..n as i64).collect();
    let mut start = Vec::with_capacity(classes.len());
    let mut residual = vec![0u32; n];
    let mut next = 0u32;
    for &(degree, count) in classes {
        start.push(next);
        for v in next..next + count {
            residual[v as usize] = degree;
        }
        next += count;
    }
    let mut b = Built::new(n, false);
    let mut edges: std::collections::HashSet<u64> = std::collections::HashSet::new();
    // `_neighbor_switch`: move one of `w`'s edges to the first node of
    // `unsat` (skipping `avoid` when it has one stub left).
    let switch = |b: &mut Built,
                  edges: &mut std::collections::HashSet<u64>,
                  residual: &mut [u32],
                  w: u32,
                  unsat: &mut PySet,
                  avoid: Option<u32>|
     -> Option<()> {
        let w_prime = match avoid {
            Some(a) if residual[a as usize] <= 1 => unsat.iter().find(|&x| x != a)?,
            _ => unsat.first()?,
        };
        let switch_node = b.succ[w as usize]
            .iter()
            .copied()
            .find(|&v| v != w_prime && !edges.contains(&pair_key(w_prime, v)))?;
        b.remove_edge(w, switch_node);
        edges.remove(&pair_key(w, switch_node));
        b.push_edge(w_prime, switch_node);
        edges.insert(pair_key(w_prime, switch_node));
        residual[w as usize] += 1;
        residual[w_prime as usize] -= 1;
        if residual[w_prime as usize] == 0 {
            unsat.discard(w_prime, &hashes);
        }
        Some(())
    };
    for &(ki, li, count) in entries {
        let (ki, li) = (ki as usize, li as usize);
        let (k_start, k_size) = (start[ki], classes[ki].1);
        let (l_start, l_size) = (start[li], classes[li].1);
        let same = ki == li;
        let unsat_of = |s: u32, size: u32, residual: &[u32]| {
            let mut set = PySet::default();
            for v in s..s + size {
                if residual[v as usize] > 0 {
                    set.add(v, &hashes);
                }
            }
            set
        };
        let mut k_unsat = unsat_of(k_start, k_size, &residual);
        let mut l_unsat = if same {
            PySet::default()
        } else {
            unsat_of(l_start, l_size, &residual)
        };
        let mut n_edges_add = if same { count / 2 } else { count };
        while n_edges_add > 0 {
            let v = k_start + rng.below(k_size as usize) as u32;
            let w = l_start + rng.below(l_size as usize) as u32;
            if v == w || edges.contains(&pair_key(v, w)) {
                continue;
            }
            if residual[v as usize] == 0 {
                switch(&mut b, &mut edges, &mut residual, v, &mut k_unsat, None)?;
            }
            if residual[w as usize] == 0 {
                if same {
                    switch(&mut b, &mut edges, &mut residual, w, &mut k_unsat, Some(v))?;
                } else {
                    switch(&mut b, &mut edges, &mut residual, w, &mut l_unsat, None)?;
                }
            }
            b.push_edge(v, w);
            edges.insert(pair_key(v, w));
            residual[v as usize] -= 1;
            residual[w as usize] -= 1;
            n_edges_add -= 1;
            if residual[v as usize] == 0 {
                k_unsat.discard(v, &hashes);
            }
            if residual[w as usize] == 0 {
                if same {
                    k_unsat.discard(w, &hashes);
                } else {
                    l_unsat.discard(w, &hashes);
                }
            }
        }
    }
    Some(b)
}

/// `is_valid_directed_joint_degree` for int sequences of one length and
/// `nkk` entries `(k, l, value)` in dict order; `None` where NetworkX
/// raises (a missing key, or dividing by a zero degree).
pub fn is_valid_directed_joint_degree(
    in_degrees: &[i64],
    out_degrees: &[i64],
    nkk: &[(i64, i64, i64)],
) -> Option<bool> {
    use std::collections::HashMap;
    let mut v_in: HashMap<i64, i64> = HashMap::new();
    let mut v_out: HashMap<i64, i64> = HashMap::new();
    let mut forbidden: HashMap<(i64, i64), i64> = HashMap::new();
    for (&i, &o) in in_degrees.iter().zip(out_degrees) {
        *v_in.entry(i).or_insert(0) += 1;
        *v_out.entry(o).or_insert(0) += 1;
        *forbidden.entry((o, i)).or_insert(0) += 1;
    }
    // `S` in insertion order: `(degree, side)` keys, side 1 out, 0 in.
    let mut order: Vec<(i64, u8)> = Vec::new();
    let mut s: HashMap<(i64, u8), i64> = HashMap::new();
    for &(k, l, val) in nkk {
        if val > 0 {
            for key in [(k, 1u8), (l, 0u8)] {
                let entry = s.entry(key).or_insert_with(|| {
                    order.push(key);
                    0
                });
                *entry += val;
            }
            let vk = *v_out.get(&k)?;
            let vl = *v_in.get(&l)?;
            if val + forbidden.get(&(k, l)).copied().unwrap_or(0) > vk * vl {
                return Some(false);
            }
        }
    }
    // `all(S[s] / s[0] == V[s] for s in S)`: below 2^53 the float quotient
    // equals the int exactly when the division is exact.
    for key in order {
        let total = s[&key];
        if key.0 == 0 {
            return None;
        }
        let count = *(if key.1 == 1 { &v_out } else { &v_in }).get(&key.0)?;
        if total != count * key.0 {
            return Some(false);
        }
    }
    Some(true)
}

/// The part of `random_labeled_rooted_forest(n)` after NetworkX picks the
/// number of roots `k` (`1 <= k < n`): the forest and its roots, in
/// `sample` order. `None` if the leaf iterator runs dry.
pub fn labeled_rooted_forest(n: usize, k: usize, rng: &mut Mt19937) -> Option<(Built, Vec<u32>)> {
    use super::pyset::PySet;
    let hashes: Vec<i64> = (0..n as i64).collect();
    let roots: Vec<u32> = rng
        .sample_range(n, k)
        .into_iter()
        .map(|x| x as u32)
        .collect();
    // `p = set(range(n)).difference(roots)`: a copy of the full set with
    // the roots discarded, in that table's order.
    let mut full = PySet::default();
    for v in 0..n as u32 {
        full.add(v, &hashes);
    }
    let mut p = PySet::default();
    p.merge(&full, &hashes);
    drop(full);
    for &r in &roots {
        p.discard(r, &hashes);
    }
    p.after_difference_update(&hashes);
    let mut in_p = vec![true; n];
    for &r in &roots {
        in_p[r as usize] = false;
    }
    let draws: Vec<u32> = (0..n - k - 1)
        .map(|_| rng.randint(0, n as i64 - 1).map(|x| x as u32))
        .collect::<Option<_>>()?;
    let mut degree = vec![0i64; n];
    for &x in &draws {
        if in_p[x as usize] {
            degree[x as usize] += 1;
        }
    }
    let order: Vec<u32> = p.iter().collect();
    // `iter(x for x in p if degree[x] == 0)`: lazy, so it sees the degrees
    // as they are when `next` is called.
    let mut at = 0usize;
    let mut next_leaf = |degree: &[i64]| -> Option<u32> {
        while at < order.len() {
            let x = order[at];
            at += 1;
            if degree[x as usize] == 0 {
                return Some(x);
            }
        }
        None
    };
    let mut b = Built::new(n, false);
    let mut u = next_leaf(&degree)?;
    let mut last = u;
    for &v in &draws {
        b.push_edge(u, v);
        degree[v as usize] -= 1;
        if v < last && degree[v as usize] == 0 {
            u = v;
        } else {
            u = next_leaf(&degree)?;
            last = u;
        }
    }
    b.push_edge(u, roots[0]);
    Some((b, roots))
}
