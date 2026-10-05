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
