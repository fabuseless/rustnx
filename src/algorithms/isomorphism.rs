//! Isomorphism tests and graph hashing: the property sequences behind
//! `could_be_isomorphic`, an exact matcher for the yes/no isomorphism,
//! subgraph isomorphism and monomorphism tests, rooted tree isomorphism and
//! Weisfeiler-Lehman hashes (with BLAKE2b, as Python's `hashlib` computes
//! it).

use std::collections::HashMap;

use rayon::prelude::*;

use crate::graph::Csr;

// --- BLAKE2b (RFC 7693), unkeyed, as `hashlib.blake2b(data, digest_size=k)` ---

const IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

#[inline(always)]
fn mix(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

fn compress(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
    let mut m = [0u64; 16];
    for (i, word) in m.iter_mut().enumerate() {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&block[i * 8..i * 8 + 8]);
        *word = u64::from_le_bytes(bytes);
    }
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= t as u64;
    v[13] ^= (t >> 64) as u64;
    if last {
        v[14] = !v[14];
    }
    for s in &SIGMA {
        mix(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        mix(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        mix(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        mix(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        mix(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        mix(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        mix(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        mix(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// `blake2b(data, digest_size=outlen).hexdigest()` (`1 <= outlen <= 64`).
pub fn blake2b_hex(data: &[u8], outlen: usize) -> String {
    let mut h = IV;
    h[0] ^= 0x0101_0000 ^ outlen as u64;
    let mut block = [0u8; 128];
    let mut t: u128 = 0;
    let mut chunks = data.chunks(128).peekable();
    if data.is_empty() {
        compress(&mut h, &block, 0, true);
    }
    while let Some(chunk) = chunks.next() {
        t += chunk.len() as u128;
        block.fill(0);
        block[..chunk.len()].copy_from_slice(chunk);
        compress(&mut h, &block, t, chunks.peek().is_none());
    }
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(2 * outlen);
    for i in 0..outlen {
        let byte = (h[i / 8] >> (8 * (i % 8))) as u8;
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 15) as usize] as char);
    }
    out
}

// --- Weisfeiler-Lehman hashes ------------------------------------------------

/// Where each node's neighbor labels come from in one WL step: a group of
/// neighbors (successors, or predecessors) and the prefix put before each
/// neighbor's label (one string for all, or one per edge).
pub struct WlGroup<'a> {
    /// Per node, `(neighbor, edge index)` pairs.
    pub rows: Vec<Vec<(u32, u32)>>,
    pub prefix: WlPrefix<'a>,
}

pub enum WlPrefix<'a> {
    Same(&'a str),
    PerEdge(&'a [String]),
}

/// Neighbor rows of `adj` with the index of each entry (its edge id).
pub fn rows_with_ids(adj: &Csr, n: usize) -> Vec<Vec<(u32, u32)>> {
    (0..n)
        .map(|v| adj.range(v).map(|e| (adj.targets[e], e as u32)).collect())
        .collect()
}

/// Predecessor rows built from `succ`, each entry carrying the id of the
/// edge in `succ`. Their order doesn't matter: WL sorts the labels.
pub fn pred_rows_with_ids(succ: &Csr, n: usize) -> Vec<Vec<(u32, u32)>> {
    let mut rows = vec![Vec::new(); n];
    for u in 0..n {
        for e in succ.range(u) {
            rows[succ.targets[e] as usize].push((u as u32, e as u32));
        }
    }
    rows
}

/// One WL step, as NetworkX's `_neighborhood_aggregate*` and `_hash_label`:
/// each node's label followed by its groups' sorted neighbor labels.
pub fn wl_step(labels: &[String], groups: &[WlGroup], digest_size: usize) -> Vec<String> {
    (0..labels.len())
        .into_par_iter()
        .map(|v| {
            let mut text = labels[v].clone();
            for group in groups {
                let row = &group.rows[v];
                match group.prefix {
                    WlPrefix::Same(prefix) => {
                        // A shared prefix keeps the order of the labels.
                        let mut parts: Vec<&str> = row
                            .iter()
                            .map(|&(w, _)| labels[w as usize].as_str())
                            .collect();
                        parts.sort_unstable();
                        for part in parts {
                            text.push_str(prefix);
                            text.push_str(part);
                        }
                    }
                    WlPrefix::PerEdge(prefixes) => {
                        let mut parts: Vec<String> = row
                            .iter()
                            .map(|&(w, e)| {
                                let mut s = prefixes[e as usize].clone();
                                s.push_str(&labels[w as usize]);
                                s
                            })
                            .collect();
                        parts.sort_unstable();
                        for part in parts {
                            text.push_str(&part);
                        }
                    }
                }
            }
            blake2b_hex(text.as_bytes(), digest_size)
        })
        .collect()
}

/// The text NetworkX hashes last in `weisfeiler_lehman_graph_hash`:
/// `str(tuple(subgraph_hash_counts))`, with each step's `Counter` items
/// sorted by label.
pub fn wl_counts_text(steps: &[Vec<String>]) -> String {
    let mut items: Vec<(String, usize)> = Vec::new();
    for labels in steps {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for label in labels {
            *counts.entry(label.as_str()).or_insert(0) += 1;
        }
        let mut step: Vec<(&str, usize)> = counts.into_iter().collect();
        step.sort_unstable();
        items.extend(step.into_iter().map(|(l, c)| (l.to_string(), c)));
    }
    let parts: Vec<String> = items
        .iter()
        .map(|(label, count)| format!("('{label}', {count})"))
        .collect();
    if parts.len() == 1 {
        format!("({},)", parts[0])
    } else {
        format!("({})", parts.join(", "))
    }
}

// --- Property sequences for `could_be_isomorphic` -----------------------------

/// Sorted neighbor sets without self-loops (both directions merged for a
/// directed graph, though only undirected graphs reach the clique code).
fn simple_neighbors(adj: &Csr, n: usize) -> Vec<Vec<u32>> {
    (0..n)
        .map(|v| {
            let mut row: Vec<u32> = adj
                .neighbors(v)
                .iter()
                .copied()
                .filter(|&w| w as usize != v)
                .collect();
            row.sort_unstable();
            row.dedup();
            row
        })
        .collect()
}

fn intersect(a: &[u32], b: &[u32]) -> Vec<u32> {
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

/// Bron-Kerbosch with Tomita pivoting: adds one to `counts[v]` for each
/// member `v` of each maximal clique `r + (subset of p)` found.
fn bron_kerbosch(
    nbrs: &[Vec<u32>],
    r: &mut Vec<u32>,
    p: Vec<u32>,
    x: Vec<u32>,
    counts: &mut [u64],
) {
    if p.is_empty() {
        if x.is_empty() {
            for &v in r.iter() {
                counts[v as usize] += 1;
            }
        }
        return;
    }
    // Pivot: the vertex of p or x with the most neighbors in p.
    let pivot = p
        .iter()
        .chain(x.iter())
        .copied()
        .max_by_key(|&u| intersect(&nbrs[u as usize], &p).len())
        .unwrap();
    let skip = &nbrs[pivot as usize];
    let mut p = p;
    let mut x = x;
    let candidates: Vec<u32> = p
        .iter()
        .copied()
        .filter(|v| skip.binary_search(v).is_err())
        .collect();
    for v in candidates {
        let nv = &nbrs[v as usize];
        r.push(v);
        bron_kerbosch(nbrs, r, intersect(&p, nv), intersect(&x, nv), counts);
        r.pop();
        if let Ok(i) = p.binary_search(&v) {
            p.remove(i);
        }
        if let Err(i) = x.binary_search(&v) {
            x.insert(i, v);
        }
    }
}

/// For each node, the number of maximal cliques containing it (what
/// `Counter(chain.from_iterable(nx.find_cliques(G)))` counts). Self-loops
/// are ignored, as `find_cliques` ignores them.
pub fn maximal_clique_counts(adj: &Csr, n: usize) -> Vec<u64> {
    let nbrs = simple_neighbors(adj, n);
    // Degeneracy order keeps each top-level candidate set small.
    let order = degeneracy_order(&nbrs, n);
    let mut rank = vec![0usize; n];
    for (i, &v) in order.iter().enumerate() {
        rank[v as usize] = i;
    }
    let parts: Vec<Vec<u64>> = order
        .par_iter()
        .fold(
            || vec![0u64; n],
            |mut counts, &v| {
                let (mut p, mut x) = (Vec::new(), Vec::new());
                for &w in &nbrs[v as usize] {
                    if rank[w as usize] > rank[v as usize] {
                        p.push(w);
                    } else {
                        x.push(w);
                    }
                }
                let mut r = vec![v];
                bron_kerbosch(&nbrs, &mut r, p, x, &mut counts);
                counts
            },
        )
        .collect();
    let mut total = vec![0u64; n];
    for part in parts {
        for (t, c) in total.iter_mut().zip(part) {
            *t += c;
        }
    }
    total
}

fn degeneracy_order(nbrs: &[Vec<u32>], n: usize) -> Vec<u32> {
    let mut degree: Vec<usize> = nbrs.iter().map(|r| r.len()).collect();
    let max_deg = degree.iter().copied().max().unwrap_or(0);
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); max_deg + 1];
    for v in 0..n {
        buckets[degree[v]].push(v as u32);
    }
    let mut removed = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut d = 0;
    while order.len() < n {
        d = d.min(max_deg);
        while buckets[d].is_empty() {
            d += 1;
        }
        let v = buckets[d].pop().unwrap() as usize;
        if removed[v] || degree[v] != d {
            continue;
        }
        removed[v] = true;
        order.push(v as u32);
        for &w in &nbrs[v] {
            let w = w as usize;
            if !removed[w] {
                degree[w] -= 1;
                buckets[degree[w]].push(w as u32);
            }
        }
        d = d.saturating_sub(1);
    }
    order
}

// --- Exact matching: isomorphism, induced subgraph isomorphism, monomorphism ---

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Problem {
    /// The two graphs are isomorphic.
    Iso,
    /// The small graph is isomorphic to an induced subgraph of the big one.
    Induced,
    /// The small graph is isomorphic to a subgraph (not necessarily induced).
    Mono,
}

/// One side of a matching problem.
pub struct Side<'a> {
    pub succ: &'a Csr,
    pub pred: Option<&'a Csr>,
    pub n: usize,
    pub labels: &'a [u32],
}

struct Prepared {
    out: Vec<Vec<u32>>,
    inn: Vec<Vec<u32>>,
    loops: Vec<u32>,
    /// NetworkX's degree: undirected (deg, 0) with a self-loop counted
    /// twice; directed (in, out).
    deg: Vec<(u32, u32)>,
    /// Undirected neighbors (both directions), for ordering and for
    /// finding a matched neighbor.
    und: Vec<Vec<u32>>,
}

fn sorted_rows(adj: &Csr, n: usize) -> Vec<Vec<u32>> {
    (0..n)
        .map(|v| {
            let mut row = adj.neighbors(v).to_vec();
            row.sort_unstable();
            row
        })
        .collect()
}

fn prepare(side: &Side, directed: bool) -> Prepared {
    let n = side.n;
    let out = sorted_rows(side.succ, n);
    let loops: Vec<u32> = (0..n)
        .map(|v| u32::from(out[v].binary_search(&(v as u32)).is_ok()))
        .collect();
    let (inn, deg, und) = if directed {
        let inn = sorted_rows(side.pred.expect("directed graphs have predecessors"), n);
        let deg = (0..n)
            .map(|v| (inn[v].len() as u32, out[v].len() as u32))
            .collect();
        let und = (0..n)
            .map(|v| {
                let mut row: Vec<u32> = out[v].iter().chain(inn[v].iter()).copied().collect();
                row.sort_unstable();
                row.dedup();
                row
            })
            .collect();
        (inn, deg, und)
    } else {
        let deg = (0..n)
            .map(|v| (out[v].len() as u32 + loops[v], 0))
            .collect();
        (Vec::new(), deg, out.clone())
    };
    Prepared {
        out,
        inn,
        loops,
        deg,
        und,
    }
}

/// Joint color refinement of both graphs (isomorphism only): colors that
/// any isomorphism must preserve. Returns `None` if the color counts
/// already differ, so the graphs can't be isomorphic.
fn refine(
    a: &Prepared,
    b: &Prepared,
    la: &[u32],
    lb: &[u32],
    directed: bool,
) -> Option<(Vec<u32>, Vec<u32>)> {
    let (na, nb) = (a.deg.len(), b.deg.len());
    let mut ids: HashMap<(u32, (u32, u32), u32), u32> = HashMap::new();
    let mut initial = |l: u32, d: (u32, u32), s: u32| {
        let next = ids.len() as u32;
        *ids.entry((l, d, s)).or_insert(next)
    };
    let mut ca: Vec<u32> = (0..na)
        .map(|v| initial(la[v], a.deg[v], a.loops[v]))
        .collect();
    let mut cb: Vec<u32> = (0..nb)
        .map(|v| initial(lb[v], b.deg[v], b.loops[v]))
        .collect();
    let mut classes = ids.len();
    loop {
        if !same_histogram(&ca, &cb, classes) {
            return None;
        }
        let signature = |p: &Prepared, c: &[u32], v: usize| {
            let mut s: Vec<u32> = vec![c[v]];
            let mut outs: Vec<u32> = p.out[v].iter().map(|&w| c[w as usize]).collect();
            outs.sort_unstable();
            s.extend(outs);
            if directed {
                s.push(u32::MAX);
                let mut ins: Vec<u32> = p.inn[v].iter().map(|&w| c[w as usize]).collect();
                ins.sort_unstable();
                s.extend(ins);
            }
            s
        };
        let sa: Vec<Vec<u32>> = (0..na)
            .into_par_iter()
            .map(|v| signature(a, &ca, v))
            .collect();
        let sb: Vec<Vec<u32>> = (0..nb)
            .into_par_iter()
            .map(|v| signature(b, &cb, v))
            .collect();
        let mut ids: HashMap<&[u32], u32> = HashMap::new();
        let mut next_a = Vec::with_capacity(na);
        for s in &sa {
            let next = ids.len() as u32;
            next_a.push(*ids.entry(s.as_slice()).or_insert(next));
        }
        let mut next_b = Vec::with_capacity(nb);
        for s in &sb {
            let next = ids.len() as u32;
            next_b.push(*ids.entry(s.as_slice()).or_insert(next));
        }
        let new_classes = ids.len();
        ca = next_a;
        cb = next_b;
        if new_classes == classes {
            if !same_histogram(&ca, &cb, new_classes) {
                return None;
            }
            return Some((ca, cb));
        }
        classes = new_classes;
    }
}

fn same_histogram(a: &[u32], b: &[u32], classes: usize) -> bool {
    let mut count = vec![0i64; classes];
    for &c in a {
        count[c as usize] += 1;
    }
    for &c in b {
        count[c as usize] -= 1;
    }
    count.iter().all(|&c| c == 0)
}

/// The order in which small-graph nodes are matched: per component, a
/// breadth-first order from a node with the rarest label and highest
/// degree, preferring nodes with more already ordered neighbors (the idea
/// of VF2++'s ordering; any order gives the same yes/no answer).
fn matching_order(p: &Prepared, labels: &[u32], label_count: &HashMap<u32, usize>) -> Vec<u32> {
    let n = p.und.len();
    let mut placed = vec![false; n];
    let mut used = vec![0u32; n];
    let mut order = Vec::with_capacity(n);
    let rarity = |v: usize| label_count.get(&labels[v]).copied().unwrap_or(0);
    let total_deg = |v: usize| p.deg[v].0 + p.deg[v].1;
    let mut starts: Vec<usize> = (0..n).collect();
    starts.sort_by_key(|&v| (rarity(v), std::cmp::Reverse(total_deg(v)), v));
    for s in starts {
        if placed[s] {
            continue;
        }
        placed[s] = true;
        let mut layer = vec![s];
        while !layer.is_empty() {
            let mut next = Vec::new();
            // Within a layer: most ordered neighbors first, then degree.
            layer.sort_by_key(|&v| {
                (
                    std::cmp::Reverse(used[v]),
                    std::cmp::Reverse(total_deg(v)),
                    v,
                )
            });
            for &v in &layer {
                order.push(v as u32);
                for &w in &p.und[v] {
                    let w = w as usize;
                    used[w] += 1;
                    if !placed[w] {
                        placed[w] = true;
                        next.push(w);
                    }
                }
            }
            layer = next;
        }
    }
    order
}

const NONE: u32 = u32::MAX;

/// Whether `small` maps into `big` as `problem` asks, nodes keeping their
/// labels. For `Iso` the graphs have the same number of nodes.
pub fn has_morphism(small: &Side, big: &Side, directed: bool, problem: Problem) -> bool {
    let (ns, nb) = (small.n, big.n);
    if ns > nb || (problem == Problem::Iso && ns != nb) {
        return false;
    }
    if ns == 0 {
        return true;
    }
    let ps = prepare(small, directed);
    let pb = prepare(big, directed);
    let mut big_by_label: HashMap<u32, Vec<u32>> = HashMap::new();
    for v in 0..nb {
        big_by_label
            .entry(big.labels[v])
            .or_default()
            .push(v as u32);
    }
    let label_count: HashMap<u32, usize> =
        big_by_label.iter().map(|(&l, v)| (l, v.len())).collect();
    let mut small_count: HashMap<u32, usize> = HashMap::new();
    for v in 0..ns {
        *small_count.entry(small.labels[v]).or_insert(0) += 1;
    }
    for (l, &c) in &small_count {
        let available = label_count.get(l).copied().unwrap_or(0);
        if c > available || (problem == Problem::Iso && c != available) {
            return false;
        }
    }
    let colors = if problem == Problem::Iso {
        match refine(&ps, &pb, small.labels, big.labels, directed) {
            None => return false,
            Some(c) => Some(c),
        }
    } else {
        None
    };
    let order = matching_order(&ps, small.labels, &label_count);

    let degree_fits = |u: usize, v: usize| {
        let (a, b) = (ps.deg[u], pb.deg[v]);
        match problem {
            Problem::Iso => a == b,
            _ => a.0 <= b.0 && a.1 <= b.1,
        }
    };
    let loops_fit = |u: usize, v: usize| match problem {
        Problem::Mono => ps.loops[u] <= pb.loops[v],
        _ => ps.loops[u] == pb.loops[v],
    };
    let induced = problem != Problem::Mono;

    let mut map = vec![NONE; ns];
    let mut rev = vec![NONE; nb];

    let candidates = |u: usize, map: &[u32], rev: &[u32]| -> Vec<u32> {
        // A matched neighbor narrows the candidates to its image's neighbors.
        let mut pool: Option<&[u32]> = None;
        for &w in &ps.out[u] {
            if w as usize != u && map[w as usize] != NONE {
                let img = map[w as usize] as usize;
                pool = Some(if directed { &pb.inn[img] } else { &pb.out[img] });
                break;
            }
        }
        if pool.is_none() && directed {
            for &w in &ps.inn[u] {
                if w as usize != u && map[w as usize] != NONE {
                    pool = Some(&pb.out[map[w as usize] as usize]);
                    break;
                }
            }
        }
        let pool = pool.unwrap_or_else(|| {
            big_by_label
                .get(&small.labels[u])
                .map(|v| v.as_slice())
                .unwrap_or(&[])
        });
        pool.iter()
            .copied()
            .filter(|&v| {
                let vi = v as usize;
                rev[vi] == NONE
                    && big.labels[vi] == small.labels[u]
                    && degree_fits(u, vi)
                    && loops_fit(u, vi)
                    && colors.as_ref().is_none_or(|(ca, cb)| ca[u] == cb[vi])
            })
            .collect()
    };

    let feasible = |u: usize, v: usize, map: &[u32], rev: &[u32]| -> bool {
        for &w in &ps.out[u] {
            let m = map[w as usize];
            if m != NONE && pb.out[v].binary_search(&m).is_err() {
                return false;
            }
        }
        if directed {
            for &w in &ps.inn[u] {
                let m = map[w as usize];
                if m != NONE && pb.inn[v].binary_search(&m).is_err() {
                    return false;
                }
            }
        }
        if induced {
            for &w in &pb.out[v] {
                let r = rev[w as usize];
                if r != NONE && ps.out[u].binary_search(&r).is_err() {
                    return false;
                }
            }
            if directed {
                for &w in &pb.inn[v] {
                    let r = rev[w as usize];
                    if r != NONE && ps.inn[u].binary_search(&r).is_err() {
                        return false;
                    }
                }
            }
        }
        true
    };

    // Iterative depth-first search over `order`.
    let mut stack: Vec<(Vec<u32>, usize)> = Vec::with_capacity(ns);
    stack.push((candidates(order[0] as usize, &map, &rev), 0));
    while !stack.is_empty() {
        let depth = stack.len() - 1;
        let (cands, next) = &mut stack[depth];
        let u = order[depth] as usize;
        if map[u] != NONE {
            // Undo this level's previous choice before trying the next.
            rev[map[u] as usize] = NONE;
            map[u] = NONE;
        }
        let mut chosen = None;
        while *next < cands.len() {
            let v = cands[*next] as usize;
            *next += 1;
            if feasible(u, v, &map, &rev) {
                chosen = Some(v);
                break;
            }
        }
        match chosen {
            None => {
                stack.pop();
            }
            Some(v) => {
                map[u] = v as u32;
                rev[v] = u as u32;
                if depth + 1 == ns {
                    return true;
                }
                let c = candidates(order[depth + 1] as usize, &map, &rev);
                stack.push((c, 0));
            }
        }
    }
    false
}

// --- Trees ---------------------------------------------------------------------

/// Centers of a tree in node order (NetworkX's `center` on a tree).
pub fn tree_centers(adj: &Csr, n: usize) -> Vec<u32> {
    let bfs = |start: usize| {
        let mut parent = vec![NONE; n];
        let mut dist = vec![usize::MAX; n];
        dist[start] = 0;
        let mut queue = vec![start];
        let mut head = 0;
        while head < queue.len() {
            let v = queue[head];
            head += 1;
            for &w in adj.neighbors(v) {
                let w = w as usize;
                if dist[w] == usize::MAX {
                    dist[w] = dist[v] + 1;
                    parent[w] = v as u32;
                    queue.push(w);
                }
            }
        }
        (*queue.last().unwrap(), parent, dist)
    };
    let (a, _, _) = bfs(0);
    let (b, parent, dist) = bfs(a);
    let length = dist[b];
    let mut path = vec![b as u32];
    let mut v = b;
    while v != a {
        v = parent[v] as usize;
        path.push(v as u32);
    }
    let mut centers = vec![path[length / 2]];
    if length % 2 == 1 {
        centers.push(path[length / 2 + 1]);
    }
    centers.sort_unstable();
    centers
}

/// Breadth-first order from `root` (as `nx.bfs_edges`), and each node's
/// children in that order, all as BFS ranks.
fn bfs_ranks(adj: &Csr, n: usize, root: usize) -> (Vec<u32>, Vec<Vec<u32>>, Vec<u32>) {
    let mut rank = vec![NONE; n];
    rank[root] = 0;
    let mut order = vec![root as u32];
    let mut children: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut depth = vec![0u32; n];
    let mut head = 0;
    while head < order.len() {
        let v = order[head] as usize;
        head += 1;
        for &w in adj.neighbors(v) {
            let w = w as usize;
            if rank[w] == NONE {
                rank[w] = order.len() as u32;
                depth[order.len()] = depth[rank[v] as usize] + 1;
                children[rank[v] as usize].push(rank[w]);
                order.push(w as u32);
            }
        }
    }
    children.truncate(order.len());
    depth.truncate(order.len());
    (order, children, depth)
}

/// NetworkX's `rooted_tree_isomorphism` on two trees: the node pairs
/// `(t1 node, t2 node)` in output order, empty if not isomorphic.
/// `descending` is 3.5+'s child order (sorted with `reverse=True`, then
/// walked with a stack); otherwise 3.4's ascending order walked
/// recursively. Both visit pairs depth first; the label ranks they assign
/// differ, and so does the order.
pub fn rooted_tree_isomorphism(
    t1: &Csr,
    n1: usize,
    root1: usize,
    t2: &Csr,
    n2: usize,
    root2: usize,
    descending: bool,
) -> Vec<(u32, u32)> {
    // NetworkX's combined tree `dT`: 0 is a fake root, t1's nodes are
    // 1..=n1 in BFS order and t2's are n1+1..=n1+n2.
    let (order1, children1, depth1) = bfs_ranks(t1, n1, root1);
    let (order2, children2, depth2) = bfs_ranks(t2, n2, root2);
    let off2 = order1.len() + 1;
    let total = off2 + order2.len();
    let mut children: Vec<Vec<u32>> = vec![Vec::new(); total];
    let mut level = vec![0u32; total];
    for (i, kids) in children1.into_iter().enumerate() {
        children[i + 1] = kids.into_iter().map(|c| c + 1).collect();
        level[i + 1] = depth1[i] + 1;
    }
    for (i, kids) in children2.into_iter().enumerate() {
        children[off2 + i] = kids.into_iter().map(|c| c + off2 as u32).collect();
        level[off2 + i] = depth2[i] + 1;
    }
    let h = *level.iter().max().unwrap() as usize;
    let mut by_level: Vec<Vec<u32>> = vec![Vec::new(); h + 1];
    for v in 1..total {
        by_level[level[v] as usize].push(v as u32);
    }
    let mut label = vec![0u32; total];
    let mut ordered_labels: Vec<Vec<u32>> = vec![Vec::new(); total];
    let mut ordered_children: Vec<Vec<u32>> = vec![Vec::new(); total];
    for i in (1..h).rev() {
        for &v in &by_level[i] {
            let v = v as usize;
            if children[v].is_empty() {
                continue;
            }
            let mut s: Vec<(u32, u32)> = children[v]
                .iter()
                .map(|&c| (label[c as usize], c))
                .collect();
            s.sort_unstable();
            if descending {
                s.reverse();
            }
            ordered_labels[v] = s.iter().map(|p| p.0).collect();
            ordered_children[v] = s.iter().map(|p| p.1).collect();
        }
        let mut forlabel: Vec<u32> = by_level[i].clone();
        forlabel.sort_by(|&a, &b| {
            ordered_labels[a as usize]
                .cmp(&ordered_labels[b as usize])
                .then(a.cmp(&b))
        });
        let mut current = 0;
        for (k, &v) in forlabel.iter().enumerate() {
            if k != 0 && ordered_labels[v as usize] != ordered_labels[forlabel[k - 1] as usize] {
                current += 1;
            }
            label[v as usize] = current;
        }
    }
    let mut pairs = Vec::new();
    if label[1] != 0 || label[off2] != 0 {
        return pairs;
    }
    let old_name = |v: u32| -> u32 {
        let v = v as usize;
        if v < off2 {
            order1[v - 1]
        } else {
            order2[v - off2]
        }
    };
    let mut stack = vec![(1u32, off2 as u32)];
    while let Some((v, w)) = stack.pop() {
        pairs.push((old_name(v), old_name(w)));
        let zipped = ordered_children[v as usize]
            .iter()
            .zip(ordered_children[w as usize].iter())
            .map(|(&x, &y)| (x, y));
        if descending {
            stack.extend(zipped);
        } else {
            // A recursive preorder: push in reverse so the first pops first.
            let kids: Vec<(u32, u32)> = zipped.collect();
            stack.extend(kids.into_iter().rev());
        }
    }
    pairs
}

/// Depth of the deepest node below `root` (a tree).
pub fn tree_height(adj: &Csr, n: usize, root: usize) -> u32 {
    let (_, _, depth) = bfs_ranks(adj, n, root);
    depth.into_iter().max().unwrap_or(0)
}
