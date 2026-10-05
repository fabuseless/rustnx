//! Batch 18: seeded random graph generators.
//!
//! Each generator replays NetworkX's code draw for draw on CPython's
//! Mersenne Twister (`pyrandom::Mt19937`), and records the graph as the
//! adjacency rows NetworkX's dicts end up with: every row lists neighbors
//! in insertion order, with removed edges taken out (dicts keep insertion
//! order, and re-adding a key appends it). `lib.rs` turns the rows into
//! the dicts of a real NetworkX graph.
//!
//! `None` means "let NetworkX run": the rare cases where NetworkX raises
//! (an empty `choice`, a zero `log`) or where rustnx can't be sure to
//! match (a float within rounding error of a threshold).

use std::collections::HashSet;

use super::pyrandom::Mt19937;
use super::pyset::PySet;

/// Adjacency rows in NetworkX's insertion order. Undirected graphs use
/// `succ` alone (an edge sits in both rows; a self-loop once). Directed
/// graphs also keep `pred`. `order` is the node insertion order when it
/// isn't `0..n`.
pub struct Built {
    pub succ: Vec<Vec<u32>>,
    pub pred: Option<Vec<Vec<u32>>>,
    pub order: Option<Vec<u32>>,
}

impl Built {
    pub fn new(n: usize, directed: bool) -> Self {
        Built {
            succ: vec![Vec::new(); n],
            pred: directed.then(|| vec![Vec::new(); n]),
            order: None,
        }
    }

    pub fn n(&self) -> usize {
        self.succ.len()
    }

    fn add_node(&mut self) {
        self.succ.push(Vec::new());
        if let Some(pred) = &mut self.pred {
            pred.push(Vec::new());
        }
    }

    pub fn has_edge(&self, u: u32, v: u32) -> bool {
        let a = &self.succ[u as usize];
        let b = match &self.pred {
            Some(pred) => &pred[v as usize],
            None => &self.succ[v as usize],
        };
        if a.len() <= b.len() {
            a.contains(&v)
        } else {
            b.contains(&u)
        }
    }

    /// `G.add_edge(u, v)` for an edge known to be new.
    pub fn push_edge(&mut self, u: u32, v: u32) {
        self.succ[u as usize].push(v);
        match &mut self.pred {
            Some(pred) => pred[v as usize].push(u),
            None => {
                if u != v {
                    self.succ[v as usize].push(u);
                }
            }
        }
    }

    /// `G.add_edge(u, v)`: an existing edge keeps its place.
    pub fn add_edge(&mut self, u: u32, v: u32) {
        if !self.has_edge(u, v) {
            self.push_edge(u, v);
        }
    }

    /// `G.remove_edge(u, v)`; false if there is no such edge (NetworkX
    /// raises).
    pub fn remove_edge(&mut self, u: u32, v: u32) -> bool {
        fn take(row: &mut Vec<u32>, x: u32) -> bool {
            match row.iter().position(|&y| y == x) {
                Some(i) => {
                    row.remove(i);
                    true
                }
                None => false,
            }
        }
        if !take(&mut self.succ[u as usize], v) {
            return false;
        }
        match &mut self.pred {
            Some(pred) => take(&mut pred[v as usize], u),
            None => u == v || take(&mut self.succ[v as usize], u),
        };
        true
    }

    /// `G.degree(u)` of an undirected graph without self-loops.
    #[inline]
    fn degree(&self, u: u32) -> usize {
        self.succ[u as usize].len()
    }

    /// Whether an undirected graph is connected (`nx.is_connected`, for
    /// at least one node).
    fn is_connected(&self) -> bool {
        let n = self.n();
        let mut seen = vec![false; n];
        let mut stack = vec![0u32];
        seen[0] = true;
        let mut count = 1;
        while let Some(u) = stack.pop() {
            for &v in &self.succ[u as usize] {
                if !seen[v as usize] {
                    seen[v as usize] = true;
                    count += 1;
                    stack.push(v);
                }
            }
        }
        count == n
    }
}

/// `complete_graph(n)` (or the directed one): every row lists the other
/// nodes in order.
pub fn complete(n: usize, directed: bool) -> Built {
    let row = |u: usize| {
        (0..n as u32)
            .filter(|&v| v as usize != u)
            .collect::<Vec<_>>()
    };
    let succ: Vec<Vec<u32>> = (0..n).map(row).collect();
    let pred = directed.then(|| succ.clone());
    Built {
        succ,
        pred,
        order: None,
    }
}

/// `path_graph(n)`.
fn path(n: usize) -> Built {
    let mut b = Built::new(n, false);
    for u in 1..n as u32 {
        b.push_edge(u - 1, u);
    }
    b
}

/// The hashes of the ints `0..n` (CPython hashes small ints to themselves).
fn int_hashes(n: usize) -> Vec<i64> {
    (0..n as i64).collect()
}

/// `_random_subset(seq, m, rng)`: m distinct items drawn with `choice`,
/// returned in the iteration order of the set NetworkX collects them in.
fn random_subset(seq: &[u32], m: usize, rng: &mut Mt19937, hashes: &[i64]) -> Vec<u32> {
    let mut targets = PySet::default();
    let mut len = 0;
    while len < m {
        let x = rng.choice(seq);
        if !targets.contains(x, hashes) {
            targets.add(x, hashes);
            len += 1;
        }
    }
    targets.iter().collect()
}

/// `int(x)` of a non-negative float, clamped: `None` for 2^62 or more,
/// which exceeds every count a generator compares it with.
fn trunc_clamped(x: f64) -> Option<i64> {
    (x < 4.611_686_018_427_388e18).then_some(x as i64)
}

/// `gnp_random_graph` for `0 < p < 1` (or NaN): a draw per pair.
pub fn gnp(n: usize, p: f64, directed: bool, rng: &mut Mt19937) -> Built {
    let mut b = Built::new(n, directed);
    let n = n as u32;
    for u in 0..n {
        let start = if directed { 0 } else { u + 1 };
        for v in start..n {
            if v != u && rng.random() < p {
                b.push_edge(u, v);
            }
        }
    }
    b
}

/// The geometric skipping loop shared by `fast_gnp_random_graph` and
/// bipartite `random_graph`: rows `v` of width `width(v)`, calling `emit`
/// for each picked `(v, w)`. `None` where NetworkX divides by zero.
fn skip_loop(
    rows: i64,
    width: impl Fn(i64) -> i64,
    first_v: i64,
    lp: f64,
    rng: &mut Mt19937,
    mut emit: impl FnMut(i64, i64),
) {
    let mut v = first_v;
    let mut w: i64 = -1;
    while v < rows {
        let lr = (1.0 - rng.random()).ln();
        match trunc_clamped(lr / lp) {
            Some(skip) => w = w + 1 + skip,
            None => {
                // `w` exceeds what the remaining rows can absorb.
                return;
            }
        }
        while v < rows && w >= width(v) {
            w -= width(v);
            v += 1;
        }
        if v < rows {
            emit(v, w);
        }
    }
}

/// `fast_gnp_random_graph` for `0 < p < 1`; `None` if `log(1 - p)` is 0
/// (NetworkX divides by it).
pub fn fast_gnp(n: usize, p: f64, directed: bool, rng: &mut Mt19937) -> Option<Built> {
    let lp = (1.0 - p).ln();
    if lp == 0.0 || lp.is_nan() {
        return None;
    }
    let mut b = Built::new(n, directed);
    let rows = n as i64;
    if directed {
        skip_loop(
            rows,
            |v| v,
            1,
            lp,
            rng,
            |v, w| b.push_edge(w as u32, v as u32),
        );
    }
    skip_loop(
        rows,
        |v| v,
        1,
        lp,
        rng,
        |v, w| b.push_edge(v as u32, w as u32),
    );
    Some(b)
}

/// Bipartite `random_graph` for `0 < p < 1`: nodes `0..n` and `n..n+m`.
pub fn bipartite_random(
    n: usize,
    m: usize,
    p: f64,
    directed: bool,
    rng: &mut Mt19937,
) -> Option<Built> {
    let lp = (1.0 - p).ln();
    if lp == 0.0 || lp.is_nan() {
        return None;
    }
    let mut b = Built::new(n + m, directed);
    let (rows, width) = (n as i64, m as i64);
    let top = n as i64;
    skip_loop(
        rows,
        |_| width,
        0,
        lp,
        rng,
        |v, w| b.push_edge(v as u32, (top + w) as u32),
    );
    if directed {
        skip_loop(
            rows,
            |_| width,
            0,
            lp,
            rng,
            |v, w| b.push_edge((top + w) as u32, v as u32),
        );
    }
    Some(b)
}

/// `gnm_random_graph` for `m` below the number of possible edges.
pub fn gnm(n: usize, m: u64, directed: bool, rng: &mut Mt19937) -> Built {
    let mut b = Built::new(n, directed);
    let mut edges: HashSet<(u32, u32)> = HashSet::new();
    let mut count = 0;
    while count < m {
        let u = rng.below(n) as u32;
        let v = rng.below(n) as u32;
        let key = if directed || u < v { (u, v) } else { (v, u) };
        if u == v || edges.contains(&key) {
            continue;
        }
        edges.insert(key);
        b.push_edge(u, v);
        count += 1;
    }
    b
}

/// Bipartite `gnmk_random_graph` (`1 < n`, `1 < m`, `k < n * m`): `bottom`
/// is NetworkX's list of the bottom nodes.
pub fn gnmk(n: usize, bottom: &[u32], k: u64, directed: bool, rng: &mut Mt19937) -> Built {
    let mut b = Built::new(n + bottom.len(), directed);
    let mut edges: HashSet<(u32, u32)> = HashSet::new();
    let mut count = 0;
    while count < k {
        let u = rng.below(n) as u32;
        let v = rng.choice(bottom);
        if !edges.insert((u, v)) {
            continue;
        }
        b.push_edge(u, v);
        count += 1;
    }
    b
}

/// `dense_gnm_random_graph` for `1 <= m < n (n - 1) / 2`.
pub fn dense_gnm(n: usize, m: u64, rng: &mut Mt19937) -> Built {
    let mut b = Built::new(n, false);
    let mmax = (n as u64) * (n as u64 - 1) / 2;
    let (mut u, mut v) = (0u32, 1u32);
    let (mut t, mut k) = (0u64, 0u64);
    loop {
        // `t < mmax` here: once `mmax - t <= m - k` every draw succeeds.
        if rng.randbelow(mmax - t) < m - k {
            b.push_edge(u, v);
            k += 1;
            if k == m {
                return b;
            }
        }
        t += 1;
        v += 1;
        if v as usize == n {
            u += 1;
            v = u + 1;
        }
    }
}

/// The preferential attachment loop of `barabasi_albert_graph` and
/// `dual_barabasi_albert_graph`, from the star on `m0 + 1` nodes; `m_for`
/// picks each new node's number of edges.
fn ba_from_star(
    n: usize,
    m0: usize,
    rng: &mut Mt19937,
    mut m_for: impl FnMut(&mut Mt19937) -> usize,
) -> Built {
    let mut b = Built::new(n, false);
    for v in 1..=m0 as u32 {
        b.push_edge(0, v);
    }
    let hashes = int_hashes(n);
    // `[n for n, d in G.degree() for _ in range(d)]`
    let mut repeated: Vec<u32> = vec![0; m0];
    repeated.extend(1..=m0 as u32);
    for source in (m0 + 1) as u32..n as u32 {
        let m = m_for(rng);
        let targets = random_subset(&repeated, m, rng, &hashes);
        for &t in &targets {
            b.push_edge(source, t);
        }
        repeated.extend_from_slice(&targets);
        repeated.extend(std::iter::repeat_n(source, m));
    }
    b
}

/// `barabasi_albert_graph` with the default initial graph (`1 <= m < n`).
pub fn barabasi_albert(n: usize, m: usize, rng: &mut Mt19937) -> Built {
    ba_from_star(n, m, rng, |_| m)
}

/// `dual_barabasi_albert_graph` with the default initial graph and
/// `0 < p < 1`.
pub fn dual_barabasi_albert(n: usize, m1: usize, m2: usize, p: f64, rng: &mut Mt19937) -> Built {
    ba_from_star(
        n,
        m1.max(m2),
        rng,
        |rng| {
            if rng.random() < p {
                m1
            } else {
                m2
            }
        },
    )
}

/// `list.remove(x)`: the first occurrence.
fn remove_first(list: &mut Vec<u32>, x: u32) -> bool {
    match list.iter().position(|&y| y == x) {
        Some(i) => {
            list.remove(i);
            true
        }
        None => false,
    }
}

/// `seed.choice([nd for nd in pref if nd not in banned])`, without
/// building the list; `None` if it would be empty.
fn choice_excluding(
    pref: &[u32],
    banned: &[u32],
    mark: &mut [bool],
    rng: &mut Mt19937,
) -> Option<u32> {
    for &x in banned {
        mark[x as usize] = true;
    }
    let count = pref.iter().filter(|&&x| !mark[x as usize]).count();
    let picked = if count == 0 {
        None
    } else {
        let mut i = rng.below(count);
        pref.iter().copied().find(|&x| {
            if mark[x as usize] {
                return false;
            }
            if i == 0 {
                return true;
            }
            i -= 1;
            false
        })
    };
    for &x in banned {
        mark[x as usize] = false;
    }
    picked
}

/// `extended_barabasi_albert_graph` (`1 <= m < n`, `p + q < 1`). `None`
/// where NetworkX would call `choice` on an empty list.
pub fn extended_barabasi_albert(
    n: usize,
    m: usize,
    p: f64,
    q: f64,
    rng: &mut Mt19937,
) -> Option<Built> {
    let mut b = Built::new(m, false);
    let hashes = int_hashes(n);
    let mut mark = vec![false; n];
    let mut pref: Vec<u32> = (0..m as u32).collect();
    let mut size: usize = 0;
    let mut new_node = m;
    let pq = p + q;
    while new_node < n {
        let a = rng.random();
        let len = b.n();
        let clique_degree = len - 1;
        // `clique_size` is `len * (len - 1) / 2`, a whole number.
        let clique_size = len * clique_degree / 2;
        if a < p && size + m <= clique_size {
            let mut eligible: Vec<u32> = (0..len as u32)
                .filter(|&v| b.degree(v) < clique_degree)
                .collect();
            for _ in 0..m {
                if eligible.is_empty() {
                    return None;
                }
                let src = rng.choice(&eligible);
                let mut banned = b.succ[src as usize].clone();
                banned.push(src);
                let dest = choice_excluding(&pref, &banned, &mut mark, rng)?;
                b.push_edge(src, dest);
                size += 1;
                pref.push(src);
                pref.push(dest);
                if b.degree(src) == clique_degree {
                    remove_first(&mut eligible, src);
                }
                if b.degree(dest) == clique_degree {
                    remove_first(&mut eligible, dest);
                }
            }
        } else if p <= a && a < pq && m <= size && size < clique_size {
            let mut eligible: Vec<u32> = (0..len as u32)
                .filter(|&v| {
                    let d = b.degree(v);
                    0 < d && d < clique_degree
                })
                .collect();
            for _ in 0..m {
                if eligible.is_empty() {
                    return None;
                }
                let node = rng.choice(&eligible);
                let mut nbrs = b.succ[node as usize].clone();
                if nbrs.is_empty() {
                    return None; // can't happen: eligible nodes have edges
                }
                let src = rng.choice(&nbrs);
                nbrs.push(node);
                let dest = choice_excluding(&pref, &nbrs, &mut mark, rng)?;
                b.remove_edge(node, src);
                b.push_edge(node, dest);
                remove_first(&mut pref, src);
                pref.push(dest);
                if b.degree(src) == 0 {
                    remove_first(&mut eligible, src);
                }
                if eligible.contains(&dest) {
                    if b.degree(dest) == clique_degree {
                        remove_first(&mut eligible, dest);
                    }
                } else if b.degree(dest) == 1 {
                    eligible.push(dest);
                }
            }
        } else {
            let targets = random_subset(&pref, m, rng, &hashes);
            b.add_node();
            for &t in &targets {
                b.push_edge(new_node as u32, t);
            }
            size += m;
            pref.extend_from_slice(&targets);
            pref.extend(std::iter::repeat_n(new_node as u32, m + 1));
            new_node += 1;
        }
    }
    Some(b)
}

/// `watts_strogatz_graph` for `k < n`; `None` if NetworkX would fail to
/// remove an edge.
pub fn watts_strogatz(n: usize, k: usize, p: f64, rng: &mut Mt19937) -> Option<Built> {
    let mut b = Built::new(n, false);
    let nn = n as u32;
    for j in 1..=(k / 2) as u32 {
        for u in 0..nn {
            b.add_edge(u, (u + j) % nn);
        }
    }
    for j in 1..=(k / 2) as u32 {
        for u in 0..nn {
            let v = (u + j) % nn;
            if rng.random() < p {
                let mut w = rng.below(n) as u32;
                let mut skipped = false;
                while w == u || b.has_edge(u, w) {
                    w = rng.below(n) as u32;
                    if b.degree(u) >= n - 1 {
                        skipped = true;
                        break;
                    }
                }
                if !skipped {
                    if !b.remove_edge(u, v) {
                        return None;
                    }
                    b.push_edge(u, w);
                }
            }
        }
    }
    Some(b)
}

/// `connected_watts_strogatz_graph`: up to `tries` Watts-Strogatz graphs
/// (`k < n`), the first connected one, or `Ok(None)` when all fail.
pub fn connected_watts_strogatz(
    n: usize,
    k: usize,
    p: f64,
    tries: u64,
    rng: &mut Mt19937,
) -> Option<Option<Built>> {
    for _ in 0..tries {
        let b = watts_strogatz(n, k, p, rng)?;
        if b.is_connected() {
            return Some(Some(b));
        }
    }
    Some(None)
}

/// `newman_watts_strogatz_graph` for `k < n`.
pub fn newman_watts_strogatz(n: usize, k: usize, p: f64, rng: &mut Mt19937) -> Built {
    let mut b = Built::new(n, false);
    let nn = n as u32;
    for j in 1..=(k / 2) as u32 {
        for u in 0..nn {
            b.add_edge(u, (u + j) % nn);
        }
    }
    // `list(G.edges())`: each edge from its earlier end.
    let edges: Vec<(u32, u32)> = (0..nn)
        .flat_map(|u| {
            b.succ[u as usize]
                .iter()
                .filter(move |&&v| v >= u)
                .map(move |&v| (u, v))
        })
        .collect();
    for (u, _) in edges {
        if rng.random() < p {
            let mut w = rng.below(n) as u32;
            let mut skipped = false;
            while w == u || b.has_edge(u, w) {
                w = rng.below(n) as u32;
                if b.degree(u) >= n - 1 {
                    skipped = true;
                    break;
                }
            }
            if !skipped {
                b.push_edge(u, w);
            }
        }
    }
    b
}

/// `powerlaw_cluster_graph` (`1 <= m <= n`). NetworkX pops the targets
/// from a fresh set, which yields them in iteration order.
pub fn powerlaw_cluster(n: usize, m: usize, p: f64, rng: &mut Mt19937) -> Built {
    let mut b = Built::new(m, false);
    let hashes = int_hashes(n);
    let mut repeated: Vec<u32> = (0..m as u32).collect();
    for source in m as u32..n as u32 {
        let possible = random_subset(&repeated, m, rng, &hashes);
        let mut pops = possible.into_iter();
        b.add_node();
        let mut target = pops.next().expect("m targets");
        b.add_edge(source, target);
        repeated.push(target);
        let mut count = 1;
        while count < m {
            if rng.random() < p {
                let hood: Vec<u32> = b.succ[target as usize]
                    .iter()
                    .copied()
                    .filter(|&x| x != source && !b.has_edge(source, x))
                    .collect();
                if !hood.is_empty() {
                    let nbr = rng.choice(&hood);
                    b.add_edge(source, nbr);
                    repeated.push(nbr);
                    count += 1;
                    continue;
                }
            }
            target = pops.next().expect("m targets");
            b.add_edge(source, target);
            repeated.push(target);
            count += 1;
        }
        repeated.extend(std::iter::repeat_n(source, m));
    }
    b
}

/// CPython's `hash((a, b))` for ints `0 <= a, b < 2^61 - 1` (tuplehash,
/// the xxHash-based one of 3.8 and later).
pub fn pair_hash(a: u64, b: u64) -> i64 {
    const P1: u64 = 11_400_714_785_074_694_791;
    const P2: u64 = 14_029_467_366_897_019_727;
    const P5: u64 = 2_870_177_450_012_600_261;
    let mut acc = P5;
    for lane in [a, b] {
        acc = acc.wrapping_add(lane.wrapping_mul(P2));
        acc = acc.rotate_left(31);
        acc = acc.wrapping_mul(P1);
    }
    acc = acc.wrapping_add(2 ^ (P5 ^ 3_527_539));
    if acc == u64::MAX {
        return 1_546_275_796;
    }
    acc as i64
}

/// `random_regular_graph` (`0 < d < n`, `n * d` even): NetworkX's
/// `_try_creation` until it succeeds, then the edges in the iteration
/// order of its set of tuples.
pub fn random_regular(d: usize, n: usize, rng: &mut Mt19937) -> Built {
    loop {
        if let Some(edges) = regular_try(d, n, rng) {
            let mut b = Built::new(n, false);
            for (u, v) in edges {
                b.push_edge(u, v);
            }
            return b;
        }
    }
}

fn regular_try(d: usize, n: usize, rng: &mut Mt19937) -> Option<Vec<(u32, u32)>> {
    let mut members: HashSet<(u32, u32)> = HashSet::new();
    let mut ids: Vec<(u32, u32)> = Vec::new();
    let mut hashes: Vec<i64> = Vec::new();
    let mut table = PySet::default();
    let mut stubs: Vec<u32> = Vec::with_capacity(n * d);
    for _ in 0..d {
        stubs.extend(0..n as u32);
    }
    // `potential_edges`: a defaultdict, so counts in first-seen order.
    let mut slot = vec![u32::MAX; n];
    while !stubs.is_empty() {
        let mut potential: Vec<(u32, u32)> = Vec::new();
        rng.shuffle(&mut stubs);
        // `zip(stubiter, stubiter)`: consecutive pairs.
        for i in 0..stubs.len() / 2 {
            let (a, c) = (stubs[2 * i], stubs[2 * i + 1]);
            let (s1, s2) = (a.min(c), a.max(c));
            if s1 != s2 && !members.contains(&(s1, s2)) {
                members.insert((s1, s2));
                hashes.push(pair_hash(s1 as u64, s2 as u64));
                table.add(ids.len() as u32, &hashes);
                ids.push((s1, s2));
            } else {
                for s in [s1, s2] {
                    if slot[s as usize] == u32::MAX {
                        slot[s as usize] = potential.len() as u32;
                        potential.push((s, 0));
                    }
                    potential[slot[s as usize] as usize].1 += 1;
                }
            }
        }
        for &(s, _) in &potential {
            slot[s as usize] = u32::MAX;
        }
        // `_suitable`, quirk included: the inner loop swaps `s1` with the
        // smaller node, so `s1` becomes the running minimum and the loop
        // may run past the outer node (only `s1 == s2` stops it).
        if !potential.is_empty() {
            let suitable = potential.iter().any(|&(x, _)| {
                let mut s1 = x;
                for &(s2, _) in &potential {
                    if s1 == s2 {
                        break;
                    }
                    let (a, b) = (s1.min(s2), s1.max(s2));
                    s1 = a;
                    if !members.contains(&(a, b)) {
                        return true;
                    }
                }
                false
            });
            if !suitable {
                return None;
            }
        }
        stubs = potential
            .iter()
            .flat_map(|&(s, c)| std::iter::repeat_n(s, c as usize))
            .collect();
    }
    Some(table.iter().map(|id| ids[id as usize]).collect())
}

/// `gn_graph` with the default kernel (`n >= 2`); `cumulative` picks the
/// NetworkX 3.6+ `cumulative_distribution` (exact prefix sums, each
/// divided by the total) over the older running sum of `d / total`.
/// `None` if a draw is exactly 0.0 (NetworkX would pick node -1).
pub fn gn(n: usize, cumulative: bool, rng: &mut Mt19937) -> Option<Built> {
    let mut b = Built::new(n, true);
    b.push_edge(1, 0);
    let mut ds: Vec<u64> = vec![1, 1];
    // A Fenwick tree over `ds` for the newer distribution.
    let mut tree = vec![0u64; n + 1];
    let fen_add = |tree: &mut Vec<u64>, i: usize, x: u64| {
        let mut i = i + 1;
        while i < tree.len() {
            tree[i] += x;
            i += 1 << i.trailing_zeros();
        }
    };
    fen_add(&mut tree, 0, 1);
    fen_add(&mut tree, 1, 1);
    let mut total: u64 = 2;
    let top = if n > 0 {
        1usize << (usize::BITS - 1 - n.leading_zeros())
    } else {
        0
    };
    for source in 2..n {
        let s = rng.random();
        if s == 0.0 {
            return None;
        }
        let len = ds.len();
        // `bisect_left(cdf, s) - 1`: the last index whose cdf is below s.
        let target = if cumulative {
            let t = total as f64;
            let (mut pos, mut sum) = (0usize, 0u64);
            let mut step = top;
            while step > 0 {
                let next = pos + step;
                if next <= len && ((sum + tree[next]) as f64) / t < s {
                    pos = next;
                    sum += tree[next];
                }
                step >>= 1;
            }
            pos
        } else {
            let t = total as f64;
            let mut c = 0.0f64;
            let mut i = 0;
            while i < len {
                let next = c + ds[i] as f64 / t;
                if next >= s {
                    break;
                }
                c = next;
                i += 1;
            }
            if i == len {
                // NetworkX would add a node past the end, then fail.
                return None;
            }
            i
        };
        b.push_edge(source as u32, target as u32);
        ds.push(1);
        fen_add(&mut tree, source, 1);
        ds[target] += 1;
        fen_add(&mut tree, target, 1);
        total += 2;
    }
    Some(b)
}

/// `gnr_graph` (`n >= 2`).
pub fn gnr(n: usize, p: f64, rng: &mut Mt19937) -> Built {
    let mut b = Built::new(n, true);
    for source in 1..n {
        let mut target = rng.below(source);
        if rng.random() < p && target != 0 {
            target = b.succ[target][0] as usize;
        }
        b.push_edge(source as u32, target as u32);
    }
    b
}

/// `gnc_graph` (`n >= 2`).
pub fn gnc(n: usize, rng: &mut Mt19937) -> Built {
    let mut b = Built::new(n, true);
    for source in 1..n {
        let target = rng.below(source);
        let succs = b.succ[target].clone();
        for s in succs {
            b.push_edge(source as u32, s);
        }
        b.push_edge(source as u32, target as u32);
    }
    b
}

/// `random_uniform_k_out_graph`: each node's out-neighbors, drawn from
/// the sorted list of nodes (less the node itself without self-loops;
/// NetworkX lists a set of small ints, which iterates in order). With
/// replacement a neighbor may repeat (a multigraph). `None` where
/// NetworkX would fail (an empty `choice`, a `sample` too large).
pub fn uniform_k_out(
    n: usize,
    k: usize,
    self_loops: bool,
    with_replacement: bool,
    rng: &mut Mt19937,
) -> Option<Vec<Vec<u32>>> {
    let all: Vec<u32> = (0..n as u32).collect();
    let mut rows = Vec::with_capacity(n);
    let mut others: Vec<u32> = Vec::with_capacity(n);
    for u in 0..n as u32 {
        let nodes: &[u32] = if self_loops {
            &all
        } else {
            others.clear();
            others.extend((0..n as u32).filter(|&v| v != u));
            &others
        };
        if with_replacement {
            if k > 0 && nodes.is_empty() {
                return None;
            }
            rows.push((0..k).map(|_| rng.choice(nodes)).collect());
        } else {
            if k > nodes.len() {
                return None;
            }
            rows.push(rng.sample(nodes, k));
        }
    }
    Some(rows)
}

/// `random_lobster_graph` (`0 <= n < 2^52`, `p1, p2 < 1`).
pub fn lobster(n: u64, p1: f64, p2: f64, rng: &mut Mt19937) -> Built {
    // `int(2 * seed.random() * n + 0.5)`
    let llen = (2.0 * rng.random() * n as f64 + 0.5) as usize;
    let mut b = path(llen);
    let mut current = llen as i64 - 1;
    for v in 0..llen as u32 {
        while rng.random() < p1 {
            current += 1;
            b.add_node();
            b.push_edge(v, current as u32);
            let cat = current as u32;
            while rng.random() < p2 {
                current += 1;
                b.add_node();
                b.push_edge(cat, current as u32);
            }
        }
    }
    b
}

/// `random_tournament(n)`: nodes join in the order the edges name them.
pub fn tournament(n: usize, rng: &mut Mt19937) -> Built {
    let mut b = Built::new(n, true);
    let mut seen = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for u in 0..n as u32 {
        for v in u + 1..n as u32 {
            let (a, c) = if rng.random() < 0.5 { (u, v) } else { (v, u) };
            for x in [a, c] {
                if !seen[x as usize] {
                    seen[x as usize] = true;
                    order.push(x);
                }
            }
            b.push_edge(a, c);
        }
    }
    b.order = Some(order);
    b
}

/// One block pair's candidate edges in `stochastic_block_model`, in
/// `itertools` order, as a cursor that can skip ahead.
enum Pairs<'a> {
    /// `product(a, b)`.
    Product { a: &'a [u32], b: &'a [u32], i: u64 },
    /// `permutations(a, 2)`.
    Permutations { a: &'a [u32], i: u64 },
    /// `combinations(a, 2)`, then (`loops`) `zip(a, a)`.
    Combinations {
        a: &'a [u32],
        row: usize,
        col: usize,
        loops: bool,
        tail: usize,
    },
}

impl Pairs<'_> {
    /// Skips `skip` pairs and returns the next one, if any.
    fn advance(&mut self, mut skip: u64) -> Option<(u32, u32)> {
        match self {
            Pairs::Product { a, b, i } => {
                let total = a.len() as u64 * b.len() as u64;
                let at = i.checked_add(skip).filter(|&x| x < total)?;
                *i = at + 1;
                let w = b.len() as u64;
                Some((a[(at / w) as usize], b[(at % w) as usize]))
            }
            Pairs::Permutations { a, i } => {
                let k = a.len() as u64;
                let total = k * k.saturating_sub(1);
                let at = i.checked_add(skip).filter(|&x| x < total)?;
                *i = at + 1;
                let (r, c) = (at / (k - 1), at % (k - 1));
                let c = if c < r { c } else { c + 1 };
                Some((a[r as usize], a[c as usize]))
            }
            Pairs::Combinations {
                a,
                row,
                col,
                loops,
                tail,
            } => {
                let k = a.len();
                // Within the combinations: rows `row`, columns `col..k`.
                while *row < k {
                    let left = (k - *col) as u64;
                    if skip < left {
                        let c = *col + skip as usize;
                        let pair = (a[*row], a[c]);
                        *col = c + 1;
                        if *col == k {
                            *row += 1;
                            *col = *row + 1;
                        }
                        return Some(pair);
                    }
                    skip -= left;
                    *row += 1;
                    *col = *row + 1;
                }
                if !*loops {
                    return None;
                }
                let at = (*tail as u64).checked_add(skip).filter(|&x| x < k as u64)?;
                *tail = at as usize + 1;
                Some((a[at as usize], a[at as usize]))
            }
        }
    }
}

/// `stochastic_block_model` over nodes `0..` grouped in `parts` (each in
/// its set's iteration order). `legacy`: NetworkX 3.4's loop, which first
/// draws for every pair of a diagonal block and then runs the sparse or
/// dense loop on the spent iterator. `None` where NetworkX takes the log
/// of 0.0 or divides by `log(1 - p) == 0`.
pub fn stochastic_block_model(
    parts: &[Vec<u32>],
    p: &[Vec<f64>],
    directed: bool,
    selfloops: bool,
    sparse: bool,
    legacy: bool,
    rng: &mut Mt19937,
) -> Option<Built> {
    let n: usize = parts.iter().map(Vec::len).sum();
    let mut b = Built::new(n, directed);
    let nb = parts.len();
    let blocks: Vec<(usize, usize)> = if directed {
        (0..nb).flat_map(|i| (0..nb).map(move |j| (i, j))).collect()
    } else {
        (0..nb).flat_map(|i| (i..nb).map(move |j| (i, j))).collect()
    };
    for (i, j) in blocks {
        let prob = p[i][j];
        let make = || {
            if i != j {
                Pairs::Product {
                    a: &parts[i],
                    b: &parts[j],
                    i: 0,
                }
            } else if directed && selfloops {
                Pairs::Product {
                    a: &parts[i],
                    b: &parts[i],
                    i: 0,
                }
            } else if directed {
                Pairs::Permutations { a: &parts[i], i: 0 }
            } else {
                Pairs::Combinations {
                    a: &parts[i],
                    row: 0,
                    col: 1,
                    loops: selfloops,
                    tail: 0,
                }
            }
        };
        let mut pairs = make();
        if legacy && i == j {
            while let Some((u, v)) = pairs.advance(0) {
                if rng.random() < prob {
                    b.push_edge(u, v);
                }
            }
        }
        if sparse {
            if prob == 1.0 {
                while let Some((u, v)) = pairs.advance(0) {
                    b.push_edge(u, v);
                }
            } else if prob > 0.0 {
                let denom = (1.0 - prob).ln();
                if denom == 0.0 {
                    return None;
                }
                loop {
                    let r = rng.random();
                    if r == 0.0 {
                        return None;
                    }
                    let skip = (r.ln() / denom).floor();
                    let skip = if skip >= 1.8e19 {
                        u64::MAX
                    } else {
                        skip as u64
                    };
                    match pairs.advance(skip) {
                        Some((u, v)) => b.push_edge(u, v),
                        None => break,
                    }
                }
            }
        } else {
            while let Some((u, v)) = pairs.advance(0) {
                if rng.random() < prob {
                    b.push_edge(u, v);
                }
            }
        }
    }
    Some(b)
}

/// Pairs closer than `radius` (Minkowski `p`) among `n` points of `dim`
/// coordinates in `[0, 1)`, sorted. `None` if some pair's distance is
/// within rounding error of the radius (SciPy's KD-tree and the
/// pure-Python check may round differently there).
pub fn geometric_pairs(coords: &[f64], dim: usize, radius: f64, p: f64) -> Option<Vec<(u32, u32)>> {
    let n = coords.len() / dim;
    let rp = if p == 2.0 {
        radius * radius
    } else {
        radius.powf(p)
    };
    let margin = rp * 1e-9;
    let dist = |u: usize, v: usize| -> f64 {
        let (a, b) = (
            &coords[u * dim..(u + 1) * dim],
            &coords[v * dim..(v + 1) * dim],
        );
        if p == 2.0 {
            a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
        } else if p == 1.0 {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
        } else {
            a.iter().zip(b).map(|(x, y)| (x - y).abs().powf(p)).sum()
        }
    };
    // A grid on the first (up to) two coordinates, with cells wider than
    // the radius, so close pairs sit in neighboring cells.
    let gd = dim.min(2);
    let reach = radius * (1.0 + 1e-6);
    let cap = ((n as f64).sqrt() as usize).max(1);
    let cells = if reach >= 1.0 {
        1
    } else {
        ((1.0 / reach).floor() as usize).clamp(1, cap)
    };
    let cell_of = |x: f64| ((x * cells as f64) as usize).min(cells - 1);
    let key = |u: usize| -> (usize, usize) {
        let c0 = cell_of(coords[u * dim]);
        let c1 = if gd == 2 {
            cell_of(coords[u * dim + 1])
        } else {
            0
        };
        (c0, c1)
    };
    let ny = if gd == 2 { cells } else { 1 };
    let mut grid: Vec<Vec<u32>> = vec![Vec::new(); cells * ny];
    for u in 0..n {
        let (c0, c1) = key(u);
        grid[c0 * ny + c1].push(u as u32);
    }
    let mut pairs = Vec::new();
    for u in 0..n {
        let (c0, c1) = key(u);
        for x in c0.saturating_sub(1)..=(c0 + 1).min(cells - 1) {
            for y in c1.saturating_sub(1)..=(c1 + 1).min(ny - 1) {
                for &v in &grid[x * ny + y] {
                    let v = v as usize;
                    if v <= u {
                        continue;
                    }
                    let d = dist(u, v);
                    if (d - rp).abs() <= margin {
                        return None;
                    }
                    if d <= rp {
                        pairs.push((u as u32, v as u32));
                    }
                }
            }
        }
    }
    pairs.sort_unstable();
    Some(pairs)
}

/// `Built` from pairs added in order to an undirected graph on `n` nodes.
pub fn from_pairs(n: usize, pairs: &[(u32, u32)]) -> Built {
    let mut b = Built::new(n, false);
    for &(u, v) in pairs {
        b.push_edge(u, v);
    }
    b
}

/// `waxman_graph` with the default metric: positions from `uniform`, then
/// a draw per pair. `l`: the given `L` (Waxman-2: the distance is
/// `random() * L`), or `None` to use the largest distance (Waxman-1).
/// Waxman-1 compares against `math.dist`, which rounds differently from
/// `hypot`, so it gives up (`None`) when a draw is within rounding error of
/// its threshold.
#[allow(clippy::too_many_arguments)]
pub fn waxman(
    n: usize,
    beta: f64,
    alpha: f64,
    l: Option<f64>,
    x0: f64,
    dx: f64,
    y0: f64,
    dy: f64,
    rng: &mut Mt19937,
) -> Option<(Vec<f64>, Built)> {
    let mut pos = Vec::with_capacity(2 * n);
    for _ in 0..n {
        pos.push(x0 + dx * rng.random());
        pos.push(y0 + dy * rng.random());
    }
    let dist =
        |u: usize, v: usize| (pos[2 * u] - pos[2 * v]).hypot(pos[2 * u + 1] - pos[2 * v + 1]);
    let mut b = Built::new(n, false);
    match l {
        Some(l) => {
            let scale = alpha * l;
            for u in 0..n {
                for v in u + 1..n {
                    let r = rng.random();
                    let d = rng.random() * l;
                    if r < beta * (-d / scale).exp() {
                        b.push_edge(u as u32, v as u32);
                    }
                }
            }
        }
        None => {
            let mut l = f64::NEG_INFINITY;
            for u in 0..n {
                for v in u + 1..n {
                    l = l.max(dist(u, v));
                }
            }
            let scale = alpha * l;
            if !(scale != 0.0 && scale.is_finite()) {
                return None;
            }
            for u in 0..n {
                for v in u + 1..n {
                    let r = rng.random();
                    let x = -dist(u, v) / scale;
                    let threshold = beta * x.exp();
                    if (r - threshold).abs() <= threshold.abs() * 1e-12 * (1.0 + x.abs()) {
                        return None;
                    }
                    if r < threshold {
                        b.push_edge(u as u32, v as u32);
                    }
                }
            }
        }
    }
    Some((pos, b))
}
