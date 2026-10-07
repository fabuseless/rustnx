//! Batch 23: growth, geometric and community random generators.
//!
//! Like batch 18 (`random_generators`), each generator replays NetworkX's
//! code draw for draw on CPython's Mersenne Twister (`pyrandom::Mt19937`)
//! and records the adjacency rows NetworkX's dicts end up with. `None`
//! means "let NetworkX run": where NetworkX raises, or where rustnx can't
//! be sure to match (a float within rounding error of a threshold).

use std::collections::{HashMap, HashSet};

use super::generators::Sim;
use super::pyrandom::Mt19937;
use super::pyset::PySet;
use super::random_generators::{complete, geometric_pairs, gnm, pair_hash, Built};
use super::spectral::py_sum;

/// `list(G.edges())` of an undirected `Built` with nodes `0..n` in order:
/// each edge from the endpoint whose row comes first, a self-loop once.
fn undirected_edges(b: &Built) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let mut seen = vec![false; b.n()];
    for (u, row) in b.succ.iter().enumerate() {
        for &v in row {
            if !seen[v as usize] {
                out.push((u as u32, v));
            }
        }
        seen[u] = true;
    }
    out
}

/// A graph with the edges of `edges` added in order to `n` new nodes
/// (all known to be new edges).
fn from_edges(n: usize, edges: impl IntoIterator<Item = (u32, u32)>) -> Built {
    let mut b = Built::new(n, false);
    for (u, v) in edges {
        b.push_edge(u, v);
    }
    b
}

/// `duplication_divergence_graph` (`n >= 2`, `0 < p <= 1`). A replica that
/// keeps no edge is removed again, so the node list is always `0..i`.
pub fn duplication_divergence(n: usize, p: f64, rng: &mut Mt19937) -> Built {
    let mut b = Built::new(n, false);
    b.push_edge(0, 1);
    let mut i = 2;
    while i < n {
        let src = rng.below(i);
        let mut kept = false;
        // Adding (i, nbr) never touches `src`'s row.
        for k in 0..b.succ[src].len() {
            let nbr = b.succ[src][k];
            if rng.random() < p {
                b.push_edge(i as u32, nbr);
                kept = true;
            }
        }
        if kept {
            i += 1;
        }
    }
    b
}

/// `partial_duplication_graph` (`1 <= n <= big_n`).
pub fn partial_duplication(big_n: usize, n: usize, p: f64, q: f64, rng: &mut Mt19937) -> Built {
    let mut b = complete(n, false);
    b.succ.resize(big_n.max(n), Vec::new());
    for new in n..big_n {
        let src = rng.below(new);
        let nbrs = b.succ[src].clone();
        for nbr in nbrs {
            if rng.random() < p {
                b.push_edge(new as u32, nbr);
            }
        }
        if rng.random() < q {
            b.push_edge(new as u32, src as u32);
        }
    }
    b
}

/// `scale_free_graph` from the default 3-cycle: a `MultiDiGraph`'s rows
/// (a parallel edge repeats its neighbor). New nodes take the next int, so
/// the node list is `0..count` and `choice(node_list)` is `randbelow`.
pub fn scale_free(
    n: usize,
    alpha: f64,
    beta: f64,
    delta_in: f64,
    delta_out: f64,
    rng: &mut Mt19937,
) -> Built {
    fn choose(cands: &[u32], count: usize, delta: f64, rng: &mut Mt19937) -> u32 {
        if delta > 0.0 {
            let bias_sum = count as f64 * delta;
            let p_delta = bias_sum / (bias_sum + cands.len() as f64);
            if rng.random() < p_delta {
                return rng.below(count) as u32;
            }
        }
        cands[rng.below(cands.len())]
    }
    let mut succ: Vec<Vec<u32>> = vec![vec![1], vec![2], vec![0]];
    let mut pred: Vec<Vec<u32>> = vec![vec![2], vec![0], vec![1]];
    let mut vs: Vec<u32> = vec![0, 1, 2];
    let mut ws: Vec<u32> = vec![0, 1, 2];
    let mut count = 3usize;
    let alpha_beta = alpha + beta;
    while count < n {
        let r = rng.random();
        let (v, w);
        if r < alpha {
            v = count as u32;
            count += 1;
            w = choose(&ws, count, delta_in, rng);
        } else if r < alpha_beta {
            v = choose(&vs, count, delta_out, rng);
            w = choose(&ws, count, delta_in, rng);
        } else {
            v = choose(&vs, count, delta_out, rng);
            w = count as u32;
            count += 1;
        }
        if succ.len() < count {
            succ.push(Vec::new());
            pred.push(Vec::new());
        }
        succ[v as usize].push(w);
        pred[w as usize].push(v);
        vs.push(v);
        ws.push(w);
    }
    Built {
        succ,
        pred: Some(pred),
        order: None,
    }
}

/// One shell of `random_shell_graph`: its node count, how NetworkX's
/// `gnm_random_graph` builds it (0 no edges, 1 complete, 2 `m` drawn
/// edges) and the number of edges to the next shell.
pub struct Shell {
    pub n: usize,
    pub kind: u8,
    pub m: u64,
    pub intra: u64,
}

/// `random_shell_graph`: each shell's `gnm_random_graph`, relabelled
/// (`convert_node_labels_to_integers` re-adds its edges in `edges()`
/// order) and merged with `union` (which re-adds both graphs' edges), then
/// the edges between consecutive shells.
pub fn random_shell(shells: &[Shell], rng: &mut Mt19937) -> Built {
    let mut g = Built::new(0, false);
    let mut offsets = Vec::with_capacity(shells.len());
    let mut total = 0usize;
    for s in shells {
        let h = match s.kind {
            0 => Built::new(s.n, false),
            1 => complete(s.n, false),
            _ => gnm(s.n, s.m, false, rng),
        };
        let relabeled = from_edges(s.n, undirected_edges(&h));
        let shift = total as u32;
        let merged = undirected_edges(&g).into_iter().chain(
            undirected_edges(&relabeled)
                .into_iter()
                .map(|(u, v)| (u + shift, v + shift)),
        );
        g = from_edges(total + s.n, merged);
        offsets.push(total);
        total += s.n;
    }
    for gi in 0..shells.len().saturating_sub(1) {
        let (o1, n1) = (offsets[gi], shells[gi].n);
        let (o2, n2) = (offsets[gi + 1], shells[gi + 1].n);
        // Only this loop links shells gi and gi + 1.
        let mut added: HashSet<(usize, usize)> = HashSet::new();
        let mut count = 0;
        while count < shells[gi].intra {
            let u = o1 + rng.below(n1);
            let v = o2 + rng.below(n2);
            if !added.insert((u, v)) {
                continue;
            }
            g.push_edge(u as u32, v as u32);
            count += 1;
        }
    }
    g
}

/// `navigable_small_world_graph` on the `n^dim` lattice points in
/// `product(range(n), repeat=dim)` order. `weights[d]` is NetworkX's
/// `d ** -r` for each lattice distance `d >= 1`. A `DiGraph`, with nodes
/// in the order `add_edge` first meets them.
pub fn navigable_small_world(
    n: usize,
    dim: usize,
    p: f64,
    q: u64,
    weights: &[f64],
    rng: &mut Mt19937,
) -> Built {
    let total: usize = (0..dim).fold(1usize, |acc, _| acc * n);
    let mut coords = vec![0u32; total * dim];
    for i in 0..total {
        let mut x = i;
        for c in (0..dim).rev() {
            coords[i * dim + c] = (x % n) as u32;
            x /= n;
        }
    }
    let mut b = Built::new(total, true);
    let mut present = vec![false; total];
    let mut order = Vec::with_capacity(total);
    let mut mark = vec![usize::MAX; total];
    let mut cdf: Vec<f64> = Vec::with_capacity(total);
    for a in 0..total {
        let mut touch = |x: usize, order: &mut Vec<u32>| {
            if !present[x] {
                present[x] = true;
                order.push(x as u32);
            }
        };
        let mut add = |a: usize, x: usize, b: &mut Built, order: &mut Vec<u32>| {
            touch(a, order);
            touch(x, order);
            if mark[x] != a {
                mark[x] = a;
                b.push_edge(a as u32, x as u32);
            }
        };
        cdf.clear();
        cdf.push(0.0);
        let mut c = 0.0f64;
        let pa = &coords[a * dim..(a + 1) * dim];
        for x in 0..total {
            if x == a {
                continue;
            }
            let px = &coords[x * dim..(x + 1) * dim];
            let d: usize = pa
                .iter()
                .zip(px)
                .map(|(&s, &t)| s.abs_diff(t) as usize)
                .sum();
            if (d as f64) <= p {
                add(a, x, &mut b, &mut order);
            }
            c += weights[d];
            cdf.push(c);
        }
        for _ in 0..q {
            // `uniform(0, cdf[-1])`, then `bisect_left`.
            let x = c * rng.random();
            let idx = cdf.partition_point(|&y| y < x);
            add(a, idx, &mut b, &mut order);
        }
    }
    b.order = Some(order);
    b
}

/// `n * dim` coordinates from `random()`, as `[seed.random() for i in
/// range(dim)]` per node.
fn draw_coords(n: usize, dim: usize, rng: &mut Mt19937) -> Vec<f64> {
    (0..n * dim).map(|_| rng.random()).collect()
}

/// `soft_random_geometric_graph` with drawn positions and the default
/// `p_dist` (`exp(-dist)`): the pairs within `radius` (sorted, as from
/// SciPy's KD-tree), each kept with one draw. `None` if a pair sits
/// within rounding error of the radius.
pub fn soft_geometric(
    n: usize,
    dim: usize,
    radius: f64,
    p: f64,
    compensated: bool,
    rng: &mut Mt19937,
) -> Option<(Vec<f64>, Built)> {
    let coords = draw_coords(n, dim, rng);
    let pairs = geometric_pairs(&coords, dim, radius, p)?;
    let inv = 1.0 / p;
    let mut b = Built::new(n, false);
    for (u, v) in pairs {
        let (pu, pv) = (
            &coords[u as usize * dim..(u as usize + 1) * dim],
            &coords[v as usize * dim..(v as usize + 1) * dim],
        );
        // `sum(abs(a - b) ** p ...) ** (1 / p)`, with libm's `pow` as in
        // Python's float `**`.
        let s = py_sum(
            pu.iter().zip(pv).map(|(a, b)| (a - b).abs().powf(p)),
            compensated,
        );
        let dist = s.powf(inv);
        if rng.random() < (-dist).exp() {
            b.push_edge(u, v);
        }
    }
    Some((coords, b))
}

/// `n` draws of `expovariate(1)`.
fn draw_weights(n: usize, rng: &mut Mt19937) -> Vec<f64> {
    (0..n).map(|_| rng.expovariate(1.0)).collect()
}

/// `thresholded_random_geometric_graph` with drawn weights and positions:
/// the pairs within `radius` whose weights sum to at least `theta`.
pub fn thresholded_geometric(
    n: usize,
    dim: usize,
    radius: f64,
    theta: f64,
    p: f64,
    rng: &mut Mt19937,
) -> Option<(Vec<f64>, Vec<f64>, Built)> {
    let weights = draw_weights(n, rng);
    let coords = draw_coords(n, dim, rng);
    let pairs = geometric_pairs(&coords, dim, radius, p)?;
    let b = from_edges(
        n,
        pairs
            .into_iter()
            .filter(|&(u, v)| weights[u as usize] + weights[v as usize] >= theta),
    );
    Some((weights, coords, b))
}

/// The Euclidean distance of two points (`math.dist` up to an ulp or so).
fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    match a.len() {
        1 => (a[0] - b[0]).abs(),
        2 => (a[0] - b[0]).hypot(a[1] - b[1]),
        _ => {
            let scale = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max);
            if scale == 0.0 {
                return 0.0;
            }
            let s: f64 = a
                .iter()
                .zip(b)
                .map(|(x, y)| {
                    let t = (x - y) / scale;
                    t * t
                })
                .sum();
            scale * s.sqrt()
        }
    }
}

/// `geographical_threshold_graph` with drawn weights and positions, the
/// Euclidean metric and the default `p_dist` (`r ** -2`): every pair with
/// `(w_u + w_v) * dist ** -2 >= theta`. `math.dist` rounds its own way, so
/// `None` when a pair is within rounding error of `theta` (or two points
/// coincide, where NetworkX divides by zero).
pub fn geographical_threshold(
    n: usize,
    dim: usize,
    theta: f64,
    rng: &mut Mt19937,
) -> Option<(Vec<f64>, Vec<f64>, Built)> {
    let weights = draw_weights(n, rng);
    let coords = draw_coords(n, dim, rng);
    let mut b = Built::new(n, false);
    let margin = theta.abs() * 1e-12;
    for u in 0..n {
        let pu = &coords[u * dim..(u + 1) * dim];
        for v in u + 1..n {
            let pv = &coords[v * dim..(v + 1) * dim];
            let d = euclidean(pu, pv);
            if d == 0.0 {
                return None;
            }
            let lhs = (weights[u] + weights[v]) * d.powf(-2.0);
            if (lhs - theta).abs() <= margin.max(lhs.abs() * 1e-12) || !lhs.is_finite() {
                return None;
            }
            if lhs >= theta {
                b.push_edge(u as u32, v as u32);
            }
        }
    }
    Some((weights, coords, b))
}

/// The constants of `geometric_soft_configuration_graph` that NetworkX
/// computes before and around its draws (from Python, with its arithmetic).
pub struct SoftConfig {
    pub kappa_0: f64,
    pub base: f64,
    pub power: f64,
    pub two_pi: f64,
    pub big_r: f64,
    pub beta: f64,
    /// `max(1, beta)`.
    pub beta_max: f64,
    pub mu: f64,
    /// `2 / zeta * math.log(n / math.pi)`.
    pub r_hat_head: f64,
    pub r_c: f64,
}

/// `geometric_soft_configuration_graph` without `kappas`: the kappas, the
/// angles, a draw per pair (`v` before `u` in node order), then the radii.
/// Returns `(thetas, kappas, radii, graph)`; `None` where NetworkX raises
/// (`math.pow` overflow, a zero division, `log` of a non-positive number).
#[allow(clippy::type_complexity)]
pub fn soft_configuration(
    n: usize,
    c: &SoftConfig,
    rng: &mut Mt19937,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Built)> {
    use std::f64::consts::PI;
    let kappas: Vec<f64> = (0..n)
        .map(|_| c.kappa_0 * (1.0 - rng.random() * c.base).powf(c.power))
        .collect();
    let thetas: Vec<f64> = (0..n).map(|_| c.two_pi * rng.random()).collect();
    let mut b = Built::new(n, false);
    for u in 0..n {
        for v in 0..u {
            let angle = PI - (PI - (thetas[u] - thetas[v]).abs()).abs();
            let dij = (c.big_r * angle).powf(c.beta);
            let mu_kappas = (c.mu * kappas[u] * kappas[v]).powf(c.beta_max);
            if !dij.is_finite() || !mu_kappas.is_finite() || mu_kappas == 0.0 {
                return None;
            }
            let p_ij = 1.0 / (1.0 + dij / mu_kappas);
            if rng.random() < p_ij {
                b.push_edge(u as u32, v as u32);
            }
        }
    }
    let kappa_min = kappas.iter().copied().fold(f64::INFINITY, f64::min);
    let low = c.mu * kappa_min;
    if low.is_nan() || low <= 0.0 || kappas.iter().any(|&k| k.is_nan() || k <= 0.0) {
        return None;
    }
    let r_hat = c.r_hat_head - c.r_c * low.ln();
    let radii = kappas.iter().map(|&k| r_hat - c.r_c * k.ln()).collect();
    Some((thetas, kappas, radii, b))
}

/// `projected_graph(B, range(n))` of a bipartite graph whose top nodes are
/// `0..n`: for each `u`, the set `{v for nbr in B[u] for v in B[nbr] if v
/// != u}` in CPython's set order. An edge to an earlier node already
/// exists (the relation is symmetric).
fn project(b: &Built, n: usize) -> Built {
    let hashes: Vec<i64> = (0..b.n() as i64).collect();
    let mut out = Built::new(n, false);
    for u in 0..n {
        let mut s = PySet::default();
        for &w in &b.succ[u] {
            for &x in &b.succ[w as usize] {
                if x as usize != u {
                    s.add(x, &hashes);
                }
            }
        }
        for x in s.iter() {
            if x as usize > u {
                out.push_edge(u as u32, x);
            }
        }
    }
    out
}

/// `uniform_random_intersection_graph` for `0 < p < 1`: bipartite
/// `random_graph(n, m, p)`, projected onto `range(n)`.
pub fn uniform_intersection(n: usize, m: usize, p: f64, rng: &mut Mt19937) -> Option<Built> {
    let b = super::random_generators::bipartite_random(n, m, p, false, rng)?;
    Some(project(&b, n))
}

/// `k_random_intersection_graph` (`0 <= k <= m`): `sample(range(n, n + m),
/// k)` per top node, projected.
pub fn k_intersection(n: usize, m: usize, k: usize, rng: &mut Mt19937) -> Built {
    let pop: Vec<u32> = (n as u32..(n + m) as u32).collect();
    let mut b = Built::new(n + m, false);
    for v in 0..n {
        for t in rng.sample(&pop, k) {
            b.push_edge(v as u32, t);
        }
    }
    project(&b, n)
}

/// `general_random_intersection_graph`: a draw per (top, bottom) pair
/// against the bottom node's probability, projected.
pub fn general_intersection(n: usize, p: &[f64], rng: &mut Mt19937) -> Built {
    let m = p.len();
    let mut b = Built::new(n + m, false);
    for u in 0..n {
        for (j, &q) in p.iter().enumerate() {
            if rng.random() < q {
                b.push_edge(u as u32, (n + j) as u32);
            }
        }
    }
    project(&b, n)
}

/// `random_k_lift`: node `(v, i)` is `v * k + i`; a shuffle of one
/// permutation (kept between edges) per edge of `G` in `edges()` order.
pub fn k_lift(
    n: usize,
    edges: &[(u32, u32)],
    k: usize,
    directed: bool,
    multigraph: bool,
    rng: &mut Mt19937,
) -> Sim {
    let mut sim = Sim::new(directed, multigraph, n * k, edges.len() * k);
    sim.add_nodes(0..(n * k) as u32);
    let mut perm: Vec<u32> = (0..k as u32).collect();
    let mut lifted = Vec::with_capacity(edges.len() * k);
    let k32 = k as u32;
    for &(u, v) in edges {
        rng.shuffle(&mut perm);
        for (i, &j) in perm.iter().enumerate() {
            lifted.push((u * k32 + i as u32, v * k32 + j));
        }
    }
    for (a, b) in lifted {
        sim.add_edge(a, b);
    }
    sim
}

/// Bipartite `preferential_attachment_graph`: nodes `0..aseq.len()`, then
/// each new bottom node. A stub picks an existing bottom node with
/// `choice` over the degree-weighted list NetworkX rebuilds each time,
/// found here with a Fenwick tree over the bottom nodes' degrees.
pub fn bipartite_preferential(aseq: &[i64], p: f64, multigraph: bool, rng: &mut Mt19937) -> Sim {
    let na = aseq.len();
    let stubs: usize = aseq.iter().map(|&a| a.max(0) as usize).sum();
    let mut sim = Sim::new(false, multigraph, na + stubs, stubs);
    sim.add_nodes(0..na as u32);
    // Fenwick tree (1-based) over bottom node degrees.
    let cap = stubs.max(1);
    let mut tree = vec![0u64; cap + 1];
    let mut total = 0u64;
    let mut bottoms = 0usize;
    for (v, &a) in aseq.iter().enumerate() {
        for _ in 0..a.max(0) {
            let r = rng.random();
            let target = if r < p || sim.len() == na {
                bottoms += 1;
                let t = (na + bottoms - 1) as u32;
                sim.add_node(t);
                t
            } else {
                // `choice(bbstubs)`: the bottom node holding stub `idx`.
                let mut idx = rng.randbelow(total);
                let mut pos = 0usize;
                let mut step = cap.next_power_of_two();
                while step > 0 {
                    let next = pos + step;
                    if next <= cap && tree[next] <= idx {
                        pos = next;
                        idx -= tree[next];
                    }
                    step >>= 1;
                }
                (na + pos) as u32
            };
            let before = sim.keys.len();
            sim.add_edge(v as u32, target);
            if multigraph || sim.keys.len() > before {
                let mut i = target as usize - na + 1;
                while i <= cap {
                    tree[i] += 1;
                    i += 1 << i.trailing_zeros();
                }
                total += 1;
            }
        }
    }
    sim
}

/// `relaxed_caveman_graph`: `caveman_graph(l, k)`, then a draw per edge of
/// `G.edges()` while it rewires the graph. `EdgeView` iterates a copy of
/// each row (`list(nbrs)`) taken when it reaches the row, and skips rows
/// it has finished.
pub fn relaxed_caveman(l: usize, k: usize, p: f64, rng: &mut Mt19937) -> Built {
    let n = l * k;
    let mut b = Built::new(n, false);
    let mut member: HashSet<(u32, u32)> = HashSet::new();
    if k > 1 {
        for start in (0..n).step_by(k) {
            for u in start..start + k {
                for v in u + 1..start + k {
                    b.push_edge(u as u32, v as u32);
                    member.insert((u as u32, v as u32));
                    member.insert((v as u32, u as u32));
                }
            }
        }
    }
    let mut seen = vec![false; n];
    for u in 0..n {
        let uu = u as u32;
        for v in b.succ[u].clone() {
            if seen[v as usize] {
                continue;
            }
            if rng.random() < p {
                let x = rng.below(n) as u32;
                if member.contains(&(uu, x)) {
                    continue;
                }
                b.remove_edge(uu, v);
                member.remove(&(uu, v));
                member.remove(&(v, uu));
                b.push_edge(uu, x);
                member.insert((uu, x));
                member.insert((x, uu));
            }
        }
        seen[u] = true;
    }
    b
}

/// `maybe_regular_expander_graph` (`maybe_regular_expander` before 3.6):
/// `d / 2` random Hamiltonian cycles from NumPy's legacy
/// `RandomState.permutation`, with no edge used twice; the edges in the
/// iteration order of NetworkX's set of tuples. `None` where NetworkX
/// runs out of tries (it raises).
pub fn maybe_regular_expander(
    n: usize,
    d: usize,
    max_tries: i64,
    rng: &mut Mt19937,
) -> Option<Vec<(u32, u32)>> {
    let mut ids: HashMap<(u32, u32), u32> = HashMap::new();
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    let mut hashes: Vec<i64> = Vec::new();
    let mut id_of = |e: (u32, u32), hashes: &mut Vec<i64>, pairs: &mut Vec<(u32, u32)>| -> u32 {
        *ids.entry(e).or_insert_with(|| {
            pairs.push(e);
            hashes.push(pair_hash(e.0 as u64, e.1 as u64));
            (pairs.len() - 1) as u32
        })
    };
    let mut edges = PySet::default();
    let mut members: HashSet<(u32, u32)> = HashSet::new();
    for i in 0..d / 2 {
        let mut iterations = max_tries;
        while edges.len() != (i + 1) * n {
            iterations -= 1;
            let mut cycle = rng.np_permutation(n - 1);
            cycle.push((n - 1) as u32);
            let mut new_edges = PySet::default();
            let mut fresh: Vec<(u32, u32)> = Vec::new();
            for j in 0..n {
                let (u, v) = (cycle[j], cycle[(j + 1) % n]);
                if !members.contains(&(u, v)) && !members.contains(&(v, u)) {
                    let id = id_of((u, v), &mut hashes, &mut pairs);
                    new_edges.add(id, &hashes);
                    fresh.push((u, v));
                }
            }
            if new_edges.len() == n {
                edges.merge(&new_edges, &hashes);
                members.extend(fresh);
            }
            if iterations == 0 {
                return None;
            }
        }
    }
    Some(edges.iter().map(|id| pairs[id as usize]).collect())
}

/// CPython's `a.union(b)` of two sets: a copy of `a` (a new set merged
/// with it), then merged with `b`.
fn set_union(a: &PySet, b: &PySet, hashes: &[i64]) -> PySet {
    let mut r = PySet::default();
    r.merge(a, hashes);
    r.merge(b, hashes);
    r
}

/// CPython's `a.intersection(b)` of two sets: iterate the smaller (`b` on
/// a tie), adding the members of the other.
fn set_intersection(a: &PySet, b: &PySet, hashes: &[i64]) -> PySet {
    let (iterated, other) = if b.len() > a.len() { (a, b) } else { (b, a) };
    let mut r = PySet::default();
    for k in iterated.iter() {
        if other.contains(k, hashes) {
            r.add(k, hashes);
        }
    }
    r
}

/// CPython's `a.difference(b)` of two sets: a copy of `a` with `b`'s keys
/// discarded when `a` is over four times larger, else the keys of `a` not
/// in `b`, added in `a`'s order.
fn set_difference(a: &PySet, b: &PySet, hashes: &[i64]) -> PySet {
    if (a.len() >> 2) > b.len() {
        let mut r = PySet::default();
        r.merge(a, hashes);
        for k in b.iter() {
            r.discard(k, hashes);
        }
        r.after_difference_update(hashes);
        return r;
    }
    let mut r = PySet::default();
    for k in a.iter() {
        if !b.contains(k, hashes) {
            r.add(k, hashes);
        }
    }
    r
}

/// Replays set algebra for the runtime check that the replicas above match
/// the running interpreter: `(op, dst, a, b)` with op 0 `s[dst].add(a)`,
/// 1 `s[dst].discard(a)`, 2 `s[dst] = s[a].union(s[b])`, 3 intersection,
/// 4 difference. Returns every set's keys in iteration order.
pub fn replay_set_algebra(
    hashes: &[i64],
    nsets: usize,
    ops: &[(u8, u32, u32, u32)],
) -> Vec<Vec<u32>> {
    let mut sets: Vec<PySet> = (0..nsets).map(|_| PySet::default()).collect();
    for &(op, dst, a, b) in ops {
        let (dst, ai, bi) = (dst as usize, a as usize, b as usize);
        match op {
            0 => sets[dst].add(a, hashes),
            1 => {
                sets[dst].discard(a, hashes);
            }
            2 => sets[dst] = set_union(&sets[ai], &sets[bi], hashes),
            3 => sets[dst] = set_intersection(&sets[ai], &sets[bi], hashes),
            _ => sets[dst] = set_difference(&sets[ai], &sets[bi], hashes),
        }
    }
    sets.iter().map(|s| s.iter().collect()).collect()
}

/// Python's `round(x)` for a float (ties to even), as an int.
fn py_round(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// The constants `AS_graph_generator.__init__` computes from `n` alone
/// (Python's arithmetic, done by the caller).
pub struct AsParams {
    pub n: i64,
    pub n_m: i64,
    pub n_cp: i64,
    pub d_m: f64,
    pub d_cp: f64,
    pub d_c: f64,
    pub p_m_m: f64,
    pub p_cp_m: f64,
    pub p_cp_cp: f64,
}

/// `random_internet_as_graph`'s result: the graph, each node's kind (0 T,
/// 1 M, 2 CP, 3 C) and `peers` count, and each edge (in insertion order)
/// with whether it is a transit link (`customer` is then `str(u)`).
pub struct AsGraph {
    pub b: Built,
    pub kinds: Vec<u8>,
    pub peers: Vec<u32>,
    pub edges: Vec<(u32, u32, bool)>,
}

/// NetworkX's `uniform_int_from_avg(a, m, seed)`.
fn uniform_int_from_avg(a: i64, m: f64, rng: &mut Mt19937) -> i64 {
    let b = 2.0 * m - a as f64;
    let fb = b.floor();
    let p = (b - fb) / 2.0;
    let x1 = py_round(rng.random() * (fb - a as f64) + a as f64);
    let x2 = i64::from(rng.random() < p);
    x1 + x2
}

/// NetworkX's `choose_pref_attach(degs, seed)` for a non-empty dict given as
/// keys and int values in order.
fn choose_pref_attach(keys: &[u32], values: &[u64], rng: &mut Mt19937) -> u32 {
    let s: u64 = values.iter().sum();
    if s == 0 {
        return rng.choice(keys);
    }
    let v = rng.random() * s as f64;
    let mut i = 0;
    let mut acc = values[0];
    while v > acc as f64 {
        i += 1;
        acc += values[i];
    }
    keys[i]
}

struct AsState<'a> {
    rng: &'a mut Mt19937,
    hashes: Vec<i64>,
    regions: Vec<PySet>,
    /// `nodes['T']`, `['M']`, `['CP']`, `['C']`.
    nodes: [PySet; 4],
    customers: Vec<PySet>,
    providers: Vec<PySet>,
    out: AsGraph,
}

impl AsState<'_> {
    fn add_edge(&mut self, i: u32, j: u32, transit: bool) {
        self.out.b.push_edge(i, j);
        self.out.edges.push((i, j, transit));
    }

    fn degree(&self, v: u32) -> u64 {
        self.out.b.succ[v as usize].len() as u64
    }

    fn choose_node_pref_attach(&mut self, options: &PySet) -> u32 {
        let keys: Vec<u32> = options.iter().collect();
        let values: Vec<u64> = keys.iter().map(|&v| self.degree(v)).collect();
        choose_pref_attach(&keys, &values, self.rng)
    }

    fn add_customer(&mut self, i: u32, j: u32) {
        let h = &self.hashes;
        self.customers[j as usize].add(i, h);
        self.providers[i as usize].add(j, h);
        let ps: Vec<u32> = self.providers[j as usize].iter().collect();
        for z in ps {
            self.customers[z as usize].add(i, h);
            self.providers[i as usize].add(z, h);
        }
    }

    fn add_node(&mut self, i: u32, kind: u8, reg2prob: f64, avg_deg: f64, t_edge_prob: f64) {
        let regs = if self.rng.random() < reg2prob { 2 } else { 1 };
        let mut node_options = PySet::default();
        self.out.b.succ.push(Vec::new());
        self.out.kinds.push(kind);
        self.out.peers.push(0);
        self.customers.push(PySet::default());
        self.providers.push(PySet::default());
        self.nodes[kind as usize].add(i, &self.hashes);
        for r in self.rng.sample(&[0usize, 1, 2, 3, 4], regs) {
            node_options = set_union(&node_options, &self.regions[r], &self.hashes);
            self.regions[r].add(i, &self.hashes);
        }
        let edge_num = uniform_int_from_avg(1, avg_deg, self.rng);
        let mut t_options = set_intersection(&node_options, &self.nodes[0], &self.hashes);
        let mut m_options = set_intersection(&node_options, &self.nodes[1], &self.hashes);
        m_options.discard(i, &self.hashes);
        let mut d = 0;
        while d < edge_num && (!t_options.is_empty() || !m_options.is_empty()) {
            let j = if m_options.is_empty()
                || (!t_options.is_empty() && self.rng.random() < t_edge_prob)
            {
                let j = self.choose_node_pref_attach(&t_options);
                t_options.discard(j, &self.hashes);
                j
            } else {
                let j = self.choose_node_pref_attach(&m_options);
                m_options.discard(j, &self.hashes);
                j
            };
            self.add_edge(i, j, true);
            self.add_customer(i, j);
            d += 1;
        }
    }

    /// Discards `v` and `v`'s neighbors from `options`.
    fn drop_self_and_neighbors(&self, options: &mut PySet, v: u32) {
        options.discard(v, &self.hashes);
        for &j in &self.out.b.succ[v as usize] {
            options.discard(j, &self.hashes);
        }
    }

    fn add_m_peering_link(&mut self, m: u32) {
        let h = &self.hashes;
        let options = set_difference(&self.nodes[1], &self.customers[m as usize], h);
        let mut options = set_difference(&options, &self.providers[m as usize], h);
        self.drop_self_and_neighbors(&mut options, m);
        if options.is_empty() {
            return;
        }
        let keys: Vec<u32> = options.iter().collect();
        let values: Vec<u64> = keys
            .iter()
            .map(|&v| self.out.peers[v as usize] as u64)
            .collect();
        let j = choose_pref_attach(&keys, &values, self.rng);
        self.add_edge(m, j, false);
        self.out.peers[m as usize] += 1;
        self.out.peers[j as usize] += 1;
    }

    fn add_cp_peering_link(&mut self, cp: u32, to_kind: usize) {
        let h = &self.hashes;
        let mut options = PySet::default();
        for r in &self.regions {
            if r.contains(cp, h) {
                options = set_union(&options, r, h);
            }
        }
        let mut options = set_intersection(&self.nodes[to_kind], &options, h);
        options.discard(cp, h);
        let mut options = set_difference(&options, &self.providers[cp as usize], h);
        self.drop_self_and_neighbors(&mut options, cp);
        if options.is_empty() {
            return;
        }
        let keys: Vec<u32> = options.iter().collect();
        let j = self.rng.sample(&keys, 1)[0];
        self.add_edge(cp, j, false);
        self.out.peers[cp as usize] += 1;
        self.out.peers[j as usize] += 1;
    }

    fn add_peering_links(&mut self, from_kind: usize, to_kind: usize, m: f64) {
        let members: Vec<u32> = self.nodes[from_kind].iter().collect();
        for i in members {
            let num = uniform_int_from_avg(0, m, self.rng);
            for _ in 0..num.max(0) {
                if from_kind == 1 {
                    self.add_m_peering_link(i);
                } else {
                    self.add_cp_peering_link(i, to_kind);
                }
            }
        }
    }
}

/// `random_internet_as_graph(n)`: `AS_graph_generator(n, seed).generate()`
/// with every set replayed as CPython's (the order of node options, of
/// preferential choices and of the peering loops follows them).
pub fn internet_as(c: &AsParams, rng: &mut Mt19937) -> AsGraph {
    let n_t = c.n.min(py_round(rng.random() * 2.0 + 4.0)).max(0);
    let n_c = (c.n - n_t - c.n_m - c.n_cp).max(0);
    let total = (n_t + c.n_m.max(0) + c.n_cp.max(0) + n_c) as usize;
    let mut st = AsState {
        rng,
        hashes: (0..total as i64).collect(),
        regions: (0..5).map(|_| PySet::default()).collect(),
        nodes: Default::default(),
        customers: Vec::with_capacity(total),
        providers: Vec::with_capacity(total),
        out: AsGraph {
            b: Built::new(0, false),
            kinds: Vec::with_capacity(total),
            peers: Vec::with_capacity(total),
            edges: Vec::new(),
        },
    };
    // t_graph: a clique of T nodes, each in every region.
    for i in 0..n_t as u32 {
        st.out.b.succ.push(Vec::new());
        st.out.kinds.push(0);
        st.out.peers.push(0);
        for r in 0..5 {
            st.regions[r].add(i, &st.hashes);
        }
        for j in 0..i {
            st.add_edge(i, j, false);
        }
        st.customers.push(PySet::default());
        st.providers.push(PySet::default());
    }
    let mut t = PySet::default();
    for i in 0..n_t as u32 {
        t.add(i, &st.hashes);
    }
    st.nodes[0] = t;
    let mut i = n_t as u32;
    for _ in 0..c.n_m.max(0) {
        st.add_node(i, 1, 0.2, c.d_m, 0.375);
        i += 1;
    }
    for _ in 0..c.n_cp.max(0) {
        st.add_node(i, 2, 0.05, c.d_cp, 0.375);
        i += 1;
    }
    for _ in 0..n_c {
        st.add_node(i, 3, 0.0, c.d_c, 0.125);
        i += 1;
    }
    st.add_peering_links(1, 1, c.p_m_m);
    st.add_peering_links(2, 1, c.p_cp_m);
    st.add_peering_links(2, 2, c.p_cp_cp);
    st.out
}

/// NumPy's `kahan_sum` (used by `RandomState.choice` to check `p`).
fn np_kahan_sum(values: &[f64]) -> f64 {
    let mut sum = values[0];
    let mut c = 0.0;
    for &x in &values[1..] {
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Legacy `RandomState.choice(len(p), p=p)`: NumPy's checks (`None` where
/// it would raise or see NaN), then `cdf = p.cumsum(); cdf /= cdf[-1]` and
/// the first index whose `cdf` exceeds one `random_sample()`.
fn np_choice_p(rng: &mut Mt19937, p: &[f64], cdf: &mut Vec<f64>) -> Option<usize> {
    if p.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return None;
    }
    let atol = f64::EPSILON.sqrt();
    if (np_kahan_sum(p) - 1.0).abs() > atol {
        return None;
    }
    cdf.clear();
    let mut acc = 0.0;
    for &x in p {
        acc += x;
        cdf.push(acc);
    }
    let last = *cdf.last()?;
    for c in cdf.iter_mut() {
        *c /= last;
    }
    let u = rng.random();
    // searchsorted(side="right"): the number of entries <= u.
    let idx = cdf.partition_point(|&c| c <= u);
    (idx < p.len()).then_some(idx)
}

/// `random_k_out_graph` from NetworkX 3.5 on (`_random_k_out_graph_numpy`,
/// on a legacy `RandomState`): the `(u, v)` edges in the order NetworkX
/// adds them. `alpha` and the running weights are exact as `f64`.
pub fn k_out_numpy(
    rng: &mut Mt19937,
    n: usize,
    k: usize,
    alpha: f64,
    self_loops: bool,
) -> Option<Vec<(u32, u32)>> {
    let mut weights = vec![alpha; n];
    let mut total = n as f64 * alpha;
    let mut out = vec![0usize; n];
    let mut remaining: Vec<u32> = (0..n as u32).collect();
    let mut edges = Vec::with_capacity(n * k);
    let mut p = vec![0.0; n];
    let mut cdf = Vec::with_capacity(n);
    for _ in 0..n * k {
        if remaining.is_empty() {
            return None; // NumPy raises on an empty choice
        }
        let u = remaining[rng.np_interval((remaining.len() - 1) as u32) as usize] as usize;
        let v = if self_loops {
            for (pi, w) in p.iter_mut().zip(&weights) {
                *pi = w / total;
            }
            np_choice_p(rng, &p, &mut cdf)?
        } else {
            let u_weight = weights[u];
            let denom = total - u_weight;
            for (i, (pi, w)) in p.iter_mut().zip(&weights).enumerate() {
                *pi = if i == u { 0.0 / denom } else { w / denom };
            }
            np_choice_p(rng, &p, &mut cdf)?
        };
        edges.push((u as u32, v as u32));
        weights[v] += 1.0;
        total += 1.0;
        out[u] += 1;
        if out[u] == k {
            remaining.retain(|&x| x as usize != u);
        }
    }
    Some(edges)
}

/// `random_k_out_graph` in NetworkX 3.4 (pure Python): `u` is a uniform
/// choice among nodes with out-degree below `k`, `v` a roulette choice over
/// `weights - adjustment` (a `Counter` difference, which drops entries that
/// aren't positive).
pub fn k_out_py34(
    rng: &mut Mt19937,
    n: usize,
    k: usize,
    alpha: f64,
    self_loops: bool,
) -> Option<Vec<(u32, u32)>> {
    let mut weights = vec![alpha; n];
    let mut out = vec![0usize; n];
    let mut edges = Vec::with_capacity(n * k);
    let mut candidates: Vec<usize> = Vec::with_capacity(n);
    for _ in 0..n * k {
        candidates.clear();
        candidates.extend((0..n).filter(|&v| out[v] < k));
        if candidates.is_empty() {
            return None;
        }
        let u = candidates[rng.below(candidates.len())];
        // Counter subtraction keeps positive values only, in order.
        let kept = |i: usize| {
            let w = if !self_loops && i == u {
                0.0
            } else {
                weights[i]
            };
            (w > 0.0).then_some(w)
        };
        let total = (0..n).filter_map(kept).fold(0.0, |s, w| s + w);
        let mut rnd = rng.random() * total;
        let mut chosen = None;
        for i in 0..n {
            if let Some(w) = kept(i) {
                rnd -= w;
                if rnd < 0.0 {
                    chosen = Some(i);
                    break;
                }
            }
        }
        let v = chosen?; // NetworkX would add an edge to None
        edges.push((u as u32, v as u32));
        out[u] += 1;
        weights[v] += 1.0;
    }
    Some(edges)
}
