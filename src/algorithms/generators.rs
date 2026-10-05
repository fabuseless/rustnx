//! Batch 17: deterministic graph generators. Each generator replays the
//! `add_node` / `add_edge` / `remove_*` calls NetworkX's generator makes on
//! a [`Sim`], a model of a NetworkX graph's dicts: node order, the order of
//! every adjacency row, and the number of parallel edges between each pair.
//! `lib.rs` then writes the dicts in one pass.

use std::collections::HashMap;
use std::hash::BuildHasherDefault;

use super::pyset::PySet;
use super::structure_more::PairHasher;

type PairMap = HashMap<u64, u32, BuildHasherDefault<PairHasher>>;

#[inline]
fn pair(u: u32, v: u32) -> u64 {
    ((u as u64) << 32) | v as u64
}

/// A NetworkX graph's structure, built with NetworkX's insertion semantics:
/// a new node goes to the end of the node dict, a new neighbor to the end
/// of a row, and adding an edge that exists changes no order (a multigraph
/// gets one more key). Every adjacency entry carries its pair's slot, so
/// the two rows of an undirected edge (or `succ` and `pred` of an arc) can
/// share one data dict, as in NetworkX.
pub struct Sim {
    pub directed: bool,
    pub multigraph: bool,
    present: Vec<bool>,
    pub order: Vec<u32>,
    pub succ: Vec<Vec<(u32, u32)>>,
    pub pred: Vec<Vec<(u32, u32)>>,
    /// Parallel edges per slot (always 1 for simple graphs).
    pub keys: Vec<u32>,
    slots: PairMap,
}

impl Sim {
    pub fn new(directed: bool, multigraph: bool, nodes: usize, edges: usize) -> Self {
        Sim {
            directed,
            multigraph,
            present: vec![false; nodes],
            order: Vec::with_capacity(nodes),
            succ: vec![Vec::new(); nodes],
            pred: if directed {
                vec![Vec::new(); nodes]
            } else {
                Vec::new()
            },
            keys: Vec::with_capacity(edges),
            slots: PairMap::with_capacity_and_hasher(edges, Default::default()),
        }
    }

    /// One more than the largest node id seen.
    pub fn capacity(&self) -> usize {
        self.present.len()
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    fn ensure(&mut self, u: u32) {
        let need = u as usize + 1;
        if self.present.len() < need {
            self.present.resize(need, false);
            self.succ.resize(need, Vec::new());
            if self.directed {
                self.pred.resize(need, Vec::new());
            }
        }
    }

    pub fn add_node(&mut self, u: u32) {
        self.ensure(u);
        if !self.present[u as usize] {
            self.present[u as usize] = true;
            self.order.push(u);
        }
    }

    pub fn add_nodes(&mut self, nodes: std::ops::Range<u32>) {
        for u in nodes {
            self.add_node(u);
        }
    }

    #[inline]
    fn key(&self, u: u32, v: u32) -> u64 {
        if self.directed || u <= v {
            pair(u, v)
        } else {
            pair(v, u)
        }
    }

    pub fn add_edge(&mut self, u: u32, v: u32) {
        self.add_node(u);
        self.add_node(v);
        let key = self.key(u, v);
        if let Some(&slot) = self.slots.get(&key) {
            if self.multigraph {
                self.keys[slot as usize] += 1;
            }
            return;
        }
        let slot = self.keys.len() as u32;
        self.keys.push(1);
        self.slots.insert(key, slot);
        self.succ[u as usize].push((v, slot));
        if self.directed {
            self.pred[v as usize].push((u, slot));
        } else if u != v {
            self.succ[v as usize].push((u, slot));
        }
    }

    /// `G.remove_edge(u, v)` for an edge of a simple graph.
    pub fn remove_edge(&mut self, u: u32, v: u32) {
        let key = self.key(u, v);
        if self.slots.remove(&key).is_none() {
            return;
        }
        self.succ[u as usize].retain(|&(x, _)| x != v);
        if self.directed {
            self.pred[v as usize].retain(|&(x, _)| x != u);
        } else if u != v {
            self.succ[v as usize].retain(|&(x, _)| x != u);
        }
    }

    /// `G.remove_node(u)`: the other rows keep their order.
    pub fn remove_node(&mut self, u: u32) {
        if (u as usize) >= self.present.len() || !self.present[u as usize] {
            return;
        }
        let out = std::mem::take(&mut self.succ[u as usize]);
        for &(v, _) in &out {
            let key = self.key(u, v);
            self.slots.remove(&key);
            if v == u {
                continue;
            }
            if self.directed {
                self.pred[v as usize].retain(|&(x, _)| x != u);
            } else {
                self.succ[v as usize].retain(|&(x, _)| x != u);
            }
        }
        if self.directed {
            for (v, _) in std::mem::take(&mut self.pred[u as usize]) {
                self.slots.remove(&pair(v, u));
                if v != u {
                    self.succ[v as usize].retain(|&(x, _)| x != u);
                }
            }
        }
        self.present[u as usize] = false;
        self.order.retain(|&x| x != u);
    }

    /// A graph given as rows of neighbors in insertion order (an edge in
    /// both rows of an undirected graph, a self-loop once; a parallel edge
    /// of a multigraph repeats its neighbor) and its node order.
    pub fn from_rows(
        succ: &[Vec<u32>],
        pred: Option<&[Vec<u32>]>,
        order: &[u32],
        multigraph: bool,
    ) -> Self {
        let n = succ.len();
        let directed = pred.is_some();
        let edges = succ.iter().map(Vec::len).sum();
        let mut g = Sim::new(directed, multigraph, n, edges);
        for &u in order {
            g.add_node(u);
        }
        // Marks the neighbors already in the row being built (parallel
        // edges share one entry).
        let mut seen = vec![u32::MAX; n];
        for (u, row) in succ.iter().enumerate() {
            let u = u as u32;
            for &v in row {
                let key = g.key(u, v);
                let first = seen[v as usize] != u;
                seen[v as usize] = u;
                if !directed && v < u {
                    // The other row of an edge from `v`'s row.
                    if first {
                        let slot = g.slots[&key];
                        g.succ[u as usize].push((v, slot));
                    }
                    continue;
                }
                match g.slots.get(&key) {
                    Some(&slot) => g.keys[slot as usize] += 1,
                    None => {
                        let slot = g.keys.len() as u32;
                        g.keys.push(1);
                        g.slots.insert(key, slot);
                        g.succ[u as usize].push((v, slot));
                    }
                }
            }
        }
        if let Some(pred) = pred {
            seen.fill(u32::MAX);
            for (v, row) in pred.iter().enumerate() {
                let v = v as u32;
                for &u in row {
                    if seen[u as usize] != v {
                        seen[u as usize] = v;
                        let slot = g.slots[&pair(u, v)];
                        g.pred[v as usize].push((u, slot));
                    }
                }
            }
        }
        g
    }

    /// `list(G.edges())`, one pair per parallel edge, in NetworkX's order
    /// (an undirected edge is reported from the endpoint whose row comes
    /// first).
    pub fn edges(&self) -> Vec<(u32, u32)> {
        let mut out = Vec::with_capacity(self.keys.len());
        let mut seen = vec![false; self.present.len()];
        for &n in &self.order {
            for &(v, slot) in &self.succ[n as usize] {
                if self.directed || !seen[v as usize] {
                    let reps = if self.multigraph {
                        self.keys[slot as usize]
                    } else {
                        1
                    };
                    for _ in 0..reps {
                        out.push((n, v));
                    }
                }
            }
            seen[n as usize] = true;
        }
        out
    }
}

/// A generator's result: the graph, plus integer tuples as node labels
/// (`width` values per node id) or, if `width` is 0, the ids themselves.
pub struct Built {
    pub sim: Sim,
    pub width: usize,
    pub labels: Vec<i64>,
    pub attr: Attr,
}

/// Node attributes NetworkX's generator sets, by node id.
pub enum Attr {
    None,
    /// `name` set to an int (`bipartite`, `subset`).
    Ints(&'static str, Vec<i64>),
    /// `pos` set to a tuple of floats.
    Pos(Vec<(f64, f64)>),
}

impl Built {
    fn plain(sim: Sim) -> Self {
        Built {
            sim,
            width: 0,
            labels: Vec::new(),
            attr: Attr::None,
        }
    }
}

fn path_edges(g: &mut Sim, nodes: std::ops::Range<u32>) {
    let mut prev = None;
    for v in nodes {
        if let Some(u) = prev {
            g.add_edge(u, v);
        }
        prev = Some(v);
    }
}

/// `pairwise(nodes, cyclic=True)`: a self-loop for one node, and the edge
/// back for two.
fn cycle_edges(g: &mut Sim, nodes: std::ops::Range<u32>) {
    let (first, end) = (nodes.start, nodes.end);
    path_edges(g, nodes);
    if end > first {
        g.add_edge(end - 1, first);
    }
}

fn clique_edges(g: &mut Sim, nodes: std::ops::Range<u32>) {
    for u in nodes.clone() {
        for v in u + 1..nodes.end {
            g.add_edge(u, v);
        }
    }
}

fn sim(directed: bool, multigraph: bool, nodes: usize, edges: usize) -> Sim {
    Sim::new(directed, multigraph, nodes, edges)
}

fn edge_estimate(nodes: usize, per_node: usize) -> usize {
    nodes.saturating_mul(per_node).min(1 << 26)
}

/// `(i - j) % n` with Python's sign rule (n > 0).
#[inline]
fn py_mod(x: i128, n: i128) -> u32 {
    x.rem_euclid(n) as u32
}

/// Builds generator `kind` from its integer parameters `p` and integer
/// lists `lists`, for a graph that is `directed` and/or a `multigraph`.
/// `None` for an unknown kind, or a case rustnx leaves to NetworkX.
pub fn build(
    kind: &str,
    p: &[i64],
    lists: &[Vec<i64>],
    directed: bool,
    multigraph: bool,
) -> Option<Built> {
    let int = |i: usize| -> Option<u32> { p.get(i).and_then(|&x| u32::try_from(x).ok()) };
    Some(match kind {
        "empty" => {
            let n = int(0)?;
            let mut g = sim(directed, multigraph, n as usize, 0);
            g.add_nodes(0..n);
            Built::plain(g)
        }
        "path" | "cycle" => {
            let n = int(0)?;
            let mut g = sim(directed, multigraph, n as usize, n as usize);
            g.add_nodes(0..n);
            if kind == "path" {
                path_edges(&mut g, 0..n);
            } else {
                cycle_edges(&mut g, 0..n);
            }
            Built::plain(g)
        }
        "complete" => {
            let n = int(0)?;
            let m = (n as usize).saturating_mul(n as usize);
            let mut g = sim(directed, multigraph, n as usize, m.min(1 << 26));
            g.add_nodes(0..n);
            if n > 1 {
                if directed {
                    for u in 0..n {
                        for v in 0..n {
                            if u != v {
                                g.add_edge(u, v);
                            }
                        }
                    }
                } else {
                    clique_edges(&mut g, 0..n);
                }
            }
            Built::plain(g)
        }
        "star" => {
            // `n` nodes in all: the hub, then the spokes.
            let n = int(0)?;
            let mut g = sim(directed, multigraph, n as usize, n as usize);
            g.add_nodes(0..n);
            for v in 1..n {
                g.add_edge(0, v);
            }
            Built::plain(g)
        }
        "wheel" => {
            let n = int(0)?;
            let mut g = sim(directed, multigraph, n as usize, 2 * n as usize);
            g.add_nodes(0..n);
            for v in 1..n {
                g.add_edge(0, v);
            }
            if n > 2 {
                cycle_edges(&mut g, 1..n);
            }
            Built::plain(g)
        }
        "ladder" | "circular_ladder" => {
            let n = int(0)?;
            let total = n.checked_mul(2)?;
            let mut g = sim(directed, multigraph, total as usize, 3 * n as usize);
            g.add_nodes(0..total);
            path_edges(&mut g, 0..n);
            path_edges(&mut g, n..total);
            for v in 0..n {
                g.add_edge(v, v + n);
            }
            if kind == "circular_ladder" {
                g.add_edge(0, n.checked_sub(1)?);
                g.add_edge(n, total.checked_sub(1)?);
            }
            Built::plain(g)
        }
        "lollipop" => {
            // A clique on ids 0..m, then a path on m..m+n joined to it.
            let (m, n) = (int(0)?, int(1)?);
            let total = m.checked_add(n)?;
            let mut g = sim(
                directed,
                multigraph,
                total as usize,
                edge_estimate(m as usize, m as usize),
            );
            g.add_nodes(0..m);
            clique_edges(&mut g, 0..m);
            g.add_nodes(m..total);
            path_edges(&mut g, m..total);
            if m > 0 && n > 0 {
                g.add_edge(m - 1, m);
            }
            Built::plain(g)
        }
        "barbell" => {
            let (m1, m2) = (int(0)?, int(1)?);
            let end = m1.checked_mul(2)?.checked_add(m2)?;
            let mut g = sim(
                directed,
                multigraph,
                end as usize,
                edge_estimate(m1 as usize, m1 as usize),
            );
            g.add_nodes(0..m1);
            clique_edges(&mut g, 0..m1);
            g.add_nodes(m1..(m1 + m2).saturating_sub(1).max(m1));
            if m2 > 1 {
                path_edges(&mut g, m1..m1 + m2);
            }
            clique_edges(&mut g, m1 + m2..end);
            g.add_edge(m1 - 1, m1);
            if m2 > 0 {
                g.add_edge(m1 + m2 - 1, m1 + m2);
            }
            Built::plain(g)
        }
        "tadpole" => {
            // A cycle on ids 0..m, then `add_path` from its last node.
            let (m, n) = (int(0)?, int(1)?);
            let total = m.checked_add(n)?;
            let mut g = sim(directed, multigraph, total as usize, total as usize);
            g.add_nodes(0..m);
            cycle_edges(&mut g, 0..m);
            if m > 0 {
                g.add_node(m - 1);
                let mut prev = m - 1;
                for v in m..total {
                    g.add_edge(prev, v);
                    prev = v;
                }
            }
            Built::plain(g)
        }
        "full_rary" => {
            let (r, n) = (p.first().copied()?, int(1)?);
            let mut g = sim(directed, multigraph, n as usize, n as usize);
            g.add_nodes(0..n);
            if r > 0 {
                for child in 1..n {
                    g.add_edge(((child as i64 - 1) / r) as u32, child);
                }
            }
            Built::plain(g)
        }
        "binomial" => {
            let n = p.first().copied()?;
            if n > 22 {
                return None;
            }
            let size = 1usize << n.max(0);
            let mut g = sim(directed, multigraph, size, size);
            g.add_node(0);
            let mut big_n: u32 = 1;
            for _ in 0..n.max(0) {
                for (u, v) in g.edges() {
                    g.add_edge(u + big_n, v + big_n);
                }
                g.add_edge(0, big_n);
                big_n *= 2;
            }
            Built::plain(g)
        }
        "bipartite" => {
            let (n1, n2) = (int(0)?, int(1)?);
            let total = n1.checked_add(n2)?;
            let mut g = sim(
                directed,
                multigraph,
                total as usize,
                edge_estimate(n1 as usize, n2 as usize),
            );
            g.add_nodes(0..total);
            for u in 0..n1 {
                for v in n1..total {
                    g.add_edge(u, v);
                }
            }
            let side = (0..total).map(|v| (v >= n1) as i64).collect();
            Built {
                attr: Attr::Ints("bipartite", side),
                ..Built::plain(g)
            }
        }
        "grid_2d" => grid_2d(int(0)?, int(1)?, p[2] != 0, p[3] != 0, directed, multigraph)?,
        "circulant" => {
            let n = int(0)?;
            let two_pass = p.get(1).copied()? != 0;
            let offsets = lists.first()?;
            let per = offsets.len().saturating_mul(2);
            let mut g = sim(
                directed,
                multigraph,
                n as usize,
                edge_estimate(n as usize, per),
            );
            g.add_nodes(0..n);
            let nn = n as i128;
            if two_pass {
                for sign in [-1i128, 1] {
                    for i in 0..n {
                        for &j in offsets {
                            g.add_edge(i, py_mod(i as i128 + sign * j as i128, nn));
                        }
                    }
                }
            } else {
                for i in 0..n {
                    for &j in offsets {
                        g.add_edge(i, py_mod(i as i128 - j as i128, nn));
                        g.add_edge(i, py_mod(i as i128 + j as i128, nn));
                    }
                }
            }
            Built::plain(g)
        }
        "caveman" | "connected_caveman" => {
            let (l, k) = (int(0)?, int(1)?);
            let total = l.checked_mul(k)?;
            let mut g = sim(
                directed,
                multigraph,
                total as usize,
                edge_estimate(total as usize, k as usize / 2 + 1),
            );
            g.add_nodes(0..total);
            if k > 1 {
                for start in (0..total).step_by(k as usize) {
                    clique_edges(&mut g, start..start + k);
                }
            }
            if kind == "connected_caveman" && k > 1 {
                for start in (0..total).step_by(k as usize) {
                    g.remove_edge(start, start + 1);
                    g.add_edge(start, py_mod(start as i128 - 1, total as i128));
                }
            }
            Built::plain(g)
        }
        "ring_of_cliques" => {
            let (c, s) = (int(0)?, int(1)?);
            let total = c.checked_mul(s)?;
            let mut g = sim(
                directed,
                multigraph,
                total as usize,
                edge_estimate(total as usize, s as usize / 2 + 1),
            );
            for i in 0..c {
                clique_edges(&mut g, i * s..i * s + s);
                g.add_edge(i * s + 1, ((i + 1) * s) % total);
            }
            Built::plain(g)
        }
        "windmill" => {
            let (n, k) = (int(0)?, int(1)?);
            let total = k.checked_add(n.checked_sub(1)?.checked_mul(k.checked_sub(1)?)?)?;
            let mut g = sim(
                directed,
                multigraph,
                total as usize,
                edge_estimate(total as usize, k as usize / 2 + 2),
            );
            g.add_nodes(0..k);
            clique_edges(&mut g, 0..k);
            let mut start = k;
            for _ in 1..n {
                g.add_nodes(start..start + k - 1);
                clique_edges(&mut g, start..start + k - 1);
                start += k - 1;
            }
            for v in k..total {
                g.add_edge(0, v);
            }
            Built::plain(g)
        }
        "multipartite" => {
            // Parts of the given sizes on consecutive ids; `subset` is the
            // part's index.
            let sizes = lists.first()?;
            let mut bounds = vec![0u32];
            for &s in sizes {
                let s = u32::try_from(s).ok()?;
                bounds.push(bounds.last()?.checked_add(s)?);
            }
            let total = *bounds.last()?;
            let mut g = sim(
                directed,
                multigraph,
                total as usize,
                edge_estimate(total as usize, 8),
            );
            let mut part = Vec::with_capacity(total as usize);
            for (i, w) in bounds.windows(2).enumerate() {
                g.add_nodes(w[0]..w[1]);
                part.extend(std::iter::repeat_n(i as i64, (w[1] - w[0]) as usize));
            }
            for a in 0..sizes.len() {
                for b in a + 1..sizes.len() {
                    for u in bounds[a]..bounds[a + 1] {
                        for v in bounds[b]..bounds[b + 1] {
                            g.add_edge(u, v);
                        }
                    }
                }
            }
            Built {
                attr: Attr::Ints("subset", part),
                ..Built::plain(g)
            }
        }
        "sudoku" => {
            let n = int(0)?;
            let n2 = n.checked_mul(n)?;
            let n3 = n2.checked_mul(n)?;
            let n4 = n3.checked_mul(n)?;
            let mut g = sim(
                directed,
                multigraph,
                n4 as usize,
                edge_estimate(n4 as usize, 3 * n2 as usize),
            );
            g.add_nodes(0..n4);
            if n >= 2 {
                for row_no in 0..n2 {
                    let row_start = row_no * n2;
                    for j in 1..n2 {
                        for i in 0..j {
                            g.add_edge(row_start + i, row_start + j);
                        }
                    }
                }
                for col_no in 0..n2 {
                    for j in (col_no..n4).step_by(n2 as usize) {
                        for i in (col_no..j).step_by(n2 as usize) {
                            g.add_edge(i, j);
                        }
                    }
                }
                for band_no in 0..n {
                    for stack_no in 0..n {
                        let box_start = n3 * band_no + n * stack_no;
                        for j in 1..n2 {
                            for i in 0..j {
                                let u = box_start + i % n + n2 * (i / n);
                                let v = box_start + j % n + n2 * (j / n);
                                g.add_edge(u, v);
                            }
                        }
                    }
                }
            }
            Built::plain(g)
        }
        "lcf" => {
            let n = int(0)?;
            let repeats = p.get(1).copied()?;
            let shifts = lists.first()?;
            let extra = (repeats as i128) * shifts.len() as i128;
            if extra > (1 << 31) {
                return None;
            }
            let mut g = sim(
                directed,
                multigraph,
                n as usize,
                n as usize + extra.max(0) as usize,
            );
            g.add_nodes(0..n);
            cycle_edges(&mut g, 0..n);
            let nn = n as i128;
            for i in 0..extra.max(0) {
                let shift = shifts[(i % shifts.len() as i128) as usize] as i128;
                g.add_edge(py_mod(i, nn), py_mod(i + shift, nn));
            }
            Built::plain(g)
        }
        "petersen" => {
            let (n, k) = (int(0)?, int(1)?);
            let mut g = sim(directed, multigraph, 2 * n as usize, 3 * n as usize);
            g.add_nodes(0..n);
            cycle_edges(&mut g, 0..n);
            for i in 0..n {
                g.add_edge(i, n + i);
                g.add_edge(n + i, n + (i + k) % n);
            }
            Built::plain(g)
        }
        "dgm" => {
            let n = int(0)?;
            if n > 13 {
                return None;
            }
            let edges = 3usize.pow(n + 1);
            let mut g = sim(directed, multigraph, edges, edges);
            g.add_edge(0, 1);
            let mut new_node = 2u32;
            for _ in 0..n {
                let mut new_edges = Vec::new();
                for (u, v) in g.edges() {
                    new_edges.push((u, new_node));
                    new_edges.push((v, new_node));
                    new_node += 1;
                }
                for (u, v) in new_edges {
                    g.add_edge(u, v);
                }
            }
            Built::plain(g)
        }
        "mycielski" => {
            let n = int(0)?;
            if !(1..=15).contains(&n) {
                return None;
            }
            let mut g = sim(directed, multigraph, 0, 0);
            if n == 1 {
                g.add_node(0);
                return Some(Built::plain(g));
            }
            // `convert_node_labels_to_integers(path_graph(2))`.
            g.add_nodes(0..2);
            g.add_edge(0, 1);
            for _ in 0..n - 2 {
                let m = g.len() as u32;
                g.add_nodes(m..2 * m);
                let old = g.edges();
                for &(u, v) in &old {
                    g.add_edge(u, v + m);
                }
                for &(u, v) in &old {
                    g.add_edge(u + m, v);
                }
                g.add_node(2 * m);
                for u in 0..m {
                    g.add_edge(u + m, 2 * m);
                }
            }
            Built::plain(g)
        }
        "paley" => {
            let p0 = p.first().copied()?;
            if p0 > u32::MAX as i64 / 2 {
                return None;
            }
            let pp = p0.max(0) as u32;
            // `{x**2 % p for x in range(1, p) if x**2 % p != 0}`, iterated in
            // CPython's set order (ints hash to themselves).
            let hashes: Vec<i64> = (0..pp as i64).collect();
            let mut squares = PySet::default();
            for x in 1..pp as u64 {
                let r = (x * x % pp as u64) as u32;
                if r != 0 {
                    squares.add(r, &hashes);
                }
            }
            let squares: Vec<u32> = squares.iter().collect();
            let mut g = sim(
                directed,
                multigraph,
                pp as usize,
                edge_estimate(pp as usize, squares.len()),
            );
            for x in 0..pp {
                for &x2 in &squares {
                    g.add_edge(x, ((x as u64 + x2 as u64) % pp as u64) as u32);
                }
            }
            Built::plain(g)
        }
        "kneser" => kneser(int(0)?, int(1)?, directed, multigraph)?,
        "grid" => grid(lists, &p[..lists.len()])?,
        "hexagonal" => hexagonal(int(0)?, int(1)?, p[2] != 0, directed, multigraph)?,
        "triangular" => triangular(int(0)?, int(1)?, p[2] != 0, directed, multigraph)?,
        _ => return super::transforms::build(kind, p, lists, directed, multigraph),
    })
}

fn grid_2d(
    rows: u32,
    cols: u32,
    pr: bool,
    pc: bool,
    directed: bool,
    multigraph: bool,
) -> Option<Built> {
    let total = rows.checked_mul(cols)?;
    let id = |i: u32, j: u32| i * cols + j;
    let mut g = sim(
        directed,
        multigraph,
        total as usize,
        edge_estimate(total as usize, 4),
    );
    g.add_nodes(0..total);
    for a in 1..rows {
        for j in 0..cols {
            g.add_edge(id(a, j), id(a - 1, j));
        }
    }
    for i in 0..rows {
        for b in 1..cols {
            g.add_edge(id(i, b), id(i, b - 1));
        }
    }
    if pr && rows > 2 {
        for j in 0..cols {
            g.add_edge(id(0, j), id(rows - 1, j));
        }
    }
    if pc && cols > 2 {
        for i in 0..rows {
            g.add_edge(id(i, 0), id(i, cols - 1));
        }
    }
    if directed {
        // `G.add_edges_from((v, u) for u, v in G.edges())` consumes the edge
        // view lazily, so rows not reached yet already hold reversed edges
        // (adding them back is a no-op, except for a new key in a
        // multigraph).
        for idx in 0..g.order.len() {
            let n = g.order[idx];
            let mut k = 0;
            while k < g.succ[n as usize].len() {
                let (v, slot) = g.succ[n as usize][k];
                let reps = if multigraph { g.keys[slot as usize] } else { 1 };
                for _ in 0..reps {
                    g.add_edge(v, n);
                }
                k += 1;
            }
        }
    }
    Some(Built::plain(g))
}

/// `binom(n, k)`, saturating.
fn binom(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut r: u128 = 1;
    for i in 0..k {
        r = r * (n - i) as u128 / (i + 1) as u128;
        if r > u64::MAX as u128 {
            return u64::MAX;
        }
    }
    r as u64
}

/// Position of a sorted k-subset of `0..n` among `combinations(range(n), k)`.
fn combination_rank(c: &[u32], n: u32, table: &[Vec<u64>]) -> u64 {
    let k = c.len();
    let mut rank = 0u64;
    let mut prev: i64 = -1;
    for (i, &x) in c.iter().enumerate() {
        for y in (prev + 1) as u32..x {
            // subsets starting with y at position i
            rank += table[(n - y - 1) as usize][k - i - 1];
        }
        prev = x as i64;
    }
    rank
}

fn kneser(n: u32, k: u32, directed: bool, multigraph: bool) -> Option<Built> {
    let count = binom(n as u64, k as u64);
    if count > (1 << 24) {
        return None;
    }
    let count = count as usize;
    let table: Vec<Vec<u64>> = (0..=n as u64)
        .map(|a| (0..=k as u64).map(|b| binom(a, b)).collect())
        .collect();
    // All k-subsets in `combinations` order.
    let mut subsets: Vec<u32> = Vec::with_capacity(count * k as usize);
    let mut c: Vec<u32> = (0..k).collect();
    if k <= n {
        loop {
            subsets.extend_from_slice(&c);
            let mut i = k as usize;
            let mut advanced = false;
            while i > 0 {
                i -= 1;
                if c[i] < n - k + i as u32 {
                    c[i] += 1;
                    for j in i + 1..k as usize {
                        c[j] = c[j - 1] + 1;
                    }
                    advanced = true;
                    break;
                }
            }
            if !advanced {
                break;
            }
        }
    }
    let hashes: Vec<i64> = (0..n as i64).collect();
    let mut universe = PySet::default();
    for v in 0..n {
        universe.add(v, &hashes);
    }
    let universe_order: Vec<u32> = universe.iter().collect();
    let mut g = sim(directed, multigraph, count, 0);
    let k_us = k as usize;
    if 2 * k > n {
        g.add_nodes(0..count as u32);
    } else {
        let mut member = vec![false; n as usize];
        let mut rest: Vec<u32> = Vec::with_capacity(n as usize);
        let mut t = vec![0u32; k_us];
        for si in 0..count {
            let s = &subsets[si * k_us..(si + 1) * k_us];
            for &x in s {
                member[x as usize] = true;
            }
            // `universe - set(s)`, in CPython's order (set_difference copies
            // the universe when it is more than four times larger).
            rest.clear();
            if (n as usize >> 2) > k_us {
                let mut copy = PySet::default();
                copy.merge(&universe, &hashes);
                for &x in s {
                    copy.discard(x, &hashes);
                }
                copy.after_difference_update(&hashes);
                rest.extend(copy.iter());
            } else {
                let mut diff = PySet::default();
                for &x in &universe_order {
                    if !member[x as usize] {
                        diff.add(x, &hashes);
                    }
                }
                rest.extend(diff.iter());
            }
            for &x in s {
                member[x as usize] = false;
            }
            if rest.windows(2).any(|w| w[0] > w[1]) {
                return None; // tuples out of order would be new nodes
            }
            // combinations(rest, k)
            let r = rest.len();
            let mut idx: Vec<usize> = (0..k_us).collect();
            let u = si as u32;
            if r >= k_us {
                loop {
                    for (slot, &ix) in t.iter_mut().zip(&idx) {
                        *slot = rest[ix];
                    }
                    let v = combination_rank(&t, n, &table) as u32;
                    g.add_edge(u, v);
                    let mut i = k_us;
                    let mut advanced = false;
                    while i > 0 {
                        i -= 1;
                        if idx[i] < r - k_us + i {
                            idx[i] += 1;
                            for j in i + 1..k_us {
                                idx[j] = idx[j - 1] + 1;
                            }
                            advanced = true;
                            break;
                        }
                    }
                    if !advanced {
                        break;
                    }
                }
            }
        }
    }
    Some(Built {
        sim: g,
        width: k_us,
        labels: subsets.iter().map(|&x| x as i64).collect(),
        attr: Attr::None,
    })
}

/// `grid_graph(dim, periodic)`: dimension `d` has the int labels
/// `atoms[d]` and is a cycle if `cyclic[d]`. Replays the Cartesian products
/// (`cartesian_product(new_dim, G)`) and the final `relabel_nodes(G,
/// flatten)` copy, which re-adds the edges in `G.edges()` order.
fn grid(atoms: &[Vec<i64>], cyclic: &[i64]) -> Option<Built> {
    let dims = atoms.len();
    let factor = |d: usize| -> Option<Sim> {
        let n = u32::try_from(atoms[d].len()).ok()?;
        let mut f = sim(false, false, n as usize, n as usize);
        f.add_nodes(0..n);
        if cyclic[d] != 0 {
            cycle_edges(&mut f, 0..n);
        } else {
            path_edges(&mut f, 0..n);
        }
        Some(f)
    };
    let mut total: usize = 1;
    for a in atoms {
        total = total.checked_mul(a.len())?;
    }
    if total > (1 << 26) {
        return None;
    }
    // Node id -> per-dimension indices (outermost dimension first).
    let mut g = factor(0)?;
    let mut coords: Vec<Vec<u32>> = g.order.iter().map(|&v| vec![v]).collect();
    for d in 1..dims {
        let f = factor(d)?;
        let (nf, ng) = (f.len(), g.len());
        // Product node (a, b) for a in f's order, b in g's order.
        let id = |a_pos: usize, b_pos: usize| (a_pos * ng + b_pos) as u32;
        let mut pos_f = vec![0usize; f.capacity()];
        for (i, &v) in f.order.iter().enumerate() {
            pos_f[v as usize] = i;
        }
        let mut pos_g = vec![0usize; g.capacity()];
        for (i, &v) in g.order.iter().enumerate() {
            pos_g[v as usize] = i;
        }
        let mut h = sim(false, false, nf * ng, edge_estimate(nf * ng, dims));
        h.add_nodes(0..(nf * ng) as u32);
        for (u, v) in f.edges() {
            for x in 0..ng {
                h.add_edge(id(pos_f[u as usize], x), id(pos_f[v as usize], x));
            }
        }
        let g_edges = g.edges();
        for x in 0..nf {
            for &(u, v) in &g_edges {
                h.add_edge(id(x, pos_g[u as usize]), id(x, pos_g[v as usize]));
            }
        }
        let mut new_coords = Vec::with_capacity(nf * ng);
        for &a in &f.order {
            for &b in &g.order {
                let mut c = Vec::with_capacity(d + 1);
                c.push(a);
                c.extend_from_slice(&coords[b as usize]);
                new_coords.push(c);
            }
        }
        coords = new_coords;
        g = h;
    }
    // relabel_nodes(G, flatten) copies: nodes in order, then G.edges().
    let mut out = sim(false, false, g.capacity(), g.keys.len());
    for &v in &g.order {
        out.add_node(v);
    }
    for (u, v) in g.edges() {
        out.add_edge(u, v);
    }
    let mut labels = Vec::with_capacity(coords.len() * dims);
    for (v, c) in coords.iter().enumerate() {
        debug_assert!(v < out.capacity());
        for (d, &i) in c.iter().enumerate() {
            // `c[0]` is the last dimension added.
            labels.push(atoms[dims - 1 - d][i as usize]);
        }
    }
    Some(Built {
        sim: out,
        width: if dims == 1 { 0 } else { dims },
        labels: if dims == 1 { atoms[0].clone() } else { labels },
        attr: Attr::None,
    })
}

fn lattice_labels(cols: u32, rows: u32) -> Vec<i64> {
    let mut labels = Vec::with_capacity(2 * (cols * rows) as usize);
    for i in 0..cols {
        for j in 0..rows {
            labels.push(i as i64);
            labels.push(j as i64);
        }
    }
    labels
}

/// Non-periodic `hexagonal_lattice_graph`; node `(i, j)` has id
/// `i * (2m + 2) + j`.
fn hexagonal(
    m: u32,
    n: u32,
    with_positions: bool,
    directed: bool,
    multigraph: bool,
) -> Option<Built> {
    let big_m = m.checked_mul(2)?;
    let rows = big_m.checked_add(2)?;
    let cols = n.checked_add(1)?;
    let total = (rows as usize).checked_mul(cols as usize)?;
    if total > (1 << 26) {
        return None;
    }
    let id = |i: u32, j: u32| i * rows + j;
    let mut g = sim(directed, multigraph, total, edge_estimate(total, 2));
    for i in 0..cols {
        for j in 0..=big_m {
            g.add_edge(id(i, j), id(i, j + 1));
        }
    }
    for i in 0..n {
        for j in 0..rows {
            if i % 2 == j % 2 {
                g.add_edge(id(i, j), id(i + 1, j));
            }
        }
    }
    g.remove_node(id(0, big_m + 1));
    g.remove_node(id(n, (big_m + 1) * (n % 2)));
    let attr = if with_positions {
        let h = 3f64.sqrt() / 2.0;
        let mut pos = vec![(0.0, 0.0); total];
        for i in 0..cols {
            for j in 0..rows {
                // 0.5 + i + i // 2 + j % 2 * (i % 2 - 0.5), and h * j
                let x = 0.5 + i as f64 + (i / 2) as f64 + (j % 2) as f64 * ((i % 2) as f64 - 0.5);
                pos[id(i, j) as usize] = (x, h * j as f64);
            }
        }
        Attr::Pos(pos)
    } else {
        Attr::None
    };
    Some(Built {
        sim: g,
        width: 2,
        labels: lattice_labels(cols, rows),
        attr,
    })
}

/// Non-periodic `triangular_lattice_graph`; node `(i, j)` has id
/// `i * (m + 1) + j`.
fn triangular(
    m: u32,
    n: u32,
    with_positions: bool,
    directed: bool,
    multigraph: bool,
) -> Option<Built> {
    let big_n = n.div_ceil(2); // (n + 1) // 2
    let rows = m.checked_add(1)?;
    let cols = big_n.checked_add(1)?;
    let total = (rows as usize).checked_mul(cols as usize)?;
    if total > (1 << 26) {
        return None;
    }
    let id = |i: u32, j: u32| i * rows + j;
    let mut g = sim(directed, multigraph, total, edge_estimate(total, 3));
    for j in 0..rows {
        for i in 0..big_n {
            g.add_edge(id(i, j), id(i + 1, j));
        }
    }
    for j in 0..m {
        for i in 0..cols {
            g.add_edge(id(i, j), id(i, j + 1));
        }
    }
    for j in (1..m).step_by(2) {
        for i in 0..big_n {
            g.add_edge(id(i, j), id(i + 1, j + 1));
        }
    }
    for j in (0..m).step_by(2) {
        for i in 0..big_n {
            g.add_edge(id(i + 1, j), id(i, j + 1));
        }
    }
    if n % 2 == 1 {
        for j in (1..rows).step_by(2) {
            g.remove_node(id(big_n, j));
        }
    }
    let attr = if with_positions {
        let h = 3f64.sqrt() / 2.0;
        let mut pos = vec![(0.0, 0.0); total];
        for i in 0..cols {
            for j in 0..rows {
                // 0.5 * (j % 2) + i, and h * j
                pos[id(i, j) as usize] = (0.5 * (j % 2) as f64 + i as f64, h * j as f64);
            }
        }
        Attr::Pos(pos)
    } else {
        Attr::None
    };
    Some(Built {
        sim: g,
        width: 2,
        labels: lattice_labels(cols, rows),
        attr,
    })
}
