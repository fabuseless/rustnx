//! NetworkX's VF2++ (`vf2pp_isomorphism`, `vf2pp_all_isomorphisms` and, in
//! 3.7, the subgraph-isomorphism and monomorphism variants), replayed so
//! the mappings come out exactly as NetworkX finds them, in the same order.
//!
//! What depends on Python's set order is replayed with `PySet` (from the
//! nodes' `hash()`): the matching order (ties among `set(G.nodes())`, and
//! for directed graphs the neighbor order of the undirected view, a set
//! union), and every candidate set, built with the same set operations as
//! NetworkX (copies, `&` and `-` of dict views, `intersection_update` with
//! sets or dicts, set comprehensions), including `T2_tilde`, whose table
//! NetworkX mutates throughout the search. The feasibility tests only
//! count, so they use plain arrays. Two variants: NetworkX 3.4 to 3.6
//! (`old`) and 3.7 (`new`). Simple graphs only.

use super::pyset::PySet;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Iso,
    Ind,
    Mono,
}

/// One graph: neighbor lists in dict order (`pred` empty when undirected),
/// each node's `hash()`, label id, degree (`(deg, 0)` undirected, `(in,
/// out)` directed), degree-value id (3.4 to 3.6) and self-loop count.
pub struct Side {
    pub succ: Vec<Vec<u32>>,
    pub pred: Vec<Vec<u32>>,
    pub hashes: Vec<i64>,
    pub label: Vec<u32>,
    pub deg: Vec<(i64, i64)>,
    pub deg_id: Vec<u32>,
    pub loops: Vec<u32>,
}

impl Side {
    fn n(&self) -> usize {
        self.succ.len()
    }
}

/// Sorted neighbor lists, for membership tests.
fn sorted(rows: &[Vec<u32>]) -> Vec<Vec<u32>> {
    rows.iter()
        .map(|r| {
            let mut s = r.clone();
            s.sort_unstable();
            s
        })
        .collect()
}

fn has(rows: &[Vec<u32>], a: u32, b: u32) -> bool {
    rows[a as usize].binary_search(&b).is_ok()
}

/// `set(it) & other`-style filtering into a fresh set: one `add` per
/// element of `items` (in order) that `keep` accepts.
fn collect(items: impl IntoIterator<Item = u32>, hashes: &[i64]) -> PySet {
    PySet::from_iter(items, hashes)
}

pub struct Vf2pp {
    /// Pattern (3.7's SG, 3.4's G1) and target (FG / G2).
    p: Side,
    t: Side,
    p_succ_sorted: Vec<Vec<u32>>,
    p_pred_sorted: Vec<Vec<u32>>,
    t_succ_sorted: Vec<Vec<u32>>,
    t_pred_sorted: Vec<Vec<u32>>,
    directed: bool,
    kind: Kind,
    new: bool,
    /// Target nodes by label, and (3.4 to 3.6) by degree value.
    t_by_label: Vec<PySet>,
    t_by_deg: Vec<PySet>,
    order: Vec<u32>,
    order_index: Vec<usize>,
    // State.
    mapping: Vec<Option<u32>>,
    rev: Vec<Option<u32>>,
    /// Pattern nodes in the order they were mapped (the dicts' order).
    mapped: Vec<u32>,
    t1: Vec<bool>,
    t1_in: Vec<bool>,
    t1_tilde: Vec<bool>,
    t2: Vec<bool>,
    t2_in: Vec<bool>,
    t2_tilde_mem: Vec<bool>,
    t2_tilde: PySet,
    stack: Vec<(u32, Vec<u32>, usize)>,
    node_number: usize,
    done: bool,
    // Scratch for label counts.
    scratch_u: Vec<[i64; 4]>,
    scratch_v: Vec<[i64; 4]>,
    touched: Vec<u32>,
}

impl Vf2pp {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        p: Side,
        t: Side,
        directed: bool,
        kind: Kind,
        new: bool,
        n_labels: usize,
        n_degs: usize,
    ) -> Option<Self> {
        let n1 = p.n();
        let n2 = t.n();
        if n1 == 0 || n2 == 0 {
            return None;
        }
        let mut t_by_label = vec![PySet::default(); n_labels];
        for v in 0..n2 as u32 {
            t_by_label[t.label[v as usize] as usize].add(v, &t.hashes);
        }
        let mut t_by_deg = Vec::new();
        if !new {
            t_by_deg = vec![PySet::default(); n_degs];
            for v in 0..n2 as u32 {
                t_by_deg[t.deg_id[v as usize] as usize].add(v, &t.hashes);
            }
        }
        let mut s = Vf2pp {
            p_succ_sorted: sorted(&p.succ),
            p_pred_sorted: sorted(&p.pred),
            t_succ_sorted: sorted(&t.succ),
            t_pred_sorted: sorted(&t.pred),
            t2_tilde: collect(0..n2 as u32, &t.hashes),
            t2_tilde_mem: vec![true; n2],
            p,
            t,
            directed,
            kind,
            new,
            t_by_label,
            t_by_deg,
            order: Vec::new(),
            order_index: vec![0; n1],
            mapping: vec![None; n1],
            rev: vec![None; n2],
            mapped: Vec::new(),
            t1: vec![false; n1],
            t1_in: vec![false; n1],
            t1_tilde: vec![true; n1],
            t2: vec![false; n2],
            t2_in: vec![false; n2],
            stack: Vec::new(),
            node_number: 0,
            done: false,
            scratch_u: vec![[0; 4]; n_labels],
            scratch_v: vec![[0; 4]; n_labels],
            touched: Vec::new(),
        };
        s.order = s.matching_order(n_labels);
        for (i, &u) in s.order.iter().enumerate() {
            s.order_index[u as usize] = i;
        }
        let first = s.order[0];
        let cands = s.candidates(first);
        s.stack.push((first, cands, 0));
        Some(s)
    }

    fn sg_fits(&self, a: i64, b: i64) -> bool {
        if self.kind == Kind::Iso {
            a == b
        } else {
            a <= b
        }
    }

    fn mono_fits(&self, a: i64, b: i64) -> bool {
        if self.kind == Kind::Mono {
            a <= b
        } else {
            a == b
        }
    }

    /// Neighbors of `u` in the pattern's undirected view: its neighbor list,
    /// or for a directed graph `set(succ.keys()) | set(pred.keys())`.
    fn view_neighbors(&self, u: usize) -> Vec<u32> {
        if !self.directed {
            return self.p.succ[u].clone();
        }
        let h = &self.p.hashes;
        let a = PySet::from_iter(self.p.succ[u].iter().copied(), h);
        let b = PySet::from_iter(self.p.pred[u].iter().copied(), h);
        a.union(&b, h).iter().collect()
    }

    /// `_matching_order`.
    fn matching_order(&self, n_labels: usize) -> Vec<u32> {
        let n1 = self.p.n();
        let h = &self.p.hashes;
        let nbrs: Vec<Vec<u32>> = (0..n1).map(|u| self.view_neighbors(u)).collect();
        let deg: Vec<i64> = (0..n1)
            .map(|u| nbrs[u].len() as i64 + i64::from(nbrs[u].contains(&(u as u32))))
            .collect();
        let mut rarity = vec![0i64; n_labels];
        for &l in &self.t.label {
            rarity[l as usize] += 1;
        }
        let label = &self.p.label;
        let mut v1 = PySet::from_iter(0..n1 as u32, h);
        let mut used = vec![0i64; n1];
        let mut order = Vec::with_capacity(n1);
        let mut visited = vec![false; n1];
        while !v1.is_empty() {
            let max_rarity = v1
                .iter()
                .map(|x| rarity[label[x as usize] as usize])
                .min()
                .unwrap();
            let mut max_node = None;
            for n in v1
                .iter()
                .filter(|&n| rarity[label[n as usize] as usize] == max_rarity)
            {
                if max_node.is_none_or(|m: u32| deg[n as usize] > deg[m as usize]) {
                    max_node = Some(n);
                }
            }
            let max_node = max_node.unwrap();
            visited[max_node as usize] = true;
            let mut layer = vec![max_node];
            while !layer.is_empty() {
                let mut to_add = layer.clone();
                while !to_add.is_empty() {
                    let max_used = to_add.iter().map(|&n| used[n as usize]).max().unwrap();
                    let mu: Vec<u32> = to_add
                        .iter()
                        .copied()
                        .filter(|&n| used[n as usize] == max_used)
                        .collect();
                    let max_deg = mu.iter().map(|&n| deg[n as usize]).max().unwrap();
                    let mut next = None;
                    for &n in mu.iter().filter(|&&n| deg[n as usize] == max_deg) {
                        if next.is_none_or(|m: u32| {
                            rarity[label[n as usize] as usize] < rarity[label[m as usize] as usize]
                        }) {
                            next = Some(n);
                        }
                    }
                    let next = next.unwrap();
                    order.push(next);
                    for &x in &nbrs[next as usize] {
                        used[x as usize] += 1;
                    }
                    let pos = to_add.iter().position(|&x| x == next).unwrap();
                    to_add.remove(pos);
                    rarity[label[next as usize] as usize] -= 1;
                    v1.discard(next, h);
                }
                let mut next_layer = Vec::new();
                for &x in &layer {
                    for &c in &nbrs[x as usize] {
                        if !visited[c as usize] {
                            visited[c as usize] = true;
                            next_layer.push(c);
                        }
                    }
                }
                layer = next_layer;
            }
        }
        order
    }

    /// `rev_map`'s keys (in its order).
    fn rev_keys(&self) -> impl Iterator<Item = u32> + '_ {
        self.mapped
            .iter()
            .map(|&u| self.mapping[u as usize].unwrap())
    }

    /// `s.intersection_update(d)` for a dict or view `d`: the elements of
    /// `items` (in order) that `s` holds, added to a fresh set.
    fn intersect_iter(&self, s: &PySet, items: &[u32]) -> PySet {
        let h = &self.t.hashes;
        collect(items.iter().copied().filter(|&x| s.contains(x, h)), h)
    }

    fn candidates(&self, u: u32) -> Vec<u32> {
        if self.new {
            self.candidates_new(u)
        } else {
            self.candidates_old(u)
        }
    }

    fn candidates_old(&self, u: u32) -> Vec<u32> {
        let h = &self.t.hashes;
        let ui = u as usize;
        let covered_s: Vec<u32> = self.p.succ[ui]
            .iter()
            .copied()
            .filter(|&x| self.mapping[x as usize].is_some())
            .collect();
        let mut covered_p: Vec<u32> = if self.directed {
            self.p.pred[ui]
                .iter()
                .copied()
                .filter(|&x| self.mapping[x as usize].is_some())
                .collect()
        } else {
            Vec::new()
        };
        let label_set = &self.t_by_label[self.p.label[ui] as usize];
        let deg_set = &self.t_by_deg[self.p.deg_id[ui] as usize];
        if covered_s.is_empty() && covered_p.is_empty() {
            let mut c = label_set.copy(h);
            c.intersection_update(deg_set, h);
            c.intersection_update(&self.t2_tilde, h);
            c.discard_all(self.rev_keys(), h);
            return c.iter().collect();
        }
        let m = |x: u32| self.mapping[x as usize].unwrap() as usize;
        let mut common;
        if !self.directed {
            common = collect(self.t.succ[m(covered_s[0])].iter().copied(), h);
            for &x in &covered_s[1..] {
                common = self.intersect_iter(&common, &self.t.succ[m(x)]);
            }
        } else {
            if !covered_s.is_empty() {
                common = collect(self.t.pred[m(covered_s[0])].iter().copied(), h);
                for &x in &covered_s[1..] {
                    common = self.intersect_iter(&common, &self.t.pred[m(x)]);
                }
            } else {
                let p1 = covered_p.pop().unwrap();
                common = collect(self.t.succ[m(p1)].iter().copied(), h);
            }
            for &x in &covered_p {
                common = self.intersect_iter(&common, &self.t.succ[m(x)]);
            }
        }
        common.discard_all(self.rev_keys(), h);
        common.intersection_update(deg_set, h);
        common.intersection_update(label_set, h);
        common.iter().collect()
    }

    fn deg_fits(&self, u: usize, v: usize) -> bool {
        let (a0, a1) = self.p.deg[u];
        let (b0, b1) = self.t.deg[v];
        self.sg_fits(a0, b0) && (!self.directed || self.sg_fits(a1, b1))
    }

    fn loops_fit(&self, u: usize, v: usize) -> bool {
        self.mono_fits(i64::from(self.p.loops[u]), i64::from(self.t.loops[v]))
    }

    fn candidates_new(&self, u: u32) -> Vec<u32> {
        let h = &self.t.hashes;
        let ui = u as usize;
        let label_set = &self.t_by_label[self.p.label[ui] as usize];
        let m = |x: u32| self.mapping[x as usize].unwrap() as usize;
        let covered_s: Vec<u32> = self.p.succ[ui]
            .iter()
            .copied()
            .filter(|&x| self.mapping[x as usize].is_some())
            .collect();
        if !self.directed {
            if !covered_s.is_empty() {
                let keys = &self.t.succ[m(covered_s[0])];
                let lt = &self.t_succ_sorted[m(covered_s[0])];
                // dict_keys & set
                let mut cands = if keys.len() <= label_set.len() {
                    collect(
                        keys.iter().copied().filter(|&x| label_set.contains(x, h)),
                        h,
                    )
                } else {
                    collect(label_set.iter().filter(|x| lt.binary_search(x).is_ok()), h)
                };
                // cands -= rev_map.keys(): a copy, then difference_update.
                cands = cands.copy(h);
                cands.discard_all(self.rev_keys(), h);
                for &x in &covered_s[1..] {
                    cands = self.intersect_iter(&cands, &self.t.succ[m(x)]);
                }
                let result = collect(
                    cands.iter().filter(|&v| {
                        self.loops_fit(ui, v as usize) && self.deg_fits(ui, v as usize)
                    }),
                    h,
                );
                if !result.is_empty() || self.kind != Kind::Mono {
                    return result.iter().collect();
                }
            }
            let mut cands = collect(
                label_set.iter().filter(|&v| self.deg_fits(ui, v as usize)),
                h,
            );
            if self.kind != Kind::Mono {
                cands.intersection_update(&self.t2_tilde, h);
            } else {
                cands.discard_all(self.rev_keys(), h);
            }
            return collect(cands.iter().filter(|&v| self.loops_fit(ui, v as usize)), h)
                .iter()
                .collect();
        }
        let mut covered_p: Vec<u32> = self.p.pred[ui]
            .iter()
            .copied()
            .filter(|&x| self.mapping[x as usize].is_some())
            .collect();
        if !covered_s.is_empty() || !covered_p.is_empty() {
            let mut cands;
            if !covered_s.is_empty() {
                cands = PySet::from_dict(&self.t.pred[m(covered_s[0])], h);
                cands.discard_all(self.rev_keys(), h);
                for &x in &covered_s[1..] {
                    cands = self.intersect_iter(&cands, &self.t.pred[m(x)]);
                }
            } else {
                let p1 = covered_p.pop().unwrap();
                cands = PySet::from_dict(&self.t.succ[m(p1)], h);
                cands.discard_all(self.rev_keys(), h);
            }
            for &x in &covered_p {
                cands = self.intersect_iter(&cands, &self.t.succ[m(x)]);
            }
            cands.intersection_update(label_set, h);
            let result = collect(
                cands
                    .iter()
                    .filter(|&v| self.loops_fit(ui, v as usize) && self.deg_fits(ui, v as usize)),
                h,
            );
            if !result.is_empty() || self.kind != Kind::Mono {
                return result.iter().collect();
            }
        }
        let mut cands = label_set.copy(h);
        if self.kind != Kind::Mono {
            cands.intersection_update(&self.t2_tilde, h);
        } else {
            cands.discard_all(self.rev_keys(), h);
        }
        collect(
            cands
                .iter()
                .filter(|&v| self.loops_fit(ui, v as usize) && self.deg_fits(ui, v as usize)),
            h,
        )
        .iter()
        .collect()
    }

    /// Per-label counts of `nbrs`: [present, in T, in T_in, in T_tilde].
    fn label_counts(
        nbrs: &[u32],
        label: &[u32],
        t: &[bool],
        t_in: &[bool],
        tilde: &[bool],
        out: &mut [[i64; 4]],
        touched: &mut Vec<u32>,
    ) {
        for &x in nbrs {
            let l = label[x as usize] as usize;
            if out[l][0] == 0 {
                touched.push(l as u32);
            }
            out[l][0] += 1;
            out[l][1] += i64::from(t[x as usize]);
            out[l][2] += i64::from(t_in[x as usize]);
            out[l][3] += i64::from(tilde[x as usize]);
        }
    }

    /// Compares the per-label counts of `u`'s and `v`'s successors (or
    /// predecessors) as NetworkX's look-ahead does: the label sets (equal
    /// for ISO and 3.4 to 3.6, else u's within v's), then unless
    /// `sets_only` the counts in T, T_in and T_tilde. `old` is 3.4 to
    /// 3.6's `_cut_PT` (equality throughout, T_in only when directed).
    fn label_check(&mut self, u: usize, v: usize, pred: bool, old: bool, sets_only: bool) -> bool {
        let mut touched = std::mem::take(&mut self.touched);
        touched.clear();
        let (mut su, mut sv) = (
            std::mem::take(&mut self.scratch_u),
            std::mem::take(&mut self.scratch_v),
        );
        let (unbrs, vnbrs) = if pred {
            (&self.p.pred[u], &self.t.pred[v])
        } else {
            (&self.p.succ[u], &self.t.succ[v])
        };
        Self::label_counts(
            unbrs,
            &self.p.label,
            &self.t1,
            &self.t1_in,
            &self.t1_tilde,
            &mut su,
            &mut touched,
        );
        let mark = touched.len();
        Self::label_counts(
            vnbrs,
            &self.t.label,
            &self.t2,
            &self.t2_in,
            &self.t2_tilde_mem,
            &mut sv,
            &mut touched,
        );
        let subset = !old && self.kind != Kind::Iso;
        let mut ok = touched.iter().all(|&l| {
            let (pu, pv) = (su[l as usize][0] > 0, sv[l as usize][0] > 0);
            if subset {
                !pu || pv
            } else {
                pu == pv
            }
        });
        if ok && !sets_only && (old || !vnbrs.is_empty()) {
            let le = !old && self.kind != Kind::Iso;
            let fits = |x: i64, y: i64| if le { x <= y } else { x == y };
            for &l in &touched[..mark] {
                let (a, b) = (su[l as usize], sv[l as usize]);
                let good = fits(a[1], b[1])
                    && (!self.directed || fits(a[2], b[2]))
                    && (if old {
                        a[3] == b[3]
                    } else {
                        self.kind == Kind::Mono || fits(a[3], b[3])
                    });
                if !good {
                    ok = false;
                    break;
                }
            }
        }
        for &l in &touched {
            su[l as usize] = [0; 4];
            sv[l as usize] = [0; 4];
        }
        self.scratch_u = su;
        self.scratch_v = sv;
        self.touched = touched;
        ok
    }

    fn feasible(&mut self, u: u32, v: u32) -> bool {
        let (ui, vi) = (u as usize, v as usize);
        if self.new {
            // _feasible_node_pair
            let eq = self.kind != Kind::Mono;
            for &n in &self.p.succ[ui] {
                if let Some(mn) = self.mapping[n as usize] {
                    if !has(&self.t_succ_sorted, v, mn) {
                        return false;
                    }
                }
            }
            if eq {
                for &n in &self.t.succ[vi] {
                    if let Some(rn) = self.rev[n as usize] {
                        if !has(&self.p_succ_sorted, u, rn) {
                            return false;
                        }
                    }
                }
            }
            if self.directed {
                for &n in &self.p.pred[ui] {
                    if let Some(mn) = self.mapping[n as usize] {
                        if !has(&self.t_pred_sorted, v, mn) {
                            return false;
                        }
                    }
                }
                if eq {
                    for &n in &self.t.pred[vi] {
                        if let Some(rn) = self.rev[n as usize] {
                            if !has(&self.p_pred_sorted, u, rn) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        // _feasible_look_ahead (3.7) / _cut_PT (3.4 to 3.6): all tests are
        // conjunctions, so their order doesn't matter.
        let old = !self.new;
        if self.directed && !self.label_check(ui, vi, true, old, true) {
            return false;
        }
        if !self.label_check(ui, vi, false, old, false) {
            return false;
        }
        !self.directed || self.label_check(ui, vi, true, old, false)
    }

    fn update(&mut self, n1: u32, n2: u32) {
        let h = std::mem::take(&mut self.t.hashes);
        self.update_with(n1, n2, &h);
        self.t.hashes = h;
    }

    fn update_with(&mut self, n1: u32, n2: u32, h: &[i64]) {
        let unc1: Vec<u32> = self.p.succ[n1 as usize]
            .iter()
            .copied()
            .filter(|&x| self.mapping[x as usize].is_none())
            .collect();
        let unc2: Vec<u32> = self.t.succ[n2 as usize]
            .iter()
            .copied()
            .filter(|&x| self.rev[x as usize].is_none())
            .collect();
        for &x in &unc1 {
            self.t1[x as usize] = true;
            self.t1_tilde[x as usize] = false;
        }
        for &x in &unc2 {
            self.t2[x as usize] = true;
            self.t2_tilde_mem[x as usize] = false;
        }
        self.t1[n1 as usize] = false;
        self.t2[n2 as usize] = false;
        self.t2_tilde.discard_all(unc2.iter().copied(), h);
        self.t1_tilde[n1 as usize] = false;
        self.t2_tilde.discard(n2, h);
        self.t2_tilde_mem[n2 as usize] = false;
        if !self.directed {
            return;
        }
        let unc1: Vec<u32> = self.p.pred[n1 as usize]
            .iter()
            .copied()
            .filter(|&x| self.mapping[x as usize].is_none())
            .collect();
        let unc2: Vec<u32> = self.t.pred[n2 as usize]
            .iter()
            .copied()
            .filter(|&x| self.rev[x as usize].is_none())
            .collect();
        for &x in &unc1 {
            self.t1_in[x as usize] = true;
            self.t1_tilde[x as usize] = false;
        }
        for &x in &unc2 {
            self.t2_in[x as usize] = true;
            self.t2_tilde_mem[x as usize] = false;
        }
        self.t1_in[n1 as usize] = false;
        self.t2_in[n2 as usize] = false;
        self.t2_tilde.discard_all(unc2.iter().copied(), h);
        self.t1_tilde[n1 as usize] = false;
        self.t2_tilde.discard(n2, h);
        self.t2_tilde_mem[n2 as usize] = false;
    }

    fn tilde2_add(&mut self, x: u32) {
        let h = std::mem::take(&mut self.t.hashes);
        self.t2_tilde.add(x, &h);
        self.t.hashes = h;
        self.t2_tilde_mem[x as usize] = true;
    }

    fn restore(&mut self, n1: u32, n2: u32) {
        if !self.directed {
            let mut added = false;
            for i in 0..self.p.succ[n1 as usize].len() {
                let nb = self.p.succ[n1 as usize][i];
                if self.mapping[nb as usize].is_some() {
                    added = true;
                    self.t1[n1 as usize] = true;
                } else {
                    if self.p.succ[nb as usize]
                        .iter()
                        .any(|&x| self.mapping[x as usize].is_some())
                    {
                        continue;
                    }
                    self.t1[nb as usize] = false;
                    self.t1_tilde[nb as usize] = true;
                }
            }
            if !added {
                self.t1_tilde[n1 as usize] = true;
            }
            let mut added = false;
            for i in 0..self.t.succ[n2 as usize].len() {
                let nb = self.t.succ[n2 as usize][i];
                if self.rev[nb as usize].is_some() {
                    added = true;
                    self.t2[n2 as usize] = true;
                } else {
                    if self.t.succ[nb as usize]
                        .iter()
                        .any(|&x| self.rev[x as usize].is_some())
                    {
                        continue;
                    }
                    self.t2[nb as usize] = false;
                    self.tilde2_add(nb);
                }
            }
            if !added {
                self.tilde2_add(n2);
            }
            return;
        }
        // _restore_Tinout_Di, pattern side.
        let mut added = false;
        for i in 0..self.p.succ[n1 as usize].len() {
            let s = self.p.succ[n1 as usize][i] as usize;
            if self.mapping[s].is_some() {
                added = true;
                self.t1_in[n1 as usize] = true;
            } else {
                if !self.p.pred[s]
                    .iter()
                    .any(|&x| self.mapping[x as usize].is_some())
                {
                    self.t1[s] = false;
                }
                if !self.p.succ[s]
                    .iter()
                    .any(|&x| self.mapping[x as usize].is_some())
                {
                    self.t1_in[s] = false;
                }
                if !self.t1[s] && !self.t1_in[s] {
                    self.t1_tilde[s] = true;
                }
            }
        }
        for i in 0..self.p.pred[n1 as usize].len() {
            let s = self.p.pred[n1 as usize][i] as usize;
            if self.mapping[s].is_some() {
                added = true;
                self.t1[n1 as usize] = true;
            } else {
                if !self.p.pred[s]
                    .iter()
                    .any(|&x| self.mapping[x as usize].is_some())
                {
                    self.t1[s] = false;
                }
                if !self.p.succ[s]
                    .iter()
                    .any(|&x| self.mapping[x as usize].is_some())
                {
                    self.t1_in[s] = false;
                }
                if !(self.t1[s] || self.t1_in[s]) {
                    self.t1_tilde[s] = true;
                }
            }
        }
        if !added {
            self.t1_tilde[n1 as usize] = true;
        }
        // Target side.
        let mut added = false;
        for i in 0..self.t.succ[n2 as usize].len() {
            let s = self.t.succ[n2 as usize][i] as usize;
            if self.rev[s].is_some() {
                added = true;
                self.t2_in[n2 as usize] = true;
            } else {
                if !self.t.pred[s]
                    .iter()
                    .any(|&x| self.rev[x as usize].is_some())
                {
                    self.t2[s] = false;
                }
                if !self.t.succ[s]
                    .iter()
                    .any(|&x| self.rev[x as usize].is_some())
                {
                    self.t2_in[s] = false;
                }
                if !self.t2[s] && !self.t2_in[s] {
                    self.tilde2_add(s as u32);
                }
            }
        }
        for i in 0..self.t.pred[n2 as usize].len() {
            let s = self.t.pred[n2 as usize][i] as usize;
            if self.rev[s].is_some() {
                added = true;
                self.t2[n2 as usize] = true;
            } else {
                if !self.t.pred[s]
                    .iter()
                    .any(|&x| self.rev[x as usize].is_some())
                {
                    self.t2[s] = false;
                }
                if !self.t.succ[s]
                    .iter()
                    .any(|&x| self.rev[x as usize].is_some())
                {
                    self.t2_in[s] = false;
                }
                if !(self.t2[s] || self.t2_in[s]) {
                    self.tilde2_add(s as u32);
                }
            }
        }
        if !added {
            self.tilde2_add(n2);
        }
    }

    fn map(&mut self, u: u32, v: u32) {
        self.mapping[u as usize] = Some(v);
        self.rev[v as usize] = Some(u);
        self.mapped.push(u);
    }

    fn unmap(&mut self, u: u32) -> u32 {
        let v = self.mapping[u as usize].take().unwrap();
        self.rev[v as usize] = None;
        let last = self.mapped.pop();
        debug_assert_eq!(last, Some(u));
        v
    }

    /// The mapping NetworkX yields: (first graph's node, second graph's
    /// node) pairs in the dict's order, plus the new pair.
    fn emit(&self, u: u32, v: u32) -> Vec<(u32, u32)> {
        let mut out: Vec<(u32, u32)> = self
            .mapped
            .iter()
            .map(|&a| (a, self.mapping[a as usize].unwrap()))
            .collect();
        out.push((u, v));
        if self.new {
            // 3.7 yields rev_map: target (FG, first) -> pattern (SG).
            out.iter().map(|&(a, b)| (b, a)).collect()
        } else {
            out
        }
    }

    /// Up to `max` more mappings (fewer only when the search is over).
    pub fn next_batch(&mut self, max: usize) -> Vec<Vec<(u32, u32)>> {
        let mut out = Vec::new();
        let n1 = self.p.n();
        let n2 = self.t.n();
        while !self.done && out.len() < max {
            let Some(top) = self.stack.last_mut() else {
                self.done = true;
                break;
            };
            if top.2 < top.1.len() {
                let (cur, cand) = (top.0, top.1[top.2]);
                top.2 += 1;
                if self.feasible(cur, cand) {
                    let target = if self.new { n1 } else { n2 };
                    if self.mapped.len() == target - 1 {
                        out.push(self.emit(cur, cand));
                        continue;
                    }
                    self.map(cur, cand);
                    self.update(cur, cand);
                    self.node_number += 1;
                    let node = self.order[self.node_number];
                    let cands = self.candidates(node);
                    self.stack.push((node, cands, 0));
                }
                continue;
            }
            // Exhausted.
            if self.new
                && (n1 as i64 - n2 as i64) > (self.node_number as i64 - self.mapped.len() as i64)
            {
                self.node_number += 1;
                let node = self.order[self.node_number];
                let cands = self.candidates(node);
                *self.stack.last_mut().unwrap() = (node, cands, 0);
                continue;
            }
            self.stack.pop();
            if !self.new {
                self.node_number = self.node_number.saturating_sub(1);
            }
            if let Some(&(node, _, _)) = self.stack.last() {
                if self.new {
                    self.node_number = self.order_index[node as usize];
                }
                let cand = self.unmap(node);
                self.restore(node, cand);
            } else {
                self.done = true;
            }
        }
        out
    }
}
