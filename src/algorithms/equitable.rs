//! `equitable_color`: NetworkX's port of Kierstead, Kostochka, Mydlarz and
//! Szemerédi's algorithm (`networkx/algorithms/coloring/equitable_coloring.py`,
//! unchanged from 3.4 to 3.7), followed step by step on node positions so
//! every choice NetworkX makes (first free color, first witness, BFS order
//! over colors) comes out the same.
//!
//! NetworkX's state, by name:
//! - `F[node]`: the node's color; `C[color]`: the color's nodes, a list in
//!   the order nodes joined it. Colors are `0..k`, in `C`'s key order.
//! - `L[node]`: neighbors added so far (a self-loop twice).
//! - `N[(node, color)]`: how many of the node's neighbors have that color.
//! - `H[(c1, c2)]`: how many nodes of `c1` have no neighbor in `c2`.
//!
//! The one set whose iteration order reaches the result (`B_cal_prime` in
//! Case II) is a [`PySet`] replica. Wherever NetworkX would raise
//! (`StopIteration`, `KeyError`, `assert`), the port returns `None` so the
//! caller hands the call to NetworkX.

use super::pyset::PySet;

struct State {
    k: usize,
    f: Vec<usize>,
    c: Vec<Vec<u32>>,
    nn: Vec<i32>,
    h: Vec<i64>,
    l: Vec<Vec<u32>>,
    /// `hash(color)` for the set replica: small ints hash to themselves.
    hashes: Vec<i64>,
}

impl State {
    #[inline]
    fn n_at(&self, node: usize, color: usize) -> i32 {
        self.nn[node * self.k + color]
    }

    #[inline]
    fn n_add(&mut self, node: usize, color: usize, d: i32) {
        self.nn[node * self.k + color] += d;
    }

    #[inline]
    fn h_at(&self, c1: usize, c2: usize) -> i64 {
        self.h[c1 * self.k + c2]
    }

    #[inline]
    fn h_add(&mut self, c1: usize, c2: usize, d: i64) {
        self.h[c1 * self.k + c2] += d;
    }

    /// `change_color(u, X, Y)`.
    fn change_color(&mut self, u: usize, x: usize, y: usize) -> Option<()> {
        if self.f[u] != x || x == y {
            return None; // NetworkX asserts
        }
        self.f[u] = y;
        for kk in 0..self.k {
            if self.n_at(u, kk) == 0 {
                self.h_add(x, kk, -1);
                self.h_add(y, kk, 1);
            }
        }
        for i in 0..self.l[u].len() {
            let v = self.l[u][i] as usize;
            self.n_add(v, x, -1);
            self.n_add(v, y, 1);
            if self.n_at(v, x) == 0 {
                self.h_add(self.f[v], x, 1);
            }
            if self.n_at(v, y) == 1 {
                self.h_add(self.f[v], y, -1);
            }
        }
        let pos = self.c[x].iter().position(|&w| w as usize == u)?;
        self.c[x].remove(pos);
        self.c[y].push(u as u32);
        Some(())
    }

    /// `move_witnesses(src, dst, ..., T_cal)`.
    fn move_witnesses(&mut self, src: usize, dst: usize, t_cal: &[Option<usize>]) -> Option<()> {
        let mut x = src;
        while x != dst {
            let y = t_cal[x]?;
            let w = *self.c[x].iter().find(|&&w| self.n_at(w as usize, y) == 0)? as usize;
            self.change_color(w, x, y)?;
            x = y;
        }
        Some(())
    }

    /// `procedure_P(V_minus, V_plus, ..., excluded_colors)`.
    fn procedure_p(
        &mut self,
        v_minus: usize,
        v_plus: usize,
        mut excluded: Vec<bool>,
    ) -> Option<()> {
        let k = self.k;
        // BFS over colors from V-: `A_cal`, `T_cal`, `R_cal`.
        let mut a_cal = vec![false; k];
        let mut a_cal_len = 0;
        let mut t_cal: Vec<Option<usize>> = vec![None; k];
        let mut r_cal = Vec::new();
        let mut reachable = vec![v_minus];
        let mut marked = vec![false; k];
        marked[v_minus] = true;
        let mut idx = 0;
        while idx < reachable.len() {
            let pop = reachable[idx];
            idx += 1;
            if !a_cal[pop] {
                a_cal[pop] = true;
                a_cal_len += 1;
            }
            r_cal.push(pop);
            let next_layer: Vec<usize> = (0..k)
                .filter(|&kk| self.h_at(kk, pop) > 0 && !a_cal[kk] && !excluded[kk] && !marked[kk])
                .collect();
            for &dst in &next_layer {
                t_cal[dst] = Some(pop);
                marked[dst] = true;
            }
            reachable.extend(next_layer);
        }
        let b = k - a_cal_len;

        if a_cal[v_plus] {
            return self.move_witnesses(v_plus, v_minus, &t_cal);
        }

        let mut a_cal_0 = vec![false; k];
        let mut num_terminal = 0;
        let mut made = false;
        for &w_1 in r_cal.iter().rev() {
            let mut broke = false;
            let members = self.c[w_1].clone();
            for &v in &members {
                let v = v as usize;
                // The last color in A_cal, other than W_1, with no neighbor of v.
                let x = (0..k)
                    .rev()
                    .find(|&u| self.n_at(v, u) == 0 && a_cal[u] && u != w_1);
                let Some(x) = x else { continue };
                for u in 0..k {
                    if self.n_at(v, u) >= 1 && !a_cal[u] {
                        let x_prime = u;
                        let w = v;
                        // w's solo neighbor in X_prime.
                        let y = self.l[w].iter().copied().find(|&node| {
                            let node = node as usize;
                            self.f[node] == x_prime && self.n_at(node, w_1) == 1
                        });
                        if let Some(y) = y {
                            let ww = w_1;
                            self.change_color(w, ww, x)?;
                            self.move_witnesses(x, v_minus, &t_cal)?;
                            self.change_color(y as usize, x_prime, ww)?;
                            let mut ex = excluded.clone();
                            for (e, &a) in ex.iter_mut().zip(&a_cal) {
                                *e |= a;
                            }
                            self.procedure_p(x_prime, v_plus, ex)?;
                            made = true;
                            break;
                        }
                    }
                }
                if made {
                    broke = true;
                    break;
                }
            }
            if !broke {
                // No node in W_1 had a solo neighbor (the for loop's `else`).
                a_cal_0[w_1] = true;
                num_terminal += 1;
            }

            if num_terminal == b {
                // Case II. BFS over colors from V+: `B_cal_prime`, `T_cal_prime`.
                let mut b_set = PySet::default();
                let mut in_b = vec![false; k];
                let mut t_cal_prime: Vec<Option<usize>> = vec![None; k];
                let mut reachable = vec![v_plus];
                let mut marked = vec![false; k];
                marked[v_plus] = true;
                let mut idx = 0;
                while idx < reachable.len() {
                    let pop = reachable[idx];
                    idx += 1;
                    if !in_b[pop] {
                        in_b[pop] = true;
                        b_set.add(pop as u32, &self.hashes);
                    }
                    let next_layer: Vec<usize> = (0..k)
                        .filter(|&kk| self.h_at(pop, kk) > 0 && !in_b[kk] && !marked[kk])
                        .collect();
                    for &dst in &next_layer {
                        t_cal_prime[pop] = Some(dst);
                        marked[dst] = true;
                    }
                    reachable.extend(next_layer);
                }

                let n = self.f.len();
                let mut covered = vec![false; n];
                let mut w_covering: Vec<Option<u32>> = vec![None; n];
                // `C[V_plus] + B_prime`, B_prime in the set's iteration order.
                let mut zs = self.c[v_plus].clone();
                for kk in b_set.iter() {
                    zs.extend_from_slice(&self.c[kk as usize]);
                }
                let mut done = false;
                for &z in &zs {
                    let z = z as usize;
                    if covered[z] || !in_b[self.f[z]] {
                        continue;
                    }
                    covered[z] = true;
                    for &w in &self.l[z] {
                        covered[w as usize] = true;
                    }
                    let lz = self.l[z].clone();
                    for &w in &lz {
                        let w = w as usize;
                        let fw = self.f[w];
                        if a_cal_0[fw] && self.n_at(z, fw) == 1 {
                            match w_covering[w] {
                                None => w_covering[w] = Some(z as u32),
                                Some(z_1) => {
                                    let z_1 = z_1 as usize;
                                    let zc = self.f[z_1];
                                    let wc = self.f[w];
                                    self.move_witnesses(wc, v_minus, &t_cal)?;
                                    self.move_witnesses(v_plus, zc, &t_cal_prime)?;
                                    self.change_color(z_1, zc, wc)?;
                                    let w_plus =
                                        (0..k).find(|&kk| self.n_at(w, kk) == 0 && !a_cal[kk])?;
                                    self.change_color(w, wc, w_plus)?;
                                    for kk in 0..k {
                                        if kk != wc && !in_b[kk] {
                                            excluded[kk] = true;
                                        }
                                    }
                                    self.procedure_p(wc, w_plus, excluded.clone())?;
                                    made = true;
                                    done = true;
                                    break;
                                }
                            }
                        }
                    }
                    if done {
                        break;
                    }
                }
                if !done {
                    return None; // NetworkX: `assert False`
                }
            }
            if made {
                break;
            }
        }
        Some(())
    }
}

/// `equitable_color(G, num_colors)` on a simple undirected graph given as
/// sorted neighbor rows (`sorted(G.neighbors(u))` after NetworkX relabels
/// the nodes `0..n` in `G.nodes` order; a self-loop lists `u` once). The
/// caller has checked `max degree < num_colors`. Returns each node's
/// color, or `None` where NetworkX's own run fails (or the padding clique
/// would be unreasonably large).
pub fn equitable_color(rows: &[Vec<u32>], num_colors: usize) -> Option<Vec<usize>> {
    let n = rows.len();
    if n == 0 {
        return Some(Vec::new());
    }
    let kk = num_colors;
    // `pad_graph`: a clique K_p on new nodes n.. so that k divides the
    // count. `add_edges_from(K.edges)` adds no node when p == 1.
    let mut rows: Vec<Vec<u32>> = rows.to_vec();
    if !n.is_multiple_of(kk) {
        let p = kk - n % kk;
        if p >= 2 {
            if p > 4096 {
                return None;
            }
            for i in 0..p {
                rows.push((0..p).filter(|&j| j != i).map(|j| (n + j) as u32).collect());
            }
        }
    }
    let total = rows.len();
    // C's keys: the colors `idx % k` in first-appearance order.
    let k = kk.min(total);
    if total.checked_mul(k)? > 200_000_000 {
        return None;
    }
    let f: Vec<usize> = (0..total).map(|i| i % kk).collect();
    let mut c = vec![Vec::new(); k];
    for (i, &col) in f.iter().enumerate() {
        c[col].push(i as u32);
    }
    let mut h = vec![0i64; k * k];
    for c1 in 0..k {
        for c2 in 0..k {
            h[c1 * k + c2] = c[c1].len() as i64;
        }
    }
    let mut st = State {
        k,
        f,
        c,
        nn: vec![0; total * k],
        h,
        l: vec![Vec::new(); total],
        hashes: (0..k as i64).collect(),
    };

    for (u, row) in rows.iter().enumerate() {
        for &v in row {
            let v = v as usize;
            if v < u {
                continue; // (v, u) was seen when v's row was read
            }
            st.l[u].push(v as u32);
            st.l[v].push(u as u32);
            let (fu, fv) = (st.f[u], st.f[v]);
            st.n_add(u, fv, 1);
            st.n_add(v, fu, 1);
            if fu != fv {
                if st.n_at(u, fv) == 1 {
                    st.h_add(fu, fv, -1);
                }
                if st.n_at(v, fu) == 1 {
                    st.h_add(fv, fu, -1);
                }
            }
        }
        let fu = st.f[u];
        if st.n_at(u, fu) != 0 {
            let y = (0..k).find(|&kc| st.n_at(u, kc) == 0)?;
            st.change_color(u, fu, y)?;
            st.procedure_p(fu, y, vec![false; k])?;
        }
    }
    st.f.truncate(n);
    Some(st.f)
}

/// `procedure_P` on a state built as NetworkX's tests build it
/// (`make_C_from_F`, `make_N_from_L_C`, `make_H_from_C_N`): `l[node]`,
/// colors as indices into `C`'s key order (`f`, `c`), and each color's
/// `hash()` (its value) for the set replica. Returns `(F, C)` afterwards,
/// or `None` where NetworkX raises. For the tests that reach Case II,
/// which `equitable_color` itself practically never does.
pub fn procedure_p_on(
    l: Vec<Vec<u32>>,
    f: Vec<usize>,
    c: Vec<Vec<u32>>,
    hashes: Vec<i64>,
    v_minus: usize,
    v_plus: usize,
) -> Option<(Vec<usize>, Vec<Vec<u32>>)> {
    let k = c.len();
    let n = l.len();
    if f.len() != n || hashes.len() != k || v_minus >= k || v_plus >= k || f.iter().any(|&x| x >= k)
    {
        return None;
    }
    let mut nn = vec![0i32; n * k];
    for (node, row) in l.iter().enumerate() {
        for &v in row {
            let v = v as usize;
            if v >= n {
                return None;
            }
            nn[node * k + f[v]] += 1;
        }
    }
    let mut h = vec![0i64; k * k];
    for c1 in 0..k {
        for c2 in 0..k {
            h[c1 * k + c2] = c[c1]
                .iter()
                .filter(|&&x| nn[x as usize * k + c2] == 0)
                .count() as i64;
        }
    }
    let mut st = State {
        k,
        f,
        c,
        nn,
        h,
        l,
        hashes,
    };
    st.procedure_p(v_minus, v_plus, vec![false; k])?;
    Some((st.f, st.c))
}
