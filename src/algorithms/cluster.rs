//! Triangle counts for `triangles`, `clustering` and `transitivity`, as
//! NetworkX's `_triangles_and_degree_iter` and
//! `_directed_triangles_and_degree_iter` count them (unweighted). All counts
//! are integers, so results match NetworkX exactly.

use rayon::prelude::*;

use crate::graph::Csr;

/// Per node: `(t, d, db)`. Undirected: `t` is twice the triangles through
/// the node and `d` its degree without self-loops (`db` is 0). Directed:
/// `t` is NetworkX's directed triangle count, `d` in- plus out-degree and
/// `db` the number of reciprocated neighbors.
pub fn triangle_counts(
    succ: &Csr,
    pred: Option<&Csr>,
    n: usize,
    nodes: &[u32],
) -> Vec<(u64, u64, u64)> {
    const PRED: u8 = 1;
    const SUCC: u8 = 2;
    let ones = |m: u8| u64::from(m & PRED) + u64::from((m & SUCC) >> 1);
    nodes
        .par_iter()
        .map_init(
            || vec![0u8; n],
            |mark, &i| {
                let i = i as usize;
                let mut members: Vec<u32> = Vec::new();
                let mut add = |adj: &Csr, bit: u8, mark: &mut Vec<u8>| {
                    for &j in adj.neighbors(i) {
                        if j as usize != i && mark[j as usize] & bit == 0 {
                            if mark[j as usize] == 0 {
                                members.push(j);
                            }
                            mark[j as usize] |= bit;
                        }
                    }
                };
                let result = match pred {
                    None => {
                        add(succ, SUCC, mark);
                        let mut t = 0u64;
                        for &j in &members {
                            for &k in succ.neighbors(j as usize) {
                                if k != j && mark[k as usize] != 0 {
                                    t += 1;
                                }
                            }
                        }
                        (t, members.len() as u64, 0)
                    }
                    Some(pred) => {
                        add(pred, PRED, mark);
                        add(succ, SUCC, mark);
                        let mut t = 0u64;
                        let (mut d, mut db) = (0u64, 0u64);
                        for &j in &members {
                            let mult = ones(mark[j as usize]);
                            d += mult;
                            db += u64::from(mult == 2);
                            let mut c = 0u64;
                            for adj in [pred, succ] {
                                for &k in adj.neighbors(j as usize) {
                                    if k != j {
                                        c += ones(mark[k as usize]);
                                    }
                                }
                            }
                            t += mult * c;
                        }
                        (t, d, db)
                    }
                };
                for &j in &members {
                    mark[j as usize] = 0;
                }
                result
            },
        )
        .collect()
}

/// Weighted triangles per node in `nodes`, as NetworkX's weighted
/// `clustering` computes them: the cube root of the product of each
/// triangle's three edge weights, each divided by `max_weight`.
/// Undirected (`pred` is `None`): `(t, d, 0)` with `t` counting each
/// triangle twice, as NetworkX's `2 * weighted_triangles`. Directed:
/// `(t, d_total, d_bidirectional)` with NetworkX's directed triangles.
/// `succ_w` and `pred_w` are aligned with `succ` and `pred`. Self-loops are
/// ignored. The cube roots are added in Rust's order, not NetworkX's, so
/// `t` can differ from NetworkX's in the last bits.
pub fn weighted_triangles(
    succ: &Csr,
    succ_w: &[f64],
    pred: Option<(&Csr, &[f64])>,
    n: usize,
    nodes: &[u32],
    max_weight: f64,
) -> Vec<(f64, u64, u64)> {
    nodes
        .par_iter()
        .map_init(
            || (vec![f64::NAN; n], vec![f64::NAN; n]),
            |(to_i, from_i), &i| {
                let i = i as usize;
                // to_i[k] = wt(k, i) for predecessors (all neighbors when
                // undirected); from_i[k] = wt(i, k) for successors.
                let set = |adj: &Csr, w: &[f64], into: &mut Vec<f64>| {
                    let mut members = Vec::new();
                    for e in adj.range(i) {
                        let k = adj.targets[e] as usize;
                        if k != i {
                            into[k] = w[e] / max_weight;
                            members.push(k);
                        }
                    }
                    members
                };
                match pred {
                    None => {
                        let members = set(succ, succ_w, to_i);
                        let mut t = 0.0;
                        for &j in &members {
                            let wij = to_i[j];
                            for e in succ.range(j) {
                                let k = succ.targets[e] as usize;
                                if k != j && !to_i[k].is_nan() {
                                    t += (wij * (succ_w[e] / max_weight) * to_i[k]).cbrt();
                                }
                            }
                        }
                        for &j in &members {
                            to_i[j] = f64::NAN;
                        }
                        (t, members.len() as u64, 0)
                    }
                    Some((pred, pred_w)) => {
                        let preds = set(pred, pred_w, to_i);
                        let succs = set(succ, succ_w, from_i);
                        let mut t = 0.0;
                        // For each neighbor j of i (either direction, weight
                        // a), each arc between j and k (weight c), and each
                        // arc between k and i (weight b): cbrt(a * b * c).
                        let mut add_for = |j: usize, a: f64| {
                            for (adj, w) in [(pred, pred_w), (succ, succ_w)] {
                                for e in adj.range(j) {
                                    let k = adj.targets[e] as usize;
                                    if k == j {
                                        continue;
                                    }
                                    let c = w[e] / max_weight;
                                    if !to_i[k].is_nan() {
                                        t += (a * to_i[k] * c).cbrt();
                                    }
                                    if !from_i[k].is_nan() {
                                        t += (a * from_i[k] * c).cbrt();
                                    }
                                }
                            }
                        };
                        for &j in &preds {
                            add_for(j, to_i[j]);
                        }
                        for &j in &succs {
                            add_for(j, from_i[j]);
                        }
                        let db = preds.iter().filter(|&&j| !from_i[j].is_nan()).count();
                        for &j in &preds {
                            to_i[j] = f64::NAN;
                        }
                        for &j in &succs {
                            from_i[j] = f64::NAN;
                        }
                        (t, (preds.len() + succs.len()) as u64, db as u64)
                    }
                }
            },
        )
        .collect()
}
