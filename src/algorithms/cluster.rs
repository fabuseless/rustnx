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
