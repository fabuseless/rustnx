//! PageRank, ported from `networkx.algorithms.link_analysis.pagerank_alg._pagerank_scipy`.

use rayon::prelude::*;

use crate::graph::Csr;

/// Below this many nodes, a sequential sweep beats spawning parallel work.
const PARALLEL_MIN_NODES: usize = 20_000;

pub struct NoConvergence;

pub struct PagerankInput<'a> {
    pub n: usize,
    /// Out-edges and their weights (`None` = all 1), to compute out-strength.
    pub out_adj: &'a Csr,
    pub out_weights: Option<&'a [f64]>,
    /// In-edges and their weights, aligned with `in_adj.targets`.
    pub in_adj: &'a Csr,
    pub in_weights: Option<&'a [f64]>,
    pub alpha: f64,
    pub personalization: Option<Vec<f64>>,
    pub nstart: Option<Vec<f64>>,
    pub dangling: Option<Vec<f64>>,
    pub max_iter: usize,
    pub tol: f64,
}

fn normalized(mut v: Vec<f64>) -> Vec<f64> {
    let s: f64 = v.iter().sum();
    for x in &mut v {
        *x /= s;
    }
    v
}

/// Power iteration exactly as NetworkX does it: row-normalize the weighted
/// adjacency, give dangling nodes the `dangling` distribution, and stop when
/// the L1 change is below `n * tol`.
pub fn pagerank(input: PagerankInput) -> Result<Vec<f64>, NoConvergence> {
    let n = input.n;
    if n == 0 {
        return Ok(Vec::new());
    }
    let weight = |w: Option<&[f64]>, e: usize| w.map_or(1.0, |w| w[e]);

    // Out-strength S and its inverse (NetworkX: `S[~dangling] = 1 / S`).
    let mut inv_s = vec![0.0; n];
    let mut is_dangling = vec![false; n];
    for u in 0..n {
        let s: f64 = input
            .out_adj
            .range(u)
            .map(|e| weight(input.out_weights, e))
            .sum();
        if s == 0.0 {
            is_dangling[u] = true;
        } else {
            inv_s[u] = 1.0 / s;
        }
    }
    let dangling_nodes: Vec<usize> = (0..n).filter(|&u| is_dangling[u]).collect();

    // Transition coefficient for each in-edge u -> v: w(u, v) / S[u].
    let coef: Vec<f64> = (0..n)
        .flat_map(|v| input.in_adj.range(v).map(move |e| (v, e)))
        .map(|(_, e)| inv_s[input.in_adj.targets[e] as usize] * weight(input.in_weights, e))
        .collect();

    let uniform = || vec![1.0 / n as f64; n];
    let mut x = input.nstart.map_or_else(uniform, normalized);
    let p = input.personalization.map_or_else(uniform, normalized);
    let dangling_weights = input.dangling.map_or_else(|| p.clone(), normalized);

    let alpha = input.alpha;
    let in_adj = input.in_adj;
    let step = |v: usize, x: &[f64], danglesum: f64| -> f64 {
        let y: f64 = in_adj
            .range(v)
            .map(|e| x[in_adj.targets[e] as usize] * coef[e])
            .sum();
        alpha * (y + danglesum * dangling_weights[v]) + (1.0 - alpha) * p[v]
    };

    let mut next = vec![0.0; n];
    for _ in 0..input.max_iter {
        let danglesum: f64 = dangling_nodes.iter().map(|&u| x[u]).sum();
        if n >= PARALLEL_MIN_NODES {
            next.par_iter_mut()
                .enumerate()
                .for_each(|(v, out)| *out = step(v, &x, danglesum));
        } else {
            for (v, out) in next.iter_mut().enumerate() {
                *out = step(v, &x, danglesum);
            }
        }
        let err: f64 = next.iter().zip(&x).map(|(a, b)| (a - b).abs()).sum();
        std::mem::swap(&mut x, &mut next);
        if err < n as f64 * input.tol {
            return Ok(x);
        }
    }
    Err(NoConvergence)
}
