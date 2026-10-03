//! Power iterations for `eigenvector_centrality` and `katz_centrality`,
//! ported from NetworkX so that every value is accumulated in the same order
//! (results are bit-for-bit NetworkX's).

use crate::graph::Csr;

/// CPython's `sum()` of floats: from Python 3.12 it uses Neumaier
/// compensated summation; before that, plain left-to-right addition.
pub fn py_sum(values: impl Iterator<Item = f64>, compensated: bool) -> f64 {
    let mut s = 0.0f64;
    if !compensated {
        for x in values {
            s += x;
        }
        return s;
    }
    let mut c = 0.0f64;
    for x in values {
        let t = s + x;
        if s.abs() >= x.abs() {
            c += (s - t) + x;
        } else {
            c += (x - t) + s;
        }
        s = t;
    }
    if c != 0.0 && c.is_finite() {
        s += c;
    }
    s
}

pub struct Iteration<'a> {
    pub adj: &'a Csr,
    pub weights: Option<&'a [f64]>,
    /// Node positions in the order of the vector's keys (NetworkX's
    /// `for n in x`).
    pub order: &'a [u32],
    pub max_iter: usize,
    pub tol: f64,
    pub compensated_sum: bool,
}

impl Iteration<'_> {
    fn spread(&self, from: &[f64], into: &mut [f64]) {
        for &n in self.order {
            let n = n as usize;
            for e in self.adj.range(n) {
                let w = self.weights.map_or(1.0, |w| w[e]);
                into[self.adj.targets[e] as usize] += from[n] * w;
            }
        }
    }

    fn error(&self, x: &[f64], last: &[f64]) -> f64 {
        py_sum(
            self.order
                .iter()
                .map(|&n| (x[n as usize] - last[n as usize]).abs()),
            self.compensated_sum,
        )
    }

    /// `eigenvector_centrality`. `norm` is Python's `math.hypot` over the
    /// vector in key order. Returns `Ok(None)` if it doesn't converge.
    pub fn eigenvector<E>(
        &self,
        mut x: Vec<f64>,
        mut norm: impl FnMut(&[f64]) -> Result<f64, E>,
    ) -> Result<Option<Vec<f64>>, E> {
        let threshold = self.order.len() as f64 * self.tol;
        let mut ordered = vec![0.0; self.order.len()];
        for _ in 0..self.max_iter {
            let last = x.clone();
            self.spread(&last, &mut x);
            for (o, &n) in ordered.iter_mut().zip(self.order) {
                *o = x[n as usize];
            }
            let mut h = norm(&ordered)?;
            if h == 0.0 {
                h = 1.0; // `math.hypot(...) or 1`
            }
            for v in x.iter_mut() {
                *v /= h;
            }
            if self.error(&x, &last) < threshold {
                return Ok(Some(x));
            }
        }
        Ok(None)
    }

    /// `katz_centrality` before normalization, with a scalar `beta`.
    pub fn katz(&self, mut x: Vec<f64>, alpha: f64, beta: f64) -> Option<Vec<f64>> {
        let threshold = self.order.len() as f64 * self.tol;
        let n = x.len();
        for _ in 0..self.max_iter {
            let last = x;
            x = vec![0.0; n];
            self.spread(&last, &mut x);
            for &v in self.order {
                let v = v as usize;
                x[v] = alpha * x[v] + beta;
            }
            if self.error(&x, &last) < threshold {
                return Some(x);
            }
        }
        None
    }
}
