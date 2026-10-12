//! Randomized measures replayed on Python's `random.Random` (Mersenne
//! Twister, `pyrandom`): `approximate_average_clustering`, and the edge
//! swaps of `random_reference` and `lattice_reference`, applied to the
//! NetworkX graph's own dicts exactly as `add_edge` / `remove_edge` do.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::pyrandom::Mt19937;

/// `approximate_average_clustering`'s triangle count: `trials` node draws
/// (`int(seed.random() * n)`, all first), then for each node with two or
/// more neighbors `seed.sample(nbrs, 2)` and a membership test. `None` for
/// an empty graph (NetworkX raises).
pub fn approx_triangles(rows: &[&[u32]], trials: usize, rng: &mut Mt19937) -> Option<u64> {
    let n = rows.len();
    if n == 0 {
        return None;
    }
    let picks: Vec<usize> = (0..trials)
        .map(|_| (rng.random() * n as f64) as usize)
        .collect();
    let mut sorted: Vec<Option<Vec<u32>>> = vec![None; n];
    let mut triangles = 0;
    for i in picks {
        let nbrs = rows[i];
        if nbrs.len() < 2 {
            continue;
        }
        let pair = rng.sample(nbrs, 2);
        let (u, v) = (pair[0], pair[1]);
        let row_v = sorted[v as usize].get_or_insert_with(|| {
            let mut r = rows[v as usize].to_vec();
            r.sort_unstable();
            r
        });
        if row_v.binary_search(&u).is_ok() {
            triangles += 1;
        }
    }
    Some(triangles)
}

/// An undirected simple graph's rows, kept in NetworkX's dict order.
struct Rows {
    rows: Vec<Vec<u32>>,
}

impl Rows {
    fn has(&self, u: u32, v: u32) -> bool {
        self.rows[u as usize].contains(&v)
    }

    fn add(&mut self, u: u32, v: u32) {
        if !self.has(u, v) {
            self.rows[u as usize].push(v);
            if u != v {
                self.rows[v as usize].push(u);
            }
        }
    }

    fn remove(&mut self, u: u32, v: u32) {
        self.rows[u as usize].retain(|&x| x != v);
        if u != v {
            self.rows[v as usize].retain(|&x| x != u);
        }
    }

    /// Whether `t` is reachable from `s` (local edge connectivity > 0).
    fn connected(
        &self,
        s: u32,
        t: u32,
        seen: &mut [u32],
        stamp: u32,
        queue: &mut Vec<u32>,
    ) -> bool {
        queue.clear();
        queue.push(s);
        seen[s as usize] = stamp;
        let mut head = 0;
        while head < queue.len() {
            let u = queue[head];
            head += 1;
            for &w in &self.rows[u as usize] {
                if w == t {
                    return true;
                }
                if seen[w as usize] != stamp {
                    seen[w as usize] = stamp;
                    queue.push(w);
                }
            }
        }
        false
    }
}

/// An edge operation on the graph: add or remove `(u, v)`.
pub enum Op {
    Add(u32, u32),
    Remove(u32, u32),
}

/// The swaps of `random_reference` (`lattice` false) or
/// `lattice_reference` (`lattice` true, with D the ring distance
/// `min(|i - j|, n - |i - j|)`), as operations in NetworkX's order.
/// `old_cdf` selects `cumulative_distribution` before NetworkX 3.6.
/// `None` where NetworkX raises (a choice from an empty neighbor list).
#[allow(clippy::too_many_arguments)]
pub fn reference_swaps(
    rows: Vec<Vec<u32>>,
    niter: usize,
    connectivity: bool,
    lattice: bool,
    old_cdf: bool,
    rng: &mut Mt19937,
) -> Option<Vec<Op>> {
    let n = rows.len();
    let mut g = Rows { rows };
    // keys, degrees = zip(*G.degree())
    let degrees: Vec<u64> = (0..n)
        .map(|u| g.rows[u].len() as u64 + u64::from(g.rows[u].contains(&(u as u32))))
        .collect();
    let cdf: Vec<f64> = if old_cdf {
        let psum: u64 = degrees.iter().sum();
        let mut cdf = vec![0.0f64];
        for (i, &d) in degrees.iter().enumerate() {
            cdf.push(cdf[i] + d as f64 / psum as f64);
        }
        cdf
    } else {
        let mut cumulative = 0.0f64;
        let mut cdf = vec![0.0f64];
        for &d in &degrees {
            cumulative += d as f64;
            cdf.push(cumulative);
        }
        cdf.into_iter().map(|c| c / cumulative).collect()
    };
    let nedges: usize = (0..n)
        .map(|u| g.rows[u].iter().filter(|&&v| v as usize >= u).count())
        .sum();
    let niter = niter.checked_mul(nedges)?;
    let tries = ((n * nedges) as f64 / ((n * (n - 1)) as f64 / 2.0)) as usize;
    // discrete_sequence: bisect_left(cdf, s) - 1, a Python index (-1 is
    // the last node).
    let pick = |s: f64| -> usize {
        let i = cdf.partition_point(|&c| c < s);
        if i == 0 {
            n - 1
        } else {
            i - 1
        }
    };
    let dist = |i: u32, j: u32| -> f64 {
        let d = (i as i64 - j as i64).unsigned_abs() as usize;
        d.min(n - d) as f64
    };
    let mut ops = Vec::new();
    let mut seen = vec![0u32; n];
    let mut stamp = 0u32;
    let mut queue = Vec::new();
    for _ in 0..niter {
        let mut tried = 0;
        while tried < tries {
            let (s1, s2) = (rng.random(), rng.random());
            let (ai, ci) = (pick(s1), pick(s2));
            if ai == ci {
                continue;
            }
            let (a, c) = (ai as u32, ci as u32);
            let row_a = &g.rows[ai];
            if row_a.is_empty() {
                return None; // seed.choice([]) raises
            }
            let b = row_a[rng.below(row_a.len())];
            let row_c = &g.rows[ci];
            if row_c.is_empty() {
                return None;
            }
            let d = row_c[rng.below(row_c.len())];
            if b == a || b == c || b == d || d == a || d == c {
                continue;
            }
            if !g.has(a, d) && !g.has(c, b) {
                if lattice && dist(a, b) + dist(c, d) < dist(a, c) + dist(b, d) {
                    tried += 1;
                    continue;
                }
                g.add(a, d);
                g.add(c, b);
                g.remove(a, b);
                g.remove(c, d);
                ops.extend([
                    Op::Add(a, d),
                    Op::Add(c, b),
                    Op::Remove(a, b),
                    Op::Remove(c, d),
                ]);
                stamp = stamp.wrapping_add(1);
                if stamp == 0 {
                    seen.iter_mut().for_each(|s| *s = 0);
                    stamp = 1;
                }
                if connectivity && !g.connected(a, b, &mut seen, stamp, &mut queue) {
                    g.remove(a, d);
                    g.remove(c, b);
                    g.add(a, b);
                    g.add(c, d);
                    ops.extend([
                        Op::Remove(a, d),
                        Op::Remove(c, b),
                        Op::Add(a, b),
                        Op::Add(c, d),
                    ]);
                } else {
                    break;
                }
            }
            tried += 1;
        }
    }
    Some(ops)
}

/// Applies `ops` to an undirected NetworkX graph's `_adj` (node objects
/// `nodes`), as `G.add_edge(u, v)` / `G.remove_edge(u, v)` do.
pub fn apply_ops<'py>(
    py: Python<'py>,
    adj: &Bound<'py, PyDict>,
    nodes: &[Bound<'py, PyAny>],
    ops: &[Op],
) -> PyResult<()> {
    let row = |u: u32| -> PyResult<Bound<'py, PyDict>> {
        Ok(adj
            .get_item(&nodes[u as usize])?
            .expect("node row")
            .cast_into::<PyDict>()?)
    };
    for op in ops {
        match *op {
            Op::Add(u, v) => {
                let (ru, rv) = (row(u)?, row(v)?);
                let (nu, nv) = (&nodes[u as usize], &nodes[v as usize]);
                // datadict = adj[u].get(v, {}); adj[u][v] = adj[v][u] = datadict
                let data = match ru.get_item(nv)? {
                    Some(d) => d,
                    None => PyDict::new(py).into_any(),
                };
                ru.set_item(nv, &data)?;
                rv.set_item(nu, &data)?;
            }
            Op::Remove(u, v) => {
                row(u)?.del_item(&nodes[v as usize])?;
                if u != v {
                    row(v)?.del_item(&nodes[u as usize])?;
                }
            }
        }
    }
    Ok(())
}

/// `generate_random_paths`' walks on a legacy NumPy `RandomState`: the
/// transition matrix NetworkX builds, each row checked as
/// `RandomState.choice(n, p=row)` checks it, with its normalized cumulative
/// sums kept for the draws.
#[pyclass(module = "rustnx._core", frozen)]
pub struct RandomPaths {
    n: usize,
    cdfs: Vec<Vec<f64>>,
}

/// A path's node indices, and the MT19937 key (bytes) and position after it.
type PathAndState<'py> = (Vec<u32>, Bound<'py, pyo3::types::PyBytes>, usize);

/// NumPy's `kahan_sum` (legacy `mtrand`).
fn kahan_sum(values: &[f64]) -> f64 {
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

#[pymethods]
impl RandomPaths {
    /// `probs`: the `n` by `n` matrix as native-endian float64 bytes. Fails
    /// (ValueError) if any row is one `choice` would refuse.
    #[new]
    fn new(probs: &[u8], n: usize) -> PyResult<Self> {
        let bad = || pyo3::exceptions::PyValueError::new_err("rows NumPy's choice refuses");
        if n == 0 || probs.len() != n * n * 8 {
            return Err(bad());
        }
        let atol = f64::EPSILON.sqrt();
        let mut cdfs = Vec::with_capacity(n);
        for r in 0..n {
            let row: Vec<f64> = probs[r * n * 8..(r + 1) * n * 8]
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_ne_bytes(*b))
                .collect();
            if row.iter().any(|x| x.is_nan() || *x < 0.0) || (kahan_sum(&row) - 1.0).abs() > atol {
                return Err(bad());
            }
            // cdf = p.cumsum(); cdf /= cdf[-1]
            let mut acc = 0.0;
            let mut cdf: Vec<f64> = row
                .iter()
                .map(|&x| {
                    acc += x;
                    acc
                })
                .collect();
            let last = cdf[n - 1];
            cdf.iter_mut().for_each(|c| *c /= last);
            cdfs.push(cdf);
        }
        Ok(RandomPaths { n, cdfs })
    }

    /// As `path`, on the MT19937 state at `address` (NumPy's
    /// `mt19937_state`: 624 key words, then the position as a C int),
    /// advanced in place. The caller holds the bit generator's lock and has
    /// checked the layout against `get_state()`.
    #[pyo3(signature = (start, path_length, address))]
    fn path_at(
        &self,
        start: Option<usize>,
        path_length: usize,
        address: usize,
    ) -> Option<Vec<u32>> {
        if address == 0 || start.is_some_and(|s| s >= self.n) {
            return None;
        }
        // SAFETY: the caller passes the address of a live mt19937_state and
        // holds its lock; nothing else touches it during this call.
        let (key, pos) = unsafe {
            (
                std::slice::from_raw_parts_mut(address as *mut u32, 624),
                &mut *((address + 624 * 4) as *mut i32),
            )
        };
        let mut state: Vec<u32> = key.to_vec();
        state.push(u32::try_from(*pos).ok()?);
        let mut rng = Mt19937::from_state(&state)?;
        let mut idx = match start {
            Some(s) => s,
            None => rng.np_interval((self.n - 1) as u32) as usize,
        };
        let mut out = Vec::with_capacity(path_length + 1);
        out.push(idx as u32);
        for _ in 0..path_length {
            let u = rng.random();
            let next = self.cdfs[idx].partition_point(|&c| c <= u);
            if next >= self.n {
                return None;
            }
            idx = next;
            out.push(idx as u32);
        }
        let new = rng.state();
        key.copy_from_slice(&new[..624]);
        *pos = new[624] as i32;
        Some(out)
    }

    /// One path: its node indices (the start from `randint(n)` when `start`
    /// is None), and the MT19937 key (bytes) and position afterwards.
    #[pyo3(signature = (start, path_length, key, pos))]
    fn path<'py>(
        &self,
        py: Python<'py>,
        start: Option<usize>,
        path_length: usize,
        key: &[u8],
        pos: usize,
    ) -> PyResult<Option<PathAndState<'py>>> {
        if key.len() != 624 * 4 || start.is_some_and(|s| s >= self.n) {
            return Ok(None);
        }
        let mut state: Vec<u32> = key
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_ne_bytes(*b))
            .collect();
        state.push(pos as u32);
        let Some(mut rng) = Mt19937::from_state(&state) else {
            return Ok(None);
        };
        let mut idx = match start {
            Some(s) => s,
            None => rng.np_interval((self.n - 1) as u32) as usize,
        };
        let mut path = Vec::with_capacity(path_length + 1);
        path.push(idx as u32);
        for _ in 0..path_length {
            let u = rng.random();
            // searchsorted(side="right")
            let next = self.cdfs[idx].partition_point(|&c| c <= u);
            if next >= self.n {
                return Ok(None);
            }
            idx = next;
            path.push(idx as u32);
        }
        let new = rng.state();
        let bytes: Vec<u8> = new[..624].iter().flat_map(|w| w.to_ne_bytes()).collect();
        Ok(Some((
            path,
            pyo3::types::PyBytes::new(py, &bytes),
            new[624] as usize,
        )))
    }
}
