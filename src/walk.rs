//! NetworkX 3.7's `random_walk` stepped in Rust. Each step makes the draw
//! `seed.choice` or `seed.choices` would make, on the `random.Random`
//! object's own Mersenne Twister state, updated in place (under the
//! object's lock, as its own methods take it), so the generator is always
//! exactly where NetworkX's walk would leave it. `_mt_layout` finds the
//! state inside the object and the caller checks the replay against
//! Python's own draws before using it.

use pyo3::exceptions::{PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyInt};

const N: usize = 624;
const M: usize = 397;

/// Where `random.Random`'s `index` and `state[624]` sit in the object.
#[derive(Clone, Copy)]
struct Layout {
    index: usize,
    state: usize,
}

/// The object's generator, borrowed in place. Only built while the object
/// is locked (or the GIL is held) and alive.
struct Mt {
    index: *mut i32,
    state: *mut u32,
}

impl Mt {
    /// # Safety
    /// `obj` must be a `random.Random` whose layout `layout` describes, kept
    /// alive and locked while the result is used.
    unsafe fn new(obj: &Bound<'_, PyAny>, layout: Layout) -> Self {
        let base = obj.as_ptr() as *mut u8;
        unsafe {
            Mt {
                index: base.add(layout.index) as *mut i32,
                state: base.add(layout.state) as *mut u32,
            }
        }
    }

    /// `genrand_uint32`.
    fn next_u32(&mut self) -> u32 {
        unsafe {
            let mt = std::slice::from_raw_parts_mut(self.state, N);
            let mut index = *self.index;
            if index < 0 || index as usize >= N {
                let mag = |y: u32| if y & 1 == 1 { 0x9908_b0df } else { 0 };
                for kk in 0..N - M {
                    let y = (mt[kk] & 0x8000_0000) | (mt[kk + 1] & 0x7fff_ffff);
                    mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag(y);
                }
                for kk in N - M..N - 1 {
                    let y = (mt[kk] & 0x8000_0000) | (mt[kk + 1] & 0x7fff_ffff);
                    mt[kk] = mt[kk + M - N] ^ (y >> 1) ^ mag(y);
                }
                let y = (mt[N - 1] & 0x8000_0000) | (mt[0] & 0x7fff_ffff);
                mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag(y);
                index = 0;
            }
            let mut y = mt[index as usize];
            *self.index = index + 1;
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c_5680;
            y ^= (y << 15) & 0xefc6_0000;
            y ^= y >> 18;
            y
        }
    }

    /// `_randbelow(n)` for 1 <= n < 2**32: `getrandbits(n.bit_length())`
    /// until below `n`.
    fn below(&mut self, n: usize) -> usize {
        let k = usize::BITS - n.leading_zeros();
        loop {
            let r = (self.next_u32() >> (32 - k)) as usize;
            if r < n {
                return r;
            }
        }
    }

    /// `random()` (`genrand_res53`).
    fn random(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) * (1.0 / 9_007_199_254_740_992.0)
    }
}

/// The offsets of `index` and `state` in `obj` (a `random.Random` whose C
/// part is `size` bytes), found by matching `getstate()`'s words; None if
/// they aren't found exactly once.
#[pyfunction]
pub fn _mt_layout(
    obj: &Bound<'_, PyAny>,
    words: Vec<u32>,
    index: i32,
    size: usize,
) -> Option<(usize, usize)> {
    if words.len() != N || size < 4 * (N + 1) {
        return None;
    }
    let bytes = unsafe { std::slice::from_raw_parts(obj.as_ptr() as *const u8, size) };
    let read = |off: usize| u32::from_ne_bytes(bytes[off..off + 4].try_into().unwrap());
    let mut found = None;
    let mut off = 0;
    while off + 4 * (N + 1) <= size {
        if read(off) as i32 == index && (0..N).all(|i| read(off + 4 + 4 * i) == words[i]) {
            if found.is_some() {
                return None;
            }
            found = Some((off, off + 4));
        }
        off += 4;
    }
    found
}

/// Draws on `obj`'s state for the caller's self-check: `kind` 0 is
/// `_randbelow(n)`, 1 is `random()`.
#[pyfunction]
pub fn _mt_draws(
    obj: &Bound<'_, PyAny>,
    layout: (usize, usize),
    draws: Vec<(u8, usize)>,
) -> Vec<f64> {
    let layout = Layout {
        index: layout.0,
        state: layout.1,
    };
    pyo3::sync::critical_section::with_critical_section(obj, || {
        let mut mt = unsafe { Mt::new(obj, layout) };
        draws
            .iter()
            .map(|&(kind, n)| {
                if kind == 0 {
                    mt.below(n) as f64
                } else {
                    mt.random()
                }
            })
            .collect()
    })
}

enum Entry {
    /// `seed.choice(list(nbrs))`.
    Uniform(Vec<Py<PyAny>>),
    /// `seed.choices(list(nbrs), weights=wts)`: the running sums (exact:
    /// ints stay below 2**53) and their total.
    Weighted(Vec<Py<PyAny>>, Vec<f64>, f64),
    /// No neighbors, or every weight zero: the walk ends.
    Stop,
    /// Weights of other types: NetworkX's own step, in Python.
    Python,
}

#[pyclass(module = "rustnx._core")]
pub struct RandomWalker {
    graph: Py<PyAny>,
    seed: Py<PyAny>,
    layout: Layout,
    weight: Option<Py<PyAny>>,
    /// NetworkX's step for weights `Entry::Weighted` can't hold: called
    /// with the neighbor dict, returns the next node, or `stop` to end.
    slow: Py<PyAny>,
    stop: Py<PyAny>,
    node: Py<PyAny>,
    done: bool,
    /// The graph's `__networkx_cache__` this walk registered `sentinel` in;
    /// the entries are dropped once it changes or loses the key.
    cache: Option<Py<PyAny>>,
    sentinel: Py<PyAny>,
    index: Py<PyDict>,
    entries: Vec<Entry>,
    /// Per entry, the entry index of each neighbor once looked up
    /// (`u32::MAX` until then), and the current node's entry.
    links: Vec<Vec<u32>>,
    cur: Option<usize>,
}

impl RandomWalker {
    fn entry(&self, py: Python<'_>, nbrs: &Bound<'_, PyAny>) -> PyResult<Entry> {
        // Plain dicts only (as `Graph` builds them); anything else takes
        // NetworkX's own step.
        if !nbrs.is_exact_instance_of::<PyDict>() {
            return Ok(Entry::Python);
        }
        let nbrs = nbrs.cast::<PyDict>()?;
        if nbrs.is_empty() {
            return Ok(Entry::Stop);
        }
        let pop: Vec<Py<PyAny>> = nbrs.keys().iter().map(|k| k.unbind()).collect();
        if pop.len() >= 1 << 32 {
            return Ok(Entry::Python);
        }
        let Some(weight) = &self.weight else {
            return Ok(Entry::Uniform(pop));
        };
        let weight = weight.bind(py);
        let mut wts = Vec::with_capacity(pop.len());
        let mut int_total: u64 = 0;
        for data in nbrs.values().iter() {
            if !data.is_exact_instance_of::<PyDict>() {
                return Ok(Entry::Python);
            }
            let data = data.cast::<PyDict>()?;
            let w = match data.get_item(weight)? {
                None => 1.0,
                Some(w) if w.is_exact_instance_of::<PyFloat>() => w.extract::<f64>()?,
                Some(w)
                    if w.is_exact_instance_of::<PyInt>()
                        || w.is_instance_of::<pyo3::types::PyBool>() =>
                {
                    let Ok(v) = w.extract::<i64>() else {
                        return Ok(Entry::Python);
                    };
                    int_total = int_total.saturating_add(v.unsigned_abs());
                    if int_total >= 1 << 53 {
                        return Ok(Entry::Python);
                    }
                    v as f64
                }
                Some(_) => return Ok(Entry::Python),
            };
            wts.push(w);
        }
        if wts.iter().any(|&w| w < 0.0) {
            return Err(PyValueError::new_err(
                "random_walk doesn't support negative weights",
            ));
        }
        // With no negative weights, sum(wts) == 0 exactly when all are zero.
        if wts.iter().all(|&w| w == 0.0) {
            return Ok(Entry::Stop);
        }
        let mut cum = Vec::with_capacity(wts.len());
        let mut running = 0.0;
        for w in wts {
            running += w;
            cum.push(running);
        }
        let total = running + 0.0;
        Ok(Entry::Weighted(pop, cum, total))
    }

    /// The cache state: whether entries may be kept, after dropping them if
    /// the graph changed.
    fn refresh(&mut self, py: Python<'_>) -> PyResult<bool> {
        let current = self.graph.bind(py).getattr("__networkx_cache__").ok();
        let valid = match (&self.cache, &current) {
            (Some(c), Some(cur)) => {
                c.bind(py).is(cur) && cur.contains(self.sentinel.bind(py)).unwrap_or(false)
            }
            _ => false,
        };
        if valid {
            return Ok(true);
        }
        self.entries.clear();
        self.links.clear();
        self.cur = None;
        self.index.bind(py).clear();
        self.cache = None;
        if let Some(cur) = current {
            if let Ok(d) = cur.cast::<PyDict>() {
                if d.is_exact_instance_of::<PyDict>() {
                    d.set_item(self.sentinel.bind(py), py.None())?;
                    self.cache = Some(cur.unbind());
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn lookup(&self, py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
        match self.index.bind(py).get_item(node)? {
            Some(i) => Ok(Some(i.extract::<usize>()?)),
            None => Ok(None),
        }
    }

    fn step(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let cached = self.refresh(py)?;
        let node = self.node.bind(py).clone();
        let mut at = match (cached, self.cur) {
            (false, _) => None,
            (true, Some(i)) => Some(i),
            (true, None) => self.lookup(py, &node)?,
        };
        let fresh;
        let entry = match at {
            Some(i) => &self.entries[i],
            None => {
                let nbrs = self.graph.bind(py).getattr("_adj")?.get_item(&node)?;
                let e = self.entry(py, &nbrs)?;
                if let Entry::Python = e {
                    self.cur = None;
                    let next = self.slow.bind(py).call1((nbrs,))?;
                    if next.is(self.stop.bind(py)) {
                        return Err(PyStopIteration::new_err(()));
                    }
                    return Ok(next.unbind());
                }
                if cached {
                    let i = self.entries.len();
                    self.index.bind(py).set_item(&node, i)?;
                    let width = match &e {
                        Entry::Uniform(pop) | Entry::Weighted(pop, _, _) => pop.len(),
                        _ => 0,
                    };
                    self.links.push(vec![u32::MAX; width]);
                    self.entries.push(e);
                    at = Some(i);
                    &self.entries[i]
                } else {
                    fresh = e;
                    &fresh
                }
            }
        };
        let seed = self.seed.bind(py);
        let layout = self.layout;
        let (pop, i) = match entry {
            Entry::Stop => return Err(PyStopIteration::new_err(())),
            Entry::Python => unreachable!(),
            Entry::Uniform(pop) => {
                let i = pyo3::sync::critical_section::with_critical_section(seed, || {
                    unsafe { Mt::new(seed, layout) }.below(pop.len())
                });
                (pop, i)
            }
            Entry::Weighted(pop, cum, total) => {
                if !total.is_finite() {
                    return Err(PyValueError::new_err("Total of weights must be finite"));
                }
                let x = pyo3::sync::critical_section::with_critical_section(seed, || {
                    unsafe { Mt::new(seed, layout) }.random()
                }) * total;
                // bisect(cum_weights, x, 0, n - 1)
                let (mut lo, mut hi) = (0, pop.len() - 1);
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    if x < cum[mid] {
                        hi = mid;
                    } else {
                        lo = mid + 1;
                    }
                }
                (pop, lo)
            }
        };
        let picked = pop[i].clone_ref(py);
        self.cur = match at {
            None => None,
            Some(a) => match self.links[a][i] {
                u32::MAX => {
                    let found = self.lookup(py, picked.bind(py))?;
                    if let Some(f) = found {
                        self.links[a][i] = f as u32;
                    }
                    found
                }
                f => Some(f as usize),
            },
        };
        Ok(picked)
    }
}

#[pymethods]
impl RandomWalker {
    /// A walk from `start` (already yielded by the caller) on `graph` with
    /// `seed` (a `random.Random` whose state is at `layout`).
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        graph: Py<PyAny>,
        start: Py<PyAny>,
        weight: Option<Py<PyAny>>,
        seed: Py<PyAny>,
        layout: (usize, usize),
        slow: Py<PyAny>,
        stop: Py<PyAny>,
    ) -> PyResult<Self> {
        Ok(RandomWalker {
            graph,
            seed,
            layout: Layout {
                index: layout.0,
                state: layout.1,
            },
            weight,
            slow,
            stop,
            node: start,
            done: false,
            cache: None,
            sentinel: py.import("builtins")?.getattr("object")?.call0()?.unbind(),
            index: PyDict::new(py).unbind(),
            entries: Vec::new(),
            links: Vec::new(),
            cur: None,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.done {
            return Err(PyStopIteration::new_err(()));
        }
        match self.step(py) {
            Ok(node) => {
                self.node = node.clone_ref(py);
                Ok(node)
            }
            Err(e) => {
                self.done = true;
                Err(e)
            }
        }
    }
}
