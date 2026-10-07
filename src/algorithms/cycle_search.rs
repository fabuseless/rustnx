//! The inner searches of NetworkX's `simple_cycles` and `chordless_cycles`
//! (`_johnson_cycle_search`, `_bounded_cycle_search` and
//! `_chordless_cycle_search`), run on the neighbor lists NetworkX's
//! `_NeighborhoodCache` would hold. NetworkX's own code still picks the
//! components, start nodes and stems (all of which depend on set order);
//! these searches depend only on the order of the neighbor lists, and their
//! sets (`blocked`, `B`, the unblock and relax stacks) end in the same state
//! whatever order they are visited in. Cycles are yielded as lists of the
//! very node objects NetworkX yields, in the same order.

use std::collections::HashSet;
use std::hash::BuildHasherDefault;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::structure_more::PairHasher;

type Set = HashSet<u32, BuildHasherDefault<PairHasher>>;

/// Each node's neighbor objects, in list order.
type NeighborObjects = Vec<Vec<Py<PyAny>>>;

enum Mode {
    /// `_johnson_cycle_search`.
    Johnson {
        closed: Vec<bool>,
        blocked: Vec<bool>,
        b: Vec<Set>,
    },
    /// `_bounded_cycle_search`.
    Bounded {
        bound: i64,
        lock: Vec<Option<i64>>,
        blen: Vec<i64>,
        b: Vec<Set>,
    },
    /// `_chordless_cycle_search`.
    Chordless {
        bound: Option<i64>,
        /// B's neighbor lists (F's when B is F).
        bnbrs: Arc<Vec<Vec<u32>>>,
        blocked: Vec<i64>,
    },
}

/// A graph's neighbor lists as `_NeighborhoodCache` holds them: node
/// indices, the neighbor objects themselves, and optionally a second graph
/// B's lists (`chordless_cycles`' undirected copy) over the same nodes.
#[pyclass(module = "rustnx._core", frozen)]
pub struct CycleLists {
    lists: Arc<Vec<Vec<u32>>>,
    objs: Arc<Vec<Vec<Py<PyAny>>>>,
    blists: Option<Arc<Vec<Vec<u32>>>>,
}

#[pymethods]
impl CycleLists {
    /// `index` maps each node to its position; `rows[i]` is node i's
    /// neighbor list (and `b_rows[i]` its list in B).
    #[new]
    #[pyo3(signature = (index, rows, b_rows=None))]
    fn new(
        index: &Bound<'_, PyDict>,
        rows: &Bound<'_, PyList>,
        b_rows: Option<&Bound<'_, PyList>>,
    ) -> PyResult<Self> {
        let (lists, objs) = index_lists(index, rows)?;
        let blists = match b_rows {
            Some(b) => {
                if b.len() != rows.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "B must have the same nodes",
                    ));
                }
                Some(Arc::new(index_lists(index, b)?.0))
            }
            None => None,
        };
        Ok(CycleLists {
            lists: Arc::new(lists),
            objs: Arc::new(objs),
            blists,
        })
    }
}

/// One search, advanced in batches of cycles.
#[pyclass(module = "rustnx._core")]
pub struct CycleSearch {
    /// F's neighbor lists as node indices, and the neighbor objects.
    nbrs: Arc<Vec<Vec<u32>>>,
    objs: Arc<Vec<Vec<Py<PyAny>>>>,
    start: u32,
    path: Vec<u32>,
    path_objs: Vec<Py<PyAny>>,
    /// Open frames: a node and the position of its next neighbor.
    stack: Vec<(u32, usize)>,
    mode: Mode,
}

impl CycleSearch {
    fn cycle<'py>(
        &self,
        py: Python<'py>,
        extra: Option<&Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::new(py, self.path_objs.iter().map(|o| o.bind(py)))?;
        if let Some(e) = extra {
            list.append(e.bind(py))?;
        }
        Ok(list)
    }

    fn push(&mut self, py: Python<'_>, v: u32, pos: usize, w: u32) {
        let obj = self.objs[v as usize][pos].clone_ref(py);
        self.path.push(w);
        self.path_objs.push(obj);
    }

    fn pop_path(&mut self) -> u32 {
        self.path_objs.pop();
        // Every open frame has its node on the path.
        self.path.pop().expect("a path node per frame")
    }

    /// Up to `max` more cycles (fewer only when the search is over).
    fn run<'py>(&mut self, py: Python<'py>, max: usize, out: &Bound<'py, PyList>) -> PyResult<()> {
        while let Some(&(v, pos)) = self.stack.last() {
            if out.len() >= max {
                return Ok(());
            }
            let row = &self.nbrs[v as usize];
            if pos >= row.len() {
                self.backtrack();
                continue;
            }
            let w = row[pos];
            self.stack.last_mut().unwrap().1 = pos + 1;
            match &mut self.mode {
                Mode::Johnson {
                    closed, blocked, ..
                } => {
                    if w == self.start {
                        *closed.last_mut().unwrap() = true;
                        out.append(self.cycle(py, None)?)?;
                    } else if !blocked[w as usize] {
                        closed.push(false);
                        blocked[w as usize] = true;
                        self.push(py, v, pos, w);
                        self.stack.push((w, 0));
                    }
                }
                Mode::Bounded {
                    bound, lock, blen, ..
                } => {
                    if w == self.start {
                        *blen.last_mut().unwrap() = 1;
                        out.append(self.cycle(py, None)?)?;
                    } else if (self.path.len() as i64) < lock[w as usize].unwrap_or(*bound) {
                        blen.push(*bound);
                        lock[w as usize] = Some(self.path.len() as i64 + 1);
                        self.push(py, v, pos, w);
                        self.stack.push((w, 0));
                    }
                }
                Mode::Chordless {
                    bound,
                    bnbrs,
                    blocked,
                } => {
                    let open = bound.is_none_or(|b| (self.path.len() as i64) < b);
                    if blocked[w as usize] == 1 && open {
                        if self.nbrs[w as usize].contains(&self.start) {
                            let extra = self.objs[v as usize][pos].clone_ref(py);
                            out.append(self.cycle(py, Some(&extra))?)?;
                        } else if !bnbrs[w as usize].contains(&self.start) {
                            for &x in &bnbrs[w as usize] {
                                blocked[x as usize] += 1;
                            }
                            self.push(py, v, pos, w);
                            self.stack.push((w, 0));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// The `else:` branch once `v`'s neighbors are exhausted.
    fn backtrack(&mut self) {
        self.stack.pop();
        // The node whose frame closed (the last on the path).
        let v = self.pop_path();
        match &mut self.mode {
            Mode::Johnson { closed, blocked, b } => {
                if closed.pop().unwrap_or(false) {
                    if let Some(last) = closed.last_mut() {
                        *last = true;
                    }
                    // unblock(v): the closure is the same in any order.
                    let mut todo = vec![v];
                    while let Some(u) = todo.pop() {
                        if blocked[u as usize] {
                            blocked[u as usize] = false;
                            todo.extend(b[u as usize].drain());
                        }
                    }
                } else {
                    for &w in &self.nbrs[v as usize] {
                        b[w as usize].insert(v);
                    }
                }
            }
            Mode::Bounded {
                bound,
                lock,
                blen,
                b,
            } => {
                let bl = blen.pop().unwrap_or(*bound);
                if let Some(last) = blen.last_mut() {
                    *last = (*last).min(bl);
                }
                if bl < *bound {
                    let mut relax = vec![(bl, v)];
                    while let Some((bl, u)) = relax.pop() {
                        let value = *bound - bl + 1;
                        if lock[u as usize].unwrap_or(*bound) < value {
                            lock[u as usize] = Some(value);
                            for &w in &b[u as usize] {
                                if !self.path.contains(&w) {
                                    relax.push((bl + 1, w));
                                }
                            }
                        }
                    }
                } else {
                    for &w in &self.nbrs[v as usize] {
                        b[w as usize].insert(v);
                    }
                }
            }
            Mode::Chordless { bnbrs, blocked, .. } => {
                for &x in &bnbrs[v as usize] {
                    blocked[x as usize] -= 1;
                }
            }
        }
    }
}

#[pymethods]
impl CycleSearch {
    /// A search over `graph`'s lists from `path` (node indices, and
    /// `path_objects`, the objects NetworkX's path list holds). `kind` is
    /// "johnson", "bounded" (with `bound`) or "chordless" (with `bound` or
    /// None; B's lists come from `graph` too).
    #[new]
    #[pyo3(signature = (graph, kind, path, path_objects, bound=None))]
    fn new(
        graph: &CycleLists,
        kind: &str,
        path: Vec<u32>,
        path_objects: Vec<Py<PyAny>>,
        bound: Option<i64>,
    ) -> PyResult<Self> {
        let lists = graph.lists.clone();
        let n = lists.len();
        let bad = || pyo3::exceptions::PyValueError::new_err("bad cycle search input");
        if path.iter().any(|&w| w as usize >= n)
            || path.len() != path_objects.len()
            || path.is_empty()
        {
            return Err(bad());
        }
        let start = path[0];
        let mode = match kind {
            "johnson" => {
                let mut blocked = vec![false; n];
                for &p in &path {
                    blocked[p as usize] = true;
                }
                Mode::Johnson {
                    closed: vec![false],
                    blocked,
                    b: vec![Set::default(); n],
                }
            }
            "bounded" => {
                let bound = bound.ok_or_else(bad)?;
                let mut lock = vec![None; n];
                for &p in &path {
                    lock[p as usize] = Some(0);
                }
                Mode::Bounded {
                    bound,
                    lock,
                    blen: vec![bound],
                    b: vec![Set::default(); n],
                }
            }
            "chordless" => {
                if path.len() < 3 {
                    return Err(bad());
                }
                let bnbrs = graph.blists.clone().unwrap_or_else(|| lists.clone());
                let mut blocked = vec![0i64; n];
                blocked[path[1] as usize] = 1;
                for &w in &path[1..] {
                    for &x in &bnbrs[w as usize] {
                        blocked[x as usize] += 1;
                    }
                }
                Mode::Chordless {
                    bound,
                    bnbrs,
                    blocked,
                }
            }
            _ => return Err(bad()),
        };
        let first = match mode {
            Mode::Chordless { .. } => path[2],
            _ => *path.last().unwrap(),
        };
        // The chordless search starts from path[2] and backtracks over the
        // whole path it was given.
        Ok(CycleSearch {
            nbrs: lists,
            objs: graph.objs.clone(),
            start,
            path,
            path_objs: path_objects,
            stack: vec![(first, 0)],
            mode,
        })
    }

    /// The next cycles, at most `max`; an empty list once the search is over.
    fn next_batch<'py>(&mut self, py: Python<'py>, max: usize) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        self.run(py, max.max(1), &out)?;
        Ok(out)
    }
}

/// `(lists, objects)` for neighbor lists `rows` (a list of lists, as
/// `_NeighborhoodCache` holds them): each neighbor's index (by `index`, a
/// dict from node to index) and the neighbor object.
fn index_lists<'py>(
    index: &Bound<'py, PyDict>,
    rows: &Bound<'py, PyList>,
) -> PyResult<(Vec<Vec<u32>>, NeighborObjects)> {
    let mut lists = Vec::with_capacity(rows.len());
    let mut objects = Vec::with_capacity(rows.len());
    for row in rows.iter() {
        let row = row.cast_into::<PyList>()?;
        let mut l = Vec::with_capacity(row.len());
        let mut o = Vec::with_capacity(row.len());
        for w in row.iter() {
            let Some(i) = index.get_item(&w)? else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "neighbor outside the graph",
                ));
            };
            l.push(i.extract::<u32>()?);
            o.push(w.unbind());
        }
        lists.push(l);
        objects.push(o);
    }
    Ok((lists, objects))
}
