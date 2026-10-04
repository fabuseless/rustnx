//! Graph operators and structure functions (batch 21).
//!
//! The operators (`union`, `compose`, `reverse`, `moral_graph`, ...) spend
//! all their time in NetworkX's `add_nodes_from` / `add_edges_from` loops.
//! Here the result graph's dicts are filled directly, replaying exactly the
//! dict operations those loops perform, in the same order: the same keys
//! inserted in the same order, the same attribute dicts shared or copied.

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::graph::CoreGraph;

fn changed() -> PyErr {
    PyNotImplementedError::new_err("the graph does not match its conversion")
}

/// One NetworkX graph as its methods iterate it: nodes and their attribute
/// dicts in `G._node` order, and each `G._adj` row's entries (target
/// position, the key object, the edge's data dict) in order.
#[pyclass(frozen, module = "rustnx._core")]
pub struct OpView {
    directed: bool,
    nodes: Vec<Py<PyAny>>,
    ndata: Vec<Py<PyAny>>,
    offsets: Vec<usize>,
    targets: Vec<u32>,
    keys: Vec<Py<PyAny>>,
    data: Vec<Py<PyAny>>,
}

/// `target.update(src)`, or `target.update(deepcopy(src))`. An empty dict
/// changes nothing either way, so it skips the call.
fn update_from<'py>(
    target: &Bound<'py, PyDict>,
    src: &Bound<'py, PyAny>,
    deepcopy: Option<&Bound<'py, PyAny>>,
) -> PyResult<()> {
    if let Ok(d) = src.cast::<PyDict>() {
        if d.is_empty() {
            return Ok(());
        }
    }
    let src = match deepcopy {
        Some(f) => f.call1((src,))?,
        None => src.clone(),
    };
    match src.cast::<PyDict>() {
        Ok(d) => target.update(d.as_mapping()),
        Err(_) => target.call_method1("update", (src,)).map(|_| ()),
    }
}

/// Where each node of a view ended up in the result graph: its
/// successor (or adjacency) row and, for directed results, predecessor row.
struct Rows<'py> {
    succ: Vec<Bound<'py, PyDict>>,
    pred: Option<Vec<Bound<'py, PyDict>>>,
}

fn row_of<'py>(adj: &Bound<'py, PyDict>, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    match adj.get_item(key)? {
        Some(row) => Ok(row.cast_into::<PyDict>()?),
        None => Err(changed()),
    }
}

/// `R.add_edges_from` for one edge `(a, b, dd)` whose endpoints are in `R`.
fn add_edge_rows<'py>(
    py: Python<'py>,
    row_a: &Bound<'py, PyDict>,
    back_row_b: &Bound<'py, PyDict>,
    akey: &Bound<'py, PyAny>,
    bkey: &Bound<'py, PyAny>,
    dd: Option<&Bound<'py, PyAny>>,
    deepcopy: Option<&Bound<'py, PyAny>>,
) -> PyResult<()> {
    let datadict = match row_a.get_item(bkey)? {
        Some(d) => d.cast_into::<PyDict>()?,
        None => PyDict::new(py),
    };
    if let Some(dd) = dd {
        update_from(&datadict, dd, deepcopy)?;
    }
    row_a.set_item(bkey, &datadict)?;
    back_row_b.set_item(akey, &datadict)
}

impl OpView {
    pub fn read<'py>(
        core: &CoreGraph,
        nodes: &Bound<'py, PyList>,
        node_dict: &Bound<'py, PyDict>,
        adj: &Bound<'py, PyDict>,
    ) -> PyResult<Self> {
        let n = core.n;
        if nodes.len() != n || node_dict.len() != n || adj.len() != n {
            return Err(changed());
        }
        let mut nv = Vec::with_capacity(n);
        let mut nd = Vec::with_capacity(n);
        for (i, (k, d)) in node_dict.iter().enumerate() {
            if !k.is(&nodes.get_item(i)?) {
                return Err(changed());
            }
            nv.push(k.unbind());
            nd.push(d.unbind());
        }
        let m = core.succ.targets.len();
        let mut keys = Vec::with_capacity(m);
        let mut data = Vec::with_capacity(m);
        for (i, (k, row)) in adj.iter().enumerate() {
            let row = row.cast_into::<PyDict>().map_err(|_| changed())?;
            if !k.is(nv[i].bind(adj.py())) || row.len() != core.succ.range(i).len() {
                return Err(changed());
            }
            for (key, dd) in row.iter() {
                keys.push(key.unbind());
                data.push(dd.unbind());
            }
        }
        Ok(OpView {
            directed: core.directed,
            nodes: nv,
            ndata: nd,
            offsets: core.succ.offsets.clone(),
            targets: core.succ.targets.clone(),
            keys,
            data,
        })
    }

    fn n(&self) -> usize {
        self.nodes.len()
    }

    fn range(&self, u: usize) -> std::ops::Range<usize> {
        self.offsets[u]..self.offsets[u + 1]
    }

    /// Entries `(u, e)` in `G.edges()` order: undirected graphs give each
    /// edge once, from the endpoint NetworkX reaches first.
    fn edge_entries(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for u in 0..self.n() {
            for e in self.range(u) {
                if self.directed || self.targets[e] as usize >= u {
                    out.push((u, e));
                }
            }
        }
        out
    }

    /// `R.add_nodes_from(G.nodes(data=True))` (or with `deepcopy(d)`):
    /// returns each node's rows in R.
    fn add_nodes_to<'py>(
        &self,
        py: Python<'py>,
        node: &Bound<'py, PyDict>,
        succ: &Bound<'py, PyDict>,
        pred: Option<&Bound<'py, PyDict>>,
        deepcopy: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Rows<'py>> {
        let n = self.n();
        let mut rows = Rows {
            succ: Vec::with_capacity(n),
            pred: pred.map(|_| Vec::with_capacity(n)),
        };
        for (k, d) in self.nodes.iter().zip(&self.ndata) {
            let k = k.bind(py);
            let target = match node.get_item(k)? {
                Some(existing) => {
                    rows.succ.push(row_of(succ, k)?);
                    if let (Some(p), Some(pr)) = (pred, rows.pred.as_mut()) {
                        pr.push(row_of(p, k)?);
                    }
                    existing.cast_into::<PyDict>()?
                }
                None => {
                    let s = PyDict::new(py);
                    succ.set_item(k, &s)?;
                    rows.succ.push(s);
                    if let (Some(p), Some(pr)) = (pred, rows.pred.as_mut()) {
                        let r = PyDict::new(py);
                        p.set_item(k, &r)?;
                        pr.push(r);
                    }
                    let nd = PyDict::new(py);
                    node.set_item(k, &nd)?;
                    nd
                }
            };
            update_from(&target, d.bind(py), deepcopy)?;
        }
        Ok(rows)
    }
}

#[pymethods]
impl OpView {
    /// The graph `nx.relabel_nodes(G, mapping)` builds (with `labels[i]` the
    /// new label of node `i`, all distinct): `_relabel_copy` re-adds the
    /// edges in `G.edges()` order, which reorders undirected rows. Data
    /// dicts stay the originals; every user of the result copies them.
    fn relabeled(&self, py: Python<'_>, labels: Vec<Py<PyAny>>) -> PyResult<OpView> {
        let n = self.n();
        if labels.len() != n {
            return Err(changed());
        }
        if self.directed {
            return Ok(OpView {
                directed: true,
                nodes: labels.iter().map(|l| l.clone_ref(py)).collect(),
                ndata: self.ndata.iter().map(|d| d.clone_ref(py)).collect(),
                offsets: self.offsets.clone(),
                targets: self.targets.clone(),
                keys: self
                    .targets
                    .iter()
                    .map(|&t| labels[t as usize].clone_ref(py))
                    .collect(),
                data: self.data.iter().map(|d| d.clone_ref(py)).collect(),
            });
        }
        let mut rows: Vec<Vec<(u32, usize)>> = vec![Vec::new(); n];
        for (u, e) in self.edge_entries() {
            let t = self.targets[e] as usize;
            rows[u].push((t as u32, e));
            if t != u {
                rows[t].push((u as u32, e));
            }
        }
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        let (mut targets, mut keys, mut data) = (Vec::new(), Vec::new(), Vec::new());
        for row in &rows {
            for &(t, e) in row {
                targets.push(t);
                keys.push(labels[t as usize].clone_ref(py));
                data.push(self.data[e].clone_ref(py));
            }
            offsets.push(targets.len());
        }
        Ok(OpView {
            directed: false,
            nodes: labels,
            ndata: self.ndata.iter().map(|d| d.clone_ref(py)).collect(),
            offsets,
            targets,
            keys,
            data,
        })
    }

    /// Whether any node is already a key of `node` (`union_all`'s
    /// disjointness test against the nodes added so far).
    fn shares_node(&self, py: Python<'_>, node: &Bound<'_, PyDict>) -> PyResult<bool> {
        for k in &self.nodes {
            if node.contains(k.bind(py))? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Adds this graph to `R` (given as its `_node`, `_succ`/`_adj` and, if
    /// directed, `_pred` dicts) as `R.add_nodes_from(G.nodes(data=True))`
    /// then `R.add_edges_from(...)` would. `edges`: 0 none, 1 `G.edges(data=
    /// True)`, 2 every `G._adj` entry (`to_undirected`, `copy`). `reverse`
    /// adds `(v, u, d)` for each `(u, v, d)`; `deep` passes `deepcopy(d)`.
    #[pyo3(signature = (node, succ, pred, edges, reverse=false, deep=false))]
    fn add_to<'py>(
        &self,
        py: Python<'py>,
        node: &Bound<'py, PyDict>,
        succ: &Bound<'py, PyDict>,
        pred: Option<Bound<'py, PyDict>>,
        edges: u8,
        reverse: bool,
        deep: bool,
    ) -> PyResult<()> {
        let deepcopy = if deep {
            Some(py.import("copy")?.getattr("deepcopy")?)
        } else {
            None
        };
        let deepcopy = deepcopy.as_ref();
        let rows = self.add_nodes_to(py, node, succ, pred.as_ref(), deepcopy)?;
        let entries: Vec<(usize, usize)> = match edges {
            0 => Vec::new(),
            1 => self.edge_entries(),
            _ => (0..self.n())
                .flat_map(|u| self.range(u).map(move |e| (u, e)))
                .collect(),
        };
        for (u, e) in entries {
            let t = self.targets[e] as usize;
            let (a, b, akey, bkey) = if reverse {
                (t, u, self.keys[e].bind(py), self.nodes[u].bind(py))
            } else {
                (u, t, self.nodes[u].bind(py), self.keys[e].bind(py))
            };
            let back = match &rows.pred {
                Some(p) => &p[b],
                None => &rows.succ[b],
            };
            add_edge_rows(
                py,
                &rows.succ[a],
                back,
                akey,
                bkey,
                Some(self.data[e].bind(py)),
                deepcopy,
            )?;
        }
        Ok(())
    }
}

/// `R.add_edge(i, j)` for every `i` in `left` then `j` in `right` (all
/// already nodes of R), as `full_join` does.
#[pyfunction]
#[pyo3(signature = (succ, pred, left, right))]
pub fn _op_join<'py>(
    py: Python<'py>,
    succ: &Bound<'py, PyDict>,
    pred: Option<Bound<'py, PyDict>>,
    left: Vec<Bound<'py, PyAny>>,
    right: Vec<Bound<'py, PyAny>>,
) -> PyResult<()> {
    let lrows = left
        .iter()
        .map(|k| row_of(succ, k))
        .collect::<PyResult<Vec<_>>>()?;
    let back = pred.as_ref().unwrap_or(succ);
    let rrows = right
        .iter()
        .map(|k| row_of(back, k))
        .collect::<PyResult<Vec<_>>>()?;
    for (i, row_i) in left.iter().zip(&lrows) {
        for (j, row_j) in right.iter().zip(&rrows) {
            add_edge_rows(py, row_i, row_j, i, j, None, None)?;
        }
    }
    Ok(())
}

/// `H.add_edges_from(combinations(preds, 2))` for each row of `pred` (a
/// DiGraph's `_pred`), into the undirected `H` (`moral_graph`).
#[pyfunction]
pub fn _op_pred_combinations<'py>(
    py: Python<'py>,
    adj: &Bound<'py, PyDict>,
    pred: &Bound<'py, PyDict>,
) -> PyResult<()> {
    for (_, preds) in pred.iter() {
        let preds = preds.cast_into::<PyDict>().map_err(|_| changed())?;
        if preds.len() < 2 {
            continue;
        }
        let keys: Vec<Bound<'py, PyAny>> = preds.keys().iter().collect();
        let rows = keys
            .iter()
            .map(|k| row_of(adj, k))
            .collect::<PyResult<Vec<_>>>()?;
        for i in 0..keys.len() {
            for j in i + 1..keys.len() {
                add_edge_rows(py, &rows[i], &rows[j], &keys[i], &keys[j], None, None)?;
            }
        }
    }
    Ok(())
}
