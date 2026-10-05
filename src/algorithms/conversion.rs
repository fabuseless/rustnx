//! Matrices and conversions (`to_scipy_sparse_array`, `incidence_matrix`,
//! `to_numpy_array`, `biadjacency_matrix`, `relabel_nodes`, ...).
//!
//! The matrix functions return the coordinates NetworkX builds edge by edge
//! in Python, in the same order, so that SciPy and NumPy turn them into
//! exactly the same arrays. The rest read the source graph's dicts.

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use super::nxdicts::{simple_edge, NodeRows, NxDicts};
use crate::graph::Csr;

/// Coordinates and values for `scipy.sparse.coo_array`.
#[derive(Default)]
pub struct Coo {
    pub row: Vec<i64>,
    pub col: Vec<i64>,
    pub data: Vec<f64>,
}

/// Weight of CSR entry `e`: the attribute value, or NetworkX's default 1.
#[inline]
fn weight_at(w: Option<&[f64]>, e: usize) -> f64 {
    w.map_or(1.0, |w| w[e])
}

/// `to_scipy_sparse_array`'s COO input. `map[v]` is node `v`'s row in the
/// result, or -1 if `nodelist` leaves it out (NetworkX then works on
/// `G.subgraph(nodelist)`, whose edges keep `G`'s order).
///
/// Edges come in `G.edges()` order: rows in node order, and for undirected
/// graphs each edge once, from the endpoint seen first. Undirected graphs
/// then get the mirrored entries, and a `-wt` entry per self-loop (in node
/// order) to undo its double count, as NetworkX appends them.
pub fn adjacency_coo(succ: &Csr, n: usize, directed: bool, map: &[i64], w: Option<&[f64]>) -> Coo {
    let mut row = Vec::new();
    let mut col = Vec::new();
    let mut data = Vec::new();
    let mut loops = Vec::new();
    for u in 0..n {
        let mu = map[u];
        if mu < 0 {
            continue;
        }
        for e in succ.range(u) {
            let v = succ.targets[e] as usize;
            let mv = map[v];
            if mv < 0 || (!directed && v < u) {
                continue;
            }
            row.push(mu);
            col.push(mv);
            data.push(weight_at(w, e));
            if !directed && v == u {
                loops.push((mu, weight_at(w, e)));
            }
        }
    }
    if directed {
        return Coo { row, col, data };
    }
    let m = row.len();
    let mut r = Vec::with_capacity(2 * m + loops.len());
    let mut c = Vec::with_capacity(2 * m + loops.len());
    let mut d = Vec::with_capacity(2 * m + loops.len());
    r.extend_from_slice(&row);
    r.extend_from_slice(&col);
    c.extend_from_slice(&col);
    c.extend_from_slice(&row);
    d.extend_from_slice(&data);
    d.extend_from_slice(&data);
    for (i, x) in loops {
        r.push(i);
        c.push(i);
        d.push(-x);
    }
    Coo {
        row: r,
        col: c,
        data: d,
    }
}

/// `bipartite.biadjacency_matrix`'s COO input: the edges of
/// `G.edges(row_order)` whose far end has a column. `rows` lists, in
/// `row_order` order, `(position, row)` for the nodes of `row_order` in `G`;
/// `col[v]` is node `v`'s column or -1. In undirected graphs an edge back to
/// an earlier row node is skipped, as `G.edges(nbunch)` reports it once.
pub fn biadjacency_coo(
    succ: &Csr,
    n: usize,
    directed: bool,
    rows: &[(u32, i64)],
    col: &[i64],
    w: Option<&[f64]>,
) -> Coo {
    let mut seen = vec![false; n];
    let mut out = Coo::default();
    for &(u, r) in rows {
        let u = u as usize;
        for e in succ.range(u) {
            let v = succ.targets[e] as usize;
            if (!directed && seen[v]) || col[v] < 0 {
                continue;
            }
            out.row.push(r);
            out.col.push(col[v]);
            out.data.push(weight_at(w, e));
        }
        seen[u] = true;
    }
    out
}

/// `to_numpy_array`'s entries: every adjacency entry between kept nodes
/// (both directions of an undirected edge), as `(row, col, weight)`.
pub fn dense_entries(succ: &Csr, n: usize, map: &[i64], w: Option<&[f64]>) -> Coo {
    let mut out = Coo::default();
    for u in 0..n {
        if map[u] < 0 {
            continue;
        }
        for e in succ.range(u) {
            let v = succ.targets[e] as usize;
            if map[v] < 0 {
                continue;
            }
            out.row.push(map[u]);
            out.col.push(map[v]);
            out.data.push(weight_at(w, e));
        }
    }
    out
}

/// CSR arrays `(indptr, indices, data)` and the number of columns.
pub type IncidenceCsr = (Vec<i64>, Vec<i64>, Vec<f64>, usize);

/// `incidence_matrix` as the CSR arrays of the LIL matrix NetworkX fills:
/// one column per edge of `G.edges()`, entries at the endpoints' rows
/// (`map`), `-wt` at the tail when `oriented`. Self-loops leave their
/// column empty, and zero values are not stored (a LIL matrix deletes an
/// entry set to zero). `Err(column)` names the first edge with an endpoint
/// that has no row (NetworkX raises there).
pub fn incidence_csr(
    succ: &Csr,
    n: usize,
    directed: bool,
    rows: usize,
    map: &[i64],
    w: Option<&[f64]>,
    oriented: bool,
) -> Result<IncidenceCsr, (u32, u32)> {
    // Entries (row, column, value), in column order.
    let mut entries: Vec<(i64, i64, f64)> = Vec::new();
    let mut column = 0i64;
    for u in 0..n {
        for e in succ.range(u) {
            let v = succ.targets[e] as usize;
            if !directed && v < u {
                continue;
            }
            let ei = column;
            column += 1;
            if u == v {
                continue;
            }
            let (ui, vi) = (map[u], map[v]);
            if ui < 0 || vi < 0 {
                return Err((u as u32, v as u32));
            }
            let wt = weight_at(w, e);
            let tail = if oriented { -wt } else { wt };
            if tail != 0.0 {
                entries.push((ui, ei, tail));
            }
            if wt != 0.0 {
                entries.push((vi, ei, wt));
            }
        }
    }
    // Stable bucket sort by row keeps each row's columns ascending.
    let mut indptr = vec![0i64; rows + 1];
    for &(r, _, _) in &entries {
        indptr[r as usize + 1] += 1;
    }
    for r in 0..rows {
        indptr[r + 1] += indptr[r];
    }
    let mut next: Vec<i64> = indptr[..rows].to_vec();
    let mut indices = vec![0i64; entries.len()];
    let mut data = vec![0.0f64; entries.len()];
    for &(r, c, x) in &entries {
        let slot = next[r as usize] as usize;
        next[r as usize] += 1;
        indices[slot] = c;
        data[slot] = x;
    }
    Ok((indptr, indices, data, column as usize))
}

/// Whether every edge's data dict has `key` (`is_weighted`). `adj` is the
/// graph's `_adj`, a dict of dicts of dicts.
pub fn all_edges_have(adj: &Bound<'_, PyDict>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
    for (_, nbrs) in adj.iter() {
        let nbrs = plain_dict(&nbrs)?;
        for (_, data) in nbrs.iter() {
            if !plain_dict(&data)?.contains(key)? {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Check that `d`'s keys are exactly `nodes`, in order (the positions the
/// CSR arrays use).
fn check_order(nodes: &Bound<'_, PyList>, d: &Bound<'_, PyDict>) -> PyResult<()> {
    if d.len() != nodes.len() || d.keys().iter().zip(nodes.iter()).any(|(k, n)| !k.is(&n)) {
        return Err(PyNotImplementedError::new_err(
            "the graph changed since it was converted",
        ));
    }
    Ok(())
}

pub fn plain_dict<'a, 'py>(obj: &'a Bound<'py, PyAny>) -> PyResult<&'a Bound<'py, PyDict>> {
    if obj.is_exact_instance_of::<PyDict>() {
        Ok(obj.cast::<PyDict>()?)
    } else {
        Err(PyNotImplementedError::new_err(
            "graph storage is not plain dicts",
        ))
    }
}

/// `get_node_attributes`: `{n: d[name]}` for nodes whose data has `name`,
/// or `{n: d.get(name, default)}` for all when `default` is given.
pub fn node_attributes<'py>(
    node: &Bound<'py, PyDict>,
    name: &Bound<'py, PyAny>,
    default: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(node.py());
    for (n, d) in node.iter() {
        let d = plain_dict(&d)?;
        match (d.get_item(name)?, default) {
            (Some(x), _) => out.set_item(n, x)?,
            (None, Some(default)) => out.set_item(n, default)?,
            (None, None) => {}
        }
    }
    Ok(out)
}

/// `get_edge_attributes` for a simple graph: keys `(u, v)` in `G.edges()`
/// order. `succ` gives the positions of `adj`'s entries (same order), to
/// tell which undirected edges were already reported.
pub fn edge_attributes<'py>(
    nodes: &Bound<'py, PyList>,
    adj: &Bound<'py, PyDict>,
    succ: &Csr,
    directed: bool,
    name: &Bound<'py, PyAny>,
    default: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let py = adj.py();
    let out = PyDict::new(py);
    check_order(nodes, adj)?;
    for (u, (node, nbrs)) in adj.iter().enumerate() {
        let nbrs = plain_dict(&nbrs)?;
        let row = succ.neighbors(u);
        if nbrs.len() != row.len() {
            return Err(PyNotImplementedError::new_err(
                "the graph changed since it was converted",
            ));
        }
        for (&v, (nbr, d)) in row.iter().zip(nbrs.iter()) {
            if !directed && (v as usize) < u {
                continue;
            }
            let d = plain_dict(&d)?;
            let value = match (d.get_item(name)?, default) {
                (Some(x), _) => x,
                (None, Some(default)) => default.clone(),
                (None, None) => continue,
            };
            out.set_item(PyTuple::new(py, [&node, &nbr])?, value)?;
        }
    }
    Ok(out)
}

/// `relabel_nodes(G, mapping)` (copy) for a simple graph, filling the new
/// graph's dicts. `labels[i]` is the new label of node `i`, `None` where
/// the mapping leaves it alone (then the original key objects are kept, as
/// NetworkX does). Rows follow the order in which NetworkX's
/// `add_edges_from(G.edges(data=True))` fills them; each edge gets a fresh
/// copy of its data dict, shared by both directions of an undirected edge.
#[allow(clippy::too_many_arguments)]
pub fn relabel_copy<'py>(
    nodes: &Bound<'py, PyList>,
    adj: &Bound<'py, PyDict>,
    node: &Bound<'py, PyDict>,
    succ: &Csr,
    directed: bool,
    labels: &[Option<Bound<'py, PyAny>>],
    new_node: &Bound<'py, PyDict>,
    new_succ: &Bound<'py, PyDict>,
    new_pred: Option<&Bound<'py, PyDict>>,
) -> PyResult<()> {
    let py = adj.py();
    let n = labels.len();
    check_order(nodes, adj)?;
    check_order(nodes, node)?;
    let label = |i: usize, obj: &Bound<'py, PyAny>| -> Bound<'py, PyAny> {
        labels[i].clone().unwrap_or_else(|| obj.clone())
    };
    // Nodes, with copies of their data, in G's order.
    let mut rows: Vec<Bound<'py, PyDict>> = Vec::with_capacity(n);
    let mut preds: Vec<Bound<'py, PyDict>> = Vec::new();
    for (i, (key, d)) in node.iter().enumerate() {
        let new = label(i, &key);
        new_node.set_item(&new, plain_dict(&d)?.copy()?)?;
        let row = PyDict::new(py);
        new_succ.set_item(&new, &row)?;
        rows.push(row);
        if let Some(pred) = new_pred {
            let row = PyDict::new(py);
            pred.set_item(&new, &row)?;
            preds.push(row);
        }
    }
    if new_node.len() != n {
        return Err(PyNotImplementedError::new_err("labels collide"));
    }
    for (u, (key, nbrs)) in adj.iter().enumerate() {
        let nbrs = plain_dict(&nbrs)?;
        let row = succ.neighbors(u);
        if nbrs.len() != row.len() {
            return Err(PyNotImplementedError::new_err(
                "the graph changed since it was converted",
            ));
        }
        let lu = label(u, &key);
        for (&v, (nbr, d)) in row.iter().zip(nbrs.iter()) {
            let v = v as usize;
            if !directed && v < u {
                continue;
            }
            let lv = label(v, &nbr);
            let data = plain_dict(&d)?.copy()?;
            rows[u].set_item(&lv, &data)?;
            if directed {
                preds[v].set_item(&lu, &data)?;
            } else {
                rows[v].set_item(&lu, &data)?;
            }
        }
    }
    Ok(())
}

/// `to_dict_of_lists`: `{key: [neighbors of key's node kept by `keep`]}` for
/// each `(key, position)`. Neighbors are the keys of the source graph's
/// adjacency rows (`adj`, `G._adj`, which can hold objects equal to but
/// distinct from the node keys, as `G.neighbors` yields them), or `nodes`
/// for native graphs.
pub fn dict_of_lists<'py>(
    nodes: &Bound<'py, PyList>,
    adj: Option<&Bound<'py, PyDict>>,
    succ: &Csr,
    keys: &[(Bound<'py, PyAny>, u32)],
    keep: Option<&[bool]>,
) -> PyResult<Bound<'py, PyDict>> {
    let py = nodes.py();
    let out = PyDict::new(py);
    let rows: Option<Vec<Bound<'py, PyDict>>> = match adj {
        Some(adj) => {
            check_order(nodes, adj)?;
            Some(
                adj.values()
                    .iter()
                    .map(|row| plain_dict(&row).cloned())
                    .collect::<PyResult<_>>()?,
            )
        }
        None => None,
    };
    for (key, u) in keys {
        let u = *u as usize;
        let row = succ.neighbors(u);
        let list = PyList::empty(py);
        match &rows {
            Some(rows) => {
                let d = &rows[u];
                if d.len() != row.len() {
                    return Err(PyNotImplementedError::new_err(
                        "the graph changed since it was converted",
                    ));
                }
                for (&v, nbr) in row.iter().zip(d.keys().iter()) {
                    if keep.is_none_or(|k| k[v as usize]) {
                        list.append(nbr)?;
                    }
                }
            }
            None => {
                for &v in row {
                    if keep.is_none_or(|k| k[v as usize]) {
                        list.append(nodes.get_item(v as usize)?)?;
                    }
                }
            }
        }
        out.set_item(key, list)?;
    }
    Ok(out)
}

/// A new NetworkX `Graph` or `DiGraph` being filled through its dicts
/// (`_node`, `_adj`, and `_pred` for directed graphs), exactly as
/// `add_nodes_from` and `add_edges_from` fill them.
pub struct NxBuilder<'py>(pub NxDicts<'py>);

fn none_node() -> PyErr {
    pyo3::exceptions::PyValueError::new_err("None cannot be a node")
}

/// A node that isn't hashable makes NetworkX take other paths (e.g.
/// `add_nodes_from` reads it as a `(node, attrdict)` pair): fall back.
fn hashable(err: PyErr, py: Python<'_>) -> PyErr {
    if err.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
        PyNotImplementedError::new_err("unhashable node")
    } else {
        err
    }
}

impl<'py> NxBuilder<'py> {
    /// Node `n`'s entries, creating it (empty rows and attributes) if new.
    fn ensure(&self, n: &Bound<'py, PyAny>) -> PyResult<NodeRows<'py>> {
        let py = self.0.node.py();
        if !self.0.node.contains(n).map_err(|e| hashable(e, py))? && n.is_none() {
            return Err(none_node());
        }
        Ok(self.0.node(n)?.0)
    }

    /// `add_nodes_from(nodes, **attr)` (`attr`: `None` for no attributes).
    pub fn add_nodes(
        &self,
        nodes: &Bound<'py, PyAny>,
        attr: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<()> {
        for n in nodes.try_iter()? {
            let rows = self.ensure(&n?)?;
            if let Some(attr) = attr {
                rows.attrs.update(attr.as_mapping())?;
            }
        }
        Ok(())
    }

    /// One edge of `add_edges_from`: the existing data dict of `(u, v)` or a
    /// new one, stored in both directions, then updated by `fill`.
    pub fn add_edge(
        &self,
        u: &Bound<'py, PyAny>,
        v: &Bound<'py, PyAny>,
        fill: impl FnOnce(&Bound<'py, PyDict>) -> PyResult<()>,
    ) -> PyResult<()> {
        let ru = self.ensure(u)?;
        let rv = self.ensure(v)?;
        fill(&simple_edge(&ru.succ, rv.back(), u, v)?)
    }

    /// `add_edges_from(edges)` for 2-tuples and 3-tuples with a dict; any
    /// other item falls back (NetworkX unpacks arbitrary sequences).
    pub fn add_edges(&self, edges: &Bound<'py, PyAny>) -> PyResult<()> {
        for e in edges.try_iter()? {
            let e = e?;
            let t = e
                .cast::<PyTuple>()
                .map_err(|_| PyNotImplementedError::new_err("edges must be tuples"))?;
            match t.len() {
                2 => self.add_edge(&t.get_item(0)?, &t.get_item(1)?, |_| Ok(()))?,
                3 => {
                    let dd = t.get_item(2)?;
                    let dd = plain_dict(&dd)?;
                    self.add_edge(&t.get_item(0)?, &t.get_item(1)?, |d| {
                        d.update(dd.as_mapping())
                    })?
                }
                _ => {
                    return Err(PyNotImplementedError::new_err(
                        "edges must be 2-tuples or 3-tuples",
                    ))
                }
            }
        }
        Ok(())
    }
}

/// `from_dict_of_lists` into a simple graph: the nodes of `d`, then an edge
/// per `(node, nbr)`.
pub fn from_dict_of_lists<'py>(b: &NxBuilder<'py>, d: &Bound<'py, PyDict>) -> PyResult<()> {
    b.add_nodes(d.as_any(), None)?;
    for (u, nbrs) in d.iter() {
        for v in nbrs.try_iter()? {
            b.add_edge(&u, &v?, |_| Ok(()))?;
        }
    }
    Ok(())
}

/// `from_dict_of_dicts` (no `multigraph_input`) into a simple graph.
pub fn from_dict_of_dicts<'py>(b: &NxBuilder<'py>, d: &Bound<'py, PyDict>) -> PyResult<()> {
    b.add_nodes(d.as_any(), None)?;
    for (u, nbrs) in d.iter() {
        for (v, data) in plain_dict(&nbrs)?.iter() {
            let data = plain_dict(&data)?;
            b.add_edge(&u, &v, |dd| dd.update(data.as_mapping()))?;
        }
    }
    Ok(())
}
