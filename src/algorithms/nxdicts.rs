//! Filling a NetworkX graph's dicts from Rust the way `add_node` and
//! `add_edge` fill them: `_node`, `_adj` (`_succ`) and, for directed
//! graphs, `_pred`. Shared by the generators, readers, graph builders and
//! operators that return NetworkX graphs.

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use super::generators::Sim;

/// A NetworkX graph's top-level dicts.
pub struct NxDicts<'py> {
    pub node: Bound<'py, PyDict>,
    pub adj: Bound<'py, PyDict>,
    pub pred: Option<Bound<'py, PyDict>>,
}

/// One node's entries: its attribute dict and its adjacency rows.
pub struct NodeRows<'py> {
    pub attrs: Bound<'py, PyDict>,
    pub succ: Bound<'py, PyDict>,
    pub pred: Option<Bound<'py, PyDict>>,
}

impl<'py> NodeRows<'py> {
    /// Where an edge into this node is stored: `_pred[v]`, or `_adj[v]`
    /// for undirected graphs.
    pub fn back(&self) -> &Bound<'py, PyDict> {
        self.pred.as_ref().unwrap_or(&self.succ)
    }
}

impl<'py> NxDicts<'py> {
    pub fn new(
        node: Bound<'py, PyDict>,
        adj: Bound<'py, PyDict>,
        pred: Option<Bound<'py, PyDict>>,
    ) -> Self {
        NxDicts { node, adj, pred }
    }

    /// The dicts of the NetworkX graph `g` (`_pred` if it is directed).
    pub fn of_graph(g: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = g.py();
        let get =
            |name| -> PyResult<Bound<'py, PyDict>> { Ok(g.getattr(name)?.cast_into::<PyDict>()?) };
        let directed = g
            .call_method0(pyo3::intern!(py, "is_directed"))?
            .is_truthy()?;
        Ok(NxDicts {
            node: get(pyo3::intern!(py, "_node"))?,
            adj: get(pyo3::intern!(py, "_adj"))?,
            pred: if directed {
                Some(get(pyo3::intern!(py, "_pred"))?)
            } else {
                None
            },
        })
    }

    pub fn is_empty(&self) -> bool {
        self.node.is_empty() && self.adj.is_empty()
    }

    /// `add_node(key)` for a node not in the graph: empty rows and an
    /// empty attribute dict.
    pub fn create_node(&self, key: &Bound<'py, PyAny>) -> PyResult<NodeRows<'py>> {
        let py = self.node.py();
        let succ = PyDict::new(py);
        self.adj.set_item(key, &succ)?;
        let pred = match &self.pred {
            Some(p) => {
                let row = PyDict::new(py);
                p.set_item(key, &row)?;
                Some(row)
            }
            None => None,
        };
        let attrs = PyDict::new(py);
        self.node.set_item(key, &attrs)?;
        Ok(NodeRows { attrs, succ, pred })
    }

    /// The entries of `key`, adding the node first if it is new (the `if u
    /// not in self._node` part of `add_node` and `add_edge`). Returns
    /// whether it was new.
    pub fn node(&self, key: &Bound<'py, PyAny>) -> PyResult<(NodeRows<'py>, bool)> {
        let Some(attrs) = self.node.get_item(key)? else {
            return Ok((self.create_node(key)?, true));
        };
        let row = |d: &Bound<'py, PyDict>| -> PyResult<Bound<'py, PyDict>> {
            d.get_item(key)?
                .ok_or_else(|| {
                    pyo3::exceptions::PyNotImplementedError::new_err("the graph's dicts disagree")
                })?
                .cast_into::<PyDict>()
                .map_err(Into::into)
        };
        Ok((
            NodeRows {
                attrs: attrs.cast_into::<PyDict>()?,
                succ: row(&self.adj)?,
                pred: match &self.pred {
                    Some(p) => Some(row(p)?),
                    None => None,
                },
            },
            false,
        ))
    }
}

/// `add_edge(u, v)` on a simple graph, given `u`'s row and `v`'s back row
/// (see [`NodeRows::back`]): the edge's data dict, the existing one or a
/// new one, stored under `v` in `u`'s row and under `u` in `v`'s.
pub fn simple_edge<'py>(
    row_u: &Bound<'py, PyDict>,
    back_v: &Bound<'py, PyDict>,
    u: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = match row_u.get_item(v)? {
        Some(d) => d.cast_into::<PyDict>()?,
        None => PyDict::new(row_u.py()),
    };
    row_u.set_item(v, &data)?;
    back_v.set_item(u, &data)?;
    Ok(data)
}

/// Writes `g` into the empty graph `dicts`: one `_node` / `_adj` (and
/// `_pred`) entry per node in node order, each row in insertion order, and
/// one attribute dict per edge (a key dict `{0: {}, 1: {}, ...}` per pair
/// for multigraphs) shared by both rows that hold it. `labels[v]` is node
/// `v`'s key; `fill(v, attrs)` sets its attributes.
pub fn write_sim<'py>(
    dicts: &NxDicts<'py>,
    g: &Sim,
    labels: &[Option<Bound<'py, PyAny>>],
    mut fill: impl FnMut(u32, &Bound<'py, PyDict>) -> PyResult<()>,
) -> PyResult<()> {
    let py = dicts.node.py();
    let label = |v: u32| -> PyResult<&Bound<'py, PyAny>> {
        labels[v as usize]
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("node without a label"))
    };
    let mut slots: Vec<Option<Bound<'py, PyDict>>> = vec![None; g.keys.len()];
    let mut slot = |s: u32| -> PyResult<Bound<'py, PyDict>> {
        let entry = &mut slots[s as usize];
        if let Some(d) = entry.take() {
            return Ok(d); // the pair's second row: nothing else needs it
        }
        let d = PyDict::new(py);
        if g.multigraph {
            for k in 0..g.keys[s as usize] {
                d.set_item(k, PyDict::new(py))?;
            }
        }
        *entry = Some(d.clone());
        Ok(d)
    };
    for &u in &g.order {
        let rows = dicts.create_node(label(u)?)?;
        fill(u, &rows.attrs)?;
        for &(v, s) in &g.succ[u as usize] {
            rows.succ.set_item(label(v)?, slot(s)?)?;
        }
        if let Some(pred) = &rows.pred {
            for &(v, s) in &g.pred[u as usize] {
                pred.set_item(label(v)?, slot(s)?)?;
            }
        }
    }
    Ok(())
}
