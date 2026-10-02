//! Graphs built directly in Rust, without a NetworkX graph in between.
//!
//! The result is laid out exactly as NetworkX would lay out the same edges:
//! nodes in order of first appearance, each node's neighbors in the order
//! its edges were added, duplicate edges merged into the first one (later
//! attribute values win). So every algorithm returns what it would on the
//! equivalent `networkx.Graph`.

use std::collections::HashMap;
use std::sync::OnceLock;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

use crate::graph::{kind, CoreGraph, Csr, EdgeAttr, NativeEdges, Weights};

/// Largest integer that an f64 represents exactly (2^53).
const MAX_EXACT_INT: u64 = 1 << 53;

/// Edges being collected, deduplicated as NetworkX does.
struct Builder {
    directed: bool,
    src: Vec<u32>,
    dst: Vec<u32>,
    lookup: HashMap<(u32, u32), u32>,
    attrs: Vec<EdgeAttr>,
    attr_index: HashMap<String, usize>,
}

impl Builder {
    fn new(directed: bool) -> Self {
        Builder {
            directed,
            src: Vec::new(),
            dst: Vec::new(),
            lookup: HashMap::new(),
            attrs: Vec::new(),
            attr_index: HashMap::new(),
        }
    }

    /// The edge id for `u -> v`, adding the edge if it's new.
    fn edge(&mut self, u: u32, v: u32) -> PyResult<usize> {
        let key = if self.directed || u <= v {
            (u, v)
        } else {
            (v, u)
        };
        if let Some(&e) = self.lookup.get(&key) {
            return Ok(e as usize);
        }
        let e = self.src.len();
        if e >= u32::MAX as usize {
            return Err(PyValueError::new_err("too many edges"));
        }
        self.lookup.insert(key, e as u32);
        self.src.push(u);
        self.dst.push(v);
        for a in &mut self.attrs {
            a.values.push(0.0);
            a.kinds.push(kind::ABSENT);
        }
        Ok(e)
    }

    fn set_attr(&mut self, e: usize, name: &str, value: f64, k: u8) {
        let i = match self.attr_index.get(name) {
            Some(&i) => i,
            None => {
                let m = self.src.len();
                self.attrs.push(EdgeAttr {
                    name: name.to_string(),
                    values: vec![0.0; m],
                    kinds: vec![kind::ABSENT; m],
                });
                self.attr_index
                    .insert(name.to_string(), self.attrs.len() - 1);
                self.attrs.len() - 1
            }
        };
        self.attrs[i].values[e] = value;
        self.attrs[i].kinds[e] = k;
    }
}

fn parse_value(value: &Bound<'_, PyAny>) -> PyResult<(f64, u8)> {
    if value.is_none() {
        return Ok((f64::NAN, kind::NONE));
    }
    if value.is_instance_of::<pyo3::types::PyBool>() {
        return Ok((if value.is_truthy()? { 1.0 } else { 0.0 }, kind::BOOL));
    }
    if value.is_instance_of::<PyInt>() || value.hasattr("__index__")? {
        let i: i64 = value
            .extract()
            .map_err(|_| PyValueError::new_err("integer edge attribute is too large"))?;
        if i.unsigned_abs() > MAX_EXACT_INT {
            return Err(PyValueError::new_err("integer edge attribute is too large"));
        }
        return Ok((i as f64, kind::INT));
    }
    if value.is_instance_of::<PyFloat>() || value.hasattr("dtype")? {
        let x: f64 = value.extract()?;
        if x.is_nan() {
            return Err(PyValueError::new_err(
                "NaN edge attributes are not supported by rustnx graphs; use a networkx graph",
            ));
        }
        return Ok((x, kind::FLOAT));
    }
    Err(PyTypeError::new_err(format!(
        "rustnx graphs store numeric edge attributes only, got {}",
        value.get_type().name()?
    )))
}

/// Stable bucket sort of `(row, target, edge)` entries into CSR rows.
fn rows(n: usize, entries: &[(u32, u32, u32)]) -> (Csr, Vec<u32>) {
    let mut offsets = vec![0usize; n + 1];
    for &(r, _, _) in entries {
        offsets[r as usize + 1] += 1;
    }
    for v in 0..n {
        offsets[v + 1] += offsets[v];
    }
    let mut next = offsets.clone();
    let mut targets = vec![0u32; entries.len()];
    let mut edge_of = vec![0u32; entries.len()];
    for &(r, t, e) in entries {
        let slot = next[r as usize];
        next[r as usize] += 1;
        targets[slot] = t;
        edge_of[slot] = e;
    }
    (Csr { offsets, targets }, edge_of)
}

/// Algorithm weights for one attribute, aligned with CSR entries.
fn entry_weights(attr: &EdgeAttr, edge_of: &[u32]) -> Vec<f64> {
    edge_of
        .iter()
        .map(|&e| match attr.kinds[e as usize] {
            kind::ABSENT => 1.0, // NetworkX's default for a missing weight
            kind::NONE => f64::NAN,
            _ => attr.values[e as usize],
        })
        .collect()
}

fn assemble(n: usize, b: Builder) -> CoreGraph {
    let m = b.src.len();
    let mut succ_entries = Vec::with_capacity(if b.directed { m } else { 2 * m });
    let mut pred_entries = Vec::new();
    for e in 0..m {
        let (u, v) = (b.src[e], b.dst[e]);
        succ_entries.push((u, v, e as u32));
        if b.directed {
            pred_entries.push((v, u, e as u32));
        } else if u != v {
            succ_entries.push((v, u, e as u32));
        }
    }
    // Entries were generated in edge order, so a stable sort by row gives
    // each row its neighbors in insertion order, as in NetworkX's dicts.
    let (succ, succ_edge) = rows(n, &succ_entries);
    let pred = b.directed.then(|| rows(n, &pred_entries));

    let mut weights = HashMap::new();
    for attr in &b.attrs {
        let all_int = attr.kinds.iter().all(|&k| k != kind::FLOAT);
        let has_hidden = attr.kinds.contains(&kind::NONE);
        weights.insert(
            attr.name.clone(),
            Weights {
                succ: entry_weights(attr, &succ_edge),
                pred: pred
                    .as_ref()
                    .map(|(_, edge_of)| entry_weights(attr, edge_of)),
                all_int,
                has_hidden,
            },
        );
    }
    CoreGraph {
        n,
        directed: b.directed,
        succ,
        pred: pred.map(|(csr, _)| csr),
        exact_pred: OnceLock::new(),
        pred_is_exact: true,
        weights,
        native: Some(NativeEdges {
            src: b.src,
            dst: b.dst,
            attrs: b.attrs,
            succ_edge,
        }),
        ones: OnceLock::new(),
    }
}

/// Build from Python edges. Returns the graph, the node list and the
/// node -> position dict.
///
/// Each edge is `(u, v)`, `(u, v, weight)` (stored under `weight`) or
/// `(u, v, {attr: value})`. `nodes`, if given, are added first, in order.
#[pyfunction]
#[pyo3(signature = (edges, nodes, directed, weight))]
pub fn build_native<'py>(
    py: Python<'py>,
    edges: &Bound<'py, PyAny>,
    nodes: Option<&Bound<'py, PyAny>>,
    directed: bool,
    weight: &str,
) -> PyResult<(CoreGraph, Bound<'py, PyList>, Bound<'py, PyDict>)> {
    let node_list = PyList::empty(py);
    let index = PyDict::new(py);
    let add_node = |node: &Bound<'py, PyAny>| -> PyResult<u32> {
        if let Some(i) = index.get_item(node)? {
            return i.extract();
        }
        let i = node_list.len();
        if i >= u32::MAX as usize {
            return Err(PyValueError::new_err("too many nodes"));
        }
        if node.is_none() {
            return Err(PyValueError::new_err("None cannot be a node"));
        }
        index.set_item(node, i)?;
        node_list.append(node)?;
        Ok(i as u32)
    };
    if let Some(nodes) = nodes {
        for node in nodes.try_iter()? {
            add_node(&node?)?;
        }
    }

    let mut b = Builder::new(directed);
    for item in edges.try_iter()? {
        let item = item?;
        let tuple = item
            .cast::<PyTuple>()
            .map(|t| t.to_owned())
            .or_else(|_| PyTuple::new(py, item.try_iter()?.collect::<PyResult<Vec<_>>>()?))?;
        let (u, v, data) = match tuple.len() {
            2 => (tuple.get_item(0)?, tuple.get_item(1)?, None),
            3 => (
                tuple.get_item(0)?,
                tuple.get_item(1)?,
                Some(tuple.get_item(2)?),
            ),
            n => {
                return Err(PyValueError::new_err(format!(
                    "edge tuple must have 2 or 3 items, got {n}"
                )))
            }
        };
        let (u, v) = (add_node(&u)?, add_node(&v)?);
        let e = b.edge(u, v)?;
        match data {
            None => {}
            Some(d) if d.is_instance_of::<PyDict>() => {
                for (key, value) in d.cast::<PyDict>()?.iter() {
                    let name = key.cast::<PyString>().map_err(|_| {
                        PyTypeError::new_err("edge attribute names must be strings")
                    })?;
                    let (x, k) = parse_value(&value)?;
                    b.set_attr(e, &name.to_cow()?, x, k);
                }
            }
            Some(w) => {
                let (x, k) = parse_value(&w)?;
                b.set_attr(e, weight, x, k);
            }
        }
    }
    let n = node_list.len();
    Ok((assemble(n, b), node_list, index))
}

/// Build from raw little-endian arrays: `src`/`dst` int64 node positions and
/// optional float64 `weights`. Nodes are `0..num_nodes`.
#[pyfunction]
#[pyo3(signature = (src, dst, weights, num_nodes, directed, weight))]
pub fn build_native_arrays(
    src: &Bound<'_, PyBytes>,
    dst: &Bound<'_, PyBytes>,
    weights: Option<&Bound<'_, PyBytes>>,
    num_nodes: Option<usize>,
    directed: bool,
    weight: &str,
) -> PyResult<CoreGraph> {
    let ints = |b: &Bound<'_, PyBytes>| -> Vec<i64> {
        b.as_bytes()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&c| i64::from_le_bytes(c))
            .collect()
    };
    let (s, d) = (ints(src), ints(dst));
    if s.len() != d.len() {
        return Err(PyValueError::new_err(
            "src and dst must have the same length",
        ));
    }
    let w: Option<Vec<f64>> = weights.map(|b| {
        b.as_bytes()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&c| f64::from_le_bytes(c))
            .collect()
    });
    if w.as_ref().is_some_and(|w| w.len() != s.len()) {
        return Err(PyValueError::new_err(
            "weights must have one value per edge",
        ));
    }
    let max_node = s.iter().chain(&d).copied().max().map_or(0, |x| x + 1);
    if s.iter().chain(&d).any(|&x| x < 0) {
        return Err(PyValueError::new_err("node ids must be non-negative"));
    }
    let n = match num_nodes {
        Some(n) if (n as i64) < max_node => {
            return Err(PyValueError::new_err(
                "num_nodes is smaller than the largest node id",
            ))
        }
        Some(n) => n,
        None => max_node as usize,
    };
    if n >= u32::MAX as usize {
        return Err(PyValueError::new_err("too many nodes"));
    }
    let mut b = Builder::new(directed);
    for i in 0..s.len() {
        let e = b.edge(s[i] as u32, d[i] as u32)?;
        if let Some(w) = &w {
            if w[i].is_nan() {
                return Err(PyValueError::new_err(
                    "NaN edge weights are not supported by rustnx graphs",
                ));
            }
            b.set_attr(e, weight, w[i], kind::FLOAT);
        }
    }
    Ok(assemble(n, b))
}
