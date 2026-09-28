//! Compressed sparse row (CSR) storage built from a NetworkX adjacency.
//!
//! Neighbor order within each row is exactly the iteration order of the
//! NetworkX adjacency dict, so traversals visit nodes in the same order as
//! NetworkX does. That is what lets results (and dict ordering) match.

use std::collections::HashMap;
use std::ops::Range;

use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyInt, PyList};

/// Largest integer that an f64 represents exactly (2^53).
const MAX_EXACT_INT: u64 = 1 << 53;

pub struct Csr {
    pub offsets: Vec<usize>,
    pub targets: Vec<u32>,
}

impl Csr {
    #[inline]
    pub fn range(&self, v: usize) -> Range<usize> {
        self.offsets[v]..self.offsets[v + 1]
    }

    #[inline]
    pub fn neighbors(&self, v: usize) -> &[u32] {
        &self.targets[self.range(v)]
    }
}

/// One edge attribute read as numbers, aligned with the CSR `targets`.
/// Hidden edges (attribute value `None`) are stored as NaN.
pub struct Weights {
    pub succ: Vec<f64>,
    pub pred: Option<Vec<f64>>,
    pub all_int: bool,
    pub has_hidden: bool,
}

#[pyclass(frozen, module = "rustnx._core")]
pub struct CoreGraph {
    pub n: usize,
    pub directed: bool,
    pub succ: Csr,
    /// Predecessor rows; only stored for directed graphs.
    pub pred: Option<Csr>,
    pub weights: HashMap<String, Weights>,
}

impl CoreGraph {
    /// Adjacency to traverse: successors, or predecessors when `reverse`.
    pub fn adj(&self, reverse: bool) -> &Csr {
        match (&self.pred, reverse) {
            (Some(pred), true) => pred,
            _ => &self.succ,
        }
    }

    /// Weights aligned with `adj(reverse)`, or `None` for unit weights.
    pub fn weight_slice(&self, attr: Option<&str>, reverse: bool) -> PyResult<Option<&[f64]>> {
        let Some(attr) = attr else { return Ok(None) };
        let w = self.weights.get(attr).ok_or_else(|| {
            PyNotImplementedError::new_err(format!("edge attribute {attr:?} was not converted"))
        })?;
        Ok(Some(match (&w.pred, reverse && self.directed) {
            (Some(pred), true) => pred.as_slice(),
            _ => w.succ.as_slice(),
        }))
    }

    pub fn weights_info(&self, attr: Option<&str>) -> (bool, bool) {
        match attr.and_then(|a| self.weights.get(a)) {
            Some(w) => (w.all_int, w.has_hidden),
            None => (true, false),
        }
    }
}

fn parse_weight(value: &Bound<'_, PyAny>, all_int: &mut bool, hidden: &mut bool) -> PyResult<f64> {
    if value.is_none() {
        *hidden = true;
        return Ok(f64::NAN);
    }
    if let Ok(f) = value.cast::<PyFloat>() {
        let x = f.value();
        if x.is_nan() {
            return Err(PyNotImplementedError::new_err(
                "NaN edge weights are not supported",
            ));
        }
        *all_int = false;
        return Ok(x);
    }
    // Python ints (and bools) must be exact in f64 for results to match.
    if value.cast::<PyInt>().is_ok() || value.hasattr("__index__")? {
        let i: i64 = value
            .extract()
            .map_err(|_| PyNotImplementedError::new_err("integer edge weight is too large"))?;
        if i.unsigned_abs() > MAX_EXACT_INT {
            return Err(PyNotImplementedError::new_err(
                "integer edge weight is too large",
            ));
        }
        return Ok(i as f64);
    }
    // NumPy floating scalars that don't subclass float (e.g. float32).
    if value.hasattr("dtype")? {
        if let Ok(x) = value.extract::<f64>() {
            if x.is_nan() {
                return Err(PyNotImplementedError::new_err(
                    "NaN edge weights are not supported",
                ));
            }
            *all_int = false;
            return Ok(x);
        }
    }
    Err(PyNotImplementedError::new_err(format!(
        "unsupported edge weight type: {}",
        value.get_type().name()?
    )))
}

struct Attr<'py> {
    name: Bound<'py, PyAny>,
    default: Bound<'py, PyAny>,
}

/// Read one adjacency mapping (`G._adj` or `G._pred`) into CSR form.
fn read_adj<'py>(
    nodes: &Bound<'py, PyList>,
    index: &Bound<'py, PyDict>,
    adj: &Bound<'py, PyAny>,
    attrs: &[Attr<'py>],
    flags: &mut [(bool, bool)],
) -> PyResult<(Csr, Vec<Vec<f64>>)> {
    let mut offsets = Vec::with_capacity(nodes.len() + 1);
    let mut targets = Vec::new();
    let mut values: Vec<Vec<f64>> = attrs.iter().map(|_| Vec::new()).collect();
    offsets.push(0);

    let mut push_edge = |targets: &mut Vec<u32>,
                         nbr: &Bound<'py, PyAny>,
                         data: &Bound<'py, PyAny>|
     -> PyResult<()> {
        let idx = index
            .get_item(nbr)?
            .ok_or_else(|| PyValueError::new_err("adjacency refers to a node missing from G"))?;
        targets.push(idx.extract::<u32>()?);
        for (k, attr) in attrs.iter().enumerate() {
            let value = match data.cast::<PyDict>() {
                Ok(d) => d
                    .get_item(&attr.name)?
                    .unwrap_or_else(|| attr.default.clone()),
                Err(_) => data.call_method1("get", (&attr.name, &attr.default))?,
            };
            let (all_int, hidden) = &mut flags[k];
            values[k].push(parse_weight(&value, all_int, hidden)?);
        }
        Ok(())
    };

    for node in nodes.iter() {
        let nbrs = adj.get_item(&node)?;
        if let Ok(d) = nbrs.cast::<PyDict>() {
            for (nbr, data) in d.iter() {
                push_edge(&mut targets, &nbr, &data)?;
            }
        } else {
            // Graph views (subgraphs, reverse views) expose mapping-like adjacency.
            for item in nbrs.call_method0("items")?.try_iter()? {
                let (nbr, data): (Bound<'py, PyAny>, Bound<'py, PyAny>) = item?.extract()?;
                push_edge(&mut targets, &nbr, &data)?;
            }
        }
        offsets.push(targets.len());
    }
    Ok((Csr { offsets, targets }, values))
}

/// Build a `CoreGraph` from NetworkX internals.
///
/// `nodes` is `list(G)`, `index` maps node -> position in `nodes`, and
/// `weight_attrs` is a list of `(attribute name, default value)` pairs.
#[pyfunction]
#[pyo3(signature = (nodes, index, succ, pred, weight_attrs))]
pub fn build_graph<'py>(
    nodes: &Bound<'py, PyList>,
    index: &Bound<'py, PyDict>,
    succ: &Bound<'py, PyAny>,
    pred: Option<&Bound<'py, PyAny>>,
    weight_attrs: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
) -> PyResult<CoreGraph> {
    if nodes.len() >= u32::MAX as usize {
        return Err(PyNotImplementedError::new_err("graph has too many nodes"));
    }
    let attrs: Vec<Attr> = weight_attrs
        .into_iter()
        .map(|(name, default)| Attr { name, default })
        .collect();
    let mut flags = vec![(true, false); attrs.len()];

    let (succ_csr, succ_vals) = read_adj(nodes, index, succ, &attrs, &mut flags)?;
    let (pred_csr, mut pred_vals) = match pred {
        Some(pred) => {
            let (csr, vals) = read_adj(nodes, index, pred, &attrs, &mut flags)?;
            (Some(csr), vals.into_iter().map(Some).collect())
        }
        None => (None, attrs.iter().map(|_| None).collect::<Vec<_>>()),
    };

    let mut weights = HashMap::new();
    for (k, (attr, succ)) in attrs.iter().zip(succ_vals).enumerate() {
        let (all_int, has_hidden) = flags[k];
        weights.insert(
            attr.name.extract::<String>()?,
            Weights {
                succ,
                pred: pred_vals[k].take(),
                all_int,
                has_hidden,
            },
        );
    }

    Ok(CoreGraph {
        n: nodes.len(),
        directed: pred_csr.is_some(),
        succ: succ_csr,
        pred: pred_csr,
        weights,
    })
}
