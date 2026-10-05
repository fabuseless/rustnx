//! Compressed sparse row (CSR) storage built from a NetworkX adjacency.
//!
//! Neighbor order within each row is exactly the iteration order of the
//! NetworkX adjacency dict, so traversals visit nodes in the same order as
//! NetworkX does. That is what lets results (and dict ordering) match.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::ops::Range;
use std::sync::OnceLock;

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
    /// Some value is an int (missing values count: they default to 1).
    /// With `!all_int`, NetworkX's distances are ints or floats depending
    /// on the path.
    pub any_int: bool,
    /// Every value is an exact Python `int` or `float` (or missing, which
    /// counts as the int default), so NumPy infers `int64` or `float64` from
    /// a list of them. Bools and NumPy scalars give other dtypes.
    pub plain: bool,
}

/// Predecessor rows in NetworkX's own `G._pred` order, with weights.
pub struct ExactPred {
    pub csr: Csr,
    pub weights: HashMap<String, Vec<f64>>,
}

#[pyclass(frozen, module = "rustnx._core")]
pub struct CoreGraph {
    pub n: usize,
    pub directed: bool,
    pub succ: Csr,
    /// Predecessor rows (directed graphs only), built by transposing `succ`,
    /// so in-edges are ordered by source position rather than by NetworkX's
    /// insertion order. Only weighted directed closeness can tell the
    /// difference (tie order changes the last bits of a float sum); it uses
    /// `exact_pred`, read from `G._pred` on demand.
    pub pred: Option<Csr>,
    pub exact_pred: OnceLock<ExactPred>,
    /// `pred` is already in NetworkX's insertion order (native graphs).
    pub pred_is_exact: bool,
    pub weights: HashMap<String, Weights>,
    /// The original edges of a graph built natively (see `native.rs`).
    pub native: Option<NativeEdges>,
    /// Unit weights, for native graphs asked about an attribute no edge has.
    pub ones: OnceLock<Vec<f64>>,
}

/// Values of one edge attribute, per edge, with the kind of each value.
pub struct EdgeAttr {
    pub name: String,
    pub values: Vec<f64>,
    pub kinds: Vec<u8>,
}

pub mod kind {
    pub const ABSENT: u8 = 0;
    pub const INT: u8 = 1;
    pub const FLOAT: u8 = 2;
    pub const NONE: u8 = 3;
    pub const BOOL: u8 = 4;
}

/// Edges of a natively built graph, in insertion order.
pub struct NativeEdges {
    pub src: Vec<u32>,
    pub dst: Vec<u32>,
    pub attrs: Vec<EdgeAttr>,
    /// Edge id of each `succ` entry.
    pub succ_edge: Vec<u32>,
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
        if self.native.is_some() && !self.weights.contains_key(attr) {
            // NetworkX treats a missing attribute as weight 1.
            let m = self.succ.targets.len();
            return Ok(Some(self.ones.get_or_init(|| vec![1.0; m]).as_slice()));
        }
        let w = self.weights.get(attr).ok_or_else(|| {
            PyNotImplementedError::new_err(format!("edge attribute {attr:?} was not converted"))
        })?;
        Ok(Some(match (&w.pred, reverse && self.directed) {
            (Some(pred), true) => pred.as_slice(),
            _ => w.succ.as_slice(),
        }))
    }

    /// Adjacency and weights for NetworkX's `G.reverse()`, exactly ordered.
    /// Weighted traversal of a directed graph needs `exact_pred` loaded.
    pub fn reverse_exact(&self, attr: Option<&str>) -> PyResult<(&Csr, Option<&[f64]>)> {
        if !self.directed || attr.is_none() || self.pred_is_exact {
            return Ok((self.adj(true), self.weight_slice(attr, true)?));
        }
        let exact = self
            .exact_pred
            .get()
            .ok_or_else(|| PyNotImplementedError::new_err("exact predecessor order not loaded"))?;
        let w = attr.and_then(|a| exact.weights.get(a)).ok_or_else(|| {
            PyNotImplementedError::new_err(format!("edge attribute {attr:?} was not converted"))
        })?;
        Ok((&exact.csr, Some(w.as_slice())))
    }

    /// Like `reverse_exact`, but exact even without weights: path-returning
    /// searches break ties by neighbor order, so the reversed graph's rows
    /// must be in NetworkX's `G._pred` order.
    pub fn reverse_exact_order(&self, attr: Option<&str>) -> PyResult<(&Csr, Option<&[f64]>)> {
        if !self.directed || self.pred_is_exact {
            return Ok((self.adj(true), self.weight_slice(attr, true)?));
        }
        let exact = self
            .exact_pred
            .get()
            .ok_or_else(|| PyNotImplementedError::new_err("exact predecessor order not loaded"))?;
        let w = match attr {
            None => None,
            Some(a) => Some(
                exact
                    .weights
                    .get(a)
                    .ok_or_else(|| {
                        PyNotImplementedError::new_err(format!(
                            "edge attribute {a:?} was not converted"
                        ))
                    })?
                    .as_slice(),
            ),
        };
        Ok((&exact.csr, w))
    }

    pub fn weights_info(&self, attr: Option<&str>) -> (bool, bool) {
        match attr.and_then(|a| self.weights.get(a)) {
            Some(w) => (w.all_int, w.has_hidden),
            None => (true, false),
        }
    }

    /// Whether an attribute mixes int and float values, so that NetworkX's
    /// path lengths are ints or floats depending on the path taken.
    pub fn weights_mixed(&self, attr: Option<&str>) -> bool {
        attr.and_then(|a| self.weights.get(a))
            .is_some_and(|w| w.any_int && !w.all_int)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ValueKind {
    Hidden,
    Int,
    Float,
}

/// One edge attribute value as a number, and what Python type it was.
fn parse_value(value: &Bound<'_, PyAny>) -> PyResult<(f64, ValueKind)> {
    if value.is_none() {
        return Ok((f64::NAN, ValueKind::Hidden));
    }
    if let Ok(f) = value.cast::<PyFloat>() {
        let x = f.value();
        if x.is_nan() {
            return Err(PyNotImplementedError::new_err(
                "NaN edge weights are not supported",
            ));
        }
        return Ok((x, ValueKind::Float));
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
        return Ok((i as f64, ValueKind::Int));
    }
    // NumPy floating scalars that don't subclass float (e.g. float32).
    if value.hasattr("dtype")? {
        if let Ok(x) = value.extract::<f64>() {
            if x.is_nan() {
                return Err(PyNotImplementedError::new_err(
                    "NaN edge weights are not supported",
                ));
            }
            return Ok((x, ValueKind::Float));
        }
    }
    Err(PyNotImplementedError::new_err(format!(
        "unsupported edge weight type: {}",
        value.get_type().name()?
    )))
}

fn record(kind: ValueKind, all_int: &mut bool, hidden: &mut bool, any_int: &mut bool) {
    match kind {
        ValueKind::Hidden => *hidden = true,
        ValueKind::Int => *any_int = true,
        ValueKind::Float => *all_int = false,
    }
}

/// Hasher for object addresses: they are already well spread apart, so a
/// single multiply beats SipHash by a wide margin.
#[derive(Default)]
struct PtrHasher(u64);

impl Hasher for PtrHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, _: &[u8]) {
        unreachable!("PtrHasher only hashes usize")
    }

    fn write_usize(&mut self, x: usize) {
        self.0 = ((x as u64) >> 4).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

/// Resolves a neighbor key from an adjacency dict to its node position.
///
/// A dict lookup per edge dominates conversion time, so two shortcuts come
/// first. Both only ever return an answer the dict lookup would also give.
struct NodeIndex<'a, 'py> {
    index: &'a Bound<'py, PyDict>,
    /// Nodes are exactly the ints `0..n` in order, so an int key is its own
    /// position.
    range_ints: bool,
    n: usize,
    /// Positions of the node objects themselves, by address. Neighbor keys
    /// are usually the very same objects as the node keys.
    by_ptr: HashMap<usize, u32, BuildHasherDefault<PtrHasher>>,
}

impl<'a, 'py> NodeIndex<'a, 'py> {
    fn new(nodes: &Bound<'py, PyList>, index: &'a Bound<'py, PyDict>) -> PyResult<Self> {
        let n = nodes.len();
        let mut range_ints = true;
        for (i, node) in nodes.iter().enumerate() {
            if !node.is_exact_instance_of::<PyInt>() || node.extract::<usize>().ok() != Some(i) {
                range_ints = false;
                break;
            }
        }
        let mut by_ptr = HashMap::default();
        if !range_ints {
            by_ptr.reserve(n);
            for (i, node) in nodes.iter().enumerate() {
                by_ptr.insert(node.as_ptr() as usize, i as u32);
            }
        }
        Ok(NodeIndex {
            index,
            range_ints,
            n,
            by_ptr,
        })
    }

    #[inline]
    fn resolve(&self, key: &Bound<'py, PyAny>) -> PyResult<u32> {
        if self.range_ints {
            if key.is_exact_instance_of::<PyInt>() {
                if let Ok(i) = key.extract::<usize>() {
                    if i < self.n {
                        return Ok(i as u32);
                    }
                }
            }
        } else if let Some(&i) = self.by_ptr.get(&(key.as_ptr() as usize)) {
            return Ok(i);
        }
        let idx = self
            .index
            .get_item(key)?
            .ok_or_else(|| PyValueError::new_err("adjacency refers to a node missing from G"))?;
        idx.extract::<u32>()
    }
}

struct Attr<'py> {
    name: Bound<'py, PyAny>,
    default: Bound<'py, PyAny>,
}

/// Read one adjacency mapping (`G._adj` or `G._pred`) into CSR form.
fn read_adj<'py>(
    nodes: &Bound<'py, PyList>,
    index: &NodeIndex<'_, 'py>,
    adj: &Bound<'py, PyAny>,
    attrs: &[Attr<'py>],
    flags: &mut [(bool, bool, bool)],
    plain: &mut [bool],
    multigraph: bool,
) -> PyResult<(Csr, Vec<Vec<f64>>)> {
    let mut offsets = Vec::with_capacity(nodes.len() + 1);
    let mut targets = Vec::new();
    let mut values: Vec<Vec<f64>> = attrs.iter().map(|_| Vec::new()).collect();
    offsets.push(0);

    let mut push_edge = |targets: &mut Vec<u32>,
                         nbr: &Bound<'py, PyAny>,
                         data: &Bound<'py, PyAny>|
     -> PyResult<()> {
        targets.push(index.resolve(nbr)?);
        let get = |d: &Bound<'py, PyAny>, attr: &Attr<'py>| -> PyResult<Bound<'py, PyAny>> {
            match d.cast::<PyDict>() {
                Ok(d) => Ok(d
                    .get_item(&attr.name)?
                    .unwrap_or_else(|| attr.default.clone())),
                Err(_) => d.call_method1("get", (&attr.name, &attr.default)),
            }
        };
        for (k, attr) in attrs.iter().enumerate() {
            let (x, kind) = if multigraph {
                plain[k] = false;
                // `data` maps edge keys to attribute dicts. NetworkX's
                // multigraph weight is `min(attr.get(weight, 1) for attr in
                // data.values())`: the first smallest value, keeping its type.
                let mut best: Option<(f64, ValueKind)> = None;
                let mut count = 0;
                for item in data.call_method0("values")?.try_iter()? {
                    let (x, kind) = parse_value(&get(&item?, attr)?)?;
                    count += 1;
                    best = match best {
                        Some((bx, bk)) if bk != ValueKind::Hidden && kind != ValueKind::Hidden => {
                            Some(if x < bx { (x, kind) } else { (bx, bk) })
                        }
                        None => Some((x, kind)),
                        Some(_) => {
                            return Err(PyNotImplementedError::new_err(
                                "None weights among parallel edges",
                            ))
                        }
                    };
                }
                if count == 0 {
                    return Err(PyNotImplementedError::new_err("edge without keys"));
                }
                best.expect("at least one edge")
            } else {
                let value = get(data, attr)?;
                if !(value.is_none()
                    || value.is_exact_instance_of::<PyInt>()
                    || value.is_exact_instance_of::<PyFloat>())
                {
                    plain[k] = false;
                }
                parse_value(&value)?
            };
            let (all_int, hidden, any_int) = &mut flags[k];
            record(kind, all_int, hidden, any_int);
            values[k].push(x);
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

/// In-edge rows for a directed graph, ordered by source position, with each
/// weight array permuted to match.
fn transpose(succ: &Csr, n: usize, values: &[Vec<f64>]) -> (Csr, Vec<Vec<f64>>) {
    let m = succ.targets.len();
    let mut offsets = vec![0usize; n + 1];
    for &t in &succ.targets {
        offsets[t as usize + 1] += 1;
    }
    for v in 0..n {
        offsets[v + 1] += offsets[v];
    }
    let mut next = offsets.clone();
    let mut targets = vec![0u32; m];
    let mut order = vec![0usize; m];
    for u in 0..n {
        for e in succ.range(u) {
            let t = succ.targets[e] as usize;
            let slot = next[t];
            next[t] += 1;
            targets[slot] = u as u32;
            order[slot] = e;
        }
    }
    let values = values
        .iter()
        .map(|vals| order.iter().map(|&e| vals[e]).collect())
        .collect();
    (Csr { offsets, targets }, values)
}

fn to_attrs<'py>(weight_attrs: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>) -> Vec<Attr<'py>> {
    weight_attrs
        .into_iter()
        .map(|(name, default)| Attr { name, default })
        .collect()
}

/// Build a `CoreGraph` from NetworkX internals.
///
/// `nodes` is `list(G)`, `index` maps node -> position in `nodes`, `succ` is
/// `G._adj`, and `weight_attrs` is a list of `(attribute name, default)`.
#[pyfunction]
#[pyo3(signature = (nodes, index, succ, directed, weight_attrs, multigraph=false))]
pub fn build_graph<'py>(
    nodes: &Bound<'py, PyList>,
    index: &Bound<'py, PyDict>,
    succ: &Bound<'py, PyAny>,
    directed: bool,
    weight_attrs: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
    multigraph: bool,
) -> PyResult<CoreGraph> {
    if nodes.len() >= u32::MAX as usize {
        return Err(PyNotImplementedError::new_err("graph has too many nodes"));
    }
    let n = nodes.len();
    let attrs = to_attrs(weight_attrs);
    let mut flags = vec![(true, false, false); attrs.len()];
    let mut plain = vec![true; attrs.len()];
    let index = NodeIndex::new(nodes, index)?;

    let (succ_csr, succ_vals) = read_adj(
        nodes, &index, succ, &attrs, &mut flags, &mut plain, multigraph,
    )?;
    let (pred_csr, mut pred_vals) = if directed {
        let (csr, vals) = transpose(&succ_csr, n, &succ_vals);
        (Some(csr), vals.into_iter().map(Some).collect())
    } else {
        (None, attrs.iter().map(|_| None).collect::<Vec<_>>())
    };

    let mut weights = HashMap::new();
    for (k, (attr, succ)) in attrs.iter().zip(succ_vals).enumerate() {
        let (all_int, has_hidden, any_int) = flags[k];
        weights.insert(
            attr.name.extract::<String>()?,
            Weights {
                succ,
                pred: pred_vals[k].take(),
                all_int,
                has_hidden,
                any_int,
                plain: plain[k],
            },
        );
    }

    Ok(CoreGraph {
        n,
        directed,
        succ: succ_csr,
        pred: pred_csr,
        exact_pred: OnceLock::new(),
        pred_is_exact: false,
        weights,
        native: None,
        ones: OnceLock::new(),
    })
}

impl CoreGraph {
    /// Read `G._pred` so that reverse traversals follow NetworkX's exact
    /// in-edge order. The caller must pass the same nodes, index and weight
    /// attributes used to build this graph, from an unchanged graph.
    pub fn load_exact_pred<'py>(
        &self,
        nodes: &Bound<'py, PyList>,
        index: &Bound<'py, PyDict>,
        pred: &Bound<'py, PyAny>,
        weight_attrs: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
        multigraph: bool,
    ) -> PyResult<()> {
        if !self.directed || self.exact_pred.get().is_some() {
            return Ok(());
        }
        if nodes.len() != self.n {
            return Err(PyValueError::new_err("node list does not match this graph"));
        }
        let attrs = to_attrs(weight_attrs);
        let mut flags = vec![(true, false, false); attrs.len()];
        let mut plain = vec![true; attrs.len()];
        let index = NodeIndex::new(nodes, index)?;
        let (csr, vals) = read_adj(
            nodes, &index, pred, &attrs, &mut flags, &mut plain, multigraph,
        )?;
        if csr.targets.len() != self.succ.targets.len() {
            return Err(PyValueError::new_err(
                "predecessors do not match this graph",
            ));
        }
        let mut weights = HashMap::new();
        for (attr, vals) in attrs.iter().zip(vals) {
            weights.insert(attr.name.extract::<String>()?, vals);
        }
        let _ = self.exact_pred.set(ExactPred { csr, weights });
        Ok(())
    }

    pub fn has_exact_pred(&self) -> bool {
        !self.directed || self.pred_is_exact || self.exact_pred.get().is_some()
    }
}
