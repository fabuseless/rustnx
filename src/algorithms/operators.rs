//! Graph operators and structure functions (batch 21).
//!
//! The operators (`union`, `compose`, `reverse`, `moral_graph`, ...) spend
//! all their time in NetworkX's `add_nodes_from` / `add_edges_from` loops.
//! Here the result graph's dicts are filled directly, replaying exactly the
//! dict operations those loops perform, in the same order: the same keys
//! inserted in the same order, the same attribute dicts shared or copied.

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyTuple};

use super::pyset::PySet as SetReplica;
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

    /// Whether every adjacency key is interchangeable with the node it
    /// stands for (see `same_key`), so results can use the node objects.
    fn keys_canonical(&self, py: Python<'_>) -> PyResult<bool> {
        self.keys_canonical_impl(py)
    }

    /// `line_graph(G)` into the empty graph given by its dicts.
    fn line_graph_into<'py>(
        &self,
        py: Python<'py>,
        node: Bound<'py, PyDict>,
        succ: Bound<'py, PyDict>,
        pred: Option<Bound<'py, PyDict>>,
    ) -> PyResult<()> {
        self.line_graph_impl(py, node, succ, pred)
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

/// CPython's `hash()` of a tuple, from its items' hashes (`tuplehash`, the
/// same xxHash-based loop in 3.8 to 3.14; checked at runtime by
/// `_b21_tuple_hashes_match`).
pub fn tuple_hash(items: &[i64]) -> i64 {
    const P1: u64 = 11400714785074694791;
    const P2: u64 = 14029467366897019727;
    const P5: u64 = 2870177450012600261;
    let mut acc = P5;
    for &h in items {
        acc = acc.wrapping_add((h as u64).wrapping_mul(P2));
        acc = acc.rotate_left(31);
        acc = acc.wrapping_mul(P1);
    }
    acc = acc.wrapping_add(items.len() as u64 ^ (P5 ^ 3527539));
    if acc == u64::MAX {
        return 1546275796;
    }
    acc as i64
}

/// Whether the dict key `a` and the node `b` it equals are interchangeable
/// in a result: the same object, or the same type all the way down (`1`
/// and `1.0`, or `0.0` and `-0.0`, are equal keys a caller can tell apart).
fn same_key(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<bool> {
    if a.is(b) {
        return Ok(true);
    }
    if !a.get_type().is(b.get_type()) {
        return Ok(false);
    }
    if let (Ok(ta), Ok(tb)) = (a.cast::<PyTuple>(), b.cast::<PyTuple>()) {
        if ta.len() != tb.len() {
            return Ok(false);
        }
        for (x, y) in ta.iter().zip(tb.iter()) {
            if !same_key(&x, &y)? {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    if let (Ok(fa), Ok(fb)) = (a.cast::<PyFloat>(), b.cast::<PyFloat>()) {
        return Ok(fa.value().to_bits() == fb.value().to_bits());
    }
    let simple = a.cast::<pyo3::types::PyInt>().is_ok() || a.cast::<pyo3::types::PyString>().is_ok();
    Ok(simple)
}

/// The result graph's dicts, for builders that track its nodes by
/// position (every node new, so no lookups).
struct Out<'py> {
    py: Python<'py>,
    node: Bound<'py, PyDict>,
    succ: Bound<'py, PyDict>,
    pred: Option<Bound<'py, PyDict>>,
}

type RowPair<'py> = (Bound<'py, PyDict>, Option<Bound<'py, PyDict>>);

impl<'py> Out<'py> {
    /// `add_node` of a node known to be new: its rows (and attribute dict
    /// `data`, or a new empty one).
    fn create(&self, key: &Bound<'py, PyAny>, data: Option<Bound<'py, PyDict>>) -> PyResult<RowPair<'py>> {
        let s = PyDict::new(self.py);
        self.succ.set_item(key, &s)?;
        let p = match &self.pred {
            Some(pred) => {
                let r = PyDict::new(self.py);
                pred.set_item(key, &r)?;
                Some(r)
            }
            None => None,
        };
        let d = data.unwrap_or_else(|| PyDict::new(self.py));
        self.node.set_item(key, d)?;
        Ok((s, p))
    }

    /// `add_node(key)` (no attributes) for a node that may exist.
    fn ensure(&self, key: &Bound<'py, PyAny>) -> PyResult<RowPair<'py>> {
        if self.node.contains(key)? {
            let s = row_of(&self.succ, key)?;
            let p = match &self.pred {
                Some(pred) => Some(row_of(pred, key)?),
                None => None,
            };
            return Ok((s, p));
        }
        self.create(key, None)
    }

    /// `add_edges_from([(a, b, dd)])` with both endpoints present, given
    /// their rows. A fresh `dd` nobody else holds can serve as the new
    /// edge's dict itself (NetworkX copies it into a new dict).
    #[allow(clippy::too_many_arguments)]
    fn edge(
        &self,
        a: (&Bound<'py, PyAny>, &RowPair<'py>),
        b: (&Bound<'py, PyAny>, &RowPair<'py>),
        dd: Option<&Bound<'py, PyDict>>,
        fresh: bool,
    ) -> PyResult<()> {
        let datadict = match a.1 .0.get_item(b.0)? {
            Some(d) => {
                let d = d.cast_into::<PyDict>()?;
                if let Some(dd) = dd {
                    if !dd.is_empty() {
                        d.update(dd.as_mapping())?;
                    }
                }
                d
            }
            None => match dd {
                Some(dd) if fresh => dd.clone(),
                Some(dd) => {
                    let d = PyDict::new(self.py);
                    if !dd.is_empty() {
                        d.update(dd.as_mapping())?;
                    }
                    d
                }
                None => PyDict::new(self.py),
            },
        };
        a.1 .0.set_item(b.0, &datadict)?;
        match &b.1 .1 {
            Some(p) => p.set_item(a.0, &datadict),
            None => b.1 .0.set_item(a.0, &datadict),
        }
    }
}

/// NetworkX's `_dict_product(d1, d2)`: `{k: (d1.get(k), d2.get(k)) for k
/// in set(d1) | set(d2)}`, built with Python's own sets (the key order is
/// their iteration order). Two empty dicts give a new empty dict.
fn dict_product<'py>(py: Python<'py>, d1: &Bound<'py, PyDict>, d2: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    if d1.is_empty() && d2.is_empty() {
        return Ok(out);
    }
    let set = py.get_type::<pyo3::types::PySet>();
    let keys = set.call1((d1,))?.bitor(set.call1((d2,))?)?;
    for k in keys.try_iter()? {
        let k = k?;
        let a = d1.get_item(&k)?.unwrap_or_else(|| py.None().into_bound(py));
        let b = d2.get_item(&k)?.unwrap_or_else(|| py.None().into_bound(py));
        out.set_item(k, PyTuple::new(py, [a, b])?)?;
    }
    Ok(out)
}

fn as_dict<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    obj.cast::<PyDict>().cloned().map_err(|_| changed())
}

impl OpView {
    /// Whether every adjacency key is interchangeable with the node it
    /// stands for (see `same_key`), so results can use the node objects.
    fn keys_canonical_impl(&self, py: Python<'_>) -> PyResult<bool> {
        for (e, key) in self.keys.iter().enumerate() {
            let t = self.targets[e] as usize;
            if !same_key(key.bind(py), self.nodes[t].bind(py))? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// `line_graph(G)` into the empty graph given by its dicts.
    fn line_graph_impl<'py>(
        &self,
        py: Python<'py>,
        node: Bound<'py, PyDict>,
        succ: Bound<'py, PyDict>,
        pred: Option<Bound<'py, PyDict>>,
    ) -> PyResult<()> {
        let out = Out { py, node, succ, pred };
        let n = self.n();
        let m = self.targets.len();
        if self.directed {
            // `_lg_directed`: each edge (u, v) becomes a node, joined to
            // every edge (v, w) out of v. Entry e is edge e's id.
            let mut rows: Vec<Option<RowPair<'py>>> = (0..m).map(|_| None).collect();
            let mut tuples: Vec<Option<Bound<'py, PyTuple>>> = (0..m).map(|_| None).collect();
            let mut source = vec![0u32; m];
            for u in 0..n {
                for e in self.range(u) {
                    source[e] = u as u32;
                }
            }
            let tuple_of = |e: usize, tuples: &mut Vec<Option<Bound<'py, PyTuple>>>| -> PyResult<Bound<'py, PyTuple>> {
                if tuples[e].is_none() {
                    let u = source[e] as usize;
                    let t = self.targets[e] as usize;
                    tuples[e] = Some(PyTuple::new(py, [self.nodes[u].bind(py), self.nodes[t].bind(py)])?);
                }
                Ok(tuples[e].clone().unwrap())
            };
            for u in 0..n {
                for e in self.range(u) {
                    let te = tuple_of(e, &mut tuples)?;
                    if rows[e].is_none() {
                        rows[e] = Some(out.create(te.as_any(), None)?);
                    }
                    let v = self.targets[e] as usize;
                    for f in self.range(v) {
                        let tf = tuple_of(f, &mut tuples)?;
                        if rows[f].is_none() {
                            rows[f] = Some(out.create(tf.as_any(), None)?);
                        }
                        let (ra, rb) = (rows[e].as_ref().unwrap(), rows[f].as_ref().unwrap());
                        out.edge((te.as_any(), ra), (tf.as_any(), rb), None, false)?;
                    }
                }
            }
            return Ok(());
        }
        // `_lg_undirected`: edge ids for the edges (p, q), p <= q by
        // position, in `G.edges()` order.
        let mut id_of = std::collections::HashMap::new();
        let mut ends: Vec<(u32, u32)> = Vec::new();
        for (u, e) in self.edge_entries() {
            let t = self.targets[e];
            id_of.insert((u as u32, t), ends.len() as u32);
            ends.push((u as u32, t));
        }
        let node_hashes = self
            .nodes
            .iter()
            .map(|v| v.bind(py).hash().map(|h| h as i64))
            .collect::<PyResult<Vec<i64>>>()?;
        let edge_hashes: Vec<i64> = ends
            .iter()
            .map(|&(p, q)| tuple_hash(&[node_hashes[p as usize], node_hashes[q as usize]]))
            .collect();
        // The `edges` set, built as `edges.update([...])` does.
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        let mut pair_hashes: Vec<i64> = Vec::new();
        let mut set = SetReplica::default();
        let mut singles: Vec<u32> = Vec::new();
        let mut ids: Vec<u32> = Vec::new();
        for u in 0..n {
            ids.clear();
            for e in self.range(u) {
                let t = self.targets[e];
                let key = if (t as usize) < u { (t, u as u32) } else { (u as u32, t) };
                ids.push(*id_of.get(&key).ok_or_else(changed)?);
            }
            if ids.len() == 1 {
                singles.push(ids[0]);
            }
            for i in 0..ids.len() {
                for j in i + 1..ids.len() {
                    let (a, b) = (ids[i], ids[j]);
                    let (a, b) = if ends[a as usize] <= ends[b as usize] { (a, b) } else { (b, a) };
                    if pairs.len() >= (u32::MAX - 2) as usize {
                        return Err(PyNotImplementedError::new_err("line graph too large"));
                    }
                    pair_hashes.push(tuple_hash(&[edge_hashes[a as usize], edge_hashes[b as usize]]));
                    set.add(pairs.len() as u32, &pair_hashes);
                    pairs.push((a, b));
                }
            }
        }
        let mut rows: Vec<Option<(Bound<'py, PyTuple>, RowPair<'py>)>> = (0..ends.len()).map(|_| None).collect();
        let node_of = |id: u32, rows: &mut Vec<Option<(Bound<'py, PyTuple>, RowPair<'py>)>>| -> PyResult<()> {
            let id = id as usize;
            if rows[id].is_none() {
                let (p, q) = ends[id];
                let t = PyTuple::new(py, [self.nodes[p as usize].bind(py), self.nodes[q as usize].bind(py)])?;
                let r = out.create(t.as_any(), None)?;
                rows[id] = Some((t, r));
            }
            Ok(())
        };
        for id in singles {
            node_of(id, &mut rows)?;
        }
        for k in set.iter() {
            let (a, b) = pairs[k as usize];
            node_of(a, &mut rows)?;
            node_of(b, &mut rows)?;
            let (ta, ra) = rows[a as usize].as_ref().unwrap();
            let (tb, rb) = rows[b as usize].as_ref().unwrap();
            out.edge((ta.as_any(), ra), (tb.as_any(), rb), None, false)?;
        }
        Ok(())
    }
}

/// The graph products (`kind`: "tensor", "cartesian", "lexicographic",
/// "strong") of the views `g` and `h` into the empty graph given by its
/// dicts, adding nodes and edges in NetworkX's order.
#[pyfunction]
#[pyo3(signature = (kind, g, h, node, succ, pred))]
pub fn _op_product<'py>(
    py: Python<'py>,
    kind: &str,
    g: &OpView,
    h: &OpView,
    node: Bound<'py, PyDict>,
    succ: Bound<'py, PyDict>,
    pred: Option<Bound<'py, PyDict>>,
) -> PyResult<()> {
    let out = Out { py, node, succ, pred };
    let (ng, nh) = (g.n(), h.n());
    let directed = out.pred.is_some();
    // `_node_product`: (u, x) for u in G for x in H.
    let mut keys = Vec::with_capacity(ng * nh);
    let mut rows = Vec::with_capacity(ng * nh);
    for u in 0..ng {
        let du = as_dict(g.ndata[u].bind(py))?;
        for x in 0..nh {
            let dx = as_dict(h.ndata[x].bind(py))?;
            let t = PyTuple::new(py, [g.nodes[u].bind(py), h.nodes[x].bind(py)])?.into_any();
            rows.push(out.create(&t, Some(dict_product(py, &du, &dx)?))?);
            keys.push(t);
        }
    }
    let id = |u: usize, x: usize| u * nh + x;
    let add = |a: usize, b: usize, dd: Option<&Bound<'py, PyDict>>, fresh: bool| {
        out.edge((&keys[a], &rows[a]), (&keys[b], &rows[b]), dd, fresh)
    };
    let gedges = g.edge_entries();
    let hedges = h.edge_entries();
    let cross_edges = |undirected: bool| -> PyResult<()> {
        for &(u, e) in &gedges {
            let v = g.targets[e] as usize;
            let c = as_dict(g.data[e].bind(py))?;
            for &(x, f) in &hedges {
                let y = h.targets[f] as usize;
                let d = as_dict(h.data[f].bind(py))?;
                let dd = dict_product(py, &c, &d)?;
                if undirected {
                    add(id(v, x), id(u, y), Some(&dd), true)?;
                } else {
                    add(id(u, x), id(v, y), Some(&dd), true)?;
                }
            }
        }
        Ok(())
    };
    let edges_cross_nodes = || -> PyResult<()> {
        for &(u, e) in &gedges {
            let v = g.targets[e] as usize;
            let d = as_dict(g.data[e].bind(py))?;
            for x in 0..nh {
                add(id(u, x), id(v, x), Some(&d), false)?;
            }
        }
        Ok(())
    };
    let nodes_cross_edges = || -> PyResult<()> {
        for x in 0..ng {
            for &(u, f) in &hedges {
                let v = h.targets[f] as usize;
                let d = as_dict(h.data[f].bind(py))?;
                add(id(x, u), id(x, v), Some(&d), false)?;
            }
        }
        Ok(())
    };
    match kind {
        "tensor" => {
            cross_edges(false)?;
            if !directed {
                cross_edges(true)?;
            }
        }
        "cartesian" => {
            edges_cross_nodes()?;
            nodes_cross_edges()?;
        }
        "lexicographic" => {
            for &(u, e) in &gedges {
                let v = g.targets[e] as usize;
                let d = as_dict(g.data[e].bind(py))?;
                for x in 0..nh {
                    for y in 0..nh {
                        add(id(u, x), id(v, y), Some(&d), false)?;
                    }
                }
            }
            nodes_cross_edges()?;
        }
        "strong" => {
            nodes_cross_edges()?;
            edges_cross_nodes()?;
            cross_edges(false)?;
            if !directed {
                cross_edges(true)?;
            }
        }
        _ => return Err(PyNotImplementedError::new_err("unknown product")),
    }
    Ok(())
}

/// `rooted_product(G, H, root)` into an empty `nx.Graph` (`root` is the
/// root as passed, `r` its position in H).
#[pyfunction]
pub fn _op_rooted_product<'py>(
    py: Python<'py>,
    g: &OpView,
    h: &OpView,
    root: Bound<'py, PyAny>,
    r: usize,
    node: Bound<'py, PyDict>,
    succ: Bound<'py, PyDict>,
) -> PyResult<()> {
    let out = Out { py, node, succ, pred: None };
    let (ng, nh) = (g.n(), h.n());
    if r >= nh {
        return Err(changed());
    }
    let mut keys = Vec::with_capacity(ng * nh);
    let mut rows = Vec::with_capacity(ng * nh);
    for u in 0..ng {
        for x in 0..nh {
            let t = PyTuple::new(py, [g.nodes[u].bind(py), h.nodes[x].bind(py)])?.into_any();
            rows.push(out.create(&t, None)?);
            keys.push(t);
        }
    }
    for (u, e) in g.edge_entries() {
        let v = g.targets[e] as usize;
        let a = PyTuple::new(py, [g.nodes[u].bind(py), &root])?.into_any();
        let b = PyTuple::new(py, [g.nodes[v].bind(py), &root])?.into_any();
        out.edge((&a, &rows[u * nh + r]), (&b, &rows[v * nh + r]), None, false)?;
    }
    let hedges = h.edge_entries();
    for x in 0..ng {
        for &(u, f) in &hedges {
            let v = h.targets[f] as usize;
            let (a, b) = (x * nh + u, x * nh + v);
            out.edge((&keys[a], &rows[a]), (&keys[b], &rows[b]), None, false)?;
        }
    }
    Ok(())
}

/// `corona_product(G, H)` into an empty `nx.Graph`. Its nodes mix G's
/// nodes and `(g, h)` tuples, which may collide, so it looks nodes up.
#[pyfunction]
pub fn _op_corona_product<'py>(
    py: Python<'py>,
    g: &OpView,
    h: &OpView,
    node: Bound<'py, PyDict>,
    succ: Bound<'py, PyDict>,
) -> PyResult<()> {
    let out = Out { py, node, succ, pred: None };
    let mut grows = Vec::with_capacity(g.n());
    for v in &g.nodes {
        grows.push(out.ensure(v.bind(py))?);
    }
    for (u, e) in g.edge_entries() {
        let t = g.targets[e] as usize;
        out.edge(
            (g.nodes[u].bind(py), &grows[u]),
            (g.nodes[t].bind(py), &grows[t]),
            None,
            false,
        )?;
    }
    let hedges = h.edge_entries();
    for (i, gv) in g.nodes.iter().enumerate() {
        let gv = gv.bind(py);
        let mut keys = Vec::with_capacity(h.n());
        let mut rows = Vec::with_capacity(h.n());
        for hv in &h.nodes {
            let t = PyTuple::new(py, [gv, hv.bind(py)])?.into_any();
            rows.push(out.ensure(&t)?);
            keys.push(t);
        }
        for &(u, f) in &hedges {
            let v = h.targets[f] as usize;
            let d = as_dict(h.data[f].bind(py))?;
            // The rows again: an earlier add may have replaced nothing,
            // but a collision can make two keys one node.
            out.edge((&keys[u], &rows[u]), (&keys[v], &rows[v]), Some(&d), false)?;
        }
        let gi = out.ensure(gv)?;
        let _ = i;
        for (k, r) in keys.iter().zip(&rows) {
            out.edge((gv, &gi), (k, r), None, false)?;
        }
    }
    Ok(())
}

/// CPython's tuple hash from item hashes (for the runtime check).
#[pyfunction]
pub fn _tuple_hash(items: Vec<i64>) -> i64 {
    tuple_hash(&items)
}

// --- Structural holes ---------------------------------------------------------------

use super::flow::{self, Fail, Val};
use std::collections::HashMap;

extern "C" {
    fn pow(x: f64, y: f64) -> f64;
}

fn unsupported(_: Fail) -> PyErr {
    PyNotImplementedError::new_err("rustnx can't reproduce this arithmetic")
}

/// Python's `a / b` for ints and floats (`b` nonzero).
fn true_div(a: Val, b: Val) -> PyResult<Val> {
    const EXACT: u64 = 1 << 53;
    match (a, b) {
        (Val::I(x), Val::I(y)) => {
            if x.unsigned_abs() > EXACT || y.unsigned_abs() > EXACT {
                return Err(unsupported(Fail::Unsupported));
            }
            Ok(Val::F(x as f64 / y as f64))
        }
        _ => Ok(Val::F(a.to_f64() / b.to_f64())),
    }
}

/// Python's `x ** 2`: exact for ints, the C library's `pow` for floats.
fn square(x: Val) -> PyResult<Val> {
    match x {
        Val::I(a) => a.checked_mul(a).map(Val::I).ok_or_else(|| unsupported(Fail::Unsupported)),
        Val::F(f) => Ok(Val::F(unsafe { pow(f, std::hint::black_box(2.0)) })),
    }
}

/// `nx.algorithms.structuralholes` on one graph, computing what each node
/// needs on first use: its neighbor set `set(nx.all_neighbors(G, v))` in
/// Python's iteration order, its out-edge weights read from `G._adj`, and
/// the `sum` / `max` normalizations.
pub struct Holes<'a, 'py> {
    g: &'a CoreGraph,
    pred: Option<&'a crate::graph::Csr>,
    nodes: &'a Bound<'py, PyList>,
    adj: Option<&'a Bound<'py, PyDict>>,
    weight: &'a Bound<'py, PyAny>,
    compensated: bool,
    hashes: HashMap<u32, i64>,
    out: HashMap<u32, HashMap<u32, Val>>,
    nbrs: HashMap<u32, std::rc::Rc<Vec<u32>>>,
    sums: HashMap<u32, Val>,
    maxes: HashMap<u32, Val>,
}

impl<'a, 'py> Holes<'a, 'py> {
    pub fn new(
        g: &'a CoreGraph,
        nodes: &'a Bound<'py, PyList>,
        adj: Option<&'a Bound<'py, PyDict>>,
        weight: &'a Bound<'py, PyAny>,
        compensated: bool,
    ) -> PyResult<Self> {
        let pred = if !g.directed {
            None
        } else if g.pred_is_exact {
            g.pred.as_ref()
        } else {
            Some(&g.exact_pred.get().ok_or_else(changed)?.csr)
        };
        if nodes.len() != g.n {
            return Err(changed());
        }
        Ok(Holes {
            g,
            pred,
            nodes,
            adj,
            weight,
            compensated,
            hashes: HashMap::new(),
            out: HashMap::new(),
            nbrs: HashMap::new(),
            sums: HashMap::new(),
            maxes: HashMap::new(),
        })
    }

    fn hash(&mut self, v: u32) -> PyResult<i64> {
        if let Some(&h) = self.hashes.get(&v) {
            return Ok(h);
        }
        let h = self.nodes.get_item(v as usize)?.hash()? as i64;
        self.hashes.insert(v, h);
        Ok(h)
    }

    /// `set(nx.all_neighbors(G, v))` in iteration order: predecessors then
    /// successors for directed graphs.
    fn neighbors(&mut self, v: u32) -> PyResult<std::rc::Rc<Vec<u32>>> {
        if let Some(n) = self.nbrs.get(&v) {
            return Ok(n.clone());
        }
        let mut seq: Vec<u32> = Vec::new();
        if let Some(pred) = self.pred {
            seq.extend_from_slice(pred.neighbors(v as usize));
        }
        seq.extend_from_slice(self.g.succ.neighbors(v as usize));
        // Local ids for the distinct nodes, in first-seen order.
        let mut local: HashMap<u32, u32> = HashMap::new();
        let mut members: Vec<u32> = Vec::new();
        let mut hashes: Vec<i64> = Vec::new();
        let mut set = SetReplica::default();
        for &w in &seq {
            let id = match local.get(&w) {
                Some(&id) => id,
                None => {
                    let id = members.len() as u32;
                    local.insert(w, id);
                    members.push(w);
                    hashes.push(self.hash(w)?);
                    id
                }
            };
            set.add(id, &hashes);
        }
        let order = std::rc::Rc::new(set.iter().map(|id| members[id as usize]).collect::<Vec<u32>>());
        self.nbrs.insert(v, order.clone());
        Ok(order)
    }

    /// `G[v][w].get(weight, 1)` for each successor `w` of `v`.
    fn out_weights(&mut self, v: u32) -> PyResult<&HashMap<u32, Val>> {
        if !self.out.contains_key(&v) {
            let targets = self.g.succ.neighbors(v as usize);
            let mut map = HashMap::with_capacity(targets.len());
            match self.adj {
                None => {
                    for &t in targets {
                        map.insert(t, Val::I(1));
                    }
                }
                Some(adj) => {
                    let row = row_of(adj, &self.nodes.get_item(v as usize)?)?;
                    if row.len() != targets.len() {
                        return Err(changed());
                    }
                    for ((_, data), &t) in row.iter().zip(targets) {
                        let data = data.cast_into::<PyDict>().map_err(|_| changed())?;
                        let val = match data.get_item(self.weight)? {
                            None => Val::I(1),
                            Some(x) => crate::py_val(&x)?,
                        };
                        map.insert(t, val);
                    }
                }
            }
            self.out.insert(v, map);
        }
        Ok(&self.out[&v])
    }

    /// `mutual_weight(G, u, v, weight)`.
    fn mutual(&mut self, u: u32, v: u32) -> PyResult<Val> {
        let a = self.out_weights(u)?.get(&v).copied().unwrap_or(Val::I(0));
        let b = self.out_weights(v)?.get(&u).copied().unwrap_or(Val::I(0));
        a.add(b).map_err(unsupported)
    }

    /// `norm(mutual_weight(G, u, w) for w in set(nx.all_neighbors(G, u)))`
    /// for `norm` `sum` or `max`.
    fn scale(&mut self, u: u32, use_max: bool) -> PyResult<Val> {
        let cache = if use_max { &self.maxes } else { &self.sums };
        if let Some(&s) = cache.get(&u) {
            return Ok(s);
        }
        let nbrs = self.neighbors(u)?;
        let mut vals = Vec::with_capacity(nbrs.len());
        for &w in nbrs.iter() {
            vals.push(self.mutual(u, w)?);
        }
        let s = if use_max {
            let mut it = vals.into_iter();
            // `max()` of nothing raises; NetworkX never gets here empty.
            let first = it.next().ok_or_else(|| unsupported(Fail::Unsupported))?;
            it.fold(first, |m, x| m.max(x))
        } else {
            flow::py_sum(vals, self.compensated).map_err(unsupported)?
        };
        if use_max {
            self.maxes.insert(u, s);
        } else {
            self.sums.insert(u, s);
        }
        Ok(s)
    }

    /// `normalized_mutual_weight(G, u, v, norm=sum or max, weight)`.
    fn normalized(&mut self, u: u32, v: u32, use_max: bool) -> PyResult<Val> {
        let scale = self.scale(u, use_max)?;
        if scale.eq(Val::I(0)) {
            return Ok(Val::I(0));
        }
        true_div(self.mutual(u, v)?, scale)
    }

    /// `local_constraint(G, u, v, weight)`.
    pub fn local_constraint(&mut self, u: u32, v: u32) -> PyResult<Val> {
        let direct = self.normalized(u, v, false)?;
        let nbrs = self.neighbors(u)?;
        let mut terms = Vec::with_capacity(nbrs.len());
        for &w in nbrs.iter() {
            let a = self.normalized(u, w, false)?;
            let b = self.normalized(w, v, false)?;
            terms.push(a.mul(b).map_err(unsupported)?);
        }
        let indirect = flow::py_sum(terms, self.compensated).map_err(unsupported)?;
        square(direct.add(indirect).map_err(unsupported)?)
    }

    /// `constraint(G, [v], weight)[v]` (`None` for NaN).
    pub fn constraint(&mut self, v: u32) -> PyResult<Option<Val>> {
        if self.g.succ.neighbors(v as usize).is_empty() {
            return Ok(None);
        }
        let nbrs = self.neighbors(v)?;
        let mut terms = Vec::with_capacity(nbrs.len());
        for &n in nbrs.iter() {
            terms.push(self.local_constraint(v, n)?);
        }
        flow::py_sum(terms, self.compensated).map(Some).map_err(unsupported)
    }

    /// Whether `effective_size` gives `v` NaN: `len(G[v]) == 0` (NetworkX
    /// 3.4) or `all(u == v for u in G[v])` (3.5+).
    fn isolated(&self, v: u32, by_len: bool) -> bool {
        let row = self.g.succ.neighbors(v as usize);
        if by_len {
            row.is_empty()
        } else {
            row.iter().all(|&u| u == v)
        }
    }

    /// `effective_size(G, [v], weight)[v]` through redundancy (directed or
    /// weighted graphs).
    pub fn effective_size(&mut self, v: u32, by_len: bool) -> PyResult<Option<Val>> {
        if self.isolated(v, by_len) {
            return Ok(None);
        }
        let nbrs = self.neighbors(v)?;
        let mut terms = Vec::with_capacity(nbrs.len());
        for &u in nbrs.iter() {
            let mut parts = Vec::with_capacity(nbrs.len());
            for &w in nbrs.iter() {
                let a = self.normalized(v, w, false)?;
                let b = self.normalized(u, w, true)?;
                parts.push(a.mul(b).map_err(unsupported)?);
            }
            let r = flow::py_sum(parts, self.compensated).map_err(unsupported)?;
            terms.push(Val::I(1).sub(r).map_err(unsupported)?);
        }
        flow::py_sum(terms, self.compensated).map(Some).map_err(unsupported)
    }

    /// `effective_size` of an undirected graph with `weight=None`: from
    /// `E = nx.ego_graph(G, v, center=False, undirected=True)`, `len(E) -
    /// 2 * E.size() / len(E)`.
    pub fn ego_effective_size(&self, v: u32, by_len: bool, mark: &mut [bool]) -> PyResult<Option<Val>> {
        if self.isolated(v, by_len) {
            return Ok(None);
        }
        let row = self.g.succ.neighbors(v as usize);
        let others: Vec<u32> = row.iter().copied().filter(|&u| u != v).collect();
        if others.is_empty() {
            // NetworkX 3.4 divides by zero here.
            return Err(unsupported(Fail::Unsupported));
        }
        for &u in &others {
            mark[u as usize] = true;
        }
        // Twice E's edge count: degrees within E, self-loops counting two.
        let mut twice: u64 = 0;
        for &u in &others {
            for &w in self.g.succ.neighbors(u as usize) {
                if mark[w as usize] {
                    twice += if w == u { 2 } else { 1 };
                }
            }
        }
        for &u in &others {
            mark[u as usize] = false;
        }
        let n = others.len() as f64;
        let size = (twice / 2) as f64;
        Ok(Some(Val::F(n - 2.0 * size / n)))
    }
}

// --- Subgraph copies ----------------------------------------------------------------

/// `G.subgraph(nodes).copy()` into the empty graph given by its dicts:
/// `members` are the subgraph's node positions in the order the subgraph
/// view iterates them. `copy()` re-adds every adjacency entry of the view
/// (each undirected edge from both ends) with `d.copy()`.
#[allow(clippy::too_many_arguments)]
pub fn subgraph_copy<'py>(
    py: Python<'py>,
    g: &CoreGraph,
    nodes: &Bound<'py, PyList>,
    node_dict: &Bound<'py, PyDict>,
    adj: &Bound<'py, PyDict>,
    members: &[u32],
    out_node: Bound<'py, PyDict>,
    succ: Bound<'py, PyDict>,
    pred: Option<Bound<'py, PyDict>>,
) -> PyResult<()> {
    if nodes.len() != g.n || members.iter().any(|&p| p as usize >= g.n) {
        return Err(changed());
    }
    let out = Out { py, node: out_node, succ, pred };
    let mut slot: HashMap<u32, usize> = HashMap::with_capacity(members.len());
    let mut objs = Vec::with_capacity(members.len());
    let mut rows = Vec::with_capacity(members.len());
    for &p in members {
        let v = nodes.get_item(p as usize)?;
        let data = node_dict.get_item(&v)?.ok_or_else(changed)?;
        let d = PyDict::new(py);
        update_from(&d, &data, None)?;
        slot.insert(p, rows.len());
        rows.push(out.create(&v, Some(d))?);
        objs.push(v);
    }
    for (i, &p) in members.iter().enumerate() {
        let row = row_of(adj, &objs[i])?;
        let targets = g.succ.neighbors(p as usize);
        if row.len() != targets.len() {
            return Err(changed());
        }
        for ((key, data), &t) in row.iter().zip(targets) {
            let Some(&j) = slot.get(&t) else { continue };
            if !same_key(&key, &objs[j])? {
                return Err(PyNotImplementedError::new_err("adjacency keys differ from the nodes"));
            }
            let data = data.cast_into::<PyDict>().map_err(|_| changed())?;
            out.edge((&objs[i], &rows[i]), (&key, &rows[j]), Some(&data), false)?;
        }
    }
    Ok(())
}

// --- Broadcasting -------------------------------------------------------------------

/// `_get_max_broadcast_value(G, U, v, values)`: the neighbors of `v` in
/// `U` sorted by value, largest first; the largest `values[u] + i` (ties
/// in the sort don't change it). `None` where NetworkX's `max()` raises.
fn max_broadcast(adj: &crate::graph::Csr, in_u: &[bool], values: &[i64], v: usize) -> Option<i64> {
    let mut vals: Vec<i64> = adj
        .neighbors(v)
        .iter()
        .filter(|&&u| in_u[u as usize])
        .map(|&u| values[u as usize])
        .collect();
    vals.sort_unstable_by(|a, b| b.cmp(a));
    vals.iter().enumerate().map(|(i, &x)| x + i as i64 + 1).max()
}

/// `tree_broadcast_center(G)` for a tree with at least 3 nodes, given each
/// node's `hash()`: `(b_T, [v] + adj[:j])`, the list NetworkX makes its
/// center set from. `None` where NetworkX would raise. `W` is a Python set
/// whose iteration order decides `min(W, key=values.get)` among ties, so
/// it is replayed with CPython's set table.
pub fn tree_broadcast_center(g: &CoreGraph, hashes: &[i64]) -> Option<(i64, Vec<u32>)> {
    let n = g.n;
    let adj = &g.succ;
    let deg: Vec<usize> = (0..n).map(|v| adj.neighbors(v).len()).collect();
    let mut in_u: Vec<bool> = deg.iter().map(|&d| d == 1).collect();
    let mut values: Vec<Option<i64>> = in_u.iter().map(|&u| if u { Some(0) } else { None }).collect();
    let mut alive: Vec<bool> = in_u.iter().map(|&u| !u).collect();
    let mut tdeg: Vec<usize> = (0..n)
        .map(|v| adj.neighbors(v).iter().filter(|&&u| alive[u as usize]).count())
        .collect();
    let mut t_len = alive.iter().filter(|&&a| a).count();
    let mut w_set = SetReplica::default();
    for v in 0..n {
        if alive[v] && tdeg[v] == 1 {
            w_set.add(v as u32, hashes);
            values[v] = Some(deg[v] as i64 - 1);
        }
    }
    let plain = |values: &[Option<i64>]| -> Vec<i64> { values.iter().map(|x| x.unwrap_or(0)).collect() };
    while t_len >= 2 {
        let mut best: Option<(u32, i64)> = None;
        for w in w_set.iter() {
            let val = values[w as usize]?;
            if best.is_none_or(|(_, b)| val < b) {
                best = Some((w, val));
            }
        }
        let (w, _) = best?;
        let w = w as usize;
        let v = *adj.neighbors(w).iter().find(|&&u| alive[u as usize])? as usize;
        in_u[w] = true;
        w_set.discard(w as u32, hashes);
        alive[w] = false;
        t_len -= 1;
        for &u in adj.neighbors(w) {
            if alive[u as usize] {
                tdeg[u as usize] -= 1;
            }
        }
        if tdeg[v] == 1 {
            values[v] = Some(max_broadcast(adj, &in_u, &plain(&values), v)?);
            w_set.add(v as u32, hashes);
        }
    }
    let v = (0..n).find(|&v| alive[v])?;
    let flat = plain(&values);
    let b_t = max_broadcast(adj, &in_u, &flat, v)?;
    // `_get_broadcast_centers`: neighbors sorted by value (stable), largest
    // first, cut where `values[u] + i == b_T`.
    let mut nbrs: Vec<u32> = adj.neighbors(v).to_vec();
    if nbrs.iter().any(|&u| values[u as usize].is_none()) {
        return None; // NetworkX compares None with ints and raises
    }
    nbrs.sort_by(|&a, &b| flat[b as usize].cmp(&flat[a as usize]));
    let j = nbrs
        .iter()
        .enumerate()
        .position(|(i, &u)| flat[u as usize] + i as i64 + 1 == b_t)?;
    let mut centers = vec![v as u32];
    centers.extend_from_slice(&nbrs[..=j]);
    Some((b_t, centers))
}

/// Multi-source BFS distances from `sources` (`-1`: unreached).
pub fn bfs_distances(g: &CoreGraph, sources: &[u32]) -> Vec<i64> {
    let mut dist = vec![-1i64; g.n];
    let mut queue = std::collections::VecDeque::new();
    for &s in sources {
        if dist[s as usize] < 0 {
            dist[s as usize] = 0;
            queue.push_back(s as usize);
        }
    }
    while let Some(v) = queue.pop_front() {
        for &u in g.succ.neighbors(v) {
            if dist[u as usize] < 0 {
                dist[u as usize] = dist[v] + 1;
                queue.push_back(u as usize);
            }
        }
    }
    dist
}

/// Bipartite `degree_centrality`: `{n: d * s_top for n, d in G.degree(top)}`
/// then the same for `bottom` (`index` maps nodes to positions; set
/// members not in G are skipped, as `G.degree(nbunch)` does).
pub fn bipartite_degree_centrality<'py>(
    g: &CoreGraph,
    index: &Bound<'py, PyDict>,
    top: &Bound<'py, pyo3::types::PySet>,
    bottom: &Bound<'py, pyo3::types::PySet>,
    s_top: f64,
    s_bottom: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let py = index.py();
    let out = PyDict::new(py);
    let degree = |v: usize| -> usize {
        let row = g.succ.neighbors(v);
        if g.directed {
            row.len() + g.adj(true).neighbors(v).len()
        } else {
            row.len() + row.iter().filter(|&&u| u as usize == v).count()
        }
    };
    for (set, s) in [(top, s_top), (bottom, s_bottom)] {
        for n in set.iter() {
            let Some(i) = index.get_item(&n)? else { continue };
            let i: usize = i.extract()?;
            if i >= g.n {
                return Err(changed());
            }
            out.set_item(&n, PyFloat::new(py, degree(i) as f64 * s))?;
        }
    }
    Ok(out)
}

// --- Bipartite projections ------------------------------------------------------------

/// `projected_graph(B, nodes)` (`weighted` false) or
/// `weighted_projected_graph(B, nodes, ratio)` (`ratio`: `n_top` when the
/// weights are ratios) into the empty graph given by its dicts. `members`
/// are the positions of `nodes`, in order. The second neighbors of each
/// node are a Python set, added to the result in its iteration order, so
/// it is replayed with CPython's set table.
#[pyfunction]
#[pyo3(signature = (view, members, weighted, ratio, node, succ, pred))]
#[allow(clippy::too_many_arguments)]
pub fn _op_projection<'py>(
    py: Python<'py>,
    view: &OpView,
    members: Vec<u32>,
    weighted: bool,
    ratio: Option<i64>,
    node: Bound<'py, PyDict>,
    succ: Bound<'py, PyDict>,
    pred: Option<Bound<'py, PyDict>>,
) -> PyResult<()> {
    let n = view.n();
    if members.iter().any(|&p| p as usize >= n) {
        return Err(changed());
    }
    let out = Out { py, node, succ, pred };
    let mut rows: Vec<Option<RowPair<'py>>> = (0..n).map(|_| None).collect();
    // `G.add_nodes_from((n, B.nodes[n]) for n in nodes)`.
    for &p in &members {
        let p = p as usize;
        let key = view.nodes[p].bind(py);
        let data = view.ndata[p].bind(py);
        match &rows[p] {
            Some(_) => {
                let d = out.node.get_item(key)?.ok_or_else(changed)?.cast_into::<PyDict>()?;
                update_from(&d, data, None)?;
            }
            None => {
                let d = PyDict::new(py);
                update_from(&d, data, None)?;
                rows[p] = Some(out.create(key, Some(d))?);
            }
        }
    }
    let mut hashes = vec![0i64; n];
    for (h, v) in hashes.iter_mut().zip(&view.nodes) {
        *h = v.bind(py).hash()? as i64;
    }
    let targets = |v: usize| &view.targets[view.range(v)];
    let weight_key = pyo3::types::PyString::new(py, "weight");
    let mut count = vec![0i64; n];
    for &u in &members {
        let u = u as usize;
        let mut second = SetReplica::default();
        if weighted {
            // `{n for nbr in set(B[u]) for n in B[nbr]} - {u}`
            let mut first = SetReplica::default();
            for &w in targets(u) {
                first.add(w, &hashes);
            }
            for w in first.iter() {
                for &x in targets(w as usize) {
                    second.add(x, &hashes);
                    count[x as usize] += 1;
                }
            }
            if (second.len() >> 2) > 1 {
                let mut copy = SetReplica::default();
                copy.merge(&second, &hashes);
                copy.discard(u as u32, &hashes);
                copy.after_difference_update(&hashes);
                second = copy;
            } else {
                let mut fresh = SetReplica::default();
                for x in second.iter() {
                    if x as usize != u {
                        fresh.add(x, &hashes);
                    }
                }
                second = fresh;
            }
        } else {
            // `{v for nbr in B[u] for v in B[nbr] if v != u}`
            for &w in targets(u) {
                for &x in targets(w as usize) {
                    if x as usize != u {
                        second.add(x, &hashes);
                    }
                }
            }
        }
        let order: Vec<u32> = second.iter().collect();
        for &v in &order {
            let v = v as usize;
            if rows[v].is_none() {
                rows[v] = Some(out.create(view.nodes[v].bind(py), None)?);
            }
            let (ru, rv) = (rows[u].as_ref().unwrap(), rows[v].as_ref().unwrap());
            let (ku, kv) = (view.nodes[u].bind(py), view.nodes[v].bind(py));
            if !weighted {
                out.edge((ku, ru), (kv, rv), None, false)?;
                continue;
            }
            let c = count[v];
            let w = match ratio {
                None => c.into_pyobject(py)?.into_any(),
                Some(top) => {
                    let q = true_div(Val::I(c), Val::I(top))?;
                    PyFloat::new(py, q.to_f64()).into_any()
                }
            };
            // `G.add_edge(u, v, weight=w)`
            let datadict = match ru.0.get_item(kv)? {
                Some(d) => d.cast_into::<PyDict>()?,
                None => PyDict::new(py),
            };
            datadict.set_item(&weight_key, w)?;
            ru.0.set_item(kv, &datadict)?;
            match &rv.1 {
                Some(p) => p.set_item(ku, &datadict)?,
                None => rv.0.set_item(ku, &datadict)?,
            }
        }
        if weighted {
            for w in targets(u) {
                for &x in targets(*w as usize) {
                    count[x as usize] = 0;
                }
            }
        }
    }
    Ok(())
}
