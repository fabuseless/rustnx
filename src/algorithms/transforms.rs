//! Batch 24: deterministic generators and graph transformations.
//!
//! The generators replay NetworkX's `add_node` / `add_edge` calls on batch
//! 17's [`Sim`] model (or compute edge lists that the Python side adds with
//! `_add_plain_edges`); the transformations fill the result graph's dicts
//! directly, with the same keys, row orders and shared or copied attribute
//! dicts as NetworkX's code.

use std::collections::{HashMap, HashSet};

use pyo3::exceptions::{PyIndexError, PyNotImplementedError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyString, PyTuple};

use super::flow::{self, Val};
use super::generators::{Attr, Built, Sim};
use super::nxdicts::{simple_edge, NodeRows, NxDicts};
use super::operators::{as_dict, dict_product, OpView, Out};
use super::pyset::PySet;

fn plain(sim: Sim) -> Built {
    Built {
        sim,
        width: 0,
        labels: Vec::new(),
        attr: Attr::None,
    }
}

/// `x ** e % m` (Python's three-argument `pow`, `m > 0`).
fn mod_pow(x: u64, mut e: u64, m: u64) -> u64 {
    let m128 = m as u128;
    let mut base = x as u128 % m128;
    let mut acc = 1u128 % m128;
    while e > 0 {
        if e & 1 == 1 {
            acc = acc * base % m128;
        }
        base = base * base % m128;
        e >>= 1;
    }
    acc as u64
}

/// `nx.circulant_graph(n, range(1, offset + 1))` on the nodes `0..n`:
/// NetworkX 3.6+ adds every `(i, (i - j) % n)` and then every `(i, (i + j)
/// % n)` (`two_pass`); before, the Harary generators add both per `(i, j)`.
fn circulant(g: &mut Sim, n: u32, offset: u32, two_pass: bool) {
    g.add_nodes(0..n);
    if n == 0 {
        return;
    }
    let nn = n as i64;
    let at = |i: u32, d: i64| (i as i64 + d).rem_euclid(nn) as u32;
    if two_pass {
        for sign in [-1i64, 1] {
            for i in 0..n {
                for j in 1..=offset as i64 {
                    g.add_edge(i, at(i, sign * j));
                }
            }
        }
    } else {
        for i in 0..n {
            for j in 1..=offset as i64 {
                g.add_edge(i, at(i, -j));
                g.add_edge(i, at(i, j));
            }
        }
    }
}

/// Batch 24's deterministic generators (see [`super::generators::build`]).
pub fn build(
    kind: &str,
    p: &[i64],
    _lists: &[Vec<i64>],
    directed: bool,
    multigraph: bool,
) -> Option<Built> {
    let int = |i: usize| -> Option<u32> { p.get(i).and_then(|&x| u32::try_from(x).ok()) };
    Some(match kind {
        "margulis" => {
            // Nodes `(x, y)` as ids `x * n + y`, labelled with both ints.
            let n = int(0)?;
            let total = n.checked_mul(n)?;
            let mut g = Sim::new(directed, multigraph, total as usize, 4 * total as usize);
            let mut labels = Vec::with_capacity(2 * total as usize);
            for x in 0..n as i64 {
                for y in 0..n as i64 {
                    labels.push(x);
                    labels.push(y);
                }
            }
            let nn = n as u64;
            for x in 0..nn {
                for y in 0..nn {
                    let u = (x * nn + y) as u32;
                    for (a, b) in [
                        ((x + 2 * y) % nn, y),
                        ((x + 2 * y + 1) % nn, y),
                        (x, (y + 2 * x) % nn),
                        (x, (y + 2 * x + 1) % nn),
                    ] {
                        g.add_edge(u, (a * nn + b) as u32);
                    }
                }
            }
            Built {
                width: 2,
                labels,
                ..plain(g)
            }
        }
        "chordal_cycle" => {
            let n = int(0)?;
            let mut g = Sim::new(directed, multigraph, n as usize, 3 * n as usize);
            let nn = n as u64;
            for x in 0..nn {
                let left = (x + nn - 1) % nn;
                let right = (x + 1) % nn;
                let chord = if x > 0 { mod_pow(x, nn - 2, nn) } else { 0 };
                for y in [left, right, chord] {
                    g.add_edge(x as u32, y as u32);
                }
            }
            plain(g)
        }
        "hkn" => {
            // `hkn_harary_graph(k, n)` for k >= 2 (k == 1 is a path).
            let (k, n, two_pass) = (int(0)?, int(1)?, p.get(2).copied()? != 0);
            let offset = k / 2;
            let mut g = Sim::new(
                directed,
                multigraph,
                n as usize,
                (n as usize).saturating_mul(k as usize).min(1 << 26),
            );
            circulant(&mut g, n, offset, two_pass);
            let half = n / 2;
            if k.is_multiple_of(2) || n.is_multiple_of(2) {
                if k % 2 == 1 {
                    for i in 0..half {
                        g.add_edge(i, i + half);
                    }
                }
            } else {
                for i in 0..=half {
                    g.add_edge(i, (i + half) % n);
                }
            }
            plain(g)
        }
        "hnm" => {
            let (n, m, two_pass) = (int(0)?, p.get(1).copied()?, p.get(2).copied()? != 0);
            if n == 0 || m < 0 {
                return None;
            }
            let d = (2 * m / n as i64) as u32;
            let offset = d / 2;
            let mut g = Sim::new(directed, multigraph, n as usize, (m as usize).min(1 << 26));
            circulant(&mut g, n, offset, two_pass);
            let half = n / 2;
            if n.is_multiple_of(2) || d.is_multiple_of(2) {
                if d % 2 == 1 {
                    for i in 0..half {
                        g.add_edge(i, i + half);
                    }
                }
                let r = (2 * m % n as i64) as u32;
                for i in 0..r / 2 {
                    g.add_edge(i, i + offset + 1);
                }
            } else {
                let extra = m - n as i64 * offset as i64;
                for i in 0..extra.max(0) as u32 {
                    g.add_edge(i, (i + half) % n);
                }
            }
            plain(g)
        }
        _ => return None,
    })
}

// --- nonisomorphic_trees ----------------------------------------------------------

/// `_next_rooted_tree(predecessor, p)` (Beyer-Hedetniemi).
fn next_rooted_tree(pred: &[i64], p: Option<usize>) -> Option<Vec<i64>> {
    let p = match p {
        Some(p) => p,
        None => {
            let mut p = pred.len() - 1;
            while pred[p] == 1 {
                p -= 1;
            }
            p
        }
    };
    if p == 0 {
        return None;
    }
    let mut q = p - 1;
    while pred[q] != pred[p] - 1 {
        q -= 1;
    }
    let mut result = pred.to_vec();
    for i in p..result.len() {
        result[i] = result[i - p + q];
    }
    Some(result)
}

/// `_split_tree(layout)`: the root's left subtree and the rest.
fn split_tree(layout: &[i64]) -> (Vec<i64>, Vec<i64>) {
    let mut one_found = false;
    let mut m = None;
    for (i, &x) in layout.iter().enumerate() {
        if x == 1 {
            if one_found {
                m = Some(i);
                break;
            }
            one_found = true;
        }
    }
    let m = m.unwrap_or(layout.len());
    let left = layout[1..m].iter().map(|&x| x - 1).collect();
    let mut rest = vec![0];
    rest.extend_from_slice(&layout[m..]);
    (left, rest)
}

/// `_next_tree(candidate)` (Wright, Richmond, Odlyzko and McKay).
fn next_tree(candidate: Vec<i64>) -> Option<Vec<i64>> {
    let (left, rest) = split_tree(&candidate);
    let left_height = left.iter().copied().max()?;
    let rest_height = rest.iter().copied().max()?;
    let mut valid = rest_height >= left_height;
    if valid && rest_height == left_height {
        // Left must not have more nodes, nor come after rest
        // lexicographically when they have as many.
        if left.len() > rest.len() || (left.len() == rest.len() && left > rest) {
            valid = false;
        }
    }
    if valid {
        return Some(candidate);
    }
    let p = left.len();
    let mut new = next_rooted_tree(&candidate, Some(p))?;
    if candidate[p] > 2 {
        let (new_left, _) = split_tree(&new);
        let h = new_left.iter().copied().max()?;
        let suffix: Vec<i64> = (1..h + 2).collect();
        // `new[-len(suffix):] = suffix`, which replaces the whole list if
        // the suffix is longer.
        let keep = new.len().saturating_sub(suffix.len());
        new.truncate(keep);
        new.extend(suffix);
    }
    Some(new)
}

/// The state of NetworkX's `nonisomorphic_trees` loop for `order >= 2`.
pub struct NonisoTrees {
    layout: Option<Vec<i64>>,
}

impl NonisoTrees {
    pub fn new(order: usize) -> Self {
        // The path graph rooted at its center.
        let mut layout: Vec<i64> = (0..(order / 2 + 1) as i64).collect();
        layout.extend(1..order.div_ceil(2) as i64);
        NonisoTrees {
            layout: Some(layout),
        }
    }

    /// The next layout NetworkX turns into a graph.
    pub fn next_layout(&mut self) -> Option<Vec<i64>> {
        while let Some(layout) = self.layout.take() {
            if let Some(tree) = next_tree(layout) {
                self.layout = next_rooted_tree(&tree, None);
                return Some(tree);
            }
        }
        None
    }
}

/// `_layout_to_graph(layout)`.
pub fn layout_to_sim(layout: &[i64]) -> Sim {
    let n = layout.len();
    let mut g = Sim::new(false, false, n, n);
    let mut stack: Vec<usize> = Vec::new();
    for (i, &level) in layout.iter().enumerate() {
        if let Some(&top) = stack.last() {
            let mut j = top;
            while layout[j] >= level {
                stack.pop();
                j = *stack.last().expect("the root is never popped");
            }
            g.add_edge(i as u32, j as u32);
        }
        stack.push(i);
    }
    g
}

/// `_rooted_trees(k)` (OEIS A000081) for every `k` whose value fits.
fn rooted_trees_table() -> &'static [u128] {
    static TABLE: std::sync::OnceLock<Vec<u128>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t: Vec<u128> = vec![0, 1];
        'grow: loop {
            let n = t.len();
            let mut value: u128 = 0;
            for j in 1..n {
                for d in 1..n {
                    if j % d == 0 {
                        let term = (d as u128)
                            .checked_mul(t[d])
                            .and_then(|x| x.checked_mul(t[n - j]));
                        match term.and_then(|x| value.checked_add(x)) {
                            Some(v) => value = v,
                            None => break 'grow,
                        }
                    }
                }
            }
            t.push(value / (n as u128 - 1));
        }
        t
    })
}

/// `_unlabeled_trees(n)` (OEIS A000055), or `None` past `u128`.
pub fn unlabeled_trees(n: usize) -> Option<u128> {
    let r = rooted_trees_table();
    if n >= r.len() {
        return None;
    }
    let mut value: u128 = 0;
    for k in 0..=n {
        value = value.checked_add(r[k].checked_mul(r[n - k])?)?;
    }
    if n.is_multiple_of(2) {
        value -= r[n / 2];
    }
    r[n].checked_sub(value / 2)
}

// --- interval_graph and visibility_graph -------------------------------------------

/// The edges `interval_graph` adds, as positions: it pops the last
/// interval and joins it to every earlier one it overlaps, in order.
pub fn interval_edges(lo: &[f64], hi: &[f64]) -> (Vec<u32>, Vec<u32>) {
    let (mut us, mut vs) = (Vec::new(), Vec::new());
    for k in (0..lo.len()).rev() {
        let (min1, max1) = (lo[k], hi[k]);
        for j in 0..k {
            if max1 >= lo[j] && hi[j] >= min1 {
                us.push(k as u32);
                vs.push(j as u32);
            }
        }
    }
    (us, vs)
}

/// The edges `visibility_graph` adds: the path, then every pair `(n1, n2)`
/// in `combinations` order that no value between obstructs, with
/// NetworkX's float arithmetic (the values are exact in a double).
pub fn visibility_edges(t: &[f64]) -> (Vec<u32>, Vec<u32>) {
    let n = t.len();
    let (mut us, mut vs) = (Vec::new(), Vec::new());
    for i in 1..n {
        us.push(i as u32 - 1);
        vs.push(i as u32);
    }
    for n1 in 0..n {
        let t1 = t[n1];
        for n2 in n1 + 1..n {
            let t2 = t[n2];
            let slope = (t2 - t1) / ((n2 - n1) as f64);
            let offset = t2 - slope * n2 as f64;
            let obstructed = (n1 + 1..n2).any(|k| t[k] >= slope * k as f64 + offset);
            if !obstructed {
                us.push(n1 as u32);
                vs.push(n2 as u32);
            }
        }
    }
    (us, vs)
}

// --- prefix_tree --------------------------------------------------------------------

fn decline(msg: &'static str) -> PyErr {
    PyNotImplementedError::new_err(msg)
}

/// A path's items: lists and tuples item by item, strings by character.
fn path_items<'py>(path: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if let Ok(l) = path.cast::<PyList>() {
        return Ok(l.iter().collect());
    }
    if let Ok(t) = path.cast::<PyTuple>() {
        return Ok(t.iter().collect());
    }
    if path.is_exact_instance_of::<PyString>() {
        return path.try_iter()?.collect();
    }
    Err(decline("rustnx needs paths as lists, tuples or strings"))
}

/// `prefix_tree(paths)` / `prefix_tree_recursive(paths)` into the empty
/// DiGraph given by its dicts: both number the trie's nodes in preorder
/// (each node's children in order of first appearance), add `(v, NIL)`
/// when a path ends at `v` just after creating `v`, and store the first
/// item object seen at each trie node as its `source`. Returns the
/// longest path's length. Items are matched through one dict, which groups
/// them exactly as NetworkX's per-node `defaultdict`s do.
pub fn prefix_tree<'py>(
    py: Python<'py>,
    paths: &Bound<'py, PyAny>,
    node: &Bound<'py, PyDict>,
    succ: &Bound<'py, PyDict>,
    pred: &Bound<'py, PyDict>,
) -> PyResult<usize> {
    let paths: Vec<Bound<'py, PyAny>> = if let Ok(l) = paths.cast::<PyList>() {
        l.iter().collect()
    } else if let Ok(t) = paths.cast::<PyTuple>() {
        t.iter().collect()
    } else {
        return Err(decline("rustnx needs paths in a list or tuple"));
    };
    let ids = PyDict::new(py);
    let mut children: Vec<Vec<u32>> = vec![Vec::new()];
    let mut source: Vec<Option<Bound<'py, PyAny>>> = vec![None];
    let mut nil = vec![false];
    let mut child_of: HashMap<(u32, u32), u32> = HashMap::new();
    let mut depth = 0;
    for path in &paths {
        let items = path_items(path)?;
        depth = depth.max(items.len());
        let mut cur = 0u32;
        for item in items {
            let id = match ids.get_item(&item) {
                Ok(Some(id)) => id.extract::<u32>()?,
                Ok(None) => {
                    let id = ids.len() as u32;
                    ids.set_item(&item, id)
                        .map_err(|_| decline("NetworkX raises for this item"))?;
                    id
                }
                Err(_) => return Err(decline("NetworkX raises for this item")),
            };
            cur = match child_of.get(&(cur, id)) {
                Some(&c) => c,
                None => {
                    let c = children.len() as u32;
                    children.push(Vec::new());
                    source.push(Some(item));
                    nil.push(false);
                    children[cur as usize].push(c);
                    child_of.insert((cur, id), c);
                    c
                }
            };
        }
        nil[cur as usize] = true;
    }
    let dicts = NxDicts::new(node.clone(), succ.clone(), Some(pred.clone()));
    let source_key = pyo3::intern!(py, "source");
    let root_key = 0i64.into_pyobject(py)?.into_any();
    let nil_key = (-1i64).into_pyobject(py)?.into_any();
    let root = dicts.create_node(&root_key)?;
    root.attrs.set_item(source_key, py.None())?;
    let nil_rows = dicts.create_node(&nil_key)?;
    nil_rows.attrs.set_item(source_key, "NIL")?;
    let edge = |a: (&Bound<'py, PyAny>, &NodeRows<'py>),
                b: (&Bound<'py, PyAny>, &NodeRows<'py>)| {
        simple_edge(&a.1.succ, b.1.back(), a.0, b.0).map(|_| ())
    };
    if nil[0] {
        edge((&root_key, &root), (&nil_key, &nil_rows))?;
    }
    let mut keys: Vec<Option<(Bound<'py, PyAny>, NodeRows<'py>)>> =
        (0..children.len()).map(|_| None).collect();
    keys[0] = Some((root_key.clone(), root));
    let mut next_name = 1i64;
    let mut stack: Vec<(u32, usize)> = vec![(0, 0)];
    while let Some(top) = stack.last_mut() {
        let (parent, idx) = (top.0 as usize, top.1);
        let Some(&child) = children[parent].get(idx) else {
            stack.pop();
            continue;
        };
        top.1 += 1;
        let key = next_name.into_pyobject(py)?.into_any();
        next_name += 1;
        let rows = dicts.create_node(&key)?;
        rows.attrs
            .set_item(source_key, source[child as usize].as_ref().unwrap())?;
        {
            let (pk, prow) = keys[parent].as_ref().unwrap();
            edge((pk, prow), (&key, &rows))?;
        }
        if nil[child as usize] {
            edge((&key, &rows), (&nil_key, &nil_rows))?;
        }
        keys[child as usize] = Some((key, rows));
        stack.push((child, 0));
    }
    Ok(depth)
}

// --- mycielskian --------------------------------------------------------------------

/// `mycielskian`'s loop on `M = convert_node_labels_to_integers(G)`, given
/// as its dicts (nodes `0..n` in order): adds the new nodes and edges to
/// them in NetworkX's order, each new edge with a new dict.
pub fn mycielskian<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyDict>,
    adj: &Bound<'py, PyDict>,
    iterations: usize,
) -> PyResult<()> {
    let n = node.len();
    if adj.len() != n {
        return Err(decline("the graph's dicts disagree"));
    }
    let mut rows: Vec<Vec<u32>> = Vec::with_capacity(n);
    let mut py_rows = Vec::with_capacity(n);
    for (i, (k, row)) in adj.iter().enumerate() {
        if k.extract::<usize>().ok() != Some(i) {
            return Err(decline("the nodes are not 0..n in order"));
        }
        let row = row.cast_into::<PyDict>()?;
        let mut r = Vec::with_capacity(row.len());
        for key in row.keys() {
            match key.extract::<usize>() {
                Ok(v) if v < n => r.push(v as u32),
                _ => return Err(decline("the nodes are not 0..n")),
            }
        }
        rows.push(r);
        py_rows.push(row);
    }
    let order: Vec<u32> = (0..n as u32).collect();
    let mut g = Sim::from_rows(&rows, None, &order, false);
    let first_slot = g.keys.len();
    let initial: Vec<usize> = rows.iter().map(Vec::len).collect();
    for _ in 0..iterations {
        let m = g.len() as u32;
        let end = m.checked_mul(2).and_then(|x| x.checked_add(1));
        if end.is_none_or(|e| e > (1 << 30)) {
            return Err(decline("the graph would be too large"));
        }
        g.add_nodes(m..2 * m);
        let old = g.edges();
        for &(u, v) in &old {
            g.add_edge(u, v + m);
        }
        for &(u, v) in &old {
            g.add_edge(u + m, v);
        }
        g.add_node(2 * m);
        for u in 0..m {
            g.add_edge(u + m, 2 * m);
        }
    }
    let dicts = NxDicts::new(node.clone(), adj.clone(), None);
    let total = g.len();
    let mut keys = Vec::with_capacity(total);
    for v in 0..total {
        let key = v.into_pyobject(py)?.into_any();
        if v >= n {
            py_rows.push(dicts.create_node(&key)?.succ);
        }
        keys.push(key);
    }
    let mut slots: Vec<Option<Bound<'py, PyDict>>> = vec![None; g.keys.len() - first_slot];
    for u in 0..total {
        let start = if u < n { initial[u] } else { 0 };
        for &(v, s) in &g.succ[u][start..] {
            let s = s as usize - first_slot;
            let d = match &slots[s] {
                Some(d) => d.clone(),
                None => {
                    let d = PyDict::new(py);
                    slots[s] = Some(d.clone());
                    d
                }
            };
            py_rows[u].set_item(&keys[v as usize], d)?;
        }
    }
    Ok(())
}

// --- stochastic_graph ---------------------------------------------------------------

/// Python's `a / b` for ints and floats (`b` nonzero).
fn true_div(a: Val, b: Val) -> PyResult<Val> {
    const EXACT: u64 = 1 << 53;
    match (a, b) {
        (Val::I(x), Val::I(y)) => {
            if x.unsigned_abs() > EXACT || y.unsigned_abs() > EXACT {
                return Err(decline("integers too large for exact division"));
            }
            Ok(Val::F(x as f64 / y as f64))
        }
        _ => Ok(Val::F(a.to_f64() / b.to_f64())),
    }
}

fn val_of(x: &Bound<'_, PyAny>) -> PyResult<Val> {
    if x.is_exact_instance_of::<pyo3::types::PyInt>() {
        return x
            .extract::<i64>()
            .map(Val::I)
            .map_err(|_| decline("integer too large"));
    }
    if let Ok(f) = x.cast::<pyo3::types::PyFloat>() {
        if x.is_exact_instance_of::<pyo3::types::PyFloat>() {
            return Ok(Val::F(f.value()));
        }
    }
    Err(decline("rustnx needs int or float weights here"))
}

fn val_obj<'py>(py: Python<'py>, v: Val) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Val::I(i) => i.into_pyobject(py)?.into_any(),
        Val::F(f) => pyo3::types::PyFloat::new(py, f).into_any(),
    })
}

/// `stochastic_graph(G)` for a DiGraph: fills the DiGraph `H` (holding G's
/// nodes, no edges) as `DiGraph(G)` does (each edge dict copied, `_pred`
/// rows in edge order), with each edge's `weight` set to its share of the
/// source's weighted out-degree (a `sum()` with `compensated` semantics).
/// Every value is checked before anything is written.
#[allow(clippy::too_many_arguments)]
pub fn stochastic<'py>(
    py: Python<'py>,
    g_node: &Bound<'py, PyDict>,
    g_succ: &Bound<'py, PyDict>,
    weight: &Bound<'py, PyAny>,
    h_node: &Bound<'py, PyDict>,
    h_succ: &Bound<'py, PyDict>,
    h_pred: &Bound<'py, PyDict>,
    compensated: bool,
) -> PyResult<()> {
    let mut values: Vec<Val> = Vec::new();
    let mut shares: Vec<Val> = Vec::new();
    for (_, row) in g_succ.iter() {
        let row = row.cast_into::<PyDict>()?;
        let start = values.len();
        for (_, d) in row.iter() {
            let d = d.cast_into::<PyDict>()?;
            values.push(match d.get_item(weight)? {
                Some(x) => val_of(&x)?,
                None => Val::I(1),
            });
        }
        let degree = flow::py_sum(values[start..].iter().copied(), compensated)
            .map_err(|_| decline("rustnx can't reproduce this sum"))?;
        let zero = match degree {
            Val::I(x) => x == 0,
            Val::F(x) => x == 0.0,
        };
        for &x in &values[start..] {
            shares.push(if zero {
                Val::I(0)
            } else {
                true_div(x, degree)?
            });
        }
    }
    let mut e = 0;
    for (u, row) in g_succ.iter() {
        let row = row.cast_into::<PyDict>()?;
        let hrow = h_succ
            .get_item(&u)?
            .ok_or_else(|| decline("the graphs disagree"))?
            .cast_into::<PyDict>()?;
        for (v, d) in row.iter() {
            let d = d.cast_into::<PyDict>()?;
            let new = PyDict::new(py);
            if !d.is_empty() {
                new.update(d.as_mapping())?;
            }
            new.set_item(weight, val_obj(py, shares[e])?)?;
            e += 1;
            hrow.set_item(&v, &new)?;
            h_pred
                .get_item(&v)?
                .ok_or_else(|| decline("the graphs disagree"))?
                .set_item(&u, &new)?;
        }
    }
    for (n, dd) in g_node.iter() {
        let dd = dd.cast_into::<PyDict>()?;
        if !dd.is_empty() {
            h_node
                .get_item(&n)?
                .ok_or_else(|| decline("the graphs disagree"))?
                .cast_into::<PyDict>()?
                .update(dd.as_mapping())?;
        }
    }
    Ok(())
}

// --- inverse_line_graph -------------------------------------------------------------

/// Why `inverse_line_graph` stops: NetworkX's error, or a case rustnx
/// leaves to it.
pub enum LineError {
    Message(&'static str),
    PopEmpty,
    Fallback,
}

struct Lg<'a> {
    g: &'a crate::graph::CoreGraph,
    edges: HashSet<u64>,
    hashes: Option<&'a [i64]>,
}

#[inline]
fn key(u: u32, v: u32) -> u64 {
    let (a, b) = if u <= v { (u, v) } else { (v, u) };
    ((a as u64) << 32) | b as u64
}

impl Lg<'_> {
    fn adjacent(&self, u: u32, v: u32) -> bool {
        self.edges.contains(&key(u, v))
    }

    /// `_triangles(G, (u, v))`.
    fn triangles(&self, u: u32, v: u32) -> Vec<(u32, u32, u32)> {
        self.g
            .succ
            .neighbors(u as usize)
            .iter()
            .filter(|&&x| self.adjacent(x, v))
            .map(|&x| (u, v, x))
            .collect()
    }

    /// `_odd_triangle(G, T)`.
    fn odd_triangle(&self, t: (u32, u32, u32)) -> bool {
        let tri = [t.0, t.1, t.2];
        let mut counts: HashMap<u32, u32> = HashMap::new();
        for &a in &tri {
            for &v in self.g.succ.neighbors(a as usize) {
                if !tri.contains(&v) {
                    *counts.entry(v).or_insert(0) += 1;
                }
            }
        }
        counts.values().any(|&c| c == 1 || c == 3)
    }

    /// `_select_starting_cell(G, starting_edge)`.
    fn starting_cell(&self, e: (u32, u32), depth: usize) -> Result<Vec<u32>, LineError> {
        if depth > 200 {
            // NetworkX would recurse further (or hit the recursion limit).
            return Err(LineError::Fallback);
        }
        let tris = self.triangles(e.0, e.1);
        let r = tris.len();
        if r == 0 {
            return Ok(vec![e.0, e.1]);
        }
        if r == 1 {
            let (a, b, c) = tris[0];
            let ac = self.triangles(a, c).len();
            let bc = self.triangles(b, c).len();
            return if ac == 1 {
                if bc == 1 {
                    Ok(vec![a, b, c])
                } else {
                    self.starting_cell((b, c), depth + 1)
                }
            } else {
                self.starting_cell((a, c), depth + 1)
            };
        }
        let odd: Vec<(u32, u32, u32)> = tris
            .iter()
            .copied()
            .filter(|&t| self.odd_triangle(t))
            .collect();
        let s = odd.len();
        if r == 2 && s == 0 {
            let t = tris[r - 1];
            return Ok(vec![t.0, t.1, t.2]);
        }
        if r - 1 <= s && s <= r {
            let hashes = self.hashes.ok_or(LineError::Fallback)?;
            let mut set = PySet::default();
            for t in &odd {
                for x in [t.0, t.1, t.2] {
                    set.add(x, hashes);
                }
            }
            let nodes: Vec<u32> = set.iter().collect();
            for &u in &nodes {
                for &v in &nodes {
                    if u != v && !self.adjacent(u, v) {
                        return Err(LineError::Message(
                            "G is not a line graph (odd triangles do not form complete subgraph)",
                        ));
                    }
                }
            }
            return Ok(nodes);
        }
        Err(LineError::Message(
            "G is not a line graph (incorrect number of odd triangles around starting edge)",
        ))
    }
}

/// `_select_starting_cell` and `_find_partition` for `inverse_line_graph`
/// on a graph with edges and no self-loops: the partition's cells as node
/// positions. `hashes` (each node's `hash()`) replay the one set NetworkX
/// iterates; without them that case falls back.
pub fn inverse_line_partition(
    g: &crate::graph::CoreGraph,
    hashes: Option<&[i64]>,
) -> Result<Vec<Vec<u32>>, LineError> {
    let n = g.n;
    let mut edges = HashSet::new();
    for u in 0..n {
        for &v in g.succ.neighbors(u) {
            edges.insert(key(u as u32, v));
        }
    }
    let lg = Lg { g, edges, hashes };
    let first = (0..n)
        .find_map(|u| g.succ.neighbors(u).first().map(|&v| (u as u32, v)))
        .ok_or(LineError::Fallback)?;
    let cell = lg.starting_cell(first, 0)?;
    // `G.copy()`: the rows are rebuilt by re-adding every adjacency entry.
    let mut part = Sim::new(false, false, n, lg.edges.len());
    part.add_nodes(0..n as u32);
    for u in 0..n {
        for &v in g.succ.neighbors(u) {
            part.add_edge(u as u32, v);
        }
    }
    let mut remaining = lg.edges.len();
    let remove_clique = |part: &mut Sim, cell: &[u32], remaining: &mut usize| {
        for i in 0..cell.len() {
            for j in i + 1..cell.len() {
                let (a, b) = (cell[i], cell[j]);
                if part.succ[a as usize].iter().any(|&(x, _)| x == b) {
                    part.remove_edge(a, b);
                    *remaining -= 1;
                }
            }
        }
    };
    remove_clique(&mut part, &cell, &mut remaining);
    let mut stack = cell.clone();
    let mut cells = vec![cell];
    while remaining > 0 {
        let u = stack.pop().ok_or(LineError::PopEmpty)?;
        if part.succ[u as usize].is_empty() {
            continue;
        }
        let mut new_cell = vec![u];
        new_cell.extend(part.succ[u as usize].iter().map(|&(x, _)| x));
        for &a in &new_cell {
            for &b in &new_cell {
                if a != b && !part.succ[a as usize].iter().any(|&(x, _)| x == b) {
                    return Err(LineError::Message(
                        "G is not a line graph (partition cell not a complete subgraph)",
                    ));
                }
            }
        }
        remove_clique(&mut part, &new_cell, &mut remaining);
        stack.extend_from_slice(&new_cell);
        cells.push(new_cell);
    }
    Ok(cells)
}

/// The edges of `inverse_line_graph`'s result: pairs of positions in its
/// node list (the cells, then the single nodes) that share a node of G, in
/// `combinations` order. `cells_of[v]` are the result nodes holding `v`.
pub fn inverse_line_edges(cells_of: &[Vec<u32>]) -> (Vec<u32>, Vec<u32>) {
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for c in cells_of {
        for i in 0..c.len() {
            for j in i + 1..c.len() {
                let (a, b) = (c[i].min(c[j]), c[i].max(c[j]));
                pairs.push((a, b));
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs.into_iter().unzip()
}

pub fn line_error(py: Python<'_>, e: LineError) -> PyResult<PyErr> {
    Ok(match e {
        LineError::Fallback => decline("rustnx leaves this graph to NetworkX"),
        LineError::PopEmpty => PyIndexError::new_err("pop from empty list"),
        LineError::Message(m) => {
            let cls = py.import("networkx")?.getattr("NetworkXError")?;
            PyErr::from_value(cls.call1((m,))?)
        }
    })
}

// --- modular_product ----------------------------------------------------------------

/// The complement's edges in `nx.complement(G).edges()` order, as
/// positions (`complement` adds `(n, n2)` for each node pair not adjacent,
/// in node order, then `edges()` walks the rows it built).
fn complement_edges(view: &OpView) -> Vec<(u32, u32)> {
    let n = view.n();
    let mut g = Sim::new(false, false, n, 0);
    g.add_nodes(0..n as u32);
    let mut mark = vec![u32::MAX; n];
    for u in 0..n {
        for e in view.range(u) {
            mark[view.targets[e] as usize] = u as u32;
        }
        for (v, &m) in mark.iter().enumerate() {
            if v != u && m != u as u32 {
                g.add_edge(u as u32, v as u32);
            }
        }
    }
    g.edges()
}

/// `modular_product(G, H)` for two undirected simple graphs into the empty
/// `nx.Graph` given by its dicts. Every edge pair adds `((u, x), (v, y))`
/// then `((v, x), (u, y))` with `**_dict_product(c, d)`; the complements'
/// edge pairs follow with empty dicts.
#[pyfunction]
#[pyo3(signature = (g, h, node, succ))]
pub fn _b24_modular_product<'py>(
    py: Python<'py>,
    g: &OpView,
    h: &OpView,
    node: Bound<'py, PyDict>,
    succ: Bound<'py, PyDict>,
) -> PyResult<()> {
    let out = Out {
        py,
        node,
        succ,
        pred: None,
    };
    let (ng, nh) = (g.n(), h.n());
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
    for &(u, e) in &gedges {
        let v = g.targets[e] as usize;
        let c = as_dict(g.data[e].bind(py))?;
        for &(x, f) in &hedges {
            let y = h.targets[f] as usize;
            let d = as_dict(h.data[f].bind(py))?;
            let dd = dict_product(py, &c, &d)?;
            add(id(u, x), id(v, y), Some(&dd), true)?;
            add(id(v, x), id(u, y), Some(&dd), false)?;
        }
    }
    let gc = complement_edges(g);
    let hc = complement_edges(h);
    for &(u, v) in &gc {
        let (u, v) = (u as usize, v as usize);
        for &(x, y) in &hc {
            let (x, y) = (x as usize, y as usize);
            add(id(u, x), id(v, y), None, false)?;
            add(id(v, x), id(u, y), None, false)?;
        }
    }
    Ok(())
}

// --- quotient_graph -----------------------------------------------------------------

/// `quotient`'s result: edges inside each block, joined block pairs and
/// their weight sums.
pub type Quotient = (Vec<i64>, Vec<(u32, u32)>, Vec<i64>);

/// What `quotient_graph`'s default node and edge data need, for blocks
/// given as each node's block index: the edges inside each block
/// (`S.number_of_edges()`), and the block pairs its default edge relation
/// joins (`combinations` order, or `permutations` order when directed: an
/// arc from the first block to the second) with the summed weights of the
/// edges between them in either direction. `weights` are exact ints (unit
/// weights if `None`); `None` if a sum leaves `i64`.
pub fn quotient(
    g: &crate::graph::CoreGraph,
    block: &[u32],
    nblocks: usize,
    weights: Option<&[f64]>,
) -> Option<Quotient> {
    let mut inside = vec![0i64; nblocks];
    let mut joined: HashSet<(u32, u32)> = HashSet::new();
    let mut sums: HashMap<(u32, u32), i64> = HashMap::new();
    for u in 0..g.n {
        let bu = block[u];
        for e in g.succ.range(u) {
            let v = g.succ.targets[e] as usize;
            let bv = block[v];
            if !g.directed && v < u {
                continue; // the edge's other entry
            }
            if bu == bv {
                inside[bu as usize] += 1;
                continue;
            }
            let w = weights.map_or(1, |w| w[e] as i64);
            let pair = (bu.min(bv), bu.max(bv));
            let s = sums.entry(pair).or_insert(0);
            *s = s.checked_add(w)?;
            joined.insert(if g.directed { (bu, bv) } else { pair });
        }
    }
    let mut pairs: Vec<(u32, u32)> = joined.into_iter().collect();
    pairs.sort_unstable();
    let totals = pairs
        .iter()
        .map(|&(b, c)| sums[&(b.min(c), b.max(c))])
        .collect();
    Some((inside, pairs, totals))
}
