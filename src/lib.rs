//! Rust core for the `rustnx` NetworkX backend.
//!
//! Python hands over NetworkX's adjacency once (`build_graph`), and every
//! algorithm then runs on integer-indexed CSR arrays with the GIL released.

mod algorithms;
mod graph;
mod native;
mod rx;
mod serialize;

use pyo3::exceptions::{PyIndexError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyFloat, PyInt, PyList, PyTuple};

use algorithms::link_analysis::{self, PagerankInput};
use algorithms::random_generators_more as more_random;
use algorithms::shortest_paths_more as more_paths;
use algorithms::traversal::{self, DijkstraState, NegativeCycle};
use algorithms::{
    approximation, bipartite, bipartite_more, centrality, centrality_more, cliques, cluster,
    communities, connectivity, conversion, cores_more, dag, degree_generators, directed, distance,
    flow, generators, graph_classes, isomorphism, leftovers, matching, measures, nxdicts,
    operators, paths, pyrandom, pyset, random_generators, readwrite, spectral, structure,
    structure_more, transforms, trees_more,
};
use graph::CoreGraph;
use rayon::prelude::*;

impl From<NegativeCycle> for PyErr {
    fn from(_: NegativeCycle) -> PyErr {
        // Same arguments as NetworkX's `_dijkstra_multisource`.
        PyValueError::new_err(("Contradictory paths found:", "negative weights?"))
    }
}

fn planarity_bail() -> PyErr {
    PyNotImplementedError::new_err("NetworkX's planarity test fails on this graph")
}

fn unbounded() -> PyErr {
    PyNotImplementedError::new_err("infinite capacity path")
}

fn all_nodes(n: usize) -> Vec<u32> {
    (0..n as u32).collect()
}

/// COO arrays (row, col, data) as native-endian int64/f64 bytes, for
/// `numpy.frombuffer` (writable, unlike `bytes`).
type CooBytes<'py> = (
    Bound<'py, PyByteArray>,
    Bound<'py, PyByteArray>,
    Bound<'py, PyByteArray>,
);

fn i64_bytes<'py>(py: Python<'py>, v: &[i64]) -> Bound<'py, PyByteArray> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_ne_bytes()).collect();
    PyByteArray::new(py, &bytes)
}

fn f64_bytes<'py>(py: Python<'py>, v: &[f64]) -> Bound<'py, PyByteArray> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_ne_bytes()).collect();
    PyByteArray::new(py, &bytes)
}

fn coo_bytes<'py>(py: Python<'py>, coo: &conversion::Coo) -> PyResult<CooBytes<'py>> {
    Ok((
        i64_bytes(py, &coo.row),
        i64_bytes(py, &coo.col),
        f64_bytes(py, &coo.data),
    ))
}

#[pymethods]
impl CoreGraph {
    fn __len__(&self) -> usize {
        self.n
    }

    /// Pickle support: NetworkX keeps converted graphs in the original
    /// graph's cache, so they must survive `pickle` and `copy.deepcopy`.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let rebuild = py
            .import("rustnx._core")?
            .getattr("_core_graph_from_bytes")?;
        let data = PyBytes::new(py, &serialize::to_bytes(self));
        (rebuild, (data,)).into_pyobject(py)
    }

    #[pyo3(name = "load_exact_pred")]
    #[pyo3(signature = (nodes, index, pred, weight_attrs, multigraph=false))]
    fn py_load_exact_pred<'py>(
        &self,
        nodes: &Bound<'py, PyList>,
        index: &Bound<'py, PyDict>,
        pred: &Bound<'py, PyAny>,
        weight_attrs: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
        multigraph: bool,
    ) -> PyResult<()> {
        self.load_exact_pred(nodes, index, pred, weight_attrs, multigraph)
    }

    #[pyo3(name = "has_exact_pred")]
    fn py_has_exact_pred(&self) -> bool {
        self.has_exact_pred()
    }

    #[getter]
    fn directed(&self) -> bool {
        self.directed
    }

    /// Built natively (not converted from a NetworkX graph).
    fn is_native(&self) -> bool {
        self.native.is_some()
    }

    /// Neighbors of `v` (successors for directed graphs), in NetworkX order.
    fn neighbors(&self, v: usize) -> PyResult<Vec<u32>> {
        self.check_index(v)?;
        Ok(self.succ.neighbors(v).to_vec())
    }

    fn predecessors(&self, v: usize) -> PyResult<Vec<u32>> {
        self.check_index(v)?;
        Ok(self.adj(true).neighbors(v).to_vec())
    }

    fn has_edge(&self, u: usize, v: usize) -> PyResult<bool> {
        self.check_index(u)?;
        self.check_index(v)?;
        Ok(self.succ.neighbors(u).contains(&(v as u32)))
    }

    /// NetworkX degrees: self-loops count twice; directed = in + out.
    fn degrees(&self) -> Vec<usize> {
        (0..self.n)
            .map(|v| {
                let out = self.succ.neighbors(v);
                let loops = out.iter().filter(|&&w| w as usize == v).count();
                if self.directed {
                    out.len() + self.adj(true).neighbors(v).len()
                } else {
                    out.len() + loops
                }
            })
            .collect()
    }

    /// Edges in `G.edges` order: `(u, v, edge id or None)` positions. For
    /// undirected graphs each edge once, from its earlier endpoint.
    #[allow(clippy::type_complexity)]
    fn edges_in_order(&self) -> (Vec<u32>, Vec<u32>, Option<Vec<u32>>) {
        let (mut us, mut vs) = (Vec::new(), Vec::new());
        let mut ids = self.native.as_ref().map(|_| Vec::new());
        for u in 0..self.n {
            for e in self.succ.range(u) {
                let v = self.succ.targets[e];
                if self.directed || v as usize >= u {
                    us.push(u as u32);
                    vs.push(v);
                    if let (Some(ids), Some(nat)) = (ids.as_mut(), self.native.as_ref()) {
                        ids.push(nat.succ_edge[e]);
                    }
                }
            }
        }
        (us, vs, ids)
    }

    /// Native graphs: edges in insertion order, and each attribute's
    /// `(name, values, kinds)` per edge (kinds: 0 absent, 1 int, 2 float,
    /// 3 None, 4 bool).
    #[allow(clippy::type_complexity)]
    fn native_edges(&self) -> Option<(Vec<u32>, Vec<u32>, Vec<(String, Vec<f64>, Vec<u8>)>)> {
        self.native.as_ref().map(|nat| {
            let attrs = nat
                .attrs
                .iter()
                .map(|a| (a.name.clone(), a.values.clone(), a.kinds.clone()))
                .collect();
            (nat.src.clone(), nat.dst.clone(), attrs)
        })
    }

    fn attribute_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.weights.keys().cloned().collect();
        names.sort();
        names
    }

    fn number_of_edges(&self) -> usize {
        let m = self.succ.targets.len();
        if self.directed {
            m
        } else {
            // Self-loops appear once in an undirected adjacency row.
            let loops = (0..self.n)
                .filter(|&v| self.succ.neighbors(v).contains(&(v as u32)))
                .count();
            (m + loops) / 2
        }
    }

    /// `(all_int, has_hidden)` for an edge attribute (unit weights: `(True, False)`).
    #[pyo3(signature = (attr=None))]
    fn weight_info(&self, attr: Option<&str>) -> (bool, bool) {
        self.weights_info(attr)
    }

    /// Whether an edge attribute mixes int and float values.
    #[pyo3(signature = (attr=None))]
    fn weight_mixed(&self, attr: Option<&str>) -> bool {
        self.weights_mixed(attr)
    }

    fn bfs_lengths(
        &self,
        py: Python<'_>,
        source: usize,
        cutoff: f64,
    ) -> PyResult<(Vec<u32>, Vec<u32>)> {
        self.check_index(source)?;
        Ok(py.detach(|| traversal::bfs_lengths(&self.succ, self.n, source, cutoff)))
    }

    #[pyo3(signature = (source, weight=None, cutoff=None))]
    fn dijkstra_lengths(
        &self,
        py: Python<'_>,
        source: usize,
        weight: Option<&str>,
        cutoff: Option<f64>,
    ) -> PyResult<(Vec<u32>, Vec<f64>)> {
        self.check_index(source)?;
        let w = self.weight_slice(weight, false)?;
        py.detach(|| {
            let mut state = DijkstraState::new(self.n);
            state.run(&self.succ, w, source, cutoff)?;
            let dists = state
                .order
                .iter()
                .map(|&v| state.dist[v as usize])
                .collect();
            Ok((std::mem::take(&mut state.order), dists))
        })
    }

    fn connected_components(&self, py: Python<'_>) -> Vec<Vec<u32>> {
        py.detach(|| traversal::connected_components(&self.succ, self.n))
    }

    /// Unscaled betweenness; `ordered` sums in NetworkX's source order (bit
    /// for bit), otherwise in faster parallel blocks.
    #[pyo3(signature = (weight=None, endpoints=false, sources=None, ordered=true))]
    fn betweenness(
        &self,
        py: Python<'_>,
        weight: Option<&str>,
        endpoints: bool,
        sources: Option<Vec<u32>>,
        ordered: bool,
    ) -> PyResult<Vec<f64>> {
        let w = self.weight_slice(weight, false)?;
        let sources = sources.unwrap_or_else(|| all_nodes(self.n));
        for &s in &sources {
            self.check_index(s as usize)?;
        }
        Ok(py.detach(|| {
            centrality::betweenness(
                &self.succ,
                self.adj(true),
                self.n,
                w,
                endpoints,
                &sources,
                ordered,
            )
        }))
    }

    /// Unscaled edge betweenness, one value per edge in `edges_in_order`
    /// (`ordered` as for `betweenness`).
    #[pyo3(signature = (weight=None, sources=None, ordered=true))]
    fn edge_betweenness(
        &self,
        py: Python<'_>,
        weight: Option<&str>,
        sources: Option<Vec<u32>>,
        ordered: bool,
    ) -> PyResult<Vec<f64>> {
        let w = self.weight_slice(weight, false)?;
        let sources = self.sources_or_all(sources)?;
        Ok(py.detach(|| {
            let (edge_id, m) = self.edge_ids();
            let in_adj = self.adj(true);
            // Directed: edge ids are arc ids, so number the in-arcs by the
            // arcs they mirror. Undirected: `in_adj` is `succ`.
            let in_edge_id = if self.directed {
                dag::pred_arc_ids(&self.succ, in_adj, self.n)
            } else {
                edge_id.clone()
            };
            centrality::edge_betweenness(
                &self.succ,
                in_adj,
                self.n,
                w,
                &edge_id,
                &in_edge_id,
                m,
                &sources,
                ordered,
            )
        }))
    }

    #[pyo3(signature = (distance=None, wf_improved=true, sources=None, compensated=false))]
    fn closeness(
        &self,
        py: Python<'_>,
        distance: Option<&str>,
        wf_improved: bool,
        sources: Option<Vec<u32>>,
        compensated: bool,
    ) -> PyResult<Vec<f64>> {
        // NetworkX runs closeness on `G.reverse()` for directed graphs.
        let (adj, w) = self.reverse_exact(distance)?;
        let sources = sources.unwrap_or_else(|| all_nodes(self.n));
        for &s in &sources {
            self.check_index(s as usize)?;
        }
        Ok(py
            .detach(|| centrality::closeness(adj, self.n, w, wf_improved, &sources, compensated))?)
    }

    /// PageRank scores in node order, or `None` if it didn't converge.
    #[pyo3(signature = (alpha, personalization, max_iter, tol, nstart, weight, dangling, return_previous=false))]
    #[allow(clippy::too_many_arguments)]
    fn pagerank(
        &self,
        py: Python<'_>,
        alpha: f64,
        personalization: Option<Vec<f64>>,
        max_iter: usize,
        tol: f64,
        nstart: Option<Vec<f64>>,
        weight: Option<&str>,
        dangling: Option<Vec<f64>>,
        return_previous: bool,
    ) -> PyResult<Option<Vec<f64>>> {
        for v in [&personalization, &nstart, &dangling].into_iter().flatten() {
            if v.len() != self.n {
                return Err(PyValueError::new_err(
                    "vector length must equal the node count",
                ));
            }
        }
        let input = PagerankInput {
            n: self.n,
            out_adj: &self.succ,
            out_weights: self.weight_slice(weight, false)?,
            in_adj: self.adj(true),
            in_weights: self.weight_slice(weight, true)?,
            alpha,
            personalization,
            nstart,
            dangling,
            max_iter,
            tol,
            return_previous,
        };
        Ok(py.detach(|| link_analysis::pagerank(input).ok()))
    }

    /// `(reached, total distance, max distance)` per source, by bit-parallel BFS.
    #[pyo3(signature = (sources=None, reverse=false))]
    fn bfs_stats(
        &self,
        py: Python<'_>,
        sources: Option<Vec<u32>>,
        reverse: bool,
    ) -> PyResult<Vec<(usize, u64, u32)>> {
        let sources = self.sources_or_all(sources)?;
        Ok(py.detach(|| {
            distance::bfs_stats(self.adj(reverse), self.n, &sources)
                .into_iter()
                .map(|s| (s.reached, s.total, s.max))
                .collect()
        }))
    }

    /// Per source `(reached, max distance)` or `None` on a negative cycle,
    /// plus the NetworkX-ordered sum of all distances (`None` on any error).
    #[pyo3(signature = (weight, sources=None, compensated=false))]
    #[allow(clippy::type_complexity)]
    fn dijkstra_stats(
        &self,
        py: Python<'_>,
        weight: &str,
        sources: Option<Vec<u32>>,
        compensated: bool,
    ) -> PyResult<(Vec<Option<(usize, f64)>>, Option<f64>)> {
        let w = self
            .weight_slice(Some(weight), false)?
            .expect("weight given");
        let sources = self.sources_or_all(sources)?;
        Ok(py.detach(|| {
            let (stats, total) =
                distance::dijkstra_stats(&self.succ, self.n, w, &sources, compensated);
            let stats = stats
                .into_iter()
                .map(|r| r.ok().map(|s| (s.reached, s.max)))
                .collect();
            (stats, total)
        }))
    }

    /// BFS orders and levels for each source (parallel).
    fn bfs_many(
        &self,
        py: Python<'_>,
        sources: Vec<u32>,
        cutoff: f64,
    ) -> PyResult<Vec<(Vec<u32>, Vec<u32>)>> {
        let sources = self.sources_or_all(Some(sources))?;
        Ok(py.detach(|| distance::bfs_many(&self.succ, self.n, &sources, cutoff)))
    }

    /// Dijkstra orders and distances per source, `None` on a negative cycle.
    #[pyo3(signature = (sources, weight=None, cutoff=None))]
    #[allow(clippy::type_complexity)]
    fn dijkstra_many(
        &self,
        py: Python<'_>,
        sources: Vec<u32>,
        weight: Option<&str>,
        cutoff: Option<f64>,
    ) -> PyResult<Vec<Option<(Vec<u32>, Vec<f64>)>>> {
        let w = self.weight_slice(weight, false)?;
        let sources = self.sources_or_all(Some(sources))?;
        Ok(py.detach(|| {
            distance::dijkstra_many(&self.succ, self.n, w, &sources, cutoff)
                .into_iter()
                .map(Result::ok)
                .collect()
        }))
    }

    /// BFS discovery order and each node's discoverer (`NO_PARENT` for the
    /// source). `reverse` searches the reversed graph in exact order.
    #[pyo3(signature = (source, cutoff, reverse=false))]
    fn bfs_tree(
        &self,
        py: Python<'_>,
        source: usize,
        cutoff: f64,
        reverse: bool,
    ) -> PyResult<(Vec<u32>, Vec<u32>)> {
        self.check_index(source)?;
        let adj = self.path_adj(None, reverse)?.0;
        Ok(py.detach(|| paths::bfs_tree(adj, self.n, source, cutoff)))
    }

    /// Dijkstra with parents: `(order, dist, parent of each order entry,
    /// first-seen order)`. Raises ValueError on a negative cycle.
    #[pyo3(signature = (source, weight=None, cutoff=None, target=None, reverse=false))]
    #[allow(clippy::type_complexity)]
    fn dijkstra_tree(
        &self,
        py: Python<'_>,
        source: usize,
        weight: Option<&str>,
        cutoff: Option<f64>,
        target: Option<usize>,
        reverse: bool,
    ) -> PyResult<(Vec<u32>, Vec<f64>, Vec<u32>, Vec<u32>)> {
        self.check_index(source)?;
        let (adj, w) = self.path_adj(weight, reverse)?;
        let t = py.detach(|| paths::dijkstra_tree(adj, self.n, w, source, cutoff, target))?;
        let parents = t.order.iter().map(|&v| t.parent[v as usize]).collect();
        Ok((t.order, t.dist, parents, t.seen_order))
    }

    /// Dijkstra from `source` until `target` is finalized: `(distance,
    /// path)`, or `None` if `target` isn't reached.
    #[pyo3(signature = (source, target, weight=None, cutoff=None))]
    fn dijkstra_path(
        &self,
        py: Python<'_>,
        source: usize,
        target: usize,
        weight: Option<&str>,
        cutoff: Option<f64>,
    ) -> PyResult<Option<(f64, Vec<u32>)>> {
        self.check_index(source)?;
        self.check_index(target)?;
        let w = self.weight_slice(weight, false)?;
        let t = py
            .detach(|| paths::dijkstra_tree(&self.succ, self.n, w, source, cutoff, Some(target)))?;
        if t.order.last() != Some(&(target as u32)) {
            return Ok(None);
        }
        let mut path = vec![target as u32];
        while *path.last().expect("non-empty") != source as u32 {
            path.push(t.parent[*path.last().expect("non-empty") as usize]);
        }
        path.reverse();
        Ok(Some((*t.dist.last().expect("non-empty"), path)))
    }

    /// Nodes reachable from `source` (following in-edges if `reverse`),
    /// excluding `source` itself. In no particular order.
    #[pyo3(signature = (source, reverse=false))]
    fn reachable(&self, py: Python<'_>, source: usize, reverse: bool) -> PyResult<Vec<u32>> {
        self.check_index(source)?;
        // `nx.ancestors` walks `G._pred` in its exact order (the caller loads it).
        let adj = self.path_adj(None, reverse && self.directed)?.0;
        let (mut order, _) = py.detach(|| paths::bfs_tree(adj, self.n, source, f64::INFINITY));
        order.remove(0);
        Ok(order)
    }

    /// `(t, d, db)` triangle counts per node (see `cluster::triangle_counts`),
    /// for `nodes` or every node. `successors_only` counts a directed graph
    /// with the undirected formula over its successors.
    #[pyo3(signature = (nodes=None, successors_only=false))]
    fn triangle_counts(
        &self,
        py: Python<'_>,
        nodes: Option<Vec<u32>>,
        successors_only: bool,
    ) -> PyResult<Vec<(u64, u64, u64)>> {
        let nodes = self.sources_or_all(nodes)?;
        let pred = (self.directed && !successors_only).then(|| self.adj(true));
        Ok(py.detach(|| cluster::triangle_counts(&self.succ, pred, self.n, &nodes)))
    }

    /// The largest value of edge attribute `weight` (missing counts as 1),
    /// NaN if any value is NaN.
    fn max_weight(&self, weight: &str) -> PyResult<f64> {
        let w = self
            .weight_slice(Some(weight), false)?
            .ok_or_else(|| PyNotImplementedError::new_err("weights were not converted"))?;
        Ok(w.iter().fold(f64::NEG_INFINITY, |m, &x| {
            if m.is_nan() || x.is_nan() {
                f64::NAN
            } else {
                m.max(x)
            }
        }))
    }

    /// Weighted triangle sums per node for weighted `clustering` (see
    /// `cluster::weighted_triangles`), for `nodes` or every node.
    #[pyo3(signature = (weight, max_weight, nodes=None))]
    fn weighted_triangles(
        &self,
        py: Python<'_>,
        weight: &str,
        max_weight: f64,
        nodes: Option<Vec<u32>>,
    ) -> PyResult<Vec<(f64, u64, u64)>> {
        let nodes = self.sources_or_all(nodes)?;
        let succ_w = self
            .weight_slice(Some(weight), false)?
            .ok_or_else(|| PyNotImplementedError::new_err("weights were not converted"))?;
        let pred = if self.directed {
            let pred_w = self
                .weight_slice(Some(weight), true)?
                .ok_or_else(|| PyNotImplementedError::new_err("weights were not converted"))?;
            Some((self.adj(true), pred_w))
        } else {
            None
        };
        Ok(py.detach(|| {
            cluster::weighted_triangles(&self.succ, succ_w, pred, self.n, &nodes, max_weight)
        }))
    }

    /// Core numbers in node order, or `None` if the graph has self-loops.
    fn core_number(&self, py: Python<'_>) -> Option<Vec<u32>> {
        let pred = self.directed.then(|| self.adj(true));
        py.detach(|| {
            if structure::has_self_loops(&self.succ, self.n) {
                None
            } else {
                Some(structure::core_number(&self.succ, pred, self.n))
            }
        })
    }

    fn is_bipartite(&self, py: Python<'_>) -> bool {
        let pred = self.directed.then(|| self.adj(true));
        py.detach(|| structure::is_bipartite(&self.succ, pred, self.n))
    }

    /// `eigenvector_centrality` from the normalized start vector `x0` (by
    /// node position), iterating nodes in `order`; `hypot` is Python's
    /// `math.hypot`. `None` if it doesn't converge.
    #[pyo3(signature = (order, x0, weight, max_iter, tol, hypot, compensated_sum))]
    #[allow(clippy::too_many_arguments)]
    fn eigenvector(
        &self,
        py: Python<'_>,
        order: Vec<u32>,
        x0: Vec<f64>,
        weight: Option<&str>,
        max_iter: usize,
        tol: f64,
        hypot: Bound<'_, PyAny>,
        compensated_sum: bool,
    ) -> PyResult<Option<Vec<f64>>> {
        let order = self.sources_or_all(Some(order))?;
        if x0.len() != self.n {
            return Err(PyValueError::new_err("x0 must have one value per node"));
        }
        let it = spectral::Iteration {
            adj: &self.succ,
            weights: self.weight_slice(weight, false)?,
            order: &order,
            max_iter,
            tol,
            compensated_sum,
        };
        it.eigenvector(x0, |values| {
            hypot.call1(PyTuple::new(py, values)?)?.extract::<f64>()
        })
    }

    /// `katz_centrality` before normalization, with scalar `beta` and a
    /// zero start vector. `None` if it doesn't converge.
    #[pyo3(signature = (order, alpha, beta, weight, max_iter, tol, compensated_sum))]
    #[allow(clippy::too_many_arguments)]
    fn katz(
        &self,
        py: Python<'_>,
        order: Vec<u32>,
        alpha: f64,
        beta: f64,
        weight: Option<&str>,
        max_iter: usize,
        tol: f64,
        compensated_sum: bool,
    ) -> PyResult<Option<Vec<f64>>> {
        let order = self.sources_or_all(Some(order))?;
        let it = spectral::Iteration {
            adj: &self.succ,
            weights: self.weight_slice(weight, false)?,
            order: &order,
            max_iter,
            tol,
            compensated_sum,
        };
        Ok(py.detach(|| it.katz(vec![0.0; self.n], alpha, beta)))
    }

    /// Harmonic centrality sums (see `centrality::harmonic`).
    #[pyo3(signature = (sources, in_nbunch, weight=None))]
    fn harmonic(
        &self,
        py: Python<'_>,
        sources: Vec<u32>,
        in_nbunch: Vec<bool>,
        weight: Option<&str>,
    ) -> PyResult<(Vec<f64>, Vec<bool>)> {
        let sources = self.sources_or_all(Some(sources))?;
        if in_nbunch.len() != self.n {
            return Err(PyValueError::new_err(
                "in_nbunch must have one value per node",
            ));
        }
        let w = self.weight_slice(weight, false)?;
        Ok(py.detach(|| centrality::harmonic(&self.succ, self.n, w, &sources, &in_nbunch))?)
    }

    /// `bidirectional_dijkstra` for `source != target`: `(distance, path)`
    /// or `None` if there is no path. Directed graphs need the exact
    /// predecessor order loaded.
    #[pyo3(signature = (source, target, weight=None))]
    fn bidirectional_dijkstra(
        &self,
        py: Python<'_>,
        source: usize,
        target: usize,
        weight: Option<&str>,
    ) -> PyResult<Option<(f64, Vec<u32>)>> {
        self.check_index(source)?;
        self.check_index(target)?;
        let forward = (&self.succ, self.weight_slice(weight, false)?);
        let backward = self.path_adj(weight, true)?;
        match py
            .detach(|| paths::bidirectional_dijkstra([forward, backward], self.n, source, target))
        {
            Ok(found) => Ok(Some(found)),
            Err(paths::BidirectionalError::NoPath) => Ok(None),
            Err(paths::BidirectionalError::Contradictory) => Err(PyValueError::new_err(
                "Contradictory paths found: negative weights?",
            )),
        }
    }

    /// `generic_bfs_edges` tree edges as `(parents, children)`. `reverse`
    /// follows in-edges in exact order (directed graphs).
    #[pyo3(signature = (source, depth_limit, reverse=false))]
    fn bfs_edges(
        &self,
        py: Python<'_>,
        source: usize,
        depth_limit: i64,
        reverse: bool,
    ) -> PyResult<(Vec<u32>, Vec<u32>)> {
        self.check_index(source)?;
        let adj = self.path_adj(None, reverse && self.directed)?.0;
        Ok(py.detach(|| {
            traversal::bfs_edges(adj, self.n, source, depth_limit)
                .into_iter()
                .unzip()
        }))
    }

    /// DFS forward edges from `starts` (all nodes if `None`), with
    /// `(start, start)` marking each new start.
    #[pyo3(signature = (starts, depth_limit))]
    fn dfs_forward(
        &self,
        py: Python<'_>,
        starts: Option<Vec<u32>>,
        depth_limit: i64,
    ) -> PyResult<(Vec<u32>, Vec<u32>)> {
        let starts = self.sources_or_all(starts)?;
        Ok(py.detach(|| {
            traversal::dfs_forward(&self.succ, self.n, &starts, depth_limit)
                .into_iter()
                .unzip()
        }))
    }

    /// `nx.bfs_layers` after the first layer: later nodes in yield order and
    /// where each layer ends.
    fn bfs_layers(
        &self,
        py: Python<'_>,
        starts: Vec<u32>,
        visited: Vec<u32>,
    ) -> PyResult<(Vec<u32>, Vec<u32>)> {
        let starts = self.sources_or_all(Some(starts))?;
        let visited = self.sources_or_all(Some(visited))?;
        Ok(py.detach(|| traversal::bfs_layers(&self.succ, self.n, &starts, &visited)))
    }

    /// `nx.dfs_postorder_nodes` from `starts` (all nodes if `None`).
    #[pyo3(signature = (starts, depth_limit))]
    fn dfs_postorder(
        &self,
        py: Python<'_>,
        starts: Option<Vec<u32>>,
        depth_limit: i64,
    ) -> PyResult<Vec<u32>> {
        let starts = self.sources_or_all(starts)?;
        Ok(py.detach(|| traversal::dfs_postorder(&self.succ, self.n, &starts, depth_limit)))
    }

    /// `nx.articulation_points` in yield order.
    fn articulation_points(&self, py: Python<'_>) -> Vec<u32> {
        match py.detach(|| structure::biconnected(&self.succ, self.n, false)) {
            structure::Biconnected::Articulation(points) => points,
            structure::Biconnected::Components(..) => unreachable!(),
        }
    }

    /// `nx.biconnected_component_edges`: all components' edges as
    /// `(us, vs)`, and where each component ends.
    #[allow(clippy::type_complexity)]
    fn biconnected_components(&self, py: Python<'_>) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        match py.detach(|| structure::biconnected(&self.succ, self.n, true)) {
            structure::Biconnected::Components(edges, ends) => {
                let (us, vs) = edges.into_iter().unzip();
                (us, vs, ends)
            }
            structure::Biconnected::Articulation(_) => unreachable!(),
        }
    }

    /// `nx.is_biconnected`: exactly one biconnected component, covering
    /// every node.
    fn is_biconnected(&self, py: Python<'_>) -> bool {
        match py.detach(|| structure::biconnected(&self.succ, self.n, true)) {
            structure::Biconnected::Components(edges, ends) => {
                if ends.len() != 1 {
                    return false;
                }
                let mut seen = vec![false; self.n];
                let mut count = 0;
                for (u, v) in edges {
                    for w in [u, v] {
                        if !seen[w as usize] {
                            seen[w as usize] = true;
                            count += 1;
                        }
                    }
                }
                count == self.n
            }
            structure::Biconnected::Articulation(_) => unreachable!(),
        }
    }

    /// Strongly connected components (NetworkX order) with no edge leaving
    /// them: `nx.attracting_components`.
    #[pyo3(signature = (early_exit=true))]
    fn attracting_components(&self, py: Python<'_>, early_exit: bool) -> Vec<Vec<u32>> {
        py.detach(|| {
            let comps = directed::strongly_connected_components(&self.succ, self.n, early_exit);
            let mut comp_of = vec![0usize; self.n];
            for (i, comp) in comps.iter().enumerate() {
                for &v in comp {
                    comp_of[v as usize] = i;
                }
            }
            comps
                .into_iter()
                .enumerate()
                .filter(|(i, comp)| {
                    comp.iter().all(|&v| {
                        self.succ
                            .neighbors(v as usize)
                            .iter()
                            .all(|&w| comp_of[w as usize] == *i)
                    })
                })
                .map(|(_, comp)| comp)
                .collect()
        })
    }

    /// `(in_degrees, out_degrees)` of a directed graph (self-loops once each).
    fn in_out_degrees(&self) -> (Vec<usize>, Vec<usize>) {
        let pred = self.adj(true);
        (0..self.n)
            .map(|v| (pred.neighbors(v).len(), self.succ.neighbors(v).len()))
            .unzip()
    }

    // --- Batch 2: shortest paths ---

    /// Dijkstra from several sources: `(order, dist, parent of each order
    /// entry, first-seen order, order indices of unparented nodes)`.
    #[pyo3(signature = (sources, weight=None, cutoff=None))]
    #[allow(clippy::type_complexity)]
    fn dijkstra_forest(
        &self,
        py: Python<'_>,
        sources: Vec<u32>,
        weight: Option<&str>,
        cutoff: Option<f64>,
    ) -> PyResult<(Vec<u32>, Vec<f64>, Vec<u32>, Vec<u32>, Vec<u32>)> {
        let sources = self.sources_or_all(Some(sources))?;
        let w = self.weight_slice(weight, false)?;
        let t = py.detach(|| {
            more_paths::dijkstra_forest(&self.succ, self.n, w, &sources, cutoff, None)
        })?;
        let parents: Vec<u32> = t.order.iter().map(|&v| t.parent[v as usize]).collect();
        let roots = (0..parents.len() as u32)
            .filter(|&k| parents[k as usize] == paths::NO_PARENT)
            .collect();
        Ok((t.order, t.dist, parents, t.seen_order, roots))
    }

    /// Multi-source Dijkstra until `target` is popped: `(distance, path)`,
    /// or `None` if `target` isn't reached.
    #[pyo3(signature = (sources, target, weight=None, cutoff=None))]
    fn dijkstra_forest_path(
        &self,
        py: Python<'_>,
        sources: Vec<u32>,
        target: usize,
        weight: Option<&str>,
        cutoff: Option<f64>,
    ) -> PyResult<Option<(f64, Vec<u32>)>> {
        let sources = self.sources_or_all(Some(sources))?;
        self.check_index(target)?;
        let w = self.weight_slice(weight, false)?;
        let t = py.detach(|| {
            more_paths::dijkstra_forest(&self.succ, self.n, w, &sources, cutoff, Some(target))
        })?;
        if t.order.last() != Some(&(target as u32)) {
            return Ok(None);
        }
        let mut path = vec![target as u32];
        loop {
            let p = t.parent[*path.last().expect("non-empty") as usize];
            if p == paths::NO_PARENT {
                break;
            }
            path.push(p);
        }
        path.reverse();
        Ok(Some((*t.dist.last().expect("non-empty"), path)))
    }

    /// `dijkstra_predecessor_and_distance`: `(pop order, distances, pred key
    /// order, pred lists flattened, where each list ends)`.
    #[pyo3(signature = (source, weight=None, cutoff=None))]
    #[allow(clippy::type_complexity)]
    fn dijkstra_pred_dist(
        &self,
        py: Python<'_>,
        source: usize,
        weight: Option<&str>,
        cutoff: Option<f64>,
    ) -> PyResult<(Vec<u32>, Vec<f64>, Vec<u32>, Vec<u32>, Vec<u32>)> {
        self.check_index(source)?;
        let w = self.weight_slice(weight, false)?;
        let (order, dist, pred) =
            py.detach(|| more_paths::dijkstra_pred_dist(&self.succ, self.n, w, source, cutoff))?;
        let (flat, ends) = pred.flatten();
        Ok((order, dist, pred.order, flat, ends))
    }

    /// `nx.predecessor`: `(discovery order, levels, pred lists flattened,
    /// where each list ends)`.
    #[pyo3(signature = (source, max_level=None))]
    #[allow(clippy::type_complexity)]
    fn bfs_pred(
        &self,
        py: Python<'_>,
        source: usize,
        max_level: Option<u32>,
    ) -> PyResult<(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>)> {
        self.check_index(source)?;
        let (levels, pred) =
            py.detach(|| more_paths::bfs_pred(&self.succ, self.n, source, max_level));
        let (flat, ends) = pred.flatten();
        Ok((pred.order, levels, flat, ends))
    }

    /// `single_target_shortest_path_length`: BFS over in-edges in exact order.
    fn bfs_lengths_reverse(
        &self,
        py: Python<'_>,
        target: usize,
        cutoff: f64,
    ) -> PyResult<(Vec<u32>, Vec<u32>)> {
        self.check_index(target)?;
        let adj = self.path_adj(None, true)?.0;
        Ok(py.detach(|| traversal::bfs_lengths(adj, self.n, target, cutoff)))
    }

    /// Predecessor lists from `source` for `_build_paths_from_predecessors`:
    /// `kind` 0 = `nx.predecessor`, 1 = Dijkstra (ValueError on a negative
    /// cycle), 2 = `bellman_ford_predecessor_and_distance` (`None` on a
    /// negative cycle).
    #[pyo3(signature = (source, kind, weight=None))]
    fn pred_paths(
        &self,
        py: Python<'_>,
        source: usize,
        kind: u8,
        weight: Option<&str>,
    ) -> PyResult<Option<PredPaths>> {
        self.check_index(source)?;
        let w = self.weight_slice(weight, false)?;
        let pred = match kind {
            0 => py.detach(|| more_paths::bfs_pred(&self.succ, self.n, source, None).1),
            1 => {
                py.detach(|| more_paths::dijkstra_pred_dist(&self.succ, self.n, w, source, None))?
                    .2
            }
            _ => match py.detach(|| self.bf_preds(w, source, false)) {
                Some(pred) => pred,
                None => return Ok(None),
            },
        };
        Ok(Some(PredPaths {
            pred,
            source: source as u32,
        }))
    }

    /// `_inner_bellman_ford` from `source`, or `None` on a negative cycle:
    /// `(dist key order, distances, flat, ends)`. `mode` 0 gives the pred
    /// lists, 1 the first path to every node (`_bellman_ford`'s `paths`), 2
    /// the first path to `target` only (none if unreached), 3 nothing.
    /// `single_node_shortcut`: `bellman_ford_predecessor_and_distance`
    /// returns at once for a one-node graph.
    #[pyo3(signature = (source, weight, heuristic, single_node_shortcut, mode, target=None))]
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn bellman_ford(
        &self,
        py: Python<'_>,
        source: usize,
        weight: Option<&str>,
        heuristic: bool,
        single_node_shortcut: bool,
        mode: u8,
        target: Option<usize>,
    ) -> PyResult<Option<(Vec<u32>, Vec<f64>, Vec<u32>, Vec<u32>)>> {
        self.check_index(source)?;
        if let Some(t) = target {
            self.check_index(t)?;
        }
        let w = self.weight_slice(weight, false)?;
        py.detach(|| {
            if single_node_shortcut && self.n == 1 {
                return Ok(Some((vec![source as u32], vec![0.0], vec![], vec![0])));
            }
            let bf = more_paths::bellman_ford(&self.succ, self.n, w, source, heuristic);
            if bf.cycle.is_some() {
                return Ok(None);
            }
            let (flat, ends) = self.bf_output(&bf, source, mode, target)?;
            Ok(Some((bf.order, bf.dist, flat, ends)))
        })
    }

    /// `bellman_ford` (heuristic on, `mode` 1 or 3) for each source, in
    /// parallel; `None` on a negative cycle.
    #[pyo3(signature = (sources, weight, mode))]
    #[allow(clippy::type_complexity)]
    fn bellman_ford_many(
        &self,
        py: Python<'_>,
        sources: Vec<u32>,
        weight: Option<&str>,
        mode: u8,
    ) -> PyResult<Vec<Option<(Vec<u32>, Vec<f64>, Vec<u32>, Vec<u32>)>>> {
        let sources = self.sources_or_all(Some(sources))?;
        let w = self.weight_slice(weight, false)?;
        py.detach(|| {
            use rayon::prelude::*;
            sources
                .par_iter()
                .map(|&s| {
                    let bf = more_paths::bellman_ford(&self.succ, self.n, w, s as usize, true);
                    if bf.cycle.is_some() {
                        return Ok(None);
                    }
                    let (flat, ends) = self.bf_output(&bf, s as usize, mode, None)?;
                    Ok(Some((bf.order, bf.dist, flat, ends)))
                })
                .collect()
        })
    }

    /// Whether some self-loop has a negative weight.
    #[pyo3(signature = (weight=None))]
    fn negative_selfloop(&self, weight: Option<&str>) -> PyResult<bool> {
        let w = self.weight_slice(weight, false)?;
        Ok(more_paths::negative_selfloop(&self.succ, self.n, w))
    }

    /// Whether some edge has a negative weight.
    #[pyo3(signature = (weight=None))]
    fn has_negative_weight(&self, weight: Option<&str>) -> PyResult<bool> {
        let w = self.weight_slice(weight, false)?;
        Ok(w.is_some_and(|w| w.iter().any(|&x| x < 0.0)))
    }

    /// `find_negative_cycle`: `None` if no cycle is detected, else
    /// `(true, cycle)`, or `(false, [v])` where NetworkX's search through the
    /// predecessors of `v` fails ("should not reach here").
    #[pyo3(signature = (source, weight=None))]
    fn find_negative_cycle(
        &self,
        py: Python<'_>,
        source: usize,
        weight: Option<&str>,
    ) -> PyResult<Option<(bool, Vec<u32>)>> {
        self.check_index(source)?;
        let w = self.weight_slice(weight, false)?;
        Ok(py.detach(|| {
            let bf = more_paths::bellman_ford(&self.succ, self.n, w, source, true);
            let v = bf.cycle?;
            Some(
                match more_paths::negative_cycle_from(&bf.pred.lists, self.n, v) {
                    Some(cycle) => (true, cycle),
                    None => (false, vec![v]),
                },
            )
        }))
    }

    /// `nx.negative_edge_cycle` for a graph with edges.
    #[pyo3(signature = (weight=None, heuristic=true))]
    fn negative_edge_cycle(
        &self,
        py: Python<'_>,
        weight: Option<&str>,
        heuristic: bool,
    ) -> PyResult<bool> {
        let w = self.weight_slice(weight, false)?;
        Ok(py.detach(|| {
            more_paths::negative_edge_cycle(&self.succ, self.n, w, self.directed, heuristic)
        }))
    }

    /// `nx.astar_path` with no heuristic: `(path, edge weights)`, or `None`.
    #[pyo3(signature = (source, target, weight=None, cutoff=None))]
    fn astar(
        &self,
        py: Python<'_>,
        source: usize,
        target: usize,
        weight: Option<&str>,
        cutoff: Option<f64>,
    ) -> PyResult<Option<(Vec<u32>, Vec<f64>)>> {
        self.check_index(source)?;
        self.check_index(target)?;
        let w = self.weight_slice(weight, false)?;
        Ok(py.detach(|| more_paths::astar(&self.succ, self.n, w, source, target, cutoff)))
    }

    // --- Batch 3: DAGs, traversal and components ---

    /// `nx.dag_longest_path`: `(has_cycle, path)`; the path is `None` when
    /// distances get too large for exact f64 sums. `weight=None` gives every
    /// edge the weight `constant`.
    #[pyo3(signature = (weight, constant))]
    fn dag_longest_path(
        &self,
        py: Python<'_>,
        weight: Option<&str>,
        constant: f64,
    ) -> PyResult<(bool, Option<Vec<u32>>)> {
        let (pred, w) = self.path_adj(weight, true)?;
        Ok(py.detach(|| {
            let (generations, cycle) =
                directed::topological_generations(&self.succ, self.adj(true), self.n);
            if cycle {
                return (true, None);
            }
            let topo: Vec<u32> = generations.into_iter().flatten().collect();
            (false, dag::longest_path(pred, w, constant, &topo))
        }))
    }

    /// Weights of the edges along `path` (one value per step).
    fn path_weights(&self, path: Vec<u32>, weight: &str) -> PyResult<Vec<f64>> {
        let w = self.weight_slice(Some(weight), false)?.expect("weights");
        let mut out = Vec::with_capacity(path.len().saturating_sub(1));
        for pair in path.windows(2) {
            self.check_index(pair[0] as usize)?;
            let e = self
                .succ
                .range(pair[0] as usize)
                .find(|&e| self.succ.targets[e] == pair[1])
                .ok_or_else(|| PyValueError::new_err("not an edge"))?;
            out.push(w[e]);
        }
        Ok(out)
    }

    /// `nx.lexicographical_topological_sort` with each node's rank in sort
    /// order: `(order, has_cycle)`.
    fn lexicographical_topological_sort(
        &self,
        py: Python<'_>,
        rank: Vec<u32>,
    ) -> PyResult<(Vec<u32>, bool)> {
        if rank.len() != self.n {
            return Err(PyValueError::new_err("one rank per node"));
        }
        Ok(py.detach(|| dag::lexicographical_topological_sort(&self.succ, self.adj(true), &rank)))
    }

    /// `nx.all_topological_sorts`, one sort at a time.
    fn all_topological_sorts(&self) -> AllTopoSorts {
        AllTopoSorts(dag::AllTopologicalSorts::new(
            &self.succ,
            self.adj(true),
            self.n,
        ))
    }

    /// `nx.transitive_reduction`: `None` if the graph has a cycle, else for
    /// each arc `dag::KEPT` or which successor's descendants removed it.
    fn transitive_reduction(&self, py: Python<'_>) -> Option<Vec<u32>> {
        py.detach(|| {
            let (generations, cycle) =
                directed::topological_generations(&self.succ, self.adj(true), self.n);
            if cycle {
                return None;
            }
            let mut pos = vec![0u32; self.n];
            for (i, &v) in generations.iter().flatten().enumerate() {
                pos[v as usize] = i as u32;
            }
            Some(dag::transitive_reduction(&self.succ, self.n, &pos))
        })
    }

    /// For `nx.transitive_closure`: per source, the heads of `edge_bfs`
    /// (`edge_bfs=true`) or the `descendants` in insertion order.
    fn closure_heads(
        &self,
        py: Python<'_>,
        sources: Vec<u32>,
        edge_bfs: bool,
    ) -> PyResult<Vec<Vec<u32>>> {
        let sources = self.sources_or_all(Some(sources))?;
        Ok(py.detach(|| {
            sources
                .par_iter()
                .map(|&v| {
                    dag::closure_heads(&self.succ, self.n, self.directed, v as usize, edge_bfs)
                })
                .collect()
        }))
    }

    /// State for `nx.transitive_closure_dag`.
    fn closure_dag(&self) -> ClosureDag {
        ClosureDag(dag::ClosureDag::new(&self.succ, self.n))
    }

    /// `nx.dag.root_to_leaf_paths`, one path at a time.
    fn root_to_leaf_paths(&self) -> RootLeafPaths {
        RootLeafPaths(dag::RootLeafPaths::new(&self.succ, self.adj(true), self.n))
    }

    /// `nx.dag_to_branching` on a DAG: each tree node's original node and
    /// parent (`u32::MAX` for none), in NetworkX's numbering.
    fn dag_to_branching(&self, py: Python<'_>) -> (Vec<u32>, Vec<u32>) {
        py.detach(|| {
            let mut paths = dag::RootLeafPaths::new(&self.succ, self.adj(true), self.n);
            dag::branching(&mut paths)
        })
    }

    /// `nx.dag.colliders` / `v_structures` from node `start`, about `limit`
    /// triples at a time: `(flat triples, next start)`.
    fn colliders(
        &self,
        py: Python<'_>,
        start: usize,
        limit: usize,
        v_structures: bool,
    ) -> PyResult<(Vec<u32>, usize)> {
        let pred = self.reverse_exact_order(None)?.0;
        Ok(py.detach(|| dag::colliders(&self.succ, pred, self.n, start, limit, v_structures)))
    }

    /// The BFS of NetworkX 3.4 to 3.6's `is_aperiodic` from node 0:
    /// `(nodes reached, gcd)`.
    fn aperiodic_bfs(&self, py: Python<'_>) -> PyResult<(usize, u64)> {
        self.check_index(0)?;
        Ok(py.detach(|| dag::aperiodic_bfs(&self.succ, self.n, 0)))
    }

    /// `nx.dfs_labeled_edges` from `starts` (all nodes if `None`):
    /// `(us, vs, labels)`.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (starts, depth_limit))]
    fn dfs_labeled_edges(
        &self,
        py: Python<'_>,
        starts: Option<Vec<u32>>,
        depth_limit: i64,
    ) -> PyResult<(Vec<u32>, Vec<u32>, Vec<u8>)> {
        let starts = self.sources_or_all(starts)?;
        Ok(py.detach(|| unzip3(dag::dfs_labeled(&self.succ, self.n, &starts, depth_limit))))
    }

    /// `nx.bfs_labeled_edges` from `sources` (`u32::MAX` marks a missing
    /// one): `(us, vs, labels, position of the missing source reached)`.
    #[allow(clippy::type_complexity)]
    fn bfs_labeled_edges(
        &self,
        py: Python<'_>,
        sources: Vec<u32>,
    ) -> PyResult<(Vec<u32>, Vec<u32>, Vec<u8>, Option<usize>)> {
        for &s in &sources {
            if s != dag::MISSING {
                self.check_index(s as usize)?;
            }
        }
        Ok(py.detach(|| {
            let (edges, missing) = dag::bfs_labeled(&self.succ, self.n, self.directed, &sources);
            let (us, vs, labels) = unzip3(edges);
            (us, vs, labels, missing)
        }))
    }

    /// `nx.edge_bfs` (or `edge_dfs` with `dfs`) from `starts`, following
    /// out-edges (`out`) and/or in-edges (`inward`, directed graphs):
    /// `(us, vs, labels)`.
    #[allow(clippy::type_complexity)]
    fn edge_traversal(
        &self,
        py: Python<'_>,
        starts: Vec<u32>,
        out: bool,
        inward: bool,
        dfs: bool,
    ) -> PyResult<(Vec<u32>, Vec<u32>, Vec<u8>)> {
        let starts = self.sources_or_all(Some(starts))?;
        let inward = inward && self.directed;
        let pred = if inward {
            Some(self.reverse_exact_order(None)?.0)
        } else {
            None
        };
        Ok(py.detach(|| {
            let (succ_id, edges) = self.edge_ids();
            let pred_id = pred.map(|p| dag::pred_arc_ids(&self.succ, p, self.n));
            let src = dag::EdgeSource {
                succ: &self.succ,
                succ_id: &succ_id,
                pred: pred.zip(pred_id.as_deref()),
                out: out || !self.directed,
                inward,
                edges,
            };
            let found = if dfs {
                dag::edge_dfs(&src, self.n, &starts)
            } else {
                dag::edge_bfs(&src, self.n, &starts)
            };
            unzip3(found)
        }))
    }

    /// `nx.kosaraju_strongly_connected_components` from `source` (all
    /// nodes if `None`); `stack_order`: NetworkX 3.7's version.
    #[pyo3(signature = (source, stack_order))]
    fn kosaraju(
        &self,
        py: Python<'_>,
        source: Option<u32>,
        stack_order: bool,
    ) -> PyResult<Vec<Vec<u32>>> {
        let starts = self.sources_or_all(source.map(|s| vec![s]))?;
        let pred = self.reverse_exact_order(None)?.0;
        Ok(py.detach(|| {
            let post = traversal::dfs_postorder(pred, self.n, &starts, self.n as i64);
            dag::kosaraju(&self.succ, self.n, post, stack_order)
        }))
    }

    /// `nx.condensation`: the strongly connected components (NetworkX
    /// order) and the condensed edges `(us, vs)` in insertion order.
    #[allow(clippy::type_complexity)]
    fn condensation(
        &self,
        py: Python<'_>,
        early_exit: bool,
    ) -> (Vec<Vec<u32>>, Vec<u32>, Vec<u32>) {
        py.detach(|| {
            let comps = directed::strongly_connected_components(&self.succ, self.n, early_exit);
            let comp = dag::component_of(&comps, self.n);
            let (us, vs) = dag::condensation_edges(&self.succ, self.n, &comp)
                .into_iter()
                .unzip();
            (comps, us, vs)
        })
    }

    /// `nx.is_semiconnected` for a weakly connected, non-empty graph.
    fn is_semiconnected(&self, py: Python<'_>) -> bool {
        py.detach(|| {
            let comps = directed::strongly_connected_components(&self.succ, self.n, true);
            dag::is_semiconnected(&self.succ, self.n, &comps)
        })
    }

    // --- Batch 4: centrality ---

    /// Unscaled `betweenness_centrality_subset`, in node order.
    #[pyo3(signature = (sources, targets, weight=None))]
    fn betweenness_subset(
        &self,
        py: Python<'_>,
        sources: Vec<u32>,
        targets: Vec<u32>,
        weight: Option<&str>,
    ) -> PyResult<Vec<f64>> {
        let w = self.weight_slice(weight, false)?;
        let sources = self.sources_or_all(Some(sources))?;
        let is_target = self.membership(&targets)?;
        Ok(py.detach(|| {
            centrality_more::betweenness_subset(&self.succ, self.n, w, &sources, &is_target)
        }))
    }

    /// Unscaled `edge_betweenness_centrality_subset`, per edge in
    /// `edges_in_order`.
    #[pyo3(signature = (sources, targets, weight=None))]
    fn edge_betweenness_subset(
        &self,
        py: Python<'_>,
        sources: Vec<u32>,
        targets: Vec<u32>,
        weight: Option<&str>,
    ) -> PyResult<Vec<f64>> {
        let w = self.weight_slice(weight, false)?;
        let sources = self.sources_or_all(Some(sources))?;
        let is_target = self.membership(&targets)?;
        Ok(py.detach(|| {
            let (edge_id, m) = self.edge_ids();
            centrality_more::edge_betweenness_subset(
                &self.succ, self.n, w, &sources, &is_target, &edge_id, m,
            )
        }))
    }

    /// Unnormalized load centrality, in node order. `rank[v]` is `v`'s
    /// position when the nodes are sorted.
    #[pyo3(signature = (rank, weight=None, cutoff=None))]
    fn load(
        &self,
        py: Python<'_>,
        rank: Vec<u32>,
        weight: Option<&str>,
        cutoff: Option<f64>,
    ) -> PyResult<Vec<f64>> {
        if rank.len() != self.n {
            return Err(PyValueError::new_err(
                "rank length must equal the node count",
            ));
        }
        let w = self.weight_slice(weight, false)?;
        Ok(py.detach(|| centrality_more::load(&self.succ, self.n, w, cutoff, &rank))?)
    }

    /// `edge_load_centrality`: key node pairs and their values.
    #[pyo3(signature = (cutoff=None))]
    #[allow(clippy::type_complexity)]
    fn edge_load(&self, py: Python<'_>, cutoff: Option<f64>) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
        let (us, vs, _) = self.edges_in_order();
        py.detach(|| {
            let (keys, values) = centrality_more::edge_load(&self.succ, self.n, (&us, &vs), cutoff);
            let (a, b) = keys.into_iter().unzip();
            (a, b, values)
        })
    }

    /// Unscaled `percolation_centrality`, in node order.
    #[pyo3(signature = (states, total, weight=None))]
    fn percolation(
        &self,
        py: Python<'_>,
        states: Vec<f64>,
        total: f64,
        weight: Option<&str>,
    ) -> PyResult<Vec<f64>> {
        if states.len() != self.n {
            return Err(PyValueError::new_err(
                "states length must equal the node count",
            ));
        }
        let w = self.weight_slice(weight, false)?;
        Ok(py.detach(|| centrality_more::percolation(&self.succ, self.n, w, &states, total)))
    }

    /// `voterank`: elected nodes in order.
    fn voterank(&self, py: Python<'_>, number: usize, avg_degree: f64) -> Vec<u32> {
        let (us, vs, _) = self.edges_in_order();
        py.detach(|| {
            centrality_more::voterank(
                &self.succ,
                self.n,
                self.directed,
                (&us, &vs),
                number,
                avg_degree,
            )
        })
    }

    /// `(total, embeddedness)` of `_dispersion` for each `(u, v)` pair; all
    /// arcs in CSR order when `pairs` is `None`.
    #[pyo3(signature = (pairs=None))]
    fn dispersion(
        &self,
        py: Python<'_>,
        pairs: Option<(Vec<u32>, Vec<u32>)>,
    ) -> PyResult<Vec<(u64, u64)>> {
        let (us, vs) = match pairs {
            Some((us, vs)) => {
                for &v in us.iter().chain(&vs) {
                    self.check_index(v as usize)?;
                }
                (us, vs)
            }
            None => {
                let us = (0..self.n)
                    .flat_map(|u| std::iter::repeat_n(u as u32, self.succ.neighbors(u).len()))
                    .collect();
                (us, self.succ.targets.clone())
            }
        };
        Ok(py.detach(|| centrality_more::dispersion(&self.succ, self.n, &us, &vs)))
    }

    fn has_self_loops(&self) -> bool {
        (0..self.n).any(|v| self.succ.neighbors(v).contains(&(v as u32)))
    }

    /// `group_degree_centrality`'s numerator (`reverse`: predecessors).
    #[pyo3(signature = (group, reverse=false))]
    fn group_degree(&self, group: Vec<u32>, reverse: bool) -> PyResult<usize> {
        let group = self.sources_or_all(Some(group))?;
        Ok(centrality_more::group_degree(
            self.adj(reverse),
            self.n,
            &group,
        ))
    }

    /// `group_closeness_centrality`'s sum of distances from `group` (on the
    /// reversed graph if directed), added in `order`. Non-negative weights.
    #[pyo3(signature = (group, order, weight=None))]
    fn group_distance_sum(
        &self,
        py: Python<'_>,
        group: Vec<u32>,
        order: Vec<u32>,
        weight: Option<&str>,
    ) -> PyResult<f64> {
        let group = self.sources_or_all(Some(group))?;
        let order = self.sources_or_all(Some(order))?;
        let w = self.weight_slice(weight, true)?;
        let adj = self.adj(true);
        Ok(py.detach(|| {
            let dist = centrality_more::multi_source_distances(adj, self.n, w, &group);
            let mut total = 0.0;
            for &v in &order {
                if let Some(d) = dist[v as usize] {
                    total += d;
                }
            }
            total
        }))
    }

    /// Unweighted reaching: `(nodes reached, sum of 1 / distance)` per source.
    fn reaching_unweighted(
        &self,
        py: Python<'_>,
        sources: Vec<u32>,
        compensated: bool,
    ) -> PyResult<Vec<(usize, f64)>> {
        let sources = self.sources_or_all(Some(sources))?;
        Ok(py.detach(|| {
            centrality_more::reaching_unweighted(&self.succ, self.n, &sources, compensated)
        }))
    }

    /// Weighted reaching: the sum of average path weights per source.
    fn reaching_weighted(
        &self,
        py: Python<'_>,
        sources: Vec<u32>,
        weight: &str,
        total: f64,
        pop_order: bool,
        compensated: bool,
    ) -> PyResult<Vec<f64>> {
        let sources = self.sources_or_all(Some(sources))?;
        let w = self.weight_slice(Some(weight), false)?.expect("weighted");
        Ok(py.detach(|| {
            centrality_more::reaching_weighted(
                &self.succ,
                self.n,
                w,
                total,
                &sources,
                pop_order,
                compensated,
            )
        }))
    }

    /// `_group_preprocessing` for the nodes `set_v`.
    #[pyo3(signature = (set_v, weight=None))]
    fn group_preprocessing(
        &self,
        py: Python<'_>,
        set_v: Vec<u32>,
        weight: Option<&str>,
    ) -> PyResult<GroupPre> {
        let set_v = self.sources_or_all(Some(set_v))?;
        let w = self.weight_slice(weight, false)?;
        let data =
            py.detach(|| centrality_more::group_preprocessing(&self.succ, self.n, w, &set_v));
        let rev_reach =
            py.detach(|| centrality_more::reverse_reach_counts(self.adj(true), self.n, &set_v));
        Ok(GroupPre {
            data,
            k: set_v.len(),
            rev_reach,
        })
    }

    // --- Batch 5: cores, clustering, distance and coloring ---

    /// `nx.onion_layers` as `(nodes, layers)` in NetworkX's dict order, or
    /// `None` if the graph has self-loops.
    fn onion_layers(&self, py: Python<'_>) -> Option<(Vec<u32>, Vec<u32>)> {
        py.detach(|| {
            if structure::has_self_loops(&self.succ, self.n) {
                None
            } else {
                Some(
                    cores_more::onion_layers(&self.succ, self.n)
                        .into_iter()
                        .unzip(),
                )
            }
        })
    }

    /// `nx.k_truss`: edges to drop as `(us, vs)`, whether each node is
    /// kept and whether each arc (CSR order) is, or `None` if the graph has
    /// self-loops.
    #[allow(clippy::type_complexity)]
    fn k_truss(
        &self,
        py: Python<'_>,
        need: u64,
    ) -> Option<(Vec<u32>, Vec<u32>, Vec<bool>, Vec<bool>)> {
        py.detach(|| {
            if structure::has_self_loops(&self.succ, self.n) {
                return None;
            }
            let (dropped, keep, arcs) = cores_more::k_truss(&self.succ, self.n, need);
            let (us, vs) = dropped.into_iter().unzip();
            Some((us, vs, keep, arcs))
        })
    }

    /// Nodes of `nx.k_corona` for core numbers `core`.
    fn k_corona(&self, py: Python<'_>, core: Vec<u32>, k: u32) -> PyResult<Vec<u32>> {
        if core.len() != self.n {
            return Err(PyValueError::new_err("one core number per node"));
        }
        Ok(py.detach(|| cores_more::k_corona(&self.succ, &core, k)))
    }

    /// `square_clustering` `(squares, potential)` per node (see
    /// `cores_more::square_clustering`).
    #[pyo3(signature = (nodes, old))]
    fn square_clustering(
        &self,
        py: Python<'_>,
        nodes: Option<Vec<u32>>,
        old: bool,
    ) -> PyResult<Vec<(i64, i64)>> {
        let nodes = self.sources_or_all(nodes)?;
        Ok(py.detach(|| cores_more::square_clustering(&self.succ, self.n, &nodes, old)))
    }

    /// `generalized_degree` counts (see `cores_more::generalized_degree`).
    fn generalized_degree(
        &self,
        py: Python<'_>,
        ws: Vec<u32>,
        ends: Vec<u32>,
    ) -> PyResult<Vec<u32>> {
        let ws = self.sources_or_all(Some(ws))?;
        if ends.windows(2).any(|w| w[0] > w[1])
            || ends.last().is_some_and(|&e| e as usize > ws.len())
        {
            return Err(PyValueError::new_err("bad group ends"));
        }
        Ok(py.detach(|| cores_more::generalized_degree(&self.succ, self.n, &ws, &ends)))
    }

    /// `nx.all_triangles` groups (see `cores_more::all_triangles`), for
    /// `nbunch` (all nodes, with their positions as ids, if `None`).
    #[allow(clippy::type_complexity)]
    fn all_triangles(
        &self,
        py: Python<'_>,
        nbunch: Option<Vec<u32>>,
    ) -> PyResult<(Vec<(u32, u32, u32, u32)>, Vec<u32>, Vec<bool>)> {
        let all = nbunch.is_none();
        let nbunch = self.sources_or_all(nbunch)?;
        Ok(py.detach(|| {
            let ids = if all {
                (0..self.n as i64).collect()
            } else {
                cores_more::triangle_ids(&self.succ, self.n, &nbunch)
            };
            cores_more::all_triangles(&self.succ, self.n, &nbunch, &ids)
        }))
    }

    /// `nx.intersection_array`: `(0, b, c)`, or `(1, ..)` / `(2, ..)` for
    /// NetworkX's two error messages (with and without the final period).
    #[pyo3(signature = (pairs_once, bound))]
    fn intersection_array(
        &self,
        py: Python<'_>,
        pairs_once: bool,
        bound: f64,
    ) -> (u8, Vec<u32>, Vec<u32>) {
        match py.detach(|| cores_more::intersection_array(&self.succ, self.n, pairs_once, bound)) {
            cores_more::Intersection::Ok(b, c) => (0, b, c),
            cores_more::Intersection::NotRegular => (1, vec![], vec![]),
            cores_more::Intersection::Inconsistent => (2, vec![], vec![]),
        }
    }

    /// `nx.tree.centroid` of a tree (the caller checks it is one).
    fn tree_centroid(&self, py: Python<'_>) -> Vec<u32> {
        if self.n == 0 {
            return vec![];
        }
        py.detach(|| cores_more::tree_centroid(&self.succ, self.n))
    }

    /// `harmonic_diameter`'s sum of inverse distances and whether any term
    /// was added. Raises ValueError on a negative cycle.
    #[pyo3(signature = (weight=None))]
    fn harmonic_sum(&self, py: Python<'_>, weight: Option<&str>) -> PyResult<(f64, bool)> {
        let w = self.weight_slice(weight, false)?;
        Ok(py.detach(|| cores_more::harmonic_sum(&self.succ, self.n, w))?)
    }

    /// Per source `(reached, Python sum of its distances)`, for `centroid`,
    /// and whether a negative cycle stopped the later sources.
    #[pyo3(signature = (weight, compensated))]
    #[allow(clippy::type_complexity)]
    fn distance_sums(
        &self,
        py: Python<'_>,
        weight: Option<&str>,
        compensated: bool,
    ) -> PyResult<(Vec<(usize, f64)>, bool)> {
        let w = self.weight_slice(weight, false)?;
        Ok(py.detach(|| cores_more::distance_sums(&self.succ, self.n, w, compensated)))
    }

    /// `greedy_color`'s colors for nodes processed in `order`.
    fn greedy_with_order(&self, py: Python<'_>, order: Vec<u32>) -> PyResult<Vec<u32>> {
        let order = self.sources_or_all(Some(order))?;
        Ok(py.detach(|| cores_more::greedy_with_order(&self.succ, self.n, &order)))
    }

    /// `greedy_color` (saturation_largest_first): order and colors.
    fn dsatur(&self, py: Python<'_>) -> (Vec<u32>, Vec<u32>) {
        let degree = self.degrees();
        py.detach(|| cores_more::dsatur(&self.succ, self.n, &degree))
    }

    /// `strategy_connected_sequential` order from each component's source.
    fn connected_sequential(
        &self,
        py: Python<'_>,
        sources: Vec<u32>,
        dfs: bool,
    ) -> PyResult<Vec<u32>> {
        let sources = self.sources_or_all(Some(sources))?;
        Ok(py.detach(|| cores_more::connected_sequential(&self.succ, self.n, &sources, dfs)))
    }

    /// `is_coloring` for color ids per node (negative: not colored):
    /// `(result, None)`, or `(False, node)` for the node NetworkX's lookup
    /// fails on.
    fn is_coloring(&self, py: Python<'_>, ids: Vec<i64>) -> PyResult<(bool, Option<u32>)> {
        if ids.len() != self.n {
            return Err(PyValueError::new_err("one color id per node"));
        }
        Ok(
            match py.detach(|| cores_more::is_coloring(&self.succ, self.n, self.directed, &ids)) {
                Ok(ok) => (ok, None),
                Err(node) => (false, Some(node)),
            },
        )
    }

    // --- Batch 6: trees and structural tests ---

    /// Positions of nodes with no edges (`nx.isolates`).
    fn isolates(&self) -> Vec<u32> {
        let pred = self.directed.then(|| self.adj(true));
        (0..self.n)
            .filter(|&v| {
                self.succ.neighbors(v).is_empty() && pred.is_none_or(|p| p.neighbors(v).is_empty())
            })
            .map(|v| v as u32)
            .collect()
    }

    /// NetworkX's `G.degree(v)` for one node.
    fn degree_of(&self, v: usize) -> PyResult<usize> {
        self.check_index(v)?;
        let out = self.succ.neighbors(v);
        Ok(if self.directed {
            out.len() + self.adj(true).neighbors(v).len()
        } else {
            out.len() + out.iter().filter(|&&w| w as usize == v).count()
        })
    }

    /// `nx.is_regular` on a non-empty graph.
    fn is_regular(&self) -> bool {
        if self.directed {
            let (ins, outs) = self.in_out_degrees();
            ins.iter().all(|&d| d == ins[0]) && outs.iter().all(|&d| d == outs[0])
        } else {
            let d = self.degrees();
            d.iter().all(|&x| x == d[0])
        }
    }

    /// Whether every degree equals `k` (`nx.is_k_regular`).
    fn all_degrees_equal(&self, k: i64) -> bool {
        self.degrees().iter().all(|&d| d as i64 == k)
    }

    /// `nx.is_tournament`: one arc between each pair, no self-loops.
    fn is_tournament(&self, py: Python<'_>) -> bool {
        let n = self.n;
        if structure::has_self_loops(&self.succ, n) {
            return false;
        }
        let pairs = (n as u128) * (n.saturating_sub(1) as u128) / 2;
        if self.succ.targets.len() as u128 != pairs {
            return false;
        }
        let pred = self.adj(true);
        py.detach(|| {
            let mut mark = vec![u32::MAX; n];
            (0..n).all(|u| {
                for &v in self.succ.neighbors(u) {
                    mark[v as usize] = u as u32;
                }
                pred.neighbors(u)
                    .iter()
                    .all(|&w| mark[w as usize] != u as u32)
            })
        })
    }

    /// `nx.chain_decomposition`: all chains' edges as `(us, vs)`, and
    /// where each chain ends.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (root=None))]
    fn chain_decomposition(
        &self,
        py: Python<'_>,
        root: Option<u32>,
    ) -> PyResult<(Vec<u32>, Vec<u32>, Vec<u32>)> {
        if let Some(r) = root {
            self.check_index(r as usize)?;
        }
        Ok(py.detach(|| {
            let (chains, _) = structure_more::chain_decomposition(&self.succ, self.n, root);
            let (mut us, mut vs, mut ends) = (Vec::new(), Vec::new(), Vec::new());
            for chain in chains {
                for (u, v) in chain {
                    us.push(u);
                    vs.push(v);
                }
                ends.push(us.len() as u32);
            }
            (us, vs, ends)
        }))
    }

    /// `nx.bridges` in yield order (with `root`: only its component, in
    /// `G.edges` order); `first_only` stops at the first.
    #[pyo3(signature = (root=None, first_only=false))]
    fn bridges(
        &self,
        py: Python<'_>,
        root: Option<u32>,
        first_only: bool,
    ) -> PyResult<(Vec<u32>, Vec<u32>)> {
        if let Some(r) = root {
            self.check_index(r as usize)?;
        }
        Ok(py.detach(|| {
            structure_more::bridges(&self.succ, self.n, root, first_only)
                .into_iter()
                .unzip()
        }))
    }

    /// `nx.local_bridges` edges (without spans) in yield order.
    fn local_bridges(&self, py: Python<'_>) -> (Vec<u32>, Vec<u32>) {
        py.detach(|| {
            structure_more::local_bridges(&self.succ, self.n)
                .into_iter()
                .unzip()
        })
    }

    /// Span of the local bridge `u`-`v` (`None`: infinite). Weighted spans
    /// need non-negative integer weights.
    #[pyo3(signature = (u, v, weight=None))]
    fn local_bridge_span(
        &self,
        py: Python<'_>,
        u: u32,
        v: u32,
        weight: Option<&str>,
    ) -> PyResult<Option<f64>> {
        self.check_index(u as usize)?;
        self.check_index(v as usize)?;
        let w = self.weight_slice(weight, false)?;
        Ok(py.detach(|| structure_more::hidden_edge_distance(&self.succ, self.n, u, v, w)))
    }

    /// Degree balance for the Euler tests: undirected `(odd, 0, 0)`;
    /// directed `(in - out == 1, out - in == 1, other unbalanced)` counts.
    fn euler_balance(&self) -> (usize, usize, usize) {
        if self.directed {
            let (ins, outs) = self.in_out_degrees();
            let (mut plus_in, mut plus_out, mut bad) = (0, 0, 0);
            for (i, o) in ins.into_iter().zip(outs) {
                if i == o + 1 {
                    plus_in += 1;
                } else if o == i + 1 {
                    plus_out += 1;
                } else if i != o {
                    bad += 1;
                }
            }
            (plus_in, plus_out, bad)
        } else {
            (self.degrees().iter().filter(|&&d| d % 2 == 1).count(), 0, 0)
        }
    }

    /// `_simplegraph_eulerian_circuit` from `source` on NetworkX's copy
    /// (undirected) or reverse (directed) of the graph: yielded pairs.
    fn euler_walk(&self, py: Python<'_>, source: u32) -> PyResult<(Vec<u32>, Vec<u32>)> {
        self.check_index(source as usize)?;
        Ok(py.detach(|| {
            let (adj, ids, m) = if self.directed {
                let adj = structure_more::transpose(&self.succ, self.n);
                let m = adj.targets.len();
                (adj, (0..m as u32).collect(), m)
            } else {
                structure_more::undirected_copy(&self.succ, self.n)
            };
            structure_more::euler_walk(&adj, &ids, m, source)
                .into_iter()
                .unzip()
        }))
    }

    /// `nx.cycle_basis`, starting from `root` if given, as lists of the
    /// objects in `nodes` (built here: the cycles can be long).
    #[pyo3(signature = (nodes, root=None))]
    fn cycle_basis<'py>(
        &self,
        py: Python<'py>,
        nodes: &Bound<'py, PyList>,
        root: Option<u32>,
    ) -> PyResult<Bound<'py, PyList>> {
        if let Some(r) = root {
            self.check_index(r as usize)?;
        }
        if nodes.len() != self.n {
            return Err(PyValueError::new_err("nodes must list every node"));
        }
        let cycles = py.detach(|| structure_more::cycle_basis(&self.succ, self.n, root));
        let objects: Vec<Bound<'py, PyAny>> = nodes.iter().collect();
        let out = PyList::empty(py);
        for cycle in cycles {
            out.append(PyList::new(
                py,
                cycle.iter().map(|&v| &objects[v as usize]),
            )?)?;
        }
        Ok(out)
    }

    /// `nx.girth` (`None`: no cycle).
    fn girth(&self, py: Python<'_>) -> Option<i64> {
        py.detach(|| structure_more::girth(&self.succ, self.n))
    }

    /// `nx.find_cycle`: orientation 0 plain, 1 forward, 2 reverse, 3
    /// ignore; `starts` defaults to all nodes. Returns `(u, v, reverse)`
    /// edges, or `None`. Reverse and ignore need exact in-edge order.
    #[pyo3(signature = (orientation, starts=None))]
    fn find_cycle(
        &self,
        py: Python<'_>,
        orientation: u8,
        starts: Option<Vec<u32>>,
    ) -> PyResult<Option<Vec<(u32, u32, bool)>>> {
        use structure_more::Orientation;
        let starts = self.sources_or_all(starts)?;
        let orientation = match orientation {
            0 => Orientation::Plain,
            1 => Orientation::Forward,
            2 => Orientation::Reverse,
            _ => Orientation::Ignore,
        };
        let pred =
            if self.directed && matches!(orientation, Orientation::Reverse | Orientation::Ignore) {
                Some(self.reverse_exact_order(None)?.0)
            } else {
                None
            };
        Ok(py.detach(|| {
            structure_more::find_cycle(
                &self.succ,
                pred,
                self.n,
                self.directed,
                orientation,
                &starts,
            )
        }))
    }

    /// `nx.immediate_dominators` from `start`: dict order and values.
    fn immediate_dominators(&self, py: Python<'_>, start: u32) -> PyResult<(Vec<u32>, Vec<u32>)> {
        self.check_index(start as usize)?;
        Ok(py.detach(|| {
            let post = traversal::dfs_postorder(&self.succ, self.n, &[start], self.n as i64);
            structure_more::immediate_dominators(self.adj(true), self.n, start, &post)
        }))
    }

    /// `nx.dominance_frontiers` from `start`: dict order, then the set
    /// additions `(v, u)` (meaning `df[v].add(u)`) in order. Needs exact
    /// in-edge order.
    #[allow(clippy::type_complexity)]
    fn dominance_frontiers(
        &self,
        py: Python<'_>,
        start: u32,
        new_style: bool,
    ) -> PyResult<(Vec<u32>, Vec<u32>, Vec<u32>)> {
        self.check_index(start as usize)?;
        let exact = self.reverse_exact_order(None)?.0;
        Ok(py.detach(|| {
            let post = traversal::dfs_postorder(&self.succ, self.n, &[start], self.n as i64);
            let (mut nodes, idom) =
                structure_more::immediate_dominators(self.adj(true), self.n, start, &post);
            let adds =
                structure_more::dominance_frontier_adds(exact, self.n, &nodes, &idom, new_style);
            if new_style {
                nodes.remove(0);
                nodes.push(start);
            }
            let (vs, us) = adds.into_iter().unzip();
            (nodes, vs, us)
        }))
    }

    /// `nx.to_prufer_sequence` for a tree with the node labelled `k` at
    /// position `pos[k]`: positions of the sequence.
    fn prufer_sequence(&self, py: Python<'_>, pos: Vec<u32>) -> PyResult<Vec<u32>> {
        let pos = self.sources_or_all(Some(pos))?;
        let degree = self.degrees();
        Ok(py.detach(|| structure_more::prufer(&self.succ, self.n, &pos, &degree)))
    }

    // --- Batch 7: shortest paths, DAG and cycle leftovers ---

    /// `(finite, int sums exact, has -0.0)` for edge attribute `weight`.
    #[pyo3(signature = (weight=None))]
    fn weight_flags(&self, weight: Option<&str>) -> PyResult<(bool, bool, bool)> {
        let w = self.weight_slice(weight, false)?;
        Ok((
            leftovers::weights_finite(w),
            leftovers::int_sums_exact(w),
            leftovers::has_negative_zero(w),
        ))
    }

    /// `floyd_warshall_predecessor_and_distance` (`tree`: 3.7's
    /// `floyd_warshall_tree`) as NetworkX's dicts `(pred, dist)`; `dist`'s
    /// rows come from `make_row()`. `old`: NetworkX 3.4/3.5's version.
    /// `None` where NetworkX raises `NetworkXUnbounded`.
    #[pyo3(signature = (nodes, make_row, weight, int_weights, old, tree, want_pred))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn floyd_warshall<'py>(
        &self,
        py: Python<'py>,
        nodes: &Bound<'py, PyList>,
        make_row: &Bound<'py, PyAny>,
        weight: Option<&str>,
        int_weights: bool,
        old: bool,
        tree: bool,
        want_pred: bool,
    ) -> PyResult<Option<(Bound<'py, PyDict>, Bound<'py, PyDict>)>> {
        if nodes.len() != self.n {
            return Err(PyValueError::new_err("nodes must list every node"));
        }
        let w = self.weight_slice(weight, false)?;
        let found = py.detach(|| {
            if tree {
                leftovers::floyd_warshall_tree(&self.succ, self.n, w, int_weights)
            } else {
                leftovers::floyd_warshall(&self.succ, self.n, w, int_weights, self.directed, old)
            }
        });
        let fw = match found {
            Ok(fw) => fw,
            Err(leftovers::FwError::Negative) => return Ok(None),
            Err(leftovers::FwError::Unsupported) => {
                return Err(PyNotImplementedError::new_err(
                    "rustnx can't reproduce this Floyd-Warshall case",
                ))
            }
        };
        let node = |i: u32| nodes.get_item(i as usize);
        let pred = PyDict::new(py);
        if want_pred {
            for &u in &fw.pred_rows {
                let row = PyDict::new(py);
                for &v in &fw.pred_order[u as usize] {
                    row.set_item(node(v)?, node(fw.pred[u as usize * self.n + v as usize])?)?;
                }
                pred.set_item(node(u)?, row)?;
            }
        }
        let dist = PyDict::new(py);
        for u in 0..self.n {
            let row = make_row.call0()?;
            let row_dict = row.cast::<PyDict>()?;
            for &v in &fw.dist_order[u] {
                let (x, int) = fw.value(u, v as usize);
                if int {
                    row_dict.set_item(node(v)?, x as i64)?;
                } else {
                    row_dict.set_item(node(v)?, x)?;
                }
            }
            dist.set_item(node(u as u32)?, row)?;
        }
        Ok(Some((pred, dist)))
    }

    /// `floyd_warshall_numpy`'s result as row-major f64 bytes, with node `i`
    /// at row `order[i]`; `None` where NetworkX raises `NetworkXUnbounded`.
    #[pyo3(signature = (order, weight, check_negative))]
    fn floyd_warshall_dense<'py>(
        &self,
        py: Python<'py>,
        order: Vec<u32>,
        weight: Option<&str>,
        check_negative: bool,
    ) -> PyResult<Option<Bound<'py, PyByteArray>>> {
        let n = self.n;
        let mut seen = vec![false; n];
        if order.len() != n
            || order
                .iter()
                .any(|&i| i as usize >= n || std::mem::replace(&mut seen[i as usize], true))
        {
            return Err(PyValueError::new_err("order must be a permutation"));
        }
        let w = self.weight_slice(weight, false)?;
        let found = py.detach(|| {
            let mut a = vec![f64::INFINITY; n * n];
            for u in 0..n {
                for e in self.succ.range(u) {
                    let v = self.succ.targets[e] as usize;
                    a[order[u] as usize * n + order[v] as usize] = w.map_or(1.0, |w| w[e]);
                }
            }
            leftovers::floyd_warshall_dense(&mut a, n, check_negative).map(|_| a)
        });
        Ok(found.ok().map(|a| {
            let bytes: Vec<u8> = a.iter().flat_map(|x| x.to_ne_bytes()).collect();
            PyByteArray::new(py, &bytes)
        }))
    }

    /// `johnson`'s Bellman-Ford potentials, or `None` on a negative cycle.
    #[pyo3(signature = (weight=None))]
    fn johnson_potentials(
        &self,
        py: Python<'_>,
        weight: Option<&str>,
    ) -> PyResult<Option<Vec<f64>>> {
        let w = self.weight_slice(weight, false)?;
        Ok(py.detach(|| leftovers::bellman_ford_all(&self.succ, self.n, w)))
    }

    /// `johnson`'s Dijkstra from each of `sources` on the reweighted graph,
    /// in parallel: `(pop order, parents in pop order, first-push order)`,
    /// or `None` where Dijkstra finds contradictory paths.
    #[pyo3(signature = (sources, h, weight=None))]
    #[allow(clippy::type_complexity)]
    fn johnson_trees(
        &self,
        py: Python<'_>,
        sources: Vec<u32>,
        h: Vec<f64>,
        weight: Option<&str>,
    ) -> PyResult<Vec<Option<(Vec<u32>, Vec<u32>, Vec<u32>)>>> {
        let sources = self.sources_or_all(Some(sources))?;
        if h.len() != self.n {
            return Err(PyValueError::new_err("one potential per node"));
        }
        let w = self.weight_slice(weight, false)?;
        Ok(py.detach(|| {
            let rw = leftovers::reweight(&self.succ, self.n, w, &h);
            sources
                .par_iter()
                .map(|&s| {
                    paths::dijkstra_tree(&self.succ, self.n, Some(&rw), s as usize, None, None)
                        .ok()
                        .map(|t| {
                            let parents = t.order.iter().map(|&v| t.parent[v as usize]).collect();
                            (t.order, parents, t.seen_order)
                        })
                })
                .collect()
        }))
    }

    /// `goldberg_radzik`'s state from `source`; Python runs the rounds.
    #[pyo3(signature = (source, int_weights, weight=None))]
    fn goldberg_radzik(
        &self,
        source: usize,
        int_weights: bool,
        weight: Option<&str>,
    ) -> PyResult<GoldbergRadzikState> {
        self.check_index(source)?;
        let w = self.weight_slice(weight, false)?;
        Ok(GoldbergRadzikState(leftovers::GoldbergRadzik::new(
            &self.succ,
            self.n,
            w,
            int_weights,
            source,
        )))
    }

    /// `nx.antichains` for topological order `topo` (positions), one batch
    /// at a time; `None` if `topo` isn't a topological order of G.
    fn antichains(&self, py: Python<'_>, topo: Vec<u32>) -> PyResult<Option<AntichainIter>> {
        let topo = self.sources_or_all(Some(topo))?;
        Ok(py.detach(|| {
            if !leftovers::is_topological_order(&self.succ, self.n, &topo) {
                return None;
            }
            let reach = leftovers::Reach::new(&self.succ, self.n, &topo);
            Some(AntichainIter(leftovers::Antichains::new(reach, &topo)))
        }))
    }

    /// `antichain_width` (NetworkX 3.7): `None` if G has a cycle.
    fn antichain_width(&self, py: Python<'_>) -> Option<usize> {
        py.detach(|| {
            let (generations, cycle) =
                directed::topological_generations(&self.succ, self.adj(true), self.n);
            if cycle {
                return None;
            }
            let topo: Vec<u32> = generations.into_iter().flatten().collect();
            let reach = leftovers::Reach::new(&self.succ, self.n, &topo);
            Some(self.n - leftovers::reach_matching_size(&reach, self.n))
        })
    }

    /// `_all_simple_edge_paths` from `source` to `targets` (positions;
    /// `extra`: the target set also holds objects that aren't nodes), with
    /// at most `limit` nodes on a path that is still extended.
    fn simple_paths(
        &self,
        source: usize,
        targets: Vec<u32>,
        extra: bool,
        limit: usize,
    ) -> PyResult<SimplePathIter> {
        self.check_index(source)?;
        let targets = self.sources_or_all(Some(targets))?;
        Ok(SimplePathIter(leftovers::SimplePaths::new(
            &self.succ,
            self.n,
            source as u32,
            &targets,
            extra,
            limit,
        )))
    }

    /// `shortest_simple_paths` from `source` to `target`, lazily.
    #[pyo3(signature = (source, target, compensated, weight=None))]
    fn shortest_simple_paths(
        &self,
        source: usize,
        target: usize,
        compensated: bool,
        weight: Option<&str>,
    ) -> PyResult<YenIter> {
        self.check_index(source)?;
        self.check_index(target)?;
        let (pred, w_pred) = self.reverse_exact_order(weight)?;
        let w_succ = self.weight_slice(weight, false)?;
        let weights = match (w_succ, w_pred) {
            (Some(a), Some(b)) => Some((a, b)),
            _ => None,
        };
        Ok(YenIter(leftovers::SimpleShortestPaths::new(
            &self.succ,
            pred,
            self.n,
            self.directed,
            weights,
            compensated,
            source as u32,
            target as u32,
        )))
    }

    /// Whether each consecutive pair of `path` is an edge.
    fn is_path(&self, path: Vec<u32>) -> PyResult<bool> {
        let path = self.sources_or_all(Some(path))?;
        Ok(path
            .windows(2)
            .all(|p| self.succ.neighbors(p[0] as usize).contains(&p[1])))
    }

    /// `_min_cycle_basis` of one component (see
    /// `leftovers::min_cycle_basis`): `(0, cycles)`, or `(1, None)` /
    /// `(2, None)` where `_dijkstra` / `bidirectional_dijkstra` find
    /// contradictory paths.
    #[pyo3(signature = (nodes, edges, chords, weight=None))]
    fn min_cycle_basis(
        &self,
        py: Python<'_>,
        nodes: Vec<u32>,
        edges: Vec<(u32, u32)>,
        chords: Vec<(u32, u32)>,
        weight: Option<&str>,
    ) -> PyResult<(u8, Option<Vec<Vec<u32>>>)> {
        let nodes = self.sources_or_all(Some(nodes))?;
        let w = self.weight_slice(weight, false)?;
        let mut weighted = Vec::with_capacity(edges.len());
        for &(u, v) in edges.iter().chain(&chords) {
            self.check_index(u as usize)?;
            self.check_index(v as usize)?;
        }
        for &(u, v) in &edges {
            let e = self
                .succ
                .range(u as usize)
                .find(|&e| self.succ.targets[e] == v)
                .ok_or_else(|| PyValueError::new_err("not an edge"))?;
            weighted.push((u, v, w.map_or(1.0, |w| w[e])));
        }
        match py.detach(|| leftovers::min_cycle_basis(self.n, &nodes, &weighted, &chords)) {
            Ok(cb) => Ok((0, Some(cb))),
            Err(leftovers::McbError::Contradictory) => Ok((1, None)),
            Err(leftovers::McbError::ContradictoryBidirectional) => Ok((2, None)),
            Err(leftovers::McbError::Unsupported) => Err(PyNotImplementedError::new_err(
                "rustnx can't reproduce this minimum cycle basis case",
            )),
        }
    }

    // --- Batch 8: trees, branchings and lowest common ancestors ---

    /// NetworkX's `maximum_branching` on edges `us[k] -> vs[k]` (key `k`)
    /// with Python int or float `weights` and partition states (0 open,
    /// 1 included, 2 excluded): the final branching's keys and, for each
    /// contraction from the last, its circuit and the key removed. `None`
    /// where rustnx can't follow NetworkX (other weight types, ints leaving
    /// `i64`, inputs on which NetworkX raises).
    #[staticmethod]
    #[allow(clippy::type_complexity)]
    fn edmonds<'py>(
        py: Python<'py>,
        n: usize,
        us: Vec<u32>,
        vs: Vec<u32>,
        weights: Vec<Bound<'py, PyAny>>,
        part: Vec<u8>,
    ) -> PyResult<Option<(Vec<u32>, Vec<(Vec<u32>, u32)>)>> {
        let m = us.len();
        if vs.len() != m || weights.len() != m || part.len() != m {
            return Err(PyValueError::new_err("edge arrays differ in length"));
        }
        if us.iter().chain(&vs).any(|&x| x as usize >= n) {
            return Err(PyIndexError::new_err("node index out of range"));
        }
        let Some(w) = weights.iter().map(py_num).collect::<Option<Vec<_>>>() else {
            return Ok(None);
        };
        Ok(py
            .detach(|| trees_more::edmonds(n, &us, &vs, &w, &part))
            .map(|b| (b.initial, b.log)))
    }

    /// `greedy_branching`'s edges (indices, in the order they are added),
    /// sorting by `(weight, rank[u], rank[v])`. `None` for weights that
    /// aren't Python ints or floats, or NaN.
    #[staticmethod]
    fn greedy_branching<'py>(
        py: Python<'py>,
        us: Vec<u32>,
        vs: Vec<u32>,
        weights: Vec<Bound<'py, PyAny>>,
        rank: Vec<u32>,
        maximum: bool,
    ) -> PyResult<Option<Vec<u32>>> {
        let n = rank.len();
        if vs.len() != us.len() || weights.len() != us.len() {
            return Err(PyValueError::new_err("edge arrays differ in length"));
        }
        if us.iter().chain(&vs).any(|&x| x as usize >= n) {
            return Err(PyIndexError::new_err("node index out of range"));
        }
        let w: Option<Vec<_>> = weights
            .iter()
            .map(|x| py_num(x).filter(|w| !matches!(w, trees_more::Num::Float(f) if f.is_nan())))
            .collect();
        let Some(w) = w else { return Ok(None) };
        Ok(Some(py.detach(|| {
            trees_more::greedy_branching(n, &us, &vs, &w, &rank, maximum)
        })))
    }

    /// `greedy_branching` on the converted `weight` (exact for ints and
    /// floats alike): kept edges as `(us, vs)`, in the order added.
    fn greedy_branching_edges(
        &self,
        py: Python<'_>,
        weight: &str,
        rank: Vec<u32>,
        maximum: bool,
    ) -> PyResult<(Vec<u32>, Vec<u32>)> {
        if rank.len() != self.n {
            return Err(PyValueError::new_err("one rank per node is needed"));
        }
        let w = self.weight_slice(Some(weight), false)?.unwrap_or(&[]);
        Ok(py.detach(|| {
            let (mut us, mut vs, mut ws) = (Vec::new(), Vec::new(), Vec::new());
            for u in 0..self.n {
                for e in self.succ.range(u) {
                    let v = self.succ.targets[e];
                    if self.directed || v as usize >= u {
                        us.push(u as u32);
                        vs.push(v);
                        ws.push(trees_more::Num::Float(w[e]));
                    }
                }
            }
            trees_more::greedy_branching(self.n, &us, &vs, &ws, &rank, maximum)
                .into_iter()
                .map(|i| (us[i as usize], vs[i as usize]))
                .unzip()
        }))
    }

    /// `prim_mst_edges` (undirected), growing a tree from each of `starts`.
    #[pyo3(signature = (starts, weight=None, minimum=true))]
    fn prim_edges(
        &self,
        py: Python<'_>,
        starts: Vec<u32>,
        weight: Option<&str>,
        minimum: bool,
    ) -> PyResult<(Vec<u32>, Vec<u32>)> {
        let starts = self.sources_or_all(Some(starts))?;
        let w = self.weight_slice(weight, false)?;
        let sign = if minimum { 1.0 } else { -1.0 };
        Ok(py.detach(|| {
            trees_more::prim(&self.succ, w, &starts, sign)
                .into_iter()
                .unzip()
        }))
    }

    /// Kruskal with a partition: `state` per edge in `G.edges` order (0
    /// open, 1 included, 2 excluded); kept edges as `(us, vs)`.
    #[pyo3(signature = (state, weight=None, maximum=false))]
    fn kruskal_partition(
        &self,
        py: Python<'_>,
        state: Vec<u8>,
        weight: Option<&str>,
        maximum: bool,
    ) -> PyResult<(Vec<u32>, Vec<u32>)> {
        let w = self.weight_slice(weight, false)?;
        let mut edges = Vec::new();
        for u in 0..self.n {
            for e in self.succ.range(u) {
                let v = self.succ.targets[e];
                if self.directed || v as usize >= u {
                    edges.push((u as u32, v, w.map_or(1.0, |w| w[e])));
                }
            }
        }
        if state.len() != edges.len() {
            return Err(PyValueError::new_err("one state per edge is needed"));
        }
        Ok(py.detach(|| {
            trees_more::kruskal_partition(self.n, &edges, &state, maximum)
                .into_iter()
                .map(|i| (edges[i as usize].0, edges[i as usize].1))
                .unzip()
        }))
    }

    /// `sum()` of the attribute over `G.edges` (`branching_weight`): an
    /// int if every value is an int, else CPython's float `sum()`.
    fn edge_weight_sum<'py>(
        &self,
        py: Python<'py>,
        weight: &str,
        compensated: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let w = self.weight_slice(Some(weight), false)?.unwrap_or(&[]);
        let values = (0..self.n).flat_map(|u| {
            self.succ
                .range(u)
                .filter(move |&e| self.directed || self.succ.targets[e] as usize >= u)
                .map(|e| w[e])
        });
        if self.weights_info(Some(weight)).0 {
            let total: i128 = values.map(|x| x as i128).sum();
            Ok(total.into_pyobject(py)?.into_any())
        } else {
            let total = spectral::py_sum(values, compensated);
            Ok(total.into_pyobject(py)?.into_any())
        }
    }

    /// `from_prufer_sequence`'s edges (before the orphans' edge), with `-1`
    /// marking out-of-range entries: `(us, vs, error)`, where `error` is
    /// the index of the first bad entry, or `-2` if NetworkX's search for a
    /// leaf would fail.
    #[staticmethod]
    fn prufer_edges(seq: Vec<i64>) -> PyResult<(Vec<u32>, Vec<u32>, i64)> {
        let n = seq.len() + 2;
        if seq.iter().any(|&v| v < -1 || v >= n as i64) {
            return Err(PyValueError::new_err("entries must be in -1..n"));
        }
        Ok(match trees_more::prufer_edges(&seq) {
            Ok(edges) => {
                let (us, vs) = edges.into_iter().unzip();
                (us, vs, -1)
            }
            Err(usize::MAX) => (Vec::new(), Vec::new(), -2),
            Err(i) => (Vec::new(), Vec::new(), i as i64),
        })
    }

    /// `from_nested_tuple(sequence, sensible_relabeling)`: node order and
    /// edges in insertion order. `None` if a level has no `len()` or can't
    /// be iterated, or nesting is deeper than `max_depth`.
    #[staticmethod]
    #[allow(clippy::type_complexity)]
    fn nested_tuple_tree(
        py: Python<'_>,
        sequence: &Bound<'_, PyAny>,
        sensible: bool,
        max_depth: usize,
    ) -> PyResult<Option<(Vec<u32>, Vec<u32>, Vec<u32>)>> {
        let mut children = Vec::new();
        if parse_nested(sequence, 0, max_depth, &mut children).is_none() {
            return Ok(None);
        }
        let shape = trees_more::Shape { children };
        let (order, edges) = py.detach(|| trees_more::nested_tuple_tree(&shape, sensible));
        let (us, vs) = edges.into_iter().unzip();
        Ok(Some((order, us, vs)))
    }

    /// `to_nested_tuple(T, root, canonical_form=True)` for a tree, or `None`
    /// if the tree is more than `max_height` levels deep.
    fn canonical_nested_tuple<'py>(
        &self,
        py: Python<'py>,
        root: usize,
        max_height: usize,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.check_index(root)?;
        let (post, children) =
            py.detach(|| trees_more::canonical_children(&self.succ, root as u32));
        let mut depth = vec![0usize; self.n];
        for &x in post.iter().rev() {
            for &c in &children[x as usize] {
                depth[c as usize] = depth[x as usize] + 1;
            }
        }
        if depth.iter().any(|&d| d > max_height) {
            return Ok(None);
        }
        let mut tuples: Vec<Option<Bound<'py, PyTuple>>> = vec![None; self.n];
        for &x in &post {
            let items: Vec<Bound<'py, PyTuple>> = children[x as usize]
                .iter()
                .map(|&c| tuples[c as usize].take().expect("child tuple"))
                .collect();
            tuples[x as usize] = Some(PyTuple::new(py, items)?);
        }
        Ok(tuples[root].take().map(|t| t.into_any()))
    }

    /// Lowest common ancestor queries for `all_pairs_lowest_common_ancestor`.
    fn dag_lca(&self) -> DagLca {
        let copy = |c: &graph::Csr| graph::Csr {
            offsets: c.offsets.clone(),
            targets: c.targets.clone(),
        };
        DagLca(trees_more::DagLca::new(
            copy(&self.succ),
            copy(self.adj(true)),
        ))
    }

    /// `tree_all_pairs_lowest_common_ancestor(G, root)` without `pairs`;
    /// each node's parent is its first predecessor in NetworkX's order.
    fn tree_lca(&self, root: usize) -> PyResult<TreeLca> {
        self.check_index(root)?;
        let (pred, _) = self.reverse_exact_order(None)?;
        let parent = (0..self.n)
            .map(|v| pred.neighbors(v).first().copied().unwrap_or(u32::MAX))
            .collect();
        Ok(TreeLca(trees_more::TreeLca::new(
            &self.succ,
            root as u32,
            parent,
        )))
    }

    /// `boruvka_mst_edges`, run a round at a time from Python.
    #[pyo3(signature = (weight=None, minimum=true))]
    fn boruvka(&self, weight: Option<&str>, minimum: bool) -> PyResult<Boruvka> {
        let w = self.weight_slice(weight, false)?.map(|w| w.to_vec());
        let adj = graph::Csr {
            offsets: self.succ.offsets.clone(),
            targets: self.succ.targets.clone(),
        };
        let sign = if minimum { 1.0 } else { -1.0 };
        Ok(Boruvka(trees_more::Boruvka::new(adj, w, sign)))
    }

    // --- Batch 9: planarity, chordal graphs and graph classes ---

    /// NetworkX's left-right planarity test: `(planar, embedding, depth)`.
    /// `embed` 0 leaves the embedding out; 1 gives each node's first
    /// half-edges in order, then the `(kind, a, b, ref)` calls of the
    /// depth-first phase (kind 0: `add_half_edge_first(a, b)`, 1:
    /// `ccw=ref`, 2: `cw=ref`); 2 gives the dicts those calls build (see
    /// `graph_classes::Layout`). `depth` bounds the recursive variant's
    /// recursion.
    fn planarity<'py>(&self, py: Python<'py>, embed: u8) -> PyResult<Bound<'py, PyTuple>> {
        // The embedding is always built: NetworkX builds it for a planar
        // graph even when only the answer is wanted (`is_planar`).
        let result = py.detach(|| {
            let adj = graph_classes::planarity_graph(&self.succ, self.n, self.directed);
            let result = graph_classes::lr_planarity(&adj, true)?;
            let layout = match (&result.embedding, embed) {
                (Some(emb), 2) => Some(graph_classes::embedding_layout(self.n, emb)?),
                _ => None,
            };
            Ok::<_, graph_classes::Bail>((result, layout))
        });
        let (result, layout) = result.map_err(|_| planarity_bail())?;
        let embedding: Py<PyAny> = match (result.embedding, layout) {
            (_, Some(l)) => (
                l.offsets,
                l.target,
                l.cw,
                l.ccw,
                l.ccw_first,
                l.pred_offsets,
                l.pred,
            )
                .into_pyobject(py)?
                .into_any()
                .unbind(),
            (Some(emb), None) if embed == 1 => {
                let calls: Vec<(u8, u32, u32, u32)> = emb
                    .calls
                    .into_iter()
                    .map(|call| match call {
                        graph_classes::HalfEdge::First(a, b) => (0, a, b, 0),
                        graph_classes::HalfEdge::Ccw(a, b, r) => (1, a, b, r),
                        graph_classes::HalfEdge::Cw(a, b, r) => (2, a, b, r),
                    })
                    .collect();
                (emb.ordered, calls).into_pyobject(py)?.into_any().unbind()
            }
            _ => py.None(),
        };
        (result.planar, embedding, result.depth).into_pyobject(py)
    }

    /// NetworkX's `get_counterexample`: the edges it adds to the
    /// counterexample, in order (`None` if planar), and the recursion
    /// depth the recursive variant would need.
    #[allow(clippy::type_complexity)]
    fn planarity_counterexample(
        &self,
        py: Python<'_>,
    ) -> PyResult<(Option<(Vec<u32>, Vec<u32>)>, usize)> {
        let result = py.detach(|| {
            let adj = graph_classes::planarity_graph(&self.succ, self.n, self.directed);
            graph_classes::counterexample(&adj)
        });
        let (edges, depth) = result.map_err(|_| planarity_bail())?;
        Ok((edges.map(|e| e.into_iter().unzip()), depth))
    }

    /// `(is_chordal, treewidth)` of an undirected graph without self-loops.
    fn chordal(&self, py: Python<'_>) -> (bool, u32) {
        py.detach(|| graph_classes::chordal(&self.succ, self.n))
    }

    /// `complete_to_chordal_graph` on a non-chordal graph without
    /// self-loops: each node's alpha, and the chords `(z, y)` in order.
    fn complete_to_chordal(&self, py: Python<'_>) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let (alpha, chords) = py.detach(|| graph_classes::complete_to_chordal(&self.succ, self.n));
        let (zs, ys) = chords.into_iter().unzip();
        (alpha, zs, ys)
    }

    /// `nx.is_at_free` on an undirected graph.
    fn is_at_free(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| graph_classes::is_at_free(&self.succ, self.n))
            .ok_or_else(|| PyNotImplementedError::new_err("graph too large for is_at_free"))
    }

    /// `nx.tournament.is_reachable` for positions (`None`: not in G).
    #[pyo3(signature = (s=None, t=None))]
    fn tournament_reachable(
        &self,
        py: Python<'_>,
        s: Option<u32>,
        t: Option<u32>,
    ) -> PyResult<bool> {
        let s = s.unwrap_or(u32::MAX);
        let t = t.unwrap_or(u32::MAX);
        for v in [s, t] {
            if v != u32::MAX {
                self.check_index(v as usize)?;
            }
        }
        Ok(py.detach(|| graph_classes::tournament_reachable(&self.succ, self.n, s, t)))
    }

    /// `nx.is_perfect_graph` on an undirected graph.
    fn is_perfect(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| graph_classes::is_perfect(&self.succ, self.n))
            .ok_or_else(|| PyNotImplementedError::new_err("graph too large for is_perfect_graph"))
    }

    /// Out-degrees in increasing order (`nx.tournament.score_sequence`).
    fn sorted_out_degrees(&self) -> Vec<usize> {
        let mut degrees: Vec<usize> = (0..self.n).map(|v| self.succ.neighbors(v).len()).collect();
        degrees.sort_unstable();
        degrees
    }

    /// `nx.tournament.is_strongly_connected`.
    fn tournament_strongly_connected(&self, py: Python<'_>) -> bool {
        py.detach(|| graph_classes::tournament_strongly_connected(&self.succ, self.n))
    }

    // --- Batch 10: triads, d-separation, degree sequences and matching ---

    /// `nx.triadic_census` counts in `TRIAD_NAMES` order, over `nodeset`
    /// (all nodes if `None`).
    #[pyo3(signature = (nodeset=None))]
    fn triadic_census(&self, py: Python<'_>, nodeset: Option<Vec<u32>>) -> PyResult<Vec<i128>> {
        let nodeset = self.sources_or_all(nodeset)?;
        let pred = self.adj(true);
        Ok(py.detach(|| matching::triadic_census(&self.succ, pred, self.n, &nodeset).to_vec()))
    }

    /// `nx.is_d_separator` on a DAG, after NetworkX's input checks.
    fn is_d_separator(
        &self,
        py: Python<'_>,
        x: Vec<u32>,
        y: Vec<u32>,
        z: Vec<u32>,
    ) -> PyResult<bool> {
        self.membership(&x)?;
        self.membership(&y)?;
        self.membership(&z)?;
        let pred = self.adj(true);
        Ok(py.detach(|| matching::is_d_separator(&self.succ, pred, self.n, &x, &y, &z)))
    }

    /// `nx.ancestors` of each of `nodes`, each in its set's insertion order
    /// (needs exact in-edge order).
    fn ancestor_lists(&self, py: Python<'_>, nodes: Vec<u32>) -> PyResult<Vec<Vec<u32>>> {
        self.membership(&nodes)?;
        let pred = self.reverse_exact_order(None)?.0;
        Ok(py.detach(|| {
            let mut seen = Vec::new();
            nodes
                .iter()
                .enumerate()
                .map(|(i, &v)| matching::ancestors_in_order(pred, self.n, v, &mut seen, i as u32))
                .collect()
        }))
    }

    /// `_reachable(G, x, a, z)` of `networkx.algorithms.d_separation`, with
    /// `a` the nodes in `nodeset` and their ancestors: the nodes of
    /// `processed` in order (needs exact in-edge order).
    fn d_reachable(
        &self,
        py: Python<'_>,
        x: Vec<u32>,
        nodeset: Vec<u32>,
        z: Vec<u32>,
    ) -> PyResult<Vec<u32>> {
        self.membership(&x)?;
        let in_z = self.membership(&z)?;
        self.membership(&nodeset)?;
        let pred = self.reverse_exact_order(None)?.0;
        let transposed = self.adj(true);
        Ok(py.detach(|| {
            let anc = matching::closure_mask(transposed, self.n, &nodeset);
            matching::d_reachable(&self.succ, pred, self.n, &x, &anc, &in_z)
        }))
    }

    /// `nx.is_minimal_d_separator` on a DAG, after its input checks.
    fn is_minimal_d_separator(
        &self,
        py: Python<'_>,
        x: Vec<u32>,
        y: Vec<u32>,
        z: Vec<u32>,
        included: Vec<u32>,
    ) -> PyResult<bool> {
        for nodes in [&x, &y, &z, &included] {
            self.membership(nodes)?;
        }
        let pred = self.adj(true);
        Ok(py.detach(|| {
            matching::is_minimal_d_separator(&self.succ, pred, self.n, &x, &y, &z, &included)
        }))
    }

    /// Successors of `nodes`, each once, first occurrence first.
    fn neighbor_union(&self, py: Python<'_>, nodes: Vec<u32>) -> PyResult<Vec<u32>> {
        self.membership(&nodes)?;
        Ok(py.detach(|| matching::neighbor_union(&self.succ, self.n, &nodes)))
    }

    /// `nx.edge_boundary` without data: `(index into order, neighbor)`.
    #[pyo3(signature = (order, nset2=None))]
    fn edge_boundary(
        &self,
        py: Python<'_>,
        order: Vec<u32>,
        nset2: Option<Vec<u32>>,
    ) -> PyResult<(Vec<u32>, Vec<u32>)> {
        self.membership(&order)?;
        if let Some(s) = &nset2 {
            self.membership(s)?;
        }
        Ok(py.detach(|| {
            matching::edge_boundary(&self.succ, self.n, self.directed, &order, nset2.as_deref())
                .into_iter()
                .unzip()
        }))
    }

    /// Index of the first pair where `nx.is_matching` returns False.
    fn first_matching_failure(&self, us: Vec<u32>, vs: Vec<u32>) -> PyResult<Option<usize>> {
        self.membership(&us)?;
        self.membership(&vs)?;
        Ok(matching::first_matching_failure(
            &self.succ, self.n, &us, &vs,
        ))
    }

    /// Whether an edge (not a self-loop) has neither end in `matched`.
    fn has_unmatched_edge(&self, py: Python<'_>, matched: Vec<u32>) -> PyResult<bool> {
        self.membership(&matched)?;
        Ok(py.detach(|| matching::has_unmatched_edge(&self.succ, self.n, &matched)))
    }

    /// `nx.maximal_matching`: chosen edges in order.
    fn maximal_matching(&self, py: Python<'_>) -> (Vec<u32>, Vec<u32>) {
        py.detach(|| {
            matching::maximal_matching(&self.succ, self.n)
                .into_iter()
                .unzip()
        })
    }

    /// `nx.max_weight_matching` (or, with `inverted`, `min_weight_matching`)
    /// as matching pairs in set-insertion order. `None` if integer weights
    /// are too large for f64 arithmetic to match Python's exact ints.
    #[pyo3(signature = (weight=None, maxcardinality=false, inverted=false))]
    fn max_weight_matching(
        &self,
        py: Python<'_>,
        weight: Option<&str>,
        maxcardinality: bool,
        inverted: bool,
    ) -> PyResult<Option<(Vec<u32>, Vec<u32>)>> {
        let w = self.weight_slice(weight, false)?;
        let any_int = weight
            .and_then(|a| self.weights.get(a))
            .is_none_or(|x| x.any_int);
        if any_int && w.is_some_and(|w| w.iter().any(|x| x.abs() > (1u64 << 49) as f64)) {
            return Ok(None);
        }
        Ok(Some(py.detach(|| {
            if inverted {
                let (g, nodes) = matching::inverted_graph(&self.succ, self.n, w);
                matching::max_weight_matching(&g, true)
                    .into_iter()
                    .map(|(u, v)| (nodes[u as usize], nodes[v as usize]))
                    .unzip()
            } else {
                let g = matching::WeightedAdj::from_csr(&self.succ, w);
                matching::max_weight_matching(&g, maxcardinality)
                    .into_iter()
                    .unzip()
            }
        })))
    }

    /// Whether `nodes` and their successors cover every node.
    fn is_dominating(&self, py: Python<'_>, nodes: Vec<u32>) -> PyResult<bool> {
        self.membership(&nodes)?;
        Ok(py.detach(|| matching::is_dominating(&self.succ, self.n, &nodes)))
    }

    /// Whether the subgraph induced by `nodes` (non-empty) is connected.
    fn induced_connected(&self, py: Python<'_>, nodes: Vec<u32>) -> PyResult<bool> {
        self.membership(&nodes)?;
        if nodes.is_empty() {
            return Err(PyValueError::new_err("nodes must not be empty"));
        }
        Ok(py.detach(|| matching::induced_connected(&self.succ, self.n, &nodes)))
    }

    /// `nx.connected_dominating_set` (connected, two or more nodes): nodes
    /// in the order they join.
    fn connected_dominating_set(&self, py: Python<'_>) -> PyResult<Vec<u32>> {
        let degree = self.degrees();
        py.detach(|| matching::connected_dominating_set(&self.succ, self.n, &degree))
            .ok_or_else(|| PyIndexError::new_err("index out of range"))
    }

    /// `nx.enumerate_all_cliques` as a resumable queue.
    fn all_cliques(&self, py: Python<'_>) -> CliqueQueue {
        CliqueQueue(py.detach(|| matching::AllCliques::new(&self.succ, self.n)))
    }

    /// `nx.node_clique_number` for each of `nodes`.
    fn node_clique_numbers(&self, py: Python<'_>, nodes: Vec<u32>) -> PyResult<Vec<u32>> {
        self.membership(&nodes)?;
        Ok(py.detach(|| matching::node_clique_numbers(&self.succ, self.n, &nodes)))
    }

    /// `nx.max_weight_clique` with integer node weights (`None`: all 1):
    /// the clique and its weight.
    #[pyo3(signature = (weights=None))]
    fn max_weight_clique(
        &self,
        py: Python<'_>,
        weights: Option<Vec<i64>>,
    ) -> PyResult<(Vec<u32>, i64)> {
        let weights = weights.unwrap_or_else(|| vec![1; self.n]);
        if weights.len() != self.n {
            return Err(PyValueError::new_err("one weight per node"));
        }
        let total: i128 = weights.iter().filter(|&&w| w > 0).map(|&w| w as i128).sum();
        if total > (1i128 << 62) {
            return Err(PyNotImplementedError::new_err("node weights are too large"));
        }
        let degree = self.degrees();
        Ok(py.detach(|| matching::MaxWeightClique::run(&self.succ, self.n, &weights, &degree)))
    }

    // --- Batch 11: isomorphism and graph hashing ---

    /// Weisfeiler-Lehman steps from the initial `labels`: the graph hash,
    /// or with `per_node` each node's hashed labels (after
    /// `initial_hashes` copies of its hashed initial label). `edge_text`
    /// holds `str(G[u][v][edge_attr])` per `succ` entry; `split` is
    /// NetworkX 3.5+'s directed aggregation (successors then predecessors,
    /// prefixed `s_`/`p_` without edge text).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (labels, edge_text, split, steps, digest_size, initial_hashes=0, per_node=false))]
    fn wl_hashes(
        &self,
        py: Python<'_>,
        labels: Vec<String>,
        edge_text: Option<Vec<String>>,
        split: bool,
        steps: usize,
        digest_size: usize,
        initial_hashes: usize,
        per_node: bool,
    ) -> PyResult<(String, Vec<Vec<String>>)> {
        if labels.len() != self.n || !(1..=64).contains(&digest_size) {
            return Err(PyValueError::new_err("bad labels or digest size"));
        }
        if edge_text
            .as_ref()
            .is_some_and(|t| t.len() != self.succ.targets.len())
        {
            return Err(PyValueError::new_err("edge text doesn't match the edges"));
        }
        Ok(py.detach(|| {
            use isomorphism::{WlGroup, WlPrefix};
            let n = self.n;
            let text = edge_text.as_deref();
            let prefix = |same: &'static str| match text {
                Some(t) => WlPrefix::PerEdge(t),
                None => WlPrefix::Same(same),
            };
            let mut groups = Vec::new();
            if split {
                groups.push(WlGroup {
                    rows: isomorphism::rows_with_ids(&self.succ, n),
                    prefix: prefix("s_"),
                });
                groups.push(WlGroup {
                    rows: isomorphism::pred_rows_with_ids(&self.succ, n),
                    prefix: prefix("p_"),
                });
            } else {
                groups.push(WlGroup {
                    rows: isomorphism::rows_with_ids(&self.succ, n),
                    prefix: prefix(""),
                });
            }
            let mut history: Vec<Vec<String>> = Vec::new();
            if initial_hashes > 0 {
                let first: Vec<String> = labels
                    .par_iter()
                    .map(|l| isomorphism::blake2b_hex(l.as_bytes(), digest_size))
                    .collect();
                for _ in 1..initial_hashes {
                    history.push(first.clone());
                }
                history.push(first);
            }
            let start = history.len();
            for i in 0..steps {
                let previous = if i == 0 {
                    &labels
                } else {
                    &history[history.len() - 1]
                };
                let next = isomorphism::wl_step(previous, &groups, digest_size);
                history.push(next);
            }
            if !per_node {
                let text = isomorphism::wl_counts_text(&history[start..]);
                let graph_hash = isomorphism::blake2b_hex(text.as_bytes(), digest_size);
                return (graph_hash, Vec::new());
            }
            // Per node, its labels across steps.
            let per_node: Vec<Vec<String>> = (0..n)
                .into_par_iter()
                .map(|v| history.iter().map(|step| step[v].clone()).collect())
                .collect();
            (String::new(), per_node)
        }))
    }

    /// Whether the sorted per-node property rows of the two graphs agree,
    /// as `could_be_isomorphic` builds them: degree, triangles (undirected
    /// graphs only) and the number of maximal cliques, for the columns
    /// asked for.
    fn iso_tables_match(
        &self,
        py: Python<'_>,
        other: &Bound<'_, CoreGraph>,
        degree: bool,
        triangles: bool,
        cliques: bool,
    ) -> bool {
        let other = other.get();
        let deg_a = degree.then(|| self.degrees());
        let deg_b = degree.then(|| other.degrees());
        py.detach(|| {
            let table = |g: &CoreGraph, deg: Option<Vec<usize>>| {
                let n = g.n;
                let tri =
                    triangles.then(|| cluster::triangle_counts(&g.succ, None, n, &all_nodes(n)));
                let clq = cliques.then(|| isomorphism::maximal_clique_counts(&g.succ, n));
                let mut rows: Vec<[u64; 3]> = (0..n)
                    .map(|v| {
                        [
                            deg.as_ref().map_or(0, |d| d[v] as u64),
                            tri.as_ref().map_or(0, |t| t[v].0 / 2),
                            clq.as_ref().map_or(0, |c| c[v]),
                        ]
                    })
                    .collect();
                rows.par_sort_unstable();
                rows
            };
            table(self, deg_a) == table(other, deg_b)
        })
    }

    /// Whether `small` maps into this graph: `problem` 0 isomorphism,
    /// 1 induced subgraph isomorphism, 2 monomorphism. Labels are integer
    /// classes (equal labels, equal classes); `None` means all equal.
    fn has_morphism(
        &self,
        py: Python<'_>,
        small: &Bound<'_, CoreGraph>,
        labels: Option<Vec<u32>>,
        small_labels: Option<Vec<u32>>,
        problem: u8,
    ) -> PyResult<bool> {
        let small = small.get();
        if small.directed != self.directed {
            return Err(PyValueError::new_err("graphs differ in directedness"));
        }
        let labels = labels.unwrap_or_else(|| vec![0; self.n]);
        let small_labels = small_labels.unwrap_or_else(|| vec![0; small.n]);
        if labels.len() != self.n || small_labels.len() != small.n {
            return Err(PyValueError::new_err("wrong number of labels"));
        }
        let problem = match problem {
            0 => isomorphism::Problem::Iso,
            1 => isomorphism::Problem::Induced,
            _ => isomorphism::Problem::Mono,
        };
        let directed = self.directed;
        Ok(py.detach(|| {
            let big = isomorphism::Side {
                succ: &self.succ,
                pred: directed.then(|| self.adj(true)),
                n: self.n,
                labels: &labels,
            };
            let sm = isomorphism::Side {
                succ: &small.succ,
                pred: directed.then(|| small.adj(true)),
                n: small.n,
                labels: &small_labels,
            };
            isomorphism::has_morphism(&sm, &big, directed, problem)
        }))
    }

    /// Centers of a tree (undirected, connected, n - 1 edges), in order.
    fn tree_centers(&self, py: Python<'_>) -> PyResult<Vec<u32>> {
        if self.n == 0 {
            return Err(PyValueError::new_err("empty graph"));
        }
        Ok(py.detach(|| isomorphism::tree_centers(&self.succ, self.n)))
    }

    /// Height of the tree below `root`.
    fn tree_height(&self, py: Python<'_>, root: usize) -> PyResult<u32> {
        self.check_index(root)?;
        Ok(py.detach(|| isomorphism::tree_height(&self.succ, self.n, root)))
    }

    /// `rooted_tree_isomorphism` of this tree from `root` and `other` from
    /// `other_root`: position pairs in NetworkX's output order.
    fn rooted_tree_isomorphism(
        &self,
        py: Python<'_>,
        root: usize,
        other: &Bound<'_, CoreGraph>,
        other_root: usize,
        descending: bool,
    ) -> PyResult<Vec<(u32, u32)>> {
        let other = other.get();
        self.check_index(root)?;
        other.check_index(other_root)?;
        Ok(py.detach(|| {
            isomorphism::rooted_tree_isomorphism(
                &self.succ,
                self.n,
                root,
                &other.succ,
                other.n,
                other_root,
                descending,
            )
        }))
    }

    // Bipartite graphs (todo item 37), in the same batch.

    /// `nx.bipartite.color`: nodes in dict order and their colors, or
    /// `None` if the graph isn't bipartite. Directed graphs need exact
    /// in-edge order loaded.
    #[allow(clippy::type_complexity)]
    fn bipartite_color(&self, py: Python<'_>) -> PyResult<Option<(Vec<u32>, Vec<u8>)>> {
        let pred = if self.directed {
            Some(self.reverse_exact_order(None)?.0)
        } else {
            None
        };
        let degree = self.degrees();
        Ok(py.detach(|| bipartite::color(&self.succ, pred, self.n, &degree).ok()))
    }

    /// `is_bipartite_node_set` for an undirected graph, given which nodes
    /// are in the set: `None` if a component isn't bipartite.
    fn is_bipartite_node_set(&self, py: Python<'_>, in_set: Vec<bool>) -> PyResult<Option<bool>> {
        if in_set.len() != self.n {
            return Err(PyValueError::new_err("wrong number of flags"));
        }
        Ok(py.detach(|| {
            let components = traversal::connected_components(&self.succ, self.n);
            bipartite::is_node_set(&self.succ, self.n, &components, &in_set).ok()
        }))
    }

    /// `hopcroft_karp_matching` with `left` in NetworkX's set order: each
    /// node's match (`None` if unmatched) and the recursion depth NetworkX
    /// would reach, or `None` if a left node has a neighbor on its side.
    #[allow(clippy::type_complexity)]
    fn hopcroft_karp(
        &self,
        py: Python<'_>,
        left: Vec<u32>,
    ) -> PyResult<Option<(Vec<Option<u32>>, usize)>> {
        let left = self.sources_or_all(Some(left))?;
        let mut is_left = vec![false; self.n];
        for &v in &left {
            is_left[v as usize] = true;
        }
        if left.iter().any(|&v| {
            self.succ
                .neighbors(v as usize)
                .iter()
                .any(|&w| is_left[w as usize])
        }) {
            return Ok(None);
        }
        Ok(py.detach(|| {
            let (mate, depth) = bipartite::hopcroft_karp(&self.succ, self.n, &left);
            let mate = mate
                .into_iter()
                .map(|m| (m != u32::MAX).then_some(m))
                .collect();
            Some((mate, depth))
        }))
    }

    /// For `to_vertex_cover`: which nodes are targets or reach one by an
    /// alternating path. `pairs` are the matching's items as positions.
    fn alternating_reach(
        &self,
        py: Python<'_>,
        targets: Vec<bool>,
        pairs: Vec<(u32, u32)>,
    ) -> PyResult<Vec<bool>> {
        if targets.len() != self.n {
            return Err(PyValueError::new_err("wrong number of flags"));
        }
        for &(u, v) in &pairs {
            self.check_index(u as usize)?;
            self.check_index(v as usize)?;
        }
        Ok(py.detach(|| {
            let n = self.n;
            let mut pair_set: std::collections::HashSet<(u32, u32)> =
                std::collections::HashSet::new();
            for &(u, v) in &pairs {
                pair_set.insert((u.min(v), u.max(v)));
            }
            let m = self.succ.targets.len();
            let mut matched = vec![false; m];
            let mut unmatched = vec![false; m];
            for u in 0..n {
                for e in self.succ.range(u) {
                    let v = self.succ.targets[e];
                    let key = ((u as u32).min(v), (u as u32).max(v));
                    let in_matching = pair_set.contains(&key);
                    // A matching item (a, a) becomes the 1-tuple `(a,)` in
                    // NetworkX's matched edges, so self-loops never count
                    // as matched; the loop edge still leaves the unmatched
                    // edges.
                    matched[e] = in_matching && v as usize != u;
                    unmatched[e] = !in_matching;
                }
            }
            bipartite::alternating_reach(&self.succ, n, &targets, &matched, &unmatched)
        }))
    }

    /// `(neighbors, overlap count)` of `_node_redundancy` for `nodes`.
    fn redundancy_overlaps(&self, py: Python<'_>, nodes: Vec<u32>) -> PyResult<Vec<(u64, u64)>> {
        let nodes = self.sources_or_all(Some(nodes))?;
        Ok(py.detach(|| bipartite::redundancy_overlaps(&self.succ, self.n, &nodes)))
    }

    /// Per-node butterfly counts, as NetworkX 3.7's `butterflies`.
    fn butterflies(&self, py: Python<'_>) -> Vec<u64> {
        let degree = self.degrees();
        py.detach(|| bipartite::butterflies(&self.succ, self.n, &degree))
    }

    // --- Batch 12: flows and cut measures ---

    /// `build_residual_network(G, capacity)`. `rows` is `list(G._adj.values())`
    /// of the NetworkX graph this snapshot was made from (rows in node order,
    /// as in `succ`); `capacity` is the attribute name.
    fn residual_network(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyList>,
        capacity: &Bound<'_, PyAny>,
        compensated: bool,
    ) -> PyResult<FlowRun> {
        let caps = self.edge_values(rows, &[(capacity, flow::FLOAT_INF)])?;
        let mut edges = Vec::new();
        for u in 0..self.n {
            for e in self.succ.range(u) {
                let v = self.succ.targets[e] as usize;
                // Undirected `G.edges` reports each edge from its first end.
                if self.directed || v >= u {
                    edges.push((u as u32, v as u32, caps[0][e]));
                }
            }
        }
        let (n, directed) = (self.n, self.directed);
        let res = py
            .detach(|| flow::Residual::build(n, directed, &edges, compensated))
            .map_err(fail_err)?;
        Ok(FlowRun {
            res,
            g_offsets: self.succ.offsets.clone(),
            g_targets: self.succ.targets.clone(),
            outcome: None,
        })
    }

    /// `network_simplex` on G (`as_directed`: on `nx.DiGraph(G)` of an
    /// undirected G, as `max_flow_min_cost` does). `rows` as for
    /// `residual_network`, `node_rows` is `list(G._node.values())`, and
    /// `overrides` replaces some demands. Returns `(0, cost, flow_dict)`,
    /// or an error code with the node or edge positions it is about.
    #[allow(clippy::too_many_arguments)]
    fn network_simplex<'py>(
        &self,
        py: Python<'py>,
        rows: &Bound<'py, PyList>,
        node_rows: &Bound<'py, PyList>,
        nodes: &Bound<'py, PyList>,
        demand: &Bound<'py, PyAny>,
        capacity: &Bound<'py, PyAny>,
        weight: &Bound<'py, PyAny>,
        overrides: Vec<(u32, Bound<'py, PyAny>)>,
        as_directed: bool,
        max_single_demand: bool,
        compensated: bool,
    ) -> PyResult<(u8, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        if !self.directed && !as_directed {
            return Err(PyNotImplementedError::new_err(
                "network_simplex needs a directed graph",
            ));
        }
        if node_rows.len() != self.n || nodes.len() != self.n {
            return Err(PyValueError::new_err("node rows do not match this graph"));
        }
        let mut demands = Vec::with_capacity(self.n);
        for row in node_rows.iter() {
            let row = row
                .cast::<PyDict>()
                .map_err(|_| PyNotImplementedError::new_err("node data is not a dict"))?;
            demands.push(match row.get_item(demand)? {
                Some(d) => py_val(&d)?,
                None => flow::Val::I(0),
            });
        }
        for (u, value) in &overrides {
            self.check_index(*u as usize)?;
            demands[*u as usize] = py_val(value)?;
        }
        let vals = self.edge_values(
            rows,
            &[(capacity, flow::FLOAT_INF), (weight, flow::Val::I(0))],
        )?;
        let (caps, weights) = (&vals[0], &vals[1]);
        // Arcs in `G.edges` order: edges with a nonzero capacity take
        // part; zero-capacity edges and self-loops don't.
        let (mut src, mut dst, mut cap, mut w) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut arcs = Vec::new();
        let mut loops = Vec::new();
        let mut loop_arcs = Vec::new();
        let mut zero_arcs = Vec::new(); // in `G.edges` order, with self-loops
        for u in 0..self.n {
            for e in self.succ.range(u) {
                let v = self.succ.targets[e] as usize;
                if v == u {
                    loops.push((caps[e], weights[e]));
                    loop_arcs.push(e);
                    zero_arcs.push((e, true));
                } else if caps[e].eq(flow::Val::I(0)) {
                    zero_arcs.push((e, false));
                } else {
                    src.push(u as u32);
                    dst.push(v as u32);
                    cap.push(caps[e]);
                    w.push(weights[e]);
                    arcs.push(e);
                }
            }
        }
        let input = flow::SimplexInput {
            demands: &demands,
            src: &src,
            dst: &dst,
            cap: &cap,
            weight: &w,
            loops: &loops,
        };
        let result = py.detach(|| flow::network_simplex(&input, max_single_demand, compensated));
        let err = |code: u8,
                   a: usize,
                   b: usize|
         -> PyResult<(u8, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
            Ok((
                code,
                a.into_pyobject(py)?.into_any(),
                b.into_pyobject(py)?.into_any(),
            ))
        };
        let arc_ends = |e: usize| -> (usize, usize) {
            let u = self.succ.offsets.partition_point(|&o| o <= e) - 1;
            (u, self.succ.targets[e] as usize)
        };
        let res = match result {
            Ok(res) => res,
            Err(flow::SimplexError::Fail(f)) => return Err(fail_err(f)),
            Err(flow::SimplexError::InfiniteDemand(i)) => return err(1, i, i),
            Err(flow::SimplexError::InfiniteWeight(k)) => {
                let (a, b) = arc_ends(arcs[k]);
                return err(2, a, b);
            }
            Err(flow::SimplexError::SelfLoopInfiniteWeight(k)) => {
                let (a, b) = arc_ends(loop_arcs[k]);
                return err(2, a, b);
            }
            Err(flow::SimplexError::DemandNotZero) => return err(3, 0, 0),
            Err(flow::SimplexError::NegativeCapacity(k)) => {
                let (a, b) = arc_ends(arcs[k]);
                return err(4, a, b);
            }
            Err(flow::SimplexError::SelfLoopNegativeCapacity(k)) => {
                let (a, b) = arc_ends(loop_arcs[k]);
                return err(4, a, b);
            }
            Err(flow::SimplexError::NoFeasibleFlow) => return err(5, 0, 0),
            Err(flow::SimplexError::Unbounded) => return err(6, 0, 0),
        };
        // flow_dict: every node, then each edge's flow in edge order, then
        // the zero entries (and self-loop flows) in `G.edges` order.
        let objects: Vec<Bound<'py, PyAny>> = nodes.iter().collect();
        let rows_out: Vec<Bound<'py, PyDict>> = (0..self.n).map(|_| PyDict::new(py)).collect();
        let flow_dict = PyDict::new(py);
        for (obj, row) in objects.iter().zip(&rows_out) {
            flow_dict.set_item(obj, row)?;
        }
        for (k, &e) in arcs.iter().enumerate() {
            let (u, v) = arc_ends(e);
            rows_out[u].set_item(&objects[v], val_obj(py, res.flows[k])?)?;
        }
        let zero = 0i64.into_pyobject(py)?.into_any();
        let mut next_loop = 0;
        for &(e, is_loop) in &zero_arcs {
            let (u, v) = arc_ends(e);
            let value = if is_loop {
                let f = res.loop_flows[next_loop];
                next_loop += 1;
                match f {
                    Some(c) => val_obj(py, c)?,
                    None => zero.clone(),
                }
            } else {
                zero.clone()
            };
            rows_out[u].set_item(&objects[v], value)?;
        }
        Ok((0, val_obj(py, res.cost)?, flow_dict.into_any()))
    }

    /// `build_flow_dict(G, R)`: `r_rows` holds `R._succ[u]` for each node u
    /// of G, in order. Values and R's neighbor keys go in as they are.
    fn build_flow_dict<'py>(
        &self,
        py: Python<'py>,
        nodes: &Bound<'py, PyList>,
        r_rows: &Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if nodes.len() != self.n || r_rows.len() != self.n {
            return Err(PyValueError::new_err("rows do not match this graph"));
        }
        let objects: Vec<Bound<'py, PyAny>> = nodes.iter().collect();
        let zero = 0i64.into_pyobject(py)?.into_any();
        let flow_key = pyo3::intern!(py, "flow");
        let unsupported = |_| PyNotImplementedError::new_err("NetworkX raises here");
        let out = PyDict::new(py);
        for (u, r_row) in r_rows.iter().enumerate() {
            let row = PyDict::new(py);
            for &v in self.succ.neighbors(u) {
                row.set_item(&objects[v as usize], &zero)?;
            }
            let r_row = r_row
                .cast::<PyDict>()
                .map_err(|_| PyNotImplementedError::new_err("R's rows are not dicts"))?;
            for (v, attr) in r_row.iter() {
                let f = attr.get_item(flow_key).map_err(unsupported)?;
                if f.gt(&zero).map_err(unsupported)? {
                    row.set_item(v, f)?;
                }
            }
            out.set_item(&objects[u], row)?;
        }
        Ok(out)
    }

    /// `cost_of_flow(G, flowDict, weight)`. `rows` as for `residual_network`.
    fn cost_of_flow<'py>(
        &self,
        py: Python<'py>,
        rows: &Bound<'py, PyList>,
        nodes: &Bound<'py, PyList>,
        flow_dict: &Bound<'py, PyAny>,
        weight: &Bound<'py, PyAny>,
        compensated: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let weights = self.edge_values(rows, &[(weight, flow::Val::I(0))])?;
        let objects: Vec<Bound<'py, PyAny>> = nodes.iter().collect();
        let mut products = Vec::new();
        for u in 0..self.n {
            let row = flow_dict
                .get_item(&objects[u])
                .map_err(|_| PyNotImplementedError::new_err("flowDict lacks a node"))?;
            for e in self.succ.range(u) {
                let v = self.succ.targets[e] as usize;
                if !self.directed && v < u {
                    continue;
                }
                let f = row
                    .get_item(&objects[v])
                    .map_err(|_| PyNotImplementedError::new_err("flowDict lacks an edge"))?;
                products.push(py_val(&f)?.mul(weights[0][e]).map_err(fail_err)?);
            }
        }
        val_obj(py, flow::py_sum(products, compensated).map_err(fail_err)?)
    }

    /// `cut_size`: the parts are `(nset1 order, nset2)` (see `flow::cut_size`);
    /// `rows` as for `residual_network`, needed when weights mix ints and floats.
    #[pyo3(signature = (parts, compensated, weight=None, rows=None))]
    fn cut_size_value<'py>(
        &self,
        py: Python<'py>,
        parts: Vec<(Vec<u32>, Option<Vec<u32>>)>,
        compensated: bool,
        weight: Option<&str>,
        rows: Option<&Bound<'py, PyList>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        for (order, nset2) in &parts {
            self.membership(order)?;
            if let Some(s) = nset2 {
                self.membership(s)?;
            }
        }
        let w = self.cut_weights(weight, rows)?;
        let (n, directed) = (self.n, self.directed);
        let value = py
            .detach(|| flow::cut_size(&self.succ, n, directed, &w, &parts, compensated))
            .map_err(fail_err)?;
        val_obj(py, value)
    }

    /// `volume(G, S, weight)` for the positions of `nbunch_iter(S)`; `rows`
    /// as for `cut_size_value`.
    #[pyo3(signature = (nodes, compensated, weight=None, rows=None))]
    fn volume_value<'py>(
        &self,
        py: Python<'py>,
        nodes: Vec<u32>,
        compensated: bool,
        weight: Option<&str>,
        rows: Option<&Bound<'py, PyList>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.membership(&nodes)?;
        let w = self.cut_weights(weight, rows)?;
        let directed = self.directed;
        let value = py
            .detach(|| flow::volume(&self.succ, directed, &w, &nodes, compensated))
            .map_err(fail_err)?;
        val_obj(py, value)
    }

    /// The number of distinct neighbors of `nodes`, and of those not in
    /// `nodes` (`node_expansion`, `boundary_expansion`).
    fn neighborhood_sizes(&self, py: Python<'_>, nodes: Vec<u32>) -> PyResult<(usize, usize)> {
        let inside = self.membership(&nodes)?;
        Ok(py.detach(|| {
            let union = matching::neighbor_union(&self.succ, self.n, &nodes);
            let outside = union.iter().filter(|&&v| !inside[v as usize]).count();
            (union.len(), outside)
        }))
    }

    // --- Batch 13: connectivity, disjoint paths and augmentation ---

    /// Edmonds-Karp flow value from `s` to `t` on the auxiliary digraph of
    /// `local_node_connectivity` (`node_split`: from `sB` to `tA`) or
    /// `local_edge_connectivity`. `s != t` for edge connectivity.
    #[pyo3(signature = (node_split, s, t, cutoff=None))]
    fn conn_local_flow(
        &self,
        py: Python<'_>,
        node_split: bool,
        s: u32,
        t: u32,
        cutoff: Option<f64>,
    ) -> PyResult<i64> {
        self.check_index(s as usize)?;
        self.check_index(t as usize)?;
        let values = self.conn_flows(py, node_split, vec![(s, t)], cutoff)?;
        Ok(values[0])
    }

    /// Flow values (no cutoff) for each pair `(ss[i], ts[i])`, as
    /// `conn_local_flow`; pairs run in parallel.
    fn conn_pair_flows(
        &self,
        py: Python<'_>,
        node_split: bool,
        ss: Vec<u32>,
        ts: Vec<u32>,
    ) -> PyResult<Vec<i64>> {
        if ss.len() != ts.len() {
            return Err(PyValueError::new_err("one target per source"));
        }
        for (&s, &t) in ss.iter().zip(&ts) {
            self.check_index(s as usize)?;
            self.check_index(t as usize)?;
            if !node_split && s == t {
                return Err(PyValueError::new_err("source and sink are the same node"));
            }
        }
        self.conn_flows(py, node_split, ss.into_iter().zip(ts).collect(), None)
    }

    /// `node_connectivity(G)` (no source and target given); `isolating`
    /// is NetworkX 3.7's version (see `connectivity::node_connectivity`).
    fn conn_node_connectivity(&self, py: Python<'_>, isolating: bool) -> PyResult<i64> {
        let degree = self.degrees();
        let pred = self.adj(true);
        py.detach(|| {
            let (v, k) = if isolating {
                let (v, k, _) =
                    connectivity::isolating_cut(&self.succ, pred, self.n, self.directed);
                (v, k)
            } else {
                // min(G.degree(), key=itemgetter(1)): the first minimum.
                let v = (0..self.n).min_by_key(|&u| degree[u]).unwrap_or(0);
                (v, degree[v])
            };
            connectivity::node_connectivity(
                &self.succ,
                pred,
                self.n,
                self.directed,
                v,
                k as i64,
                isolating,
            )
        })
        .map_err(|_| unbounded())
    }

    /// `edge_connectivity(G)`'s minimum local connectivity with `cutoff`;
    /// see `connectivity::edge_connectivity`.
    fn conn_edge_connectivity(&self, py: Python<'_>, cutoff: f64) -> PyResult<Option<i64>> {
        py.detach(|| connectivity::edge_connectivity(&self.succ, self.n, self.directed, cutoff))
            .map_err(|_| unbounded())
    }

    /// One s-t minimum cut: `(flow value, sink side in insertion order,
    /// cut arcs as us/offsets/vs)`, positions in the auxiliary digraph.
    /// `graph_rows` reads the cut edges from G's rows instead of the
    /// auxiliary digraph's (`minimum_st_edge_cut` called on G itself).
    #[allow(clippy::type_complexity)]
    fn conn_st_cut(
        &self,
        py: Python<'_>,
        node_split: bool,
        s: u32,
        t: u32,
        graph_rows: bool,
    ) -> PyResult<(i64, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>)> {
        self.check_index(s as usize)?;
        self.check_index(t as usize)?;
        if !node_split && s == t {
            return Err(PyValueError::new_err("source and sink are the same node"));
        }
        py.detach(|| {
            let (h, s, t) = self.conn_aux(node_split, s, t);
            let mut r = connectivity::residual(&h);
            let rows = if graph_rows && !node_split {
                connectivity::CutRows::Graph(&self.succ)
            } else {
                connectivity::CutRows::Aux
            };
            let cut = connectivity::st_cut(&h, &mut r, s, t, &rows, false)?;
            Ok((cut.value, cut.sink_order, cut.us, cut.offsets, cut.vs))
        })
        .map_err(|_: connectivity::Unbounded| unbounded())
    }

    /// `minimum_node_cut` / `minimum_edge_cut`'s search over the pairs
    /// `(ss[i], ts[i])` on one shared residual network. Returns `None`
    /// (the initial cut stands) or `(pair index, cut)`, the cut as in
    /// `conn_st_cut` without the value, or `None` for adjacent nodes.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn conn_search_cuts(
        &self,
        py: Python<'_>,
        node_split: bool,
        ss: Vec<u32>,
        ts: Vec<u32>,
        skips: Vec<bool>,
        initial_len: usize,
        adjacent_both: bool,
        requeue: bool,
    ) -> PyResult<Option<(usize, Option<(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>)>)>> {
        if ss.len() != ts.len() || ss.len() != skips.len() {
            return Err(PyValueError::new_err("one target and flag per source"));
        }
        let mut pairs = Vec::with_capacity(ss.len());
        for ((&s, &t), &skip_if_edge) in ss.iter().zip(&ts).zip(&skips) {
            self.check_index(s as usize)?;
            self.check_index(t as usize)?;
            if !node_split && s == t {
                return Err(PyValueError::new_err("source and sink are the same node"));
            }
            pairs.push(connectivity::CutPair { s, t, skip_if_edge });
        }
        py.detach(|| {
            let h = if node_split {
                connectivity::node_aux(&self.succ, self.n, self.directed)
            } else {
                connectivity::edge_aux(&self.succ, self.n, self.directed)
            };
            let chosen = connectivity::search_cuts(
                &self.succ,
                &h,
                node_split,
                &pairs,
                initial_len,
                adjacent_both,
                requeue,
            )?;
            Ok(chosen.map(|(i, cut)| (i, cut.map(|c| (c.sink_order, c.us, c.offsets, c.vs)))))
        })
        .map_err(|_: connectivity::Unbounded| unbounded())
    }

    /// `edge_disjoint_paths` / `node_disjoint_paths` (node positions of G).
    /// Status 0: the paths; 1: `NetworkXNoPath`; 2: source and sink are the
    /// same node (edge paths only).
    #[pyo3(signature = (node_split, s, t, cutoff=None))]
    fn conn_disjoint_paths(
        &self,
        py: Python<'_>,
        node_split: bool,
        s: u32,
        t: u32,
        cutoff: Option<f64>,
    ) -> PyResult<(u8, Vec<Vec<u32>>)> {
        self.check_index(s as usize)?;
        self.check_index(t as usize)?;
        py.detach(|| {
            let (h, s, t) = self.conn_aux(node_split, s, t);
            // possible = min(H.out_degree(s), H.in_degree(t))
            let possible = h.succ[s as usize].len().min(h.pred[t as usize].len());
            if possible == 0 {
                return Ok((1, Vec::new()));
            }
            if s == t {
                return Ok((2, Vec::new()));
            }
            let possible = possible as f64;
            let cutoff = cutoff.map_or(possible, |c| if c <= possible { c } else { possible });
            let mut r = connectivity::residual(&h);
            if connectivity::max_flow(&mut r, s, t, cutoff)? == 0 {
                return Ok((1, Vec::new()));
            }
            let mut paths = connectivity::disjoint_paths(&r, s, t, cutoff);
            if node_split {
                // Each auxiliary node's original node, first occurrences only.
                let mut seen = vec![false; self.n];
                for path in paths.iter_mut() {
                    let mut ids = Vec::new();
                    for &a in path.iter() {
                        let i = a / 2;
                        if !seen[i as usize] {
                            seen[i as usize] = true;
                            ids.push(i);
                        }
                    }
                    for &i in &ids {
                        seen[i as usize] = false;
                    }
                    *path = ids;
                }
            }
            Ok((0, paths))
        })
        .map_err(|_: connectivity::Unbounded| unbounded())
    }

    /// `minimum_node_cut` (3.7+): the first node with the smallest
    /// isolating cut, and whether that cut is its predecessors (directed).
    fn conn_isolating_cut(&self) -> (usize, bool) {
        let (v, _, use_pred) =
            connectivity::isolating_cut(&self.succ, self.adj(true), self.n, self.directed);
        (v, use_pred)
    }

    /// Predecessors of `v` in NetworkX's `G.pred[v]` order (call
    /// `_ensure_exact_pred` first for converted graphs).
    fn conn_exact_predecessors(&self, v: usize) -> PyResult<Vec<u32>> {
        self.check_index(v)?;
        let (rows, _) = self.reverse_exact_order(None)?;
        Ok(rows.neighbors(v).to_vec())
    }

    /// `stoer_wagner` on a connected graph with two or more nodes:
    /// `(status, cut value, nodes in the rebuilt graph's order, reachable
    /// side in breadth-first order)`. Status 1: a negative weight (outside
    /// self-loops); 2: an infinite weight, or int weights too large to sum
    /// exactly in f64; both skip the computation.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (weight=None))]
    fn conn_stoer_wagner(
        &self,
        py: Python<'_>,
        weight: Option<&str>,
    ) -> PyResult<(u8, f64, Vec<u32>, Vec<u32>)> {
        let w = self.weight_slice(weight, false)?;
        let (all_int, _) = self.weights_info(weight);
        if let Some(ws) = w {
            let mut total = 0.0f64;
            let mut status = 0;
            for u in 0..self.n {
                for e in self.succ.range(u) {
                    if self.succ.targets[e] as usize == u {
                        continue;
                    }
                    let x = ws[e];
                    if x < 0.0 {
                        return Ok((1, 0.0, Vec::new(), Vec::new()));
                    }
                    if !x.is_finite() {
                        status = 2;
                    }
                    total += x;
                }
            }
            if status == 2 || (all_int && total > (1u64 << 52) as f64) {
                return Ok((2, 0.0, Vec::new(), Vec::new()));
            }
        }
        Ok(py.detach(|| {
            let r = connectivity::stoer_wagner(&self.succ, self.n, w);
            (0, r.cut_value, r.node_order, r.reachable)
        }))
    }

    /// `bridge_components`: node positions per component, in insertion order.
    fn conn_bridge_components(&self, py: Python<'_>) -> Vec<Vec<u32>> {
        py.detach(|| {
            let bridges = structure_more::bridges(&self.succ, self.n, None, false);
            connectivity::bridge_components(&self.succ, self.n, &bridges)
        })
    }

    /// `complement_edges` for `u` in `start..end`: flat position pairs.
    fn conn_complement_edges(&self, py: Python<'_>, start: usize, end: usize) -> Vec<u32> {
        let end = end.min(self.n);
        py.detach(|| {
            connectivity::complement_edges(
                &self.succ,
                self.adj(true),
                self.n,
                self.directed,
                start.min(end),
                end,
            )
        })
    }

    // --- Batch 14: assortativity, link prediction and reciprocity ---

    /// A scorer for link prediction pairs (see `measures::LinkMode`; `mode`
    /// 0 counts, 1 communities, 2 sums of `table[degree]`), holding the
    /// per-node data once for all batches of pairs.
    #[pyo3(signature = (mode, hashes=None, table=None, classes=None, compensated=false))]
    fn link_scorer(
        slf: Bound<'_, Self>,
        mode: u8,
        hashes: Option<Vec<i64>>,
        table: Option<Vec<f64>>,
        classes: Option<Vec<i64>>,
        compensated: bool,
    ) -> PyResult<LinkScorer> {
        let g = slf.get();
        let degree = g.degrees();
        let max_deg = degree.iter().copied().max().unwrap_or(0);
        let hashes = hashes.unwrap_or_else(|| vec![0; g.n]);
        let table = table.unwrap_or_default();
        let mode = match mode {
            0 => measures::LinkMode::Counts,
            1 => measures::LinkMode::Community,
            2 if table.len() > max_deg => measures::LinkMode::Sum,
            _ => return Err(PyValueError::new_err("bad link scorer mode or table")),
        };
        if hashes.len() != g.n || classes.as_ref().is_some_and(|c| c.len() != g.n) {
            return Err(PyValueError::new_err("per-node data has the wrong length"));
        }
        Ok(LinkScorer {
            graph: slf.unbind(),
            mode,
            hashes,
            degree,
            table,
            classes,
            compensated,
        })
    }

    /// `list(G._adj[u].keys() & G._adj[v].keys() - {u, v})` per pair, as
    /// rustnx's replay of CPython's set table orders it.
    fn common_neighbors_in_set_order(
        &self,
        us: Vec<u32>,
        vs: Vec<u32>,
        hashes: Vec<i64>,
    ) -> PyResult<Vec<Vec<u32>>> {
        self.check_pairs(&us, &vs)?;
        if hashes.len() != self.n {
            return Err(PyValueError::new_err("hashes has the wrong length"));
        }
        Ok(measures::common_neighbors_in_set_order(
            &self.succ, self.n, &hashes, &us, &vs,
        ))
    }

    /// Unweighted distance per pair, -1 if unreachable.
    fn pair_distances(&self, py: Python<'_>, us: Vec<u32>, vs: Vec<u32>) -> PyResult<Vec<i64>> {
        self.check_pairs(&us, &vs)?;
        Ok(py.detach(|| measures::pair_distances(&self.succ, self.n, &us, &vs)))
    }

    /// Integer degrees of one kind (0 `G.degree`, 1 out, 2 in), summing an
    /// integer edge attribute when `weight` is given.
    #[pyo3(signature = (kind, weight=None))]
    fn kind_degrees(&self, kind: u8, weight: Option<&str>) -> PyResult<Vec<i64>> {
        let n = self.n;
        let w_out = self.weight_slice(weight, false)?;
        let out = measures::row_sums(&self.succ, n, w_out);
        Ok(match (kind, self.directed) {
            (1, true) => out,
            (2, true) => measures::row_sums(self.adj(true), n, self.weight_slice(weight, true)?),
            (0, true) => {
                let ins = measures::row_sums(self.adj(true), n, self.weight_slice(weight, true)?);
                out.iter().zip(ins).map(|(a, b)| a + b).collect()
            }
            (0, false) => (0..n)
                .map(|v| {
                    // A self-loop counts twice.
                    let own = self
                        .succ
                        .range(v)
                        .find(|&e| self.succ.targets[e] as usize == v)
                        .map_or(0, |e| w_out.map_or(1, |w| w[e] as i64));
                    out[v] + own
                })
                .collect(),
            _ => return Err(PyValueError::new_err("bad degree kind")),
        })
    }

    /// Pairs of `node_degree_xy`/`node_attribute_xy` as node positions.
    #[pyo3(signature = (order, nbr_mask=None))]
    fn xy_pairs(
        &self,
        order: Vec<u32>,
        nbr_mask: Option<Vec<bool>>,
    ) -> PyResult<(Vec<u32>, Vec<u32>)> {
        self.sources_or_all(Some(order.clone()))?;
        if nbr_mask.as_ref().is_some_and(|m| m.len() != self.n) {
            return Err(PyValueError::new_err("mask has the wrong length"));
        }
        Ok(measures::xy_pairs(&self.succ, &order, nbr_mask.as_deref()))
    }

    /// NetworkX's `mixing_dict` over the `xy_pairs` stream, with per-node
    /// x and y classes: `(number of pairs, [(class, node, [(class, node,
    /// count)])])` in insertion order.
    #[pyo3(signature = (order, xcls, ycls, nbr_mask=None))]
    fn mixing(
        &self,
        py: Python<'_>,
        order: Vec<u32>,
        xcls: Vec<i64>,
        ycls: Vec<i64>,
        nbr_mask: Option<Vec<bool>>,
    ) -> PyResult<(u64, Vec<measures::MixingRow>)> {
        self.sources_or_all(Some(order.clone()))?;
        if xcls.len() != self.n
            || ycls.len() != self.n
            || nbr_mask.as_ref().is_some_and(|m| m.len() != self.n)
        {
            return Err(PyValueError::new_err("per-node data has the wrong length"));
        }
        Ok(py.detach(|| measures::mixing(&self.succ, &order, nbr_mask.as_deref(), &xcls, &ycls)))
    }

    /// `average_neighbor_degree` numerators and source degrees for `nodes`
    /// (kinds as in `kind_degrees`; the source kind picks the rows).
    #[pyo3(signature = (nodes, source, target, weight=None))]
    fn neighbor_degree_terms(
        &self,
        nodes: Vec<u32>,
        source: u8,
        target: u8,
        weight: Option<&str>,
    ) -> PyResult<Vec<(i64, i64)>> {
        let nodes = self.sources_or_all(Some(nodes))?;
        let target_degree = self.kind_degrees(target, None)?;
        let source_degree = self.kind_degrees(source, weight)?;
        let mut rows = Vec::new();
        if !self.directed || source != 2 {
            rows.push((&self.succ, self.weight_slice(weight, false)?));
        }
        if self.directed && source != 1 {
            rows.push((self.adj(true), self.weight_slice(weight, true)?));
        }
        measures::neighbor_degree_terms(&rows, &target_degree, &source_degree, &nodes)
            .ok_or_else(|| PyNotImplementedError::new_err("integer overflow"))
    }

    /// `average_degree_connectivity` sums per source degree, in first-seen
    /// order: `(degrees, neighbor degree sums, weighted degree sums)`.
    #[pyo3(signature = (nodes, source, target, weight=None))]
    #[allow(clippy::type_complexity)]
    fn degree_connectivity(
        &self,
        nodes: Vec<u32>,
        source: u8,
        target: u8,
        weight: Option<&str>,
    ) -> PyResult<(Vec<i64>, Vec<i64>, Vec<i64>)> {
        let nodes = self.sources_or_all(Some(nodes))?;
        let k_degree = self.kind_degrees(source, None)?;
        let target_degree = self.kind_degrees(target, None)?;
        let weighted_source = self.kind_degrees(source, weight)?;
        // Neighbors: predecessors for "in", else successors (NetworkX's
        // `G.neighbors` for "in+out").
        let reverse = self.directed && source == 2;
        measures::degree_connectivity(
            self.adj(reverse),
            self.weight_slice(weight, reverse)?,
            &k_degree,
            &target_degree,
            &weighted_source,
            &nodes,
        )
        .ok_or_else(|| PyNotImplementedError::new_err("integer overflow"))
    }

    /// `(len(pred & succ), len(pred) + len(succ))` per node (directed).
    fn reciprocity_counts(&self, nodes: Vec<u32>) -> PyResult<Vec<(u64, u64)>> {
        let nodes = self.sources_or_all(Some(nodes))?;
        Ok(measures::reciprocity_counts(
            &self.succ,
            self.adj(true),
            self.n,
            &nodes,
        ))
    }

    /// Directed edges `u -> v` (u != v) whose reverse edge exists.
    fn reciprocated_edges(&self, py: Python<'_>) -> u64 {
        py.detach(|| measures::reciprocated_edges(&self.succ, self.n))
    }

    /// `(nk, ek)` per degree, for `rich_club_coefficient`.
    fn rich_club_counts(&self) -> Vec<(u64, u64)> {
        let degree = self.degrees();
        measures::rich_club_counts(&self.succ, self.n, &degree)
    }

    /// `sum(G.degree(u) * G.degree(v) for u, v in G.edges())`.
    fn s_metric_sum(&self, py: Python<'_>) -> u128 {
        let degree = self.degrees();
        py.detach(|| measures::s_metric(&self.succ, self.n, self.directed, &degree))
    }

    /// Rows of the adjacency matrix to the power `k` (wrapping int64).
    fn walk_counts(&self, py: Python<'_>, k: u64) -> Vec<Vec<i64>> {
        py.detach(|| measures::walk_counts(&self.succ, self.n, k))
    }

    // --- Batch 15: communities, efficiency and structural holes ---

    /// `global_efficiency`'s running total of `1 / d`, before the division.
    fn global_efficiency_total(&self, py: Python<'_>) -> f64 {
        py.detach(|| communities::global_efficiency(&self.succ, self.n))
    }

    /// `local_efficiency`'s per-node efficiencies (`None`: NetworkX's int
    /// 0). `orders[v]`: the nodes of `G.subgraph(G[v])` in its iteration
    /// order when that is a set's order, else `None` (G's order).
    fn local_efficiencies(
        &self,
        py: Python<'_>,
        orders: Vec<Option<Vec<u32>>>,
    ) -> PyResult<Vec<Option<f64>>> {
        if orders.len() != self.n {
            return Err(PyValueError::new_err("one order per node expected"));
        }
        for order in orders.iter().flatten() {
            for &v in order {
                self.check_index(v as usize)?;
            }
        }
        Ok(py.detach(|| communities::local_efficiency(&self.succ, self.n, &orders)))
    }

    /// The sum in `gutman_index` (kind 0), `schultz_index` (1) or
    /// `hyper_wiener_index` (2), before halving, in NetworkX's order;
    /// `None` if it can't be matched (overflow, a negative cycle).
    fn distance_index(
        &self,
        py: Python<'_>,
        kind: u8,
        weight: Option<&str>,
        float: bool,
        compensated: bool,
    ) -> PyResult<Option<Py<PyAny>>> {
        let kind = match kind {
            0 => communities::IndexKind::Gutman,
            1 => communities::IndexKind::Schultz,
            _ => communities::IndexKind::HyperWiener,
        };
        let w = self.weight_slice(weight, false)?;
        // NetworkX's `dict(G.degree, weight=weight)` holds unweighted degrees.
        let degrees = self.degrees();
        let total = py.detach(|| {
            if float {
                let deg: Vec<f64> = degrees.iter().map(|&d| d as f64).collect();
                communities::distance_index_float(&self.succ, self.n, w?, &deg, kind, compensated)
                    .map(communities::Num::Float)
            } else {
                let deg: Vec<i128> = degrees.iter().map(|&d| d as i128).collect();
                communities::distance_index_int(&self.succ, self.n, w, &deg, kind)
                    .map(communities::Num::Int)
            }
        });
        total.map(|t| num_object(py, t)).transpose()
    }

    /// `closeness_vitality`'s Wiener index totals: G's (if `whole`), and
    /// G's without each node of `removals` (`None`: not connected).
    #[allow(clippy::type_complexity)]
    fn vitality_totals(
        &self,
        py: Python<'_>,
        weight: Option<&str>,
        float: bool,
        compensated: bool,
        removals: Vec<u32>,
        whole: bool,
    ) -> PyResult<(Option<Py<PyAny>>, Vec<Option<Py<PyAny>>>)> {
        let w = self.weight_slice(weight, false)?;
        let removals = self.sources_or_all(Some(removals))?;
        let (whole, without) = py.detach(|| {
            communities::vitality_totals(
                &self.succ,
                self.n,
                w,
                float,
                compensated,
                &removals,
                whole,
            )
        });
        let whole = whole.map(|t| num_object(py, t)).transpose()?;
        let without = without
            .into_iter()
            .map(|t| t.map(|t| num_object(py, t)).transpose())
            .collect::<PyResult<_>>()?;
        Ok((whole, without))
    }

    /// `flow_hierarchy`'s integer sums: arc weight inside strongly
    /// connected components, and in total.
    #[pyo3(signature = (weight=None))]
    fn scc_arc_weights(&self, py: Python<'_>, weight: Option<&str>) -> PyResult<(i128, i128)> {
        let w = self.weight_slice(weight, false)?;
        Ok(py.detach(|| communities::scc_arc_weights(&self.succ, self.n, w)))
    }

    /// Edges inside one block and between two blocks, given each node's
    /// block (`-1`: none). `None` if `strict` and an edge has an end in no
    /// block.
    fn block_edge_counts(
        &self,
        py: Python<'_>,
        block: Vec<i64>,
        strict: bool,
    ) -> PyResult<Option<(u64, u64)>> {
        if block.len() != self.n {
            return Err(PyValueError::new_err("one block per node expected"));
        }
        Ok(py.detach(|| {
            communities::block_edge_counts(&self.succ, self.n, self.directed, &block, strict)
        }))
    }

    /// Edges inside each block (positions, flattened with `ends`), summed.
    fn edges_inside_blocks(
        &self,
        py: Python<'_>,
        flat: Vec<u32>,
        ends: Vec<usize>,
    ) -> PyResult<u64> {
        self.check_flat(&flat, &ends)?;
        Ok(py.detach(|| {
            communities::edges_inside_blocks(&self.succ, self.n, self.directed, &flat, &ends)
        }))
    }

    /// `modularity`'s sums (see `communities::modularity_stats`). Float
    /// weights on a directed graph need exact in-edge order loaded.
    #[allow(clippy::type_complexity)]
    fn modularity_stats(
        &self,
        py: Python<'_>,
        flat: Vec<u32>,
        ends: Vec<usize>,
        weight: Option<&str>,
        float: bool,
        compensated: bool,
    ) -> PyResult<(Py<PyAny>, Vec<(Py<PyAny>, Py<PyAny>, Py<PyAny>)>)> {
        self.check_flat(&flat, &ends)?;
        let w = self.weight_slice(weight, false)?;
        if float && w.is_none() {
            return Err(PyValueError::new_err("float sums need a weight"));
        }
        if !float && communities::degrees_int(&self.succ, self.n, w).is_none() {
            return Err(PyNotImplementedError::new_err(
                "integer weights are too large",
            ));
        }
        let pred = if !self.directed {
            None
        } else if float {
            Some(self.reverse_exact(weight)?)
        } else {
            Some((self.adj(true), self.weight_slice(weight, true)?))
        };
        let (total, per) = py.detach(|| {
            communities::modularity_stats(
                &self.succ,
                pred,
                self.n,
                w,
                float,
                compensated,
                &flat,
                &ends,
            )
        });
        let per = per
            .into_iter()
            .map(|(l, o, i)| Ok((num_object(py, l)?, num_object(py, o)?, num_object(py, i)?)))
            .collect::<PyResult<_>>()?;
        Ok((num_object(py, total)?, per))
    }

    /// `greedy_modularity_communities`' merges and whether the generator ran
    /// out (`None`: hand the call to NetworkX). `rank[v]`: v's position in
    /// sorted order, for ties. Float weights on a directed graph need exact
    /// in-edge order loaded.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn greedy_modularity(
        &self,
        py: Python<'_>,
        weight: Option<&str>,
        float: bool,
        compensated: bool,
        resolution: f64,
        rank: Vec<u32>,
        cutoff: f64,
        best_n: f64,
    ) -> PyResult<Option<(Vec<(u32, u32)>, bool)>> {
        if rank.len() != self.n {
            return Err(PyValueError::new_err("one rank per node expected"));
        }
        let n = self.n;
        let w = self.weight_slice(weight, false)?;
        // Degrees as `G.degree`, `G.out_degree` and `G.in_degree` give them.
        let (out_deg, in_deg, total): (Vec<f64>, Vec<f64>, f64) = if self.directed {
            let (pred, pw) = if float {
                self.reverse_exact(weight)?
            } else {
                (self.adj(true), self.weight_slice(weight, true)?)
            };
            if float {
                let out =
                    communities::row_sums_float(&self.succ, n, w.expect("weight"), compensated);
                let inn = communities::row_sums_float(pred, n, pw.expect("weight"), compensated);
                let deg: Vec<f64> = out.iter().zip(&inn).map(|(a, b)| a + b).collect();
                let s = spectral::py_sum(deg.into_iter(), compensated);
                (out, inn, s)
            } else {
                let out = communities::row_sums_int(&self.succ, n, w);
                let inn = communities::row_sums_int(pred, n, pw);
                let s: i128 = out.iter().chain(&inn).sum();
                let f = |v: Vec<i128>| v.into_iter().map(|x| x as f64).collect::<Vec<f64>>();
                (f(out), f(inn), s as f64)
            }
        } else if float {
            let deg = communities::degrees_float(&self.succ, n, w.expect("weight"), compensated);
            let s = spectral::py_sum(deg.iter().copied(), compensated);
            (deg, Vec::new(), s)
        } else {
            let Some(deg) = communities::degrees_int(&self.succ, n, w) else {
                return Ok(None);
            };
            let s: i128 = deg.iter().sum();
            (
                deg.into_iter().map(|x| x as f64).collect(),
                Vec::new(),
                s as f64,
            )
        };
        // `G.size(weight)`: the degree sum halved (exact: it is even).
        let m = total / 2.0;
        if m == 0.0 || !m.is_finite() {
            return Ok(None);
        }
        let q0 = 1.0 / m;
        let (a, b) = if self.directed {
            (
                out_deg.iter().map(|&d| d * q0).collect(),
                in_deg.iter().map(|&d| d * q0).collect(),
            )
        } else {
            (out_deg.iter().map(|&d| d * q0 * 0.5).collect(), Vec::new())
        };
        let mut edges = Vec::new();
        for u in 0..n {
            for e in self.succ.range(u) {
                let v = self.succ.targets[e];
                if self.directed || v as usize >= u {
                    edges.push((u as u32, v, w.map_or(1.0, |w| w[e])));
                }
            }
        }
        let input = communities::GreedyInput {
            n,
            directed: self.directed,
            edges,
            a,
            b,
            q0,
            resolution,
            rank: &rank,
            cutoff,
            best_n,
        };
        Ok(py.detach(|| communities::greedy_modularity(input)))
    }

    /// `naive_greedy_modularity_communities`' merges for unit or integer
    /// weights (`None`: hand the call to NetworkX). `m` and `norm` are
    /// computed by Python, as `modularity` does.
    #[allow(clippy::too_many_arguments)]
    fn naive_greedy_modularity(
        &self,
        py: Python<'_>,
        weight: Option<&str>,
        m: f64,
        norm: f64,
        res_int: Option<i128>,
        res_float: f64,
        compensated: bool,
    ) -> PyResult<Option<Vec<(u32, u32)>>> {
        let w = self.weight_slice(weight, false)?;
        let res = match res_int {
            Some(r) => communities::Resolution::Int(r),
            None => communities::Resolution::Float(res_float),
        };
        Ok(py.detach(|| {
            communities::naive_greedy_modularity(&self.succ, self.n, w, m, norm, res, compensated)
        }))
    }

    /// `girvan_newman`'s state after `G.copy().to_undirected()` and removing
    /// self-loops. `scale`: `edge_betweenness_centrality`'s normalization.
    #[pyo3(signature = (scale=None))]
    fn girvan_newman(&self, scale: Option<f64>) -> GirvanNewman {
        let mut g = communities::EditableGraph::rebuilt(&self.succ, self.n, None);
        g.remove_self_loops();
        GirvanNewman { g, scale }
    }

    /// `edge_betweenness_partition` for an undirected graph (on `G.copy()`).
    #[pyo3(signature = (number_of_sets, weight=None, scale=None))]
    fn edge_betweenness_partition(
        &self,
        py: Python<'_>,
        number_of_sets: usize,
        weight: Option<&str>,
        scale: Option<f64>,
    ) -> PyResult<Vec<Vec<u32>>> {
        let w = self.weight_slice(weight, false)?;
        Ok(py.detach(|| {
            let mut g = communities::EditableGraph::rebuilt(&self.succ, self.n, w);
            g.betweenness_partition(number_of_sets, w.is_some(), scale)
        }))
    }

    /// `asyn_lpa_communities`: final labels, and the generator's new state
    /// (`state`: `random.Random.getstate()[1]`). `None` for a bad state.
    #[allow(clippy::type_complexity)]
    fn asyn_lpa(
        &self,
        py: Python<'_>,
        weight: Option<&str>,
        state: Vec<u32>,
    ) -> PyResult<Option<(Vec<u32>, Vec<u32>)>> {
        let w = self.weight_slice(weight, false)?;
        Ok(py.detach(|| {
            let mut rng = communities::Mt19937::from_state(&state)?;
            let labels = communities::asyn_lpa(&self.succ, self.n, w, &mut rng);
            Some((labels, rng.state()))
        }))
    }

    /// `fast_label_propagation_communities`: final labels and the
    /// generator's new state. Directed graphs need exact in-edge order.
    #[allow(clippy::type_complexity)]
    fn fast_label_propagation(
        &self,
        py: Python<'_>,
        weight: Option<&str>,
        state: Vec<u32>,
    ) -> PyResult<Option<(Vec<u32>, Vec<u32>)>> {
        let w = self.weight_slice(weight, false)?;
        let pred = if self.directed {
            Some(self.reverse_exact_order(weight)?)
        } else {
            None
        };
        Ok(py.detach(|| {
            let mut rng = communities::Mt19937::from_state(&state)?;
            let labels = communities::fast_label_propagation(&self.succ, pred, self.n, w, &mut rng);
            Some((labels, rng.state()))
        }))
    }

    /// `asyn_fluidc`: each node's community (`None`: unassigned), the order
    /// nodes joined NetworkX's dict, and the generator's new state. `None`
    /// where NetworkX would fail or for a bad state.
    #[allow(clippy::type_complexity)]
    fn asyn_fluidc(
        &self,
        py: Python<'_>,
        k: usize,
        max_iter: i64,
        legacy: bool,
        state: Vec<u32>,
    ) -> Option<(Vec<Option<u32>>, Vec<u32>, Vec<u32>)> {
        if k == 0 || k > self.n {
            return None;
        }
        py.detach(|| {
            let mut rng = communities::Mt19937::from_state(&state)?;
            let (com, order) =
                communities::asyn_fluidc(&self.succ, self.n, k, max_iter, legacy, &mut rng)?;
            let com = com
                .into_iter()
                .map(|c| (c != u32::MAX).then_some(c))
                .collect();
            Some((com, order, rng.state()))
        })
    }

    /// `overlapping_modularity`'s sums (see `communities::overlap_stats`).
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn overlap_stats(
        &self,
        py: Python<'_>,
        flat: Vec<u32>,
        ends: Vec<usize>,
        membership: Vec<u64>,
        weight: Option<&str>,
        float: bool,
        compensated: bool,
    ) -> PyResult<(Py<PyAny>, Vec<(f64, f64)>)> {
        self.check_flat(&flat, &ends)?;
        if membership.len() != self.n {
            return Err(PyValueError::new_err("one count per node expected"));
        }
        let w = self.weight_slice(weight, false)?;
        if float && w.is_none() {
            return Err(PyValueError::new_err("float sums need a weight"));
        }
        if !float && communities::degrees_int(&self.succ, self.n, w).is_none() {
            return Err(PyNotImplementedError::new_err(
                "integer weights are too large",
            ));
        }
        let (total, per) = py.detach(|| {
            communities::overlap_stats(
                &self.succ,
                self.n,
                w,
                float,
                compensated,
                &membership,
                &flat,
                &ends,
            )
        });
        Ok((num_object(py, total)?, per))
    }

    // --- Batch 16: approximation algorithms and graph operations ---

    /// `min_weighted_vertex_cover`'s cover in insertion order. `costs` are
    /// the node weights (`None`: all 1); `None` if one isn't a Python int
    /// or float, or int arithmetic would leave `i64`.
    #[pyo3(signature = (costs=None))]
    fn local_ratio_cover(
        &self,
        py: Python<'_>,
        costs: Option<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<Option<Vec<u32>>> {
        let costs = match costs {
            None => vec![trees_more::Num::Int(1); self.n],
            Some(values) => {
                if values.len() != self.n {
                    return Err(PyValueError::new_err("one cost per node"));
                }
                let mut out = Vec::with_capacity(self.n);
                for value in &values {
                    match py_num(value) {
                        Some(x) => out.push(x),
                        None => return Ok(None),
                    }
                }
                out
            }
        };
        Ok(
            py.detach(|| {
                approximation::local_ratio_cover(&self.succ, self.n, self.directed, costs)
            }),
        )
    }

    /// `min_weighted_dominating_set`'s nodes in insertion order.
    #[pyo3(signature = (weights, uncovered_rule))]
    fn min_weighted_dominating(
        &self,
        py: Python<'_>,
        weights: Option<Vec<f64>>,
        uncovered_rule: bool,
    ) -> PyResult<Vec<u32>> {
        if weights.as_ref().is_some_and(|w| w.len() != self.n) {
            return Err(PyValueError::new_err("one weight per node"));
        }
        Ok(py.detach(|| {
            approximation::min_weighted_dominating(
                &self.succ,
                self.n,
                weights.as_deref(),
                uncovered_rule,
            )
        }))
    }

    /// Whether every node is adjacent to all others (self-loops aside).
    fn tsp_is_complete(&self) -> bool {
        approximation::is_complete(&self.succ, self.n)
    }

    /// `greedy_tsp` from `source`: `(0, cycle)`, or `(1, [])` if G isn't
    /// complete, or `(2, [])` on a tie between nearest nodes.
    fn greedy_tsp(&self, py: Python<'_>, weight: &str, source: usize) -> PyResult<(u8, Vec<u32>)> {
        self.check_index(source)?;
        let w = self
            .weight_slice(Some(weight), false)?
            .expect("an attribute");
        Ok(py.detach(
            || match approximation::greedy_tsp(&self.succ, w, self.n, source) {
                Ok(cycle) => (0, cycle),
                Err(approximation::TspError::NotComplete) => (1, Vec::new()),
                Err(approximation::TspError::Tie) => (2, Vec::new()),
            },
        ))
    }

    /// The tour state of `simulated_annealing_tsp` and
    /// `threshold_accepting_tsp`; `nodes[k]` is the position of
    /// `init_cycle[k]`. The caller checked that G is complete.
    fn tsp_tour(
        &self,
        weight: &str,
        nodes: Vec<u32>,
        ints: bool,
        compensated: bool,
    ) -> PyResult<TspTour> {
        for &v in &nodes {
            self.check_index(v as usize)?;
        }
        let w = self
            .weight_slice(Some(weight), false)?
            .expect("an attribute");
        Ok(TspTour(approximation::Tour::new(
            &self.succ,
            w,
            self.n,
            nodes,
            ints,
            compensated,
        )))
    }

    /// The nodes `treewidth_decomp` eliminates with the min fill-in heuristic.
    fn min_fill_in_order(&self, py: Python<'_>) -> Vec<u32> {
        py.detach(|| approximation::min_fill_in_order(&self.succ, self.n))
    }

    /// `approximate_diameter`'s two sweeps from `source`, or `None` if the
    /// graph isn't (strongly) connected.
    fn two_sweep(&self, py: Python<'_>, source: usize) -> PyResult<Option<u32>> {
        self.check_index(source)?;
        let pred = if self.directed {
            Some(self.adj(true))
        } else {
            None
        };
        Ok(py.detach(|| approximation::two_sweep(&self.succ, pred, self.n, source).ok()))
    }

    /// `one_exchange`'s state, starting from the nodes with `side` set.
    /// The caller checked that the weights are ints.
    #[pyo3(signature = (side, weight=None))]
    fn max_cut(&self, side: Vec<bool>, weight: Option<&str>) -> PyResult<MaxCutState> {
        if side.len() != self.n {
            return Err(PyValueError::new_err("one side per node"));
        }
        let w = match self.weight_slice(weight, false)? {
            Some(w) => w.iter().map(|&x| x as i64).collect(),
            None => vec![1; self.succ.targets.len()],
        };
        Ok(MaxCutState(approximation::MaxCut::new(&self.succ, w, side)))
    }

    /// `cut_size` between the nodes with `side` set and the rest (int weights).
    #[pyo3(signature = (side, weight=None))]
    fn partition_cut_value(
        &self,
        py: Python<'_>,
        side: Vec<bool>,
        weight: Option<&str>,
    ) -> PyResult<i128> {
        if side.len() != self.n {
            return Err(PyValueError::new_err("one side per node"));
        }
        let w = self.weight_slice(weight, false)?;
        Ok(py.detach(|| approximation::cut_value(&self.succ, w, &side)))
    }

    /// Edges `is_kl_connected` rejects for path count `limit`, in edge order
    /// (only the first with `first_only`). Directed graphs need the exact
    /// predecessor order loaded.
    fn kl_rejected(
        &self,
        py: Python<'_>,
        limit: u64,
        first_only: bool,
    ) -> PyResult<(Vec<u32>, Vec<u32>)> {
        let pred = self.path_adj(None, true)?.0;
        Ok(py.detach(|| {
            approximation::kl_rejected(&self.succ, pred, self.n, self.directed, limit, first_only)
                .into_iter()
                .unzip()
        }))
    }

    /// `complement`'s edges in the order NetworkX adds them.
    fn complement_pairs(&self, py: Python<'_>) -> (Vec<u32>, Vec<u32>) {
        py.detach(|| approximation::complement_edges(&self.succ, self.n, self.directed))
    }

    /// `power(G, k)`'s edges in the order NetworkX adds them.
    fn power_pairs(&self, py: Python<'_>, k: usize) -> (Vec<u32>, Vec<u32>) {
        py.detach(|| approximation::power_edges(&self.succ, self.n, k))
    }

    /// Mehlhorn's spanning tree over the terminals (`steiner_tree`):
    /// `(missing, us, vs)`, where `missing` is the first node no terminal
    /// reaches. Distances use `dist_weight` (hop counts with `hops`); the
    /// `G_1'` weights add the `weight` attribute.
    #[pyo3(signature = (sources, dist_weight, hops, weight))]
    #[allow(clippy::type_complexity)]
    fn mehlhorn_terminal_mst(
        &self,
        py: Python<'_>,
        sources: Vec<u32>,
        dist_weight: Option<&str>,
        hops: bool,
        weight: Option<&str>,
    ) -> PyResult<(Option<u32>, Vec<u32>, Vec<u32>)> {
        let sources = self.sources_or_all(Some(sources))?;
        let wd = self.weight_slice(dist_weight, false)?;
        let w = self.weight_slice(weight, false)?;
        let tree = py
            .detach(|| more_paths::dijkstra_forest(&self.succ, self.n, wd, &sources, None, None))?;
        Ok(
            match py
                .detach(|| approximation::mehlhorn_terminal_mst(&self.succ, self.n, &tree, hops, w))
            {
                Ok(pairs) => {
                    let (us, vs) = pairs.into_iter().unzip();
                    (None, us, vs)
                }
                Err(v) => (Some(v), Vec::new(), Vec::new()),
            },
        )
    }

    /// `densest_subgraph` by greedy++ (`heap_init` is `None`) or FISTA
    /// (`heap_init`: the peeling heap's `(node, index into b)` fill order):
    /// `(best density, nodes removed in the best run, prefix)`. The caller
    /// checked FISTA's graph has no self-loops.
    #[pyo3(signature = (iterations, fista, heap_init=None))]
    #[allow(clippy::type_complexity)]
    fn densest_peeling(
        &self,
        py: Python<'_>,
        iterations: usize,
        fista: bool,
        heap_init: Option<Vec<(u32, u32)>>,
    ) -> PyResult<(f64, Vec<u32>, Option<usize>)> {
        let p = if fista {
            let init = heap_init.unwrap_or_else(|| (0..self.n as u32).map(|v| (v, v)).collect());
            if init.len() != self.n
                || init
                    .iter()
                    .any(|&(a, b)| a as usize >= self.n || b as usize >= self.n)
            {
                return Err(PyValueError::new_err("one heap entry per node"));
            }
            py.detach(|| approximation::fista_peeling(&self.succ, self.n, iterations, &init))
        } else {
            let degree = self.degrees();
            let m = self.number_of_edges();
            py.detach(|| {
                approximation::greedy_plus_plus(&self.succ, self.n, &degree, m, iterations)
            })
        };
        Ok((p.density, p.removed, p.prefix))
    }

    /// Edges of this graph, in `G.edges()` order, that `other` lacks;
    /// `map[v]` is node `v`'s position in `other`.
    fn edges_missing_from(
        &self,
        py: Python<'_>,
        other: &Bound<'_, CoreGraph>,
        map: Vec<u32>,
    ) -> PyResult<(Vec<u32>, Vec<u32>)> {
        let other = other.get();
        if map.len() != self.n || map.iter().any(|&h| h as usize >= other.n) {
            return Err(PyValueError::new_err(
                "map must give a node of other per node",
            ));
        }
        Ok(py.detach(|| {
            approximation::edges_missing_from(&self.succ, self.n, self.directed, &other.succ, &map)
        }))
    }

    // --- Batch 19: matrices and conversion ---

    /// `to_scipy_sparse_array`'s COO input `(row, col, data)` as native
    /// int64/f64 bytes; `map[v]` is node `v`'s row, -1 to leave it out
    /// (`None`: all nodes, in order).
    #[pyo3(signature = (map=None, weight=None))]
    fn adjacency_coo<'py>(
        &self,
        py: Python<'py>,
        map: Option<Vec<i64>>,
        weight: Option<&str>,
    ) -> PyResult<CooBytes<'py>> {
        let map = self.node_map(map)?;
        let w = self.weight_slice(weight, false)?;
        let coo =
            py.detach(|| conversion::adjacency_coo(&self.succ, self.n, self.directed, &map, w));
        coo_bytes(py, &coo)
    }

    /// `bipartite.biadjacency_matrix`'s COO input: `rows` holds
    /// `(position, row)` for the row nodes in `G`, `col[v]` node `v`'s column
    /// or -1.
    #[pyo3(signature = (rows, col, weight=None))]
    fn biadjacency_coo<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<(u32, i64)>,
        col: Vec<i64>,
        weight: Option<&str>,
    ) -> PyResult<CooBytes<'py>> {
        if col.len() != self.n || rows.iter().any(|&(u, _)| u as usize >= self.n) {
            return Err(PyValueError::new_err("rows and col must index this graph"));
        }
        let w = self.weight_slice(weight, false)?;
        let coo = py.detach(|| {
            conversion::biadjacency_coo(&self.succ, self.n, self.directed, &rows, &col, w)
        });
        coo_bytes(py, &coo)
    }

    /// `to_numpy_array`'s `(row, col, weight)` entries (see `adjacency_coo`).
    #[pyo3(signature = (map=None, weight=None))]
    fn dense_entries<'py>(
        &self,
        py: Python<'py>,
        map: Option<Vec<i64>>,
        weight: Option<&str>,
    ) -> PyResult<CooBytes<'py>> {
        let map = self.node_map(map)?;
        let w = self.weight_slice(weight, false)?;
        let coo = py.detach(|| conversion::dense_entries(&self.succ, self.n, &map, w));
        coo_bytes(py, &coo)
    }

    /// `incidence_matrix` as CSR bytes `(indptr, indices, data)` with
    /// `rows` rows, and the number of columns (edges); or the positions of
    /// the first edge with an endpoint outside `map`.
    #[pyo3(signature = (rows, map=None, weight=None, oriented=false))]
    #[allow(clippy::type_complexity)]
    fn incidence_csr<'py>(
        &self,
        py: Python<'py>,
        rows: usize,
        map: Option<Vec<i64>>,
        weight: Option<&str>,
        oriented: bool,
    ) -> PyResult<(Option<(CooBytes<'py>, usize)>, Option<(u32, u32)>)> {
        let map = self.node_map(map)?;
        if map.iter().any(|&i| i >= rows as i64) {
            return Err(PyValueError::new_err("map must give rows below `rows`"));
        }
        let w = self.weight_slice(weight, false)?;
        let found = py.detach(|| {
            conversion::incidence_csr(&self.succ, self.n, self.directed, rows, &map, w, oriented)
        });
        Ok(match found {
            Ok((indptr, indices, data, columns)) => (
                Some((
                    (
                        i64_bytes(py, &indptr),
                        i64_bytes(py, &indices),
                        f64_bytes(py, &data),
                    ),
                    columns,
                )),
                None,
            ),
            Err(edge) => (None, Some(edge)),
        })
    }

    /// Number of self-loops (`number_of_selfloops`).
    fn number_of_selfloops(&self) -> usize {
        (0..self.n)
            .filter(|&v| self.succ.neighbors(v).contains(&(v as u32)))
            .count()
    }

    /// Whether every value of an edge attribute is an exact Python int or
    /// float (see `Weights::plain`); true for `None` (unit weights).
    #[pyo3(signature = (weight=None))]
    fn weight_plain(&self, weight: Option<&str>) -> bool {
        weight
            .and_then(|a| self.weights.get(a))
            .is_none_or(|w| w.plain)
    }

    /// `is_weighted`: whether every edge's data in `adj` (`G._adj`) has `key`.
    #[staticmethod]
    fn all_edges_have(adj: &Bound<'_, PyAny>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        conversion::all_edges_have(conversion::plain_dict(adj)?, key)
    }

    /// `get_node_attributes` on `node` (`G._node`); `default=None` keeps only
    /// the nodes that have the attribute.
    #[staticmethod]
    #[pyo3(signature = (node, name, default=None))]
    fn node_attributes<'py>(
        node: &Bound<'py, PyAny>,
        name: &Bound<'py, PyAny>,
        default: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        conversion::node_attributes(conversion::plain_dict(node)?, name, default)
    }

    /// `get_edge_attributes` on the source graph's `_adj` (unchanged since
    /// this graph was built from it, with node order `nodes`).
    #[pyo3(signature = (nodes, adj, name, default=None))]
    fn edge_attributes<'py>(
        &self,
        nodes: &Bound<'py, PyList>,
        adj: &Bound<'py, PyAny>,
        name: &Bound<'py, PyAny>,
        default: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        conversion::edge_attributes(
            nodes,
            conversion::plain_dict(adj)?,
            &self.succ,
            self.directed,
            name,
            default,
        )
    }

    /// `relabel_nodes(copy=True)`: fill the new graph's `_node`, `_succ`
    /// (`_adj`) and `_pred` dicts from the source graph's `_node` and `_adj`.
    /// `labels[i]` is node `i`'s new label (`None`: unchanged).
    #[pyo3(signature = (nodes, node, adj, labels, new_node, new_succ, new_pred=None))]
    #[allow(clippy::too_many_arguments)]
    fn relabel_copy<'py>(
        &self,
        nodes: &Bound<'py, PyList>,
        node: &Bound<'py, PyAny>,
        adj: &Bound<'py, PyAny>,
        labels: Vec<Option<Bound<'py, PyAny>>>,
        new_node: &Bound<'py, PyDict>,
        new_succ: &Bound<'py, PyDict>,
        new_pred: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<()> {
        if labels.len() != self.n || nodes.len() != self.n || new_pred.is_some() != self.directed {
            return Err(PyValueError::new_err("arguments do not match this graph"));
        }
        conversion::relabel_copy(
            nodes,
            conversion::plain_dict(adj)?,
            conversion::plain_dict(node)?,
            &self.succ,
            self.directed,
            &labels,
            new_node,
            new_succ,
            new_pred,
        )
    }

    /// `to_dict_of_lists`: `keys` holds `(key, position)` in output order;
    /// `keep[v]` whether node `v` is in `nodelist` (`None`: all). `adj` is the
    /// source graph's `_adj` (`None` for native graphs).
    #[pyo3(signature = (nodes, adj, keys, keep=None))]
    fn dict_of_lists<'py>(
        &self,
        nodes: &Bound<'py, PyList>,
        adj: Option<&Bound<'py, PyAny>>,
        keys: Vec<(Bound<'py, PyAny>, u32)>,
        keep: Option<Vec<bool>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if nodes.len() != self.n
            || keep.as_ref().is_some_and(|k| k.len() != self.n)
            || keys.iter().any(|&(_, u)| u as usize >= self.n)
        {
            return Err(PyValueError::new_err("arguments do not match this graph"));
        }
        let adj = adj.map(conversion::plain_dict).transpose()?;
        conversion::dict_of_lists(nodes, adj, &self.succ, &keys, keep.as_deref())
    }

    /// Fill a new NetworkX graph's dicts (`node`, `adj`, and `pred` for a
    /// `DiGraph`) as NetworkX's builders do. `kind` picks the input:
    /// `"nodes"` (`add_nodes_from(data, **attr)`), `"edges"`
    /// (`add_edges_from(data)`), `"lists"` (`from_dict_of_lists`) or
    /// `"dicts"` (`from_dict_of_dicts`).
    #[staticmethod]
    #[pyo3(signature = (node, adj, pred, kind, data, attr=None))]
    fn build_into<'py>(
        node: Bound<'py, PyDict>,
        adj: Bound<'py, PyDict>,
        pred: Option<Bound<'py, PyDict>>,
        kind: &str,
        data: &Bound<'py, PyAny>,
        attr: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<()> {
        let b = conversion::NxBuilder(nxdicts::NxDicts::new(node, adj, pred));
        match kind {
            "nodes" => b.add_nodes(data, attr),
            "edges" => b.add_edges(data),
            "lists" => conversion::from_dict_of_lists(&b, conversion::plain_dict(data)?),
            "dicts" => conversion::from_dict_of_dicts(&b, conversion::plain_dict(data)?),
            _ => Err(PyValueError::new_err("unknown kind")),
        }
    }

    /// `add_weighted_edges_from` on a new NetworkX graph's dicts: edge `i`
    /// joins `labels[us[i]]` and `labels[vs[i]]` (the ints themselves if
    /// `labels` is `None`), with data `{attr: values[i]}`, or `{}` if
    /// `values` is `None`.
    #[staticmethod]
    #[pyo3(signature = (node, adj, pred, us, vs, labels=None, attr=None, values=None))]
    #[allow(clippy::too_many_arguments)]
    fn build_weighted_into<'py>(
        py: Python<'py>,
        node: Bound<'py, PyDict>,
        adj: Bound<'py, PyDict>,
        pred: Option<Bound<'py, PyDict>>,
        us: Vec<i64>,
        vs: Vec<i64>,
        labels: Option<Vec<Bound<'py, PyAny>>>,
        attr: Option<Bound<'py, PyAny>>,
        values: Option<Bound<'py, PyList>>,
    ) -> PyResult<()> {
        if us.len() != vs.len() || values.as_ref().is_some_and(|v| v.len() != us.len()) {
            return Err(PyValueError::new_err("edge arrays must agree"));
        }
        let label = |i: i64| -> PyResult<Bound<'py, PyAny>> {
            match &labels {
                Some(l) => usize::try_from(i)
                    .ok()
                    .and_then(|i| l.get(i).cloned())
                    .ok_or_else(|| PyValueError::new_err("label index out of range")),
                None => Ok(PyInt::new(py, i).into_any()),
            }
        };
        let b = conversion::NxBuilder(nxdicts::NxDicts::new(node, adj, pred));
        for (i, (&u, &v)) in us.iter().zip(&vs).enumerate() {
            let (u, v) = (label(u)?, label(v)?);
            b.add_edge(&u, &v, |d| match &values {
                Some(vals) => d.set_item(&attr, vals.get_item(i)?),
                None => Ok(()),
            })?;
        }
        Ok(())
    }

    // --- Batch 17: deterministic generators ---

    /// Builds generator `kind` (see `generators::build`) straight into a new
    /// NetworkX graph's dicts `node`, `adj` and, for a directed graph,
    /// `pred`. Node labels are the ids, `labels[id]`, or `(labels[i],
    /// cols[j])` for `grid_2d`. `False` (with the graph untouched) for a
    /// case rustnx leaves to NetworkX.
    #[staticmethod]
    #[pyo3(signature = (kind, params, lists, node, adj, pred, multigraph, labels=None, cols=None))]
    #[allow(clippy::too_many_arguments)]
    fn generate_graph<'py>(
        py: Python<'py>,
        kind: &str,
        params: Vec<i64>,
        lists: Vec<Vec<i64>>,
        node: &Bound<'py, PyDict>,
        adj: &Bound<'py, PyDict>,
        pred: Option<Bound<'py, PyDict>>,
        multigraph: bool,
        labels: Option<Vec<Bound<'py, PyAny>>>,
        cols: Option<Vec<Bound<'py, PyAny>>>,
    ) -> PyResult<bool> {
        let directed = pred.is_some();
        let kind_owned = kind.to_owned();
        let Some(built) =
            py.detach(|| generators::build(&kind_owned, &params, &lists, directed, multigraph))
        else {
            return Ok(false);
        };
        let cap = built.sim.capacity();
        let labels = match (labels, cols) {
            (Some(rows), Some(cols)) => {
                if rows.len().checked_mul(cols.len()) != Some(cap) && cap != 0 {
                    return Err(PyValueError::new_err("labels must cover every node"));
                }
                let mut out = vec![None; cap];
                for &v in &built.sim.order {
                    let (i, j) = (v as usize / cols.len(), v as usize % cols.len());
                    out[v as usize] = Some(PyTuple::new(py, [&rows[i], &cols[j]])?.into_any());
                }
                out
            }
            (Some(given), None) => {
                if given.len() < cap {
                    return Err(PyValueError::new_err("labels must cover every node"));
                }
                given.into_iter().take(cap).map(Some).collect()
            }
            _ => generated_labels(py, &built)?,
        };
        write_generated(py, &built, &labels, node, adj, pred.as_ref())?;
        Ok(true)
    }

    // --- Batch 18: random generators ---

    /// Replays `random.Random` calls (`pyrandom::replay`) from `state`, for
    /// the tests: each result as a float, an int, a list of ints or `None`.
    #[staticmethod]
    #[allow(clippy::type_complexity)]
    fn pyrandom_replay<'py>(
        py: Python<'py>,
        state: Vec<u32>,
        ops: Vec<(u8, i64, i64)>,
    ) -> PyResult<Option<(Vec<Bound<'py, PyAny>>, Vec<u32>)>> {
        let Some(mut rng) = pyrandom::Mt19937::from_state(&state) else {
            return Ok(None);
        };
        let draws = pyrandom::replay(&mut rng, &ops);
        let mut out = Vec::with_capacity(draws.len());
        for d in draws {
            out.push(match d {
                pyrandom::Draw::Float(x) => x.into_pyobject(py)?.into_any(),
                pyrandom::Draw::Int(x) => x.into_pyobject(py)?.into_any(),
                pyrandom::Draw::Ints(x) => PyList::new(py, x)?.into_any(),
                pyrandom::Draw::Unsupported => py.None().into_bound(py),
            });
        }
        Ok(Some((out, rng.state())))
    }

    /// `complete_graph(n)` (directed or not) into the empty graph `g`.
    #[staticmethod]
    fn rg_complete(py: Python<'_>, n: usize, g: &Bound<'_, PyAny>) -> PyResult<()> {
        let directed = g.call_method0("is_directed")?.is_truthy()?;
        let b = py.detach(|| random_generators::complete(n, directed));
        fill_generated(py, &b, g, None)
    }

    /// The nodes `0..n` and no edges, into `g`.
    #[staticmethod]
    fn rg_empty(n: usize, g: &Bound<'_, PyAny>) -> PyResult<()> {
        let directed = g.call_method0("is_directed")?.is_truthy()?;
        fill_generated(g.py(), &random_generators::Built::new(n, directed), g, None)
    }

    /// `gnp_random_graph` (`fast`: `fast_gnp_random_graph`) for `0 < p < 1`
    /// into `g`; the generator's new state, or `None` to let NetworkX run.
    #[staticmethod]
    fn rg_gnp(
        py: Python<'_>,
        n: usize,
        p: f64,
        fast: bool,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        let directed = g.call_method0("is_directed")?.is_truthy()?;
        rg_run(py, &state, g, None, |rng| {
            if fast {
                random_generators::fast_gnp(n, p, directed, rng)
            } else {
                Some(random_generators::gnp(n, p, directed, rng))
            }
        })
    }

    /// `gnm_random_graph` (`dense`: `dense_gnm_random_graph`) for `m` below
    /// the number of possible edges.
    #[staticmethod]
    fn rg_gnm(
        py: Python<'_>,
        n: usize,
        m: u64,
        dense: bool,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        let directed = g.call_method0("is_directed")?.is_truthy()?;
        rg_run(py, &state, g, None, |rng| {
            Some(if dense {
                random_generators::dense_gnm(n, m, rng)
            } else {
                random_generators::gnm(n, m, directed, rng)
            })
        })
    }

    /// `barabasi_albert_graph` (`m2 == 0`) or `dual_barabasi_albert_graph`
    /// with the default initial graph.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn rg_barabasi_albert(
        py: Python<'_>,
        n: usize,
        m1: usize,
        m2: usize,
        p: f64,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        rg_run(py, &state, g, None, |rng| {
            Some(if m2 == 0 {
                random_generators::barabasi_albert(n, m1, rng)
            } else {
                random_generators::dual_barabasi_albert(n, m1, m2, p, rng)
            })
        })
    }

    /// `extended_barabasi_albert_graph`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn rg_extended_barabasi_albert(
        py: Python<'_>,
        n: usize,
        m: usize,
        p: f64,
        q: f64,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        rg_run(py, &state, g, None, |rng| {
            random_generators::extended_barabasi_albert(n, m, p, q, rng)
        })
    }

    /// `watts_strogatz_graph` (kind 0), `newman_watts_strogatz_graph` (1)
    /// or `connected_watts_strogatz_graph` (2, with `tries`) for `k < n`.
    /// Returns `(found, state)`; `found` is false when every try of kind 2
    /// fails (NetworkX then raises).
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn rg_watts_strogatz(
        py: Python<'_>,
        kind: u8,
        n: usize,
        k: usize,
        p: f64,
        tries: u64,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<(bool, Vec<u32>)>> {
        let Some(mut rng) = pyrandom::Mt19937::from_state(&state) else {
            return Ok(None);
        };
        let built = py.detach(|| match kind {
            0 => random_generators::watts_strogatz(n, k, p, &mut rng).map(Some),
            1 => Some(Some(random_generators::newman_watts_strogatz(
                n, k, p, &mut rng,
            ))),
            _ => random_generators::connected_watts_strogatz(n, k, p, tries, &mut rng),
        });
        match built {
            None => Ok(None),
            Some(None) => Ok(Some((false, rng.state()))),
            Some(Some(b)) => {
                fill_generated(py, &b, g, None)?;
                Ok(Some((true, rng.state())))
            }
        }
    }

    /// `powerlaw_cluster_graph` (`1 <= m <= n`).
    #[staticmethod]
    fn rg_powerlaw_cluster(
        py: Python<'_>,
        n: usize,
        m: usize,
        p: f64,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        rg_run(py, &state, g, None, |rng| {
            Some(random_generators::powerlaw_cluster(n, m, p, rng))
        })
    }

    /// `random_regular_graph` (`0 < d < n`, `n * d` even).
    #[staticmethod]
    fn rg_random_regular(
        py: Python<'_>,
        d: usize,
        n: usize,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        rg_run(py, &state, g, None, |rng| {
            Some(random_generators::random_regular(d, n, rng))
        })
    }

    /// `gn_graph` (kind 0, `cumulative`: NetworkX 3.6's distribution),
    /// `gnr_graph` (1, with `p`) or `gnc_graph` (2), for `n >= 2`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn rg_growing_network(
        py: Python<'_>,
        kind: u8,
        n: usize,
        p: f64,
        cumulative: bool,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        rg_run(py, &state, g, None, |rng| match kind {
            0 => random_generators::gn(n, cumulative, rng),
            1 => Some(random_generators::gnr(n, p, rng)),
            _ => Some(random_generators::gnc(n, rng)),
        })
    }

    /// `random_uniform_k_out_graph`.
    #[staticmethod]
    fn rg_uniform_k_out(
        py: Python<'_>,
        n: usize,
        k: usize,
        self_loops: bool,
        with_replacement: bool,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        rg_run(py, &state, g, None, |rng| {
            let rows = random_generators::uniform_k_out(n, k, self_loops, with_replacement, rng)?;
            let mut b = random_generators::Built::new(n, true);
            for (u, row) in rows.into_iter().enumerate() {
                for v in row {
                    b.push_edge(u as u32, v);
                }
            }
            Some(b)
        })
    }

    /// `random_lobster_graph` (`0 <= n < 2^52`, `p1, p2 < 1`).
    #[staticmethod]
    fn rg_lobster(
        py: Python<'_>,
        n: u64,
        p1: f64,
        p2: f64,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        rg_run(py, &state, g, None, |rng| {
            Some(random_generators::lobster(n, p1, p2, rng))
        })
    }

    /// `random_tournament(n)`.
    #[staticmethod]
    fn rg_tournament(
        py: Python<'_>,
        n: usize,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        rg_run(py, &state, g, None, |rng| {
            Some(random_generators::tournament(n, rng))
        })
    }

    /// `stochastic_block_model` with nodes `0..` grouped in `parts` (each
    /// in its set's order) and the `block` attribute values `blocks`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn rg_stochastic_block_model<'py>(
        py: Python<'py>,
        parts: Vec<Vec<u32>>,
        p: Vec<Vec<f64>>,
        directed: bool,
        selfloops: bool,
        sparse: bool,
        legacy: bool,
        state: Vec<u32>,
        g: &Bound<'py, PyAny>,
        block_key: &Bound<'py, PyAny>,
        blocks: &Bound<'py, PyList>,
    ) -> PyResult<Option<Vec<u32>>> {
        let n: usize = parts.iter().map(Vec::len).sum();
        let nb = parts.len();
        if p.len() != nb
            || p.iter().any(|row| row.len() != nb)
            || blocks.len() != n
            || parts.iter().flatten().any(|&v| v as usize >= n)
        {
            return Err(PyValueError::new_err("inconsistent block model"));
        }
        rg_run(py, &state, g, Some((block_key, blocks)), |rng| {
            let mut b = random_generators::stochastic_block_model(
                &parts, &p, directed, selfloops, sparse, legacy, rng,
            )?;
            b.order = Some(parts.iter().flatten().copied().collect());
            Some(b)
        })
    }

    /// `random_geometric_graph` with drawn positions: `dim` draws per node,
    /// then the pairs within `radius`. Fills `g` (node attribute `pos_key`,
    /// each a list) and returns the new state; `None` to let NetworkX run.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn rg_geometric<'py>(
        py: Python<'py>,
        n: usize,
        radius: f64,
        dim: usize,
        p: f64,
        state: Vec<u32>,
        g: &Bound<'py, PyAny>,
        pos_key: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        let Some(mut rng) = pyrandom::Mt19937::from_state(&state) else {
            return Ok(None);
        };
        let found = py.detach(|| {
            let coords: Vec<f64> = (0..n * dim).map(|_| rng.random()).collect();
            let pairs = random_generators::geometric_pairs(&coords, dim, radius, p)?;
            Some((coords, random_generators::from_pairs(n, &pairs)))
        });
        let Some((coords, b)) = found else {
            return Ok(None);
        };
        let pos = PyList::empty(py);
        for row in coords.chunks(dim.max(1)).take(n) {
            pos.append(PyList::new(py, row)?)?;
        }
        if dim == 0 {
            for _ in 0..n {
                pos.append(PyList::empty(py))?;
            }
        }
        fill_generated(py, &b, g, Some((pos_key, &pos)))?;
        Ok(Some(rng.state()))
    }

    /// `waxman_graph` with the default metric over the rectangle with
    /// corner `(x0, y0)` and sides `(dx, dy)`; `l`: the given `L`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn rg_waxman<'py>(
        py: Python<'py>,
        n: usize,
        beta: f64,
        alpha: f64,
        l: Option<f64>,
        domain: (f64, f64, f64, f64),
        state: Vec<u32>,
        g: &Bound<'py, PyAny>,
        pos_key: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        let Some(mut rng) = pyrandom::Mt19937::from_state(&state) else {
            return Ok(None);
        };
        let (x0, dx, y0, dy) = domain;
        let found =
            py.detach(|| random_generators::waxman(n, beta, alpha, l, x0, dx, y0, dy, &mut rng));
        let Some((coords, b)) = found else {
            return Ok(None);
        };
        let pos = PyList::empty(py);
        for xy in coords.chunks(2) {
            pos.append(PyTuple::new(py, xy)?)?;
        }
        fill_generated(py, &b, g, Some((pos_key, &pos)))?;
        Ok(Some(rng.state()))
    }

    /// Bipartite `random_graph` (`gnmk == false`, `0 < p < 1`) or
    /// `gnmk_random_graph` (`k` edges, `bottom`: NetworkX's bottom list),
    /// with the `bipartite` attribute values `labels`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn rg_bipartite<'py>(
        py: Python<'py>,
        n: usize,
        m: usize,
        p: f64,
        k: u64,
        bottom: Option<Vec<u32>>,
        state: Vec<u32>,
        g: &Bound<'py, PyAny>,
        label_key: &Bound<'py, PyAny>,
        labels: &Bound<'py, PyList>,
    ) -> PyResult<Option<Vec<u32>>> {
        if labels.len() != n + m
            || bottom
                .as_ref()
                .is_some_and(|b| b.is_empty() || b.iter().any(|&v| v as usize >= n + m))
        {
            return Err(PyValueError::new_err("inconsistent bipartite sets"));
        }
        let directed = g.call_method0("is_directed")?.is_truthy()?;
        rg_run(
            py,
            &state,
            g,
            Some((label_key, labels)),
            |rng| match &bottom {
                None => random_generators::bipartite_random(n, m, p, directed, rng),
                Some(bottom) => Some(random_generators::gnmk(n, bottom, k, directed, rng)),
            },
        )
    }

    // --- Batch 20: readers and parsers ---

    /// `parse_edgelist` (and bipartite `parse_edgelist`) into the empty
    /// graph `graph`. `data`: 0 ignores edge data, 1 is `literal_eval`
    /// (joined with "," if `comma`, then stripped if `strip`), 2 converts
    /// with `keys` and `types` (nodetype codes). Returns False, leaving
    /// `graph` untouched, where NetworkX must run instead.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (graph, lines, mode, comments, delimiter, nodetype, data, keys, types, comma, strip, bipartite))]
    fn rw_edgelist(
        py: Python<'_>,
        graph: &Bound<'_, PyAny>,
        lines: &Bound<'_, PyAny>,
        mode: u8,
        comments: Option<&str>,
        delimiter: Option<&str>,
        nodetype: u8,
        data: u8,
        keys: Vec<String>,
        types: Vec<u8>,
        comma: bool,
        strip: bool,
        bipartite: bool,
    ) -> PyResult<bool> {
        let (Some(items), Some(nodetype)) = (
            readwrite::string_items(lines, mode),
            readwrite::NodeType::from_code(nodetype),
        ) else {
            return Ok(false);
        };
        let Some(strs) = readwrite::split_lines(&items, mode) else {
            return Ok(false);
        };
        let data = match data {
            0 => readwrite::EdgeData::Ignore,
            1 => readwrite::EdgeData::Literal { comma, strip },
            _ => {
                let Some(spec) = keys
                    .into_iter()
                    .zip(types)
                    .map(|(k, t)| readwrite::NodeType::from_code(t).map(|t| (k, t)))
                    .collect::<Option<Vec<_>>>()
                else {
                    return Ok(false);
                };
                readwrite::EdgeData::Typed(spec)
            }
        };
        let parsed = py
            .detach(|| readwrite::edgelist(&strs, comments, delimiter, nodetype, &data, bipartite));
        match parsed {
            Some(parsed) => readwrite::apply(py, &parsed, graph).map(|_| true),
            None => Ok(false),
        }
    }

    /// `parse_adjlist` into the empty graph `graph` (see `rw_edgelist`).
    #[staticmethod]
    fn rw_adjlist(
        py: Python<'_>,
        graph: &Bound<'_, PyAny>,
        lines: &Bound<'_, PyAny>,
        mode: u8,
        comments: &str,
        delimiter: Option<&str>,
        nodetype: u8,
    ) -> PyResult<bool> {
        let (Some(items), Some(nodetype)) = (
            readwrite::string_items(lines, mode),
            readwrite::NodeType::from_code(nodetype),
        ) else {
            return Ok(false);
        };
        let Some(strs) = readwrite::split_lines(&items, mode) else {
            return Ok(false);
        };
        match py.detach(|| readwrite::adjlist(&strs, comments, delimiter, nodetype)) {
            Some(parsed) => readwrite::apply(py, &parsed, graph).map(|_| true),
            None => Ok(false),
        }
    }

    /// `parse_multiline_adjlist` into the empty graph `graph`; `edgetype` is
    /// 0 for `literal_eval`, else a nodetype code plus one.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn rw_multiline_adjlist(
        py: Python<'_>,
        graph: &Bound<'_, PyAny>,
        lines: &Bound<'_, PyAny>,
        mode: u8,
        comments: &str,
        delimiter: Option<&str>,
        nodetype: u8,
        edgetype: u8,
    ) -> PyResult<bool> {
        let edgetype = match edgetype {
            0 => None,
            t => match readwrite::NodeType::from_code(t - 1) {
                Some(t) => Some(t),
                None => return Ok(false),
            },
        };
        let (Some(items), Some(nodetype)) = (
            readwrite::string_items(lines, mode),
            readwrite::NodeType::from_code(nodetype),
        ) else {
            return Ok(false);
        };
        let Some(strs) = readwrite::split_lines(&items, mode) else {
            return Ok(false);
        };
        let parsed = py.detach(|| {
            readwrite::multiline_adjlist(&strs, comments, delimiter, nodetype, edgetype)
        });
        match parsed {
            Some(parsed) => readwrite::apply(py, &parsed, graph).map(|_| true),
            None => Ok(false),
        }
    }

    /// `parse_leda` (`kind` 0) or `parse_pajek` (`kind` 1): the new graph, or
    /// None where NetworkX must run instead.
    #[staticmethod]
    fn rw_leda_pajek<'py>(
        py: Python<'py>,
        lines: &Bound<'py, PyAny>,
        mode: u8,
        kind: u8,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(items) = readwrite::string_items(lines, mode) else {
            return Ok(None);
        };
        let Some(strs) = readwrite::split_lines(&items, mode) else {
            return Ok(None);
        };
        let parsed = py.detach(|| {
            if kind == 0 {
                readwrite::leda(&strs)
            } else {
                readwrite::pajek(&strs)
            }
        });
        let Some(parsed) = parsed else {
            return Ok(None);
        };
        let graph = readwrite::new_nx_graph(py, parsed.class.0, parsed.class.1)?;
        readwrite::apply(py, &parsed, &graph)?;
        Ok(Some(graph))
    }

    /// `from_graph6_bytes` (`sparse` false) or `from_sparse6_bytes` for each
    /// of `chunks`; `file` reads them as `read_graph6` / `read_sparse6` do
    /// (one per stripped, non-empty line of the one chunk). None if any
    /// fails, so NetworkX runs instead.
    #[staticmethod]
    fn rw_graph6<'py>(
        py: Python<'py>,
        data: &Bound<'py, PyBytes>,
        sparse: bool,
        file: bool,
        strip_newline: bool,
    ) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
        let bytes = data.as_bytes();
        let chunks: Vec<&[u8]> = if file {
            bytes
                .split_inclusive(|&c| c == b'\n')
                .map(readwrite::py_bytes_strip)
                .filter(|line| !line.is_empty())
                .collect()
        } else {
            vec![bytes]
        };
        let parsed = py.detach(|| {
            chunks
                .iter()
                .map(|c| {
                    if sparse {
                        readwrite::sparse6(c)
                    } else {
                        readwrite::graph6(c, strip_newline)
                    }
                })
                .collect::<Option<Vec<_>>>()
        });
        let Some(parsed) = parsed else {
            return Ok(None);
        };
        let mut graphs = Vec::with_capacity(parsed.len());
        for p in &parsed {
            let graph = readwrite::new_nx_graph(py, p.class.0, p.class.1)?;
            readwrite::apply(py, p, &graph)?;
            graphs.push(graph);
        }
        Ok(Some(graphs))
    }

    /// `parse_gml` / `read_gml` with `destringizer=None`: the new graph, or
    /// None where NetworkX must run instead. `mode` as for
    /// `readwrite::gml_lines` (lists are mode 0, single strings 1 or 3).
    #[staticmethod]
    #[pyo3(signature = (lines, mode, label))]
    fn rw_gml<'py>(
        py: Python<'py>,
        lines: &Bound<'py, PyAny>,
        mode: u8,
        label: Option<&str>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(items) = readwrite::string_items(lines, mode.min(1)) else {
            return Ok(None);
        };
        let Some(strs) = readwrite::gml_lines(&items, mode) else {
            return Ok(None);
        };
        let Some(parsed) = py.detach(|| readwrite::gml(&strs, label)) else {
            return Ok(None);
        };
        let graph = readwrite::new_nx_graph(py, parsed.class.0, parsed.class.1)?;
        readwrite::apply(py, &parsed, &graph)?;
        Ok(Some(graph))
    }

    /// The loops of `node_link_graph` into the new, empty graph `graph`.
    /// False (leaving `graph` partly built) where NetworkX must run instead.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn rw_node_link<'py>(
        graph: &Bound<'py, PyAny>,
        nodes: &Bound<'py, PyAny>,
        edges: &Bound<'py, PyAny>,
        source: Bound<'py, PyAny>,
        target: Bound<'py, PyAny>,
        name: Bound<'py, PyAny>,
        key: Bound<'py, PyAny>,
    ) -> bool {
        let names = readwrite::NodeLinkNames {
            source,
            target,
            name,
            key,
        };
        readwrite::PyBuilder::new(graph)
            .and_then(|b| readwrite::node_link(&b, nodes, edges, &names))
            .is_some()
    }

    /// The loops of `adjacency_graph` (see `rw_node_link`).
    #[staticmethod]
    fn rw_adjacency<'py>(
        graph: &Bound<'py, PyAny>,
        nodes: &Bound<'py, PyAny>,
        adjacency: &Bound<'py, PyAny>,
        id: &Bound<'py, PyAny>,
        key: &Bound<'py, PyAny>,
    ) -> bool {
        readwrite::PyBuilder::new(graph)
            .and_then(|b| readwrite::adjacency(&b, nodes, adjacency, id, key))
            .is_some()
    }

    /// The loops of `cytoscape_graph` (see `rw_node_link`).
    #[staticmethod]
    fn rw_cytoscape<'py>(
        graph: &Bound<'py, PyAny>,
        nodes: &Bound<'py, PyAny>,
        edges: &Bound<'py, PyAny>,
        name: &Bound<'py, PyAny>,
        ident: &Bound<'py, PyAny>,
    ) -> bool {
        readwrite::PyBuilder::new(graph)
            .and_then(|b| readwrite::cytoscape(&b, nodes, edges, name, ident))
            .is_some()
    }

    /// `tree_graph` into the new, empty DiGraph `graph` (see `rw_node_link`).
    #[staticmethod]
    fn rw_tree<'py>(
        graph: &Bound<'py, PyAny>,
        data: &Bound<'py, PyAny>,
        ident: &Bound<'py, PyAny>,
        children: &Bound<'py, PyAny>,
    ) -> bool {
        readwrite::PyBuilder::new(graph)
            .and_then(|b| readwrite::tree(&b, data, ident, children))
            .is_some()
    }

    // --- Batch 21: operators and structure ---

    /// This graph's NetworkX dicts (`list(G)`, `G._node`, `G._adj`) read
    /// in the order NetworkX iterates them, for the operators.
    fn op_view(
        &self,
        nodes: &Bound<'_, PyList>,
        node_dict: &Bound<'_, PyDict>,
        adj: &Bound<'_, PyDict>,
    ) -> PyResult<operators::OpView> {
        operators::OpView::read(self, nodes, node_dict, adj)
    }

    /// The structural holes measures (`kind`: 0 `constraint`, 1
    /// `effective_size` by redundancy, 2 `effective_size` by ego graphs,
    /// 3 `local_constraint` of `(targets[i], others[i])`). `adj` is
    /// `G._adj` to read weights from (`None`: every weight is 1). `None`
    /// in the result stands for NaN.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (kind, nodes, adj, weight, targets, others, by_len, compensated))]
    fn structural_holes<'py>(
        &self,
        py: Python<'py>,
        kind: u8,
        nodes: &Bound<'py, PyList>,
        adj: Option<Bound<'py, PyDict>>,
        weight: &Bound<'py, PyAny>,
        targets: Vec<u32>,
        others: Vec<u32>,
        by_len: bool,
        compensated: bool,
    ) -> PyResult<Vec<Option<Bound<'py, PyAny>>>> {
        if targets.iter().chain(&others).any(|&v| v as usize >= self.n) {
            return Err(PyIndexError::new_err("node index out of range"));
        }
        let mut holes = operators::Holes::new(self, nodes, adj.as_ref(), weight, compensated)?;
        let mut mark = vec![false; if kind == 2 { self.n } else { 0 }];
        let mut out = Vec::with_capacity(targets.len());
        for (i, &v) in targets.iter().enumerate() {
            let val = match kind {
                0 => holes.constraint(v)?,
                1 => holes.effective_size(v, by_len)?,
                2 => holes.ego_effective_size(v, by_len, &mut mark)?,
                _ => {
                    let o = *others
                        .get(i)
                        .ok_or_else(|| PyIndexError::new_err("missing pair"))?;
                    Some(holes.local_constraint(v, o)?)
                }
            };
            out.push(match val {
                Some(x) => Some(val_obj(py, x)?),
                None => None,
            });
        }
        Ok(out)
    }

    /// `G.subgraph(nodes).copy()` (see `operators::subgraph_copy`).
    #[allow(clippy::too_many_arguments)]
    fn subgraph_copy_into<'py>(
        &self,
        py: Python<'py>,
        nodes: &Bound<'py, PyList>,
        node_dict: &Bound<'py, PyDict>,
        adj: &Bound<'py, PyDict>,
        members: Vec<u32>,
        node: Bound<'py, PyDict>,
        succ: Bound<'py, PyDict>,
        pred: Option<Bound<'py, PyDict>>,
    ) -> PyResult<()> {
        operators::subgraph_copy(py, self, nodes, node_dict, adj, &members, node, succ, pred)
    }

    /// `tree_broadcast_center` on a tree of 3 or more nodes (see
    /// `operators::tree_broadcast_center`), with `hashes` each node's
    /// `hash()`; `None` where NetworkX raises.
    fn tree_broadcast_center(
        &self,
        py: Python<'_>,
        hashes: Vec<i64>,
    ) -> PyResult<Option<(i64, Vec<u32>)>> {
        if hashes.len() != self.n || self.directed {
            return Err(PyValueError::new_err(
                "one hash per node of an undirected graph",
            ));
        }
        Ok(py.detach(|| operators::tree_broadcast_center(self, &hashes)))
    }

    /// Distances from the nearest of `sources` (`-1` where unreached).
    fn multi_source_distances(&self, py: Python<'_>, sources: Vec<u32>) -> PyResult<Vec<i64>> {
        if sources.iter().any(|&s| s as usize >= self.n) {
            return Err(PyIndexError::new_err("node index out of range"));
        }
        Ok(py.detach(|| operators::bfs_distances(self, &sources)))
    }

    /// Bipartite `degree_centrality` (see
    /// `operators::bipartite_degree_centrality`).
    fn bipartite_degree_centrality<'py>(
        &self,
        index: &Bound<'py, PyDict>,
        top: &Bound<'py, pyo3::types::PySet>,
        bottom: &Bound<'py, pyo3::types::PySet>,
        s_top: f64,
        s_bottom: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        operators::bipartite_degree_centrality(self, index, top, bottom, s_top, s_bottom)
    }

    // --- Batch 26: multigraphs and bipartite measures ---

    /// A multigraph snapshot's per-entry parallel-edge data (see
    /// `bipartite_more::MultiEdges`), read from `adj` (the source graph's
    /// `G._adj`, rows in `nodes` order): counts, and with `weighted` the
    /// values of edge attribute `attr` (looked up as NetworkX's
    /// `d.get(attr, 1)` does, so `None` is a key like any other).
    fn b26_multi_edges<'py>(
        &self,
        nodes: &Bound<'py, PyList>,
        adj: &Bound<'py, PyAny>,
        weighted: bool,
        attr: Bound<'py, PyAny>,
    ) -> PyResult<bipartite_more::MultiEdges> {
        let one = 1i64.into_pyobject(nodes.py())?.into_any();
        bipartite_more::read_multi(self, nodes, adj, weighted.then_some((attr, one)))
    }

    /// Multigraph `(degree, in-degree, out-degree)`, counting parallel edges.
    fn b26_multi_degrees(
        &self,
        me: &bipartite_more::MultiEdges,
    ) -> PyResult<(Vec<u64>, Vec<u64>, Vec<u64>)> {
        self.b26_check_multi(me)?;
        Ok(bipartite_more::multi_degrees(self, me))
    }

    /// PageRank of a multigraph: NetworkX's sparse matrix adds up parallel
    /// edges' weights (each edge counts 1 without a weight).
    #[allow(clippy::too_many_arguments)]
    fn b26_multi_pagerank(
        &self,
        py: Python<'_>,
        me: &bipartite_more::MultiEdges,
        alpha: f64,
        personalization: Option<Vec<f64>>,
        max_iter: usize,
        tol: f64,
        nstart: Option<Vec<f64>>,
        dangling: Option<Vec<f64>>,
    ) -> PyResult<Option<Vec<f64>>> {
        self.b26_check_multi(me)?;
        for v in [&personalization, &nstart, &dangling].into_iter().flatten() {
            if v.len() != self.n {
                return Err(PyValueError::new_err(
                    "vector length must equal the node count",
                ));
            }
        }
        let out_w: Vec<f64> = if me.weighted {
            me.sum.clone()
        } else {
            me.mult.iter().map(|&k| k as f64).collect()
        };
        let in_w: Vec<f64> = if self.directed {
            bipartite_more::pred_entries(self)?
                .into_iter()
                .map(|e| out_w[e])
                .collect()
        } else {
            out_w.clone()
        };
        let input = PagerankInput {
            n: self.n,
            out_adj: &self.succ,
            out_weights: Some(&out_w),
            in_adj: self.adj(true),
            in_weights: Some(&in_w),
            alpha,
            personalization,
            nstart,
            dangling,
            max_iter,
            tol,
            return_previous: false,
        };
        Ok(py.detach(|| link_analysis::pagerank(input).ok()))
    }

    /// Kruskal on an undirected multigraph: `(us, vs, key positions)`.
    fn b26_multi_kruskal(
        &self,
        py: Python<'_>,
        me: &bipartite_more::MultiEdges,
        maximum: bool,
    ) -> PyResult<(Vec<u32>, Vec<u32>, Vec<u32>)> {
        self.b26_check_multi(me)?;
        if !me.weighted {
            return Err(PyValueError::new_err("weights were not read"));
        }
        Ok(py.detach(|| bipartite_more::multi_kruskal(self, me, maximum)))
    }

    /// Prim on an undirected multigraph from `starts`: `(us, vs, key positions)`.
    fn b26_multi_prim(
        &self,
        py: Python<'_>,
        me: &bipartite_more::MultiEdges,
        starts: Vec<u32>,
        minimum: bool,
    ) -> PyResult<(Vec<u32>, Vec<u32>, Vec<u32>)> {
        self.b26_check_multi(me)?;
        if !me.weighted {
            return Err(PyValueError::new_err("weights were not read"));
        }
        let starts = self.sources_or_all(Some(starts))?;
        Ok(py.detach(|| bipartite_more::multi_prim(self, me, &starts, minimum)))
    }

    /// Bipartite `latapy_clustering` of `sources` (`mode` 0 dot, 1 min,
    /// 2 max), given each node's `hash()`.
    fn b26_latapy(
        &self,
        py: Python<'_>,
        sources: Vec<u32>,
        mode: u8,
        hashes: Vec<i64>,
    ) -> PyResult<Vec<f64>> {
        let sources = self.sources_or_all(Some(sources))?;
        if hashes.len() != self.n {
            return Err(PyValueError::new_err("one hash per node is needed"));
        }
        py.detach(|| bipartite_more::latapy(&self.succ, self.n, &sources, mode, &hashes))
            .ok_or_else(|| pyo3::exceptions::PyZeroDivisionError::new_err("division by zero"))
    }

    /// `(4 x 4-cycles, 2 x 3-paths)` for `robins_alexander_clustering`.
    fn b26_cycles_and_paths(&self, py: Python<'_>) -> (u64, u64) {
        py.detach(|| bipartite_more::cycles_and_paths(&self.succ, self.n))
    }

    /// `eppstein_matching` from its left nodes (positions, in the order
    /// the set iterates): the matching's items in dict order, or `None` if
    /// it would recurse deeper than `max_depth`.
    fn b26_eppstein_from(
        &self,
        py: Python<'_>,
        left: Vec<u32>,
        max_depth: usize,
    ) -> PyResult<Option<Vec<(u32, u32)>>> {
        let left = self.sources_or_all(Some(left))?;
        Ok(py.detach(|| {
            // `G.edges(left)`: undirected graphs skip neighbors already
            // listed as a start.
            let mut seen = vec![false; self.n];
            let mut edges = Vec::new();
            for &u in &left {
                for &v in self.succ.neighbors(u as usize) {
                    if self.directed || !seen[v as usize] {
                        edges.push((u, v));
                    }
                }
                seen[u as usize] = true;
            }
            bipartite_more::eppstein(self.n, &edges, max_depth)
        }))
    }

    /// `maximal_extendability` after its checks: the value, or `None` if
    /// the residual digraph is not strongly connected. `mate` gives each
    /// node's partner in the perfect matching.
    fn b26_extendability(
        &self,
        py: Python<'_>,
        in_u: Vec<bool>,
        in_v: Vec<bool>,
        mate: Vec<u32>,
    ) -> PyResult<Option<i64>> {
        if in_u.len() != self.n || in_v.len() != self.n || mate.len() != self.n {
            return Err(PyValueError::new_err("one entry per node is needed"));
        }
        if mate.iter().any(|&m| m as usize >= self.n) {
            return Err(PyIndexError::new_err("node index out of range"));
        }
        Ok(py.detach(|| {
            match bipartite_more::extendability(&self.succ, self.n, &in_u, &in_v, &mate) {
                bipartite_more::Extendability::NotStronglyConnected => None,
                bipartite_more::Extendability::Value(k) => Some(k),
            }
        }))
    }

    /// Adds the weighted projection's edges (`kind` 0 Jaccard overlap, 1
    /// min overlap, 2 collaboration) to the NetworkX graph `target`, whose
    /// nodes NetworkX's `add_nodes_from` already added. `nodes` are the
    /// node objects by position.
    #[allow(clippy::too_many_arguments)]
    fn b26_projection<'py>(
        &self,
        py: Python<'py>,
        nodes: &Bound<'py, PyList>,
        members: Vec<u32>,
        kind: u8,
        hashes: Vec<i64>,
        compensated: bool,
        target: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let members = self.sources_or_all(Some(members))?;
        if hashes.len() != self.n || nodes.len() != self.n {
            return Err(PyValueError::new_err(
                "one hash and node per node is needed",
            ));
        }
        let kind = match kind {
            0 => bipartite_more::Projection::Overlap(true),
            1 => bipartite_more::Projection::Overlap(false),
            _ => bipartite_more::Projection::Collaboration,
        };
        let (pred, _) = self.reverse_exact_order(None)?;
        let edges = py.detach(|| {
            bipartite_more::projection_edges(
                &self.succ,
                pred,
                self.n,
                &members,
                kind,
                &hashes,
                compensated,
            )
        })?;
        let dicts = nxdicts::NxDicts::of_graph(target)?;
        let weight = pyo3::intern!(py, "weight");
        let mut rows: Vec<Option<nxdicts::NodeRows<'py>>> = (0..self.n).map(|_| None).collect();
        for (u, v, w) in edges {
            for x in [u, v] {
                if rows[x as usize].is_none() {
                    rows[x as usize] = Some(dicts.node(&nodes.get_item(x as usize)?)?.0);
                }
            }
            let (ru, rv) = (
                rows[u as usize].as_ref().unwrap(),
                rows[v as usize].as_ref().unwrap(),
            );
            let (ku, kv) = (nodes.get_item(u as usize)?, nodes.get_item(v as usize)?);
            let data = nxdicts::simple_edge(&ru.succ, rv.back(), &ku, &kv)?;
            match w {
                flow::Val::I(i) => data.set_item(weight, i)?,
                flow::Val::F(x) => data.set_item(weight, x)?,
            }
        }
        Ok(())
    }

    /// Multigraph `(number_of_edges, number_of_selfloops)`.
    fn b26_multi_counts(&self, me: &bipartite_more::MultiEdges) -> PyResult<(u64, u64)> {
        self.b26_check_multi(me)?;
        Ok(bipartite_more::multi_counts(self, me))
    }

    /// Multigraph `s_metric` total (before `float`).
    fn b26_multi_s_metric(&self, me: &bipartite_more::MultiEdges) -> PyResult<u128> {
        self.b26_check_multi(me)?;
        let (degree, _, _) = bipartite_more::multi_degrees(self, me);
        Ok(bipartite_more::multi_s_metric(self, me, &degree))
    }

    /// `bipartite_modularity`'s pieces: each node's weighted degree, and
    /// each community's internal weight (communities as positions in set
    /// order). `rows` as for `cut_size_value`.
    #[pyo3(signature = (communities, compensated, weight=None, rows=None))]
    #[allow(clippy::type_complexity)]
    fn b26_modularity_parts<'py>(
        &self,
        py: Python<'py>,
        communities: Vec<Vec<u32>>,
        compensated: bool,
        weight: Option<&str>,
        rows: Option<&Bound<'py, PyList>>,
    ) -> PyResult<(Vec<Bound<'py, PyAny>>, Vec<Bound<'py, PyAny>>)> {
        for c in &communities {
            self.membership(c)?;
        }
        let w = self.cut_weights(weight, rows)?;
        let n = self.n;
        let (deg, internal) = py
            .detach(|| -> flow::Res<_> {
                Ok((
                    bipartite_more::weighted_degrees(&self.succ, n, &w, compensated)?,
                    bipartite_more::internal_weights(&self.succ, n, &w, &communities, compensated)?,
                ))
            })
            .map_err(fail_err)?;
        let objs = |v: Vec<flow::Val>| -> PyResult<Vec<Bound<'py, PyAny>>> {
            v.into_iter().map(|x| val_obj(py, x)).collect()
        };
        Ok((objs(deg)?, objs(internal)?))
    }

    /// `all(key in d for d in node.values())` over `G._node` (`cd_index`).
    #[staticmethod]
    fn b26_all_nodes_have(node: &Bound<'_, PyAny>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        for (_, d) in conversion::plain_dict(node)?.iter() {
            if !conversion::plain_dict(&d)?.contains(key)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn b26_check_multi(&self, me: &bipartite_more::MultiEdges) -> PyResult<()> {
        if me.mult.len() != self.succ.targets.len() {
            return Err(PyValueError::new_err("multigraph data of another graph"));
        }
        Ok(())
    }

    // --- Batch 24: generators and transforms ---

    /// `prefix_tree(paths)` / `prefix_tree_recursive(paths)` into the empty
    /// DiGraph given by its dicts; returns the longest path's length.
    #[staticmethod]
    fn b24_prefix_tree<'py>(
        py: Python<'py>,
        paths: &Bound<'py, PyAny>,
        node: &Bound<'py, PyDict>,
        succ: &Bound<'py, PyDict>,
        pred: &Bound<'py, PyDict>,
    ) -> PyResult<usize> {
        transforms::prefix_tree(py, paths, node, succ, pred)
    }

    /// The edges `interval_graph` adds, as `(us, vs)` positions.
    #[staticmethod]
    fn b24_interval_edges(py: Python<'_>, lo: Vec<f64>, hi: Vec<f64>) -> (Vec<u32>, Vec<u32>) {
        py.detach(|| transforms::interval_edges(&lo, &hi))
    }

    /// The edges `visibility_graph` adds (the path first).
    #[staticmethod]
    fn b24_visibility_edges(py: Python<'_>, values: Vec<f64>) -> (Vec<u32>, Vec<u32>) {
        py.detach(|| transforms::visibility_edges(&values))
    }

    /// `mycielskian`'s loop on the integer-labelled graph given by its dicts.
    #[staticmethod]
    fn b24_mycielskian<'py>(
        py: Python<'py>,
        node: &Bound<'py, PyDict>,
        adj: &Bound<'py, PyDict>,
        iterations: usize,
    ) -> PyResult<()> {
        transforms::mycielskian(py, node, adj, iterations)
    }

    /// `stochastic_graph(G)` into `H` (see `transforms::stochastic`).
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn b24_stochastic<'py>(
        py: Python<'py>,
        g_node: &Bound<'py, PyDict>,
        g_succ: &Bound<'py, PyDict>,
        weight: &Bound<'py, PyAny>,
        h_node: &Bound<'py, PyDict>,
        h_succ: &Bound<'py, PyDict>,
        h_pred: &Bound<'py, PyDict>,
        compensated: bool,
    ) -> PyResult<()> {
        transforms::stochastic(
            py,
            g_node,
            g_succ,
            weight,
            h_node,
            h_succ,
            h_pred,
            compensated,
        )
    }

    /// `inverse_line_graph`'s partition of the nodes into cells (positions),
    /// raising NetworkX's errors.
    fn b24_inverse_line_partition(
        &self,
        py: Python<'_>,
        hashes: Option<Vec<i64>>,
    ) -> PyResult<Vec<Vec<u32>>> {
        let found = py.detach(|| transforms::inverse_line_partition(self, hashes.as_deref()));
        found.map_err(|e| transforms::line_error(py, e).unwrap_or_else(|err| err))
    }

    /// The result edges of `inverse_line_graph` (see
    /// `transforms::inverse_line_edges`).
    #[staticmethod]
    fn b24_inverse_line_edges(cells_of: Vec<Vec<u32>>) -> (Vec<u32>, Vec<u32>) {
        transforms::inverse_line_edges(&cells_of)
    }

    /// `quotient_graph`'s default data for the blocks `block` (each node's
    /// block index): edges inside each block, the joined block pairs and
    /// their weight sums. `None` unless the weights are ints (or missing).
    #[allow(clippy::type_complexity)]
    fn b24_quotient(
        &self,
        py: Python<'_>,
        block: Vec<u32>,
        nblocks: usize,
        weight: Option<&str>,
    ) -> PyResult<Option<(Vec<i64>, Vec<(u32, u32)>, Vec<i64>)>> {
        if block.len() != self.n || block.iter().any(|&b| b as usize >= nblocks) {
            return Err(PyValueError::new_err("one block per node"));
        }
        let (all_int, hidden) = self.weights_info(weight);
        if !all_int || hidden {
            return Ok(None);
        }
        let w = self.weight_slice(weight, false)?;
        Ok(py.detach(|| transforms::quotient(self, &block, nblocks, w)))
    }

    // --- Batch 25: cliques, structure and approximation ---

    /// `find_cliques`' setup for the caller's `nodes` (positions, `None`
    /// for objects that aren't nodes): 0 they don't form a clique, 1 they
    /// are the only clique, 2 the search (in the returned iterator).
    fn clique_search(
        &self,
        py: Python<'_>,
        hashes: Vec<i64>,
        prefix: Vec<Option<u32>>,
    ) -> PyResult<(u8, Option<CliqueSearchIter>)> {
        self.b25_check_hashes(&hashes)?;
        if prefix.iter().flatten().any(|&v| v as usize >= self.n) {
            return Err(PyIndexError::new_err("node index out of range"));
        }
        let start = py.detach(|| cliques::CliqueSearch::start(&self.succ, self.n, hashes, &prefix));
        Ok(match start {
            cliques::CliqueStart::NotAClique => (0, None),
            cliques::CliqueStart::Only => (1, None),
            cliques::CliqueStart::Search(s) => (2, Some(CliqueSearchIter(s))),
        })
    }

    /// `make_clique_bipartite` into the empty `nx.Graph` dicts `b_node`,
    /// `b_adj`: G's nodes with `bipartite=1`, then per clique `i` the node
    /// `-i - 1` with `bipartite=0` and its edges.
    fn clique_bipartite_fill<'py>(
        &self,
        py: Python<'py>,
        hashes: Vec<i64>,
        nodes: &Bound<'py, PyList>,
        b_node: Bound<'py, PyDict>,
        b_adj: Bound<'py, PyDict>,
    ) -> PyResult<()> {
        self.b25_check_hashes(&hashes)?;
        if nodes.len() != self.n {
            return Err(PyValueError::new_err("node list does not match this graph"));
        }
        let found = py.detach(|| cliques::all_cliques(&self.succ, self.n, hashes));
        let dicts = nxdicts::NxDicts::new(b_node, b_adj, None);
        let bipartite = pyo3::intern!(py, "bipartite");
        let keys: Vec<Bound<'py, PyAny>> = nodes.iter().collect();
        for key in &keys {
            let (rows, _) = dicts.node(key)?;
            rows.attrs.set_item(bipartite, 1)?;
        }
        for (i, clique) in found.iter().enumerate() {
            let name = (-(i as i64) - 1).into_pyobject(py)?.into_any();
            let (top, _) = dicts.node(&name)?;
            top.attrs.set_item(bipartite, 0)?;
            for &v in clique {
                let (rows, _) = dicts.node(&keys[v as usize])?;
                nxdicts::simple_edge(&rows.succ, &top.succ, &keys[v as usize], &name)?;
            }
        }
        Ok(())
    }

    /// Every maximal clique in `find_cliques`' order (positions).
    fn maximal_cliques(&self, py: Python<'_>, hashes: Vec<i64>) -> PyResult<Vec<Vec<u32>>> {
        self.b25_check_hashes(&hashes)?;
        Ok(py.detach(|| cliques::all_cliques(&self.succ, self.n, hashes)))
    }

    /// `make_max_clique_graph`'s node count and edges `(us[k], vs[k])`.
    fn max_clique_graph_edges(
        &self,
        py: Python<'_>,
        hashes: Vec<i64>,
    ) -> PyResult<(usize, Vec<u32>, Vec<u32>)> {
        self.b25_check_hashes(&hashes)?;
        Ok(py.detach(|| {
            let found = cliques::all_cliques(&self.succ, self.n, hashes);
            let edges = cliques::clique_overlaps(&found, self.n);
            (
                found.len(),
                edges.iter().map(|e| e.0).collect(),
                edges.iter().map(|e| e.1).collect(),
            )
        }))
    }

    /// `dominating_set(G, start)` on a non-empty graph (`None`: NetworkX's
    /// arbitrary start): the nodes in the order they join it.
    #[pyo3(signature = (hashes, start=None))]
    fn dominating_set_order(
        &self,
        py: Python<'_>,
        hashes: Vec<i64>,
        start: Option<usize>,
    ) -> PyResult<Vec<u32>> {
        self.b25_check_hashes(&hashes)?;
        if let Some(s) = start {
            self.check_index(s)?;
        }
        if self.n == 0 {
            return Err(PyValueError::new_err("the graph is empty"));
        }
        let start = start.map(|s| s as u32);
        Ok(py.detach(|| cliques::dominating_set(&self.succ, self.n, &hashes, start)))
    }

    /// `maximal_independent_set`'s loop from the set `nodes` and the
    /// generator state `state`: the nodes added and the new state.
    fn maximal_independent_draws(
        &self,
        py: Python<'_>,
        hashes: Vec<i64>,
        nodes: Vec<u32>,
        state: Vec<u32>,
    ) -> PyResult<Option<(Vec<u32>, Vec<u32>)>> {
        self.b25_check_hashes(&hashes)?;
        if nodes.iter().any(|&v| v as usize >= self.n) {
            return Err(PyIndexError::new_err("node index out of range"));
        }
        Ok(py.detach(|| {
            let mut rng = pyrandom::Mt19937::from_state(&state)?;
            let added =
                cliques::maximal_independent_set(&self.succ, self.n, &hashes, &nodes, &mut rng);
            Some((added, rng.state()))
        }))
    }

    /// `approximation.large_clique_size`.
    fn large_clique_size(&self, py: Python<'_>, hashes: Vec<i64>) -> PyResult<usize> {
        self.b25_check_hashes(&hashes)?;
        let degree = self.degrees();
        Ok(py.detach(|| cliques::large_clique_size(&self.succ, self.n, &hashes, &degree)))
    }

    /// `approximation.ramsey_R2`: the clique and the independent set, each
    /// in the order its nodes were added.
    fn ramsey_r2(&self, py: Python<'_>, hashes: Vec<i64>) -> PyResult<(Vec<u32>, Vec<u32>)> {
        self.b25_check_hashes(&hashes)?;
        Ok(py.detach(|| cliques::ramsey_r2(&self.succ, self.n, &hashes)))
    }

    /// `approximation.clique_removal` on G (or on `nx.complement(G)`, for
    /// `max_clique`): the index of the largest independent set, the
    /// independent sets and the cliques, each in add order.
    #[allow(clippy::type_complexity)]
    fn clique_removal(
        &self,
        py: Python<'_>,
        hashes: Vec<i64>,
        complement: bool,
    ) -> PyResult<(usize, Vec<Vec<u32>>, Vec<Vec<u32>>)> {
        self.b25_check_hashes(&hashes)?;
        Ok(py.detach(|| {
            let g = if complement {
                cliques::SimGraph::complement(&self.succ, self.n)
            } else {
                cliques::SimGraph::of(&self.succ, self.n)
            };
            cliques::clique_removal(g, self.n, &hashes)
        }))
    }

    /// `tournament.hamiltonian_path` (positions).
    fn hamiltonian_path(&self, py: Python<'_>, hashes: Vec<i64>) -> PyResult<Vec<u32>> {
        self.b25_check_hashes(&hashes)?;
        Ok(py.detach(|| {
            let rows: Vec<Vec<u32>> = (0..self.n)
                .map(|v| {
                    let mut row = self.succ.neighbors(v).to_vec();
                    row.sort_unstable();
                    row
                })
                .collect();
            cliques::hamiltonian_path(&rows, &hashes)
        }))
    }

    /// `chordal_graph_cliques`: each clique in the order its set was built,
    /// and how the generator ends (0 normally, 1 "Input graph is not
    /// chordal.", 2 the self-loop error of `_is_complete_graph`).
    fn chordal_cliques(&self, py: Python<'_>, hashes: Vec<i64>) -> PyResult<(Vec<Vec<u32>>, u8)> {
        self.b25_check_hashes(&hashes)?;
        Ok(py.detach(|| {
            let (found, error) = cliques::chordal_graph_cliques(&self.succ, self.n, &hashes);
            let code = match error {
                None => 0,
                Some(cliques::ChordalError::NotChordal) => 1,
                Some(cliques::ChordalError::SelfLoop) => 2,
            };
            (found, code)
        }))
    }

    /// `treewidth_min_degree`'s elimination order and, per bag added, the
    /// index of the bag it joins (see `cliques::treewidth_min_degree`).
    fn min_degree_eliminations(
        &self,
        py: Python<'_>,
        hashes: Vec<i64>,
    ) -> PyResult<(Vec<u32>, Vec<u32>)> {
        self.b25_check_hashes(&hashes)?;
        Ok(py.detach(|| cliques::treewidth_min_degree(&self.succ, self.n, &hashes)))
    }

    /// `approximation.local_node_connectivity` for each `(us[k], vs[k])`
    /// (distinct nodes), with `cutoff` (`None`: no cutoff). Directed graphs
    /// need exact in-edge order.
    #[pyo3(signature = (us, vs, cutoff=None))]
    fn approx_local_connectivity(
        &self,
        py: Python<'_>,
        us: Vec<u32>,
        vs: Vec<u32>,
        cutoff: Option<usize>,
    ) -> PyResult<Vec<usize>> {
        if us.len() != vs.len() || us.iter().chain(&vs).any(|&v| v as usize >= self.n) {
            return Err(PyIndexError::new_err("node index out of range"));
        }
        let back = self.b25_exact_back()?;
        let pairs: Vec<(u32, u32)> = us.into_iter().zip(vs).collect();
        let cutoff = cutoff.unwrap_or(usize::MAX);
        Ok(py.detach(|| cliques::approx_pairs(&self.succ, back, self.n, &pairs, cutoff)))
    }

    /// `approximation.node_connectivity(G)` for a connected G, from `v`, the
    /// first node of the smallest degree `min_degree`.
    fn approx_node_connectivity(
        &self,
        py: Python<'_>,
        v: usize,
        min_degree: usize,
    ) -> PyResult<usize> {
        self.check_index(v)?;
        let back = self.b25_exact_back()?;
        Ok(py.detach(|| {
            cliques::approx_node_connectivity(
                &self.succ,
                back,
                self.n,
                self.directed,
                v as u32,
                min_degree,
            )
        }))
    }

    /// Fills the new `nx.DiGraph` dicts as `build_auxiliary_edge_connectivity`
    /// (`node_ids` None) or `build_auxiliary_node_connectivity` (`node_ids`
    /// the nodes, giving `f"{i}A"` / `f"{i}B"` nodes with `id` attributes)
    /// leaves them; `nodes` are G's nodes.
    #[pyo3(signature = (nodes, h_node, h_succ, h_pred, node_ids))]
    fn auxiliary_fill<'py>(
        &self,
        py: Python<'py>,
        nodes: &Bound<'py, PyList>,
        h_node: Bound<'py, PyDict>,
        h_succ: Bound<'py, PyDict>,
        h_pred: Bound<'py, PyDict>,
        node_ids: bool,
    ) -> PyResult<()> {
        if nodes.len() != self.n {
            return Err(PyValueError::new_err("node list does not match this graph"));
        }
        let pairs = py.detach(|| cliques::auxiliary_edge_pairs(&self.succ, self.n, self.directed));
        let dicts = nxdicts::NxDicts::new(h_node, h_succ, Some(h_pred));
        let capacity = pyo3::intern!(py, "capacity");
        let one = 1i64.into_pyobject(py)?;
        let edge = |a: &nxdicts::NodeRows<'py>,
                    b: &nxdicts::NodeRows<'py>,
                    ka: &Bound<'py, PyAny>,
                    kb: &Bound<'py, PyAny>|
         -> PyResult<()> {
            let d = PyDict::new(py);
            d.set_item(capacity, &one)?;
            a.succ.set_item(kb, &d)?;
            b.back().set_item(ka, d)
        };
        if !node_ids {
            let keys: Vec<Bound<'py, PyAny>> = nodes.iter().collect();
            let rows = keys
                .iter()
                .map(|k| dicts.create_node(k))
                .collect::<PyResult<Vec<_>>>()?;
            for (a, b) in pairs {
                let (a, b) = (a as usize, b as usize);
                edge(&rows[a], &rows[b], &keys[a], &keys[b])?;
            }
            return Ok(());
        }
        let id = pyo3::intern!(py, "id");
        let mut a_keys = Vec::with_capacity(self.n);
        let mut b_keys = Vec::with_capacity(self.n);
        let mut a_rows = Vec::with_capacity(self.n);
        let mut b_rows = Vec::with_capacity(self.n);
        for (i, node) in nodes.iter().enumerate() {
            let ka = pyo3::types::PyString::new(py, &format!("{i}A")).into_any();
            let kb = pyo3::types::PyString::new(py, &format!("{i}B")).into_any();
            let ra = dicts.create_node(&ka)?;
            ra.attrs.set_item(id, &node)?;
            let rb = dicts.create_node(&kb)?;
            rb.attrs.set_item(id, &node)?;
            edge(&ra, &rb, &ka, &kb)?;
            a_keys.push(ka);
            b_keys.push(kb);
            a_rows.push(ra);
            b_rows.push(rb);
        }
        for (s, t) in pairs {
            let (s, t) = (s as usize, t as usize);
            edge(&b_rows[s], &a_rows[t], &b_keys[s], &a_keys[t])?;
        }
        Ok(())
    }

    /// `find_asteroidal_triple` (undirected, 6 or more nodes); `tuple_set`
    /// for NetworkX 3.4's order of non-edges.
    fn asteroidal_triple(
        &self,
        py: Python<'_>,
        hashes: Vec<i64>,
        tuple_set: bool,
    ) -> PyResult<Option<(u32, u32, u32)>> {
        self.b25_check_hashes(&hashes)?;
        Ok(py.detach(|| cliques::asteroidal_triple(&self.succ, self.n, &hashes, tuple_set)))
    }

    /// `k_clique_communities`: the maximal cliques of size `k` or more and,
    /// per community, the indices of its cliques in the order
    /// `frozenset.union` takes them. `clique_hashes(cliques)` gives each
    /// clique's `frozenset` hash.
    #[allow(clippy::type_complexity)]
    fn k_clique_communities<'py>(
        &self,
        py: Python<'py>,
        hashes: Vec<i64>,
        k: usize,
        clique_hashes: &Bound<'py, PyAny>,
    ) -> PyResult<(Vec<Vec<u32>>, Vec<Vec<u32>>)> {
        self.b25_check_hashes(&hashes)?;
        let found: Vec<Vec<u32>> = py.detach(|| {
            cliques::all_cliques(&self.succ, self.n, hashes.clone())
                .into_iter()
                .filter(|c| c.len() >= k)
                .collect()
        });
        let fs_hashes: Vec<i64> = clique_hashes.call1((found.clone(),))?.extract()?;
        if fs_hashes.len() != found.len() {
            return Err(PyValueError::new_err("one hash per clique"));
        }
        let comps =
            py.detach(|| cliques::k_clique_communities(&found, &fs_hashes, &hashes, self.n, k));
        Ok((found, comps))
    }

    /// The iteration order of `set(G)` (one `add` per node, in order).
    fn set_iteration_order(&self, hashes: Vec<i64>) -> PyResult<Vec<u32>> {
        self.b25_check_hashes(&hashes)?;
        Ok(pyset::PySet::from_iter(0..self.n as u32, &hashes)
            .iter()
            .collect())
    }

    /// `metric_closure` into the empty `nx.Graph` dicts `m_node`, `m_adj`:
    /// for each node `u` in order, an edge to every node still in `Gnodes`
    /// (`set(G)`, iterating as `set_order`) with `distance` and `path` from
    /// Dijkstra. Returns False, filling nothing, when the first node doesn't
    /// reach every node (NetworkX raises).
    #[allow(clippy::too_many_arguments)]
    fn metric_closure_fill<'py>(
        &self,
        py: Python<'py>,
        weight: Option<&str>,
        all_int: bool,
        nodes: &Bound<'py, PyList>,
        set_order: Vec<u32>,
        m_node: Bound<'py, PyDict>,
        m_adj: Bound<'py, PyDict>,
    ) -> PyResult<bool> {
        if nodes.len() != self.n || set_order.len() != self.n {
            return Err(PyValueError::new_err("node list does not match this graph"));
        }
        if set_order.iter().any(|&v| v as usize >= self.n) {
            return Err(PyIndexError::new_err("node index out of range"));
        }
        let w = self.weight_slice(weight, false)?;
        let dicts = nxdicts::NxDicts::new(m_node, m_adj, None);
        let keys: Vec<Bound<'py, PyAny>> = nodes.iter().collect();
        let distance = pyo3::intern!(py, "distance");
        let path_key = pyo3::intern!(py, "path");
        let mut alive = vec![true; self.n];
        let batch = 64usize;
        let mut start = 0;
        while start < self.n {
            let end = (start + batch).min(self.n);
            let trees = py.detach(|| {
                use rayon::prelude::*;
                (start..end)
                    .into_par_iter()
                    .map(|s| paths::dijkstra_tree(&self.succ, self.n, w, s, None, None))
                    .collect::<Vec<_>>()
            });
            for (k, tree) in trees.into_iter().enumerate() {
                let u = start + k;
                let tree = tree?;
                if u == 0 && tree.order.len() != self.n {
                    return Ok(false);
                }
                let mut dist = vec![0.0; self.n];
                for (&v, &d) in tree.order.iter().zip(&tree.dist) {
                    dist[v as usize] = d;
                }
                alive[u] = false;
                let mut chain = Vec::new();
                for &v in &set_order {
                    let v = v as usize;
                    if !alive[v] {
                        continue;
                    }
                    // add_edge(u, v): u, then v, join the graph if new
                    let (row_u, _) = dicts.node(&keys[u])?;
                    let (row_v, _) = dicts.node(&keys[v])?;
                    chain.clear();
                    let mut x = v as u32;
                    while x as usize != u {
                        chain.push(x);
                        x = tree.parent[x as usize];
                    }
                    chain.push(u as u32);
                    let path = PyList::new(py, chain.iter().rev().map(|&x| &keys[x as usize]))?;
                    let d = PyDict::new(py);
                    if all_int {
                        d.set_item(distance, dist[v] as i64)?;
                    } else {
                        d.set_item(distance, dist[v])?;
                    }
                    d.set_item(path_key, path)?;
                    row_u.succ.set_item(&keys[v], &d)?;
                    row_v.succ.set_item(&keys[u], d)?;
                }
            }
            start = end;
        }
        Ok(true)
    }

    /// Whether `kernighan_lin_bisection` can run in rustnx with `weight`
    /// (`skip_hidden`: `None` weights hide edges, 3.6+; before, they fail).
    fn kl_supported(&self, weight: Option<&str>, skip_hidden: bool) -> bool {
        self.b25_kl_weights(weight, skip_hidden).is_some()
    }

    /// `kernighan_lin_bisection`'s sweeps (see `cliques::kernighan_lin`):
    /// the final sides, or `None` when a sweep has no moves.
    #[allow(clippy::too_many_arguments)]
    fn kernighan_lin(
        &self,
        py: Python<'_>,
        weight: Option<&str>,
        skip_hidden: bool,
        order: Vec<u32>,
        side: Vec<bool>,
        max_iter: usize,
        compensated: bool,
    ) -> PyResult<Option<Vec<bool>>> {
        if side.len() != self.n
            || order.len() != self.n
            || order.iter().any(|&v| v as usize >= self.n)
        {
            return Err(PyIndexError::new_err("node index out of range"));
        }
        let weights = self
            .b25_kl_weights(weight, skip_hidden)
            .ok_or_else(|| PyNotImplementedError::new_err("unsupported weights"))?;
        py.detach(|| {
            cliques::kernighan_lin(&self.succ, &weights, &order, side, max_iter, compensated)
        })
        .map_err(|_| PyNotImplementedError::new_err("an int overflowed"))
    }

    // --- Batch 22: degree-sequence generators ---

    /// `configuration_model` (undirected class) into the empty graph `g`;
    /// the generator's new state.
    #[staticmethod]
    fn dg_configuration(
        py: Python<'_>,
        degree: Vec<u32>,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        rg_run(py, &state, g, None, |rng| {
            Some(degree_generators::configuration_model(&degree, rng))
        })
    }

    /// `directed_configuration_model` (any class) into `g`.
    #[staticmethod]
    fn dg_directed_configuration(
        py: Python<'_>,
        out_degree: Vec<u32>,
        in_degree: Vec<u32>,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        let directed = g.call_method0("is_directed")?.is_truthy()?;
        rg_run(py, &state, g, None, |rng| {
            Some(degree_generators::directed_configuration_model(
                &out_degree,
                &in_degree,
                directed,
                rng,
            ))
        })
    }

    /// Bipartite `configuration_model` into `g`, with node attribute `key`
    /// set to `labels[v]`.
    #[staticmethod]
    fn dg_bipartite_configuration<'py>(
        py: Python<'py>,
        aseq: Vec<u32>,
        bseq: Vec<u32>,
        state: Vec<u32>,
        g: &Bound<'py, PyAny>,
        key: &Bound<'py, PyAny>,
        labels: &Bound<'py, PyList>,
    ) -> PyResult<Option<Vec<u32>>> {
        rg_run(py, &state, g, Some((key, labels)), |rng| {
            Some(degree_generators::bipartite_configuration_model(
                &aseq, &bseq, rng,
            ))
        })
    }

    /// `random_clustered_graph` into `g`.
    #[staticmethod]
    fn dg_clustered(
        py: Python<'_>,
        single: Vec<u32>,
        triangle: Vec<u32>,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        rg_run(py, &state, g, None, |rng| {
            Some(degree_generators::random_clustered(&single, &triangle, rng))
        })
    }

    /// `expected_degree_graph` into `g` (a `Graph`); `None` where NetworkX
    /// raises after drawing.
    #[staticmethod]
    fn dg_expected_degree(
        py: Python<'_>,
        w: Vec<f64>,
        rho: f64,
        selfloops: bool,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        if w.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err(PyValueError::new_err(
                "weights must be finite and non-negative",
            ));
        }
        rg_run(py, &state, g, None, |rng| {
            degree_generators::expected_degree(&w, rho, selfloops, rng)
        })
    }

    /// `havel_hakimi_graph` for a graphical sequence, into `g`.
    #[staticmethod]
    fn dg_havel_hakimi(py: Python<'_>, degree: Vec<u32>, g: &Bound<'_, PyAny>) -> PyResult<()> {
        if degree.iter().any(|&d| d as usize >= degree.len()) {
            return Err(PyValueError::new_err("degrees must be below the length"));
        }
        let b = py.detach(|| degree_generators::havel_hakimi(&degree));
        fill_generated(py, &b, g, None)
    }

    /// `directed_havel_hakimi_graph` for non-negative sequences of one
    /// length with equal sums, into `g`; false for a non-digraphical pair
    /// (`g` untouched).
    #[staticmethod]
    fn dg_directed_havel_hakimi(
        py: Python<'_>,
        in_degree: Vec<u32>,
        out_degree: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        if in_degree.len() != out_degree.len() {
            return Err(PyValueError::new_err("sequences of one length"));
        }
        let directed = g.call_method0("is_directed")?.is_truthy()?;
        let found = py
            .detach(|| degree_generators::directed_havel_hakimi(&in_degree, &out_degree, directed));
        match found {
            Ok(b) => fill_generated(py, &b, g, None).map(|_| true),
            Err(()) => Ok(false),
        }
    }

    /// `degree_sequence_tree` for a valid sequence, into `g` (`legacy`:
    /// NetworkX 3.4 and 3.5, which may remove node 0).
    #[staticmethod]
    fn dg_degree_sequence_tree(
        py: Python<'_>,
        degree: Vec<i64>,
        legacy: bool,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let leaves: i128 = degree
            .iter()
            .filter(|&&d| d > 1)
            .map(|&d| d as i128 - 2)
            .sum();
        if leaves + degree.len() as i128 + 2 >= 1 << 31 {
            return Err(PyValueError::new_err("too many nodes"));
        }
        let b = py.detach(|| degree_generators::degree_sequence_tree(&degree, legacy));
        fill_generated(py, &b, g, None)
    }

    /// The bipartite Havel-Hakimi generators (`kind` 0 `havel_hakimi_graph`,
    /// 1 `reverse_havel_hakimi_graph`, 2 `alternating_havel_hakimi_graph`)
    /// into `g`, with node attribute `key` set to `labels[v]`.
    #[staticmethod]
    fn dg_bipartite_havel_hakimi<'py>(
        py: Python<'py>,
        aseq: Vec<u32>,
        bseq: Vec<u32>,
        kind: u8,
        g: &Bound<'py, PyAny>,
        key: &Bound<'py, PyAny>,
        labels: &Bound<'py, PyList>,
    ) -> PyResult<()> {
        let b = py.detach(|| degree_generators::bipartite_havel_hakimi(&aseq, &bseq, kind));
        fill_generated(py, &b, g, Some((key, labels)))
    }

    /// `random_powerlaw_tree_sequence(n, gamma, tries)` with `alpha = gamma
    /// - 1`: `(sequence, state)`, the sequence `None` where NetworkX gives
    /// up after `tries`; `None` where its arithmetic raises.
    #[staticmethod]
    #[allow(clippy::type_complexity)]
    fn dg_powerlaw_tree_sequence(
        py: Python<'_>,
        n: usize,
        alpha: f64,
        tries: usize,
        legacy: bool,
        state: Vec<u32>,
    ) -> PyResult<Option<(Option<Vec<u64>>, Vec<u32>)>> {
        let Some(mut rng) = pyrandom::Mt19937::from_state(&state) else {
            return Ok(None);
        };
        let found = py.detach(|| {
            degree_generators::powerlaw_tree_sequence(n, alpha, tries, legacy, &mut rng)
        });
        Ok(found.map(|r| (r.ok(), rng.state())))
    }

    /// `random_labeled_tree(n)` for `n >= 2` into `g`; with `rooted`, also
    /// `random_labeled_rooted_tree`'s root. `(root, state)`.
    #[staticmethod]
    fn dg_labeled_tree(
        py: Python<'_>,
        n: usize,
        rooted: bool,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<(Option<i64>, Vec<u32>)>> {
        let Some(mut rng) = pyrandom::Mt19937::from_state(&state) else {
            return Ok(None);
        };
        let (b, root) = py.detach(|| {
            let b = degree_generators::random_labeled_tree(n, &mut rng);
            let root = if rooted {
                rng.randint(0, n as i64 - 1)
            } else {
                None
            };
            (b, root)
        });
        fill_generated(py, &b, g, None)?;
        Ok(Some((root, rng.state())))
    }

    /// `random_cograph(n)` into `g`.
    #[staticmethod]
    fn dg_cograph(
        py: Python<'_>,
        n: u32,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        if n > 26 {
            return Err(PyValueError::new_err("too many nodes"));
        }
        rg_run(py, &state, g, None, |rng| {
            Some(degree_generators::random_cograph(n, rng))
        })
    }

    /// `random_degree_sequence_graph(sequence, tries)` for a graphical
    /// sequence, into `g` (a `Graph`): `(built, state)`, `built` false when
    /// every try failed (`g` untouched); `None` to let NetworkX run.
    #[staticmethod]
    fn dg_random_degree_sequence(
        py: Python<'_>,
        degree: Vec<u32>,
        tries: usize,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<(bool, Vec<u32>)>> {
        let total: u64 = degree.iter().map(|&d| d as u64).sum();
        let dmax = degree.iter().copied().max().unwrap_or(0) as u64;
        if total >= 1 << 53 || dmax * dmax >= 1 << 53 || degree.len() >= 1 << 31 {
            return Err(PyValueError::new_err("degrees too large"));
        }
        let Some(mut rng) = pyrandom::Mt19937::from_state(&state) else {
            return Ok(None);
        };
        let mut interrupted = None;
        let found = py.detach(|| {
            let mut stop = || signalled(&mut interrupted);
            degree_generators::random_degree_sequence(&degree, tries, &mut rng, &mut stop)
        });
        if let Some(err) = interrupted {
            return Err(err);
        }
        let Some(found) = found else {
            return Ok(None);
        };
        let built = found.is_some();
        if let Some(b) = found {
            fill_generated(py, &b, g, None)?;
        }
        Ok(Some((built, rng.state())))
    }

    /// `joint_degree_graph` into `g`: `classes` holds each degree's node
    /// count in NetworkX's order, `entries` the `(k index, l index, count)`
    /// to realise.
    #[staticmethod]
    fn dg_joint_degree(
        py: Python<'_>,
        classes: Vec<(u32, u32)>,
        entries: Vec<(u32, u32, u64)>,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        let n: u64 = classes.iter().map(|&(_, c)| c as u64).sum();
        if n >= 1 << 31
            || entries
                .iter()
                .any(|&(k, l, _)| k as usize >= classes.len() || l as usize >= classes.len())
        {
            return Err(PyValueError::new_err("bad joint degree classes"));
        }
        let mut interrupted = None;
        let found = rg_run(py, &state, g, None, |rng| {
            let mut stop = || signalled(&mut interrupted);
            degree_generators::joint_degree(&classes, &entries, rng, &mut stop)
        });
        match interrupted {
            Some(err) => Err(err),
            None => found,
        }
    }

    /// `is_valid_directed_joint_degree` for int sequences of one length and
    /// `nkk` as `(k, l, value)` entries; `None` where NetworkX raises.
    #[staticmethod]
    fn dg_is_valid_directed_joint_degree(
        py: Python<'_>,
        in_degrees: Vec<i64>,
        out_degrees: Vec<i64>,
        nkk: Vec<(i64, i64, i64)>,
    ) -> PyResult<Option<bool>> {
        if in_degrees.len() != out_degrees.len() {
            return Err(PyValueError::new_err("sequences of one length"));
        }
        Ok(py.detach(|| {
            degree_generators::is_valid_directed_joint_degree(&in_degrees, &out_degrees, &nkk)
        }))
    }

    /// `random_labeled_rooted_forest(n)` after NetworkX picked `k` roots
    /// (`1 <= k < n`), into `g`: `(roots, state)`.
    #[staticmethod]
    #[allow(clippy::type_complexity)]
    fn dg_labeled_rooted_forest(
        py: Python<'_>,
        n: usize,
        k: usize,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<(Vec<u32>, Vec<u32>)>> {
        if k == 0 || k >= n || n >= 1 << 31 {
            return Err(PyValueError::new_err("need 1 <= k < n"));
        }
        let Some(mut rng) = pyrandom::Mt19937::from_state(&state) else {
            return Ok(None);
        };
        let Some((b, roots)) =
            py.detach(|| degree_generators::labeled_rooted_forest(n, k, &mut rng))
        else {
            return Ok(None);
        };
        fill_generated(py, &b, g, None)?;
        Ok(Some((roots, rng.state())))
    }

    /// `directed_joint_degree_graph` into `g`, for valid non-negative
    /// degrees and the positive `nkk` entries `(k, l, count)`.
    #[staticmethod]
    fn dg_directed_joint_degree(
        py: Python<'_>,
        in_degrees: Vec<u32>,
        out_degrees: Vec<u32>,
        nkk: Vec<(u32, u32, u64)>,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        if in_degrees.len() != out_degrees.len() || in_degrees.len() >= 1 << 31 {
            return Err(PyValueError::new_err("sequences of one length"));
        }
        rg_run(py, &state, g, None, |rng| {
            degree_generators::directed_joint_degree(&in_degrees, &out_degrees, &nkk, rng)
        })
    }

    /// `PySet::pop` replayed against `ops` (see `replay_set_pops`).
    #[staticmethod]
    fn dg_replay_set_pops(hashes: Vec<i64>, ops: Vec<(u8, u32)>) -> PyResult<(Vec<i64>, Vec<u32>)> {
        if ops.iter().any(|&(_, k)| k as usize >= hashes.len()) {
            return Err(PyValueError::new_err("keys index the hashes"));
        }
        Ok(degree_generators::replay_set_pops(&hashes, &ops))
    }

    // --- Batch 23: growth and geometric generators ---

    /// `duplication_divergence_graph` (`n >= 2`, `0 < p <= 1`) into `g`.
    #[staticmethod]
    fn b23_duplication_divergence(
        py: Python<'_>,
        n: usize,
        p: f64,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        rg_run(py, &state, g, None, |rng| {
            Some(more_random::duplication_divergence(n, p, rng))
        })
    }

    /// `partial_duplication_graph` (`1 <= n <= big_n`) into `g`.
    #[staticmethod]
    fn b23_partial_duplication(
        py: Python<'_>,
        big_n: usize,
        n: usize,
        p: f64,
        q: f64,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        rg_run(py, &state, g, None, |rng| {
            Some(more_random::partial_duplication(big_n, n, p, q, rng))
        })
    }

    /// `scale_free_graph` from the default 3-cycle into the empty
    /// `MultiDiGraph` `g`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn b23_scale_free(
        py: Python<'_>,
        n: usize,
        alpha: f64,
        beta: f64,
        delta_in: f64,
        delta_out: f64,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        rg_run(py, &state, g, None, |rng| {
            Some(more_random::scale_free(
                n, alpha, beta, delta_in, delta_out, rng,
            ))
        })
    }

    /// `random_shell_graph`: per shell `(n, kind, m, intra)` (see
    /// `random_generators_more::Shell`).
    #[staticmethod]
    fn b23_random_shell(
        py: Python<'_>,
        shells: Vec<(usize, u8, u64, u64)>,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        let shells: Vec<more_random::Shell> = shells
            .into_iter()
            .map(|(n, kind, m, intra)| more_random::Shell { n, kind, m, intra })
            .collect();
        rg_run(py, &state, g, None, |rng| {
            Some(more_random::random_shell(&shells, rng))
        })
    }

    /// `navigable_small_world_graph` into the empty `DiGraph` `g`, with the
    /// lattice points as tuples; `weights[d]` is `d ** -r`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn b23_navigable_small_world<'py>(
        py: Python<'py>,
        n: usize,
        dim: usize,
        p: f64,
        q: u64,
        weights: Vec<f64>,
        state: Vec<u32>,
        g: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        let total = (0..dim).try_fold(1usize, |acc, _| acc.checked_mul(n));
        let Some(total) = total.filter(|&t| t < (1 << 31)) else {
            return Ok(None);
        };
        if weights.len() < dim * n.saturating_sub(1) + 1 {
            return Err(PyValueError::new_err("one weight per lattice distance"));
        }
        let Some(mut rng) = pyrandom::Mt19937::from_state(&state) else {
            return Ok(None);
        };
        let b = py.detach(|| more_random::navigable_small_world(n, dim, p, q, &weights, &mut rng));
        let mut labels = Vec::with_capacity(total);
        let mut digits = vec![0usize; dim];
        for _ in 0..total {
            labels.push(PyTuple::new(py, &digits)?.into_any());
            for c in (0..dim).rev() {
                digits[c] += 1;
                if digits[c] < n {
                    break;
                }
                digits[c] = 0;
            }
        }
        fill_built(py, &b, g, Some(labels), &[])?;
        Ok(Some(rng.state()))
    }

    /// `soft_random_geometric_graph` with drawn positions and the default
    /// `p_dist` into `g` (node attribute `pos_key`, each a list).
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn b23_soft_geometric<'py>(
        py: Python<'py>,
        n: usize,
        dim: usize,
        radius: f64,
        p: f64,
        compensated: bool,
        state: Vec<u32>,
        g: &Bound<'py, PyAny>,
        pos_key: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        let Some(mut rng) = pyrandom::Mt19937::from_state(&state) else {
            return Ok(None);
        };
        let found =
            py.detach(|| more_random::soft_geometric(n, dim, radius, p, compensated, &mut rng));
        let Some((coords, b)) = found else {
            return Ok(None);
        };
        let pos = b23_rows(py, &coords, dim)?;
        fill_built(py, &b, g, None, &[(pos_key, &pos)])?;
        Ok(Some(rng.state()))
    }

    /// `thresholded_random_geometric_graph` with drawn weights and
    /// positions into `g` (attributes `weight_key`, then `pos_key`).
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn b23_thresholded_geometric<'py>(
        py: Python<'py>,
        n: usize,
        dim: usize,
        radius: f64,
        theta: f64,
        p: f64,
        state: Vec<u32>,
        g: &Bound<'py, PyAny>,
        pos_key: &Bound<'py, PyAny>,
        weight_key: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        let Some(mut rng) = pyrandom::Mt19937::from_state(&state) else {
            return Ok(None);
        };
        let found =
            py.detach(|| more_random::thresholded_geometric(n, dim, radius, theta, p, &mut rng));
        let Some((weights, coords, b)) = found else {
            return Ok(None);
        };
        let pos = b23_rows(py, &coords, dim)?;
        let w = PyList::new(py, weights)?;
        fill_built(py, &b, g, None, &[(weight_key, &w), (pos_key, &pos)])?;
        Ok(Some(rng.state()))
    }

    /// `geographical_threshold_graph` with drawn weights and positions, the
    /// default metric and `p_dist`, into `g`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn b23_geographical_threshold<'py>(
        py: Python<'py>,
        n: usize,
        dim: usize,
        theta: f64,
        state: Vec<u32>,
        g: &Bound<'py, PyAny>,
        pos_key: &Bound<'py, PyAny>,
        weight_key: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        let Some(mut rng) = pyrandom::Mt19937::from_state(&state) else {
            return Ok(None);
        };
        let found = py.detach(|| more_random::geographical_threshold(n, dim, theta, &mut rng));
        let Some((weights, coords, b)) = found else {
            return Ok(None);
        };
        let pos = b23_rows(py, &coords, dim)?;
        let w = PyList::new(py, weights)?;
        fill_built(py, &b, g, None, &[(weight_key, &w), (pos_key, &pos)])?;
        Ok(Some(rng.state()))
    }

    /// `geometric_soft_configuration_graph` without `kappas`, given the
    /// constants NetworkX computes (`consts`: `kappa_0, base, power,
    /// 2 * pi, R, beta, max(1, beta), mu, 2 / zeta * log(n / pi), R_c`).
    #[staticmethod]
    fn b23_soft_configuration<'py>(
        py: Python<'py>,
        n: usize,
        consts: Vec<f64>,
        state: Vec<u32>,
        g: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        let [kappa_0, base, power, two_pi, big_r, beta, beta_max, mu, r_hat_head, r_c] = consts[..]
        else {
            return Err(PyValueError::new_err("ten constants"));
        };
        let c = more_random::SoftConfig {
            kappa_0,
            base,
            power,
            two_pi,
            big_r,
            beta,
            beta_max,
            mu,
            r_hat_head,
            r_c,
        };
        let Some(mut rng) = pyrandom::Mt19937::from_state(&state) else {
            return Ok(None);
        };
        let found = py.detach(|| more_random::soft_configuration(n, &c, &mut rng));
        let Some((thetas, kappas, radii, b)) = found else {
            return Ok(None);
        };
        let (kt, kk, kr) = (
            pyo3::types::PyString::new(py, "theta"),
            pyo3::types::PyString::new(py, "kappa"),
            pyo3::types::PyString::new(py, "radius"),
        );
        let (t, k, r) = (
            PyList::new(py, thetas)?,
            PyList::new(py, kappas)?,
            PyList::new(py, radii)?,
        );
        fill_built(
            py,
            &b,
            g,
            None,
            &[(kt.as_any(), &t), (kk.as_any(), &k), (kr.as_any(), &r)],
        )?;
        Ok(Some(rng.state()))
    }

    /// The random intersection graphs into `g`: `kind` 0 uniform (`p`:
    /// `[p]`, `m` bottom nodes; node attribute `bipartite` 0), 1 `k` random
    /// (`m`, `k`), 2 general (`p` per bottom node).
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn b23_intersection<'py>(
        py: Python<'py>,
        kind: u8,
        n: usize,
        m: usize,
        k: usize,
        p: Vec<f64>,
        state: Vec<u32>,
        g: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        if (kind == 0 && p.len() != 1) || (kind == 1 && n > 0 && k > m) {
            return Err(PyValueError::new_err("bad intersection graph arguments"));
        }
        if kind == 0 {
            let key = pyo3::types::PyString::new(py, "bipartite");
            let zeros = PyList::new(py, vec![0i64; n])?;
            return rg_run(py, &state, g, Some((key.as_any(), &zeros)), |rng| {
                more_random::uniform_intersection(n, m, p[0], rng)
            });
        }
        rg_run(py, &state, g, None, |rng| {
            Some(match kind {
                1 => more_random::k_intersection(n, m, k, rng),
                _ => more_random::general_intersection(n, &p, rng),
            })
        })
    }

    /// `random_k_lift(G, k)` into the empty graph `h` of `G`'s class.
    #[staticmethod]
    fn b23_k_lift<'py>(
        py: Python<'py>,
        g: &Bound<'py, PyAny>,
        k: usize,
        state: Vec<u32>,
        h: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        let src = nxdicts::NxDicts::of_graph(g)?;
        let directed = src.pred.is_some();
        let multigraph = g
            .call_method0(pyo3::intern!(py, "is_multigraph"))?
            .is_truthy()?;
        let index = PyDict::new(py);
        let mut nodes = Vec::with_capacity(src.node.len());
        for (i, key) in src.node.keys().iter().enumerate() {
            index.set_item(&key, i)?;
            nodes.push(key);
        }
        if nodes.len().checked_mul(k).is_none_or(|t| t >= 1 << 31) {
            return Ok(None);
        }
        let position = |key: &Bound<'py, PyAny>| -> PyResult<usize> {
            index
                .get_item(key)?
                .ok_or_else(|| PyNotImplementedError::new_err("the graph's dicts disagree"))?
                .extract()
        };
        // `G.edges()`: an undirected edge from the row met first, one pair
        // per parallel edge.
        let mut edges: Vec<(u32, u32)> = Vec::new();
        let mut seen = vec![false; nodes.len()];
        for (key, row) in src.adj.iter() {
            let u = position(&key)?;
            for (nbr, data) in row.cast_into::<PyDict>()?.iter() {
                let v = position(&nbr)?;
                if !directed && seen[v] {
                    continue;
                }
                let reps = if multigraph {
                    data.cast_into::<PyDict>()?.len()
                } else {
                    1
                };
                for _ in 0..reps {
                    edges.push((u as u32, v as u32));
                }
            }
            seen[u] = true;
        }
        let Some(mut rng) = pyrandom::Mt19937::from_state(&state) else {
            return Ok(None);
        };
        let n = nodes.len();
        let sim = py.detach(|| more_random::k_lift(n, &edges, k, directed, multigraph, &mut rng));
        let mut labels = Vec::with_capacity(n * k);
        for v in &nodes {
            for i in 0..k {
                labels.push(Some((v, i).into_pyobject(py)?.into_any()));
            }
        }
        let dicts = nxdicts::NxDicts::of_graph(h)?;
        if !dicts.is_empty() || dicts.pred.is_some() != directed {
            return Err(PyValueError::new_err("the graph to fill is not empty"));
        }
        nxdicts::write_sim(&dicts, &sim, &labels, |_, _| Ok(()))?;
        Ok(Some(rng.state()))
    }

    /// Bipartite `preferential_attachment_graph` into the empty undirected
    /// graph `g` (node attribute `bipartite`).
    #[staticmethod]
    fn b23_bipartite_preferential<'py>(
        py: Python<'py>,
        aseq: Vec<i64>,
        p: f64,
        state: Vec<u32>,
        g: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        let multigraph = g
            .call_method0(pyo3::intern!(py, "is_multigraph"))?
            .is_truthy()?;
        let Some(mut rng) = pyrandom::Mt19937::from_state(&state) else {
            return Ok(None);
        };
        let sim = py.detach(|| more_random::bipartite_preferential(&aseq, p, multigraph, &mut rng));
        let na = aseq.len();
        let key = pyo3::types::PyString::new(py, "bipartite");
        let (zero, one) = (PyInt::new(py, 0), PyInt::new(py, 1));
        b23_write(py, &sim, g, |u, data| {
            data.set_item(&key, if (u as usize) < na { &zero } else { &one })
        })?;
        Ok(Some(rng.state()))
    }

    /// `relaxed_caveman_graph` into the empty `Graph` `g`.
    #[staticmethod]
    fn b23_relaxed_caveman(
        py: Python<'_>,
        l: usize,
        k: usize,
        p: f64,
        state: Vec<u32>,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        if l.checked_mul(k).is_none_or(|n| n >= 1 << 31) {
            return Ok(None);
        }
        rg_run(py, &state, g, None, |rng| {
            Some(more_random::relaxed_caveman(l, k, p, rng))
        })
    }

    /// `random_k_out_graph`'s edges, in the order NetworkX adds them, and
    /// the generator's new state: `numpy` replays NetworkX 3.5+'s NumPy
    /// version on a legacy `RandomState` state, otherwise 3.4's pure-Python
    /// version on a `random.Random` state. `None` lets NetworkX run.
    #[staticmethod]
    #[allow(clippy::type_complexity)]
    fn k_out_edges(
        py: Python<'_>,
        n: usize,
        k: usize,
        alpha: f64,
        self_loops: bool,
        numpy: bool,
        state: Vec<u32>,
    ) -> Option<(Vec<u32>, Vec<u32>, Vec<u32>)> {
        let mut rng = pyrandom::Mt19937::from_state(&state)?;
        let edges = py.detach(|| {
            if numpy {
                algorithms::random_generators_more::k_out_numpy(&mut rng, n, k, alpha, self_loops)
            } else {
                algorithms::random_generators_more::k_out_py34(&mut rng, n, k, alpha, self_loops)
            }
        })?;
        let (us, vs) = edges.into_iter().unzip();
        Some((us, vs, rng.state()))
    }

    /// `geometric_edges`: the pairs of `nodes` (positions `positions`, each
    /// a sequence of numbers of one length) within `radius`, sorted by
    /// position, as `(u, v)` tuples; `None` to let NetworkX run (positions
    /// SciPy reads differently, or a pair within rounding error of the
    /// radius).
    #[staticmethod]
    fn b23_geometric_edges<'py>(
        py: Python<'py>,
        nodes: Vec<Bound<'py, PyAny>>,
        positions: Vec<Bound<'py, PyAny>>,
        radius: f64,
        p: f64,
    ) -> PyResult<Option<Bound<'py, PyList>>> {
        if nodes.len() != positions.len() || nodes.is_empty() {
            return Ok(None);
        }
        let mut coords = Vec::new();
        let mut dim = None;
        for pos in &positions {
            if pos.is_instance_of::<pyo3::types::PyString>() {
                return Ok(None);
            }
            let Ok(row) = pos.extract::<Vec<f64>>() else {
                return Ok(None);
            };
            if *dim.get_or_insert(row.len()) != row.len() || row.iter().any(|x| !x.is_finite()) {
                return Ok(None);
            }
            coords.extend(row);
        }
        let dim = dim.unwrap_or(0);
        if dim == 0 {
            return Ok(None);
        }
        let Some(pairs) = py.detach(|| random_generators::geometric_pairs(&coords, dim, radius, p))
        else {
            return Ok(None);
        };
        let out = PyList::empty(py);
        for (u, v) in pairs {
            out.append((&nodes[u as usize], &nodes[v as usize]))?;
        }
        Ok(Some(out))
    }

    /// `random_internet_as_graph` into the empty `Graph` `g`, given the
    /// constants `AS_graph_generator` computes from `n` (`n, n_m, n_cp` and
    /// `d_m, d_cp, d_c, p_m_m, p_cp_m, p_cp_cp`).
    #[staticmethod]
    fn b23_internet_as<'py>(
        py: Python<'py>,
        counts: (i64, i64, i64),
        rates: Vec<f64>,
        state: Vec<u32>,
        g: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        let [d_m, d_cp, d_c, p_m_m, p_cp_m, p_cp_cp] = rates[..] else {
            return Err(PyValueError::new_err("six rates"));
        };
        let (n, n_m, n_cp) = counts;
        if n >= 1 << 24 {
            return Ok(None);
        }
        let c = more_random::AsParams {
            n,
            n_m,
            n_cp,
            d_m,
            d_cp,
            d_c,
            p_m_m,
            p_cp_m,
            p_cp_cp,
        };
        let Some(mut rng) = pyrandom::Mt19937::from_state(&state) else {
            return Ok(None);
        };
        let out = py.detach(|| more_random::internet_as(&c, &mut rng));
        let s = |x: &str| pyo3::types::PyString::new(py, x);
        let (k_type, k_peers, k_customer) = (s("type"), s("peers"), s("customer"));
        let kinds = [s("T"), s("M"), s("CP"), s("C")];
        fill_built_with(py, &out.b, g, None, |u, data| {
            let kind = out.kinds[u as usize];
            data.set_item(&k_type, &kinds[kind as usize])?;
            if kind != 0 {
                data.set_item(&k_peers, out.peers[u as usize])?;
            }
            Ok(())
        })?;
        // `add_edge(i, j, type=kind, customer=customer)`: one dict per edge,
        // shared by both rows.
        let adj = g.getattr("_adj")?.cast_into::<PyDict>()?;
        let (transit, peer, none) = (s("transit"), s("peer"), s("none"));
        for &(u, v, is_transit) in &out.edges {
            let row = adj
                .get_item(u)?
                .ok_or_else(|| PyValueError::new_err("missing row"))?
                .cast_into::<PyDict>()?;
            let data = row
                .get_item(v)?
                .ok_or_else(|| PyValueError::new_err("missing edge"))?
                .cast_into::<PyDict>()?;
            if is_transit {
                data.set_item(&k_type, &transit)?;
                data.set_item(&k_customer, u.to_string())?;
            } else {
                data.set_item(&k_type, &peer)?;
                data.set_item(&k_customer, &none)?;
            }
        }
        Ok(Some(rng.state()))
    }

    /// Replays set algebra on `pyset::PySet` (the runtime check of
    /// `random_internet_as_graph`'s union, intersection and difference);
    /// see `random_generators_more::replay_set_algebra`.
    #[staticmethod]
    fn b23_replay_set_algebra(
        hashes: Vec<i64>,
        nsets: usize,
        ops: Vec<(u8, u32, u32, u32)>,
    ) -> PyResult<Vec<Vec<u32>>> {
        for &(op, dst, a, b) in &ops {
            let bad_key = op <= 1 && a as usize >= hashes.len();
            let bad_set =
                dst as usize >= nsets || (op >= 2 && (a as usize >= nsets || b as usize >= nsets));
            if bad_key || bad_set {
                return Err(PyIndexError::new_err("set or key out of range"));
            }
        }
        Ok(more_random::replay_set_algebra(&hashes, nsets, &ops))
    }

    /// `maybe_regular_expander_graph` into the empty graph `g` (`n` nodes),
    /// from a NumPy `RandomState`'s MT19937 `state` (624 key words and the
    /// position).
    #[staticmethod]
    fn b23_maybe_regular_expander<'py>(
        py: Python<'py>,
        n: usize,
        d: usize,
        max_tries: i64,
        state: Vec<u32>,
        g: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        if n < 3 || d < 2 || n - 1 < d || n >= 1 << 31 {
            return Err(PyValueError::new_err("bad expander arguments"));
        }
        let Some(mut rng) = pyrandom::Mt19937::from_state(&state) else {
            return Ok(None);
        };
        let Some(edges) =
            py.detach(|| more_random::maybe_regular_expander(n, d, max_tries, &mut rng))
        else {
            return Ok(None);
        };
        let directed = g.call_method0("is_directed")?.is_truthy()?;
        let multigraph = g.call_method0("is_multigraph")?.is_truthy()?;
        let mut sim = generators::Sim::new(directed, multigraph, n, edges.len());
        sim.add_nodes(0..n as u32);
        for (u, v) in edges {
            sim.add_edge(u, v);
        }
        b23_write(py, &sim, g, |_, _| Ok(()))?;
        Ok(Some(rng.state()))
    }

    /// `greedy_color` (largest_first): processing order and each node's color.
    fn greedy_color(&self, py: Python<'_>) -> (Vec<u32>, Vec<u32>) {
        let degree = self.degrees();
        py.detach(|| structure::greedy_color(&self.succ, self.n, &degree))
    }

    /// Final labels of `label_propagation_communities`, visiting `order`.
    fn label_propagation(&self, py: Python<'_>, order: Vec<u32>) -> PyResult<Vec<u32>> {
        let order = self.sources_or_all(Some(order))?;
        Ok(py.detach(|| structure::label_propagation(&self.succ, self.n, &order)))
    }

    /// Kruskal's spanning forest: kept edges as `(us, vs)` in yield order.
    #[pyo3(signature = (weight=None, maximum=false))]
    fn kruskal(
        &self,
        py: Python<'_>,
        weight: Option<&str>,
        maximum: bool,
    ) -> PyResult<(Vec<u32>, Vec<u32>)> {
        let w = self.weight_slice(weight, false)?;
        Ok(py.detach(|| {
            let mut edges = Vec::new();
            for u in 0..self.n {
                for e in self.succ.range(u) {
                    let v = self.succ.targets[e];
                    if self.directed || v as usize >= u {
                        edges.push((u as u32, v, w.map_or(1.0, |w| w[e])));
                    }
                }
            }
            structure::kruskal(self.n, &edges, maximum)
                .into_iter()
                .map(|i| (edges[i as usize].0, edges[i as usize].1))
                .unzip()
        }))
    }

    /// `all_shortest_paths` from `source` to `target` as a lazy path
    /// iterator, or `None` if `target` isn't reached. `weight=None` uses
    /// `nx.predecessor` (BFS).
    #[pyo3(signature = (source, target, weight=None, skip_ends_pass=false))]
    fn all_shortest_paths(
        &self,
        py: Python<'_>,
        source: usize,
        target: usize,
        weight: Option<&str>,
        skip_ends_pass: bool,
    ) -> PyResult<Option<AllPaths>> {
        self.check_index(source)?;
        self.check_index(target)?;
        let pred = match weight {
            None => py.detach(|| paths::bfs_predecessors(&self.succ, self.n, source)),
            Some(_) => {
                let w = self.weight_slice(weight, false)?;
                py.detach(|| paths::dijkstra_predecessors(&self.succ, self.n, w, source))?
            }
        };
        Ok(
            paths::PathsFromPreds::new(pred, source as u32, target as u32, skip_ends_pass)
                .map(AllPaths),
        )
    }

    /// `bidirectional_shortest_path` as node positions, or `None` if no path.
    fn bidirectional_bfs(
        &self,
        py: Python<'_>,
        source: usize,
        target: usize,
    ) -> PyResult<Option<Vec<u32>>> {
        self.check_index(source)?;
        self.check_index(target)?;
        let rev = self.path_adj(None, true)?.0;
        Ok(py.detach(|| paths::bidirectional_bfs(&self.succ, rev, self.n, source, target)))
    }

    /// `bfs_tree` for each source, in parallel.
    fn bfs_tree_many(
        &self,
        py: Python<'_>,
        sources: Vec<u32>,
        cutoff: f64,
    ) -> PyResult<Vec<(Vec<u32>, Vec<u32>)>> {
        let sources = self.sources_or_all(Some(sources))?;
        Ok(py.detach(|| {
            use rayon::prelude::*;
            sources
                .par_iter()
                .map(|&s| paths::bfs_tree(&self.succ, self.n, s as usize, cutoff))
                .collect()
        }))
    }

    /// `dijkstra_tree` for each source, in parallel; `None` on a negative cycle.
    #[pyo3(signature = (sources, weight=None, cutoff=None))]
    #[allow(clippy::type_complexity)]
    fn dijkstra_tree_many(
        &self,
        py: Python<'_>,
        sources: Vec<u32>,
        weight: Option<&str>,
        cutoff: Option<f64>,
    ) -> PyResult<Vec<Option<(Vec<u32>, Vec<f64>, Vec<u32>, Vec<u32>)>>> {
        let sources = self.sources_or_all(Some(sources))?;
        let w = self.weight_slice(weight, false)?;
        Ok(py.detach(|| {
            use rayon::prelude::*;
            sources
                .par_iter()
                .map(|&s| {
                    paths::dijkstra_tree(&self.succ, self.n, w, s as usize, cutoff, None)
                        .ok()
                        .map(|t| {
                            let parents = t.order.iter().map(|&v| t.parent[v as usize]).collect();
                            (t.order, t.dist, parents, t.seen_order)
                        })
                })
                .collect()
        }))
    }

    /// rustworkx's strongly connected components (petgraph Kosaraju order).
    fn rx_scc(&self, py: Python<'_>) -> Vec<Vec<u32>> {
        py.detach(|| rx::kosaraju_scc(&self.succ, self.adj(true), self.n))
    }

    /// rustworkx's topological order, or `None` on a cycle.
    fn rx_toposort(&self, py: Python<'_>) -> Option<Vec<u32>> {
        py.detach(|| rx::toposort(&self.succ, self.adj(true), self.n))
    }

    /// rustworkx Dijkstra lengths from `source` (NaN = unreached). Raises
    /// ValueError, as rustworkx does, on a NaN or negative cost it reaches.
    #[pyo3(signature = (source, goal=None))]
    fn rx_dijkstra(
        &self,
        py: Python<'_>,
        source: usize,
        goal: Option<usize>,
    ) -> PyResult<Vec<f64>> {
        self.check_index(source)?;
        let costs = self
            .weight_slice(Some(rx::RX_WEIGHT), false)?
            .expect("costs");
        py.detach(|| rx::rx_dijkstra(&self.succ, self.n, costs, source, goal))
            .map_err(|e| match e {
                rx::CostError::NaN => PyValueError::new_err("NaN weights not supported."),
                rx::CostError::Negative => PyValueError::new_err("Negative weights not supported."),
            })
    }

    /// `early_exit`: see `directed::strongly_connected_components`.
    #[pyo3(signature = (early_exit=true))]
    fn strongly_connected_components(&self, py: Python<'_>, early_exit: bool) -> Vec<Vec<u32>> {
        py.detach(|| directed::strongly_connected_components(&self.succ, self.n, early_exit))
    }

    fn weakly_connected_components(&self, py: Python<'_>) -> Vec<Vec<u32>> {
        py.detach(|| directed::weakly_connected_components(&self.succ, self.adj(true), self.n))
    }

    /// In-degrees left after removing the out-edges of `processed` nodes:
    /// the state of `nx.topological_generations` after those generations.
    fn indegrees_after(&self, processed: Vec<u32>) -> PyResult<Vec<usize>> {
        let pred = self.adj(true);
        let mut indegree: Vec<usize> = (0..self.n).map(|v| pred.neighbors(v).len()).collect();
        for &v in &processed {
            self.check_index(v as usize)?;
            for &child in self.succ.neighbors(v as usize) {
                indegree[child as usize] -= 1;
            }
        }
        Ok(indegree)
    }

    /// `(generations, has_cycle)`
    fn topological_generations(&self, py: Python<'_>) -> (Vec<Vec<u32>>, bool) {
        py.detach(|| directed::topological_generations(&self.succ, self.adj(true), self.n))
    }
}

impl CoreGraph {
    /// `bellman_ford_predecessor_and_distance`'s pred lists (`None` on a
    /// negative cycle). The caller checks for negative self-loops first.
    fn bf_preds(
        &self,
        w: Option<&[f64]>,
        source: usize,
        heuristic: bool,
    ) -> Option<more_paths::PredLists> {
        if self.n == 1 {
            return Some(more_paths::PredLists::new(1, source));
        }
        let bf = more_paths::bellman_ford(&self.succ, self.n, w, source, heuristic);
        bf.cycle.is_none().then_some(bf.pred)
    }

    /// The flat lists `bellman_ford` returns for `mode` (see there).
    fn bf_output(
        &self,
        bf: &more_paths::BellmanFord,
        source: usize,
        mode: u8,
        target: Option<usize>,
    ) -> PyResult<(Vec<u32>, Vec<u32>)> {
        let dsts: Vec<u32> = match mode {
            0 => return Ok(bf.pred.flatten()),
            1 => bf.order.clone(),
            2 => target
                .filter(|&t| bf.pred.reached[t])
                .map(|t| vec![t as u32])
                .unwrap_or_default(),
            _ => return Ok((Vec::new(), Vec::new())),
        };
        let mut on_path = vec![false; self.n];
        let (mut flat, mut ends) = (Vec::new(), Vec::new());
        for dst in dsts {
            // Only the first path is used, which is the same with and
            // without the pre-3.7 "skip ends the pass" behaviour.
            let found = more_paths::paths_from_preds(
                &bf.pred.lists,
                source as u32,
                dst,
                false,
                1,
                &mut on_path,
                &mut flat,
                &mut ends,
            );
            if found == 0 {
                // NetworkX would raise StopIteration here.
                return Err(PyNotImplementedError::new_err("no path from predecessors"));
            }
        }
        Ok((flat, ends))
    }
}

impl CoreGraph {
    fn path_adj(
        &self,
        weight: Option<&str>,
        reverse: bool,
    ) -> PyResult<(&graph::Csr, Option<&[f64]>)> {
        if reverse {
            self.reverse_exact_order(weight)
        } else {
            Ok((&self.succ, self.weight_slice(weight, false)?))
        }
    }

    /// For each arc of `succ`, the position of its edge in `edges_in_order`
    /// (both arcs of an undirected edge share one), and the edge count.
    fn edge_ids(&self) -> (Vec<u32>, usize) {
        let m = self.succ.targets.len();
        if self.directed {
            return ((0..m as u32).collect(), m);
        }
        // An undirected edge {t, v} (t < v) is numbered while scanning row t;
        // its mirror arc sits in row v. For a fixed v those edges are
        // numbered in increasing t, so pair them with row v's arcs to
        // smaller targets, sorted by target.
        let low: Vec<Vec<u32>> = (0..self.n)
            .map(|v| {
                let mut arcs: Vec<u32> = self
                    .succ
                    .range(v)
                    .filter(|&e| (self.succ.targets[e] as usize) < v)
                    .map(|e| e as u32)
                    .collect();
                arcs.sort_unstable_by_key(|&e| self.succ.targets[e as usize]);
                arcs
            })
            .collect();
        let mut cursor = vec![0usize; self.n];
        let mut ids = vec![0u32; m];
        let mut next = 0u32;
        for t in 0..self.n {
            for e in self.succ.range(t) {
                let v = self.succ.targets[e] as usize;
                if v >= t {
                    ids[e] = next;
                    if v > t {
                        ids[low[v][cursor[v]] as usize] = next;
                        cursor[v] += 1;
                    }
                    next += 1;
                }
            }
        }
        (ids, next as usize)
    }

    fn sources_or_all(&self, sources: Option<Vec<u32>>) -> PyResult<Vec<u32>> {
        let sources = sources.unwrap_or_else(|| all_nodes(self.n));
        for &v in &sources {
            self.check_index(v as usize)?;
        }
        Ok(sources)
    }

    /// A flag per node: whether it is in `nodes`.
    fn membership(&self, nodes: &[u32]) -> PyResult<Vec<bool>> {
        let mut flags = vec![false; self.n];
        for &v in nodes {
            self.check_index(v as usize)?;
            flags[v as usize] = true;
        }
        Ok(flags)
    }

    /// Equal-length lists of valid node positions.
    fn check_pairs(&self, us: &[u32], vs: &[u32]) -> PyResult<()> {
        if us.len() != vs.len() {
            return Err(PyValueError::new_err("pair lists differ in length"));
        }
        for &v in us.iter().chain(vs) {
            self.check_index(v as usize)?;
        }
        Ok(())
    }

    /// The auxiliary digraph for local connectivity, and `s`, `t` mapped
    /// into it (`sB`, `tA` when node-split).
    fn conn_aux(&self, node_split: bool, s: u32, t: u32) -> (connectivity::Aux, u32, u32) {
        if node_split {
            let h = connectivity::node_aux(&self.succ, self.n, self.directed);
            (h, 2 * s + 1, 2 * t)
        } else {
            (
                connectivity::edge_aux(&self.succ, self.n, self.directed),
                s,
                t,
            )
        }
    }

    fn conn_flows(
        &self,
        py: Python<'_>,
        node_split: bool,
        pairs: Vec<(u32, u32)>,
        cutoff: Option<f64>,
    ) -> PyResult<Vec<i64>> {
        py.detach(|| {
            let h = if node_split {
                connectivity::node_aux(&self.succ, self.n, self.directed)
            } else {
                connectivity::edge_aux(&self.succ, self.n, self.directed)
            };
            let pairs: Vec<(u32, u32)> = if node_split {
                pairs.into_iter().map(|(s, t)| (2 * s + 1, 2 * t)).collect()
            } else {
                pairs
            };
            connectivity::pair_flows(&h, &pairs, cutoff.unwrap_or(f64::INFINITY))
        })
        .map_err(|_| unbounded())
    }

    /// A node map for the batch 19 matrix functions: identity if `None`.
    fn node_map(&self, map: Option<Vec<i64>>) -> PyResult<Vec<i64>> {
        match map {
            None => Ok((0..self.n as i64).collect()),
            Some(map) if map.len() == self.n => Ok(map),
            Some(_) => Err(PyValueError::new_err("map must have one entry per node")),
        }
    }

    fn check_index(&self, v: usize) -> PyResult<()> {
        if v < self.n {
            Ok(())
        } else {
            Err(PyIndexError::new_err("node index out of range"))
        }
    }
}

/// Predecessor lists from one source, for `_build_paths_from_predecessors`.
#[pyclass(module = "rustnx._core", frozen)]
pub struct PredPaths {
    pred: more_paths::PredLists,
    source: u32,
}

#[pymethods]
impl PredPaths {
    /// Keys of NetworkX's `pred` dict, in order.
    fn order(&self) -> Vec<u32> {
        self.pred.order.clone()
    }

    /// Every path to each of `targets` (unreached ones are skipped):
    /// `(targets reached, paths flattened, path ends, where each target's
    /// paths end)`.
    #[allow(clippy::type_complexity)]
    fn all_paths(
        &self,
        py: Python<'_>,
        targets: Vec<u32>,
        skip_ends_pass: bool,
    ) -> PyResult<(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>)> {
        let n = self.pred.lists.len();
        if targets.iter().any(|&t| t as usize >= n) {
            return Err(PyIndexError::new_err("node index out of range"));
        }
        Ok(py.detach(|| {
            let mut on_path = vec![false; n];
            let (mut reached, mut flat, mut ends, mut groups) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for t in targets {
                if !self.pred.reached[t as usize] {
                    continue;
                }
                reached.push(t);
                more_paths::paths_from_preds(
                    &self.pred.lists,
                    self.source,
                    t,
                    skip_ends_pass,
                    usize::MAX,
                    &mut on_path,
                    &mut flat,
                    &mut ends,
                );
                groups.push(ends.len() as u32);
            }
            (reached, flat, ends, groups)
        }))
    }
}

/// `_group_preprocessing` results for `group_betweenness_centrality`
/// (`K x K` matrices over the group nodes, by position).
#[pyclass(frozen, module = "rustnx._core")]
pub struct GroupPre {
    data: centrality_more::GroupData,
    k: usize,
    rev_reach: Vec<u32>,
}

#[pymethods]
impl GroupPre {
    /// `(reached, positions in D[x] order, len(D[x]), nodes reaching x)`.
    #[allow(clippy::type_complexity)]
    fn reach(&self) -> (Vec<bool>, Vec<u32>, Vec<u32>, Vec<u32>) {
        (
            self.data.reached.clone(),
            self.data.pos.clone(),
            self.data.reach_len.clone(),
            self.rev_reach.clone(),
        )
    }

    /// `prominent_group` search over all nodes: `(max_GBC, max_group)`
    /// (`None` for NetworkX's initial `0, []`), or `None` where NetworkX
    /// raises.
    #[allow(clippy::type_complexity)]
    fn prominent(
        &self,
        py: Python<'_>,
        k: usize,
        greedy: bool,
        rank: Vec<u32>,
    ) -> PyResult<Option<(f64, Option<Vec<u32>>)>> {
        if rank.len() != self.k || self.data.reach_len.len() != self.k {
            return Err(PyValueError::new_err(
                "rank length must equal the node count",
            ));
        }
        Ok(py.detach(|| centrality_more::prominent_group(&self.data, k, greedy, &rank).ok()))
    }

    /// `PB_m[v][v]` for each `v` of one group, or `None` where NetworkX
    /// raises `KeyError`.
    #[pyo3(signature = (group, y_orders=None))]
    fn main(
        &self,
        py: Python<'_>,
        group: Vec<u32>,
        y_orders: Option<Vec<Vec<u32>>>,
    ) -> PyResult<Option<Vec<f64>>> {
        let k = self.k as u32;
        if group
            .iter()
            .chain(y_orders.iter().flatten().flatten())
            .any(|&v| v >= k)
            || y_orders.as_ref().is_some_and(|y| y.len() != group.len())
        {
            return Err(PyIndexError::new_err("group position out of range"));
        }
        Ok(py.detach(|| {
            centrality_more::group_main(&self.data, self.k, &group, y_orders.as_deref()).ok()
        }))
    }
}

/// Lazy iterator over `all_shortest_paths` results (node positions).
#[pyclass(module = "rustnx._core")]
pub struct AllPaths(paths::PathsFromPreds);

#[pymethods]
impl AllPaths {
    fn next_path(&mut self) -> Option<Vec<u32>> {
        self.0.next_path()
    }
}

/// Lazy iterator over `all_topological_sorts` results (node positions).
#[pyclass(module = "rustnx._core")]
pub struct AllTopoSorts(dag::AllTopologicalSorts);

#[pymethods]
impl AllTopoSorts {
    /// `(False, None)` when a cycle stops the search, else `(True, sort)`
    /// (`None` once all sorts are out).
    fn next_sort(&mut self) -> (bool, Option<Vec<u32>>) {
        match self.0.next_sort() {
            Ok(sort) => (true, sort),
            Err(dag::HasCycle) => (false, None),
        }
    }
}

/// `nx.transitive_closure_dag`'s growing closure.
#[pyclass(module = "rustnx._core")]
pub struct ClosureDag(dag::ClosureDag);

#[pymethods]
impl ClosureDag {
    fn layer2(&mut self, v: usize) -> PyResult<Vec<u32>> {
        if v >= self.0.rows.len() {
            return Err(PyIndexError::new_err("node index out of range"));
        }
        Ok(self.0.layer2(v))
    }

    fn append(&mut self, v: usize, heads: Vec<u32>) -> PyResult<()> {
        let n = self.0.rows.len();
        if v >= n || heads.iter().any(|&h| h as usize >= n) {
            return Err(PyIndexError::new_err("node index out of range"));
        }
        self.0.rows[v].extend(heads);
        Ok(())
    }
}

/// Lazy iterator over `root_to_leaf_paths` results (node positions).
#[pyclass(module = "rustnx._core")]
pub struct RootLeafPaths(dag::RootLeafPaths);

#[pymethods]
impl RootLeafPaths {
    fn next_path(&mut self) -> Option<Vec<u32>> {
        self.0.next_path()
    }
}

/// `nx.enumerate_all_cliques`'s queue (batch 10).
#[pyclass(module = "rustnx._core")]
pub struct CliqueQueue(matching::AllCliques);

#[pymethods]
impl CliqueQueue {
    /// Up to `limit` more cliques, as lists of the objects in `nodes`.
    fn next_batch<'py>(
        &mut self,
        py: Python<'py>,
        nodes: &Bound<'py, PyList>,
        limit: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let batch = py.detach(|| self.0.next_batch(limit));
        let objects: Vec<Bound<'py, PyAny>> = nodes.iter().collect();
        if batch.iter().flatten().any(|&v| v as usize >= objects.len()) {
            return Err(PyIndexError::new_err("node index out of range"));
        }
        let out = PyList::empty(py);
        for clique in batch {
            out.append(PyList::new(
                py,
                clique.iter().map(|&v| &objects[v as usize]),
            )?)?;
        }
        Ok(out)
    }
}

/// `find_cliques`' search after its setup (batch 25).
#[pyclass(module = "rustnx._core")]
pub struct CliqueSearchIter(Box<cliques::CliqueSearch>);

#[pymethods]
impl CliqueSearchIter {
    /// Up to `limit` more cliques, each `prefix` followed by the objects in
    /// `nodes` (`find_cliques` yields copies of its list `Q`).
    fn next_batch<'py>(
        &mut self,
        py: Python<'py>,
        nodes: &Bound<'py, PyList>,
        prefix: &Bound<'py, PyList>,
        limit: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let batch = py.detach(|| self.0.next_batch(limit));
        let head: Vec<Bound<'py, PyAny>> = prefix.iter().collect();
        let out = PyList::empty(py);
        for clique in batch {
            let mut items = head.clone();
            for &v in &clique {
                items.push(nodes.get_item(v as usize)?);
            }
            out.append(PyList::new(py, items)?)?;
        }
        Ok(out)
    }
}

impl CoreGraph {
    fn b25_check_hashes(&self, hashes: &[i64]) -> PyResult<()> {
        if hashes.len() != self.n {
            return Err(PyValueError::new_err("one hash per node"));
        }
        Ok(())
    }

    /// Edge weights as Python numbers aligned with `succ` (`None` for an
    /// edge `kernighan_lin_bisection` skips), or `None` where rustnx can't
    /// follow NetworkX: mixed or unusual types, `None` before 3.6, values
    /// that aren't finite or that could overflow an int64 total.
    fn b25_kl_weights(
        &self,
        weight: Option<&str>,
        skip_hidden: bool,
    ) -> Option<Vec<Option<flow::Val>>> {
        let m = self.succ.targets.len();
        let Some(attr) = weight else {
            return Some(vec![Some(flow::Val::I(1)); m]);
        };
        let w = self.weight_slice(Some(attr), false).ok()??;
        let info = self.weights.get(attr);
        let (all_int, plain) = info.map_or((true, true), |i| (i.all_int, i.plain));
        if !plain || self.weights_mixed(Some(attr)) {
            return None;
        }
        w.iter()
            .map(|&x| {
                if x.is_nan() {
                    return skip_hidden.then_some(None);
                }
                if !x.is_finite() || (all_int && x.abs() >= (1u64 << 31) as f64) {
                    return None;
                }
                Some(Some(if all_int {
                    flow::Val::I(x as i64)
                } else {
                    flow::Val::F(x)
                }))
            })
            .collect()
    }

    /// In-edges in NetworkX's `G._pred` order (rows of `G._adj` for
    /// undirected graphs).
    fn b25_exact_back(&self) -> PyResult<&graph::Csr> {
        if !self.directed {
            return Ok(&self.succ);
        }
        if self.pred_is_exact {
            return Ok(self.adj(true));
        }
        self.exact_pred
            .get()
            .map(|e| &e.csr)
            .ok_or_else(|| PyNotImplementedError::new_err("exact predecessors not loaded"))
    }
}

// --- Batch 17: deterministic generators ---

/// Labels made in Rust: the ids, ints, or tuples of ints.
fn generated_labels<'py>(
    py: Python<'py>,
    built: &generators::Built,
) -> PyResult<Vec<Option<Bound<'py, PyAny>>>> {
    let mut out = vec![None; built.sim.capacity()];
    let w = built.width;
    for &v in &built.sim.order {
        let v = v as usize;
        out[v] = Some(if w == 0 {
            let x = if built.labels.is_empty() {
                v as i64
            } else {
                built.labels[v]
            };
            x.into_pyobject(py)?.into_any()
        } else {
            PyTuple::new(py, &built.labels[v * w..(v + 1) * w])?.into_any()
        });
    }
    Ok(out)
}

/// Writes a generated graph into a new NetworkX graph's dicts (see
/// `nxdicts::write_sim`), with the node attributes the generator sets.
fn write_generated<'py>(
    py: Python<'py>,
    built: &generators::Built,
    labels: &[Option<Bound<'py, PyAny>>],
    node: &Bound<'py, PyDict>,
    adj: &Bound<'py, PyDict>,
    pred: Option<&Bound<'py, PyDict>>,
) -> PyResult<()> {
    let dicts = nxdicts::NxDicts::new(node.clone(), adj.clone(), pred.cloned());
    let attr_name = match &built.attr {
        generators::Attr::Ints(name, _) => Some(pyo3::types::PyString::intern(py, name)),
        _ => None,
    };
    let pos_name = pyo3::intern!(py, "pos");
    nxdicts::write_sim(&dicts, &built.sim, labels, |u, data| match &built.attr {
        generators::Attr::None => Ok(()),
        generators::Attr::Ints(_, values) => {
            data.set_item(attr_name.as_ref().unwrap(), values[u as usize])
        }
        generators::Attr::Pos(values) => {
            let (x, y) = values[u as usize];
            data.set_item(pos_name, (x, y))
        }
    })
}

// --- Batch 16: approximation algorithms and graph operations ---

/// The cycle of `simulated_annealing_tsp` / `threshold_accepting_tsp`.
#[pyclass(module = "rustnx._core")]
pub struct TspTour(approximation::Tour);

fn tour_cost<'py>(py: Python<'py>, cost: approximation::Cost) -> PyResult<Bound<'py, PyAny>> {
    Ok(match cost {
        approximation::Cost::Int(x) => x.into_pyobject(py)?.into_any(),
        approximation::Cost::Float(x) => PyFloat::new(py, x).into_any(),
    })
}

#[pymethods]
impl TspTour {
    /// The current cycle's cost, as NetworkX's `sum()` gives it.
    fn cost<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tour_cost(py, self.0.cost())
    }

    /// Apply the move (0: swap two nodes, 1: move one node) for the sampled
    /// indices, and return the new cost.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        kind: u8,
        a: usize,
        b: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !self.0.apply(kind, a, b) {
            return Err(PyIndexError::new_err("move index out of range"));
        }
        tour_cost(py, self.0.cost())
    }

    fn save_best(&mut self) {
        self.0.save_best();
    }

    /// Indices into `init_cycle` of the best cycle saved.
    fn best(&self) -> Vec<u32> {
        self.0.best()
    }
}

/// `one_exchange`'s cut state.
#[pyclass(module = "rustnx._core")]
pub struct MaxCutState(approximation::MaxCut);

#[pymethods]
impl MaxCutState {
    fn cut(&self) -> i128 {
        self.0.cut()
    }

    /// The first node in `order` whose switch gives the largest cut, and
    /// that cut.
    fn best(&self, py: Python<'_>, order: Vec<u32>) -> PyResult<Option<(u32, i128)>> {
        let state = &self.0;
        match py.detach(|| state.best(&order)) {
            Some(found) => Ok(Some(found)),
            None if order.is_empty() => Ok(None),
            None => Err(PyIndexError::new_err("node index out of range")),
        }
    }

    fn switch(&mut self, v: usize) -> PyResult<()> {
        if v >= self.0.len() {
            return Err(PyIndexError::new_err("node index out of range"));
        }
        self.0.switch(v);
        Ok(())
    }
}

/// Add edges `(nodes[u], nodes[v])` to a new NetworkX graph's adjacency
/// dicts as `add_edges_from` would (batch 16): one shared empty attribute
/// dict per edge, and edges already present left as they are. `pred_rows`
/// are a directed graph's `_pred` dicts.
#[pyfunction]
#[pyo3(signature = (succ_rows, pred_rows, nodes, us, vs))]
fn _add_plain_edges(
    succ_rows: Vec<Bound<'_, PyDict>>,
    pred_rows: Option<Vec<Bound<'_, PyDict>>>,
    nodes: Vec<Bound<'_, PyAny>>,
    us: Vec<u32>,
    vs: Vec<u32>,
) -> PyResult<()> {
    let n = nodes.len();
    if succ_rows.len() != n
        || pred_rows.as_ref().is_some_and(|p| p.len() != n)
        || us.len() != vs.len()
        || us.iter().chain(&vs).any(|&v| v as usize >= n)
    {
        return Err(PyValueError::new_err("rows, nodes and edges must agree"));
    }
    for (&u, &v) in us.iter().zip(&vs) {
        let (u, v) = (u as usize, v as usize);
        if succ_rows[u].contains(&nodes[v])? {
            continue;
        }
        let data = PyDict::new(succ_rows[u].py());
        succ_rows[u].set_item(&nodes[v], &data)?;
        match &pred_rows {
            Some(pred) => pred[v].set_item(&nodes[u], &data)?,
            None => succ_rows[v].set_item(&nodes[u], &data)?,
        }
    }
    Ok(())
}

/// A degree-sequence test (batch 10) on a list of Python ints: `kind` is
/// `"hh"`, `"eg"`, `"multi"` or `"pseudo"`. `None` unless every item is an
/// int (or bool) that fits in 64 bits.
#[pyfunction]
fn _degree_sequence_test(kind: &str, seq: &Bound<'_, PyList>) -> PyResult<Option<bool>> {
    let Some(values) = int_values(seq) else {
        return Ok(None);
    };
    Ok(Some(match kind {
        "hh" => matching::is_valid_degree_sequence_havel_hakimi(&values),
        "eg" => matching::is_valid_degree_sequence_erdos_gallai(&values),
        "multi" => matching::is_multigraphical(&values),
        "pseudo" => {
            let sum: i128 = values.iter().map(|&d| d as i128).sum();
            sum % 2 == 0 && values.iter().min().is_none_or(|&d| d >= 0)
        }
        _ => return Err(PyValueError::new_err("unknown degree sequence test")),
    }))
}

/// `nx.is_digraphical` on two lists of Python ints (`None` as above, or
/// where NetworkX would allocate a huge list).
#[pyfunction]
fn _digraphical(ins: &Bound<'_, PyList>, outs: &Bound<'_, PyList>) -> Option<bool> {
    matching::is_digraphical(&int_values(ins)?, &int_values(outs)?)
}

/// Whether every item of `seq` is a Python int (or bool): then
/// `make_list_of_ints` leaves the list as it is.
#[pyfunction]
fn _plain_int_list(seq: &Bound<'_, PyList>) -> bool {
    seq.iter()
        .all(|item| item.cast::<pyo3::types::PyInt>().is_ok())
}

fn int_values(seq: &Bound<'_, PyList>) -> Option<Vec<i64>> {
    seq.iter()
        .map(|item| {
            if item.cast::<pyo3::types::PyInt>().is_ok() {
                item.extract::<i64>().ok()
            } else {
                None
            }
        })
        .collect()
}

/// A Python int (within `i64`) or float as a `Num`; `None` otherwise.
fn py_num(value: &Bound<'_, PyAny>) -> Option<trees_more::Num> {
    if let Ok(f) = value.cast::<PyFloat>() {
        return Some(trees_more::Num::Float(f.value()));
    }
    if value.cast::<PyInt>().is_ok() {
        return value.extract::<i64>().ok().map(trees_more::Num::Int);
    }
    None
}

/// Reads a nested tuple as `from_nested_tuple` walks it: `len()` first,
/// then the children (only for a non-empty level). Node ids in preorder.
fn parse_nested(
    obj: &Bound<'_, PyAny>,
    depth: usize,
    max_depth: usize,
    children: &mut Vec<Vec<u32>>,
) -> Option<u32> {
    if depth > max_depth {
        return None;
    }
    let id = children.len() as u32;
    children.push(Vec::new());
    if obj.len().ok()? == 0 {
        return Some(id);
    }
    for child in obj.try_iter().ok()? {
        let child = parse_nested(&child.ok()?, depth + 1, max_depth, children)?;
        children[id as usize].push(child);
    }
    Some(id)
}

/// Cached ancestor sets for `all_pairs_lowest_common_ancestor`.
#[pyclass(module = "rustnx._core")]
pub struct DagLca(trees_more::DagLca);

#[pymethods]
impl DagLca {
    /// Each pair's lowest common ancestor, `-1` if there is none, or `-2`
    /// if there are several lowest ones.
    fn query(&mut self, py: Python<'_>, us: Vec<u32>, vs: Vec<u32>) -> PyResult<Vec<i64>> {
        let n = self.0.node_count();
        if us.len() != vs.len() || us.iter().chain(&vs).any(|&x| x as usize >= n) {
            return Err(PyIndexError::new_err("node index out of range"));
        }
        let inner = &mut self.0;
        Ok(py.detach(|| inner.query(&us, &vs)))
    }
}

/// Lazy `tree_all_pairs_lowest_common_ancestor` results.
#[pyclass(module = "rustnx._core")]
pub struct TreeLca(trees_more::TreeLca);

#[pymethods]
impl TreeLca {
    /// The next results as `(vs, nodes, ancestors)`; empty when done.
    fn next_batch(&mut self, py: Python<'_>, limit: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let inner = &mut self.0;
        let items = py.detach(|| inner.next_batch(limit));
        let mut out = (Vec::new(), Vec::new(), Vec::new());
        for (v, node, a) in items {
            out.0.push(v);
            out.1.push(node);
            out.2.push(a);
        }
        out
    }
}

/// `boruvka_mst_edges`' state between rounds.
#[pyclass(module = "rustnx._core")]
pub struct Boruvka(trees_more::Boruvka);

#[pymethods]
impl Boruvka {
    fn components(&mut self) -> Vec<Vec<u32>> {
        self.0.components()
    }

    /// One round: `(us, vs, any_best)`, or `None` if `orders` is needed.
    #[pyo3(signature = (orders=None))]
    #[allow(clippy::type_complexity)]
    fn round(
        &mut self,
        py: Python<'_>,
        orders: Option<Vec<Vec<u32>>>,
    ) -> Option<(Vec<u32>, Vec<u32>, bool)> {
        let inner = &mut self.0;
        py.detach(|| inner.round(orders.as_deref()))
            .map(|(edges, any)| {
                let (us, vs) = edges.into_iter().unzip();
                (us, vs, any)
            })
    }
}

// --- Batch 7: lazy iterators and state ---

/// `goldberg_radzik`'s state between the rounds Python drives.
#[pyclass(module = "rustnx._core")]
pub struct GoldbergRadzikState(leftovers::GoldbergRadzik);

#[pymethods]
impl GoldbergRadzikState {
    /// One `topo_sort` over `order`; `False` where NetworkX finds a
    /// negative cycle.
    fn topo_sort(&mut self, order: Vec<u32>, skip_counted: bool) -> PyResult<bool> {
        let n = self.0.d.len();
        if order.iter().any(|&v| v as usize >= n) {
            return Err(PyIndexError::new_err("node index out of range"));
        }
        match self.0.topo_sort(&order, skip_counted) {
            Ok(()) => Ok(true),
            Err(leftovers::GrError::Negative) => Ok(false),
            Err(leftovers::GrError::Unsupported) => Err(gr_unsupported()),
        }
    }

    /// One `relax`: the nodes added to `relabeled`, in order.
    fn relax(&mut self) -> PyResult<Vec<u32>> {
        self.0.relax().map_err(|_| gr_unsupported())
    }

    /// `(pred keys, pred values, distances, distance is int)` in key order
    /// (`u32::MAX` for the source's `None`).
    #[allow(clippy::type_complexity)]
    fn result(&self) -> (Vec<u32>, Vec<u32>, Vec<f64>, Vec<bool>) {
        let keys = self.0.pred_order.clone();
        let preds = keys.iter().map(|&v| self.0.pred[v as usize]).collect();
        let d = keys.iter().map(|&v| self.0.d[v as usize]).collect();
        let ints = keys.iter().map(|&v| self.0.d_int[v as usize]).collect();
        (keys, preds, d, ints)
    }
}

fn gr_unsupported() -> PyErr {
    PyNotImplementedError::new_err("integer distances too large for exact sums")
}

/// Flattens up to `limit` items of a lazy search: `(flat, ends)`.
fn batch_of(limit: usize, mut next: impl FnMut() -> Option<Vec<u32>>) -> (Vec<u32>, Vec<u32>) {
    let (mut flat, mut ends) = (Vec::new(), Vec::new());
    while ends.len() < limit {
        let Some(item) = next() else { break };
        flat.extend_from_slice(&item);
        ends.push(flat.len() as u32);
        if flat.len() > (1 << 20) {
            break;
        }
    }
    (flat, ends)
}

/// Lazy iterator over `nx.antichains` results (node positions).
#[pyclass(module = "rustnx._core")]
pub struct AntichainIter(leftovers::Antichains);

#[pymethods]
impl AntichainIter {
    /// Up to `limit` antichains, flattened: `(flat, ends)`.
    fn next_batch(&mut self, py: Python<'_>, limit: usize) -> (Vec<u32>, Vec<u32>) {
        py.detach(|| batch_of(limit, || self.0.next_antichain()))
    }
}

/// Lazy iterator over `_all_simple_edge_paths` results (node positions).
#[pyclass(module = "rustnx._core")]
pub struct SimplePathIter(leftovers::SimplePaths);

#[pymethods]
impl SimplePathIter {
    /// Up to `limit` paths, flattened: `(flat, ends)`.
    fn next_batch(&mut self, py: Python<'_>, limit: usize) -> (Vec<u32>, Vec<u32>) {
        py.detach(|| batch_of(limit, || self.0.next_path()))
    }
}

/// Lazy iterator over `shortest_simple_paths` results (node positions).
#[pyclass(module = "rustnx._core")]
pub struct YenIter(leftovers::SimpleShortestPaths);

#[pymethods]
impl YenIter {
    /// `(0, path)` (`None` once all are out), `(1, None)` where NetworkX
    /// raises `NetworkXNoPath`, `(2, None)` for its contradictory paths
    /// `ValueError`.
    fn next_path(&mut self, py: Python<'_>) -> (u8, Option<Vec<u32>>) {
        py.detach(|| match self.0.next_path() {
            Ok(p) => (0, p),
            Err(leftovers::YenError::NoPath) => (1, None),
            Err(leftovers::YenError::Contradictory) => (2, None),
        })
    }
}

fn unzip3(items: Vec<(u32, u32, u8)>) -> (Vec<u32>, Vec<u32>, Vec<u8>) {
    let mut us = Vec::with_capacity(items.len());
    let mut vs = Vec::with_capacity(items.len());
    let mut labels = Vec::with_capacity(items.len());
    for (u, v, l) in items {
        us.push(u);
        vs.push(v);
        labels.push(l);
    }
    (us, vs, labels)
}

// --- Batch 15: communities, efficiency and structural holes ---

impl CoreGraph {
    /// Checks node lists flattened with `ends` (each list's end).
    fn check_flat(&self, flat: &[u32], ends: &[usize]) -> PyResult<()> {
        if ends.windows(2).any(|w| w[0] > w[1]) || ends.last().is_some_and(|&e| e != flat.len()) {
            return Err(PyValueError::new_err("bad list ends"));
        }
        for &v in flat {
            self.check_index(v as usize)?;
        }
        Ok(())
    }
}

/// A batch 15 sum as the Python int or float NetworkX gets.
fn num_object(py: Python<'_>, x: communities::Num) -> PyResult<Py<PyAny>> {
    Ok(match x {
        communities::Num::Int(i) => i.into_pyobject(py)?.into_any().unbind(),
        communities::Num::Float(f) => PyFloat::new(py, f).into_any().unbind(),
    })
}

/// `girvan_newman`'s working graph, one level of communities per call.
#[pyclass(module = "rustnx._core")]
pub struct GirvanNewman {
    g: communities::EditableGraph,
    scale: Option<f64>,
}

#[pymethods]
impl GirvanNewman {
    /// The next tuple of components, or `None` once the graph has no edges.
    fn next_level(&mut self, py: Python<'_>) -> Option<Vec<Vec<u32>>> {
        if self.g.number_of_edges() == 0 {
            return None;
        }
        let scale = self.scale;
        let g = &mut self.g;
        Some(py.detach(|| g.girvan_newman_step(scale)))
    }
}

/// Link prediction scores for batches of pairs (`CoreGraph.link_scorer`).
#[pyclass(module = "rustnx._core", frozen)]
pub struct LinkScorer {
    graph: Py<CoreGraph>,
    mode: measures::LinkMode,
    hashes: Vec<i64>,
    degree: Vec<usize>,
    table: Vec<f64>,
    classes: Option<Vec<i64>>,
    compensated: bool,
}

#[pymethods]
impl LinkScorer {
    /// `(codes, a, b, f, w)` per pair; see `measures::code`.
    #[allow(clippy::type_complexity)]
    fn scores(
        &self,
        py: Python<'_>,
        us: Vec<u32>,
        vs: Vec<u32>,
    ) -> PyResult<(Vec<u8>, Vec<i64>, Vec<i64>, Vec<f64>, Vec<u32>)> {
        let g = self.graph.get();
        g.check_pairs(&us, &vs)?;
        let inp = measures::LinkInput {
            adj: &g.succ,
            n: g.n,
            hashes: &self.hashes,
            degree: &self.degree,
            table: &self.table,
            classes: self.classes.as_deref(),
            compensated: self.compensated,
        };
        let out = py.detach(|| measures::link_scores(&inp, self.mode, &us, &vs));
        Ok((out.codes, out.a, out.b, out.f, out.w))
    }
}

// --- Batch 12: flows and cut measures ---

impl CoreGraph {
    /// Edge attributes read from NetworkX's adjacency rows (`list(G._adj
    /// .values())`, aligned with `succ`), one vector per `(name, default)`.
    fn edge_values(
        &self,
        rows: &Bound<'_, PyList>,
        attrs: &[(&Bound<'_, PyAny>, flow::Val)],
    ) -> PyResult<Vec<Vec<flow::Val>>> {
        if rows.len() != self.n {
            return Err(PyNotImplementedError::new_err(
                "adjacency does not match this graph",
            ));
        }
        let m = self.succ.targets.len();
        let mut out: Vec<Vec<flow::Val>> = attrs.iter().map(|_| Vec::with_capacity(m)).collect();
        for (u, row) in rows.iter().enumerate() {
            let row = row
                .cast::<PyDict>()
                .map_err(|_| PyNotImplementedError::new_err("adjacency rows are not dicts"))?;
            if row.len() != self.succ.range(u).len() {
                return Err(PyNotImplementedError::new_err(
                    "adjacency does not match this graph",
                ));
            }
            for (_, data) in row.iter() {
                let data = data
                    .cast::<PyDict>()
                    .map_err(|_| PyNotImplementedError::new_err("edge data is not a dict"))?;
                for (k, (name, default)) in attrs.iter().enumerate() {
                    out[k].push(match data.get_item(name)? {
                        Some(x) => py_val(&x)?,
                        None => *default,
                    });
                }
            }
        }
        Ok(out)
    }

    /// Weights for the cut measures, as Python numbers: the converted ones,
    /// or, for an attribute mixing ints and floats, read from `rows` (see
    /// `edge_values`). `None` values fall back (NetworkX raises).
    fn cut_weights(
        &self,
        weight: Option<&str>,
        rows: Option<&Bound<'_, PyList>>,
    ) -> PyResult<flow::Weights<'_>> {
        let Some(attr) = weight else {
            return Ok(flow::Weights::Unit);
        };
        let (all_int, hidden) = self.weights_info(Some(attr));
        if hidden {
            return Err(PyNotImplementedError::new_err("None weights"));
        }
        if self.weights_mixed(Some(attr)) {
            let rows =
                rows.ok_or_else(|| PyNotImplementedError::new_err("weights mix ints and floats"))?;
            let name = pyo3::types::PyString::new(rows.py(), attr).into_any();
            let mut vals = self.edge_values(rows, &[(&name, flow::Val::I(1))])?;
            return Ok(flow::Weights::Exact(vals.pop().unwrap_or_default()));
        }
        Ok(match self.weight_slice(Some(attr), false)? {
            Some(w) => flow::Weights::Stored(w, all_int),
            None => flow::Weights::Unit,
        })
    }
}

/// A Python int (within `i64`, not a bool) or float (not NaN) as a `Val`.
fn py_val(x: &Bound<'_, PyAny>) -> PyResult<flow::Val> {
    if x.is_exact_instance_of::<PyInt>() {
        return x
            .extract::<i64>()
            .map(flow::Val::I)
            .map_err(|_| PyNotImplementedError::new_err("integer too large"));
    }
    if x.is_exact_instance_of::<PyFloat>() {
        let f = x.cast::<PyFloat>()?.value();
        if !f.is_nan() {
            return Ok(flow::Val::F(f));
        }
    }
    Err(PyNotImplementedError::new_err(
        "rustnx needs int or float values here",
    ))
}

fn val_obj(py: Python<'_>, v: flow::Val) -> PyResult<Bound<'_, PyAny>> {
    Ok(match v {
        flow::Val::I(i) => i.into_pyobject(py)?.into_any(),
        flow::Val::F(f) => PyFloat::new(py, f).into_any(),
    })
}

fn fail_err(f: flow::Fail) -> PyErr {
    let raise = |name: &str, msg: &str| {
        Python::attach(|py| -> PyErr {
            match py
                .import("networkx")
                .and_then(|nx| nx.getattr(name))
                .and_then(|cls| cls.call1((msg,)))
            {
                Ok(exc) => PyErr::from_value(exc),
                Err(e) => e,
            }
        })
    };
    match f {
        flow::Fail::Unsupported => PyNotImplementedError::new_err(
            "NetworkX raises an error rustnx doesn't reproduce, or a value is too large",
        ),
        flow::Fail::Unbounded(msg) => raise("NetworkXUnbounded", msg),
    }
}

/// A residual network and the flow last computed on it (batch 12).
#[pyclass(module = "rustnx._core")]
pub struct FlowRun {
    res: flow::Residual,
    /// G's own adjacency, for `build_flow_dict`.
    g_offsets: Vec<usize>,
    g_targets: Vec<u32>,
    outcome: Option<flow::Outcome>,
}

fn flow_algo(name: &str, two_phase: bool, d: i64, threshold: f64) -> PyResult<flow::Algo> {
    Ok(match name {
        "edmonds_karp" => flow::Algo::EdmondsKarp,
        "shortest_augmenting_path" => flow::Algo::ShortestAugmentingPath(two_phase, d),
        "dinitz" => flow::Algo::Dinitz,
        "boykov_kolmogorov" => flow::Algo::BoykovKolmogorov,
        "preflow_push" => flow::Algo::PreflowPush(threshold),
        _ => return Err(PyValueError::new_err("unknown flow algorithm")),
    })
}

#[pymethods]
impl FlowRun {
    /// `R.size()`.
    fn edge_count(&self) -> usize {
        self.res.tail.len()
    }

    /// `R.graph["inf"]`.
    fn inf<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        val_obj(py, self.res.inf)
    }

    /// Run a maximum flow algorithm from scratch; returns the flow value.
    /// `cutoff` is the caller's value (None for NetworkX's default),
    /// `d` and `threshold` as `flow::Algo` describes.
    #[pyo3(signature = (algorithm, s, t, cutoff=None, value_only=false, two_phase=false, d=0, threshold=0.0, hashes=None))]
    #[allow(clippy::too_many_arguments)]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        algorithm: &str,
        s: u32,
        t: u32,
        cutoff: Option<&Bound<'py, PyAny>>,
        value_only: bool,
        two_phase: bool,
        d: i64,
        threshold: f64,
        hashes: Option<Vec<i64>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let algo = flow_algo(algorithm, two_phase, d, threshold)?;
        let cutoff = match cutoff {
            Some(c) if !c.is_none() => Some(py_val(c)?),
            _ => None,
        };
        let n = self.res.n;
        if s as usize >= n || t as usize >= n {
            return Err(PyIndexError::new_err("node index out of range"));
        }
        let hashes = hashes.unwrap_or_default();
        let res = &mut self.res;
        let outcome = py
            .detach(|| flow::run_flow(res, algo, s, t, cutoff, value_only, &hashes))
            .map_err(fail_err)?;
        let value = val_obj(py, outcome.value)?;
        self.outcome = Some(outcome);
        Ok(value)
    }

    /// `minimum_cut`'s reverse search from `t` (see `Residual::cut_order`).
    fn cut_order(&self, t: u32, strict: bool) -> PyResult<Vec<u32>> {
        if t as usize >= self.res.n {
            return Err(PyIndexError::new_err("node index out of range"));
        }
        Ok(self.res.cut_order(t, strict))
    }

    /// Fill a fresh `R`: `succ_rows`/`pred_rows` are `R._succ` and
    /// `R._pred` rows in node order; edge dicts hold the capacity (and the
    /// flow, `with_flow`). Each edge's dict is shared by both rows.
    fn fill<'py>(
        &self,
        py: Python<'py>,
        succ_rows: &Bound<'py, PyList>,
        pred_rows: &Bound<'py, PyList>,
        nodes: &Bound<'py, PyList>,
        with_flow: bool,
    ) -> PyResult<()> {
        let r = &self.res;
        if succ_rows.len() != r.n || pred_rows.len() != r.n || nodes.len() != r.n {
            return Err(PyValueError::new_err(
                "rows do not match the residual network",
            ));
        }
        let objects: Vec<Bound<'py, PyAny>> = nodes.iter().collect();
        let capacity = pyo3::intern!(py, "capacity");
        let flow_key = pyo3::intern!(py, "flow");
        let mut dicts: Vec<Option<Bound<'py, PyDict>>> = vec![None; r.tail.len()];
        for (u, row) in succ_rows.iter().enumerate() {
            let row = row.cast::<PyDict>()?;
            for &e in &r.succ[u] {
                let e = e as usize;
                let d = PyDict::new(py);
                d.set_item(capacity, val_obj(py, r.cap[e])?)?;
                if with_flow {
                    d.set_item(flow_key, val_obj(py, r.flow[e])?)?;
                }
                row.set_item(&objects[r.head[e] as usize], &d)?;
                dicts[e] = Some(d);
            }
        }
        for (v, row) in pred_rows.iter().enumerate() {
            let row = row.cast::<PyDict>()?;
            for &e in &r.pred[v] {
                let e = e as usize;
                let d = dicts[e]
                    .as_ref()
                    .ok_or_else(|| PyValueError::new_err("edge missing"))?;
                row.set_item(&objects[r.tail[e] as usize], d)?;
            }
        }
        Ok(())
    }

    /// Set the `excess` and `height` node attributes the last run left
    /// (`node_rows` is `list(R._node.values())`); returns the positions of
    /// the `curr_edge`s, if the run set them.
    fn set_node_attrs<'py>(
        &self,
        py: Python<'py>,
        node_rows: &Bound<'py, PyList>,
    ) -> PyResult<Option<Vec<u32>>> {
        let Some(out) = &self.outcome else {
            return Ok(None);
        };
        if node_rows.len() != self.res.n {
            return Err(PyValueError::new_err(
                "rows do not match the residual network",
            ));
        }
        if let Some(excess) = &out.excess {
            let key = pyo3::intern!(py, "excess");
            for (row, &x) in node_rows.iter().zip(excess) {
                row.set_item(key, val_obj(py, x)?)?;
            }
        }
        let Some(state) = &out.state else {
            return Ok(None);
        };
        let key = pyo3::intern!(py, "height");
        for (row, &h) in node_rows.iter().zip(&state.height) {
            row.set_item(key, h)?;
        }
        Ok(Some(state.curr.clone()))
    }

    /// `R.graph["trees"]` after `boykov_kolmogorov`.
    fn trees<'py>(
        &self,
        py: Python<'py>,
        nodes: &Bound<'py, PyList>,
    ) -> PyResult<Option<Bound<'py, PyTuple>>> {
        let Some((source, target)) = self.outcome.as_ref().and_then(|o| o.trees.as_ref()) else {
            return Ok(None);
        };
        let objects: Vec<Bound<'py, PyAny>> = nodes.iter().collect();
        let to_dict = |tree: &Vec<(u32, Option<u32>)>| -> PyResult<Bound<'py, PyDict>> {
            let d = PyDict::new(py);
            for &(k, p) in tree {
                match p {
                    Some(p) => d.set_item(&objects[k as usize], &objects[p as usize])?,
                    None => d.set_item(&objects[k as usize], py.None())?,
                }
            }
            Ok(d)
        };
        Ok(Some(PyTuple::new(
            py,
            [to_dict(source)?, to_dict(target)?],
        )?))
    }

    /// `build_flow_dict(G, R)` for the last run.
    fn flow_dict<'py>(
        &self,
        py: Python<'py>,
        nodes: &Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let r = &self.res;
        let objects: Vec<Bound<'py, PyAny>> = nodes.iter().collect();
        if objects.len() != r.n {
            return Err(PyValueError::new_err(
                "nodes do not match the residual network",
            ));
        }
        let zero = 0i64.into_pyobject(py)?.into_any();
        let out = PyDict::new(py);
        for u in 0..r.n {
            let row = PyDict::new(py);
            for &v in &self.g_targets[self.g_offsets[u]..self.g_offsets[u + 1]] {
                row.set_item(&objects[v as usize], &zero)?;
            }
            for &e in &r.succ[u] {
                let f = r.flow[e as usize];
                if f.gt(flow::Val::I(0)) {
                    row.set_item(&objects[r.head[e as usize] as usize], val_obj(py, f)?)?;
                }
            }
            out.set_item(&objects[u], row)?;
        }
        Ok(out)
    }

    /// `gomory_hu_tree` from this (fresh) residual network: each non-root
    /// node's parent position and edge weight, in node order.
    #[pyo3(signature = (algorithm, strict_cut, two_phase=false, d=0, threshold=0.0, hashes=None))]
    #[allow(clippy::too_many_arguments)]
    fn gomory_hu<'py>(
        &mut self,
        py: Python<'py>,
        algorithm: &str,
        strict_cut: bool,
        two_phase: bool,
        d: i64,
        threshold: f64,
        hashes: Option<Vec<i64>>,
    ) -> PyResult<Vec<(u32, Bound<'py, PyAny>)>> {
        let algo = flow_algo(algorithm, two_phase, d, threshold)?;
        let hashes = hashes.unwrap_or_default();
        let res = &mut self.res;
        let tree = py
            .detach(|| flow::gomory_hu(res, algo, &hashes, strict_cut))
            .map_err(fail_err)?;
        tree.into_iter()
            .map(|(p, w)| Ok((p, val_obj(py, w)?)))
            .collect()
    }
}

/// Replays set operations on `pyset::PySet` (the runtime check that it
/// matches the interpreter's sets); see `pyset::replay_sets`.
#[pyfunction]
fn _replay_sets(
    hashes: Vec<i64>,
    nsets: usize,
    ops: Vec<(u8, u32, u32)>,
) -> PyResult<(Vec<i64>, Vec<Vec<u32>>)> {
    for &(op, s, k) in &ops {
        let key_ok = match op {
            0 | 1 => (k as usize) < hashes.len(),
            3 | 5..=10 => (k as usize) < nsets,
            _ => true,
        };
        if s as usize >= nsets || !key_ok {
            return Err(PyIndexError::new_err("set or key out of range"));
        }
    }
    Ok(pyset::replay_sets(&hashes, nsets, &ops))
}

/// Tells the core whether `sum()` compensates ints after the first float
/// (Python 3.14+), as detected by `rustnx.algorithms` at import.
#[pyfunction]
fn _set_sum_ints_compensated(on: bool) {
    flow::SUM_INTS_COMPENSATED.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// CPython's float `sum()` (exposed for tests).
#[pyfunction]
fn _py_sum(values: Vec<f64>, compensated: bool) -> f64 {
    spectral::py_sum(values.into_iter(), compensated)
}

// --- Batch 22: degree-sequence generators (helpers) ---

/// For long loops running without the GIL: takes it to check for signals
/// (Ctrl-C), keeping the error; true if the loop should stop.
fn signalled(error: &mut Option<PyErr>) -> bool {
    if let Err(err) = Python::attach(|py| py.check_signals()) {
        *error = Some(err);
        return true;
    }
    false
}

// --- Batch 18: random generators (helpers) ---

/// Runs a seeded generator from `random.Random` state `state` without the
/// GIL, then fills the empty NetworkX graph `g`; the new state, or `None`
/// (bad state, or the generator can't follow NetworkX) with `g` untouched.
fn rg_run<'py>(
    py: Python<'py>,
    state: &[u32],
    g: &Bound<'py, PyAny>,
    node_attr: Option<(&Bound<'py, PyAny>, &Bound<'py, PyList>)>,
    generate: impl FnOnce(&mut pyrandom::Mt19937) -> Option<random_generators::Built> + Send,
) -> PyResult<Option<Vec<u32>>> {
    let Some(mut rng) = pyrandom::Mt19937::from_state(state) else {
        return Ok(None);
    };
    let Some(b) = py.detach(|| generate(&mut rng)) else {
        return Ok(None);
    };
    fill_generated(py, &b, g, node_attr)?;
    Ok(Some(rng.state()))
}

/// Fills the empty NetworkX graph `g` (any of the four classes) with the
/// nodes `0..n` and adjacency rows of `b`, as NetworkX's `add_node`/`add_edge`
/// calls leave them (see `nxdicts::write_sim`). Node `u`'s data is
/// `{key: values[u]}` with `node_attr`, else `{}`.
fn fill_generated<'py>(
    py: Python<'py>,
    b: &random_generators::Built,
    g: &Bound<'py, PyAny>,
    node_attr: Option<(&Bound<'py, PyAny>, &Bound<'py, PyList>)>,
) -> PyResult<()> {
    let attrs: Vec<_> = node_attr.into_iter().collect();
    fill_built(py, b, g, None, &attrs)
}

/// [`fill_generated`] with node keys `labels` (default: the ints `0..n`)
/// and node attributes `{key: values[u], ...}` set in the order given.
/// Nodes missing from `b.order` are left out.
fn fill_built<'py>(
    py: Python<'py>,
    b: &random_generators::Built,
    g: &Bound<'py, PyAny>,
    labels: Option<Vec<Bound<'py, PyAny>>>,
    attrs: &[(&Bound<'py, PyAny>, &Bound<'py, PyList>)],
) -> PyResult<()> {
    if attrs.iter().any(|(_, values)| values.len() != b.n()) {
        return Err(PyValueError::new_err("one attribute value per node"));
    }
    fill_built_with(py, b, g, labels, |u, data| {
        for (key, values) in attrs {
            data.set_item(key, values.get_item(u as usize)?)?;
        }
        Ok(())
    })
}

/// [`fill_built`] with each node's attributes set by `fill(u, data)`.
fn fill_built_with<'py>(
    py: Python<'py>,
    b: &random_generators::Built,
    g: &Bound<'py, PyAny>,
    labels: Option<Vec<Bound<'py, PyAny>>>,
    fill: impl FnMut(u32, &Bound<'py, PyDict>) -> PyResult<()>,
) -> PyResult<()> {
    let n = b.n();
    let dicts = nxdicts::NxDicts::of_graph(g)?;
    if !dicts.is_empty() {
        return Err(PyValueError::new_err("the graph to fill is not empty"));
    }
    if dicts.pred.is_some() != b.pred.is_some() {
        return Err(PyValueError::new_err("the graph's direction differs"));
    }
    if labels.as_ref().is_some_and(|l| l.len() != n) {
        return Err(PyValueError::new_err("one label per node"));
    }
    let multi = g
        .call_method0(pyo3::intern!(py, "is_multigraph"))?
        .is_truthy()?;
    let all: Vec<u32>;
    let order: &[u32] = match &b.order {
        Some(order) => order,
        None => {
            all = (0..n as u32).collect();
            &all
        }
    };
    let sim = py.detach(|| generators::Sim::from_rows(&b.succ, b.pred.as_deref(), order, multi));
    let labels: Vec<Option<Bound<'py, PyAny>>> = match labels {
        Some(l) => l.into_iter().map(Some).collect(),
        None => (0..n as u64)
            .map(|i| Some(PyInt::new(py, i).into_any()))
            .collect(),
    };
    nxdicts::write_sim(&dicts, &sim, &labels, fill)
}

// --- Batch 23: growth and geometric generators (helpers) ---

/// `coords` as one list of `dim` floats per node (`[seed.random() for i in
/// range(dim)]`).
fn b23_rows<'py>(py: Python<'py>, coords: &[f64], dim: usize) -> PyResult<Bound<'py, PyList>> {
    let pos = PyList::empty(py);
    if dim == 0 {
        return Ok(pos);
    }
    for row in coords.chunks(dim) {
        pos.append(PyList::new(py, row)?)?;
    }
    Ok(pos)
}

/// Writes `sim` (int node keys) into the empty graph `g`, setting each
/// node's attributes with `fill`.
fn b23_write<'py>(
    py: Python<'py>,
    sim: &generators::Sim,
    g: &Bound<'py, PyAny>,
    fill: impl FnMut(u32, &Bound<'py, PyDict>) -> PyResult<()>,
) -> PyResult<()> {
    let dicts = nxdicts::NxDicts::of_graph(g)?;
    if !dicts.is_empty() || dicts.pred.is_some() != sim.directed {
        return Err(PyValueError::new_err("the graph to fill is not empty"));
    }
    let labels: Vec<Option<Bound<'py, PyAny>>> = (0..sim.capacity() as u64)
        .map(|i| Some(PyInt::new(py, i).into_any()))
        .collect();
    nxdicts::write_sim(&dicts, sim, &labels, fill)
}

// --- Batch 24: generators and transforms ---

/// NetworkX's `nonisomorphic_trees` loop, one tree at a time.
#[pyclass(module = "rustnx._core")]
struct B24NonisoTrees(std::sync::Mutex<transforms::NonisoTrees>);

#[pymethods]
impl B24NonisoTrees {
    #[new]
    fn new(order: usize) -> Self {
        B24NonisoTrees(std::sync::Mutex::new(transforms::NonisoTrees::new(order)))
    }

    /// Writes the next tree into the empty `nx.Graph` given by its dicts;
    /// false when there are no more.
    fn fill_next<'py>(
        &self,
        py: Python<'py>,
        node: &Bound<'py, PyDict>,
        adj: &Bound<'py, PyDict>,
    ) -> PyResult<bool> {
        let layout = self
            .0
            .lock()
            .map_err(|_| PyValueError::new_err("iterator state poisoned"))?
            .next_layout();
        let Some(layout) = layout else {
            return Ok(false);
        };
        let sim = transforms::layout_to_sim(&layout);
        let labels = (0..sim.capacity())
            .map(|v| Ok(Some(v.into_pyobject(py)?.into_any())))
            .collect::<PyResult<Vec<_>>>()?;
        let dicts = nxdicts::NxDicts::new(node.clone(), adj.clone(), None);
        nxdicts::write_sim(&dicts, &sim, &labels, |_, _| Ok(()))?;
        Ok(true)
    }
}

#[pymodule(gil_used = false)]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CoreGraph>()?;
    m.add_class::<operators::OpView>()?;
    m.add_class::<B24NonisoTrees>()?;
    m.add_function(wrap_pyfunction!(transforms::_b24_modular_product, m)?)?;
    m.add_function(wrap_pyfunction!(operators::_op_join, m)?)?;
    m.add_function(wrap_pyfunction!(operators::_op_pred_combinations, m)?)?;
    m.add_function(wrap_pyfunction!(operators::_op_product, m)?)?;
    m.add_function(wrap_pyfunction!(operators::_op_rooted_product, m)?)?;
    m.add_function(wrap_pyfunction!(operators::_op_corona_product, m)?)?;
    m.add_function(wrap_pyfunction!(operators::_tuple_hash, m)?)?;
    m.add_function(wrap_pyfunction!(operators::_op_projection, m)?)?;
    m.add_function(wrap_pyfunction!(operators::_op_intersection, m)?)?;
    m.add_class::<AllPaths>()?;
    m.add_class::<PredPaths>()?;
    m.add_class::<LinkScorer>()?;
    m.add_class::<AllTopoSorts>()?;
    m.add_class::<ClosureDag>()?;
    m.add_class::<RootLeafPaths>()?;
    m.add_class::<GroupPre>()?;
    m.add_class::<GoldbergRadzikState>()?;
    m.add_class::<AntichainIter>()?;
    m.add_class::<SimplePathIter>()?;
    m.add_class::<YenIter>()?;
    m.add_class::<DagLca>()?;
    m.add_class::<TreeLca>()?;
    m.add_class::<Boruvka>()?;
    m.add_class::<CliqueQueue>()?;
    m.add_class::<CliqueSearchIter>()?;
    m.add_class::<FlowRun>()?;
    m.add_class::<GirvanNewman>()?;
    m.add_class::<TspTour>()?;
    m.add_class::<MaxCutState>()?;
    m.add_class::<bipartite_more::MultiEdges>()?;
    m.add_function(wrap_pyfunction!(graph::build_graph, m)?)?;
    m.add_function(wrap_pyfunction!(serialize::_core_graph_from_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(native::build_native, m)?)?;
    m.add_function(wrap_pyfunction!(native::build_native_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(rx::build_rx, m)?)?;
    m.add_function(wrap_pyfunction!(_py_sum, m)?)?;
    m.add_function(wrap_pyfunction!(_set_sum_ints_compensated, m)?)?;
    m.add_function(wrap_pyfunction!(_degree_sequence_test, m)?)?;
    m.add_function(wrap_pyfunction!(_digraphical, m)?)?;
    m.add_function(wrap_pyfunction!(_plain_int_list, m)?)?;
    m.add_function(wrap_pyfunction!(_replay_sets, m)?)?;
    m.add_function(wrap_pyfunction!(_add_plain_edges, m)?)?;
    Ok(())
}
