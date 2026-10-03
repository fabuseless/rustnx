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
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

use algorithms::link_analysis::{self, PagerankInput};
use algorithms::shortest_paths_more as more_paths;
use algorithms::traversal::{self, DijkstraState, NegativeCycle};
use algorithms::{centrality, cluster, directed, distance, paths, spectral, structure};
use graph::CoreGraph;

impl From<NegativeCycle> for PyErr {
    fn from(_: NegativeCycle) -> PyErr {
        // Same arguments as NetworkX's `_dijkstra_multisource`.
        PyValueError::new_err(("Contradictory paths found:", "negative weights?"))
    }
}

fn all_nodes(n: usize) -> Vec<u32> {
    (0..n as u32).collect()
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

    #[pyo3(signature = (weight=None, endpoints=false, sources=None))]
    fn betweenness(
        &self,
        py: Python<'_>,
        weight: Option<&str>,
        endpoints: bool,
        sources: Option<Vec<u32>>,
    ) -> PyResult<Vec<f64>> {
        let w = self.weight_slice(weight, false)?;
        let sources = sources.unwrap_or_else(|| all_nodes(self.n));
        for &s in &sources {
            self.check_index(s as usize)?;
        }
        Ok(py.detach(|| {
            centrality::betweenness(&self.succ, self.adj(true), self.n, w, endpoints, &sources)
        }))
    }

    /// Unscaled edge betweenness, one value per edge in `edges_in_order`.
    #[pyo3(signature = (weight=None, sources=None))]
    fn edge_betweenness(
        &self,
        py: Python<'_>,
        weight: Option<&str>,
        sources: Option<Vec<u32>>,
    ) -> PyResult<Vec<f64>> {
        let w = self.weight_slice(weight, false)?;
        let sources = self.sources_or_all(sources)?;
        Ok(py.detach(|| {
            let (edge_id, m) = self.edge_ids();
            centrality::edge_betweenness(&self.succ, self.n, w, &edge_id, m, &sources)
        }))
    }

    #[pyo3(signature = (distance=None, wf_improved=true, sources=None))]
    fn closeness(
        &self,
        py: Python<'_>,
        distance: Option<&str>,
        wf_improved: bool,
        sources: Option<Vec<u32>>,
    ) -> PyResult<Vec<f64>> {
        // NetworkX runs closeness on `G.reverse()` for directed graphs.
        let (adj, w) = self.reverse_exact(distance)?;
        let sources = sources.unwrap_or_else(|| all_nodes(self.n));
        for &s in &sources {
            self.check_index(s as usize)?;
        }
        Ok(py.detach(|| centrality::closeness(adj, self.n, w, wf_improved, &sources))?)
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
    #[pyo3(signature = (weight, sources=None))]
    #[allow(clippy::type_complexity)]
    fn dijkstra_stats(
        &self,
        py: Python<'_>,
        weight: &str,
        sources: Option<Vec<u32>>,
    ) -> PyResult<(Vec<Option<(usize, f64)>>, Option<f64>)> {
        let w = self
            .weight_slice(Some(weight), false)?
            .expect("weight given");
        let sources = self.sources_or_all(sources)?;
        Ok(py.detach(|| {
            let (stats, total) = distance::dijkstra_stats(&self.succ, self.n, w, &sources);
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
        let adj = self.adj(reverse);
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

    /// `find_negative_cycle`: the cycle, or `None` if none is detected.
    /// Raises NotImplementedError where NetworkX reaches code it never
    /// expects to (its errors there are accidental).
    #[pyo3(signature = (source, weight=None))]
    fn find_negative_cycle(
        &self,
        py: Python<'_>,
        source: usize,
        weight: Option<&str>,
    ) -> PyResult<Option<Vec<u32>>> {
        self.check_index(source)?;
        let w = self.weight_slice(weight, false)?;
        py.detach(|| {
            let bf = more_paths::bellman_ford(&self.succ, self.n, w, source, true);
            let Some(v) = bf.cycle else { return Ok(None) };
            more_paths::negative_cycle_from(&bf.pred.lists, self.n, v)
                .map(Some)
                .ok_or_else(|| PyNotImplementedError::new_err("negative cycle not identified"))
        })
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

/// Lazy iterator over `all_shortest_paths` results (node positions).
#[pyclass(module = "rustnx._core")]
pub struct AllPaths(paths::PathsFromPreds);

#[pymethods]
impl AllPaths {
    fn next_path(&mut self) -> Option<Vec<u32>> {
        self.0.next_path()
    }
}

/// CPython's float `sum()` (exposed for tests).
#[pyfunction]
fn _py_sum(values: Vec<f64>, compensated: bool) -> f64 {
    spectral::py_sum(values.into_iter(), compensated)
}

#[pymodule(gil_used = false)]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CoreGraph>()?;
    m.add_class::<AllPaths>()?;
    m.add_class::<PredPaths>()?;
    m.add_function(wrap_pyfunction!(graph::build_graph, m)?)?;
    m.add_function(wrap_pyfunction!(serialize::_core_graph_from_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(native::build_native, m)?)?;
    m.add_function(wrap_pyfunction!(native::build_native_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(rx::build_rx, m)?)?;
    m.add_function(wrap_pyfunction!(_py_sum, m)?)?;
    Ok(())
}
