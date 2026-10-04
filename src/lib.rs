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
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList, PyTuple};

use algorithms::link_analysis::{self, PagerankInput};
use algorithms::shortest_paths_more as more_paths;
use algorithms::traversal::{self, DijkstraState, NegativeCycle};
use algorithms::{
    centrality, centrality_more, cluster, cores_more, dag, directed, distance, leftovers, paths,
    spectral, structure, structure_more,
};
use graph::CoreGraph;
use rayon::prelude::*;

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
    /// `leftovers::min_cycle_basis`); `None` where Dijkstra finds
    /// contradictory paths.
    #[pyo3(signature = (nodes, edges, chords, weight=None))]
    fn min_cycle_basis(
        &self,
        py: Python<'_>,
        nodes: Vec<u32>,
        edges: Vec<(u32, u32)>,
        chords: Vec<(u32, u32)>,
        weight: Option<&str>,
    ) -> PyResult<Option<Vec<Vec<u32>>>> {
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
            Ok(cb) => Ok(Some(cb)),
            Err(leftovers::McbError::Contradictory) => Ok(None),
            Err(leftovers::McbError::Unsupported) => Err(PyNotImplementedError::new_err(
                "rustnx can't reproduce this minimum cycle basis case",
            )),
        }
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
    m.add_class::<AllTopoSorts>()?;
    m.add_class::<ClosureDag>()?;
    m.add_class::<RootLeafPaths>()?;
    m.add_class::<GroupPre>()?;
    m.add_class::<GoldbergRadzikState>()?;
    m.add_class::<AntichainIter>()?;
    m.add_class::<SimplePathIter>()?;
    m.add_class::<YenIter>()?;
    m.add_function(wrap_pyfunction!(graph::build_graph, m)?)?;
    m.add_function(wrap_pyfunction!(serialize::_core_graph_from_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(native::build_native, m)?)?;
    m.add_function(wrap_pyfunction!(native::build_native_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(rx::build_rx, m)?)?;
    m.add_function(wrap_pyfunction!(_py_sum, m)?)?;
    Ok(())
}
