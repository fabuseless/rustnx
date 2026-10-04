//! Rust core for the `rustnx` NetworkX backend.
//!
//! Python hands over NetworkX's adjacency once (`build_graph`), and every
//! algorithm then runs on integer-indexed CSR arrays with the GIL released.

mod algorithms;
mod graph;
mod native;
mod rx;
mod serialize;

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

use algorithms::link_analysis::{self, PagerankInput};
use algorithms::traversal::{self, DijkstraState, NegativeCycle};
use algorithms::{
    centrality, centrality_more, cluster, directed, distance, paths, spectral, structure,
};
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

    /// Whether some edge has a negative value of `attr`.
    fn has_negative_weight(&self, attr: &str) -> PyResult<bool> {
        Ok(self
            .weight_slice(Some(attr), false)?
            .is_some_and(|w| w.iter().any(|&x| x < 0.0)))
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

/// CPython's float `sum()` (exposed for tests).
#[pyfunction]
fn _py_sum(values: Vec<f64>, compensated: bool) -> f64 {
    spectral::py_sum(values.into_iter(), compensated)
}

#[pymodule(gil_used = false)]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CoreGraph>()?;
    m.add_class::<AllPaths>()?;
    m.add_class::<GroupPre>()?;
    m.add_function(wrap_pyfunction!(graph::build_graph, m)?)?;
    m.add_function(wrap_pyfunction!(serialize::_core_graph_from_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(native::build_native, m)?)?;
    m.add_function(wrap_pyfunction!(native::build_native_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(rx::build_rx, m)?)?;
    m.add_function(wrap_pyfunction!(_py_sum, m)?)?;
    Ok(())
}
