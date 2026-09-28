//! Rust core for the `rustnx` NetworkX backend.
//!
//! Python hands over NetworkX's adjacency once (`build_graph`), and every
//! algorithm then runs on integer-indexed CSR arrays with the GIL released.

mod algorithms;
mod graph;

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

use algorithms::centrality;
use algorithms::traversal::{self, DijkstraState, NegativeCycle};
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

    #[getter]
    fn directed(&self) -> bool {
        self.directed
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

    #[pyo3(signature = (distance=None, wf_improved=true, sources=None))]
    fn closeness(
        &self,
        py: Python<'_>,
        distance: Option<&str>,
        wf_improved: bool,
        sources: Option<Vec<u32>>,
    ) -> PyResult<Vec<f64>> {
        // NetworkX runs closeness on `G.reverse()` for directed graphs.
        let w = self.weight_slice(distance, true)?;
        let sources = sources.unwrap_or_else(|| all_nodes(self.n));
        for &s in &sources {
            self.check_index(s as usize)?;
        }
        Ok(py.detach(|| centrality::closeness(self.adj(true), self.n, w, wf_improved, &sources))?)
    }
}

impl CoreGraph {
    fn check_index(&self, v: usize) -> PyResult<()> {
        if v < self.n {
            Ok(())
        } else {
            Err(PyIndexError::new_err("node index out of range"))
        }
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CoreGraph>()?;
    m.add_function(wrap_pyfunction!(graph::build_graph, m)?)?;
    Ok(())
}
