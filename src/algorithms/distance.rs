//! Many-source shortest path work: bit-parallel BFS statistics, and batched
//! BFS / Dijkstra for the all-pairs functions.

use rayon::prelude::*;

use super::traversal::{bfs_lengths, DijkstraState, NegativeCycle};
use crate::graph::Csr;

/// Per-source BFS statistics.
#[derive(Clone, Copy)]
pub struct BfsStats {
    /// Nodes reached, including the source.
    pub reached: usize,
    /// Sum of distances to reached nodes.
    pub total: u64,
    /// Largest distance to a reached node (the eccentricity if all reached).
    pub max: u32,
}

/// Up to 64 breadth-first searches at once, one per bit of a `u64`.
///
/// Each node carries a bitmask of the searches that have reached it, so one
/// pass over a node's edges advances every search at once, cutting edge
/// scans by up to 64x. Distances are the same as separate BFS runs, so the
/// counts, sums and maxima are identical.
struct BitParallelBfs {
    seen: Vec<u64>,
    frontier: Vec<u64>,
    next: Vec<u64>,
}

impl BitParallelBfs {
    fn new(n: usize) -> Self {
        BitParallelBfs {
            seen: vec![0; n],
            frontier: vec![0; n],
            next: vec![0; n],
        }
    }

    fn run(&mut self, adj: &Csr, batch: &[u32]) -> Vec<BfsStats> {
        debug_assert!(batch.len() <= 64);
        let n = self.seen.len();
        self.seen.fill(0);
        self.frontier.fill(0);
        let mut stats = vec![
            BfsStats {
                reached: 1,
                total: 0,
                max: 0
            };
            batch.len()
        ];
        for (bit, &s) in batch.iter().enumerate() {
            self.seen[s as usize] |= 1 << bit;
            self.frontier[s as usize] |= 1 << bit;
        }
        let mut level = 0u32;
        loop {
            level += 1;
            self.next.fill(0);
            for v in 0..n {
                let f = self.frontier[v];
                if f != 0 {
                    for &w in adj.neighbors(v) {
                        self.next[w as usize] |= f;
                    }
                }
            }
            let mut any = false;
            for w in 0..n {
                let new = self.next[w] & !self.seen[w];
                self.frontier[w] = new;
                if new != 0 {
                    any = true;
                    self.seen[w] |= new;
                    let mut bits = new;
                    while bits != 0 {
                        let st = &mut stats[bits.trailing_zeros() as usize];
                        st.reached += 1;
                        st.total += level as u64;
                        st.max = level;
                        bits &= bits - 1;
                    }
                }
            }
            if !any {
                break;
            }
        }
        stats
    }
}

/// BFS statistics for each source, 64 sources per parallel batch.
pub fn bfs_stats(adj: &Csr, n: usize, sources: &[u32]) -> Vec<BfsStats> {
    sources
        .par_chunks(64)
        .map_init(
            || BitParallelBfs::new(n),
            |state, batch| state.run(adj, batch),
        )
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect()
}

/// Per-source Dijkstra statistics; `max` is in the weights' units.
#[derive(Clone, Copy)]
pub struct DijkstraStats {
    pub reached: usize,
    pub max: f64,
}

/// `(reached, max distance, distances in pop order)` for one source.
type SourceDists = (usize, f64, Vec<f64>);

/// Dijkstra from each source, plus the sum of every distance found.
///
/// The sum is taken in the order NetworkX's
/// `sum(l for u in G for l in path_length(u).values())` takes it (sources
/// in order, each source's distances in pop order, one running total), so
/// float results match exactly. It is `None` if any source hit a negative
/// cycle; per-source results say which.
/// The total of all distances follows Python's `sum()` over NetworkX's
/// sequence (sources in order, each in Dijkstra pop order): compensated
/// (Neumaier) summation from Python 3.12 when `compensated` is set.
pub fn dijkstra_stats(
    adj: &Csr,
    n: usize,
    weights: &[f64],
    sources: &[u32],
    compensated: bool,
) -> (Vec<Result<DijkstraStats, NegativeCycle>>, Option<f64>) {
    const CHUNK: usize = 64;
    let mut stats = Vec::with_capacity(sources.len());
    let mut total = Some(0.0f64);
    let mut correction = 0.0f64;
    for chunk in sources.chunks(CHUNK * rayon::current_num_threads().max(1)) {
        let results: Vec<Result<SourceDists, NegativeCycle>> = chunk
            .par_iter()
            .map_init(
                || DijkstraState::new(n),
                |state, &s| {
                    state.run(adj, Some(weights), s as usize, None)?;
                    let dists: Vec<f64> = state
                        .order
                        .iter()
                        .map(|&v| state.dist[v as usize])
                        .collect();
                    let max = dists.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    Ok((dists.len(), max, dists))
                },
            )
            .collect();
        for r in results {
            match r {
                Ok((reached, max, dists)) => {
                    if let Some(s) = total.as_mut() {
                        for d in dists {
                            if compensated {
                                let t = *s + d;
                                if s.abs() >= d.abs() {
                                    correction += (*s - t) + d;
                                } else {
                                    correction += (d - t) + *s;
                                }
                                *s = t;
                            } else {
                                *s += d;
                            }
                        }
                    }
                    stats.push(Ok(DijkstraStats { reached, max }));
                }
                Err(e) => {
                    total = None;
                    stats.push(Err(e));
                }
            }
        }
    }
    if let Some(s) = total.as_mut() {
        if correction != 0.0 && correction.is_finite() {
            *s += correction;
        }
    }
    (stats, total)
}

/// `single_source_shortest_path_length` for each source, in parallel.
pub fn bfs_many(adj: &Csr, n: usize, sources: &[u32], cutoff: f64) -> Vec<(Vec<u32>, Vec<u32>)> {
    sources
        .par_iter()
        .map(|&s| bfs_lengths(adj, n, s as usize, cutoff))
        .collect()
}

pub type Lengths = (Vec<u32>, Vec<f64>);

/// `single_source_dijkstra_path_length` for each source, in parallel.
pub fn dijkstra_many(
    adj: &Csr,
    n: usize,
    weights: Option<&[f64]>,
    sources: &[u32],
    cutoff: Option<f64>,
) -> Vec<Result<Lengths, NegativeCycle>> {
    sources
        .par_iter()
        .map_init(
            || DijkstraState::new(n),
            |state, &s| {
                state.run(adj, weights, s as usize, cutoff)?;
                let dists = state
                    .order
                    .iter()
                    .map(|&v| state.dist[v as usize])
                    .collect();
                Ok((state.order.clone(), dists))
            },
        )
        .collect()
}
