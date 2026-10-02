# Changelog

All notable changes to rustnx are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). While rustnx is at 0.x, minor
versions may change behavior.

## [Unreleased]

### Added
- Distance measures: `eccentricity`, `diameter`, `radius`, `center`,
  `periphery`, `average_shortest_path_length` and `wiener_index`. Unweighted
  graphs use bit-parallel BFS (about 800× faster than NetworkX); weighted ones
  use parallel Dijkstra (about 40×).
- `all_pairs_shortest_path_length` and `all_pairs_dijkstra_path_length`,
  computed in parallel batches.

- NetworkX backend (opt in with `nx.config.backend_priority = ["rustnx"]`)
  with Rust implementations of:
  - `betweenness_centrality` (unweighted and weighted, `normalized`,
    `endpoints`, `k` sampling), run in parallel
  - `closeness_centrality` (unweighted and `distance=`, `wf_improved`, `u=`),
    run in parallel, with results identical to NetworkX
  - `single_source_shortest_path_length` and
    `single_source_dijkstra_path_length`, with `cutoff`
  - `connected_components`, `number_connected_components`, `is_connected`
  - `pagerank`, with every NetworkX option
  - `strongly_connected_components` and `weakly_connected_components`, plus
    their `number_*` and `is_*` variants
  - `topological_sort`, `topological_generations`, `is_directed_acyclic_graph`
- Generators (`topological_sort`, the component functions) notice when the
  graph changes during iteration. `topological_sort` then behaves exactly as
  NetworkX does; the component generators raise `RuntimeError` instead of
  returning stale results.
- Automatic fallback to NetworkX for anything unsupported: other functions,
  multigraphs, callable or non-numeric weights, and parameters rustnx doesn't
  implement.
- On small graphs (under 500 nodes), the linear-time functions stay in
  NetworkX, because it's faster there.
- Converting a graph to rustnx is up to 3× faster. Directed graphs are read
  once, not twice, and node lookups skip the Python dict for integer-labelled
  graphs and for neighbor keys that are the same objects as the node keys.
- `closeness_centrality` on unweighted graphs runs 64 BFS searches at once
  (bit-parallel BFS): about 10× faster than before, with identical results.
- `benchmarks/compare.py` compares rustnx with nx-rustworkx and
  FrankenNetworkX, checking every result against NetworkX.
- Graphs that rustnx has worked on can be pickled and deep-copied (e.g. sent
  to `multiprocessing` workers). Previously this failed, because NetworkX
  caches rustnx's converted graph on the original graph.
- Supports NetworkX 3.4 through 3.7 and Python 3.10+.
- Prebuilt wheels for Linux (x86-64, ARM), macOS (Intel, Apple Silicon) and
  Windows, each tested on its platform, published to PyPI by pushing a
  version tag.
