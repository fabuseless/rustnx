# Changelog

All notable changes to rustnx are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). While rustnx is at 0.x, minor
versions may change behavior.

## [Unreleased]

## [0.1.0a3] - 2026-10-03

### Added
- `bidirectional_dijkstra`; weighted `shortest_path(G, source, target)` now
  runs in Rust too.
- `harmonic_centrality`, `eigenvector_centrality` and `katz_centrality`,
  with results bit-for-bit identical to NetworkX.
- `core_number`, `k_core` and `is_bipartite`.
- `bfs_edges`, `dfs_edges` and `dfs_preorder_nodes`.
- `bfs_tree` and `dfs_tree`.
- `minimum_spanning_edges`, `maximum_spanning_edges`, `minimum_spanning_tree`
  and `maximum_spanning_tree` (Kruskal).
- `all_shortest_paths`, with paths generated lazily.
- `greedy_color` (`largest_first`) and `label_propagation_communities`.
- Multigraph support (`MultiGraph`, `MultiDiGraph`) for components,
  traversals, the shortest path family, betweenness, closeness and harmonic
  centrality, the distance measures and `is_bipartite`. Parallel edges count
  once, with the minimum weight, as in NetworkX's code for these functions.
- Free-threaded Python (3.14t) support: the extension declares it doesn't
  need the GIL, wheels are built for 3.14t on Linux, macOS and Windows, and a
  test runs algorithms from many threads at once and compares the results.
- README section "When rustnx helps, and when it doesn't", covering
  conversion cost, caching, small graphs and inputs that run in NetworkX.
- `CONTRIBUTING.md` and `SECURITY.md`.
- `docs/API.md`: every supported function with the parameters rustnx handles,
  multigraph support, the small-graph rule and what falls back to NetworkX.
  It is generated from the code by `scripts/gen_api_docs.py`.
- A speed and dispatch guard in CI (`benchmarks/guard.py`): speedups over
  NetworkX must stay above floors, and calls must really run in Rust.
- Issue forms for bug reports and function requests.
- Dependabot: weekly grouped update PRs for GitHub Actions and Rust crates.

### Fixed
- Functions that return path lengths now hand graphs whose weights mix ints
  and floats to NetworkX. NetworkX returns an int or a float depending on the
  path; rustnx returned floats.
- Graph subclasses that override how their structure is read (such as
  NetworkX's internal `_AntiGraph`) are no longer converted, since rustnx
  reads the underlying adjacency directly.
- A cached conversion that NetworkX made for a function keeping all
  attributes on the NetworkX side could be reused without the edge weights a
  later call needed; missing weights are now converted on demand.
- Functions that live in NetworkX subpackages (such as
  `nx.community.label_propagation_communities`) are now found when binding
  arguments.

## [0.1.0a2] - 2026-10-02

### Fixed
- `rustnx.enable()` now actually runs rustnx on NetworkX graphs. It used to
  add `"networkx"` to `nx.config.backend_priority`, and NetworkX tries a
  listed backend that is also the input's own backend first, so every call
  on a NetworkX graph ran in NetworkX. `enable()` now sets the priority to
  `["rustnx"]` and turns on `nx.config.fallback_to_nx` for rustnx graphs.

## [0.1.0a1] - 2026-10-02

First public pre-release.

### Added
- `rustnx.rx`, a rustworkx-compatible API (`PyGraph`, `PyDiGraph`, and
  rustworkx-named algorithm functions) on the same Rust core. It matches
  rustworkx's index reuse, multigraph rules, neighbor order and exceptions,
  and is tested against rustworkx itself.
- Native graphs: `rustnx.Graph` and `rustnx.DiGraph` are built directly in
  Rust from edge tuples, or from NumPy arrays with `from_arrays`. No
  conversion is needed, they use about 5× less memory than NetworkX graphs,
  and they give exactly the results of the equivalent `networkx.Graph`.
  They're read-only, with a small NetworkX-like query API and
  `to_networkx()`.
- `rustnx.enable()` sets NetworkX's backend priority to rustnx, falling back
  to NetworkX for everything else.
- Distance measures: `eccentricity`, `diameter`, `radius`, `center`,
  `periphery`, `average_shortest_path_length` and `wiener_index`. Unweighted
  graphs use bit-parallel BFS (about 800× faster than NetworkX); weighted ones
  use parallel Dijkstra (about 40×).
- `all_pairs_shortest_path_length` and `all_pairs_dijkstra_path_length`,
  computed in parallel batches.
- Shortest paths that return the paths: `shortest_path`,
  `shortest_path_length`, `single_source_shortest_path`,
  `single_target_shortest_path`, `bidirectional_shortest_path`, `has_path`,
  `dijkstra_path`, `dijkstra_path_length`, `single_source_dijkstra`,
  `single_source_dijkstra_path`, `all_pairs_shortest_path`,
  `all_pairs_dijkstra_path` and `all_pairs_dijkstra`. They return the same
  paths as NetworkX, ties included.
- `descendants` and `ancestors`.
- `triangles`, `clustering`, `average_clustering` and `transitivity`
  (unweighted), with results identical to NetworkX.
- `edge_betweenness_centrality` (unweighted and weighted, `k` sampling), run
  in parallel.

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
