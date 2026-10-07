# Changelog

All notable changes to rustnx are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). While rustnx is at 0.x, minor
versions may change behavior.

## [Unreleased]

### Added
- `simple_cycles` and `chordless_cycles` as hybrids: NetworkX's own code
  makes every choice that depends on set order, and the cycle searches inside
  it (Johnson's, the length-bounded one and the chordless one) run in Rust on
  the same neighbor lists, yielding the same cycles in the same order, lazily.
  4x to 10x faster when there are many cycles. Multigraphs too.
- `eulerize` (2x to 3x, its matching in rustnx), `local_and_global_consistency`
  (3x to 5x) and `harmonic_function` (1.2x) as hybrids.
- GraphML and GEXF readers in Rust: `read_graphml` and `parse_graphml`
  (about 8x faster) and `read_gexf` (about 5x), with an XML parser
  (quick-xml) and NetworkX's readers followed step by step, so node, row and
  attribute order, value types, edge ids and keys, and the choice of graph
  class all match. Documents NetworkX reads specially (yFiles data, ports,
  hyperedges, nested GraphML graphs, a DOCTYPE, non-UTF-8 encodings) go to
  NetworkX, and expat checks every document, so malformed files raise
  NetworkX's own error.
- pandas: `from_pandas_edgelist` (1.4x to 2.6x faster), `to_pandas_edgelist`
  (2.4x to 5x) and `from_pandas_adjacency` (2x to 3x).
- Generators: `random_k_out_graph` (replays NetworkX 3.5+'s NumPy draws, 3x
  to 5x faster, and 3.4's pure-Python draws, 70x to 150x), `graph_atlas`
  (about 600x: the atlas is parsed once per process) and `graph_atlas_g`,
  and `random_unlabeled_tree`, `random_unlabeled_rooted_tree` and
  `random_unlabeled_rooted_forest` (NetworkX's own sampling with cached tree
  counts, 5x to 160x).
- `docs/COVERAGE.md` gives measured reasons for 55 more functions that stay
  in NetworkX (small named graphs, in-place attribute setters, graph
  constructors, eigensolver-bound expanders, ...).
- NumPy/SciPy round. Hybrids run NetworkX's own SciPy code with the sparse
  matrix it builds supplied by rustnx (identical inputs, so identical
  results): `normalized_laplacian_matrix` and `bethe_hessian_matrix` (3x to
  8x faster), `tournament_matrix` (20x to 35x), `eigenvector_centrality_numpy`
  (2x to 4x) and `hits` (2x to 5x). The last two use ARPACK, which starts
  from a random vector, so their last bits vary between runs in NetworkX
  itself.
- Hybrids whose expensive inner call runs in rustnx: `k_factor` (its
  `max_weight_matching`, 12x to 29x), `junction_tree` (chordal completion,
  cliques and spanning tree, about 35x) and `find_induced_nodes`
  (`is_chordal`, about 1.6x). Bit for bit identical.
- `attr_matrix` and `attr_sparse_matrix` (without `node_attr`, 12x to 14x)
  and `magnetic_laplacian_matrix` (NetworkX 3.7+, 9x to 19x), built in Rust,
  bit for bit identical.
- Weighted `clustering` and `average_clustering` in Rust with fast floats
  (about 90x faster); with exact floats they run in NetworkX.
- `docs/COVERAGE.md` lists functions that stay in NetworkX on purpose, each
  with its measured reason (for example, LAPACK-bound spectra, where building
  the matrix in Rust measured about 1x).
- Float settings. `betweenness_centrality`, `edge_betweenness_centrality`
  and `pagerank` have a fast mode (the default, whose floats can differ from
  NetworkX's in the last bits) and an exact mode (bit for bit NetworkX's).
  Choose globally with `nx.config.backends.rustnx.exact_floats` or per
  function with `nx.config.backends.rustnx.exact_floats_overrides` (per
  function wins), also through `rustnx.enable(...)`, `RUSTNX_EXACT_FLOATS`
  and `RUSTNX_EXACT_FLOATS_OVERRIDES`. `rustnx.float_settings()` and
  `rustnx.explain_floats()` report what applies; `verbose` (or
  `RUSTNX_VERBOSE=1`) logs it as functions run, and every call is logged at
  DEBUG on the `rustnx` logger. See the README's "Fast and exact floats".
- 20 more functions: `degree_centrality`, `in_degree_centrality`,
  `out_degree_centrality`, `node_connected_component`,
  `articulation_points`, `biconnected_components`,
  `biconnected_component_edges`, `is_biconnected`, `attracting_components`,
  `number_attracting_components`, `is_attracting_component`,
  `bfs_predecessors`, `bfs_successors`, `bfs_layers`,
  `descendants_at_distance`, `dfs_postorder_nodes`, `dfs_predecessors`,
  `dfs_successors`, `is_tree` and `is_forest`.
- 93 more functions in batches 2 to 6, each matching NetworkX 3.4, 3.5 and
  3.7 exactly (values, order, set iteration order and errors), following
  each version's algorithm where it changed:
  - Shortest paths: multi-source Dijkstra, `predecessor`,
    `dijkstra_predecessor_and_distance`, `single_target_shortest_path_length`,
    all shortest paths from one source or all pairs, NetworkX's queue-based
    Bellman-Ford family with its negative-cycle detection,
    `negative_edge_cycle`, `find_negative_cycle`, and A* without a heuristic.
  - DAGs and traversal: longest path, lexicographical and all topological
    sorts, `has_cycle`, transitive closure and reduction, `is_aperiodic`,
    `v_structures`, `root_to_leaf_paths`, `dag_to_branching`, labeled and
    edge BFS/DFS, `generic_bfs_edges`, Kosaraju components, `condensation`
    and `is_semiconnected`.
  - Centrality: subset and edge-subset betweenness, load and edge load,
    percolation, group betweenness, closeness and degree centrality,
    `prominent_group`, local and global reaching centrality, VoteRank and
    dispersion, all bit-for-bit identical.
  - Cores, clustering and distance: k-shell, k-crust, k-corona, k-truss,
    onion layers, square clustering, generalized degree, `all_triangles`,
    `centroid` (`barycenter`), `harmonic_diameter` and the distance-regular
    tests; `is_coloring`, `is_equitable`, and more `greedy_color` strategies
    (DSATUR, connected sequential, random sequential).
  - Trees and structure: arborescence and branching tests,
    `to_prufer_sequence`, `kruskal_mst_edges`, bridges, local bridges,
    chain decomposition, isolates, regular and tournament tests, dominators
    and dominance frontiers, Euler tests, circuits and paths, `cycle_basis`,
    `find_cycle` and `girth`.
- 88 more functions in batches 7 to 11, matching NetworkX 3.4, 3.5 and 3.7
  exactly:
  - Paths and cycles: the Floyd-Warshall family (including
    `floyd_warshall_numpy`), `johnson`, `goldberg_radzik`, `antichains`,
    `antichain_width`, `all_simple_paths`, `all_simple_edge_paths`,
    `shortest_simple_paths`, `is_simple_path` and `minimum_cycle_basis`.
  - Trees: Edmonds branchings and spanning arborescences (with the weight
    rewrites NetworkX makes), `greedy_branching`, `branching_weight`,
    `prim_mst_edges`, `boruvka_mst_edges`, `partition_spanning_tree`,
    Prufer and nested tuple conversions, `nx.tree.centroid` and the three
    lowest-common-ancestor functions.
  - Graph classes: `is_planar`, `check_planarity` (the same embedding) and
    the counterexample functions, `is_chordal`, `chordal_graph_treewidth`,
    `complete_to_chordal_graph`, `is_at_free`, `is_perfect_graph`, and the
    tournament module's `is_reachable`, `is_strongly_connected` and
    `score_sequence`.
  - Triads, d-separation, degree-sequence tests, boundaries, matching (a
    port of NetworkX's blossom algorithm), edge covers, dominating sets and
    cliques.
  - Isomorphism: `is_isomorphic`, the VF2++ yes/no tests, the
    `could_be_isomorphic` family, tree isomorphism and Weisfeiler-Lehman
    hashes; and eight bipartite functions.
- 117 more functions in batches 12 to 16, matching NetworkX 3.4 to 3.7
  exactly:
  - Flows and cuts: every maximum flow algorithm (including the default
    `preflow_push`), `minimum_cut`, residual networks identical to
    NetworkX's, `gomory_hu_tree`, network simplex and minimum cost flow,
    and the cut measures (`cut_size`, `conductance`, expansions).
  - Connectivity: node and edge connectivity (local, global, average, all
    pairs), minimum node and edge cuts, disjoint paths, `stoer_wagner`,
    bridge components, k-edge components (k up to 2) and augmentation.
  - Assortativity and mixing, link prediction, reciprocity, rich club,
    `s_metric` and `number_of_walks`.
  - Communities: modularity and partition measures, greedy modularity,
    Girvan-Newman, seeded label propagation and `asyn_fluidc`; efficiency,
    vitality, Wiener-type indices, `flow_hierarchy` and `voronoi_cells`.
  - Approximation: vertex cover, dominating sets, maximal matching, greedy
    and annealing TSP, treewidth, diameter, max cut, Steiner tree, densest
    subgraph; and `complement`, `power`, `difference`,
    `symmetric_difference`, `is_kl_connected`, `kl_connected_subgraph`.
- 131 more functions in batches 17 to 21, each matching NetworkX 3.4 to 3.7
  exactly:
  - Deterministic generators (34): complete, cycle, path, star, wheel,
    ladders, lollipop, barbell, tadpole, trees, bipartite and multipartite,
    Turan, grids and hypercubes, hexagonal and triangular lattices,
    circulant, caveman, ring of cliques, windmill, Sudoku, LCF, generalized
    Petersen, Dorogovtsev-Goltsev-Mendes, Mycielski, Paley and Kneser graphs,
    built in Rust straight into NetworkX graphs, with `create_using`.
  - Seeded random generators (25, plus the old `random_lobster` name):
    G(n, p) and G(n, m), Barabasi-Albert (plain, dual, extended),
    Watts-Strogatz (plain, Newman, connected), `powerlaw_cluster_graph`,
    `random_regular_graph`, growing networks, `random_uniform_k_out_graph`,
    `random_lobster_graph`, `random_tournament`, stochastic block, random and
    planted partition graphs, random geometric and Waxman graphs, and the
    bipartite `random_graph` and `gnmk_random_graph`. They replay NetworkX's
    draws on a copy of CPython's `random.Random` (`src/algorithms/pyrandom.rs`),
    so the same seed gives the same graph and leaves the generator in the
    same state.
  - Matrices and conversion (20): SciPy and NumPy adjacency, Laplacian,
    incidence and biadjacency matrices, `to_dict_of_lists`, the attribute
    getters, `is_weighted`, `is_negatively_weighted`, `number_of_selfloops`,
    `relabel_nodes` and `convert_node_labels_to_integers` (copies), and graph
    builders from dicts, edge lists, NumPy and SciPy arrays.
  - Readers and parsers (23): edge lists, adjacency lists (and multiline),
    LEDA, Pajek, GML, graph6 and sparse6, and the JSON node-link, adjacency,
    Cytoscape and tree formats, parsed in Rust.
  - Operators and structure (28): union, compose, disjoint union and
    intersection (and their `_all` forms), `full_join`, `reverse`, the six
    graph products, `line_graph`, `ego_graph`, `moral_graph`, structural
    holes (`constraint`, `effective_size`, `local_constraint`), tree
    broadcasting, and the bipartite `density`, `degree_centrality` and
    projections.
  - Generators, readers and graph builders take no graph argument, so they
    run in rustnx with `backend="rustnx"` or with rustnx listed in
    `nx.config.backend_priority.generators`.
- 93 more functions in batches 22 to 26, each matching NetworkX 3.4 to 3.7
  exactly:
  - Degree-sequence and tree generators (21): configuration models (plain,
    directed, bipartite), Havel-Hakimi graphs (plain, directed and the
    three bipartite variants), `expected_degree_graph`,
    `degree_sequence_tree`, `random_degree_sequence_graph`, power-law trees
    and sequences, random labeled trees, rooted trees and forests,
    `random_cograph`, `random_clustered_graph`, joint degree graphs and
    `is_valid_directed_joint_degree`.
  - Growth, geometric and community generators (20): duplication models,
    `scale_free_graph`, `random_shell_graph`, `relaxed_caveman_graph`,
    `gaussian_random_partition_graph`, `navigable_small_world_graph`,
    geographical threshold, soft and thresholded random geometric graphs,
    `geometric_soft_configuration_graph`, `geometric_edges`, the random
    intersection graphs, `random_k_lift`, `maybe_regular_expander_graph`,
    `random_internet_as_graph` and the bipartite preferential attachment
    graph.
  - Generators and transformations (17): Margulis-Gabber-Galil, chordal
    cycle and Harary graphs, prefix trees, interval and visibility graphs,
    `nonisomorphic_trees`, `join_trees`, `mycielskian`, `stochastic_graph`,
    `contracted_nodes`, `contracted_edge`, `quotient_graph`,
    `modular_product` and `inverse_line_graph`.
  - Cliques, communities, structure and approximation (23): `find_cliques`
    and the clique graphs, `k_clique_communities`,
    `kernighan_lin_bisection`, `dominating_set`, `maximal_independent_set`,
    `chordal_graph_cliques`, `find_asteroidal_triple`, `hamiltonian_path`,
    the auxiliary connectivity graphs, and the approximation module's clique,
    independent set, Ramsey, treewidth, node connectivity and
    `metric_closure` functions.
  - Bipartite measures (12): clustering, betweenness, weighted projections,
    `min_edge_cover`, `eppstein_matching`, `maximal_extendability`,
    `modularity`, and `cd_index`.
- Multigraph support for `pagerank`, the degree centralities, minimum and
  maximum spanning trees and edges (with keys), `number_of_selfloops`,
  bipartite `density`, `s_metric`, `has_eulerian_path`, `is_semieulerian`
  and the group degree centralities.
- `docs/COVERAGE.md`, generated by `scripts/gen_coverage.py`: every function
  NetworkX lets a backend implement, and which ones rustnx does.

### Fixed
- On Python 3.12+, weighted `wiener_index`, `average_shortest_path_length`
  and `closeness_centrality` now add float distances the way Python 3.12's
  compensated `sum()` does, so the last bits match NetworkX (they differed
  for weights whose sums aren't exact).
- Sums that mix ints and floats follow Python 3.14's `sum()`, which also
  compensates ints met after the first float (3.12 and 3.13 don't); detected
  at import, not by version number.
- Function arguments that aren't graphs (such as a node view passed to
  `is_partition`) are handed to NetworkX instead of failing in dispatch.
- `ancestors` now returns its set in the same iteration order as NetworkX
  (it followed in-edges in a different order; the nodes were right).
- On NetworkX 3.6, `nx.tree.centroid` (registered there as `centroid`) runs
  the tree centroid, not 3.7's distance-measure `centroid`.
- `docs/API.md` lists functions that live in NetworkX submodules (such as
  `nx.tree.mst.kruskal_mst_edges`) under their real path.
- `docs/API.md` no longer lists internal keyword arguments as parameters.
- `strongly_connected_components` on NetworkX 3.4 to 3.6: when the whole
  graph is one component, the set's iteration order now matches (3.7 added an
  early exit that fills the set in a different order).

### Changed
- `LICENSE` now includes NetworkX's copyright notice and license, which
  covers the parts of rustnx ported from or taken from NetworkX.
- `betweenness_centrality` and `edge_betweenness_centrality` with exact
  floats add per-source contributions in NetworkX's order, matching it bit
  for bit (a few percent slower than fast floats).
- `pagerank` with exact floats builds the sparse matrix in Rust and runs the
  installed NetworkX's own SciPy code on it, matching it bit for bit; with
  exact floats, multigraphs run in NetworkX.
- Pickled rustnx graphs also store whether each weight attribute holds only
  plain ints and floats (format `RNX4`), so an unpickled graph builds SciPy
  matrices in Rust as the original does. Older pickles still load.
- `rustnx.enable()` also puts rustnx first in
  `nx.config.backend_priority.generators`, which NetworkX uses for every
  function that returns a graph (generators, readers, graph builders, and
  operations such as `union`, `k_core` or `minimum_spanning_tree`). Before,
  those ran in rustnx only with `backend="rustnx"`. Calls whose input is
  small (under about 100 nodes, edges or lines) stay in NetworkX, which is
  faster there; an explicit `backend="rustnx"` always runs rustnx.
- Internal: the generators, readers, graph builders and operators fill new
  NetworkX graphs through one shared module (`src/algorithms/nxdicts.rs`)
  instead of five separate helpers.
- Internal: one replica of CPython's set table (`src/algorithms/pyset.rs`)
  now serves both the flow algorithms and link prediction, and the
  connectivity functions use the flow module's Edmonds-Karp instead of a
  second copy. Results and speed are unchanged.

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
