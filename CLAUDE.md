# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

rustnx is a Rust-accelerated backend for NetworkX (PyO3 + maturin), plus a
rustworkx-compatible API (`rustnx.rx`). Every function must return exactly
what the installed NetworkX returns (same values, dict order, ties and
errors) on NetworkX 3.4, 3.5 and 3.7; anything rustnx can't match raises
`NotImplementedError` so NetworkX falls back to its own code.
Floats too: where NetworkX uses NumPy or SciPy, pass identical arrays to the
same calls. The one exception is a function with a float setting
(`_config.FLOAT_FUNCTIONS`): its fast mode (the default) may differ in the
last bits, but its exact mode must match bit for bit. Adding one: register
it in `FLOAT_FUNCTIONS`, call `_config.note(name)` in the function, list
inputs only the fast mode can run in `interface.INEXACT_ONLY`, test both
modes, and update the README's "Fast and exact floats" table.

- Build: `maturin develop --release`
- Tests: `pytest tests` (compare against NetworkX), plus NetworkX's own suite
  with `NETWORKX_TEST_BACKEND=rustnx NETWORKX_FALLBACK_TO_NX=True pytest --pyargs networkx`
- Lint: `cargo fmt --check` and `cargo clippy --release -- -D warnings`
- New algorithms go in `python/rustnx/algorithms.py`, and must be added to
  `algorithms.__all__` and `_info.FUNCTIONS` (a test checks they match).
  Then add them to `scripts/gen_api_docs.py` and regenerate `docs/API.md`
  (a test checks it is up to date). See `CONTRIBUTING.md` for the full steps.
- Coverage list: after adding functions, regenerate `docs/COVERAGE.md` with
  `python scripts/gen_coverage.py`, run with the newest NetworkX.
- Speed guard: `python benchmarks/guard.py` (also the `speed-guard` CI job)
  checks speedup floors and that calls really run in Rust.

## Rules

- **No emojis.** Don't use emojis anywhere: replies, the todo list, docs,
  code comments or commit messages.
- **No session links.** Never put Claude Code session URLs
  (`claude.ai/code/session_...`) in commit messages, PR or issue
  descriptions, comments, or any file in the repo. That includes the
  `Claude-Session:` commit trailer; a `Co-Authored-By:` line is fine.
- **Always display the todo list as a table.** Whenever the todo list (the
  "To do" section below) is shown, use a Markdown table with the columns
  `#`, `Item`, `Status`, `Group`, `Owner` and `Notes`. `Status` is one of
  `Done`, `Partly done`, `Blocked` or `To do`. Keep the numbering
  stable so items can be referred to by number, and update this file when
  items are added or finished (finished items stay in the table).
- Releases: bump the version in `pyproject.toml`, `Cargo.toml` (semver form,
  e.g. `0.1.0-alpha.2`) and `python/rustnx/__init__.py`, date the CHANGELOG
  section, and merge that first. Only then does the owner publish a GitHub
  Release tagged `vX.Y.Z` and approve the `pypi` deployment. The wheels
  workflow fails at once if the tag doesn't match `pyproject.toml`, and a test
  checks the three version numbers agree.

## To do

| # | Item | Status | Group | Owner | Notes |
|---|------|--------|-------|-------|-------|
| 1 | README section: what gets faster and what stays at NetworkX speed | Done | Getting users | Claude | Written with item 18: "When rustnx helps, and when it doesn't" |
| 2 | GitHub issue templates (bug report, function request) | Done | Getting users | Claude | Issue forms in `.github/ISSUE_TEMPLATE/`; security reports link to private reporting |
| 3 | Announcement post (NetworkX Discussions, Scientific Python forum, r/Python) | To do | Getting users | Owner | Claude can draft it |
| 4 | `bidirectional_dijkstra` | Done | Algorithms | Claude | |
| 5 | `harmonic_centrality` | Done | Algorithms | Claude | |
| 6 | `eigenvector_centrality` | Done | Algorithms | Claude | |
| 7 | `katz_centrality` | Done | Algorithms | Claude | Katz `nstart` and per-node `beta` fall back |
| 8 | `hits` | Done | Algorithms | Claude | NumPy/SciPy round: NetworkX's own SciPy code on the sparse matrix rustnx builds, 2x to 5x faster. Before 3.7 it uses ARPACK (random start), so NetworkX's own last bits vary between runs; rustnx's are as close to NetworkX as NetworkX is to itself |
| 9 | `core_number`, `k_core` | Done | Algorithms | Claude | |
| 10 | `minimum_spanning_tree`, `minimum_spanning_edges` | Done | Algorithms | Claude | Kruskal, minimum and maximum. Prim and Borůvka run in Rust too since batch 8 (rustnx replays NetworkX's set operations for Prim's start nodes and Borůvka's ties) |
| 11 | Traversals: `bfs_edges`, `bfs_tree`, `dfs_preorder_nodes`, `dfs_tree` | Done | Algorithms | Claude | Also `dfs_edges` |
| 12 | `all_shortest_paths` | Done | Algorithms | Claude | Bellman-Ford falls back |
| 13 | `is_bipartite` | Done | Algorithms | Claude | |
| 14 | `simple_cycles` | Done | Algorithms | Claude | Set-order round: a hybrid. NetworkX's own code picks components and start nodes (by set order); Johnson's search and the length-bounded search inside it run in Rust on the same neighbor lists (`src/algorithms/cycle_search.rs`), lazily. 4x to 6x faster with many cycles, about even when NetworkX's component loop dominates |
| 15 | `louvain_communities`, `label_propagation_communities` | Done | Algorithms | Claude | Ports round: `louvain_partitions` in Rust with Python's own arithmetic (int/float mixing, CPython's `sum()`, libm `pow`), both NetworkX variants (3.4 to 3.6, and 3.7's), Python replaying the set operations; `louvain_communities` is NetworkX's code around it. 4x to 10x. Done earlier: `label_propagation_communities` (plus `greedy_color`) |
| 16 | Weighted clustering | Done | Algorithms | Claude | NumPy/SciPy round: fast floats in Rust (about 90x faster, within about 1e-15 relative); with exact floats it runs in NetworkX, whose sums follow set order and NumPy's pairwise summation |
| 17 | Multigraph support (`MultiGraph`, `MultiDiGraph`) | Done | Engineering | Claude | Components, traversals, shortest paths, betweenness/closeness/harmonic, distance measures, `is_bipartite`, and (batch 26) `pagerank`, degree centralities, spanning trees and edges with keys, `s_metric`, Euler path tests, group degree centralities (`interface.MULTIGRAPH_FUNCTIONS`). Early-exit tests (`is_regular`, `is_eulerian`) stay in NetworkX, which is faster there |
| 18 | Faster conversion from NetworkX, or docs steering to native `rustnx.Graph` | Done | Engineering | Claude | Profiled: conversion already costs about one pure-Python walk of the graph (the floor); documented in the README section "When rustnx helps, and when it doesn't" |
| 19 | Free-threaded Python 3.14 wheels | Done | Engineering | Claude | `gil_used = false`; cp314t wheels in `wheels.yml`; `tests/test_free_threading.py` |
| 20 | CI benchmark guard that fails on slowdowns | Done | Release quality | Claude | `benchmarks/guard.py`, CI job `speed-guard`: speedup floors plus dispatch checks |
| 21 | API docs listing every supported function and parameter | Done | Release quality | Claude | `docs/API.md`, generated by `scripts/gen_api_docs.py`; a test checks it is up to date |
| 22 | `CONTRIBUTING.md` | Done | Release quality | Claude | |
| 23 | `SECURITY.md` | Done | Release quality | Claude | Reports go through GitHub private vulnerability reporting (item 27) |
| 24 | Turn on "Automatically delete head branches" | Done | Release quality | Owner | |
| 25 | Enable Dependabot for CI actions | Done | Release quality | Claude | `.github/dependabot.yml`: weekly grouped PRs for GitHub Actions and Rust crates |
| 26 | Release `0.1.0` (drop the alpha label) after feedback | To do | Release quality | Owner + Claude | |
| 27 | Turn on private vulnerability reporting | Done | Release quality | Owner | `SECURITY.md` points to it |
| 28 | Shortest paths | Done | NetworkX coverage | Claude | Batches 2 and 7. `reconstruct_path` stays in NetworkX: it takes no graph, so there's nothing to convert or speed up |
| 29 | Centrality: the rest (current flow, subgraph and Estrada, trophic, Laplacian, second order, communicability, NumPy variants) | Done | NetworkX coverage | Claude | NumPy/SciPy round: `eigenvector_centrality_numpy` (hybrid, 2x to 4x). The other 18 stay in NetworkX with measured reasons in `docs/COVERAGE.md`: their time is in LAPACK or SciPy solves, so building the matrix in Rust gave about 1x |
| 30 | Components | Done | NetworkX coverage | Claude | Batches 1 and 3 |
| 31 | Directed acyclic graphs | Done | NetworkX coverage | Claude | Batches 3 and 7. `colliders` stays in NetworkX: its C-level loop beats building the tuples from Rust (measured twice) |
| 32 | Traversal | Done | NetworkX coverage | Claude | Batches 1 and 3. `bfs_beam_edges` stays in NetworkX: it calls a Python callable per neighbor, so nothing to speed up |
| 33 | Distance measures: the rest (resistance distance, effective graph resistance, Kemeny constant) | Done | NetworkX coverage | Claude | NumPy/SciPy round: all 3 stay in NetworkX (LAPACK-bound; measured about 1x with the matrix built in Rust). Done in batch 5: centroid/barycenter, harmonic diameter, distance-regular tests |
| 34 | Clustering and cores | Done | NetworkX coverage | Claude | Batch 5; weighted clustering is item 16 |
| 35 | Trees, branchings and arborescences: the rest (`random_spanning_tree`) | Done | NetworkX coverage | Claude | Ports round: `random_spanning_tree` makes NetworkX's draws on the same Python sets and hands each contracted Laplacian (built directly, the same integers in the same order) to the same `np.linalg.det` (multiplicative, unweighted or int weights, NetworkX 3.5+; 15x to 20x; with the contractions in Rust in the full-Rust round, 35x to 85x). NumPy/SciPy round: `junction_tree` (about 35x); `number_of_spanning_trees` stays in NetworkX (LAPACK). Done in batches 1, 6, 8 and 24 |
| 36 | Link analysis: `google_matrix` | Done | NetworkX coverage | Claude | NumPy/SciPy round: `hits` done (item 8); `google_matrix` stays in NetworkX (dense NumPy arithmetic, measured 1.1x) |
| 37 | Bipartite graphs: the rest (`birank`, `spectral_bipartivity`, `minimum_weight_full_matching`, `degrees`) | Done | NetworkX coverage | Claude | NumPy/SciPy round: all 4 stay in NetworkX (SciPy-bound, measured about 1x; `degrees` returns NetworkX's views). Done in batches 11, 18 to 23 and 26 |
| 38 | Connectivity, flows and cuts: the rest (`capacity_scaling`, `all_node_cuts`, `k_components`, k-edge components for k >= 3, weighted augmentation) | Done | NetworkX coverage | Claude | Flows and structure round: `capacity_scaling` (phases in Rust with Python's int arithmetic and set order, about 12x); `k_edge_augmentation` with k >= 3 or `avail`, and its helpers (`greedy_`, `partial_k_edge_augmentation`, `weighted_bridge_`, `weighted_one_edge_augmentation`, `general_k_edge_subgraphs`) as hybrids with their flows and connectivity tests in rustnx (2x to 26x). 5 stay in NetworkX with measured reasons: `all_node_cuts` and `k_components` (half their time is NetworkX's own antichain loop, whose set orders decide the cuts' order), `collapse`, `complement_edges`, `detect_unboundedness`. Done in batches 12, 13 and 25 |
| 39 | Approximation algorithms: the rest (Christofides, `traveling_salesman_problem`, `asadpour_atsp`, `approximate_k_components`, ...) | Done | NetworkX coverage | Claude | Ports round: `approximate_average_clustering` (Python's random draws replayed, about 20x), `christofides` (5x) and `traveling_salesman_problem` (4x), `asadpour_atsp` and `held_karp_ascent` (their arborescences from rustnx, about 30x), `approximate_k_components` (10x to 25x). `spanning_tree_distribution` stays in NetworkX (LAPACK determinants over float weights summed in SciPy's order). Done in batches 16 and 25 |
| 40 | Communities: the rest (`lukes_partitioning`, Louvain and Leiden) | Done | NetworkX coverage | Claude | Ports round: Louvain (item 15) and `lukes_partitioning` (its `descendants` calls in Rust, about 2.5x; replayed fully in Rust in the full-Rust round, 100x to 1,000x). Leiden stays in NetworkX (NetworkX has no implementation of its own, only backends), and `edge_current_flow_betweenness_partition` (NumPy/SciPy solves). Done in batches 15 and 25 |
| 41 | Coloring: the rest (`equitable_color`, `pad_graph`) | Blocked | NetworkX coverage | Claude | `equitable_color` depends on set order; `pad_graph` mutates its input. Done in batch 5: `is_coloring`, `is_equitable` and more `greedy_color` strategies |
| 42 | Isomorphism and graph hashing: the rest (the six mapping-returning `vf2pp_*` functions) | Done | NetworkX coverage | Claude | Full-Rust round: the VF2++ search in Rust replaying NetworkX 3.4-3.6 and 3.7's matching order and candidate set orders (`src/algorithms/vf2pp_map.rs`), so mappings come out in the same order. Multigraphs go to NetworkX. 10x to 25x. Done in batch 11: VF2 and VF2++ yes/no tests, `could_be_isomorphic` family, tree isomorphism, Weisfeiler-Lehman hashes |
| 43 | Matching, cliques, covers and independent sets: the rest (`is_edge_cover`) | Done | NetworkX coverage | Claude | `is_edge_cover` stays in NetworkX (one C-level set comparison). Done in batches 10 and 25 |
| 44 | Cycles, Euler tours and simple paths: the rest | Done | NetworkX coverage | Claude | Set-order round: `chordless_cycles` (hybrid with the chordless search in Rust, up to 10x) and `eulerize` (hybrid with the matching in rustnx, 2x to 3x); `recursive_simple_cycles` stays in NetworkX (it removes the caller's self-loops in place). Done in batches 6 and 7 (Euler tests, circuits and paths, cycle basis, simple paths, minimum cycle basis) |
| 45 | Structural tests and graph classes: the rest (`create_component_structure`) | Done | NetworkX coverage | Claude | Flows and structure round: `create_component_structure` in Rust with CPython's set orders replayed (about 20x). NumPy/SciPy round: `k_factor` (12x to 29x), `find_induced_nodes` (1.6x), `tournament_matrix` (20x to 35x); triad helpers stay in NetworkX (O(1) or subgraph copies). Done in batches 6, 8, 9, 10, 21 and 25 |
| 46 | Graph measures: the rest (`simrank_similarity`, `communicability`, random reference graphs, edit distance, polynomials, ...) | Done | NetworkX coverage | Claude | Ports round: `random_reference` and `lattice_reference` (Python's random draws replayed, 150x to 1,400x), `sigma` and `omega` (400x to 1,200x), `generate_random_paths` (NumPy `RandomState` replayed in place, 15x to 25x), `panther_similarity` (20x) and `panther_vector_similarity` (1.5x to 5x), `random_walk` (3.7+, 2x to 14x; stepped in Rust on the generator's own state in the full-Rust round, 2.5x to 50x). Edit distance, `simrank_similarity`, `non_randomness` and the polynomials stay in NetworkX with measured reasons. Done in batches 14, 15, 21 and 26 |
| 47 | Graph operations and transforms: the rest (`dedensify`, `snap_aggregation`, `spanner`) | Done | NetworkX coverage | Claude | Flows and structure round: `snap_aggregation` (each split decided in linear time, NetworkX's own set operations; 80x to 800x), `spanner` (replayed with NetworkX's id() ties, random draws and set orders; 4x to 5x). `dedensify` stays in NetworkX (most of its time is copying the graph it returns). Set-order round: `local_and_global_consistency` and `harmonic_function` as hybrids; the edge swaps stay in NetworkX. Done in batches 16, 21 and 24 |
| 48 | Linear algebra: the rest (spectra, normalized, directed and modularity matrices, ...) | Done | NetworkX coverage | Claude | NumPy/SciPy round: `normalized_laplacian_matrix`, `bethe_hessian_matrix`, `tournament_matrix` (hybrids, 3x to 35x), `attr_matrix`, `attr_sparse_matrix` (12x to 14x), `magnetic_laplacian_matrix` (3.7+, 9x to 19x), all bit-for-bit. The other 14 stay in NetworkX with measured reasons (spectra and eigensolvers are LAPACK/ARPACK-bound). Done in batch 19: `adjacency_matrix`, `incidence_matrix`, `laplacian_matrix` |
| 49 | Graph generators: the rest | Done | NetworkX coverage | Claude | Generators and I/O round: `random_k_out_graph` (NumPy replay, 3x to 5x; 70x to 150x on 3.4), `graph_atlas` (about 600x) and `graph_atlas_g`, and the unlabeled random trees with cached tree counts (5x to 160x). 37 stay in NetworkX with measured reasons (small named graphs, `LFR_benchmark_graph` (half its time is SciPy's zeta), ARPACK- and LAPACK-bound expanders, `spectral_graph_forge`, `random_kernel_graph`). Done in batches 17, 18, 22, 23 and 24 |
| 50 | Reading, writing, conversion and drawing: the rest | Done | NetworkX coverage | Claude | Ports round: `pydot_read_dot` with pydot 4.0.1's grammar in Rust (about 100x; other pydot versions go to pydot); `agraph_read_dot` and `from_agraph` stay in NetworkX (Graphviz's C parser, then pygraphviz's own accessors). Generators and I/O round: `read_graphml`/`parse_graphml` (about 8x), `read_gexf` (about 5x), the pandas converters (1.4x to 5x). 15 stay in NetworkX with measured reasons. Done in batches 19 and 20 |
| 51 | Deduplicate code added in parallel batches | Done | Engineering | Claude | One CPython set replica (`pyset.rs`) for `flow` and `measures`; `connectivity` runs `flow`'s Edmonds-Karp (with reusable scratch space) on `flow::Residual`. Speed unchanged |
| 52 | Deduplicate the graph-building helpers added in batches 17 to 21 | Done | Engineering | Claude | One module, `nxdicts.rs`: node creation, shared edge dicts and the row writer for batch 17's graph model (batch 18 converts its rows to it with `Sim::from_rows`); batches 19 to 21 use its node and edge helpers |
| 53 | Early exit in `is_regular`, `is_k_regular`, `is_eulerian` and `is_strongly_connected` | Done | Engineering | Claude | They stop at the first counter-example (strong connectivity first scans for a node without in- or out-edges); a repeat call on 262k nodes went from 3.6 ms (45 ms) to about 30 µs, level with NetworkX's dispatch overhead. Automatic dispatch leaves them, and the single queries in `interface._QUERIES`, to NetworkX until the graph is converted |
| 54 | Graph operators decline unpickled graphs | Done | Engineering | Claude | `operators::same_node`: identity, else equal and interchangeable (`same_key`); used by `OpView::read` and `conversion::check_order` |
| 55 | Slow generators: `gnc_graph`, `partial_duplication_graph` | Done | Engineering | Claude | Cause: `PairHasher::finish` returned the bare multiply, so pairs `(u, 0)` (and pairs sharing `v`) shared a bucket; it now rotates as rustc-hash does. `gnc_graph` 0.2x to 2.2x, `partial_duplication_graph` 1.7x, `complete_graph`, `barbell_graph`, `turan_graph`, `complete_multipartite_graph` 1.6x to 2.1x. `generate_random_paths` (1.4x) prepares its rows in one parallel pass |
| 56 | `minimum_spanning_tree` on very large graphs | Done | Engineering | Claude | `CoreGraph::fill_edge_subgraph` fills the tree's dicts as `add_nodes_from` / `add_edges_from` would (plain `Graph`s; subclasses keep the Python path), in one call with Kruskal and the cyclic GC paused (`_gc_paused`). 11x at 200k nodes / 1M edges; 2.9x on the road network, where allocating the tree's 3M dicts dominates (NetworkX pays the same) |

Items 28 to 50 cover every NetworkX function rustnx doesn't implement yet,
grouped by area; `docs/COVERAGE.md` (generated by `scripts/gen_coverage.py`)
lists each function and its status. They are worked through in batches of
20 functions, easiest and most useful first. Batches 1 to 26 and the
NumPy/SciPy round are done (rustnx now covers 616 of NetworkX 3.7's 797; 54
more stay in NetworkX on purpose, each with a measured reason in
`docs/COVERAGE.md`: add a function to `STAYS` in `scripts/gen_coverage.py`
only after measuring that rustnx can't make it faster).

Suggested order: 3 (announcement), then more item 28 to 47 batches (every area is now started; most leftovers depend on set order, randomness or NumPy), then 17
(remaining multigraph functions), then 26.
