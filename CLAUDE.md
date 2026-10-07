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
| 15 | `louvain_communities`, `label_propagation_communities` | Partly done | Algorithms | Claude | Done: `label_propagation_communities` (plus `greedy_color`). Louvain blocked: internals differ in all three versions and depend on set order and float sums |
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
| 35 | Trees, branchings and arborescences: the rest (`random_spanning_tree`) | Partly done | NetworkX coverage | Claude | 1 left: it picks edges by comparing Python's random numbers with ratios of LAPACK determinants, so a Rust version would pick different trees. NumPy/SciPy round: `junction_tree` (NetworkX's code with its chordal completion, cliques and spanning tree run in rustnx, about 35x); `number_of_spanning_trees` stays in NetworkX (LAPACK). Done in batches 1, 6, 8 and 24 |
| 36 | Link analysis: `google_matrix` | Done | NetworkX coverage | Claude | NumPy/SciPy round: `hits` done (item 8); `google_matrix` stays in NetworkX (dense NumPy arithmetic, measured 1.1x) |
| 37 | Bipartite graphs: the rest (`birank`, `spectral_bipartivity`, `minimum_weight_full_matching`, `degrees`) | Done | NetworkX coverage | Claude | NumPy/SciPy round: all 4 stay in NetworkX (SciPy-bound, measured about 1x; `degrees` returns NetworkX's views). Done in batches 11, 18 to 23 and 26 |
| 38 | Connectivity, flows and cuts: the rest (`capacity_scaling`, `all_node_cuts`, `k_components`, k-edge components for k >= 3, weighted augmentation) | Partly done | NetworkX coverage | Claude | 11 left. Done in batches 12, 13 and 25 (auxiliary connectivity graphs). Most of the rest depend deeply on set order (antichains, frozenset-keyed structures) |
| 39 | Approximation algorithms: the rest (Christofides, `traveling_salesman_problem`, `asadpour_atsp`, `approximate_k_components`, ...) | Partly done | NetworkX coverage | Claude | 7 left. Done in batches 16 and 25 (Ramsey-based clique and independent set approximations, `treewidth_min_degree`, approximate node connectivity, `metric_closure`). The rest use SciPy linear programming or blossom internals, or depend on set order |
| 40 | Communities: the rest (`lukes_partitioning`, Louvain and Leiden) | Partly done | NetworkX coverage | Claude | 5 left. Done in batches 15 and 25 (`k_clique_communities`, `kernighan_lin_bisection`). Louvain is item 15 |
| 41 | Coloring: the rest (`equitable_color`, `pad_graph`) | Blocked | NetworkX coverage | Claude | `equitable_color` depends on set order; `pad_graph` mutates its input. Done in batch 5: `is_coloring`, `is_equitable` and more `greedy_color` strategies |
| 42 | Isomorphism and graph hashing: the rest (the six mapping-returning `vf2pp_*` functions) | Blocked | NetworkX coverage | Claude | Their mappings follow set order (`_matching_order`, candidate sets). Done in batch 11: VF2 and VF2++ yes/no tests, `could_be_isomorphic` family, tree isomorphism, Weisfeiler-Lehman hashes |
| 43 | Matching, cliques, covers and independent sets: the rest (`is_edge_cover`) | Done | NetworkX coverage | Claude | `is_edge_cover` stays in NetworkX (one C-level set comparison). Done in batches 10 and 25 |
| 44 | Cycles, Euler tours and simple paths: the rest | Done | NetworkX coverage | Claude | Set-order round: `chordless_cycles` (hybrid with the chordless search in Rust, up to 10x) and `eulerize` (hybrid with the matching in rustnx, 2x to 3x); `recursive_simple_cycles` stays in NetworkX (it removes the caller's self-loops in place). Done in batches 6 and 7 (Euler tests, circuits and paths, cycle basis, simple paths, minimum cycle basis) |
| 45 | Structural tests and graph classes: the rest (`create_component_structure`) | Partly done | NetworkX coverage | Claude | 1 left: per-node components of subgraph views keyed in set order; a hybrid measured 0.3x, so it needs a full Rust port with CPython's set order. NumPy/SciPy round: `k_factor` (12x to 29x), `find_induced_nodes` (1.6x), `tournament_matrix` (20x to 35x); triad helpers stay in NetworkX (O(1) or subgraph copies). Done in batches 6, 8, 9, 10, 21 and 25 |
| 46 | Graph measures: the rest (`simrank_similarity`, `communicability`, random reference graphs, edit distance, polynomials, ...) | Partly done | NetworkX coverage | Claude | 20 left. Done in batches 14, 15, 21 and 26 (`cd_index`). `mutual_weight` and `normalized_mutual_weight` are O(degree), so dispatch makes them slower; most others use NumPy/LAPACK or random graphs |
| 47 | Graph operations and transforms: the rest (`dedensify`, `snap_aggregation`, `spanner`) | Partly done | NetworkX coverage | Claude | 3 left: names and random draws taken in frozenset and set order. Set-order round: `local_and_global_consistency` (3x to 5x) and `harmonic_function` (1.2x) as hybrids; the edge swaps stay in NetworkX (they change the caller's graph in place). Done in batches 16, 21 and 24 (contraction, `quotient_graph`, `modular_product`, `stochastic_graph`, `mycielskian`, `inverse_line_graph`) |
| 48 | Linear algebra: the rest (spectra, normalized, directed and modularity matrices, ...) | Done | NetworkX coverage | Claude | NumPy/SciPy round: `normalized_laplacian_matrix`, `bethe_hessian_matrix`, `tournament_matrix` (hybrids, 3x to 35x), `attr_matrix`, `attr_sparse_matrix` (12x to 14x), `magnetic_laplacian_matrix` (3.7+, 9x to 19x), all bit-for-bit. The other 14 stay in NetworkX with measured reasons (spectra and eigensolvers are LAPACK/ARPACK-bound). Done in batch 19: `adjacency_matrix`, `incidence_matrix`, `laplacian_matrix` |
| 49 | Graph generators: the rest | Done | NetworkX coverage | Claude | Generators and I/O round: `random_k_out_graph` (NumPy replay, 3x to 5x; 70x to 150x on 3.4), `graph_atlas` (about 600x) and `graph_atlas_g`, and the unlabeled random trees with cached tree counts (5x to 160x). 37 stay in NetworkX with measured reasons (small named graphs, `LFR_benchmark_graph` (half its time is SciPy's zeta), ARPACK- and LAPACK-bound expanders, `spectral_graph_forge`, `random_kernel_graph`). Done in batches 17, 18, 22, 23 and 24 |
| 50 | Reading, writing, conversion and drawing: the rest | Partly done | NetworkX coverage | Claude | 3 left: `pydot_read_dot` (99.5% of its time is pydot's own parser, so it needs a DOT parser in Rust that matches pydot's), `agraph_read_dot` and `from_agraph` (pygraphviz isn't installable here, so not measured). Generators and I/O round: `read_graphml`/`parse_graphml` (about 8x) and `read_gexf` (about 5x) with an XML parser in Rust, `from_pandas_edgelist` (1.4x to 2.6x), `to_pandas_edgelist` (2.4x to 5x), `from_pandas_adjacency` (2x to 3x). 13 stay in NetworkX with measured reasons (in-place attribute setters, graph constructors, `forceatlas2_layout`, `to_pandas_adjacency`, `from_pydot`, `is_empty`, `to_edgelist`). Done in batches 19 and 20 |
| 51 | Deduplicate code added in parallel batches | Done | Engineering | Claude | One CPython set replica (`pyset.rs`) for `flow` and `measures`; `connectivity` runs `flow`'s Edmonds-Karp (with reusable scratch space) on `flow::Residual`. Speed unchanged |
| 52 | Deduplicate the graph-building helpers added in batches 17 to 21 | Done | Engineering | Claude | One module, `nxdicts.rs`: node creation, shared edge dicts and the row writer for batch 17's graph model (batch 18 converts its rows to it with `Sim::from_rows`); batches 19 to 21 use its node and edge helpers |

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
