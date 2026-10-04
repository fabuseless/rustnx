# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

rustnx is a Rust-accelerated backend for NetworkX (PyO3 + maturin), plus a
rustworkx-compatible API (`rustnx.rx`). Every function must return exactly
what the installed NetworkX returns (same values, dict order, ties and
errors) on NetworkX 3.4, 3.5 and 3.7; anything rustnx can't match raises
`NotImplementedError` so NetworkX falls back to its own code.

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
| 8 | `hits` | Blocked | Algorithms | Claude | NetworkX 3.4 to 3.6 use SciPy's sparse SVD; 3.7's power iteration runs through SciPy sparse kernels and NumPy pairwise sums, so an exact match would depend on SciPy internals |
| 9 | `core_number`, `k_core` | Done | Algorithms | Claude | |
| 10 | `minimum_spanning_tree`, `minimum_spanning_edges` | Done | Algorithms | Claude | Kruskal, minimum and maximum. Prim and Borůvka run in Rust too since batch 8 (rustnx replays NetworkX's set operations for Prim's start nodes and Borůvka's ties) |
| 11 | Traversals: `bfs_edges`, `bfs_tree`, `dfs_preorder_nodes`, `dfs_tree` | Done | Algorithms | Claude | Also `dfs_edges` |
| 12 | `all_shortest_paths` | Done | Algorithms | Claude | Bellman-Ford falls back |
| 13 | `is_bipartite` | Done | Algorithms | Claude | |
| 14 | `simple_cycles` | Blocked | Algorithms | Claude | Picks each component's start with `next(iter(set))` while mutating subgraphs, so cycle order depends on Python's set layout; 3.5 also differs |
| 15 | `louvain_communities`, `label_propagation_communities` | Partly done | Algorithms | Claude | Done: `label_propagation_communities` (plus `greedy_color`). Louvain blocked: internals differ in all three versions and depend on set order and float sums |
| 16 | Weighted clustering | Blocked | Algorithms | Claude | Sums NumPy `cbrt` arrays (pairwise summation) over set intersections in hash order |
| 17 | Multigraph support (`MultiGraph`, `MultiDiGraph`) | Partly done | Engineering | Claude | Done: components, traversals, shortest paths, betweenness/closeness/harmonic, distance measures, `is_bipartite` (`interface.MULTIGRAPH_FUNCTIONS`). Left: `pagerank` (sums parallel weights), degree-based functions, MST with keys |
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
| 29 | Centrality: the rest (current flow, subgraph and Estrada, trophic, Laplacian, second order, communicability, NumPy variants) | Partly done | NetworkX coverage | Claude | 19 left. Done: degree (batch 1); subset, load, percolation, group, reaching, VoteRank, dispersion (batch 4). Most of the rest use LAPACK/SciPy, so can't match bit for bit; `incremental_closeness_centrality` mutates its input |
| 30 | Components | Done | NetworkX coverage | Claude | Batches 1 and 3 |
| 31 | Directed acyclic graphs | Done | NetworkX coverage | Claude | Batches 3 and 7. `colliders` stays in NetworkX: its C-level loop beats building the tuples from Rust (measured twice) |
| 32 | Traversal | Done | NetworkX coverage | Claude | Batches 1 and 3. `bfs_beam_edges` stays in NetworkX: it calls a Python callable per neighbor, so nothing to speed up |
| 33 | Distance measures: the rest (resistance distance, effective graph resistance, Kemeny constant) | Partly done | NetworkX coverage | Claude | 3 left, all NumPy/LAPACK, so can't match bit for bit. Done in batch 5: centroid/barycenter, harmonic diameter, distance-regular tests |
| 34 | Clustering and cores | Done | NetworkX coverage | Claude | Batch 5; weighted clustering is item 16 |
| 35 | Trees, branchings and arborescences: the rest (`join_trees`, `junction_tree`, `number_of_spanning_trees`, `random_spanning_tree`) | Partly done | NetworkX coverage | Claude | 4 left. Done in batches 1, 6 and 8 (branchings and arborescences, Prim and Boruvka, Prufer and nested tuples, tree centroid). The rest: NumPy determinant, set order, or plain graph copying |
| 36 | Link analysis: `google_matrix` | Blocked | NetworkX coverage | Claude | Returns a NumPy matrix built with NumPy arithmetic; `hits` is item 8 |
| 37 | Bipartite graphs: the rest (projections, clustering, generators, matrices, `minimum_weight_full_matching`, ...) | Partly done | NetworkX coverage | Claude | 31 left. Done in batch 11: `color`, `sets`, `is_bipartite_node_set`, Hopcroft-Karp, vertex cover, closeness, redundancy, butterflies. Projections and clustering depend on set order; generators and matrices gain little |
| 38 | Connectivity, flows and cuts: the rest (`capacity_scaling`, `all_node_cuts`, `k_components`, k-edge components for k >= 3, weighted augmentation) | Partly done | NetworkX coverage | Claude | 13 left. Done in batches 12 and 13: all maximum flow algorithms, minimum cut, residual networks, Gomory-Hu, network simplex and min-cost flow, cut measures, node and edge connectivity, minimum cuts, disjoint paths, Stoer-Wagner, bridge components, 1- and 2-edge augmentation. Most of the rest depend on set order |
| 39 | Approximation algorithms: the rest (`max_clique`, `maximum_independent_set`, Christofides, `traveling_salesman_problem`, `treewidth_min_degree`, approximate connectivity, ...) | Partly done | NetworkX coverage | Claude | 17 left. Done in batch 16: vertex cover, dominating sets, maximal matching, greedy and annealing TSP, min fill-in treewidth, diameter, max cut, Steiner tree, densest subgraph. The rest depend on set order or SciPy, or need batch 13's connectivity code |
| 40 | Communities: the rest (`k_clique_communities`, `lukes_partitioning`, `kernighan_lin_bisection`, Louvain and Leiden) | Partly done | NetworkX coverage | Claude | 8 left. Done in batch 15: modularity and partition measures, greedy modularity, Girvan-Newman, label propagation (seeded, with Python's random draws replayed), `asyn_fluidc`. `kernighan_lin_bisection` is feasible; Louvain is item 15 |
| 41 | Coloring: the rest (`equitable_color`, `pad_graph`) | Blocked | NetworkX coverage | Claude | `equitable_color` depends on set order; `pad_graph` mutates its input. Done in batch 5: `is_coloring`, `is_equitable` and more `greedy_color` strategies |
| 42 | Isomorphism and graph hashing: the rest (the six mapping-returning `vf2pp_*` functions) | Blocked | NetworkX coverage | Claude | Their mappings follow set order (`_matching_order`, candidate sets). Done in batch 11: VF2 and VF2++ yes/no tests, `could_be_isomorphic` family, tree isomorphism, Weisfeiler-Lehman hashes |
| 43 | Matching, cliques, covers and independent sets: the rest (`find_cliques`, `dominating_set`, `maximal_independent_set`, clique graphs, `is_edge_cover`) | Partly done | NetworkX coverage | Claude | 7 left, depending on set order or randomness (or no gain for `is_edge_cover`). Done in batch 10: matchings (blossom port), edge cover, dominating set tests, `enumerate_all_cliques`, `node_clique_number`, `max_weight_clique` |
| 44 | Cycles, Euler tours and simple paths: the rest (`chordless_cycles`, `recursive_simple_cycles`, `eulerize`) | Blocked | NetworkX coverage | Claude | Set order, input mutation, or blossom internals; `simple_cycles` is item 14. Done in batches 6 and 7 (Euler tests, circuits and paths, cycle basis, simple paths, minimum cycle basis) |
| 45 | Structural tests and graph classes: the rest (`k_factor`, `moral_graph`, chordal cliques, asteroidal triples, Hamiltonian path, triads by type, ...) | Partly done | NetworkX coverage | Claude | 14 left. Done in batches 6, 8, 9 and 10 (bridges, isolates, dominators, planarity with embeddings, chordal tests, LCAs, d-separation, triadic census, degree sequences, boundaries). `k_factor` could now reuse batch 10's blossom port |
| 46 | Graph measures: the rest (structural holes, `simrank_similarity`, `communicability`, random reference graphs, edit distance, polynomials, ...) | Partly done | NetworkX coverage | Claude | 26 left. Done in batches 14 and 15: assortativity and mixing, link prediction, reciprocity, rich club, `s_metric`, `number_of_walks`, efficiency, vitality, Wiener-type indices, `flow_hierarchy`, `voronoi_cells`. Structural holes are feasible for explicit `nodes`; most others use NumPy/LAPACK or random graphs |
| 47 | Graph operations and transforms: the rest (products, `compose`, `union`, contraction, quotient graphs, swaps, ...) | Partly done | NetworkX coverage | Claude | 28 left. Done in batch 16: `complement`, `power`, `difference`, `symmetric_difference`, `is_kl_connected`, `kl_connected_subgraph`. Most of the rest spend their time building the NetworkX result, so little to gain |
| 48 | Linear algebra (Laplacian, adjacency and incidence matrices, spectra) | To do | NetworkX coverage | Claude | 22 left; returns SciPy/NumPy objects, so little to gain |
| 49 | Graph generators | To do | NetworkX coverage | Claude | 147 left; they build NetworkX graphs, so little to gain |
| 50 | Reading, writing, conversion and drawing | To do | NetworkX coverage | Claude | 58 left; they move graphs in and out of NetworkX, so little to gain |
| 51 | Deduplicate code added in parallel batches | To do | Engineering | Claude | Batches 12 (`flow::PySet`) and 14 (`measures::PySet`) each ported CPython's set table; batch 13 has its own Edmonds-Karp in `connectivity.rs` beside batch 12's `flow.rs`. Both copies are tested; merge each into one |

Items 28 to 50 cover every NetworkX function rustnx doesn't implement yet,
grouped by area; `docs/COVERAGE.md` (generated by `scripts/gen_coverage.py`)
lists each function and its status. They are worked through in batches of
20 functions, easiest and most useful first. Batches 1 to 16 are done (318
functions; rustnx now covers 383 of NetworkX 3.7's 797).

Suggested order: 3 (announcement), then more item 28 to 47 batches (every area is now started; most leftovers depend on set order, randomness or NumPy), then 17
(remaining multigraph functions), then 26.
