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
  section, then the owner publishes a GitHub Release tagged `vX.Y.Z` and
  approves the `pypi` deployment.

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
| 10 | `minimum_spanning_tree`, `minimum_spanning_edges` | Done | Algorithms | Claude | Kruskal, minimum and maximum; Prim starts from `set(G).pop()` (hash order) and Borůvka fall back |
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

Suggested order: a `0.1.0a3` pre-release of the unreleased work, then 3 (announcement), then 17 (remaining multigraph functions), then 26.
