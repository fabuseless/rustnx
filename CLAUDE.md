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

## Rules

- **Always display the todo list as a table.** Whenever the todo list (the
  "To do" section below) is shown, use a Markdown table with the columns
  `#`, `Item`, `Group`, `Owner` and `Notes`. Keep the numbering stable so
  items can be referred to by number, and update this file when items are
  added or finished (finished items stay in the table, marked "✅ Done" in
  `Notes`).
- Releases: bump the version in `pyproject.toml`, `Cargo.toml` (semver form,
  e.g. `0.1.0-alpha.2`) and `python/rustnx/__init__.py`, date the CHANGELOG
  section, then the owner publishes a GitHub Release tagged `vX.Y.Z` and
  approves the `pypi` deployment.

## To do

| # | Item | Group | Owner | Notes |
|---|------|-------|-------|-------|
| 1 | README section: what gets faster and what stays at NetworkX speed | Getting users | Claude | Small graphs, one-off calls, unsupported functions, multigraphs |
| 2 | GitHub issue templates (bug report, function request) | Getting users | Claude | |
| 3 | Announcement post (NetworkX Discussions, Scientific Python forum, r/Python) | Getting users | Owner | Claude can draft it |
| 4 | `bidirectional_dijkstra` | Algorithms | Claude | ✅ Done |
| 5 | `harmonic_centrality` | Algorithms | Claude | ✅ Done |
| 6 | `eigenvector_centrality` | Algorithms | Claude | ✅ Done |
| 7 | `katz_centrality` | Algorithms | Claude | ✅ Done (Katz `nstart` and per-node `beta` fall back) |
| 8 | `hits` | Algorithms | Claude | Blocked: NetworkX 3.4/3.5 use SciPy's sparse SVD, which can't be matched exactly; 3.7's power iteration could be ported for 3.7 only |
| 9 | `core_number`, `k_core` | Algorithms | Claude | ✅ Done |
| 10 | `minimum_spanning_tree`, `minimum_spanning_edges` | Algorithms | Claude | |
| 11 | Traversals: `bfs_edges`, `bfs_tree`, `dfs_preorder_nodes`, `dfs_tree` | Algorithms | Claude | Partly done: `bfs_edges`, `dfs_edges`, `dfs_preorder_nodes`; `bfs_tree`, `dfs_tree` left |
| 12 | `all_shortest_paths` | Algorithms | Claude | |
| 13 | `is_bipartite` | Algorithms | Claude | ✅ Done |
| 14 | `simple_cycles` | Algorithms | Claude | |
| 15 | `louvain_communities`, `label_propagation_communities` | Algorithms | Claude | Exact match is hard (randomness) |
| 16 | Weighted clustering | Algorithms | Claude | Bit-for-bit match is hard |
| 17 | Multigraph support (`MultiGraph`, `MultiDiGraph`) | Engineering | Claude | |
| 18 | Faster conversion from NetworkX, or docs steering to native `rustnx.Graph` | Engineering | Claude | |
| 19 | Free-threaded Python 3.14 wheels | Engineering | Claude | Optional |
| 20 | CI benchmark guard that fails on slowdowns | Release quality | Claude | |
| 21 | API docs listing every supported function and parameter | Release quality | Claude | |
| 22 | `CONTRIBUTING.md` | Release quality | Claude | |
| 23 | `SECURITY.md` | Release quality | Claude | |
| 24 | Turn on "Automatically delete head branches" | Release quality | Owner | Settings → General |
| 25 | Enable Dependabot for CI actions | Release quality | Owner | Optional |
| 26 | Release `0.1.0` (drop the alpha label) after feedback | Release quality | Owner + Claude | |

Suggested order: 1–2, 3, 20, 10, 11, then the rest.
