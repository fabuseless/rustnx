# rustnx

**NetworkX, but fast.** rustnx is a Rust-powered backend for
[NetworkX](https://networkx.org). You keep writing normal NetworkX code, and
supported algorithms run in Rust instead of Python, often 50–100× faster.

```python
import networkx as nx

nx.config.backend_priority = ["rustnx"]   # or: NETWORKX_BACKEND_PRIORITY=rustnx

G = nx.barabasi_albert_graph(4000, 4, seed=1)
nx.betweenness_centrality(G)              # runs in Rust: 0.4s instead of 41s
```

Nothing else changes. Anything rustnx doesn't support, such as other
functions, multigraphs or callable weights, keeps running in NetworkX, so
turning it on never breaks working code.

## Supported algorithms (v0.1)

| Function | Notes |
|---|---|
| `betweenness_centrality` | Unweighted and weighted, `normalized`, `endpoints`, and `k` sampling (picks the same nodes as NetworkX for a given `seed`). Parallel. |
| `closeness_centrality` | Unweighted and `distance=`, `wf_improved`, single node `u=`. Parallel. Results are **bit-for-bit identical** to NetworkX. |
| `single_source_shortest_path_length` | Same nodes and same dict order as NetworkX, with `cutoff`. |
| `single_source_dijkstra_path_length` | Same order as NetworkX, with `cutoff`. Integer weights give integer distances. Raises the same error on negative cycles. |
| `connected_components`, `number_connected_components`, `is_connected` | Components come out in the same order as NetworkX. |

## Benchmarks

`python benchmarks/bench.py` on a 4-core machine (NetworkX 3.6.1):

| Function | Graph | NetworkX | rustnx | Speedup |
|---|---|---|---|---|
| `betweenness_centrality` | 4,000 nodes / 16k edges | 40.9 s | 0.38 s | **107×** |
| `betweenness_centrality` (weighted) | 4,000 / 16k | 115.3 s | 1.24 s | **93×** |
| `closeness_centrality` | 4,000 / 16k | 6.0 s | 0.11 s | **54×** |
| `single_source_dijkstra_path_length` | 160,000 / 319k | 0.38 s | 0.063 s | 6× |
| `single_source_shortest_path_length` | 160,000 / 319k | 0.11 s | 0.020 s | 5× |
| `connected_components` | 200,000 / 300k | 0.23 s | 0.040 s | 6× |

The rustnx column is a repeat call. The first call on a graph also converts
it to rustnx's format, and NetworkX caches that conversion on the graph. For
the heavy algorithms conversion is negligible. For linear-time ones the first
call is roughly break-even.

On small graphs (under 500 nodes), the linear-time functions stay in
NetworkX automatically, because the dispatch overhead outweighs the work.
The centrality functions are faster in Rust at every size.

## Correctness

Results must match NetworkX, or the speed is worthless. Two test layers
check that:

1. **`tests/`**: about 1,400 randomized comparisons against NetworkX, run on
   directed and undirected graphs, int, float and missing weights,
   self-loops, and shuffled or non-integer node labels. They check values,
   dict ordering and error messages.
2. **NetworkX's own test suite**, with every supported call routed through
   rustnx:

   ```bash
   NETWORKX_TEST_BACKEND=rustnx NETWORKX_FALLBACK_TO_NX=True \
       pytest --pyargs networkx
   ```

   Result: 0 failures on NetworkX 3.4.2, 3.5 and 3.7 (9,066 tests passed on
   3.7), with over 300,000 calls handled by rustnx. CI runs both layers on
   NetworkX 3.4, 3.5 and the latest release.

rustnx checks every call against the installed NetworkX's own signature.
If a newer NetworkX adds a parameter, rustnx ignores it while it is left at
its default. If the caller actually uses it, rustnx hands the call back to
NetworkX.

Betweenness sums per-source contributions in parallel. It matches NetworkX
to about 1e-15 relative error rather than bit-for-bit, and it gives the same
result on any machine regardless of thread count.

## How it works

```
nx.betweenness_centrality(G)
        │  NetworkX dispatch (backend_priority = ["rustnx"])
        ▼
rustnx.interface      convert G once: nodes → 0..n-1, adjacency → CSR arrays
        │             (NetworkX caches this on G for later calls)
        ▼
rustnx._core (Rust)   algorithm on flat arrays, GIL released, parallel via rayon
        │
        ▼
dict keyed by your original nodes, in NetworkX's order
```

Neighbor order in the Rust arrays follows NetworkX's adjacency dicts exactly.
That's why traversal order, tie-breaking and result ordering all match.

## Releasing

The `Wheels` workflow builds packages for Linux (x86-64, ARM), macOS (Intel,
Apple Silicon) and Windows, plus a source package. It then installs each
wheel on that platform and runs the test suite against it. The workflow runs
on every pull request.

To publish a release:

1. One-time setup: on PyPI, add a
   [trusted publisher](https://docs.pypi.org/trusted-publishers/) for
   `fabuseless/rustnx`, workflow `wheels.yml`, environment `pypi`. Then create
   an environment named `pypi` in the repo's GitHub settings. No API token is
   needed.
2. Set the version in `pyproject.toml` and `Cargo.toml`, and move the
   changelog's *Unreleased* entries under the new version.
3. Push a tag such as `v0.1.0a1`. If every build and test passes, the
   workflow uploads the packages to PyPI.

## Development

```bash
uv venv && source .venv/bin/activate
uv pip install maturin networkx pytest numpy scipy
maturin develop --release
pytest                                  # rustnx comparison tests
python benchmarks/bench.py --quick      # benchmarks
```

Layout:

- `src/`: Rust core. `graph.rs` handles conversion and CSR storage;
  `algorithms/` holds the traversals and centrality.
- `python/rustnx/`: the NetworkX backend. `interface.py` is the entry point;
  `algorithms.py` holds the NetworkX-compatible wrappers.
- `tests/`: comparisons against NetworkX.
- `benchmarks/`: speed comparisons.

## Roadmap

- More algorithms: `pagerank`, all-pairs shortest paths, `shortest_path`
  (actual paths), `edge_betweenness_centrality`, `topological_sort`, strongly
  and weakly connected components, `clustering`/`triangles`.
- A native `rustnx.Graph` that lives in Rust, so there's no conversion at all.
- Multigraph support.
- A rustworkx-compatible API over the same core.

## License

BSD 3-Clause, the same license as NetworkX. See [LICENSE](LICENSE). Release
notes are in [CHANGELOG.md](CHANGELOG.md).
