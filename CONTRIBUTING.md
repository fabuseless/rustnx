# Contributing to rustnx

Thanks for helping. rustnx is a NetworkX backend: NetworkX hands supported
calls to Rust and runs everything else itself. This guide covers setting up,
the one rule every change must follow, and how to add an algorithm.

## The rule: match NetworkX exactly

A rustnx function must return exactly what the installed NetworkX returns for
the same call: the same values, the same types (`3` is not `3.0`), the same
dict and list order, the same tie-breaking, and the same exceptions with the
same messages.

When rustnx can't match NetworkX for some input (an unsupported parameter, a
callable weight, a version difference it can't reproduce), it must raise
`NotImplementedError`. NetworkX then runs its own implementation, so the user
still gets the right answer, just not faster.

rustnx supports NetworkX 3.4 and newer, and CI tests 3.4, 3.5 and the latest
release. Several NetworkX functions changed between these versions; read each
version's source before porting, and detect differences at runtime (see
`_dijkstra_paths_in_pop_order` in `python/rustnx/algorithms.py` for an
example).

## Setup

You need Rust (stable, via [rustup](https://rustup.rs)) and Python 3.10 or
newer.

```bash
git clone https://github.com/fabuseless/rustnx
cd rustnx
uv venv && source .venv/bin/activate      # or python -m venv .venv
uv pip install maturin networkx pytest numpy scipy rustworkx
maturin develop --release                 # rebuild after every Rust change
```

## Checks to run before a pull request

```bash
cargo fmt --check
cargo clippy --release -- -D warnings
pytest                                    # comparisons against NetworkX

# NetworkX's own test suite, with supported calls routed through rustnx:
cd /tmp && NETWORKX_TEST_BACKEND=rustnx NETWORKX_FALLBACK_TO_NX=True \
    python -m pytest -q -p no:cacheprovider --pyargs networkx
```

Both test layers must pass with no failures. CI runs them on NetworkX 3.4,
3.5 and the latest release, and on free-threaded Python 3.14t. If you can,
run the comparison tests against the oldest and newest supported NetworkX
locally too, since that is where most mismatches show up.

## Adding an algorithm

1. **Read NetworkX's source** for the function in every supported version,
   including the helpers it calls. Note how it orders results, breaks ties,
   sums floats and raises errors.
2. **Write the Rust part** in `src/algorithms/`, and expose it as a method on
   `CoreGraph` in `src/lib.rs`. Work on node positions `0..n-1`; neighbor
   order in the CSR arrays matches NetworkX's adjacency dicts. Release the GIL
   (`py.detach`) around long computations.
3. **Write the Python wrapper** in `python/rustnx/algorithms.py`, with the same
   signature as the NetworkX function. It validates inputs, raises
   NetworkX's exceptions, maps positions back to nodes and raises
   `NotImplementedError` for anything it can't match.
4. **Register it** in `algorithms.__all__` and `_info.FUNCTIONS` (a test checks
   that they match). If it is cheap enough that NetworkX is faster on small
   graphs, add it to `_LINEAR_TIME` in `interface.py`. If it supports
   multigraphs, add it to `MULTIGRAPH_FUNCTIONS`.
5. **Test it** in `tests/test_against_networkx.py` against NetworkX on random
   graphs: directed and undirected, unweighted, int and float weights, missing
   weights, self-loops, non-integer labels, missing nodes and error cases.
   Compare outputs exactly, including order and types; the `exact_outcome`
   helper does this. Add a native-graph case in `tests/test_native.py`.
6. **Document it** in the README's supported algorithms table, add a benchmark
   row if it is notably faster, and add a CHANGELOG entry under
   "Unreleased".

## Pull requests

- Keep each pull request focused, and describe what changed and how you
  tested it.
- Add a CHANGELOG entry under "Unreleased" for anything users would notice.
- Don't change the version number; releases are cut separately.
- By contributing, you agree that your contribution is licensed under the
  project's BSD 3-Clause license.

## Reporting bugs and requesting functions

Open an issue. For a wrong result, include a small graph that reproduces it,
the NetworkX and rustnx versions, and what NetworkX returns. For security
problems, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.
