"""NumPy/SciPy functions: hybrids (NetworkX's own arithmetic on matrices
rustnx builds), attribute matrices and weighted clustering.

Hybrids must match NetworkX bit for bit, except where NetworkX itself
doesn't: functions using ARPACK (``eigs``/``eigsh``/``svds``) start from a
random vector, so NetworkX's own results vary in the last bits between
runs; those are compared with a tolerance.
"""

import inspect
import math
import random
import warnings

import networkx as nx
import numpy as np
import pytest
import scipy as sp

import rustnx
from rustnx import algorithms


def weighted(seed, n=40, m=120, directed=False, connected=True, missing=False):
    G = nx.gnm_random_graph(n, m, seed=seed, directed=directed)
    if connected:
        comps = nx.strongly_connected_components(G) if directed else nx.connected_components(G)
        G = G.subgraph(max(comps, key=len)).copy()
    # Shuffled, non-integer labels catch any node-order slip.
    G = nx.relabel_nodes(G, {v: f"n{(v * 37) % 101}" for v in G})
    rng = random.Random(seed)
    for u, v in G.edges:
        if not (missing and rng.random() < 0.2):
            G[u][v]["weight"] = rng.choice([1, 2, 0.5, 3.25])
    return G


def same(a, b):
    """Bit for bit: types, dtypes, shapes, sparse structure, values."""
    if isinstance(a, dict):
        return type(a) is type(b) and list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if sp.sparse.issparse(a):
        if type(a) is not type(b) or a.dtype != b.dtype or a.shape != b.shape:
            return False
        if a.format in ("csr", "csc"):
            return (
                np.array_equal(a.indptr, b.indptr)
                and np.array_equal(a.indices, b.indices)
                and np.array_equal(a.data, b.data, equal_nan=True)
            )
        return np.array_equal(a.toarray(), b.toarray(), equal_nan=True)
    if isinstance(a, np.ndarray):
        return (
            type(a) is type(b)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and np.array_equal(a, b, equal_nan=True)
        )
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a):
        return math.isnan(b)
    return type(a) is type(b) and a == b


def close(a, b):
    if isinstance(a, dict):
        return list(a) == list(b) and np.allclose(list(a.values()), list(b.values()), atol=1e-9)
    if isinstance(a, tuple):
        return all(close(x, y) for x, y in zip(a, b))
    if sp.sparse.issparse(a):
        a, b = a.toarray(), b.toarray()
    return np.allclose(np.asarray(a), np.asarray(b), atol=1e-9)


def both(func, *args, **kwargs):
    def run(backend):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return "ok", func(*args, backend=backend, **kwargs)
        except Exception as exc:
            return type(exc), str(exc)

    return run("rustnx"), run("networkx")


def check(func, *args, compare=same, **kwargs):
    """Compare both backends. Returns False if rustnx handed the call to
    NetworkX (``NotImplementedError``), which automatic dispatch does
    silently."""
    ours, ref = both(func, *args, **kwargs)
    if ours[0] is NotImplementedError:
        return False
    if ref[0] != "ok" or ours[0] != "ok":
        assert ours == ref
    else:
        assert compare(ours[1], ref[1]), func.__name__
    return True


EXACT = [
    (nx.normalized_laplacian_matrix, False, {}),
    (nx.normalized_laplacian_matrix, False, {"weight": None}),
    (nx.bethe_hessian_matrix, False, {}),
    (nx.bethe_hessian_matrix, False, {"r": 1.5}),
]


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("case", range(len(EXACT)))
def test_hybrids_exact(seed, missing, case):
    func, directed, kwargs = EXACT[case]
    G = weighted(seed, directed=directed, missing=missing)
    assert check(func, G, **kwargs) or missing
    if "nodelist" in inspect.signature(func).parameters:
        check(func, G, nodelist=list(G)[::-1], **kwargs)


ARPACK = [
    (nx.eigenvector_centrality_numpy, False, {}),
    (nx.eigenvector_centrality_numpy, True, {"weight": None}),
    (nx.hits, True, {}),
    (nx.hits, False, {"normalized": False}),
]


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("case", range(len(ARPACK)))
def test_hybrids_arpack(seed, case):
    func, directed, kwargs = ARPACK[case]
    G = weighted(seed, n=60, m=240, directed=directed)
    check(func, G, compare=close, **kwargs)


def test_hybrids_errors():
    # NetworkX's own checks and errors, unchanged.
    check(nx.eigenvector_centrality_numpy, nx.Graph())
    check(nx.eigenvector_centrality_numpy, nx.Graph([(0, 1), (2, 3)]))
    check(nx.normalized_laplacian_matrix, nx.Graph())
    check(nx.hits, nx.DiGraph())
    check(nx.tournament.tournament_matrix, nx.DiGraph())


@pytest.mark.parametrize("seed", range(10))
def test_tournament_matrix(seed):
    T = nx.tournament.random_tournament(30, seed=seed)
    T = nx.relabel_nodes(T, {v: f"t{(v * 7) % 31}" for v in T})
    rng = random.Random(seed)
    for u, v in T.edges:
        if seed % 2:
            T[u][v]["weight"] = rng.choice([1, 2.5])
    check(nx.tournament.tournament_matrix, T)


magnetic = pytest.mark.skipif(
    not hasattr(nx, "magnetic_laplacian_matrix"), reason="NetworkX 3.7+"
)


@magnetic
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("directed", [False, True])
def test_magnetic_laplacian_matrix(seed, directed):
    G = weighted(seed, n=40, m=150, directed=directed, connected=False, missing=True)
    rng = random.Random(seed)
    for u, v in G.edges:
        if "weight" in G[u][v]:
            G[u][v]["weight"] = rng.choice([1, 2, 0.5, 3.25, 0, -1.5])
    G.add_edge("n3", "n3", weight=2)
    nodes = list(G)

    def same_signs(a, b):
        return same(a, b) and np.array_equal(np.signbit(a.data.real), np.signbit(b.data.real)) and np.array_equal(
            np.signbit(a.data.imag), np.signbit(b.data.imag)
        )

    for kwargs in [{}, {"normalized": True}, {"q": 0.1}, {"q": 0.5, "normalized": True},
                   {"nodelist": nodes[::-1]}, {"nodelist": nodes[:20]}, {"weight": None}, {"q": 0.7}]:
        check(nx.magnetic_laplacian_matrix, G, compare=same_signs, **kwargs)


def test_hybrids_use_rustnx_matrices(monkeypatch):
    # The matrix comes from rustnx: NetworkX's builder isn't called.
    calls = []
    real = nx.utils.backends._registered_algorithms["to_scipy_sparse_array"].orig_func

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(
        nx.utils.backends._registered_algorithms["to_scipy_sparse_array"], "orig_func", spy
    )
    G = weighted(1)
    nx.normalized_laplacian_matrix(G, backend="rustnx")
    assert calls == []


# --- Attribute matrices ---------------------------------------------------------


def _same_lil(a, b):
    if type(a) is not type(b) or a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.format != "lil":
        return same(a.tocsr(), b.tocsr())
    return all(
        list(x) == list(y) and [type(v) for v in x] == [type(v) for v in y]
        for x, y in zip(a.rows, b.rows)
    ) and all(
        len(x) == len(y) and all(type(p) is type(q) and (p == q or (p != p and q != q)) for p, q in zip(x, y))
        for x, y in zip(a.data, b.data)
    )


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("directed", [False, True])
def test_attr_matrices(seed, directed):
    G = weighted(seed, directed=directed, connected=False)
    rng = random.Random(seed)
    for u, v in G.edges:
        G[u][v]["cap"] = rng.choice([1, 3, 0])
    G.add_edge("n5", "n5", weight=4, cap=2)
    order = sorted(G)
    ran = 0
    for kwargs in [
        {},
        {"edge_attr": "weight"},
        {"edge_attr": "cap"},
        {"edge_attr": "cap", "normalized": True},
        {"rc_order": order},
        {"rc_order": order, "edge_attr": "weight", "normalized": True},
        {"edge_attr": "missing"},
        {"rc_order": order[:-1]},
    ]:
        ran += check(nx.attr_matrix, G, **kwargs)
        ran += check(nx.attr_sparse_matrix, G, compare=lambda a, b: (
            _same_lil(a[0], b[0]) and a[1] == b[1] if isinstance(a, tuple) else _same_lil(a, b)
        ), **kwargs)
    assert ran >= 10  # most of these run in rustnx


# --- Weighted clustering ----------------------------------------------------------


@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("directed", [False, True])
def test_weighted_clustering(seed, directed, restore_config):
    G = weighted(seed, n=50, m=400, directed=directed, connected=False, missing=seed % 2 == 0)
    G.add_edge("n1", "n1", weight=2)
    nodes = list(G)
    for kwargs in [{}, {"nodes": nodes[::3]}, {"nodes": nodes[0]}]:
        ref = nx.clustering(G, weight="weight", backend="networkx", **kwargs)
        ours = nx.clustering(G, weight="weight", backend="rustnx", **kwargs)
        if not isinstance(ref, dict):
            ref, ours = {0: ref}, {0: ours}
        assert list(ours) == list(ref)
        for k in ref:
            assert type(ours[k]) is type(ref[k])  # int 0 where NetworkX has it
            assert ours[k] == pytest.approx(ref[k], rel=1e-12, abs=1e-15)
    for count_zeros in [True, False]:
        ref = nx.average_clustering(G, weight="weight", count_zeros=count_zeros, backend="networkx")
        ours = nx.average_clustering(G, weight="weight", count_zeros=count_zeros, backend="rustnx")
        assert ours == pytest.approx(ref, rel=1e-12)


def test_weighted_clustering_exact_floats(restore_config):
    G = weighted(3, n=50, m=300)
    nx.config.backends.rustnx.exact_floats_overrides = {"clustering": True}
    with pytest.raises(NotImplementedError) as info:
        nx.clustering(G, weight="weight", backend="rustnx")
    assert "exact_floats_overrides['clustering'] = False" in str(info.value.__cause__)
    # Unweighted clustering is exact in Rust either way.
    assert nx.clustering(G, backend="rustnx") == nx.clustering(G, backend="networkx")
    rustnx.enable()
    assert nx.clustering(G, weight="weight") == nx.clustering(G, weight="weight", backend="networkx")


def test_weighted_clustering_non_positive_max_falls_back():
    G = nx.complete_graph(4)
    for u, v in G.edges:
        G[u][v]["weight"] = -1
    with pytest.raises(NotImplementedError):
        nx.clustering(G, weight="weight", backend="rustnx")


@pytest.fixture
def restore_config():
    cfg = nx.config.backends.rustnx
    priority = nx.config.backend_priority
    saved = (cfg.exact_floats, dict(cfg.exact_floats_overrides), cfg.verbose,
             priority.algos, priority.generators, nx.config.fallback_to_nx)
    yield
    (cfg.exact_floats, cfg.exact_floats_overrides, cfg.verbose,
     priority.algos, priority.generators, nx.config.fallback_to_nx) = saved


# --- Hybrids with nested rustnx calls ----------------------------------------------


def _graph_outcome(func, *args, **kwargs):
    def run(backend):
        try:
            g = func(*args, backend=backend, **kwargs)
            if isinstance(g, nx.Graph):
                return "ok", list(g.nodes(data=True)), list(g.edges(data=True)), g.graph
            return "ok", g
        except Exception as exc:
            return type(exc), str(exc)

    return run("rustnx"), run("networkx")


@pytest.mark.parametrize("seed", range(12))
def test_k_factor(seed):
    G = nx.random_regular_graph(4, 24, seed=seed)
    G = nx.relabel_nodes(G, {v: f"v{(v * 7) % 31}" for v in G})
    rng = random.Random(seed)
    for u, v in G.edges:
        G[u][v]["weight"] = rng.randint(1, 5)
        G[u][v]["tag"] = "kept"  # non-numeric edge data is copied as NetworkX does
    for k in (1, 2, 3, 5):
        ours, ref = _graph_outcome(nx.k_factor, G, k)
        assert ours == ref


@pytest.mark.parametrize("seed", range(15))
def test_junction_tree(seed):
    G = nx.gnm_random_graph(22, 40 + seed, seed=seed, directed=seed % 3 == 0)
    G = nx.relabel_nodes(G, {v: (v * 7) % 31 for v in G})
    ours, ref = _graph_outcome(nx.junction_tree, G)
    assert ours == ref


def _chordal(n, seed):
    rng = random.Random(seed)
    G = nx.complete_graph(3)
    for v in range(3, n):
        a = rng.randrange(v)
        G.add_edge(v, a)
        for b in list(G[a])[:1]:
            if b != v:
                G.add_edge(v, b)
    return G


@pytest.mark.parametrize("seed", range(12))
def test_find_induced_nodes(seed):
    G = _chordal(50, seed)
    nodes = list(G)
    for s, t in [(nodes[0], nodes[-1]), (nodes[3], nodes[40]), (nodes[5], nodes[5])]:
        for bound in (None, 3):
            ours, ref = _graph_outcome(nx.find_induced_nodes, G, s, t, bound)
            assert ours == ref
    ours, ref = _graph_outcome(nx.find_induced_nodes, nx.cycle_graph(5), 0, 2)
    assert ours == ref  # not chordal: NetworkX's error


def test_hybrid_input_checks_still_run():
    # NetworkX's decorators (not_implemented_for, ...) run before dispatch.
    for func, args in [(nx.k_factor, (nx.DiGraph([(0, 1)]), 1)),
                       (nx.junction_tree, (nx.MultiGraph([(0, 1)]),)),
                       (nx.find_induced_nodes, (nx.DiGraph([(0, 1)]), 0, 1))]:
        ours, ref = _graph_outcome(func, *args)
        assert ours == ref or ours[0] is NotImplementedError
