"""Full-Rust round: functions moved fully into Rust, results exactly
NetworkX's (in the same order)."""

import itertools
import random
import warnings

import networkx as nx
import pytest

from rustnx import algorithms


def outcome(func, *args, **kwargs):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return "ok", func(*args, **kwargs)
    except Exception as exc:
        return type(exc).__name__, str(exc)


# --- VF2++ mappings -------------------------------------------------------------

_VF2PP = [
    "vf2pp_isomorphism", "vf2pp_all_isomorphisms", "vf2pp_subgraph_isomorphism",
    "vf2pp_all_subgraph_isomorphisms", "vf2pp_monomorphism", "vf2pp_all_monomorphisms",
]


def _norm(r):
    if r is None or isinstance(r, bool):
        return r
    if isinstance(r, dict):
        return [(repr(k), repr(v)) for k, v in r.items()]
    return [_norm(x) for x in itertools.islice(r, 500)]


def _relabel(G, rnd, mode):
    if mode == 1:
        return nx.relabel_nodes(G, {v: f"n{v}" for v in G})
    if mode == 2:
        return nx.relabel_nodes(G, {v: (v, len(repr(v)) % 3) for v in G})
    if mode == 3:
        return nx.relabel_nodes(G, {v: (v * 8 if isinstance(v, int) else (v, 8)) for v in G})
    if mode == 4:
        order = list(G)
        rnd.shuffle(order)
        H = G.__class__()
        H.add_nodes_from(order)
        H.add_edges_from(G.edges)
        return H
    return G


@pytest.mark.parametrize("s", range(120))
def test_vf2pp_mappings(s):
    names = [n for n in _VF2PP if n in nx.utils.backends._registered_algorithms]
    rnd = random.Random(s)
    directed = rnd.random() < 0.35
    n = rnd.randint(1, 9)
    G = nx.gnp_random_graph(n, rnd.choice([0.2, 0.4, 0.7]), seed=s, directed=directed)
    if rnd.random() < 0.3:
        G.add_edge(0, 0)
    G = _relabel(G, rnd, rnd.randint(0, 4))
    perm = list(G)
    rnd.shuffle(perm)
    H = _relabel(nx.relabel_nodes(G, dict(zip(G, perm))), rnd, rnd.randint(0, 4))
    sub = G.subgraph(rnd.sample(list(G), max(1, n // 2))).copy()
    K = _relabel(nx.gnp_random_graph(n, 0.4, seed=s + 1, directed=directed), rnd, rnd.randint(0, 4))
    kws = [{}]
    if rnd.random() < 0.3:
        for X in (G, H, sub, K):
            for v in X:
                X.nodes[v]["c"] = hash(repr(v)) % 2 if rnd.random() < 0.5 else "a"
        kws.append({"node_label": "c", "default_label": "z"})
    for name in names:
        func = getattr(nx, name)
        for A, B in ((G, H), (G, K), (G, sub), (H, G)):
            for kw in kws:
                ours = outcome(lambda: _norm(func(A, B, backend="rustnx", **kw)))
                ref = outcome(lambda: _norm(func(A, B, backend="networkx", **kw)))
                assert ours == ref


def test_vf2pp_regular_graph():
    H = nx.random_regular_graph(4, 200, seed=1)
    P = nx.relabel_nodes(H, {v: (v * 7919) % 200 for v in H})
    ours = nx.vf2pp_isomorphism(H, P, backend="rustnx")
    assert list(ours.items()) == list(nx.vf2pp_isomorphism(H, P, backend="networkx").items())


def test_vf2pp_declines_multigraphs_and_mixed_objects():
    M = nx.MultiGraph([(0, 1), (0, 1)])
    with pytest.raises(NotImplementedError):
        nx.vf2pp_isomorphism(M, M, backend="rustnx")
    G = nx.Graph()
    G.add_node(1)
    G.add_edge(1.0, 2)
    with pytest.raises(NotImplementedError):
        nx.vf2pp_isomorphism(G, G, backend="rustnx")


def test_vf2pp_directedness_error():
    if not algorithms._vf2pp_variant():
        pytest.skip("NetworkX 3.7 raises here")
    with pytest.raises(nx.NetworkXError):
        nx.vf2pp_isomorphism(nx.path_graph(3), nx.path_graph(3, create_using=nx.DiGraph), backend="rustnx")


# --- lukes_partitioning -------------------------------------------------------------


def _lukes_tree(rnd, directed):
    n = rnd.randint(1, 30)
    T = nx.random_labeled_tree(n, seed=rnd.randrange(10**9)) if hasattr(nx, "random_labeled_tree") \
        else nx.random_tree(n, seed=rnd.randrange(10**9))
    r = rnd.random()
    if r < 0.2:
        T = nx.relabel_nodes(T, {v: (f"s{v}" if v % 2 else (v, v)) for v in T})
    elif r < 0.35:
        T = nx.relabel_nodes(T, {v: v * 8 for v in T})
    elif r < 0.5:
        order, edges = list(T), list(T.edges)
        rnd.shuffle(order)
        rnd.shuffle(edges)
        T = nx.Graph()
        T.add_nodes_from(order)
        T.add_edges_from(edges)
    if directed:
        T = nx.bfs_tree(T, rnd.choice(list(T)))
    kind = rnd.randrange(5)
    for u, v in T.edges:
        T.edges[u, v]["w"] = [
            lambda: rnd.randint(-3, 9),
            lambda: rnd.random() * 10,
            lambda: rnd.choice([rnd.random(), rnd.randint(1, 5)]),
            # Large and small floats: 3.12+'s compensated sum() differs here.
            lambda: rnd.choice([rnd.random() * 1e16, rnd.random(), -rnd.random() * 1e15, 1]),
            lambda: rnd.choice([0.1, 0.2, 0.3, 1, True]),
        ][kind]()
    for v in T:
        T.nodes[v]["nw"] = rnd.choice([1, 1, 2, 3, 0, True])
    if rnd.random() < 0.1:
        T.nodes[rnd.choice(list(T))]["nw"] = -1
    return T


@pytest.mark.parametrize("s", range(150))
def test_lukes_partitioning(s, monkeypatch):
    rnd = random.Random(s)
    G = _lukes_tree(rnd, rnd.random() < 0.3)
    declined = []
    run = algorithms._hybrid.run
    monkeypatch.setattr(algorithms._hybrid, "run", lambda *a: declined.append(1) or run(*a))
    for kw in (
        {"max_size": rnd.randint(0, 6)},
        {"max_size": rnd.randint(1, 8), "node_weight": "nw", "edge_weight": "w"},
        {"max_size": rnd.randint(1, 5), "edge_weight": "w"},
        {"max_size": rnd.randint(1, 5), "node_weight": "nw"},
        {"max_size": 2.5},
    ):
        results = []
        for backend in ("rustnx", "networkx"):
            random.seed(s)
            out = outcome(nx.community.lukes_partitioning, G, backend=backend, **kw)
            if out[0] == "ok" and out[1] is not None:
                # Each set's own iteration order counts.
                out = "ok", [list(c) for c in out[1]]
            results.append((out, random.random()))
        assert results[0] == results[1]
    # Only the float max_size (and a lone node) goes to the hybrid.
    assert len(declined) == (5 if len(G) < 2 else 1)


def test_lukes_sum_mode():
    # sum()'s float mode is recognized on every supported interpreter.
    assert algorithms._sum_mode() is not None


# --- random_walk stepped in Rust ---------------------------------------------------


def _walk(G, backend, n, rng, between, **kw):
    out = []
    try:
        for i, x in enumerate(itertools.islice(nx.random_walk(G, backend=backend, seed=rng, **kw), n)):
            # The caller drawing from the same generator between steps.
            out.append((x, rng.random() if between and i % 3 == 0 else None))
    except Exception as exc:
        out.append((type(exc).__name__, str(exc)))
    return out, rng.getstate()


@pytest.mark.parametrize("s", range(80))
def test_random_walk_in_rust(s):
    from fractions import Fraction

    if not algorithms._random_walk_recognized():
        pytest.skip("this NetworkX has no random_walk rustnx follows")
    assert algorithms._mt_layout() is not None
    rnd = random.Random(s)
    n = rnd.randint(1, 12)
    G = nx.gnp_random_graph(n, rnd.random() * 0.6, seed=s, directed=rnd.random() < 0.5)
    for u, v in G.edges:
        r = rnd.random()
        G.edges[u, v]["w"] = 0 if r < 0.2 else (-1 if r < 0.23 else (rnd.randint(1, 4) if r < 0.5 else rnd.random()))
        if s % 7 == 1:
            # Odd weights: infinities, bools, huge ints, other numeric types.
            G.edges[u, v]["w"] = rnd.choice([float("inf"), True, False, 2**60, 0.0, float("nan"), Fraction(1, 3), -0.0])
    if s % 5 == 2:
        G = nx.relabel_nodes(G, {v: (f"n{v}" if v % 2 else (v, "t")) for v in G})
    start = next(iter(G))
    for kw in ({}, {"weight": "w"}, {"weight": "missing"}):
        for between in (False, True):
            ours = _walk(G, "rustnx", 200, random.Random(s), between, start=start, **kw)
            ref = _walk(G, "networkx", 200, random.Random(s), between, start=start, **kw)
            assert ours == ref
