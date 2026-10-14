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
