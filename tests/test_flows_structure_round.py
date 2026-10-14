"""Flows and structure round: functions from rows 38, 45 and 47. Results
must be exactly NetworkX's (in the same order where NetworkX's order is
deterministic)."""

import random
import warnings

import networkx as nx
import pytest
from networkx.algorithms.connectivity import edge_augmentation as ea
from networkx.algorithms.connectivity import edge_kcomponents as ek


def typed(x):
    if isinstance(x, dict):
        return ("dict", [(typed(k), typed(v)) for k, v in x.items()])
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, [typed(v) for v in x])
    if isinstance(x, (set, frozenset)):
        return (type(x).__name__, [typed(v) for v in x])
    return (type(x).__name__, repr(x))


def graph_snapshot(G):
    kw = {"keys": True} if G.is_multigraph() else {}
    return (
        type(G).__name__,
        typed(list(G.nodes(data=True))),
        typed(list(G.edges(data=True, **kw))),
        [(repr(u), [repr(v) for v in G._adj[u]]) for u in G],
    )


def outcome(func, *args, **kwargs):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return "ok", func(*args, **kwargs)
    except Exception as exc:
        return type(exc).__name__, str(exc)


# --- snap_aggregation -------------------------------------------------------------


@pytest.mark.parametrize("s", range(60))
def test_snap_aggregation(s):
    rnd = random.Random(s)
    n = rnd.randint(0, 40)
    cls = rnd.choice([nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
    G = cls(nx.gnp_random_graph(n, rnd.random() * 0.3, seed=s, directed=cls().is_directed()))
    if rnd.random() < 0.3:
        G = nx.relabel_nodes(G, {v: (f"s{v}" if v % 3 else (v, v)) for v in G})
    if G.is_multigraph():
        for u, v in list(G.edges())[:5]:
            G.add_edge(u, v)
    for v in G:
        G.nodes[v]["c"] = rnd.choice("ab")
        G.nodes[v]["d"] = rnd.choice([1, 2])
    kw = {"keys": True} if G.is_multigraph() else {}
    for e in G.edges(**kw):
        G.edges[e]["t"] = rnd.choice("xyz")
        G.edges[e]["u"] = rnd.choice([0, 1])
    for args in (
        {"node_attributes": ("c",)},
        {"node_attributes": ("c", "d"), "edge_attributes": ("t",)},
        {"node_attributes": ("c",), "edge_attributes": ("t", "u"), "prefix": "S",
         "supernode_attribute": "g", "superedge_attribute": "ty"},
        {"node_attributes": ()},
        {"node_attributes": ("missing",)},
    ):
        def run(backend):
            out = outcome(nx.snap_aggregation, G, backend=backend, **args)
            return (out[0], graph_snapshot(out[1])) if out[0] == "ok" else out

        assert run("rustnx") == run("networkx")


# --- create_component_structure ---------------------------------------------------


@pytest.mark.parametrize("s", range(60))
def test_create_component_structure(s):
    from networkx.algorithms import asteroidal

    rnd = random.Random(s)
    n = rnd.randint(0, 50)
    G = nx.gnp_random_graph(n, rnd.random() * 0.3, seed=s)
    r = rnd.random()
    if r < 0.25:
        G = nx.relabel_nodes(G, {v: f"n{v}" for v in G})
    elif r < 0.4:
        G = nx.relabel_nodes(G, {v: (v, v % 3) for v in G})
    elif r < 0.55:
        G = nx.relabel_nodes(G, {v: v * 8 for v in G})
    elif r < 0.7:
        order = list(G)
        rnd.shuffle(order)
        H = nx.Graph()
        H.add_nodes_from(order)
        H.add_edges_from(G.edges)
        G = H
    if n and rnd.random() < 0.3:
        G.add_edge(next(iter(G)), next(iter(G)))

    def run(backend):
        out = outcome(asteroidal.create_component_structure, G, backend=backend)
        return (out[0], typed(out[1])) if out[0] == "ok" else out

    assert run("rustnx") == run("networkx")


# --- capacity_scaling ---------------------------------------------------------------


@pytest.mark.parametrize("s", range(150))
def test_capacity_scaling(s):
    rnd = random.Random(s)
    n = rnd.randint(1, 25)
    cls = rnd.choice([nx.DiGraph, nx.MultiDiGraph])
    G = cls(nx.gnm_random_graph(n, rnd.randint(0, n * 3), seed=s, directed=True))
    if rnd.random() < 0.3:
        G = nx.relabel_nodes(G, {v: (f"n{v}" if v % 2 else v * 8) for v in G})
    nodes = list(G)
    if G.is_multigraph():
        for u, v in list(G.edges())[:4]:
            G.add_edge(u, v)
    if rnd.random() < 0.4:
        G.add_edge(nodes[0], nodes[0])
    kw = {"keys": True} if G.is_multigraph() else {}
    for e in G.edges(**kw):
        if rnd.random() < 0.85:
            G.edges[e]["capacity"] = rnd.randint(0, 15)
        G.edges[e]["weight"] = rnd.randint(-3, 12)
    for _ in range(rnd.randint(0, 4)):
        a, b = rnd.sample(nodes, 2) if len(nodes) > 1 else (nodes[0], nodes[0])
        x = rnd.randint(1, 10)
        G.nodes[a]["demand"] = G.nodes[a].get("demand", 0) - x
        G.nodes[b]["demand"] = G.nodes[b].get("demand", 0) + x
    if rnd.random() < 0.1:
        G.nodes[nodes[0]]["demand"] = G.nodes[nodes[0]].get("demand", 0) + 1

    def run(backend):
        out = outcome(nx.capacity_scaling, G, backend=backend)
        return (out[0], typed(out[1])) if out[0] == "ok" else out

    assert run("rustnx") == run("networkx")


def test_capacity_scaling_declines_floats():
    G = nx.DiGraph([(0, 1, {"capacity": 2.5})])
    G.nodes[0]["demand"] = -1
    G.nodes[1]["demand"] = 1
    with pytest.raises(NotImplementedError):
        nx.capacity_scaling(G, backend="rustnx")


# --- k-edge augmentation hybrids ----------------------------------------------------


def _norm(out):
    # NetworkX iterates sets of graph objects (memory-address order) in the
    # partial and general k-edge-subgraph paths; compare those unordered.
    if out[0] != "ok":
        return out
    return "ok", sorted(repr(x) for x in out[1])


def _subgraphs(gen):
    return [(list(H.nodes), list(H.edges)) for H in gen]


@pytest.mark.parametrize("s", range(40))
def test_k_edge_augmentation(s):
    rnd = random.Random(s)
    n = rnd.randint(1, 14)
    G = nx.gnm_random_graph(n, rnd.randint(0, n * 2), seed=s)
    if rnd.random() < 0.3:
        G = nx.relabel_nodes(G, {v: f"n{v}" for v in G})
    nodes = list(G)
    non = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:] if not G.has_edge(u, v)]
    rnd.shuffle(non)
    av_list = non[: rnd.randint(0, len(non))]
    av_dict = {e: rnd.randint(1, 9) for e in av_list}
    av_tri = [(u, v, {"w": rnd.randint(1, 9)}) for u, v in av_list]
    for k in (1, 2, 3, 4):
        for kw in ({}, {"partial": True}, {"avail": av_list}, {"avail": av_dict},
                   {"avail": av_tri, "weight": "w"}, {"avail": av_dict, "partial": True}):
            ours = outcome(lambda: list(nx.k_edge_augmentation(G, k, backend="rustnx", **kw)))
            ref = outcome(lambda: list(nx.k_edge_augmentation(G, k, backend="networkx", **kw)))
            if kw.get("partial") and "avail" in kw:
                assert _norm(ours) == _norm(ref)
            else:
                assert ours == ref
        for backend_pair in (("rustnx", "networkx"),):
            a, b = (outcome(lambda be=be: _subgraphs(ek.general_k_edge_subgraphs(G, k, backend=be)))
                    for be in backend_pair)
            assert _norm(a) == _norm(b)
            a, b = (outcome(lambda be=be: list(ea.greedy_k_edge_augmentation(G, k, avail=av_dict, seed=s, backend=be)))
                    for be in backend_pair)
            assert a == b
            a, b = (outcome(lambda be=be: list(ea.partial_k_edge_augmentation(G, k, avail=av_dict, backend=be)))
                    for be in backend_pair)
            assert _norm(a) == _norm(b)
    for func, kw in ((ea.weighted_bridge_augmentation, {}), (ea.weighted_one_edge_augmentation, {"partial": True})):
        a, b = (outcome(lambda be=be: list(func(G, av_dict, backend=be, **kw))) for be in ("rustnx", "networkx"))
        assert a == b


# --- spanner ----------------------------------------------------------------------


@pytest.mark.parametrize("s", range(60))
def test_spanner(s):
    rnd = random.Random(s)
    n = rnd.randint(1, 60)
    G = nx.gnm_random_graph(n, rnd.randint(0, n * 4), seed=s)
    r = rnd.random()
    if r < 0.2:
        G = nx.relabel_nodes(G, {v: f"n{v}" for v in G})
    elif r < 0.35:
        G = nx.relabel_nodes(G, {v: (v, v * 3) for v in G})
    if rnd.random() < 0.2:
        G.add_edge(next(iter(G)), next(iter(G)))
    for u, v in G.edges:
        G.edges[u, v]["w"] = rnd.choice([rnd.randint(1, 5), rnd.random(), 2.0])
    for stretch in (1, 2, 3, 5, 7.0):
        for weight in (None, "w"):
            def run(backend):
                out = outcome(nx.spanner, G, stretch, weight=weight, seed=s, backend=backend)
                return (out[0], graph_snapshot(out[1])) if out[0] == "ok" else out

            assert run("rustnx") == run("networkx")
    # The generator ends where NetworkX leaves it.
    states = []
    for backend in ("rustnx", "networkx"):
        rng = random.Random(s)
        nx.spanner(G, 3, seed=rng, backend=backend)
        states.append(rng.random())
    assert states[0] == states[1]
