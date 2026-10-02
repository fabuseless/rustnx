"""rustnx.rx against the real rustworkx.

Random sequences of graph edits are applied to both libraries; every query
and algorithm must then agree (floats to within 1e-12, since rustworkx sums
some results in parallel).
"""

import math
import random

import pytest

RX = pytest.importorskip("rustworkx")
import rustnx.rx as rx  # noqa: E402


def random_ops(seed):
    rng = random.Random(seed)
    ops = []
    for _ in range(rng.randint(5, 60)):
        r = rng.random()
        if r < 0.3:
            ops.append(("add_node", rng.randint(0, 99)))
        elif r < 0.8:
            ops.append(("add_edge", rng.randint(0, 25), rng.randint(0, 25), rng.choice([1, 2, 3, 0.5])))
        elif r < 0.88:
            ops.append(("remove_node", rng.randint(0, 25)))
        elif r < 0.95:
            ops.append(("remove_edge", rng.randint(0, 25), rng.randint(0, 25)))
        else:
            ops.append(("remove_edge_from_index", rng.randint(0, 40)))
    return ops


def apply(lib, directed, multigraph, ops):
    g = lib.PyDiGraph(multigraph=multigraph) if directed else lib.PyGraph(multigraph=multigraph)
    for op in ops:
        name, *args = op
        try:
            if name == "add_node":
                g.add_node(args[0])
            elif name == "add_edge":
                g.add_edge(*args)
            elif name == "remove_node":
                g.remove_node(args[0])
            elif name == "remove_edge":
                g.remove_edge(*args)
            else:
                g.remove_edge_from_index(args[0])
        except (IndexError, OverflowError, lib.NoEdgeBetweenNodes):
            pass
    return g


def outcome(func):
    try:
        return ("ok", normalize(func()))
    except Exception as exc:
        return (type(exc).__name__, str(exc))


def normalize(x):
    if hasattr(x, "items"):
        return {k: normalize(v) for k, v in x.items()}
    if isinstance(x, (str, bytes)):
        return x
    if isinstance(x, (set, frozenset)):
        return frozenset(x)
    if hasattr(x, "__iter__") and not isinstance(x, tuple):
        return [normalize(v) for v in x]
    if isinstance(x, tuple):
        return tuple(normalize(v) for v in x)
    return x


def close(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    return a == b


CASES = [(d, m, s) for d in (False, True) for m in (True, False) for s in range(40)]


@pytest.mark.parametrize("directed, multigraph, seed", CASES)
def test_graph_structure(directed, multigraph, seed):
    ops = random_ops(seed)
    a, b = apply(RX, directed, multigraph, ops), apply(rx, directed, multigraph, ops)
    for method in ["node_indices", "nodes", "edge_list", "edge_indices", "weighted_edge_list",
                   "edges", "num_nodes", "num_edges"]:
        assert normalize(getattr(a, method)()) == normalize(getattr(b, method)()), method
    assert normalize(a.edge_index_map()) == normalize(b.edge_index_map())
    for v in range(28):
        for method in (["incident_edges", "adj"] if directed else ["degree", "incident_edges", "adj"]):
            assert outcome(lambda: getattr(a, method)(v)) == outcome(lambda: getattr(b, method)(v)), (method, v)
        assert set(a.neighbors(v)) == set(b.neighbors(v))
        for method in ["out_edges", "in_edges", "out_edge_indices", "in_edge_indices",
                       "incident_edge_index_map"]:
            assert outcome(lambda: getattr(a, method)(v)) == outcome(lambda: getattr(b, method)(v)), (method, v)
        for u in range(0, 28, 4):
            assert normalize(a.edge_indices_from_endpoints(v, u)) == b.edge_indices_from_endpoints(v, u)
        if directed:
            assert set(a.neighbors_undirected(v)) == set(b.neighbors_undirected(v))
            assert normalize(a.incident_edge_index_map(v, all_edges=True)) == normalize(
                b.incident_edge_index_map(v, all_edges=True)
            )
            for method in ["successor_indices", "predecessor_indices",
                           "in_degree", "out_degree"]:
                assert outcome(lambda: getattr(a, method)(v)) == outcome(lambda: getattr(b, method)(v)), (method, v)
        for u in range(0, 28, 3):
            assert a.has_edge(v, u) == b.has_edge(v, u)
            assert outcome(lambda: a.get_edge_data(v, u)) == outcome(lambda: b.get_edge_data(v, u))
    assert a.has_parallel_edges() == b.has_parallel_edges()
    assert a.find_node_by_weight(7) == b.find_node_by_weight(7)
    assert normalize(a.filter_nodes(lambda x: x % 2 == 0)) == b.filter_nodes(lambda x: x % 2 == 0)
    assert normalize(a.filter_edges(lambda w: w > 1)) == b.filter_edges(lambda w: w > 1)
    if directed:
        for kwargs in [{}, {"multigraph": False}, {"multigraph": False, "weight_combo_fn": lambda x, y: x + y}]:
            ua, ub = a.to_undirected(**kwargs), b.to_undirected(**kwargs)
            assert normalize(ua.weighted_edge_list()) == normalize(ub.weighted_edge_list())
            assert ua.node_indices() == ub.node_indices()
    # Index reuse after the edits.
    assert [a.add_node("x") for _ in range(3)] == [b.add_node("x") for _ in range(3)]
    live = a.node_indices()
    if len(live) >= 2:
        assert a.add_edge(live[0], live[1], 9) == b.add_edge(live[0], live[1], 9)


def algorithm_calls(directed):
    calls = {
        "betweenness": lambda m, g: m.betweenness_centrality(g),
        "betweenness_raw": lambda m, g: m.betweenness_centrality(g, normalized=False),
        "betweenness_endpoints": lambda m, g: m.betweenness_centrality(g, endpoints=True),
        "closeness": lambda m, g: m.closeness_centrality(g),
        "closeness_no_wf": lambda m, g: m.closeness_centrality(g, wf_improved=False),
        "all_pairs": lambda m, g: m.all_pairs_dijkstra_path_lengths(g, float),
    }
    if directed:
        calls.update({
            "scc": lambda m, g: m.strongly_connected_components(g),
            "n_scc": lambda m, g: m.number_strongly_connected_components(g),
            "is_scc": lambda m, g: m.is_strongly_connected(g),
            "wcc": lambda m, g: m.weakly_connected_components(g),
            "n_wcc": lambda m, g: m.number_weakly_connected_components(g),
            "is_wcc": lambda m, g: m.is_weakly_connected(g),
            "topo": lambda m, g: m.topological_sort(g),
            "dag": lambda m, g: m.is_directed_acyclic_graph(g),
            "pagerank": lambda m, g: m.pagerank(g),
            "pagerank_w": lambda m, g: m.pagerank(g, weight_fn=float, alpha=0.7),
        })
    else:
        calls.update({
            "cc": lambda m, g: m.connected_components(g),
            "n_cc": lambda m, g: m.number_connected_components(g),
            "is_cc": lambda m, g: m.is_connected(g),
        })
    return calls


@pytest.mark.parametrize("directed, multigraph, seed", CASES)
def test_algorithms(directed, multigraph, seed):
    ops = random_ops(seed)
    a, b = apply(RX, directed, multigraph, ops), apply(rx, directed, multigraph, ops)
    for name, call in algorithm_calls(directed).items():
        ours, ref = outcome(lambda: call(rx, b)), outcome(lambda: call(RX, a))
        assert close(ours, ref), name
    for v in a.node_indices()[:4]:
        assert close(outcome(lambda: rx.dijkstra_shortest_path_lengths(b, v, float)),
                     outcome(lambda: RX.dijkstra_shortest_path_lengths(a, v, float)))
        for goal in a.node_indices()[:3]:
            assert close(outcome(lambda: rx.dijkstra_shortest_path_lengths(b, v, float, goal=goal)),
                         outcome(lambda: RX.dijkstra_shortest_path_lengths(a, v, float, goal=goal)))


def test_pagerank_options():
    ops = random_ops(3)
    a, b = apply(RX, True, True, ops), apply(rx, True, True, ops)
    live = a.node_indices()
    for kwargs in [
        {"personalization": {live[0]: 1, live[-1]: 2}},
        {"nstart": {v: i + 1 for i, v in enumerate(live)}},
        {"dangling": {live[0]: 1}},
        {"max_iter": 1},
        {"tol": 1e-12, "max_iter": 500},
    ]:
        assert close(outcome(lambda: rx.pagerank(b, **kwargs)), outcome(lambda: RX.pagerank(a, **kwargs))), kwargs


def test_dijkstra_cost_errors():
    for lib_pair in [(RX, rx)]:
        results = []
        for lib in lib_pair:
            g = lib.PyDiGraph()
            g.add_nodes_from(range(4))
            g.add_edge(0, 1, 1.0)
            g.add_edge(2, 3, -1.0)  # unreachable from 0: no error
            g.add_edge(1, 2, float("nan"))
            results.append([
                outcome(lambda: lib.dijkstra_shortest_path_lengths(g, 0, float, goal=1)),
                outcome(lambda: lib.dijkstra_shortest_path_lengths(g, 0, float)),
                outcome(lambda: lib.dijkstra_shortest_path_lengths(g, 2, float)),
                outcome(lambda: lib.dijkstra_shortest_path_lengths(g, 3, float)),
                outcome(lambda: lib.dijkstra_shortest_path_lengths(g, 9, float)),
            ])
        assert results[0] == results[1]


def test_extend_from_edge_list_and_null_graph():
    for lib in (RX, rx):
        pass
    a, b = RX.PyGraph(), rx.PyGraph()
    for g in (a, b):
        g.extend_from_edge_list([(0, 3), (2, 1)])
        g.extend_from_weighted_edge_list([(4, 0, 7)])
    assert normalize(a.weighted_edge_list()) == normalize(b.weighted_edge_list())
    assert a.node_indices() == b.node_indices()
    assert outcome(lambda: RX.is_connected(RX.PyGraph())) [0] == outcome(lambda: rx.is_connected(rx.PyGraph()))[0] == "NullGraph"
    with pytest.raises(TypeError):
        rx.pagerank(rx.PyGraph())
    with pytest.raises(rx.DAGHasCycle):
        g = rx.PyDiGraph()
        g.extend_from_edge_list([(0, 1), (1, 0)])
        rx.topological_sort(g)


def test_networkx_converter():
    import networkx as nx

    G = nx.les_miserables_graph()
    a, b = RX.networkx_converter(G, keep_attributes=True), rx.networkx_converter(G, keep_attributes=True)
    assert normalize(a.weighted_edge_list()) == normalize(b.weighted_edge_list())
    assert normalize(a.nodes()) == normalize(b.nodes())
    assert close(normalize(RX.betweenness_centrality(a)), rx.betweenness_centrality(b))


def test_copy_is_independent():
    g = rx.PyGraph()
    g.add_nodes_from("abc")
    g.add_edge(0, 1, None)
    h = g.copy()
    h.remove_node(0)
    assert g.num_nodes() == 3 and g.num_edges() == 1 and h.num_nodes() == 2


@pytest.mark.parametrize("seed", range(20))
def test_digraph_edits(seed):
    ops = random_ops(seed)
    a, b = apply(RX, True, True, ops), apply(rx, True, True, ops)
    live = a.node_indices()
    if live:
        assert a.add_child(live[0], "c", 5) == b.add_child(live[0], "c", 5)
        assert a.add_parent(live[-1], "p", 6) == b.add_parent(live[-1], "p", 6)
    a.reverse()
    b.reverse()
    assert normalize(a.edge_index_map()) == normalize(b.edge_index_map())
    for v in a.node_indices():
        assert normalize(a.out_edges(v)) == normalize(b.out_edges(v))
    assert normalize(RX.strongly_connected_components(a)) == rx.strongly_connected_components(b)
    a.clear_edges()
    b.clear_edges()
    assert a.add_edge(a.node_indices()[0], a.node_indices()[0], 1) == b.add_edge(b.node_indices()[0], b.node_indices()[0], 1) if live else True


@pytest.mark.parametrize("directed", [False, True])
def test_larger_graph(directed):
    """Above rustworkx's parallel threshold (50 nodes)."""
    rng = random.Random(7)
    edges = [(rng.randrange(400), rng.randrange(400), rng.choice([1.0, 2.0, 3.5])) for _ in range(1500)]
    a = RX.PyDiGraph() if directed else RX.PyGraph()
    b = rx.PyDiGraph() if directed else rx.PyGraph()
    for g in (a, b):
        g.extend_from_weighted_edge_list(edges)
        g.remove_nodes_from([3, 50, 51, 200])
    for name, call in algorithm_calls(directed).items():
        if name == "all_pairs":
            continue
        assert close(outcome(lambda: call(rx, b)), outcome(lambda: call(RX, a))), name
    assert close(normalize(RX.dijkstra_shortest_path_lengths(a, 0, float)),
                 rx.dijkstra_shortest_path_lengths(b, 0, float))
