"""Compare every rustnx algorithm with NetworkX on many random graphs.

``backend="rustnx"`` forces the Rust path; ``backend="networkx"`` forces the
reference implementation.
"""

import logging
import math
import random

import networkx as nx
import pytest

import rustnx


def random_graph(seed, directed, weights):
    rng = random.Random(seed)
    n = rng.randint(1, 40)
    p = rng.choice([0.02, 0.08, 0.2, 0.5])
    G = nx.gnp_random_graph(n, p, seed=seed, directed=directed)
    # Shuffle node insertion order and use non-integer labels sometimes, so
    # we don't accidentally depend on nodes being 0..n-1.
    if rng.random() < 0.5:
        mapping = {v: f"n{v}" for v in G}
        G = nx.relabel_nodes(G, mapping)
    H = G.__class__()
    nodes = list(G)
    rng.shuffle(nodes)
    H.add_nodes_from(nodes)
    edges = list(G.edges)
    rng.shuffle(edges)
    H.add_edges_from(edges)
    if rng.random() < 0.3 and nodes:
        v = rng.choice(nodes)
        H.add_edge(v, v)  # self-loop
    for u, v, d in H.edges(data=True):
        if weights == "int":
            d["weight"] = rng.randint(1, 5)
        elif weights == "float":
            d["weight"] = rng.choice([0.5, 1.0, 1.5, 2.25, 3.0])
        elif weights == "missing" and rng.random() < 0.5:
            d["weight"] = rng.randint(1, 3)
    return H


SEEDS = range(60)
WEIGHTS = ["none", "int", "float", "missing"]


def both(func, *args, **kwargs):
    ours = func(*args, backend="rustnx", **kwargs)
    ref = func(*args, backend="networkx", **kwargs)
    return ours, ref


def assert_close_dicts(ours, ref):
    assert list(ours) == list(ref)  # same keys in the same order
    for key in ref:
        assert ours[key] == pytest.approx(ref[key], rel=1e-12, abs=1e-12), key


@pytest.mark.parametrize("seed", SEEDS)
def test_connected_components(seed):
    G = random_graph(seed, directed=False, weights="none")
    ours, ref = both(lambda G, **kw: list(nx.connected_components(G, **kw)), G)
    assert ours == ref
    assert nx.number_connected_components(G, backend="rustnx") == len(ref)
    assert nx.is_connected(G, backend="rustnx") == nx.is_connected(G)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_shortest_path_length(seed, directed):
    G = random_graph(seed, directed, "none")
    for source in list(G)[:5]:
        for cutoff in [None, 0, 1, 2, 2.5]:
            ours, ref = both(
                nx.single_source_shortest_path_length, G, source, cutoff=cutoff
            )
            assert list(ours.items()) == list(ref.items())


@pytest.mark.parametrize("weights", WEIGHTS)
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_dijkstra_path_length(seed, directed, weights):
    G = random_graph(seed, directed, weights)
    for source in list(G)[:5]:
        for cutoff in [None, 1, 2.5]:
            for weight in ["weight", None]:
                ours, ref = both(
                    nx.single_source_dijkstra_path_length,
                    G,
                    source,
                    cutoff=cutoff,
                    weight=weight,
                )
                # Same nodes, same order, same values (and types for ints).
                assert list(ours.items()) == list(ref.items())
                for key in ref:
                    assert type(ours[key]) is type(ref[key]) or weights == "float"


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_betweenness(seed, directed, weights):
    G = random_graph(seed, directed, weights)
    weight = None if weights == "none" else "weight"
    for normalized in [True, False]:
        for endpoints in [False, True]:
            ours, ref = both(
                nx.betweenness_centrality,
                G,
                normalized=normalized,
                weight=weight,
                endpoints=endpoints,
            )
            assert_close_dicts(ours, ref)


@pytest.mark.parametrize("seed", range(20))
def test_betweenness_sampled(seed):
    G = random_graph(seed, directed=seed % 2 == 0, weights="int")
    k = max(1, len(G) // 3)
    for endpoints in [False, True]:
        ours, ref = both(
            nx.betweenness_centrality, G, k=k, endpoints=endpoints, seed=seed
        )
        assert list(ours) == list(ref)
        for key in ref:
            if math.isnan(ref[key]):
                assert math.isnan(ours[key])
            else:
                assert ours[key] == pytest.approx(ref[key], rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_closeness(seed, directed, weights):
    G = random_graph(seed, directed, weights)
    distance = None if weights == "none" else "weight"
    for wf in [True, False]:
        ours, ref = both(
            nx.closeness_centrality, G, distance=distance, wf_improved=wf
        )
        assert ours == ref  # exact: same per-node summation order as NetworkX
        u = next(iter(G))
        assert nx.closeness_centrality(
            G, u=u, distance=distance, backend="rustnx"
        ) == nx.closeness_centrality(G, u=u, distance=distance)


def test_bigger_graph_matches():
    G = nx.barabasi_albert_graph(2000, 3, seed=1)
    ours, ref = both(nx.betweenness_centrality, G)
    assert_close_dicts(ours, ref)
    ours, ref = both(nx.closeness_centrality, G)
    assert ours == ref


# --- Errors match NetworkX ------------------------------------------------


def test_missing_source():
    G = nx.path_graph(3)
    with pytest.raises(nx.NodeNotFound, match="Source 9 is not in G"):
        nx.single_source_shortest_path_length(G, 9, backend="rustnx")
    with pytest.raises(nx.NodeNotFound, match="Node 9 not found in graph"):
        nx.single_source_dijkstra_path_length(G, 9, backend="rustnx")


def test_negative_weights():
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=1)
    G.add_edge(1, 2, weight=1)
    G.add_edge(0, 2, weight=3)
    G.add_edge(2, 1, weight=-5)
    with pytest.raises(ValueError, match="Contradictory paths"):
        nx.single_source_dijkstra_path_length(G, 0, backend="rustnx")


def test_directed_components_not_implemented():
    with pytest.raises(nx.NetworkXNotImplemented):
        list(nx.connected_components(nx.DiGraph([(0, 1)]), backend="rustnx"))


def test_null_graph_connectivity():
    with pytest.raises(nx.NetworkXPointlessConcept):
        nx.is_connected(nx.Graph(), backend="rustnx")


# --- Dispatch through backend priority --------------------------------------


@pytest.fixture
def rustnx_priority():
    old = nx.config.backend_priority.algos
    nx.config.backend_priority.algos = ["rustnx"]
    yield
    nx.config.backend_priority.algos = old


def test_priority_dispatches_to_rust(rustnx_priority, caplog):
    G = nx.path_graph(10)
    with caplog.at_level(logging.DEBUG, logger="networkx"):
        nx.betweenness_centrality(G)
    assert any("Using backend 'rustnx'" in r.getMessage() for r in caplog.records)


def test_small_graphs_stay_in_networkx(rustnx_priority, caplog):
    small = nx.path_graph(10)
    large = nx.path_graph(1000)
    with caplog.at_level(logging.DEBUG, logger="networkx"):
        nx.single_source_shortest_path_length(small, 0)
    assert not any("Using backend 'rustnx'" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="networkx"):
        nx.single_source_shortest_path_length(large, 0)
    assert any("Using backend 'rustnx'" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "call",
    [
        # Multigraphs and callable weights fall back to NetworkX.
        lambda: nx.betweenness_centrality(nx.MultiGraph([(0, 1), (0, 1), (1, 2)])),
        lambda: nx.single_source_dijkstra_path_length(
            nx.path_graph(4), 0, weight=lambda u, v, d: 2
        ),
        # Non-numeric weights fail conversion and fall back.
        lambda: nx.closeness_centrality(
            nx.Graph([(0, 1, {"weight": __import__("fractions").Fraction(1, 3)})]),
            distance="weight",
        ),
    ],
)
def test_unsupported_inputs_fall_back(rustnx_priority, call):
    result = call()
    old = nx.config.backend_priority.algos
    nx.config.backend_priority.algos = []
    try:
        assert result == call()
    finally:
        nx.config.backend_priority.algos = old


def test_graph_wrapper():
    G = nx.DiGraph([(0, 1), (1, 2)])
    R = rustnx.from_networkx(G, weights=["weight"])
    assert len(R) == 3 and R.number_of_edges() == 2 and R.is_directed()
    assert 1 in R and "x" not in R and [] not in R
    assert rustnx.to_networkx(R) is G
    assert nx.betweenness_centrality(R) == nx.betweenness_centrality(G)


def test_unrecognized_rescale_falls_back(rustnx_priority, monkeypatch):
    from networkx.algorithms.centrality import betweenness as nx_bc

    from rustnx import algorithms

    def future_rescale(betweenness, n, *, normalized, directed, new_required_arg):
        raise AssertionError("rustnx must not call an unrecognized _rescale")

    monkeypatch.setattr(nx_bc, "_rescale", future_rescale)
    algorithms._rescale_params.cache_clear()
    try:
        with pytest.raises(NotImplementedError):
            algorithms.betweenness_centrality(rustnx.from_networkx(nx.path_graph(5)))
    finally:
        algorithms._rescale_params.cache_clear()


def test_new_networkx_parameters_are_tolerated():
    """NetworkX passes every parameter to backends, including ones added in
    newer releases (closeness_centrality gained ``sp`` in 3.7)."""
    import inspect

    from rustnx import interface

    G = nx.path_graph(5)
    R = rustnx.from_networkx(G)
    params = inspect.signature(nx.closeness_centrality).parameters
    if "sp" not in params:
        pytest.skip("installed NetworkX has no closeness_centrality(sp=...)")
    # Left at its default: rustnx runs.
    assert interface.closeness_centrality(R, sp=None) == nx.closeness_centrality(G)
    # Actually used: rustnx declines, so NetworkX runs it.
    sp = dict(nx.all_pairs_shortest_path_length(G))
    assert interface.can_run("closeness_centrality", (G,), {"sp": sp}) != True  # noqa: E712
    with pytest.raises(NotImplementedError):
        interface.closeness_centrality(R, sp=sp)


def test_backend_specific_keywords_decline():
    from rustnx import interface

    G = nx.path_graph(5)
    assert isinstance(
        interface.can_run("betweenness_centrality", (G,), {"made_up": 1}), str
    )


# --- Directed components ------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_strongly_connected_components(seed):
    G = random_graph(seed, directed=True, weights="none")
    ours, ref = both(lambda G, **kw: list(nx.strongly_connected_components(G, **kw)), G)
    assert ours == ref  # same components in the same order
    assert nx.number_strongly_connected_components(G, backend="rustnx") == len(ref)
    assert nx.is_strongly_connected(G, backend="rustnx") == nx.is_strongly_connected(G)


def test_strongly_connected_single_component():
    # Exercises NetworkX's early exit when the whole graph is one SCC.
    G = nx.DiGraph(nx.cycle_graph(50))
    G.add_edges_from([(10, 3), (40, 20), (7, 30)])
    ours, ref = both(lambda G, **kw: list(nx.strongly_connected_components(G, **kw)), G)
    assert ours == ref and len(ours) == 1


@pytest.mark.parametrize("seed", SEEDS)
def test_weakly_connected_components(seed):
    G = random_graph(seed, directed=True, weights="none")
    ours, ref = both(lambda G, **kw: list(nx.weakly_connected_components(G, **kw)), G)
    assert ours == ref
    assert nx.number_weakly_connected_components(G, backend="rustnx") == len(ref)
    assert nx.is_weakly_connected(G, backend="rustnx") == nx.is_weakly_connected(G)


def test_directed_components_reject_undirected_and_null():
    with pytest.raises(nx.NetworkXNotImplemented):
        list(nx.strongly_connected_components(nx.path_graph(3), backend="rustnx"))
    with pytest.raises(nx.NetworkXNotImplemented):
        list(nx.weakly_connected_components(nx.path_graph(3), backend="rustnx"))
    for func in [nx.is_strongly_connected, nx.is_weakly_connected]:
        with pytest.raises(nx.NetworkXPointlessConcept):
            func(nx.DiGraph(), backend="rustnx")


# --- Topological order ----------------------------------------------------------


def random_dag(seed):
    G = random_graph(seed, directed=True, weights="none")
    rng = random.Random(seed)
    rank = {v: rng.random() for v in G}
    G.remove_edges_from([(u, v) for u, v in G.edges if rank[u] >= rank[v]])
    return G


def collect_until_error(iterable):
    out = []
    try:
        for item in iterable:
            out.append(item)
    except nx.NetworkXUnfeasible:
        return out, "unfeasible"
    return out, None


@pytest.mark.parametrize("seed", SEEDS)
def test_topological_sort(seed):
    for G in [random_dag(seed), random_graph(seed, directed=True, weights="none")]:
        for func in [nx.topological_sort, nx.topological_generations]:
            ours = collect_until_error(func(G, backend="rustnx"))
            ref = collect_until_error(func(G, backend="networkx"))
            assert ours == ref  # same order, and same point of failure on cycles
        assert nx.is_directed_acyclic_graph(
            G, backend="rustnx"
        ) == nx.is_directed_acyclic_graph(G)


def test_topological_sort_undirected():
    G = nx.path_graph(4)
    with pytest.raises(nx.NetworkXError, match="undirected"):
        list(nx.topological_sort(G, backend="rustnx"))
    assert nx.is_directed_acyclic_graph(G, backend="rustnx") is False


# --- PageRank ---------------------------------------------------------------------


def assert_pagerank_close(ours, ref):
    assert list(ours) == list(ref)
    for key in ref:
        assert ours[key] == pytest.approx(ref[key], rel=1e-9, abs=1e-12), key


@pytest.mark.parametrize("weights", ["none", "int", "float", "missing"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_pagerank(seed, directed, weights):
    G = random_graph(seed, directed, weights)
    weight = None if weights == "none" else "weight"
    ours, ref = both(nx.pagerank, G, weight=weight)
    assert_pagerank_close(ours, ref)


@pytest.mark.parametrize("seed", range(20))
def test_pagerank_options(seed):
    G = random_graph(seed, directed=True, weights="int")
    rng = random.Random(seed)
    nodes = list(G)
    subset = rng.sample(nodes, max(1, len(nodes) // 3))
    personalization = {v: rng.randint(0, 5) for v in subset}
    personalization[subset[0]] = 1  # at least one non-zero
    options = [
        {"alpha": 0.5},
        {"personalization": personalization},
        {"nstart": {v: rng.random() for v in nodes}},
        {"dangling": {v: rng.randint(1, 3) for v in subset}},
        {"personalization": personalization, "dangling": personalization},
        {"tol": 1e-10, "max_iter": 1000},
    ]
    for kwargs in options:
        ours, ref = both(nx.pagerank, G, **kwargs)
        assert_pagerank_close(ours, ref)


def test_pagerank_errors():
    G = nx.DiGraph(nx.path_graph(10))
    with pytest.raises(nx.PowerIterationFailedConvergence):
        nx.pagerank(G, max_iter=1, backend="rustnx")
    with pytest.raises(ZeroDivisionError):
        nx.pagerank(G, personalization={0: 0}, backend="rustnx")
    assert nx.pagerank(nx.DiGraph(), backend="rustnx") == {}


# --- Changing the graph during iteration ---------------------------------------


def _mutate_during(algorithm, mutate):
    DG = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
    first = True
    for x in algorithm(DG):
        if first:
            first = False
            mutate(DG, x)


@pytest.mark.parametrize(
    "mutate, error",
    [
        (lambda DG, x: DG.add_edge(5 - x, 5), RuntimeError),
        (lambda DG, x: DG.remove_node(2), RuntimeError),
        (lambda DG, x: DG.remove_node(4), nx.NetworkXUnfeasible),
        (lambda DG, x: DG.add_edge(9, 10), None),
    ],
)
def test_topological_sort_graph_changed(mutate, error):
    """Matches NetworkX's behavior when the graph changes mid-iteration."""
    for backend in ["rustnx", "networkx"]:

        def algorithm(G):
            return nx.topological_sort(G, backend=backend)

        if error is None:
            _mutate_during(algorithm, mutate)
        else:
            with pytest.raises(error):
                _mutate_during(algorithm, mutate)


def test_topological_sort_changed_before_iterating():
    DG = nx.DiGraph([(1, 2), (2, 3)])
    it = nx.topological_sort(DG, backend="rustnx")
    DG.add_edge(0, 1)
    assert list(it) == [0, 1, 2, 3]


def test_components_graph_changed():
    for func, G in [
        (nx.connected_components, nx.Graph([(0, 1), (2, 3)])),
        (nx.weakly_connected_components, nx.DiGraph([(0, 1), (2, 3)])),
        (nx.strongly_connected_components, nx.DiGraph([(0, 1), (2, 3)])),
    ]:
        it = func(G, backend="rustnx")
        next(it)
        G.add_edge(1, 2)
        with pytest.raises(RuntimeError, match="changed during iteration"):
            list(it)


def test_iteration_marker_is_cleaned_up():
    G = nx.DiGraph([(0, 1), (1, 2)])
    list(nx.topological_sort(G, backend="rustnx"))
    list(nx.strongly_connected_components(G, backend="rustnx"))
    assert not any(
        isinstance(k, tuple) and k[0] == "rustnx-iteration"
        for k in G.__networkx_cache__
    )


def test_views_fall_back(rustnx_priority):
    G = nx.DiGraph(nx.path_graph(600, create_using=nx.DiGraph))
    view = G.subgraph(range(550))
    assert list(nx.topological_sort(view)) == list(
        nx.topological_sort(view, backend="networkx")
    )


# --- Conversion -----------------------------------------------------------------


def _fresh_int(x):
    """An int equal to x but (for large x) a different object."""
    return int(str(x))


@pytest.mark.parametrize(
    "make",
    [
        # Nodes 0..n-1 in order (fast path).
        lambda: nx.gnm_random_graph(300, 900, seed=1, directed=True),
        # Same labels, but not in order.
        lambda: nx.DiGraph(
            [(v, u) for u, v in nx.gnm_random_graph(300, 900, seed=2).edges]
        ),
        # Equal-but-not-identical int objects in the adjacency.
        lambda: nx.DiGraph(
            [(_fresh_int(u + 1000), _fresh_int(v + 1000))
             for u, v in nx.gnm_random_graph(300, 900, seed=3).edges]
        ),
        # Strings, tuples and mixed types.
        lambda: nx.relabel_nodes(
            nx.gnm_random_graph(300, 900, seed=4, directed=True),
            lambda v: (v, "x") if v % 3 == 0 else (f"s{v}" if v % 3 == 1 else v + 0.5),
        ),
    ],
)
def test_conversion_node_labels(make):
    G = make()
    for u, v, d in G.edges(data=True):
        d["weight"] = (hash((u, v)) % 7) + 1
    source = next(iter(G))
    ours, ref = both(nx.single_source_dijkstra_path_length, G, source)
    assert list(ours.items()) == list(ref.items())
    ours, ref = both(nx.closeness_centrality, G, distance="weight")
    assert ours == ref


def test_bool_and_int_labels():
    # True == 1 as dict keys: NetworkX treats them as the same node.
    G = nx.Graph()
    G.add_edges_from([(0, 1), (True, 2), (2, 3)])
    ours, ref = both(nx.single_source_shortest_path_length, G, 0)
    assert list(ours.items()) == list(ref.items())


def test_exact_pred_requires_unchanged_graph():
    from rustnx import algorithms

    G = nx.DiGraph()
    G.add_weighted_edges_from([(0, 1, 1), (1, 2, 2), (2, 0, 3)])
    R = rustnx.from_networkx(G, ["weight"])
    G.add_edge(2, 3, weight=1)  # clears G.__networkx_cache__
    with pytest.raises(NotImplementedError, match="changed"):
        algorithms.closeness_centrality(R, distance="weight")


def test_pickle_and_deepcopy_after_rustnx():
    import copy
    import pickle

    G = nx.DiGraph()
    G.add_weighted_edges_from([(0, 1, 2), (1, 2, 3.5), (2, 0, 1), (2, 3, 1)])
    pr = nx.pagerank(G, backend="rustnx")
    cc = nx.closeness_centrality(G, distance="weight", backend="rustnx")
    for H in [pickle.loads(pickle.dumps(G)), copy.deepcopy(G)]:
        assert nx.pagerank(H, backend="rustnx") == pr
        assert nx.closeness_centrality(H, distance="weight", backend="rustnx") == cc
        (cached,) = H.__networkx_cache__["backends"]["rustnx"].values()
        assert cached._source is H and cached._source_unchanged()


def _pagerank_in_worker(G):
    return nx.pagerank(G, backend="rustnx")


def test_send_graph_to_another_process():
    import multiprocessing

    G = nx.gnm_random_graph(50, 150, seed=1, directed=True)
    expected = nx.pagerank(G, backend="rustnx")  # caches a converted graph on G
    pool = multiprocessing.get_context("spawn").Pool(1)
    try:
        result = pool.apply(_pagerank_in_worker, (G,))
    finally:
        # close() and join() rather than the context manager's terminate():
        # on free-threaded Windows, terminate() can race the pool's own
        # handler threads ("concurrent send_bytes() calls").
        pool.close()
        pool.join()
    assert result == expected


def test_malformed_pickle_data_rejected():
    from rustnx import _core

    with pytest.raises(ValueError, match="malformed"):
        _core._core_graph_from_bytes(b"RNX1" + b"\xff" * 20)


# --- Distance measures ----------------------------------------------------------


def outcome(func, *args, **kwargs):
    """A function's result, or its exception type and arguments."""
    try:
        result = func(*args, **kwargs)
        if hasattr(result, "__next__"):
            result = [(k, list(v.items())) for k, v in result]
        return ("ok", result)
    except Exception as exc:  # compare errors too
        return (type(exc), exc.args)


def connected_random_graph(seed, directed, weights):
    G = random_graph(seed, directed, weights)
    if seed % 3 == 0:
        return G  # often disconnected: exercises the errors
    # Otherwise add a cycle through all nodes so it is (strongly) connected.
    nodes = list(G)
    for u, v in zip(nodes, nodes[1:] + nodes[:1]):
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=2.0 if weights == "float" else 2)
    return G


DISTANCE_FUNCS = [
    nx.eccentricity,
    nx.diameter,
    nx.radius,
    nx.center,
    nx.periphery,
    nx.average_shortest_path_length,
    nx.wiener_index,
]


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_distance_measures(seed, directed, weights):
    G = connected_random_graph(seed, directed, weights)
    weight = None if weights == "none" else "weight"
    for func in DISTANCE_FUNCS:
        ours = outcome(func, G, weight=weight, backend="rustnx")
        ref = outcome(func, G, weight=weight, backend="networkx")
        assert ours == ref, func.__name__
    v = next(iter(G))
    assert outcome(nx.eccentricity, G, v=v, weight=weight, backend="rustnx") == outcome(
        nx.eccentricity, G, v=v, weight=weight
    )


def test_distance_measures_negative_weights():
    G = nx.DiGraph()
    G.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (2, 0, -5), (0, 2, 3)])
    for func in [nx.eccentricity, nx.average_shortest_path_length, nx.wiener_index]:
        ours = outcome(func, G, weight="weight", backend="rustnx")
        assert ours == outcome(func, G, weight="weight", backend="networkx")
        assert ours[0] is ValueError


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_all_pairs_lengths(seed, directed, weights):
    G = random_graph(seed, directed, weights)
    for cutoff in [None, 1, 2.5]:
        assert outcome(
            nx.all_pairs_shortest_path_length, G, cutoff=cutoff, backend="rustnx"
        ) == outcome(nx.all_pairs_shortest_path_length, G, cutoff=cutoff)
        for weight in ["weight", None]:
            ours = outcome(
                nx.all_pairs_dijkstra_path_length, G, cutoff=cutoff, weight=weight,
                backend="rustnx",
            )
            ref = outcome(
                nx.all_pairs_dijkstra_path_length, G, cutoff=cutoff, weight=weight
            )
            assert ours == ref


def test_all_pairs_larger_than_one_batch():
    G = nx.gnm_random_graph(2500, 6000, seed=5)
    ours = dict(nx.all_pairs_shortest_path_length(G, backend="rustnx"))
    assert list(ours) == list(G)
    ref = dict(nx.all_pairs_shortest_path_length(G))
    assert all(list(ours[k].items()) == list(ref[k].items()) for k in ref)


def test_all_pairs_graph_changed():
    G = nx.path_graph(5)
    it = nx.all_pairs_shortest_path_length(G, backend="rustnx")
    next(it)
    G.add_edge(0, 4)
    with pytest.raises(RuntimeError, match="changed during iteration"):
        list(it)


@pytest.mark.parametrize(
    "call",
    [
        lambda G: nx.diameter(G, usebounds=True),
        lambda G: nx.center(nx.path_graph(7)),  # trees take NetworkX's tree path
        lambda G: nx.eccentricity(G, sp=dict(nx.shortest_path_length(G))),
        lambda G: nx.eccentricity(G, v=[0, 1]),
        lambda G: nx.average_shortest_path_length(G, method="floyd-warshall"),
        lambda G: outcome(nx.diameter, nx.Graph()),
    ],
)
def test_distance_measures_fall_back(rustnx_priority, call):
    G = nx.cycle_graph(8)
    result = call(G)
    old = nx.config.backend_priority.algos
    nx.config.backend_priority.algos = []
    try:
        assert result == call(G)
    finally:
        nx.config.backend_priority.algos = old


def test_backend_function_list_matches_implementations():
    from rustnx import _info, algorithms, interface

    assert sorted(_info.FUNCTIONS) == sorted(algorithms.__all__)
    # Functions only some supported NetworkX releases have.
    version_specific = {
        "all_triangles",
        "antichain_width",
        "barycenter",
        "butterflies",
        "centroid",
        "connected_dominating_set",
        "floyd_warshall_tree",
        "hyper_wiener_index",
        "is_connected_dominating_set",
        "is_cover",
        "is_perfect_graph",
        "overlapping_modularity",
        "tree_centroid",
        "vf2pp_is_monomorphic",
        "vf2pp_subgraph_is_isomorphic",
    }
    for name in algorithms.__all__:
        assert hasattr(interface, name), name
        try:
            assert interface._nx_function(name) is not None, name
        except AttributeError:
            assert name in version_specific, name


# --- Shortest paths that return the paths ---------------------------------------


def ordered(value):
    """``value`` with dict order and generator contents made comparable."""
    if isinstance(value, dict):
        return [(k, ordered(v)) for k, v in value.items()]
    if isinstance(value, (list, tuple)) or hasattr(value, "__next__"):
        return [ordered(v) for v in value]
    if isinstance(value, float) and value.is_integer():
        return ("float", value)
    return value


def same_outcome(func, *args, **kwargs):
    def run(backend):
        try:
            return ("ok", ordered(func(*args, backend=backend, **kwargs)))
        except Exception as exc:
            return (type(exc), exc.args)

    ours, ref = run("rustnx"), run("networkx")
    assert ours == ref
    return ref


def path_queries(G, rng):
    nodes = list(G)
    picks = rng.sample(nodes, min(4, len(nodes))) + ["not-a-node"]
    return [(s, t) for s in picks for t in picks]


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_unweighted_paths(seed, directed):
    G = random_graph(seed, directed, "none")
    rng = random.Random(seed)
    for s, t in path_queries(G, rng):
        same_outcome(nx.bidirectional_shortest_path, G, s, t)
        same_outcome(nx.has_path, G, s, t)
        same_outcome(nx.shortest_path, G, s, t)
        same_outcome(nx.shortest_path_length, G, s, t)
    for v in list(G)[:4] + ["not-a-node"]:
        for cutoff in [None, 0, 1, 2.5]:
            same_outcome(nx.single_source_shortest_path, G, v, cutoff=cutoff)
            same_outcome(nx.single_target_shortest_path, G, v, cutoff=cutoff)
        same_outcome(nx.shortest_path, G, source=v)
        same_outcome(nx.shortest_path, G, target=v)
        same_outcome(nx.shortest_path_length, G, source=v)
        same_outcome(nx.shortest_path_length, G, target=v)
    for cutoff in [None, 1]:
        same_outcome(nx.all_pairs_shortest_path, G, cutoff=cutoff)
    same_outcome(nx.shortest_path_length, G)


@pytest.mark.parametrize("weights", WEIGHTS)
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_dijkstra_paths(seed, directed, weights):
    G = random_graph(seed, directed, weights)
    rng = random.Random(seed)
    for s, t in path_queries(G, rng):
        same_outcome(nx.dijkstra_path, G, s, t)
        same_outcome(nx.dijkstra_path_length, G, s, t)
        same_outcome(nx.single_source_dijkstra, G, s, target=t)
        same_outcome(nx.single_source_dijkstra, G, s, target=t, cutoff=2)
        same_outcome(nx.shortest_path_length, G, s, t, weight="weight")
    for v in list(G)[:4] + ["not-a-node"]:
        for cutoff in [None, 0, 1, 2.5]:
            same_outcome(nx.single_source_dijkstra_path, G, v, cutoff=cutoff)
            same_outcome(nx.single_source_dijkstra, G, v, cutoff=cutoff)
        same_outcome(nx.shortest_path, G, source=v, weight="weight")
        same_outcome(nx.shortest_path, G, target=v, weight="weight")
        same_outcome(nx.shortest_path_length, G, source=v, weight="weight")
        same_outcome(nx.shortest_path_length, G, target=v, weight="weight")
    for cutoff in [None, 2]:
        same_outcome(nx.all_pairs_dijkstra_path, G, cutoff=cutoff)
        same_outcome(nx.all_pairs_dijkstra, G, cutoff=cutoff)
    same_outcome(nx.shortest_path_length, G, weight="weight")


def test_dijkstra_path_ties_follow_networkx():
    # Equal-length routes: NetworkX keeps the route found last before the
    # node is finalized, and orders the paths dict by version.
    G = nx.DiGraph()
    G.add_weighted_edges_from([(0, 1, 5), (0, 2, 1), (2, 1, 1), (1, 3, 1), (0, 3, 3)])
    same_outcome(nx.single_source_dijkstra_path, G, 0)
    same_outcome(nx.single_source_dijkstra, G, 0)
    same_outcome(nx.dijkstra_path, G, 0, 3)


def test_paths_negative_weights():
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 1), (1, 2, -3), (2, 3, 1)])
    G.add_node(9)
    for target in [3, 9, "not-a-node"]:
        same_outcome(nx.dijkstra_path, G, 0, target)
        same_outcome(nx.dijkstra_path_length, G, 0, target)
    same_outcome(nx.single_source_dijkstra_path, G, 0)
    same_outcome(nx.all_pairs_dijkstra_path, G)
    same_outcome(nx.shortest_path, G, target=3, weight="weight")


def test_shortest_path_methods():
    G = random_graph(3, True, "int")
    same_outcome(nx.shortest_path, G, 0, 1, method="nope")
    same_outcome(nx.shortest_path_length, G, 0, 1, method="nope")
    with pytest.raises(NotImplementedError):
        nx.shortest_path(G, 0, weight="weight", method="bellman-ford", backend="rustnx")
    # weight=None ignores the method
    same_outcome(nx.shortest_path, G, 0, method="bellman-ford")


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_descendants_ancestors(seed, directed):
    G = random_graph(seed, directed, "none")
    for v in list(G)[:5] + ["not-a-node"]:
        same_outcome(nx.descendants, G, v)
        same_outcome(nx.ancestors, G, v)


# --- Clustering -----------------------------------------------------------------


def exact(func, *args, **kwargs):
    """Like ``same_outcome``, but floats must match bit for bit."""
    def run(backend):
        try:
            result = func(*args, backend=backend, **kwargs)
            if isinstance(result, dict):
                result = [(k, type(v), v) for k, v in result.items()]
            else:
                result = (type(result), result)
            return ("ok", result)
        except Exception as exc:
            return (type(exc), exc.args)

    assert run("rustnx") == run("networkx")


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_clustering(seed, directed):
    G = random_graph(seed, directed, "none")
    if not directed:
        # random_graph is sparse; add triangles.
        nodes = list(G)
        rng = random.Random(seed)
        for _ in range(len(nodes)):
            a, b, c = (rng.choice(nodes) for _ in range(3))
            G.add_edges_from([(a, b), (b, c), (c, a)])
    else:
        G.add_edges_from([(v, u) for u, v in list(G.edges)[::3]])  # reciprocal
    nodes = list(G)
    subsets = [None, nodes[0], nodes[:5], nodes[3:1:-1] + nodes[:2], ["not-a-node", nodes[-1]], []]
    for subset in subsets:
        exact(nx.clustering, G, subset)
        if not directed:
            exact(nx.triangles, G, subset)
        if subset is None or (isinstance(subset, list) and subset):
            for count_zeros in [True, False]:
                exact(nx.average_clustering, G, subset, count_zeros=count_zeros)
    exact(nx.transitivity, G)
    if directed:
        exact(nx.triangles, G)


def test_clustering_edge_cases():
    for G in [nx.Graph(), nx.DiGraph(), nx.path_graph(3), nx.complete_graph(1)]:
        exact(nx.clustering, G)
        exact(nx.transitivity, G)
        exact(nx.average_clustering, G)
    G = nx.Graph([(0, 0), (0, 1), (1, 2), (2, 0)])  # self-loop
    exact(nx.clustering, G)
    exact(nx.triangles, G)
    exact(nx.transitivity, G)


# --- Edge betweenness -----------------------------------------------------------


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_edge_betweenness(seed, directed, weights):
    G = random_graph(seed, directed, weights)
    weight = None if weights == "none" else "weight"
    for normalized in [True, False]:
        ours, ref = both(
            nx.edge_betweenness_centrality, G, normalized=normalized, weight=weight
        )
        assert_close_dicts(ours, ref)


@pytest.mark.parametrize("seed", range(20))
def test_edge_betweenness_sampled(seed):
    G = random_graph(seed, directed=seed % 2 == 0, weights="int")
    k = max(1, len(G) // 3)
    for weight in [None, "weight"]:
        ours, ref = both(nx.edge_betweenness_centrality, G, k=k, weight=weight, seed=seed)
        assert_close_dicts(ours, ref)


def test_edge_betweenness_small_graphs():
    for G in [nx.Graph(), nx.Graph([(0, 0)]), nx.path_graph(2), nx.DiGraph([(1, 0)])]:
        ours, ref = both(nx.edge_betweenness_centrality, G)
        assert_close_dicts(ours, ref)


# --- rustnx.enable() ---------------------------------------------------------------


@pytest.fixture
def restore_config():
    old = nx.config.backend_priority.algos, nx.config.fallback_to_nx
    yield
    nx.config.backend_priority.algos, nx.config.fallback_to_nx = old


@pytest.mark.parametrize("before", [[], ["networkx"], ["rustnx", "networkx"], ["networkx", "rustnx"]])
def test_enable_runs_rustnx_on_networkx_graphs(restore_config, monkeypatch, before):
    # Regression: enable() used to add "networkx" to the priority list, which
    # made NetworkX run every call on NetworkX graphs before rustnx could.
    from rustnx import interface

    calls = []
    original = interface.pagerank

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(interface, "pagerank", counting)
    nx.config.backend_priority.algos = before
    rustnx.enable()
    assert nx.config.backend_priority.algos == ["rustnx"]
    G = nx.gnm_random_graph(600, 2000, seed=1)  # above the small-graph cutoff
    assert nx.pagerank(G) == pytest.approx(nx.pagerank(G, backend="networkx"))
    assert calls, "rustnx was not used"
    # Unsupported functions still run in NetworkX.
    assert nx.is_tree(G) is False



# --- Ten more algorithms ------------------------------------------------------------


def test_py_sum_matches_builtin_sum():
    from rustnx import _core
    from rustnx.algorithms import _COMPENSATED_SUM

    rng = random.Random(0)
    for _ in range(2000):
        values = [rng.choice([1e16, -1e16, 1.0, 1e-16, 0.1, -0.0]) * rng.random() for _ in range(rng.randint(0, 30))]
        values += [rng.uniform(-1, 1) for _ in range(rng.randint(0, 5))]
        rng.shuffle(values)
        assert _core._py_sum(values, _COMPENSATED_SUM) == sum(values) or (
            math.isnan(sum(values))
        ), values


def exact_outcome(func, *args, **kwargs):
    """Both backends' results (or errors), with generators expanded and
    float values compared bit for bit, including int-vs-float."""
    def norm(value):
        if hasattr(value, "__next__"):
            out = []
            try:
                for item in value:
                    out.append(norm(item))
            except Exception as exc:
                out.append(("raised", type(exc), exc.args))
            return ("iter", out)
        if isinstance(value, dict):
            return ("dict", [(k, norm(v)) for k, v in value.items()])
        if isinstance(value, (list, tuple)):
            return (type(value).__name__, [norm(v) for v in value])
        if isinstance(value, nx.Graph):
            return ("graph", type(value).__name__, list(value.nodes(data=True)), list(value.edges(data=True)), value.graph)
        return (type(value).__name__, value)

    def run(backend):
        try:
            extra = {} if backend is None else {"backend": backend}
            return ("ok", norm(func(*args, **extra, **kwargs)))
        except Exception as exc:
            if isinstance(exc, nx.PowerIterationFailedConvergence):
                return (type(exc), str(exc))  # its args hold the exception itself
            return (type(exc), exc.args)

    ours = run("rustnx")
    if ours[0] is NotImplementedError:
        # A case rustnx hands to NetworkX: check it falls back cleanly.
        old = nx.config.backend_priority.algos
        nx.config.backend_priority.algos = ["rustnx"]
        try:
            ours = run(None)
        finally:
            nx.config.backend_priority.algos = old
    ref = run("networkx")
    assert ours == ref
    return ref


def graph_for(seed, directed, weights="none"):
    G = random_graph(seed, directed, weights)
    nodes = list(G)
    rng = random.Random(seed)
    # Denser variety: cycles, reciprocal edges.
    for _ in range(len(nodes) // 2):
        a, b = rng.choice(nodes), rng.choice(nodes)
        if a != b:
            if weights == "none":
                G.add_edge(a, b)
            elif weights == "float":
                G.add_edge(a, b, weight=rng.choice([0.5, 1.25, 2.0, 3.5]))
            else:
                G.add_edge(a, b, weight=rng.randint(1, 4))
    return G


@pytest.mark.parametrize("seed", range(20))
def test_mixed_int_float_weights_fall_back(seed, restore_config):
    # NetworkX returns int or float lengths depending on the path when
    # weights mix both; rustnx hands these calls to NetworkX.
    G = graph_for(seed, seed % 2 == 0, "int")
    for i, (u, v, d) in enumerate(G.edges(data=True)):
        if i % 3 == 0:
            d["weight"] = d["weight"] + 0.5
    nx.config.backend_priority.algos = ["rustnx"]
    nodes = list(G)
    s, t = nodes[0], nodes[-1]
    calls = [
        lambda: nx.single_source_dijkstra_path_length(G, s),
        lambda: nx.single_source_dijkstra(G, s),
        lambda: dict(nx.all_pairs_dijkstra_path_length(G)),
        lambda: nx.bidirectional_dijkstra(G, s, t),
        lambda: nx.dijkstra_path_length(G, s, t),
        lambda: nx.shortest_path_length(G, s, weight="weight"),
    ]
    for call in calls:
        try:
            ours = ("ok", ordered(call()))
        except Exception as exc:
            ours = (type(exc), exc.args)
        nx.config.backend_priority.algos = []
        try:
            try:
                ref = ("ok", ordered(call()))
            except Exception as exc:
                ref = (type(exc), exc.args)
        finally:
            nx.config.backend_priority.algos = ["rustnx"]
        assert ours == ref


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_core_number_k_core_bipartite(seed, directed):
    G = graph_for(seed, directed)
    exact_outcome(nx.core_number, G)  # self-loops raise like NetworkX
    H = G.copy()
    H.remove_edges_from(list(nx.selfloop_edges(H)))
    exact_outcome(nx.core_number, H)
    for k in [None, 0, 1, 2, 3]:
        exact_outcome(nx.k_core, H, k=k)
    exact_outcome(nx.is_bipartite, G)
    exact_outcome(nx.is_bipartite, H)
    B = nx.bipartite.random_graph(seed % 7 + 1, seed % 5 + 2, 0.4, seed=seed, directed=directed)
    exact_outcome(nx.is_bipartite, B)


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_bidirectional_dijkstra(seed, directed, weights):
    G = graph_for(seed, directed, weights)
    rng = random.Random(seed)
    for s, t in path_queries(G, rng):
        exact_outcome(nx.bidirectional_dijkstra, G, s, t)
        exact_outcome(nx.shortest_path, G, s, t, weight="weight")


def test_bidirectional_dijkstra_negative_weights():
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 1), (1, 2, -3), (2, 3, 1), (3, 4, 1)])
    for t in [2, 3, 4]:
        exact_outcome(nx.bidirectional_dijkstra, G, 0, t)


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_harmonic_centrality(seed, directed, weights):
    G = graph_for(seed, directed, weights)
    distance = None if weights == "none" else "weight"
    nodes = list(G)
    exact_outcome(nx.harmonic_centrality, G, distance=distance)
    exact_outcome(nx.harmonic_centrality, G, sources=nodes[:3], distance=distance)
    exact_outcome(nx.harmonic_centrality, G, nbunch=nodes[0])  # transposed: falls back
    exact_outcome(nx.harmonic_centrality, G, nbunch=nodes, sources=nodes[::2] + ["missing"])


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_eigenvector_and_katz(seed, directed, weights):
    G = graph_for(seed, directed, weights)
    weight = None if weights == "none" else "weight"
    exact_outcome(nx.eigenvector_centrality, G, weight=weight, max_iter=500)
    exact_outcome(nx.eigenvector_centrality, G, weight=weight, max_iter=3)  # fails to converge
    nodes = list(G)
    nstart = {v: 1 + i % 3 for i, v in enumerate(reversed(nodes))}
    exact_outcome(nx.eigenvector_centrality, G, nstart=nstart, weight=weight, max_iter=500)
    for alpha in [0.01, 0.05, 0.1]:
        for normalized in [True, False]:
            exact_outcome(nx.katz_centrality, G, alpha=alpha, weight=weight, normalized=normalized)
    exact_outcome(nx.katz_centrality, G, alpha=0.9, max_iter=50)  # may not converge


def test_spectral_edge_cases():
    for G in [nx.Graph(), nx.DiGraph(), nx.Graph([(0, 1)]), nx.empty_graph(3), nx.DiGraph([(0, 1), (1, 2)])]:
        exact_outcome(nx.eigenvector_centrality, G)
        exact_outcome(nx.katz_centrality, G)
        exact_outcome(nx.harmonic_centrality, G)
    exact_outcome(nx.eigenvector_centrality, nx.path_graph(3), nstart={0: 0, 1: 0, 2: 0})


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_traversals(seed, directed):
    G = graph_for(seed, directed)
    nodes = list(G)
    for source in nodes[:3] + ["missing"]:
        for depth in [None, 0, 1, 2]:
            exact_outcome(nx.bfs_edges, G, source, depth_limit=depth)
            exact_outcome(nx.dfs_edges, G, source, depth_limit=depth)
            exact_outcome(nx.dfs_preorder_nodes, G, source, depth_limit=depth)
        if directed:
            exact_outcome(nx.bfs_edges, G, source, reverse=True)
    for depth in [None, 1]:
        exact_outcome(nx.dfs_edges, G, depth_limit=depth)
        exact_outcome(nx.dfs_preorder_nodes, G, depth_limit=depth)


def test_traversal_graph_changed():
    G = nx.path_graph(10)
    it = nx.bfs_edges(G, 0, backend="rustnx")
    next(it)
    G.add_edge(0, 5)
    with pytest.raises(RuntimeError):
        list(it)


def test_graph_subclass_overriding_structure_is_not_converted():
    from networkx.algorithms.approximation.kcomponents import _AntiGraph

    G = nx.karate_club_graph()
    A = _AntiGraph(nx.complement(G))  # presents G itself
    with pytest.raises(NotImplementedError):
        rustnx.from_networkx(A)

    class Plain(nx.Graph):
        label = "just adds an attribute"

    rustnx.from_networkx(Plain(G))  # still converted


# --- Trees, spanning trees, all_shortest_paths, coloring, communities -------------


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_bfs_dfs_tree(seed, directed):
    G = graph_for(seed, directed)
    for source in list(G)[:3] + ["missing"]:
        for depth in [None, 1, 2]:
            exact_outcome(nx.bfs_tree, G, source, depth_limit=depth)
            exact_outcome(nx.dfs_tree, G, source, depth_limit=depth)
    exact_outcome(nx.dfs_tree, G)
    if directed:
        exact_outcome(nx.bfs_tree, G, list(G)[0], reverse=True)


@pytest.mark.parametrize("weights", WEIGHTS)
@pytest.mark.parametrize("seed", SEEDS)
def test_spanning_trees(seed, weights):
    G = graph_for(seed, False, "int" if weights == "missing" else weights)
    if weights == "missing":
        for i, (_, _, d) in enumerate(G.edges(data=True)):
            if i % 2:
                del d["weight"]
    for _, _, d in G.edges(data=True):
        d["label"] = "keep me"  # non-numeric attributes must come through
    weight = None if weights == "none" else "weight"
    for func in [nx.minimum_spanning_edges, nx.maximum_spanning_edges]:
        exact_outcome(func, G, weight=weight)
        exact_outcome(func, G, weight=weight, data=False)
        exact_outcome(func, G, algorithm="prim", weight=weight)  # falls back
        exact_outcome(func, G, algorithm="nope")
    for func in [nx.minimum_spanning_tree, nx.maximum_spanning_tree]:
        exact_outcome(func, G, weight=weight)
    # The yielded data dicts are the graph's own, as in NetworkX.
    edges = list(nx.minimum_spanning_edges(G, backend="rustnx"))
    if edges:
        u, v, d = edges[0]
        assert d is G[u][v]
    exact_outcome(nx.minimum_spanning_edges, G.to_directed())


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_all_shortest_paths(seed, directed, weights):
    G = graph_for(seed, directed, weights)
    weight = None if weights == "none" else "weight"
    rng = random.Random(seed)
    for s, t in path_queries(G, rng):
        exact_outcome(nx.all_shortest_paths, G, s, t, weight=weight)
    nodes = list(G)
    if nodes:
        exact_outcome(nx.all_shortest_paths, G, nodes[0], nodes[-1], weight="weight", method="nope")


def test_all_shortest_paths_ties_and_zero_weights():
    G = nx.grid_2d_graph(4, 4)  # many equal-length paths
    exact_outcome(nx.all_shortest_paths, G, (0, 0), (3, 3))
    H = nx.DiGraph()
    H.add_weighted_edges_from([(0, 1, 0), (1, 0, 0), (1, 2, 1), (0, 2, 1), (2, 3, 0)])
    exact_outcome(nx.all_shortest_paths, H, 0, 3, weight="weight")
    exact_outcome(nx.all_shortest_paths, H, 3, 0, weight="weight")


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_greedy_color_and_label_propagation(seed, directed):
    G = graph_for(seed, directed)
    exact_outcome(nx.greedy_color, G)
    exact_outcome(nx.greedy_color, G, strategy="smallest_last")  # falls back
    if not directed:
        def communities(G, **kw):
            return [sorted(map(str, c)) for c in nx.community.label_propagation_communities(G, **kw)]

        exact_outcome(communities, G)
        exact_outcome(lambda G, **kw: list(nx.community.label_propagation_communities(G, **kw)), G)
    if directed:
        exact_outcome(nx.community.label_propagation_communities, G)


def test_spanning_tree_algorithm_names():
    G = nx.path_graph(5)
    for name in ["kruskal", "prim", "boruvka", "borůvka", "nope"]:
        exact_outcome(lambda G, **kw: list(nx.minimum_spanning_edges(G, algorithm=name, **kw)), G)


# --- Multigraphs ------------------------------------------------------------------


def random_multigraph(seed, directed, weights):
    rng = random.Random(seed)
    base = graph_for(seed, directed, "none")
    M = nx.MultiDiGraph() if directed else nx.MultiGraph()
    M.add_nodes_from(base)
    for u, v in base.edges():
        for _ in range(rng.choice([1, 1, 2, 3])):  # parallel edges
            if weights == "none":
                M.add_edge(u, v)
            elif weights == "int":
                M.add_edge(u, v, weight=rng.randint(1, 5))
            else:
                M.add_edge(u, v, weight=rng.choice([0.5, 1.25, 2.0, 3.5]))
    return M


MULTI_PAIR_FUNCS = [
    nx.dijkstra_path, nx.dijkstra_path_length, nx.bidirectional_dijkstra,
    nx.bidirectional_shortest_path, nx.has_path,
    lambda G, s, t, **kw: nx.shortest_path(G, s, t, weight="weight", **kw),
    lambda G, s, t, **kw: list(nx.all_shortest_paths(G, s, t, weight="weight", **kw)),
    lambda G, s, t, **kw: list(nx.all_shortest_paths(G, s, t, **kw)),
]
MULTI_SOURCE_FUNCS = [
    nx.single_source_shortest_path_length, nx.single_source_dijkstra_path_length,
    nx.single_source_shortest_path, nx.single_target_shortest_path,
    nx.single_source_dijkstra_path, nx.single_source_dijkstra,
    nx.descendants, nx.ancestors,
    lambda G, s, **kw: list(nx.bfs_edges(G, s, **kw)),
    lambda G, s, **kw: list(nx.dfs_edges(G, s, **kw)),
    lambda G, s, **kw: list(nx.dfs_preorder_nodes(G, s, **kw)),
    nx.bfs_tree, nx.dfs_tree,
]
MULTI_GRAPH_FUNCS = [
    lambda G, **kw: dict(nx.all_pairs_shortest_path_length(G, **kw)),
    lambda G, **kw: dict(nx.all_pairs_dijkstra_path_length(G, **kw)),
    lambda G, **kw: dict(nx.all_pairs_shortest_path(G, **kw)),
    lambda G, **kw: dict(nx.all_pairs_dijkstra_path(G, **kw)),
    lambda G, **kw: dict(nx.all_pairs_dijkstra(G, **kw)),
    lambda G, **kw: nx.betweenness_centrality(G, **kw),
    lambda G, **kw: nx.betweenness_centrality(G, weight="weight", **kw),
    lambda G, **kw: nx.closeness_centrality(G, **kw),
    lambda G, **kw: nx.closeness_centrality(G, distance="weight", **kw),
    lambda G, **kw: nx.harmonic_centrality(G, distance="weight", **kw),
    nx.is_bipartite,
    # Not supported for multigraphs: must fall back cleanly.
    lambda G, **kw: nx.pagerank(G, **kw),
    lambda G, **kw: nx.pagerank(G, weight="weight", **kw),
    lambda G, **kw: nx.greedy_color(G, **kw),
]


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_multigraphs(seed, directed, weights, restore_config):
    M = random_multigraph(seed, directed, weights)
    rng = random.Random(seed)
    for s, t in path_queries(M, rng):
        for func in MULTI_PAIR_FUNCS:
            exact_outcome(func, M, s, t)
    for s in list(M)[:3]:
        for func in MULTI_SOURCE_FUNCS:
            exact_outcome(func, M, s)
    for func in MULTI_GRAPH_FUNCS:
        exact_outcome(func, M)
    if directed:
        for func in [nx.strongly_connected_components, nx.weakly_connected_components]:
            exact_outcome(lambda G, **kw: list(func(G, **kw)), M)
    else:
        exact_outcome(lambda G, **kw: list(nx.connected_components(G, **kw)), M)


@pytest.mark.parametrize("weights", ["none", "int"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_multigraph_distance_measures(seed, directed, weights):
    M = random_multigraph(seed, directed, weights)
    for u, v in zip(list(M), list(M)[1:] + list(M)[:1]):
        M.add_edge(u, v, **({} if weights == "none" else {"weight": 2}))  # connected
    weight = None if weights == "none" else "weight"
    for func in DISTANCE_FUNCS:
        exact_outcome(func, M, weight=weight)


def test_multigraph_weight_types_and_none():
    M = nx.MultiGraph()
    M.add_edge(0, 1, weight=2)
    M.add_edge(0, 1, weight=2.0)  # min keeps the first: the int 2
    M.add_edge(1, 2, weight=1.5)
    M.add_edge(2, 3, weight=None)  # a single None edge is hidden
    for t in [1, 2, 3]:
        exact_outcome(nx.dijkstra_path_length, M, 0, t)
        exact_outcome(nx.single_source_dijkstra_path_length, M, 0)
    M.add_edge(2, 3, weight=1)  # None mixed with numbers: NetworkX raises
    exact_outcome(nx.single_source_dijkstra_path_length, M, 0)


def test_cached_multigraph_conversion_not_reused(restore_config):
    M = random_multigraph(3, False, "int")
    M = nx.MultiGraph(M)
    nx.config.backend_priority.algos = ["rustnx"]
    nx.single_source_shortest_path_length(M, list(M)[0])  # caches a conversion
    nx.config.backend_priority.algos = []
    ref = nx.pagerank(M)
    nx.config.backend_priority.algos = ["rustnx"]
    assert nx.pagerank(M) == ref  # pagerank sums parallel weights: NetworkX runs it


def test_version_numbers_agree():
    # pyproject.toml decides the PyPI version; the others must match it
    # (Cargo needs the semver spelling, e.g. 0.1.0a3 -> 0.1.0-alpha.3).
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    pyproject = re.search(r'^version = "(.+)"', (root / "pyproject.toml").read_text(), re.M)
    cargo = re.search(r'^version = "(.+)"', (root / "Cargo.toml").read_text(), re.M)
    if not (pyproject and cargo):
        pytest.skip("source tree not available")
    py_version = pyproject.group(1)
    m = re.fullmatch(r"(\d+\.\d+\.\d+)(?:(a|b|rc)(\d+))?", py_version)
    assert m, py_version
    labels = {"a": "alpha", "b": "beta", "rc": "rc"}
    expected_cargo = m.group(1) + (f"-{labels[m.group(2)]}.{m.group(3)}" if m.group(2) else "")
    assert cargo.group(1) == expected_cargo
    assert rustnx.__version__ == py_version
