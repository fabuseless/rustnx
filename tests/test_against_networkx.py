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
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        assert pool.apply(_pagerank_in_worker, (G,)) == expected


def test_malformed_pickle_data_rejected():
    from rustnx import _core

    with pytest.raises(ValueError, match="malformed"):
        _core._core_graph_from_bytes(b"RNX1" + b"\xff" * 20)
