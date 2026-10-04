"""Functions added in the NetworkX coverage batches (todo items 28 to 50).

Same approach as test_against_networkx.py: run each function with
``backend="rustnx"`` and ``backend="networkx"`` and require identical
results, including the iteration order of returned sets.
"""

import inspect
import random
import warnings

import networkx as nx
from networkx.algorithms.coloring.equitable_coloring import is_coloring, is_equitable
import pytest

from test_against_networkx import exact_outcome, graph_for, random_multigraph


def with_set_order(func):
    """Wrap ``func`` so sets in its result become lists in iteration order
    (``exact_outcome`` alone compares sets by equality)."""

    def convert(value):
        if isinstance(value, (set, frozenset)):
            return ("set", list(value))
        if hasattr(value, "__next__"):
            return [convert(v) for v in value]
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(convert(v) for v in value)
        return value

    def run(*args, **kwargs):
        return convert(func(*args, **kwargs))

    return run


def listed(func):
    return with_set_order(lambda *a, **kw: list(func(*a, **kw)))


def sources_for(G, seed):
    rng = random.Random(seed)
    nodes = list(G)
    return rng.sample(nodes, min(3, len(nodes))) + ["missing"]


# --- Batch 1: traversal, components, degree centrality, trees ---------------------


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch1_traversal(seed, directed):
    G = graph_for(seed, directed)
    for source in sources_for(G, seed):
        for depth in [None, 0, 1, 2]:
            exact_outcome(listed(nx.bfs_successors), G, source, depth_limit=depth)
            exact_outcome(listed(nx.dfs_postorder_nodes), G, source, depth_limit=depth)
            exact_outcome(nx.dfs_predecessors, G, source, depth_limit=depth)
            exact_outcome(nx.dfs_successors, G, source, depth_limit=depth)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                exact_outcome(listed(nx.bfs_predecessors), G, source, depth_limit=depth)
        exact_outcome(listed(nx.bfs_layers), G, source)
        for distance in [0, 1, 2, 3, 100, -1, 2.0, 2.5, True]:
            exact_outcome(with_set_order(nx.descendants_at_distance), G, source, distance)
    for depth in [None, 1, 3]:
        exact_outcome(listed(nx.dfs_postorder_nodes), G, depth_limit=depth)
        exact_outcome(nx.dfs_predecessors, G, depth_limit=depth)
        exact_outcome(nx.dfs_successors, G, depth_limit=depth)
    nodes = list(G)
    exact_outcome(listed(nx.bfs_layers), G, nodes[:3])
    exact_outcome(listed(nx.bfs_layers), G, nodes[:2] + nodes[:2])  # repeats
    # An iterator (a fresh one for each backend).
    exact_outcome(with_set_order(lambda G, **kw: list(nx.bfs_layers(G, iter(nodes[:2]), **kw))), G)
    exact_outcome(listed(nx.bfs_layers), G, [])
    exact_outcome(listed(nx.bfs_layers), G, nodes[:1] + ["missing"])


def test_bfs_predecessors_warns_like_networkx():
    G = nx.path_graph(600)

    def warning_messages(backend):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = list(nx.bfs_predecessors(G, 0, backend=backend))
        return result, [(w.category, str(w.message)) for w in caught]

    assert warning_messages("rustnx") == warning_messages("networkx")


@pytest.mark.parametrize("seed", range(60))
def test_batch1_components(seed):
    G = graph_for(seed, False)
    rng = random.Random(seed)
    if seed % 3 == 0:  # a few disjoint cycles and bridges
        G.add_edges_from(nx.cycle_graph(5).edges)
        G.add_edge(0, list(G)[0])
    exact_outcome(listed(nx.articulation_points), G)
    exact_outcome(listed(nx.biconnected_components), G)
    exact_outcome(listed(nx.biconnected_component_edges), G)
    exact_outcome(nx.is_biconnected, G)
    for n in rng.sample(list(G), min(3, len(G))) + ["missing", [1]]:
        exact_outcome(with_set_order(nx.node_connected_component), G, n)
    D = graph_for(seed, True)
    exact_outcome(listed(nx.attracting_components), D)
    exact_outcome(listed(nx.strongly_connected_components), D)  # set order too
    exact_outcome(nx.number_attracting_components, D)
    exact_outcome(nx.is_attracting_component, D)
    # Directed graphs are rejected (and undirected ones for attracting).
    exact_outcome(nx.is_biconnected, D)
    exact_outcome(nx.is_attracting_component, G)


@pytest.mark.parametrize(
    "G",
    [
        nx.empty_graph(0),
        nx.empty_graph(1),
        nx.path_graph(2),
        nx.complete_graph(4),
        nx.barbell_graph(4, 2),
        nx.Graph([(0, 0), (0, 1)]),
        nx.Graph([(0, 0)]),
        nx.DiGraph([(0, 1), (1, 0), (1, 2)]),
        nx.DiGraph([(0, 0)]),
    ],
    ids=lambda G: f"{type(G).__name__}{list(G.edges)}",
)
def test_batch1_small_cases(G):
    funcs = [nx.is_tree, nx.is_forest, nx.degree_centrality, nx.in_degree_centrality,
             nx.out_degree_centrality, nx.is_biconnected, nx.is_attracting_component,
             nx.number_attracting_components, listed(nx.articulation_points),
             listed(nx.biconnected_component_edges), listed(nx.attracting_components)]
    for func in funcs:
        exact_outcome(func, G)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(60))
def test_batch1_degree_centrality_and_trees(seed, directed):
    G = graph_for(seed, directed)
    exact_outcome(nx.degree_centrality, G)
    exact_outcome(nx.in_degree_centrality, G)
    exact_outcome(nx.out_degree_centrality, G)
    exact_outcome(nx.is_tree, G)
    exact_outcome(nx.is_forest, G)
    # Trees and forests: a random spanning forest of G.
    F = G.__class__()
    F.add_nodes_from(G)
    F.add_edges_from(nx.bfs_edges(G, list(G)[0], backend="networkx"))
    exact_outcome(nx.is_tree, F)
    exact_outcome(nx.is_forest, F)
    F.add_edges_from(list(F.edges)[:1])  # unchanged, already present
    first = list(F)[0]
    F.add_edge(first, first)  # a self-loop breaks both
    exact_outcome(nx.is_tree, F)
    exact_outcome(nx.is_forest, F)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(20))
def test_batch1_multigraphs(seed, directed, restore_config):
    M = random_multigraph(seed, directed, "none")
    source = list(M)[0]
    exact_outcome(listed(nx.bfs_layers), M, source)
    exact_outcome(listed(nx.bfs_successors), M, source)
    exact_outcome(listed(nx.dfs_postorder_nodes), M)
    exact_outcome(nx.dfs_successors, M)
    exact_outcome(with_set_order(nx.descendants_at_distance), M, source, 2)
    if directed:
        exact_outcome(listed(nx.attracting_components), M)
    else:
        exact_outcome(listed(nx.biconnected_components), M)
        exact_outcome(listed(nx.articulation_points), M)
        exact_outcome(with_set_order(nx.node_connected_component), M, source)
    # Parallel edges change degrees and edge counts: these fall back.
    exact_outcome(nx.degree_centrality, M)
    exact_outcome(nx.is_tree, M)
    exact_outcome(nx.is_forest, M)


@pytest.fixture
def restore_config():
    old = nx.config.backend_priority.algos, nx.config.fallback_to_nx
    yield
    nx.config.backend_priority.algos, nx.config.fallback_to_nx = old


def test_batch1_graph_changes_during_iteration():
    for func in [nx.articulation_points, nx.biconnected_components]:
        G = nx.barbell_graph(5, 3)
        it = func(G, backend="rustnx")
        next(it)
        G.add_edge("x", "y")
        with pytest.raises(RuntimeError):
            list(it)
    # attracting_components computes everything when iteration starts.
    for backend in ["rustnx", "networkx"]:
        D = nx.DiGraph([(0, 1), (2, 3), (4, 5)])
        it = nx.attracting_components(D, backend=backend)
        D.add_edge(1, 0)  # before the first item: seen by NetworkX
        first = next(it)
        D.add_edge(3, 2)  # after: not seen
        assert [first, *it] == [{0, 1}, {3}, {5}]


# --- Batch 2: shortest paths -----------------------------------------------------


def weighted_graph(seed, directed, weights):
    """``graph_for`` with ``weights`` "none", "int", "float", "negative"
    (ints, some below zero) or "zero" (ints, a few of them zero)."""
    G = graph_for(seed, directed, "int" if weights in ("negative", "zero") else weights)
    rng = random.Random(seed)
    for u, v, d in G.edges(data=True):
        if weights == "negative" and rng.random() < 0.15:
            d["weight"] = -rng.randint(1, 3)
    if weights == "zero":
        # A few only: zero-weight cycles multiply the number of shortest paths.
        edges = list(G.edges(data=True))
        for u, v, d in rng.sample(edges, min(len(edges), 4)):
            d["weight"] = 0
    return G


def node_choices(G, seed):
    rng = random.Random(seed + 1)
    nodes = list(G)
    return rng.sample(nodes, min(3, len(nodes)))


B2_WEIGHTS = ["none", "int", "float", "zero"]


def quiet(func):
    """Ignore the FutureWarning NetworkX 3.4 gives for
    ``single_target_shortest_path_length``."""

    def run(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return func(*args, **kwargs)

    return run


@pytest.mark.parametrize("weights", B2_WEIGHTS + ["negative"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch2_multi_source_dijkstra(seed, directed, weights):
    G = weighted_graph(seed, directed, weights)
    picks = node_choices(G, seed)
    for sources in [picks[:1], picks, picks[:2] + picks[:1], set(picks), [], ["missing"]]:
        for cutoff in [None, 2, 0]:
            exact_outcome(nx.multi_source_dijkstra_path_length, G, sources, cutoff=cutoff)
            exact_outcome(nx.multi_source_dijkstra_path, G, sources, cutoff=cutoff)
            exact_outcome(nx.multi_source_dijkstra, G, sources, cutoff=cutoff)
        for target in list(G)[-2:] + picks[:1] + ["missing"]:
            exact_outcome(nx.multi_source_dijkstra, G, sources, target=target)
            exact_outcome(nx.multi_source_dijkstra, G, sources, target=target, cutoff=2)
    exact_outcome(nx.multi_source_dijkstra_path_length, G, picks, weight=None)


@pytest.mark.parametrize("weights", B2_WEIGHTS + ["negative"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch2_predecessors(seed, directed, weights):
    G = weighted_graph(seed, directed, weights)
    for source in node_choices(G, seed) + ["missing"]:
        for cutoff in [None, 0, 1, 2.5, -1]:
            exact_outcome(nx.dijkstra_predecessor_and_distance, G, source, cutoff=cutoff)
            exact_outcome(nx.predecessor, G, source, cutoff=cutoff)
            exact_outcome(nx.predecessor, G, source, cutoff=cutoff, return_seen=True)
            exact_outcome(quiet(listed(nx.single_target_shortest_path_length)), G, source, cutoff=cutoff)
        for target in list(G)[-2:] + ["missing"]:
            exact_outcome(nx.predecessor, G, source, target=target)
            exact_outcome(nx.predecessor, G, source, target=target, return_seen=True)


def all_paths(G, source, **kw):
    return list(nx.single_source_all_shortest_paths(G, source, **kw))


def all_pairs_paths(G, **kw):
    return list(nx.all_pairs_all_shortest_paths(G, **kw))


@pytest.mark.parametrize("weights", B2_WEIGHTS + ["negative"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch2_all_shortest_paths(seed, directed, weights):
    G = weighted_graph(seed, directed, weights)
    weight = None if weights == "none" else "weight"
    for source in node_choices(G, seed) + ["missing", [1]]:
        for method in ["dijkstra", "bellman-ford", "other"]:
            exact_outcome(all_paths, G, source, weight=weight, method=method)
        exact_outcome(all_paths, G, source)
    if len(G) <= 20:
        for method in ["dijkstra", "bellman-ford", "other"]:
            exact_outcome(all_pairs_paths, G, weight=weight, method=method)


@pytest.mark.parametrize("weights", B2_WEIGHTS + ["negative"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch2_bellman_ford(seed, directed, weights):
    G = weighted_graph(seed, directed, weights)
    weight = "weight" if seed % 4 else None
    picks = node_choices(G, seed)
    for source in picks + ["missing", [1]]:
        exact_outcome(nx.single_source_bellman_ford_path_length, G, source, weight=weight)
        exact_outcome(nx.single_source_bellman_ford_path, G, source, weight=weight)
        exact_outcome(nx.single_source_bellman_ford, G, source, weight=weight)
        exact_outcome(nx.find_negative_cycle, G, source, weight=weight)
        for heuristic in [False, True]:
            exact_outcome(nx.bellman_ford_predecessor_and_distance, G, source, weight=weight, heuristic=heuristic)
        for target in list(G)[-2:] + picks[:1] + ["missing", [1]]:
            exact_outcome(nx.bellman_ford_path, G, source, target, weight=weight)
            exact_outcome(nx.bellman_ford_path_length, G, source, target, weight=weight)
            exact_outcome(nx.single_source_bellman_ford, G, source, target=target, weight=weight)
    exact_outcome(listed(nx.all_pairs_bellman_ford_path_length), G, weight=weight)
    exact_outcome(listed(nx.all_pairs_bellman_ford_path), G, weight=weight)
    for heuristic in [False, True]:
        exact_outcome(nx.negative_edge_cycle, G, weight=weight, heuristic=heuristic)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch2_negative_cycles(seed, directed):
    # Mostly positive weights with a few negative edges, so that some graphs
    # have negative cycles and some don't.
    G = weighted_graph(seed, directed, "int")
    rng = random.Random(seed)
    edges = list(G.edges(data=True))
    for u, v, d in rng.sample(edges, min(len(edges), rng.randint(0, 3))):
        d["weight"] = -rng.randint(1, 6)
    if seed % 5 == 0 and len(G) > 1:
        a = list(G)[0]
        G.add_edge(a, a, weight=-1 if seed % 10 == 0 else 0)
    for heuristic in [False, True]:
        exact_outcome(nx.negative_edge_cycle, G, heuristic=heuristic)
    for source in list(G)[:6]:
        exact_outcome(nx.find_negative_cycle, G, source)
        exact_outcome(nx.bellman_ford_predecessor_and_distance, G, source)
        exact_outcome(nx.single_source_bellman_ford, G, source)
        exact_outcome(all_paths, G, source, weight="weight", method="bellman-ford")


@pytest.mark.parametrize("weights", B2_WEIGHTS)
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch2_astar(seed, directed, weights):
    G = weighted_graph(seed, directed, weights)
    picks = node_choices(G, seed)
    for source in picks + ["missing"]:
        for target in list(G)[-3:] + picks[:1] + ["missing"]:
            for cutoff in [None, 0, 2, 3.5]:
                exact_outcome(nx.astar_path, G, source, target, cutoff=cutoff)
                exact_outcome(nx.astar_path_length, G, source, target, cutoff=cutoff)
            exact_outcome(nx.astar_path_length, G, source, target, weight=None)
            exact_outcome(nx.astar_path, G, source, target, heuristic=lambda u, v: 0)


@pytest.mark.parametrize("seed", range(10))
def test_batch2_hidden_and_mixed_weights(seed):
    G = weighted_graph(seed, seed % 2 == 0, "int")
    for i, (u, v, d) in enumerate(G.edges(data=True)):
        if i % 4 == 0:
            d["weight"] = None
        elif i % 4 == 1:
            d["weight"] = 1.5
    source, target = list(G)[0], list(G)[-1]
    funcs = [
        lambda G, **kw: nx.multi_source_dijkstra(G, [source], **kw),
        lambda G, **kw: nx.dijkstra_predecessor_and_distance(G, source, **kw),
        lambda G, **kw: nx.astar_path(G, source, target, **kw),
        lambda G, **kw: nx.astar_path_length(G, source, target, **kw),
        lambda G, **kw: nx.single_source_bellman_ford(G, source, **kw),
        lambda G, **kw: nx.negative_edge_cycle(G, **kw),
        lambda G, **kw: all_paths(G, source, weight="weight", **kw),
    ]
    for func in funcs:
        exact_outcome(func, G)


@pytest.mark.parametrize(
    "G",
    [
        nx.empty_graph(0),
        nx.empty_graph(1),
        nx.Graph([(0, 0)]),
        nx.DiGraph([(0, 0)]),
        nx.path_graph(3),
        nx.DiGraph([(0, 1), (1, 2), (2, 0)]),
    ],
    ids=lambda G: f"{type(G).__name__}{list(G.edges)}",
)
@pytest.mark.parametrize("w", [0, 1, -1, 2.5])
def test_batch2_small_cases(G, w):
    G = G.copy()
    nx.set_edge_attributes(G, w, "weight")
    nodes = list(G) or [0]
    s, t = nodes[0], nodes[-1]
    calls = [
        lambda G, **kw: nx.multi_source_dijkstra(G, nodes, **kw),
        lambda G, **kw: nx.dijkstra_predecessor_and_distance(G, s, **kw),
        lambda G, **kw: nx.predecessor(G, s, **kw),
        lambda G, **kw: nx.bellman_ford_predecessor_and_distance(G, s, **kw),
        lambda G, **kw: nx.single_source_bellman_ford(G, s, **kw),
        lambda G, **kw: nx.bellman_ford_path_length(G, s, t, **kw),
        lambda G, **kw: nx.negative_edge_cycle(G, **kw),
        lambda G, **kw: nx.find_negative_cycle(G, s, **kw),
        lambda G, **kw: nx.astar_path_length(G, s, t, **kw),
        lambda G, **kw: all_paths(G, s, weight="weight", method="bellman-ford", **kw),
        lambda G, **kw: all_pairs_paths(G, weight="weight", **kw),
        lambda G, **kw: all_pairs_paths(G, weight="weight", method="other", **kw),
        lambda G, **kw: list(nx.all_pairs_bellman_ford_path_length(G, **kw)),
    ]
    for call in calls:
        exact_outcome(call, G)


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(15))
def test_batch2_multigraphs(seed, directed, weights):
    M = random_multigraph(seed, directed, weights)
    picks = node_choices(M, seed)
    s, t = picks[0], list(M)[-1]
    exact_outcome(nx.multi_source_dijkstra, M, picks)
    exact_outcome(nx.dijkstra_predecessor_and_distance, M, s)
    exact_outcome(nx.predecessor, M, s)
    exact_outcome(quiet(listed(nx.single_target_shortest_path_length)), M, s)
    exact_outcome(all_paths, M, s, weight="weight")
    exact_outcome(nx.single_source_bellman_ford, M, s)
    exact_outcome(nx.bellman_ford_predecessor_and_distance, M, s)
    exact_outcome(nx.negative_edge_cycle, M)
    exact_outcome(nx.find_negative_cycle, M, s)
    exact_outcome(nx.astar_path_length, M, s, t)
    exact_outcome(listed(nx.all_pairs_bellman_ford_path), M)


def test_batch2_runs_in_rust():
    """The common cases really run in rustnx (``exact_outcome`` alone would
    also pass if they all fell back)."""
    for directed in [False, True]:
        for weights in ["int", "float"]:
            G = weighted_graph(3, directed, weights)
            s, t = list(G)[0], list(G)[-1]
            calls = [
                lambda: nx.multi_source_dijkstra(G, [s, t], backend="rustnx"),
                lambda: nx.multi_source_dijkstra(G, [s], target=t, backend="rustnx"),
                lambda: nx.multi_source_dijkstra_path_length(G, [s], backend="rustnx"),
                lambda: nx.dijkstra_predecessor_and_distance(G, s, backend="rustnx"),
                lambda: nx.predecessor(G, s, backend="rustnx"),
                lambda: quiet(nx.single_target_shortest_path_length)(G, s, backend="rustnx"),
                lambda: list(nx.single_source_all_shortest_paths(G, s, weight="weight", backend="rustnx")),
                lambda: list(nx.all_pairs_all_shortest_paths(G, backend="rustnx")),
                lambda: nx.single_source_bellman_ford(G, s, backend="rustnx"),
                lambda: nx.bellman_ford_path(G, s, t, backend="rustnx"),
                lambda: nx.bellman_ford_path_length(G, s, t, backend="rustnx"),
                lambda: nx.bellman_ford_predecessor_and_distance(G, s, backend="rustnx"),
                lambda: list(nx.all_pairs_bellman_ford_path(G, backend="rustnx")),
                lambda: nx.negative_edge_cycle(G, backend="rustnx"),
                lambda: nx.astar_path_length(G, s, t, backend="rustnx"),
            ]
            for call in calls:
                try:
                    call()
                except nx.NetworkXException:
                    pass
            with pytest.raises(nx.NetworkXError):
                nx.find_negative_cycle(G, s, backend="rustnx")


def test_batch2_graph_changes_during_iteration():
    for backend in ["rustnx", "networkx"]:
        G = nx.path_graph(5)
        it = nx.single_source_all_shortest_paths(G, 0, backend=backend)
        G.add_edge(0, 4)  # before iteration starts: seen
        assert dict(it)[4] == [[0, 4]]
    G = nx.path_graph(5)
    it = nx.all_pairs_bellman_ford_path_length(G, backend="rustnx")
    next(it)
    G.add_edge(0, 4)
    with pytest.raises(RuntimeError):
        list(it)


def test_batch2_single_target_length_type():
    G = nx.path_graph(600)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ours = nx.single_target_shortest_path_length(G, 0, backend="rustnx")
        ref = nx.single_target_shortest_path_length(G, 0, backend="networkx")
    assert type(ours) is dict or not isinstance(ref, dict)
    assert isinstance(ours, dict) == isinstance(ref, dict)
    assert list(dict(ours).items()) == list(dict(ref).items())
    messages = [(w.category, str(w.message)) for w in caught]
    assert messages[: len(messages) // 2] == messages[len(messages) // 2 :]


def test_batch2_equal_nodes_of_other_types():
    # NetworkX puts the caller's object (1.0) in some results and the graph's
    # (1) in others; rustnx hands these calls to NetworkX.
    G = nx.path_graph(4)
    nx.set_edge_attributes(G, 1, "weight")
    calls = [
        lambda G, **kw: nx.multi_source_dijkstra(G, [1.0], **kw),
        lambda G, **kw: nx.multi_source_dijkstra(G, [0], target=3.0, **kw),
        lambda G, **kw: nx.astar_path(G, 0, 3.0, **kw),
        lambda G, **kw: nx.single_source_bellman_ford(G, True, **kw),
        lambda G, **kw: nx.predecessor(G, 1.0, **kw),
        lambda G, **kw: nx.dijkstra_predecessor_and_distance(G, 1.0, **kw),
    ]
    for call in calls:
        exact_outcome(call, G)
        with pytest.raises(NotImplementedError):
            call(G, backend="rustnx")


def test_batch2_probes_do_not_dispatch(restore_config):
    # NetworkX's single_source_dijkstra_path calls the dispatchable
    # multi_source_dijkstra_path; the version probes must not run through
    # rustnx again (that recursed).
    from rustnx import algorithms

    probes = [algorithms._dijkstra_paths_in_pop_order, algorithms._path_skip_ends_pass,
              algorithms._all_paths_over_pred]
    for probe in probes:
        probe.cache_clear()
    nx.config.backend_priority.algos = ["rustnx"]
    G = nx.path_graph(5)
    nx.set_edge_attributes(G, 1, "weight")
    assert nx.multi_source_dijkstra_path(G, [0], backend="rustnx")[4] == [0, 1, 2, 3, 4]
    assert list(nx.single_source_all_shortest_paths(G, 0, weight="weight", backend="rustnx"))[-1] == (4, [[0, 1, 2, 3, 4]])


# --- Batch 3: DAGs, traversal and components -------------------------------------


from itertools import islice


def graph_parts(func):
    """Wrap a function returning a graph: compare its nodes, adjacency and
    predecessor order, data and graph attributes, with set order."""

    def run(*args, **kwargs):
        G = func(*args, **kwargs)
        parts = [type(G).__name__, list(G.nodes(data=True)), list(G.edges(data=True)),
                 dict(G.graph), [(n, list(G.adj[n])) for n in G]]
        if G.is_directed():
            parts.append([(n, list(G.pred[n])) for n in G])
        return parts

    return with_set_order(run)


def first(func, k=300):
    return with_set_order(lambda *a, **kw: list(islice(func(*a, **kw), k)))


def random_dag(seed, weights="none", small=False):
    """A DAG on shuffled nodes: graph_for's edges oriented by a random order."""
    rng = random.Random(seed)
    G = graph_for(seed, True, weights)
    order = list(G)
    rng.shuffle(order)
    rank = {v: i for i, v in enumerate(order)}
    D = nx.DiGraph()
    D.add_nodes_from(G)
    for u, v, d in G.edges(data=True):
        if rank[u] < rank[v]:
            D.add_edge(u, v, **d)
        elif rank[v] < rank[u] and rng.random() < 0.5:
            D.add_edge(v, u, **d)
    if small:
        keep = list(D)[: rng.randint(0, 9)]
        D = D.subgraph(keep).copy()
    return D


DAG_ONLY = ["colliders", "v_structures", "root_to_leaf_paths", "has_cycle"]


def dag_func(name):
    return getattr(nx.dag, name)


def runs_in_rustnx(func, *args, **kwargs):
    """Call with backend="rustnx" (no fallback) and expand generators."""
    result = func(*args, backend="rustnx", **kwargs)
    if hasattr(result, "__next__"):
        result = list(result)
    return result


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("seed", range(40))
def test_batch3_dags(seed, weights):
    D = random_dag(seed, weights)
    C = graph_for(seed, True, weights)  # usually cyclic
    U = graph_for(seed, False, weights)
    for G in [D, C, U]:
        exact_outcome(dag_func("has_cycle"), G)
        for kw in [{}, {"default_weight": 2}, {"default_weight": 2.5}, {"weight": None},
                   {"weight": None, "default_weight": 0.5}, {"weight": "nope"}]:
            exact_outcome(nx.dag_longest_path, G, **kw)
            exact_outcome(nx.dag_longest_path_length, G, **kw)
        exact_outcome(listed(nx.lexicographical_topological_sort), G)
        exact_outcome(graph_parts(nx.transitive_reduction), G)
        exact_outcome(graph_parts(nx.transitive_closure_dag), G)
        for reflexive in [False, True, None, 1]:
            exact_outcome(graph_parts(nx.transitive_closure), G, reflexive)
        exact_outcome(listed(dag_func("colliders")), G)  # falls back
        exact_outcome(listed(dag_func("v_structures")), G)
        exact_outcome(first(dag_func("root_to_leaf_paths")), G)
    S = random_dag(seed, weights, small=True)
    for G in [S, C.subgraph(list(C)[:6]).copy(), U]:
        exact_outcome(first(nx.all_topological_sorts, 2000), G)
        exact_outcome(graph_parts(nx.dag_to_branching), G)
        exact_outcome(listed(dag_func("root_to_leaf_paths")), G)
    if weights == "none":
        assert runs_in_rustnx(nx.dag_longest_path, D) == nx.dag_longest_path(D, backend="networkx")
        for name in ["has_cycle", "v_structures"]:
            runs_in_rustnx(dag_func(name), D)
        next(dag_func("root_to_leaf_paths")(D, backend="rustnx"), None)
        for func in [nx.transitive_reduction, nx.transitive_closure, nx.transitive_closure_dag,
                     nx.dag_longest_path_length]:
            runs_in_rustnx(func, D)
        runs_in_rustnx(nx.dag_to_branching, S)
        if all(isinstance(v, int) for v in D) or all(isinstance(v, str) for v in D):
            runs_in_rustnx(nx.lexicographical_topological_sort, D)


def test_batch3_dag_weights():
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=-3)
    G.add_edge(1, 2, weight=0)
    G.add_edge(0, 2, weight=0.0)
    G.add_edge(2, 3, weight=True)
    G.add_edge(4, 3)  # missing weight
    G.add_edge(3, 5, weight=2**52)
    for kw in [{}, {"default_weight": 7}, {"default_weight": -1}]:
        exact_outcome(nx.dag_longest_path, G, **kw)
        exact_outcome(nx.dag_longest_path_length, G, **kw)
    exact_outcome(nx.dag_longest_path, G, topo_order=[4, 0, 1, 2, 3, 5])
    exact_outcome(nx.dag_longest_path, G, weight=lambda u, v, d: 1)
    G.add_edge(5, 6, weight=None)
    exact_outcome(nx.dag_longest_path, G)
    G.add_edge(6, 7, weight=2**60)
    exact_outcome(nx.dag_longest_path, G)


@pytest.mark.parametrize(
    "G",
    [
        nx.DiGraph(),
        nx.DiGraph([(0, 0)]),
        nx.DiGraph([(0, 1), (1, 0)]),
        nx.DiGraph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 2)]),
        nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]),
        nx.DiGraph([(0, 1), (1, 2), (2, 0), (3, 4)]),
        nx.DiGraph([(0, 1), (1, 2), (2, 0), (2, 3)]),
        nx.DiGraph([(3, 2), (2, 1), (1, 0)]),
        nx.DiGraph([("b", "a"), ("c", "a"), ("a", "d")]),
        nx.DiGraph([(1, 2.5), (0.5, 2.5), (2.5, 3)]),
        nx.DiGraph([(1, "a"), ("b", 2)]),
        nx.Graph([(0, 1), (1, 2), (2, 0)]),
        nx.Graph([(0, 0)]),
        nx.empty_graph(3, create_using=nx.DiGraph),
    ],
    ids=lambda G: f"{type(G).__name__}{list(G.edges)}",
)
def test_batch3_small_cases(G):
    funcs = [nx.is_aperiodic, nx.is_semiconnected, nx.dag_longest_path,
             nx.dag_longest_path_length, dag_func("has_cycle"),
             listed(nx.lexicographical_topological_sort), listed(nx.all_topological_sorts),
             listed(nx.kosaraju_strongly_connected_components),
             graph_parts(nx.condensation), graph_parts(nx.transitive_reduction),
             graph_parts(nx.transitive_closure), graph_parts(nx.transitive_closure_dag),
             graph_parts(nx.dag_to_branching), listed(dag_func("colliders")),
             listed(dag_func("v_structures")), listed(dag_func("root_to_leaf_paths")),
             listed(nx.bfs_labeled_edges), listed(nx.dfs_labeled_edges),
             listed(nx.edge_bfs), listed(nx.edge_dfs)]
    for func in funcs:
        if func in (listed(nx.bfs_labeled_edges),):
            continue
        exact_outcome(func, G)
    for source in list(G)[:2] + ["missing"]:
        exact_outcome(listed(nx.bfs_labeled_edges), G, source)
        exact_outcome(listed(nx.generic_bfs_edges), G, source)


@pytest.mark.parametrize("seed", range(60))
def test_batch3_components(seed):
    D = graph_for(seed, True)
    rng = random.Random(seed)
    exact_outcome(graph_parts(nx.condensation), D)
    exact_outcome(nx.is_semiconnected, D)
    exact_outcome(listed(nx.kosaraju_strongly_connected_components), D)
    for source in rng.sample(list(D), min(3, len(D))) + ["missing", [0]]:
        exact_outcome(listed(nx.kosaraju_strongly_connected_components), D, source)
    # Semiconnected cases: a path through components, with extra edges.
    P = nx.DiGraph()
    for i in range(rng.randint(1, 6)):
        size = rng.randint(1, 4)
        nx.add_cycle(P, [(i, j) for j in range(size)]) if size > 1 else P.add_node((i, 0))
        if i:
            P.add_edge((i - 1, 0), (i, size - 1))
    exact_outcome(nx.is_semiconnected, P)
    exact_outcome(graph_parts(nx.condensation), P)
    if len(P) > 1:
        P.remove_edge(*list(P.edges)[-1])
        exact_outcome(nx.is_semiconnected, P)
    # is_aperiodic: on each strongly connected component, and on the whole graph.
    exact_outcome(nx.is_aperiodic, D)
    for comp in nx.strongly_connected_components(D, backend="networkx"):
        H = D.subgraph(comp).copy()
        exact_outcome(nx.is_aperiodic, H)
    U = graph_for(seed, False)
    exact_outcome(graph_parts(nx.condensation), U)
    exact_outcome(nx.is_semiconnected, U)
    exact_outcome(nx.is_aperiodic, U)
    exact_outcome(listed(nx.kosaraju_strongly_connected_components), U)
    exact_outcome(graph_parts(lambda G, **kw: nx.condensation(G, list(nx.strongly_connected_components(G)), **kw)), D)
    runs_in_rustnx(nx.condensation, D)
    runs_in_rustnx(nx.kosaraju_strongly_connected_components, D)
    runs_in_rustnx(nx.is_semiconnected, D)


def test_batch3_lexicographical_nan():
    # (NetworkX's dag_longest_path loops forever on a NaN node: nan != nan.)
    G = nx.DiGraph([(float("nan"), 1), (0, 1), (2.5, 0)])
    exact_outcome(listed(nx.lexicographical_topological_sort), G)


def test_batch3_aperiodic_cycles():
    for sizes in [[3], [2, 4], [3, 3], [3, 4], [2, 2], [1], [5, 10, 15]]:
        G = nx.DiGraph()
        for k, size in enumerate(sizes):
            nx.add_cycle(G, [0] + [(k, j) for j in range(1, size)]) if size > 1 else G.add_edge(0, 0)
        exact_outcome(nx.is_aperiodic, G)
        assert runs_in_rustnx(nx.is_aperiodic, G) == nx.is_aperiodic(G, backend="networkx")


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch3_traversal(seed, directed):
    G = graph_for(seed, directed)
    nodes = list(G)
    rng = random.Random(seed)
    for source in sources_for(G, seed):
        exact_outcome(listed(nx.bfs_labeled_edges), G, source)
        for depth in [None, 0, 1, 2, 5]:
            exact_outcome(listed(nx.dfs_labeled_edges), G, source, depth_limit=depth)
            exact_outcome(listed(nx.generic_bfs_edges), G, source, depth_limit=depth)
        for orientation in [None, "original", "reverse", "ignore", "bad"]:
            exact_outcome(listed(nx.edge_bfs), G, source, orientation)
            exact_outcome(listed(nx.edge_dfs), G, source, orientation)
    for depth in [None, 1, 3]:
        exact_outcome(listed(nx.dfs_labeled_edges), G, depth_limit=depth)
    some = rng.sample(nodes, min(3, len(nodes)))
    for sources in [some, some + some[:1], some + ["missing"], ["missing"] + some, [], iter(some)]:
        if hasattr(sources, "__next__"):
            exact_outcome(with_set_order(lambda G, **kw: list(nx.bfs_labeled_edges(G, iter(some), **kw))), G)
            continue
        exact_outcome(listed(nx.bfs_labeled_edges), G, sources)
        for orientation in [None, "original", "reverse", "ignore", "bad"]:
            exact_outcome(listed(nx.edge_bfs), G, sources, orientation)
            exact_outcome(listed(nx.edge_dfs), G, sources, orientation)
            exact_outcome(listed(nx.edge_bfs), G, set(sources), orientation)
            exact_outcome(listed(nx.edge_dfs), G, tuple(sources), orientation)
    for orientation in [None, "ignore"]:
        exact_outcome(listed(nx.edge_bfs), G, None, orientation)
        exact_outcome(listed(nx.edge_dfs), G, None, orientation)
        exact_outcome(listed(nx.edge_bfs), G, 12345, orientation)  # neither node nor list
        exact_outcome(listed(nx.edge_dfs), G, [[1]], orientation)
    exact_outcome(listed(nx.generic_bfs_edges), G, nodes[0], neighbors=G.neighbors)
    exact_outcome(listed(nx.dfs_labeled_edges), G, nodes[0], sort_neighbors=sorted)
    src = nodes[0]
    for func in [nx.bfs_labeled_edges, nx.dfs_labeled_edges, nx.generic_bfs_edges]:
        runs_in_rustnx(func, G, src)
    for orientation in [None, "original", "reverse", "ignore"]:
        runs_in_rustnx(nx.edge_bfs, G, src, orientation)
        runs_in_rustnx(nx.edge_dfs, G, nodes, orientation)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(20))
def test_batch3_multigraphs(seed, directed, restore_config):
    M = random_multigraph(seed, directed, "none")
    source = list(M)[0]
    exact_outcome(listed(nx.bfs_labeled_edges), M, source)
    exact_outcome(listed(nx.dfs_labeled_edges), M)
    exact_outcome(listed(nx.generic_bfs_edges), M, source)
    # Edge keys: these fall back.
    exact_outcome(listed(nx.edge_bfs), M, source, "ignore")
    exact_outcome(listed(nx.edge_dfs), M)
    exact_outcome(graph_parts(nx.transitive_closure), M)
    if directed:
        exact_outcome(listed(nx.kosaraju_strongly_connected_components), M)
        exact_outcome(graph_parts(nx.condensation), M)
        exact_outcome(nx.is_semiconnected, M)
        exact_outcome(nx.dag_longest_path, M)
        D = nx.MultiDiGraph(random_dag(seed))
        D.add_edges_from(list(D.edges)[:3])
        exact_outcome(graph_parts(nx.transitive_reduction), D)
        exact_outcome(nx.dag_longest_path, D)
        exact_outcome(listed(nx.all_topological_sorts), nx.MultiDiGraph(random_dag(seed, small=True)))
        exact_outcome(graph_parts(nx.dag_to_branching), D)
        runs_in_rustnx(nx.kosaraju_strongly_connected_components, M)
        runs_in_rustnx(nx.transitive_reduction, D)


def test_batch3_graph_changes_during_iteration():
    calls = [
        lambda G, b: nx.dfs_labeled_edges(G, 0, backend=b),
        lambda G, b: nx.bfs_labeled_edges(G, 0, backend=b),
        lambda G, b: nx.edge_bfs(G, 0, backend=b),
        lambda G, b: nx.edge_dfs(G, 0, backend=b),
        lambda G, b: nx.kosaraju_strongly_connected_components(G, backend=b),
        lambda G, b: nx.dag.v_structures(G, backend=b),
        lambda G, b: nx.dag.root_to_leaf_paths(G, backend=b),
        lambda G, b: nx.all_topological_sorts(G, backend=b),
    ]
    for call in calls:
        G = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (2, 4), (5, 4), (6, 4)])
        it = call(G, "rustnx")
        next(it)
        G.add_edge("x", "y")
        with pytest.raises(RuntimeError):
            list(it)


def test_batch3_lexicographical_changes_like_networkx():
    # lexicographical_topological_sort continues on the changed graph, as
    # NetworkX does (its own test_topological_sort6 checks the errors).
    changes = [
        lambda G, x: G.add_edge(5 - x, 5),
        lambda G, x: G.remove_node(4),
        lambda G, x: G.remove_node(2),
        lambda G, x: G.add_edge(4, 9),
        lambda G, x: G.add_edge(1, 4),
        lambda G, x: G.add_edge(0, 3),
    ]
    for change in changes:
        for when in [0, 1, 2]:
            def run(backend):
                G = nx.DiGraph([(1, 2), (2, 3), (3, 4), (1, 3)])
                out = []
                try:
                    it = nx.lexicographical_topological_sort(G, backend=backend)
                    if when == 0:
                        change(G, 1)
                    for k, x in enumerate(it):
                        out.append(x)
                        if k + 1 == when:
                            change(G, x)
                except Exception as exc:
                    out.append((type(exc), exc.args))
                return out

            assert run("rustnx") == run("networkx")


def test_batch3_lazy_errors():
    # Errors NetworkX raises at the first next() are raised there too.
    U = nx.path_graph(3)
    D = nx.DiGraph([(0, 1), (1, 0)])
    for backend in ["rustnx", "networkx"]:
        it = nx.lexicographical_topological_sort(U, backend=backend)
        with pytest.raises(nx.NetworkXError):
            next(it)
        it = nx.edge_bfs(nx.DiGraph([(0, 1)]), 0, "bad", backend=backend)
        with pytest.raises(nx.NetworkXError):
            next(it)
        it = nx.bfs_labeled_edges(D, "missing", backend=backend)
        with pytest.raises(KeyError):
            next(it)
        it = nx.kosaraju_strongly_connected_components(D, "missing", backend=backend)
        with pytest.raises(nx.NetworkXError):
            next(it)
        it = nx.all_topological_sorts(D, backend=backend)
        with pytest.raises(nx.NetworkXUnfeasible):
            next(it)
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.all_topological_sorts(U, backend=backend)
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.dag.v_structures(U, backend=backend)


# --- Batch 4: centrality ------------------------------------------------------------


def subset_args(G, seed):
    rng = random.Random(seed)
    nodes = list(G)
    k = min(len(nodes), rng.randint(1, 6))
    return rng.sample(nodes, k), rng.sample(nodes, min(len(nodes), rng.randint(1, 6)))


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch4_subset_betweenness(seed, directed, weights):
    G = graph_for(seed, directed, weights)
    weight = None if weights == "none" else "weight"
    sources, targets = subset_args(G, seed)
    for normalized in [False, True]:
        for func in [nx.betweenness_centrality_subset, nx.edge_betweenness_centrality_subset]:
            exact_outcome(func, G, sources, targets, normalized=normalized, weight=weight)
            exact_outcome(func, G, list(G), list(G), normalized=normalized, weight=weight)
            exact_outcome(func, G, sources + sources[:1], targets + ["missing"], weight=weight)
            exact_outcome(func, G, [], targets, weight=weight)
            exact_outcome(func, G, sources + ["missing"], targets, weight=weight)


@pytest.mark.parametrize("weights", ["none", "int", "float", "missing"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch4_load(seed, directed, weights):
    G = graph_for(seed, directed, "none" if weights == "missing" else weights)
    if weights == "missing":
        for i, (u, v, d) in enumerate(G.edges(data=True)):
            if i % 2:
                d["weight"] = 2
    weight = None if weights == "none" else "weight"
    nodes = list(G)
    for cutoff in [None, 1, 2, 2.5]:
        for normalized in [True, False]:
            exact_outcome(nx.load_centrality, G, cutoff=cutoff, normalized=normalized, weight=weight)
        exact_outcome(nx.load_centrality, G, nodes[0], cutoff=cutoff, weight=weight)
    exact_outcome(nx.load_centrality, G, "missing", weight=weight)
    if weights == "none":
        for cutoff in [False, 1, 2, 3.5]:
            exact_outcome(nx.edge_load_centrality, G, cutoff=cutoff)


def test_batch4_load_quirks():
    # Weighted predecessors listed after the source are skipped.
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (0, 2, 2), (2, 3, 1), (0, 3, 3)])
    exact_outcome(nx.load_centrality, G, weight="weight")
    G.add_edge(2, 2, weight=0)  # zero-weight self-loop
    exact_outcome(nx.load_centrality, G, weight="weight")
    G.add_edge(3, 4, weight=-1)
    exact_outcome(nx.load_centrality, G, weight="weight")  # contradictory paths
    # Mixed node types fall back (NetworkX's sort may raise).
    H = nx.Graph([(0, "a"), ("a", 1), (0, "b"), ("b", 1)])
    exact_outcome(nx.load_centrality, H)
    for G in [nx.empty_graph(0), nx.empty_graph(1), nx.path_graph(2), nx.path_graph(3)]:
        exact_outcome(nx.load_centrality, G)
        exact_outcome(nx.load_centrality, G, 0)
        exact_outcome(nx.edge_load_centrality, G)


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch4_percolation(seed, directed, weights):
    G = graph_for(seed, directed, weights)
    weight = None if weights == "none" else "weight"
    rng = random.Random(seed)
    exact_outcome(nx.percolation_centrality, G, weight=weight)
    for v in G:
        if rng.random() < 0.7:
            G.nodes[v]["percolation"] = rng.choice([0.25, 0.5, 1, 2])
    exact_outcome(nx.percolation_centrality, G, weight=weight)
    states = {v: rng.random() for v in G}
    exact_outcome(nx.percolation_centrality, G, states=states, weight=weight)
    exact_outcome(nx.percolation_centrality, G, attribute="other", weight=weight)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(60))
def test_batch4_voterank(seed, directed):
    G = graph_for(seed, directed)
    for number in [None, 0, 1, 3, -1, 1000]:
        exact_outcome(nx.voterank, G, number)


@pytest.mark.parametrize("seed", range(40))
def test_batch4_dispersion(seed):
    G = graph_for(seed, False)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    nodes = list(G)
    exact_outcome(nx.dispersion, G)
    exact_outcome(nx.dispersion, G, normalized=False)
    exact_outcome(nx.dispersion, G, alpha=2, b=1, c=0.5)
    for u in nodes[:3]:
        exact_outcome(nx.dispersion, G, u)
        exact_outcome(nx.dispersion, G, v=u)
        for v in nodes[-3:]:
            exact_outcome(nx.dispersion, G, u, v)
    exact_outcome(nx.dispersion, G, "missing")
    # Self-loops and directed graphs fall back.
    G.add_edge(nodes[0], nodes[0])
    exact_outcome(nx.dispersion, G, nodes[0])
    exact_outcome(nx.dispersion, graph_for(seed, True))


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch4_group(seed, directed, weights):
    G = graph_for(seed, directed, weights)
    weight = None if weights == "none" else "weight"
    rng = random.Random(seed)
    nodes = list(G)
    groups = [rng.sample(nodes, min(len(nodes), k)) for k in [1, 2, 5]]
    groups += [set(groups[-1]), groups[-1] + groups[-1][:1], [], nodes, groups[0] + ["missing"]]
    for S in groups:
        exact_outcome(nx.group_closeness_centrality, G, S, weight=weight)
        exact_outcome(nx.group_degree_centrality, G, S)
        exact_outcome(nx.group_in_degree_centrality, G, S)
        exact_outcome(nx.group_out_degree_centrality, G, S)


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch4_reaching(seed, directed, weights):
    G = graph_for(seed, directed, weights)
    weight = None if weights == "none" else "weight"
    for normalized in [True, False]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            exact_outcome(nx.global_reaching_centrality, G, weight=weight, normalized=normalized)
        for v in list(G)[:4] + ["missing"]:
            exact_outcome(nx.local_reaching_centrality, G, v, weight=weight, normalized=normalized)


@pytest.mark.parametrize(
    "G",
    [
        nx.empty_graph(0),
        nx.empty_graph(1),
        nx.empty_graph(3),
        nx.path_graph(2),
        nx.Graph([(0, 0)]),
        nx.Graph([(0, 0), (0, 1)]),
        nx.DiGraph([(0, 1), (1, 0), (1, 2)]),
        nx.DiGraph([(0, 0)]),
        nx.complete_graph(4),
        nx.star_graph(4),
    ],
    ids=lambda G: f"{type(G).__name__}{list(G.edges)}",
)
def test_batch4_small_cases(G):
    nodes = list(G)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        exact_outcome(nx.global_reaching_centrality, G)
    funcs = [nx.voterank, nx.percolation_centrality, nx.edge_load_centrality, nx.load_centrality]
    if not G.is_directed():
        funcs.append(nx.dispersion)
    for func in funcs:
        exact_outcome(func, G)
    for v in nodes[:2]:
        exact_outcome(nx.local_reaching_centrality, G, v)
        exact_outcome(nx.group_closeness_centrality, G, [v])
        exact_outcome(nx.group_degree_centrality, G, [v])
        exact_outcome(nx.betweenness_centrality_subset, G, nodes, [v], normalized=True)
        exact_outcome(nx.edge_betweenness_centrality_subset, G, [v], nodes, normalized=True)


def test_batch4_reaching_warns_like_networkx():
    G = nx.path_graph(600)

    def run(backend):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = nx.global_reaching_centrality(G, backend=backend)
        return result, [(w.category, str(w.message)) for w in caught]

    assert run("rustnx") == run("networkx")


def test_batch4_reaching_weight_errors():
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 2), (1, 2, -1)])
    exact_outcome(nx.global_reaching_centrality, G, weight="weight")
    exact_outcome(nx.local_reaching_centrality, G, 0, weight="weight")
    G[1][2]["weight"] = 0
    exact_outcome(nx.local_reaching_centrality, G, 0, weight="weight")
    G[1][2]["weight"] = 1.5  # mixed ints and floats fall back
    exact_outcome(nx.local_reaching_centrality, G, 0, weight="weight")
    del G[1][2]["weight"]
    exact_outcome(nx.local_reaching_centrality, G, 0, weight="weight")


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(20))
def test_batch4_multigraphs(seed, directed, restore_config):
    M = random_multigraph(seed, directed, "int")
    nodes = list(M)
    exact_outcome(nx.betweenness_centrality_subset, M, nodes[:3], nodes[-3:], weight="weight")
    exact_outcome(nx.load_centrality, M)
    exact_outcome(nx.load_centrality, M, weight="weight")
    # These see parallel edges: they fall back.
    exact_outcome(nx.edge_betweenness_centrality_subset, M, nodes[:3], nodes[-3:])
    exact_outcome(nx.edge_load_centrality, M)
    exact_outcome(nx.voterank, M)
    exact_outcome(nx.group_degree_centrality, M, nodes[:2])


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch4_group_betweenness(seed, directed, weights):
    G = graph_for(seed, directed, weights)
    weight = None if weights == "none" else "weight"
    rng = random.Random(seed)
    nodes = list(G)
    groups = [rng.sample(nodes, min(len(nodes), k)) for k in [1, 2, 3, 6]]
    for normalized in [True, False]:
        for endpoints in [False, True]:
            for S in groups:
                exact_outcome(nx.group_betweenness_centrality, G, S, normalized=normalized,
                              weight=weight, endpoints=endpoints)
            exact_outcome(nx.group_betweenness_centrality, G, groups, normalized=normalized,
                          weight=weight, endpoints=endpoints)
    exact_outcome(nx.group_betweenness_centrality, G, nodes[:2] + ["missing", "other"])
    exact_outcome(nx.group_betweenness_centrality, G, [nodes[:2], ["missing"]])
    # Strongly connected / connected graphs take another endpoint count.
    H = nx.cycle_graph(7, create_using=G.__class__)
    H.add_edges_from([(0, 3), (2, 5)])
    for S in [[0], [0, 3], [1, 4, 5], [[1, 2], [3, 6]]]:
        exact_outcome(nx.group_betweenness_centrality, H, S)
        exact_outcome(nx.group_betweenness_centrality, H, S, normalized=False)


@pytest.mark.parametrize("weights", ["none", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(15))
def test_batch4_prominent_group(seed, directed, weights):
    pytest.importorskip("pandas")
    G = graph_for(seed, directed, weights)
    if len(G) > 12:  # NetworkX's search is slow
        G = G.subgraph(list(G)[:12]).copy()
    weight = None if weights == "none" else "weight"
    for k in [0, 1, 3, len(G) + 1]:
        for greedy in [False, True]:
            exact_outcome(nx.prominent_group, G, k, weight=weight, greedy=greedy)
        exact_outcome(nx.prominent_group, G, k, weight=weight, normalized=False, endpoints=True)
    exact_outcome(nx.prominent_group, G, 2, C=list(G)[:2])


# --- Batch 5: cores, clustering, distance and coloring -----------------------------


def nx_has(name):
    return pytest.mark.skipif(not hasattr(nx, name), reason=f"NetworkX lacks {name}")


def without_self_loops(G):
    H = G.copy()
    H.remove_edges_from(list(nx.selfloop_edges(H)))
    return H


def counters_as_lists(func):
    """Counters compare as dicts; check their key order too."""
    def run(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, dict) and not hasattr(result, "most_common"):
            return {k: list(v.items()) for k, v in result.items()}
        return list(result.items())
    return run


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch5_cores(seed, directed):
    G = graph_for(seed, directed)
    for H in [G, without_self_loops(G)]:
        exact_outcome(nx.k_shell, H)
        exact_outcome(nx.k_crust, H)
        for k in [None, 0, 1, 2, 3, 2.0, 2.5]:
            exact_outcome(nx.k_shell, H, k=k)
            exact_outcome(nx.k_crust, H, k=k)
            exact_outcome(nx.k_corona, H, k)
        exact_outcome(nx.onion_layers, H)
        for k in [-1, 0, 2, 3, 4, 2.5, 4.0, float("nan"), float("inf"), True]:
            exact_outcome(nx.k_truss, H, k)
    H = without_self_loops(G)
    core = nx.core_number(H, backend="networkx")
    exact_outcome(nx.k_shell, H, core_number=core)
    exact_outcome(nx.k_corona, H, 1, core_number=core)
    exact_outcome(nx.k_crust, H, 1, core_number=core)


@pytest.mark.parametrize(
    "G",
    [nx.empty_graph(0), nx.empty_graph(3), nx.path_graph(5), nx.complete_graph(5),
     nx.karate_club_graph(), nx.Graph([(0, 0), (0, 1)])],
    ids=lambda G: f"{len(G)}-{G.number_of_edges()}",
)
def test_batch5_core_edge_cases(G):
    for func in [nx.k_shell, nx.k_crust, nx.onion_layers]:
        exact_outcome(func, G)
    for k in [None, 1, 2, 3]:
        exact_outcome(nx.k_corona, G, k)
        exact_outcome(nx.k_truss, G, k if k is not None else 3)
    G2 = G.copy()
    G2.graph["name"] = "g"
    nx.set_node_attributes(G2, 1, "a")
    nx.set_edge_attributes(G2, 2, "w")
    exact_outcome(nx.k_truss, G2, 3)
    exact_outcome(nx.k_shell, G2)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(60))
def test_batch5_clustering(seed, directed):
    G = graph_for(seed, directed)
    nodes = list(G)
    rng = random.Random(seed)
    subset = rng.sample(nodes, min(4, len(nodes))) + ["missing"] + nodes[:1]
    exact_outcome(nx.square_clustering, G)
    exact_outcome(nx.square_clustering, G, subset)
    exact_outcome(nx.square_clustering, G, nodes[0])
    exact_outcome(counters_as_lists(nx.generalized_degree), G)
    exact_outcome(counters_as_lists(nx.generalized_degree), G, subset)
    exact_outcome(counters_as_lists(nx.generalized_degree), G, nodes[0])
    if hasattr(nx, "all_triangles"):
        exact_outcome(listed(nx.all_triangles), G)
        exact_outcome(listed(nx.all_triangles), G, subset)
        exact_outcome(listed(nx.all_triangles), G, nodes[0])


@nx_has("all_triangles")
@pytest.mark.parametrize("seed", range(10))
def test_batch5_all_triangles_dense(seed):
    # Many triangles per edge: yield order follows Python's set order.
    for G in [nx.gnp_random_graph(30, 0.6, seed=seed),
              nx.relabel_nodes(nx.gnp_random_graph(30, 0.6, seed=seed), lambda v: f"n{v}")]:
        exact_outcome(listed(nx.all_triangles), G)
        exact_outcome(listed(nx.all_triangles), G, list(G)[::3])
    # A view falls back.
    exact_outcome(listed(nx.all_triangles), nx.complete_graph(8).subgraph(range(6)))


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch5_distance(seed, directed, weights):
    from test_against_networkx import connected_random_graph

    G = connected_random_graph(seed, directed, weights)
    weight = None if weights == "none" else "weight"
    exact_outcome(nx.barycenter, G, weight=weight)
    if hasattr(nx, "centroid"):
        exact_outcome(nx.centroid, G, weight=weight)
    exact_outcome(nx.harmonic_diameter, G)
    if "weight" in inspect.signature(nx.harmonic_diameter).parameters:
        exact_outcome(nx.harmonic_diameter, G, weight=weight)
    if not directed:
        # Trees take NetworkX 3.7's tree path.
        T = nx.bfs_tree(G, list(G)[0], backend="networkx").to_undirected()
        exact_outcome(nx.barycenter, T)
        exact_outcome(nx.harmonic_diameter, T)


@pytest.mark.parametrize(
    "G",
    [nx.empty_graph(0), nx.empty_graph(1), nx.empty_graph(2), nx.path_graph(2), nx.path_graph(5),
     nx.star_graph(4), nx.balanced_tree(2, 3), nx.cycle_graph(6), nx.cycle_graph(40),
     nx.petersen_graph(), nx.complete_graph(5), nx.hypercube_graph(3), nx.dodecahedral_graph(),
     nx.icosahedral_graph(), nx.heawood_graph(), nx.circular_ladder_graph(5),
     nx.disjoint_union(nx.cycle_graph(4), nx.cycle_graph(4)),
     nx.Graph([(0, 0)]), nx.Graph([(0, 0), (1, 1), (0, 1)]), nx.DiGraph([(0, 1), (1, 0)]),
     nx.complete_bipartite_graph(3, 3), nx.paley_graph(13).to_undirected()],
    ids=lambda G: f"{type(G).__name__}-{len(G)}-{G.number_of_edges()}",
)
def test_batch5_distance_regular_and_small_cases(G):
    exact_outcome(nx.intersection_array, G)
    exact_outcome(nx.is_distance_regular, G)
    exact_outcome(nx.is_strongly_regular, G)
    exact_outcome(nx.barycenter, G)
    exact_outcome(nx.harmonic_diameter, G)
    if hasattr(nx, "centroid"):
        exact_outcome(nx.centroid, G)


@pytest.mark.parametrize("seed", range(30))
def test_batch5_distance_regular_random(seed):
    rng = random.Random(seed)
    n = rng.choice([4, 6, 8, 10, 12, 16])
    d = rng.choice([2, 3, 4])
    n = max(n, d + 1)
    if n * d % 2:
        n += 1
    G = nx.random_regular_graph(d, n, seed=seed)
    exact_outcome(nx.intersection_array, G)
    exact_outcome(nx.is_distance_regular, G)
    exact_outcome(nx.is_strongly_regular, G)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch5_coloring(seed, directed):
    G = graph_for(seed, directed)
    nodes = list(G)
    strategies = ["largest_first", "saturation_largest_first", "DSATUR",
                  "connected_sequential", "connected_sequential_bfs", "connected_sequential_dfs",
                  "smallest_last", "independent_set"]
    for strategy in strategies:
        exact_outcome(nx.greedy_color, G, strategy)
    exact_outcome(nx.greedy_color, G, "nonsense")
    exact_outcome(nx.greedy_color, without_self_loops(G).to_undirected(), "largest_first", interchange=True)

    def seeded(G, **kwargs):
        random.seed(seed)
        result = nx.greedy_color(G, "random_sequential", **kwargs)
        return result, random.random()  # the global state moved on the same way

    exact_outcome(seeded, G)
    colorings = [
        nx.greedy_color(G, backend="networkx"),
        {v: i % 3 for i, v in enumerate(nodes)},
        {v: str(i % 2) for i, v in enumerate(nodes)},
        {v: 1.0 if i % 2 else 1 for i, v in enumerate(nodes)},
        {v: [i] for i, v in enumerate(nodes)},  # unhashable colors fall back
        {v: 0 for v in nodes[1:]},  # a missing node
        {v: float("nan") for v in nodes},
    ]
    for coloring in colorings:
        exact_outcome(is_coloring, G, coloring)
        exact_outcome(is_equitable, G, coloring)
        exact_outcome(is_equitable, G, coloring, num_colors=5)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(15))
def test_batch5_multigraphs(seed, directed):
    M = random_multigraph(seed, directed, "int")
    exact_outcome(nx.square_clustering, M)
    exact_outcome(nx.harmonic_diameter, M)
    # NetworkX's is_coloring unpacks multigraph edge triples into two names
    # and raises; rustnx falls back.
    exact_outcome(is_coloring, M, nx.greedy_color(nx.Graph(M) if not directed else nx.DiGraph(M), backend="networkx"))
    exact_outcome(nx.k_truss if not directed else nx.k_shell, M, 2)
    exact_outcome(nx.barycenter, M)
    if not directed:
        exact_outcome(counters_as_lists(nx.generalized_degree), M)
        if hasattr(nx, "all_triangles"):
            exact_outcome(listed(nx.all_triangles), M)
        exact_outcome(nx.onion_layers, M)


@nx_has("all_triangles")
def test_batch5_graph_changes_during_iteration():
    G = nx.complete_graph(600)
    it = nx.all_triangles(G, backend="rustnx")
    next(it)
    G.add_edge("x", "y")
    with pytest.raises(RuntimeError):
        list(it)


# --- Batch 6: trees and structural tests ------------------------------------------


def _eulerian_graph(seed, directed):
    """A random Eulerian graph: a closed walk's edges (no repeats), plus
    sometimes one edge removed so only a path remains."""
    rng = random.Random(seed)
    n = rng.randint(2, 12)
    labels = [f"v{i}" for i in range(n)] if seed % 2 else list(range(n))
    G = nx.DiGraph() if directed else nx.Graph()
    v = labels[0]
    G.add_node(v)
    for _ in range(rng.randint(2, 30)):
        w = rng.choice(labels)
        if w != v and not G.has_edge(v, w):
            G.add_edge(v, w)
            v = w
    if v != labels[0] and not G.has_edge(v, labels[0]):
        G.add_edge(v, labels[0])
    if seed % 3 == 0 and G.number_of_edges():
        G.remove_edge(*list(G.edges)[rng.randrange(G.number_of_edges())])
    if seed % 5 == 0:
        G.add_edge(labels[0], labels[0])
    return G


def _batch6_graphs(seed, directed):
    G = graph_for(seed, directed)
    yield G
    rng = random.Random(seed)
    T = nx.random_labeled_tree(rng.randint(1, 30), seed=seed) if hasattr(nx, "random_labeled_tree") else nx.random_tree(rng.randint(1, 30), seed=seed)
    if directed:
        T = nx.bfs_tree(T, 0, backend="networkx")
    yield T
    yield _eulerian_graph(seed, directed)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(60))
def test_batch6_structure(seed, directed):
    for G in _batch6_graphs(seed, directed):
        nodes = list(G)
        exact_outcome(listed(nx.isolates), G)
        exact_outcome(nx.number_of_isolates, G)
        exact_outcome(nx.is_regular, G)
        for k in [0, 1, 2, 3, 2.0, True]:
            exact_outcome(nx.is_k_regular, G, k)
        exact_outcome(nx.is_tournament, G)
        exact_outcome(nx.is_eulerian, G)
        exact_outcome(nx.is_semieulerian, G)
        for source in [None] + nodes[:4] + ["missing"]:
            exact_outcome(nx.has_eulerian_path, G, source)
            exact_outcome(listed(nx.eulerian_circuit), G, source)
            exact_outcome(listed(nx.eulerian_path), G, source)
        exact_outcome(nx.is_arborescence, G)
        exact_outcome(nx.is_branching, G)
        exact_outcome(nx.to_prufer_sequence, G)
        exact_outcome(listed(nx.algorithms.tree.mst.kruskal_mst_edges), G, True)
        exact_outcome(listed(nx.algorithms.tree.mst.kruskal_mst_edges), G, False, data=False)
        for orientation in [None, "original", "reverse", "ignore", "bogus"]:
            exact_outcome(nx.find_cycle, G, orientation=orientation)
            for source in nodes[:3]:
                exact_outcome(nx.find_cycle, G, source, orientation=orientation)
        for start in nodes[:4] + ["missing"]:
            exact_outcome(nx.immediate_dominators, G, start)
            exact_outcome(with_set_order(nx.dominance_frontiers), G, start)
        if not directed:
            exact_outcome(listed(nx.bridges), G)
            exact_outcome(nx.has_bridges, G)
            exact_outcome(listed(nx.local_bridges), G)
            exact_outcome(listed(nx.local_bridges), G, with_span=False)
            exact_outcome(listed(nx.chain_decomposition), G)
            exact_outcome(nx.cycle_basis, G)
            exact_outcome(nx.girth, G)
            for root in nodes[:3] + ["missing"]:
                exact_outcome(nx.has_bridges, G, root)
                exact_outcome(listed(nx.bridges), G, root)
                exact_outcome(listed(nx.chain_decomposition), G, root)
                exact_outcome(nx.cycle_basis, G, root)


@pytest.mark.parametrize("weights", ["int", "float", "missing"])
@pytest.mark.parametrize("seed", range(30))
def test_batch6_weighted(seed, weights):
    G = graph_for(seed, False, weights)
    exact_outcome(listed(nx.local_bridges), G, weight="weight")
    exact_outcome(listed(nx.algorithms.tree.mst.kruskal_mst_edges), G, True)
    exact_outcome(listed(nx.algorithms.tree.mst.kruskal_mst_edges), G, False, data=False)
    D = graph_for(seed, True, weights)
    exact_outcome(listed(nx.algorithms.tree.mst.kruskal_mst_edges), D, True, weight="weight")


def test_batch6_dominators_and_cycles_on_known_graphs():
    # Classic examples with several dominance frontiers and cycles.
    D = nx.DiGraph([(1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 2), (5, 6), (6, 6), (6, 1)])
    for start in D:
        exact_outcome(nx.immediate_dominators, D, start)
        exact_outcome(with_set_order(nx.dominance_frontiers), D, start)
    exact_outcome(nx.find_cycle, nx.DiGraph([(0, 1), (2, 1), (2, 3), (3, 2)]), orientation="ignore")
    exact_outcome(nx.find_cycle, nx.DiGraph([(0, 1), (1, 2)]))  # no cycle
    exact_outcome(nx.find_cycle, nx.DiGraph([(0, 1), (1, 2)]), orientation="ignore")
    for G in [nx.petersen_graph(), nx.complete_graph(5), nx.path_graph(5), nx.Graph([(0, 0)])]:
        exact_outcome(nx.girth, G)
        exact_outcome(nx.cycle_basis, G)
        exact_outcome(listed(nx.chain_decomposition), G)
        exact_outcome(listed(nx.bridges), G)


@pytest.mark.parametrize(
    "G",
    [
        nx.empty_graph(0),
        nx.empty_graph(1),
        nx.empty_graph(3),
        nx.path_graph(2),
        nx.complete_graph(4),
        nx.Graph([(0, 0)]),
        nx.DiGraph(),
        nx.DiGraph([(0, 0)]),
        nx.DiGraph([(0, 1), (1, 2), (0, 2)]),
        nx.DiGraph([(0, 1), (1, 0)]),
    ],
    ids=lambda G: f"{type(G).__name__}{list(G.edges)}",
)
def test_batch6_small_cases(G):
    funcs = [listed(nx.isolates), nx.number_of_isolates, nx.is_regular, nx.is_tournament,
             nx.is_eulerian, nx.is_semieulerian, nx.has_eulerian_path, listed(nx.eulerian_circuit),
             listed(nx.eulerian_path), nx.is_arborescence, nx.is_branching, nx.to_prufer_sequence,
             nx.find_cycle, listed(nx.bridges), nx.has_bridges, listed(nx.local_bridges),
             listed(nx.chain_decomposition), nx.cycle_basis, nx.girth]
    for func in funcs:
        exact_outcome(func, G)
    exact_outcome(nx.is_k_regular, G, 0)
    exact_outcome(nx.immediate_dominators, G, 0)
    exact_outcome(with_set_order(nx.dominance_frontiers), G, 0)


def test_batch6_prufer_labels():
    exact_outcome(nx.to_prufer_sequence, nx.path_graph(["a", "b", "c"]))
    T = nx.Graph([(3, 1), (1, 0), (0, 2), (2, 4), (1, 5)])
    exact_outcome(nx.to_prufer_sequence, T)
    T.add_edge(5, 3)
    exact_outcome(nx.to_prufer_sequence, T)  # not a tree


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(20))
def test_batch6_multigraphs(seed, directed, restore_config):
    M = random_multigraph(seed, directed, "none")
    M.add_node("lonely")
    start = list(M)[0]
    exact_outcome(listed(nx.isolates), M)
    exact_outcome(nx.number_of_isolates, M)
    if directed:
        exact_outcome(nx.immediate_dominators, M, start)
        exact_outcome(with_set_order(nx.dominance_frontiers), M, start)
    # Parallel edges change degrees and cycles: these fall back.
    exact_outcome(nx.is_eulerian, M)
    exact_outcome(listed(nx.eulerian_circuit), M)
    exact_outcome(nx.find_cycle, M) if directed else exact_outcome(listed(nx.bridges), M)
    exact_outcome(nx.is_regular, M)


def test_batch6_graph_changes_during_iteration():
    for func in [nx.isolates, nx.bridges, nx.local_bridges]:
        G = nx.path_graph(6)
        G.add_nodes_from(["a", "b"])
        it = func(G, backend="rustnx")
        next(it)
        G.add_edge("x", "y")
        with pytest.raises(RuntimeError):
            list(it)
    # These work on a copy made when iteration starts.
    for func in [nx.chain_decomposition, nx.eulerian_circuit, nx.eulerian_path]:
        results = []
        for backend in ["rustnx", "networkx"]:
            G = nx.cycle_graph(5)
            G.add_edges_from([(0, 2), (2, 4), (4, 1), (1, 3), (3, 0)])
            it = func(G, backend=backend)
            G.remove_edge(0, 1)  # before the first item: seen
            G.add_edge(0, 1) if func is not nx.chain_decomposition else None
            first = next(it)
            G.add_edge("x", "y")  # after: not seen
            results.append([first, *it])
        assert results[0] == results[1]


# --- Batch 7: shortest paths, DAG and cycle leftovers -----------------------------


from collections import defaultdict  # noqa: E402
from itertools import islice  # noqa: E402



def typed_dicts(func):
    """Also compare dict types and defaultdict defaults (Floyd-Warshall's
    ``dist`` rows are ``defaultdict(lambda: inf)``)."""

    def convert(value):
        if isinstance(value, dict):
            default = value.default_factory() if isinstance(value, defaultdict) else None
            return (type(value).__name__, default, [(k, convert(v)) for k, v in value.items()])
        if isinstance(value, tuple):
            return tuple(convert(v) for v in value)
        return value

    return lambda *a, **kw: convert(func(*a, **kw))


def first_paths(func, k=200):
    """The first ``k`` items of a generator (simple paths can explode)."""
    return lambda *a, **kw: list(islice(func(*a, **kw), k))


def as_array(func):
    def run(*a, **kw):
        A = func(*a, **kw)
        values = [[x if x == x else "nan" for x in row] for row in A.tolist()]
        return (type(A).__name__, str(A.dtype), A.shape, values)

    return run


FW_TREE = getattr(nx, "floyd_warshall_tree", None)
ANTICHAIN_WIDTH = getattr(nx.dag, "antichain_width", None)


def _fw_calls(G, **kw):
    exact_outcome(typed_dicts(nx.floyd_warshall_predecessor_and_distance), G, **kw)
    exact_outcome(typed_dicts(nx.floyd_warshall), G, **kw)
    if FW_TREE is not None:
        exact_outcome(typed_dicts(FW_TREE), G, **kw)
    exact_outcome(as_array(nx.floyd_warshall_numpy), G, **kw)


def _signed_graph(seed, directed, cycles):
    """Random weights in -3..5: negative edges, and negative cycles unless
    ``cycles`` is False (then a DAG)."""
    rng = random.Random(seed)
    G = random_dag(seed, "none") if not cycles else graph_for(seed, directed)
    if not directed:
        G = G.to_undirected()
    for u, v, d in G.edges(data=True):
        d["weight"] = rng.randint(-3, 5) if rng.random() < 0.3 else rng.randint(0, 5)
    return G


@pytest.mark.parametrize("weights", ["none", "int", "float", "missing"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch7_dense_shortest_paths(seed, directed, weights):
    G = graph_for(seed, directed, weights)
    _fw_calls(G)
    _fw_calls(G, weight=None)
    nodes = list(G)
    random.Random(seed).shuffle(nodes)
    exact_outcome(as_array(nx.floyd_warshall_numpy), G, nodelist=nodes)
    exact_outcome(nx.johnson, G)
    exact_outcome(nx.johnson, G, weight=None)
    for source in nodes[:3] + ["missing"]:
        exact_outcome(nx.goldberg_radzik, G, source)
        exact_outcome(nx.goldberg_radzik, G, source, weight=None)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch7_negative_weights(seed, directed):
    for cycles in [False, True]:
        G = _signed_graph(seed, directed, cycles)
        _fw_calls(G)
        exact_outcome(nx.johnson, G)
        for source in list(G)[:4]:
            exact_outcome(nx.goldberg_radzik, G, source)
        if len(G) > 1:
            s, t = list(G)[0], list(G)[-1]
            exact_outcome(first_paths(nx.shortest_simple_paths, 20), G, s, t, weight="weight")
        if not directed and len(G) < 20:
            exact_outcome(nx.minimum_cycle_basis, G, weight="weight")


@pytest.mark.parametrize(
    "edges",
    [
        [],
        [(0, 0, 1)],
        [(0, 0, -1)],
        [(0, 0, 0.0)],
        [(0, 1, 2), (1, 0, -3)],
        [(0, 1, -2), (1, 2, 1), (2, 0, 1)],
        [(0, 1, None), (1, 2, 1), (0, 2, 5)],
        [(0, 1, 2**60), (1, 2, 2**60)],
        [(0, 1, 1), (1, 2, 2.5)],
        [(0, 1, float("inf")), (1, 2, 1)],
        [(0, 1, -0.0), (1, 2, 1.0)],
        [("a", "b", 1), ("b", "c", 2), ("a", "c", 3), ("c", "c", 1)],
    ],
    ids=str,
)
@pytest.mark.parametrize("directed", [False, True])
def test_batch7_weight_edge_cases(edges, directed):
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_node(0)
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    _fw_calls(G)
    exact_outcome(nx.johnson, G)
    nodes = list(G)
    for source in nodes:
        exact_outcome(nx.goldberg_radzik, G, source)
        for target in nodes:
            exact_outcome(first_paths(nx.shortest_simple_paths, 20), G, source, target, weight="weight")
    if not directed:
        exact_outcome(nx.minimum_cycle_basis, G, weight="weight")
    exact_outcome(nx.floyd_warshall_numpy, G, nodelist=nodes[:-1])  # too short
    exact_outcome(nx.floyd_warshall_numpy, G, nodelist=nodes[:-1] + ["x"])  # not in G


def test_batch7_empty_and_single_node():
    for G in [nx.Graph(), nx.DiGraph(), nx.empty_graph(1), nx.DiGraph([(0, 0)])]:
        _fw_calls(G)
        exact_outcome(nx.johnson, G)
        exact_outcome(nx.goldberg_radzik, G, 0)
        exact_outcome(listed(nx.all_simple_paths), G, 0, 0)
        exact_outcome(listed(nx.shortest_simple_paths), G, 0, 0)
        exact_outcome(nx.is_simple_path, G, [0])
        exact_outcome(nx.is_simple_path, G, [])
        if G.is_directed():
            exact_outcome(listed(nx.antichains), G)
            if ANTICHAIN_WIDTH is not None:
                exact_outcome(ANTICHAIN_WIDTH, G)
        else:
            exact_outcome(nx.minimum_cycle_basis, G)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch7_simple_paths(seed, directed):
    G = graph_for(seed, directed)
    nodes = list(G)
    rng = random.Random(seed)
    picks = rng.sample(nodes, min(3, len(nodes)))
    # Unbounded cutoffs only on small graphs: with an unreachable target
    # both backends walk every simple path.
    unbounded = [None, float("inf")] if len(G) <= 10 else []
    for source in picks + ["missing"]:
        for target in picks[:2] + ["missing", picks, set(picks), ["missing"], [], [None] + picks[:1], 5]:
            for cutoff in [0, 1, 2, 2.5, True, -1, float("nan"), *unbounded]:
                exact_outcome(first_paths(nx.all_simple_paths), G, source, target, cutoff=cutoff)
            exact_outcome(first_paths(nx.all_simple_edge_paths), G, source, target, cutoff=3)
        for target in picks + ["missing"]:
            exact_outcome(first_paths(nx.shortest_simple_paths, 20), G, source, target)
    for k in range(5):
        path = rng.sample(nodes, min(len(nodes), k))
        exact_outcome(nx.is_simple_path, G, path)
        exact_outcome(nx.is_simple_path, G, path + path[:1])
        exact_outcome(nx.is_simple_path, G, path + ["missing"])
    walk = list(nx.dfs_preorder_nodes(G, nodes[0], backend="networkx"))[:6]
    exact_outcome(nx.is_simple_path, G, walk)
    exact_outcome(nx.is_simple_path, G, [[1], [2]])
    exact_outcome(nx.is_simple_path, G, ([v] for v in nodes))  # no len()


@pytest.mark.parametrize("weights", ["int", "float", "missing"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch7_shortest_simple_paths_weighted(seed, directed, weights):
    G = graph_for(seed, directed, weights)
    nodes = list(G)
    for source in nodes[:3]:
        for target in nodes[-3:]:
            exact_outcome(first_paths(nx.shortest_simple_paths, 40), G, source, target, weight="weight")
    if weights == "float":
        # Many equal-length paths: ties go by NetworkX's push order.
        H = nx.grid_2d_graph(4, 4)
        H = nx.relabel_nodes(H, {v: i for i, v in enumerate(H)})
        if directed:
            H = H.to_directed()
        for u, v, d in H.edges(data=True):
            d["weight"] = [0.5, 1.0, 1.5][(u + v) % 3]
        exact_outcome(first_paths(nx.shortest_simple_paths, 100), H, 0, 15, weight="weight")
        exact_outcome(first_paths(nx.shortest_simple_paths, 100), H, 0, 15)


@pytest.mark.parametrize("seed", range(40))
def test_batch7_antichains(seed):
    D = random_dag(seed, "none")
    exact_outcome(first_paths(nx.antichains, 2000), D)
    topo = list(nx.topological_sort(D, backend="networkx"))
    lex = list(nx.lexicographical_topological_sort(D, key=str, backend="networkx"))
    exact_outcome(first_paths(nx.antichains, 2000), D, topo_order=topo)
    exact_outcome(first_paths(nx.antichains, 2000), D, topo_order=lex)
    exact_outcome(first_paths(nx.antichains, 2000), D, topo_order=topo[::-1])  # not a valid order
    exact_outcome(first_paths(nx.antichains, 2000), D, topo_order=topo[1:])
    if ANTICHAIN_WIDTH is not None:
        exact_outcome(ANTICHAIN_WIDTH, D)
    C = graph_for(seed, True)  # usually cyclic
    exact_outcome(first_paths(nx.antichains, 500), C)
    if ANTICHAIN_WIDTH is not None:
        exact_outcome(ANTICHAIN_WIDTH, C)
    exact_outcome(listed(nx.antichains), graph_for(seed, False))  # undirected


@pytest.mark.parametrize("weights", ["none", "int", "float", "missing"])
@pytest.mark.parametrize("seed", range(40))
def test_batch7_minimum_cycle_basis(seed, weights):
    rng = random.Random(seed)
    G = nx.gnm_random_graph(rng.randint(1, 18), rng.randint(0, 30), seed=seed)
    G.add_edges_from((v + 100, w + 100) for v, w in nx.cycle_graph(rng.randint(3, 6)).edges)
    if seed % 3 == 0:
        G.add_edge(0, 0)
        G = nx.relabel_nodes(G, {v: f"n{v}" for v in G})
    for u, v, d in G.edges(data=True):
        if weights == "int":
            d["weight"] = rng.randint(1, 4)
        elif weights == "float":
            d["weight"] = rng.choice([0.5, 1.0, 1.5])
        elif weights == "missing" and rng.random() < 0.5:
            d["weight"] = 2
    exact_outcome(nx.minimum_cycle_basis, G)
    exact_outcome(nx.minimum_cycle_basis, G, weight="weight")
    exact_outcome(nx.minimum_cycle_basis, nx.DiGraph(G))  # directed: rejected
    T = nx.relabel_nodes(G, {v: (v, 0) for v in G})  # tuple nodes fall back
    exact_outcome(nx.minimum_cycle_basis, T)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(15))
def test_batch7_multigraphs(seed, directed, restore_config):
    M = random_multigraph(seed, directed, "int" if seed % 2 else "float")
    nodes = list(M)
    exact_outcome(nx.johnson, M)
    exact_outcome(nx.is_simple_path, M, nodes[:3])
    # Parallel edges count separately in these: they fall back.
    exact_outcome(typed_dicts(nx.floyd_warshall), M)
    exact_outcome(nx.goldberg_radzik, M, nodes[0])
    exact_outcome(first_paths(nx.all_simple_paths), M, nodes[0], nodes[-1], cutoff=3)
    exact_outcome(first_paths(nx.shortest_simple_paths), M, nodes[0], nodes[-1])
    if not directed:
        exact_outcome(nx.minimum_cycle_basis, M)


def test_batch7_graph_changes_during_iteration():
    calls = [
        lambda G, b: nx.all_simple_paths(G, 0, 5, backend=b),
        lambda G, b: nx.all_simple_edge_paths(G, 0, 5, backend=b),
        lambda G, b: nx.shortest_simple_paths(G, 0, 5, backend=b),
    ]
    for call in calls:
        G = nx.complete_graph(7)
        it = call(G, "rustnx")
        next(it)
        G.add_edge("x", "y")
        with pytest.raises(RuntimeError):
            list(it)
        # A change before the first item: NetworkX runs on the changed graph.
        results = []
        for backend in ["rustnx", "networkx"]:
            G = nx.complete_graph(7)
            it = call(G, backend)
            G.remove_edge(0, 5)
            results.append(list(islice(it, 50)))
        assert results[0] == results[1]
    # antichains copies the graph when iteration starts.
    results = []
    for backend in ["rustnx", "networkx"]:
        D = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)])
        it = nx.antichains(D, backend=backend)
        D.remove_edge(0, 2)
        first_item = next(it)
        D.add_edge(4, 5)
        results.append([first_item, *it])
    assert results[0] == results[1]


# --- Batch 8: trees, branchings and lowest common ancestors -------------------------


def _graph_state(G):
    """Everything about a graph that order can show: node and adjacency
    order (and predecessor order), and all attributes."""
    state = [type(G).__name__, dict(G.graph), list(G._node.items()),
             [(u, list(nbrs.items())) for u, nbrs in G._adj.items()]]
    if G.is_directed():
        state.append([(u, list(nbrs)) for u, nbrs in G._pred.items()])
    return state


def strict(func):
    """``func`` with graphs in its result replaced by their full state."""

    def convert(value):
        if isinstance(value, nx.Graph):
            return ("graph", _graph_state(value))
        if hasattr(value, "__next__"):
            return [convert(v) for v in value]
        if isinstance(value, (list, tuple)):
            return type(value)(convert(v) for v in value)
        return value

    return lambda *a, **kw: convert(func(*a, **kw))


def mutating_outcome(func, G, *args, **kwargs):
    """``exact_outcome`` for functions that change their input: each backend
    runs on its own copy, and the copies must end up identical too."""
    import copy

    def run(backend):
        H = copy.deepcopy(G)
        H.__networkx_cache__.clear()
        try:
            result = ("ok", strict(func)(H, *args, backend=backend, **kwargs))
        except NotImplementedError:
            raise
        except Exception as exc:
            result = (type(exc), exc.args)
        return result, _graph_state(H)

    try:
        ours = run("rustnx")
    except NotImplementedError:
        ours = None  # falls back; NetworkX's own run is the reference
    ref = run("networkx")
    if ours is not None:
        assert ours == ref
    return ref


def _weighted_graph(seed, directed, weights):
    """A random graph whose weights include negatives, zeros and ties."""
    rng = random.Random(seed)
    G = graph_for(seed, directed, "none")
    for u, v, d in G.edges(data=True):
        if weights == "int":
            d["weight"] = rng.randint(-3, 6)
        elif weights == "float":
            d["weight"] = rng.choice([-2.0, 0.0, 0.1, 0.2, 0.3, 1.25, 3.5])
        elif weights == "mixed":
            d["weight"] = rng.choice([1, 2, 0.5, 3.25])
            d["color"] = rng.choice(["red", "blue"])
        elif weights == "missing" and rng.random() < 0.5:
            d["weight"] = rng.randint(-1, 4)
    return G


def _with_partition(G, seed):
    rng = random.Random(seed)
    P = G.copy()
    for _, _, d in P.edges(data=True):
        r = rng.random()
        if r < 0.15:
            d["partition"] = nx.EdgePartition.INCLUDED
        elif r < 0.35:
            d["partition"] = nx.EdgePartition.EXCLUDED
        elif r < 0.45:
            d["partition"] = nx.EdgePartition.OPEN
    return P


_TRANSFORMED = [nx.minimum_branching, nx.tree.minimal_branching,
                nx.maximum_spanning_arborescence, nx.minimum_spanning_arborescence]


@pytest.mark.parametrize("weights", ["int", "float", "mixed", "missing", "none"])
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("seed", range(40))
def test_batch8_branchings(seed, directed, weights):
    G = _weighted_graph(seed, directed, weights)
    for preserve in [False, True]:
        exact_outcome(strict(nx.maximum_branching), G, preserve_attrs=preserve)
        for func in _TRANSFORMED:
            mutating_outcome(func, G, preserve_attrs=preserve)
    exact_outcome(strict(nx.maximum_branching), G, default=2.5)
    exact_outcome(strict(nx.maximum_branching), G, attr="color", default=0)
    mutating_outcome(nx.minimum_branching, G, default=-1)
    mutating_outcome(nx.minimum_spanning_arborescence, G, attr="other", default=3)
    P = _with_partition(G, seed)
    for preserve in [False, True]:
        exact_outcome(strict(nx.maximum_branching), P, partition="partition", preserve_attrs=preserve)
        mutating_outcome(nx.minimum_spanning_arborescence, P, partition="partition",
                         preserve_attrs=preserve)
        mutating_outcome(nx.maximum_spanning_arborescence, P, partition="partition")
    for kind in ["max", "min", "bogus"]:
        exact_outcome(strict(nx.tree.greedy_branching), G, kind=kind)
        exact_outcome(strict(nx.tree.greedy_branching), G, default=2, kind=kind)
    exact_outcome(nx.tree.branching_weight, G)
    exact_outcome(nx.tree.branching_weight, G, default=0.5)
    exact_outcome(nx.tree.branching_weight, G, attr="color", default=2)


@pytest.mark.parametrize("weights", ["int", "float", "missing", "none"])
@pytest.mark.parametrize("seed", range(40))
def test_batch8_spanning_trees(seed, weights):
    from networkx.algorithms.tree import mst

    G = _weighted_graph(seed, False, weights)
    if seed % 4 == 0:  # several components, isolated nodes
        G.add_edges_from([("a", "b"), ("b", "c"), ("x", "x")])
        G.add_nodes_from(["lonely", 10**6])
    for minimum in [True, False]:
        exact_outcome(listed(mst.prim_mst_edges), G, minimum)
        exact_outcome(listed(mst.prim_mst_edges), G, minimum, data=False)
        exact_outcome(listed(mst.prim_mst_edges), G, minimum, weight=None)
        exact_outcome(listed(mst.boruvka_mst_edges), G, minimum)
        exact_outcome(listed(mst.boruvka_mst_edges), G, minimum, data=False)
        exact_outcome(listed(mst.boruvka_mst_edges), G, minimum, weight=None)
        exact_outcome(strict(nx.partition_spanning_tree), G, minimum)
        P = _with_partition(G, seed)
        exact_outcome(strict(nx.partition_spanning_tree), P, minimum)
        exact_outcome(strict(nx.partition_spanning_tree), P, minimum, weight=None)
    D = _with_partition(_weighted_graph(seed, True, weights), seed)
    exact_outcome(strict(nx.partition_spanning_tree), D)
    exact_outcome(listed(mst.prim_mst_edges), D, True)  # directed: falls back
    exact_outcome(listed(mst.boruvka_mst_edges), D, True)
    exact_outcome(listed(mst.boruvka_mst_edges), D, False, weight=None)


def _random_tree(seed, n):
    if hasattr(nx, "random_labeled_tree"):
        return nx.random_labeled_tree(n, seed=seed)
    return nx.random_tree(n, seed=seed)


@pytest.mark.parametrize("seed", range(60))
def test_batch8_tree_codes(seed):
    rng = random.Random(seed)
    n = rng.randint(0, 25)
    exact_outcome(strict(nx.from_prufer_sequence), [rng.randrange(n + 2) for _ in range(n)])

    def nested(depth):
        width = rng.randint(0, 3 if depth < 5 else 0)
        kids = [nested(depth + 1) for _ in range(width)]
        return list(kids) if rng.random() < 0.2 else tuple(kids)

    t = nested(0)
    for sensible in [False, True]:
        exact_outcome(strict(nx.from_nested_tuple), t, sensible_relabeling=sensible)
    if n:
        T = _random_tree(seed, n)
        if seed % 2:
            T = nx.relabel_nodes(T, {v: f"v{v}" for v in T})
        nodes = list(T)
        for root in nodes[:4] + ["missing"]:
            exact_outcome(nx.to_nested_tuple, T, root, canonical_form=True)
            exact_outcome(nx.to_nested_tuple, T, root)  # falls back
        if hasattr(nx.tree, "centroid"):
            exact_outcome(nx.tree.centroid, T)
    G = graph_for(seed, False)
    exact_outcome(nx.to_nested_tuple, G, list(G)[0], canonical_form=True)
    if hasattr(nx.tree, "centroid"):
        exact_outcome(nx.tree.centroid, G)
        exact_outcome(nx.tree.centroid, graph_for(seed, True))


def test_batch8_tree_code_edge_cases():
    for seq in [[], [0], [3], [-1], [1, 1], [0, 0, 0], [5, 0, 1], [1.0], [True], (2, 2, 2)]:
        exact_outcome(strict(nx.from_prufer_sequence), seq)
    for t in [(), ((),), ((), ()), (((),), ()), [[], [[]]], ("ab",), (1,), ((),) * 6]:
        for sensible in [False, True]:
            exact_outcome(strict(nx.from_nested_tuple), t, sensible_relabeling=sensible)
    deep = ()
    for _ in range(120):
        deep = (deep,)
    exact_outcome(strict(nx.from_nested_tuple), deep)  # too deep: falls back
    for T in [nx.empty_graph(0), nx.empty_graph(1), nx.path_graph(2), nx.path_graph(5),
              nx.star_graph(4), nx.Graph([(0, 0)]), nx.balanced_tree(2, 3)]:
        for root in [0, 1]:
            exact_outcome(nx.to_nested_tuple, T, root, canonical_form=True)
        if hasattr(nx.tree, "centroid"):
            exact_outcome(nx.tree.centroid, T)
    exact_outcome(nx.to_nested_tuple, nx.path_graph(150), 0, canonical_form=True)
    exact_outcome(nx.to_nested_tuple, nx.DiGraph([(0, 1)]), 0, canonical_form=True)


def _dag(seed):
    D = graph_for(seed, True)
    order = {v: i for i, v in enumerate(D)}
    dag = nx.DiGraph()
    dag.add_nodes_from(D)
    dag.add_edges_from((u, v) for u, v in D.edges if order[u] < order[v])
    return dag


@pytest.mark.parametrize("seed", range(60))
def test_batch8_lowest_common_ancestors(seed):
    rng = random.Random(seed)
    dag = _dag(seed)
    nodes = list(dag)
    exact_outcome(listed(nx.all_pairs_lowest_common_ancestor), dag)
    pairs = [(rng.choice(nodes), rng.choice(nodes)) for _ in range(30)]
    exact_outcome(listed(nx.all_pairs_lowest_common_ancestor), dag, pairs)
    exact_outcome(listed(nx.all_pairs_lowest_common_ancestor), dag, pairs + [(nodes[0], "missing")])
    for u, v in pairs[:5] + [(nodes[0], "missing")]:
        exact_outcome(nx.lowest_common_ancestor, dag, u, v)
        exact_outcome(nx.lowest_common_ancestor, dag, u, v, default="none")
    D = graph_for(seed, True)
    exact_outcome(listed(nx.all_pairs_lowest_common_ancestor), D)  # cycles
    exact_outcome(listed(nx.all_pairs_lowest_common_ancestor), graph_for(seed, False))
    T = _random_tree(seed, rng.randint(1, 30))
    T = nx.bfs_tree(T, 0, backend="networkx")
    for G in [T, dag, D]:
        exact_outcome(listed(nx.tree_all_pairs_lowest_common_ancestor), G)
        for root in list(G)[:3] + ["missing"]:
            exact_outcome(listed(nx.tree_all_pairs_lowest_common_ancestor), G, root)
        exact_outcome(listed(nx.tree_all_pairs_lowest_common_ancestor), G,
                      pairs=[(list(G)[0], list(G)[-1])])  # falls back


def test_batch8_lca_edge_cases():
    # Two lowest common ancestors: NetworkX's answer depends on set order.
    D = nx.DiGraph([(0, 2), (0, 3), (1, 2), (1, 3), (4, 0), (4, 1)])
    for pairs in [None, [(2, 3), (3, 2), (2, 2), (0, 1)]]:
        exact_outcome(listed(nx.all_pairs_lowest_common_ancestor), D, pairs)
    exact_outcome(nx.lowest_common_ancestor, D, 2, 3)
    S = nx.relabel_nodes(D, {v: f"s{v}" for v in D})
    exact_outcome(listed(nx.all_pairs_lowest_common_ancestor), S)
    for G in [nx.DiGraph(), nx.DiGraph([(0, 1), (2, 1)]), nx.DiGraph([(0, 1), (1, 0)]),
              nx.DiGraph([(0, 1), (0, 2), (3, 4)]), nx.DiGraph([(0, 0)])]:
        exact_outcome(listed(nx.all_pairs_lowest_common_ancestor), G)
        exact_outcome(listed(nx.tree_all_pairs_lowest_common_ancestor), G)
    # Odd pairs: an iterator, a non-pair, an equal node of another type.
    exact_outcome(listed(lambda G, **kw: nx.all_pairs_lowest_common_ancestor(
        G, iter([(2, 3), (0, 1)]), **kw)), D)  # a fresh iterator for each backend
    exact_outcome(listed(lambda G, **kw: nx.all_pairs_lowest_common_ancestor(
        G, [iter([2, 3])], **kw)), D)
    exact_outcome(listed(nx.all_pairs_lowest_common_ancestor), D, [(2, 3, 4)])
    exact_outcome(listed(nx.all_pairs_lowest_common_ancestor), D, [(2.0, 3)])
    exact_outcome(nx.lowest_common_ancestor, D, 2.0, 3)
    exact_outcome(nx.lowest_common_ancestor, nx.Graph([(0, 1)]), 0, 1)


def test_batch8_mutating_functions_change_the_input_like_networkx():
    # Float weights that NetworkX's rewrite doesn't restore exactly, and
    # edges without the attribute.
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=0.1)
    G.add_edge(1, 2, weight=0.7)
    G.add_edge(2, 0)
    G.add_edge(0, 3, weight=0.3)
    for func in _TRANSFORMED:
        mutating_outcome(func, G)
    mutating_outcome(nx.minimum_spanning_arborescence, nx.DiGraph([(0, 1), (2, 3)]))
    mutating_outcome(nx.maximum_spanning_arborescence, nx.DiGraph())
    mutating_outcome(nx.minimum_spanning_arborescence, nx.Graph([(0, 1)]))


def test_batch8_branching_edge_cases():
    big = 2**70
    for G in [nx.DiGraph([(0, 1, {"weight": big}), (1, 0, {"weight": 1})]),
              nx.DiGraph([(0, 1, {"weight": "x"})]),
              nx.DiGraph([(0, 0, {"weight": 5}), (0, 1, {"weight": 2})]),
              nx.DiGraph([("edmonds new node base name 0", 1, {"weight": 1}), (1, 2)]),
              nx.DiGraph([(0, 1, {"weight": float("nan")}), (1, 2, {"weight": 1.0})])]:
        exact_outcome(strict(nx.maximum_branching), G)
        exact_outcome(strict(nx.tree.greedy_branching), G)
        exact_outcome(lambda G, **kw: repr(nx.tree.branching_weight(G, **kw)), G)  # NaN
    exact_outcome(strict(nx.maximum_branching), nx.DiGraph([(0, 1, {1: 2})]), preserve_attrs=True)
    exact_outcome(strict(nx.maximum_branching), nx.DiGraph([(0, 1)]), attr=None)
    exact_outcome(strict(nx.maximum_branching), nx.DiGraph([(0, 1)]), partition="weight")
    # attr=None draws a random attribute name (from the seed's state).
    for seed in [0, 1, None]:
        exact_outcome(strict(lambda G, **kw: nx.tree.greedy_branching(
            G, attr=None, seed=random.Random(5) if seed is None else seed, **kw)),
            nx.DiGraph([(0, 1), (1, 2)]))
    exact_outcome(strict(nx.tree.greedy_branching), nx.DiGraph([(0, 1), ("a", 2)]))


@pytest.mark.parametrize("directed", [False, True])
def test_batch8_multigraphs_fall_back(directed, restore_config):
    from networkx.algorithms.tree import mst

    M = random_multigraph(3, directed, "int")
    exact_outcome(strict(nx.maximum_branching), M)
    exact_outcome(nx.tree.branching_weight, M)
    if directed:
        exact_outcome(listed(nx.all_pairs_lowest_common_ancestor), M)
    else:
        exact_outcome(listed(mst.prim_mst_edges), M, True)
    exact_outcome(listed(mst.boruvka_mst_edges), M)


def test_batch8_graph_changes_during_iteration():
    from networkx.algorithms.tree import mst

    cases = [
        (nx.Graph, lambda G, **kw: mst.prim_mst_edges(G, True, **kw)),
        (nx.Graph, lambda G, **kw: mst.boruvka_mst_edges(G, **kw)),
        (nx.DiGraph, lambda G, **kw: nx.tree_all_pairs_lowest_common_ancestor(G, 0, **kw)),
    ]
    for cls, call in cases:
        results = []
        for backend in ["rustnx", "networkx"]:
            G = nx.path_graph(8, create_using=cls)
            it = call(G, backend=backend)
            G.add_edge(7, 8)  # before the first item: seen
            results.append(list(it))
        assert results[0] == results[1]
    G = nx.path_graph(8)
    it = mst.prim_mst_edges(G, True, backend="rustnx")
    next(it)
    G.add_edge("x", "y")
    with pytest.raises(RuntimeError):
        list(it)


# --- Batch 9: planarity, chordal graphs and graph classes -------------------------

planarity = nx.algorithms.planarity
tournament = nx.algorithms.tournament


def _triangulation(seed, n):
    """A random maximal planar graph: triangles split around a new node."""
    rng = random.Random(seed)
    G = nx.Graph([(0, 1), (1, 2), (2, 0)])
    faces = [(0, 1, 2)]
    for v in range(3, n):
        a, b, c = faces.pop(rng.randrange(len(faces)))
        G.add_edges_from([(v, a), (v, b), (v, c)])
        faces += [(a, b, v), (b, c, v), (c, a, v)]
    return G


def _reordered(G, seed, directed=False, labels=False):
    """G with shuffled node and edge order (and edge directions), some
    reciprocal arcs if directed, and string labels if asked."""
    rng = random.Random(seed)
    H = nx.DiGraph() if directed else nx.Graph()
    name = (lambda v: f"v{v}") if labels else (lambda v: v)
    nodes = list(G)
    rng.shuffle(nodes)
    H.add_nodes_from(name(v) for v in nodes)
    edges = list(G.edges)
    rng.shuffle(edges)
    for u, v in edges:
        if rng.random() < 0.5:
            u, v = v, u
        H.add_edge(name(u), name(v))
        if directed and rng.random() < 0.2:
            H.add_edge(name(v), name(u))
    return H


def _planarity_graphs(seed, directed):
    rng = random.Random(seed)
    yield graph_for(seed, directed)
    T = _triangulation(seed, rng.randint(3, 30))
    yield _reordered(T, seed, directed, labels=seed % 2 == 1)
    # Nearly planar: a triangulation with one edge replaced, or with a K5
    # or K3,3 attached, or subdivided.
    H = _reordered(T, seed + 1, directed)
    edges = list(H.edges)
    H.remove_edge(*edges[seed % len(edges)])
    if seed % 3 == 0:
        H.add_edge(*rng.sample(list(H), 2))
    yield H
    K = nx.complete_graph(5) if seed % 2 else nx.complete_bipartite_graph(3, 3)
    K = nx.relabel_nodes(K, {v: 100 + v for v in K})
    if seed % 4 < 2:
        u, v = next(iter(K.edges))
        K.remove_edge(u, v)
        nx.add_path(K, [u, 200, v])
    S = nx.union(nx.Graph(T), K)
    S.add_edge(0, 100)
    yield _reordered(S, seed + 2, directed)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch9_planarity(seed, directed):
    for G in _planarity_graphs(seed, directed):
        exact_outcome(nx.is_planar, G)
        for counterexample in [False, True]:
            exact_outcome(nx.check_planarity, G, counterexample)
            exact_outcome(planarity.check_planarity_recursive, G, counterexample)
        exact_outcome(planarity.get_counterexample, G)
        exact_outcome(planarity.get_counterexample_recursive, G)


def _chordal_graph(seed):
    """A random chordal graph: each new node joins a clique of earlier
    nodes (a random partial k-tree), relabeled and reordered."""
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_node(0)
    for v in range(1, rng.randint(2, 30)):
        anchor = rng.randrange(v)
        clique = [anchor] + [u for u in G[anchor] if rng.random() < 0.6]
        clique = [u for u in clique if all(G.has_edge(u, w) for w in clique if w != u)]
        G.add_edges_from((v, u) for u in clique)
    return _reordered(G, seed, labels=seed % 2 == 0)


def _interval_graph(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 30)
    starts = [rng.randint(0, 50) for _ in range(n)]
    return nx.interval_graph([(s, s + rng.randint(0, 10)) for s in starts])


@pytest.mark.parametrize("seed", range(60))
def test_batch9_chordal_and_at_free(seed):
    G = graph_for(seed, False)
    C = _chordal_graph(seed)
    graphs = [G, C, _interval_graph(seed), nx.cycle_graph(4 + seed % 5), _reordered(_triangulation(seed, 12), seed)]
    # A few chords missing from a chordal graph make it non-chordal.
    D = C.copy()
    edges = list(D.edges)
    if edges:
        D.remove_edges_from(edges[: 1 + seed % 3])
    graphs.append(D)
    for H in graphs:
        exact_outcome(nx.is_chordal, H)
        exact_outcome(nx.chordal_graph_treewidth, H)
        exact_outcome(nx.complete_to_chordal_graph, H)
        exact_outcome(nx.is_at_free, H)
    # Directed graphs are rejected.
    for func in [nx.is_chordal, nx.chordal_graph_treewidth, nx.complete_to_chordal_graph, nx.is_at_free]:
        exact_outcome(func, graph_for(seed, True))


@nx_has("is_perfect_graph")
@pytest.mark.parametrize("seed", range(60))
def test_batch9_perfect_graphs(seed):
    rng = random.Random(seed)
    # Small graphs: NetworkX enumerates every chordless cycle of G and of
    # its complement for a perfect graph.
    G = graph_for(seed, False)
    G = G.subgraph(list(G)[:16]).copy()
    C = _chordal_graph(seed)
    C = C.subgraph(list(C)[:16]).copy()
    graphs = [G, C, nx.cycle_graph(5 + seed % 4), nx.complement(nx.cycle_graph(5 + seed % 4)),
              nx.grid_2d_graph(2 + seed % 3, 3), _interval_graph(seed).subgraph(range(0, 50, 3)).copy(),
              nx.bipartite.random_graph(rng.randint(1, 8), rng.randint(1, 8), 0.4, seed=seed)]
    # An odd hole through a node with a self-loop doesn't count in G.
    L = nx.cycle_graph(7)
    L.add_edge(0, 0)
    graphs.append(L)
    for H in graphs:
        exact_outcome(nx.is_perfect_graph, H)
    exact_outcome(nx.is_perfect_graph, graph_for(seed, True))


@pytest.mark.parametrize("seed", range(30))
def test_batch9_tournaments(seed):
    rng = random.Random(seed)
    T = tournament.random_tournament(rng.randint(0, 10), seed=seed)
    # NetworkX 3.4 and 3.5 take time quintic in the number of nodes here.
    D = graph_for(seed, True)
    D = D.subgraph(list(D)[:10]).copy()
    # A tournament with one arc reversed or removed, and the random digraph.
    U = T.copy()
    if U.number_of_edges():
        u, v = list(U.edges)[seed % U.number_of_edges()]
        U.remove_edge(u, v)
        if seed % 2:
            U.add_edge(v, u)
    for G in [T, U, D, _reordered(T, seed, directed=True, labels=True)]:
        nodes = list(G)
        for s in nodes[:3] + ["missing", [1]]:
            for t in nodes[-2:] + ["missing", [2]]:
                exact_outcome(tournament.is_reachable, G, s, t)
        exact_outcome(tournament.is_strongly_connected, G)
        exact_outcome(tournament.score_sequence, G)
    G = graph_for(seed, False)
    exact_outcome(tournament.is_reachable, G, 0, 1)
    exact_outcome(tournament.is_strongly_connected, G)
    exact_outcome(tournament.score_sequence, G)


@pytest.mark.parametrize(
    "G",
    [
        nx.empty_graph(0),
        nx.empty_graph(1),
        nx.empty_graph(3),
        nx.path_graph(3),
        nx.complete_graph(4),
        nx.complete_graph(5),
        nx.complete_bipartite_graph(3, 3),
        nx.petersen_graph(),
        nx.Graph([(0, 0)]),
        nx.Graph([(0, 0), (1, 2)]),
        nx.Graph([(0, 0), (1, 2), (2, 3), (3, 4)]),
        nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (1, 1)]),
        nx.DiGraph(),
        nx.DiGraph([(0, 0)]),
        nx.DiGraph([(0, 1), (1, 0)]),
        nx.DiGraph(nx.complete_graph(5)),
    ],
    ids=lambda G: f"{type(G).__name__}{list(G.edges)}",
)
def test_batch9_small_cases(G):
    funcs = [nx.is_planar, nx.check_planarity, planarity.check_planarity_recursive,
             planarity.get_counterexample, planarity.get_counterexample_recursive,
             nx.is_chordal, nx.chordal_graph_treewidth, nx.complete_to_chordal_graph,
             nx.is_at_free, tournament.is_strongly_connected, tournament.score_sequence]
    for func in funcs:
        exact_outcome(func, G)
    exact_outcome(nx.check_planarity, G, counterexample=True)
    exact_outcome(tournament.is_reachable, G, 0, 1)


def test_batch9_recursion_limit():
    # The recursive variants recurse once per DFS level: rustnx leaves deep
    # graphs to NetworkX, which may raise RecursionError.
    G = nx.path_graph(5000)
    exact_outcome(planarity.check_planarity_recursive, G)
    exact_outcome(nx.check_planarity, G)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_batch9_multigraphs(seed, directed, restore_config):
    M = random_multigraph(seed, directed, "none")
    # NetworkX merges parallel edges here; these fall back.
    exact_outcome(nx.is_planar, M)
    exact_outcome(nx.check_planarity, M, True)
    if directed:
        exact_outcome(tournament.score_sequence, M)
    else:
        exact_outcome(nx.complete_to_chordal_graph, M)
        exact_outcome(nx.chordal_graph_treewidth, M)


# --- Batch 10: triads, d-separation, degree sequences and matching -----------------


def _random_dag(seed):
    """A random DAG from ``graph_for``: edges kept only from earlier to
    later nodes in a shuffled order (node and edge order stay random)."""
    G = graph_for(seed, True)
    rng = random.Random(seed)
    rank = list(G)
    rng.shuffle(rank)
    rank = {v: i for i, v in enumerate(rank)}
    G.remove_edges_from([(u, v) for u, v in G.edges if rank[u] >= rank[v]])
    return G


def _node_sets(G, rng):
    """Disjoint random node sets (some as single nodes)."""
    nodes = list(G)
    rng.shuffle(nodes)
    k = len(nodes)
    cuts = sorted(rng.randint(0, k) for _ in range(3))
    x, y, z = nodes[: cuts[0]], nodes[cuts[0] : cuts[1]], nodes[cuts[1] : cuts[2]]
    out = []
    for s in (x, y, z):
        if len(s) == 1 and rng.random() < 0.5:
            out.append(s[0])
        else:
            out.append(set(s) if rng.random() < 0.8 else frozenset(s))
    return out


@pytest.mark.parametrize("seed", range(60))
def test_batch10_triadic_census(seed):
    D = graph_for(seed, True)
    rng = random.Random(seed)
    nodes = list(D)
    exact_outcome(nx.triadic_census, D)
    for k in [0, 1, 3, len(nodes) // 2, len(nodes)]:
        exact_outcome(nx.triadic_census, D, rng.sample(nodes, min(k, len(nodes))))
    exact_outcome(nx.triadic_census, D, nodes[:2] + nodes[:1])  # duplicates
    exact_outcome(nx.triadic_census, D, nodes[:2] + ["missing"])
    exact_outcome(nx.triadic_census, D, tuple(nodes[:3]))
    exact_outcome(nx.triadic_census, D, nodes[0])  # one node: len() fails
    exact_outcome(nx.triadic_census, graph_for(seed, False))


@pytest.mark.parametrize("seed", range(60))
def test_batch10_d_separation(seed):
    D = _random_dag(seed)
    rng = random.Random(seed)
    for _ in range(4):
        x, y, z = _node_sets(D, rng)
        exact_outcome(nx.is_d_separator, D, x, y, z)
        exact_outcome(with_set_order(nx.find_minimal_d_separator), D, x, y)
        exact_outcome(nx.is_minimal_d_separator, D, x, y, z)
        found = nx.find_minimal_d_separator(D, x, y, backend="networkx")
        if found is not None:
            exact_outcome(nx.is_minimal_d_separator, D, x, y, found)
            extra = set(found) | ({next(iter(z))} if isinstance(z, (set, frozenset)) and z else set())
            exact_outcome(nx.is_minimal_d_separator, D, x, y, extra)
        # included and restricted
        rest = [v for v in D if v not in (x if isinstance(x, (set, frozenset)) else {x})
                and v not in (y if isinstance(y, (set, frozenset)) else {y})]
        inc = set(rng.sample(rest, min(1, len(rest))))
        res = set(rng.sample(rest, min(len(rest), rng.randint(0, 6)))) | inc
        for kw in [{"included": inc}, {"restricted": res}, {"included": inc, "restricted": res},
                   {"restricted": inc}, {"included": res, "restricted": inc}]:
            exact_outcome(with_set_order(nx.find_minimal_d_separator), D, x, y, **kw)
            exact_outcome(nx.is_minimal_d_separator, D, x, y, z, **kw)
    nodes = list(D)
    a, b = nodes[0], nodes[-1]
    # Errors: overlaps, missing nodes, non-sets, cycles, undirected graphs.
    for args in [({a}, {a}, set()), ({a}, {b}, {a}), ({a}, {"missing"}, set()),
                 ({a}, {b, "gone"}, {"missing"}), ([a], [b], []), (a, b, [a])]:
        exact_outcome(nx.is_d_separator, D, *args)
        exact_outcome(nx.is_minimal_d_separator, D, *args)
        exact_outcome(with_set_order(nx.find_minimal_d_separator), D, *args[:2])
        exact_outcome(nx.is_minimal_d_separator, D, *args, included={"missing"})
        exact_outcome(nx.is_minimal_d_separator, D, *args, restricted={a})
        exact_outcome(with_set_order(nx.find_minimal_d_separator), D, *args[:2], restricted=set(nodes[1:3]))
    C = graph_for(seed, True)
    if not nx.is_directed_acyclic_graph(C):
        exact_outcome(nx.is_d_separator, C, {a}, {b}, set())
        exact_outcome(nx.find_minimal_d_separator, C, a, b)
    exact_outcome(nx.is_d_separator, graph_for(seed, False), a, b, set())


def test_batch10_d_separation_known_graphs():
    # NetworkX's documentation examples (collider, chain, fork).
    D = nx.path_graph(4, create_using=nx.DiGraph)
    exact_outcome(nx.is_d_separator, D, 0, 2, {1})
    exact_outcome(nx.is_d_separator, D, 0, 2, set())
    D = nx.DiGraph([("x", "z"), ("y", "z"), ("z", "w"), ("a", "x"), ("a", "y")])
    for z in [set(), {"z"}, {"w"}, {"a"}, {"a", "w"}]:
        exact_outcome(nx.is_d_separator, D, "x", "y", z)
        exact_outcome(nx.is_minimal_d_separator, D, "x", "y", z)
    exact_outcome(with_set_order(nx.find_minimal_d_separator), D, "x", "y")
    exact_outcome(with_set_order(nx.find_minimal_d_separator), D, {"x"}, {"w"})
    D = nx.DiGraph([(1, 2), (2, 3)])
    D.add_node(1.0)  # equal to 1: the given object differs from G's
    exact_outcome(with_set_order(nx.find_minimal_d_separator), D, {1.0}, {3})


def _sequences(seed):
    rng = random.Random(seed)
    G = graph_for(seed, False)
    degrees = [d for _, d in G.degree()]
    yield degrees
    yield sorted(degrees, reverse=True)
    yield [rng.randint(0, 6) for _ in range(rng.randint(0, 12))]
    yield [rng.randint(-1, 4) for _ in range(rng.randint(1, 8))]
    yield [float(d) for d in degrees]
    yield degrees[:3] + [2.5]
    yield degrees[:2] + ["x"]
    yield [True, True, 2, 0]
    yield [2**70, 2**70]
    yield []
    yield [0]
    yield [3, 3, 3, 3]


def _sequence_outcome(func, *seqs):
    """Both backends' results on fresh copies, with what each did to the
    given lists (NetworkX converts some in place)."""
    def run(backend):
        copies = [list(s) if isinstance(s, list) else s for s in seqs]
        try:
            result = ("ok", func(*copies, backend=backend))
        except Exception as exc:
            result = (type(exc), exc.args)
        return result, [(type(c), list(c)) if isinstance(c, list) else None for c in copies]

    assert run("rustnx") == run("networkx")


@pytest.mark.parametrize("seed", range(60))
def test_batch10_degree_sequences(seed):
    for seq in _sequences(seed):
        for func in [nx.is_valid_degree_sequence_erdos_gallai, nx.is_valid_degree_sequence_havel_hakimi,
                     nx.is_graphical, nx.is_multigraphical, nx.is_pseudographical]:
            _sequence_outcome(func, seq)
        _sequence_outcome(lambda s, **kw: nx.is_graphical(s, method="hh", **kw), seq)
        _sequence_outcome(lambda s, **kw: nx.is_graphical(iter(s), **kw), seq)
        _sequence_outcome(lambda s, **kw: nx.is_multigraphical(tuple(s), **kw), seq)
        _sequence_outcome(lambda s, **kw: nx.is_pseudographical(iter(s), **kw), seq)
        _sequence_outcome(nx.is_digraphical, seq, seq)
        _sequence_outcome(nx.is_digraphical, seq, list(reversed(seq)))
    D = graph_for(seed, True)
    ins = [d for _, d in D.in_degree()]
    outs = [d for _, d in D.out_degree()]
    _sequence_outcome(nx.is_digraphical, ins, outs)
    _sequence_outcome(nx.is_digraphical, ins, outs[:-1])
    _sequence_outcome(nx.is_digraphical, ins + [1], outs + [0])
    _sequence_outcome(lambda s, **kw: nx.is_graphical(s, method="bogus", **kw), [1, 1])


def test_batch10_degree_sequences_dispatch_without_graphs(restore_config):
    # No graph argument: rustnx runs only when asked, or by priority.
    nx.config.backend_priority.algos = ["rustnx"]
    assert nx.is_graphical([2, 2, 2]) is True
    assert nx.is_valid_degree_sequence_erdos_gallai([3, 1]) is False


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(60))
def test_batch10_boundaries(seed, directed):
    G = graph_for(seed, directed)
    rng = random.Random(seed)
    nodes = list(G)
    for _ in range(3):
        a = rng.sample(nodes, rng.randint(0, len(nodes)))
        b = rng.sample(nodes, rng.randint(0, len(nodes)))
        for n1, n2 in [(a, None), (a, b), (a + ["missing", [1]], None), (iter(a), b + ["missing"]),
                       (set(a), set(b)), (a, a)]:
            if hasattr(n1, "__next__"):
                exact_outcome(with_set_order(lambda G, **kw: nx.node_boundary(G, iter(a), b, **kw)), G)
                exact_outcome(listed(lambda G, **kw: nx.edge_boundary(G, iter(a), b, **kw)), G)
                continue
            exact_outcome(with_set_order(nx.node_boundary), G, n1, n2)
            exact_outcome(listed(nx.edge_boundary), G, n1, n2)
        exact_outcome(listed(nx.edge_boundary), G, a, data=True)


@pytest.mark.parametrize("weights", ["none", "int", "float", "missing"])
@pytest.mark.parametrize("seed", range(60))
def test_batch10_matching(seed, weights):
    G = graph_for(seed, False, weights)
    if seed % 4 == 0:
        # Dense graphs make blossoms (and expand them) more often.
        rng = random.Random(seed)
        G = nx.gnp_random_graph(rng.randint(5, 25), 0.5, seed=seed)
        for u, v, d in G.edges(data=True):
            if weights == "int":
                d["weight"] = rng.randint(1, 9)
            elif weights == "float":
                d["weight"] = rng.choice([0.1, 0.25, 0.7, 1.3, 2.9])
            elif weights == "missing" and rng.random() < 0.5:
                d["weight"] = rng.randint(-3, 9)
    exact_outcome(with_set_order(nx.maximal_matching), G)
    for card in [False, True]:
        exact_outcome(with_set_order(nx.max_weight_matching), G, card)
        exact_outcome(with_set_order(nx.max_weight_matching), G, card, weight=None)
    exact_outcome(with_set_order(nx.min_weight_matching), G)
    exact_outcome(with_set_order(nx.min_weight_matching), G, weight=None)
    exact_outcome(with_set_order(nx.min_edge_cover), G)
    H = G.copy()
    H.remove_nodes_from(list(nx.isolates(H)))
    exact_outcome(with_set_order(nx.min_edge_cover), H)
    rng = random.Random(seed)
    mate = nx.max_weight_matching(G, backend="networkx")
    maximal = nx.maximal_matching(G, backend="networkx")
    edges = list(G.edges)
    matchings = [mate, maximal, list(mate)[:-1], dict(nx.utils.flatten([[(u, v), (v, u)] for u, v in mate]) and
                 {k: v for u, w in mate for k, v in [(u, w), (w, u)]}), set()]
    if edges:
        matchings += [list(mate) + [rng.choice(edges)], [rng.choice(edges)] + list(mate)]
    nodes = list(G)
    matchings += [[(nodes[0], nodes[0])], [(nodes[0], "missing")], [(nodes[0],)],
                  [(nodes[0], nodes[0]), (nodes[0], "missing")], [(nodes[0], nodes[0]), 5],
                  [(nodes[0], "missing"), (nodes[0], nodes[0])], {nodes[0]: nodes[0]}]
    for m in matchings:
        for func in [nx.is_matching, nx.is_maximal_matching, nx.is_perfect_matching]:
            exact_outcome(func, G, m)
            exact_outcome(lambda G, **kw: func(G, iter(m) if not isinstance(m, dict) else m, **kw), G)
    D = graph_for(seed, True, weights)
    exact_outcome(nx.max_weight_matching, D)
    exact_outcome(nx.maximal_matching, D)
    exact_outcome(nx.is_matching, D, list(D.edges)[:2])


def test_batch10_matching_known_graphs():
    # NetworkX's blossom tests: nested blossoms, expansion, and S-blossom
    # relabeling, with integer and float weights.
    cases = [
        [(1, 2, 8), (1, 3, 9), (2, 3, 10), (3, 4, 7)],
        [(1, 2, 9), (1, 3, 8), (2, 3, 10), (1, 4, 5), (4, 5, 4), (1, 6, 3)],
        [(1, 2, 9), (1, 3, 9), (2, 3, 10), (2, 4, 8), (3, 5, 8), (4, 5, 10), (5, 6, 6)],
        [(1, 2, 10), (1, 7, 10), (2, 3, 12), (3, 4, 20), (3, 5, 20), (4, 5, 25), (5, 6, 10),
         (6, 7, 10), (7, 8, 8)],
        [(1, 2, 8), (1, 3, 8), (2, 3, 10), (2, 4, 12), (3, 5, 12), (4, 5, 14), (4, 6, 12),
         (5, 7, 12), (6, 7, 14), (7, 8, 12)],
        [(1, 2, 23), (1, 5, 22), (1, 6, 15), (2, 3, 25), (3, 4, 22), (4, 5, 25), (4, 8, 14),
         (5, 7, 13)],
        [(1, 2, 19), (1, 3, 20), (1, 8, 8), (2, 3, 25), (2, 4, 18), (3, 5, 18), (4, 5, 13),
         (4, 7, 7), (5, 6, 7)],
        [(1, 2, 45), (1, 5, 45), (2, 3, 50), (3, 4, 45), (4, 5, 50), (1, 6, 30), (3, 9, 35),
         (4, 8, 35), (5, 7, 26), (9, 10, 5)],
        [(1, 2, 45), (1, 5, 45), (2, 3, 50), (3, 4, 45), (4, 5, 50), (1, 6, 30), (3, 9, 35),
         (4, 8, 26), (5, 7, 40), (9, 10, 5)],
        [(1, 2, 40), (1, 3, 40), (2, 3, 60), (2, 4, 55), (3, 5, 55), (4, 5, 50), (1, 8, 15),
         (5, 7, 30), (7, 6, 10), (8, 10, 10), (4, 9, 30)],
    ]
    for edges in cases:
        for scale in [1, 0.1, 1.5]:
            G = nx.Graph()
            G.add_weighted_edges_from((u, v, w * scale if scale != 1 else w) for u, v, w in edges)
            for card in [False, True]:
                exact_outcome(with_set_order(nx.max_weight_matching), G, card)
            exact_outcome(with_set_order(nx.min_weight_matching), G)
    G = nx.Graph([(0, 1, {"weight": 2**60})])
    exact_outcome(with_set_order(nx.max_weight_matching), G)  # falls back
    G = nx.Graph([(0, 1, {"weight": None})])
    exact_outcome(with_set_order(nx.max_weight_matching), G)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(60))
def test_batch10_dominating(seed, directed):
    G = graph_for(seed, directed)
    rng = random.Random(seed)
    nodes = list(G)
    if not nodes:
        return
    ds = nx.dominating_set(G, backend="networkx")
    for nb in [ds, list(ds)[:-1], nodes, nodes[:1], [], nodes + ["missing", [1]],
               rng.sample(nodes, len(nodes) // 2)]:
        exact_outcome(nx.is_dominating_set, G, nb)
        exact_outcome(lambda G, **kw: nx.is_dominating_set(G, iter(nb), **kw), G)
        if hasattr(nx, "is_connected_dominating_set"):
            exact_outcome(nx.is_connected_dominating_set, G, nb)
            exact_outcome(lambda G, **kw: nx.is_connected_dominating_set(G, iter(nb), **kw), G)
    exact_outcome(nx.is_dominating_set, G, 5)
    if hasattr(nx, "connected_dominating_set"):
        exact_outcome(with_set_order(nx.connected_dominating_set), G)
        if not directed and nodes:
            H = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            exact_outcome(with_set_order(nx.connected_dominating_set), H)
            cds = nx.connected_dominating_set(H, backend="networkx")
            exact_outcome(nx.is_connected_dominating_set, H, cds)


@nx_has("connected_dominating_set")
def test_batch10_dominating_small_cases():
    for G in [nx.Graph(), nx.empty_graph(1), nx.Graph([(0, 0)]), nx.path_graph(2), nx.star_graph(5),
              nx.Graph([(0, 0), (0, 1), (1, 2)])]:
        exact_outcome(with_set_order(nx.connected_dominating_set), G)
        exact_outcome(nx.is_connected_dominating_set, G, list(G))
        exact_outcome(nx.is_connected_dominating_set, G, [])
    G = nx.Graph([("ab", "c"), ("c", "a"), ("a", "b")])
    exact_outcome(nx.is_connected_dominating_set, G, "ab")  # a node, and a string


@pytest.mark.parametrize("seed", range(60))
def test_batch10_cliques(seed):
    G = graph_for(seed, False)
    rng = random.Random(seed)
    if seed % 3 == 0:
        G = nx.gnp_random_graph(rng.randint(3, 18), 0.6, seed=seed)
    nodes = list(G)
    exact_outcome(listed(nx.enumerate_all_cliques), G)
    exact_outcome(nx.node_clique_number, G, nodes)
    exact_outcome(nx.node_clique_number, G, nodes[:3] + nodes[:1])
    exact_outcome(nx.node_clique_number, G, nodes[0])
    exact_outcome(nx.node_clique_number, G, nodes[:2] + ["missing"])
    exact_outcome(nx.node_clique_number, G)
    exact_outcome(nx.max_weight_clique, G, weight=None)
    exact_outcome(nx.max_weight_clique, G)  # no node weights: KeyError
    for v in G:
        G.nodes[v]["weight"] = rng.randint(-2, 9)
    exact_outcome(nx.max_weight_clique, G)
    G.nodes[nodes[-1]]["weight"] = 1.5
    exact_outcome(nx.max_weight_clique, G)
    D = graph_for(seed, True)
    exact_outcome(listed(nx.enumerate_all_cliques), D)
    exact_outcome(nx.max_weight_clique, D)
    exact_outcome(nx.node_clique_number, D, list(D))


@pytest.mark.parametrize("seed", range(20))
def test_batch10_multigraphs(seed, restore_config):
    M = random_multigraph(seed, False, "none")
    nodes = list(M)
    mate = list(nx.maximal_matching(nx.Graph(M), backend="networkx"))
    exact_outcome(nx.is_matching, M, mate)
    exact_outcome(nx.is_perfect_matching, M, mate)
    exact_outcome(nx.is_dominating_set, M, nodes[: len(nodes) // 2])
    exact_outcome(with_set_order(nx.node_boundary), M, nodes[: len(nodes) // 2])
    # These fall back.
    exact_outcome(nx.is_maximal_matching, M, mate)
    exact_outcome(listed(nx.edge_boundary), M, nodes[:3])
    exact_outcome(listed(nx.enumerate_all_cliques), M)
    D = random_multigraph(seed, True, "none")
    exact_outcome(nx.triadic_census, D)


def test_batch10_graph_changes_during_iteration():
    G = nx.path_graph(6)
    it = nx.edge_boundary(G, [0, 1, 2], backend="rustnx")
    G.add_edge(1, 9)
    with pytest.raises(RuntimeError):
        list(it)
    # enumerate_all_cliques reads the graph when iteration starts.
    results = []
    for backend in ["rustnx", "networkx"]:
        G = nx.complete_graph(4)
        it = nx.enumerate_all_cliques(G, backend=backend)
        G.remove_edge(0, 1)  # before the first item: seen
        first = next(it)
        G.add_edge("x", "y")  # after: not seen
        results.append([first, *it])
    assert results[0] == results[1]


def test_batch10_runs_in_rust():
    # Check rustnx really handles the common calls (no silent fallback).
    G = nx.gnp_random_graph(30, 0.2, seed=1)
    D = nx.gnp_random_graph(30, 0.1, seed=1, directed=True)
    dag = nx.DiGraph([(u, v) for u, v in D.edges if u < v])
    calls = [
        lambda: nx.triadic_census(D, backend="rustnx"),
        lambda: nx.is_d_separator(dag, 0, 1, set(), backend="rustnx"),
        lambda: nx.find_minimal_d_separator(dag, {0}, {29}, backend="rustnx"),
        lambda: nx.is_minimal_d_separator(dag, {0}, {29}, set(), backend="rustnx"),
        lambda: nx.is_graphical([2, 2, 2], backend="rustnx"),
        lambda: nx.is_digraphical([1, 1], [1, 1], backend="rustnx"),
        lambda: nx.node_boundary(G, [0, 1], backend="rustnx"),
        lambda: list(nx.edge_boundary(G, [0, 1], backend="rustnx")),
        lambda: nx.max_weight_matching(G, backend="rustnx"),
        lambda: nx.min_weight_matching(G, backend="rustnx"),
        lambda: nx.maximal_matching(G, backend="rustnx"),
        lambda: nx.is_matching(G, {(0, 1)}, backend="rustnx"),
        lambda: nx.is_dominating_set(G, [0], backend="rustnx"),
        lambda: list(nx.enumerate_all_cliques(G, backend="rustnx")),
        lambda: nx.node_clique_number(G, [0, 1], backend="rustnx"),
        lambda: nx.max_weight_clique(G, weight=None, backend="rustnx"),
    ]
    for call in calls:
        call()


# --- Batch 11: isomorphism and graph hashing ---------------------------------------

from networkx.algorithms.isomorphism.tree_isomorphism import root_trees  # noqa: E402


def _shuffled_copy(G, seed, labels=True):
    """An isomorphic copy with new node names and a new insertion order."""
    rng = random.Random(seed)
    nodes = list(G)
    names = nodes[:]
    rng.shuffle(names)
    rename = {v: (f"c{w}" if labels else w) for v, w in zip(nodes, names)}
    H = G.__class__()
    order = nodes[:]
    rng.shuffle(order)
    H.add_nodes_from((rename[v], G.nodes[v]) for v in order)
    edges = list(G.edges(data=True))
    rng.shuffle(edges)
    H.add_edges_from((rename[u], rename[v], d) for u, v, d in edges)
    return H


def _swapped(G, seed):
    """Same degrees (usually), different edges: hard cases for the checks."""
    H = G.copy()
    try:
        if H.is_directed():
            nx.directed_edge_swap(H, nswap=2, max_tries=200, seed=seed)
        else:
            nx.double_edge_swap(H, nswap=2, max_tries=200, seed=seed)
    except (nx.NetworkXError, nx.NetworkXAlgorithmError):
        pass
    return H


def _batch11_pairs(seed, directed):
    G = graph_for(seed, directed)
    H = _shuffled_copy(G, seed)
    K = _swapped(G, seed)
    other = graph_for(seed + 1000, directed)
    return G, [(G, H), (G, K), (H, K), (G, G), (G, other), (K, _shuffled_copy(K, seed + 1))]


def _could_be_calls():
    calls = [nx.could_be_isomorphic, nx.fast_could_be_isomorphic, nx.faster_could_be_isomorphic]
    if "properties" in inspect.signature(nx.could_be_isomorphic).parameters:
        for props in ["d", "t", "c", "dt", "tc", "cd", "", "xyz", ["d", "c"]]:
            calls.append(lambda A, B, props=props, **kw: nx.could_be_isomorphic(A, B, properties=props, **kw))
    return calls


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(60))
def test_batch11_isomorphism(seed, directed):
    G, pairs = _batch11_pairs(seed, directed)
    for A, B in pairs:
        exact_outcome(nx.is_isomorphic, A, B)
        exact_outcome(nx.vf2pp_is_isomorphic, A, B)
        for func in _could_be_calls():
            exact_outcome(func, A, B)
    # Mixed directedness, and a graph against an empty one.
    U = graph_for(seed, not directed)
    for func in [nx.is_isomorphic, nx.vf2pp_is_isomorphic, *_could_be_calls()]:
        exact_outcome(func, G, U)
        exact_outcome(func, U, G)
        exact_outcome(func, G, G.__class__())
        exact_outcome(func, G.__class__(), G.__class__())


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch11_vf2pp_labels(seed, directed):
    rng = random.Random(seed)
    G = graph_for(seed, directed)
    colors = ["red", "blue", 1, 1.0, True, None, (1, 2)]
    for v in G:
        if rng.random() < 0.8:
            G.nodes[v]["color"] = rng.choice(colors[: rng.randint(1, len(colors))])
    H = _shuffled_copy(G, seed)
    K = H.copy()
    if len(K):
        K.nodes[rng.choice(list(K))]["color"] = "green"
    for A, B in [(G, H), (G, K), (H, G)]:
        for label, default in [("color", None), ("color", "red"), ("shape", None), (None, None), (None, 5)]:
            exact_outcome(nx.vf2pp_is_isomorphic, A, B, node_label=label, default_label=default)
            exact_outcome(nx.vf2pp_is_isomorphic, A, B, label, default)
            if hasattr(nx, "vf2pp_is_monomorphic"):
                exact_outcome(nx.vf2pp_is_monomorphic, A, B, node_label=label, default_label=default)
                exact_outcome(nx.vf2pp_subgraph_is_isomorphic, A, B, node_label=label, default_label=default)
    G.nodes[next(iter(G))]["color"] = ["unhashable"]
    exact_outcome(nx.vf2pp_is_isomorphic, G, H, node_label="color")


def _random_subgraph(G, seed, induced):
    rng = random.Random(seed)
    nodes = [v for v in G if rng.random() < 0.6] or list(G)[:1]
    S = G.subgraph(nodes).copy()
    if not induced:
        edges = list(S.edges)
        S.remove_edges_from(rng.sample(edges, len(edges) // 4))
    return _shuffled_copy(S, seed)


@nx_has("vf2pp_is_monomorphic")
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(60))
def test_batch11_vf2pp_subgraphs(seed, directed):
    # At most 12 nodes: subgraph searches on larger random graphs can take
    # NetworkX minutes.
    G = graph_for(seed, directed)
    G = G.subgraph(list(G)[:12]).copy()
    small = [
        _random_subgraph(G, seed, True),
        _random_subgraph(G, seed + 1, False),
        nx.gnp_random_graph(4 + seed % 5, 0.4, seed=seed, directed=directed),  # small: no hard searches
        nx.path_graph(3, create_using=G.__class__),
        nx.cycle_graph(4, create_using=G.__class__),
        nx.complete_graph(3, create_using=G.__class__),
        G.__class__([(0, 0), (0, 1)]),
        G.__class__([(0, 1)]),
    ]
    for S in small:
        for func in [nx.vf2pp_is_monomorphic, nx.vf2pp_subgraph_is_isomorphic]:
            exact_outcome(func, G, S)
            exact_outcome(func, S, G)
            exact_outcome(func, S, S)
    for func in [nx.vf2pp_is_monomorphic, nx.vf2pp_subgraph_is_isomorphic]:
        exact_outcome(func, G, G.__class__())
        exact_outcome(func, G, graph_for(seed, not directed))


def test_batch11_known_isomorphism_cases():
    # Regular graphs and other cases where degrees say nothing.
    cases = [
        (nx.petersen_graph(), nx.circulant_graph(10, [1, 3])),
        (nx.cycle_graph(6), nx.Graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])),
        (nx.circular_ladder_graph(5), nx.cycle_graph(10)),
        (nx.hypercube_graph(3), nx.circulant_graph(8, [1, 4])),
        (nx.complete_bipartite_graph(3, 3), nx.circulant_graph(6, [1, 3])),
        (nx.Graph([(0, 0), (1, 1)]), nx.Graph([(5, 5), (6, 6)])),
        (nx.Graph([(0, 0), (0, 1)]), nx.Graph([(0, 1), (1, 1)])),
        (nx.DiGraph([(0, 1), (1, 2), (2, 0)]), nx.DiGraph([(0, 2), (2, 1), (1, 0)])),
        (nx.DiGraph([(0, 1), (1, 0)]), nx.DiGraph([(0, 1)])),
        (nx.empty_graph(3), nx.empty_graph(3)),
    ]
    for A, B in cases:
        for func in [nx.is_isomorphic, nx.vf2pp_is_isomorphic, *_could_be_calls()]:
            exact_outcome(func, A, B)
        if hasattr(nx, "vf2pp_is_monomorphic"):
            exact_outcome(nx.vf2pp_is_monomorphic, A, B)
            exact_outcome(nx.vf2pp_subgraph_is_isomorphic, A, B)


def test_batch11_runs_in_rust():
    G = nx.gnm_random_graph(80, 200, seed=3)
    H = _shuffled_copy(G, 3)
    T = nx.Graph([(i, (i - 1) // 2) for i in range(1, 50)])
    calls = [
        lambda: nx.is_isomorphic(G, H, backend="rustnx"),
        lambda: nx.vf2pp_is_isomorphic(G, H, backend="rustnx"),
        lambda: nx.vf2pp_is_isomorphic(G, H, node_label="color", backend="rustnx"),
        lambda: nx.could_be_isomorphic(G, H, backend="rustnx"),
        lambda: nx.fast_could_be_isomorphic(G, H, backend="rustnx"),
        lambda: nx.faster_could_be_isomorphic(G, H, backend="rustnx"),
        lambda: nx.isomorphism.tree_isomorphism(T, T, backend="rustnx"),
        lambda: nx.isomorphism.rooted_tree_isomorphism(T, 0, T, 0, backend="rustnx"),
        lambda: root_trees(T, 0, T, 1, backend="rustnx"),
        lambda: nx.weisfeiler_lehman_graph_hash(G, backend="rustnx"),
        lambda: nx.weisfeiler_lehman_subgraph_hashes(G, backend="rustnx"),
    ]
    if hasattr(nx, "vf2pp_is_monomorphic"):
        P = nx.path_graph(4)
        calls.append(lambda: nx.vf2pp_is_monomorphic(G, P, backend="rustnx"))
        calls.append(lambda: nx.vf2pp_subgraph_is_isomorphic(G, P, backend="rustnx"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for call in calls:
            call()  # NotImplementedError here would mean a fallback


def _shuffled_tree(n, seed, labels=False):
    rng = random.Random(seed)
    edges = [(i, rng.randrange(i)) for i in range(1, n)]
    rng.shuffle(edges)
    T = nx.Graph()
    order = list(range(n))
    rng.shuffle(order)
    T.add_nodes_from(order)
    T.add_edges_from((u, v) if rng.random() < 0.5 else (v, u) for u, v in edges)
    if labels:
        T = nx.relabel_nodes(T, {v: f"t{v}" for v in T})
    return T


@pytest.mark.parametrize("seed", range(80))
def test_batch11_trees(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 40)
    T = _shuffled_tree(n, seed)
    U = _shuffled_copy(T, seed)
    V = _shuffled_tree(n, seed + 1, labels=True)
    W = _shuffled_tree(rng.randint(1, 40), seed + 2)
    for A, B in [(T, U), (U, T), (T, V), (T, W), (T, T)]:
        exact_outcome(nx.isomorphism.tree_isomorphism, A, B)
        for _ in range(3):
            r1, r2 = rng.choice(list(A)), rng.choice(list(B))
            exact_outcome(nx.isomorphism.rooted_tree_isomorphism, A, r1, B, r2)
            exact_outcome(root_trees, A, r1, B, r2)
    # Not trees, empty graphs, missing roots, directed trees.
    C = T.copy()
    C.add_edge("x", "y")
    G = graph_for(seed, False)
    D = nx.DiGraph(T)
    for A, B in [(C, T), (T, C), (G, G), (T, nx.Graph()), (nx.Graph(), T), (T, D), (D, D)]:
        exact_outcome(nx.isomorphism.tree_isomorphism, A, B)
        a = next(iter(A), 0)
        b = next(iter(B), 0)
        exact_outcome(nx.isomorphism.rooted_tree_isomorphism, A, a, B, b)
        exact_outcome(root_trees, A, a, B, b)
    exact_outcome(nx.isomorphism.rooted_tree_isomorphism, T, "missing", T, next(iter(T)))
    exact_outcome(root_trees, T, "missing", T, next(iter(T)))
    exact_outcome(root_trees, G, next(iter(G)), D, next(iter(D)))


def test_batch11_deep_trees():
    # NetworkX 3.4 walks the result recursively; rustnx leaves deep trees
    # to it there (it may hit the recursion limit).
    P = nx.path_graph(1200)
    Q = _shuffled_copy(P, 1)
    exact_outcome(nx.isomorphism.tree_isomorphism, P, Q)
    exact_outcome(nx.isomorphism.rooted_tree_isomorphism, P, 0, Q, "c0")
    exact_outcome(nx.isomorphism.rooted_tree_isomorphism, P, 0, P, 0)


def _labelled_graph(seed, directed, weights):
    rng = random.Random(seed)
    G = graph_for(seed, directed, weights)
    for v in G:
        G.nodes[v]["label"] = rng.choice(["A", "B", 3, 2.5, None])
    return G


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch11_weisfeiler_lehman(seed, directed, weights):
    G = _labelled_graph(seed, directed, weights)
    edge_attrs = [None] if weights == "none" else [None, "weight"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for edge_attr in edge_attrs:
            for node_attr in [None, "label"]:
                for iterations in [1, 2, 3]:
                    for digest_size in [16, 5]:
                        kw = dict(edge_attr=edge_attr, node_attr=node_attr, iterations=iterations, digest_size=digest_size)
                        exact_outcome(nx.weisfeiler_lehman_graph_hash, G, **kw)
                        exact_outcome(nx.weisfeiler_lehman_subgraph_hashes, G, **kw)
                        exact_outcome(nx.weisfeiler_lehman_subgraph_hashes, G, include_initial_labels=True, **kw)


def test_batch11_weisfeiler_lehman_edge_cases():
    G = nx.Graph([(0, 1, {"w": 1}), (1, 2, {"w": 2.5}), (2, 2, {"w": True})])
    G.nodes[0]["label"] = "x"
    cases = [
        (nx.Graph(), {}),
        (nx.DiGraph(), {}),
        (G, {"edge_attr": "w"}),
        (G, {"edge_attr": "missing"}),
        (G, {"node_attr": "label"}),  # missing on most nodes: KeyError
        (G, {"iterations": 0}),
        (G, {"iterations": -1, "edge_attr": "w"}),
        (G, {"digest_size": 64}),
        (G, {"digest_size": 1}),
        (nx.Graph([("é", "a")]), {}),
        (nx.Graph([(0, 1, {"w": "é"})]), {"edge_attr": "w"}),
        (nx.Graph([(0, 1, {"w": "red"})]), {"edge_attr": "w"}),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for H, kw in cases:
            exact_outcome(nx.weisfeiler_lehman_graph_hash, H, **kw)
            exact_outcome(nx.weisfeiler_lehman_subgraph_hashes, H, **kw)
    H = nx.Graph([(0, 1)])
    nx.set_node_attributes(H, {0: "é", 1: "a"}, "label")
    exact_outcome(nx.weisfeiler_lehman_graph_hash, H, node_attr="label")
    # Long inputs span several BLAKE2b blocks.
    exact_outcome(nx.weisfeiler_lehman_graph_hash, nx.star_graph(200))
    exact_outcome(nx.weisfeiler_lehman_subgraph_hashes, nx.complete_graph(30), digest_size=64)


@pytest.mark.parametrize("directed", [False, True])
def test_batch11_weisfeiler_lehman_warns_like_networkx(directed):
    G = nx.gnm_random_graph(30, 60, seed=1, directed=directed)

    def run(backend, func, **kw):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = ("ok", func(G, backend=backend, **kw))
            except Exception as exc:
                result = (type(exc), exc.args)
        return result, [(w.category, str(w.message)) for w in caught]

    for func in [nx.weisfeiler_lehman_graph_hash, nx.weisfeiler_lehman_subgraph_hashes]:
        for kw in [{}, {"node_attr": "x"}, {"iterations": 0}]:
            assert run("rustnx", func, **kw) == run("networkx", func, **kw)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_batch11_multigraphs(seed, directed, restore_config):
    M = random_multigraph(seed, directed, "none")
    M2 = _shuffled_copy(M, seed)
    for func in [nx.is_isomorphic, nx.vf2pp_is_isomorphic, *_could_be_calls()]:
        exact_outcome(func, M, M2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        exact_outcome(nx.weisfeiler_lehman_graph_hash, M)
    if not directed:
        T = nx.MultiGraph(_shuffled_tree(10, seed).edges)
        exact_outcome(nx.isomorphism.tree_isomorphism, T, T)
        exact_outcome(nx.isomorphism.rooted_tree_isomorphism, T, 0, T, 0)


def test_batch11_callables_fall_back():
    G = nx.cycle_graph(5)
    nx.set_node_attributes(G, "a", "color")
    H = _shuffled_copy(G, 2)
    nm = nx.algorithms.isomorphism.categorical_node_match("color", None)
    em = nx.algorithms.isomorphism.categorical_edge_match("w", 1)
    exact_outcome(nx.is_isomorphic, G, H, node_match=nm)
    exact_outcome(nx.is_isomorphic, G, H, edge_match=em)
    with pytest.raises(NotImplementedError):
        nx.is_isomorphic(G, H, node_match=nm, backend="rustnx")


# Bipartite graphs (todo item 37), added in batch 11.

bipartite = nx.algorithms.bipartite


def _bipartite_graph(seed, directed=False, connected=False):
    rng = random.Random(seed)
    n1, n2 = rng.randint(1, 15), rng.randint(1, 15)
    B = bipartite.random_graph(n1, n2, rng.choice([0.1, 0.25, 0.5]), seed=seed)
    if connected:
        top = [v for v in B if v < n1]
        bottom = [v for v in B if v >= n1]
        for a, b in zip(top, bottom):
            B.add_edge(a, b)
        for v in top[len(bottom):]:
            B.add_edge(v, bottom[0])
        for v in bottom[len(top):]:
            B.add_edge(top[0], v)
    if rng.random() < 0.5:
        B = nx.relabel_nodes(B, {v: f"b{v}" for v in B})
    if directed:
        D = nx.DiGraph()
        D.add_nodes_from(B)
        D.add_edges_from((u, v) if rng.random() < 0.5 else (v, u) for u, v in B.edges)
        B = D
    return _shuffled_copy(B, seed, labels=False)


def _top_side(B):
    return [v for v in B if B.nodes[v].get("bipartite") == 0]


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(60))
def test_batch11_bipartite_basic(seed, directed):
    rng = random.Random(seed)
    B = _bipartite_graph(seed, directed)
    C = _bipartite_graph(seed, directed, connected=True)
    G = graph_for(seed, directed)
    for H in [B, C, G, H_empty := B.__class__()]:
        exact_outcome(bipartite.color, H)
        exact_outcome(with_set_order(bipartite.sets), H)
        top = _top_side(H)
        exact_outcome(with_set_order(bipartite.sets), H, top)
        exact_outcome(with_set_order(bipartite.sets), H, top + ["missing"])
        for nodes in [top, [v for v in H if v not in top], top[:1], top + top[:1], list(H)[:3]]:
            exact_outcome(bipartite.is_bipartite_node_set, H, nodes)
    assert H_empty is not None
    exact_outcome(bipartite.is_bipartite_node_set, B, [[1]])  # unhashable
    exact_outcome(bipartite.is_bipartite_node_set, B, iter(list(B)))  # no len
    # A self-loop, an isolated node.
    H = C.copy()
    H.add_node("isolated")
    exact_outcome(bipartite.color, H)
    H.add_edge(rng.choice(list(C)), "isolated")
    exact_outcome(bipartite.color, H)
    exact_outcome(with_set_order(bipartite.sets), H)
    H.add_edge("isolated", "isolated")
    exact_outcome(bipartite.color, H)
    exact_outcome(bipartite.is_bipartite_node_set, H, _top_side(C))


@pytest.mark.parametrize("seed", range(60))
def test_batch11_bipartite_matching(seed):
    rng = random.Random(seed)
    B = _bipartite_graph(seed)
    C = _bipartite_graph(seed, connected=True)
    for H in [B, C]:
        top = _top_side(H)
        exact_outcome(bipartite.hopcroft_karp_matching, H)
        exact_outcome(bipartite.hopcroft_karp_matching, H, top)
        exact_outcome(bipartite.maximum_matching, H, top)
        exact_outcome(bipartite.hopcroft_karp_matching, H, [v for v in H if v not in top])
        exact_outcome(bipartite.hopcroft_karp_matching, H, list(H)[:4])  # maybe not a side
        exact_outcome(bipartite.hopcroft_karp_matching, H, top + ["missing"])
        matching = bipartite.hopcroft_karp_matching(H, top, backend="networkx")
        partial = dict(list(matching.items())[: len(matching) // 2])
        for M in [matching, partial, {}]:
            exact_outcome(with_set_order(bipartite.to_vertex_cover), H, M, top)
            exact_outcome(with_set_order(bipartite.to_vertex_cover), H, M)
        # An invalid "matching": arbitrary pairs, self pairs, unknown nodes.
        nodes = list(H)
        odd = {rng.choice(nodes): rng.choice(nodes) for _ in range(3)}
        odd["unknown"] = nodes[0]
        exact_outcome(with_set_order(bipartite.to_vertex_cover), H, odd, top)
    G = graph_for(seed, False)
    exact_outcome(bipartite.hopcroft_karp_matching, G)
    exact_outcome(bipartite.hopcroft_karp_matching, G, list(G)[::2])
    exact_outcome(bipartite.hopcroft_karp_matching, _bipartite_graph(seed, True, True))
    exact_outcome(with_set_order(bipartite.to_vertex_cover), G, {}, list(G)[::2])


def test_batch11_bipartite_long_augmenting_paths():
    # Matching a long path pairs nodes along it; with this top order the
    # augmenting paths get long.
    for n in [50, 1500]:
        P = nx.path_graph(n)
        top = list(range(0, n, 2))[::-1]
        exact_outcome(bipartite.hopcroft_karp_matching, P, top)
        exact_outcome(bipartite.hopcroft_karp_matching, P)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch11_bipartite_measures(seed, directed):
    rng = random.Random(seed)
    for H in [_bipartite_graph(seed, directed), _bipartite_graph(seed, directed, True), graph_for(seed, directed)]:
        top = _top_side(H) or list(H)[:2]
        for normalized in [True, False]:
            exact_outcome(bipartite.closeness_centrality, H, top, normalized=normalized)
        exact_outcome(bipartite.closeness_centrality, H, list(H)[:3])
        exact_outcome(bipartite.closeness_centrality, H, list(H))
        exact_outcome(bipartite.closeness_centrality, H, top + ["missing"])
        exact_outcome(bipartite.closeness_centrality, H, [])
        rich = [v for v in H if len(H[v]) >= 2]
        for nodes in [None, rich, rich[:3] + rich[:1], set(rich[:4]), list(H)[:5], ["missing"]]:
            exact_outcome(bipartite.node_redundancy, H, nodes)
        R = H.subgraph(rich).copy()
        exact_outcome(bipartite.node_redundancy, R)
        if hasattr(bipartite, "butterflies") and not directed:
            for nodes in [None, rich[:3], list(H)[0] if len(H) else None, ["missing"], 7]:
                exact_outcome(bipartite.butterflies, H, nodes)
    if hasattr(bipartite, "butterflies"):
        for H in [nx.complete_bipartite_graph(4, 5), nx.complete_graph(6), nx.Graph([(0, 0), (0, 1), (1, 2), (2, 3), (3, 0)]), nx.empty_graph(3)]:
            exact_outcome(bipartite.butterflies, H)
            exact_outcome(bipartite.robins_alexander_clustering, H)
    assert rng is not None


def test_batch11_bipartite_runs_in_rust():
    C = _bipartite_graph(5, connected=True)
    top = _top_side(C)
    D = _bipartite_graph(5, True, True)
    calls = [
        lambda: bipartite.color(C, backend="rustnx"),
        lambda: bipartite.color(D, backend="rustnx"),
        lambda: bipartite.sets(C, backend="rustnx"),
        lambda: bipartite.is_bipartite_node_set(C, top, backend="rustnx"),
        lambda: bipartite.hopcroft_karp_matching(C, backend="rustnx"),
        lambda: bipartite.to_vertex_cover(C, {}, backend="rustnx"),
        lambda: bipartite.closeness_centrality(C, top, backend="rustnx"),
        lambda: bipartite.node_redundancy(nx.complete_bipartite_graph(3, 3), backend="rustnx"),
    ]
    if hasattr(bipartite, "butterflies"):
        calls.append(lambda: bipartite.butterflies(C, backend="rustnx"))
    for call in calls:
        call()


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_batch11_bipartite_multigraphs(seed, directed, restore_config):
    base = _bipartite_graph(seed, directed, connected=True)
    M = nx.MultiDiGraph() if directed else nx.MultiGraph()
    M.add_nodes_from(base)
    for u, v in base.edges:
        for _ in range(random.Random(seed).choice([1, 2])):
            M.add_edge(u, v)
    top = _top_side(base)
    exact_outcome(bipartite.color, M)
    exact_outcome(with_set_order(bipartite.sets), M)
    exact_outcome(bipartite.is_bipartite_node_set, M, top)
    exact_outcome(bipartite.closeness_centrality, M, top)
    exact_outcome(bipartite.node_redundancy, M, [v for v in M if len(M[v]) >= 2])
    if not directed:
        exact_outcome(bipartite.hopcroft_karp_matching, M, top)
        exact_outcome(with_set_order(bipartite.to_vertex_cover), M, {}, top)
        if hasattr(bipartite, "butterflies"):
            exact_outcome(bipartite.butterflies, M)


# --- Fixes found while integrating batches 7 to 11 ------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_ancestors_set_order(seed):
    # NetworkX builds the set from a reverse BFS over `G._pred`; string labels
    # make the set's iteration order depend on that insertion order.
    D = graph_for(seed, True)
    D = nx.relabel_nodes(D, {v: f"s{v}" for v in D})
    for source in list(D)[:5]:
        exact_outcome(with_set_order(nx.ancestors), D, source)
        exact_outcome(with_set_order(nx.descendants), D, source)


# --- Batch 12: flows and cut measures ---------------------------------------------


from networkx.algorithms import flow as nx_flow

_B12_FLOW_FUNCS = ["edmonds_karp", "shortest_augmenting_path", "dinitz", "boykov_kolmogorov", "preflow_push"]


def _b12_norm(value):
    """``value`` made comparable: residual networks with both adjacency
    orders and node attributes (``CurrentEdge``s by state), sets in order."""
    from networkx.algorithms.flow.utils import CurrentEdge

    if isinstance(value, CurrentEdge):
        # The iterator's remaining items show its position (this uses it up).
        it = getattr(value, "_it", None)
        return ("curr_edge", type(value._edges).__name__, _b12_norm(getattr(value, "_curr", None)),
                None if it is None else [_b12_norm(item) for item in it])
    if isinstance(value, nx.Graph):
        pred = value._pred if value.is_directed() else {}
        return (
            "graph", type(value).__name__, _b12_norm(value.graph),
            [(n, _b12_norm(d)) for n, d in value._node.items()],
            [(u, [(v, _b12_norm(d)) for v, d in nbrs.items()]) for u, nbrs in value._adj.items()],
            [(u, list(nbrs)) for u, nbrs in pred.items()],
            getattr(value, "__networkx_cache__", "unset"),
        )
    if isinstance(value, dict):
        return ("dict", [(k, _b12_norm(v)) for k, v in value.items()])
    if isinstance(value, (set, frozenset)):
        return ("set", list(value))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [_b12_norm(v) for v in value])
    return (type(value).__name__, value)


def _b12(func):
    return lambda *args, **kwargs: _b12_norm(func(*args, **kwargs))


def _b12_network(seed, directed, kind, size=14):
    """A random flow network: capacities ``kind`` "int", "float", "mixed"
    (ints and floats) or "none", some missing (infinite), some zero;
    int or float weights; int, string or tuple labels."""
    rng = random.Random(seed)
    n = rng.randint(2, size)
    labels = list(range(n))
    if seed % 3 == 1:
        labels = [f"n{i}" for i in range(n)]
    elif seed % 3 == 2:
        labels = [(i % 4, str(i)) for i in range(n)]
    rng.shuffle(labels)
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(labels)
    for _ in range(rng.randint(0, 3 * n)):
        u, v = rng.choice(labels), rng.choice(labels)
        d = {}
        r = rng.random()
        if kind == "int" and r < 0.85:
            d["capacity"] = rng.randint(0, 9)
        elif kind == "float" and r < 0.85:
            d["capacity"] = rng.choice([0.1, 0.3, 0.7, 1.5, 2.25, 1 / 3, 0.0])
        elif kind == "mixed" and r < 0.85:
            d["capacity"] = rng.choice([0, 1, 4, 7, 0.1, 0.3, 2.5])
        d["weight"] = rng.randint(-2, 6) if kind == "int" else rng.choice([1, 2, -1, 0.5, 1.25])
        G.add_edge(u, v, **d)
    return G


def _b12_ends(G, seed):
    rng = random.Random(seed)
    nodes = list(G)
    pairs = [tuple(rng.sample(nodes, 2)) for _ in range(2)]
    return pairs + [(nodes[0], nodes[0]), (nodes[0], "missing"), ("missing", nodes[-1])]


@pytest.mark.parametrize("kind", ["int", "float", "mixed", "none"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(25))
def test_batch12_maximum_flow(seed, directed, kind):
    G = _b12_network(seed, directed, kind)
    exact_outcome(_b12(nx_flow.build_residual_network), G, "capacity")
    for s, t in _b12_ends(G, seed):
        for name in _B12_FLOW_FUNCS:
            func = getattr(nx_flow, name)
            exact_outcome(_b12(func), G, s, t)
            exact_outcome(_b12(nx.maximum_flow), G, s, t, flow_func=func)
            exact_outcome(_b12(nx.minimum_cut), G, s, t, flow_func=func)
            exact_outcome(nx.maximum_flow_value, G, s, t, flow_func=func)
            exact_outcome(nx.minimum_cut_value, G, s, t, flow_func=func)
            try:
                R = func(G, s, t, backend="networkx")
            except nx.NetworkXException:
                R = None
            if R is not None:
                exact_outcome(nx_flow.build_flow_dict, G, R)
            if name != "preflow_push":
                for cutoff in [0, 2, 2.5]:
                    exact_outcome(_b12(func), G, s, t, cutoff=cutoff)
                    exact_outcome(_b12(nx.minimum_cut), G, s, t, flow_func=func, cutoff=cutoff)
        exact_outcome(_b12(nx_flow.shortest_augmenting_path), G, s, t, two_phase=True)
        exact_outcome(_b12(nx.maximum_flow), G, s, t, flow_func=nx_flow.shortest_augmenting_path, two_phase=True)
        for freq in [0, None, 0.5, 3]:
            exact_outcome(_b12(nx_flow.preflow_push), G, s, t, global_relabel_freq=freq)
        exact_outcome(_b12(nx_flow.preflow_push), G, s, t, value_only=True)
        exact_outcome(_b12(nx.maximum_flow), G, s, t)
        exact_outcome(_b12(nx.minimum_cut), G, s, t)
        exact_outcome(nx.maximum_flow_value, G, s, t)
        exact_outcome(nx.minimum_cut_value, G, s, t)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(8))
def test_batch12_maximum_flow_larger(seed, directed):
    # Larger networks reach preflow-push's gap and global relabel heuristics.
    for kind in ["int", "float"]:
        G = _b12_network(100 + seed, directed, kind, size=70)
        for s, t in _b12_ends(G, seed)[:2]:
            for name in _B12_FLOW_FUNCS:
                exact_outcome(_b12(getattr(nx_flow, name)), G, s, t)
            exact_outcome(_b12(nx_flow.preflow_push), G, s, t, value_only=True)
            exact_outcome(_b12(nx.minimum_cut), G, s, t)


def test_batch12_flow_errors(restore_config):
    G = nx.DiGraph()
    G.add_edge("a", "b")  # infinite capacity
    G.add_edge("b", "c", capacity=3)
    G.add_edge("b", "d")
    for name in _B12_FLOW_FUNCS:
        func = getattr(nx_flow, name)
        exact_outcome(_b12(func), G, "a", "c")
        exact_outcome(_b12(func), G, "a", "d")  # unbounded
        exact_outcome(_b12(nx.minimum_cut), G, "a", "d", flow_func=func)
    exact_outcome(nx.maximum_flow, G, "a", "c", cutoff=2)  # kwargs without flow_func
    exact_outcome(nx.maximum_flow, G, "a", "c", flow_func=3)
    exact_outcome(nx.minimum_cut, G, "a", "c", flow_func=nx_flow.preflow_push, cutoff=2)
    exact_outcome(nx.minimum_cut_value, G, "a", "c", flow_func=nx_flow.preflow_push, cutoff=2)
    exact_outcome(nx.maximum_flow, G, "a", "c", flow_func=nx_flow.edmonds_karp, bogus=1)
    exact_outcome(_b12(nx_flow.preflow_push), G, "a", "c", global_relabel_freq=-1)
    R = nx_flow.dinitz(G, "a", "c", backend="networkx")
    exact_outcome(nx_flow.build_flow_dict, G, R)
    exact_outcome(nx_flow.build_flow_dict, G, nx_flow.build_residual_network(G, "capacity"))  # no flows
    exact_outcome(nx_flow.build_flow_dict, nx.DiGraph([("a", "z")]), R)  # z not in R
    R["a"]["b"]["flow"] = 1.5
    exact_outcome(nx_flow.build_flow_dict, G, R)
    # Inputs rustnx hands to NetworkX.
    R = nx_flow.build_residual_network(G, "capacity")
    exact_outcome(_b12(nx_flow.edmonds_karp), G, "a", "c", residual=R)
    exact_outcome(nx.maximum_flow_value, G, "a", "c", flow_func=lambda *a, **k: nx_flow.dinitz(*a, **k))
    H = G.copy()
    H["b"]["c"]["capacity"] = 2**70
    H["b"]["d"]["capacity"] = True
    exact_outcome(_b12(nx_flow.edmonds_karp), H, "a", "c")
    exact_outcome(nx.maximum_flow_value, nx.MultiDiGraph(G), "a", "c")
    # Overflow of the faux infinity falls back too.
    H = nx.DiGraph([(0, 1, {"capacity": 2**62}), (1, 2, {"capacity": 2**62})])
    exact_outcome(_b12(nx_flow.preflow_push), H, 0, 2)
    exact_outcome(_b12(nx.maximum_flow), H, 0, 2, flow_func=nx_flow.dinitz)
    # Flows run in Rust for ordinary inputs (and kwargs reach them).
    nx.config.backend_priority.algos = []
    nx.config.fallback_to_nx = False
    D = _b12_network(3, True, "int")
    s, t = list(D)[:2]
    for name in _B12_FLOW_FUNCS:
        getattr(nx_flow, name)(D, s, t, backend="rustnx")
        nx.minimum_cut(D, s, t, flow_func=getattr(nx_flow, name), backend="rustnx")
    nx.maximum_flow(D, s, t, flow_func=nx_flow.dinitz, cutoff=2, backend="rustnx")
    nx_flow.build_flow_dict(D, nx_flow.dinitz(D, s, t), backend="rustnx")
    U = _b12_network(3, False, "float")
    for _, _, d in U.edges(data=True):
        d.setdefault("capacity", 1)
    nx.gomory_hu_tree(U, backend="rustnx")
    try:
        nx.network_simplex(_b12_demands(D, 3), backend="rustnx")
    except nx.NetworkXUnfeasible:
        pass
    nx.cut_size(U, list(U)[:3], weight="weight", backend="rustnx")


@pytest.mark.parametrize("kind", ["int", "float", "mixed", "none"])
@pytest.mark.parametrize("seed", range(25))
def test_batch12_gomory_hu(seed, kind):
    G = _b12_network(seed, False, kind)
    exact_outcome(_b12(nx.gomory_hu_tree), G)
    for name in _B12_FLOW_FUNCS:
        exact_outcome(_b12(nx.gomory_hu_tree), G, flow_func=getattr(nx_flow, name))
    if seed == 0:
        exact_outcome(nx.gomory_hu_tree, nx.Graph())
        exact_outcome(nx.gomory_hu_tree, nx.DiGraph([(0, 1)]))
        exact_outcome(_b12(nx.gomory_hu_tree), nx.Graph([(0, 1), (1, 2)]))  # unbounded


def _b12_demands(G, seed, total_zero=True):
    rng = random.Random(seed)
    H = G.copy()
    nodes = list(H)
    for v in nodes:
        if rng.random() < 0.7:
            H.nodes[v]["demand"] = rng.choice([-3, -1, 0, 1, 2, 4] if seed % 2 else [-1.5, 0.5, 1, -2])
    if total_zero:
        total = sum(d for _, d in H.nodes(data="demand", default=0))
        H.nodes[nodes[0]]["demand"] = H.nodes[nodes[0]].get("demand", 0) - total
    return H


@pytest.mark.parametrize("kind", ["int", "float", "mixed", "none"])
@pytest.mark.parametrize("seed", range(30))
def test_batch12_min_cost_flow(seed, kind):
    D = _b12_demands(_b12_network(seed, True, kind), seed)
    for func in [nx.network_simplex, nx.min_cost_flow, nx.min_cost_flow_cost]:
        exact_outcome(_b12(func), D)
    try:
        flow = nx.min_cost_flow(D, backend="networkx")
    except nx.NetworkXException:
        flow = None
    if flow is not None:
        exact_outcome(nx.cost_of_flow, D, flow)
    for directed in [False, True]:
        G = _b12_network(seed, directed, kind)
        for s, t in _b12_ends(G, seed)[:3]:
            exact_outcome(_b12(nx.max_flow_min_cost), G, s, t)
        try:
            flow = nx.maximum_flow(G, *_b12_ends(G, seed)[0], backend="networkx")[1]
        except nx.NetworkXException:
            continue
        exact_outcome(nx.cost_of_flow, G, flow)
        exact_outcome(nx.cost_of_flow, G, flow, weight="missing")
    exact_outcome(nx.network_simplex, _b12_network(seed, False, kind))
    exact_outcome(_b12(nx.network_simplex), _b12_demands(_b12_network(seed, True, kind), seed, False))


def test_batch12_min_cost_flow_errors():
    inf = float("inf")
    cases = []
    G = nx.DiGraph([(0, 1, {"capacity": 2, "weight": 1}), (1, 2, {"weight": 2})])
    for attr, value in [("demand", inf), ("demand", -inf)]:
        H = G.copy()
        H.nodes[0][attr] = value
        cases.append(H)
    H = G.copy()
    H[1][2]["weight"] = -inf
    cases.append(H)
    H = G.copy()
    H.add_edge(2, 2, weight=inf)
    cases.append(H)
    H = G.copy()
    H.nodes[0]["demand"] = 1
    cases.append(H)  # demands don't sum to zero
    H = G.copy()
    H[0][1]["capacity"] = -1
    cases.append(H)
    H = G.copy()
    H.add_edge(2, 2, capacity=-1)
    cases.append(H)
    H = G.copy()
    H.nodes[0]["demand"], H.nodes[2]["demand"] = -3, 3  # infeasible
    cases.append(H)
    H = nx.DiGraph([(0, 1, {"weight": -1}), (1, 0, {"weight": -1})])
    cases.append(H)  # negative cycle of infinite capacity
    H = nx.DiGraph([(0, 1, {"weight": 1}), (1, 1, {"weight": -1})])
    cases.append(H)  # negative self-loop of infinite capacity
    H = nx.DiGraph([(0, 1, {"weight": 1}), (1, 1, {"weight": -1, "capacity": 3}), (1, 0, {"capacity": 0})])
    cases.append(H)
    H = nx.DiGraph([(0, 1, {"weight": 1}), (1, 2, {"weight": 3})])
    H.nodes[0]["demand"], H.nodes[2]["demand"] = -2**62, 2**62
    cases.append(H)  # overflow: falls back
    cases += [nx.DiGraph(), nx.DiGraph([(0, 1)])]
    for H in cases:
        for func in [nx.network_simplex, nx.min_cost_flow, nx.min_cost_flow_cost]:
            exact_outcome(_b12(func), H)
    exact_outcome(nx.cost_of_flow, G, {0: {1: 1}, 1: {}})  # KeyError, from NetworkX
    exact_outcome(nx.cost_of_flow, G, {0: {1: 1.5}, 1: {2: 2}})


def _b12_weighted(seed, directed, kind):
    G = graph_for(seed, directed, "float" if kind == "float" else "int" if kind in ("int", "mixed") else "none")
    if kind == "mixed":
        for i, (u, v, d) in enumerate(G.edges(data=True)):
            if i % 3 == 0:
                d["weight"] = d["weight"] + 0.5
            elif i % 5 == 0:
                del d["weight"]
    return G


@pytest.mark.parametrize("kind", ["none", "int", "float", "mixed"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(25))
def test_batch12_cut_measures(seed, directed, kind):
    G = _b12_weighted(seed, directed, kind)
    if seed % 2:
        G = nx.relabel_nodes(G, {v: f"s{v}" for v in G})
    nodes = list(G)
    rng = random.Random(seed)
    halves = [nodes[: len(nodes) // 2], nodes[len(nodes) // 2:]]
    sets = [halves[0], set(halves[0]), rng.sample(nodes, min(3, len(nodes))), nodes[:1] + nodes[:1],
            halves[0] + ["missing"], []]
    weights = [None, "weight"] if kind != "none" else [None, "weight", "missing"]
    for S in sets:
        others = [None, halves[1], nodes, [v for v in nodes if v not in S][:2]]
        for T in others:
            for weight in weights:
                exact_outcome(nx.cut_size, G, S, T, weight=weight)
                exact_outcome(nx.normalized_cut_size, G, S, T, weight=weight)
                exact_outcome(nx.conductance, G, S, T, weight=weight)
                exact_outcome(nx.edge_expansion, G, S, T, weight=weight)
                exact_outcome(nx.mixing_expansion, G, S, T, weight=weight)
        for weight in weights:
            exact_outcome(nx.volume, G, S, weight=weight)
        exact_outcome(nx.node_expansion, G, S)
        exact_outcome(nx.boundary_expansion, G, S)
    exact_outcome(nx.volume, G, nodes[0])  # a single node: NetworkX fails
    # An iterator (a fresh one for each backend) falls back.
    exact_outcome(lambda G, **kw: nx.cut_size(G, iter(halves[0]), halves[1], **kw), G)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_batch12_cut_measures_multigraphs(seed, directed, restore_config):
    M = random_multigraph(seed, directed, "int")
    S = list(M)[: len(M) // 2]
    exact_outcome(nx.node_expansion, M, S)
    exact_outcome(nx.boundary_expansion, M, S)
    exact_outcome(nx.cut_size, M, S, list(M)[len(M) // 2:])  # falls back
    exact_outcome(nx.volume, M, S, weight="weight")


def test_batch12_cut_measures_float_sums():
    # Float sums whose order and compensation show in the last bits.
    G = nx.Graph()
    for i in range(30):
        G.add_edge(i, i + 30, weight=0.1 * (i + 1) + 1e16 * (i % 2))
        G.add_edge(i, (i + 1) % 30, weight=1e-3 * i)
    S = list(range(30))
    for func in [nx.cut_size, nx.volume, nx.normalized_cut_size, nx.conductance]:
        exact_outcome(func, G, S, weight="weight")


def test_batch12_set_replica():
    # rustnx's copy of CPython's set table against real sets: many ops on
    # colliding hashes, and large sets (the resize rule changes at 50000).
    from rustnx import _core, algorithms

    assert algorithms._sets_replayable()
    rng = random.Random(7)
    keys = [rng.randrange(-(2**63), 2**63) for _ in range(70000)] + list(range(200))
    keys = list(dict.fromkeys(keys))
    position = {k: i for i, k in enumerate(keys)}
    hashes = [hash(k) for k in keys]
    big = set()
    ops = []
    for i in range(60000):
        big.add(keys[i])
        ops.append((0, 0, i))
    for i in range(0, 60000, 3):
        big.discard(keys[i])
        ops.append((1, 0, i))
    for i in range(60000, len(keys)):
        big.add(keys[i])
        ops.append((0, 0, i))
    small = set()
    small.update(big)
    ops.append((3, 1, 0))
    big.clear()
    ops.append((2, 0, 0))
    for i in range(500):
        big.add(keys[i])
        ops.append((0, 0, i))
    _, got = _core._replay_sets(hashes, 2, ops)
    assert got == [[position[k] for k in big], [position[k] for k in small]]



def test_batch12_runs_in_rust(restore_config):
    # Each function on ordinary inputs, with no fallback to NetworkX.
    nx.config.fallback_to_nx = False
    D = nx.DiGraph()
    rng = random.Random(1)
    for i in range(40):
        for _ in range(3):
            j = rng.randrange(40)
            if i != j:
                D.add_edge(i, j, capacity=rng.randint(1, 9), weight=rng.choice([1, 2, 0.5]))
    U = D.to_undirected()
    s, t = 0, 1
    value = nx.maximum_flow_value(D, s, t, backend="networkx")
    C = D.copy()
    C.nodes[s]["demand"], C.nodes[t]["demand"] = -value, value
    S, T = list(range(0, 40, 2)), list(range(1, 40, 2))
    calls = [lambda b, f=getattr(nx_flow, name): f(D, s, t, backend=b) for name in _B12_FLOW_FUNCS]
    calls += [
        lambda b: nx_flow.build_residual_network(D, "capacity", backend=b),
        lambda b: nx_flow.build_flow_dict(D, nx_flow.dinitz(D, s, t), backend=b),
        lambda b: nx.maximum_flow(D, s, t, backend=b),
        lambda b: nx.maximum_flow(D, s, t, flow_func=nx_flow.shortest_augmenting_path, two_phase=True, backend=b),
        lambda b: nx.maximum_flow_value(D, s, t, backend=b),
        lambda b: nx.minimum_cut(D, s, t, backend=b),
        lambda b: nx.minimum_cut_value(D, s, t, flow_func=nx_flow.edmonds_karp, cutoff=3, backend=b),
        lambda b: nx.gomory_hu_tree(U, backend=b),
        lambda b: nx.network_simplex(C, backend=b),
        lambda b: nx.min_cost_flow(C, backend=b),
        lambda b: nx.min_cost_flow_cost(C, backend=b),
        lambda b: nx.max_flow_min_cost(D, s, t, backend=b),
        lambda b: nx.max_flow_min_cost(U, s, t, backend=b),
        lambda b: nx.cost_of_flow(C, nx.min_cost_flow(C, backend="networkx"), backend=b),
        lambda b: nx.cut_size(U, S, weight="weight", backend=b),
        lambda b: nx.cut_size(D, S, T, weight="weight", backend=b),
        lambda b: nx.volume(U, S, weight="weight", backend=b),
        lambda b: nx.normalized_cut_size(U, S, weight="weight", backend=b),
        lambda b: nx.conductance(U, S, weight="weight", backend=b),
        lambda b: nx.edge_expansion(U, S, backend=b),
        lambda b: nx.mixing_expansion(U, S, backend=b),
        lambda b: nx.node_expansion(D, S, backend=b),
        lambda b: nx.boundary_expansion(D, S, backend=b),
    ]
    for call in calls:
        assert _b12_norm(call("rustnx")) == _b12_norm(call("networkx"))


# --- Batch 13: connectivity, disjoint paths and augmentation -----------------------

import itertools  # noqa: E402

from networkx.algorithms import connectivity as _b13_conn  # noqa: E402
from networkx.algorithms.connectivity import edge_augmentation as _b13_aug  # noqa: E402


def _b13_connected_graph(seed, directed, weights="none"):
    """graph_for's graph, joined up along a random node order."""
    G = graph_for(seed, directed, weights)
    rng = random.Random(seed)
    order = list(G)
    rng.shuffle(order)
    for a, b in zip(order, order[1:]):
        if rng.random() < 0.3 or not (G.has_edge(a, b) or G.has_edge(b, a)):
            if rng.random() < 0.5:
                a, b = b, a
            if weights == "int":
                G.add_edge(a, b, weight=rng.randint(1, 4))
            elif weights == "float":
                G.add_edge(a, b, weight=rng.choice([0.5, 1.25, 0.1, 0.0]))
            else:
                G.add_edge(a, b)
    return G


def _b13_pairs(G, seed):
    rng = random.Random(seed)
    nodes = list(G)
    pairs = [tuple(rng.sample(nodes, 2)) for _ in range(3)] if len(nodes) > 1 else []
    if nodes:
        pairs += [(nodes[0], nodes[0]), (nodes[0], "missing"), ("missing", nodes[0])]
    return pairs


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch13_local_connectivity_and_paths(seed, directed):
    for G in [graph_for(seed, directed), _b13_connected_graph(seed, directed)]:
        for s, t in _b13_pairs(G, seed):
            for cutoff in [None, 0, 1, 2, 2.5]:
                exact_outcome(_b13_conn.local_node_connectivity, G, s, t, cutoff=cutoff)
                exact_outcome(_b13_conn.local_edge_connectivity, G, s, t, cutoff=cutoff)
                exact_outcome(listed(_b13_conn.node_disjoint_paths), G, s, t, cutoff=cutoff)
                exact_outcome(listed(_b13_conn.edge_disjoint_paths), G, s, t, cutoff=cutoff)
            exact_outcome(nx.node_connectivity, G, s, t)
            exact_outcome(nx.edge_connectivity, G, s, t)
            exact_outcome(nx.edge_connectivity, G, s, t, cutoff=1)
            # Cut sets in iteration order (string labels make it depend on
            # NetworkX's set construction).
            exact_outcome(with_set_order(_b13_conn.minimum_st_edge_cut), G, s, t)
            exact_outcome(with_set_order(_b13_conn.minimum_st_node_cut), G, s, t)
            exact_outcome(with_set_order(nx.minimum_node_cut), G, s, t)
            exact_outcome(with_set_order(nx.minimum_edge_cut), G, s, t)
        exact_outcome(nx.node_connectivity, G, list(G)[0])  # only a source
        exact_outcome(nx.minimum_edge_cut, G, t=list(G)[0])


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch13_global_connectivity(seed, directed):
    for G in [graph_for(seed, directed), _b13_connected_graph(seed, directed)]:
        exact_outcome(nx.node_connectivity, G)
        for cutoff in [None, 1, 2, 2.5, 3.0, 100]:
            exact_outcome(nx.edge_connectivity, G, cutoff=cutoff)
        # Many cuts on one shared residual network (3.4 to 3.6 reorder it).
        exact_outcome(with_set_order(nx.minimum_node_cut), G)
        exact_outcome(with_set_order(nx.minimum_edge_cut), G)
        nodes = list(G)
        small = G.subgraph(nodes[:12]).copy()
        exact_outcome(nx.average_node_connectivity, small)
        exact_outcome(nx.all_pairs_node_connectivity, small)
        for nbunch in [nodes[:5], nodes[:1], nodes[:3] + ["missing"], ["missing"] + nodes[:2], []]:
            exact_outcome(nx.all_pairs_node_connectivity, small, nbunch=nbunch)


@pytest.mark.parametrize(
    "G",
    [
        nx.empty_graph(0),
        nx.empty_graph(1),
        nx.empty_graph(2),
        nx.path_graph(2),
        nx.complete_graph(5),
        nx.cycle_graph(6),
        nx.barbell_graph(4, 1),
        nx.Graph([(0, 0), (0, 1)]),
        nx.Graph([(0, 0), (1, 1), (0, 1)]),
        nx.DiGraph([(0, 0)]),
        nx.DiGraph([(0, 1), (1, 0)]),
        nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2)]),
        nx.complete_graph(4, create_using=nx.DiGraph),
    ],
    ids=lambda G: f"{type(G).__name__}{list(G.edges)}",
)
def test_batch13_small_cases(G):
    for func in [nx.node_connectivity, nx.edge_connectivity, nx.average_node_connectivity,
                 nx.all_pairs_node_connectivity, with_set_order(nx.minimum_node_cut),
                 with_set_order(nx.minimum_edge_cut)]:
        exact_outcome(func, G)
    for s, t in itertools.product(list(G)[:2], repeat=2):
        exact_outcome(listed(_b13_conn.edge_disjoint_paths), G, s, t)
        exact_outcome(listed(_b13_conn.node_disjoint_paths), G, s, t)
        exact_outcome(with_set_order(_b13_conn.minimum_st_node_cut), G, s, t)
        exact_outcome(with_set_order(_b13_conn.minimum_st_edge_cut), G, s, t)
    if not G.is_directed():
        exact_outcome(with_set_order(nx.stoer_wagner), G)
        exact_outcome(listed(_b13_conn.bridge_components), G)
        for k in [1, 2]:
            exact_outcome(listed(_b13_conn.k_edge_augmentation), G, k)
            exact_outcome(listed(_b13_conn.k_edge_augmentation), G, k, partial=True)
            exact_outcome(_b13_conn.is_k_edge_connected, G, k)


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("weights", ["none", "int", "float", "missing"])
def test_batch13_stoer_wagner(seed, weights):
    C = _b13_connected_graph(seed, False, "none" if weights == "missing" else weights)
    if weights == "missing":
        for i, (u, v, d) in enumerate(C.edges(data=True)):
            if i % 2:
                d["weight"] = random.Random(seed + i).randint(1, 5)
    exact_outcome(with_set_order(nx.stoer_wagner), C)
    exact_outcome(with_set_order(nx.stoer_wagner), C, weight=None)
    exact_outcome(with_set_order(nx.stoer_wagner), graph_for(seed, False, weights))  # maybe disconnected
    if weights == "int" and C.number_of_edges():
        D = C.copy()
        u, v = next(iter(D.edges))
        D.add_edge(u, v, weight=-1)  # add_edge clears the conversion cache
        exact_outcome(nx.stoer_wagner, D)
        D.add_edge(u, v, weight=0)
        exact_outcome(with_set_order(nx.stoer_wagner), D)


def test_batch13_stoer_wagner_falls_back():
    G = nx.cycle_graph(5)
    for u, v in list(G.edges):
        G.add_edge(u, v, weight=1 if u % 2 else 1.5)  # mixed ints and floats
    with pytest.raises(NotImplementedError):
        nx.stoer_wagner(G, backend="rustnx")
    G.add_edge(0, 1, weight=float("inf"))
    with pytest.raises(NotImplementedError):
        nx.stoer_wagner(G, backend="rustnx")


@pytest.mark.parametrize("seed", range(40))
def test_batch13_edge_components(seed):
    D = graph_for(seed, True)
    for G in [graph_for(seed, False), _b13_connected_graph(seed, False), D]:
        if not G.is_directed():
            exact_outcome(listed(_b13_conn.bridge_components), G)
        for k in [1, 2, 0]:
            exact_outcome(listed(_b13_conn.k_edge_components), G, k)
            if k < 2 or not G.is_directed():
                # Otherwise NetworkX pops subgraphs from a set of graphs
                # (address order), so even its own runs differ.
                exact_outcome(listed(_b13_conn.k_edge_subgraphs), G, k)
    G = _b13_connected_graph(seed, False)
    for k in [1, 2, 3, 4, 0]:
        exact_outcome(_b13_conn.is_k_edge_connected, G, k)
        for s, t in _b13_pairs(G, seed)[:4]:
            exact_outcome(_b13_conn.is_locally_k_edge_connected, G, s, t, k)
    # Dense enough for k = 3 to be decided by flows.
    H = nx.relabel_nodes(nx.random_regular_graph(4, 12, seed=seed), {i: f"r{i}" for i in range(12)})
    exact_outcome(_b13_conn.is_k_edge_connected, H, 3)
    exact_outcome(_b13_conn.is_k_edge_connected, H, 4)


@pytest.mark.parametrize("seed", range(40))
def test_batch13_augmentation(seed):
    for G in [graph_for(seed, False), _b13_connected_graph(seed, False)]:
        exact_outcome(listed(_b13_aug.unconstrained_one_edge_augmentation), G)
        exact_outcome(listed(_b13_aug.one_edge_augmentation), G)
        exact_outcome(listed(_b13_aug.unconstrained_bridge_augmentation), G)
        exact_outcome(listed(_b13_aug.bridge_augmentation), G)
        for k in [1, 2, 0]:
            for partial in [False, True]:
                exact_outcome(listed(_b13_conn.k_edge_augmentation), G, k, partial=partial)
    D = graph_for(seed, True)
    exact_outcome(listed(_b13_aug.unconstrained_one_edge_augmentation), D)


def _b13_converted(G, name):
    from rustnx import interface

    return interface.convert_from_nx(G, name=name)


def test_batch13_generators_follow_mutation():
    # NetworkX's generators start work on the first next(): changes made
    # before that count.
    makers = {
        "edge_disjoint_paths": lambda G: _b13_conn.edge_disjoint_paths(G, 0, 5),
        "node_disjoint_paths": lambda G: _b13_conn.node_disjoint_paths(G, 0, 5),
        "bridge_components": lambda G: _b13_conn.bridge_components(G),
        "k_edge_augmentation": lambda G: _b13_conn.k_edge_augmentation(G, 2),
    }
    for name, make in makers.items():
        results = []
        for ours in [True, False]:
            G = nx.path_graph(8)
            gen = make(_b13_converted(G, name) if ours else G)
            G.add_edge(0, 4)
            G.add_edge(1, 5)
            G.add_edge(7, 3)
            results.append([sorted(x) if isinstance(x, set) else x for x in gen])
        assert results[0] == results[1], name


def test_batch13_declines():
    G = nx.complete_graph(5)
    G.add_edge(0, 0)
    calls = [
        lambda: nx.edge_connectivity(G, backend="rustnx"),  # self-loops
        lambda: _b13_conn.is_k_edge_connected(G, 3, backend="rustnx"),
        lambda: list(_b13_conn.k_edge_components(G, 3, backend="rustnx")),
        lambda: list(_b13_conn.k_edge_subgraphs(G, 3, backend="rustnx")),
        lambda: list(_b13_conn.k_edge_augmentation(G, 3, backend="rustnx")),
        lambda: list(_b13_aug.one_edge_augmentation(G, avail=[(0, 1)], backend="rustnx")),
        lambda: nx.node_connectivity(G, flow_func=nx.flow.dinitz, backend="rustnx"),
        lambda: _b13_conn.local_edge_connectivity(G, [1], 0, backend="rustnx"),
    ]
    for call in calls:
        with pytest.raises(NotImplementedError):
            call()
    D = nx.MultiGraph(nx.cycle_graph(4))
    with pytest.raises(NotImplementedError):
        nx.node_connectivity(D, backend="rustnx")


def test_batch13_runs_in_rust():
    G = _b13_connected_graph(5, False)
    D = _b13_connected_graph(5, True)
    s, t = list(G)[:2]
    for H in [G, D]:
        nx.node_connectivity(H, backend="rustnx")
        nx.edge_connectivity(H, backend="rustnx")
        nx.minimum_node_cut(H, backend="rustnx")
        nx.minimum_edge_cut(H, backend="rustnx")
        list(_b13_conn.node_disjoint_paths(H, s, t, backend="rustnx"))
    nx.stoer_wagner(G, backend="rustnx")
    list(_b13_conn.k_edge_augmentation(G, 2, backend="rustnx"))


# --- Batch 14: assortativity, link prediction and reciprocity ----------------------


def _b14_with_communities(G, seed, coverage=0.85):
    rng = random.Random(seed)
    for v in G:
        if rng.random() < coverage:
            G.nodes[v]["community"] = rng.choice([0, 1, 2, 1.0, "a"])
    return G


def _b14_ebunch(G, seed, k=40):
    rng = random.Random(seed)
    nodes = list(G)
    if not nodes:
        return []
    pairs = [(rng.choice(nodes), rng.choice(nodes)) for _ in range(k)]
    return pairs + [(nodes[0], nodes[0])]


def _b14_repr(func):
    """Compare float results through ``repr`` (NaN compares unequal)."""
    def run(*args, **kwargs):
        return repr(func(*args, **kwargs))
    return run


def _b14_array(func):
    def run(*args, **kwargs):
        a = func(*args, **kwargs)
        return (a.dtype.str, a.shape, a.tobytes())
    return run


_B14_LINK = [
    nx.jaccard_coefficient,
    nx.adamic_adar_index,
    nx.resource_allocation_index,
    nx.preferential_attachment,
    nx.cn_soundarajan_hopcroft,
    nx.ra_index_soundarajan_hopcroft,
    nx.within_inter_cluster,
    nx.common_neighbor_centrality,
]


def test_batch14_set_order_replay_matches():
    from rustnx import algorithms

    assert algorithms._b14_set_order_matches()


@pytest.mark.parametrize("seed", range(40))
def test_batch14_link_prediction(seed):
    G = _b14_with_communities(graph_for(seed, False), seed)
    ebunch = _b14_ebunch(G, seed)
    for func in _B14_LINK:
        exact_outcome(listed(func), G, ebunch)
        exact_outcome(listed(func), G)
    for alpha in [0.8, 1, 0, 0.5]:
        exact_outcome(listed(nx.common_neighbor_centrality), G, ebunch[:-1], alpha=alpha)
    for delta in [0.001, 2, 0, -1]:
        exact_outcome(listed(nx.within_inter_cluster), G, ebunch[:-1], delta=delta)
    exact_outcome(listed(nx.cn_soundarajan_hopcroft), G, ebunch, community="nope")
    exact_outcome(listed(nx.jaccard_coefficient), graph_for(seed, True), ebunch)


@pytest.mark.parametrize("seed", range(12))
def test_batch14_link_prediction_set_order(seed):
    # Dense graphs with string or mixed labels: float sums over common
    # neighbors follow the order of NetworkX's set, which rustnx replays.
    rng = random.Random(seed)
    n = rng.choice([50, 120])
    base = nx.gnp_random_graph(n, rng.choice([0.1, 0.3, 0.6]), seed=seed)
    labels = {v: f"s{v}" if seed % 3 == 0 or (seed % 3 == 2 and v % 2) else v * 7919 - 300 for v in base}
    G = _b14_with_communities(nx.relabel_nodes(base, labels), seed, 0.97)
    ebunch = _b14_ebunch(G, seed, 600)
    for func in [nx.resource_allocation_index, nx.adamic_adar_index, nx.ra_index_soundarajan_hopcroft,
                 nx.cn_soundarajan_hopcroft, nx.within_inter_cluster]:
        exact_outcome(listed(func), G, ebunch)
    exact_outcome(listed(nx.resource_allocation_index), G)


def test_batch14_link_prediction_edge_cases():
    G = _b14_with_communities(nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 3)]), 1, 1.0)
    G.add_node(4)
    for func in _B14_LINK:
        exact_outcome(listed(func), G, iter([(0, 3), (1, 3)]))  # consumed by the check
        exact_outcome(listed(func), G, [(0, 3, 1)])
        exact_outcome(listed(func), G, [(0, "missing")])
        exact_outcome(listed(func), G, [([0], 1)])
        exact_outcome(listed(func), G, [(0, 3), (4, 4), (2, 2), (3, 3)])
        exact_outcome(listed(func), G, [(1.0, 3), (True, 2)])
        exact_outcome(listed(func), nx.Graph())
        exact_outcome(listed(func), nx.DiGraph([(0, 1)]))
        exact_outcome(listed(func), nx.MultiGraph([(0, 1), (1, 2)]))
    # A neighbor of degree 1 makes Adamic-Adar divide by zero.
    exact_outcome(listed(nx.adamic_adar_index), nx.path_graph(3), [(0, 2), (1, 1), (0, 2)])
    H = nx.complete_graph(6)
    H.nodes[0]["community"] = H.nodes[1]["community"] = 0
    H.nodes[2]["community"] = 0
    for func in [nx.cn_soundarajan_hopcroft, nx.ra_index_soundarajan_hopcroft, nx.within_inter_cluster]:
        exact_outcome(listed(func), H, [(0, 1), (0, 2), (1, 5), (5, 1)])
    H.nodes[3]["community"] = [1]  # unhashable: rustnx hands it to NetworkX
    exact_outcome(listed(nx.cn_soundarajan_hopcroft), H, [(0, 1)])


def test_batch14_link_prediction_runs_in_rust():
    G = _b14_with_communities(nx.relabel_nodes(nx.gnp_random_graph(30, 0.3, seed=1), str), 1, 1.0)
    for func in _B14_LINK:
        assert len(list(func(G, backend="rustnx"))) == 30 * 29 // 2 - G.number_of_edges()


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch14_degree_mixing(seed, directed):
    for weights in ["none", "int", "float"]:
        G = graph_for(seed, directed, weights)
        weight = None if weights == "none" else "weight"
        nodes = list(G)
        some = nodes[: len(nodes) // 2] + ["missing"]
        kinds = [("out", "in"), ("in", "out"), ("out", "out"), ("in", "in")] if directed else [("out", "in")]
        for x, y in kinds:
            for nbunch in [None, some]:
                kw = {"x": x, "y": y, "weight": weight, "nodes": nbunch}
                exact_outcome(listed(nx.node_degree_xy), G, **kw)
                exact_outcome(nx.degree_mixing_dict, G, **kw)
                exact_outcome(nx.degree_mixing_dict, G, normalized=True, **kw)
                exact_outcome(_b14_array(nx.degree_mixing_matrix), G, **kw)
                exact_outcome(_b14_array(nx.degree_mixing_matrix), G, normalized=False, **kw)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    exact_outcome(_b14_repr(nx.degree_assortativity_coefficient), G, **kw)
                    exact_outcome(_b14_repr(nx.degree_pearson_correlation_coefficient), G, **kw)
        exact_outcome(nx.degree_mixing_dict, G, x="bad", y="in")
        if nodes:
            exact_outcome(_b14_repr(nx.degree_assortativity_coefficient), G, nodes=nodes[0])


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch14_attribute_mixing(seed, directed):
    G = graph_for(seed, directed)
    rng = random.Random(seed)
    for v in G:
        if rng.random() < 0.8:
            G.nodes[v]["color"] = rng.choice([0, 1, 2, "a", None, 1.0, True])
        G.nodes[v]["size"] = rng.choice([0, 1, 2.5, 4])
    nodes = list(G)
    some = nodes[: len(nodes) // 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for nbunch in [None, some, some + ["missing"]]:
            exact_outcome(listed(nx.node_attribute_xy), G, "color", nodes=nbunch)
            exact_outcome(nx.attribute_mixing_dict, G, "color", nodes=nbunch)
            exact_outcome(nx.attribute_mixing_dict, G, "color", nodes=nbunch, normalized=True)
            exact_outcome(_b14_array(nx.attribute_mixing_matrix), G, "color", nodes=nbunch)
            exact_outcome(_b14_repr(nx.attribute_assortativity_coefficient), G, "color", nodes=nbunch)
            exact_outcome(_b14_repr(nx.numeric_assortativity_coefficient), G, "size", nodes=nbunch)
            exact_outcome(_b14_repr(nx.numeric_assortativity_coefficient), G, "color", nodes=nbunch)
        mapping = {0: 0, 1: 1, 2: 2, "a": 3, None: 4}
        exact_outcome(_b14_array(nx.attribute_mixing_matrix), G, "color", mapping=mapping)
        exact_outcome(listed(nx.node_attribute_xy), G, "nope")


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch14_neighbor_degree(seed, directed):
    for weights in ["none", "int", "float"]:
        G = graph_for(seed, directed, weights)
        weight = None if weights == "none" else "weight"
        nodes = list(G)
        options = ["in", "out", "in+out", "bad"] if directed else ["out", "in+out", "in"]
        for nbunch in [None, nodes[::3] + nodes[:2] + ["missing"], nodes[0] if nodes else None, 3.5]:
            for source in options:
                for target in options:
                    kw = {"source": source, "target": target, "nodes": nbunch, "weight": weight}
                    exact_outcome(nx.average_neighbor_degree, G, **kw)
                    exact_outcome(nx.average_degree_connectivity, G, **kw)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch14_reciprocity_rich_club_walks(seed, directed):
    G = graph_for(seed, directed)
    nodes = list(G)
    exact_outcome(nx.reciprocity, G)
    exact_outcome(nx.overall_reciprocity, G)
    exact_outcome(nx.reciprocity, G, nodes[::2] + ["missing"])
    if nodes:
        exact_outcome(nx.reciprocity, G, nodes[0])
    exact_outcome(nx.s_metric, G)
    exact_outcome(nx.rich_club_coefficient, G, normalized=False)
    H = G.copy()
    H.remove_edges_from(list(nx.selfloop_edges(H)))
    exact_outcome(nx.rich_club_coefficient, H, normalized=False)
    for k in [0, 1, 2, 3, 6, -1, 2.0]:
        exact_outcome(nx.number_of_walks, G, k)
    for E in [nx.empty_graph(3, create_using=G.__class__), G.__class__()]:
        exact_outcome(nx.overall_reciprocity, E)
        exact_outcome(nx.s_metric, E)
        exact_outcome(nx.number_of_walks, E, 2)
        if not directed:
            exact_outcome(nx.rich_club_coefficient, E, normalized=False)


def test_batch14_walks_wrap_like_int64():
    exact_outcome(nx.number_of_walks, nx.complete_graph(5), 40)


def test_batch14_runs_in_rust():
    G = graph_for(3, False, "int")
    D = graph_for(3, True, "int")
    for v in G:
        G.nodes[v]["c"] = hash(v) % 3
    calls = [
        lambda: list(nx.node_degree_xy(D, weight="weight", backend="rustnx")),
        lambda: nx.degree_mixing_dict(G, backend="rustnx"),
        lambda: nx.degree_mixing_matrix(D, backend="rustnx"),
        lambda: nx.degree_assortativity_coefficient(G, backend="rustnx"),
        lambda: nx.degree_pearson_correlation_coefficient(D, backend="rustnx"),
        lambda: list(nx.node_attribute_xy(G, "c", backend="rustnx")),
        lambda: nx.attribute_mixing_dict(G, "c", backend="rustnx"),
        lambda: nx.attribute_mixing_matrix(G, "c", backend="rustnx"),
        lambda: nx.attribute_assortativity_coefficient(G, "c", backend="rustnx"),
        lambda: nx.numeric_assortativity_coefficient(G, "c", backend="rustnx"),
        lambda: nx.average_neighbor_degree(D, source="in", weight="weight", backend="rustnx"),
        lambda: nx.average_degree_connectivity(D, backend="rustnx"),
        lambda: nx.reciprocity(D, list(D), backend="rustnx"),
        lambda: nx.overall_reciprocity(D, backend="rustnx"),
        lambda: nx.s_metric(G, backend="rustnx"),
        lambda: nx.rich_club_coefficient(nx.path_graph(6), normalized=False, backend="rustnx"),
        lambda: nx.number_of_walks(D, 3, backend="rustnx"),
    ]
    for call in calls:
        call()


@pytest.mark.parametrize("seed", range(6))
def test_batch14_multigraphs_fall_back(seed, restore_config):
    M = random_multigraph(seed, seed % 2 == 0, "none")
    for v in M:
        M.nodes[v]["c"] = 1
    exact_outcome(nx.degree_mixing_dict, M)
    exact_outcome(nx.attribute_mixing_dict, M, "c")
    exact_outcome(nx.average_neighbor_degree, M)
    exact_outcome(nx.average_degree_connectivity, M)
    exact_outcome(nx.s_metric, M)
    exact_outcome(nx.number_of_walks, M, 2)
    exact_outcome(nx.overall_reciprocity, M)
    if not M.is_directed():
        exact_outcome(listed(nx.jaccard_coefficient), M)


def test_batch14_link_prediction_notices_changes():
    G = nx.gnp_random_graph(30, 0.3, seed=2)
    it = nx.jaccard_coefficient(G, list(nx.non_edges(G)), backend="rustnx")
    next(it)
    G.add_edge(0, 29)
    with pytest.raises(RuntimeError):
        list(it)


# --- Batch 15: communities, efficiency and structural holes -----------------------

import math  # noqa: E402

community = nx.community


def _b15_nan_safe(func):
    """``func`` with NaN floats (inf - inf in closeness_vitality) replaced by
    a marker, since NaN never compares equal."""

    def convert(value):
        if isinstance(value, float) and math.isnan(value):
            return "nan"
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return value

    return lambda *args, **kwargs: convert(func(*args, **kwargs))


def _b15_rough_weights(G, seed, kind):
    """``G`` with arbitrary float (or int) weights: sums of these are inexact,
    so any change in summation order shows up in the last bits."""
    rng = random.Random(seed)
    H = G.copy()
    for _, _, d in H.edges(data=True):
        d["weight"] = rng.random() * 3 if kind == "float" else rng.randint(1, 9)
    return H


def _b15_connected(seed, kind="float"):
    """A connected undirected graph with rough weights, half with str labels."""
    rng = random.Random(seed)
    n = rng.randint(4, 40)
    G = nx.connected_watts_strogatz_graph(n, min(4, n - 1), 0.3, seed=seed)
    if seed % 2:
        G = nx.relabel_nodes(G, {v: f"v{(v * 7) % n}" for v in G})
    H = nx.Graph()
    nodes = list(G)
    rng.shuffle(nodes)
    H.add_nodes_from(nodes)
    edges = list(G.edges)
    rng.shuffle(edges)
    H.add_edges_from(edges)
    return _b15_rough_weights(H, seed, kind)


def _b15_partitions(G, seed):
    """Partitions of G's nodes in several container types, and a few
    collections that aren't partitions."""
    rng = random.Random(seed)
    nodes = list(G)
    rng.shuffle(nodes)
    k = rng.randint(1, 5)
    blocks = [nodes[i::k] for i in range(k)]
    good = [
        [set(b) for b in blocks],
        [frozenset(b) for b in blocks],
        [list(b) for b in blocks],
        tuple(set(b) for b in blocks),
        [set(b) for b in blocks] + [set()],
    ]
    bad = [[set(nodes[:-1])] if nodes else [{"x"}], [set(nodes), set(nodes[:1])], [set(nodes) | {"missing"}]]
    return good, bad


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch15_measures(seed, directed, weights):
    G = graph_for(seed, directed, weights)
    wt = None if weights == "none" else "weight"
    nodes = list(G)
    exact_outcome(nx.global_efficiency, G)
    exact_outcome(nx.local_efficiency, G)
    for u, v in [(nodes[0], nodes[-1]), (nodes[0], nodes[0]), (nodes[0], "missing")]:
        exact_outcome(nx.efficiency, G, u, v)
    exact_outcome(nx.gutman_index, G, weight=wt)
    exact_outcome(nx.schultz_index, G, weight=wt)
    if hasattr(nx, "hyper_wiener_index"):
        exact_outcome(nx.hyper_wiener_index, G, weight=wt)
    exact_outcome(_b15_nan_safe(nx.closeness_vitality), G, weight=wt)
    exact_outcome(_b15_nan_safe(nx.closeness_vitality), G, nodes[0], weight=wt)
    exact_outcome(_b15_nan_safe(nx.closeness_vitality), G, "missing", weight=wt)
    exact_outcome(_b15_nan_safe(nx.closeness_vitality), G, nodes[-1], weight=wt, wiener_index=7)
    exact_outcome(nx.flow_hierarchy, G, weight=wt)
    for centers in [nodes[:1], nodes[:3], nodes[:2] + nodes[:1], [], ["missing"]]:
        exact_outcome(with_set_order(nx.voronoi_cells), G, centers)
        exact_outcome(with_set_order(nx.voronoi_cells), G, set(centers), weight=wt)


@pytest.mark.parametrize("kind", ["float", "int"])
@pytest.mark.parametrize("seed", range(30))
def test_batch15_rough_weights(seed, kind):
    # Inexact float sums: these match only if every sum runs in NetworkX's order.
    G = _b15_connected(seed, kind)
    nodes = list(G)
    exact_outcome(nx.gutman_index, G, weight="weight")
    exact_outcome(nx.schultz_index, G, weight="weight")
    exact_outcome(nx.closeness_vitality, G, weight="weight")
    exact_outcome(nx.local_efficiency, G)
    exact_outcome(nx.global_efficiency, G)
    good, _ = _b15_partitions(G, seed)
    for resolution in [1, 0.7, 2]:
        exact_outcome(community.modularity, G, good[0], resolution=resolution)
    exact_outcome(with_set_order(community.greedy_modularity_communities), G, weight="weight")
    exact_outcome(with_set_order(community.edge_betweenness_partition), G, 3, weight="weight")
    exact_outcome(listed(community.asyn_lpa_communities), G, weight="weight", seed=seed)
    exact_outcome(listed(community.fast_label_propagation_communities), G, weight="weight", seed=seed)
    D = _b15_rough_weights(nx.gnp_random_graph(len(nodes), 0.15, seed=seed, directed=True), seed, kind)
    exact_outcome(community.modularity, D, [set(D)], weight="weight")
    exact_outcome(with_set_order(community.greedy_modularity_communities), D, weight="weight")
    exact_outcome(listed(community.fast_label_propagation_communities), D, weight="weight", seed=seed)
    if hasattr(community, "overlapping_modularity"):
        cover = [set(nodes[::2]), set(nodes[1::2]) | set(nodes[:3])]
        exact_outcome(community.overlapping_modularity, G, cover)


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch15_quality(seed, directed, weights):
    G = graph_for(seed, directed, weights)
    wt = None if weights == "none" else "weight"
    good, bad = _b15_partitions(G, seed)
    quality = community.quality
    for part in good + bad:
        exact_outcome(community.is_partition, G, part)
        exact_outcome(community.modularity, G, part, weight=wt)
        exact_outcome(community.modularity, G, part, weight=wt, resolution=0.5)
        exact_outcome(community.partition_quality, G, part)
        exact_outcome(quality.intra_community_edges, G, part)
        exact_outcome(quality.inter_community_edges, G, part)
        exact_outcome(quality.inter_community_non_edges, G, part)
        if hasattr(community, "is_cover"):
            exact_outcome(community.is_cover, G, part)
            if not directed:
                exact_outcome(community.overlapping_modularity, G, part, weight=wt)
    # Generators, dict partitions, and partitions of only some of the nodes.
    blocks = good[0]
    exact_outcome(lambda G, **kw: community.is_partition(G, (set(b) for b in blocks), **kw), G)
    exact_outcome(lambda G, **kw: community.modularity(G, (set(b) for b in blocks), **kw), G)
    exact_outcome(lambda G, **kw: community.partition_quality(G, (set(b) for b in blocks), **kw), G)
    exact_outcome(lambda G, **kw: quality.inter_community_edges(G, (set(b) for b in blocks), **kw), G)
    exact_outcome(quality.inter_community_edges, G, dict(enumerate(good[0])))
    exact_outcome(quality.inter_community_edges, G, good[0][1:])
    exact_outcome(quality.inter_community_non_edges, G, good[0][1:])
    exact_outcome(quality.intra_community_edges, G, [list(G)[0], ["missing"]])


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch15_greedy_modularity(seed, directed, weights):
    G = graph_for(seed, directed, weights)
    wt = None if weights == "none" else "weight"
    n = len(G)
    greedy = with_set_order(community.greedy_modularity_communities)
    exact_outcome(greedy, G, weight=wt)
    for kwargs in [{"resolution": 0.5}, {"resolution": 2}, {"cutoff": 2}, {"best_n": 2},
                   {"cutoff": 2, "best_n": 3}, {"best_n": 1}, {"best_n": n}, {"cutoff": 0},
                   {"cutoff": n + 1}, {"best_n": 1, "cutoff": 2}, {"cutoff": 1.5}]:
        exact_outcome(greedy, G, weight=wt, **kwargs)
    if not directed and n <= 20:
        naive = with_set_order(community.naive_greedy_modularity_communities)
        exact_outcome(naive, G, weight=wt)
        exact_outcome(naive, G, weight=wt, resolution=0.5)
        exact_outcome(naive, G, weight=wt, resolution=2)


@pytest.mark.parametrize("weights", ["none", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch15_divisive(seed, directed, weights):
    G = graph_for(seed, directed, weights)
    wt = None if weights == "none" else "weight"
    if len(G) <= 25:
        exact_outcome(listed(community.girvan_newman), G)
    else:
        first = lambda G, **kw: list(zip(range(3), community.girvan_newman(G, **kw)))  # noqa: E731
        exact_outcome(with_set_order(first), G)
    for k in [0, 1, 2, 3, len(G), len(G) + 1, 2.0]:
        exact_outcome(with_set_order(community.edge_betweenness_partition), G, k, weight=wt)


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch15_label_propagation(seed, directed, weights):
    G = graph_for(seed, directed, weights)
    wt = None if weights == "none" else "weight"
    for func in [community.asyn_lpa_communities, community.fast_label_propagation_communities]:
        exact_outcome(listed(func), G, weight=wt, seed=seed)
        # A fresh generator for each backend.
        exact_outcome(listed(lambda G, **kw: func(G, weight=wt, seed=random.Random(seed), **kw)), G)
        # The global generator: the draws and the state left behind match.
        def with_global(G, func=func, **kw):
            random.seed(seed)
            result = list(func(G, weight=wt, **kw))
            return result, random.random()

        exact_outcome(with_set_order(with_global), G)
    if not directed:
        for k in [1, 2, 3, 0, 2.0, len(G) + 1]:
            exact_outcome(listed(community.asyn_fluidc), G, k, seed=seed)
        exact_outcome(listed(community.asyn_fluidc), G, 2, max_iter=1, seed=seed)
        exact_outcome(listed(community.asyn_fluidc), G, 2, max_iter=0, seed=seed)


@pytest.mark.parametrize(
    "G",
    [
        nx.empty_graph(0),
        nx.empty_graph(1),
        nx.empty_graph(3),
        nx.path_graph(2),
        nx.path_graph(4),
        nx.complete_graph(4),
        nx.barbell_graph(4, 2),
        nx.Graph([(0, 0)]),
        nx.Graph([(0, 0), (0, 1)]),
        nx.DiGraph([(0, 1), (1, 0), (1, 2)]),
        nx.DiGraph([(0, 0)]),
    ],
    ids=lambda G: f"{type(G).__name__}{list(G.edges)}",
)
def test_batch15_small_cases(G):
    nodes = list(G)
    part = [set(nodes[::2]), set(nodes[1::2])]
    funcs = [
        nx.global_efficiency, nx.local_efficiency, nx.gutman_index, nx.schultz_index,
        _b15_nan_safe(nx.closeness_vitality), nx.flow_hierarchy,
        lambda G, **kw: community.modularity(G, part, **kw),
        lambda G, **kw: community.partition_quality(G, part, **kw),
        lambda G, **kw: community.quality.inter_community_non_edges(G, part, **kw),
        with_set_order(community.greedy_modularity_communities),
        with_set_order(community.naive_greedy_modularity_communities),
        listed(community.girvan_newman),
        listed(lambda G, **kw: community.asyn_lpa_communities(G, seed=1, **kw)),
        listed(lambda G, **kw: community.fast_label_propagation_communities(G, seed=1, **kw)),
        listed(lambda G, **kw: community.asyn_fluidc(G, 1, seed=1, **kw)),
        with_set_order(lambda G, **kw: community.edge_betweenness_partition(G, 2, **kw)),
    ]
    if hasattr(nx, "hyper_wiener_index"):
        funcs.append(nx.hyper_wiener_index)
    for func in funcs:
        exact_outcome(func, G)
    if nodes:
        exact_outcome(_b15_nan_safe(nx.closeness_vitality), G, nodes[0])
        exact_outcome(nx.efficiency, G, nodes[0], nodes[-1])
        exact_outcome(with_set_order(nx.voronoi_cells), G, nodes[:1])


def test_batch15_known_cases():
    G = nx.karate_club_graph()
    exact_outcome(with_set_order(community.greedy_modularity_communities), G)
    exact_outcome(with_set_order(community.greedy_modularity_communities), G, weight="weight")
    exact_outcome(with_set_order(community.naive_greedy_modularity_communities), G)
    exact_outcome(with_set_order(lambda G, **kw: list(zip(range(4), community.girvan_newman(G, **kw)))), G)
    exact_outcome(community.modularity, G, community.label_propagation_communities(G))
    exact_outcome(nx.local_efficiency, G)
    exact_outcome(nx.closeness_vitality, G, weight="weight")
    # Ties everywhere: equal betweenness, equal modularity gains.
    for H in [nx.cycle_graph(8), nx.complete_graph(6), nx.grid_2d_graph(3, 4), nx.star_graph(5)]:
        exact_outcome(listed(community.girvan_newman), H)
        exact_outcome(with_set_order(community.greedy_modularity_communities), H)
        exact_outcome(with_set_order(community.naive_greedy_modularity_communities), H)
        exact_outcome(with_set_order(community.edge_betweenness_partition), H, 3)
    # A directed graph whose undirected copy reorders neighbors.
    D = nx.DiGraph([(3, 0), (2, 0), (1, 0), (0, 4), (4, 1), (2, 3)])
    exact_outcome(listed(community.girvan_newman), D)
    exact_outcome(nx.flow_hierarchy, D)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_batch15_multigraphs(seed, directed, restore_config):
    M = random_multigraph(seed, directed, "int")
    nodes = list(M)
    if not directed:
        exact_outcome(nx.global_efficiency, M)
        exact_outcome(nx.local_efficiency, M)
        exact_outcome(nx.efficiency, M, nodes[0], nodes[-1])
    exact_outcome(_b15_nan_safe(nx.closeness_vitality), M, weight="weight")
    exact_outcome(with_set_order(nx.voronoi_cells), M, nodes[:2])
    # These fall back.
    exact_outcome(community.modularity, M, [set(M)])
    exact_outcome(with_set_order(community.greedy_modularity_communities), M)
    exact_outcome(nx.flow_hierarchy, M)
    exact_outcome(listed(lambda G, **kw: community.asyn_lpa_communities(G, seed=1, **kw)), M)


def test_batch15_unsupported_fall_back(restore_config):
    G = graph_for(3, False, "int")
    nodes = list(G)
    G.add_edge(nodes[0], nodes[1], weight=0.5)  # mixed int and float weights
    exact_outcome(community.modularity, G, [set(G)])
    exact_outcome(with_set_order(community.greedy_modularity_communities), G, weight="weight")
    exact_outcome(nx.gutman_index, G, weight="weight")
    exact_outcome(with_set_order(community.naive_greedy_modularity_communities), _b15_connected(1), weight="weight")
    exact_outcome(listed(community.girvan_newman), G, most_valuable_edge=lambda G: next(iter(G.edges)))
    T = nx.relabel_nodes(nx.path_graph(5), {i: (i, i) for i in range(5)})  # 2-tuple labels
    exact_outcome(listed(community.girvan_newman), T)
    exact_outcome(with_set_order(community.edge_betweenness_partition), T, 2)
    # Node labels that can't be sorted break greedy modularity's ties.
    Mixed = nx.relabel_nodes(nx.cycle_graph(6), {0: "a", 1: (1,)})
    exact_outcome(with_set_order(community.greedy_modularity_communities), Mixed)
    W = nx.path_graph(4)
    W.add_node("weight")
    exact_outcome(nx.gutman_index, W)
    import numpy as np

    exact_outcome(
        listed(lambda G, **kw: community.asyn_lpa_communities(G, seed=np.random.RandomState(1), **kw)), G
    )


def test_batch15_graph_changes_before_iteration():
    # girvan_newman and the label propagation generators read the graph
    # when iteration starts, as NetworkX does.
    for make in [lambda G, b: community.girvan_newman(G, backend=b),
                 lambda G, b: community.asyn_lpa_communities(G, seed=1, backend=b),
                 lambda G, b: community.fast_label_propagation_communities(G, seed=1, backend=b)]:
        results = []
        for backend in ["rustnx", "networkx"]:
            G = nx.barbell_graph(4, 1)
            it = make(G, backend)
            G.remove_edge(0, 1)
            first = next(it)
            G.add_edge("x", "y")
            results.append([first, *it])
        assert results[0] == results[1]


def test_batch15_runs_in_rust():
    G = nx.connected_watts_strogatz_graph(30, 4, 0.2, seed=1)
    D = nx.gnp_random_graph(30, 0.1, seed=1, directed=True)
    part = [set(range(15)), set(range(15, 30))]
    calls = [
        lambda: nx.efficiency(G, 0, 9, backend="rustnx"),
        lambda: nx.global_efficiency(G, backend="rustnx"),
        lambda: nx.local_efficiency(G, backend="rustnx"),
        lambda: nx.closeness_vitality(G, backend="rustnx"),
        lambda: nx.gutman_index(G, backend="rustnx"),
        lambda: nx.schultz_index(G, backend="rustnx"),
        lambda: nx.flow_hierarchy(D, backend="rustnx"),
        lambda: nx.voronoi_cells(G, [0, 5], backend="rustnx"),
        lambda: community.modularity(G, part, backend="rustnx"),
        lambda: community.modularity(D, part, backend="rustnx"),
        lambda: community.partition_quality(G, part, backend="rustnx"),
        lambda: community.is_partition(G, part, backend="rustnx"),
        lambda: community.quality.intra_community_edges(G, part, backend="rustnx"),
        lambda: community.quality.inter_community_edges(G, part, backend="rustnx"),
        lambda: community.quality.inter_community_non_edges(G, part, backend="rustnx"),
        lambda: community.greedy_modularity_communities(G, backend="rustnx"),
        lambda: community.naive_greedy_modularity_communities(nx.path_graph(8), backend="rustnx"),
        lambda: list(community.girvan_newman(G, backend="rustnx")),
        lambda: community.edge_betweenness_partition(G, 3, backend="rustnx"),
        lambda: list(community.asyn_lpa_communities(G, seed=1, backend="rustnx")),
        lambda: list(community.fast_label_propagation_communities(D, seed=1, backend="rustnx")),
        lambda: list(community.asyn_fluidc(G, 3, seed=1, backend="rustnx")),
    ]
    if hasattr(nx, "hyper_wiener_index"):
        calls.append(lambda: nx.hyper_wiener_index(G, backend="rustnx"))
    if hasattr(community, "is_cover"):
        calls.append(lambda: community.is_cover(G, part, backend="rustnx"))
        calls.append(lambda: community.overlapping_modularity(G, part, backend="rustnx"))
    for call in calls:
        call()


# --- Batch 16: approximation algorithms and graph operations -----------------------

from networkx.algorithms import approximation as approx  # noqa: E402
from networkx.algorithms.approximation.treewidth import treewidth_decomp  # noqa: E402


def _b16_graph_state(G):
    """Everything about a result graph a caller could see: its class, node
    and adjacency order, attribute dicts, and which edge dicts are shared."""
    if not isinstance(G, nx.Graph):
        return G
    state = [type(G).__name__, G.graph, list(G.nodes(data=True))]
    state.append([(u, list(nbrs.items())) for u, nbrs in G._adj.items()])
    if G.is_directed():
        state.append([(u, list(nbrs.items())) for u, nbrs in G._pred.items()])
        state.append([G._succ[u][v] is G._pred[v][u] for u in G for v in G._succ[u]])
    else:
        state.append([G._adj[u][v] is G._adj[v][u] for u in G for v in G._adj[u]])
    return state


def _b16_graphs(func):
    """Compare graph results (also inside tuples) with `_b16_graph_state`."""

    def run(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, tuple):
            return tuple(_b16_graph_state(r) for r in result)
        return _b16_graph_state(result)

    return run


def _b16_node_weights(G, seed, kind):
    rng = random.Random(seed)
    for v in G:
        if kind == "int":
            G.nodes[v]["w"] = rng.randint(-2, 6)
        elif kind == "float":
            G.nodes[v]["w"] = rng.choice([0.5, 1.0, 2.25, 3.0, 0.1])
        elif kind == "mixed" and rng.random() < 0.7:
            G.nodes[v]["w"] = rng.choice([1, 2, 0.5, 3.75, True])
    return G


@pytest.mark.parametrize("kind", ["none", "int", "float", "mixed"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch16_covers_and_dominating_sets(seed, directed, kind):
    G = _b16_node_weights(graph_for(seed, directed), seed, kind)
    for weight in [None, "w", "missing"]:
        exact_outcome(with_set_order(approx.min_weighted_vertex_cover), G, weight=weight)
        exact_outcome(with_set_order(approx.min_weighted_dominating_set), G, weight=weight)
    exact_outcome(with_set_order(approx.min_edge_dominating_set), G)
    exact_outcome(with_set_order(approx.min_maximal_matching), G)


def test_batch16_covers_edge_cases():
    graphs = [nx.Graph(), nx.DiGraph(), nx.empty_graph(3), nx.Graph([(0, 0)]),
              nx.Graph([(0, 0), (0, 1)]), nx.star_graph(5), nx.complete_graph(5)]
    for G in graphs:
        for func in [approx.min_weighted_vertex_cover, approx.min_weighted_dominating_set,
                     approx.min_edge_dominating_set, approx.min_maximal_matching]:
            exact_outcome(with_set_order(func), G)
    G = nx.path_graph(6)
    for value in [2**60, 2**70, -(2**62), float("inf"), float("nan"), "x", None, 1.5]:
        G.nodes[2]["w"] = value
        exact_outcome(with_set_order(approx.min_weighted_vertex_cover), G, weight="w")
        exact_outcome(with_set_order(approx.min_weighted_dominating_set), G, weight="w")
    # String labels: set iteration order follows insertion order.
    S = nx.relabel_nodes(nx.gnp_random_graph(30, 0.15, seed=3), lambda v: f"s{v}")
    exact_outcome(with_set_order(approx.min_weighted_vertex_cover), S)
    exact_outcome(with_set_order(approx.min_weighted_dominating_set), S)


def _b16_complete(seed, directed, kind):
    rng = random.Random(seed)
    n = rng.randint(1, 12)
    G = nx.complete_graph(n, nx.DiGraph() if directed else nx.Graph())
    if rng.random() < 0.5:
        G = nx.relabel_nodes(G, {v: f"t{v}" for v in G})
    for u, v, d in G.edges(data=True):
        if kind == "float":
            d["weight"] = rng.random() * 10
        elif kind == "int":
            d["weight"] = rng.randint(1, 4)
        elif kind == "wide":
            d["weight"] = rng.randint(1, 10**6)
        elif kind == "mixed":
            d["weight"] = rng.choice([1, 2.5])
    if rng.random() < 0.2 and n:
        v = rng.choice(list(G))
        G.add_edge(v, v, weight=0.5)
    return G


@pytest.mark.parametrize("kind", ["float", "int", "wide", "mixed", "none"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch16_tsp(seed, directed, kind):
    G = _b16_complete(seed, directed, kind)
    nodes = list(G)
    rng = random.Random(seed)
    sources = [None] + nodes[:2] + ["missing"]
    for source in sources:
        exact_outcome(approx.greedy_tsp, G, source=source)
    exact_outcome(approx.greedy_tsp, G, weight=None)
    cycle = nodes + nodes[:1]
    rng.shuffle(cycle[1:-1]) if len(cycle) > 3 else None
    for func in [approx.simulated_annealing_tsp, approx.threshold_accepting_tsp]:
        for move in ["1-1", "1-0"]:
            exact_outcome(func, G, cycle, move=move, seed=seed, max_iterations=5, N_inner=40)
            exact_outcome(func, G, "greedy", move=move, seed=seed, max_iterations=3, N_inner=30)
        exact_outcome(func, G, cycle, source=nodes[-1] if nodes else None, seed=1)
        exact_outcome(func, G, cycle[:-1], seed=1)
        exact_outcome(func, G, cycle[:-1] + ["missing", cycle[0]], seed=1)
        exact_outcome(func, G, cycle, seed=2, N_inner=10, max_iterations=2)
    exact_outcome(approx.simulated_annealing_tsp, G, cycle, temp=0.5, alpha=0.3, seed=3)
    exact_outcome(approx.threshold_accepting_tsp, G, cycle, threshold=-1, seed=3)
    exact_outcome(approx.threshold_accepting_tsp, G, cycle, threshold=0.75, alpha=0.5, seed=4)
    if len(G) > 3:
        H = G.copy()
        H.remove_edge(nodes[0], nodes[1])
        exact_outcome(approx.greedy_tsp, H)
        exact_outcome(approx.simulated_annealing_tsp, H, cycle, seed=1)


@pytest.mark.parametrize("seed", range(60))
def test_batch16_treewidth(seed):
    G = graph_for(seed, False)

    def bags(func):
        def run(G, **kw):
            width, decomp = func(G, **kw)
            return (width, [list(b) for b in decomp], [(list(u), list(v)) for u, v in decomp.edges])

        return run

    exact_outcome(bags(approx.treewidth_min_fill_in), G)
    exact_outcome(bags(treewidth_decomp), G)
    S = nx.relabel_nodes(G, {v: ("t", str(v)) for v in G})
    exact_outcome(bags(approx.treewidth_min_fill_in), S)
    exact_outcome(approx.treewidth_min_fill_in, graph_for(seed, True))
    exact_outcome(lambda G, **kw: treewidth_decomp(G, **kw)[0], graph_for(seed, True))
    # A custom heuristic falls back.
    exact_outcome(bags(lambda G, **kw: treewidth_decomp(G, lambda g: None, **kw)), G)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch16_approximate_diameter(seed, directed):
    G = graph_for(seed, directed)
    for s in range(3):
        exact_outcome(approx.diameter, G, seed=s)
    C = nx.cycle_graph(7, nx.DiGraph() if directed else nx.Graph())
    nx.add_path(C, [0, 10, 11, 12, 3])
    for s in range(5):
        exact_outcome(approx.diameter, C, seed=s)
    exact_outcome(approx.diameter, nx.Graph())
    exact_outcome(approx.diameter, nx.empty_graph(1))
    exact_outcome(approx.diameter, nx.empty_graph(2))


@pytest.mark.parametrize("weights", ["none", "int", "float"])
@pytest.mark.parametrize("seed", range(40))
def test_batch16_maxcut(seed, weights):
    G = graph_for(seed, False, weights)
    G = nx.relabel_nodes(G, {v: f"m{v}" for v in G}) if seed % 2 else G
    nodes = list(G)
    rng = random.Random(seed)
    weight = None if weights == "none" else "weight"
    for s in range(3):
        exact_outcome(with_set_order(approx.one_exchange), G, seed=s, weight=weight)
        start = rng.sample(nodes, len(nodes) // 2) + ["not a node"]
        exact_outcome(with_set_order(approx.one_exchange), G, start, seed=s, weight=weight)
        for p in [0.5, 0.1, 0.9]:
            exact_outcome(with_set_order(approx.randomized_partitioning), G, s, p, weight)
    exact_outcome(with_set_order(approx.one_exchange), graph_for(seed, True), seed=1)
    exact_outcome(with_set_order(approx.randomized_partitioning), graph_for(seed, True), seed=1)


def _b16_kl_graph(seed, directed):
    rng = random.Random(seed)
    G = nx.gnp_random_graph(rng.randint(2, 14), rng.choice([0.2, 0.4, 0.7]), seed=seed,
                            directed=directed)
    if rng.random() < 0.5:
        G = nx.relabel_nodes(G, {v: f"k{v}" for v in G})
    if rng.random() < 0.3:
        v = rng.choice(list(G))
        G.add_edge(v, v)
    for u, v, d in G.edges(data=True):
        d["w"] = rng.randint(1, 3)
    G.graph["name"] = "kl"
    return G


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch16_kl_connected(seed, directed):
    G = _b16_kl_graph(seed, directed)
    pairs = [(1, 1), (2, 2), (3, 3), (2, 2.5), (1, 0), (2, True)]
    if nx.number_of_selfloops(G) == 0:
        pairs.append((2, float("inf")))  # NetworkX never finishes with self-loops
    for k, l in pairs:
        exact_outcome(nx.is_kl_connected, G, k, l)
        exact_outcome(_b16_graphs(nx.kl_connected_subgraph), G, k, l)
        exact_outcome(_b16_graphs(nx.kl_connected_subgraph), G, k, l, same_as_graph=True)
    exact_outcome(nx.is_kl_connected, G, 2, 2, low_memory=True)
    exact_outcome(_b16_graphs(nx.kl_connected_subgraph), G, 2, 2, low_memory=True)


def _b16_pair(seed, directed):
    G = graph_for(seed, directed)
    H = G.__class__()
    nodes = list(G)
    random.Random(seed).shuffle(nodes)
    H.add_nodes_from(nodes)
    rng = random.Random(seed + 1)
    H.add_edges_from(e for e in G.edges if rng.random() < 0.6)
    for _ in range(len(nodes)):
        a, b = rng.choice(nodes), rng.choice(nodes)
        H.add_edge(a, b)
    return G, H


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_batch16_operators(seed, directed):
    G, H = _b16_pair(seed, directed)
    G.graph["name"] = "G"
    for v in list(G)[:3]:
        G.nodes[v]["color"] = "red"
    exact_outcome(_b16_graphs(nx.complement), G)
    for k in [1, 2, 3, 2.5, 0, -1, 100, float("inf"), True]:
        exact_outcome(_b16_graphs(nx.power), G, k)
    for func in [nx.difference, nx.symmetric_difference]:
        exact_outcome(_b16_graphs(func), G, H)
        exact_outcome(_b16_graphs(func), H, G)
        exact_outcome(_b16_graphs(func), G, G)
        # Mixed directedness.
        exact_outcome(_b16_graphs(func), G, H.to_undirected() if directed else H.to_directed())
        # Different node sets.
        K = H.copy()
        K.add_node("extra")
        exact_outcome(_b16_graphs(func), G, K)
    S = nx.relabel_nodes(G, {v: f"c{v}" for v in G})
    exact_outcome(_b16_graphs(nx.complement), S)


def test_batch16_operators_on_subclasses():
    class MyGraph(nx.Graph):
        pass

    G = MyGraph(nx.path_graph(5))
    exact_outcome(_b16_graphs(nx.complement), G)
    exact_outcome(_b16_graphs(nx.difference), G, nx.path_graph(5))
    exact_outcome(_b16_graphs(nx.power), G, 2)



@pytest.mark.parametrize("weights", ["none", "int", "float", "missing"])
@pytest.mark.parametrize("seed", range(40))
def test_batch16_steiner_tree(seed, weights):
    G = graph_for(seed, False, weights)
    if seed % 3 == 0:
        # One component, so every node reaches a terminal.
        G.add_edges_from(zip(list(G), list(G)[1:]), weight=2)
    nodes = list(G)
    rng = random.Random(seed)
    for k in [1, 2, 3, 6]:
        terminals = rng.sample(nodes, min(k, len(nodes)))
        for weight in ["weight", None, "other"]:
            exact_outcome(_b16_graphs(approx.steiner_tree), G, terminals, weight=weight)
        exact_outcome(_b16_graphs(approx.steiner_tree), G, terminals, method="mehlhorn")
        exact_outcome(_b16_graphs(approx.steiner_tree), G, terminals + terminals[:1])
    exact_outcome(_b16_graphs(approx.steiner_tree), G, nodes[:2], method="kou")
    exact_outcome(_b16_graphs(approx.steiner_tree), G, nodes[:2], method="nope")
    exact_outcome(_b16_graphs(approx.steiner_tree), G, [])
    exact_outcome(_b16_graphs(approx.steiner_tree), G, ["missing"])
    exact_outcome(_b16_graphs(approx.steiner_tree), graph_for(seed, True), nodes[:2])
    if G.number_of_edges():
        u, v = next(iter(G.edges))
        H = G.copy()
        H[u][v]["weight"] = -1
        exact_outcome(_b16_graphs(approx.steiner_tree), H, nodes[:3])


_b16_densest = getattr(approx, "densest_subgraph", None)


@pytest.mark.skipif(_b16_densest is None, reason="NetworkX lacks densest_subgraph")
@pytest.mark.parametrize("seed", range(60))
def test_batch16_densest_subgraph(seed):
    G = graph_for(seed, False)
    rng = random.Random(seed)
    if seed % 2:
        # Plain 0..n-1 labels in a shuffled order (NetworkX 3.5 and 3.6 index
        # by node there).
        nodes = list(range(len(G)))
        rng.shuffle(nodes)
        H = nx.Graph()
        H.add_nodes_from(nodes)
        H.add_edges_from(nx.convert_node_labels_to_integers(G).edges)
        G = H
    for method in ["fista", "greedy++"]:
        for iterations in [1, 2, 5, 30]:
            exact_outcome(with_set_order(_b16_densest), G, iterations, method=method)
        exact_outcome(with_set_order(_b16_densest), G, 0, method=method)
    exact_outcome(with_set_order(_b16_densest), G, 3)
    exact_outcome(with_set_order(_b16_densest), G, method="nope")
    exact_outcome(with_set_order(_b16_densest), nx.Graph(), 3)
    exact_outcome(with_set_order(_b16_densest), nx.empty_graph(4), 3)
    exact_outcome(with_set_order(_b16_densest), graph_for(seed, True))

@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_batch16_multigraphs(seed, directed, restore_config):
    M = random_multigraph(seed, directed, "none")
    for v in M:
        M.nodes[v]["w"] = random.Random(str(v)).randint(1, 5)
    exact_outcome(with_set_order(approx.min_weighted_vertex_cover), M, weight="w")
    exact_outcome(approx.diameter, M, seed=seed)
    if not directed:
        exact_outcome(with_set_order(approx.min_weighted_dominating_set), M, weight="w")
    # These fall back.
    exact_outcome(with_set_order(approx.min_maximal_matching), M)
    exact_outcome(_b16_graphs(nx.complement), M)
    exact_outcome(nx.is_kl_connected, M, 2, 2)
    if not directed:
        exact_outcome(_b16_graphs(approx.steiner_tree), M, list(M)[:2])


def test_batch16_runs_in_rust():
    G = nx.gnp_random_graph(30, 0.2, seed=1)
    D = nx.gnp_random_graph(30, 0.3, seed=1, directed=True)
    K = nx.complete_graph(8)
    for u, v, d in K.edges(data=True):
        d["weight"] = random.Random(u * 10 + v).random()
    cycle = list(K) + [0]
    calls = [
        lambda: approx.min_weighted_vertex_cover(G, backend="rustnx"),
        lambda: approx.min_weighted_dominating_set(G, backend="rustnx"),
        lambda: approx.min_edge_dominating_set(G, backend="rustnx"),
        lambda: approx.min_maximal_matching(G, backend="rustnx"),
        lambda: approx.greedy_tsp(K, backend="rustnx"),
        lambda: approx.simulated_annealing_tsp(K, cycle, seed=1, backend="rustnx"),
        lambda: approx.threshold_accepting_tsp(K, "greedy", seed=1, backend="rustnx"),
        lambda: approx.treewidth_min_fill_in(G, backend="rustnx"),
        lambda: treewidth_decomp(G, backend="rustnx"),
        lambda: approx.diameter(D, seed=1, backend="rustnx"),
        lambda: approx.one_exchange(G, seed=1, backend="rustnx"),
        lambda: approx.randomized_partitioning(G, seed=1, backend="rustnx"),
        lambda: nx.is_kl_connected(G, 2, 2, backend="rustnx"),
        lambda: nx.kl_connected_subgraph(D, 2, 2, backend="rustnx"),
        lambda: nx.complement(G, backend="rustnx"),
        lambda: nx.power(G, 2, backend="rustnx"),
        lambda: nx.difference(G, G, backend="rustnx"),
        lambda: nx.symmetric_difference(D, D, backend="rustnx"),
        lambda: approx.steiner_tree(G, [0, 5, 9], backend="rustnx"),
    ]
    if _b16_densest is not None:
        calls.append(lambda: _b16_densest(G, 5, backend="rustnx"))
        calls.append(lambda: _b16_densest(G, 5, method="greedy++", backend="rustnx"))
    for call in calls:
        call()


# --- Fixes found while integrating batches 12 to 16 ------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_float_distance_totals_follow_python_sum(seed):
    # Python 3.12+ sums floats with compensated summation; random float
    # weights make the last bits depend on it (0.5, 1.25, ... sum exactly).
    rng = random.Random(seed)
    G = nx.connected_watts_strogatz_graph(30, 4, 0.3, seed=seed)
    for u, v in G.edges:
        G[u][v]["weight"] = rng.random()
    exact_outcome(nx.wiener_index, G, weight="weight")
    exact_outcome(nx.average_shortest_path_length, G, weight="weight")
    exact_outcome(nx.closeness_centrality, G, distance="weight")
    D = nx.DiGraph(G)
    for u, v in D.edges:
        D[u][v]["weight"] = rng.random()
    exact_outcome(nx.wiener_index, D, weight="weight")
    exact_outcome(nx.closeness_centrality, D, distance="weight")


# --- Batch 20: readers and parsers ----------------------------------------------------


import bz2 as _b20_bz2
import gzip as _b20_gzip
import io as _b20_io
import struct as _b20_struct


def _b20_value(value):
    """``value`` with types made explicit and floats compared bit for bit."""
    if isinstance(value, dict):
        return ("dict", [(_b20_value(k), _b20_value(v)) for k, v in value.items()])
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [_b20_value(v) for v in value])
    if type(value) is float:
        return ("float", _b20_struct.pack("<d", value))
    return (type(value).__name__, value)


def _b20_state(G):
    """Everything observable about a NetworkX graph: class, graph dict, node
    order and attributes, every adjacency row in order (with keys), and which
    rows share each edge's data dict."""
    if isinstance(G, list):
        return ("list", [_b20_state(H) for H in G])
    if not isinstance(G, nx.Graph):
        return _b20_value(G)
    multi = G.is_multigraph()

    def row(nbrs):
        if multi:
            return [(_b20_value(v), [(_b20_value(k), _b20_value(d)) for k, d in kd.items()])
                    for v, kd in nbrs.items()]
        return [(_b20_value(v), _b20_value(d)) for v, d in nbrs.items()]

    state = [type(G), _b20_value(G.graph), [(_b20_value(n), _b20_value(d)) for n, d in G._node.items()]]
    state.append([(_b20_value(u), row(nbrs)) for u, nbrs in G._adj.items()])
    assert list(G._adj) == list(G._node)
    back = G._pred if G.is_directed() else G._adj
    if G.is_directed():
        assert G._succ is G._adj and list(G._pred) == list(G._node)
        state.append([(_b20_value(u), row(nbrs)) for u, nbrs in G._pred.items()])
    for u, nbrs in G._adj.items():
        for v, d in nbrs.items():
            assert back[v][u] is d
    return state


def _b20_outcome(call):
    """Runs ``call(backend)`` with rustnx and with NetworkX and requires the
    same graph (or error). If rustnx declines, checks that NetworkX falls
    back cleanly with rustnx first in ``backend_priority.generators``.
    Returns whether rustnx ran the call itself."""

    def run(backend):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return ("ok", _b20_state(call(backend)))
        except Exception as exc:
            return (type(exc), exc.args)

    ours = run("rustnx")
    ran = ours[0] is not NotImplementedError
    if not ran:
        old = nx.config.backend_priority.generators
        nx.config.backend_priority.generators = ["rustnx"]
        try:
            ours = run(None)
        finally:
            nx.config.backend_priority.generators = old
    ref = run("networkx")
    assert ours == ref
    return ran


def _b20_attr_graph(seed, directed, multigraph=False):
    """A random graph with str, int, float, bool and None edge attributes."""
    rng = random.Random(seed)
    G = graph_for(seed, directed)
    if multigraph:
        G = (nx.MultiDiGraph if directed else nx.MultiGraph)(G)
        for u, v in list(G.edges())[: rng.randint(0, 5)]:
            G.add_edge(u, v)
    values = [1, -2, 0, 3.5, -0.25, 1e300, True, False, None, "x", "a b", "q'q", 'd"d']
    for *_, d in G.edges(data=True):
        for key in rng.sample(["weight", "color", "w2", "flag"], rng.randint(0, 3)):
            d[key] = rng.choice(values)
    return G


_B20_TOKENS = [
    "1", "2", "01", "-1", "+3", "0", "-0", "1.5", "-0.0", "0.0", "1e3", "nan", "inf",
    "-Infinity", "1_0", "x", "y", "١", "é", "{}", "{'a':", "1}", "{'a': [1]}",
    "{'key': 1}", "{'w': 2}", "{'w': 2,}", "'s'", "True", "None", "#", "a#b", ",", "\t",
    "{'w':", "0x1",
]


def _b20_random_lines(seed):
    rng = random.Random(seed)
    lines = []
    for _ in range(rng.randint(0, 25)):
        tokens = [rng.choice(_B20_TOKENS) for _ in range(rng.randint(0, 5))]
        sep = rng.choice([" ", " ", "  ", "\t", ","])
        line = sep.join(tokens)
        if rng.random() < 0.3:
            line += rng.choice(["\n", " \n", "\r\n", "  "])
        lines.append(line)
    return lines


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch20_edgelists(seed, directed):
    G = _b20_attr_graph(seed, directed, multigraph=seed % 4 == 3)
    cls = type(G)
    rng = random.Random(seed)
    for data in [True, False]:
        lines = list(nx.generate_edgelist(G, data=data))
        if rng.random() < 0.5:
            lines.insert(rng.randint(0, len(lines)), "# a comment")
            lines.append("")
        for create_using in [None, cls]:
            for nodetype in [None, int, str]:
                assert _b20_outcome(lambda backend: nx.parse_edgelist(
                    lines, create_using=create_using, nodetype=nodetype, backend=backend,
                )) or nodetype is int
                assert _b20_outcome(lambda backend: nx.bipartite.parse_edgelist(
                    lines, create_using=create_using, nodetype=nodetype, backend=backend,
                )) or nodetype is int
        text = "".join(line + "\n" for line in lines).encode()
        assert _b20_outcome(lambda backend: nx.read_edgelist(
            _b20_io.BytesIO(text), create_using=create_using, backend=backend,
        ))
        assert _b20_outcome(lambda backend: nx.bipartite.read_edgelist(
            _b20_io.BytesIO(text), create_using=create_using, backend=backend,
        ))
    # Typed data, and the "," delimiter (data joined with "," then).
    H = nx.Graph()
    for u, v in G.edges():
        H.add_edge(u, v, weight=rng.choice([1, 2.5, -3]), color=rng.choice(["r", "g"]))
    for delimiter in [None, ",", ";", " "]:
        lines = list(nx.generate_edgelist(H, delimiter=delimiter or " ", data=["weight", "color"]))
        for data in [(("weight", float), ("color", str)), [("weight", float)], True]:
            ran = _b20_outcome(lambda backend: nx.parse_edgelist(
                lines, delimiter=delimiter, data=data, backend=backend,
            ))
            assert ran or data is True or len(data) != 2
        dict_lines = list(nx.generate_edgelist(H, delimiter=delimiter or " "))
        _b20_outcome(lambda backend: nx.parse_edgelist(dict_lines, delimiter=delimiter, backend=backend))
        _b20_outcome(lambda backend: nx.bipartite.parse_edgelist(
            dict_lines, delimiter=delimiter, backend=backend))
    weighted = "".join(f"{u} {v} {d['weight']}\n" for u, v, d in H.edges(data=True)).encode()
    for nodetype in [None, str, float]:
        _b20_outcome(lambda backend: nx.read_weighted_edgelist(
            _b20_io.BytesIO(weighted), nodetype=nodetype, backend=backend))


@pytest.mark.parametrize("seed", range(150))
def test_batch20_edgelist_tokens(seed):
    # Odd tokens: numbers Python reads differently, broken dicts, reserved
    # attribute names, comments mid-token, CR, non-ASCII digits.
    lines = _b20_random_lines(seed)
    rng = random.Random(seed)
    comments = rng.choice(["#", "#", None, "", "a", "//"])
    delimiter = rng.choice([None, None, ",", "\t", " "])
    for nodetype in [None, int, float]:
        for data in [True, False, [("w", int)], (("w", float), ("c", str))]:
            for func in [nx.parse_edgelist, nx.bipartite.parse_edgelist]:
                _b20_outcome(lambda backend: func(
                    lines, comments=comments, delimiter=delimiter, nodetype=nodetype,
                    data=data, backend=backend,
                ))
            # An iterator: rustnx reads it all, then hands NetworkX a copy if needed.
            _b20_outcome(lambda backend: nx.parse_edgelist(
                iter(lines), comments=comments, delimiter=delimiter, nodetype=nodetype,
                data=data, backend=backend,
            ))
    text = "\n".join(lines).encode()
    for nodetype in [None, int]:
        _b20_outcome(lambda backend: nx.read_edgelist(
            _b20_io.BytesIO(text), comments=comments, delimiter=delimiter,
            nodetype=nodetype, backend=backend,
        ))
        _b20_outcome(lambda backend: nx.read_weighted_edgelist(
            _b20_io.BytesIO(text), comments=comments, delimiter=delimiter,
            nodetype=nodetype, backend=backend,
        ))
    for nodetype in [None, int]:
        _b20_outcome(lambda backend: nx.parse_adjlist(
            lines, comments=comments or "#", delimiter=delimiter, nodetype=nodetype,
            backend=backend,
        ))
        _b20_outcome(lambda backend: nx.read_adjlist(
            _b20_io.BytesIO(text), comments=comments or "#", delimiter=delimiter,
            nodetype=nodetype, backend=backend,
        ))
        for edgetype in [None, int, float, str]:
            _b20_outcome(lambda backend: nx.parse_multiline_adjlist(
                iter(lines), comments=comments or "#", delimiter=delimiter,
                nodetype=nodetype, edgetype=edgetype, backend=backend,
            ))
            _b20_outcome(lambda backend: nx.read_multiline_adjlist(
                _b20_io.BytesIO(text), comments=comments or "#", delimiter=delimiter,
                nodetype=nodetype, edgetype=edgetype, backend=backend,
            ))
    for func in [nx.parse_leda, nx.parse_pajek]:
        _b20_outcome(lambda backend: func(lines, backend=backend))
        _b20_outcome(lambda backend: func("\n".join(lines), backend=backend))


def test_batch20_literal_values():
    cases = [
        "{'a': 1, 'b': -2.5, 'c': 'x y', 'd': True, 'e': None, 'f': 1e400}",
        "{'a': 1, 'a': 2}",  # the last value wins, in the first key's place
        "{'a': +1, 'b': 00, 'c': .5, 'd': 5., 'e': 1E-3, 'f': -0}",
        "{ 'a' :1 , }",
        "{'a': 01}", "{'a': 1_000}", "{'a': 0x1}", "{'a': 1j}", "{'a': 'x' 'y'}",
        "{'a': \"q\"}", "{'a': b'x'}", "{'a': - 1}", "{'a': {}}", "{1: 2}",
        "{'self': 1}", "{'u_of_edge': 1}", "{'key': 1}", "[('a', 1)]", "{'a': 'b\\\\c'}",
        "{'a': 99999999999999999999}", "{'a': 1.5e}",
    ]
    for case in cases:
        for create_using in [None, nx.MultiGraph]:
            _b20_outcome(lambda backend: nx.parse_edgelist(
                [f"1 2 {case}"], create_using=create_using, backend=backend))
            _b20_outcome(lambda backend: nx.bipartite.parse_edgelist(
                [f"1 2 {case}"], create_using=create_using, backend=backend))
            _b20_outcome(lambda backend: nx.parse_multiline_adjlist(
                iter(["1 1", f"2 {case}"]), create_using=create_using, backend=backend))


def test_batch20_create_using():
    lines = ["1 2", "2 3 {'w': 1}", "3 1", "1 2 {'w': 2}"]
    for create_using in [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]:
        assert _b20_outcome(lambda backend: nx.parse_edgelist(
            lines, create_using=create_using, backend=backend))
        assert _b20_outcome(lambda backend: nx.parse_adjlist(
            lines, create_using=create_using, backend=backend))

        # An instance is cleared (graph attributes too) and filled.
        def filled(func, backend, create_using=create_using):
            G = create_using(name="old")
            G.add_edge("a", "b")
            H = func(lines, create_using=G, backend=backend)
            assert H is G
            return H

        assert _b20_outcome(lambda backend: filled(nx.parse_edgelist, backend))
        assert _b20_outcome(lambda backend: filled(nx.parse_adjlist, backend))

    class MyGraph(nx.Graph):
        pass

    frozen = nx.freeze(nx.Graph())
    for create_using in [MyGraph, MyGraph(), frozen, 3, "x"]:
        assert not _b20_outcome(lambda backend: nx.parse_edgelist(
            lines, create_using=create_using, backend=backend))


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch20_adjlists(seed, directed):
    G = _b20_attr_graph(seed, directed, multigraph=seed % 3 == 2)
    cls = type(G)
    lines = list(nx.generate_adjlist(G))
    text = "".join(line + "\n" for line in lines).encode()
    for create_using in [None, cls]:
        assert _b20_outcome(lambda backend: nx.parse_adjlist(
            lines, create_using=create_using, backend=backend))
        assert _b20_outcome(lambda backend: nx.read_adjlist(
            _b20_io.BytesIO(text), create_using=create_using, backend=backend))
        _b20_outcome(lambda backend: nx.parse_adjlist(
            lines, create_using=create_using, nodetype=int, backend=backend))
    # Multiline: dict data (literal_eval) and typed weights.
    H = nx.DiGraph() if directed else nx.Graph()
    H.add_nodes_from(G)
    rng = random.Random(seed)
    for u, v in G.edges():
        if rng.random() < 0.5:
            H.add_edge(u, v, weight=rng.choice([1, 2, 3]))
        else:
            H.add_edge(u, v)
    for graph, edgetypes in [(G, [None]), (H, [None, int, str])]:
        mlines = list(nx.generate_multiline_adjlist(graph))
        mtext = "".join(line + "\n" for line in mlines).encode()
        for edgetype in edgetypes:
            for create_using in [None, type(graph)]:
                _b20_outcome(lambda backend: nx.parse_multiline_adjlist(
                    iter(mlines), create_using=create_using, edgetype=edgetype,
                    backend=backend))
                ran = _b20_outcome(lambda backend: nx.read_multiline_adjlist(
                    _b20_io.BytesIO(mtext), create_using=create_using, edgetype=edgetype,
                    backend=backend))
                assert ran or edgetype is not None
    # A list fails in NetworkX for nodes with neighbors (next(list)); rustnx
    # leaves lists alone.
    assert not _b20_outcome(lambda backend: nx.parse_multiline_adjlist(
        ["a 1", "b"], backend=backend))
    assert not _b20_outcome(lambda backend: nx.parse_multiline_adjlist(
        ["a 0", "b 0"], backend=backend))


def test_batch20_multiline_adjlist_cases():
    cases = [
        ["a 2", "# c", "", "b {'weight': 2}", "c 7"],
        ["a 2", "b"],  # runs out of neighbors
        ["a x"], ["a"], ["a 1 2"], ["a -1", "b 0"], ["a 1", "  "], ["a 1", "b junk"],
        ["a 1", "b {'key': 1}"], ["a 1", "b 5"], ["a 1", "b {'a': 1}{'b': 2}"],
        ["a 1", "b {'a':", "1}"], ["a 2", "b {'a': 1}", "b {'b': 2}"],
    ]
    for lines in cases:
        for create_using in [None, nx.MultiDiGraph]:
            _b20_outcome(lambda backend: nx.parse_multiline_adjlist(
                iter(lines), create_using=create_using, backend=backend))
            for edgetype in [int, float]:
                _b20_outcome(lambda backend: nx.parse_multiline_adjlist(
                    iter(lines), create_using=create_using, edgetype=edgetype,
                    backend=backend))


def test_batch20_files(tmp_path):
    G = _b20_attr_graph(3, False)
    text = "".join(line + "\n" for line in nx.generate_edgelist(G)).encode()
    for suffix, opener in [("", open), (".gz", _b20_gzip.open), (".bz2", _b20_bz2.open)]:
        path = tmp_path / f"g.edgelist{suffix}"
        with opener(path, "wb") as f:
            f.write(text)
        for p in [path, str(path)]:
            assert _b20_outcome(lambda backend: nx.read_edgelist(p, backend=backend))
            assert _b20_outcome(lambda backend: nx.read_weighted_edgelist(
                p, data=None, backend=backend) if False else nx.read_adjlist(p, backend=backend))

            def from_handle(backend):
                with opener(path, "rb") as f:
                    return nx.read_edgelist(f, backend=backend)

            assert _b20_outcome(from_handle)
    # Text-mode handles fall back; reading resumes where the file was.
    path = tmp_path / "g.edgelist"

    def text_handle(backend):
        with open(path, encoding="utf-8") as f:
            return nx.read_edgelist(f, backend=backend)

    assert not _b20_outcome(text_handle)

    def resumed(backend):
        f = _b20_io.BytesIO(b"junk line\n1 2\n2 3 {'w': [1]}\n")
        f.seek(10)
        return nx.read_edgelist(f, backend=backend), f.tell()

    assert not _b20_outcome(lambda backend: resumed(backend)[0])
    # Encodings: rustnx decodes UTF-8, ASCII and Latin-1 itself.
    data = "é ÿ {'c': 'è'}\n".encode
    for encoding in ["utf-8", "UTF8", "latin-1", "ascii", "utf-16", "utf-8-sig", "nope", 3]:
        try:
            raw = data(encoding if encoding in ("utf-8", "latin-1", "utf-16", "utf-8-sig") else "utf-8")
        except (LookupError, TypeError):
            raw = b"a b\n"
        _b20_outcome(lambda backend: nx.read_edgelist(
            _b20_io.BytesIO(raw), encoding=encoding, backend=backend))
    # Invalid UTF-8 raises the same error after the fallback.
    assert not _b20_outcome(lambda backend: nx.read_edgelist(
        _b20_io.BytesIO(b"1 2\n\xff 3\n"), backend=backend))


def test_batch20_fallback_rewinds_files():
    # rustnx declines this file (a list value), so NetworkX must read it from
    # where the file was, under backend_priority.
    raw = b"skip\n1 2 {'w': 1}\n2 3 {'w': [1]}\n"
    old = nx.config.backend_priority.generators
    nx.config.backend_priority.generators = ["rustnx"]
    try:
        f = _b20_io.BytesIO(raw)
        f.seek(5)
        G = nx.read_edgelist(f)
    finally:
        nx.config.backend_priority.generators = old
    f = _b20_io.BytesIO(raw)
    f.seek(5)
    assert _b20_state(G) == _b20_state(nx.read_edgelist(f, backend="networkx"))


def _b20_leda_text(G, rng):
    nodes = list(G)
    index = {v: i + 1 for i, v in enumerate(nodes)}
    lines = ["#header", "LEDA.GRAPH", "string", "int", "-1" if G.is_directed() else "-2"]
    lines += ["#nodes", str(len(nodes))]
    lines += [rng.choice(["|{%s}|", "|{%s}|  ", "%s" if v else "|{%s}|"]) % v for v in nodes]
    lines += ["#edges", str(G.number_of_edges())]
    labels = ["", "x", "12", "long_label", "\u00e9t\u00e9"]
    lines += [f"{index[u]} {index[v]} 0 |{{{rng.choice(labels)}}}|" for u, v in G.edges()]
    return lines


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch20_leda(seed, directed):
    rng = random.Random(seed)
    G = graph_for(seed, directed)
    G = nx.relabel_nodes(G, {v: f"v{v}" for v in G})
    if seed % 5 == 0 and len(G):
        G = nx.relabel_nodes(G, {list(G)[0]: ""})  # no label: the node's number
    lines = _b20_leda_text(G, rng)
    text = "\n".join(lines)
    assert _b20_outcome(lambda backend: nx.parse_leda(text, backend=backend))
    assert _b20_outcome(lambda backend: nx.parse_leda(lines, backend=backend))
    assert _b20_outcome(lambda backend: nx.parse_leda(iter(lines), backend=backend))
    assert _b20_outcome(lambda backend: nx.read_leda(
        _b20_io.BytesIO(text.encode()), backend=backend))
    # Damaged files: missing lines, bad numbers, bad node indices.
    for _ in range(5):
        broken = list(lines)
        i = rng.randrange(len(broken))
        choice = rng.random()
        if choice < 0.4:
            del broken[i]
        elif choice < 0.7:
            broken[i] = rng.choice(["x", "0 9 0 |{}|", "1 1 0", "99 1 0 |{a}|", "-1", ""])
        else:
            broken = broken[:i]
        _b20_outcome(lambda backend: nx.parse_leda(broken, backend=backend))


def test_batch20_pajek_cases():
    cases = [
        '*network x\n*vertices 3\n1 "a b" 0.1 0.2 box ic Red\n2 b\n3 c\n*edges\n1 2 2.5\n2 3\n1 2 1 c red\n',
        "*Network\n*Vertices 2\n1 a\n2 b\n*Arcs\n1 2\n2 1 x\n1 2 1 k v\n",
        "*vertices 2\n1 a 0.5 junk box\n2 'b c'\n*arcs\n1 2 \"3\"\n",
        "*vertices 2\n1 a\n2 b\n*edges\n1 3\n3 1 1.5\n  \n1\n",
        "*vertices 2\n1 a\n2 a\n*edges\n1 2\n",
        "*vertices 1\n1 a\n*vertices 1\n1 b\n*edges\n1 1\n",
        "*edges\n1 2\n",  # no *vertices
        "*edges\n\n",
        "*vertices 2\n1 a\n2 b\n*matrix\n0 1\n1 0\n",
        "*vertices 3\n1 a\n2 b\n",  # too few vertex lines
        "*vertices x\n", "*vertices 2 3\n", "*vertices 1\n1\n",
        '*vertices 1\n1 "unclosed\n', "*vertices 1\n1 a\\\n", '*vertices 1\n1 "a\\"b" 1 2 s\n',
        "*vertices 1\n1 a\\ b\n*edges\n1 1 1 key 3\n", "*vertices 1\n1 a\n*edges\n1 1 1 self x\n",
        "*vertices 1\n1 a 1 2 s id 9 x y\n", "*network  two words  \n",
        "*vertices 1\n1 a 1_0 2 s\n", "*vertices 1\n1 a inf -nan s\n",
        "*İ\n", "*networK\n*vertices 1\n1 é\n",
        "*vertices 2\r\n1 a\r\n2 b\r\n*edges\r\n1 2 3\r\n",
    ]
    for text in cases:
        _b20_outcome(lambda backend: nx.parse_pajek(text, backend=backend))
        _b20_outcome(lambda backend: nx.parse_pajek(text.split("\n"), backend=backend))
        _b20_outcome(lambda backend: nx.read_pajek(
            _b20_io.BytesIO(text.encode()), backend=backend))


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(20))
def test_batch20_pajek(seed, directed):
    G = _b20_attr_graph(seed, directed, multigraph=seed % 2 == 1)
    rng = random.Random(seed)
    for v, d in G.nodes(data=True):
        if rng.random() < 0.5:
            d.update(x=rng.random(), y=rng.choice([1, 2.5]), shape="box")
    for *_, d in G.edges(data=True):
        for key in [k for k in d if k != "weight"]:
            d[key] = str(d[key]).replace('"', "").replace("'", "")
        if "weight" in d and not isinstance(d["weight"], (int, float)):
            del d["weight"]
    text = "\n".join(nx.generate_pajek(G))
    assert _b20_outcome(lambda backend: nx.parse_pajek(text, backend=backend))
    assert _b20_outcome(lambda backend: nx.read_pajek(
        _b20_io.BytesIO(text.encode()), backend=backend))


@pytest.mark.parametrize("seed", range(40))
def test_batch20_graph6_sparse6(seed):
    rng = random.Random(seed)
    G = nx.convert_node_labels_to_integers(graph_for(seed, False))
    G.remove_edges_from(nx.selfloop_edges(G))
    for header in [False, True]:
        g6 = nx.to_graph6_bytes(G, header=header)
        for data in [g6, g6.rstrip(b"\n"), g6 + b"\n", g6[:-2], g6[:-1] + b"\x00"]:
            _b20_outcome(lambda backend: nx.from_graph6_bytes(data, backend=backend))
        s6 = nx.to_sparse6_bytes(G, header=header)
        for data in [s6, s6.rstrip(b"\n"), s6[:-2], b">>sparse6<<" + s6[-3:]]:
            _b20_outcome(lambda backend: nx.from_sparse6_bytes(data, backend=backend))
    assert _b20_outcome(lambda backend: nx.from_graph6_bytes(
        nx.to_graph6_bytes(G, header=False).rstrip(b"\n"), backend=backend))
    assert _b20_outcome(lambda backend: nx.from_sparse6_bytes(
        nx.to_sparse6_bytes(G, header=False).rstrip(b"\n"), backend=backend))
    # Parallel edges and self-loops make sparse6 return a MultiGraph.
    M = nx.MultiGraph(G)
    for _ in range(rng.randint(0, 4)):
        if len(G):
            M.add_edge(rng.choice(list(G)), rng.choice(list(G)))
    s6 = nx.to_sparse6_bytes(M, header=False).rstrip(b"\n")
    assert _b20_outcome(lambda backend: nx.from_sparse6_bytes(s6, backend=backend))
    # Files: one graph per line, blank lines skipped, whitespace stripped.
    graphs = [nx.convert_node_labels_to_integers(graph_for(seed + k, False)) for k in range(3)]
    for H in graphs:
        H.remove_edges_from(nx.selfloop_edges(H))
    for writer, reader in [(nx.to_graph6_bytes, nx.read_graph6),
                           (nx.to_sparse6_bytes, nx.read_sparse6)]:
        chunks = [writer(H, header=rng.random() < 0.5) for H in graphs]
        for raw in [chunks[0], b"".join(chunks), b"\n \x0b".join(chunks) + b"\n\n", b"", b"\n"]:
            _b20_outcome(lambda backend: reader(_b20_io.BytesIO(raw), backend=backend))


def test_batch20_graph6_errors():
    for data in [b"", b"\n", b"A", b"A_a", b"@", b"~", b"~?", b"~~", b">>graph6<<", b"A_\n\n",
                 b"A\x7f", b"B_ ", "A_", bytearray(b"A_")]:
        _b20_outcome(lambda backend: nx.from_graph6_bytes(data, backend=backend))
    for data in [b"", b":", b"A", b":A", b":A_", b":~", b":~~", b":@", b":A\x00", b">>sparse6<<",
                 b":Fa@x^", b":Fa@x^\n", ":A_", bytearray(b":A_")]:
        _b20_outcome(lambda backend: nx.from_sparse6_bytes(data, backend=backend))


import json as _b20_json

_b20_jg = nx.readwrite.json_graph


def _b20_json_graph(seed, directed, multigraph):
    """A random graph with tuple, int, float, bool and str nodes and assorted
    attribute values (lists and dicts too)."""
    rng = random.Random(seed)
    G = _b20_attr_graph(seed, directed, multigraph)
    labels = [0, 1, 2.5, -0.0, (1, 2), ("a", (3,)), "s", True, 10**20]
    mapping = {v: rng.choice(labels) if rng.random() < 0.3 else v for v in G}
    G = nx.relabel_nodes(G, mapping)
    for v, d in G.nodes(data=True):
        if rng.random() < 0.4:
            d["label"] = rng.choice(["x", 1, [1, 2], {"k": 1}, None])
    if multigraph:
        for u, v, k, d in list(G.edges(keys=True, data=True))[:3]:
            G.add_edge(u, v, key=rng.choice(["k", 7, (1, 2)]), **d)
    G.graph.update(name=f"g{seed}", meta=[1, 2])
    return G


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch20_json_graphs(seed, directed):
    G = _b20_json_graph(seed, directed, multigraph=seed % 2 == 1)
    multi = G.is_multigraph()
    cases = []
    for edges in ["links", "edges"]:
        data = _b20_jg.node_link_data(G, edges=edges)
        cases.append((_b20_jg.node_link_graph, data, {"edges": edges}))
    cases.append((_b20_jg.adjacency_graph, _b20_jg.adjacency_data(G), {}))
    if not any(isinstance(v, bool) or v is None for v in G):
        cases.append((_b20_jg.cytoscape_graph, _b20_jg.cytoscape_data(G), {}))
    for func, data, kwargs in cases:
        # As built, and through JSON (tuples become lists, keys become str).
        loaded = _b20_json.loads(_b20_json.dumps(data, default=str))
        for d in [data, loaded]:
            for flags in [{}, {"directed": not directed, "multigraph": not multi}]:
                if func is _b20_jg.cytoscape_graph:
                    if flags:
                        continue
                ran = _b20_outcome(lambda backend: func(d, **kwargs, **flags, backend=backend))
                assert ran or d is loaded
    # node_link_graph with renamed fields and missing ids (numbered by position).
    data = _b20_jg.node_link_data(G, source="s", target="t", name="n", key="k", edges="e",
                                  nodes="v")
    for d in data["v"][::3]:
        del d["n"]
    _b20_outcome(lambda backend: _b20_jg.node_link_graph(
        data, source="s", target="t", name="n", key="k", edges="e", nodes="v", backend=backend))
    # tree_graph on a random tree (attributes on some nodes).
    rng = random.Random(seed)
    T = nx.bfs_tree(nx.random_labeled_tree(30, seed=seed) if hasattr(nx, "random_labeled_tree")
                    else nx.balanced_tree(2, 4), 0)
    for v in T:
        if rng.random() < 0.5:
            T.nodes[v]["w"] = rng.choice([1, "x", [2], None])
    for ident, children in [("id", "children"), ("name", "kids")]:
        data = _b20_jg.tree_data(T, 0, ident=ident, children=children)
        assert _b20_outcome(lambda backend: _b20_jg.tree_graph(
            data, ident=ident, children=children, backend=backend))


def test_batch20_graph_dict_identity():
    data = _b20_jg.node_link_data(nx.path_graph(3, create_using=nx.MultiGraph), edges="edges")
    G = _b20_jg.node_link_graph(data, edges="edges", backend="rustnx")
    assert G.graph is data["graph"]


def test_batch20_json_edge_cases():
    nl = _b20_jg.node_link_graph
    cases = [
        {"nodes": [{"id": 1}, {"id": 1, "a": 2}, {"id": True, "b": 3}, {"id": 1.0}], "edges": []},
        {"nodes": [{"id": 0.0}, {"id": -0.0}], "edges": [{"source": -0.0, "target": 0.0},
                                                       {"source": 1, "target": -0.0}]},
        {"nodes": [{"id": [1, [2]]}], "edges": [{"source": [1, [2]], "target": 3}]},
        {"nodes": [{"id": [1, 2]}], "edges": [{"source": [1, 2], "target": [1, 2], "key": 4}]},
        {"nodes": [{"id": None}], "edges": []},
        {"nodes": [], "edges": [{"source": None, "target": 1}]},
        {"nodes": [], "edges": [{"source": 1}]},
        {"nodes": [{"id": 1, "node_for_adding": 2}], "edges": []},
        {"nodes": [{"id": 1, "self": 2}], "edges": []},
        {"nodes": [], "edges": [{"source": 1, "target": 2, "u_of_edge": 3}]},
        {"nodes": [], "edges": [{"source": 1, "target": 2, "key": None, "w": 1}]},
        {"nodes": [], "edges": [{"source": 1, "target": 2, "key": [1]}]},
        {"nodes": [{"id": {1: 2}}], "edges": []},
        {"nodes": [{"id": 1, 5: "int key"}], "edges": []},
        {"nodes": ({"id": 1},), "edges": []},
        {"nodes": [], "edges": [], "graph": [("a", 1)], "multigraph": False, "directed": True},
        {"nodes": [], "edges": [], "multigraph": "yes"},
        {"edges": []}, {"nodes": []}, [], None,
    ]
    for data in cases:
        for kwargs in [{"edges": "edges"}, {"edges": "edges", "multigraph": False},
                       {"edges": "edges", "key": "w"}, {}]:
            _b20_outcome(lambda backend: nl(data, **kwargs, backend=backend))
    adj = _b20_jg.adjacency_graph
    cases = [
        {"nodes": [{"id": 1, "x": 1}, {"id": 2}], "adjacency": [[{"id": 2, "key": 0, "w": 1}], []]},
        {"nodes": [{"id": 1}], "adjacency": [[{"id": 2}]]},  # no key in a multigraph
        {"nodes": [{"id": 1}], "adjacency": [[{"id": 2, "key": None}]]},
        {"nodes": [{"id": 1}], "adjacency": [[], [{"id": 2}]]},  # more rows than nodes
        {"nodes": [{"x": 1}], "adjacency": [[]]},
        {"nodes": [{"id": 1}], "adjacency": [[{"key": 1}]]},
        {"nodes": [{"id": 1}]},
        {"nodes": [{"id": 1}], "adjacency": [[{"id": 1, "key": "a", 3: "int key"}]],
         "graph": {"name": "x"}, "multigraph": False},
        {"nodes": [], "adjacency": [], "graph": 5},
    ]
    for data in cases:
        for kwargs in [{}, {"multigraph": False}, {"attrs": {"id": "id", "key": "key"}},
                       {"attrs": {"id": "x", "key": "id"}}, {"attrs": {"key": "key"}}]:
            _b20_outcome(lambda backend: adj(data, **kwargs, backend=backend))

    class Ambiguous:
        def __bool__(self):
            raise ValueError("ambiguous")

    cy = _b20_jg.cytoscape_graph
    base = {"data": [], "directed": False, "multigraph": False}
    cases = [
        {**base, "elements": {"nodes": [{"data": {"value": 1, "name": "a", "id": "1"}}],
                              "edges": [{"data": {"source": 1, "target": 2, "w": 3}}]}},
        {**base, "multigraph": True, "elements": {"nodes": [], "edges": [
            {"data": {"source": 1, "target": 2}}, {"data": {"source": 1, "target": 2, "key": 0}},
            {"data": {"source": 1, "target": 2, "key": None}}]}},
        {**base, "elements": {"nodes": [{"data": {"value": 1, "name": Ambiguous()}}], "edges": []}},
        {**base, "elements": {"nodes": [{"data": {"name": "a"}}], "edges": []}},
        {**base, "elements": {"nodes": [{"value": 1}], "edges": []}},
        {**base, "elements": {"nodes": []}},
        {**base, "data": None, "elements": {"nodes": [], "edges": []}},
        {**base, "data": {"a": 1}, "directed": True, "elements": {"nodes": [], "edges": [
            {"data": {"source": (1, 2), "target": 2}}]}},
    ]
    for data in cases:
        for kwargs in [{}, {"name": "id", "ident": "id"}, {"name": "label"}]:
            _b20_outcome(lambda backend: cy(data, **kwargs, backend=backend))
    tree = _b20_jg.tree_graph
    deep = {"id": 0}
    node = deep
    for i in range(1, 400):
        node["children"] = [{"id": i}]
        node = node["children"][0]
    cases = [
        {"id": 1, "children": [{"id": 2, "a": 1}, {"id": 3, "children": [{"id": 2, "b": 2}]}]},
        {"id": 1, "children": [{"id": 2, "children": []}, {"id": 3, "children": None}]},
        {"id": 1, "children": ({"id": 2},)},
        {"id": 1, "children": [{"x": 2}]},
        {"id": 1, "self": 2},
        {"id": None},
        {"id": 1, "children": [{"id": 1}]},
        {"children": []},
        deep,
        [],
    ]
    for data in cases:
        _b20_outcome(lambda backend: tree(data, backend=backend))


def _b20_gml_graph(seed, directed, multigraph):
    rng = random.Random(seed)
    G = _b20_attr_graph(seed, directed, multigraph)
    for _, d in G.nodes(data=True):
        if rng.random() < 0.5:
            d.update(rng.choice([
                {"w": 1.5}, {"s": "café \"q\" & <x>"}, {"l": [1, 2.5, "x"]},
                {"d": {"k": 1, "n": {"m": "x"}}}, {"e": ()}, {"f": []}, {"big": 10**30},
                {"inf": float("inf"), "ninf": float("-inf")}, {"one": [7]},
            ]))
    for *_, d in G.edges(data=True):
        for k in list(d):
            if d[k] is None or isinstance(d[k], bool):
                d[k] = str(d[k])
    if multigraph:
        for u, v in list(G.edges())[:3]:
            G.add_edge(u, v, key=rng.choice(["k", 9, 2.5]))
    G.graph.update(name=f"g{seed}", tags=["a", "b"])
    return G


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_batch20_gml(seed, directed):
    G = _b20_gml_graph(seed, directed, multigraph=seed % 2 == 1)
    text = "\n".join(nx.generate_gml(G))
    for label in ["label", "id", None]:
        assert _b20_outcome(lambda backend: nx.parse_gml(text, label=label, backend=backend))
        lines = text.split("\n")
        assert _b20_outcome(lambda backend: nx.parse_gml(lines, label=label, backend=backend))
        assert _b20_outcome(lambda backend: nx.parse_gml(
            iter([line + "\n" for line in lines]), label=label, backend=backend))
        assert _b20_outcome(lambda backend: nx.read_gml(
            _b20_io.BytesIO(text.encode()), label=label, backend=backend))
    # Labels other than "label" (this one is unique: the node's own id).
    H = nx.relabel_nodes(G, {v: i for i, v in enumerate(G)})
    for v, d in H.nodes(data=True):
        d["name"] = f"n{v}"
    text = "\n".join(nx.generate_gml(H))
    assert _b20_outcome(lambda backend: nx.parse_gml(text, label="name", backend=backend))
    _b20_outcome(lambda backend: nx.parse_gml(text, label="missing", backend=backend))
    _b20_outcome(lambda backend: nx.parse_gml(text, destringizer=nx.readwrite.gml.literal_destringizer,
                                              backend=backend))


_B20_GML_TOKENS = [
    "graph", "[", "]", "node", "edge", "id", "label", "source", "target", "key", "directed",
    "multigraph", "1", "0", "-2", "1.5", "+INF", "-INF", "INF", "NAN", "-INFe5", "1e5", "1.",
    ".5", "5.e3", "1.5x", '"s"', '"()"', '"[]"', '"&#65;&amp;"', '"&nbsp;"', '"&#55296;"',
    '"&#99999999;"', '"&#x41;"', '"a', 'b"', "# c", " ", "é", "\t", "abc_9",
    "_networkx_list_start", '"_networkx_list_start"', "+", "99999999999999999999", "self",
]


@pytest.mark.parametrize("seed", range(150))
def test_batch20_gml_tokens(seed):
    rng = random.Random(seed)
    lines = []
    for _ in range(rng.randint(0, 12)):
        lines.append(" ".join(rng.choice(_B20_GML_TOKENS) for _ in range(rng.randint(0, 6))))
    # Mostly well-formed graphs with odd values.
    body = []
    n = rng.randint(0, 5)
    for i in range(n):
        body.append(f"node [ id {rng.choice([i, i, str(i), f'{i}.0'])} label \"{rng.choice('abcde')}\" "
                    f"{rng.choice(_B20_GML_TOKENS[3:])} {rng.choice(_B20_GML_TOKENS[12:])} ]")
    for _ in range(rng.randint(0, 6)):
        if n:
            body.append(f"edge [ source {rng.randrange(n)} target {rng.randrange(n)} "
                        f"{rng.choice(['', 'key 1', 'key 1.0', 'key \"k\"', 'w 2'])} ]")
    header = rng.choice(["", "directed 1", "multigraph 1", "directed 1 multigraph 1", "multigraph 0"])
    texts = ["\n".join(lines), f"graph [ {header}\n" + "\n".join(body) + "\n]",
             "graph [\n" + "\n".join(lines) + "\n]"]
    for text in texts:
        for label in ["label", "id"]:
            _b20_outcome(lambda backend: nx.parse_gml(text, label=label, backend=backend))
            _b20_outcome(lambda backend: nx.parse_gml(text.split("\n"), label=label, backend=backend))


def test_batch20_gml_cases():
    cases = [
        'graph [ node [ id 1 label "a\n  b" ] ]',  # a string over two lines
        'graph [ node [ id 1 label "a\n  b\n c" ] ]',
        'graph [ node [ id 1 label "a\n\n c" ] ]',
        'graph [ node [ id 1 label "a" x "b\nc" ] ]',
        'graph [ node [ id 1 label "open\n',
        "graph [ node [ id abc label def ] edge [ source abc target abc ] ]",
        "graph [ node [ id 1 ] node [ id 1.0 ] ]",
        "graph [ node [ id 1 label 2 ] node [ id 2 label 2.0 ] ]",
        "graph [ node [ id 1 ] edge [ source 1 target 2 ] ]",
        "graph [ node [ id 1 ] edge [ source 1.0 target 1 ] ]",
        "graph [ node [ id 1 ] edge [ source 1 target 1 ] edge [ source 1 target 1 ] ]",
        "graph [ multigraph 1 node [ id 1 ] edge [ source 1 target 1 ] edge [ source 1 target 1 key 0 ] ]",
        "graph [ multigraph 1 node [ id 1 ] edge [ source 1 target 1 key 1 ] edge [ source 1 target 1 ] edge [ source 1 target 1 ] ]",
        "graph [ multigraph 1 node [ id 1 ] edge [ source 1 target 1 key NAN ] edge [ source 1 target 1 key NAN ] ]",
        "graph [ multigraph 1 node [ id 1 ] edge [ source 1 target 1 key [ a 1 ] ] ]",
        "graph [ node [ id 1 node_for_adding 2 ] ]",
        "graph [ node [ id 1 ] edge [ source 1 target 1 u_of_edge 2 ] ]",
        "graph [ node [ id [ a 1 ] ] ]",
        'graph [ node [ id "()" label "x" ] ]',
        'graph [ node [ id 1 label "[]" ] ]',
        "graph [ node [ id 1 x NAN y INF z -INF w +INF ] ]",
        "graph [ node [ id 1 x -INFe5 ] ]",
        "graph [ node [ id 1 x 1e5 ] ]",
        "graph [ node [ id 1 x 99999999999999999999 ] ]",
        'graph [ node [ id 1 l "_networkx_list_start" l 2 ] node [ id 2 l "_networkx_list_start" ] ]',
        'graph [ name "&amp;&lt;&gt;&quot;&#65;&#x42;&#99999999;&#; & &x41; &nbsp;" ]',
        'graph [ name "&#55296;" ]',
        "graph [ directed 1 directed 0 ]",
        'graph [ directed "" multigraph 0.0 ]',
        "graph [ ] graph [ ]",
        "graph 5", "", "node [ id 1 ]", "graph [ node 5 ]", "graph [ ] ]", "graph [",
        "graph [ id ] ]", "graph [ x ]",
        "Creator \"me\"\ngraph [ node [ id 0 ] ]",
        "graph [ node [ id 0 ] ]",
        "graph [ node [ id 0 label \"café\" ] ]",
        "graph [ né 1 ]",
        "graph [ node [ id 0 ] ]\r\n# trailing\r\n",
        "graph [ node [ id 0 ] ]\x0bgraph",
        "graph [ x" + " [ y" * 150 + " ]" * 151,
    ]
    for text in cases:
        for label in ["label", "id", None]:
            _b20_outcome(lambda backend: nx.parse_gml(text, label=label, backend=backend))
            _b20_outcome(lambda backend: nx.parse_gml(text.split("\n"), label=label, backend=backend))
            _b20_outcome(lambda backend: nx.read_gml(
                _b20_io.BytesIO(text.encode()), label=label, backend=backend))
    # List items: one trailing newline is dropped; another one inside raises.
    for lines in [["graph [ ]\n"], ["graph [\n", "]\n\n"], ["graph [ ]\r\n"], [b"graph [ ]"],
                  ("graph [ ]",), ["graph [ ", 3, "]"]]:
        _b20_outcome(lambda backend: nx.parse_gml(lines, backend=backend))
