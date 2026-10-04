"""Functions added in the NetworkX coverage batches (todo items 28 to 50).

Same approach as test_against_networkx.py: run each function with
``backend="rustnx"`` and ``backend="networkx"`` and require identical
results, including the iteration order of returned sets.
"""

import random
import warnings

import networkx as nx
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
