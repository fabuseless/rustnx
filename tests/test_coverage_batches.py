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


@pytest.mark.parametrize("seed", range(40))
def test_batch9_tournaments(seed):
    rng = random.Random(seed)
    T = tournament.random_tournament(rng.randint(0, 12), seed=seed)
    # NetworkX 3.4 and 3.5 take time quintic in the number of nodes here.
    D = graph_for(seed, True)
    D = D.subgraph(list(D)[:12]).copy()
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
