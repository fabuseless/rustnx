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
