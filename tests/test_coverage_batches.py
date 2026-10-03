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
