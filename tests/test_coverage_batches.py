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
