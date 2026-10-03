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
        exact_outcome(listed(dag_func("colliders")), G)
        exact_outcome(listed(dag_func("v_structures")), G)
        exact_outcome(first(dag_func("root_to_leaf_paths")), G)
    S = random_dag(seed, weights, small=True)
    for G in [S, C.subgraph(list(C)[:6]).copy(), U]:
        exact_outcome(first(nx.all_topological_sorts, 2000), G)
        exact_outcome(graph_parts(nx.dag_to_branching), G)
        exact_outcome(listed(dag_func("root_to_leaf_paths")), G)
    if weights == "none":
        assert runs_in_rustnx(nx.dag_longest_path, D) == nx.dag_longest_path(D, backend="networkx")
        for name in ["has_cycle", "colliders", "v_structures"]:
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
        lambda G, b: nx.dag.colliders(G, backend=b),
        lambda G, b: nx.dag.root_to_leaf_paths(G, backend=b),
        lambda G, b: nx.all_topological_sorts(G, backend=b),
        lambda G, b: nx.lexicographical_topological_sort(G, backend=b),
    ]
    for call in calls:
        G = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (2, 4)])
        it = call(G, "rustnx")
        next(it)
        G.add_edge("x", "y")
        with pytest.raises(RuntimeError):
            list(it)


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
            nx.dag.colliders(U, backend=backend)
