"""rustnx.Graph / rustnx.DiGraph against networkx graphs built from the same edges.

A native graph must behave exactly like a ``networkx.Graph`` built with
``add_edge`` calls in the same order: same nodes, same adjacency order, same
merged attributes for duplicate edges, and so the same algorithm results.
"""

import math
import pickle
import random

import networkx as nx
import pytest

import rustnx


def random_edges(seed, directed):
    rng = random.Random(seed)
    n = rng.randint(1, 30)
    labels = list(range(n))
    if rng.random() < 0.5:
        labels = [f"n{i}" if i % 2 else (i, "t") for i in labels]
    edges = []
    for _ in range(rng.randint(0, 3 * n)):
        u, v = rng.choice(labels), rng.choice(labels)
        form = rng.random()
        if form < 0.3:
            edges.append((u, v))
        elif form < 0.6:
            edges.append((u, v, rng.choice([1, 2, 3, 0.5, 2.25])))
        else:
            d = {}
            if rng.random() < 0.8:
                d["weight"] = rng.choice([1, 2, 4, 1.5])
            if rng.random() < 0.3:
                d["cap"] = rng.randint(0, 9)
            edges.append((u, v, d))
    # Duplicates and reversed duplicates exercise NetworkX's merge rules.
    for _ in range(rng.randint(0, 5)):
        if edges:
            e = rng.choice(edges)
            edges.append((e[1], e[0]) + tuple(e[2:]) if rng.random() < 0.5 else e)
    extra_nodes = [f"lonely{i}" for i in range(rng.randint(0, 2))]
    return edges, extra_nodes


def to_nx(edges, nodes, directed):
    H = nx.DiGraph() if directed else nx.Graph()
    H.add_nodes_from(nodes)
    for e in edges:
        if len(e) == 2:
            H.add_edge(*e)
        elif isinstance(e[2], dict):
            H.add_edge(e[0], e[1], **e[2])
        else:
            H.add_edge(e[0], e[1], weight=e[2])
    return H


def build(seed, directed):
    edges, nodes = random_edges(seed, directed)
    cls = rustnx.DiGraph if directed else rustnx.Graph
    return cls(edges, nodes=nodes), to_nx(edges, nodes, directed)


def outcome(func):
    try:
        result = func()
        if hasattr(result, "__next__"):
            result = list(result)
            result = [
                (k, list(v.items())) if isinstance(v, dict) else (k, v)
                for k, v in result
            ] if result and isinstance(result[0], tuple) else result
        return ("ok", result)
    except Exception as exc:
        if isinstance(exc, nx.PowerIterationFailedConvergence):
            return (type(exc), str(exc))  # its args hold the exception itself
        return (type(exc), exc.args)


def same(a, b):
    """Equal, treating floats within 1e-12 as equal (parallel sums)."""
    if a == b:
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) or (
            math.isnan(a) and math.isnan(b)
        )
    return False


SEEDS = range(50)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_structure_matches_networkx(seed, directed):
    G, H = build(seed, directed)
    assert G.nodes() == list(H)
    assert G.edges() == list(H.edges())
    assert G.edges(data=True) == list(H.edges(data=True))
    assert G.number_of_edges() == H.number_of_edges()
    assert G.degree() == dict(H.degree())
    for v in H:
        assert list(G.neighbors(v)) == list(H.neighbors(v))
        if directed:
            assert list(G.predecessors(v)) == list(H.predecessors(v))
    back = G.to_networkx()
    assert list(back) == list(H)
    assert list(back.edges(data=True)) == list(H.edges(data=True))
    for v in H:  # adjacency order too
        assert list(back.adj[v]) == list(H.adj[v])


def algorithm_calls(H, directed):
    nodes = list(H)
    src = nodes[0] if nodes else None
    calls = {
        "betweenness": lambda G, b: nx.betweenness_centrality(G, backend=b),
        "betweenness_w": lambda G, b: nx.betweenness_centrality(G, weight="weight", backend=b),
        "edge_betweenness": lambda G, b: list(nx.edge_betweenness_centrality(G, backend=b).items()),
        "edge_betweenness_w": lambda G, b: list(
            nx.edge_betweenness_centrality(G, weight="weight", backend=b).items()
        ),
        "closeness": lambda G, b: nx.closeness_centrality(G, backend=b),
        "closeness_w": lambda G, b: nx.closeness_centrality(G, distance="weight", backend=b),
        "pagerank": lambda G, b: nx.pagerank(G, backend=b),
        "pagerank_cap": lambda G, b: nx.pagerank(G, weight="cap", backend=b),
        "pagerank_missing_attr": lambda G, b: nx.pagerank(G, weight="nope", backend=b),
        "all_pairs": lambda G, b: nx.all_pairs_shortest_path_length(G, backend=b),
        "all_pairs_w": lambda G, b: nx.all_pairs_dijkstra_path_length(G, backend=b),
        "all_paths": lambda G, b: nx.all_pairs_shortest_path(G, backend=b),
        "all_paths_w": lambda G, b: nx.all_pairs_dijkstra_path(G, backend=b),
        "diameter": lambda G, b: nx.diameter(G, backend=b),
        "harmonic": lambda G, b: list(nx.harmonic_centrality(G, backend=b).items()),
        "eigenvector": lambda G, b: list(nx.eigenvector_centrality(G, max_iter=1000, backend=b).items()),
        "katz": lambda G, b: list(nx.katz_centrality(G, alpha=0.02, backend=b).items()),
        "is_bipartite": lambda G, b: nx.is_bipartite(G, backend=b),
        "core_number": lambda G, b: list(nx.core_number(G, backend=b).items()),
        "k_core": lambda G, b: sorted(map(str, nx.k_core(G, backend=b).edges)),
        "clustering": lambda G, b: list(nx.clustering(G, backend=b).items()),
        "avg_clustering": lambda G, b: nx.average_clustering(G, backend=b),
        "transitivity": lambda G, b: nx.transitivity(G, backend=b),
        "avg_spl_w": lambda G, b: nx.average_shortest_path_length(G, weight="weight", backend=b),
    }
    if src is not None:
        calls["bfs"] = lambda G, b: nx.single_source_shortest_path_length(G, src, backend=b)
        calls["dijkstra"] = lambda G, b: nx.single_source_dijkstra_path_length(G, src, backend=b)
        calls["paths"] = lambda G, b: list(nx.single_source_shortest_path(G, src, backend=b).items())
        calls["paths_w"] = lambda G, b: list(nx.single_source_dijkstra_path(G, src, backend=b).items())
        calls["target_paths"] = lambda G, b: list(nx.shortest_path(G, target=src, backend=b).items())
        calls["target_paths_w"] = lambda G, b: list(
            nx.shortest_path(G, target=src, weight="weight", backend=b).items()
        )
        dst = list(H)[-1]
        calls["pair"] = lambda G, b: nx.shortest_path(G, src, dst, backend=b)
        calls["pair_w"] = lambda G, b: nx.dijkstra_path(G, src, dst, backend=b)
        calls["has_path"] = lambda G, b: nx.has_path(G, dst, src, backend=b)
        calls["descendants"] = lambda G, b: nx.descendants(G, src, backend=b)
        calls["bidir_dijkstra"] = lambda G, b: nx.bidirectional_dijkstra(G, src, dst, backend=b)
        calls["bfs_edges"] = lambda G, b: list(nx.bfs_edges(G, src, backend=b))
        calls["dfs_edges"] = lambda G, b: list(nx.dfs_edges(G, src, backend=b))
        calls["dfs_preorder"] = lambda G, b: list(nx.dfs_preorder_nodes(G, backend=b))
        calls["ancestors"] = lambda G, b: nx.ancestors(G, src, backend=b)
    if directed:
        calls["scc"] = lambda G, b: list(nx.strongly_connected_components(G, backend=b))
        calls["wcc"] = lambda G, b: list(nx.weakly_connected_components(G, backend=b))
        calls["topo"] = lambda G, b: list(nx.topological_sort(G, backend=b))
        calls["dag"] = lambda G, b: nx.is_directed_acyclic_graph(G, backend=b)
    else:
        calls["cc"] = lambda G, b: list(nx.connected_components(G, backend=b))
        calls["triangles"] = lambda G, b: list(nx.triangles(G, backend=b).items())
    return calls


@pytest.fixture
def enabled():
    old = nx.config.backend_priority.algos, nx.config.fallback_to_nx
    rustnx.enable()
    yield
    nx.config.backend_priority.algos, nx.config.fallback_to_nx = old


def check_all(G, H, directed):
    for name, call in algorithm_calls(H, directed).items():
        ours = outcome(lambda: call(G, "rustnx"))
        if ours[0] is NotImplementedError:
            # rustnx deliberately hands this case to NetworkX (e.g. empty
            # graphs); through normal dispatch the native graph is converted.
            ours = outcome(lambda: call(G, None))
        ref = outcome(lambda: call(H, "networkx"))
        assert same(ours, ref), name


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_algorithms_match_networkx(seed, directed, enabled):
    G, H = build(seed, directed)
    check_all(G, H, directed)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(20))
def test_from_arrays(seed, directed, enabled):
    rng = random.Random(seed)
    n = rng.randint(1, 40)
    m = rng.randint(0, 3 * n)
    src = [rng.randrange(n) for _ in range(m)]
    dst = [rng.randrange(n) for _ in range(m)]
    w = [rng.choice([0.5, 1.0, 2.0]) for _ in range(m)]
    cls = rustnx.DiGraph if directed else rustnx.Graph
    G = cls.from_arrays(src, dst, w, num_nodes=n)
    H = nx.DiGraph() if directed else nx.Graph()
    H.add_nodes_from(range(n))
    H.add_weighted_edges_from(zip(src, dst, w))
    assert G.nodes() == list(H)
    assert G.edges(data=True) == list(H.edges(data=True))
    check_all(G, H, directed)


def test_from_numpy_arrays():
    np = pytest.importorskip("numpy")
    src = np.array([0, 1, 2], dtype=np.int32)
    dst = np.array([1, 2, 0], dtype=np.uint16)
    G = rustnx.DiGraph.from_arrays(src, dst, np.array([1, 2, 3], dtype=np.float32))
    assert G.edges(data=True) == [(0, 1, {"weight": 1.0}), (1, 2, {"weight": 2.0}), (2, 0, {"weight": 3.0})]
    with pytest.raises(TypeError, match="integers"):
        rustnx.Graph.from_arrays(np.array([0.5]), np.array([1]))
    with pytest.raises(ValueError, match="num_nodes"):
        rustnx.Graph.from_arrays([0, 5], [1, 2], num_nodes=3)


def test_range_index_lookups():
    G = rustnx.Graph.from_arrays([0, 1], [1, 2])
    assert 1 in G and True in G and 1.0 in G
    assert 3 not in G and -1 not in G and "1" not in G and 1.5 not in G
    assert nx.single_source_shortest_path_length(G, True) == {True: 0, 0: 1, 2: 1}


def test_invalid_input():
    with pytest.raises(TypeError, match="numeric"):
        rustnx.Graph([(0, 1, {"color": "red"})])
    with pytest.raises(ValueError, match="NaN"):
        rustnx.Graph([(0, 1, float("nan"))])
    with pytest.raises(ValueError, match="2 or 3"):
        rustnx.Graph([(0, 1, 2, 3)])
    with pytest.raises(ValueError, match="None cannot be a node"):
        rustnx.Graph([(None, 1)])
    with pytest.raises(TypeError, match="strings"):
        rustnx.Graph([(0, 1, {5: 1})])


def test_unimplemented_functions_fall_back_with_enable():
    G = rustnx.Graph([(0, 1), (1, 2)])
    old = nx.config.backend_priority.algos, nx.config.fallback_to_nx
    try:
        nx.config.backend_priority.algos = []
        nx.config.fallback_to_nx = False
        with pytest.raises(NotImplementedError):
            nx.is_tree(G)
        rustnx.enable()
        assert nx.config.backend_priority.algos[0] == "rustnx"
        assert nx.is_tree(G) is True
    finally:
        nx.config.backend_priority.algos, nx.config.fallback_to_nx = old


def test_pickle_native():
    G = rustnx.DiGraph([("a", "b", 2), ("b", "c", {"weight": 1.5, "cap": 3})], kind="test")
    H = pickle.loads(pickle.dumps(G))
    assert H.graph == {"kind": "test"}
    assert H.edges(data=True) == G.edges(data=True)
    assert nx.closeness_centrality(H, distance="weight") == nx.closeness_centrality(
        G.to_networkx(), distance="weight", backend="networkx"
    )
    A = rustnx.Graph.from_arrays([0, 1], [1, 2])
    assert pickle.loads(pickle.dumps(A)).edges() == A.edges()


def test_graph_attributes_and_repr():
    G = rustnx.Graph([(1, 2)], name="g")
    assert G.graph == {"name": "g"}
    assert repr(G) == "<Graph (undirected) with 2 nodes and 1 edges>"
    assert G.has_edge(2, 1) and not G.has_edge(1, 3)
    with pytest.raises(nx.NetworkXError):
        G.successors(1)
