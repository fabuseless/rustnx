"""Generators (todo item 49) and I/O (item 50) added in the generators/I/O
round: each must give exactly NetworkX's graph, including node and
adjacency order, key types and graph attributes, and advance the random
generator exactly as NetworkX does."""

import random
import warnings

import networkx as nx
import pytest

np = pytest.importorskip("numpy")


def snapshot(G):
    def key(x):
        return (type(x).__name__, x)

    multi = G.is_multigraph()
    edges = list(G.edges(keys=True, data=True) if multi else G.edges(data=True))
    adj = [(key(u), [key(v) for v in G._adj[u]]) for u in G]
    pred = [(key(u), [key(v) for v in G._pred[u]]) for u in G] if G.is_directed() else None
    return type(G).__name__, G.graph, list(G.nodes(data=True)), edges, adj, pred


def outcome(func, *args, **kwargs):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = func(*args, **kwargs)
    except Exception as exc:  # compare errors too
        return type(exc).__name__, str(exc)
    if isinstance(result, list):
        return "ok", [snapshot(g) for g in result]
    return "ok", snapshot(result) if isinstance(result, nx.Graph) else result


def same(make_call):
    """Both backends from identical fresh arguments; rustnx may decline."""
    ref = outcome(*make_call("networkx"))
    ours = outcome(*make_call("rustnx"))
    if ours[0] == "NotImplementedError":
        return False
    assert ours == ref
    return True


def seeds(seed):
    out = [lambda: seed]
    if nx.__version__ >= "3.5":
        out.append(lambda: np.random.RandomState(seed))
    else:
        out.append(lambda: random.Random(seed))
    return out


@pytest.mark.parametrize("seed", range(15))
def test_random_k_out_graph(seed):
    ran = 0
    for n, k, alpha, self_loops in [(20, 3, 1.0, True), (20, 3, 0.5, False), (15, 4, 2, True),
                                    (30, 2, 0.0, True), (10, 1, 1, False), (1, 1, 1.0, False),
                                    (12, 12, 0.25, True), (5, 2, -1.0, True)]:
        for make_seed in seeds(seed):
            ran += same(lambda b: (lambda: nx.random_k_out_graph(
                n, k, alpha, self_loops=self_loops, seed=make_seed(), backend=b),))
    assert ran >= 8


def test_random_k_out_graph_advances_the_generator():
    if nx.__version__ >= "3.5":
        a, b = np.random.RandomState(5), np.random.RandomState(5)
        draw = lambda s: s.random_sample()  # noqa: E731
    else:
        a, b = random.Random(5), random.Random(5)
        draw = lambda s: s.random()  # noqa: E731
    nx.random_k_out_graph(30, 3, 1.0, seed=a, backend="networkx")
    nx.random_k_out_graph(30, 3, 1.0, seed=b, backend="rustnx")
    assert draw(a) == draw(b)


@pytest.mark.parametrize("i", list(range(0, 1253, 31)) + [1252, -1, 1253])
def test_graph_atlas(i):
    assert same(lambda b: (lambda: nx.graph_atlas(i, backend=b),))


def test_graph_atlas_g():
    assert same(lambda b: (lambda: nx.graph_atlas_g(backend=b),))


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("n", [0, 1, 2, 7, 30])
def test_random_unlabeled_trees(seed, n):
    for func, extra in [(nx.random_unlabeled_tree, {"number_of_trees": 3}),
                        (nx.random_unlabeled_rooted_tree, {"number_of_trees": 3}),
                        (nx.random_unlabeled_rooted_forest, {"number_of_forests": 3, "q": 4})]:
        for kwargs in ({}, extra):
            assert same(lambda b: (lambda: func(n, seed=random.Random(seed), backend=b, **kwargs),))


def test_tree_counts_match_networkx():
    from networkx.generators import trees

    from rustnx import algorithms

    counts = [0, 1]
    trees._num_rooted_trees(200, counts)
    assert counts == [algorithms._num_rooted_trees_cached(i, [0, 1]) for i in range(201)]
    for q in (1, 3, 50, 200):
        forests = [1]
        trees._num_rooted_forests(200, q, forests)
        assert forests == [algorithms._num_rooted_forests_cached(i, q, [1]) for i in range(201)]
