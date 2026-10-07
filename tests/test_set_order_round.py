"""Set-order round: functions whose NetworkX code depends on Python's set
order run as hybrids (NetworkX's own code makes every set-order choice),
with their expensive inner work in Rust. Results must be exactly
NetworkX's, in the same order."""

import itertools
import random
import warnings

import networkx as nx
import pytest

from rustnx import algorithms


def typed(cycle):
    return [(type(x).__name__, repr(x)) for x in cycle]


def outcome(func, G, **kwargs):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return "ok", [typed(c) for c in itertools.islice(func(G, **kwargs), 20000)]
    except Exception as exc:
        return type(exc).__name__, str(exc)


def test_rust_searches_are_used():
    assert algorithms._CYCLE_HELPERS_REGISTERED == {
        "_johnson_cycle_search", "_bounded_cycle_search", "_chordless_cycle_search",
    }


def _graphs(seed):
    rnd = random.Random(seed)
    n = rnd.randint(1, 13)
    p = rnd.random() * 0.6
    for cls in (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph):
        G = cls(nx.gnp_random_graph(n, p, seed=seed, directed=cls().is_directed()))
        if rnd.random() < 0.3:
            G = nx.relabel_nodes(G, {v: (f"s{v}" if v % 2 else v * 1000) for v in G})
        for v in list(G)[:2]:
            if rnd.random() < 0.3:
                G.add_edge(v, v)
        if G.is_multigraph():
            for u, v in list(G.edges())[:3]:
                G.add_edge(u, v)
        yield G


@pytest.mark.parametrize("seed", range(60))
@pytest.mark.parametrize("name", ["simple_cycles", "chordless_cycles"])
def test_cycles(seed, name):
    func = getattr(nx, name)
    for G in _graphs(seed):
        for kwargs in ({}, {"length_bound": 1}, {"length_bound": 2}, {"length_bound": 3},
                       {"length_bound": 5}, {"length_bound": -1}):
            ref = outcome(lambda G, **k: func(G, backend="networkx", **k), G, **kwargs)
            ours = outcome(lambda G, **k: func(G, backend="rustnx", **k), G, **kwargs)
            assert ours == ref


@pytest.mark.parametrize("case", range(5))
def test_cycles_many(case):
    func, G, kwargs = [
        (nx.simple_cycles, nx.gnp_random_graph(13, 0.4, seed=3, directed=True), {}),
        (nx.simple_cycles, nx.gnp_random_graph(13, 0.35, seed=4), {}),
        (nx.simple_cycles, nx.gnp_random_graph(30, 0.2, seed=3, directed=True), {"length_bound": 5}),
        (nx.chordless_cycles, nx.gnp_random_graph(35, 0.12, seed=3), {}),
        (nx.chordless_cycles, nx.gnp_random_graph(30, 0.2, seed=3, directed=True), {"length_bound": 6}),
    ][case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert list(func(G, backend="rustnx", **kwargs)) == list(func(G, backend="networkx", **kwargs))


def test_cycles_are_lazy():
    G = nx.complete_graph(12, create_using=nx.DiGraph)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = list(itertools.islice(nx.simple_cycles(G, backend="rustnx"), 5))
        ref = list(itertools.islice(nx.simple_cycles(G, backend="networkx"), 5))
    assert ours == ref


def _multi_snapshot(G):
    return list(G.nodes), list(G.edges(keys=True, data=True)), G.graph


@pytest.mark.parametrize("seed", range(8))
def test_eulerize(seed):
    for G in (nx.connected_watts_strogatz_graph(40 + 10 * seed, 4, 0.3, seed=seed),
              nx.path_graph(5), nx.cycle_graph(6), nx.MultiGraph([(0, 1), (0, 1), (1, 2)]),
              nx.empty_graph(3), nx.Graph()):
        def run(backend):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return "ok", _multi_snapshot(nx.eulerize(G, backend=backend))
            except Exception as exc:
                return type(exc).__name__, str(exc)
        assert run("rustnx") == run("networkx")


@pytest.mark.parametrize("name", ["harmonic_function", "local_and_global_consistency"])
@pytest.mark.parametrize("seed", range(4))
def test_node_classification(name, seed):
    pytest.importorskip("scipy")
    from networkx.algorithms import node_classification

    func = getattr(node_classification, name)
    G = nx.gnm_random_graph(300, 1200, seed=seed)
    rnd = random.Random(seed)
    for v in rnd.sample(list(G), 20):
        G.nodes[v]["label"] = rnd.choice(["a", "b", 3])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert func(G, backend="rustnx") == func(G, backend="networkx")
        assert func(G, max_iter=5, backend="rustnx") == func(G, max_iter=5, backend="networkx")
