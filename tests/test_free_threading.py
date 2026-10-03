"""Concurrency checks for free-threaded Python (3.13t/3.14t builds).

Runs algorithms from many threads at once on shared graphs, including the
lazy parts (conversion caching, exact predecessor loading, on-demand weight
conversion), and compares every result with a single-threaded run.
"""

import random
import sys
import sysconfig
import threading

import networkx as nx
import pytest

import rustnx

pytestmark = pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"),
    reason="needs a free-threaded Python build",
)


def test_gil_stays_disabled():
    import rustnx._core  # noqa: F401

    assert not sys._is_gil_enabled()


def test_concurrent_algorithms_match_sequential():
    rng = random.Random(0)
    G = nx.gnm_random_graph(600, 2400, seed=1, directed=True)
    for u, v, d in G.edges(data=True):
        d["weight"] = rng.randint(1, 9)
        d["other"] = rng.choice([0.5, 1.5, 2.5])
    U = nx.gnm_random_graph(600, 2400, seed=2)
    N = rustnx.DiGraph(list(G.edges(data="weight")))
    calls = [
        lambda: nx.pagerank(G, backend="rustnx"),
        lambda: nx.betweenness_centrality(G, k=50, seed=1, weight="weight", backend="rustnx"),
        lambda: nx.single_source_dijkstra_path_length(G, 0, backend="rustnx"),
        lambda: nx.single_source_dijkstra_path_length(G, 0, weight="other", backend="rustnx"),
        lambda: nx.shortest_path(G, target=5, weight="weight", backend="rustnx"),
        lambda: nx.bidirectional_dijkstra(G, 0, 7, backend="rustnx"),
        lambda: [sorted(c) for c in nx.strongly_connected_components(G, backend="rustnx")],
        lambda: nx.closeness_centrality(U, backend="rustnx"),
        lambda: nx.clustering(U, backend="rustnx"),
        lambda: list(nx.bfs_edges(U, 0, backend="rustnx")),
        lambda: list(nx.all_shortest_paths(G, 0, 9, weight="weight", backend="rustnx")),
        lambda: nx.pagerank(N, backend="rustnx"),
    ]
    expected = [f() for f in calls]
    failures = []
    lock = threading.Lock()

    def worker(seed):
        r = random.Random(seed)
        for _ in range(25):
            i = r.randrange(len(calls))
            try:
                ok = calls[i]() == expected[i]
            except Exception as exc:  # pragma: no cover - reported below
                ok = repr(exc)
            if ok is not True:
                with lock:
                    failures.append((i, ok))

    for round_ in range(3):
        # Drop cached conversions so conversion and lazy loading race too.
        G.__networkx_cache__.clear()
        U.__networkx_cache__.clear()
        threads = [threading.Thread(target=worker, args=(round_ * 100 + t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert failures == []
