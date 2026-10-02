"""Compare rustnx with other Rust-backed NetworkX backends.

    pip install nx-rustworkx franken-networkx
    python benchmarks/compare.py [--quick]

Each backend is called through NetworkX with an explicit ``backend=``, so all
of them get the same inputs. Every result is checked against NetworkX's:
"same" means identical output (values and order), "close" means equal to
within 1e-9, and "DIFFERENT" means it disagrees with NetworkX.

"cold" includes converting the NetworkX graph; "warm" reuses the conversion
NetworkX caches on the graph.
"""

import argparse
import math
import random
import time

import networkx as nx

nx.config.warnings_to_ignore.add("cache")

BACKENDS = ["rustnx", "rustworkx", "franken_networkx"]


def weighted(G, seed=0):
    rng = random.Random(seed)
    for _, _, d in G.edges(data=True):
        d["weight"] = rng.randint(1, 10)
    return G


def graphs(quick):
    s = 4 if quick else 1
    ba = nx.barabasi_albert_graph(2000 // s, 4, seed=1)
    grid = weighted(nx.convert_node_labels_to_integers(nx.grid_2d_graph(300 // s, 300 // s)))
    sparse = nx.gnm_random_graph(200_000 // s, 300_000 // s, seed=2)
    web = weighted(nx.gnm_random_graph(100_000 // s, 500_000 // s, seed=3, directed=True))
    rng = random.Random(4)
    rank = {v: rng.random() for v in web}
    dag = nx.DiGraph((u, v) for u, v in web.edges if rank[u] < rank[v])
    dag.add_nodes_from(web)
    return [
        ("betweenness_centrality", ba, lambda G, b: nx.betweenness_centrality(G, backend=b)),
        ("closeness_centrality", ba, lambda G, b: nx.closeness_centrality(G, backend=b)),
        ("pagerank", web, lambda G, b: nx.pagerank(G, backend=b)),
        ("single_source_dijkstra_path_length", grid,
         lambda G, b: nx.single_source_dijkstra_path_length(G, 0, backend=b)),
        ("connected_components", sparse,
         lambda G, b: list(nx.connected_components(G, backend=b))),
        ("strongly_connected_components", web,
         lambda G, b: list(nx.strongly_connected_components(G, backend=b))),
        ("topological_sort", dag, lambda G, b: list(nx.topological_sort(G, backend=b))),
    ]


def compare(result, reference):
    if result == reference:
        return "same"
    if isinstance(reference, dict) and isinstance(result, dict):
        if set(result) == set(reference) and all(
            math.isclose(result[k], reference[k], rel_tol=1e-9, abs_tol=1e-12)
            for k in reference
        ):
            return "close" if list(result) == list(reference) else "close (order differs)"
    if isinstance(reference, list) and isinstance(result, list):
        as_set = lambda xs: {frozenset(x) if isinstance(x, set) else x for x in xs}
        if as_set(result) == as_set(reference) and len(result) == len(reference):
            return "same contents, different order"
    return "DIFFERENT"


def timed(func):
    start = time.perf_counter()
    result = func()
    return time.perf_counter() - start, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smaller graphs")
    args = parser.parse_args()
    installed = [b for b in BACKENDS if b in nx.utils.backends.backends]

    for name, G, call in graphs(args.quick):
        print(f"\n{name}  ({G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges)")
        t_nx, reference = timed(lambda: call(G, "networkx"))
        print(f"  {'networkx':18s} {t_nx:9.3f}s")
        for backend in installed:
            G.__networkx_cache__.clear()
            try:
                t_cold, result = timed(lambda: call(G, backend))
                t_warm, _ = timed(lambda: call(G, backend))
            except Exception as exc:  # not implemented, or an error
                msg = str(exc).splitlines()[0][:70] if str(exc) else ""
                print(f"  {backend:18s} {'n/a':>10s}  {type(exc).__name__}: {msg}")
                continue
            print(
                f"  {backend:18s} {t_cold:9.3f}s cold {t_warm:9.4f}s warm "
                f"{t_nx / t_warm:7.0f}x  {compare(result, reference)}"
            )


if __name__ == "__main__":
    main()
