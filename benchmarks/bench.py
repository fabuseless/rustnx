"""Benchmark rustnx against NetworkX.

    python benchmarks/bench.py [--quick]

"cold" includes converting the NetworkX graph to rustnx's format; "warm"
reuses the conversion NetworkX caches on the graph (the normal case when
calling several algorithms on the same graph).
"""

import argparse
import os
import random
import time

import networkx as nx

nx.config.warnings_to_ignore.add("cache")


def timed(func, repeat):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - start)
    return best


def weighted(G, seed=0):
    rng = random.Random(seed)
    for _, _, d in G.edges(data=True):
        d["weight"] = rng.randint(1, 10)
    return G


def cases(quick):
    scale = 4 if quick else 1
    ba = nx.barabasi_albert_graph(4000 // scale, 4, seed=1)
    ba_w = weighted(ba.copy())
    grid = weighted(nx.convert_node_labels_to_integers(
        nx.grid_2d_graph(400 // scale, 400 // scale)
    ))
    sparse = nx.gnm_random_graph(200_000 // scale, 300_000 // scale, seed=2)
    return [
        ("betweenness_centrality", ba, lambda G, **kw: nx.betweenness_centrality(G, **kw)),
        ("betweenness_centrality (weighted)", ba_w,
         lambda G, **kw: nx.betweenness_centrality(G, weight="weight", **kw)),
        ("closeness_centrality", ba, lambda G, **kw: nx.closeness_centrality(G, **kw)),
        ("single_source_dijkstra_path_length", grid,
         lambda G, **kw: nx.single_source_dijkstra_path_length(G, 0, **kw)),
        ("single_source_shortest_path_length", grid,
         lambda G, **kw: nx.single_source_shortest_path_length(G, 0, **kw)),
        ("connected_components", sparse,
         lambda G, **kw: list(nx.connected_components(G, **kw))),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smaller graphs")
    args = parser.parse_args()

    print(f"CPU cores: {os.cpu_count()}\n")
    header = f"{'function':38s} {'graph':>18s} {'networkx':>10s} {'rustnx cold':>12s} {'rustnx warm':>12s} {'speedup (warm)':>15s}"
    print(header)
    print("-" * len(header))
    for name, G, call in cases(args.quick):
        repeat = 3
        t_nx = timed(lambda: call(G, backend="networkx"), 1 if "centrality" in name else repeat)

        def cold():
            G.__networkx_cache__.clear()
            call(G, backend="rustnx")

        t_cold = timed(cold, repeat)
        call(G, backend="rustnx")  # populate cache
        t_warm = timed(lambda: call(G, backend="rustnx"), repeat)
        size = f"{G.number_of_nodes()}n/{G.number_of_edges()}e"
        print(
            f"{name:38s} {size:>18s} {t_nx:9.3f}s {t_cold:11.3f}s {t_warm:11.4f}s "
            f"{t_nx / t_warm:14.0f}x"
        )


if __name__ == "__main__":
    main()
