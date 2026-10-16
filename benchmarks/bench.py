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
    web = weighted(nx.gnm_random_graph(
        200_000 // scale, 1_000_000 // scale, seed=3, directed=True
    ))
    rng = random.Random(4)
    rank = {v: rng.random() for v in web}
    dag = nx.DiGraph((u, v) for u, v in web.edges if rank[u] < rank[v])
    dag.add_nodes_from(web)
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
        ("pagerank", web, lambda G, **kw: nx.pagerank(G, **kw)),
        ("pagerank (weighted)", web,
         lambda G, **kw: nx.pagerank(G, weight="weight", **kw)),
        ("strongly_connected_components", web,
         lambda G, **kw: list(nx.strongly_connected_components(G, **kw))),
        ("weakly_connected_components", web,
         lambda G, **kw: list(nx.weakly_connected_components(G, **kw))),
        ("topological_sort", dag,
         lambda G, **kw: list(nx.topological_sort(G, **kw))),
    ] + more_cases(scale, ba, ba_w, web)


def more_cases(scale, ba, ba_w, web):
    dist = weighted(nx.barabasi_albert_graph(5000 // scale, 3, seed=5))
    big = weighted(nx.gnm_random_graph(200_000 // scale, 1_000_000 // scale, seed=6))
    giant = list(max(nx.connected_components(big, backend="networkx"), key=len))
    rng = random.Random(7)
    pairs = [(rng.choice(giant), rng.choice(giant)) for _ in range(50)]
    ba100k = nx.barabasi_albert_graph(100_000 // scale, 5, seed=8)
    harmonic = nx.barabasi_albert_graph(3000 // scale, 4, seed=9)
    side = int((200_000 // scale) ** 0.5)
    grid = nx.grid_2d_graph(side, side)
    small_grid = nx.grid_2d_graph(60, 60)
    src = giant[0]
    return [
        ("edge_betweenness_centrality", ba, lambda G, **kw: nx.edge_betweenness_centrality(G, **kw)),
        ("edge_betweenness_centrality (weighted)", ba_w,
         lambda G, **kw: nx.edge_betweenness_centrality(G, weight="weight", **kw)),
        ("diameter", dist, lambda G, **kw: nx.diameter(G, **kw)),
        ("average_shortest_path_length", dist, lambda G, **kw: nx.average_shortest_path_length(G, **kw)),
        ("wiener_index", dist, lambda G, **kw: nx.wiener_index(G, **kw)),
        ("eccentricity (weighted)", dist, lambda G, **kw: nx.eccentricity(G, weight="weight", **kw)),
        ("all_pairs_dijkstra_path_length", dist,
         lambda G, **kw: dict(nx.all_pairs_dijkstra_path_length(G, **kw))),
        ("all_pairs_shortest_path_length", dist,
         lambda G, **kw: dict(nx.all_pairs_shortest_path_length(G, **kw))),
        ("dijkstra_path (50 pairs)", big,
         lambda G, **kw: [nx.dijkstra_path(G, s, t, **kw) for s, t in pairs]),
        ("single_source_dijkstra_path", big, lambda G, **kw: nx.single_source_dijkstra_path(G, src, **kw)),
        ("single_source_shortest_path", big, lambda G, **kw: nx.single_source_shortest_path(G, src, **kw)),
        ("clustering", ba100k, lambda G, **kw: nx.clustering(G, **kw)),
        ("transitivity", ba100k, lambda G, **kw: nx.transitivity(G, **kw)),
        ("triangles", ba100k, lambda G, **kw: nx.triangles(G, **kw)),
        ("katz_centrality", web, lambda G, **kw: nx.katz_centrality(G, alpha=0.05, **kw)),
        ("core_number", big, lambda G, **kw: nx.core_number(G, **kw)),
        ("eigenvector_centrality", big, lambda G, **kw: nx.eigenvector_centrality(G, **kw)),
        ("harmonic_centrality", harmonic, lambda G, **kw: nx.harmonic_centrality(G, **kw)),
        ("is_bipartite", grid, lambda G, **kw: nx.is_bipartite(G, **kw)),
        ("bfs_edges", big, lambda G, **kw: list(nx.bfs_edges(G, src, **kw))),
        ("dfs_edges", big, lambda G, **kw: list(nx.dfs_edges(G, src, **kw))),
        ("bidirectional_dijkstra (20 pairs)", big,
         lambda G, **kw: [nx.bidirectional_dijkstra(G, s, t, **kw) for s, t in pairs[:20]]),
        ("k_core", big, lambda G, **kw: nx.k_core(G, **kw)),
        ("label_propagation_communities", ba100k,
         lambda G, **kw: list(nx.community.label_propagation_communities(G, **kw))),
        ("greedy_color", ba100k, lambda G, **kw: nx.greedy_color(G, **kw)),
        ("minimum_spanning_tree", big, lambda G, **kw: nx.minimum_spanning_tree(G, **kw)),
        ("bfs_tree", big, lambda G, **kw: nx.bfs_tree(G, src, **kw)),
        ("all_shortest_paths (2.7M paths)", small_grid,
         lambda G, **kw: list(nx.all_shortest_paths(G, (0, 0), (12, 12), **kw))),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smaller graphs")
    parser.add_argument("--only", nargs="*", help="run the cases whose name starts with one of these")
    args = parser.parse_args()

    print(f"CPU cores: {os.cpu_count()}\n")
    header = f"{'function':40s} {'graph':>18s} {'networkx':>10s} {'rustnx cold':>12s} {'rustnx warm':>12s} {'speedup (warm)':>15s}"
    print(header)
    print("-" * len(header))
    for name, G, call in cases(args.quick):
        if args.only and not any(name.startswith(o) for o in args.only):
            continue
        repeat = 3
        slow = "centrality" in name or "all_pairs" in name or name in (
            "diameter", "average_shortest_path_length", "wiener_index", "eccentricity (weighted)")
        t_nx = timed(lambda: call(G, backend="networkx"), 1 if slow else repeat)

        def cold():
            G.__networkx_cache__.clear()
            call(G, backend="rustnx")

        t_cold = timed(cold, repeat)
        call(G, backend="rustnx")  # populate cache
        t_warm = timed(lambda: call(G, backend="rustnx"), repeat)
        size = f"{G.number_of_nodes()}n/{G.number_of_edges()}e"
        print(
            f"{name:40s} {size:>18s} {t_nx:9.3f}s {t_cold:11.3f}s {t_warm:11.4f}s "
            f"{t_nx / t_warm:14.0f}x"
        )


if __name__ == "__main__":
    main()
