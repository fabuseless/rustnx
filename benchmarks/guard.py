"""Speed and dispatch guard, run in CI on every pull request.

    python benchmarks/guard.py

Tests check that rustnx gives the right answers; this checks that it is
still fast, and that calls really run in Rust. CI machines vary a lot in
speed, so it compares rustnx against NetworkX on the same machine and checks
each speedup against a floor set well below today's numbers. It fails on
large slowdowns and silent fallbacks to NetworkX, not on noise.

With GITHUB_STEP_SUMMARY set (in GitHub Actions), the results table is also
written to the job summary.
"""

import contextlib
import os
import random
import sys
import time

import networkx as nx

import rustnx
from rustnx import interface

nx.config.warnings_to_ignore.add("cache")


def weighted(G, seed=0):
    rng = random.Random(seed)
    for _, _, d in G.edges(data=True):
        d["weight"] = rng.randint(1, 10)
    return G


def timed(func, repeat):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - start)
    return best


# (name, graph builder, call, minimum speedup over NetworkX).
# Floors sit at roughly a quarter of the speedups measured on a 4-core
# machine, so slower CI runners and noise don't trip them.
def cases():
    ba = nx.barabasi_albert_graph(800, 4, seed=1)
    ba_big = nx.barabasi_albert_graph(20_000, 5, seed=2)
    gnm = weighted(nx.gnm_random_graph(100_000, 400_000, seed=3))
    gnm_d = nx.gnm_random_graph(20_000, 100_000, seed=4, directed=True)
    gnm_d_big = nx.gnm_random_graph(50_000, 250_000, seed=5, directed=True)
    grid = nx.convert_node_labels_to_integers(nx.grid_2d_graph(40, 40))
    return [
        ("betweenness_centrality", ba, lambda G, b: nx.betweenness_centrality(G, backend=b), 15),
        ("closeness_centrality", ba, lambda G, b: nx.closeness_centrality(G, backend=b), 30),
        ("harmonic_centrality", ba, lambda G, b: nx.harmonic_centrality(G, backend=b), 5),
        ("pagerank", gnm_d, lambda G, b: nx.pagerank(G, backend=b), 20),
        ("katz_centrality", gnm_d, lambda G, b: nx.katz_centrality(G, alpha=0.05, backend=b), 20),
        ("eigenvector_centrality", ba_big, lambda G, b: nx.eigenvector_centrality(G, max_iter=1000, backend=b), 10),
        ("clustering", ba_big, lambda G, b: nx.clustering(G, backend=b), 10),
        ("clustering (weighted)", gnm, lambda G, b: nx.clustering(G, weight="weight", backend=b), 20),
        ("normalized_laplacian_matrix", gnm, lambda G, b: nx.normalized_laplacian_matrix(G, backend=b), 2),
        ("core_number", ba_big, lambda G, b: nx.core_number(G, backend=b), 10),
        ("diameter", grid, lambda G, b: nx.diameter(G, backend=b), 40),
        ("single_source_dijkstra_path_length", gnm, lambda G, b: nx.single_source_dijkstra_path_length(G, 0, backend=b), 2),
        ("connected_components", gnm, lambda G, b: list(nx.connected_components(G, backend=b)), 2),
        ("strongly_connected_components", gnm_d_big, lambda G, b: list(nx.strongly_connected_components(G, backend=b)), 5),
    ]


@contextlib.contextmanager
def counting(name):
    """Count calls that NetworkX dispatches to rustnx's ``name``."""
    calls = []
    original = getattr(interface, name)

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    setattr(interface, name, counted)
    try:
        yield calls
    finally:
        setattr(interface, name, original)


@contextlib.contextmanager
def priority(algos, fallback=False):
    old = nx.config.backend_priority.algos, nx.config.fallback_to_nx
    nx.config.backend_priority.algos = algos
    nx.config.fallback_to_nx = fallback
    try:
        yield
    finally:
        nx.config.backend_priority.algos, nx.config.fallback_to_nx = old


def dispatch_checks():
    """Each check returns whether the call ran in rustnx."""
    big = nx.gnm_random_graph(2000, 8000, seed=6)
    weighted(big)

    def enable_runs_rustnx():
        old = nx.config.backend_priority.algos, nx.config.fallback_to_nx
        try:
            rustnx.enable()
            with counting("pagerank") as calls:
                nx.pagerank(big)
            return bool(calls)
        finally:
            nx.config.backend_priority.algos, nx.config.fallback_to_nx = old

    def priority_runs_rustnx():
        with priority(["rustnx"]), counting("betweenness_centrality") as calls:
            nx.betweenness_centrality(big, k=50, seed=1)
        return bool(calls)

    def multigraph_runs_rustnx():
        M = nx.MultiGraph(big)
        M.add_edges_from((u, v, {"weight": 1}) for u, v in list(big.edges)[:500])
        with priority(["rustnx"]), counting("dijkstra_path_length") as calls:
            nx.dijkstra_path_length(M, 0, 1999)
        return bool(calls)

    def weights_after_all_attr_conversion():
        G = big.copy()
        for _, _, d in G.edges(data=True):
            d["label"] = "text"  # forces an all-attributes conversion below
        with priority(["rustnx"]):
            nx.minimum_spanning_tree(G)  # caches a conversion without weights
            # The cached conversion must not make this fall back.
            with counting("single_source_dijkstra_path_length") as calls:
                result = nx.single_source_dijkstra_path_length(G, 0)
        ref = nx.single_source_dijkstra_path_length(G, 0, backend="networkx")
        return bool(calls) and result == ref

    def native_runs_rustnx():
        N = rustnx.DiGraph(list(big.edges(data="weight")))
        with counting("pagerank") as calls:
            nx.pagerank(N)
        return bool(calls)

    return [
        ("rustnx.enable() runs rustnx on NetworkX graphs", enable_runs_rustnx),
        ('backend_priority = ["rustnx"] runs rustnx', priority_runs_rustnx),
        ("multigraphs run rustnx for supported functions", multigraph_runs_rustnx),
        ("weights load after an all-attributes conversion", weights_after_all_attr_conversion),
        ("native graphs run rustnx", native_runs_rustnx),
    ]


def main():
    rows, failures = [], []
    for name, G, call, floor in cases():
        call(G, "rustnx")  # warm: convert once, as repeated calls do
        nx_time = timed(lambda: call(G, "networkx"), 1)
        rx_time = timed(lambda: call(G, "rustnx"), 3)
        speedup = nx_time / rx_time
        ok = speedup >= floor
        rows.append(f"| `{name}` | {nx_time:.3f} s | {rx_time:.4f} s | {speedup:.0f}x | {floor}x | {'ok' if ok else 'FAIL'} |")
        if not ok:
            failures.append(f"{name}: {speedup:.1f}x is below the {floor}x floor")

    checks = []
    for label, check in dispatch_checks():
        try:
            ok = check()
        except Exception as exc:  # report, don't hide
            ok = False
            label += f" ({type(exc).__name__}: {exc})"
        checks.append(f"| {label} | {'ok' if ok else 'FAIL'} |")
        if not ok:
            failures.append(f"dispatch: {label}")

    report = "\n".join(
        [
            f"## rustnx speed guard (NetworkX {nx.__version__}, {os.cpu_count()} CPUs)",
            "",
            "| Function | NetworkX | rustnx (warm) | Speedup | Floor | |",
            "|---|---|---|---|---|---|",
            *rows,
            "",
            "| Dispatch check | |",
            "|---|---|",
            *checks,
            "",
        ]
    )
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(report + "\n")
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("All speed floors and dispatch checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
