"""The smaller measured tables in the README.

    python benchmarks/readme_numbers.py [conversion] [native] [floats] [rx]

conversion: converting a NetworkX graph to rustnx against walking it once in
            pure Python (200,000 nodes, 1M edges).
native:     building a 1M-edge directed graph as networkx.DiGraph and as
            rustnx.DiGraph (from edges and from arrays): time, memory (RSS
            growth, measured in a fresh process each) and a first pagerank.
floats:     the fast-float functions in their fast and exact modes.
rx:         rustnx.rx against rustworkx (needs rustworkx installed).
"""

import json
import os
import random
import subprocess
import sys
import time
import warnings

import networkx as nx

nx.config.warnings_to_ignore.add("cache")
warnings.simplefilter("ignore")


def best(f, repeat=3):
    out = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        out = min(out, time.perf_counter() - t)
    return out


def weighted(G, seed=0, floats=False):
    rng = random.Random(seed)
    for _, _, d in G.edges(data=True):
        d["weight"] = rng.random() if floats else rng.randint(1, 10)
    return G


def conversion():
    from rustnx import interface

    rows = []
    for label, directed, w in (("Undirected, no weights", False, False),
                               ("Undirected, with weights", False, True),
                               ("Directed, with weights", True, True)):
        G = nx.gnm_random_graph(200_000, 1_000_000, seed=1, directed=directed)
        if w:
            weighted(G)
        attrs = {"weight": 1} if w else None

        def convert():
            interface.convert_from_nx(G, edge_attrs=attrs, name="pagerank")

        def walk():
            if w:
                for nbrs in G._adj.values():
                    for d in nbrs.values():
                        d.get("weight", 1)
            else:
                for nbrs in G._adj.values():
                    for _ in nbrs:
                        pass

        rows.append((label, best(convert), best(walk)))
    print("| 200,000 nodes / 1M edges | Converting to rustnx | Just walking the graph in pure Python |")
    print("|---|---|---|")
    for label, c, w in rows:
        print(f"| {label} | {c:.2f} s | {w:.2f} s |")


_NATIVE_CHILD = r"""
import gc, json, sys, time, random
import networkx as nx, numpy as np
import rustnx
nx.config.warnings_to_ignore.add("cache")
rustnx.enable()
kind = sys.argv[1]
rng = np.random.default_rng(1)
n, m = 200_000, 1_000_000
src = rng.integers(0, n, m); dst = rng.integers(0, n, m); wts = rng.random(m)
edges = list(zip(src.tolist(), dst.tolist(), wts.tolist()))
gc.collect()
def rss():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096
before = rss()
t = time.perf_counter()
if kind == "networkx":
    G = nx.DiGraph(); G.add_weighted_edges_from(edges)
elif kind == "edges":
    G = rustnx.DiGraph(edges)
else:
    G = rustnx.DiGraph.from_arrays(src, dst, wts, num_nodes=n)
build = time.perf_counter() - t
gc.collect()
mem = rss() - before
t = time.perf_counter()
nx.pagerank(G)
pr = time.perf_counter() - t
print(json.dumps({"build": build, "mem": mem, "pagerank": pr}))
"""


def native():
    res = {}
    for kind in ("networkx", "edges", "arrays"):
        p = subprocess.run([sys.executable, "-c", _NATIVE_CHILD, kind], capture_output=True, text=True)
        if p.returncode:
            print(p.stderr[-2000:])
            return
        res[kind] = json.loads(p.stdout.strip().splitlines()[-1])
    print("| 1M-edge directed graph | `networkx.DiGraph` | `rustnx.DiGraph(edges)` | `rustnx.DiGraph.from_arrays` |")
    print("|---|---|---|---|")
    print("| Build time | " + " | ".join(f"{res[k]['build']:.2f} s" for k in res) + " |")
    print("| Memory | " + " | ".join(f"{res[k]['mem'] / 2**20:.0f} MiB" for k in res) + " |")
    print("| `pagerank`, first call | " + " | ".join(f"{res[k]['pagerank']:.3f} s" for k in res) + " |")


def floats():
    cfg = nx.config.backends.rustnx
    ba = nx.barabasi_albert_graph(5000, 3, seed=1)
    web = nx.gnm_random_graph(200_000, 1_000_000, seed=3, directed=True)
    wc = weighted(nx.gnm_random_graph(20_000, 200_000, seed=4), floats=True)
    cases = [
        ("betweenness_centrality", "5,000 nodes", lambda b: nx.betweenness_centrality(ba, backend=b)),
        ("edge_betweenness_centrality", "5,000 nodes", lambda b: nx.edge_betweenness_centrality(ba, backend=b)),
        ("pagerank", "200,000 nodes", lambda b: nx.pagerank(web, backend=b)),
        ("clustering (weighted)", "20,000 nodes, 200,000 edges", lambda b: nx.clustering(wc, weight="weight", backend=b)),
    ]
    print("| Function | Graph | Fast | Exact | NetworkX |")
    print("|---|---|---|---|---|")
    for name, graph, f in cases:
        f("rustnx")
        cfg.exact_floats = False
        fast = best(lambda: f("rustnx"))
        cfg.exact_floats = True
        try:
            exact = best(lambda: f("rustnx"))
        except NotImplementedError:
            exact = None  # runs in NetworkX
        cfg.exact_floats = False
        ref = best(lambda: f("networkx"), 1)
        ex = f"{exact:.3f} s" if exact is not None else "(NetworkX)"
        print(f"| {name} | {graph} | {fast:.3f} s | {ex} | {ref:.2f} s |")


def rx():
    import rustworkx

    import rustnx.rx as rnx

    def build(mod, directed, n, edges):
        g = mod.PyDiGraph() if directed else mod.PyGraph()
        g.add_nodes_from(range(n))
        g.extend_from_edge_list(edges)
        return g

    ba = list(nx.barabasi_albert_graph(2000, 4, seed=1).edges)
    dg = list(nx.gnm_random_graph(100_000, 500_000, seed=2, directed=True).edges)
    ga_r, ga_o = build(rnx, False, 2000, ba), build(rustworkx, False, 2000, ba)
    gd_r, gd_o = build(rnx, True, 100_000, dg), build(rustworkx, True, 100_000, dg)
    print("| Call | rustnx.rx | rustworkx |")
    print("|---|---|---|")
    for label, f, gr, go in (
        ("betweenness_centrality (2k nodes)", lambda m, g: m.betweenness_centrality(g), ga_r, ga_o),
        ("closeness_centrality (2k nodes)", lambda m, g: m.closeness_centrality(g), ga_r, ga_o),
        ("strongly_connected_components (100k nodes)", lambda m, g: m.strongly_connected_components(g), gd_r, gd_o),
    ):
        print(f"| `{label}` | {best(lambda: f(rnx, gr)):.4f} s | {best(lambda: f(rustworkx, go)):.4f} s |")
    print(f"| build, 500k edges | {best(lambda: build(rnx, True, 100_000, dg)):.3f} s | "
          f"{best(lambda: build(rustworkx, True, 100_000, dg)):.3f} s |")


def main():
    which = sys.argv[1:] or ["conversion", "native", "floats", "rx"]
    for w in which:
        print(f"\n## {w}")
        globals()[w]()


if __name__ == "__main__":
    main()
