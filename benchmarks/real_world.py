"""Real-world analysis pipelines, run with NetworkX and with rustnx.

    python benchmarks/real_world.py --data DIR [--only NAME ...] [--out results.json]

DIR holds SNAP edge lists (https://snap.stanford.edu/data/), gzipped as
downloaded: facebook_combined, email-Enron, ca-AstroPh, soc-Epinions1,
cit-HepPh, roadNet-PA and web-Google (``<name>.txt.gz``).

Each pipeline is a sequence of analysis steps on one graph, as a script
would run them. Every step runs once with ``backend="networkx"`` and once
with ``backend="rustnx"``; rustnx's first step includes converting the graph
(NetworkX caches the conversion for the later steps). Results are checked
against NetworkX's ("same", "close" within 1e-9 relative, or "DIFFERENT").
"""

import argparse
import gzip
import itertools
import json
import os
import random
import sys
import time
import warnings

import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from all_functions import close, materialize, norm  # noqa: E402

nx.config.warnings_to_ignore.add("cache")
warnings.simplefilter("ignore")


def load(data, name, directed):
    G = nx.DiGraph() if directed else nx.Graph()
    with gzip.open(os.path.join(data, f"{name}.txt.gz"), "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            u, v = line.split()
            u, v = int(u), int(v)
            if u != v:
                G.add_edge(u, v)
    return G


def with_weights(G, lo=1, hi=100, seed=0):
    rng = random.Random(seed)
    for _, _, d in G.edges(data=True):
        d["weight"] = rng.randint(lo, hi)
    return G


def pairs(G, k, seed):
    rng = random.Random(seed)
    V = list(G)
    return [(rng.choice(V), rng.choice(V)) for _ in range(k)]


def largest_cc(G):
    return G.subgraph(max(nx.connected_components(G, backend="networkx"), key=len)).copy()


# Each pipeline: (dataset, directed, prepare, steps). A step is (label,
# function of (G, backend)).
def social():
    return "facebook_combined", False, None, [
        ("degree_centrality", lambda G, b: nx.degree_centrality(G, backend=b)),
        ("betweenness_centrality", lambda G, b: nx.betweenness_centrality(G, backend=b)),
        ("closeness_centrality", lambda G, b: nx.closeness_centrality(G, backend=b)),
        ("eigenvector_centrality", lambda G, b: nx.eigenvector_centrality(G, backend=b)),
        ("pagerank", lambda G, b: nx.pagerank(G, backend=b)),
        ("clustering", lambda G, b: nx.clustering(G, backend=b)),
        ("average_clustering", lambda G, b: nx.average_clustering(G, backend=b)),
        ("transitivity", lambda G, b: nx.transitivity(G, backend=b)),
        ("louvain_communities", lambda G, b: nx.community.louvain_communities(G, seed=1, backend=b)),
        ("label_propagation_communities", lambda G, b: list(nx.community.label_propagation_communities(G, backend=b))),
        ("core_number", lambda G, b: nx.core_number(G, backend=b)),
        ("diameter", lambda G, b: nx.diameter(G, backend=b)),
        ("average_shortest_path_length", lambda G, b: nx.average_shortest_path_length(G, backend=b)),
    ]


def email():
    return "email-Enron", False, None, [
        ("connected_components", lambda G, b: list(nx.connected_components(G, backend=b))),
        ("core_number", lambda G, b: nx.core_number(G, backend=b)),
        ("pagerank", lambda G, b: nx.pagerank(G, backend=b)),
        ("betweenness_centrality (k=1000)", lambda G, b: nx.betweenness_centrality(G, k=1000, seed=1, backend=b)),
        ("clustering", lambda G, b: nx.clustering(G, backend=b)),
        ("triangles", lambda G, b: nx.triangles(G, backend=b)),
        ("louvain_communities", lambda G, b: nx.community.louvain_communities(G, seed=1, backend=b)),
        ("articulation_points", lambda G, b: list(nx.articulation_points(G, backend=b))),
        ("degree_assortativity_coefficient", lambda G, b: nx.degree_assortativity_coefficient(G, backend=b)),
    ]


def collaboration():
    def prepare(G):
        return largest_cc(G)

    return "ca-AstroPh", False, prepare, [
        ("closeness_centrality", lambda G, b: nx.closeness_centrality(G, backend=b)),
        ("eigenvector_centrality", lambda G, b: nx.eigenvector_centrality(G, backend=b)),
        ("k_core (k=20)", lambda G, b: nx.k_core(G, 20, backend=b)),
        ("average_clustering", lambda G, b: nx.average_clustering(G, backend=b)),
        ("bridges", lambda G, b: list(nx.bridges(G, backend=b))),
        ("greedy_color", lambda G, b: nx.greedy_color(G, backend=b)),
        ("label_propagation_communities", lambda G, b: list(nx.community.label_propagation_communities(G, backend=b))),
    ]


def trust():
    return "soc-Epinions1", True, None, [
        ("strongly_connected_components", lambda G, b: list(nx.strongly_connected_components(G, backend=b))),
        ("weakly_connected_components", lambda G, b: list(nx.weakly_connected_components(G, backend=b))),
        ("condensation", lambda G, b: nx.condensation(G, backend=b)),
        ("pagerank", lambda G, b: nx.pagerank(G, backend=b)),
        ("hits", lambda G, b: nx.hits(G, backend=b)),
        ("in_degree_centrality", lambda G, b: nx.in_degree_centrality(G, backend=b)),
        ("reciprocity", lambda G, b: nx.reciprocity(G, backend=b)),
        ("katz_centrality", lambda G, b: nx.katz_centrality(G, alpha=0.005, backend=b)),
    ]


def citations():
    return "cit-HepPh", True, None, [
        ("pagerank", lambda G, b: nx.pagerank(G, backend=b)),
        ("strongly_connected_components", lambda G, b: list(nx.strongly_connected_components(G, backend=b))),
        ("is_directed_acyclic_graph", lambda G, b: nx.is_directed_acyclic_graph(G, backend=b)),
        ("descendants (50 papers)", lambda G, b: [nx.descendants(G, v, backend=b) for v in list(G)[:50]]),
        ("ancestors (50 papers)", lambda G, b: [nx.ancestors(G, v, backend=b) for v in list(G)[:50]]),
        ("in_degree_centrality", lambda G, b: nx.in_degree_centrality(G, backend=b)),
    ]


def roads():
    def prepare(G):
        # 20 routes between random nodes of the main component, picked
        # before timing.
        G.graph["routes"] = pairs(largest_cc(G), 20, 1)
        return with_weights(G)

    return "roadNet-PA", False, prepare, [
        ("connected_components", lambda G, b: list(nx.connected_components(G, backend=b))),
        ("dijkstra_path (20 routes)", lambda G, b: [nx.dijkstra_path(G, s, t, backend=b) for s, t in G.graph["routes"]]),
        ("single_source_dijkstra_path_length", lambda G, b: nx.single_source_dijkstra_path_length(G, next(iter(G)), backend=b)),
        ("bfs_edges", lambda G, b: list(nx.bfs_edges(G, next(iter(G)), backend=b))),
        ("minimum_spanning_tree", lambda G, b: nx.minimum_spanning_tree(G, backend=b)),
        ("articulation_points", lambda G, b: list(nx.articulation_points(G, backend=b))),
    ]


def web():
    return "web-Google", True, None, [
        ("pagerank", lambda G, b: nx.pagerank(G, backend=b)),
        ("strongly_connected_components", lambda G, b: list(nx.strongly_connected_components(G, backend=b))),
        ("weakly_connected_components", lambda G, b: list(nx.weakly_connected_components(G, backend=b))),
        ("in_degree_centrality", lambda G, b: nx.in_degree_centrality(G, backend=b)),
        # (no hits: NetworkX's own fails to converge on this graph)
    ]


PIPELINES = {
    "social network (Facebook)": social,
    "email network (Enron)": email,
    "collaboration network (astrophysics)": collaboration,
    "trust network (Epinions)": trust,
    "citation network (HEP-PH)": citations,
    "road routing (Pennsylvania)": roads,
    "web graph (Google)": web,
}


def run(data, name, factory):
    dataset, directed, prepare, steps = factory()
    t = time.perf_counter()
    G = load(data, dataset, directed)
    load_time = time.perf_counter() - t
    if prepare:
        G = prepare(G)
    rec = {"pipeline": name, "dataset": dataset, "nodes": len(G), "edges": G.number_of_edges(),
           "directed": directed, "load": load_time, "steps": []}
    print(f"\n{name}: {dataset}, {len(G):,} nodes, {G.number_of_edges():,} edges", flush=True)
    results = {}
    for backend in ("networkx", "rustnx"):
        cache = getattr(G, "__networkx_cache__", None)
        if cache:
            cache.clear()
        for i, (label, step) in enumerate(steps):
            random.seed(12345)
            t = time.perf_counter()
            out = materialize(step(G, backend))
            dt = time.perf_counter() - t
            if backend == "networkx":
                results[i] = norm(out)
                rec["steps"].append({"step": label, "networkx": dt})
            else:
                a = norm(out)
                s = rec["steps"][i]
                s["rustnx"] = dt
                s["match"] = "same" if a == results[i] else ("close" if close(a, results[i]) else "DIFFERENT")
                print(f"  {label:40s} networkx {s['networkx']:9.3f}s  rustnx {dt:8.3f}s  "
                      f"{s['networkx'] / dt:7.1f}x  {s['match']}", flush=True)
    rec["networkx_total"] = sum(s["networkx"] for s in rec["steps"])
    rec["rustnx_total"] = sum(s["rustnx"] for s in rec["steps"])
    print(f"  {'total':40s} networkx {rec['networkx_total']:9.3f}s  rustnx {rec['rustnx_total']:8.3f}s  "
          f"{rec['networkx_total'] / rec['rustnx_total']:7.1f}x", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default="real_world.json")
    args = ap.parse_args()
    import rustnx  # noqa: F401

    out = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            out = json.load(f)
    done = {r["pipeline"] for r in out}
    for name, factory in PIPELINES.items():
        if args.only and not any(o.lower() in name.lower() for o in args.only):
            continue
        if name in done:
            continue
        out.append(run(args.data, name, factory))
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
