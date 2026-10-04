"""Generate docs/API.md from the code, so the list can't drift.

    python scripts/gen_api_docs.py          # rewrite docs/API.md
    python scripts/gen_api_docs.py --check  # exit 1 if it is out of date

Signatures, multigraph support and the small-graph rule come from the code;
the notes below are the hand-written part. Output does not depend on the
installed NetworkX version.
"""

import inspect
import pathlib
import sys

from rustnx import algorithms, interface, rx  # the installed (or develop-mode) package

ROOT = pathlib.Path(__file__).resolve().parent.parent

OUTPUT = ROOT / "docs" / "API.md"

SECTIONS = [
    ("Centrality", [
        "degree_centrality", "in_degree_centrality", "out_degree_centrality",
        "betweenness_centrality", "edge_betweenness_centrality", "closeness_centrality",
        "harmonic_centrality", "eigenvector_centrality", "katz_centrality", "pagerank",
    ]),
    ("Shortest paths: lengths", [
        "single_source_shortest_path_length", "single_source_dijkstra_path_length",
        "all_pairs_shortest_path_length", "all_pairs_dijkstra_path_length",
        "dijkstra_path_length", "shortest_path_length",
        "single_target_shortest_path_length", "multi_source_dijkstra_path_length",
    ]),
    ("Shortest paths: paths", [
        "shortest_path", "single_source_shortest_path", "single_target_shortest_path",
        "bidirectional_shortest_path", "single_source_dijkstra", "single_source_dijkstra_path",
        "dijkstra_path", "bidirectional_dijkstra", "all_pairs_shortest_path",
        "all_pairs_dijkstra_path", "all_pairs_dijkstra", "all_shortest_paths",
        "multi_source_dijkstra", "multi_source_dijkstra_path",
        "single_source_all_shortest_paths", "all_pairs_all_shortest_paths",
    ]),
    ("Shortest paths: predecessors", [
        "predecessor", "dijkstra_predecessor_and_distance",
        "bellman_ford_predecessor_and_distance",
    ]),
    ("Shortest paths: Bellman-Ford and negative cycles", [
        "bellman_ford_path", "bellman_ford_path_length", "single_source_bellman_ford",
        "single_source_bellman_ford_path", "single_source_bellman_ford_path_length",
        "all_pairs_bellman_ford_path", "all_pairs_bellman_ford_path_length",
        "negative_edge_cycle", "find_negative_cycle",
    ]),
    ("Shortest paths: A*", ["astar_path", "astar_path_length"]),
    ("Reachability", ["has_path", "descendants", "ancestors"]),
    ("Distance measures", [
        "eccentricity", "diameter", "radius", "center", "periphery",
        "average_shortest_path_length", "wiener_index",
    ]),
    ("Components", [
        "connected_components", "number_connected_components", "is_connected",
        "node_connected_component", "articulation_points", "biconnected_components",
        "biconnected_component_edges", "is_biconnected",
        "strongly_connected_components", "number_strongly_connected_components",
        "is_strongly_connected", "weakly_connected_components",
        "number_weakly_connected_components", "is_weakly_connected",
        "attracting_components", "number_attracting_components", "is_attracting_component",
    ]),
    ("Directed acyclic graphs", [
        "topological_sort", "topological_generations", "is_directed_acyclic_graph",
    ]),
    ("Clustering", ["triangles", "clustering", "average_clustering", "transitivity"]),
    ("Cores, coloring and communities", [
        "core_number", "k_core", "is_bipartite", "greedy_color", "label_propagation_communities",
    ]),
    ("Traversal", [
        "bfs_edges", "bfs_tree", "bfs_predecessors", "bfs_successors", "bfs_layers",
        "descendants_at_distance", "dfs_edges", "dfs_tree", "dfs_preorder_nodes",
        "dfs_postorder_nodes", "dfs_predecessors", "dfs_successors",
    ]),
    ("Trees", ["is_tree", "is_forest"]),
    ("Spanning trees", [
        "minimum_spanning_edges", "maximum_spanning_edges",
        "minimum_spanning_tree", "maximum_spanning_tree",
    ]),
]

LENGTHS = "Falls back when weights mix ints and floats (NetworkX's length types then depend on the path)."
NOTES = {
    "betweenness_centrality": "Parallel. Matches NetworkX to about 1e-15 (sums in a different order). `k` picks the same nodes as NetworkX for a given `seed`. `None` weights fall back.",
    "edge_betweenness_centrality": "Parallel. Matches NetworkX to about 1e-15. `None` weights fall back.",
    "closeness_centrality": "Bit-for-bit identical.",
    "harmonic_centrality": "Bit-for-bit identical. A small `nbunch` with many `sources` falls back.",
    "eigenvector_centrality": "Bit-for-bit identical. `nstart` must give a value for every node.",
    "katz_centrality": "Bit-for-bit identical. `nstart` and a per-node `beta` fall back.",
    "pagerank": "Parallel on large graphs. `None` weights fall back.",
    "single_source_dijkstra_path_length": LENGTHS,
    "all_pairs_dijkstra_path_length": LENGTHS + " Generator.",
    "all_pairs_shortest_path_length": "Generator.",
    "dijkstra_path_length": LENGTHS,
    "shortest_path_length": "`method='bellman-ford'` with a weight falls back. " + LENGTHS,
    "shortest_path": "No source and no target falls back (the return type differs by version). `method='bellman-ford'` with a weight falls back.",
    "single_source_dijkstra": LENGTHS + " The paths dict follows the installed NetworkX's order.",
    "single_source_dijkstra_path": "The paths dict follows the installed NetworkX's order.",
    "bidirectional_dijkstra": LENGTHS,
    "all_pairs_shortest_path": "Generator.",
    "all_pairs_dijkstra_path": "Generator.",
    "all_pairs_dijkstra": LENGTHS + " Generator.",
    "all_shortest_paths": "Paths are generated lazily. `method='bellman-ford'` falls back.",
    "multi_source_dijkstra_path_length": LENGTHS + " An iterator of sources falls back.",
    "multi_source_dijkstra": LENGTHS + " The paths dict follows the installed NetworkX's order; on NetworkX 3.6+, negative weights that pop a node before all sources fall back. An iterator of sources falls back.",
    "multi_source_dijkstra_path": "The paths dict follows the installed NetworkX's order; on NetworkX 3.6+, negative weights that pop a node before all sources fall back. An iterator of sources falls back.",
    "single_target_shortest_path_length": "Returns an iterator, with NetworkX's FutureWarning, where the installed NetworkX does (3.4); a dict otherwise.",
    "single_source_all_shortest_paths": "Generator. All three methods, including `bellman-ford`. Paths come in the installed NetworkX's order.",
    "all_pairs_all_shortest_paths": "Generator. All three methods, including `bellman-ford`.",
    "predecessor": "A `cutoff` that isn't an int or float falls back.",
    "dijkstra_predecessor_and_distance": LENGTHS,
    "bellman_ford_predecessor_and_distance": LENGTHS + " `None` weights fall back. " + "Uses NetworkX's queue order and negative cycle checks.",
    "bellman_ford_path": "`None` weights fall back.",
    "bellman_ford_path_length": LENGTHS + " `None` weights fall back.",
    "single_source_bellman_ford": LENGTHS + " `None` weights fall back.",
    "single_source_bellman_ford_path": "`None` weights fall back.",
    "single_source_bellman_ford_path_length": LENGTHS + " `None` weights fall back.",
    "all_pairs_bellman_ford_path": "Generator; parallel. `None` weights fall back.",
    "all_pairs_bellman_ford_path_length": LENGTHS + " Generator; parallel. `None` weights fall back.",
    "negative_edge_cycle": "Doesn't add (and remove) a temporary node in the graph, as NetworkX does. `None` weights fall back.",
    "find_negative_cycle": "`None` weights fall back.",
    "astar_path": "Only without a `heuristic`; graphs with negative weights fall back.",
    "astar_path_length": "Only without a `heuristic`; graphs with negative weights fall back. " + LENGTHS,
    "eccentricity": "`sp` and empty graphs fall back.",
    "diameter": "`e`, empty graphs and `usebounds=True` on undirected graphs fall back.",
    "radius": "`e`, empty graphs and `usebounds=True` on undirected graphs fall back.",
    "center": "`e`, empty graphs, `usebounds=True` on undirected graphs and unweighted trees fall back.",
    "periphery": "`e`, empty graphs and `usebounds=True` on undirected graphs fall back.",
    "average_shortest_path_length": "Weighted sums are added in NetworkX's order. Methods other than `unweighted` and `dijkstra`, and the null graph, fall back.",
    "wiener_index": "Weighted sums are added in NetworkX's order.",
    "topological_sort": "Raises NetworkX's errors if the graph changes during iteration.",
    "topological_generations": "Raises NetworkX's errors if the graph changes during iteration.",
    "triangles": "Bit-for-bit identical.",
    "clustering": "Unweighted; `weight` falls back. Bit-for-bit identical.",
    "average_clustering": "Unweighted; `weight` falls back. Bit-for-bit identical.",
    "transitivity": "Bit-for-bit identical.",
    "k_core": "Builds the subgraph in NetworkX, so only the core numbers get faster.",
    "greedy_color": "Only the default `largest_first` strategy without `interchange`; others fall back.",
    "bfs_edges": "`sort_neighbors` falls back.",
    "bfs_tree": "`sort_neighbors` falls back.",
    "dfs_edges": "`sort_neighbors` falls back.",
    "dfs_tree": "`sort_neighbors` falls back.",
    "dfs_preorder_nodes": "`sort_neighbors` falls back.",
    "dfs_postorder_nodes": "`sort_neighbors` falls back.",
    "dfs_predecessors": "`sort_neighbors` falls back.",
    "dfs_successors": "`sort_neighbors` falls back.",
    "bfs_predecessors": "`sort_neighbors` falls back. Gives the installed NetworkX's deprecation warning (3.7+).",
    "bfs_successors": "`sort_neighbors` falls back.",
    "bfs_layers": "The first layer follows the installed NetworkX's order.",
    "minimum_spanning_edges": "Kruskal; Prim and Borůvka fall back. Yields the graph's own edge data dicts.",
    "maximum_spanning_edges": "Kruskal; Prim and Borůvka fall back. Yields the graph's own edge data dicts.",
    "minimum_spanning_tree": "Kruskal; Prim and Borůvka fall back.",
    "maximum_spanning_tree": "Kruskal; Prim and Borůvka fall back.",
}


def location(name):
    return "nx.community" if name == "label_propagation_communities" else "nx"


def parameters(name):
    params = list(inspect.signature(getattr(algorithms, name)).parameters)[1:]  # drop G
    return ", ".join(f"`{p}`" for p in params) or "none"


def render():
    listed = [n for _, names in SECTIONS for n in names]
    missing = sorted(set(algorithms.__all__) - set(listed))
    extra = sorted(set(listed) - set(algorithms.__all__))
    if missing or extra:
        raise SystemExit(f"update SECTIONS in {__file__}: missing {missing}, unknown {extra}")
    out = [
        "# rustnx API reference",
        "",
        "<!-- Generated by scripts/gen_api_docs.py; edit that script, not this file. -->",
        "",
        "## NetworkX backend",
        "",
        f"rustnx implements {len(listed)} NetworkX functions. Call them as usual (for",
        "example `nx.pagerank(G)`) after `rustnx.enable()`, or pass `backend=\"rustnx\"`.",
        "Results match the installed NetworkX (3.4 or newer) exactly.",
        "",
        "- **Parameters handled in Rust** are the ones rustnx implements. Any other",
        "  parameter of the installed NetworkX is fine at its default; set to anything",
        "  else, the call runs in NetworkX.",
        "- **Multigraphs**: whether `MultiGraph`/`MultiDiGraph` inputs run in Rust.",
        "- **Under 500 nodes**: \"NetworkX\" means small NetworkX graphs stay in",
        "  NetworkX automatically (converting would cost more than it saves); pass",
        "  `backend=\"rustnx\"` to force rustnx.",
        "- Callable weights always run in NetworkX.",
        "",
    ]
    for title, names in SECTIONS:
        out += [f"### {title}", "",
                "| Function | Parameters handled in Rust | Multigraphs | Under 500 nodes | Notes |",
                "|---|---|---|---|---|"]
        for name in names:
            multi = "yes" if name in interface.MULTIGRAPH_FUNCTIONS else "no"
            small = "NetworkX" if name in interface._LINEAR_TIME else "rustnx"
            out.append(f"| `{location(name)}.{name}` | {parameters(name)} | {multi} | {small} | {NOTES.get(name, '')} |")
        out.append("")

    classes = [n for n in rx.__all__ if n in ("PyGraph", "PyDiGraph")]
    errors = [n for n in rx.__all__ if isinstance(getattr(rx, n), type) and issubclass(getattr(rx, n), Exception)]
    funcs = [n for n in rx.__all__ if n not in classes and n not in errors]
    out += [
        "## rustworkx-compatible API (`rustnx.rx`)",
        "",
        "A subset of rustworkx's API on the same Rust core, tested against rustworkx",
        "itself. See the README for the semantics it matches.",
        "",
        "- **Graph classes:** " + ", ".join(f"`{n}`" for n in classes),
        "- **Functions:** " + ", ".join(f"`{n}`" for n in funcs),
        "- **Exceptions:** " + ", ".join(f"`{n}`" for n in errors),
        "",
        "## Native graphs",
        "",
        "`rustnx.Graph` and `rustnx.DiGraph` are built in Rust from edge tuples, or",
        "with `from_arrays(src, dst, weights=None, *, num_nodes=None)` from NumPy",
        "arrays. They are read-only (`nodes`, `edges`, `neighbors`, `successors`,",
        "`predecessors`, `degree`, `has_edge`, `has_node`, `len`), every function",
        "above accepts them, and `to_networkx()` converts back.",
        "",
    ]
    return "\n".join(out)


def main():
    text = render()
    if "--check" in sys.argv:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != text:
            print(f"{OUTPUT.relative_to(ROOT)} is out of date: run python scripts/gen_api_docs.py")
            return 1
        return 0
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
