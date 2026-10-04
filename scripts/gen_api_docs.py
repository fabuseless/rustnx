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
    ("Centrality: subsets, groups and more", [
        "betweenness_centrality_subset", "edge_betweenness_centrality_subset",
        "newman_betweenness_centrality", "edge_load_centrality", "percolation_centrality",
        "group_betweenness_centrality", "group_closeness_centrality", "group_degree_centrality",
        "group_in_degree_centrality", "group_out_degree_centrality",
        "prominent_group", "local_reaching_centrality", "global_reaching_centrality", "voterank",
        "dispersion",
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
        "kosaraju_strongly_connected_components", "condensation", "is_semiconnected",
    ]),
    ("Directed acyclic graphs", [
        "topological_sort", "topological_generations", "is_directed_acyclic_graph",
        "has_cycle", "lexicographical_topological_sort", "all_topological_sorts",
        "dag_longest_path", "dag_longest_path_length", "transitive_closure",
        "transitive_closure_dag", "transitive_reduction", "is_aperiodic",
        "v_structures", "root_to_leaf_paths", "dag_to_branching",
    ]),
    ("Clustering", [
        "triangles", "clustering", "average_clustering", "transitivity",
        "square_clustering", "generalized_degree", "all_triangles",
    ]),
    ("Cores, coloring and communities", [
        "core_number", "k_core", "k_shell", "k_crust", "k_corona", "k_truss", "onion_layers",
        "is_bipartite", "greedy_color", "is_coloring", "is_equitable",
        "label_propagation_communities",
    ]),
    ("Distance-regular graphs and other distance measures", [
        "centroid", "barycenter", "harmonic_diameter",
        "is_distance_regular", "intersection_array", "is_strongly_regular",
    ]),
    ("Traversal", [
        "bfs_edges", "bfs_tree", "bfs_predecessors", "bfs_successors", "bfs_layers",
        "descendants_at_distance", "dfs_edges", "dfs_tree", "dfs_preorder_nodes",
        "dfs_postorder_nodes", "dfs_predecessors", "dfs_successors",
        "generic_bfs_edges", "bfs_labeled_edges", "dfs_labeled_edges", "edge_bfs", "edge_dfs",
    ]),
    ("Trees", ["is_tree", "is_forest", "is_arborescence", "is_branching", "to_prufer_sequence"]),
    ("Spanning trees", [
        "minimum_spanning_edges", "maximum_spanning_edges",
        "minimum_spanning_tree", "maximum_spanning_tree", "kruskal_mst_edges",
    ]),
    ("Structural tests", [
        "bridges", "has_bridges", "local_bridges", "chain_decomposition",
        "isolates", "number_of_isolates", "is_regular", "is_k_regular",
        "is_tournament", "immediate_dominators", "dominance_frontiers",
    ]),
    ("Cycles and Euler tours", [
        "is_eulerian", "has_eulerian_path", "is_semieulerian", "eulerian_circuit",
        "eulerian_path", "cycle_basis", "find_cycle", "girth",
    ]),
    ("Isomorphism and graph hashing", [
        "could_be_isomorphic", "fast_could_be_isomorphic", "faster_could_be_isomorphic",
        "is_isomorphic", "vf2pp_is_isomorphic", "vf2pp_subgraph_is_isomorphic",
        "vf2pp_is_monomorphic", "tree_isomorphism", "rooted_tree_isomorphism", "root_trees",
        "weisfeiler_lehman_graph_hash", "weisfeiler_lehman_subgraph_hashes",
    ]),
    ("Bipartite graphs", [
        "color", "sets", "is_bipartite_node_set", "hopcroft_karp_matching", "to_vertex_cover",
        "bipartite_closeness_centrality", "node_redundancy", "butterflies",
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
    "betweenness_centrality_subset": "Parallel. Bit-for-bit identical (per-source sums are added in source order). Missing sources and `None` weights fall back.",
    "edge_betweenness_centrality_subset": "Parallel. Bit-for-bit identical. Missing sources, tuple node labels and `None` weights fall back.",
    "newman_betweenness_centrality": "Also reachable as `nx.load_centrality`. Parallel. Bit-for-bit identical. Only int or str node labels (NetworkX sorts nodes on ties); others fall back.",
    "edge_load_centrality": "Parallel. Bit-for-bit identical.",
    "percolation_centrality": "Parallel. Bit-for-bit identical. Non-numeric states, 2-node graphs and states that would divide by zero fall back.",
    "group_betweenness_centrality": "Bit-for-bit identical, following the installed NetworkX's algorithm (3.7 changed it). Cases where NetworkX raises `KeyError` (directed graphs before 3.7) fall back, as does the null graph.",
    "prominent_group": "Bit-for-bit identical search. `C`, `k` outside 0 to n, non-int/str node labels, and cases where NetworkX raises fall back. Needs pandas installed, like NetworkX.",
    "group_closeness_centrality": "Negative weights fall back.",
    "group_degree_centrality": "Missing nodes and one-shot iterators fall back.",
    "group_in_degree_centrality": "Missing nodes and one-shot iterators fall back.",
    "group_out_degree_centrality": "Missing nodes and one-shot iterators fall back.",
    "local_reaching_centrality": "`paths` falls back. Weighted: all edges need an int or float weight (not mixed, not zero), else it falls back.",
    "global_reaching_centrality": "Weighted: as `local_reaching_centrality`. Gives the installed NetworkX's warnings (3.4).",
    "voterank": "Bit-for-bit identical.",
    "dispersion": "Undirected graphs without self-loops; others fall back (NetworkX's count then depends on set order).",
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
    "greedy_color": "Strategies `largest_first`, `saturation_largest_first` (`DSATUR`), `random_sequential` (draws from the global `random` state, as NetworkX does) and `connected_sequential` (`_bfs`, `_dfs`). `smallest_last`, `independent_set`, callables and `interchange` fall back.",
    "k_shell": "Builds the subgraph in NetworkX, so only the core numbers get faster.",
    "k_crust": "Builds the subgraph in NetworkX, so only the core numbers get faster.",
    "k_corona": "Builds the subgraph in NetworkX. A non-integer `k` falls back.",
    "k_truss": "Peels edges in Rust; the result is NetworkX's `G.copy()` with the dropped edges and nodes removed.",
    "square_clustering": "Bit-for-bit identical. Follows the installed NetworkX's formula (3.4 differs from 3.5+ on self-loops and directed graphs).",
    "generalized_degree": "Builds each neighbor set in Python, as NetworkX does, so the `Counter` keys come in the same order.",
    "all_triangles": "NetworkX 3.7+. Generator. Rebuilds Python's intersection sets where several triangles share an edge, for NetworkX's yield order. Graph views fall back.",
    "centroid": "NetworkX 3.7+ (`barycenter` before). Weighted sums are added in NetworkX's order; unweighted trees take NetworkX's tree path. `attr` and `sp` fall back.",
    "barycenter": "NetworkX 3.4 and 3.5 (`centroid` from 3.7). Weighted sums are added in NetworkX's order. `attr` and `sp` fall back.",
    "harmonic_diameter": "Inverse distances are added in NetworkX's order. `sp` falls back; `weight` needs NetworkX 3.5+.",
    "intersection_array": "Follows the installed NetworkX's checks (3.7+ rejects long cycles early) and error messages.",
    "is_distance_regular": "Follows the installed NetworkX's checks.",
    "is_strongly_regular": "Follows the installed NetworkX's checks.",
    "is_coloring": "A coloring that isn't a `dict` of int, str or float colors falls back.",
    "is_equitable": "A coloring that isn't a `dict` of int, str or float colors falls back.",
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
    "lexicographical_topological_sort": "`key` falls back, and so do nodes that aren't all ints and floats (no NaN) or all strings.",
    "all_topological_sorts": "Sorts are generated lazily.",
    "dag_longest_path": "`topo_order` falls back, and so do `None` weights and a `default_weight` other than the one the graph was converted with.",
    "dag_longest_path_length": "As `dag_longest_path`. Lengths are summed in Python from the edge data, as in NetworkX.",
    "transitive_closure": "Builds the closure in NetworkX from Rust searches; edges are added in NetworkX's order (including set iteration order).",
    "transitive_closure_dag": "`topo_order` falls back. Builds the closure in NetworkX, adding edges in NetworkX's order (including set iteration order).",
    "transitive_reduction": "Kept edges are added in NetworkX's (set iteration) order.",
    "is_aperiodic": "On NetworkX 3.4, graphs not reachable from their first node fall back (NetworkX recurses in set order).",
    "v_structures": "Generator.",
    "root_to_leaf_paths": "Paths are generated lazily. Undirected graphs fall back.",
    "generic_bfs_edges": "`neighbors` falls back.",
    "dfs_labeled_edges": "`sort_neighbors` falls back.",
    "edge_bfs": "A `source` that is neither a node nor a list, tuple, set or dict of hashable nodes falls back.",
    "edge_dfs": "A `source` that is neither a node nor a list, tuple, set or dict of hashable nodes falls back.",
    "kosaraju_strongly_connected_components": "Component sets are filled in the installed NetworkX's order.",
    "condensation": "`scc` falls back. Member sets follow `strongly_connected_components`' order.",
    "kruskal_mst_edges": "`partition` falls back. Yields the graph's own edge data dicts.",
    "to_prufer_sequence": "Bit-for-bit identical.",
    "bridges": "`root` falls back (NetworkX then lists a subgraph copy's edges, in set order).",
    "local_bridges": "Spans with float, mixed or negative weights fall back.",
    "chain_decomposition": "Computed when iteration starts, as NetworkX does.",
    "is_k_regular": "A non-integer `k` falls back.",
    "immediate_dominators": "Follows the installed NetworkX on whether `start` is included (3.7 leaves it out).",
    "dominance_frontiers": "Follows the installed NetworkX's version (3.7 adds `start` last). Sets iterate in NetworkX's order.",
    "eulerian_circuit": "Computed when iteration starts, as NetworkX does. A `source` not in the graph falls back.",
    "eulerian_path": "Computed when iteration starts, as NetworkX does. A `source` not in the graph falls back.",
    "has_eulerian_path": "A `source` not in the graph falls back.",
    "cycle_basis": "A `root` not in the graph falls back.",
    "find_cycle": "A `source` that isn't one node of the graph, and invalid orientations on directed graphs, fall back.",
    "could_be_isomorphic": "Takes two graphs. Follows the installed NetworkX's checks (3.5+ stops at the first property that differs, which decides whether directed graphs raise). Multigraphs fall back.",
    "fast_could_be_isomorphic": "Takes two graphs. As `could_be_isomorphic` (3.7+ stops at the first property that differs).",
    "faster_could_be_isomorphic": "Takes two graphs.",
    "is_isomorphic": "Takes two graphs. A yes/no answer, so an exact matcher in Rust (with color refinement) gives NetworkX's result without running VF2. `node_match`, `edge_match` and multigraphs fall back.",
    "vf2pp_is_isomorphic": "Takes two graphs. Node labels are read from the NetworkX graphs. Empty graphs give `False`, as in NetworkX. A directed and an undirected graph fall back before NetworkX 3.7.",
    "vf2pp_subgraph_is_isomorphic": "NetworkX 3.7+. Takes two graphs (the second is the smaller). Node labels are read from the NetworkX graphs.",
    "vf2pp_is_monomorphic": "NetworkX 3.7+. Takes two graphs (the second is the smaller). Node labels are read from the NetworkX graphs.",
    "tree_isomorphism": "Takes two trees. Follows the installed NetworkX's child order and errors (3.4 asserts; its recursive walk means very deep trees fall back there). A directed `t2` falls back.",
    "rooted_tree_isomorphism": "As `tree_isomorphism`. Directed trees and roots not in the trees fall back.",
    "root_trees": "Builds the combined tree in NetworkX from Rust searches. Roots not in the graphs fall back.",
    "weisfeiler_lehman_graph_hash": "BLAKE2b in Rust, parallel; same hashes as NetworkX's `hashlib`, for the installed version (3.5 changed them), with the same warnings. Node and edge attribute labels are read from the NetworkX graph; non-ASCII labels, non-str attribute names and an unusual `digest_size` fall back.",
    "weisfeiler_lehman_subgraph_hashes": "As `weisfeiler_lehman_graph_hash`.",
    "color": "Directed graphs visit predecessors in NetworkX's order.",
    "sets": "Sets are filled in NetworkX's order, so they iterate the same way.",
    "is_bipartite_node_set": "Directed graphs raise as in NetworkX.",
    "hopcroft_karp_matching": "Also `nx.bipartite.maximum_matching`. Follows NetworkX's search order (its `left` set's iteration order), so it finds the same matching. Directed graphs, `top_nodes` not in the graph or with neighbors among themselves, and augmenting paths deep enough to approach Python's recursion limit fall back.",
    "to_vertex_cover": "Parallel. Builds the cover with NetworkX's set operations. Directed graphs, multigraphs and `top_nodes` not in the graph fall back.",
    "bipartite_closeness_centrality": "As `nx.bipartite.closeness_centrality`. Bit-for-bit identical; parallel searches.",
    "node_redundancy": "A one-shot iterator of nodes, and nodes not in the graph, fall back.",
    "butterflies": "NetworkX 3.7+.",
}

# Functions NetworkX only exposes under `nx.isomorphism` or `nx.bipartite`
# (or their modules).
ISOMORPHISM_ONLY = {
    "tree_isomorphism": "nx.isomorphism",
    "rooted_tree_isomorphism": "nx.isomorphism",
    "root_trees": "networkx.algorithms.isomorphism.tree_isomorphism",
    "color": "nx.bipartite",
    "sets": "nx.bipartite",
    "is_bipartite_node_set": "nx.bipartite",
    "hopcroft_karp_matching": "nx.bipartite",
    "to_vertex_cover": "nx.bipartite",
    "node_redundancy": "nx.bipartite",
    "butterflies": "nx.bipartite",
}

# Functions NetworkX only exposes under `nx.dag`.
DAG_ONLY = {"v_structures", "root_to_leaf_paths", "has_cycle"}


def location(name):
    if name in DAG_ONLY:
        return "nx.dag"
    if name in ISOMORPHISM_ONLY:
        return ISOMORPHISM_ONLY[name]
    if name in ("is_coloring", "is_equitable"):
        return "nx.algorithms.coloring.equitable_coloring"
    return "nx.community" if name == "label_propagation_communities" else "nx"
    return {
        "label_propagation_communities": "nx.community",
        "kruskal_mst_edges": "nx.tree.mst",
        "is_tournament": "nx.tournament",
    }.get(name, "nx")


def parameters(name):
    params = list(inspect.signature(getattr(algorithms, name)).parameters)[1:]  # drop G
    params = [p for p in params if not p.startswith("_")]  # internal keywords
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
