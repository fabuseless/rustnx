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
    ("Shortest paths: A*", ["astar_path", "astar_path_length"]),
    ("Reachability", ["has_path", "descendants", "ancestors"]),
    ("Trees", ["is_tree", "is_forest", "is_arborescence", "is_branching", "to_prufer_sequence"]),
    ("Tournaments", ["is_reachable", "tournament_is_strongly_connected", "score_sequence"]),
    ("Cliques", ["enumerate_all_cliques", "node_clique_number", "max_weight_clique"]),
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
    ("Shortest paths: Floyd-Warshall, Johnson and Goldberg-Radzik", [
        "floyd_warshall", "floyd_warshall_predecessor_and_distance", "floyd_warshall_tree",
        "floyd_warshall_numpy", "johnson", "goldberg_radzik",
    ]),
    ("Simple paths", [
        "all_simple_paths", "all_simple_edge_paths", "shortest_simple_paths", "is_simple_path",
    ]),
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
        "v_structures", "root_to_leaf_paths", "dag_to_branching", "antichains",
        "antichain_width",
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
        "eulerian_path", "cycle_basis", "find_cycle", "girth", "minimum_cycle_basis",
    ]),
    ("Branchings, arborescences and more trees", [
        "maximum_branching", "minimum_branching", "minimal_branching",
        "maximum_spanning_arborescence", "minimum_spanning_arborescence",
        "greedy_branching", "branching_weight", "prim_mst_edges", "boruvka_mst_edges",
        "partition_spanning_tree",
        "from_prufer_sequence", "from_nested_tuple", "to_nested_tuple", "tree_centroid",
    ]),
    ("Lowest common ancestors", [
        "lowest_common_ancestor", "all_pairs_lowest_common_ancestor",
        "tree_all_pairs_lowest_common_ancestor",
    ]),
    ("Planarity and graph classes", [
        "is_planar", "check_planarity", "check_planarity_recursive", "get_counterexample",
        "get_counterexample_recursive", "is_chordal", "chordal_graph_treewidth",
        "complete_to_chordal_graph", "is_at_free", "is_perfect_graph",
    ]),
    ("Triads and d-separation", [
        "triadic_census", "is_d_separator", "is_minimal_d_separator", "find_minimal_d_separator",
    ]),
    ("Degree sequences", [
        "is_graphical", "is_digraphical", "is_multigraphical", "is_pseudographical",
        "is_valid_degree_sequence_erdos_gallai", "is_valid_degree_sequence_havel_hakimi",
    ]),
    ("Boundaries and dominating sets", [
        "node_boundary", "edge_boundary", "is_dominating_set", "connected_dominating_set",
        "is_connected_dominating_set",
    ]),
    ("Matching and covers", [
        "maximal_matching", "max_weight_matching", "min_weight_matching", "is_matching",
        "is_maximal_matching", "is_perfect_matching", "min_edge_cover",
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
    ("Flows", [
        "maximum_flow", "maximum_flow_value", "minimum_cut", "minimum_cut_value",
        "edmonds_karp", "shortest_augmenting_path", "dinitz", "boykov_kolmogorov",
        "preflow_push", "build_residual_network", "build_flow_dict", "gomory_hu_tree",
    ]),
    ("Minimum cost flows", [
        "network_simplex", "min_cost_flow", "min_cost_flow_cost", "max_flow_min_cost",
        "cost_of_flow",
    ]),
    ("Cut measures", [
        "cut_size", "volume", "normalized_cut_size", "conductance", "edge_expansion",
        "mixing_expansion", "node_expansion", "boundary_expansion",
    ]),
    ("Connectivity and cuts", [
        "node_connectivity", "edge_connectivity", "local_node_connectivity",
        "local_edge_connectivity", "average_node_connectivity", "all_pairs_node_connectivity",
        "minimum_node_cut", "minimum_edge_cut", "minimum_st_node_cut", "minimum_st_edge_cut",
        "node_disjoint_paths", "edge_disjoint_paths", "stoer_wagner",
    ]),
    ("Edge components and augmentation", [
        "bridge_components", "k_edge_components", "k_edge_subgraphs", "is_k_edge_connected",
        "is_locally_k_edge_connected", "k_edge_augmentation", "one_edge_augmentation",
        "unconstrained_one_edge_augmentation", "bridge_augmentation",
        "unconstrained_bridge_augmentation",
    ]),
    ("Assortativity and mixing", [
        "degree_assortativity_coefficient", "degree_pearson_correlation_coefficient",
        "attribute_assortativity_coefficient", "numeric_assortativity_coefficient",
        "degree_mixing_dict", "degree_mixing_matrix", "attribute_mixing_dict",
        "attribute_mixing_matrix", "node_degree_xy", "node_attribute_xy",
        "average_degree_connectivity", "average_neighbor_degree",
    ]),
    ("Link prediction", [
        "jaccard_coefficient", "adamic_adar_index", "resource_allocation_index",
        "preferential_attachment", "cn_soundarajan_hopcroft", "ra_index_soundarajan_hopcroft",
        "within_inter_cluster", "common_neighbor_centrality",
    ]),
    ("Reciprocity, rich club and walks", [
        "reciprocity", "overall_reciprocity", "rich_club_coefficient", "s_metric",
        "number_of_walks",
    ]),
    ("Communities", [
        "modularity", "overlapping_modularity", "partition_quality", "is_partition", "is_cover",
        "intra_community_edges", "inter_community_edges", "inter_community_non_edges",
        "greedy_modularity_communities", "naive_greedy_modularity_communities",
        "girvan_newman", "edge_betweenness_partition", "asyn_lpa_communities",
        "fast_label_propagation_communities", "asyn_fluidc",
    ]),
    ("Efficiency, vitality and distance indices", [
        "efficiency", "global_efficiency", "local_efficiency", "closeness_vitality",
        "gutman_index", "schultz_index", "hyper_wiener_index", "flow_hierarchy", "voronoi_cells",
    ]),
    ("Approximation algorithms", [
        "min_weighted_vertex_cover", "min_weighted_dominating_set", "min_edge_dominating_set",
        "min_maximal_matching", "greedy_tsp", "simulated_annealing_tsp",
        "threshold_accepting_tsp", "treewidth_min_fill_in", "treewidth_decomp",
        "approximate_diameter", "one_exchange", "randomized_partitioning", "steiner_tree",
        "densest_subgraph",
    ]),
    ("Random graph generators", [
        "gnp_random_graph", "fast_gnp_random_graph", "gnm_random_graph",
        "dense_gnm_random_graph", "barabasi_albert_graph", "dual_barabasi_albert_graph",
        "extended_barabasi_albert_graph", "watts_strogatz_graph", "newman_watts_strogatz_graph",
        "connected_watts_strogatz_graph", "powerlaw_cluster_graph", "random_regular_graph",
        "gn_graph", "gnr_graph", "gnc_graph", "random_uniform_k_out_graph",
        "random_lobster_graph", "random_lobster", "random_tournament", "stochastic_block_model",
        "random_partition_graph", "planted_partition_graph", "random_geometric_graph",
        "waxman_graph", "random_graph", "gnmk_random_graph",
    ]),
    ("Matrices and conversion", [
        "to_scipy_sparse_array", "adjacency_matrix", "laplacian_matrix", "incidence_matrix",
        "to_numpy_array", "biadjacency_matrix", "to_dict_of_lists", "number_of_selfloops",
        "is_weighted", "is_negatively_weighted", "get_node_attributes",
        "get_edge_attributes", "relabel_nodes", "convert_node_labels_to_integers",
        "from_dict_of_lists", "from_dict_of_dicts", "from_edgelist", "from_numpy_array",
        "from_scipy_sparse_array", "from_biadjacency_matrix",
    ]),
    ("Deterministic generators", [
        "empty_graph", "complete_graph", "cycle_graph", "path_graph", "star_graph",
        "wheel_graph", "ladder_graph", "circular_ladder_graph", "lollipop_graph",
        "barbell_graph", "tadpole_graph", "full_rary_tree", "balanced_tree", "binomial_tree",
        "complete_bipartite_graph", "complete_multipartite_graph", "turan_graph",
        "grid_2d_graph", "grid_graph", "hypercube_graph", "hexagonal_lattice_graph",
        "triangular_lattice_graph", "circulant_graph", "caveman_graph",
        "connected_caveman_graph", "ring_of_cliques", "windmill_graph", "sudoku_graph",
        "LCF_graph", "generalized_petersen_graph", "dorogovtsev_goltsev_mendes_graph",
        "mycielski_graph", "paley_graph", "kneser_graph",
    ]),
    ("Graph operations", [
        "complement", "power", "difference", "symmetric_difference", "is_kl_connected",
        "kl_connected_subgraph",
    ]),
    ("Reading and parsing", [
        "parse_edgelist", "read_edgelist", "read_weighted_edgelist", "bipartite_parse_edgelist",
        "bipartite_read_edgelist", "parse_adjlist", "read_adjlist", "parse_multiline_adjlist",
        "read_multiline_adjlist", "parse_leda", "read_leda", "parse_pajek", "read_pajek",
        "from_graph6_bytes", "read_graph6", "from_sparse6_bytes", "read_sparse6",
        "node_link_graph", "adjacency_graph", "cytoscape_graph", "tree_graph", "parse_gml",
        "read_gml",
    ]),
]

FLOWNOTE = "Capacities (and the flow values) can be ints or floats, mixed or missing (infinite): every value, its type and the order of every float addition are NetworkX's."

DEGREE_WEIGHTS = "Integer edge weights only; float weights fall back."
ATTRIBUTES = "Node attributes are read from the NetworkX graph."
LINKS = "Scores pairs in batches as the generator is consumed, so `ebunch` (default: `nx.non_edges(G)`) is streamed."
COMMUNITIES = "Community values must be ints, floats, strings, bools or `None`; others fall back."
LENGTHS = "Falls back when weights mix ints and floats (NetworkX's length types then depend on the path)."
NOTES = {
    "triadic_census": "A `nodelist` that is one node or an iterator falls back.",
    "is_d_separator": "Arguments other than nodes and sets of nodes (lists, for example) fall back.",
    "is_minimal_d_separator": "Arguments other than nodes and sets of nodes (lists, for example) fall back.",
    "find_minimal_d_separator": "Arguments other than nodes and sets of nodes fall back, as do nodes equal to but of a different type than G's. The set iterates in NetworkX's order.",
    "is_graphical": "Takes a sequence, not a graph: runs in rustnx with `backend=\"rustnx\"` or `nx.config.backend_priority`. Ints beyond 64 bits run NetworkX's code.",
    "is_digraphical": "Takes a sequence, not a graph: runs in rustnx with `backend=\"rustnx\"` or `nx.config.backend_priority`. Ints beyond 64 bits run NetworkX's code.",
    "is_multigraphical": "Takes a sequence, not a graph: runs in rustnx with `backend=\"rustnx\"` or `nx.config.backend_priority`. Ints beyond 64 bits run NetworkX's code.",
    "is_pseudographical": "Takes a sequence, not a graph: runs in rustnx with `backend=\"rustnx\"` or `nx.config.backend_priority`. Ints beyond 64 bits run NetworkX's code.",
    "is_valid_degree_sequence_erdos_gallai": "Takes a sequence, not a graph: runs in rustnx with `backend=\"rustnx\"` or `nx.config.backend_priority`. Ints beyond 64 bits run NetworkX's code.",
    "is_valid_degree_sequence_havel_hakimi": "Takes a sequence, not a graph: runs in rustnx with `backend=\"rustnx\"` or `nx.config.backend_priority`. Ints beyond 64 bits run NetworkX's code.",
    "node_boundary": "The set iterates in NetworkX's order.",
    "edge_boundary": "`data` falls back.",
    "connected_dominating_set": "NetworkX 3.5 and newer. The set iterates in NetworkX's order.",
    "is_connected_dominating_set": "NetworkX 3.5 and newer.",
    "maximal_matching": "The set iterates in NetworkX's order.",
    "max_weight_matching": "Integer weights beyond 2**49 in magnitude fall back. The set (and each pair's orientation) is NetworkX's.",
    "min_weight_matching": "Integer weights beyond 2**49 in magnitude fall back. The set (and each pair's orientation) is NetworkX's.",
    "min_edge_cover": "`matching_algorithm` falls back. The set iterates in NetworkX's order.",
    "enumerate_all_cliques": "Computed in batches when iteration starts, as NetworkX does.",
    "node_clique_number": "`nodes=None` (the dict follows `find_cliques`' set order) and `cliques` fall back, as do directed graphs and nodes not in G.",
    "max_weight_clique": "Node weights are read from the NetworkX graph; native graphs support `weight=None` only.",
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
    "reciprocity": "Undirected graphs with `nodes` fall back (NetworkX raises `AttributeError`).",
    "rich_club_coefficient": "`normalized=False` only: normalizing uses random edge swaps, so the default falls back.",
    "number_of_walks": "Exact int64 arithmetic (wrapping on overflow as NumPy and SciPy do). Graphs without edges (NetworkX returns floats) and walks longer than 100 fall back.",
    "betweenness_centrality": "Parallel. Matches NetworkX to about 1e-15 (sums in a different order). `k` picks the same nodes as NetworkX for a given `seed`. `None` weights fall back.",
    "edge_betweenness_centrality": "Parallel. Matches NetworkX to about 1e-15. `None` weights fall back.",
    "closeness_centrality": "Bit-for-bit identical.",
    "harmonic_centrality": "Bit-for-bit identical. A small `nbunch` with many `sources` falls back.",
    "eigenvector_centrality": "Bit-for-bit identical. `nstart` must give a value for every node.",
    "katz_centrality": "Bit-for-bit identical. `nstart` and a per-node `beta` fall back.",
    "pagerank": "Parallel on large graphs. Matches NetworkX to about 1e-16 (NetworkX computes it with SciPy sparse arithmetic). `None` weights fall back.",
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
    "modularity": "Bit-for-bit identical: sums follow `set(community)` order. Weights must be all ints or all floats. A non-partition, and an edgeless graph on NetworkX 3.4, fall back (to raise NetworkX's error).",
    "overlapping_modularity": "NetworkX 3.7+. Bit-for-bit identical: sums follow `set(community)` order. Weights must be all ints or all floats.",
    "partition_quality": "Counts in Rust.",
    "is_partition": "",
    "is_cover": "NetworkX 3.7+.",
    "intra_community_edges": "Blocks that NetworkX rejects (unhashable nodes) fall back.",
    "inter_community_edges": "Linear time (NetworkX checks every pair of blocks). Iterator partitions fall back.",
    "inter_community_non_edges": "Linear time (NetworkX builds the complement graph). Iterator partitions fall back.",
    "greedy_modularity_communities": "Replays NetworkX's mapped-heap merges (ties by node order: int or str labels only) and its frozenset unions, so communities iterate alike. Weights must be all ints or all floats.",
    "naive_greedy_modularity_communities": "Unit or integer weights (float sums would follow frozenset order: those fall back). Incremental modularity, added in NetworkX's order.",
    "girvan_newman": "Lazy, like NetworkX. Exact edge betweenness (sources added in order) on NetworkX's rebuilt copy of G, so the same edge is removed on ties. `most_valuable_edge` and 2-tuple node labels fall back.",
    "edge_betweenness_partition": "Exact edge betweenness (sources added in order) on NetworkX's rebuilt copy of G, so the same edge is removed on ties. 2-tuple node labels and `None` weights fall back.",
    "asyn_lpa_communities": "Replays CPython's Mersenne Twister, so a `random.Random` seed (or `seed=None`, the global generator) gives NetworkX's exact draws and leaves the generator in the same state; NumPy seeds fall back.",
    "fast_label_propagation_communities": "Replays CPython's Mersenne Twister, so a `random.Random` seed (or `seed=None`, the global generator) gives NetworkX's exact draws and leaves the generator in the same state; NumPy seeds fall back.",
    "asyn_fluidc": "Replays CPython's Mersenne Twister, so a `random.Random` seed (or `seed=None`, the global generator) gives NetworkX's exact draws and leaves the generator in the same state; NumPy seeds fall back. Follows each version's loop limit.",
    "efficiency": "",
    "global_efficiency": "Parallel BFS; `1 / d` added in NetworkX's order.",
    "local_efficiency": "Parallel. Visits each neighborhood in the order of NetworkX's subgraph view (a Python set's order for small neighborhoods), so sums match bit for bit.",
    "closeness_vitality": "Parallel. Wiener indices summed in NetworkX's order. Weights must be non-negative and all ints or all floats.",
    "gutman_index": "Parallel. Weights must be non-negative and all ints or all floats.",
    "schultz_index": "Parallel. Weights must be non-negative and all ints or all floats.",
    "hyper_wiener_index": "NetworkX 3.6+. Parallel. Unweighted or integer weights (float squares fall back).",
    "flow_hierarchy": "Unweighted or integer weights.",
    "voronoi_cells": "Multi-source Dijkstra in Rust.",
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
    "floyd_warshall": "Parallel. Follows the installed NetworkX's version (3.4/3.5 count missing weights as 1.0 and don't check for negative cycles). Dict and row key orders match. Mixed int and float weights fall back, as do `None` weights before 3.6.",
    "floyd_warshall_predecessor_and_distance": "As `floyd_warshall`; the predecessor dicts match too.",
    "floyd_warshall_tree": "NetworkX 3.7+. Mixed int and float weights fall back.",
    "floyd_warshall_numpy": "Parallel; the same float operations as NetworkX's NumPy loop. `None` and `-0.0` weights, and a `nodelist` holding nodes not in the graph, fall back.",
    "johnson": "Parallel. The paths dicts follow the installed NetworkX's order. `None` and infinite weights fall back.",
    "goldberg_radzik": "Follows the installed NetworkX's version (3.7 changed the scan order). Rebuilds NetworkX's `relabeled` sets in Python, since they are visited in set order. " + LENGTHS + " `None` and infinite weights fall back.",
    "antichains": "Generator; the closure is computed when iteration starts, as NetworkX does. A `topo_order` that isn't a topological order of the graph falls back.",
    "antichain_width": "NetworkX 3.7+. Counts a maximum matching of the closure in Rust.",
    "all_simple_paths": "Paths are generated lazily in Rust, in NetworkX's order. Raises RuntimeError if the graph changes during iteration.",
    "all_simple_edge_paths": "Paths are generated lazily in Rust, in NetworkX's order. Raises RuntimeError if the graph changes during iteration.",
    "shortest_simple_paths": "Yen's algorithm with NetworkX's bidirectional searches and tie-breaking, one path per step. Mixed int and float, and infinite, weights fall back.",
    "minimum_cycle_basis": "Parallel lifted-graph searches. Builds the subgraph views and chord sets in Python, as NetworkX does (their order can depend on set layout). `None` and infinite weights, and tuple node labels, fall back.",
    "maximum_branching": "Edmonds' algorithm as NetworkX runs it, in Rust; the result's edges come in NetworkX's (set iteration) order. Weights must be Python ints or floats, and `partition` values `EdgePartition` members or `None`.",
    "minimum_branching": "Edmonds' algorithm as NetworkX runs it, in Rust; the result's edges come in NetworkX's (set iteration) order. Weights must be Python ints or floats, and `partition` values `EdgePartition` members or `None`. Changes G's weights exactly as NetworkX does (edges without `attr` gain it).",
    "minimal_branching": "Edmonds' algorithm as NetworkX runs it, in Rust; the result's edges come in NetworkX's (set iteration) order. Weights must be Python ints or floats, and `partition` values `EdgePartition` members or `None`. Changes G's weights exactly as NetworkX does (edges without `attr` gain it).",
    "maximum_spanning_arborescence": "Edmonds' algorithm as NetworkX runs it, in Rust; the result's edges come in NetworkX's (set iteration) order. Weights must be Python ints or floats, and `partition` values `EdgePartition` members or `None`. Changes G's weights exactly as NetworkX does (edges without `attr` gain it).",
    "minimum_spanning_arborescence": "Edmonds' algorithm as NetworkX runs it, in Rust; the result's edges come in NetworkX's (set iteration) order. Weights must be Python ints or floats, and `partition` values `EdgePartition` members or `None`. Changes G's weights exactly as NetworkX does (edges without `attr` gain it).",
    "greedy_branching": "Only int or str node labels (NetworkX sorts edges by weight, then nodes); others fall back. `attr=None` uses the random state as NetworkX does.",
    "branching_weight": "Weights mixing ints and floats fall back.",
    "prim_mst_edges": "Undirected graphs; directed ones fall back. Trees start from the nodes NetworkX pops from `set(G)` (rustnx replays the same set operations). Yields the graph's own edge data dicts.",
    "boruvka_mst_edges": "Rounds run in Rust. Where a component's best edge is tied, rustnx replays NetworkX's set of the component's nodes to scan it in the same order. Yields the graph's own edge data dicts.",
    "partition_spanning_tree": "Kruskal with the partition in Rust. `partition` values must be `EdgePartition` members or `None`.",
    "from_prufer_sequence": "Takes no graph: runs in rustnx with `backend='rustnx'` or `nx.config.backend_priority.generators` (set by `rustnx.enable()`). Sequences of non-ints fall back.",
    "from_nested_tuple": "Takes no graph (see `from_prufer_sequence`). Nesting deeper than 100 levels falls back.",
    "to_nested_tuple": "`canonical_form=True` only (otherwise children follow set order). Trees deeper than 100 levels fall back.",
    "tree_centroid": "NetworkX 3.7+, as `nx.tree.centroid`.",
    "lowest_common_ancestor": "As `all_pairs_lowest_common_ancestor`.",
    "all_pairs_lowest_common_ancestor": "Pairs with a unique lowest common ancestor are answered in Rust; for pairs with several, rustnx repeats NetworkX's set-based walk.",
    "tree_all_pairs_lowest_common_ancestor": "`pairs` falls back (NetworkX keeps them in sets), and so does a `root` not in the graph.",
    "check_planarity": "Builds the same `PlanarEmbedding` (same half-edge calls) or counterexample.",
    "check_planarity_recursive": "Graphs deep enough that NetworkX might reach the recursion limit fall back.",
    "get_counterexample_recursive": "Graphs deep enough that NetworkX might reach the recursion limit fall back.",
    "is_chordal": "Self-loops fall back (NetworkX's outcome then depends on set order).",
    "chordal_graph_treewidth": "Self-loops and the null graph (whose result differs by version) fall back.",
    "complete_to_chordal_graph": "Self-loops fall back. Chords are added in NetworkX's (set iteration) order.",
    "is_reachable": "Unhashable nodes fall back.",
    "is_perfect_graph": "NetworkX 3.7 and later.",
    "maximum_flow": "Runs `flow_func` in Rust when it is one of NetworkX's five maximum flow functions (default `preflow_push`), with their keyword arguments (`cutoff`, `two_phase`, `global_relabel_freq`); other callables, `residual`, callable capacities and ints beyond 64 bits fall back. " + FLOWNOTE,
    "maximum_flow_value": "As `maximum_flow`.",
    "minimum_cut": "As `maximum_flow`. The partition's sets are built as the installed NetworkX builds them (3.7 searches from the sink one node at a time; earlier versions build a presized set), so they iterate the same way.",
    "minimum_cut_value": "As `maximum_flow`.",
    "edmonds_karp": "Returns NetworkX's residual network: same node and edge order (successors and predecessors), attributes and graph attributes. `residual` and callable capacities fall back. " + FLOWNOTE,
    "shortest_augmenting_path": "As `edmonds_karp`; the nodes' `height` and `curr_edge` attributes (a `CurrentEdge` at NetworkX's position) too.",
    "dinitz": "As `edmonds_karp`.",
    "boykov_kolmogorov": "As `edmonds_karp`, with the search trees in `R.graph[\"trees\"]`.",
    "preflow_push": "As `shortest_augmenting_path`, with `excess`. NetworkX picks active nodes with `next(iter(set))`; rustnx replays CPython's set table to pick the same ones (checked against the interpreter at first use; elsewhere it falls back).",
    "build_residual_network": "As `edmonds_karp` (no flows yet).",
    "build_flow_dict": "Takes G and a residual network R; flows are R's own objects (anything comparable with 0). Nodes of G missing from R, and edges without a flow, fall back.",
    "gomory_hu_tree": "Any of NetworkX's five maximum flow functions as `flow_func`; before 3.7 NetworkX's `minimum_cut` reorders the shared residual network, which rustnx repeats. Edge weights are bit-for-bit NetworkX's.",
    "network_simplex": "Ports NetworkX's pivot rule and spanning tree updates, so the flow (among several optimal ones) is NetworkX's; the faux infinity follows the installed version (3.6 changed it). Demands, capacities and weights can be ints or floats; other values (and ints beyond 64 bits) fall back, as do multigraphs. Errors and their messages are NetworkX's.",
    "min_cost_flow": "As `network_simplex`.",
    "min_cost_flow_cost": "As `network_simplex`.",
    "max_flow_min_cost": "`preflow_push` for the flow value, then `network_simplex` on `nx.DiGraph(G)`, as NetworkX does.",
    "cost_of_flow": "Flow values and weights can be ints or floats; anything else, and missing entries in `flowDict`, fall back.",
    "cut_size": "The `sum()` follows NetworkX's edge order and Python's summation (compensated from 3.12), so floats match bit for bit; weights mixing ints and floats are read from the NetworkX graph. Iterators as `S` or `T`, `None` weights, and directed graphs with `T=None` (NetworkX fails) fall back.",
    "volume": "As `cut_size`. A single node as `S` falls back (NetworkX fails).",
    "normalized_cut_size": "As `cut_size`.",
    "conductance": "As `cut_size`.",
    "edge_expansion": "As `cut_size`.",
    "mixing_expansion": "As `cut_size`.",
    "node_expansion": "Nodes not in G fall back.",
    "boundary_expansion": "",
    "min_weighted_vertex_cover": "Node weights are read from the NetworkX graph and must be Python ints or floats (int arithmetic within 64 bits). The set iterates in NetworkX's order.",
    "min_weighted_dominating_set": "Follows the installed NetworkX's cost rule (3.6 changed it). Node weights must be ints within 2**53 or floats other than NaN. The set iterates in NetworkX's order.",
    "min_edge_dominating_set": "As `maximal_matching`.",
    "min_maximal_matching": "As `maximal_matching`.",
    "greedy_tsp": "NetworkX picks each next node with `min()` over a set, so a tie for the nearest node falls back, as do graphs under 3 nodes, `weight=None` and a source not in G.",
    "simulated_annealing_tsp": "Moves and tour costs in Rust; the random draws come from `seed` exactly as in NetworkX. Custom `move` functions, `init_cycle=\"greedy\"` with a tie (see `greedy_tsp`), weights mixing ints and floats and graphs under 3 nodes fall back.",
    "threshold_accepting_tsp": "As `simulated_annealing_tsp`.",
    "treewidth_min_fill_in": "The min fill-in heuristic runs in Rust; rustnx then replays NetworkX's set operations, so the bags (frozensets) iterate the same way.",
    "treewidth_decomp": "The default `heuristic` only; directed graphs fall back. As `treewidth_min_fill_in`.",
    "approximate_diameter": "Both sweeps in Rust, from the node `seed` picks as in NetworkX.",
    "one_exchange": "Int weights (or none); float weights fall back, since NetworkX's float sums follow set order. The cut sets are rebuilt with NetworkX's set operations.",
    "randomized_partitioning": "Int weights (or none), as `one_exchange`.",
    "steiner_tree": "Mehlhorn's method (the default) for the installed version (3.6 changed its distances); `method=\"kou\"` falls back (it starts from `set.pop()`), as do negative and `None` weights. Returns a view of the original graph, as NetworkX does.",
    "densest_subgraph": "NetworkX 3.5 and newer. Greedy++ and FISTA (in float32, like NetworkX's NumPy arrays). FISTA falls back on self-loops, and with NetworkX 3.5 and 3.6 on nodes other than the ints 0..n-1.",
    "complement": "Plain `Graph` and `DiGraph` results with NetworkX's adjacency order; subclasses fall back.",
    "power": "Builds the result with NetworkX's adjacency order.",
    "difference": "Takes two graphs. Plain `Graph` and `DiGraph` results; subclasses fall back.",
    "symmetric_difference": "Takes two graphs. Plain `Graph` and `DiGraph` results; subclasses fall back.",
    "is_kl_connected": "`low_memory=True` falls back (its searches run on subgraph views ordered by a set).",
    "kl_connected_subgraph": "`low_memory=True` falls back. Returns a deep copy of the original graph without the rejected edges.",
}

# Functions NetworkX exposes only in a submodule, or under another name.
QUALIFIED = {
    "check_planarity_recursive": "nx.algorithms.planarity.check_planarity_recursive",
    "get_counterexample": "nx.algorithms.planarity.get_counterexample",
    "get_counterexample_recursive": "nx.algorithms.planarity.get_counterexample_recursive",
    "is_reachable": "nx.tournament.is_reachable",
    "tournament_is_strongly_connected": "nx.tournament.is_strongly_connected",
    "score_sequence": "nx.tournament.score_sequence",
    "approximate_diameter": "nx.approximation.diameter",
    "treewidth_decomp": "nx.algorithms.approximation.treewidth.treewidth_decomp",
    "degree_assortativity_coefficient": DEGREE_WEIGHTS + " Builds the mixing matrix in Rust; the last step is NetworkX's own NumPy code, so bit-for-bit identical.",
    "degree_pearson_correlation_coefficient": DEGREE_WEIGHTS + " Pairs come from Rust and go to SciPy's `pearsonr` as in NetworkX. Fewer than two pairs fall back (NetworkX versions differ there).",
    "attribute_assortativity_coefficient": ATTRIBUTES + " The last step is NetworkX's own NumPy code.",
    "numeric_assortativity_coefficient": ATTRIBUTES + " Native graphs fall back.",
    "degree_mixing_dict": DEGREE_WEIGHTS + " Same keys in the same order.",
    "degree_mixing_matrix": DEGREE_WEIGHTS,
    "attribute_mixing_dict": ATTRIBUTES + " Same keys (the same objects) in the same order.",
    "attribute_mixing_matrix": ATTRIBUTES,
    "node_degree_xy": DEGREE_WEIGHTS + " Pairs follow NetworkX's `set(nodes)` order.",
    "node_attribute_xy": ATTRIBUTES,
    "average_degree_connectivity": DEGREE_WEIGHTS,
    "average_neighbor_degree": DEGREE_WEIGHTS + " A single node as `nodes` falls back (NetworkX raises `TypeError`).",
    "jaccard_coefficient": LINKS,
    "adamic_adar_index": LINKS + " Float sums follow the order of NetworkX's common-neighbor set, which rustnx replays (checked once against the running Python).",
    "resource_allocation_index": LINKS + " Float sums follow NetworkX's set order, as for `adamic_adar_index`.",
    "preferential_attachment": LINKS,
    "cn_soundarajan_hopcroft": LINKS + " " + COMMUNITIES,
    "ra_index_soundarajan_hopcroft": LINKS + " " + COMMUNITIES,
    "within_inter_cluster": LINKS + " " + COMMUNITIES,
    "common_neighbor_centrality": LINKS + " Distances come from one search per source node of `ebunch`.",
}

# Functions NetworkX only exposes under `nx.isomorphism` or `nx.bipartite`
# (or their modules).
ISOMORPHISM_ONLY = {
    "tree_isomorphism": "nx.isomorphism",
    "rooted_tree_isomorphism": "nx.isomorphism",
    "root_trees": "networkx.algorithms.isomorphism.tree_isomorphism",
    "color": "nx.bipartite",
    "sets": "nx.bipartite",
    "random_graph": "nx.bipartite",
    "gnmk_random_graph": "nx.bipartite",
    "is_bipartite_node_set": "nx.bipartite",
    "hopcroft_karp_matching": "nx.bipartite",
    "to_vertex_cover": "nx.bipartite",
    "node_redundancy": "nx.bipartite",
    "butterflies": "nx.bipartite",
}

# Functions NetworkX only exposes under `nx.dag`.
DAG_ONLY = {"v_structures", "root_to_leaf_paths", "has_cycle", "antichain_width"}


# Functions NetworkX exposes only in a submodule, not as `nx.<name>`.
SUBMODULE = {
    "modularity": "nx.community",
    "overlapping_modularity": "nx.community",
    "partition_quality": "nx.community",
    "is_partition": "nx.community",
    "is_cover": "nx.community",
    "intra_community_edges": "nx.community.quality",
    "inter_community_edges": "nx.community.quality",
    "inter_community_non_edges": "nx.community.quality",
    "greedy_modularity_communities": "nx.community",
    "naive_greedy_modularity_communities": "nx.community",
    "girvan_newman": "nx.community",
    "edge_betweenness_partition": "nx.community",
    "asyn_lpa_communities": "nx.community",
    "fast_label_propagation_communities": "nx.community",
    "asyn_fluidc": "nx.community",
    "label_propagation_communities": "nx.community",
    "is_coloring": "nx.algorithms.coloring.equitable_coloring",
    "is_equitable": "nx.algorithms.coloring.equitable_coloring",
    "branching_weight": "nx.tree",
    "greedy_branching": "nx.tree",
    "minimal_branching": "nx.tree",
    "kruskal_mst_edges": "nx.tree.mst",
    "prim_mst_edges": "nx.tree.mst",
    "boruvka_mst_edges": "nx.tree.mst",
    "is_tournament": "nx.tournament",
    "edmonds_karp": "nx.flow",
    "shortest_augmenting_path": "nx.flow",
    "dinitz": "nx.flow",
    "boykov_kolmogorov": "nx.flow",
    "preflow_push": "nx.flow",
    "build_residual_network": "nx.flow",
    "build_flow_dict": "nx.flow",
    "min_weighted_vertex_cover": "nx.approximation",
    "min_weighted_dominating_set": "nx.approximation",
    "min_edge_dominating_set": "nx.approximation",
    "min_maximal_matching": "nx.approximation",
    "greedy_tsp": "nx.approximation",
    "simulated_annealing_tsp": "nx.approximation",
    "threshold_accepting_tsp": "nx.approximation",
    "treewidth_min_fill_in": "nx.approximation",
    "one_exchange": "nx.approximation",
    "randomized_partitioning": "nx.approximation",
    "steiner_tree": "nx.approximation",
    "densest_subgraph": "nx.approximation",
}


# Batch 13: connectivity, disjoint paths and augmentation.
FLOW = "Edmonds-Karp (NetworkX's default `flow_func`) in Rust on NetworkX's auxiliary digraph, with arcs in NetworkX's order; `flow_func`, `auxiliary` and `residual` fall back."
CUT = " The set is built with NetworkX's own set operations, so it iterates in the same order."
NOTES.update({
    "node_connectivity": FLOW + " Pairs run in parallel; follows the installed NetworkX's search (3.7 changed it).",
    "edge_connectivity": FLOW + " Pairs run in parallel. Undirected graphs with self-loops fall back (NetworkX's dominating-set shortcut can then give a set-order-dependent answer).",
    "local_node_connectivity": FLOW,
    "local_edge_connectivity": FLOW,
    "average_node_connectivity": FLOW + " Pairs run in parallel.",
    "all_pairs_node_connectivity": FLOW + " Pairs run in parallel.",
    "minimum_node_cut": FLOW + " Follows the installed NetworkX's search, including the residual network reordering of 3.4 to 3.6." + CUT,
    "minimum_edge_cut": FLOW + " Replays NetworkX's dominating set (set operations and all) for undirected graphs." + CUT,
    "minimum_st_node_cut": FLOW + " Adjacent nodes give the installed version's empty result (`{}` in 3.4)." + CUT,
    "minimum_st_edge_cut": FLOW + CUT,
    "node_disjoint_paths": FLOW + " Generator; same paths in the same order. A non-numeric `cutoff` falls back.",
    "edge_disjoint_paths": FLOW + " Generator; same paths in the same order. A non-numeric `cutoff` falls back.",
    "stoer_wagner": "Replays NetworkX's heap (ties by insertion order) and contractions; the partition lists follow NetworkX's set order. `heap`, `None` weights, weights mixing ints and floats, infinite weights and int weights summing past 2**52 fall back.",
    "bridge_components": "Sets iterate in NetworkX's order (it searches a copy of G, whose rows are reordered).",
    "k_edge_components": "`k` = 1 and 2 (undirected) and `k` = 1 (directed); others fall back (NetworkX's auxiliary graph picks cuts by set order with preflow-push).",
    "k_edge_subgraphs": "As `k_edge_components`; larger `k` falls back (NetworkX pops subgraphs from a set of graphs).",
    "is_k_edge_connected": "`k` >= 3 uses `edge_connectivity` (self-loops fall back).",
    "is_locally_k_edge_connected": "Nodes not in the graph fall back.",
    "k_edge_augmentation": "`k` = 1 and 2 without `avail`; `avail` and larger `k` (a seeded greedy search) fall back.",
    "one_edge_augmentation": "Without `avail`; `avail` falls back.",
    "bridge_augmentation": "Without `avail`; `avail` falls back.",
    "unconstrained_bridge_augmentation": "Bridge components in Rust; the small tree of components is augmented with NetworkX's code.",
})
SUBMODULE.update({
    name: "nx.connectivity"
    for name in [
        "local_node_connectivity", "local_edge_connectivity", "minimum_st_node_cut",
        "minimum_st_edge_cut", "bridge_components", "is_locally_k_edge_connected",
    ]
})
SUBMODULE.update({
    name: "nx.algorithms.connectivity.edge_augmentation"
    for name in [
        "one_edge_augmentation", "unconstrained_one_edge_augmentation", "bridge_augmentation",
        "unconstrained_bridge_augmentation",
    ]
})


# Batch 19: matrices and conversion.
SPARSE = "Rust builds the COO coordinates in NetworkX's edge order; SciPy assembles the array exactly as for NetworkX (same dtype, format and index arrays)."
NODELIST = " A `nodelist` that isn't a list, or names missing or repeated nodes, falls back."
BUILDER = " Runs with `backend=\"rustnx\"` or when rustnx is listed in `nx.config.backend_priority.generators` (set by `rustnx.enable()`). `create_using` other than `None`, `nx.Graph` or `nx.DiGraph` falls back."
NOTES.update({
    "to_scipy_sparse_array": SPARSE + NODELIST + " Weights other than plain ints and floats, `dtype`s that NumPy would convert differently from a list (narrower ints), and a `nodelist` under half the nodes with a format other than CSR or CSC (NetworkX's subgraph then iterates in set order) fall back.",
    "adjacency_matrix": "As `to_scipy_sparse_array`.",
    "laplacian_matrix": "The adjacency matrix as `to_scipy_sparse_array`, then NetworkX's own SciPy arithmetic for the installed version (3.6 changed how the degree matrix is built).",
    "incidence_matrix": "Builds the CSR arrays of NetworkX's LIL matrix in Rust (zero weights stay unstored). `edgelist`, `dtype` other than float64, and endpoints missing from `nodelist` fall back.",
    "to_numpy_array": "Entries from Rust, assigned with NumPy as NetworkX does." + NODELIST + " Structured dtypes, `None` weights, and result dtypes other than float64 (or int64 with int weights) fall back.",
    "biadjacency_matrix": SPARSE + " A `row_order` that isn't a list or tuple falls back.",
    "to_dict_of_lists": "Neighbor objects come from the graph's own adjacency dicts. A `nodelist` that isn't a list, tuple, set or dict, or names missing nodes, falls back.",
    "number_of_selfloops": "Dispatchable from NetworkX 3.5 on.",
    "is_weighted": "Reads the original graph's edge data in Rust. Dispatchable from NetworkX 3.5 on; native graphs fall back.",
    "is_negatively_weighted": "`None` weights and `weight=None` fall back.",
    "get_node_attributes": "Reads the original graph's node data in Rust. Dispatchable from NetworkX 3.5 on.",
    "get_edge_attributes": "Reads the original graph's edge data in Rust (values of any type). Dispatchable from NetworkX 3.5 on; native graphs fall back.",
    "relabel_nodes": "Fills the new graph's dicts from Rust, row by row in the order NetworkX's `add_edges_from` does. `copy=False` (it changes the input), labels that collide, `None` or unhashable labels, mappings that are neither dicts nor callables, multigraphs and graph subclasses fall back.",
    "convert_node_labels_to_integers": "Through `relabel_nodes`.",
    "from_dict_of_lists": "Fills the new graph's dicts from Rust as `add_edges_from` does." + BUILDER,
    "from_dict_of_dicts": "As `from_dict_of_lists`; `multigraph_input` and non-dict data fall back." + BUILDER,
    "from_edgelist": "As `from_dict_of_lists`; one-shot iterators and edges other than 2-tuples and 3-tuples with a dict fall back." + BUILDER,
    "from_numpy_array": "Edge positions and values from NumPy, the graph filled from Rust. Long double, string and structured dtypes fall back." + BUILDER,
    "from_scipy_sparse_array": "Edges in NetworkX's order for each format; DOK arrays fall back." + BUILDER,
    "from_biadjacency_matrix": "As `from_scipy_sparse_array`; `row_order` and `column_order` (NetworkX 3.7) fall back." + BUILDER,
})
SUBMODULE.update({"biadjacency_matrix": "nx.bipartite", "from_biadjacency_matrix": "nx.bipartite"})
# Batch 17: deterministic generators.
GENERATOR = "Generators take no graph: they run in rustnx with `backend=\"rustnx\"` or `nx.config.backend_priority.generators` (set by `rustnx.enable()`). Returns a plain NetworkX graph built in Rust, identical to NetworkX's (node and row order, attributes, keys). `create_using` may be a NetworkX graph class or instance (instances that NetworkX would fill and then reject fall back)."
NODES = " An int, or a list, tuple, range or str of distinct nodes; other iterables fall back."
INTS = " Int arguments only; others fall back."
NOTES.update({
    name: GENERATOR + INTS
    for name in [
        "ladder_graph", "lollipop_graph", "barbell_graph", "tadpole_graph", "full_rary_tree",
        "balanced_tree", "binomial_tree", "complete_bipartite_graph", "complete_multipartite_graph",
        "turan_graph", "caveman_graph", "connected_caveman_graph", "ring_of_cliques",
        "windmill_graph", "sudoku_graph", "LCF_graph", "generalized_petersen_graph",
        "dorogovtsev_goltsev_mendes_graph", "mycielski_graph", "hypercube_graph",
    ]
})
NOTES.update({
    name: GENERATOR + NODES
    for name in ["empty_graph", "complete_graph", "cycle_graph", "path_graph", "star_graph", "wheel_graph", "grid_2d_graph"]
})
NOTES.update({
    "circular_ladder_graph": GENERATOR + INTS + " `n` < 2 falls back before NetworkX 3.7 (which raises).",
    "circulant_graph": GENERATOR + " Int `n` and a list of int offsets; follows the installed NetworkX's edge order (3.6 changed it).",
    "grid_graph": GENERATOR + " Dimensions are ints or lists of distinct ints; `periodic` is a bool or a list.",
    "hexagonal_lattice_graph": GENERATOR + " Periodic lattices fall back.",
    "triangular_lattice_graph": GENERATOR + " Periodic lattices fall back.",
    "paley_graph": GENERATOR + INTS + " The squares are iterated in CPython's set order (replayed in Rust).",
    "kneser_graph": GENERATOR + INTS + " Replays CPython's set difference order; large graphs (over 2**24 nodes) fall back.",
})
# Seeded random generators (batch 18): no graph argument, so NetworkX sends
# them to rustnx only with `backend="rustnx"` or with rustnx listed in
# `nx.config.backend_priority.generators` (set by `rustnx.enable()`).
RANDOM = (
    "Replays NetworkX's draws on CPython's `random.Random` (any `seed` NetworkX "
    "turns into one, including `None`: the global generator, left where NetworkX "
    "would leave it); NumPy generators fall back. `create_using` takes `None` or a "
    "NetworkX graph class; instances and subclasses fall back."
)
NOTES.update({
    "gnp_random_graph": RANDOM,
    "fast_gnp_random_graph": RANDOM,
    "gnm_random_graph": RANDOM,
    "dense_gnm_random_graph": RANDOM,
    "barabasi_albert_graph": RANDOM + " `initial_graph` falls back.",
    "dual_barabasi_albert_graph": RANDOM + " `initial_graph` falls back.",
    "extended_barabasi_albert_graph": RANDOM,
    "watts_strogatz_graph": RANDOM,
    "newman_watts_strogatz_graph": RANDOM,
    "connected_watts_strogatz_graph": RANDOM,
    "powerlaw_cluster_graph": RANDOM + " Replays the order NetworkX pops its target set in.",
    "random_regular_graph": RANDOM + " Edges in the iteration order of NetworkX's set of pairs.",
    "gn_graph": RANDOM + " A `kernel` falls back. Follows the installed NetworkX's `cumulative_distribution` (3.6 changed its rounding).",
    "gnr_graph": RANDOM,
    "gnc_graph": RANDOM,
    "random_uniform_k_out_graph": RANDOM,
    "random_lobster_graph": RANDOM + " NetworkX 3.6+.",
    "random_lobster": RANDOM + " NetworkX 3.4 and 3.5 (3.6's deprecated alias runs in NetworkX, which calls `random_lobster_graph`).",
    "random_tournament": RANDOM,
    "stochastic_block_model": RANDOM + " `nodelist` falls back. Follows the installed NetworkX's loop (3.7 stopped drawing twice for diagonal blocks).",
    "random_partition_graph": RANDOM,
    "planted_partition_graph": RANDOM,
    "random_geometric_graph": RANDOM + " Given `pos` falls back. Pairs whose distance is within 1e-9 (relative) of the radius make the call fall back, since SciPy's KD-tree may round them either way.",
    "waxman_graph": RANDOM + " A `metric` falls back. Without `L`, a draw within rounding error of its threshold makes the call fall back (`math.dist` and Rust's `hypot` may differ in the last bit).",
    "random_graph": RANDOM + " As `nx.bipartite.random_graph`. `p >= 1` falls back.",
    "gnmk_random_graph": RANDOM + " As `nx.bipartite.gnmk_random_graph`. `k >= n * m` falls back.",
})
# Batch 20: readers and parsers.
READ = "Reads a binary file or path (`.gz` and `.bz2` too) decoded as UTF-8, ASCII or Latin-1; other encodings, text-mode files and anything rustnx declines are read by NetworkX from the same position."
PARSE = "Lines as a list or tuple of `str`, or an iterator (read once; if rustnx declines, NetworkX's code runs on the same lines)."
CREATE = " `create_using`: `None` or a plain NetworkX graph class or instance."
LITERAL = " `data=True` handles flat dicts of `str`, `int`, `float`, `bool` and `None` values; other edge data falls back."
NOTES.update({
    "parse_edgelist": PARSE + CREATE + LITERAL + " `nodetype` and typed `data`: `int`, `float` or `str`.",
    "read_edgelist": READ + CREATE + LITERAL,
    "read_weighted_edgelist": READ + CREATE,
    "bipartite_parse_edgelist": PARSE + CREATE + LITERAL,
    "bipartite_read_edgelist": READ + CREATE + LITERAL,
    "parse_adjlist": PARSE + CREATE + " `nodetype`: `int`, `float` or `str`.",
    "read_adjlist": READ + CREATE,
    "parse_multiline_adjlist": "Iterators only (NetworkX calls `next(lines)`, so lists fail there); see `parse_edgelist`." + CREATE + LITERAL,
    "read_multiline_adjlist": READ + CREATE + LITERAL,
    "parse_leda": PARSE + " A `str` is split at newlines as in NetworkX.",
    "read_leda": READ,
    "parse_pajek": PARSE + " A `str` is split at newlines as in NetworkX. Replays `shlex.split` (quotes and escapes); `*matrix` sections fall back.",
    "read_pajek": READ + " `*matrix` sections fall back.",
    "from_graph6_bytes": "`bytes` input; follows the installed NetworkX on trailing newlines (ignored from 3.5).",
    "read_graph6": READ.split(" decoded")[0] + "; one graph or a list, as in NetworkX.",
    "from_sparse6_bytes": "`bytes` input. Returns a `MultiGraph` when there are parallel edges, as NetworkX does.",
    "read_sparse6": READ.split(" decoded")[0] + "; one graph or a list, as in NetworkX.",
})
JSON = "Replays NetworkX's `add_node` / `add_edge` calls on the same objects. Nodes and keys must be `str`, `int`, `float`, `bool`, or tuples of those; plain `dict` and `list` input only; anything else falls back."
NOTES.update({
    "node_link_graph": JSON + " `G.graph` is `data[\"graph\"]` itself, as in NetworkX. In 3.4 and 3.5, leaving out `edges` (which warns there) falls back.",
    "adjacency_graph": JSON,
    "cytoscape_graph": JSON,
    "tree_graph": JSON + " Trees deeper than 200 levels fall back (NetworkX recurses per level).",
})
GML = "`destringizer=None` only. Node ids, labels and keys must be ints, floats, strings or `()`; edges must name a node by its id's own type. Named character references other than `&amp;`, `&lt;`, `&gt;` and `&quot;`, and ints beyond 64 bits, fall back."
NOTES.update({
    "parse_gml": PARSE + " A `str` is split at line boundaries as in NetworkX. " + GML,
    "read_gml": READ.replace("decoded as UTF-8, ASCII or Latin-1", "decoded as ASCII") + " " + GML,
})
SUBMODULE.update({
    name: "nx.readwrite.json_graph"
    for name in ["node_link_graph", "adjacency_graph", "cytoscape_graph", "tree_graph"]
})
QUALIFIED.update({
    "bipartite_parse_edgelist": "nx.bipartite.parse_edgelist",
    "bipartite_read_edgelist": "nx.bipartite.read_edgelist",
})
# Batch 21: operators and structure.
OPERATOR = "Builds the result's dicts in Rust with the exact sequence of NetworkX's `add_nodes_from` / `add_edges_from` calls: same node and adjacency order, attribute dicts copied (or shared) as NetworkX does."
PLAIN = " Plain `Graph` and `DiGraph` inputs and results; subclasses, graph views and multigraphs fall back."
HOLES = "Explicit `nodes`; in 3.5+ `nodes=None` uses SciPy and falls back. Sums follow `set(nx.all_neighbors(G, v))` order, which rustnx replays, and Python's `sum()`; weights must be ints or floats."
SECTIONS.append(("Operators, products and structure", [
    "union", "union_all", "compose", "compose_all", "disjoint_union", "disjoint_union_all",
    "full_join", "intersection", "intersection_all", "reverse", "moral_graph", "line_graph",
    "ego_graph", "cartesian_product", "tensor_product", "strong_product",
    "lexicographic_product", "rooted_product", "corona_product", "constraint",
    "effective_size", "local_constraint", "tree_broadcast_center", "tree_broadcast_time",
    "density", "bipartite_degree_centrality", "projected_graph", "weighted_projected_graph",
]))
NOTES.update({
    "union": OPERATOR + PLAIN + " `rename` prefixes that make two nodes' labels collide fall back.",
    "union_all": OPERATOR + PLAIN + " `rename` prefixes that make two nodes' labels collide fall back.",
    "compose": OPERATOR + PLAIN,
    "compose_all": OPERATOR + PLAIN,
    "disjoint_union": OPERATOR + PLAIN + " Replays the reordering of NetworkX's relabelled copies.",
    "disjoint_union_all": OPERATOR + PLAIN + " Replays the reordering of NetworkX's relabelled copies.",
    "full_join": OPERATOR + PLAIN,
    "intersection": "Replays NetworkX's node and edge sets (CPython's set table and tuple hash, checked once against the running Python), so nodes and edges come in the same order." + PLAIN,
    "intersection_all": "As `intersection`." + PLAIN,
    "reverse": OPERATOR + " Attributes are deep-copied, as in NetworkX. `copy=False` (a view) falls back.",
    "moral_graph": OPERATOR + " Attributes are deep-copied (`G.to_undirected()`).",
    "line_graph": "Undirected line graphs replay NetworkX's set of node pairs (CPython's set table and tuple hash). `create_using` falls back.",
    "ego_graph": "Replays the subgraph view's node order (`set(sp)` when the subgraph is under half the graph). `undirected=True` falls back.",
    "cartesian_product": OPERATOR + " Attribute tuples are built with NetworkX's set order of keys.",
    "tensor_product": OPERATOR + " Attribute tuples are built with NetworkX's set order of keys.",
    "strong_product": OPERATOR + " Attribute tuples are built with NetworkX's set order of keys.",
    "lexicographic_product": OPERATOR + " Attribute tuples are built with NetworkX's set order of keys.",
    "rooted_product": OPERATOR + " A `root` equal to, but not the same object as, H's node falls back.",
    "corona_product": OPERATOR,
    "constraint": HOLES,
    "effective_size": HOLES + " Undirected unweighted graphs count edges of each ego graph, as NetworkX does.",
    "local_constraint": "As `constraint`. Nodes not in the graph fall back.",
    "tree_broadcast_center": "Replays NetworkX's set of leaves (ties go by set order). Non-trees fall back in 3.4 (it doesn't raise there).",
    "tree_broadcast_time": "Missing nodes fall back in 3.4.",
    "density": "As `nx.bipartite.density`.",
    "bipartite_degree_centrality": "As `nx.bipartite.degree_centrality`; dict order follows NetworkX's sets.",
    "projected_graph": "As `nx.bipartite.projected_graph`; replays the set of second neighbors. `multigraph=True`, and `nodes` that is not a container, fall back.",
    "weighted_projected_graph": "As `nx.bipartite.weighted_projected_graph`; replays the set of second neighbors.",
})
QUALIFIED.update({
    "density": "nx.bipartite.density",
    "bipartite_degree_centrality": "nx.bipartite.degree_centrality",
    "projected_graph": "nx.bipartite.projected_graph",
    "weighted_projected_graph": "nx.bipartite.weighted_projected_graph",
})


def location(name):
    if name in DAG_ONLY:
        return "nx.dag"
    return ISOMORPHISM_ONLY.get(name) or SUBMODULE.get(name, "nx")


def parameters(name):
    params = list(inspect.signature(getattr(algorithms, name)).parameters)
    if params and params[0] in ("G", "T", "flowG"):
        params = params[1:]  # drop the graph
    # Drop internal keywords (the flow functions' `_s` and `_t` are NetworkX's).
    params = [p for p in params if not p.startswith("_") or p in ("_s", "_t")]
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
            qualified = QUALIFIED.get(name, f"{location(name)}.{name}")
            out.append(f"| `{qualified}` | {parameters(name)} | {multi} | {small} | {NOTES.get(name, '')} |")
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
