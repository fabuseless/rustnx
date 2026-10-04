"""The ``networkx.backends`` entry point for rustnx.

NetworkX looks up algorithms on this module by name, and calls
``convert_from_nx`` / ``convert_to_nx`` / ``can_run`` / ``should_run``.
"""

import functools
import inspect

import networkx as nx

from . import algorithms
from .graph import RustnxGraph, from_networkx, to_networkx

# Parameters that may hold an edge attribute name (or a callable).
_WEIGHT_PARAMS = ("weight", "distance")

# Linear-time algorithms where, on small graphs, converting to rustnx and
# dispatching costs more than NetworkX spends running the algorithm.
_LINEAR_TIME = {
    "all_shortest_paths",
    "ancestors",
    "articulation_points",
    "astar_path",
    "astar_path_length",
    "attracting_components",
    "bfs_edges",
    "bfs_labeled_edges",
    "bfs_layers",
    "bfs_predecessors",
    "bfs_successors",
    "bfs_tree",
    "biconnected_component_edges",
    "biconnected_components",
    "bidirectional_dijkstra",
    "bidirectional_shortest_path",
    "boundary_expansion",
    "branching_weight",
    "bridges",
    "build_residual_network",
    "chain_decomposition",
    "check_planarity",
    "check_planarity_recursive",
    "chordal_graph_treewidth",
    "color",
    "condensation",
    "conductance",
    "connected_components",
    "connected_dominating_set",
    "core_number",
    "cost_of_flow",
    "cut_size",
    "cycle_basis",
    "dag_longest_path",
    "dag_longest_path_length",
    "degree_centrality",
    "descendants",
    "descendants_at_distance",
    "dfs_edges",
    "dfs_labeled_edges",
    "dfs_postorder_nodes",
    "dfs_predecessors",
    "dfs_preorder_nodes",
    "dfs_successors",
    "dfs_tree",
    "dijkstra_path",
    "dijkstra_path_length",
    "dijkstra_predecessor_and_distance",
    "dominance_frontiers",
    "edge_bfs",
    "edge_boundary",
    "edge_dfs",
    "edge_expansion",
    "eulerian_circuit",
    "eulerian_path",
    "faster_could_be_isomorphic",
    "find_cycle",
    "generic_bfs_edges",
    "greedy_color",
    "group_closeness_centrality",
    "group_degree_centrality",
    "group_in_degree_centrality",
    "group_out_degree_centrality",
    "has_bridges",
    "has_cycle",
    "has_eulerian_path",
    "has_path",
    "immediate_dominators",
    "in_degree_centrality",
    "intersection_array",
    "is_aperiodic",
    "is_arborescence",
    "is_attracting_component",
    "is_biconnected",
    "is_bipartite",
    "is_bipartite_node_set",
    "is_branching",
    "is_chordal",
    "is_coloring",
    "is_connected",
    "is_connected_dominating_set",
    "is_d_separator",
    "is_directed_acyclic_graph",
    "is_distance_regular",
    "is_dominating_set",
    "is_equitable",
    "is_eulerian",
    "is_forest",
    "is_k_regular",
    "is_matching",
    "is_maximal_matching",
    "is_perfect_matching",
    "is_planar",
    "is_regular",
    "is_semiconnected",
    "is_semieulerian",
    "is_simple_path",
    "is_strongly_connected",
    "is_strongly_regular",
    "is_tournament",
    "is_tree",
    "is_weakly_connected",
    "isolates",
    "k_core",
    "k_corona",
    "k_crust",
    "k_shell",
    "kosaraju_strongly_connected_components",
    "kruskal_mst_edges",
    "label_propagation_communities",
    "lexicographical_topological_sort",
    "local_bridges",
    "local_reaching_centrality",
    "maximal_matching",
    "maximum_spanning_edges",
    "maximum_spanning_tree",
    "minimum_spanning_edges",
    "minimum_spanning_tree",
    "mixing_expansion",
    "multi_source_dijkstra",
    "multi_source_dijkstra_path",
    "multi_source_dijkstra_path_length",
    "node_boundary",
    "node_connected_component",
    "node_expansion",
    "normalized_cut_size",
    "number_attracting_components",
    "number_connected_components",
    "number_of_isolates",
    "number_strongly_connected_components",
    "number_weakly_connected_components",
    "onion_layers",
    "out_degree_centrality",
    "predecessor",
    "root_trees",
    "score_sequence",
    "sets",
    "shortest_path",
    "single_source_dijkstra",
    "single_source_dijkstra_path",
    "single_source_dijkstra_path_length",
    "single_source_shortest_path",
    "single_source_shortest_path_length",
    "single_target_shortest_path",
    "single_target_shortest_path_length",
    "strongly_connected_components",
    "to_prufer_sequence",
    "topological_generations",
    "topological_sort",
    "tree_centroid",
    "triadic_census",
    "v_structures",
    "volume",
    "weakly_connected_components",
}
SMALL_GRAPH_NODES = 500

# Functions whose NetworkX code sees a multigraph only through `G[v]`
# (neighbors once each) and, for weights, `_weight_function` (the minimum over
# parallel edges). rustnx converts multigraphs that way for these alone.
MULTIGRAPH_FUNCTIONS = {
    "all_pairs_all_shortest_paths",
    "all_pairs_bellman_ford_path",
    "all_pairs_bellman_ford_path_length",
    "all_pairs_dijkstra",
    "all_pairs_dijkstra_path",
    "all_pairs_dijkstra_path_length",
    "all_pairs_shortest_path",
    "all_pairs_shortest_path_length",
    "all_shortest_paths",
    "all_triangles",
    "ancestors",
    "articulation_points",
    "astar_path",
    "astar_path_length",
    "attracting_components",
    "average_shortest_path_length",
    "bellman_ford_path",
    "bellman_ford_path_length",
    "bellman_ford_predecessor_and_distance",
    "betweenness_centrality",
    "betweenness_centrality_subset",
    "bfs_edges",
    "bfs_labeled_edges",
    "bfs_layers",
    "bfs_predecessors",
    "bfs_successors",
    "bfs_tree",
    "biconnected_component_edges",
    "biconnected_components",
    "bidirectional_dijkstra",
    "bidirectional_shortest_path",
    "bipartite_closeness_centrality",
    "boundary_expansion",
    "center",
    "closeness_centrality",
    "color",
    "condensation",
    "connected_components",
    "descendants",
    "descendants_at_distance",
    "dfs_edges",
    "dfs_labeled_edges",
    "dfs_postorder_nodes",
    "dfs_predecessors",
    "dfs_preorder_nodes",
    "dfs_successors",
    "dfs_tree",
    "diameter",
    "dijkstra_path",
    "dijkstra_path_length",
    "dijkstra_predecessor_and_distance",
    "dominance_frontiers",
    "eccentricity",
    "find_negative_cycle",
    "generic_bfs_edges",
    "harmonic_centrality",
    "harmonic_diameter",
    "has_path",
    "hopcroft_karp_matching",
    "immediate_dominators",
    "is_attracting_component",
    "is_biconnected",
    "is_bipartite",
    "is_bipartite_node_set",
    "is_connected",
    "is_dominating_set",
    "is_matching",
    "is_perfect_matching",
    "is_semiconnected",
    "is_simple_path",
    "is_strongly_connected",
    "is_weakly_connected",
    "isolates",
    "johnson",
    "kosaraju_strongly_connected_components",
    "multi_source_dijkstra",
    "multi_source_dijkstra_path",
    "multi_source_dijkstra_path_length",
    "negative_edge_cycle",
    "newman_betweenness_centrality",
    "node_boundary",
    "node_connected_component",
    "node_expansion",
    "node_redundancy",
    "number_attracting_components",
    "number_connected_components",
    "number_of_isolates",
    "number_strongly_connected_components",
    "number_weakly_connected_components",
    "periphery",
    "predecessor",
    "radius",
    "sets",
    "shortest_path",
    "shortest_path_length",
    "single_source_all_shortest_paths",
    "single_source_bellman_ford",
    "single_source_bellman_ford_path",
    "single_source_bellman_ford_path_length",
    "single_source_dijkstra",
    "single_source_dijkstra_path",
    "single_source_dijkstra_path_length",
    "single_source_shortest_path",
    "single_source_shortest_path_length",
    "single_target_shortest_path",
    "single_target_shortest_path_length",
    "square_clustering",
    "strongly_connected_components",
    "transitive_reduction",
    "weakly_connected_components",
    "wiener_index",
}

# Functions that return subgraphs of the original NetworkX graph, or (the flow
# functions) read its edge attributes there.
_BUILDS_FROM_SOURCE = {
    "boruvka_mst_edges",
    "boykov_kolmogorov",
    "build_residual_network",
    "dinitz",
    "edmonds_karp",
    "gomory_hu_tree",
    "k_core",
    "k_corona",
    "k_crust",
    "k_shell",
    "k_truss",
    "kruskal_mst_edges",
    "maximum_branching",
    "maximum_flow",
    "maximum_flow_value",
    "maximum_spanning_arborescence",
    "maximum_spanning_edges",
    "maximum_spanning_tree",
    "minimal_branching",
    "minimum_branching",
    "minimum_cut",
    "minimum_cut_value",
    "minimum_spanning_arborescence",
    "minimum_spanning_edges",
    "minimum_spanning_tree",
    "partition_spanning_tree",
    "preflow_push",
    "prim_mst_edges",
    "shortest_augmenting_path",
    "transitive_closure",
    "transitive_closure_dag",
}


def convert_from_nx(
    G,
    edge_attrs=None,
    node_attrs=None,
    preserve_edge_attrs=False,
    preserve_node_attrs=False,
    preserve_graph_attrs=False,
    name=None,
    graph_name=None,
):
    if isinstance(G, RustnxGraph):
        return G
    multigraph_ok = name in MULTIGRAPH_FUNCTIONS
    if G.is_multigraph() and not multigraph_ok:
        raise NotImplementedError(f"rustnx does not support multigraphs in {name}")
    if preserve_edge_attrs is True:
        if name in _BUILDS_FROM_SOURCE:
            # These build their result from the original NetworkX graph, so
            # its attributes don't need to be copied into Rust.
            return from_networkx(G, edge_attrs or {}, multigraph_ok=multigraph_ok)
        # Arbitrary edge data (e.g. for callable weights) isn't stored in Rust.
        raise NotImplementedError("rustnx only stores numeric edge attributes")
    if isinstance(preserve_edge_attrs, dict):
        edge_attrs = preserve_edge_attrs.get(graph_name) or edge_attrs
    return from_networkx(G, edge_attrs or {}, multigraph_ok=multigraph_ok)


def convert_to_nx(obj, *, name=None):
    if isinstance(obj, RustnxGraph):
        return to_networkx(obj)
    return obj


_SIMPLE_TYPES = (bool, int, float, str)


def _is_default(value, param):
    if value is param.default:
        return True
    default = param.default
    return (
        isinstance(default, _SIMPLE_TYPES)
        and type(value) is type(default)
        and value == default
    )


def _nx_function(name):
    """The installed NetworkX's dispatchable function called ``name`` (some,
    like ``label_propagation_communities``, live in subpackages)."""
    registry = getattr(nx.utils.backends, "_registered_algorithms", {})
    return registry[name] if name in registry else getattr(nx, name)


@functools.cache
def _nx_signature(name):
    return inspect.signature(_nx_function(name))


# Parameters NetworkX renamed between releases: {function: {old: ours}}.
_RENAMED_PARAMS = {
    "vf2pp_is_isomorphic": {"G1": "FG", "G2": "SG"},  # renamed in 3.7
}


def _bind(name, args, kwargs):
    """Bind a call to the installed NetworkX function's signature.

    Returns ``(arguments, unsupported)``: the arguments rustnx implements,
    and the names of any other arguments the caller actually set. NetworkX
    adds parameters in new releases (e.g. ``closeness_centrality(sp=...)`` in
    3.7) and passes every parameter to backends, so rustnx must tolerate ones
    it doesn't know about. Raises ``TypeError`` if the call doesn't bind.
    """
    nx_sig = _nx_signature(name)
    bound = nx_sig.bind(*args, **kwargs)
    ours = _OUR_PARAMS[name]
    renamed = _RENAMED_PARAMS.get(name, {})
    arguments = {}
    unsupported = []
    for key, value in bound.arguments.items():
        param = nx_sig.parameters[key]
        if key == "backend":
            continue
        if param.kind is param.VAR_KEYWORD:
            if key in ours:  # e.g. maximum_flow's options for its flow_func
                arguments.update(value)
            else:
                unsupported.extend(value)  # backend-specific keywords
        elif renamed.get(key, key) in ours:
            arguments[renamed.get(key, key)] = value
        elif not _is_default(value, param):
            unsupported.append(key)
    return arguments, unsupported


def _make_entry(name):
    func = getattr(algorithms, name)

    @functools.wraps(func)
    def entry(*args, **kwargs):
        try:
            arguments, unsupported = _bind(name, args, kwargs)
        except TypeError:
            return func(*args, **kwargs)  # raise the usual error
        if unsupported:
            raise NotImplementedError(
                f"rustnx does not support: {', '.join(unsupported)}"
            )
        if name not in MULTIGRAPH_FUNCTIONS and any(
            getattr(value, "_multigraph", False) for value in arguments.values()
        ):
            # NetworkX caches one conversion per graph, whichever function
            # asked for it.
            raise NotImplementedError(f"rustnx does not support multigraphs in {name}")
        return func(**arguments)

    return entry


_OUR_PARAMS = {
    name: set(inspect.signature(getattr(algorithms, name)).parameters)
    for name in algorithms.__all__
}
globals().update({name: _make_entry(name) for name in algorithms.__all__})


def can_run(name, args, kwargs):
    if name not in _OUR_PARAMS:
        return False
    try:
        arguments, unsupported = _bind(name, args, kwargs)
    except TypeError:
        return True  # let the call raise the usual error
    if unsupported:
        return f"unsupported arguments: {', '.join(unsupported)}"
    G = arguments.get("G")
    if G is not None and G.is_multigraph() and name not in MULTIGRAPH_FUNCTIONS:
        return "multigraphs are not supported by this function"
    for param in _WEIGHT_PARAMS:
        if callable(arguments.get(param)):
            return "callable weights are not supported"
    return True


def should_run(name, args, kwargs):
    """Decline automatic dispatch where NetworkX is faster.

    Only consulted when rustnx is chosen via ``nx.config.backend_priority``;
    an explicit ``backend="rustnx"`` always runs rustnx.
    """
    if name not in _LINEAR_TIME:
        return True
    G = args[0] if args else kwargs.get("G")
    if isinstance(G, RustnxGraph) or G is None:
        return True
    if len(G) < SMALL_GRAPH_NODES:
        return "graph is small; NetworkX is faster"
    return True
