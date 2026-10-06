"""The ``networkx.backends`` entry point for rustnx.

NetworkX looks up algorithms on this module by name, and calls
``convert_from_nx`` / ``convert_to_nx`` / ``can_run`` / ``should_run``.
"""

import functools
import inspect
import os
import sys

import networkx as nx

from . import algorithms
from ._config import exact_floats, inexact_message
from ._config import logger as _float_logger
from .graph import RustnxGraph, from_networkx, to_networkx

# Parameters that may hold an edge attribute name (or a callable).
_WEIGHT_PARAMS = ("weight", "distance")

# Linear-time algorithms where, on small graphs, converting to rustnx and
# dispatching costs more than NetworkX spends running the algorithm.
_LINEAR_TIME = {
    "adjacency_matrix",
    "all_shortest_paths",
    "ancestors",
    "approximate_diameter",
    "articulation_points",
    "astar_path",
    "astar_path_length",
    "asyn_lpa_communities",
    "attracting_components",
    "attribute_assortativity_coefficient",
    "attribute_mixing_dict",
    "attribute_mixing_matrix",
    "average_degree_connectivity",
    "average_neighbor_degree",
    "bfs_edges",
    "bfs_labeled_edges",
    "bfs_layers",
    "bfs_predecessors",
    "bfs_successors",
    "bfs_tree",
    "biadjacency_matrix",
    "biconnected_component_edges",
    "biconnected_components",
    "bidirectional_dijkstra",
    "bidirectional_shortest_path",
    "bipartite_degree_centrality",
    "boundary_expansion",
    "branching_weight",
    "bridge_augmentation",
    "bridge_components",
    "bridges",
    "build_auxiliary_edge_connectivity",
    "build_auxiliary_node_connectivity",
    "build_flow_dict",
    "build_residual_network",
    "chain_decomposition",
    "check_planarity",
    "check_planarity_recursive",
    "chordal_graph_treewidth",
    "color",
    "compose",
    "condensation",
    "conductance",
    "connected_components",
    "connected_dominating_set",
    "contracted_edge",
    "contracted_nodes",
    "convert_node_labels_to_integers",
    "core_number",
    "cost_of_flow",
    "cut_size",
    "cycle_basis",
    "dag_longest_path",
    "dag_longest_path_length",
    "degree_assortativity_coefficient",
    "degree_centrality",
    "degree_mixing_dict",
    "degree_mixing_matrix",
    "degree_pearson_correlation_coefficient",
    "density",
    "descendants",
    "descendants_at_distance",
    "dfs_edges",
    "dfs_labeled_edges",
    "dfs_postorder_nodes",
    "dfs_predecessors",
    "dfs_preorder_nodes",
    "dfs_successors",
    "dfs_tree",
    "difference",
    "dijkstra_path",
    "dijkstra_path_length",
    "dijkstra_predecessor_and_distance",
    "disjoint_union",
    "dominance_frontiers",
    "dominating_set",
    "edge_bfs",
    "edge_boundary",
    "edge_dfs",
    "edge_expansion",
    "efficiency",
    "ego_graph",
    "eulerian_circuit",
    "eulerian_path",
    "fast_label_propagation_communities",
    "faster_could_be_isomorphic",
    "find_cycle",
    "flow_hierarchy",
    "full_join",
    "generic_bfs_edges",
    "geometric_edges",
    "get_edge_attributes",
    "get_node_attributes",
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
    "incidence_matrix",
    "intersection",
    "intersection_array",
    "intra_community_edges",
    "inverse_line_graph",
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
    "is_cover",
    "is_d_separator",
    "is_directed_acyclic_graph",
    "is_distance_regular",
    "is_dominating_set",
    "is_equitable",
    "is_eulerian",
    "is_forest",
    "is_k_edge_connected",
    "is_k_regular",
    "is_matching",
    "is_maximal_matching",
    "is_negatively_weighted",
    "is_partition",
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
    "is_weighted",
    "isolates",
    "k_core",
    "k_corona",
    "k_crust",
    "k_edge_augmentation",
    "k_edge_components",
    "k_edge_subgraphs",
    "k_shell",
    "kosaraju_strongly_connected_components",
    "kruskal_mst_edges",
    "label_propagation_communities",
    "laplacian_matrix",
    "lexicographical_topological_sort",
    "line_graph",
    "local_bridges",
    "local_reaching_centrality",
    "maximal_matching",
    "maximum_spanning_edges",
    "maximum_spanning_tree",
    "min_edge_dominating_set",
    "min_maximal_matching",
    "min_weighted_dominating_set",
    "min_weighted_vertex_cover",
    "minimum_spanning_edges",
    "minimum_spanning_tree",
    "mixing_expansion",
    "modularity",
    "moral_graph",
    "multi_source_dijkstra",
    "multi_source_dijkstra_path",
    "multi_source_dijkstra_path_length",
    "mycielskian",
    "node_attribute_xy",
    "node_boundary",
    "node_connected_component",
    "node_degree_xy",
    "node_expansion",
    "normalized_cut_size",
    "number_attracting_components",
    "number_connected_components",
    "number_of_isolates",
    "number_of_selfloops",
    "number_strongly_connected_components",
    "number_weakly_connected_components",
    "numeric_assortativity_coefficient",
    "one_edge_augmentation",
    "onion_layers",
    "out_degree_centrality",
    "overall_reciprocity",
    "overlapping_modularity",
    "partition_quality",
    "predecessor",
    "randomized_partitioning",
    "reciprocity",
    "relabel_nodes",
    "reverse",
    "rich_club_coefficient",
    "root_trees",
    "s_metric",
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
    "stochastic_graph",
    "strongly_connected_components",
    "symmetric_difference",
    "to_dict_of_lists",
    "to_numpy_array",
    "to_prufer_sequence",
    "to_scipy_sparse_array",
    "topological_generations",
    "topological_sort",
    "tree_centroid",
    "triadic_census",
    "unconstrained_bridge_augmentation",
    "unconstrained_one_edge_augmentation",
    "union",
    "v_structures",
    "volume",
    "voronoi_cells",
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
    "approximate_diameter",
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
    "bipartite_average_clustering",
    "bipartite_betweenness_centrality",
    "bipartite_closeness_centrality",
    "boundary_expansion",
    "build_auxiliary_edge_connectivity",
    "build_auxiliary_node_connectivity",
    "center",
    "closeness_centrality",
    "closeness_vitality",
    "color",
    "condensation",
    "connected_components",
    "degree_centrality",
    "density",
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
    "dominating_set",
    "eccentricity",
    "efficiency",
    "find_cliques",
    "find_cliques_recursive",
    "find_negative_cycle",
    "generic_bfs_edges",
    "geometric_edges",
    "global_efficiency",
    "group_degree_centrality",
    "group_in_degree_centrality",
    "group_out_degree_centrality",
    "harmonic_centrality",
    "harmonic_diameter",
    "has_eulerian_path",
    "has_path",
    "hopcroft_karp_matching",
    "immediate_dominators",
    "in_degree_centrality",
    "is_attracting_component",
    "is_biconnected",
    "is_bipartite",
    "is_bipartite_node_set",
    "is_connected",
    "is_dominating_set",
    "is_matching",
    "is_perfect_matching",
    "is_semiconnected",
    "is_semieulerian",
    "is_simple_path",
    "is_strongly_connected",
    "is_weakly_connected",
    "isolates",
    "johnson",
    "kosaraju_strongly_connected_components",
    "kruskal_mst_edges",
    "latapy_clustering",
    "local_efficiency",
    "maximum_spanning_edges",
    "maximum_spanning_tree",
    "min_weighted_dominating_set",
    "min_weighted_vertex_cover",
    "minimum_spanning_edges",
    "minimum_spanning_tree",
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
    "number_of_selfloops",
    "number_strongly_connected_components",
    "number_weakly_connected_components",
    "out_degree_centrality",
    "pagerank",
    "periphery",
    "predecessor",
    "prim_mst_edges",
    "radius",
    "random_k_lift",
    "s_metric",
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
    "voronoi_cells",
    "weakly_connected_components",
    "wiener_index",
}

# Functions that return subgraphs of the original NetworkX graph, or (the flow
# functions) read its edge attributes there.
_BUILDS_FROM_SOURCE = {
    "boruvka_mst_edges",
    "boykov_kolmogorov",
    "build_residual_network",
    "compose",
    "compose_all",
    "contracted_edge",
    "contracted_nodes",
    "convert_node_labels_to_integers",
    "dinitz",
    "disjoint_union",
    "disjoint_union_all",
    "edmonds_karp",
    "ego_graph",
    "full_join",
    "generic_weighted_projected_graph",
    "gomory_hu_tree",
    "is_weighted",
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
    "modular_product",
    "partition_spanning_tree",
    "preflow_push",
    "prim_mst_edges",
    "relabel_nodes",
    "reverse",
    "shortest_augmenting_path",
    "steiner_tree",
    "transitive_closure",
    "transitive_closure_dag",
    "union",
    "union_all",
}

# Functions that read edge attributes of any type from the original NetworkX
# graph's dicts, so they don't convert their `edge_attrs` to numbers.
_EDGE_DATA_FROM_SOURCE = {"get_edge_attributes", "stochastic_graph"}


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
    if name in _EDGE_DATA_FROM_SOURCE:
        # Reads any edge attribute (not only numbers) from the original graph.
        edge_attrs = None
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


# Inputs rustnx can only compute with fast floats (see _config.py): with
# exact floats set for the function, they run in NetworkX. name -> (which
# inputs, as a test on the graph; what to call them in messages).
INEXACT_ONLY = {
    "pagerank": (
        lambda G: G.is_multigraph(),
        "pagerank on a multigraph",
    ),
}


def _inexact_reason(name, G):
    """Why rustnx declines this call under exact floats, or None.

    An explicit ``backend="rustnx"`` is let through, so that the function
    itself raises with this reason (NetworkX's own error wouldn't give it).
    """
    entry = INEXACT_ONLY.get(name)
    if entry is None or G is None or not entry[0](G) or not exact_floats(name):
        return None
    if _backend_requested():
        return None
    reason = inexact_message(name, entry[1])
    _float_logger.debug("%s; running NetworkX's own code instead", reason)
    return reason


def can_run(name, args, kwargs):
    if name not in _OUR_PARAMS:
        return False
    if (
        _takes_no_graph(name)
        and _input_size(name, args, kwargs) < SMALL_INPUT
        and not _backend_requested()
    ):
        # NetworkX skips `should_run` for calls without graph inputs, so a
        # small input is declined here; NetworkX then runs its own code.
        return "input is small; NetworkX is faster"
    try:
        arguments, unsupported = _bind(name, args, kwargs)
    except TypeError:
        return True  # let the call raise the usual error
    if unsupported:
        return f"unsupported arguments: {', '.join(unsupported)}"
    G = arguments.get("G")
    if G is not None and not hasattr(G, "is_multigraph"):
        # e.g. NetworkX's own tests call is_partition(G.nodes(), ...)
        return "the graph argument is not a graph"
    if G is not None and G.is_multigraph() and name not in MULTIGRAPH_FUNCTIONS:
        return "multigraphs are not supported by this function"
    reason = _inexact_reason(name, G)
    if reason is not None:
        return reason
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


def _backend_requested():
    """Whether this call came with ``backend="rustnx"`` rather than through
    ``nx.config.backend_priority``.

    NetworkX doesn't tell ``can_run`` which, and declining a call the user
    sent to rustnx explicitly would raise instead of running NetworkX's
    code, so read the dispatcher's own ``backend`` argument (``__call__`` in
    NetworkX 3.4, ``_call_if_any_backends_installed`` from 3.5). Anything
    unexpected counts as requested, so rustnx runs as it would without the
    check. NetworkX's own test suite with rustnx enabled always runs rustnx.
    """
    from networkx.utils.backends import _dispatchable

    if _dispatchable._is_testing:
        return True
    frame = sys._getframe(2)
    for _ in range(6):
        if frame is None:
            break
        if frame.f_code.co_name in ("__call__", "_call_if_any_backends_installed"):
            local = frame.f_locals
            if isinstance(local.get("self"), _dispatchable) and "backend" in local:
                return local["backend"] is not None
        frame = frame.f_back
    return True


# Generators, readers and graph builders take no graph. Below this input
# size (nodes, edges, lines or bytes, roughly) NetworkX's own code is faster
# than rustnx's dispatch overhead (measured: rustnx wins from about 30 to
# 100 elements up).
SMALL_INPUT = 100

_LARGE = 1 << 62


@functools.cache
def _takes_no_graph(name):
    from networkx.utils.backends import _registered_algorithms

    func = _registered_algorithms.get(name)
    return func is not None and not func.graphs


def _power(base, exponent):
    """``base ** exponent`` for size estimates, without building huge ints."""
    if type(base) is not int or type(exponent) is not int or exponent < 0:
        return _LARGE
    if base <= 1 or exponent <= 1:
        return abs(base) if exponent == 1 else 1
    return base**exponent if exponent * base.bit_length() < 62 else _LARGE


def _product(*values):
    out = 1
    for v in values:
        out *= _size_of(v)
        if out >= _LARGE:
            return _LARGE
    return out


# Generators whose size isn't their largest argument.
_SIZE_ESTIMATES = {
    "balanced_tree": lambda a: _power(a.get("r"), a.get("h")),
    "binomial_tree": lambda a: _power(2, a.get("n")),
    "hypercube_graph": lambda a: _power(2, a.get("n")),
    "dorogovtsev_goltsev_mendes_graph": lambda a: _power(3, a.get("n")),
    "mycielski_graph": lambda a: _power(2, a.get("n")),
    "sudoku_graph": lambda a: _power(a.get("n", 3), 4),
    "complete_graph": lambda a: _product(a.get("n"), a.get("n")),
    "complete_bipartite_graph": lambda a: _product(a.get("n1"), a.get("n2")),
    "grid_2d_graph": lambda a: _product(a.get("m"), a.get("n")),
    "hexagonal_lattice_graph": lambda a: _product(a.get("m"), a.get("n")),
    "triangular_lattice_graph": lambda a: _product(a.get("m"), a.get("n")),
    "grid_graph": lambda a: _product(*a["dim"]) if isinstance(a.get("dim"), (list, tuple)) else _LARGE,
    "caveman_graph": lambda a: _product(a.get("l"), a.get("k"), a.get("k")),
    "connected_caveman_graph": lambda a: _product(a.get("l"), a.get("k"), a.get("k")),
    "ring_of_cliques": lambda a: _product(a.get("num_cliques"), a.get("clique_size"), a.get("clique_size")),
    "windmill_graph": lambda a: _product(a.get("n"), a.get("k"), a.get("k")),
    "kneser_graph": lambda a: _product(a.get("n"), a.get("n"), a.get("k")),
    "circulant_graph": lambda a: _product(a.get("n"), a.get("offsets")),
    "margulis_gabber_galil_graph": lambda a: _product(a.get("n"), a.get("n")),
    "nonisomorphic_trees": lambda a: _power(3, a.get("order")),
    "join_trees": lambda a: sum(len(pair[0]) for pair in a["rooted_trees"])
    if isinstance(a.get("rooted_trees"), (list, tuple))
    else _LARGE,
    "random_cograph": lambda a: _power(2, a.get("n")),
    # `tries` (default 100) doesn't make the tree larger.
    "random_powerlaw_tree": lambda a: _size_of(a.get("n")),
    "random_powerlaw_tree_sequence": lambda a: _size_of(a.get("n")),
    "joint_degree_graph": lambda a: _joint_degree_size(a.get("joint_degrees")),
    "directed_joint_degree_graph": lambda a: max(
        _size_of(a.get("in_degrees")), _joint_degree_size(a.get("nkk"))
    ),
    "navigable_small_world_graph": lambda a: _power(a.get("n"), a.get("dim", 2)),
    "relaxed_caveman_graph": lambda a: _product(a.get("l"), a.get("k"), a.get("k")),
    "random_shell_graph": lambda a: sum(_size_of(s[0]) + _size_of(s[1]) for s in a["constructor"]),
    "k_random_intersection_graph": lambda a: _product(a.get("n"), a.get("k")),
    "general_random_intersection_graph": lambda a: _product(a.get("n"), a.get("m")),
    "maybe_regular_expander": lambda a: _product(a.get("n"), a.get("d")),
    "maybe_regular_expander_graph": lambda a: _product(a.get("n"), a.get("d")),
}


def _joint_degree_size(joint_degrees):
    """The edges (twice, for ``joint_degree_graph``) of the graph a joint
    degree dict describes."""
    try:
        return sum(sum(row.values()) for row in joint_degrees.values())
    except Exception:
        return _LARGE


def _size_of(value):
    """A rough size for one argument; unknown kinds count as large."""
    if value is None or type(value) is bool or isinstance(value, type):
        return 0  # flags, and graph classes (`create_using`, `default`)
    if isinstance(value, int):
        return abs(value)
    if isinstance(value, float):
        return 0
    if isinstance(value, (str, bytes, bytearray, list, tuple, dict, set, frozenset, range)):
        return len(value)
    nnz = getattr(value, "nnz", None)  # SciPy sparse arrays
    if isinstance(nnz, int):
        return nnz
    size = getattr(value, "size", None)  # NumPy arrays
    if isinstance(size, int):
        return size
    try:  # an open file: its length in bytes
        return os.fstat(value.fileno()).st_size
    except (AttributeError, OSError, TypeError, ValueError):
        return _LARGE


@functools.cache
def _param_names(name):
    params = _nx_signature(name).parameters.values()
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return None
    return tuple(p.name for p in params)


def _input_size(name, args, kwargs):
    # Cheaper than binding the signature: this runs on every automatic call.
    names = _param_names(name)
    if names is None or len(args) > len(names):
        return _LARGE  # let rustnx raise the usual error
    arguments = dict(zip(names, args))
    arguments.update(kwargs)
    estimate = _SIZE_ESTIMATES.get(name)
    if estimate is not None:
        try:
            return estimate(arguments)
        except (TypeError, ValueError, KeyError):
            return _LARGE
    size = 0
    for key, value in arguments.items():
        if key in ("create_using", "default", "backend", "seed"):
            continue
        if isinstance(value, (str, os.PathLike)) and name.startswith("read_"):
            try:
                value = os.path.getsize(value)
            except OSError:
                return _LARGE  # NetworkX raises the error
        else:
            value = _size_of(value)
        if value >= SMALL_INPUT:
            return value
        size = max(size, value)
    return size
