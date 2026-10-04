"""NetworkX-compatible algorithms backed by the Rust core.

Each function has the same signature and return value as its NetworkX
counterpart. Inputs rustnx can't handle raise ``NotImplementedError``, which
makes NetworkX fall back to its own implementation.
"""

import copy
import functools
from collections import Counter, defaultdict
from collections.abc import Set
import inspect
from itertools import chain, islice
from itertools import chain, combinations
from itertools import chain, permutations
import math
import operator
import random
import sys
import warnings

import networkx as nx
from networkx.algorithms.centrality import betweenness as _nx_betweenness
from networkx.algorithms import matching as _nx_matching

from . import _core

__all__ = [
    "adamic_adar_index",
    "all_pairs_all_shortest_paths",
    "all_pairs_bellman_ford_path",
    "all_pairs_bellman_ford_path_length",
    "all_pairs_dijkstra",
    "all_pairs_dijkstra_path",
    "all_pairs_dijkstra_path_length",
    "all_pairs_lowest_common_ancestor",
    "all_pairs_node_connectivity",
    "all_pairs_shortest_path",
    "all_pairs_shortest_path_length",
    "all_shortest_paths",
    "all_simple_edge_paths",
    "all_simple_paths",
    "all_topological_sorts",
    "all_triangles",
    "ancestors",
    "antichain_width",
    "antichains",
    "approximate_diameter",
    "articulation_points",
    "astar_path",
    "astar_path_length",
    "asyn_fluidc",
    "asyn_lpa_communities",
    "attracting_components",
    "attribute_assortativity_coefficient",
    "attribute_mixing_dict",
    "attribute_mixing_matrix",
    "average_clustering",
    "average_degree_connectivity",
    "average_neighbor_degree",
    "average_node_connectivity",
    "average_shortest_path_length",
    "barycenter",
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
    "boruvka_mst_edges",
    "boundary_expansion",
    "boykov_kolmogorov",
    "branching_weight",
    "bridge_augmentation",
    "bridge_components",
    "bridges",
    "build_flow_dict",
    "build_residual_network",
    "butterflies",
    "center",
    "centroid",
    "chain_decomposition",
    "check_planarity",
    "check_planarity_recursive",
    "chordal_graph_treewidth",
    "closeness_centrality",
    "closeness_vitality",
    "clustering",
    "cn_soundarajan_hopcroft",
    "color",
    "common_neighbor_centrality",
    "complement",
    "complete_to_chordal_graph",
    "condensation",
    "conductance",
    "connected_components",
    "connected_dominating_set",
    "core_number",
    "cost_of_flow",
    "could_be_isomorphic",
    "cut_size",
    "cycle_basis",
    "dag_longest_path",
    "dag_longest_path_length",
    "dag_to_branching",
    "degree_assortativity_coefficient",
    "degree_centrality",
    "degree_mixing_dict",
    "degree_mixing_matrix",
    "degree_pearson_correlation_coefficient",
    "densest_subgraph",
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
    "difference",
    "dijkstra_path",
    "dijkstra_path_length",
    "dijkstra_predecessor_and_distance",
    "dinitz",
    "dispersion",
    "dominance_frontiers",
    "eccentricity",
    "edge_betweenness_centrality",
    "edge_betweenness_centrality_subset",
    "edge_betweenness_partition",
    "edge_bfs",
    "edge_boundary",
    "edge_connectivity",
    "edge_dfs",
    "edge_disjoint_paths",
    "edge_expansion",
    "edge_load_centrality",
    "edmonds_karp",
    "efficiency",
    "eigenvector_centrality",
    "enumerate_all_cliques",
    "eulerian_circuit",
    "eulerian_path",
    "fast_could_be_isomorphic",
    "fast_label_propagation_communities",
    "faster_could_be_isomorphic",
    "find_cycle",
    "find_minimal_d_separator",
    "find_negative_cycle",
    "flow_hierarchy",
    "floyd_warshall",
    "floyd_warshall_numpy",
    "floyd_warshall_predecessor_and_distance",
    "floyd_warshall_tree",
    "from_nested_tuple",
    "from_prufer_sequence",
    "generalized_degree",
    "generic_bfs_edges",
    "get_counterexample",
    "get_counterexample_recursive",
    "girth",
    "girvan_newman",
    "global_efficiency",
    "global_reaching_centrality",
    "goldberg_radzik",
    "gomory_hu_tree",
    "greedy_branching",
    "greedy_color",
    "greedy_modularity_communities",
    "greedy_tsp",
    "group_betweenness_centrality",
    "group_closeness_centrality",
    "group_degree_centrality",
    "group_in_degree_centrality",
    "group_out_degree_centrality",
    "gutman_index",
    "harmonic_centrality",
    "harmonic_diameter",
    "has_bridges",
    "has_cycle",
    "has_eulerian_path",
    "has_path",
    "hopcroft_karp_matching",
    "hyper_wiener_index",
    "immediate_dominators",
    "in_degree_centrality",
    "inter_community_edges",
    "inter_community_non_edges",
    "intersection_array",
    "intra_community_edges",
    "is_aperiodic",
    "is_arborescence",
    "is_at_free",
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
    "is_digraphical",
    "is_directed_acyclic_graph",
    "is_distance_regular",
    "is_dominating_set",
    "is_equitable",
    "is_eulerian",
    "is_forest",
    "is_graphical",
    "is_isomorphic",
    "is_k_edge_connected",
    "is_k_regular",
    "is_kl_connected",
    "is_locally_k_edge_connected",
    "is_matching",
    "is_maximal_matching",
    "is_minimal_d_separator",
    "is_multigraphical",
    "is_partition",
    "is_perfect_graph",
    "is_perfect_matching",
    "is_planar",
    "is_pseudographical",
    "is_reachable",
    "is_regular",
    "is_semiconnected",
    "is_semieulerian",
    "is_simple_path",
    "is_strongly_connected",
    "is_strongly_regular",
    "is_tournament",
    "is_tree",
    "is_valid_degree_sequence_erdos_gallai",
    "is_valid_degree_sequence_havel_hakimi",
    "is_weakly_connected",
    "isolates",
    "jaccard_coefficient",
    "johnson",
    "k_core",
    "k_corona",
    "k_crust",
    "k_edge_augmentation",
    "k_edge_components",
    "k_edge_subgraphs",
    "k_shell",
    "k_truss",
    "katz_centrality",
    "kl_connected_subgraph",
    "kosaraju_strongly_connected_components",
    "kruskal_mst_edges",
    "label_propagation_communities",
    "lexicographical_topological_sort",
    "local_bridges",
    "local_edge_connectivity",
    "local_efficiency",
    "local_node_connectivity",
    "local_reaching_centrality",
    "lowest_common_ancestor",
    "max_flow_min_cost",
    "max_weight_clique",
    "max_weight_matching",
    "maximal_matching",
    "maximum_branching",
    "maximum_flow",
    "maximum_flow_value",
    "maximum_spanning_arborescence",
    "maximum_spanning_edges",
    "maximum_spanning_tree",
    "min_cost_flow",
    "min_cost_flow_cost",
    "min_edge_cover",
    "min_edge_dominating_set",
    "min_maximal_matching",
    "min_weight_matching",
    "min_weighted_dominating_set",
    "min_weighted_vertex_cover",
    "minimal_branching",
    "minimum_branching",
    "minimum_cut",
    "minimum_cut_value",
    "minimum_cycle_basis",
    "minimum_edge_cut",
    "minimum_node_cut",
    "minimum_spanning_arborescence",
    "minimum_spanning_edges",
    "minimum_spanning_tree",
    "minimum_st_edge_cut",
    "minimum_st_node_cut",
    "mixing_expansion",
    "modularity",
    "multi_source_dijkstra",
    "multi_source_dijkstra_path",
    "multi_source_dijkstra_path_length",
    "naive_greedy_modularity_communities",
    "negative_edge_cycle",
    "network_simplex",
    "newman_betweenness_centrality",
    "node_attribute_xy",
    "node_boundary",
    "node_clique_number",
    "node_connected_component",
    "node_connectivity",
    "node_degree_xy",
    "node_disjoint_paths",
    "node_expansion",
    "node_redundancy",
    "normalized_cut_size",
    "number_attracting_components",
    "number_connected_components",
    "number_of_isolates",
    "number_of_walks",
    "number_strongly_connected_components",
    "number_weakly_connected_components",
    "numeric_assortativity_coefficient",
    "one_edge_augmentation",
    "one_exchange",
    "onion_layers",
    "out_degree_centrality",
    "overall_reciprocity",
    "overlapping_modularity",
    "pagerank",
    "partition_quality",
    "partition_spanning_tree",
    "percolation_centrality",
    "periphery",
    "power",
    "predecessor",
    "preferential_attachment",
    "preflow_push",
    "prim_mst_edges",
    "prominent_group",
    "ra_index_soundarajan_hopcroft",
    "radius",
    "randomized_partitioning",
    "reciprocity",
    "resource_allocation_index",
    "rich_club_coefficient",
    "root_to_leaf_paths",
    "root_trees",
    "rooted_tree_isomorphism",
    "s_metric",
    "schultz_index",
    "score_sequence",
    "sets",
    "shortest_augmenting_path",
    "shortest_path",
    "shortest_path_length",
    "shortest_simple_paths",
    "simulated_annealing_tsp",
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
    "steiner_tree",
    "stoer_wagner",
    "strongly_connected_components",
    "symmetric_difference",
    "threshold_accepting_tsp",
    "to_nested_tuple",
    "to_prufer_sequence",
    "to_vertex_cover",
    "topological_generations",
    "topological_sort",
    "tournament_is_strongly_connected",
    "transitive_closure",
    "transitive_closure_dag",
    "transitive_reduction",
    "transitivity",
    "tree_all_pairs_lowest_common_ancestor",
    "tree_centroid",
    "tree_isomorphism",
    "treewidth_decomp",
    "treewidth_min_fill_in",
    "triadic_census",
    "triangles",
    "unconstrained_bridge_augmentation",
    "unconstrained_one_edge_augmentation",
    "v_structures",
    "vf2pp_is_isomorphic",
    "vf2pp_is_monomorphic",
    "vf2pp_subgraph_is_isomorphic",
    "volume",
    "voronoi_cells",
    "voterank",
    "weakly_connected_components",
    "weisfeiler_lehman_graph_hash",
    "weisfeiler_lehman_subgraph_hashes",
    "wiener_index",
    "within_inter_cluster",
]


def _index_of(G, node, message):
    try:
        return G._index[node]
    except (KeyError, TypeError):
        raise nx.NodeNotFound(message) from None


def _check_weight(G, attr, distances=False):
    """Validate a weight argument; return ``(attr, all_int, has_hidden)``.

    ``distances=True`` is for functions that return path lengths: if the
    attribute mixes ints and floats, NetworkX returns an int or a float
    depending on the path, which rustnx doesn't track, so it declines.
    """
    if callable(attr):
        raise NotImplementedError("rustnx does not support callable weights")
    if attr is not None:
        G._ensure_weight(attr)
    all_int, has_hidden = G._core.weight_info(attr)
    if distances and G._core.weight_mixed(attr):
        raise NotImplementedError("edge weights mix ints and floats")
    return attr, all_int, has_hidden


def _undirected_only(G):
    if G.is_directed():
        raise nx.NetworkXNotImplemented("not implemented for directed type")


def _directed_only(G):
    if not G.is_directed():
        raise nx.NetworkXNotImplemented("not implemented for undirected type")


class _MutationGuard:
    """Notices when the source NetworkX graph changes during iteration.

    NetworkX's generators (``topological_sort``, ``connected_components``,
    ...) read the live graph between yields, so changing the graph
    mid-iteration changes what they do next. rustnx computes results up
    front, so it has to notice. Every change made through the NetworkX API
    clears ``G.__networkx_cache__``, so a marker left there vanishes when the
    graph changes. Checking it is O(1).
    """

    def __init__(self, G):
        self.graph = G._source
        if G._core.is_native():
            # Native graphs are immutable: nothing can change.
            self._cache = self._key = None
            return
        self._cache = getattr(self.graph, "__networkx_cache__", None)
        # Views (subgraph, reverse, ...) keep their own cache, which changes
        # to the underlying graph don't clear.
        if self._cache is None or getattr(self.graph, "_graph", None) is not None:
            raise NotImplementedError(
                "can't detect changes to this graph during iteration"
            )
        self._key = ("rustnx-iteration", id(self))
        self._cache[self._key] = True

    def changed(self):
        if self._key is None:
            return False
        cache = getattr(self.graph, "__networkx_cache__", None)
        return cache is not self._cache or self._key not in cache

    def release(self):
        if self._key is not None:
            self._cache.pop(self._key, None)


def _components(G, comps):
    """Yield components as node sets, stopping loudly if G changes."""
    guard = _MutationGuard(G)

    def generate():
        nodes = G._nodes
        try:
            for comp in comps:
                if guard.changed():
                    # NetworkX would continue on the changed graph; the
                    # precomputed components no longer apply.
                    raise RuntimeError("Graph changed during iteration")
                yield {nodes[i] for i in comp}
        finally:
            guard.release()

    return generate()


def _check_not_null(G):
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept(
            "Connectivity is undefined for the null graph."
        )


def connected_components(G):
    _undirected_only(G)
    return _components(G, G._core.connected_components())


def number_connected_components(G):
    _undirected_only(G)
    return len(G._core.connected_components())


def is_connected(G):
    _undirected_only(G)
    n = len(G)
    if n == 0:
        raise nx.NetworkXPointlessConcept(
            "Connectivity is undefined for the null graph."
        )
    return len(G._core.connected_components()[0]) == n


def single_source_shortest_path_length(G, source, cutoff=None):
    s = _index_of(G, source, f"Source {source} is not in G")
    if cutoff is None:
        cutoff = math.inf
    order, levels = G._core.bfs_lengths(s, float(cutoff))
    nodes = G._nodes
    return dict(zip([nodes[i] for i in order], levels))


def single_source_dijkstra_path_length(G, source, cutoff=None, weight="weight"):
    s = _index_of(G, source, f"Node {source} not found in graph")
    weight, all_int, _ = _check_weight(G, weight, distances=True)
    order, dists = G._core.dijkstra_lengths(
        s, weight, None if cutoff is None else float(cutoff)
    )
    if all_int:
        dists = map(int, dists)
    nodes = G._nodes
    result = dict(zip([nodes[i] for i in order], dists))
    result[source] = 0  # NetworkX keeps the source distance as the int 0
    return result


@functools.cache
def _rescale_params():
    """Arguments accepted by the installed NetworkX's betweenness ``_rescale``.

    Its signature and formula differ between releases (3.4, 3.5 and 3.6 all
    differ), so rustnx calls NetworkX's own function to match the installed
    version exactly. Returns ``None`` if the signature isn't recognized.
    """
    rescale = getattr(_nx_betweenness, "_rescale", None)
    if rescale is None:
        return None
    params = list(inspect.signature(rescale).parameters.values())[2:]
    known = {"normalized", "directed", "k", "endpoints", "sampled_nodes"}
    if any(p.name not in known and p.default is p.empty for p in params):
        return None
    return rescale, [p.name for p in params if p.name in known]


def betweenness_centrality(
    G, k=None, normalized=True, weight=None, endpoints=False, seed=None
):
    rescale_params = _rescale_params()
    if rescale_params is None:
        raise NotImplementedError("unrecognized NetworkX betweenness rescaling")
    n = len(G)
    if k == n:
        k = None
    weight, _, has_hidden = _check_weight(G, weight)
    if weight is not None and has_hidden:
        raise NotImplementedError("rustnx does not support None edge weights here")
    sampled = None
    sources = None
    if k is not None:
        # `seed` is already a `random.Random` (NetworkX's @py_random_state
        # runs before dispatch), so this samples exactly what NetworkX would.
        sampled = seed.sample(list(G._nodes), k)
        sources = [G._index[v] for v in sampled]
    raw = G._core.betweenness(weight, bool(endpoints), sources)
    rescale, names = rescale_params
    available = {
        "normalized": normalized,
        "directed": G.is_directed(),
        "k": k,
        "endpoints": endpoints,
        "sampled_nodes": sampled,
    }
    return rescale(
        dict(zip(G._nodes, raw)), n, **{name: available[name] for name in names}
    )


@functools.cache
def _edge_rescale():
    """How the installed NetworkX rescales edge betweenness: ``_rescale_e``
    before 3.6, the shared ``_rescale`` after. ``None`` if unrecognized."""
    func = getattr(nx.edge_betweenness_centrality, "orig_func", None)
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return None
    if "_rescale_e(" in source and hasattr(_nx_betweenness, "_rescale_e"):
        return lambda b, n, normalized, directed, sampled: _nx_betweenness._rescale_e(
            b, n, normalized=normalized, directed=directed
        )
    if "sampled_nodes=None if k is None else nodes" in source:
        return lambda b, n, normalized, directed, sampled: _nx_betweenness._rescale(
            b, n, normalized=normalized, directed=directed, sampled_nodes=sampled
        )
    return None


def edge_betweenness_centrality(G, k=None, normalized=True, weight=None, seed=None):
    rescale = _edge_rescale()
    if rescale is None:
        raise NotImplementedError("unrecognized NetworkX edge betweenness rescaling")
    weight, _, has_hidden = _check_weight(G, weight)
    if weight is not None and has_hidden:
        raise NotImplementedError("rustnx does not support None edge weights here")
    sampled = sources = None
    if k is not None:
        # As in betweenness_centrality, `seed` is already a `random.Random`.
        sampled = seed.sample(list(G._nodes), k)
        sources = [G._index[v] for v in sampled]
    raw = G._core.edge_betweenness(weight, sources)
    us, vs, _ = G._core.edges_in_order()
    nodes = G._nodes
    betweenness = dict(zip([(nodes[u], nodes[v]) for u, v in zip(us, vs)], raw))
    return rescale(betweenness, len(G), normalized, G.is_directed(), sampled)


def closeness_centrality(G, u=None, distance=None, wf_improved=True):
    distance, _, _ = _check_weight(G, distance)
    sources = None
    if u is not None:
        if distance is None:
            msg = f"Source {u} is not in G"
        else:
            msg = f"Node {u} not found in graph"
        sources = [_index_of(G, u, msg)]
    if G.is_directed() and distance is not None:
        # Tie order on the reversed graph affects the float sums.
        G._ensure_exact_pred()
    values = G._core.closeness(distance, bool(wf_improved), sources, _COMPENSATED_SUM)
    if u is not None:
        return values[0]
    return dict(zip(G._nodes, values))


# --- Directed components ------------------------------------------------------


@functools.cache
def _scc_early_exit():
    """Whether the installed NetworkX's ``strongly_connected_components``
    (3.7+) stops early when the whole graph is one component. That changes
    the order it fills that component's set in, and so the set's iteration
    order."""
    try:
        source = inspect.getsource(nx.strongly_connected_components.orig_func)
    except (AttributeError, OSError, TypeError):
        return True
    return "root_low" in source


def _scc(G):
    return G._core.strongly_connected_components(_scc_early_exit())


def strongly_connected_components(G):
    _directed_only(G)
    return _components(G, _scc(G))


def number_strongly_connected_components(G):
    _directed_only(G)
    return len(G._core.strongly_connected_components())


def is_strongly_connected(G):
    _directed_only(G)
    _check_not_null(G)
    return len(G._core.strongly_connected_components()[0]) == len(G)


def weakly_connected_components(G):
    _directed_only(G)
    return _components(G, G._core.weakly_connected_components())


def number_weakly_connected_components(G):
    _directed_only(G)
    return len(G._core.weakly_connected_components())


def is_weakly_connected(G):
    _directed_only(G)
    _check_not_null(G)
    return len(G._core.weakly_connected_components()[0]) == len(G)


# --- Topological order ---------------------------------------------------------


def topological_generations(G):
    guard = _MutationGuard(G)
    return _topological_generations(G, guard)


def _topological_generations(G, guard):
    try:
        if not G.is_directed():
            raise nx.NetworkXError(
                "Topological sort not defined on undirected graphs."
            )
        generations, has_cycle = G._core.topological_generations()
        nodes = G._nodes
        if guard.changed():
            # Changed before iteration started: NetworkX would work on the
            # changed graph from the beginning.
            yield from nx.topological_generations(guard.graph, backend="networkx")
            return
        for i, generation in enumerate(generations):
            yield [nodes[v] for v in generation]
            if guard.changed():
                yield from _resume_topological_generations(G, guard, generations, i)
                return
        if has_cycle:
            raise nx.NetworkXUnfeasible(
                "Graph contains a cycle or graph changed during iteration"
            )
    finally:
        guard.release()


def _resume_topological_generations(G, guard, generations, done):
    """Continue exactly as NetworkX would after the graph changed.

    When NetworkX yields generation ``done`` it has already processed that
    generation's out-edges and built the next one, so rebuild that state
    from the snapshot and run NetworkX's loop on the live graph.
    """
    nodes = G._nodes
    processed = [v for generation in generations[: done + 1] for v in generation]
    remaining = G._core.indegrees_after(processed)
    indegree_map = {nodes[v]: d for v, d in enumerate(remaining) if d > 0}
    if done + 1 < len(generations):
        zero_indegree = [nodes[v] for v in generations[done + 1]]
    else:
        zero_indegree = []
    live = guard.graph
    # From networkx.algorithms.dag.topological_generations (non-multigraph).
    while zero_indegree:
        this_generation = zero_indegree
        zero_indegree = []
        for node in this_generation:
            if node not in live:
                raise RuntimeError("Graph changed during iteration")
            for child in live.neighbors(node):
                try:
                    indegree_map[child] -= 1
                except KeyError as err:
                    raise RuntimeError("Graph changed during iteration") from err
                if indegree_map[child] == 0:
                    zero_indegree.append(child)
                    del indegree_map[child]
        yield this_generation
    if indegree_map:
        raise nx.NetworkXUnfeasible(
            "Graph contains a cycle or graph changed during iteration"
        )


def topological_sort(G):
    generations = topological_generations(G)

    def generate():
        for generation in generations:
            yield from generation

    return generate()


def is_directed_acyclic_graph(G):
    return G.is_directed() and not G._core.topological_generations()[1]


# --- PageRank --------------------------------------------------------------------


def _node_vector(G, mapping):
    """``[mapping.get(n, 0) for n in G]`` as floats, like NetworkX's arrays."""
    if mapping is None:
        return None
    try:
        return [float(mapping.get(n, 0)) for n in G._nodes]
    except (AttributeError, TypeError, ValueError):
        raise NotImplementedError("unsupported vector values") from None


def pagerank(
    G,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
):
    if len(G) == 0:
        return {}
    weight, _, has_hidden = _check_weight(G, weight)
    if weight is not None and has_hidden:
        raise NotImplementedError("rustnx does not support None edge weights here")
    p = _node_vector(G, personalization)
    if p is not None and sum(p) == 0:
        raise ZeroDivisionError
    scores = G._core.pagerank(
        float(alpha),
        p,
        max(0, operator.index(max_iter)),
        float(tol),
        _node_vector(G, nstart),
        weight,
        _node_vector(G, dangling),
    )
    if scores is None:
        raise nx.PowerIterationFailedConvergence(max_iter)
    return dict(zip(G._nodes, scores))


# --- Distance measures ---------------------------------------------------------


_NEGATIVE_CYCLE = ("Contradictory paths found:", "negative weights?")


def _not_connected_error(G):
    if G.is_directed():
        msg = "Found infinite path length because the digraph is not strongly connected"
    else:
        msg = "Found infinite path length because the graph is not connected"
    return nx.NetworkXError(msg)


def _eccentricity_values(G, weight, sources=None):
    """Eccentricities in source order, raising exactly as NetworkX would.

    NetworkX computes source by source and raises at the first source that
    either hits a negative cycle or doesn't reach every node; checking the
    precomputed results in the same order raises the same error.
    """
    n = len(G)
    if weight is None:
        stats = G._core.bfs_stats(sources)
        for reached, _, _ in stats:
            if reached != n:
                raise _not_connected_error(G)
        return [mx for _, _, mx in stats]
    weight, all_int, _ = _check_weight(G, weight, distances=True)
    stats, _ = G._core.dijkstra_stats(weight, sources)
    values = []
    for st in stats:
        if st is None:
            raise ValueError(*_NEGATIVE_CYCLE)
        reached, mx = st
        if reached != n:
            raise _not_connected_error(G)
        values.append(int(mx) if all_int else mx)
    return values


def eccentricity(G, v=None, sp=None, weight=None):
    if sp is not None or len(G) == 0:
        # Precomputed paths are cheap in Python; empty graphs differ by version.
        raise NotImplementedError("rustnx computes eccentricity from scratch")
    if v is None:
        return dict(zip(G._nodes, _eccentricity_values(G, weight)))
    if v in G:
        return _eccentricity_values(G, weight, [G._index[v]])[0]
    raise NotImplementedError("rustnx supports v=None or a single node")


def _all_eccentricities(G, e, usebounds, weight):
    if e is not None or len(G) == 0:
        raise NotImplementedError("rustnx computes eccentricities from scratch")
    if usebounds is True and not G.is_directed():
        # NetworkX uses a different algorithm (with its own errors) here.
        raise NotImplementedError("usebounds=True runs in NetworkX")
    return dict(zip(G._nodes, _eccentricity_values(G, weight)))


def diameter(G, e=None, usebounds=False, weight=None):
    return max(_all_eccentricities(G, e, usebounds, weight).values())


def radius(G, e=None, usebounds=False, weight=None):
    return min(_all_eccentricities(G, e, usebounds, weight).values())


def _is_tree(G):
    if G.is_directed() or len(G) == 0:
        return False
    return G.number_of_edges() == len(G) - 1 and len(
        G._core.connected_components()[0]
    ) == len(G)


def center(G, e=None, usebounds=False, weight=None):
    if e is None and weight is None and _is_tree(G):
        # NetworkX 3.7+ takes a tree-specific path with its own output order.
        raise NotImplementedError("trees run in NetworkX")
    ecc = _all_eccentricities(G, e, usebounds, weight)
    r = min(ecc.values())
    return [v for v in ecc if ecc[v] == r]


def periphery(G, e=None, usebounds=False, weight=None):
    ecc = _all_eccentricities(G, e, usebounds, weight)
    d = max(ecc.values())
    return [v for v in ecc if ecc[v] == d]


def _is_strongly_or_plainly_connected(G):
    if G.is_directed():
        return len(G._core.strongly_connected_components()) == 1
    return len(G._core.connected_components()) == 1


def _distance_total(G, weight):
    """Sum of all shortest path lengths, summed in NetworkX's order."""
    if weight is None:
        return sum(total for _, total, _ in G._core.bfs_stats(None))
    weight, all_int, _ = _check_weight(G, weight, distances=True)
    stats, total = G._core.dijkstra_stats(weight, None, _COMPENSATED_SUM)
    if total is None:
        raise ValueError(*_NEGATIVE_CYCLE)
    return int(total) if all_int else total


def average_shortest_path_length(G, weight=None, method=None):
    if method is None:
        method = "unweighted" if weight is None else "dijkstra"
    if method not in ("unweighted", "dijkstra"):
        raise NotImplementedError(f"method {method!r} runs in NetworkX")
    n = len(G)
    if n == 0:
        raise NotImplementedError("the null graph runs in NetworkX")
    if n == 1:
        return 0
    if G.is_directed() and not _is_strongly_or_plainly_connected(G):
        raise nx.NetworkXError("Graph is not strongly connected.")
    if not G.is_directed() and not _is_strongly_or_plainly_connected(G):
        raise nx.NetworkXError("Graph is not connected.")
    s = _distance_total(G, None if method == "unweighted" else weight)
    return s / (n * (n - 1))


def wiener_index(G, weight=None):
    if len(G) == 0:
        raise NotImplementedError("the null graph runs in NetworkX")
    if not _is_strongly_or_plainly_connected(G):
        return float("inf")
    total = _distance_total(G, weight)
    return total if G.is_directed() else total / 2


# Sources per Rust call in the all-pairs generators: large enough to keep
# every core busy, small enough to keep memory bounded.
_ALL_PAIRS_BATCH = 1024


def _all_pairs(G, compute, batch_size=_ALL_PAIRS_BATCH):
    guard = _MutationGuard(G)

    def generate():
        nodes = G._nodes
        try:
            for start in range(0, len(nodes), batch_size):
                batch = list(range(start, min(start + batch_size, len(nodes))))
                for s, lengths in zip(batch, compute(batch)):
                    if guard.changed():
                        raise RuntimeError("Graph changed during iteration")
                    yield nodes[s], lengths
        finally:
            guard.release()

    return generate()


def all_pairs_shortest_path_length(G, cutoff=None):
    cutoff = math.inf if cutoff is None else float(cutoff)
    nodes = G._nodes

    def compute(batch):
        for order, levels in G._core.bfs_many(batch, cutoff):
            yield dict(zip([nodes[i] for i in order], levels))

    return _all_pairs(G, compute)


def all_pairs_dijkstra_path_length(G, cutoff=None, weight="weight"):
    weight, all_int, _ = _check_weight(G, weight, distances=True)
    cutoff = None if cutoff is None else float(cutoff)
    nodes = G._nodes

    def compute(batch):
        for s, result in zip(batch, G._core.dijkstra_many(batch, weight, cutoff)):
            if result is None:
                raise ValueError(*_NEGATIVE_CYCLE)
            order, dists = result
            lengths = dict(zip([nodes[i] for i in order], map(int, dists) if all_int else dists))
            lengths[nodes[s]] = 0  # NetworkX keeps the source distance as the int 0
            yield lengths

    return _all_pairs(G, compute)


# --- Shortest paths that return the paths ------------------------------------


def _bfs_paths(G, s, cutoff, reverse=False):
    """``{node: path}`` in discovery order; reversed searches give each path
    from the node to ``s`` (NetworkX's ``single_target_shortest_path``)."""
    if reverse and G.is_directed():
        G._ensure_exact_pred()
    order, parents = G._core.bfs_tree(s, float(cutoff), reverse)
    return _tree_paths(G, order, parents, order, reverse)


def _tree_paths(G, order, parents, key_order, reverse=False):
    nodes = G._nodes
    by_index = {order[0]: [nodes[order[0]]]}
    for v, p in zip(order[1:], parents[1:]):
        path = by_index[p]
        by_index[v] = [nodes[v], *path] if reverse else [*path, nodes[v]]
    return {nodes[i]: by_index[i] for i in key_order}


@functools.cache
def _dijkstra_paths_in_pop_order():
    """Whether the installed NetworkX orders Dijkstra's paths dict by pop
    order (3.6+) rather than by when each node was first reached."""
    H = nx.DiGraph()
    H.add_weighted_edges_from([(0, 1, 5), (0, 2, 1), (2, 1, 1), (1, 3, 1)])
    # NetworkX's private helper, called directly: the public functions
    # dispatch nested calls (to `multi_source_dijkstra_path`), which can
    # land back in rustnx and in this probe.
    from networkx.algorithms.shortest_paths import weighted

    paths = {0: [0]}
    weighted._dijkstra_multisource(H, [0], lambda u, v, d: d["weight"], paths=paths)
    return list(paths) == [0, 2, 1, 3]


def _dijkstra_tree(G, s, weight, cutoff=None, target=None, reverse=False, lengths=True):
    # `lengths=False`: the caller only uses the paths, so int/float mixing
    # (which only affects the lengths' types) doesn't matter.
    weight, all_int, _ = _check_weight(G, weight, distances=lengths)
    if reverse and G.is_directed():
        G._ensure_exact_pred()
    order, dists, parents, seen = G._core.dijkstra_tree(
        s,
        weight,
        None if cutoff is None else float(cutoff),
        None if target is None else target,
        reverse,
    )
    if all_int:
        dists = [int(d) for d in dists]
    dists[0] = 0  # NetworkX keeps the source distance as the int 0
    return order, dists, parents, seen


def _dijkstra_result(G, tree):
    order, dists, parents, seen = tree
    nodes = G._nodes
    dist = dict(zip([nodes[i] for i in order], dists))
    key_order = order if _dijkstra_paths_in_pop_order() else seen
    return dist, _tree_paths(G, order, parents, key_order)


def _dijkstra_target(G, s, t, weight, cutoff=None, lengths=True):
    """``(distance, path)`` to ``t``, or ``None`` when ``t`` isn't reached."""
    weight, all_int, _ = _check_weight(G, weight, distances=lengths)
    result = G._core.dijkstra_path(s, t, weight, None if cutoff is None else float(cutoff))
    if result is None:
        return None
    dist, path = result
    nodes = G._nodes
    return int(dist) if all_int else dist, [nodes[i] for i in path]


def single_source_shortest_path(G, source, cutoff=None):
    s = _index_of(G, source, f"Source {source} not in G")
    return _bfs_paths(G, s, math.inf if cutoff is None else cutoff)


def single_target_shortest_path(G, target, cutoff=None):
    t = _index_of(G, target, f"Target {target} not in G")
    return _bfs_paths(G, t, math.inf if cutoff is None else cutoff, reverse=True)


def single_source_dijkstra_path(G, source, cutoff=None, weight="weight"):
    s = _index_of(G, source, f"Node {source} not found in graph")
    return _dijkstra_result(G, _dijkstra_tree(G, s, weight, cutoff, lengths=False))[1]


def single_source_dijkstra(G, source, target=None, cutoff=None, weight="weight", *, _lengths=True):
    s = _index_of(G, source, f"Node {source} not found in graph")
    if target is None:
        return _dijkstra_result(G, _dijkstra_tree(G, s, weight, cutoff, lengths=_lengths))
    if target == source:
        return 0, [target]
    try:
        t = G._index[target]
    except (KeyError, TypeError):
        t = None
    result = None if t is None else _dijkstra_target(G, s, t, weight, cutoff, lengths=_lengths)
    if result is None:
        if t is None:
            # NetworkX searches the whole graph first: a negative cycle
            # would raise before NoPath does.
            _dijkstra_tree(G, s, weight, cutoff, lengths=_lengths)
        raise nx.NetworkXNoPath(f"No path to {target}.")
    return result


def dijkstra_path(G, source, target, weight="weight"):
    return single_source_dijkstra(G, source, target=target, weight=weight, _lengths=False)[1]


def dijkstra_path_length(G, source, target, weight="weight"):
    s = _index_of(G, source, f"Node {source} not found in graph")
    if source == target:
        return 0
    try:
        t = G._index[target]
    except (KeyError, TypeError):
        t = None
    result = None if t is None else _dijkstra_target(G, s, t, weight)
    if result is None:
        if t is None:
            _dijkstra_tree(G, s, weight)
        raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")
    return result[0]


def bidirectional_shortest_path(G, source, target):
    s = _index_of(G, source, f"Source {source} is not in G")
    t = _index_of(G, target, f"Target {target} is not in G")
    if G.is_directed():
        G._ensure_exact_pred()
    path = G._core.bidirectional_bfs(s, t)
    if path is None:
        raise nx.NetworkXNoPath(f"No path between {source} and {target}.")
    nodes = G._nodes
    return [nodes[i] for i in path]


def _reachable(G, source, reverse):
    kind = "digraph" if G.is_directed() else "graph"
    try:
        s = G._index[source]
    except KeyError:
        raise nx.NetworkXError(f"The node {source} is not in the {kind}.") from None
    if reverse and G.is_directed():
        G._ensure_exact_pred()
    nodes = G._nodes
    return {nodes[i] for i in G._core.reachable(s, reverse)}


def descendants(G, source):
    return _reachable(G, source, False)


def ancestors(G, source):
    return _reachable(G, source, True)


def has_path(G, source, target):
    try:
        bidirectional_shortest_path(G, source, target)
    except nx.NetworkXNoPath:
        return False
    return True


def all_pairs_shortest_path(G, cutoff=None):
    cutoff = math.inf if cutoff is None else float(cutoff)

    def compute(batch):
        for order, parents in G._core.bfs_tree_many(batch, cutoff):
            yield _tree_paths(G, order, parents, order)

    return _all_pairs(G, compute)


def _dijkstra_trees(G, batch, weight, cutoff, lengths):
    weight, all_int, _ = _check_weight(G, weight, distances=lengths)
    for tree in G._core.dijkstra_tree_many(batch, weight, cutoff):
        if tree is None:
            raise ValueError(*_NEGATIVE_CYCLE)
        order, dists, parents, seen = tree
        if all_int:
            dists = [int(d) for d in dists]
        dists[0] = 0
        yield order, dists, parents, seen


def all_pairs_dijkstra_path(G, cutoff=None, weight="weight"):
    _check_weight(G, weight)
    cutoff = None if cutoff is None else float(cutoff)

    def compute(batch):
        for tree in _dijkstra_trees(G, batch, weight, cutoff, lengths=False):
            yield _dijkstra_result(G, tree)[1]

    return _all_pairs(G, compute)


def all_pairs_dijkstra(G, cutoff=None, weight="weight"):
    _check_weight(G, weight, distances=True)  # before iteration starts
    cutoff = None if cutoff is None else float(cutoff)

    def compute(batch):
        for tree in _dijkstra_trees(G, batch, weight, cutoff, lengths=True):
            yield _dijkstra_result(G, tree)

    return _all_pairs(G, compute)


def _check_method(method):
    if method not in ("dijkstra", "bellman-ford"):
        raise ValueError(f"method not supported: {method}")
    if method != "dijkstra":
        raise NotImplementedError("rustnx has no Bellman-Ford")


def shortest_path(G, source=None, target=None, weight=None, method="dijkstra"):
    if method not in ("dijkstra", "bellman-ford"):
        raise ValueError(f"method not supported: {method}")
    if source is None and target is None:
        # Returns a dict or an iterator depending on the NetworkX version.
        raise NotImplementedError("rustnx leaves all-pairs shortest_path to NetworkX")
    if weight is not None:
        _check_method(method)
    if source is None:
        t = _index_of(G, target, _target_message(G, target, weight))
        if weight is None:
            paths = _bfs_paths(G, t, math.inf, reverse=True)
        else:
            tree = _dijkstra_tree(G, t, weight, reverse=True, lengths=False)
            paths = _dijkstra_result(G, tree)[1]
            paths = {v: p[::-1] for v, p in paths.items()}
        return paths
    if target is None:
        if weight is None:
            return single_source_shortest_path(G, source)
        return single_source_dijkstra_path(G, source, weight=weight)
    if weight is None:
        return bidirectional_shortest_path(G, source, target)
    return _bidirectional_dijkstra(G, source, target, weight, lengths=False)[1]


def _target_message(G, target, weight):
    # The reversed search runs single_source_* with the target as source.
    if weight is None:
        return f"Source {target} not in G"
    return f"Node {target} not found in graph"


def shortest_path_length(G, source=None, target=None, weight=None, method="dijkstra"):
    if method not in ("dijkstra", "bellman-ford"):
        raise ValueError(f"method not supported: {method}")
    if weight is not None:
        _check_method(method)
    if source is None:
        if target is None:
            if weight is None:
                return all_pairs_shortest_path_length(G)
            return all_pairs_dijkstra_path_length(G, weight=weight)
        if weight is None:
            t = _index_of(G, target, f"Source {target} is not in G")
            if G.is_directed():
                G._ensure_exact_pred()
            order, parents = G._core.bfs_tree(t, math.inf, True)
            level = {order[0]: 0}
            for v, p in zip(order[1:], parents[1:]):
                level[v] = level[p] + 1
            nodes = G._nodes
            return {nodes[i]: level[i] for i in order}
        t = _index_of(G, target, f"Node {target} not found in graph")
        order, dists, _, _ = _dijkstra_tree(G, t, weight, reverse=True)
        nodes = G._nodes
        return dict(zip([nodes[i] for i in order], dists))
    if target is None:
        if weight is None:
            return single_source_shortest_path_length(G, source)
        return single_source_dijkstra_path_length(G, source, weight=weight)
    if weight is None:
        return len(bidirectional_shortest_path(G, source, target)) - 1
    return dijkstra_path_length(G, source, target, weight)


# --- Clustering ----------------------------------------------------------------


def _node_subset(G, nodes):
    """Node positions for a NetworkX ``nbunch`` (nodes not in G are skipped)."""
    if iter(nodes) is nodes:
        # An iterator can't be re-read if NetworkX has to take over.
        raise NotImplementedError("rustnx needs a reusable container of nodes")
    index = G._index
    picked = []
    for v in nodes:
        try:
            i = index.get(v)
        except TypeError:
            raise NotImplementedError("unhashable node in nbunch") from None
        if i is not None:
            picked.append(i)
    # Repeated nodes keep their first position, as in a dict.
    return list(dict.fromkeys(picked))


def _triangle_counts(G, nodes):
    """``(single, positions, counts)``: whether ``nodes`` named one node,
    which nodes were counted, and their ``(t, d, db)`` counts."""
    if nodes is None:
        return False, None, G._core.triangle_counts(None)
    if nodes in G:
        i = G._index[nodes]
        return True, [i], G._core.triangle_counts([i])
    positions = _node_subset(G, nodes)
    return False, positions, G._core.triangle_counts(positions)


def _per_node(G, positions, values, single):
    if single:
        return values[0]
    nodes = G._nodes
    keys = nodes if positions is None else [nodes[i] for i in positions]
    return dict(zip(keys, values))


def triangles(G, nodes=None):
    _undirected_only(G)
    single, positions, counts = _triangle_counts(G, nodes)
    return _per_node(G, positions, [t // 2 for t, _, _ in counts], single)


def _clustering_values(G, nodes, weight):
    if weight is not None:
        # Weighted clustering sums cube roots in set-iteration order.
        raise NotImplementedError("rustnx supports unweighted clustering only")
    single, positions, counts = _triangle_counts(G, nodes)
    if G.is_directed():
        values = [0 if t == 0 else t / ((d * (d - 1) - 2 * db) * 2) for t, d, db in counts]
    else:
        values = [0 if t == 0 else t / (d * (d - 1)) for t, d, _ in counts]
    return single, positions, values


def clustering(G, nodes=None, weight=None):
    single, positions, values = _clustering_values(G, nodes, weight)
    return _per_node(G, positions, values, single)


def average_clustering(G, nodes=None, weight=None, count_zeros=True):
    if nodes is not None and nodes in G:
        raise NotImplementedError("NetworkX raises on a single node here")
    _, _, c = _clustering_values(G, nodes, weight)
    if not count_zeros:
        c = [v for v in c if abs(v) > 0]
    return sum(c) / len(c)


def transitivity(G):
    # On directed graphs NetworkX applies the undirected formula to
    # successors only.
    counts = G._core.triangle_counts(None, successors_only=True)
    if not counts:
        return 0
    triangles = sum(t for t, _, _ in counts)
    contri = sum(d * (d - 1) for _, d, _ in counts)
    return 0 if triangles == 0 else triangles / contri


def bidirectional_dijkstra(G, source, target, weight="weight"):
    return _bidirectional_dijkstra(G, source, target, weight, lengths=True)


def _bidirectional_dijkstra(G, source, target, weight, lengths):
    s = _index_of(G, source, f"Source {source} is not in G")
    t = _index_of(G, target, f"Target {target} is not in G")
    if source == target:
        return 0, [source]
    weight, all_int, _ = _check_weight(G, weight, distances=lengths)
    if G.is_directed():
        G._ensure_exact_pred()
    found = G._core.bidirectional_dijkstra(s, t, weight)
    if found is None:
        raise nx.NetworkXNoPath(f"No path between {source} and {target}.")
    dist, path = found
    nodes = G._nodes
    return int(dist) if all_int else dist, [nodes[i] for i in path]


# --- Centrality: harmonic, eigenvector, Katz -------------------------------------


# Python 3.12 made sum() of floats use compensated summation; convergence
# checks that call sum() must add the same way.
_COMPENSATED_SUM = sys.version_info >= (3, 12)


def _nbunch_list(G, nbunch):
    """``list(G.nbunch_iter(nbunch))``, keeping the caller's node objects."""
    if nbunch in G:
        return [nbunch]
    if iter(nbunch) is nbunch:
        raise NotImplementedError("rustnx needs a reusable container of nodes")
    picked = []
    for v in nbunch:
        try:
            if v in G._index:
                picked.append(v)
        except TypeError:
            raise NotImplementedError("unhashable node in nbunch") from None
    return picked


def harmonic_centrality(G, nbunch=None, distance=None, sources=None):
    # Built exactly as NetworkX builds them, so they iterate in the same order.
    nbunch = set(_nbunch_list(G, nbunch) if nbunch is not None else G._nodes)
    sources = set(_nbunch_list(G, sources) if sources is not None else G._nodes)
    if len(nbunch) < len(sources):
        # NetworkX swaps the roles here and adds terms in set-intersection
        # order, which differs by version.
        raise NotImplementedError("rustnx computes harmonic centrality for many targets")
    distance, _, _ = _check_weight(G, distance)
    index = G._index
    in_nbunch = [False] * len(G)
    for u in nbunch:
        in_nbunch[index[u]] = True
    total, touched = G._core.harmonic([index[v] for v in sources], in_nbunch, distance)
    centrality = {u: 0 for u in nbunch}
    for u in centrality:
        i = index[u]
        if touched[i]:
            centrality[u] = total[i]
    return centrality


def _iterations(max_iter):
    try:
        return max(operator.index(max_iter), 0)
    except TypeError:
        raise NotImplementedError("max_iter must be an integer") from None


def _unhidden_weight(G, weight):
    weight, _, has_hidden = _check_weight(G, weight)
    if weight is not None and has_hidden:
        raise NotImplementedError("rustnx does not support None edge weights here")
    return weight


def eigenvector_centrality(G, max_iter=100, tol=1e-06, nstart=None, weight=None):
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("cannot compute centrality for the null graph")
    if nstart is None:
        nstart = {v: 1 for v in G._nodes}
    elif len(nstart) != len(G) or not all(v in G for v in nstart):
        raise NotImplementedError("nstart must give a value for every node")
    if all(v == 0 for v in nstart.values()):
        raise nx.NetworkXError("initial vector cannot have all zero values")
    nstart_sum = sum(nstart.values())
    x = {k: v / nstart_sum for k, v in nstart.items()}
    if not all(type(v) is float for v in x.values()):
        raise NotImplementedError("rustnx needs a float start vector")
    weight = _unhidden_weight(G, weight if weight else None)
    index = G._index
    order = [index[k] for k in x]
    x0 = [0.0] * len(G)
    for i, v in zip(order, x.values()):
        x0[i] = v
    result = G._core.eigenvector(
        order, x0, weight, _iterations(max_iter), float(tol), math.hypot, _COMPENSATED_SUM
    )
    if result is None:
        raise nx.PowerIterationFailedConvergence(max_iter)
    return {k: result[i] for k, i in zip(x, order)}


def katz_centrality(
    G, alpha=0.1, beta=1.0, max_iter=1000, tol=1e-06, nstart=None, normalized=True, weight=None
):
    if len(G) == 0:
        return {}
    if nstart is not None:
        raise NotImplementedError("rustnx starts Katz from zero")
    try:
        beta = float(beta)
    except (TypeError, ValueError, AttributeError):
        raise NotImplementedError("rustnx supports a scalar beta") from None
    if type(alpha) not in (int, float):
        raise NotImplementedError("rustnx supports a real alpha")
    weight = _unhidden_weight(G, weight)
    n = len(G)
    x = G._core.katz(
        list(range(n)), float(alpha), beta, weight, _iterations(max_iter), float(tol), _COMPENSATED_SUM
    )
    if x is None:
        raise nx.PowerIterationFailedConvergence(max_iter)
    if normalized:
        try:
            s = 1.0 / math.hypot(*x)
        except ZeroDivisionError:
            s = 1.0
    else:
        s = 1
    return dict(zip(G._nodes, [v * s for v in x]))


# --- Cores and bipartiteness -------------------------------------------------------


def core_number(G):
    core = G._core.core_number()
    if core is None:
        raise nx.NetworkXNotImplemented(
            "Input graph has self loops which is not permitted; "
            "Consider using G.remove_edges_from(nx.selfloop_edges(G))."
        )
    return dict(zip(G._nodes, core))


def _networkx_graph(G):
    """The NetworkX graph ``G`` stands for, to build subgraphs from."""
    if G._core.is_native():
        return G.to_networkx()
    if not G._source_unchanged():
        raise NotImplementedError("the graph changed since it was converted")
    return G._source


def k_core(G, k=None, core_number=None):
    base = _networkx_graph(G)
    core = globals()["core_number"](G) if core_number is None else core_number
    if k is None:
        k = max(core.values())
    return base.subgraph(v for v in core if core[v] >= k).copy()


def is_bipartite(G):
    return G._core.is_bipartite()


# --- Traversal -------------------------------------------------------------------


def _depth_limit(G, depth_limit):
    if depth_limit is None:
        return len(G)
    try:
        return operator.index(depth_limit)
    except TypeError:
        raise NotImplementedError("depth_limit must be an integer") from None


def _start(G, source, reverse=False):
    """``(position or None, error)`` for a traversal start; NetworkX raises
    the error lazily, when iteration reaches the missing node."""
    try:
        hash(source)
    except TypeError:
        raise NotImplementedError("unhashable source") from None
    kind = "digraph" if G.is_directed() else "graph"
    return G._index.get(source), nx.NetworkXError(f"The node {source} is not in the {kind}.")


def _traversal(G, produce):
    guard = _MutationGuard(G)

    def generate():
        try:
            for item in produce():
                if guard.changed():
                    raise RuntimeError("Graph changed during iteration")
                yield item
        finally:
            guard.release()

    return generate()


def bfs_edges(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    if sort_neighbors is not None:
        raise NotImplementedError("rustnx does not support sort_neighbors")
    s, missing = _start(G, source)
    depth = _depth_limit(G, depth_limit)
    reverse = bool(reverse) and G.is_directed()
    if reverse:
        G._ensure_exact_pred()

    def produce():
        if s is None:
            raise missing
        parents, children = G._core.bfs_edges(s, depth, reverse)
        nodes = G._nodes
        for p, c in zip(parents, children):
            yield nodes[p], nodes[c]

    return _traversal(G, produce)


def _dfs(G, source, depth_limit, sort_neighbors, preorder):
    if sort_neighbors is not None:
        raise NotImplementedError("rustnx does not support sort_neighbors")
    depth = _depth_limit(G, depth_limit)
    if source is None:
        s, missing, starts = 0, None, None
    else:
        s, missing = _start(G, source)
        starts = [s]

    def produce():
        if s is None:
            if preorder:
                yield source  # NetworkX yields the start before looking it up
            raise missing
        parents, children = G._core.dfs_forward(starts, depth)
        nodes = G._nodes
        for p, c in zip(parents, children):
            if p == c:  # a new start
                if preorder:
                    yield nodes[p]
            elif preorder:
                yield nodes[c]
            else:
                yield nodes[p], nodes[c]

    return _traversal(G, produce)


def dfs_edges(G, source=None, depth_limit=None, *, sort_neighbors=None):
    return _dfs(G, source, depth_limit, sort_neighbors, preorder=False)


def dfs_preorder_nodes(G, source=None, depth_limit=None, *, sort_neighbors=None):
    return _dfs(G, source, depth_limit, sort_neighbors, preorder=True)


def bfs_tree(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    edges = bfs_edges(G, source, reverse=reverse, depth_limit=depth_limit, sort_neighbors=sort_neighbors)
    T = nx.DiGraph()
    T.add_node(source)
    T.add_edges_from(edges)
    return T


def dfs_tree(G, source=None, depth_limit=None, *, sort_neighbors=None):
    edges = dfs_edges(G, source, depth_limit, sort_neighbors=sort_neighbors)
    T = nx.DiGraph()
    if source is None:
        T.add_nodes_from(G._nodes)
    else:
        T.add_node(source)
    T.add_edges_from(edges)
    return T


# --- Spanning trees --------------------------------------------------------------


def _mst_algorithms():
    # NetworkX's own table (it also accepts e.g. "borůvka").
    from networkx.algorithms.tree import mst

    return getattr(mst, "ALGORITHMS", {"kruskal": None, "prim": None, "boruvka": None})


def _spanning_edges(G, algorithm, weight, data, maximum):
    if G.is_directed():
        raise nx.NetworkXNotImplemented("not implemented for directed type")
    try:
        known = algorithm in _mst_algorithms()
    except TypeError:
        raise NotImplementedError("unhashable algorithm") from None
    if not known:
        raise ValueError(f"{algorithm} is not a valid choice for an algorithm.")
    if algorithm != "kruskal":
        # Prim starts from `set(G).pop()`, which follows hash order.
        raise NotImplementedError("rustnx implements Kruskal's algorithm only")
    base = _networkx_graph(G)
    _unhidden_weight(G, weight)
    us, vs = G._core.kruskal(weight, maximum)
    guard = _MutationGuard(G)
    nodes = G._nodes
    adj = base._adj

    def generate():
        try:
            for u, v in zip(us, vs):
                if guard.changed():
                    raise RuntimeError("Graph changed during iteration")
                a, b = nodes[u], nodes[v]
                yield (a, b, adj[a][b]) if data else (a, b)
        finally:
            guard.release()

    return generate()


def minimum_spanning_edges(G, algorithm="kruskal", weight="weight", keys=True, data=True, ignore_nan=False):
    return _spanning_edges(G, algorithm, weight, data, maximum=False)


def maximum_spanning_edges(G, algorithm="kruskal", weight="weight", keys=True, data=True, ignore_nan=False):
    return _spanning_edges(G, algorithm, weight, data, maximum=True)


def _spanning_tree(G, weight, algorithm, maximum):
    edges = _spanning_edges(G, algorithm, weight, True, maximum)
    base = _networkx_graph(G)
    T = base.__class__()
    T.graph.update(base.graph)
    T.add_nodes_from(base.nodes.items())
    T.add_edges_from(edges)
    return T


def minimum_spanning_tree(G, weight="weight", algorithm="kruskal", ignore_nan=False):
    return _spanning_tree(G, weight, algorithm, maximum=False)


def maximum_spanning_tree(G, weight="weight", algorithm="kruskal", ignore_nan=False):
    return _spanning_tree(G, weight, algorithm, maximum=True)


# --- all_shortest_paths ----------------------------------------------------------


@functools.cache
def _path_skip_ends_pass():
    """Whether the installed NetworkX (before 3.7) ends a pass of
    ``_build_paths_from_predecessors`` when it skips a predecessor already on
    the path, which re-yields paths through zero-weight cycles."""
    H = nx.DiGraph()
    H.add_weighted_edges_from([(0, 1, 0), (1, 0, 0), (1, 2, 1), (0, 2, 1), (2, 3, 0)])
    # NetworkX's private helpers, called directly so that no nested call is
    # dispatched (possibly back to rustnx).
    from networkx.algorithms.shortest_paths import generic, weighted

    pred = {0: []}
    weighted._dijkstra_multisource(H, [0], lambda u, v, d: d["weight"], pred=pred)
    return len(list(generic._build_paths_from_predecessors({0}, 3, pred))) == 4


def all_shortest_paths(G, source, target, weight=None, method="dijkstra"):
    method = "unweighted" if weight is None else method
    if method == "unweighted":
        s = _index_of(G, source, f"Source {source} not in G")
    elif method == "dijkstra":
        s = _index_of(G, source, f"Node {source} is not found in the graph")
        weight, _, _ = _check_weight(G, weight)
    elif method == "bellman-ford":
        raise NotImplementedError("rustnx has no Bellman-Ford")
    else:
        raise ValueError(f"method not supported: {method}")
    try:
        t = G._index.get(target)
    except TypeError:
        raise NotImplementedError("unhashable target") from None
    paths = None if t is None else G._core.all_shortest_paths(
        s, t, weight if method == "dijkstra" else None, _path_skip_ends_pass()
    )

    def generate():
        if paths is None:
            raise nx.NetworkXNoPath(f"Target {target} cannot be reached from given sources")
        nodes = G._nodes
        while (p := paths.next_path()) is not None:
            path = [nodes[i] for i in p]
            path[-1] = target  # NetworkX starts the search from the caller's object
            yield path

    return generate()


# --- Coloring and communities ----------------------------------------------------


def greedy_color(G, strategy="largest_first", interchange=False):
    if len(G) == 0:
        return {}
    if not isinstance(strategy, str) or interchange:
        raise NotImplementedError("rustnx implements string strategies without interchange")
    order, colors = _greedy_order(G, strategy)
    nodes = G._nodes
    return {nodes[i]: colors[i] for i in order}


def label_propagation_communities(G):
    _undirected_only(G)
    order, colors = G._core.greedy_color()
    nodes = G._nodes
    # NetworkX visits each color class as a set, so build the same sets and
    # let Python decide the visiting order.
    coloring = {}
    for i in order:
        c = colors[i]
        if c in coloring:
            coloring[c].add(nodes[i])
        else:
            coloring[c] = {nodes[i]}
    index = G._index
    labels = G._core.label_propagation([index[n] for members in coloring.values() for n in members])
    clusters = defaultdict(set)
    for node, label in zip(nodes, labels):
        clusters[label].add(node)
    return clusters.values()


# --- More traversal ----------------------------------------------------------------


@functools.cache
def _bfs_predecessors_deprecation():
    """The DeprecationWarning message the installed NetworkX gives for
    ``bfs_predecessors`` (3.7+), or ``None``."""
    H = nx.Graph([(0, 1)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        list(nx.bfs_predecessors(H, 0, backend="networkx"))
    for w in caught:
        if issubclass(w.category, DeprecationWarning):
            return str(w.message)
    return None


def bfs_predecessors(G, source, depth_limit=None, sort_neighbors=None):
    edges = bfs_edges(G, source, depth_limit=depth_limit, sort_neighbors=sort_neighbors)
    deprecation = _bfs_predecessors_deprecation()

    def generate():
        if deprecation is not None:
            warnings.warn(deprecation, category=DeprecationWarning, stacklevel=2)
        for s, t in edges:
            yield t, s

    return generate()


def bfs_successors(G, source, depth_limit=None, sort_neighbors=None):
    edges = bfs_edges(G, source, depth_limit=depth_limit, sort_neighbors=sort_neighbors)

    def generate():
        # As in NetworkX, the first parent is the caller's `source` object.
        parent = source
        children = []
        for p, c in edges:
            if p == parent:
                children.append(c)
                continue
            yield parent, children
            children = [c]
            parent = p
        yield parent, children

    return generate()


@functools.cache
def _bfs_layers_list_first():
    """Whether the installed NetworkX (3.4) takes the first layer as
    ``list(sources)`` before building the visited set from ``sources``
    (later releases use ``list(set(sources))``). With an iterator the
    visited set then starts empty."""
    H = nx.Graph()
    H.add_node(0)
    return next(nx.bfs_layers(H, [0, 0], backend="networkx")) == [0, 0]


def bfs_layers(G, sources):
    def produce():
        starts = sources
        if starts in G:
            starts = [starts]
        # Built exactly as NetworkX builds them, so the first layer (and an
        # iterator `sources`) behaves the same.
        if _bfs_layers_list_first():
            current = list(starts)
            visited = set(starts)
        else:
            visited = set(starts)
            current = list(visited)
        for source in current:
            if source not in G:
                raise nx.NetworkXError(f"The node {source} is not in the graph.")
        if not current:
            return
        index = G._index
        later, ends = G._core.bfs_layers([index[v] for v in current], [index[v] for v in visited])
        yield current
        nodes = G._nodes
        begin = 0
        for end in ends:
            yield [nodes[i] for i in later[begin:end]]
            begin = end

    return _traversal(G, produce)


def descendants_at_distance(G, source, distance):
    if source not in G:
        raise nx.NetworkXError(f"The node {source} is not in the graph.")
    # NetworkX compares each layer number `i` with `distance`.
    if isinstance(distance, float) and distance.is_integer():
        distance = int(distance)
    try:
        distance = operator.index(distance)
    except TypeError:
        raise NotImplementedError("rustnx needs an integer distance") from None
    if distance < 0:
        return set()
    if distance == 0:
        return {source}
    # A BFS cut off at `distance`, like NetworkX stopping at that layer.
    order, levels = G._core.bfs_lengths(G._index[source], float(distance))
    nodes = G._nodes
    return {nodes[v] for v, level in zip(order, levels) if level == distance}


def dfs_postorder_nodes(G, source=None, depth_limit=None, *, sort_neighbors=None):
    if sort_neighbors is not None:
        raise NotImplementedError("rustnx does not support sort_neighbors")
    depth = _depth_limit(G, depth_limit)
    if source is None:
        s, missing, starts = 0, None, None
    else:
        s, missing = _start(G, source)
        starts = [s]

    def produce():
        if s is None:
            raise missing
        nodes = G._nodes
        for v in G._core.dfs_postorder(starts, depth):
            yield nodes[v]

    return _traversal(G, produce)


def dfs_predecessors(G, source=None, depth_limit=None, *, sort_neighbors=None):
    return {t: s for s, t in dfs_edges(G, source, depth_limit, sort_neighbors=sort_neighbors)}


def dfs_successors(G, source=None, depth_limit=None, *, sort_neighbors=None):
    d = defaultdict(list)
    for s, t in dfs_edges(G, source=source, depth_limit=depth_limit, sort_neighbors=sort_neighbors):
        d[s].append(t)
    return dict(d)


# --- More components ---------------------------------------------------------------


def node_connected_component(G, n):
    _undirected_only(G)
    seen = {n}  # the caller's object, as in NetworkX (and its TypeError)
    try:
        s = G._index[n]
    except KeyError:
        raise KeyError(n) from None
    order, _ = G._core.bfs_lengths(s, math.inf)
    nodes = G._nodes
    seen.update([nodes[i] for i in order[1:]])
    return seen


def articulation_points(G):
    _undirected_only(G)

    def produce():
        nodes = G._nodes
        for v in G._core.articulation_points():
            yield nodes[v]

    return _traversal(G, produce)


def _biconnected_edges(G):
    """Each biconnected component as a list of ``(u, v)`` edges."""
    us, vs, ends = G._core.biconnected_components()
    nodes = G._nodes
    begin = 0
    for end in ends:
        yield [(nodes[us[i]], nodes[vs[i]]) for i in range(begin, end)]
        begin = end


def biconnected_component_edges(G):
    _undirected_only(G)
    return _traversal(G, lambda: _biconnected_edges(G))


def biconnected_components(G):
    _undirected_only(G)

    def produce():
        for edges in _biconnected_edges(G):
            yield set(chain.from_iterable(edges))

    return _traversal(G, produce)


def is_biconnected(G):
    _undirected_only(G)
    return G._core.is_biconnected()


def attracting_components(G):
    _directed_only(G)
    comps = G._core.attracting_components(_scc_early_exit())
    guard = _MutationGuard(G)

    def generate():
        # NetworkX computes every component when iteration starts, so only a
        # change before then matters.
        try:
            changed = guard.changed()
        finally:
            guard.release()
        if changed:
            yield from nx.attracting_components(guard.graph, backend="networkx")
            return
        nodes = G._nodes
        for comp in comps:
            yield {nodes[i] for i in comp}

    return generate()


def number_attracting_components(G):
    _directed_only(G)
    return len(G._core.attracting_components(_scc_early_exit()))


def is_attracting_component(G):
    _directed_only(G)
    comps = G._core.attracting_components(_scc_early_exit())
    return len(comps) == 1 and len(comps[0]) == len(G)


# --- Degree centrality and trees ---------------------------------------------------


def _degree_centrality(G, degrees):
    if len(G) <= 1:
        return {n: 1 for n in G._nodes}
    s = 1.0 / (len(G) - 1.0)
    return {n: d * s for n, d in zip(G._nodes, degrees)}


def degree_centrality(G):
    return _degree_centrality(G, G._core.degrees())


def in_degree_centrality(G):
    _directed_only(G)
    return _degree_centrality(G, G._core.in_out_degrees()[0])


def out_degree_centrality(G):
    _directed_only(G)
    return _degree_centrality(G, G._core.in_out_degrees()[1])


def _component_count(G):
    if G.is_directed():
        return len(G._core.weakly_connected_components())
    return len(G._core.connected_components())


def is_tree(G):
    n = len(G)
    if n == 0:
        raise nx.NetworkXPointlessConcept("G has no nodes.")
    return n - 1 == G._core.number_of_edges() and _component_count(G) == 1


def is_forest(G):
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("G has no nodes.")
    # Every component has at least (size - 1) edges, so each has exactly
    # that many (NetworkX's test) if and only if the totals agree.
    return G._core.number_of_edges() == len(G) - _component_count(G)


# --- Batch 2: shortest paths -------------------------------------------------------


_BELLMAN_FORD_UNBOUNDED = "Negative cycle detected."
_NO_PARENT = 2**32 - 1  # `paths::NO_PARENT`


def _same_node_type(G, node, i):
    """Decline when the caller's node object is an equal object of another
    type (``1.0`` for ``1``): NetworkX puts the caller's object in some
    results and the graph's in others, and rustnx doesn't track which."""
    if type(node) is not type(G._nodes[i]):
        raise NotImplementedError("node given as an equal object of another type")


def _require_hashable(node):
    try:
        hash(node)
    except TypeError:
        raise NotImplementedError("unhashable node") from None


def _position(G, node):
    """Position of ``node``, or ``None`` if it isn't in G (unhashable nodes
    are declined: NetworkX raises a TypeError for them at some later point)."""
    _require_hashable(node)
    i = G._index.get(node)
    if i is not None:
        _same_node_type(G, node, i)
    return i


def _cutoff(cutoff):
    return None if cutoff is None else float(cutoff)


def _lists(G, keys, flat, ends):
    """``{node: [nodes]}`` from flattened lists of positions."""
    nodes = G._nodes
    result = {}
    begin = 0
    for k, end in zip(keys, ends):
        result[nodes[k]] = [nodes[i] for i in flat[begin:end]]
        begin = end
    return result


def _length_dict(G, order, dists, all_int, zeros=(0,)):
    """``{node: distance}``; the entries at ``zeros`` (order indices) are the
    int 0 NetworkX starts its sources at."""
    if all_int:
        dists = [int(d) for d in dists]
    for k in zeros:
        dists[k] = 0
    nodes = G._nodes
    return dict(zip([nodes[i] for i in order], dists))


# Multi-source Dijkstra


def _multi_sources(G, sources):
    if iter(sources) is sources:
        # NetworkX consumes an iterator while checking it.
        raise NotImplementedError("rustnx needs a reusable container of sources")
    if not sources:
        raise ValueError("sources must not be empty")
    positions = []
    for s in sources:
        i = _index_of(G, s, f"Node {s} not found in graph")
        _same_node_type(G, s, i)
        positions.append(i)
    return positions


def _multi_forest(G, positions, weight, cutoff, lengths):
    weight, all_int, _ = _check_weight(G, weight, distances=lengths)
    order, dists, parents, seen, roots = G._core.dijkstra_forest(positions, weight, _cutoff(cutoff))
    return order, _length_dict(G, order, dists, all_int, roots), parents, seen, roots


def multi_source_dijkstra_path_length(G, sources, cutoff=None, weight="weight"):
    positions = _multi_sources(G, sources)
    return _multi_forest(G, positions, weight, cutoff, True)[1]


def _multi_paths(G, positions, order, parents, seen, roots):
    distinct = list(dict.fromkeys(positions))
    k = len(distinct)
    if _dijkstra_paths_in_pop_order():
        # NetworkX 3.6+ skips the first `len(sources)` popped nodes, assuming
        # they are the sources. Negative weights can break that (and make it
        # raise KeyError); leave those cases to NetworkX.
        if len(roots) != k or set(order[:k]) != set(distinct):
            raise NotImplementedError("negative weights reorder the sources")
        key_order = distinct + order[k:]
    else:
        key_order = seen
    nodes = G._nodes
    by_index = {}
    for v, p in zip(order, parents):
        by_index[v] = [nodes[v]] if p == _NO_PARENT else [*by_index[p], nodes[v]]
    return {nodes[i]: by_index[i] for i in key_order}


def multi_source_dijkstra(G, sources, target=None, cutoff=None, weight="weight"):
    return _multi_source_dijkstra(G, sources, target, cutoff, weight, True)


def _multi_source_dijkstra(G, sources, target, cutoff, weight, lengths):
    positions = _multi_sources(G, sources)
    if target in sources:
        return (0, [target])
    if target is None:
        order, dist, parents, seen, roots = _multi_forest(G, positions, weight, cutoff, lengths)
        return dist, _multi_paths(G, positions, order, parents, seen, roots)
    t = _position(G, target)
    weight, all_int, _ = _check_weight(G, weight, distances=lengths)
    found = None
    if t is not None:
        found = G._core.dijkstra_forest_path(positions, t, weight, _cutoff(cutoff))
    else:
        # NetworkX searches the whole graph first: a negative cycle would
        # raise before NoPath does.
        G._core.dijkstra_forest(positions, weight, _cutoff(cutoff))
    if found is None:
        raise nx.NetworkXNoPath(f"No path to {target}.")
    dist, path = found
    nodes = G._nodes
    return int(dist) if all_int else dist, [nodes[i] for i in path]


def multi_source_dijkstra_path(G, sources, cutoff=None, weight="weight"):
    return _multi_source_dijkstra(G, sources, None, cutoff, weight, False)[1]


def dijkstra_predecessor_and_distance(G, source, cutoff=None, weight="weight"):
    s = _index_of(G, source, f"Node {source} is not found in the graph")
    _same_node_type(G, source, s)
    weight, all_int, _ = _check_weight(G, weight, distances=True)
    order, dists, keys, flat, ends = G._core.dijkstra_pred_dist(s, weight, _cutoff(cutoff))
    return _lists(G, keys, flat, ends), _length_dict(G, order, dists, all_int)


# Unweighted


def _max_level(cutoff):
    """The last BFS level ``nx.predecessor`` builds: it stops after the
    first level with ``cutoff <= level`` (a falsy cutoff never stops it)."""
    if not cutoff:
        return None
    if type(cutoff) not in (int, float, bool):
        raise NotImplementedError("rustnx needs an int or float cutoff")
    if math.isnan(cutoff) or cutoff == math.inf:
        return None
    if cutoff <= 1:
        return 1
    level = max(1, math.ceil(cutoff))
    if level >= 2**32 - 1:
        return None
    return level


def predecessor(G, source, target=None, cutoff=None, return_seen=None):
    s = _index_of(G, source, f"Source {source} not in G")
    _same_node_type(G, source, s)
    t = None if target is None else _position(G, target)
    keys, levels, flat, ends = G._core.bfs_pred(s, _max_level(cutoff))
    if target is not None:
        try:
            k = keys.index(t) if t is not None else None
        except ValueError:
            k = None
        if k is None:
            return ([], -1) if return_seen else []
        nodes = G._nodes
        found = [nodes[i] for i in flat[ends[k - 1] if k else 0 : ends[k]]]
        return (found, levels[k]) if return_seen else found
    pred = _lists(G, keys, flat, ends)
    if return_seen:
        return pred, dict(zip(pred, levels))
    return pred


@functools.cache
def _single_target_length_iterator():
    """``(True, message)`` if the installed NetworkX (3.4) returns an
    iterator from ``single_target_shortest_path_length`` and warns with
    ``message``; ``(False, None)`` if it returns a dict."""
    H = nx.Graph([(0, 1)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = nx.single_target_shortest_path_length(H, 0, backend="networkx")
    if isinstance(result, dict):
        return False, None
    for w in caught:
        if issubclass(w.category, FutureWarning):
            return True, str(w.message)
    return True, None


def single_target_shortest_path_length(G, target, cutoff=None):
    t = _index_of(G, target, f"Target {target} is not in G")
    _same_node_type(G, target, t)
    if G.is_directed():
        G._ensure_exact_pred()
    cutoff = math.inf if cutoff is None else float(cutoff)
    is_iter, message = _single_target_length_iterator()
    if message is not None:
        warnings.warn(message, FutureWarning, stacklevel=2)
    order, levels = G._core.bfs_lengths_reverse(t, cutoff)
    nodes = G._nodes
    if not is_iter:
        return dict(zip([nodes[i] for i in order], levels))

    def produce():
        for i, level in zip(order, levels):
            yield nodes[i], level

    return _traversal(G, produce)


# All shortest paths from one source


@functools.cache
def _all_paths_over_pred():
    """Whether the installed NetworkX (3.5+) yields
    ``single_source_all_shortest_paths`` in predecessor-dict order, rather
    than in graph order skipping unreached nodes (3.4)."""
    # Read from the source: running it would dispatch its nested
    # `nx.predecessor` call, possibly back to rustnx.
    try:
        source = inspect.getsource(nx.single_source_all_shortest_paths.orig_func)
    except (AttributeError, OSError, TypeError):
        return True
    return "for n in G:" not in source


def _bf_weight(G, weight, lengths=True):
    weight, all_int, has_hidden = _check_weight(G, weight, distances=lengths)
    if has_hidden:
        # NetworkX's Bellman-Ford adds the None and raises TypeError, if and
        # when it reaches such an edge.
        raise NotImplementedError("rustnx does not support None edge weights here")
    return weight, all_int


def _pred_paths(G, s, source, weight, method):
    """The predecessor lists ``single_source_all_shortest_paths`` builds,
    raising what NetworkX raises (lazily, from inside its generator)."""
    if method == "unweighted":
        if s is None:
            raise nx.NodeNotFound(f"Source {source} not in G")
        return G._core.pred_paths(s, 0)
    if method == "dijkstra":
        if s is None:
            raise nx.NodeNotFound(f"Node {source} is not found in the graph")
        return G._core.pred_paths(s, 1, weight)
    if method == "bellman-ford":
        if s is None:
            raise nx.NodeNotFound(f"Node {source} is not found in the graph")
        if G._core.negative_selfloop(weight):
            raise nx.NetworkXUnbounded(_BELLMAN_FORD_UNBOUNDED)
        found = G._core.pred_paths(s, 2, weight)
        if found is None:
            raise nx.NetworkXUnbounded(_BELLMAN_FORD_UNBOUNDED)
        return found
    raise ValueError(f"method not supported: {method}")


def _check_paths_method(G, weight, method):
    method = "unweighted" if weight is None else method
    if method == "dijkstra":
        weight, _, _ = _check_weight(G, weight)
    elif method == "bellman-ford":
        weight, _ = _bf_weight(G, weight, lengths=False)
    return weight, method


def _all_paths_from(G, s, source, weight, method):
    """``[(node, [paths])]`` as ``single_source_all_shortest_paths`` yields."""
    found = _pred_paths(G, s, source, weight, method)
    if _all_paths_over_pred():
        targets = found.order()
    else:
        targets = list(range(len(G)))
    reached, flat, ends, groups = found.all_paths(targets, _path_skip_ends_pass())
    nodes = G._nodes
    result = []
    begin = 0
    path_begin = 0
    for t, group_end in zip(reached, groups):
        group = []
        for end in ends[begin:group_end]:
            group.append([nodes[i] for i in flat[path_begin:end]])
            path_begin = end
        begin = group_end
        result.append((nodes[t], group))
    return result


def single_source_all_shortest_paths(G, source, weight=None, method="dijkstra"):
    weight, method = _check_paths_method(G, weight, method)
    try:
        s = G._index.get(source)
    except TypeError:
        s = None  # NetworkX's `source not in G` is True
    if s is not None:
        _same_node_type(G, source, s)
    guard = _MutationGuard(G)

    def generate():
        # NetworkX builds the predecessors when iteration starts. From 3.5
        # it then never reads the graph again, so only a change before then
        # matters; 3.4 keeps iterating over the graph's nodes.
        try:
            if guard.changed():
                guard.release()
                yield from nx.single_source_all_shortest_paths(
                    guard.graph, source, weight=weight, method=method, backend="networkx"
                )
                return
            found = _all_paths_from(G, s, source, weight, method)
            live = not _all_paths_over_pred()
            if not live:
                guard.release()
            for item in found:
                if live and guard.changed():
                    raise RuntimeError("Graph changed during iteration")
                yield item
        finally:
            guard.release()

    return generate()


def all_pairs_all_shortest_paths(G, weight=None, method="dijkstra"):
    weight, method = _check_paths_method(G, weight, method)
    if method not in ("unweighted", "dijkstra", "bellman-ford"):
        # Raised lazily, from the first source: an empty graph never raises.
        def failing():
            if len(G):
                raise ValueError(f"method not supported: {method}")
            yield from ()

        return failing()

    def compute(batch):
        for s in batch:
            yield dict(_all_paths_from(G, s, G._nodes[s], weight, method))

    return _all_pairs(G, compute, batch_size=1)


# Bellman-Ford


def _bellman_ford(G, s, weight, heuristic=True, shortcut=False, mode=3, target=None):
    found = G._core.bellman_ford(s, weight, heuristic, shortcut, mode, target)
    if found is None:
        raise nx.NetworkXUnbounded(_BELLMAN_FORD_UNBOUNDED)
    return found


def bellman_ford_predecessor_and_distance(G, source, target=None, weight="weight", heuristic=False):
    s = _index_of(G, source, f"Node {source} is not found in the graph")
    _same_node_type(G, source, s)
    weight, all_int = _bf_weight(G, weight)
    if G._core.negative_selfloop(weight):
        raise nx.NetworkXUnbounded(_BELLMAN_FORD_UNBOUNDED)
    order, dists, flat, ends = _bellman_ford(G, s, weight, bool(heuristic), True, mode=0)
    return _lists(G, order, flat, ends), _length_dict(G, order, dists, all_int)


def _bf_source(G, source):
    # NetworkX builds `{source: [...]}` before checking `source in G`.
    _require_hashable(source)
    s = _index_of(G, source, f"Source {source} not in G")
    _same_node_type(G, source, s)
    return s


def single_source_bellman_ford_path_length(G, source, weight="weight"):
    s = _bf_source(G, source)
    weight, all_int = _bf_weight(G, weight)
    order, dists, _, _ = _bellman_ford(G, s, weight)
    return _length_dict(G, order, dists, all_int)


def bellman_ford_path_length(G, source, target, weight="weight"):
    if source == target:
        if source not in G:
            raise nx.NodeNotFound(f"Node {source} not found in graph")
        return 0
    s = _bf_source(G, source)
    t = _position(G, target)
    weight, all_int = _bf_weight(G, weight)
    order, dists, _, _ = _bellman_ford(G, s, weight)
    if t is not None:
        try:
            d = dists[order.index(t)]
        except ValueError:
            pass
        else:
            return int(d) if all_int else d
    raise nx.NetworkXNoPath(f"node {target} not reachable from {source}")


def single_source_bellman_ford(G, source, target=None, weight="weight"):
    return _single_source_bellman_ford(G, source, target, weight, True)


def _single_source_bellman_ford(G, source, target, weight, lengths):
    if source == target:
        if source not in G:
            raise nx.NodeNotFound(f"Node {source} is not found in the graph")
        return (0, [source])
    s = _bf_source(G, source)
    t = None if target is None else _position(G, target)
    weight, all_int = _bf_weight(G, weight, lengths=lengths)
    if target is None:
        order, dists, flat, ends = _bellman_ford(G, s, weight, mode=1)
        return _length_dict(G, order, dists, all_int), _lists(G, order, flat, ends)
    order, dists, flat, ends = _bellman_ford(G, s, weight, mode=2, target=t)
    if not ends:
        raise nx.NetworkXNoPath(f"Target {target} cannot be reached from given sources")
    d = dists[order.index(t)]
    nodes = G._nodes
    return int(d) if all_int else d, [nodes[i] for i in flat]


def bellman_ford_path(G, source, target, weight="weight"):
    return _single_source_bellman_ford(G, source, target, weight, False)[1]


def single_source_bellman_ford_path(G, source, weight="weight"):
    return _single_source_bellman_ford(G, source, None, weight, False)[1]


# Each source's Bellman-Ford output can be large; keep batches small.
_BELLMAN_FORD_BATCH = 64


def all_pairs_bellman_ford_path_length(G, weight="weight"):
    weight, all_int = _bf_weight(G, weight)

    def compute(batch):
        for found in G._core.bellman_ford_many(batch, weight, 3):
            if found is None:
                raise nx.NetworkXUnbounded(_BELLMAN_FORD_UNBOUNDED)
            order, dists, _, _ = found
            yield _length_dict(G, order, dists, all_int)

    return _all_pairs(G, compute, _BELLMAN_FORD_BATCH)


def all_pairs_bellman_ford_path(G, weight="weight"):
    weight, _ = _bf_weight(G, weight, lengths=False)

    def compute(batch):
        for found in G._core.bellman_ford_many(batch, weight, 1):
            if found is None:
                raise nx.NetworkXUnbounded(_BELLMAN_FORD_UNBOUNDED)
            order, _, flat, ends = found
            yield _lists(G, order, flat, ends)

    return _all_pairs(G, compute, _BELLMAN_FORD_BATCH)


def negative_edge_cycle(G, weight="weight", heuristic=True):
    if G._core.number_of_edges() == 0:
        return False
    weight, _ = _bf_weight(G, weight, lengths=False)
    return G._core.negative_edge_cycle(weight, bool(heuristic))


def find_negative_cycle(G, source, weight="weight"):
    s = _bf_source(G, source)
    weight, _ = _bf_weight(G, weight, lengths=False)
    found = G._core.find_negative_cycle(s, weight)
    if found is None:
        raise nx.NetworkXError("No negative cycles detected.")
    identified, cycle = found
    if not identified:
        v = cycle[0]
        if G._core.has_edge(v, v):
            # NetworkX calls `weight(G, v, v)` here, which fails in ways
            # that depend on the node type.
            raise NotImplementedError("negative cycle not identified")
        raise nx.NetworkXError("Negative cycle is detected but not found")
    nodes = G._nodes
    return [nodes[i] for i in cycle]


# A*


def _astar(G, s, t, source, target, heuristic, weight, cutoff, lengths):
    if heuristic is not None:
        raise NotImplementedError("rustnx supports A* without a heuristic only")
    _same_node_type(G, source, s)
    _same_node_type(G, target, t)
    weight, all_int, _ = _check_weight(G, weight, distances=lengths)
    if G._core.has_negative_weight(weight):
        # A* may never finish on a negative cycle; leave these to NetworkX.
        raise NotImplementedError("rustnx's A* needs non-negative weights")
    if cutoff:
        if type(cutoff) not in (int, float, bool):
            raise NotImplementedError("rustnx needs an int or float cutoff")
        cutoff = float(cutoff)
    else:
        cutoff = None
    found = G._core.astar(s, t, weight, cutoff)
    if found is None:
        raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")
    path, costs = found
    nodes = G._nodes
    return [nodes[i] for i in path], [int(c) for c in costs] if all_int else costs


def astar_path(G, source, target, heuristic=None, weight="weight", *, cutoff=None):
    s = _index_of(G, source, f"Source {source} is not in G")
    t = _index_of(G, target, f"Target {target} is not in G")
    return _astar(G, s, t, source, target, heuristic, weight, cutoff, False)[0]


def astar_path_length(G, source, target, heuristic=None, weight="weight", *, cutoff=None):
    if source not in G or target not in G:
        raise nx.NodeNotFound(f"Either source {source} or target {target} is not in G")
    s, t = G._index[source], G._index[target]
    # The same additions as NetworkX's `sum(weight(u, v, G[u][v]) ...)`.
    return sum(_astar(G, s, t, source, target, heuristic, weight, cutoff, True)[1])


# --- Batch 3: DAGs, traversal and components ---------------------------------------


def _topological_order_or_raise(G):
    generations, has_cycle = G._core.topological_generations()
    if has_cycle:
        raise nx.NetworkXUnfeasible(
            "Graph contains a cycle or graph changed during iteration"
        )
    return [v for generation in generations for v in generation]


def has_cycle(G):
    if not G.is_directed():
        raise nx.NetworkXError("Topological sort not defined on undirected graphs.")
    return G._core.topological_generations()[1]


def _longest_path_weight(G, weight, default_weight):
    """``(attr, constant)`` for ``G._core.dag_longest_path``.

    NetworkX reads ``data.get(weight, default_weight)``; the conversion fills
    in missing values with the default it was made with, so that must be
    ``default_weight`` (an int stays an int, a float a float).
    """
    if type(default_weight) not in (int, float) or not math.isfinite(default_weight):
        raise NotImplementedError("rustnx needs a numeric default_weight")
    if weight is None:
        return None, float(default_weight)
    if callable(weight) or not isinstance(weight, str):
        raise NotImplementedError("rustnx needs a string weight attribute")
    if G._core.is_native():
        stored = 1  # native graphs read missing values as 1
    else:
        stored = next((d for name, d in G._weight_attrs if name == weight), None)
        if stored is None:
            stored = 1
            G._ensure_weight(weight)  # converts with the default 1
    if type(stored) is not type(default_weight) or stored != default_weight:
        raise NotImplementedError("converted with a different default_weight")
    _unhidden_weight(G, weight)
    return weight, 0.0


def _dag_longest_path(G, weight, default_weight):
    _directed_only(G)
    if len(G) == 0:
        return []
    attr, constant = _longest_path_weight(G, weight, default_weight)
    G._ensure_exact_pred()  # ties go to the first predecessor in G.pred
    cycle, path = G._core.dag_longest_path(attr, constant)
    if cycle:
        raise nx.NetworkXUnfeasible(
            "Graph contains a cycle or graph changed during iteration"
        )
    if path is None:
        raise NotImplementedError("path lengths too large for exact sums")
    return path


def dag_longest_path(G, weight="weight", default_weight=1):
    nodes = G._nodes
    return [nodes[i] for i in _dag_longest_path(G, weight, default_weight)]


def dag_longest_path_length(G, weight="weight", default_weight=1):
    path = _dag_longest_path(G, weight, default_weight)
    nodes = G._nodes
    if not G._core.is_native():
        # Sum the edge data exactly as NetworkX does (ints stay ints).
        base = _networkx_graph(G)
        path_length = 0
        for u, v in zip(path, path[1:]):
            path_length += base[nodes[u]][nodes[v]].get(weight, default_weight)
        return path_length
    if weight is None:
        values = [default_weight] * (len(path) - 1)
    elif G._core.weight_mixed(weight):
        raise NotImplementedError("edge weights mix ints and floats")
    else:
        values = G._core.path_weights(path, weight)
        if G._core.weight_info(weight)[0]:
            values = map(int, values)
    path_length = 0
    for w in values:
        path_length += w
    return path_length


def _sort_ranks(G):
    """Each node's position in sorted order, if Python's ``<`` orders the
    nodes totally (ints and floats without NaN, or strings); else ``None``."""
    nodes = G._nodes
    kinds = {type(v) for v in nodes}
    if kinds <= {int, float}:
        if float in kinds and any(v != v for v in nodes):
            return None
    elif kinds != {str}:
        return None
    # NetworkX's heap entries are (node, position in G, node).
    order = sorted(range(len(nodes)), key=lambda i: (nodes[i], i))
    rank = [0] * len(nodes)
    for r, i in enumerate(order):
        rank[i] = r
    return rank


def lexicographical_topological_sort(G):
    rank = _sort_ranks(G) if G.is_directed() else None
    if G.is_directed() and rank is None:
        raise NotImplementedError("rustnx sorts ints, floats or strings only")
    guard = _MutationGuard(G)
    return _lexicographical_topological_sort(G, guard, rank)


def _lexicographical_topological_sort(G, guard, rank):
    try:
        if not G.is_directed():
            raise nx.NetworkXError("Topological sort not defined on undirected graphs.")
        order, cycle = G._core.lexicographical_topological_sort(rank)
        if guard.changed():
            # Changed before iteration started: NetworkX would sort the
            # changed graph.
            yield from nx.lexicographical_topological_sort(guard.graph, backend="networkx")
            return
        nodes = G._nodes
        for i, v in enumerate(order):
            yield nodes[v]
            if guard.changed():
                yield from _resume_lexicographical(G, guard, order, i)
                return
        if cycle:
            raise nx.NetworkXUnfeasible(
                "Graph contains a cycle or graph changed during iteration"
            )
    finally:
        guard.release()


def _resume_lexicographical(G, guard, order, done):
    """Continue exactly as NetworkX would after the graph changed.

    When NetworkX yields ``order[done]`` it has already processed that
    node's out-edges, so rebuild its heap and in-degree map from the
    snapshot and run NetworkX's loop on the live graph.
    """
    import heapq

    nodes = G._nodes
    processed = order[: done + 1]
    remaining = G._core.indegrees_after(processed)
    is_processed = [False] * len(nodes)
    for v in processed:
        is_processed[v] = True
    indegree_map = {nodes[v]: d for v, d in enumerate(remaining) if d > 0}
    nodeid_map = {n: i for i, n in enumerate(nodes)}

    def create_tuple(node):
        return node, nodeid_map[node], node

    zero_indegree = [
        create_tuple(nodes[v]) for v, d in enumerate(remaining) if d == 0 and not is_processed[v]
    ]
    heapq.heapify(zero_indegree)
    live = guard.graph
    # From networkx.algorithms.dag.lexicographical_topological_sort.
    while zero_indegree:
        _, _, node = heapq.heappop(zero_indegree)
        if node not in live:
            raise RuntimeError("Graph changed during iteration")
        for _, child in live.edges(node):
            try:
                indegree_map[child] -= 1
            except KeyError as err:
                raise RuntimeError("Graph changed during iteration") from err
            if indegree_map[child] == 0:
                try:
                    heapq.heappush(zero_indegree, create_tuple(child))
                except TypeError as err:
                    raise TypeError(
                        "Consider using `key=` parameter to resolve ambiguities in the sort order."
                    ) from err
                del indegree_map[child]
        yield node
    if indegree_map:
        raise nx.NetworkXUnfeasible(
            "Graph contains a cycle or graph changed during iteration"
        )


def all_topological_sorts(G):
    _directed_only(G)
    sorts = G._core.all_topological_sorts()

    def produce():
        nodes = G._nodes
        while True:
            ok, sort = sorts.next_sort()
            if not ok:
                raise nx.NetworkXUnfeasible("Graph contains a cycle.")
            if sort is None:
                return
            yield [nodes[i] for i in sort]

    return _traversal(G, produce)


def transitive_reduction(G):
    _directed_only(G)
    groups = G._core.transitive_reduction()
    if groups is None:
        raise nx.NetworkXError("Directed Acyclic Graph required for transitive_reduction")
    nodes = G._nodes
    TR = nx.DiGraph()
    TR.add_nodes_from(nodes)
    us, vs, _ = G._core.edges_in_order()
    kept = 0xFFFFFFFF
    begin = 0
    m = len(us)
    while begin < m:
        u = us[begin]
        end = begin
        while end < m and us[end] == u:
            end += 1
        node = nodes[u]
        if end - begin == 1:
            TR.add_edge(node, nodes[vs[begin]])
        else:
            # NetworkX adds the kept successors by iterating a set built from
            # G[u] and shrunk by one removal per successor still in it; a
            # removal may resize the set, so repeat the same removals.
            row = [nodes[vs[e]] for e in range(begin, end)]
            nbrs = set(row)
            removed = defaultdict(list)
            for k in range(end - begin):
                g = groups[begin + k]
                if g != kept:
                    removed[g].append(row[k])
            for g in sorted(removed):
                nbrs.difference_update(removed[g])
            TR.add_edges_from((node, v) for v in nbrs)
        begin = end
    return TR


def transitive_closure(G, reflexive=False):
    base = _networkx_graph(G)
    TC = base.copy()
    if reflexive not in {None, True, False}:
        raise nx.NetworkXError("Incorrect value for the parameter `reflexive`")
    if reflexive is not None and reflexive is not True and reflexive is not False:
        return TC  # e.g. 1 == True: NetworkX's branches all test identity
    nodes = G._nodes
    edge_bfs = reflexive is False
    for start in range(0, len(nodes), _ALL_PAIRS_BATCH):
        batch = list(range(start, min(start + _ALL_PAIRS_BATCH, len(nodes))))
        for i, heads in zip(batch, G._core.closure_heads(batch, edge_bfs)):
            v = nodes[i]
            if edge_bfs:
                found = [nodes[j] for j in heads]
            else:
                # NetworkX iterates `nx.descendants(G, v)` (a set built in
                # BFS order), or that set `| {v}`.
                found = {nodes[j] for j in heads}
                if reflexive is True:
                    found = found | {v}
            TC_v = TC[v]
            TC.add_edges_from((v, u) for u in found if u not in TC_v)
    return TC


def transitive_closure_dag(G):
    _directed_only(G)
    topo = _topological_order_or_raise(G)
    TC = _networkx_graph(G).copy()
    state = G._core.closure_dag()
    nodes = G._nodes
    index = G._index
    for i in reversed(topo):
        layer = state.layer2(i)
        if not layer:
            continue
        # NetworkX adds edges by iterating `descendants_at_distance(TC, v,
        # 2)`, a set: its order decides TC's adjacency order from here on.
        found = set([nodes[j] for j in layer])
        v = nodes[i]
        TC.add_edges_from((v, u) for u in found)
        state.append(i, [index[u] for u in found])
    return TC


def root_to_leaf_paths(G):
    if not G.is_directed():
        raise NotImplementedError("NetworkX raises AttributeError here")
    paths = G._core.root_to_leaf_paths()

    def produce():
        nodes = G._nodes
        while (p := paths.next_path()) is not None:
            yield [nodes[i] for i in p]

    return _traversal(G, produce)


def dag_to_branching(G):
    _directed_only(G)
    if G._core.topological_generations()[1]:
        raise nx.HasACycle("dag_to_branching is only defined for acyclic graphs")
    sources, parents = G._core.dag_to_branching()
    nodes = G._nodes
    B = nx.DiGraph()
    # nx.prefix_tree numbers tree nodes from 1 (0 and -1 are removed).
    B.add_nodes_from((i, {"source": nodes[s]}) for i, s in enumerate(sources, 1))
    B.add_edges_from(
        (p + 1, i) for i, p in enumerate(parents, 1) if p != 0xFFFFFFFF
    )
    return B


# Triples per Rust call in v_structures.
_TRIPLES_BATCH = 65536


def _colliders(G, v_structures):
    _directed_only(G)
    G._ensure_exact_pred()  # combinations of G.predecessors(node), in order
    guard = _MutationGuard(G)

    def generate():
        nodes = G._nodes
        get = nodes.__getitem__
        # guard.changed() inlined: every change made through the NetworkX
        # API clears this cache dict, removing the key. Native graphs (no
        # key) can't change.
        cache, key = guard._cache, guard._key
        start = 0
        try:
            while start < len(nodes):
                flat, start = G._core.colliders(start, _TRIPLES_BATCH, v_structures)
                # Tuples are built in C; NetworkX's loop is C-level too.
                triples = zip(map(get, flat[0::3]), map(get, flat[1::3]), map(get, flat[2::3]))
                if key is None:
                    yield from triples
                    continue
                for triple in triples:
                    if key not in cache:
                        raise RuntimeError("Graph changed during iteration")
                    yield triple
        finally:
            guard.release()

    return generate()


def v_structures(G):
    return _colliders(G, True)


@functools.cache
def _aperiodic_errors():
    """What the installed NetworkX's ``is_aperiodic`` does with an undirected
    graph (an exception type and args), and whether it requires a strongly
    connected graph (3.5+; 3.4 recurses on the unreached nodes)."""
    try:
        nx.is_aperiodic(nx.Graph([(0, 1)]), backend="networkx")
        undirected = None
    except Exception as exc:
        undirected = (type(exc), exc.args)
    try:
        nx.is_aperiodic(nx.DiGraph([(0, 1)]), backend="networkx")
        strong = False
    except nx.NetworkXError:
        strong = True
    return undirected, strong


def is_aperiodic(G):
    undirected, strong = _aperiodic_errors()
    if not G.is_directed():
        if undirected is None:
            raise NotImplementedError("unrecognized NetworkX is_aperiodic")
        raise undirected[0](*undirected[1])
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("Graph has no nodes.")
    if strong:
        if len(G._core.strongly_connected_components()[0]) != len(G):
            raise nx.NetworkXError("Graph is not strongly connected.")
    # On a strongly connected graph every version's search finds the gcd of
    # all cycle lengths (the period) from any spanning tree's levels.
    reached, g = G._core.aperiodic_bfs()
    if reached != len(G):
        # NetworkX 3.4 recurses on a subgraph whose first node follows set
        # iteration order.
        raise NotImplementedError("NetworkX recurses on the unreached nodes")
    return g == 1


_DFS_LABELS = ("forward", "nontree", "reverse", "reverse-depth_limit")


def dfs_labeled_edges(G, source=None, depth_limit=None):
    depth = _depth_limit(G, depth_limit)
    if source is None:
        s, missing, starts = 0, None, None
    else:
        s, missing = _start(G, source)
        starts = [s]

    def produce():
        if s is None:
            yield source, source, "forward"
            raise missing
        us, vs, labels = G._core.dfs_labeled_edges(starts, depth)
        nodes = G._nodes
        names = _DFS_LABELS
        for u, v, label in zip(us, vs, labels):
            yield nodes[u], nodes[v], names[label]

    return _traversal(G, produce)


# Indexed by the Rust labels (see `dag::label`).
_BFS_LABELS = ("forward", None, "reverse", None, "tree", "level")


def bfs_labeled_edges(G, sources):
    _MISSING = 0xFFFFFFFF

    def produce():
        starts = sources
        if starts in G:
            starts = [starts]
        depth = dict.fromkeys(starts, 0)  # as NetworkX builds it
        index = G._index
        positions = [index.get(s, _MISSING) for s in depth]
        us, vs, labels, missing = G._core.bfs_labeled_edges(positions)
        nodes = G._nodes
        names = _BFS_LABELS
        for u, v, label in zip(us, vs, labels):
            yield nodes[u], nodes[v], names[label]
        if missing is not None:
            raise KeyError(list(depth)[missing])

    return _traversal(G, produce)


def generic_bfs_edges(G, source, depth_limit=None):
    return bfs_edges(G, source, depth_limit=depth_limit)


def _edge_traversal(G, source, orientation, dfs):
    if G.is_multigraph():
        raise NotImplementedError("rustnx does not support multigraph edge keys")
    if source is not None and source not in G:
        if not isinstance(source, (list, tuple, set, frozenset, dict)):
            raise NotImplementedError("rustnx needs a node or a container of nodes")
        for n in source:
            try:
                hash(n)
            except TypeError:
                raise NotImplementedError("unhashable node in source") from None
    directed = G.is_directed()
    if directed and orientation in ("reverse", "ignore"):
        G._ensure_exact_pred()  # G.in_edges(node) follows G.pred

    def produce():
        index = G._index
        if source is None:
            starts = list(range(len(G)))
        elif source in G:
            starts = [index[source]]
        else:
            starts = [index[n] for n in source if n in index]
        if not starts:
            return
        if orientation is None:
            out, inward, labeled = True, False, False
        elif not directed or orientation == "original":
            out, inward, labeled = True, False, True
        elif orientation == "reverse":
            out, inward, labeled = False, True, True
        elif orientation == "ignore":
            out, inward, labeled = True, True, True
        else:
            raise nx.NetworkXError("invalid orientation argument.")
        us, vs, labels = G._core.edge_traversal(starts, out, inward, dfs)
        nodes = G._nodes
        if labeled:
            names = _DFS_LABELS
            for u, v, label in zip(us, vs, labels):
                yield nodes[u], nodes[v], names[label]
        else:
            for u, v in zip(us, vs):
                yield nodes[u], nodes[v]

    return _traversal(G, produce)


def edge_bfs(G, source=None, orientation=None):
    return _edge_traversal(G, source, orientation, dfs=False)


def edge_dfs(G, source=None, orientation=None):
    return _edge_traversal(G, source, orientation, dfs=True)


@functools.cache
def _kosaraju_stack_order():
    """Whether the installed NetworkX's Kosaraju (3.7+) fills each component
    from an explicit stack, rather than in DFS preorder."""
    try:
        source = inspect.getsource(nx.kosaraju_strongly_connected_components.orig_func)
    except (AttributeError, OSError, TypeError):
        return None
    if "dfs_preorder_nodes" in source:
        return False
    if "stack.pop()" in source:
        return True
    return None


def kosaraju_strongly_connected_components(G, source=None):
    _directed_only(G)
    stack_order = _kosaraju_stack_order()
    if stack_order is None:
        raise NotImplementedError("unrecognized NetworkX Kosaraju")
    G._ensure_exact_pred()  # the DFS runs on G.reverse(copy=False)
    if source is None:
        s, missing = None, None
    else:
        s, missing = _start(G, source)

    def produce():
        if source is not None and s is None:
            raise missing
        nodes = G._nodes
        for comp in G._core.kosaraju(s, stack_order):
            yield {nodes[i] for i in comp}

    return _traversal(G, produce)


def condensation(G):
    _directed_only(G)
    mapping = {}
    members = {}
    C = nx.DiGraph()
    C.graph["mapping"] = mapping
    if len(G) == 0:
        return C
    comps, us, vs = G._core.condensation(_scc_early_exit())
    nodes = G._nodes
    for i, comp in enumerate(comps):
        component = {nodes[v] for v in comp}
        members[i] = component
        # NetworkX fills the mapping by iterating the set.
        mapping.update((n, i) for n in component)
    C.add_nodes_from(range(len(comps)))
    C.add_edges_from(zip(us, vs))
    nx.set_node_attributes(C, members, "members")
    return C


def is_semiconnected(G):
    _directed_only(G)
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept(
            "Connectivity is undefined for the null graph."
        )
    if len(G._core.weakly_connected_components()) != 1:
        return False
    return G._core.is_semiconnected()


# --- Batch 4: centrality -----------------------------------------------------------


def _positions(G, nodes):
    """Positions of ``nodes`` in order (repeats kept), declining missing or
    unhashable nodes (NetworkX raises different errors at different times
    for those)."""
    index = G._index
    try:
        return [index[v] for v in nodes]
    except (KeyError, TypeError):
        raise NotImplementedError("rustnx needs every node to be in G") from None


def _reusable(nodes):
    if iter(nodes) is nodes:
        raise NotImplementedError("rustnx needs a reusable container of nodes")
    return nodes


def _target_positions(G, targets):
    """Positions of the nodes in ``set(targets)`` (others are ignored)."""
    index = G._index
    try:
        return [index[v] for v in set(_reusable(targets)) if v in index]
    except TypeError:
        raise NotImplementedError("unhashable target") from None


def _source_text(func):
    """The installed NetworkX's source of ``func``, whitespace collapsed."""
    try:
        return " ".join(inspect.getsource(func.orig_func).split())
    except (AttributeError, OSError, TypeError):
        return ""


@functools.cache
def _subset_rescale():
    """How the installed NetworkX rescales subset betweenness: ``(node,
    edge)``, each a function ``(b, n, normalized, directed)`` or ``None``.
    3.4 and 3.5 use the subset module's own ``_rescale``/``_rescale_e``; 3.7
    uses the shared betweenness ``_rescale`` (with ``endpoints=False`` for
    nodes)."""
    from networkx.algorithms.centrality import betweenness_subset as mod

    plain = "_rescale(b, len(G), normalized=normalized, directed=G.is_directed())"
    no_ends = (
        "_rescale( b, len(G), normalized=normalized, directed=G.is_directed(), endpoints=False )"
    )
    node_src = _source_text(nx.betweenness_centrality_subset)
    edge_src = _source_text(nx.edge_betweenness_centrality_subset)
    node = edge = None
    if no_ends in node_src:
        def node(b, n, normalized, directed):
            return mod._rescale(b, n, normalized=normalized, directed=directed, endpoints=False)
    elif "b = " + plain in node_src:
        def node(b, n, normalized, directed):
            return mod._rescale(b, n, normalized=normalized, directed=directed)
    if "b = _rescale_e(" + plain[len("_rescale("):] in edge_src:
        def edge(b, n, normalized, directed):
            return mod._rescale_e(b, n, normalized=normalized, directed=directed)
    elif "b = " + plain in edge_src:
        def edge(b, n, normalized, directed):
            return mod._rescale(b, n, normalized=normalized, directed=directed)
    return node, edge


def betweenness_centrality_subset(G, sources, targets, normalized=False, weight=None):
    rescale = _subset_rescale()[0]
    if rescale is None:
        raise NotImplementedError("unrecognized NetworkX subset betweenness rescaling")
    weight = _unhidden_weight(G, weight)
    sources = _positions(G, sources)
    targets = _target_positions(G, targets)
    raw = G._core.betweenness_subset(sources, targets, weight)
    return rescale(dict(zip(G._nodes, raw)), len(G), normalized, G.is_directed())


def edge_betweenness_centrality_subset(G, sources, targets, normalized=False, weight=None):
    rescale = _subset_rescale()[1]
    if rescale is None:
        raise NotImplementedError("unrecognized NetworkX subset betweenness rescaling")
    if any(isinstance(v, tuple) for v in G._nodes):
        # NetworkX keeps nodes and edges as keys of one dict.
        raise NotImplementedError("rustnx does not support tuple nodes here")
    weight = _unhidden_weight(G, weight)
    sources = _positions(G, sources)
    targets = _target_positions(G, targets)
    raw = G._core.edge_betweenness_subset(sources, targets, weight)
    us, vs, _ = G._core.edges_in_order()
    nodes = G._nodes
    b = dict(zip([(nodes[u], nodes[v]) for u, v in zip(us, vs)], raw))
    return rescale(b, len(G), normalized, G.is_directed())


def _node_rank(G):
    """Each node's position in ``sorted(G)``, for NetworkX's sorts of
    ``(distance, node)`` tuples. Only all-int or all-str nodes, where
    Python's comparisons are a total order."""
    nodes = G._nodes
    kinds = set(map(type, nodes))
    if not (kinds <= {int} or kinds <= {str}):
        raise NotImplementedError("rustnx sorts only int or str nodes")
    rank = [0] * len(nodes)
    for r, i in enumerate(sorted(range(len(nodes)), key=nodes.__getitem__)):
        rank[i] = r
    return rank


def _number(value):
    if type(value) not in (int, float, bool):
        raise NotImplementedError("rustnx needs a numeric cutoff")
    return float(value)


def newman_betweenness_centrality(G, v=None, cutoff=None, normalized=True, weight=None):
    weight, _, _ = _check_weight(G, weight)
    if weight is None:
        # `nx.predecessor` stops when `cutoff and cutoff <= level`.
        cutoff = _number(cutoff) if cutoff else None
    elif cutoff is not None:
        cutoff = _number(cutoff)
    if v is not None:
        try:
            hash(v)
        except TypeError:
            raise NotImplementedError("unhashable node") from None
    raw = G._core.load(_node_rank(G), weight, cutoff)
    order = len(G)
    if v is not None:
        i = G._index.get(v)
        betweenness = 0.0 if i is None else raw[i]
        if normalized:
            if order <= 2:
                return betweenness
            betweenness *= 1.0 / ((order - 1) * (order - 2))
        return betweenness
    betweenness = dict(zip(G._nodes, raw))
    if normalized:
        if order <= 2:
            return betweenness
        scale = 1.0 / ((order - 1) * (order - 2))
        for k in betweenness:
            betweenness[k] *= scale
    return betweenness


def edge_load_centrality(G, cutoff=False):
    cutoff = _number(cutoff) if cutoff else None
    us, vs, values = G._core.edge_load(cutoff)
    nodes = G._nodes
    return dict(zip([(nodes[u], nodes[v]) for u, v in zip(us, vs)], values))


_MAX_EXACT_INT = 2**53


def _state_value(x):
    if type(x) not in (int, float, bool) or (type(x) is int and abs(x) > _MAX_EXACT_INT):
        raise NotImplementedError("rustnx needs int or float percolation states")
    return float(x)


def percolation_centrality(G, attribute="percolation", states=None, weight=None):
    weight = _unhidden_weight(G, weight)
    n = len(G)
    if n == 2:
        raise NotImplementedError("NetworkX divides by zero here")
    if states is None:
        states = nx.get_node_attributes(_networkx_graph(G), attribute, default=1)
    # The same loop as NetworkX, so the total is the same.
    p_sigma_x_t = 0.0
    for x in states.values():
        p_sigma_x_t += x
    if type(p_sigma_x_t) is not float:
        raise NotImplementedError("rustnx needs int or float percolation states")
    try:
        values = [_state_value(states[v]) for v in G._nodes]
    except (KeyError, TypeError):
        raise NotImplementedError("a node has no percolation state") from None
    if n > 1 and any(p_sigma_x_t - x == 0 for x in values):
        raise NotImplementedError("NetworkX may divide by zero here")
    raw = G._core.percolation(values, p_sigma_x_t, weight)
    if n == 0:
        return {}
    scale = 1 / (n - 2)
    return {v: x * scale for v, x in zip(G._nodes, raw)}


def voterank(G, number_of_nodes=None):
    n = len(G)
    if n == 0:
        return []
    if number_of_nodes is not None and type(number_of_nodes) not in (int, bool):
        raise NotImplementedError("rustnx needs an integer number_of_nodes")
    if number_of_nodes is None or number_of_nodes > n:
        number_of_nodes = n
    m = G._core.number_of_edges()
    # Average out-degree (directed) or degree (self-loops count twice).
    avg_degree = (m if G.is_directed() else 2 * m) / n
    nodes = G._nodes
    return [nodes[i] for i in G._core.voterank(max(number_of_nodes, 0), avg_degree)]


def dispersion(G, u=None, v=None, normalized=True, alpha=1.0, b=0.0, c=0.0):
    if G.is_directed() or G._core.has_self_loops():
        # Then NetworkX's count depends on the iteration order of a set.
        raise NotImplementedError("rustnx supports undirected graphs without self-loops")

    def value(counts):
        total, embeddedness = counts
        dispersion_val = total
        if normalized:
            dispersion_val = (total + b) ** alpha
            if embeddedness + c != 0:
                dispersion_val /= embeddedness + c
        return dispersion_val

    nodes = G._nodes
    core = G._core
    if u is None and v is None:
        results = {x: {} for x in nodes}
        counts = core.dispersion()
        i = 0
        for x in range(len(nodes)):
            row = results[nodes[x]]
            for y in core.neighbors(x):
                row[nodes[y]] = value(counts[i])
                i += 1
        return results
    if u is not None and v is not None:
        a, z = _positions(G, [u, v])
        return value(core.dispersion(([a], [z]))[0])
    center = _positions(G, [u if v is None else v])[0]
    nbrs = core.neighbors(center)
    counts = core.dispersion(([center] * len(nbrs), nbrs))
    return {nodes[y]: value(k) for y, k in zip(nbrs, counts)}


def _group_degree(G, S, reverse):
    S = _reusable(S)
    count = G._core.group_degree(_positions(G, S), reverse)
    return count / (len(G) - len(S))


def group_degree_centrality(G, S):
    return _group_degree(G, S, False)


def group_in_degree_centrality(G, S):
    _directed_only(G)
    return _group_degree(G, S, True)


def group_out_degree_centrality(G, S):
    _directed_only(G)
    return _group_degree(G, S, False)


def group_closeness_centrality(G, S, weight=None):
    weight, _, _ = _check_weight(G, weight)
    if weight is not None and G._core.has_negative_weight(weight):
        raise NotImplementedError("with negative weights the visiting order matters")
    # Built exactly as NetworkX builds them, so they iterate in the same order.
    V = set(G._nodes)
    S = set(S)
    V_S = V - S
    if not S:
        raise ValueError("sources must not be empty")
    index = G._index
    for s in S:
        if s not in index:
            raise nx.NodeNotFound(f"Node {s} not found in graph")
    total = G._core.group_distance_sum([index[s] for s in S], [index[x] for x in V_S], weight)
    if total == 0:
        return 0  # NetworkX's ZeroDivisionError case
    return len(V_S) / total


@functools.cache
def _global_reaching_warnings():
    """Warnings NetworkX's ``global_reaching_centrality`` gives (3.4 warns
    that ``shortest_path`` will return an iterator)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        nx.global_reaching_centrality(nx.path_graph(3), backend="networkx")
    return [(w.category, str(w.message)) for w in caught]


def _reaching_weight(G, weight):
    """Checks for weighted reaching centrality; returns ``G.size(weight)``.
    Raises NetworkX's error for negative weights."""
    _, _, has_hidden = _check_weight(G, weight)
    if has_hidden:
        raise NotImplementedError("rustnx does not support None edge weights here")
    source = _networkx_graph(G)
    kinds = set()
    for _, _, data in source.edges(data=True):
        if weight not in data:
            raise NotImplementedError("an edge has no weight")  # KeyError in NetworkX
        x = data[weight]
        kinds.add(type(x))
        if x < 0:
            raise nx.NetworkXError("edge weights must be positive")
        if x == 0:
            raise NotImplementedError("NetworkX divides by a zero weight")
    if not (kinds <= {int} or kinds <= {float}):
        # Path weight sums then switch between int and float.
        raise NotImplementedError("rustnx needs all-int or all-float weights")
    return source.size(weight=weight)


_SINGLE_SELF_LOOP = "local_reaching_centrality of a single node with self-loop not well-defined"


def _reaching_values(G, sources, weight, normalized, total_weight):
    n = len(G)
    core = G._core
    if weight is None:
        parts = core.reaching_unweighted(sources, _COMPENSATED_SUM)
        if G.is_directed():
            return [(reached - 1) / (n - 1) for reached, _ in parts]
        return [s / 1 / (n - 1) for _, s in parts]
    sums = core.reaching_weighted(
        sources, weight, total_weight, _dijkstra_paths_in_pop_order(), _COMPENSATED_SUM
    )
    # NetworkX's `G.size(weight=weight) / G.size()`; the first is `total_weight`.
    norm = total_weight / core.number_of_edges() if normalized else 1
    return [s / norm / (n - 1) for s in sums]


def local_reaching_centrality(G, v, paths=None, weight=None, normalized=True):
    if paths is not None:
        raise NotImplementedError("rustnx computes the paths itself")
    s = _positions(G, [v])[0]
    if weight is None:
        total_weight = G._core.number_of_edges()
    else:
        total_weight = _reaching_weight(G, weight)
    if total_weight > 0 and len(G) == 1:
        raise nx.NetworkXError(_SINGLE_SELF_LOOP)
    if total_weight <= 0:
        raise nx.NetworkXError("Size of G must be positive")
    return _reaching_values(G, [s], weight, normalized, total_weight)[0]


def global_reaching_centrality(G, weight=None, normalized=True):
    if weight is None:
        total_weight = G._core.number_of_edges()
    else:
        total_weight = _reaching_weight(G, weight)
    if total_weight <= 0:
        raise nx.NetworkXError("Size of G must be positive")
    for category, message in _global_reaching_warnings():
        warnings.warn(message, category, stacklevel=2)
    if len(G) == 1:
        raise nx.NetworkXError(_SINGLE_SELF_LOOP)
    lrc = _reaching_values(G, list(range(len(G))), weight, normalized, total_weight)
    max_lrc = max(lrc)
    return sum(max_lrc - c for c in lrc) / (len(G) - 1)


@functools.cache
def _group_betweenness_version():
    """Which ``group_betweenness_centrality`` the installed NetworkX has:
    ``"inplace"`` (3.7+: updates the matrices in place and counts endpoints
    differently), ``"copy"`` (earlier) or ``None`` if unrecognized."""
    src = _source_text(nx.group_betweenness_centrality)
    if "for y in group & Dx.keys():" in src and "extra = N_in * (N - 1 + N_out)" in src:
        return "inplace"
    if "sigma_m_v = deepcopy(sigma_m)" in src and "scale = c * (2 * v - c - 1)" in src:
        return "copy"
    return None


def group_betweenness_centrality(G, C, normalized=True, weight=None, endpoints=False):
    version = _group_betweenness_version()
    if version is None:
        raise NotImplementedError("unrecognized NetworkX group betweenness")
    n = len(G)
    if n == 0:
        raise NotImplementedError("NetworkX's connectivity checks raise here")
    C = _reusable(C)
    list_of_groups = True
    if any(el in G for el in C):
        C = [C]
        list_of_groups = False
    for group in C:
        _reusable(group)  # iterated twice by NetworkX
    set_v = {node for group in C for node in group}
    index = G._index
    # As `set_v - G.nodes` builds it, so the message lists nodes in the same order.
    missing = set(x for x in set_v if x not in index)
    if missing:
        raise nx.NodeNotFound(f"The node(s) {missing} are in C but not in G.")
    weight = _unhidden_weight(G, weight)
    is_directed = G.is_directed()
    connected = _is_strongly_or_plainly_connected(G)
    sv = list(set_v)
    spos = {x: i for i, x in enumerate(sv)}
    k = len(sv)
    pre = G._core.group_preprocessing([index[x] for x in sv], weight)
    reached, pos, reach_len, rev_reach = pre.reach()

    def y_order(group, a):
        # NetworkX iterates `group & D[x].keys()`, a set whose order depends
        # on which side Python iterates (by length) and in what order. A dict
        # with the same length (when that matters) and the group members in
        # `D[x]`'s order gives the same set.
        members = [y for y in group if reached[a * k + spos[y]]]
        members.sort(key=lambda y: pos[a * k + spos[y]])
        size = min(reach_len[a], len(group) + 1)
        keys = members + [object() for _ in range(size - len(members))]
        return [spos[y] for y in group & dict.fromkeys(keys).keys()]

    def reach_count(group):
        # sum((1 if v in group else 2) for u in group for v in D[u] if v != u)
        total = 0
        for u in group:
            a = spos[u]
            inside = sum(1 for y in group if y != u and reached[a * k + spos[y]])
            total += inside + 2 * (reach_len[a] - 1 - inside)
        return total

    def pair_count(group):
        # Reachable pairs (u, v), u != v, with u or v in the group.
        total = 0
        for u in group:
            a = spos[u]
            total += reach_len[a] - 1 + rev_reach[a] - 1
            total -= sum(1 for y in group if y != u and reached[a * k + spos[y]])
        return total

    GBC = []
    for group in C:
        group = set(group)
        order = [spos[x] for x in group]
        y_orders = None
        if version == "inplace":
            y_orders = [y_order(group, a) for a in order]
        values = pre.main(order, y_orders)
        if values is None:
            raise NotImplementedError("NetworkX raises KeyError here")
        GBC_group = 0
        for value in values:
            GBC_group += value
        if version == "copy":
            v, c = n, len(group)
            if not endpoints:
                scale = 0
                if connected:
                    scale = c * (2 * v - c - 1)
                if scale == 0:
                    scale = reach_count(group)
                GBC_group -= scale
            if normalized:
                scale = 1 / ((v - c) * (v - c - 1))
                GBC_group *= scale
            elif not is_directed:
                GBC_group /= 2
        else:
            N = n
            if endpoints:
                Nscale = N
            else:
                N_in = len(group)
                N_out = Nscale = N - N_in
                if connected:
                    extra = N_in * (N - 1 + N_out)
                elif is_directed:
                    extra = pair_count(group)
                else:
                    extra = reach_count(group)
                GBC_group -= extra
            if normalized:
                GBC_group /= Nscale * (Nscale - 1)
            elif not is_directed:
                GBC_group /= 2
        GBC.append(GBC_group)
    if list_of_groups:
        return GBC
    return GBC[0]


def prominent_group(G, k, weight=None, C=None, endpoints=False, normalized=True, greedy=False):
    try:
        import pandas  # noqa: F401  (NetworkX imports it first)
    except ImportError:
        raise NotImplementedError("NetworkX needs pandas here") from None
    if C is not None:
        # NetworkX's DataFrame then pairs scores with nodes by position
        # across differently ordered rows and columns.
        raise NotImplementedError("rustnx does not support C")
    n = len(G)
    if type(k) is not int or not 0 <= k <= n or n == 0:
        raise NotImplementedError("NetworkX raises or loops here")
    rank = _node_rank(G)
    weight = _unhidden_weight(G, weight)
    pre = G._core.group_preprocessing(list(range(n)), weight)
    found = pre.prominent(k, bool(greedy), rank)
    if found is None:
        raise NotImplementedError("NetworkX raises here")
    max_GBC, group = found
    if group is None:
        max_GBC, group = 0, []
    reached, _, reach_len, _ = pre.reach()
    v = n
    if not endpoints:
        scale = 0
        if _is_strongly_or_plainly_connected(G):
            scale = k * (2 * v - k - 1)
        if scale == 0:
            members = set(group)
            for a in group:
                inside = sum(1 for b in members if b != a and reached[a * n + b])
                scale += inside + 2 * (reach_len[a] - 1 - inside)
        max_GBC -= scale
    if normalized:
        scale = 1 / ((v - k) * (v - k - 1))
        max_GBC *= scale
    elif not G.is_directed():
        max_GBC /= 2
    max_GBC = float(f"{max_GBC:.2f}")
    nodes = G._nodes
    return max_GBC, [nodes[i] for i in group]


# --- Batch 5: cores, clustering, distance and coloring ----------------------------


def _base_and_core(G, core_number):
    """The NetworkX graph to build a core subgraph from, and core numbers.
    The subgraph is built in NetworkX from the same node generator as
    NetworkX's, so its node set and order are the same."""
    base = _networkx_graph(G)
    core = globals()["core_number"](G) if core_number is None else core_number
    return base, core


def k_shell(G, k=None, core_number=None):
    base, core = _base_and_core(G, core_number)
    if k is None:
        k = max(core.values())
    return base.subgraph(v for v in core if core[v] == k).copy()


def k_crust(G, k=None, core_number=None):
    base, core = _base_and_core(G, core_number)
    if k is None:
        # One less than the other core subgraphs' default.
        k = max(core.values()) - 1
    return base.subgraph(v for v in core if core[v] <= k).copy()


def _integral(k):
    """``k`` as an int if it is an integer number, else ``None``."""
    if isinstance(k, float):
        return int(k) if k.is_integer() else None
    try:
        return operator.index(k)
    except TypeError:
        return None


def k_corona(G, k, core_number=None):
    base = _networkx_graph(G)
    if core_number is not None:
        # NetworkX's own filter, on the caller's core numbers.
        c = core_number
        if k is None:
            k = max(c.values())
        return base.subgraph(
            v for v in c if c[v] == k and k == sum(1 for w in base[v] if c[w] >= k)
        ).copy()
    core = G._core.core_number()
    if core is None:
        globals()["core_number"](G)  # raises NetworkX's self-loop error
    if k is None:
        k = max(dict(zip(G._nodes, core)).values())  # NetworkX's error if empty
    kk = _integral(k)
    if kk is None:
        raise NotImplementedError("rustnx needs an integer k")
    nodes = G._nodes
    positions = G._core.k_corona(core, kk) if 0 <= kk < 2**32 else []
    return base.subgraph(nodes[i] for i in positions).copy()


_SELF_LOOPS_MESSAGE = (
    "Input graph has self loops which is not permitted; "
    "Consider using G.remove_edges_from(nx.selfloop_edges(G))."
)


def _truss_support(k):
    """Smallest triangle count an edge needs to stay: NetworkX drops edges
    with ``len(common neighbors) < k - 2``."""
    if isinstance(k, float):
        if math.isnan(k):
            return 0  # every comparison with NaN is false: nothing is dropped
        t = k - 2
        if t == math.inf:
            return 2**63
        return 0 if t == -math.inf else max(0, math.ceil(t))
    try:
        k = operator.index(k)
    except TypeError:
        raise NotImplementedError("rustnx needs an int or float k") from None
    return min(max(0, k - 2), 2**63)


def k_truss(G, k):
    _undirected_only(G)
    need = _truss_support(k)
    result = G._core.k_truss(need)
    if result is None:
        raise nx.NetworkXNotImplemented(_SELF_LOOPS_MESSAGE)
    base = _networkx_graph(G)
    us, vs, keep, arcs = result
    nodes = G._nodes
    if type(base) is nx.Graph and type(base._adj) is dict:
        # NetworkX removes edges and isolated nodes from `G.copy()`. Taking
        # `copy`'s own steps for only what survives gives the same dicts in
        # the same order, without the copy and removals.
        H = nx.Graph()
        H.graph.update(base.graph)
        H.add_nodes_from((n, d.copy()) for (n, d), kept in zip(base._node.items(), keep) if kept)
        alive = iter(arcs)
        H.add_edges_from(
            (u, v, d.copy()) for u, nbrs in base._adj.items() for v, d in nbrs.items() if next(alive)
        )
        return H
    # Otherwise remove the dropped edges and nodes from a copy: removals
    # leave the copy's dicts in order.
    H = base.copy()
    H.remove_edges_from(zip([nodes[u] for u in us], [nodes[v] for v in vs]))
    H.remove_nodes_from([v for v, kept in zip(nodes, keep) if not kept])
    return H


def onion_layers(G):
    _undirected_only(G)
    result = G._core.onion_layers()
    if result is None:
        raise nx.NetworkXNotImplemented(
            "Input graph contains self loops which is not permitted; "
            "Consider using G.remove_edges_from(nx.selfloop_edges(G))."
        )
    order, layers = result
    nodes = G._nodes
    return dict(zip([nodes[i] for i in order], layers))


def _nbunch_positions(G, nodes):
    """``(single, positions)`` for a NetworkX ``nodes`` argument: whether it
    names one node, and the positions ``G.nbunch_iter(nodes)`` visits (each
    once, first occurrence first), or ``None`` for all nodes."""
    if nodes is None:
        return False, None
    if nodes in G:
        return True, [G._index[nodes]]
    try:
        iter(nodes)
    except TypeError:
        raise NotImplementedError("NetworkX raises its own error here") from None
    return False, _node_subset(G, nodes)


@functools.cache
def _square_clustering_by_pairs():
    """Whether the installed NetworkX (3.4) counts squares over pairs of
    neighbors including self-loops; 3.5+ count them through two-hop
    neighbors, ignoring self-loops, and give different values."""
    H = nx.Graph([(0, 1), (1, 2), (2, 0), (0, 0)])
    return nx.square_clustering(H, backend="networkx")[0] == 2


def square_clustering(G, nodes=None):
    single, positions = _nbunch_positions(G, nodes)
    old = _square_clustering_by_pairs()
    counts = G._core.square_clustering(positions, old)
    # 3.4 keeps the (int) square count where the potential isn't positive.
    values = [
        squares / potential if potential > 0 else (squares if old else 0)
        for squares, potential in counts
    ]
    return _per_node(G, positions, values, single)


def generalized_degree(G, nodes=None):
    _undirected_only(G)
    single, positions = _nbunch_positions(G, nodes)
    base = _networkx_graph(G)
    adj = base.adj
    index = G._index
    vlist = G._nodes if positions is None else [G._nodes[i] for i in positions]
    # Counter keys come in the order NetworkX meets them, iterating the set
    # `set(G[v]) - {v}`; building the same set gives the same order.
    ws = []
    ends = []
    for v in vlist:
        ws.extend([index[w] for w in set(adj[v]) - {v}])
        ends.append(len(ws))
    counts = G._core.generalized_degree(ws, ends)
    degrees = []
    begin = 0
    for end in ends:
        degrees.append(Counter(counts[begin:end]))
        begin = end
    return _per_node(G, positions, degrees, single)


def all_triangles(G, nbunch=None):
    _undirected_only(G)
    base = _networkx_graph(G)
    if not isinstance(base._adj, dict):
        # Views compute `v_nbrs & u_nbrs` with Python's Set mixin, which
        # builds the set in a different order.
        raise NotImplementedError("rustnx needs a graph, not a view")
    if nbunch is None:
        positions = None
    else:
        _, positions = _nbunch_positions(G, nbunch)

    def produce():
        groups, ws, qualify = G._core.all_triangles(positions)
        nodes = G._nodes
        for u, v, start, end in groups:
            nu, nv = nodes[u], nodes[v]
            picked = [i for i in range(start, end) if qualify[i]]
            if len(picked) == 1:
                yield nu, nv, nodes[ws[picked[0]]]
                continue
            # NetworkX yields from the set `v_nbrs & u_nbrs`: rebuild it, in
            # the order CPython inserts into it, to iterate it the same way.
            members = [nodes[i] for i in ws[start:end]]
            keep = dict(zip(members, qualify[start:end]))
            for w in set(members):
                if keep[w]:
                    yield nu, nv, w

    return _traversal(G, produce)


@functools.cache
def _centroid_tree_shortcut():
    """Whether the installed NetworkX's ``centroid`` (3.7+; ``barycenter``
    before) hands unweighted undirected trees to ``nx.tree.centroid``, which
    orders its result differently (and rejects the null graph)."""
    try:
        nx.barycenter(nx.Graph(), backend="networkx")
    except nx.NetworkXPointlessConcept:
        return True
    return False


@functools.cache
def _centroid_is_tree_centroid():
    """Whether the installed NetworkX (3.6) registers its *tree* centroid
    under the name ``centroid`` (3.7 renames it ``tree_centroid`` and gives
    ``centroid`` to the distance measure that was ``barycenter``)."""
    registry = getattr(nx.utils.backends, "_registered_algorithms", {})
    func = registry.get("centroid")
    module = getattr(getattr(func, "orig_func", func), "__module__", "")
    return module.startswith("networkx.algorithms.tree")


def centroid(G, weight=None):
    if _centroid_is_tree_centroid():
        return tree_centroid(G)
    return _distance_centroid(G, weight)


def _distance_centroid(G, weight):
    nodes = G._nodes
    if weight is None and not G.is_directed() and _centroid_tree_shortcut():
        if is_tree(G):  # raises NetworkX's error for the null graph
            return [nodes[i] for i in G._core.tree_centroid()]
    n = len(G)
    if weight is None:
        sums = [(reached, total) for reached, total, _ in G._core.bfs_stats(None)]
        failed, all_int = False, True
    else:
        weight, all_int, _ = _check_weight(G, weight, distances=True)
        sums, failed = G._core.distance_sums(weight, _COMPENSATED_SUM)
    smallest, centroid_vertices = float("inf"), []
    for v, (reached, total) in zip(nodes, sums):
        if reached < n:
            raise nx.NetworkXNoPath(
                f"Input graph {_networkx_graph(G)} is disconnected, so every induced "
                "subgraph has infinite barycentricity."
            )
        barycentricity = int(total) if all_int else total
        if barycentricity < smallest:
            smallest = barycentricity
            centroid_vertices = [v]
        elif barycentricity == smallest:
            centroid_vertices.append(v)
    if failed:
        raise ValueError(*_NEGATIVE_CYCLE)
    return centroid_vertices


def barycenter(G, weight=None):
    # NetworkX 3.4 to 3.6 name `centroid` this way.
    return _distance_centroid(G, weight)


def harmonic_diameter(G, weight=None):
    if weight is not None:
        # Mixed int and float distances give the same 1/d either way.
        weight, _, _ = _check_weight(G, weight)
    total, added = G._core.harmonic_sum(weight)
    order = len(G)
    if added and total != 0:
        return order * (order - 1) / total
    if order > 1:
        return math.inf
    return math.nan


@functools.cache
def _intersection_pairs_once():
    """Whether the installed NetworkX (3.7+) checks connectivity first,
    visits each unordered pair once and gives up when the diameter passes
    ``8 log2(n) / 3``. That rejects long cycles, which 3.4 and 3.5 accept."""
    # NetworkX's own code, not a dispatched call: `is_distance_regular`
    # would dispatch `intersection_array` back here while probing.
    try:
        nx.intersection_array.orig_func(nx.cycle_graph(40))
    except nx.NetworkXError:
        return True
    return False


def intersection_array(G):
    _undirected_only(G)
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("Graph has no nodes.")
    bound = (8 * math.log(len(G), 2)) / 3
    code, b, c = G._core.intersection_array(_intersection_pairs_once(), bound)
    if code == 1:
        raise nx.NetworkXError("Graph is not distance regular.")
    if code == 2:
        raise nx.NetworkXError("Graph is not distance regular")
    return b, c


def is_distance_regular(G):
    try:
        intersection_array(G)
        return True
    except nx.NetworkXError:
        return False


def is_strongly_regular(G):
    _undirected_only(G)
    try:
        b, _ = intersection_array(G)
    except nx.NetworkXError:
        return False
    # A distance-regular graph's diameter is the length of its arrays.
    return len(b) == 2


_MISSING = object()


def _color_ids(G, coloring):
    """Per node, an id that is equal exactly when the colors compare
    equal (-1 for nodes without a color)."""
    if type(coloring) is not dict:
        raise NotImplementedError("rustnx needs the coloring as a dict")
    ids = {}
    out = []
    get = coloring.get
    for v in G._nodes:
        c = get(v, _MISSING)
        if c is _MISSING:
            out.append(-1)
            continue
        t = type(c)
        # Equal dict keys must mean `==`, which NaN and custom types break.
        if not (t is int or t is str or t is bool or (t is float and c == c)):
            raise NotImplementedError("rustnx supports int, str and float colors")
        out.append(ids.setdefault(c, len(ids)))
    return out


def is_coloring(G, coloring):
    ok, missing = G._core.is_coloring(_color_ids(G, coloring))
    if missing is not None:
        raise KeyError(G._nodes[missing])
    return ok


def is_equitable(G, coloring, num_colors=None):
    if not is_coloring(G, coloring):
        return False
    # The rest is NetworkX's own code: it only reads the coloring.
    color_set_size = defaultdict(int)
    for color in coloring.values():
        color_set_size[color] += 1
    if num_colors is not None:
        for color in range(num_colors):
            if color not in color_set_size:
                color_set_size[color] = 0
    all_set_sizes = set(color_set_size.values())
    if len(all_set_sizes) == 0 and num_colors is None:
        return True
    elif len(all_set_sizes) == 1:
        return True
    elif len(all_set_sizes) == 2:
        a, b = list(all_set_sizes)
        return abs(a - b) <= 1
    else:
        return False


def _greedy_order(G, strategy):
    """``(order, colors)`` for a ``greedy_color`` strategy rustnx matches."""
    core = G._core
    if strategy == "largest_first":
        return core.greedy_color()
    if strategy in ("saturation_largest_first", "DSATUR"):
        return core.dsatur()
    if strategy == "random_sequential":
        # NetworkX shuffles `list(G)` with the global `random` instance (no
        # seed reaches the strategy); shuffling positions draws the same
        # numbers and gives the same permutation.
        order = list(range(len(G)))
        random._inst.shuffle(order)
        return order, core.greedy_with_order(order)
    if strategy in ("connected_sequential", "connected_sequential_bfs", "connected_sequential_dfs"):
        _undirected_only(G)
        nodes = G._nodes
        index = G._index
        # Each component's source is `arbitrary_element` of NetworkX's
        # component set, built here in the same order.
        sources = [
            index[next(iter({nodes[i] for i in comp}))] for comp in core.connected_components()
        ]
        order = core.connected_sequential(sources, strategy.endswith("_dfs"))
        return order, core.greedy_with_order(order)
    # smallest_last and independent_set pick nodes from sets (hash order).
    raise NotImplementedError(f"rustnx does not implement the {strategy} strategy")


# --- Batch 6: trees and structural tests -------------------------------------------


def _node_arg(G, node):
    """Position of a node argument that NetworkX puts in its output as
    given (a root or a start). Declines a missing node (NetworkX's errors
    for those differ by function and version) and an equal object of a
    different type, which NetworkX would return instead of G's node."""
    try:
        i = G._index.get(node)
    except TypeError:
        i = None
    if i is None or type(G._nodes[i]) is not type(node):
        raise NotImplementedError("rustnx needs a node of the graph here")
    return i


def _computed_on_first_next(G, compute, fallback):
    """A generator that runs ``compute()`` when iteration starts, as NetworkX
    does (it copies or decomposes the graph then, so later changes don't
    matter). If G changed before that, NetworkX's ``fallback()`` runs."""
    guard = _MutationGuard(G)

    def generate():
        try:
            changed = guard.changed()
        finally:
            guard.release()
        if changed:
            yield from fallback(guard.graph)
            return
        yield from compute()

    return generate()


def isolates(G):
    def produce():
        nodes = G._nodes
        for v in G._core.isolates():
            yield nodes[v]

    return _traversal(G, produce)


def number_of_isolates(G):
    return len(G._core.isolates())


def is_regular(G):
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("Graph has no nodes.")
    return G._core.is_regular()


def is_k_regular(G, k):
    _undirected_only(G)
    if type(k) not in (int, bool):
        raise NotImplementedError("rustnx needs an integer k")
    return G._core.all_degrees_equal(int(k))


def is_tournament(G):
    _directed_only(G)
    return G._core.is_tournament()


def bridges(G, root=None):
    _undirected_only(G)
    if root is not None:
        # NetworkX then lists the edges of a subgraph copy, whose order
        # follows the component set's hash order.
        raise NotImplementedError("rustnx does not support root here")

    def produce():
        us, vs = G._core.bridges()
        nodes = G._nodes
        for u, v in zip(us, vs):
            yield nodes[u], nodes[v]

    return _traversal(G, produce)


def has_bridges(G, root=None):
    _undirected_only(G)
    r = None
    if root is not None:
        if root not in G:
            raise nx.NodeNotFound(f"Root node {root} is not in graph")
        r = G._index[root]
    us, _ = G._core.bridges(r, True)
    return bool(us)


def _span_weight(G, weight):
    """The edge attribute for local bridge spans: rustnx's searches give
    NetworkX's exact (integer) distances only for non-negative ints."""
    if weight is None:
        return None
    weight, all_int, _ = _check_weight(G, weight)
    if not all_int or G._core.weight_mixed(weight):
        raise NotImplementedError("rustnx needs integer weights for spans")
    if G._core.has_negative_weight(weight):
        raise NotImplementedError("rustnx does not support negative weights here")
    return weight


def local_bridges(G, with_span=True, weight=None):
    _undirected_only(G)
    span = with_span is True
    attr = _span_weight(G, weight) if span else None

    def produce():
        us, vs = G._core.local_bridges()
        nodes = G._nodes
        core = G._core
        for u, v in zip(us, vs):
            if not span:
                yield nodes[u], nodes[v]
                continue
            d = core.local_bridge_span(u, v, attr)
            yield nodes[u], nodes[v], float("inf") if d is None else int(d)

    return _traversal(G, produce)


def _chains(G, r):
    us, vs, ends = G._core.chain_decomposition(r)
    nodes = G._nodes
    begin = 0
    for end in ends:
        yield [(nodes[us[i]], nodes[vs[i]]) for i in range(begin, end)]
        begin = end


def chain_decomposition(G, root=None):
    _undirected_only(G)
    r = None if root is None or root not in G else _node_arg(G, root)

    def compute():
        if root is not None and r is None:
            raise nx.NodeNotFound(f"Root node {root} is not in graph")
        yield from _chains(G, r)

    return _computed_on_first_next(
        G, compute, lambda H: nx.chain_decomposition(H, root=root, backend="networkx")
    )


def is_eulerian(G):
    plus_in, plus_out, bad = G._core.euler_balance()
    if G.is_directed():
        return not (plus_in or plus_out or bad) and is_strongly_connected(G)
    return not plus_in and is_connected(G)


def has_eulerian_path(G, source=None):
    if is_eulerian(G):
        return True
    s = None if source is None else _node_arg(G, source)
    plus_in, plus_out, bad = G._core.euler_balance()
    if G.is_directed():
        if s is not None:
            ins, outs = G._core.in_out_degrees()
            if outs[s] - ins[s] != 1:
                return False
        if bad:
            return False
        return plus_in <= 1 and plus_out <= 1 and is_weakly_connected(G)
    if s is not None and G._core.degree_of(s) % 2 != 1:
        return False
    return plus_in == 2 and is_connected(G)


def is_semieulerian(G):
    return has_eulerian_path(G) and not is_eulerian(G)


def _euler_pairs(G, start):
    us, vs = G._core.euler_walk(start)
    nodes = G._nodes
    return [(nodes[u], nodes[v]) for u, v in zip(us, vs)]


def eulerian_circuit(G, source=None, keys=False):
    s = None if source is None else _node_arg(G, source)

    def compute():
        if not is_eulerian(G):
            raise nx.NetworkXError("G is not Eulerian.")
        # The walk runs on G.copy() (or G.reverse()), as in NetworkX.
        yield from _euler_pairs(G, 0 if s is None else s)

    return _computed_on_first_next(
        G, compute, lambda H: nx.eulerian_circuit(H, source=source, keys=keys, backend="networkx")
    )


def _path_start(G):
    """``_find_path_start`` on the walked graph (G's reverse if directed),
    for a graph that has an Eulerian path but no circuit."""
    if G.is_directed():
        ins, outs = G._core.in_out_degrees()
        v1, v2 = [v for v, (i, o) in enumerate(zip(ins, outs)) if i != o]
        # In the reverse, out-degree is G's in-degree.
        return v1 if ins[v1] > outs[v1] else v2
    return next(v for v, d in enumerate(G._core.degrees()) if d % 2)


def eulerian_path(G, source=None, keys=False):
    s = None if source is None else _node_arg(G, source)

    def compute():
        if not has_eulerian_path(G, source):
            raise nx.NetworkXError("Graph has no Eulerian paths.")
        eulerian = is_eulerian(G)
        if G.is_directed():
            if s is None or not eulerian:
                start = 0 if eulerian else _path_start(G)
            else:
                start = s
            yield from _euler_pairs(G, start)
        else:
            if s is None:
                start = 0 if eulerian else _path_start(G)
            else:
                start = s
            yield from reversed([(v, u) for u, v in _euler_pairs(G, start)])

    return _computed_on_first_next(
        G, compute, lambda H: nx.eulerian_path(H, source=source, keys=keys, backend="networkx")
    )


def cycle_basis(G, root=None):
    _undirected_only(G)
    if len(G) == 0:
        return []
    r = None if root is None else _node_arg(G, root)
    nodes = G._nodes
    return G._core.cycle_basis(nodes if type(nodes) is list else list(nodes), r)


def girth(G):
    _undirected_only(G)
    g = G._core.girth()
    return math.inf if g is None else g


_ORIENTATIONS = {None: 0, "original": 1, "reverse": 2, "ignore": 3}


def find_cycle(G, source=None, orientation=None):
    directed = G.is_directed()
    if not directed:
        code = 0 if orientation is None else 1  # edge_dfs labels them "forward"
    elif orientation is None or (type(orientation) is str and orientation in _ORIENTATIONS):
        code = _ORIENTATIONS[orientation]
    else:
        raise NotImplementedError("invalid orientation")
    starts = None if source is None else [_node_arg(G, source)]
    if code >= 2:
        G._ensure_exact_pred()
    cycle = G._core.find_cycle(code, starts)
    if cycle is None:
        raise nx.NetworkXNoCycle("No cycle found.")
    nodes = G._nodes
    if code == 0:
        return [(nodes[u], nodes[v]) for u, v, _ in cycle]
    return [(nodes[u], nodes[v], "reverse" if rev else "forward") for u, v, rev in cycle]


@functools.cache
def _idom_includes_start():
    """Whether ``immediate_dominators`` maps start to itself (before 3.7;
    later releases leave start out)."""
    return 0 in nx.immediate_dominators(nx.DiGraph([(0, 1)]), 0, backend="networkx")


@functools.cache
def _new_dominance_frontiers():
    """Whether ``dominance_frontiers`` is NetworkX 3.7's version: start
    goes last, with dominator ``None``, and its in-edges always count."""
    D = nx.DiGraph([(0, 1), (1, 0)])
    return list(nx.dominance_frontiers(D, 0, backend="networkx")) == [1, 0]


def _start_position(G, start):
    _directed_only(G)
    if start not in G:
        raise nx.NetworkXError("start is not in G")
    return _node_arg(G, start)


def immediate_dominators(G, start):
    s = _start_position(G, start)
    order, idom = G._core.immediate_dominators(s)
    nodes = G._nodes
    result = {nodes[u]: nodes[d] for u, d in zip(order, idom)}
    if not _idom_includes_start():
        del result[start]
    return result


def dominance_frontiers(G, start):
    s = _start_position(G, start)
    G._ensure_exact_pred()
    order, vs, us = G._core.dominance_frontiers(s, _new_dominance_frontiers())
    nodes = G._nodes
    df = {nodes[u]: set() for u in order}
    # The same additions in the same order give the same set order.
    for v, u in zip(vs, us):
        df[nodes[v]].add(nodes[u])
    return df


def is_arborescence(G):
    _directed_only(G)
    return is_tree(G) and max(G._core.in_out_degrees()[0]) <= 1


def is_branching(G):
    _directed_only(G)
    return is_forest(G) and max(G._core.in_out_degrees()[0]) <= 1


def to_prufer_sequence(T):
    _undirected_only(T)
    n = len(T)
    if n < 2:
        msg = "Prüfer sequence undefined for trees with fewer than two nodes"
        raise nx.NetworkXPointlessConcept(msg)
    if not is_tree(T):
        raise nx.NotATree("provided graph is not a tree")
    index = T._index
    pos = [index.get(k) for k in range(n)]
    if None in pos:
        raise KeyError("tree must have node labels {0, ..., n - 1}")
    nodes = T._nodes
    return [nodes[v] for v in T._core.prufer_sequence(pos)]


def kruskal_mst_edges(G, minimum, weight="weight", keys=True, data=True, ignore_nan=False, partition=None):
    if partition is not None:
        raise NotImplementedError("rustnx does not support partition")
    base = _networkx_graph(G)
    weight, _, has_hidden = _check_weight(G, weight)
    core = G._core  # after `_check_weight`, which may convert `weight`
    if has_hidden and not core.is_native():
        # NetworkX converts `weight` with default None for this function,
        # so edges lacking it look hidden; NetworkX itself defaults to 1.
        # Kept on G, which NetworkX caches while the graph is unchanged.
        cores = G.__dict__.setdefault("_kruskal_cores", {})
        if weight not in cores:
            from .graph import from_networkx

            cores[weight] = from_networkx(base, {weight: 1})._core
        core = cores[weight]
        has_hidden = core.weight_info(weight)[1]
    if has_hidden:
        raise NotImplementedError("rustnx does not support None edge weights here")
    us, vs = core.kruskal(weight, not minimum)
    guard = _MutationGuard(G)
    nodes = G._nodes
    adj = base._adj

    def generate():
        try:
            for u, v in zip(us, vs):
                if guard.changed():
                    raise RuntimeError("Graph changed during iteration")
                a, b = nodes[u], nodes[v]
                yield (a, b, adj[a][b]) if data else (a, b)
        finally:
            guard.release()

    return generate()


# --- Batch 7: shortest paths, DAG and cycle leftovers ------------------------------


def _weight_flags(G, weight, all_int):
    """Decline weights rustnx's f64 arithmetic can't reproduce: infinite
    ones, and ints whose path sums could leave f64's exact range."""
    finite, exact, _ = G._core.weight_flags(weight)
    if not finite:
        raise NotImplementedError("rustnx does not support infinite weights here")
    if not exact and (all_int or G._core.weight_mixed(weight)):
        raise NotImplementedError("integer weights too large for exact sums")


# Floyd-Warshall


@functools.cache
def _floyd_warshall_inline():
    """Whether the installed NetworkX's Floyd-Warshall is 3.4/3.5's version
    (``min(e_weight, ...)`` over ``G.edges(data=True)`` with missing weights
    1.0, no negative cycle check) rather than 3.6+'s ``_init_pred_dist``
    (rows of ``G._adj``, missing weights 1, hidden edges skipped)."""
    return "_init_pred_dist" not in _source_text(nx.floyd_warshall_predecessor_and_distance)


def _inf():
    return float("inf")


# NetworkX's `dist` rows: `defaultdict(lambda: float("inf"))`.
_DIST_ROW = functools.partial(defaultdict, _inf)


def _core_with_default(G, weight, default):
    """A core converted with ``default`` for edges lacking ``weight``,
    kept on G (which NetworkX caches while the graph is unchanged)."""
    cores = G.__dict__.setdefault("_rustnx_default_cores", {})
    key = (weight, type(default), default)
    if key not in cores:
        from .graph import from_networkx

        cores[key] = from_networkx(_networkx_graph(G), {weight: default})._core
    return cores[key]


def _floyd_warshall(G, weight, tree=False, want_pred=True):
    old = not tree and _floyd_warshall_inline()
    if callable(weight):
        raise NotImplementedError("rustnx does not support callable weights")
    core = G._core
    if weight is None:
        # `d.get(None, 1.0)` before 3.6, `_weight_function`'s 1 after.
        int_weights = not old
    else:
        if old:
            if G._core.is_native():
                raise NotImplementedError("rustnx can't tell missing weights apart here")
            if not isinstance(weight, str):
                raise NotImplementedError("rustnx only supports string edge attribute names")
            core = _core_with_default(G, weight, 1.0)
        else:
            _check_weight(G, weight)
            core = G._core
        int_weights, has_hidden = core.weight_info(weight)
        if core.weight_mixed(weight):
            raise NotImplementedError("edge weights mix ints and floats")
        if has_hidden and old:
            # NetworkX 3.4/3.5 compare None with a number and raise TypeError.
            raise NotImplementedError("rustnx does not support None edge weights here")
    nodes = G._nodes
    found = core.floyd_warshall(
        nodes if type(nodes) is list else list(nodes),
        _DIST_ROW,
        weight,
        int_weights,
        old,
        tree,
        want_pred,
    )
    if found is None:
        raise nx.NetworkXUnbounded("Negative cycle detected.")
    return found


def floyd_warshall_predecessor_and_distance(G, weight="weight"):
    return _floyd_warshall(G, weight)


def floyd_warshall(G, weight="weight"):
    return _floyd_warshall(G, weight, want_pred=False)[1]


def floyd_warshall_tree(G, weight="weight"):
    return _floyd_warshall(G, weight, tree=True)


@functools.cache
def _floyd_warshall_numpy_checks():
    """Whether ``floyd_warshall_numpy`` checks the diagonal for negative
    cycles (NetworkX 3.6+)."""
    return "np.diag(A) < 0" in _source_text(nx.floyd_warshall_numpy)


def floyd_warshall_numpy(G, nodelist=None, weight="weight"):
    import numpy as np

    n = len(G)
    if nodelist is not None:
        if not len(nodelist) == len(G) == len(set(nodelist)):
            raise nx.NetworkXError(
                "nodelist must contain every node in G with no repeats."
                "If you wanted a subgraph of G use G.subgraph(nodelist)"
            )
        order = [0] * n
        for k, v in enumerate(nodelist):
            i = G._index.get(v)
            if i is None:
                # NetworkX's error message lists a set of nodes.
                raise NotImplementedError("nodelist holds nodes not in G")
            order[i] = k
    else:
        order = list(range(n))
    weight, _, has_hidden = _check_weight(G, weight)
    if has_hidden:
        raise NotImplementedError("rustnx does not support None edge weights here")
    if G._core.weight_flags(weight)[2]:
        raise NotImplementedError("NumPy may order -0.0 and 0.0 either way")
    if n == 0:
        return np.full((0, 0), np.inf)
    found = G._core.floyd_warshall_dense(order, weight, _floyd_warshall_numpy_checks())
    if found is None:
        raise nx.NetworkXUnbounded("Negative cycle detected.")
    return np.frombuffer(found, dtype=np.float64).reshape(n, n)


# Johnson and Goldberg-Radzik


def johnson(G, weight="weight"):
    weight, all_int, has_hidden = _check_weight(G, weight)
    if has_hidden:
        # NetworkX's reweighting adds the None and raises TypeError.
        raise NotImplementedError("rustnx does not support None edge weights here")
    _weight_flags(G, weight, all_int)
    h = G._core.johnson_potentials(weight)
    if h is None:
        raise nx.NetworkXUnbounded(_BELLMAN_FORD_UNBOUNDED)
    pop_order = _dijkstra_paths_in_pop_order()
    nodes = G._nodes
    result = {}
    for start in range(0, len(nodes), _ALL_PAIRS_BATCH):
        batch = list(range(start, min(start + _ALL_PAIRS_BATCH, len(nodes))))
        for s, tree in zip(batch, G._core.johnson_trees(batch, h, weight)):
            if tree is None:
                raise ValueError(*_NEGATIVE_CYCLE)
            order, parents, seen = tree
            result[nodes[s]] = _tree_paths(G, order, parents, order if pop_order else seen)
    return result


@functools.cache
def _goldberg_radzik_skips_counted():
    """Whether ``goldberg_radzik``'s ``topo_sort`` skips relabeled nodes
    already counted (3.4/3.5) rather than iterating a copy of the set made
    before any were (3.6+)."""
    return "relabeled - neg_count.keys()" not in _source_text(nx.goldberg_radzik)


def goldberg_radzik(G, source, weight="weight"):
    if source not in G:
        raise nx.NodeNotFound(f"Node {source} is not found in the graph")
    s = G._index[source]
    _same_node_type(G, source, s)
    weight, all_int, has_hidden = _check_weight(G, weight, distances=True)
    if has_hidden:
        raise NotImplementedError("rustnx does not support None edge weights here")
    if G._core.negative_selfloop(weight):
        raise nx.NetworkXUnbounded(_BELLMAN_FORD_UNBOUNDED)
    if len(G) == 1:
        return {source: None}, {source: 0}
    _weight_flags(G, weight, all_int)
    state = G._core.goldberg_radzik(s, all_int, weight)
    skip_counted = _goldberg_radzik_skips_counted()
    nodes = G._nodes
    index = G._index
    no_counts = {}.keys()
    # `relabeled` is a set: each round visits it in Python's set order, so
    # it is built here exactly as NetworkX builds it.
    relabeled = {source}
    while relabeled:
        visit = relabeled if skip_counted else relabeled - no_counts
        if not state.topo_sort([index[u] for u in visit], skip_counted):
            raise nx.NetworkXUnbounded(_BELLMAN_FORD_UNBOUNDED)
        relabeled = set(map(nodes.__getitem__, state.relax()))
    keys, preds, dists, ints = state.result()
    pred = {nodes[v]: None if p == _NO_PARENT else nodes[p] for v, p in zip(keys, preds)}
    d = {nodes[v]: int(x) if i else x for v, x, i in zip(keys, dists, ints)}
    return pred, d


# DAGs


def _flat_batches(G, it, limit=1024):
    """Lists of nodes from a Rust iterator's ``(flat, ends)`` batches."""
    nodes = G._nodes
    while True:
        flat, ends = it.next_batch(limit)
        if not ends:
            return
        begin = 0
        for end in ends:
            yield [nodes[i] for i in flat[begin:end]]
            begin = end


def antichains(G, topo_order=None):
    _directed_only(G)
    given = None
    if topo_order is not None:
        if iter(topo_order) is topo_order:
            raise NotImplementedError("rustnx needs a reusable container of nodes")
        # NetworkX trusts the order it is given; rustnx takes only true
        # topological orders, for which the closure is plain reachability.
        given = G._core.antichains([_node_arg(G, v) for v in topo_order])
        if given is None:
            raise NotImplementedError("topo_order is not a topological order of G")

    def compute():
        # NetworkX sorts and copies the graph when iteration starts.
        it = given
        if it is None:
            it = G._core.antichains(_topological_order_or_raise(G))
        yield from _flat_batches(G, it)

    return _computed_on_first_next(
        G, compute, lambda H: nx.antichains(H, topo_order=topo_order, backend="networkx")
    )


def antichain_width(G):
    _directed_only(G)
    width = G._core.antichain_width()
    if width is None:
        raise nx.NetworkXUnfeasible(
            "Graph contains a cycle or graph changed during iteration"
        )
    return width


# Simple paths


def _live_paths(G, start, fallback):
    """A generator over ``start()`` that, like NetworkX's, reads the graph
    as it goes: a change before iteration runs NetworkX on the changed
    graph, a change during it stops with a RuntimeError."""
    guard = _MutationGuard(G)

    def generate():
        try:
            if guard.changed():
                guard.release()
                yield from fallback(guard.graph)
                return
            cache, key = guard._cache, guard._key
            for item in start():
                if key is not None and key not in cache:
                    raise RuntimeError("Graph changed during iteration")
                yield item
        finally:
            guard.release()

    return generate()


def _path_limit(cutoff, n):
    """``len(current_path) - 1 < cutoff`` as ``nodes on path < limit``,
    or ``None`` when NetworkX yields nothing (``cutoff >= 0`` fails)."""
    if type(cutoff) not in (int, float, bool):
        raise NotImplementedError("rustnx needs an int or float cutoff")
    if not cutoff >= 0:
        return None
    if cutoff >= n + 1:
        return n + 1
    return math.ceil(cutoff)


def _simple_paths(G, source, target, cutoff, edges):
    try:
        s = G._index.get(source)
    except TypeError:
        s = None
    if s is not None:
        _same_node_type(G, source, s)
    error = None
    targets, extra = [], False
    if target in G:
        targets = [G._index[target]]
    else:
        try:
            target_set = set(target)
        except TypeError as err:
            error = nx.NodeNotFound(f"target node {target} not in graph")
            error.__cause__ = err
        else:
            for x in target_set:
                i = G._index.get(x)
                if i is not None:
                    targets.append(i)
                elif x is not None:  # None is always a key of current_path
                    extra = True
    limit = _path_limit(len(G) - 1 if cutoff is None else cutoff, len(G))
    nodes = G._nodes

    def start():
        if s is None:
            raise nx.NodeNotFound(f"source node {source} not in graph")
        if error is not None:
            raise error
        if limit is None or not (targets or extra):
            return
        it = G._core.simple_paths(s, targets, extra, limit)
        for path in _flat_batches(G, it):
            if edges:
                yield list(zip(path, path[1:]))
            else:
                yield path

    func = nx.all_simple_edge_paths if edges else nx.all_simple_paths
    return _live_paths(
        G, start, lambda H: func(H, source, target, cutoff=cutoff, backend="networkx")
    )


def all_simple_paths(G, source, target, cutoff=None):
    return _simple_paths(G, source, target, cutoff, edges=False)


def all_simple_edge_paths(G, source, target, cutoff=None):
    return _simple_paths(G, source, target, cutoff, edges=True)


def shortest_simple_paths(G, source, target, weight=None):
    s = _position(G, source)
    t = _position(G, target)
    if weight is not None:
        weight, all_int, _ = _check_weight(G, weight, distances=True)
        _weight_flags(G, weight, all_int)
    if G.is_directed():
        G._ensure_exact_pred()  # the reverse searches follow G.pred

    def start():
        if s is None:
            raise nx.NodeNotFound(f"source node {source} not in graph")
        if t is None:
            raise nx.NodeNotFound(f"target node {target} not in graph")
        it = G._core.shortest_simple_paths(s, t, _COMPENSATED_SUM, weight)
        nodes = G._nodes
        while True:
            code, path = it.next_path()
            if code == 1:
                raise nx.NetworkXNoPath(f"No path between {source} and {target}.")
            if code == 2:
                raise ValueError("Contradictory paths found: negative weights?")
            if path is None:
                return
            yield [nodes[i] for i in path]

    return _live_paths(
        G, start, lambda H: nx.shortest_simple_paths(H, source, target, weight=weight, backend="networkx")
    )


def is_simple_path(G, nodes):
    if len(nodes) == 0:
        return False
    if len(nodes) == 1:
        return nodes[0] in G
    index = G._index
    try:
        path = [index[v] for v in nodes]
    except (KeyError, TypeError):
        return False
    if len(set(path)) != len(path):
        return False
    return G._core.is_path(path)


# Minimum cycle basis


def _spanning_forest_edges(index, edges):
    """``minimum_spanning_edges(G, weight=None)``: with equal weights,
    Kruskal keeps each edge (in ``G.edges`` order) joining two trees."""
    parent = {}

    def find(x):
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != root:
            parent[x], x = root, parent[x]
        return root

    tree = []
    for u, v in edges:
        a, b = find(index[u]), find(index[v])
        if a != b:
            parent[a] = b
            tree.append((u, v))
    return tree


def minimum_cycle_basis(G, weight=None):
    _undirected_only(G)
    if weight is not None:
        weight, all_int, has_hidden = _check_weight(G, weight)
        if has_hidden:
            raise NotImplementedError("rustnx does not support None edge weights here")
        _weight_flags(G, weight, all_int)
    nodes = G._nodes
    if any(isinstance(v, tuple) for v in nodes):
        # NetworkX's lifted graph names copies `(v, 1)`, which a tuple node
        # could equal.
        raise NotImplementedError("rustnx does not support tuple nodes here")
    base = _networkx_graph(G)
    index = G._index
    cb = []
    for c in connected_components(G):
        # The subgraph view (and the chord set) iterate in an order that can
        # depend on Python's set layout, so they are built as NetworkX does.
        sub = base.subgraph(c)
        edges = list(sub.edges)
        tree_edges = _spanning_forest_edges(index, edges)
        chords = sub.edges - tree_edges - {(v, u) for u, v in tree_edges}
        if not chords:
            continue
        code, found = G._core.min_cycle_basis(
            [index[v] for v in sub],
            [(index[u], index[v]) for u, v in edges],
            [(index[u], index[v]) for u, v in chords],
            weight,
        )
        if code == 1:
            raise ValueError(*_NEGATIVE_CYCLE)  # from `_dijkstra`
        if code == 2:
            raise ValueError("Contradictory paths found: negative weights?")
        cb.extend([nodes[i] for i in cycle] for cycle in found)
    return cb


# --- Batch 8: trees, branchings and lowest common ancestors -------------------------

from ._core import CoreGraph as _CoreGraph  # noqa: E402  (graph-less Rust helpers)

# NetworkX recurses once per level of a nested tuple or tree; deeper inputs
# fall back, so NetworkX raises its own RecursionError where it would.
_NESTED_DEPTH = 100

# `maximum_branching`'s internal names, which a graph must not already use.
_EDMONDS_CANDIDATE = "edmonds' secret candidate attribute"
_EDMONDS_NEW_NODE = "edmonds new node base name "


def _guarded(G, produce, fallback=None):
    """A generator over ``produce()`` that starts when iteration does. If G
    changed before that, NetworkX's ``fallback(graph)`` runs instead (when
    given); a change after the first item raises, as in ``_traversal``."""
    guard = _MutationGuard(G)

    def generate():
        try:
            if fallback is not None and guard.changed():
                yield from fallback(guard.graph)
                return
            for item in produce():
                if guard.changed():
                    raise RuntimeError("Graph changed during iteration")
                yield item
        finally:
            guard.release()

    return generate()


def _weight_core(G, attr, default):
    """A core whose ``attr`` weights were converted with ``default`` for
    missing values (NetworkX's ``data.get(attr, default)``)."""
    if not isinstance(attr, str):
        raise NotImplementedError("rustnx needs a string weight attribute")
    if type(default) not in (int, float) or not math.isfinite(default):
        raise NotImplementedError("rustnx needs a finite int or float default")
    key = (attr, type(default), default, math.copysign(1, default))
    if G._core.is_native():
        if key != (attr, int, 1, 1.0):
            raise NotImplementedError("native graphs read missing values as 1")
        return G._core
    base = _networkx_graph(G)
    for name, stored in G._weight_attrs:
        if name == attr and type(stored) in (int, float) and (
            (attr, type(stored), stored, math.copysign(1, stored)) == key
        ):
            return G._core
    # Kept on G, which NetworkX caches while the graph is unchanged.
    cores = G.__dict__.setdefault("_default_cores", {})
    if key not in cores:
        from .graph import from_networkx

        cores[key] = from_networkx(base, {attr: default})._core
    return cores[key]


def branching_weight(G, attr="weight", default=1):
    core = _weight_core(G, attr, default)
    all_int, has_hidden = core.weight_info(attr)
    if has_hidden or core.weight_mixed(attr):
        # `sum()` of ints and floats mixed in edge order: rustnx keeps one
        # type per sum.
        raise NotImplementedError("edge weights mix types")
    return core.edge_weight_sum(attr, _COMPENSATED_SUM)


def _numbers(values):
    if not all(type(w) in (int, float) for w in values):
        raise NotImplementedError("rustnx needs int or float weights")


def _edge_positions(G, edges):
    index = G._index
    return [index[u] for u, _, _ in edges], [index[v] for _, v, _ in edges]


def greedy_branching(G, attr="weight", default=1, kind="max", seed=None):
    try:
        known = kind in {"max", "min"}
    except TypeError:
        raise NotImplementedError("unhashable kind") from None
    if not known:
        raise nx.NetworkXException("Unknown value for `kind`.")
    nodes = G._nodes
    types = {type(x) for x in nodes}
    if not (types <= {int} or types == {str}):
        # NetworkX sorts by (weight, u, v); other labels may not compare.
        raise NotImplementedError("rustnx needs int or str node labels")
    base = _networkx_graph(G)
    rank = [0] * len(nodes)
    for r, x in enumerate(sorted(range(len(nodes)), key=nodes.__getitem__)):
        rank[x] = r
    if isinstance(attr, str):
        # The weights as converted (ints are exact in floats, and compare
        # with floats as Python does); the result keeps the edges' values.
        core = _weight_core(G, attr, default)
        if core.weight_info(attr)[1]:
            raise NotImplementedError("rustnx needs int or float weights")
        us, vs = core.greedy_branching_edges(attr, rank, kind != "min")
        adj = base._adj
        B = nx.DiGraph()
        B.add_nodes_from(base)
        for u, v in zip(us, vs):
            a, b = nodes[u], nodes[v]
            B.add_edge(a, b, **{attr: adj[a][b].get(attr, default)})
        return B
    if attr is None:
        if type(default) not in (int, float) or math.isnan(default):
            raise NotImplementedError("rustnx needs an int or float default")
        # Generate a random string the graph probably won't have.
        from networkx.algorithms.tree.branchings import random_string

        attr = random_string(seed=seed)
    edges = list(base.edges(data=True))
    weights = [data.get(attr, default) for _, _, data in edges]
    us, vs = _edge_positions(G, edges)
    kept = _CoreGraph.greedy_branching(us, vs, weights, rank, kind != "min")
    if kept is None:
        raise NotImplementedError("rustnx needs int or float weights (no NaN)")
    B = nx.DiGraph()
    B.add_nodes_from(base)
    for k in kept:
        u, v, _ = edges[k]
        B.add_edge(u, v, **{attr: weights[k]})
    return B


def _partition_state(value):
    if value is None:
        return 0
    if type(value) is not nx.EdgePartition:
        raise NotImplementedError("rustnx needs EdgePartition values")
    return value.value  # OPEN 0, INCLUDED 1, EXCLUDED 2


def _check_edmonds(base, edges, attr, preserve_attrs, partition):
    if edges and not isinstance(attr, str):
        # NetworkX passes the data as keywords, which raises TypeError.
        raise NotImplementedError("rustnx needs a string weight attribute")
    if partition is not None and (not isinstance(partition, str) or partition == attr):
        raise NotImplementedError("rustnx needs a separate string partition attribute")
    if _EDMONDS_CANDIDATE in (attr, partition):
        raise NotImplementedError("attribute name used internally by NetworkX")
    if any(isinstance(x, str) and x.startswith(_EDMONDS_NEW_NODE) for x in base):
        raise NotImplementedError("node name used internally by NetworkX")
    if preserve_attrs:
        for _, _, data in edges:
            if _EDMONDS_CANDIDATE in data or not all(isinstance(k, str) for k in data):
                raise NotImplementedError("edge attribute names NetworkX can't copy")


def _branching(G, base, edges, weights, attr, preserve_attrs, partition):
    """NetworkX's `maximum_branching` reading ``weights`` (one per edge of
    ``edges``, in ``G.edges`` order) for ``attr``."""
    _check_edmonds(base, edges, attr, preserve_attrs, partition)
    _numbers(weights)
    if partition is None:
        part = bytes(len(edges))
        values = None
    else:
        values = [data.get(partition) for _, _, data in edges]
        part = bytes(_partition_state(p) for p in values)
    us, vs = _edge_positions(G, edges)
    result = _CoreGraph.edmonds(len(G), us, vs, weights, part)
    if result is None:
        raise NotImplementedError("rustnx can't follow NetworkX on these weights")
    return result, values


def _branching_graph(base, edges, weights, attr, preserve_attrs, partition, result, values):
    initial, log = result
    # NetworkX lists the final edges from a set of edge keys; replaying the
    # same set operations gives the same iteration order.
    keys = set(dict.fromkeys(initial))
    for circuit, removed in log:
        keys.update(circuit)
        keys.remove(removed)
    H = base.__class__()
    H.add_nodes_from(base)
    for k in keys:
        u, v, data = edges[k]
        dd = {attr: weights[k]}
        if preserve_attrs:
            if values is not None and values[k] is not None:
                dd[partition] = values[k]
            for key, value in data.items():
                if key != attr and key not in dd:
                    dd[key] = value
        H.add_edge(u, v, **dd)
    return H


def maximum_branching(G, attr="weight", default=1, preserve_attrs=False, partition=None):
    base = _networkx_graph(G)
    edges = list(base.edges(data=True))
    weights = [data.get(attr, default) for _, _, data in edges]
    result, values = _branching(G, base, edges, weights, attr, preserve_attrs, partition)
    return _branching_graph(base, edges, weights, attr, preserve_attrs, partition, result, values)


def _transformed_branching(G, attr, default, preserve_attrs, partition, forward, backward):
    """The wrappers that rewrite G's weights with ``forward``, run
    `maximum_branching`, restore them with ``backward`` and apply
    ``backward`` to the result's weights. The rewrites change G's data as
    NetworkX's do (an edge without ``attr`` gains it, and float rounding may
    stick); they happen only once rustnx knows it can finish."""
    base = _networkx_graph(G)
    edges = list(base.edges(data=True))
    weights = [data.get(attr, default) for _, _, data in edges]
    _numbers(weights)
    forward = forward(weights)
    shifted = [forward(w) for w in weights]
    result, values = _branching(G, base, edges, shifted, attr, preserve_attrs, partition)
    for _, _, d in edges:
        d[attr] = forward(d.get(attr, default))
    nx._clear_cache(base)
    for _, _, d in edges:
        d[attr] = backward(d.get(attr, default))
    nx._clear_cache(base)
    B = _branching_graph(base, edges, shifted, attr, preserve_attrs, partition, result, values)
    for _, _, d in B.edges(data=True):
        d[attr] = backward(d.get(attr, default))
    nx._clear_cache(B)
    return B


def minimum_branching(G, attr="weight", default=1, preserve_attrs=False, partition=None):
    return _transformed_branching(
        G, attr, default, preserve_attrs, partition, lambda ws: operator.neg, operator.neg
    )


def _extremes(weights):
    """``(max_weight, min_weight)`` as NetworkX's loops find them."""
    max_weight, min_weight = -math.inf, math.inf
    for w in weights:
        if w > max_weight:
            max_weight = w
        if w < min_weight:
            min_weight = w
    return max_weight, min_weight


def _minimal_transform(G, attr, default, preserve_attrs, partition):
    span = {}

    def forward(weights):
        hi, lo = span["hl"] = _extremes(weights)
        return lambda w: hi + 1 + (hi - lo) - w

    def backward(w):
        hi, lo = span["hl"]
        return hi + 1 + (hi - lo) - w

    return _transformed_branching(G, attr, default, preserve_attrs, partition, forward, backward)


def minimal_branching(G, *, attr="weight", default=1, preserve_attrs=False, partition=None):
    return _minimal_transform(G, attr, default, preserve_attrs, partition)


def _check_arborescence(B, kind):
    if not B.is_directed():
        raise nx.NetworkXNotImplemented("not implemented for undirected type")
    if len(B) == 0:
        raise nx.NetworkXPointlessConcept("G has no nodes.")  # from is_tree
    # B is a branching, so it is an arborescence when it is connected.
    if B.number_of_edges() != len(B) - 1:
        raise nx.NetworkXException(f"No {kind} spanning arborescence in G.")
    return B


def maximum_spanning_arborescence(G, attr="weight", default=1, preserve_attrs=False, partition=None):
    span = {}

    def forward(weights):
        hi, lo = _extremes(weights)
        span["hl"] = hi, lo
        return lambda w: w - lo + 1 - (lo - hi)

    def backward(w):
        hi, lo = span["hl"]
        return w + lo - 1 + (lo - hi)

    B = _transformed_branching(G, attr, default, preserve_attrs, partition, forward, backward)
    return _check_arborescence(B, "maximum")


def minimum_spanning_arborescence(G, attr="weight", default=1, preserve_attrs=False, partition=None):
    B = _minimal_transform(G, attr, default, preserve_attrs, partition)
    return _check_arborescence(B, "minimum")


def _prim_starts(G):
    """The nodes `prim_mst_edges` pops from ``set(G)`` to start each tree.
    NetworkX discards each tree's nodes from the set as it goes; doing the
    same pops and discards on a real set gives the same order."""
    nodes = G._nodes
    comps = G._core.connected_components()
    comp_of = [0] * len(nodes)
    for c, comp in enumerate(comps):
        for i in comp:
            comp_of[i] = c
    index = G._index
    remaining = set(nodes)
    discard = remaining.discard  # not difference_update, which may resize
    starts = []
    while remaining:
        i = index[remaining.pop()]
        starts.append(i)
        for j in comps[comp_of[i]]:
            discard(nodes[j])
    return starts


def prim_mst_edges(G, minimum, weight="weight", keys=True, data=True, ignore_nan=False):
    if G.is_directed():
        raise NotImplementedError("rustnx implements prim_mst_edges for undirected graphs")
    base = _networkx_graph(G)
    weight = _unhidden_weight(G, weight)  # NaN weights fall back when converted
    nodes = G._nodes

    def produce():
        adj = base._adj
        us, vs = G._core.prim_edges(_prim_starts(G), weight, bool(minimum))
        for u, v in zip(us, vs):
            a, b = nodes[u], nodes[v]
            yield (a, b, adj[a][b]) if data else (a, b)

    from networkx.algorithms.tree import mst

    def fallback(H):
        return mst.prim_mst_edges(H, minimum, weight, keys, data, ignore_nan, backend="networkx")

    return _guarded(G, produce, fallback)


def boruvka_mst_edges(G, minimum=True, weight="weight", keys=True, data=True, ignore_nan=False):
    base = _networkx_graph(G)
    weight = _unhidden_weight(G, weight)  # NaN weights fall back when converted
    nodes = G._nodes
    index = G._index

    def produce():
        state = G._core.boruvka(weight, bool(minimum))
        adj = base._adj
        while True:
            found = state.round()
            if found is None:
                # A component's best edge is tied, so the order NetworkX
                # scans it in matters: that of `{n for n in component}`,
                # where `forest.to_sets()` builds the component by adding
                # its nodes in node order.
                orders = []
                for comp in state.components():
                    members = set()
                    for i in comp:
                        members.add(nodes[i])
                    orders.append([index[x] for x in {x for x in members}])
                found = state.round(orders)
            us, vs, any_best = found
            if not any_best:
                return
            for u, v in zip(us, vs):
                a, b = nodes[u], nodes[v]
                yield (a, b, adj[a][b]) if data else (a, b)

    from networkx.algorithms.tree import mst

    def fallback(H):
        return mst.boruvka_mst_edges(H, minimum, weight, keys, data, ignore_nan, backend="networkx")

    return _guarded(G, produce, fallback)


def partition_spanning_tree(G, minimum=True, weight="weight", partition="partition", ignore_nan=False):
    base = _networkx_graph(G)
    weight = _unhidden_weight(G, weight)
    try:
        state = bytes(_partition_state(d.get(partition)) for _, _, d in base.edges(data=True))
    except TypeError:
        raise NotImplementedError("unhashable partition attribute") from None
    us, vs = G._core.kruskal_partition(state, weight, not minimum)
    nodes = G._nodes
    adj = base._adj
    T = base.__class__()
    T.graph.update(base.graph)
    T.add_nodes_from(base.nodes.items())
    T.add_edges_from((nodes[u], nodes[v], adj[nodes[u]][nodes[v]]) for u, v in zip(us, vs))
    return T


def from_prufer_sequence(sequence):
    n = len(sequence) + 2
    values = list(sequence)
    if len(values) != n - 2 or any(type(v) is not int for v in values):
        raise NotImplementedError("rustnx needs a sequence of ints")
    us, vs, error = _CoreGraph.prufer_edges([v if 0 <= v < n else -1 for v in values])
    if error == -2:
        raise NotImplementedError("NetworkX's leaf search fails here")
    if error >= 0:
        raise nx.NetworkXError(
            f"Invalid Prufer sequence: Values must be between 0 and {n - 1}, got {values[error]}"
        )
    T = nx.Graph()  # nx.empty_graph(n), without dispatching
    T.add_nodes_from(range(n))
    T.add_edges_from(zip(us, vs))
    orphans = set(T) - set(us)
    u, v = orphans
    T.add_edge(u, v)
    return T


def from_nested_tuple(sequence, sensible_relabeling=False):
    tree = _CoreGraph.nested_tuple_tree(sequence, bool(sensible_relabeling), _NESTED_DEPTH)
    if tree is None:
        raise NotImplementedError("rustnx needs a nested sequence of sized iterables")
    order, us, vs = tree
    T = nx.Graph()
    T.add_nodes_from(order)
    T.add_edges_from(zip(us, vs))
    return T


def to_nested_tuple(T, root, canonical_form=False):
    _undirected_only(T)
    if not canonical_form:
        # Children are listed in the order of a set of nodes.
        raise NotImplementedError("rustnx implements canonical_form=True only")
    if not is_tree(T):
        raise nx.NotATree("provided graph is not a tree")
    result = T._core.canonical_nested_tuple(_node_arg(T, root), _NESTED_DEPTH)
    if result is None:
        raise NotImplementedError("tree too deep")
    return result


def tree_centroid(G):
    _undirected_only(G)
    if not is_tree(G):
        raise nx.NotATree("provided graph is not a tree")
    nodes = G._nodes
    return [nodes[i] for i in G._core.tree_centroid()]


_LCA_BATCH = 4096


def _lca_by_sets(G, base, v, w, cache):
    """NetworkX's own walk for a pair with several lowest common ancestors:
    it starts from the first element of a set intersection. The ancestor
    sets are built as NetworkX builds them (``ancestors`` gives the same
    set order), and cached per node as NetworkX does."""
    for x in (v, w):
        if x not in cache:
            cache[x] = ancestors(G, x)
            cache[x].add(x)
    common = cache[v] & cache[w]
    ancestor = next(iter(common))
    while True:
        successor = None
        for lower in base._succ[ancestor]:
            if lower in common:
                successor = lower
                break
        if successor is None:
            return ancestor
        ancestor = successor


def all_pairs_lowest_common_ancestor(G, pairs=None):
    _directed_only(G)
    if not is_directed_acyclic_graph(G):
        raise nx.NetworkXError("LCA only defined on directed acyclic graphs.")
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("LCA meaningless on null graphs.")
    base = _networkx_graph(G)
    nodes = G._nodes
    if pairs is None:
        positions = None
    else:
        pairs = dict.fromkeys(pairs)
        nodeset = set(nodes)
        for pair in pairs:
            if set(pair) - nodeset:
                raise nx.NodeNotFound(f"Node(s) {set(pair) - nodeset} from pair {pair} not in G.")
        try:
            positions = [(_node_arg(G, v), _node_arg(G, w)) for v, w in pairs]
        except (NotImplementedError, TypeError, ValueError):
            # `pairs` may have been an iterator: hand NetworkX what it read.
            return nx.all_pairs_lowest_common_ancestor(base, pairs, backend="networkx")
    lca = G._core.dag_lca()

    def batches():
        n = len(nodes)
        if positions is None:
            for i in range(n):
                yield [i] * (n - i), list(range(i, n))
        else:
            for s in range(0, len(positions), _LCA_BATCH):
                chunk = positions[s:s + _LCA_BATCH]
                yield [p for p, _ in chunk], [q for _, q in chunk]

    def produce():
        cache = {}
        for us, vs in batches():
            for u, v, r in zip(us, vs, lca.query(us, vs)):
                if r == -1:
                    continue
                a, b = nodes[u], nodes[v]
                yield (a, b), (nodes[r] if r >= 0 else _lca_by_sets(G, base, a, b, cache))

    return _guarded(G, produce)


def lowest_common_ancestor(G, node1, node2, default=None):
    _directed_only(G)
    ans = list(all_pairs_lowest_common_ancestor(G, pairs=[(node1, node2)]))
    if ans:
        assert len(ans) == 1
        return ans[0][1]
    return default


def tree_all_pairs_lowest_common_ancestor(G, root=None, pairs=None):
    _directed_only(G)
    if pairs is not None:
        # NetworkX keeps pairs in sets, so results follow set order.
        raise NotImplementedError("rustnx does not support pairs")
    r = None if root is None else _node_arg(G, root)
    G._ensure_exact_pred()  # a node's parent is the first in G.pred

    def produce():
        if len(G) == 0:
            raise nx.NetworkXPointlessConcept("LCA meaningless on null graphs.")
        start = r
        if start is None:
            for v, deg in enumerate(G._core.in_out_degrees()[0]):
                if deg == 0:
                    if start is not None:
                        raise nx.NetworkXError("No root specified and tree has multiple sources.")
                    start = v
                elif deg > 1:
                    raise nx.NetworkXError("Tree LCA only defined on trees; use DAG routine.")
            if start is None:
                raise nx.NetworkXError("Graph contains a cycle.")
        results = G._core.tree_lca(start)
        nodes = G._nodes
        while True:
            vs, xs, ancestors = results.next_batch(65536)
            if not vs:
                return
            for v, x, a in zip(vs, xs, ancestors):
                yield (nodes[v], nodes[x]), nodes[a]

    return _guarded(
        G,
        produce,
        lambda H: nx.tree_all_pairs_lowest_common_ancestor(H, root=root, backend="networkx"),
    )


# --- Batch 9: planarity, chordal graphs and graph classes --------------------------


def _replay_embedding(G, ordered, calls):
    """The ``PlanarEmbedding`` NetworkX's ``lr_planarity`` builds, made with
    the same ``add_half_edge`` calls in the same order (rustnx ran the
    left-right test that decides them)."""
    nodes = G._nodes
    embedding = nx.PlanarEmbedding()
    embedding.add_nodes_from(nodes)
    add = embedding.add_half_edge
    for v, row in zip(nodes, ordered):
        previous = None
        for w in row:
            w = nodes[w]
            add(v, w, ccw=previous)
            previous = w
    first = embedding.add_half_edge_first
    for kind, a, b, r in calls:
        if kind == 0:
            first(nodes[a], nodes[b])
        elif kind == 1:
            add(nodes[a], nodes[b], ccw=nodes[r])
        else:
            add(nodes[a], nodes[b], cw=nodes[r])
    return embedding


def _embedding_from_layout(G, layout):
    """The same ``PlanarEmbedding``, with its dicts filled in directly from
    the state rustnx computed for those calls (much faster)."""
    offsets, targets, cws, ccws, ccw_first, pred_offsets, preds = layout
    nodes = G._nodes
    embedding = nx.PlanarEmbedding()
    embedding.add_nodes_from(nodes)
    succ = embedding._succ
    pred = embedding._pred
    dicts = [
        {"ccw": nodes[b], "cw": nodes[a]} if first else {"cw": nodes[a], "ccw": nodes[b]}
        for a, b, first in zip(cws, ccws, ccw_first)
    ]
    heads = [nodes[w] for w in targets]
    tails = []
    for i, v in enumerate(nodes):
        begin, end = offsets[i], offsets[i + 1]
        if begin != end:
            succ[v].update(zip(heads[begin:end], dicts[begin:end]))
            tails.extend([v] * (end - begin))
    for i, w in enumerate(nodes):
        begin, end = pred_offsets[i], pred_offsets[i + 1]
        if begin != end:
            pred[w].update((tails[p], dicts[p]) for p in preds[begin:end])
    return embedding


@functools.cache
def _embedding_layout_matches():
    """Whether filling the embedding's dicts directly gives what NetworkX's
    ``add_half_edge`` calls give (checked once on a small graph with
    reordered neighbors, in case a NetworkX release changes them)."""
    from .graph import from_networkx

    H = nx.Graph()
    H.add_nodes_from([5, 2, 7, 0, 3, 6, 1, 4, 8])
    H.add_edges_from([(1, 2), (0, 1), (2, 3), (3, 4), (4, 0), (5, 0), (5, 1), (5, 2), (5, 3),
                      (5, 4), (6, 0), (6, 1), (7, 3), (8, 7), (2, 4)])
    G = from_networkx(H, {})
    calls = G._core.planarity(1)[1]
    layout = G._core.planarity(2)[1]

    def state(E):
        return ([(u, v, list(d.items())) for u, v, d in E.edges(data=True)],
                [(v, list(E._pred[v])) for v in E])

    return state(_replay_embedding(G, *calls)) == state(_embedding_from_layout(G, layout))


def _check_recursion(depth):
    """The recursive planarity functions recurse once per level of the DFS
    tree (and of ``ref`` chains); decline graphs where NetworkX might hit
    the recursion limit, so it raises (or not) itself."""
    frames = 0
    frame = sys._getframe()
    while frame is not None:
        frames += 1
        frame = frame.f_back
    if frames + depth + 100 >= sys.getrecursionlimit():
        raise NotImplementedError("NetworkX may reach the recursion limit here")


def _counterexample(G, recursive):
    edges, depth = G._core.planarity_counterexample()
    if recursive:
        _check_recursion(depth)
    if edges is None:
        raise nx.NetworkXException("G is planar - no counter example.")
    nodes = G._nodes
    subgraph = nx.Graph()
    subgraph.add_edges_from((nodes[u], nodes[v]) for u, v in zip(*edges))
    return subgraph


def _check_planarity(G, counterexample, recursive):
    direct = _embedding_layout_matches()
    planar, embedding, depth = G._core.planarity(2 if direct else 1)
    if recursive:
        _check_recursion(depth)
    if planar:
        if direct:
            return True, _embedding_from_layout(G, embedding)
        return True, _replay_embedding(G, *embedding)
    if counterexample:
        return False, _counterexample(G, recursive)
    return False, None


def is_planar(G):
    return G._core.planarity(0)[0]


def check_planarity(G, counterexample=False):
    return _check_planarity(G, counterexample, False)


def check_planarity_recursive(G, counterexample=False):
    return _check_planarity(G, counterexample, True)


def get_counterexample(G):
    return _counterexample(G, False)


def get_counterexample_recursive(G):
    return _counterexample(G, True)


def _chordal(G):
    """``(is_chordal, treewidth)`` for an undirected graph of 4 or more
    nodes. With self-loops, NetworkX raises or not depending on the order
    its maximum cardinality search visits nodes in (set order)."""
    if G._core.has_self_loops():
        raise NotImplementedError("rustnx does not support self-loops here")
    return G._core.chordal()


def is_chordal(G):
    _undirected_only(G)
    if len(G) <= 3:
        return True
    return _chordal(G)[0]


def chordal_graph_treewidth(G):
    _undirected_only(G)  # raised by is_chordal
    if len(G) == 0:
        # NetworkX 3.4 and 3.5 return -2; later releases raise ValueError.
        raise NotImplementedError("rustnx does not support the null graph here")
    if G._core.has_self_loops():
        raise NotImplementedError("rustnx does not support self-loops here")
    chordal, treewidth = G._core.chordal()
    if not chordal:
        raise nx.NetworkXError("Input graph is not chordal.")
    return treewidth


def complete_to_chordal_graph(G):
    _undirected_only(G)
    H = _networkx_graph(G).copy()
    nodes = G._nodes
    alpha = {node: 0 for node in nodes}
    if len(G) <= 3 or _chordal(G)[0]:
        return H, alpha
    values, zs, ys = G._core.complete_to_chordal()
    alpha = dict(zip(nodes, values))
    # NetworkX collects the chords in a set and adds them in its order.
    chords = set()
    for z, y in zip(zs, ys):
        chords.add((nodes[z], nodes[y]))
    H.add_edges_from(chords)
    return H, alpha


def is_at_free(G):
    _undirected_only(G)
    return G._core.is_at_free()


def _membership_position(G, node):
    """Position of a node NetworkX only tests membership with (``None`` if
    not in G); unhashable nodes make NetworkX raise or not depending on
    what it reaches first."""
    try:
        return G._index.get(node)
    except TypeError:
        raise NotImplementedError("unhashable node") from None


def is_reachable(G, s, t):
    _directed_only(G)
    return G._core.tournament_reachable(_membership_position(G, s), _membership_position(G, t))


def tournament_is_strongly_connected(G):
    _directed_only(G)
    return G._core.tournament_strongly_connected()


def score_sequence(G):
    _directed_only(G)
    return G._core.sorted_out_degrees()


def is_perfect_graph(G):
    # NetworkX checks the graph type inside the dispatched function.
    _undirected_only(G)
    return G._core.is_perfect()


# --- Batch 10: triads, d-separation, degree sequences and matching ---------------


def triadic_census(G, nodelist=None):
    _directed_only(G)
    nodeset = None
    if nodelist is not None:
        if nodelist in G or iter(nodelist) is nodelist:
            # NetworkX takes one node, or calls len() on an iterator.
            raise NotImplementedError("rustnx needs a list of nodes")
        nodeset = _node_subset(G, nodelist)
        if len(nodelist) != len(nodeset):
            raise ValueError("nodelist includes duplicate nodes or nodes not in G")
    counts = G._core.triadic_census(nodeset)
    return dict(zip(_TRIAD_NAMES, counts))


_TRIAD_NAMES = (
    "003", "012", "102", "021D", "021U", "021C", "111D", "111U",
    "030T", "030C", "201", "120D", "120U", "120C", "210", "300",
)


class _NodeSet(Set):
    """Stands in for ``G.nodes`` in set arithmetic (``s - G.nodes``), with
    ``NodeView``'s semantics: the result is ``set(...)`` in ``s``'s order."""

    def __init__(self, G):
        self._G = G

    def __contains__(self, node):
        return node in self._G

    def __iter__(self):
        return iter(self._G._nodes)

    def __len__(self):
        return len(self._G)

    @classmethod
    def _from_iterable(cls, it):
        return set(it)


def _node_or_set(G, value):
    """NetworkX's ``{value} if value in G else value`` for the d-separation
    arguments; rustnx handles plain sets (anything else, NetworkX's own set
    arithmetic decides what happens)."""
    value = {value} if value in G else value
    if type(value) not in (set, frozenset):
        raise NotImplementedError("rustnx needs nodes or sets of nodes")
    return value


def _dag_or_raise(G):
    if not is_directed_acyclic_graph(G):
        raise nx.NetworkXError("graph should be directed acyclic")


def is_d_separator(G, x, y, z):
    _directed_only(G)
    x, y, z = (_node_or_set(G, s) for s in (x, y, z))
    intersection = x & y or x & z or y & z
    if intersection:
        raise nx.NetworkXError(f"The sets are not disjoint, with intersection {intersection}")
    set_v = x | y | z
    missing = set_v - _NodeSet(G)
    if missing:
        raise nx.NodeNotFound(f"The node(s) {missing} are not found in G")
    _dag_or_raise(G)
    index = G._index
    return G._core.is_d_separator(*([index[v] for v in s] for s in (x, y, z)))


def _included_restricted(G, included, restricted):
    if included is None:
        included = set()
    elif included in G:
        included = {included}
    if restricted is not None:
        restricted = _node_or_set(G, restricted)
    if type(included) not in (set, frozenset):
        raise NotImplementedError("rustnx needs nodes or sets of nodes")
    return included, restricted


def _check_found(G, given, restricted):
    """NetworkX's ``set_y - G.nodes`` check, with ``set_y`` the union of
    ``given`` and ``restricted`` (``set(G)`` by default)."""
    nodes = _NodeSet(G)
    if restricted is None:
        if not functools.reduce(operator.or_, given) - nodes:
            return
        restricted = set(G._nodes)
    missing = functools.reduce(operator.or_, (*given, restricted)) - nodes
    if missing:
        raise nx.NodeNotFound(f"The node(s) {missing} are not found in G")


def is_minimal_d_separator(G, x, y, z, *, included=None, restricted=None):
    _directed_only(G)
    _dag_or_raise(G)
    x, y, z = (_node_or_set(G, s) for s in (x, y, z))
    included, restricted = _included_restricted(G, included, restricted)
    _check_found(G, (x, y, included), restricted)
    if not included <= z:
        raise nx.NetworkXError(
            f"Included nodes {included} must be in proposed separating set z {x}"
        )
    if restricted is None:
        if any(v not in G for v in z):
            restricted = set(G._nodes)
    if restricted is not None and not z <= restricted:
        raise nx.NetworkXError(
            f"Separating set {z} must be contained in restricted set {restricted}"
        )
    intersection = x.intersection(y) or x.intersection(z) or y.intersection(z)
    if intersection:
        raise nx.NetworkXError(f"The sets are not disjoint, with intersection {intersection}")
    index = G._index
    return G._core.is_minimal_d_separator(
        *([index[v] for v in s] for s in (x, y, z, included))
    )


def find_minimal_d_separator(G, x, y, *, included=None, restricted=None):
    _directed_only(G)
    _dag_or_raise(G)
    x, y = (_node_or_set(G, s) for s in (x, y))
    included, restricted = _included_restricted(G, included, restricted)
    _check_found(G, (x, y, included), restricted)
    if restricted is None:
        restricted = set(G._nodes)
    index, nodes = G._index, G._nodes
    for s in (x, y, included, restricted):
        for v in s:
            if type(nodes[index[v]]) is not type(v):
                # The result mixes the given objects and G's nodes.
                raise NotImplementedError("rustnx needs G's own node objects")
    if not included <= restricted:
        raise nx.NetworkXError(
            f"Included nodes {included} must be in restricted nodes {restricted}"
        )
    intersection = x & y or x & included or y & included
    if intersection:
        raise nx.NetworkXError(
            f"The sets x, y, included are not disjoint. Overlap: {intersection}"
        )
    G._ensure_exact_pred()
    core = G._core
    nodeset = x | y | included
    seeds = [index[v] for v in nodeset]
    # The same sets built in the same order, so their iteration order (and
    # the order of set operations on them) is NetworkX's.
    ancestors = nodeset.union(*[{nodes[i] for i in a} for a in core.ancestor_lists(seeds)])
    z_init = restricted & (ancestors - (x | y))

    def reachable(start, z):
        reached = core.d_reachable([index[v] for v in start], seeds, [index[v] for v in z])
        return {nodes[i] for i in reached}

    x_closure = reachable(x, z_init)
    if x_closure & y:
        return None
    z_updated = z_init & (x_closure | included)
    y_closure = reachable(y, z_updated)
    return z_updated & (y_closure | included)


def _int_list(sequence):
    """``nx.utils.make_list_of_ints(sequence)``: a list of plain ints is
    returned as is (NetworkX changes nothing in it); anything else goes
    through NetworkX's helper, with its errors and in-place conversion."""
    if type(sequence) is list and _core._plain_int_list(sequence):
        return sequence
    return nx.utils.make_list_of_ints(sequence)


def _degree_test(kind, sequence, name):
    seq = _int_list(sequence)
    result = _core._degree_sequence_test(kind, seq)
    if result is None:
        # Ints beyond 64 bits: NetworkX's own code on the converted list.
        return getattr(nx, name).orig_func(seq)
    return result


def is_valid_degree_sequence_havel_hakimi(deg_sequence):
    return _degree_test("hh", deg_sequence, "is_valid_degree_sequence_havel_hakimi")


def is_valid_degree_sequence_erdos_gallai(deg_sequence):
    return _degree_test("eg", deg_sequence, "is_valid_degree_sequence_erdos_gallai")


def is_graphical(sequence, method="eg"):
    if method == "eg":
        return is_valid_degree_sequence_erdos_gallai(list(sequence))
    elif method == "hh":
        return is_valid_degree_sequence_havel_hakimi(list(sequence))
    msg = "`method` must be 'eg' or 'hh'"
    raise nx.NetworkXException(msg)


def is_multigraphical(sequence):
    try:
        seq = _int_list(sequence)
    except nx.NetworkXError:
        return False
    result = _core._degree_sequence_test("multi", seq)
    return nx.is_multigraphical.orig_func(seq) if result is None else result


def is_pseudographical(sequence):
    try:
        seq = _int_list(sequence)
    except nx.NetworkXError:
        return False
    result = _core._degree_sequence_test("pseudo", seq)
    if result is None or not seq:
        # Big ints, or `min()` of an empty sequence (which raises).
        return sum(seq) % 2 == 0 and min(seq) >= 0
    return result


def is_digraphical(in_sequence, out_sequence):
    try:
        ins = _int_list(in_sequence)
        outs = _int_list(out_sequence)
    except nx.NetworkXError:
        return False
    result = _core._digraphical(ins, outs)
    return nx.is_digraphical.orig_func(ins, outs) if result is None else result


def node_boundary(G, nbunch1, nbunch2=None):
    nset1 = {n for n in nbunch1 if n in G}
    index, nodes = G._index, G._nodes
    nbrs = G._core.neighbor_union([index[v] for v in nset1])
    bdy = set([nodes[i] for i in nbrs]) - nset1
    if nbunch2 is not None:
        bdy &= set(nbunch2)
    return bdy


def edge_boundary(G, nbunch1, nbunch2=None, data=False, keys=False, default=None):
    if data is not False:
        raise NotImplementedError("rustnx does not support edge data here")
    nset1 = {n for n in nbunch1 if n in G}
    order = list(nset1)
    index = G._index
    positions = [index[v] for v in order]
    second = None
    if nbunch2 is not None:
        second = [index[v] for v in set(nbunch2) if v in G]
    sources, targets = G._core.edge_boundary(positions, second)

    def produce():
        nodes = G._nodes
        for i, v in zip(sources, targets):
            # The source is `nset1`'s object, as `nbunch_iter` yields it.
            yield order[i], nodes[v]

    return _traversal(G, produce)


def _matching_pairs(G, matching):
    """``is_matching``'s loop up to the first exception it would raise: the
    node positions of each edge before it, and the exception (or ``None``).
    NetworkX raises it only if no earlier edge makes it return False."""
    if isinstance(matching, dict):
        matching = _nx_matching.matching_dict_to_set(matching)
    index = G._index
    us, vs = [], []
    edges = iter(matching)
    while True:
        try:
            edge = next(edges)
        except StopIteration:
            return us, vs, None
        except Exception as exc:
            return us, vs, exc
        try:
            if len(edge) != 2:
                return us, vs, nx.NetworkXError(f"matching has non-2-tuple edge {edge}")
            u, v = edge
        except Exception as exc:
            return us, vs, exc
        try:
            i, j = index[u], index[v]
        except (KeyError, TypeError):
            return us, vs, nx.NetworkXError(f"matching contains edge {edge} with node not in G")
        us.append(i)
        vs.append(j)


def _checked_matching(G, matching):
    """``(us, vs)`` if ``matching`` passes ``is_matching``'s loop, else
    ``None``; raises where it would."""
    us, vs, error = _matching_pairs(G, matching)
    failure = G._core.first_matching_failure(us, vs)
    if failure is not None:
        return None
    if error is not None:
        raise error
    return us, vs


def is_matching(G, matching):
    return _checked_matching(G, matching) is not None


def is_maximal_matching(G, matching):
    pairs = _checked_matching(G, matching)
    if pairs is None:
        return False
    return not G._core.has_unmatched_edge(pairs[0] + pairs[1])


def is_perfect_matching(G, matching):
    pairs = _checked_matching(G, matching)
    if pairs is None:
        return False
    return len(set(pairs[0] + pairs[1])) == len(G)


def maximal_matching(G):
    _undirected_only(G)
    us, vs = G._core.maximal_matching()
    nodes = G._nodes
    return {(nodes[u], nodes[v]) for u, v in zip(us, vs)}


def _matching_set(G, weight, maxcardinality=False, inverted=False):
    weight, _, has_hidden = _check_weight(G, weight)
    if has_hidden:
        raise NotImplementedError("rustnx does not support None edge weights here")
    pairs = G._core.max_weight_matching(weight, maxcardinality, inverted)
    if pairs is None:
        raise NotImplementedError("integer edge weights are too large")
    nodes = G._nodes
    # `matching_dict_to_set` adds them in this order.
    return {(nodes[u], nodes[v]) for u, v in zip(*pairs)}


def max_weight_matching(G, maxcardinality=False, weight="weight"):
    _undirected_only(G)
    if type(maxcardinality) not in (bool, int):
        raise NotImplementedError("rustnx needs a bool maxcardinality")
    if len(G) == 0:
        return set()
    return _matching_set(G, weight, bool(maxcardinality))


def min_weight_matching(G, weight="weight"):
    _undirected_only(G)
    if G._core.number_of_edges() == 0:
        return set()
    return _matching_set(G, weight, True, inverted=True)


def min_edge_cover(G, matching_algorithm=None):
    _undirected_only(G)
    if len(G) == 0:
        return set()
    if number_of_isolates(G) > 0:
        raise nx.NetworkXException(
            "Graph has a node with no edge incident on it, so no edge cover exists."
        )
    if matching_algorithm is not None:
        raise NotImplementedError("rustnx does not support matching_algorithm")
    min_cover = _matching_set(G, "weight", True)
    uncovered_nodes = set(G._nodes) - {v for u, v in min_cover} - {u for u, v in min_cover}
    index, nodes, core = G._index, G._nodes, G._core
    for v in uncovered_nodes:
        u = nodes[core.neighbors(index[v])[0]]
        min_cover.add((u, v))
    return min_cover


def _present_positions(G, nbunch):
    """Positions of the items of ``nbunch`` that are nodes of G (as
    ``{n for n in nbunch if n in G}`` picks them)."""
    index = G._index
    picked = []
    for n in nbunch:
        try:
            i = index.get(n)
        except TypeError:
            continue
        if i is not None:
            picked.append(i)
    return picked


def is_dominating_set(G, nbunch):
    return G._core.is_dominating(_present_positions(G, nbunch))


def is_connected_dominating_set(G, nbunch):
    _undirected_only(G)
    if not is_dominating_set(G, nbunch):
        return False
    # `nx.subgraph(G, nbunch)` reads `nbunch` again, via `nbunch_iter`.
    if nbunch in G:
        picked = [G._index[nbunch]]
    else:
        index = G._index
        picked = []
        for n in nbunch:
            try:
                i = index.get(n)
            except TypeError:
                raise nx.NetworkXError(f"Node {n} in sequence nbunch is not a valid node.") from None
            if i is not None:
                picked.append(i)
    if not picked:
        raise nx.NetworkXPointlessConcept("Connectivity is undefined for the null graph.")
    return G._core.induced_connected(picked)


def connected_dominating_set(G):
    _undirected_only(G)
    if len(G) == 0:
        return set()
    if not is_connected(G):
        raise nx.NetworkXError("G must be a connected graph")
    nodes = G._nodes
    if len(G) == 1:
        return set(nodes)
    return {nodes[i] for i in G._core.connected_dominating_set()}


_CLIQUE_BATCH = 1024


def enumerate_all_cliques(G):
    _undirected_only(G)

    def compute():
        queue = G._core.all_cliques()
        nodes = G._nodes
        nodes = nodes if type(nodes) is list else list(nodes)
        while True:
            batch = queue.next_batch(nodes, _CLIQUE_BATCH)
            yield from batch
            if len(batch) < _CLIQUE_BATCH:
                return

    return _computed_on_first_next(
        G, compute, lambda H: nx.enumerate_all_cliques(H, backend="networkx")
    )


def node_clique_number(G, nodes=None, cliques=None, separate_nodes=False):
    if cliques is not None or nodes is None:
        # Without `nodes`, the dict follows `find_cliques`' order, which
        # depends on set iteration order.
        raise NotImplementedError("rustnx needs nodes and no cliques here")
    if G.is_directed():
        raise NotImplementedError("NetworkX raises from find_cliques here")
    index = G._index
    if nodes in G:
        return G._core.node_clique_numbers([index[nodes]])[0]
    if iter(nodes) is nodes:
        raise NotImplementedError("rustnx needs a reusable container of nodes")
    try:
        positions = [index[n] for n in nodes]
    except (KeyError, TypeError):
        raise NotImplementedError("NetworkX raises for missing nodes") from None
    values = G._core.node_clique_numbers(positions)
    return {n: values[i] for i, n in enumerate(nodes)}


def max_weight_clique(G, weight="weight"):
    _undirected_only(G)
    weights = None
    if weight is not None:
        source = G._source
        if source is None or not G._source_unchanged():
            raise NotImplementedError("rustnx needs the NetworkX graph's node data")
        node_data = source._node
        weights = []
        for v in G._nodes:
            data = node_data[v]
            if weight not in data:
                errmsg = f"Node {v!r} does not have the requested weight field."
                raise KeyError(errmsg)
            if not isinstance(data[weight], int):
                errmsg = f"The {weight!r} field of node {v!r} is not an integer."
                raise ValueError(errmsg)
            weights.append(data[weight])
        if any(type(w) not in (int, bool) or not -(2**62) < w < 2**62 for w in weights):
            raise NotImplementedError("rustnx needs plain integer node weights")
    clique, total = G._core.max_weight_clique(weights)
    nodes = G._nodes
    return [nodes[i] for i in clique], total


# --- Batch 11: isomorphism and graph hashing ----------------------------------------


def _registered(name):
    """The installed NetworkX's dispatchable ``name`` (some live in
    subpackages, or share a name with their module)."""
    return nx.utils.backends._registered_algorithms[name]


@functools.cache
def _staged_could_be_isomorphic(name):
    """Whether the installed ``name`` checks properties one at a time,
    stopping at the first mismatch (3.5+ ``could_be_isomorphic``, 3.7+
    ``fast_`` and ``faster_could_be_isomorphic``). Older ones compute every
    property of G1, then of G2, and compare once; for directed graphs that
    decides whether ``triangles`` raises or a degree mismatch returns
    ``False`` first."""
    text = _source_text(_registered(name))
    return "properties_to_check" in text or "could_be_isomorphic(G1, G2, properties=" in text


def _property_tables_match(G1, G2, properties, staged):
    if len(G1) != len(G2):
        return False
    d, t, c = ("d" in properties), ("t" in properties), ("c" in properties)
    match = G1._core.iso_tables_match
    if not staged:
        # One table of every property, G1's computed (and checked) first.
        if t or c:
            _undirected_only(G1)
            _undirected_only(G2)
        return match(G2._core, d, t, c)
    if d and not match(G2._core, True, False, False):
        return False
    if t:
        _undirected_only(G1)
        _undirected_only(G2)
        if not match(G2._core, d, True, False):
            return False
    if c:
        _undirected_only(G1)  # find_cliques
        _undirected_only(G2)
        if not match(G2._core, d, t, True):
            return False
    return True


def could_be_isomorphic(G1, G2, properties="dtc"):
    try:
        properties = set(properties)
    except TypeError:
        raise NotImplementedError("unsupported properties") from None
    return _property_tables_match(G1, G2, properties, _staged_could_be_isomorphic("could_be_isomorphic"))


def fast_could_be_isomorphic(G1, G2):
    return _property_tables_match(G1, G2, "dt", _staged_could_be_isomorphic("fast_could_be_isomorphic"))


def faster_could_be_isomorphic(G1, G2):
    # Degrees only: both styles agree.
    return _property_tables_match(G1, G2, "d", True)


def is_isomorphic(G1, G2):
    if G1.is_directed() != G2.is_directed():
        raise nx.NetworkXError("Graphs G1 and G2 are not of the same type.")
    # A yes/no answer: any exact matcher agrees with NetworkX's VF2.
    return G1._core.has_morphism(G2._core, None, None, 0)


def _node_labels(G, node_label, default_label):
    """``dict(G.nodes(data=node_label, default=default_label))`` values."""
    if G._core.is_native():
        return [default_label] * len(G)  # native graphs have no node data
    if not G._source_unchanged():
        raise NotImplementedError("the graph changed since it was converted")
    try:
        return [dd[node_label] if node_label in dd else default_label for dd in G._source._node.values()]
    except TypeError:
        raise NotImplementedError("unhashable node_label") from None


def _label_classes(FG, SG, node_label, default_label):
    """Both graphs' labels as integer classes: one dict for both, so labels
    that NetworkX's dicts and sets treat as equal share a class."""
    classes = {}
    try:
        big = [classes.setdefault(x, len(classes)) for x in _node_labels(FG, node_label, default_label)]
        small = [classes.setdefault(x, len(classes)) for x in _node_labels(SG, node_label, default_label)]
    except TypeError:
        raise NotImplementedError("unhashable node labels") from None
    if len(classes) <= 1:
        return None, None
    return big, small


@functools.cache
def _vf2pp_checks_directedness():
    """Whether ``vf2pp_is_isomorphic`` rejects a directed and an undirected
    graph (3.7+); older releases run on, with results rustnx doesn't copy."""
    from networkx.algorithms.isomorphism import vf2pp

    try:
        return "must have the same directedness" in inspect.getsource(vf2pp)
    except (OSError, TypeError):
        return False


def _vf2pp_test(FG, SG, node_label, default_label, problem, mixed_raises):
    # NetworkX's generator returns before anything else on an empty graph,
    # so even two empty graphs are "not isomorphic".
    if len(SG) == 0 or len(FG) == 0:
        return False
    if SG.is_directed() != FG.is_directed():
        if mixed_raises:
            raise nx.NetworkXError("SG and FG must have the same directedness")
        raise NotImplementedError("graphs differ in directedness")
    big, small = _label_classes(FG, SG, node_label, default_label)
    # A yes/no answer: VF2++'s candidate order (set order) doesn't matter.
    return FG._core.has_morphism(SG._core, big, small, problem)


def vf2pp_is_isomorphic(FG, SG, node_label=None, default_label=None):
    return _vf2pp_test(FG, SG, node_label, default_label, 0, _vf2pp_checks_directedness())


def vf2pp_subgraph_is_isomorphic(FG, SG, node_label=None, default_label=None):
    return _vf2pp_test(FG, SG, node_label, default_label, 1, True)


def vf2pp_is_monomorphic(FG, SG, node_label=None, default_label=None):
    return _vf2pp_test(FG, SG, node_label, default_label, 2, True)


@functools.cache
def _tree_isomorphism_style():
    """``(asserts, descending)`` for the installed ``rooted_tree_isomorphism``:
    3.4 checks trees with ``assert`` and orders children ascending, walking
    them recursively; 3.5+ raises ``NetworkXError`` and sorts children with
    ``reverse=True``, walking them with a stack."""
    text = _source_text(_registered("rooted_tree_isomorphism"))
    return "assert nx.is_tree(t1)" in text, "reverse=True" in text


def _check_trees(t1, t2):
    asserts = _tree_isomorphism_style()[0]
    for T, name in ((t1, "t1"), (t2, "t2")):
        if not is_tree(T):
            if asserts:
                raise AssertionError
            raise nx.NetworkXError(f"{name} is not a tree")


def _rooted_tree_pairs(t1, r1, t2, r2):
    descending = _tree_isomorphism_style()[1]
    if not descending:
        # 3.4 walks the result recursively: deep trees hit Python's
        # recursion limit there, which rustnx leaves to NetworkX.
        height = max(t1._core.tree_height(r1), t2._core.tree_height(r2))
        if height + 2 >= sys.getrecursionlimit() // 3:
            raise NotImplementedError("deep trees run in NetworkX")
    pairs = t1._core.rooted_tree_isomorphism(r1, t2._core, r2, descending)
    n1, n2 = t1._nodes, t2._nodes
    return [(n1[a], n2[b]) for a, b in pairs]


def rooted_tree_isomorphism(t1, root1, t2, root2):
    if t1.is_directed() or t2.is_directed():
        raise NotImplementedError("rustnx supports undirected trees here")
    _check_trees(t1, t2)
    r1, r2 = _node_arg(t1, root1), _node_arg(t2, root2)
    return _rooted_tree_pairs(t1, r1, t2, r2)


def tree_isomorphism(t1, t2):
    # NetworkX rejects a directed t1 before dispatching; t2 isn't checked.
    if t1.is_directed() or t2.is_directed():
        raise NotImplementedError("rustnx supports undirected trees here")
    _check_trees(t1, t2)
    if not _property_tables_match(t1, t2, "d", True):
        return []
    center1, center2 = t1._core.tree_centers(), t2._core.tree_centers()
    if len(center1) != len(center2):
        return []
    attempt = _rooted_tree_pairs(t1, center1[0], t2, center2[0])
    if attempt or len(center1) == 1:
        return attempt
    return _rooted_tree_pairs(t1, center1[0], t2, center2[1])


def root_trees(t1, root1, t2, root2):
    r1, r2 = _node_arg(t1, root1), _node_arg(t2, root2)
    newroot1 = 1
    newroot2 = len(t1) + 1
    edges = [(0, newroot1), (0, newroot2)]
    namemap = {}
    for T, root, r, new_root in ((t1, root1, r1, newroot1), (t2, root2, r2, newroot2)):
        parents, children = T._core.bfs_edges(r, len(T), False)
        nodes = T._nodes
        # NetworkX names nodes new_root, new_root + 1, ... in BFS order.
        new = {r: new_root}
        namemap[new_root] = root
        for i, (p, c) in enumerate(zip(parents, children)):
            new[c] = new_root + 1 + i
            edges.append((new[p], new_root + 1 + i))
        namemap.update({new_root + 1 + i: nodes[c] for i, c in enumerate(children)})
    # namemap lists t1's nodes, then t2's, as NetworkX builds it.
    dT = nx.DiGraph()
    dT.add_edges_from(edges)
    return (dT, namemap, newroot1, newroot2)


@functools.cache
def _wl_new_style():
    """Whether the installed Weisfeiler-Lehman hashes are 3.5+'s: directed
    graphs aggregate successors and predecessors separately (``s_``/``p_``),
    without attributes the degree labels count as the first iteration, and
    ``iterations`` must be positive."""
    return "_neighborhood_aggregate_directed" in _source_text(_registered("weisfeiler_lehman_graph_hash"))


def _wl_prepare(G, name, edge_attr, node_attr, iterations, digest_size, extra):
    """Validate a WL call; return ``(labels, edge_text, split, steps)``."""
    for attr in (edge_attr, node_attr):
        if attr is not None and not (type(attr) is str and attr):
            raise NotImplementedError("rustnx supports str attribute names here")
    if type(iterations) is not int or type(digest_size) is not int or not 1 <= digest_size <= 64:
        raise NotImplementedError("unsupported iterations or digest_size")
    # NetworkX's own function on an empty graph gives the same warnings and
    # the same error for a non-positive `iterations`, in the same order.
    empty = nx.DiGraph() if G.is_directed() else nx.Graph()
    _registered(name).orig_func(
        empty, edge_attr=edge_attr, node_attr=node_attr, iterations=iterations,
        digest_size=digest_size, **extra
    )
    new = _wl_new_style()
    split = new and G.is_directed()
    if node_attr or edge_attr:
        source = None if G._core.is_native() else G._source
        if source is None or not G._source_unchanged():
            raise NotImplementedError("rustnx needs the NetworkX graph's attributes")
    if node_attr:
        labels = [str(dd[node_attr]) for dd in source._node.values()]
    elif edge_attr:
        labels = [""] * len(G)
    elif split:
        ins, outs = G._core.in_out_degrees()
        labels = [f"{i}_{o}" for i, o in zip(ins, outs)]
    else:
        labels = [str(d) for d in G._core.degrees()]
    steps = iterations - 1 if new and not edge_attr and not node_attr else iterations
    steps = max(steps, 0)
    edge_text = None
    if edge_attr is not None and steps:
        # `str(G[u][v][edge_attr])` per adjacency entry, in CSR order.
        adj = source._adj
        edge_text = [str(d[edge_attr]) for u in source for d in adj[u].values()]
        if not all(s.isascii() for s in edge_text):
            raise NotImplementedError("non-ASCII labels raise in NetworkX")
    if node_attr and not all(s.isascii() for s in labels):
        raise NotImplementedError("non-ASCII labels raise in NetworkX")
    return labels, edge_text, split, steps


def weisfeiler_lehman_graph_hash(G, edge_attr=None, node_attr=None, iterations=3, digest_size=16):
    labels, edge_text, split, steps = _wl_prepare(
        G, "weisfeiler_lehman_graph_hash", edge_attr, node_attr, iterations, digest_size, {}
    )
    return G._core.wl_hashes(labels, edge_text, split, steps, digest_size)[0]


def weisfeiler_lehman_subgraph_hashes(
    G, edge_attr=None, node_attr=None, iterations=3, digest_size=16, include_initial_labels=False
):
    extra = {"include_initial_labels": include_initial_labels}
    labels, edge_text, split, steps = _wl_prepare(
        G, "weisfeiler_lehman_subgraph_hashes", edge_attr, node_attr, iterations, digest_size, extra
    )
    # Initial labels hashed first: when asked, and (3.5+) without attributes,
    # where the degree labels stand for the first iteration (both can apply).
    initial = int(bool(include_initial_labels))
    if _wl_new_style() and not edge_attr and not node_attr:
        initial += 1
    if not initial and not steps:
        return {}
    per_node = G._core.wl_hashes(labels, edge_text, split, steps, digest_size, initial, True)[1]
    return dict(zip(G._nodes, per_node))


# Bipartite graphs (todo item 37), added in the same batch.


def color(G):
    if G.is_directed():
        G._ensure_exact_pred()  # predecessors come first, in NetworkX's order
    found = G._core.bipartite_color()
    if found is None:
        raise nx.NetworkXError("Graph is not bipartite.")
    order, colors = found
    nodes = G._nodes
    return dict(zip([nodes[v] for v in order], colors))


def sets(G, top_nodes=None):
    if top_nodes is not None:
        X = set(top_nodes)
        Y = set(G) - X
        return (X, Y)
    connected = is_weakly_connected(G) if G.is_directed() else is_connected(G)
    if not connected:
        raise nx.AmbiguousSolution("Disconnected graph: Ambiguous solution for bipartite sets.")
    c = color(G)
    # Built in NetworkX's insertion order, so the sets iterate the same way.
    X = {n for n, is_top in c.items() if is_top}
    Y = {n for n, is_top in c.items() if not is_top}
    return (X, Y)


def is_bipartite_node_set(G, nodes):
    S = set(nodes)
    if len(S) < len(nodes):
        raise nx.AmbiguousSolution(
            "The input node set contains duplicates.\n"
            "This may lead to incorrect results when using it in bipartite algorithms.\n"
            "Consider using set(nodes) as the input"
        )
    _undirected_only(G)  # connected_components
    result = G._core.is_bipartite_node_set([v in S for v in G._nodes])
    if result is None:
        raise nx.NetworkXError("Graph is not bipartite.")
    return result


def _bipartite_sides(G, top_nodes):
    """NetworkX's ``bipartite_sets`` for the matching functions, as node
    positions in set order (``NotImplementedError`` for nodes not in G)."""
    if G.is_directed():
        raise NotImplementedError("rustnx supports undirected graphs here")
    left, right = sets(G, top_nodes)
    index = G._index
    if any(v not in index for v in left):
        raise NotImplementedError("top_nodes has nodes not in the graph")
    return left, right, [index[v] for v in left], [index[v] for v in right]


def hopcroft_karp_matching(G, top_nodes=None):
    left, right, left_pos, right_pos = _bipartite_sides(G, top_nodes)
    found = G._core.hopcroft_karp(left_pos)
    if found is None:
        raise NotImplementedError("a top node has a neighbor among the top nodes")
    mate, depth = found
    if depth >= sys.getrecursionlimit() // 4:
        # NetworkX's depth_first_search recurses this deep; leave the
        # outcome (perhaps a RecursionError) to it.
        raise NotImplementedError("deep augmenting paths run in NetworkX")
    nodes = G._nodes
    matching = {v: nodes[mate[i]] for v, i in zip(left, left_pos) if mate[i] is not None}
    matching.update((v, nodes[mate[i]]) for v, i in zip(right, right_pos) if mate[i] is not None)
    return matching


def to_vertex_cover(G, matching, top_nodes=None):
    L, R, _, _ = _bipartite_sides(G, top_nodes)
    unmatched_vertices = set(G) - set(matching)
    U = unmatched_vertices & L
    index = G._index
    try:
        pairs = [(index[u], index[v]) for u, v in matching.items() if u in index and v in index]
    except (AttributeError, TypeError):
        raise NotImplementedError("matching must be a dict of hashable nodes") from None
    reach = G._core.alternating_reach([v in U for v in G._nodes], pairs)
    # NetworkX's set comprehension over G: the same insertions, the same set.
    Z = {v for v, ok in zip(G._nodes, reach) if ok}
    return (L - Z) | (R & Z)


def bipartite_closeness_centrality(G, nodes, normalized=True):
    top = set(nodes)
    bottom = set(G) - top
    n = len(top)
    m = len(bottom)
    index = G._index
    for node in top:
        if node not in index:
            raise nx.NodeNotFound(f"Source {node} is not in G")
    order = list(top) + list(bottom)
    stats = G._core.bfs_stats([index[v] for v in order])
    size = len(G)
    closeness = {}
    for i, node in enumerate(order):
        reached, totsp, _ = stats[i]
        if totsp > 0.0 and size > 1:
            closeness[node] = ((m + 2 * (n - 1)) if i < n else (n + 2 * (m - 1))) / totsp
            if normalized:
                s = (reached - 1) / (size - 1)
                closeness[node] *= s
        else:
            closeness[node] = 0.0
    return closeness


def node_redundancy(G, nodes=None):
    if nodes is None:
        nodes = G._nodes
    else:
        try:
            if iter(nodes) is nodes:
                raise NotImplementedError("rustnx needs a reusable container of nodes")
        except TypeError:
            raise NotImplementedError("nodes is not a container of nodes") from None
    index = G._index
    try:
        positions = [index[v] for v in nodes]
    except (KeyError, TypeError):
        raise NotImplementedError("nodes must be nodes of the graph") from None
    counts = G._core.redundancy_overlaps(positions)
    if any(d < 2 for d, _ in counts):
        raise nx.NetworkXError(
            "Cannot compute redundancy coefficient for a node"
            " that has fewer than two neighbors."
        )
    return {v: (2 * overlap) / (d * (d - 1)) for v, (d, overlap) in zip(nodes, counts)}


def butterflies(G, nodes=None):
    if G._core.number_of_edges() == 0:
        counts = [0] * len(G)
    else:
        counts = G._core.butterflies()
    if nodes is None:
        return dict(zip(G._nodes, counts))
    try:
        picked = _nbunch_list(G, nodes)
    except TypeError:
        raise NotImplementedError("nodes is not a node or a container of nodes") from None
    index = G._index
    return {v: counts[index[v]] for v in picked}


# --- Batch 12: flows and cut measures ------------------------------------------------


# NetworkX's maximum flow functions rustnx runs, and the keyword arguments
# each takes through `maximum_flow` and friends.
_FLOW_KWARGS = {
    "edmonds_karp": {"cutoff"},
    "shortest_augmenting_path": {"two_phase", "cutoff"},
    "dinitz": {"cutoff"},
    "boykov_kolmogorov": {"cutoff"},
    "preflow_push": {"global_relabel_freq"},
}


def _flow_func_name(flow_func):
    """Which of NetworkX's flow functions ``flow_func`` is, or ``None``."""
    for name in _FLOW_KWARGS:
        if flow_func is _registered(name):
            return name
    return None


@functools.cache
def _cut_by_layers():
    """Whether the installed ``minimum_cut`` (3.7+) finds the sink side with
    its own breadth-first search over edges with ``flow < capacity``,
    adding to ``{_t}`` one node at a time. Earlier versions delete the
    edges with ``flow == capacity``, take ``set(dict(shortest_path_length(R,
    target=_t)))`` (a presized set) and add the edges back, which moves them
    to the end of R's rows and so changes later flows on the same R."""
    return "non_reachable = {_t}" in _source_text(_registered("minimum_cut"))


@functools.cache
def _simplex_single_demand():
    """Whether the installed ``network_simplex`` (before 3.6) takes the
    largest single demand, rather than the demands' sum, for its faux
    infinity."""
    return "sum(abs(d) for d in DEAF.node_demands)" not in _source_text(
        _registered("network_simplex")
    )


@functools.cache
def _sets_replayable():
    """Whether rustnx's replica of CPython's set table gives this
    interpreter's set iteration order (``preflow_push`` picks nodes with
    ``next(iter(level.active))``), checked on random operations with keys
    whose hashes collide, are negative or large."""
    rng = random.Random(20261004)
    keys = list(range(48)) + [-1, -2, 2**61 - 1, 2**61, -(2**61), 2**64 + 7]
    keys += [1000 + 64 * i for i in range(60)] + [f"k{i}" for i in range(40)]
    keys += [(i, "t") for i in range(20)]
    position = {k: i for i, k in enumerate(keys)}
    nsets = 4
    sets = [set() for _ in range(nsets)]
    ops, firsts = [], []
    for _ in range(4000):
        s, r = rng.randrange(nsets), rng.random()
        if r < 0.55:
            op, k = 0, rng.randrange(len(keys))
            sets[s].add(keys[k])
        elif r < 0.9:
            op, k = 1, rng.randrange(len(keys))
            sets[s].discard(keys[k])
        elif r < 0.94:
            op, k = 2, 0
            sets[s].clear()
        else:
            op, k = 3, rng.randrange(nsets)
            sets[s].update(sets[k])
        ops.append((op, s, k))
        first = next(iter(sets[s]), None)
        firsts.append(-1 if first is None else position[first])
    try:
        got_firsts, got_sets = _core._replay_sets([hash(k) for k in keys], nsets, ops)
    except Exception:
        return False
    return got_firsts == firsts and got_sets == [[position[k] for k in s] for s in sets]


def _node_list(G):
    nodes = G._nodes
    return nodes if type(nodes) is list else list(nodes)


def _flow_rows(G):
    """The NetworkX graph's adjacency rows, aligned with the conversion."""
    return list(_networkx_graph(G)._adj.values())


def _flow_network(G, capacity):
    """``build_residual_network(G, capacity)`` as a ``FlowRun``."""
    if callable(capacity):
        raise NotImplementedError("rustnx does not support callable capacities")
    return G._core.residual_network(_flow_rows(G), capacity, _COMPENSATED_SUM)


def _flow_ends(G, s, t):
    if s not in G:
        raise nx.NetworkXError(f"node {str(s)} not in graph")
    if t not in G:
        raise nx.NetworkXError(f"node {str(t)} not in graph")
    if s == t:
        raise nx.NetworkXError("source and sink are the same node")
    return G._index[s], G._index[t]


def _run_flow(G, s, t, capacity, name, value_only, residual=None, cutoff=None,
              two_phase=False, global_relabel_freq=1):
    """NetworkX's flow function ``name`` up to building R: the ``FlowRun``
    and the flow value."""
    if residual is not None:
        raise NotImplementedError("rustnx does not reuse residual networks")
    si, ti = _flow_ends(G, s, t)
    if name == "preflow_push":
        if global_relabel_freq is None:
            global_relabel_freq = 0
        if global_relabel_freq < 0:
            raise nx.NetworkXError("global_relabel_freq must be nonnegative.")
    run = _flow_network(G, capacity)
    n, m = len(G), run.edge_count()
    if name == "preflow_push":
        if not _sets_replayable():
            raise NotImplementedError("this interpreter's sets can't be replayed")
        # GlobalRelabelThreshold
        threshold = (n + m) / global_relabel_freq if global_relabel_freq else float("inf")
        if type(threshold) is not float:
            raise NotImplementedError("unsupported global_relabel_freq")
        hashes = [hash(v) for v in G._nodes]
        value = run.run(name, si, ti, value_only=bool(value_only), threshold=threshold, hashes=hashes)
    elif name == "shortest_augmenting_path":
        half = m / 2
        d = n if not two_phase else int(min(half**0.5, 2 * n ** (2.0 / 3)))
        value = run.run(name, si, ti, cutoff, two_phase=bool(two_phase), d=d)
    else:
        value = run.run(name, si, ti, cutoff)
    return run, value


def _current_edge(edges, position):
    """A ``CurrentEdge`` over ``edges`` that has moved ``position`` times."""
    from networkx.algorithms.flow.utils import CurrentEdge

    current = CurrentEdge(edges)
    if position:
        current._it = it = iter(edges.items())
        current._curr = next(islice(it, position, None))
    return current


def _residual_graph(G, run, value=None, algorithm=None):
    """NetworkX's residual network ``R`` for ``run``: without flows from
    ``build_residual_network``, else with the flow and node attributes the
    algorithm leaves, then the graph's ``flow_value`` and ``algorithm``."""
    nodes = _node_list(G)
    R = nx.DiGraph()
    R.__networkx_cache__ = None
    R.add_nodes_from(nodes)
    run.fill(list(R._succ.values()), list(R._pred.values()), nodes, algorithm is not None)
    R.graph["inf"] = run.inf()
    if algorithm is None:
        return R
    positions = run.set_node_attrs(list(R._node.values()))
    if positions is not None:
        R_succ = R.succ  # NetworkX's CurrentEdges iterate these AtlasViews
        for (u, row), position in zip(R._node.items(), positions):
            row["curr_edge"] = _current_edge(R_succ[u], position)
    trees = run.trees(nodes)
    if trees is not None:
        R.graph["trees"] = trees
    R.graph["flow_value"] = value
    R.graph["algorithm"] = algorithm
    return R


def build_residual_network(G, capacity):
    return _residual_graph(G, _flow_network(G, capacity))


def build_flow_dict(G, R):
    # R is read as it is now, as NetworkX reads it (residual networks have
    # caching disabled, so R was converted just for this call anyway).
    R_adj = (R.to_networkx() if R._core.is_native() else R._source)._adj
    try:
        rows = [R_adj[u] for u in G._nodes]
    except (KeyError, TypeError):
        raise NotImplementedError("NetworkX raises for nodes missing from R") from None
    return G._core.build_flow_dict(_node_list(G), rows)


def edmonds_karp(G, s, t, capacity="capacity", residual=None, value_only=False, cutoff=None):
    run, value = _run_flow(G, s, t, capacity, "edmonds_karp", value_only, residual, cutoff)
    return _residual_graph(G, run, value, "edmonds_karp")


def shortest_augmenting_path(
    G, s, t, capacity="capacity", residual=None, value_only=False, two_phase=False, cutoff=None
):
    run, value = _run_flow(
        G, s, t, capacity, "shortest_augmenting_path", value_only, residual, cutoff, two_phase
    )
    return _residual_graph(G, run, value, "shortest_augmenting_path")


def dinitz(G, s, t, capacity="capacity", residual=None, value_only=False, cutoff=None):
    run, value = _run_flow(G, s, t, capacity, "dinitz", value_only, residual, cutoff)
    return _residual_graph(G, run, value, "dinitz")


def boykov_kolmogorov(G, s, t, capacity="capacity", residual=None, value_only=False, cutoff=None):
    run, value = _run_flow(G, s, t, capacity, "boykov_kolmogorov", value_only, residual, cutoff)
    return _residual_graph(G, run, value, "boykov_kolmogorov")


def preflow_push(G, s, t, capacity="capacity", residual=None, global_relabel_freq=1, value_only=False):
    run, value = _run_flow(
        G, s, t, capacity, "preflow_push", value_only, residual,
        global_relabel_freq=global_relabel_freq,
    )
    return _residual_graph(G, run, value, "preflow_push")


def _pick_flow_func(flow_func, kwargs, cut=False):
    """``maximum_flow``'s checks of ``flow_func``; the flow function's name."""
    if flow_func is None:
        if kwargs:
            raise nx.NetworkXError(
                "You have to explicitly set a flow_func if you need to pass parameters via kwargs."
            )
        flow_func = _registered("preflow_push")
    if not callable(flow_func):
        raise nx.NetworkXError("flow_func has to be callable.")
    if cut and kwargs.get("cutoff") is not None and flow_func is _registered("preflow_push"):
        raise nx.NetworkXError("cutoff should not be specified.")
    name = _flow_func_name(flow_func)
    if name is None or not set(kwargs) <= _FLOW_KWARGS[name]:
        raise NotImplementedError("rustnx only runs NetworkX's own flow functions")
    return name


def maximum_flow(flowG, _s, _t, capacity="capacity", flow_func=None, **kwargs):
    name = _pick_flow_func(flow_func, kwargs)
    run, value = _run_flow(flowG, _s, _t, capacity, name, False, **kwargs)
    return value, run.flow_dict(_node_list(flowG))


def maximum_flow_value(flowG, _s, _t, capacity="capacity", flow_func=None, **kwargs):
    name = _pick_flow_func(flow_func, kwargs)
    return _run_flow(flowG, _s, _t, capacity, name, True, **kwargs)[1]


def minimum_cut(flowG, _s, _t, capacity="capacity", flow_func=None, **kwargs):
    name = _pick_flow_func(flow_func, kwargs, cut=True)
    run, value = _run_flow(flowG, _s, _t, capacity, name, True, **kwargs)
    nodes = flowG._nodes
    strict = _cut_by_layers()
    order = run.cut_order(flowG._index[_t], strict)
    # The sink itself is the caller's object, the rest R's nodes.
    if strict:
        non_reachable = {_t}
        add = non_reachable.add
        for i in order[1:]:
            add(nodes[i])
    else:
        found = {_t: None}
        for i in order[1:]:
            found[nodes[i]] = None
        non_reachable = set(found)
    partition = (set(nodes) - non_reachable, non_reachable)
    return value, partition


def minimum_cut_value(flowG, _s, _t, capacity="capacity", flow_func=None, **kwargs):
    name = _pick_flow_func(flow_func, kwargs, cut=True)
    return _run_flow(flowG, _s, _t, capacity, name, True, **kwargs)[1]


def gomory_hu_tree(G, capacity="capacity", flow_func=None):
    _undirected_only(G)
    if flow_func is None:
        flow_func = _registered("edmonds_karp")
    if len(G) == 0:
        raise nx.NetworkXError("Empty Graph does not have a Gomory-Hu tree representation")
    name = _flow_func_name(flow_func)
    if name is None:
        raise NotImplementedError("rustnx only runs NetworkX's own flow functions")
    run = _flow_network(G, capacity)
    n, m = len(G), run.edge_count()
    params = {}
    if name == "preflow_push":
        if not _sets_replayable():
            raise NotImplementedError("this interpreter's sets can't be replayed")
        params = {"threshold": (n + m) / 1, "hashes": [hash(v) for v in G._nodes]}
    elif name == "shortest_augmenting_path":
        params = {"d": n}
    tree = run.gomory_hu(name, _cut_by_layers(), **params)
    nodes = G._nodes
    T = nx.Graph()
    T.add_nodes_from(nodes)
    T.add_weighted_edges_from((nodes[u], nodes[p], w) for u, (p, w) in enumerate(tree, 1))
    return T


def _network_simplex(G, demand, capacity, weight, overrides=(), as_directed=False):
    """``network_simplex``'s ``(flow_cost, flow_dict)`` after its first checks."""
    H = _networkx_graph(G)
    nodes = _node_list(G)
    code, a, b = G._core.network_simplex(
        list(H._adj.values()), list(H._node.values()), nodes, demand, capacity, weight,
        list(overrides), as_directed, _simplex_single_demand(), _COMPENSATED_SUM,
    )
    if code == 0:
        return a, b
    if code == 1:
        raise nx.NetworkXError(f"node {nodes[a]!r} has infinite demand")
    e = (nodes[a], nodes[b])
    if code == 2:
        raise nx.NetworkXError(f"edge {e!r} has infinite weight")
    if code == 3:
        raise nx.NetworkXUnfeasible("total node demand is not zero")
    if code == 4:
        raise nx.NetworkXUnfeasible(f"edge {e!r} has negative capacity")
    if code == 5:
        raise nx.NetworkXUnfeasible("no flow satisfies all node demands")
    raise nx.NetworkXUnbounded("negative cycle with infinite capacity found")


def network_simplex(G, demand="demand", capacity="capacity", weight="weight"):
    _directed_only(G)
    if len(G) == 0:
        raise nx.NetworkXError("graph has no nodes")
    return _network_simplex(G, demand, capacity, weight)


def min_cost_flow_cost(G, demand="demand", capacity="capacity", weight="weight"):
    return network_simplex(G, demand, capacity, weight)[0]


def min_cost_flow(G, demand="demand", capacity="capacity", weight="weight"):
    return network_simplex(G, demand, capacity, weight)[1]


def max_flow_min_cost(G, s, t, capacity="capacity", weight="weight"):
    max_flow = maximum_flow_value(G, s, t, capacity=capacity)
    # nx.DiGraph(G) with demands -max_flow at s and max_flow at t.
    index = G._index
    overrides = [(index[s], -max_flow), (index[t], max_flow)]
    return _network_simplex(G, "demand", capacity, weight, overrides, as_directed=True)[1]


def cost_of_flow(G, flowDict, weight="weight"):
    if callable(weight):
        raise NotImplementedError("rustnx does not support callable weights")
    return G._core.cost_of_flow(_flow_rows(G), _node_list(G), flowDict, weight, _COMPENSATED_SUM)


def _cut_weight(G, weight):
    """The weight attribute for the cut measures (``None``: each edge is 1)
    and, if it mixes ints and floats, the adjacency rows to read it from."""
    if weight is None:
        return None, None
    if not isinstance(weight, str):
        raise NotImplementedError("rustnx needs an edge attribute name for weight")
    G._ensure_weight(weight)
    return weight, (_flow_rows(G) if G._core.weight_mixed(weight) else None)


def _reusable_sets(*args):
    # A one-shot iterator would be consumed before rustnx could fall back.
    from collections.abc import Iterator

    if any(isinstance(a, Iterator) for a in args):
        raise NotImplementedError("rustnx needs reusable node containers")


def _nset_positions(G, nbunch):
    """``{n for n in nbunch if n in G}`` as positions, in the set's order."""
    index = G._index
    return [index[v] for v in {n for n in nbunch if n in G}]


def _set_positions(G, nbunch):
    """``set(nbunch)``'s nodes of G as positions."""
    index = G._index
    return [index[v] for v in set(nbunch) if v in G]


def cut_size(G, S, T=None, weight=None):
    _reusable_sets(S, T)
    weight, rows = _cut_weight(G, weight)
    if G.is_directed() and T is None:
        raise NotImplementedError("NetworkX fails with T=None on directed graphs")
    parts = [(_nset_positions(G, S), None if T is None else _set_positions(G, T))]
    if G.is_directed():
        parts.append((_nset_positions(G, T), _set_positions(G, S)))
    return G._core.cut_size_value(parts, _COMPENSATED_SUM, weight, rows)


def volume(G, S, weight=None):
    _reusable_sets(S)
    weight, rows = _cut_weight(G, weight)
    if S in G:
        raise NotImplementedError("NetworkX fails on a single node here")
    index = G._index
    try:
        positions = [index[n] for n in S if n in index]  # nbunch_iter
    except TypeError:
        raise NotImplementedError("NetworkX raises for this nbunch") from None
    return G._core.volume_value(positions, _COMPENSATED_SUM, weight, rows)


def normalized_cut_size(G, S, T=None, weight=None):
    _reusable_sets(S, T)
    if T is None:
        T = set(G._nodes) - set(S)
    num_cut_edges = cut_size(G, S, T=T, weight=weight)
    volume_S = volume(G, S, weight=weight)
    volume_T = volume(G, T, weight=weight)
    return num_cut_edges * (1 / volume_S + 1 / volume_T)


def conductance(G, S, T=None, weight=None):
    _reusable_sets(S, T)
    if T is None:
        T = set(G._nodes) - set(S)
    num_cut_edges = cut_size(G, S, T, weight=weight)
    volume_S = volume(G, S, weight=weight)
    volume_T = volume(G, T, weight=weight)
    return num_cut_edges / min(volume_S, volume_T)


def edge_expansion(G, S, T=None, weight=None):
    _reusable_sets(S, T)
    if T is None:
        T = set(G._nodes) - set(S)
    num_cut_edges = cut_size(G, S, T=T, weight=weight)
    return num_cut_edges / min(len(S), len(T))


def mixing_expansion(G, S, T=None, weight=None):
    _reusable_sets(S, T)
    num_cut_edges = cut_size(G, S, T=T, weight=weight)
    num_total_edges = G.number_of_edges()
    return num_cut_edges / (2 * num_total_edges)


def node_expansion(G, S):
    _reusable_sets(S)
    index = G._index
    try:
        positions = [index[v] for v in S]
    except (KeyError, TypeError):
        raise NotImplementedError("NetworkX raises for nodes not in G") from None
    union, _ = G._core.neighborhood_sizes(positions)
    return union / len(S)


def boundary_expansion(G, S):
    _reusable_sets(S)
    _, outside = G._core.neighborhood_sizes(_nset_positions(G, S))
    return outside / len(S)


# --- Batch 13: connectivity, disjoint paths and augmentation -----------------------
#
# NetworkX runs Edmonds-Karp on auxiliary digraphs (node-split for node
# connectivity) and their residual networks; the Rust core replays those
# networks in NetworkX's insertion order, so flows, disjoint paths and cuts
# come out the same. Python rebuilds the returned sets with NetworkX's own
# sequence of set operations, so their iteration order matches too.

import itertools  # noqa: E402


def _b13_positions(G, *nodes):
    """Positions of nodes NetworkX looks up in a dict (None if missing).
    Declines unhashable nodes: NetworkX's errors for those vary."""
    index = G._index
    out = []
    for node in nodes:
        try:
            hash(node)
        except TypeError:
            raise NotImplementedError("rustnx needs hashable nodes here") from None
        out.append(index.get(node))
    return out


def _b13_cutoff(cutoff):
    if cutoff is None:
        return None
    if not isinstance(cutoff, (int, float)):
        raise NotImplementedError("rustnx needs an int or float cutoff")
    return float(cutoff)


def _b13_source(func):
    try:
        return inspect.getsource(func.orig_func)
    except (AttributeError, OSError, TypeError):
        return ""


@functools.cache
def _b13_cut_style():
    """How the installed NetworkX's ``minimum_cut`` builds the sink side:
    ``(requeues, incremental)``, or None if unrecognized. 3.4 to 3.6 remove
    the saturated arcs and add them back, which moves them to the end of a
    shared residual network's rows, and build the set from a dict; 3.7
    adds the nodes to a set one at a time."""
    from networkx.algorithms.flow import maxflow

    source = _b13_source(maxflow.minimum_cut)
    requeues = "R.remove_edges_from(cutset)" in source and "R.add_edges_from(cutset)" in source
    if "non_reachable = {_t}" in source and not requeues:
        return False, True
    if requeues and "shortest_path_length(R, target=_t)" in source:
        return True, False
    return None


@functools.cache
def _b13_node_cut_style():
    """``minimum_st_node_cut``'s answer for adjacent nodes: ``(checks both
    directions, returns {} rather than set())``. 3.7 checks only s -> t;
    3.4 returns a dict."""
    source = _b13_source(nx.algorithms.connectivity.minimum_st_node_cut)
    return "G.has_edge(t, s)" in source, "return {}" in source


@functools.cache
def _b13_node_cut_search():
    """``minimum_node_cut``'s search: "isolating" (3.7: start from the
    smallest isolating cut, try both directions) or "degree" (before)."""
    source = _b13_source(nx.algorithms.connectivity.minimum_node_cut)
    if "isolating_cut" in source and "ordered_pairs" in source:
        return "isolating"
    if "min_cut = set(G[v])" in source:
        return "degree"
    return None


@functools.cache
def _b13_isolating_connectivity():
    """Whether ``node_connectivity`` starts from the smallest isolating cut
    (3.7) rather than the smallest degree (before)."""
    source = _b13_source(nx.algorithms.connectivity.node_connectivity)
    if "def degree(v)" in source and "ordered_pairs" in source:
        return True
    if "min(G.degree(), key=itemgetter(1))" in source:
        return False
    return None


@functools.lru_cache(maxsize=4)
def _b13_split_names(n):
    """Node names of the node-split auxiliary digraph (``iA``, ``iB``), and
    each name's position."""
    names = [f"{i}{side}" for i in range(n) for side in "AB"]
    return names, {name: i for i, name in enumerate(names)}


def _b13_cutset(names, sink, us, offsets, vs):
    """``minimum_st_edge_cut``'s set: ``minimum_cut``'s partition, then the
    edges from each reachable node (in set order) to the sink side."""
    style = _b13_cut_style()
    if style is None:
        raise NotImplementedError("unrecognized NetworkX minimum_cut")
    sink_nodes = [names[i] for i in sink]
    if style[1]:
        non_reachable = set(sink_nodes)  # {_t}, then one add per node
    else:
        non_reachable = set(dict.fromkeys(sink_nodes))
    reachable = set(names) - non_reachable
    targets = {
        names[u]: [names[v] for v in vs[offsets[k] : offsets[k + 1]]]
        for k, u in enumerate(us)
    }
    cutset = set()
    for u in reachable:
        out = targets.get(u)
        if out:
            cutset.update([(u, v) for v in out])
    return cutset


def _b13_node_cut(G, cut, s, t):
    """``minimum_st_node_cut``'s result from a node-split cut."""
    names, position = _b13_split_names(len(G))
    edge_cut = _b13_cutset(names, *cut)
    nodes = G._nodes
    node_cut = {nodes[position[node] // 2] for edge in edge_cut for node in edge}
    return node_cut - {s, t}


def _b13_connected(G):
    if G.is_directed():
        return is_weakly_connected(G)
    return is_connected(G)


def local_node_connectivity(G, s, t, cutoff=None):
    si, ti = _b13_positions(G, s, t)
    # f'{mapping[s]}B': a KeyError for a node not in G.
    if si is None:
        raise KeyError(s)
    if ti is None:
        raise KeyError(t)
    return G._core.conn_local_flow(True, si, ti, _b13_cutoff(cutoff))


def local_edge_connectivity(G, s, t, cutoff=None):
    si, ti = _b13_positions(G, s, t)
    cutoff = _b13_cutoff(cutoff)
    if si is None:
        raise nx.NetworkXError(f"node {str(s)} not in graph")
    if ti is None:
        raise nx.NetworkXError(f"node {str(t)} not in graph")
    if si == ti:
        raise nx.NetworkXError("source and sink are the same node")
    return G._core.conn_local_flow(False, si, ti, cutoff)


def _b13_both_or_neither(s, t):
    if (s is not None and t is None) or (s is None and t is not None):
        raise nx.NetworkXError("Both source and target must be specified.")


def _b13_given_pair(G, s, t):
    si, ti = _b13_positions(G, s, t)
    if si is None:
        raise nx.NetworkXError(f"node {s} not in graph")
    if ti is None:
        raise nx.NetworkXError(f"node {t} not in graph")
    return si, ti


def node_connectivity(G, s=None, t=None):
    _b13_both_or_neither(s, t)
    if s is not None and t is not None:
        si, ti = _b13_given_pair(G, s, t)
        return G._core.conn_local_flow(True, si, ti)
    isolating = _b13_isolating_connectivity()
    if isolating is None:
        raise NotImplementedError("unrecognized NetworkX node_connectivity")
    if not _b13_connected(G):
        return 0
    return G._core.conn_node_connectivity(isolating)


def _b13_edge_connectivity(G, cutoff):
    """``edge_connectivity(G, cutoff=cutoff)`` without source and target."""
    if not _b13_connected(G):
        return 0
    L = min(G._core.degrees())
    if cutoff is not None:
        L = min(cutoff, L)
    if not G.is_directed() and G._core.has_self_loops():
        # Self-loops inflate the degrees NetworkX's dominating-set shortcut
        # relies on, so its answer can depend on the set it picks.
        raise NotImplementedError("rustnx declines self-loops here")
    if G.is_directed() and len(G) == 1:
        # The cycle through the nodes asks for a flow from the node to itself.
        raise nx.NetworkXError("source and sink are the same node")
    if not isinstance(L, (int, float)):
        raise NotImplementedError("rustnx needs an int or float cutoff")
    # Each local run either finishes below the current minimum (its exact
    # value) or reaches it, so the result is the minimum found if that is
    # smaller than L, and L itself (as given) otherwise.
    found = G._core.conn_edge_connectivity(float(L))
    if found is not None and found < L:
        return found
    return L


def edge_connectivity(G, s=None, t=None, cutoff=None):
    _b13_both_or_neither(s, t)
    if s is not None and t is not None:
        _b13_given_pair(G, s, t)
        return local_edge_connectivity(G, s, t, cutoff=cutoff)
    return _b13_edge_connectivity(G, cutoff)


def _b13_pairs(items, directed):
    return list(itertools.permutations(items, 2) if directed else itertools.combinations(items, 2))


def average_node_connectivity(G):
    pairs = _b13_pairs(range(len(G)), G.is_directed())
    if not pairs:
        return 0
    values = G._core.conn_pair_flows(True, [u for u, _ in pairs], [v for _, v in pairs])
    return sum(values) / len(pairs)


def all_pairs_node_connectivity(G, nbunch=None):
    if nbunch is None:
        nbunch = G._nodes
    else:
        nbunch = set(nbunch)
    directed = G.is_directed()
    all_pairs = {n: {} for n in nbunch}
    order = list(nbunch)
    index = G._index
    positions = [index.get(n) for n in order]
    if len(order) >= 2 and None in positions:
        # local_node_connectivity's f'{mapping[s]}B' fails at the first pair
        # with a missing node: (order[0], order[m]) for the first missing m.
        m = positions.index(None)
        raise KeyError(order[m])
    pairs = _b13_pairs(range(len(order)), directed)
    values = G._core.conn_pair_flows(
        True, [positions[i] for i, _ in pairs], [positions[j] for _, j in pairs]
    )
    for (i, j), K in zip(pairs, values):
        u, v = order[i], order[j]
        all_pairs[u][v] = K
        if not directed:
            all_pairs[v][u] = K
    return all_pairs


def minimum_st_edge_cut(G, s, t):
    si, ti = _b13_positions(G, s, t)
    if _b13_cut_style() is None:
        raise NotImplementedError("unrecognized NetworkX minimum_cut")
    if si is None:
        raise nx.NetworkXError(f"node {str(s)} not in graph")
    if ti is None:
        raise nx.NetworkXError(f"node {str(t)} not in graph")
    if si == ti:
        raise nx.NetworkXError("source and sink are the same node")
    _, *cut = G._core.conn_st_cut(False, si, ti, True)
    return _b13_cutset(G._nodes, *cut)


def minimum_st_node_cut(G, s, t):
    if _b13_cut_style() is None:
        raise NotImplementedError("unrecognized NetworkX minimum_cut")
    both, empty_dict = _b13_node_cut_style()
    si, ti = _b13_positions(G, s, t)
    core = G._core
    if si is not None and ti is not None:
        if core.has_edge(si, ti) or (both and core.has_edge(ti, si)):
            return {} if empty_dict else set()
    if si is None:
        raise KeyError(s)
    if ti is None:
        raise KeyError(t)
    _, *cut = core.conn_st_cut(True, si, ti, False)
    return _b13_node_cut(G, cut, s, t)


def _b13_neighbors(G, v):
    """``neighbors(v)`` of the cut searches: predecessors (in ``G.pred``
    order) then successors for directed graphs, as positions."""
    core = G._core
    if G.is_directed():
        return core.conn_exact_predecessors(v) + core.neighbors(v)
    return core.neighbors(v)


def minimum_node_cut(G, s=None, t=None):
    _b13_both_or_neither(s, t)
    if s is not None and t is not None:
        _b13_given_pair(G, s, t)
        return minimum_st_node_cut(G, s, t)
    style = _b13_cut_style()
    search = _b13_node_cut_search()
    if style is None or search is None:
        raise NotImplementedError("unrecognized NetworkX minimum_node_cut")
    directed = G.is_directed()
    if not _b13_connected(G):
        raise nx.NetworkXError("Input graph is not connected")
    if directed:
        G._ensure_exact_pred()
    core = G._core
    nodes = G._nodes
    index = G._index
    both, empty_dict = _b13_node_cut_style()
    ss, ts, skips = [], [], []

    def add(x, y, skip):
        ss.append(x)
        ts.append(y)
        skips.append(skip)

    if search == "degree":
        degrees = core.degrees()
        v = min(range(len(nodes)), key=degrees.__getitem__)
        vnode = nodes[v]
        min_cut = set(nodes[w] for w in core.neighbors(v))
        nbrs = _b13_neighbors(G, v)
        for w in set(nodes) - set(nodes[x] for x in nbrs) - {vnode}:
            add(v, index[w], False)
        for x, y in _b13_pairs(nbrs, directed):
            add(x, y, True)
    else:
        v, _ = core.conn_isolating_cut()
        vnode = nodes[v]
        succ = [nodes[w] for w in core.neighbors(v)]
        if directed:
            pred = [nodes[w] for w in core.conn_exact_predecessors(v)]
            # G.pred[v].keys() - {v}: a KeysView difference, built one
            # element at a time.
            min_cut = min(
                set(w for w in pred if w not in {vnode}),
                set(w for w in succ if w not in {vnode}),
                key=len,
            )
        else:
            min_cut = set(w for w in succ if w not in {vnode})
        v_nbrs = set(nodes[x] for x in _b13_neighbors(G, v)) - {vnode}
        for w in set(nodes) - v_nbrs - {vnode}:
            add(v, index[w], False)
            if directed:
                add(index[w], v, False)
        for x, y in _b13_pairs([index[w] for w in v_nbrs], directed):
            add(x, y, True)
    chosen = core.conn_search_cuts(True, ss, ts, skips, len(min_cut), both, style[0])
    if chosen is None:
        return min_cut
    i, cut = chosen
    if cut is None:
        return {} if empty_dict else set()
    return _b13_node_cut(G, cut, nodes[ss[i]], nodes[ts[i]])


def _b13_dominating_set(G, start):
    """``nx.dominating_set(G, start_with=nodes[start])``, set operations and
    all (the order it pops nodes in follows set order)."""
    nodes = G._nodes
    core = G._core
    index = G._index
    all_nodes = set(nodes)
    dominating_set = {nodes[start]}
    dominated_nodes = set(nodes[w] for w in core.neighbors(start))
    remaining_nodes = all_nodes - dominated_nodes - dominating_set
    while remaining_nodes:
        v = remaining_nodes.pop()
        undominated_nbrs = set(nodes[w] for w in core.neighbors(index[v])) - dominating_set
        dominating_set.add(v)
        dominated_nodes |= undominated_nbrs
        remaining_nodes -= undominated_nbrs
    return dominating_set


def minimum_edge_cut(G, s=None, t=None):
    _b13_both_or_neither(s, t)
    style = _b13_cut_style()
    if style is None:
        raise NotImplementedError("unrecognized NetworkX minimum_cut")
    core = G._core
    nodes = G._nodes
    if s is not None and t is not None:
        si, ti = _b13_given_pair(G, s, t)
        if si == ti:
            raise nx.NetworkXError("source and sink are the same node")
        _, *cut = core.conn_st_cut(False, si, ti, False)
        return _b13_cutset(nodes, *cut)
    if not _b13_connected(G):
        raise nx.NetworkXError("Input graph is not connected")
    degrees = core.degrees()
    node = min(range(len(nodes)), key=degrees.__getitem__)
    min_cut = set((nodes[node], nodes[w]) for w in core.neighbors(node))
    n = len(nodes)
    if G.is_directed():
        if n == 1:
            raise nx.NetworkXError("source and sink are the same node")
        ss = list(range(n))
        ts = [(i + 1) % n for i in range(n)]
    else:
        index = G._index
        for start in range(n):
            if sum(1 for w in core.neighbors(start) if w != start) < n - 1:
                break
        else:
            return min_cut
        D = _b13_dominating_set(G, start)
        v = index[D.pop()]
        ts = [index[w] for w in D]
        ss = [v] * len(ts)
    chosen = core.conn_search_cuts(False, ss, ts, [False] * len(ss), len(min_cut), False, style[0])
    if chosen is None:
        return min_cut
    return _b13_cutset(nodes, *chosen[1])


def _b13_disjoint_paths(G, s, t, cutoff, node_split, fallback):
    _b13_positions(G, s, t)
    cutoff = _b13_cutoff(cutoff)

    def compute():
        si, ti = _b13_given_pair(G, s, t)
        status, paths = G._core.conn_disjoint_paths(node_split, si, ti, cutoff)
        if status == 1:
            raise nx.NetworkXNoPath
        if status == 2:
            raise nx.NetworkXError("source and sink are the same node")
        nodes = G._nodes
        for path in paths:
            out = [nodes[i] for i in path]
            if not node_split:
                # edge_disjoint_paths starts each path with s as given, and
                # ends it with t as given unless t was s's direct successor.
                out[0] = s
                if len(out) > 2:
                    out[-1] = t
            yield out

    return _computed_on_first_next(G, compute, fallback)


def edge_disjoint_paths(G, s, t, cutoff=None):
    return _b13_disjoint_paths(
        G, s, t, cutoff, False,
        lambda H: nx.algorithms.connectivity.edge_disjoint_paths(
            H, s, t, cutoff=cutoff, backend="networkx"
        ),
    )


def node_disjoint_paths(G, s, t, cutoff=None):
    return _b13_disjoint_paths(
        G, s, t, cutoff, True,
        lambda H: nx.algorithms.connectivity.node_disjoint_paths(
            H, s, t, cutoff=cutoff, backend="networkx"
        ),
    )


def stoer_wagner(G, weight="weight"):
    _undirected_only(G)
    n = len(G)
    if n < 2:
        raise nx.NetworkXError("graph has less than two nodes.")
    if not is_connected(G):
        raise nx.NetworkXError("graph is not connected.")
    weight, all_int, has_hidden = _check_weight(G, weight)
    if weight is not None and (has_hidden or G._core.weight_mixed(weight)):
        raise NotImplementedError("rustnx needs all-int or all-float weights here")
    status, cut_value, order, reachable = G._core.conn_stoer_wagner(weight)
    if status == 1:
        raise nx.NetworkXError("graph has a negative-weighted edge.")
    if status == 2:
        raise NotImplementedError("infinite or very large weights")
    nodes = G._nodes
    if all_int:
        cut_value = int(cut_value)
    # nodes = set(G) of the rebuilt graph; reachable = set(dict of a BFS).
    node_set = set(nodes[i] for i in order)
    reachable = set(dict.fromkeys(nodes[i] for i in reachable))
    return cut_value, (list(reachable), list(node_set - reachable))


def bridge_components(G):
    _undirected_only(G)

    def compute():
        nodes = G._nodes
        for comp in G._core.conn_bridge_components():
            yield {nodes[i] for i in comp}

    return _computed_on_first_next(
        G, compute, lambda H: nx.algorithms.connectivity.bridge_components(H, backend="networkx")
    )


def k_edge_components(G, k):
    if k < 1:
        raise ValueError("k cannot be less than 1")
    if G.is_directed():
        if k == 1:
            return strongly_connected_components(G)
    elif k == 1:
        return connected_components(G)
    elif k == 2:
        return bridge_components(G)
    # The auxiliary graph picks its cuts with set order and preflow-push.
    raise NotImplementedError("rustnx only does k = 1 and 2 (undirected) here")


def k_edge_subgraphs(G, k):
    if k < 1:
        raise ValueError("k cannot be less than 1")
    if k <= (1 if G.is_directed() else 2):
        return k_edge_components(G, k)
    # general_k_edge_subgraphs pops graphs from a set (address order).
    raise NotImplementedError("rustnx only does k = 1 and 2 (undirected) here")


def is_k_edge_connected(G, k):
    _undirected_only(G)
    if k < 1:
        raise ValueError(f"k must be positive, not {k}")
    if len(G) < k + 1:
        return False
    elif any(d < k for d in G._core.degrees()):
        return False
    elif k == 1:
        return is_connected(G)
    elif k == 2:
        return is_connected(G) and not has_bridges(G)
    else:
        return _b13_edge_connectivity(G, k) >= k


def is_locally_k_edge_connected(G, s, t, k):
    _undirected_only(G)
    if k < 1:
        raise ValueError(f"k must be positive, not {k}")
    si, ti = _b13_positions(G, s, t)
    if si is None or ti is None:
        # G.degree(missing) is a degree view, not a number.
        raise NotImplementedError("rustnx needs nodes of the graph here")
    core = G._core
    if core.degree_of(si) < k or core.degree_of(ti) < k:
        return False
    elif k == 1:
        return has_path(G, s, t)
    else:
        return local_edge_connectivity(G, s, t, cutoff=k) >= k


def _b13_complement_edges(G):
    """``complement_edges(G)`` for the tiny graphs ``k_edge_augmentation``
    hands it (fewer than k + 1 nodes). The public function stays in
    NetworkX: its loop is cheaper than yielding pairs built from Rust."""
    nodes = G._nodes
    flat = G._core.conn_complement_edges(0, len(nodes))
    return list(zip([nodes[i] for i in flat[0::2]], [nodes[i] for i in flat[1::2]]))


def unconstrained_one_edge_augmentation(G):
    def compute():
        if G.is_directed():
            raise nx.NetworkXNotImplemented("not implemented for directed type")
        nodes = G._nodes
        # collapse() maps each component's nodes in the order of set(cc).
        firsts = [
            next(iter(set({nodes[i] for i in comp})))
            for comp in G._core.connected_components()
        ]
        yield from zip(firsts, firsts[1:])

    return _computed_on_first_next(
        G,
        compute,
        lambda H: nx.algorithms.connectivity.edge_augmentation.unconstrained_one_edge_augmentation(
            H, backend="networkx"
        ),
    )


def one_edge_augmentation(G, avail=None, weight=None, partial=False):
    _undirected_only(G)
    if avail is not None:
        raise NotImplementedError("rustnx does not support avail here")
    return unconstrained_one_edge_augmentation(G)


def unconstrained_bridge_augmentation(G):
    def compute():
        if G.is_directed():
            raise nx.NetworkXNotImplemented("not implemented for directed type")
        nodes = G._nodes
        core = G._core
        index = G._index
        comps = core.conn_bridge_components()
        comp_of = [0] * len(nodes)
        for c, comp in enumerate(comps):
            for i in comp:
                comp_of[i] = c
        # collapse(G, bridge_ccs): one meta node per component (one even
        # without nodes), and the bridges between them in G.edges order.
        C = nx.Graph()
        C.add_nodes_from(range(max(len(comps), 1)))
        us, vs = core.bridges()
        C.add_edges_from((comp_of[u], comp_of[v]) for u, v in zip(us, vs))
        # From networkx.algorithms.connectivity.edge_augmentation.
        vset1 = [
            tuple(cc) * 2 if len(cc) == 1 else sorted(cc, key=C.degree)[0:2]
            for cc in nx.connected_components(C, backend="networkx")
        ]
        if len(vset1) > 1:
            nodes1 = [vs[0] for vs in vset1]
            nodes2 = [vs[1] for vs in vset1]
            A1 = list(zip(nodes1[1:], nodes2))
        else:
            A1 = []
        T = C.copy()
        T.add_edges_from(A1)
        leafs = [n for n, d in T.degree() if d == 1]
        if len(leafs) == 1:
            A2 = []
        if len(leafs) == 2:
            A2 = [tuple(leafs)]
        else:
            try:
                root = next(n for n, d in T.degree() if d > 1)
            except StopIteration:
                return
            v2 = [
                n for n in nx.dfs_preorder_nodes(T, root, backend="networkx")
                if T.degree(n) == 1
            ]
            half = math.ceil(len(v2) / 2)
            A2 = list(zip(v2[:half], v2[-half:]))
        aug_tree_edges = A1 + A2
        degrees = core.degrees()
        # inverse[mu]: list(set(bridge_ccs[mu])), sorted by (degree, node).
        inverse = {
            mu: sorted(set({nodes[i] for i in comp}), key=lambda u: (degrees[index[u]], u))
            for mu, comp in enumerate(comps)
        }
        added = set()  # edges G2 gained
        for mu, mv in aug_tree_edges:
            for u, v in itertools.product(inverse[mu], inverse[mv]):
                if not ((u, v) in added or core.has_edge(index[u], index[v])):
                    added.add((u, v))
                    added.add((v, u))
                    yield (u, v)
                    break

    return _computed_on_first_next(
        G,
        compute,
        lambda H: nx.algorithms.connectivity.edge_augmentation.unconstrained_bridge_augmentation(
            H, backend="networkx"
        ),
    )


def bridge_augmentation(G, avail=None, weight=None):
    _undirected_only(G)
    if len(G) < 3:
        raise nx.NetworkXUnfeasible("impossible to bridge connect less than 3 nodes")
    if avail is not None:
        raise NotImplementedError("rustnx does not support avail here")
    return unconstrained_bridge_augmentation(G)


def k_edge_augmentation(G, k, avail=None, weight=None, partial=False):
    _undirected_only(G)
    if avail is not None:
        raise NotImplementedError("rustnx does not support avail here")
    if type(k) is not int or k >= 3:
        # k >= 3 is a greedy search with a seeded shuffle.
        raise NotImplementedError("rustnx only does k = 1 and 2 here")

    def compute():
        try:
            if k <= 0:
                raise ValueError(f"k must be a positive integer, not {k}")
            elif len(G) < k + 1:
                msg = f"impossible to {k} connect in graph with less than {k + 1} nodes"
                raise nx.NetworkXUnfeasible(msg)
            elif k == 1:
                aug_edges = one_edge_augmentation(G)
            else:
                aug_edges = bridge_augmentation(G)
            yield from list(aug_edges)
        except nx.NetworkXUnfeasible:
            if partial:
                yield from _b13_complement_edges(G)
            else:
                raise

    return _computed_on_first_next(
        G,
        compute,
        lambda H: nx.algorithms.connectivity.k_edge_augmentation(
            H, k, avail=avail, weight=weight, partial=partial, backend="networkx"
        ),
    )


# --- Batch 14: assortativity, link prediction and reciprocity -----------------------


def _b14_int_weight(G, weight):
    """A degree weight rustnx can sum: integer values only (summing floats
    would have to follow Python's mixed int/float ``sum()`` exactly)."""
    if weight is None:
        return None
    attr, all_int, has_hidden = _check_weight(G, weight)
    if not all_int or has_hidden:
        raise NotImplementedError("rustnx sums integer edge weights only here")
    return attr


def _b14_reusable(nodes):
    """Decline a one-shot iterator of nodes: NetworkX reads ``nodes`` more
    than once, and must be able to take over."""
    if nodes is not None and iter(nodes) is nodes:
        raise NotImplementedError("rustnx needs a reusable container of nodes")


def _b14_nbunch_list(G, nodes):
    """Positions of ``list(G.nbunch_iter(nodes))``, repeats kept."""
    try:
        _b14_reusable(nodes)
        index = G._index
        return [index[v] for v in nodes if v in index]
    except TypeError:
        raise NotImplementedError("NetworkX raises its own error here") from None


def _b14_node_values(G, attribute):
    """``G.nodes[v].get(attribute)`` for every node, in node order."""
    try:
        hash(attribute)
    except TypeError:
        raise NotImplementedError("unhashable attribute name") from None
    if G._core.is_native():
        return [None] * len(G)  # native graphs have no node data
    if not G._source_unchanged():
        raise NotImplementedError("the graph changed since it was converted")
    return [dd.get(attribute) for dd in G._source._node.values()]


def _b14_lazy(G, compute, fallback):
    """A generator like NetworkX's: it reads the graph when iteration starts
    (if G changed before then, NetworkX's own ``fallback(graph)`` runs) and
    stops loudly if G changes while it yields."""
    guard = _MutationGuard(G)

    def generate():
        try:
            if guard.changed():
                yield from fallback(guard.graph)
                return
            for item in compute():
                if guard.changed():
                    raise RuntimeError("Graph changed during iteration")
                yield item
        finally:
            guard.release()

    return generate()


# Link prediction. NetworkX checks every pair of `ebunch` when called, then
# scores the pairs lazily; rustnx scores them in growing batches in Rust.

_B14_MAX_BATCH = 4096


@functools.cache
def _b14_set_order_matches():
    """Whether rustnx's replay of CPython's set table orders
    ``G._adj[u].keys() & G._adj[v].keys() - {u, v}`` (NetworkX's
    ``common_neighbors``) as this interpreter does. Checked once, on graphs
    whose sets resize, keep dummies and take both intersection paths."""
    from .graph import from_networkx

    rng = random.Random(14)
    for labels in ("int", "str", "mixed"):
        H = nx.Graph()
        names = []
        for i in range(70):
            name = i if labels == "int" or (labels == "mixed" and i % 2) else f"v{i}"
            names.append(name)
        H.add_nodes_from(names)
        hubs = names[:3]
        for name in names:
            for hub in hubs:
                if rng.random() < 0.9:
                    H.add_edge(hub, name)  # some self-loops, too
            for _ in range(rng.randrange(12)):
                H.add_edge(name, rng.choice(names))
        core = from_networkx(H)._core
        index = {v: i for i, v in enumerate(names)}
        pairs = [(a, b) for a in names[:12] for b in names[:24]]
        us = [index[a] for a, _ in pairs]
        vs = [index[b] for _, b in pairs]
        got = core.common_neighbors_in_set_order(us, vs, [hash(v) for v in names])
        for (a, b), order in zip(pairs, got):
            expected = list(H._adj[a].keys() & H._adj[b].keys() - {a, b})
            if [names[i] for i in order] != expected:
                return False
    return True


def _b14_pairs(G, ebunch):
    """NetworkX's ``_apply_prediction``: check every pair of ``ebunch`` now,
    and give the pairs to score (``nx.non_edges(G)`` without an ebunch)."""
    _undirected_only(G)
    if ebunch is None:
        return nx.non_edges(_networkx_graph(G))
    index = G._index
    for u, v in ebunch:
        # `u not in G`: an unhashable node is just not in G.
        try:
            found = u in index
        except TypeError:
            found = False
        if not found:
            raise nx.NodeNotFound(f"Node {u} not in G.")
        try:
            found = v in index
        except TypeError:
            found = False
        if not found:
            raise nx.NodeNotFound(f"Node {v} not in G.")
    return ebunch


def _b14_predictions(G, pairs, score):
    """Yield ``(u, v, score)`` for each pair, scoring growing batches.

    ``score(us, vs)`` takes a batch's node positions and returns ``(values,
    fail)``: the scores up to the first pair whose score raises in
    NetworkX, and ``fail(k, u, v)`` raising that error for pair ``k``
    (``None`` if no pair raises).
    """
    guard = _MutationGuard(G)
    index = G._index

    def generate():
        try:
            it = iter(pairs)
            size = 64
            key, cache, graph = guard._key, guard._cache, guard.graph
            while True:
                batch = list(islice(it, size))
                if not batch:
                    return
                size = min(2 * size, _B14_MAX_BATCH)
                try:
                    us = [index[u] for u, _ in batch]
                    vs = [index[v] for _, v in batch]
                except (KeyError, TypeError):
                    raise RuntimeError("ebunch changed during iteration") from None
                values, fail = score(us, vs)
                if key is None:  # a native graph: it can't change
                    yield from [(u, v, value) for (u, v), value in zip(batch, values)]
                else:
                    for (u, v), value in zip(batch, values):
                        # `guard.changed()`, inlined.
                        if key not in cache or graph.__networkx_cache__ is not cache:
                            raise RuntimeError("Graph changed during iteration")
                        yield (u, v, value)
                if fail is not None:
                    k = len(values)
                    if guard.changed():
                        raise RuntimeError("Graph changed during iteration")
                    fail(k, *batch[k])
        finally:
            guard.release()

    return generate()


# `LinkScorer` outcome codes (see `measures::code` in Rust).
_B14_OK, _B14_INT_ZERO, _B14_NO_U, _B14_NO_V, _B14_NO_W, _B14_ZERO_DIVISION = range(6)
_B14_COMMUNITY_TYPES = (bool, float, int, str, type(None))


def _b14_community_classes(G, community):
    """Each node's community as an integer class (-1 for none). Classes are
    equal exactly when NetworkX's ``==`` says the communities are."""
    try:
        hash(community)
    except TypeError:
        raise NotImplementedError("unhashable community attribute name") from None
    if G._core.is_native():
        return [-1] * len(G)
    if not G._source_unchanged():
        raise NotImplementedError("the graph changed since it was converted")
    classes = {}
    out = []
    missing = object()
    for dd in G._source._node.values():
        c = dd.get(community, missing)
        if c is missing:
            out.append(-1)
            continue
        if type(c) not in _B14_COMMUNITY_TYPES or c != c:
            raise NotImplementedError("rustnx compares plain community values only")
        out.append(classes.setdefault(c, len(classes)))
    return out


def _b14_scorer(G, mode, table=None, community=None):
    """A Rust scorer; set-order-dependent modes need the set replay check."""
    hashes = classes = None
    if mode != 0:
        if not _b14_set_order_matches():
            raise NotImplementedError("this Python orders sets differently")
        hashes = [hash(v) for v in G._nodes]
    if community is not None:
        classes = _b14_community_classes(G, community)
    return G._core.link_scorer(mode, hashes, table, classes, _COMPENSATED_SUM)


def _b14_degree_table(G, term):
    """``term(d)`` per degree ``d`` up to the maximum (NaN where NetworkX's
    expression divides by zero)."""
    top = max(G._core.degrees(), default=0)
    table = []
    for d in range(top + 1):
        try:
            table.append(term(d))
        except (ZeroDivisionError, ValueError):
            table.append(math.nan)
    return table


def _b14_split(G, codes, w):
    """``(n, fail)``: how many pairs score without raising, and the error of
    the next one (see ``_b14_predictions``)."""
    if not codes or max(codes) < _B14_NO_U:
        return len(codes), None
    k = next(i for i, c in enumerate(codes) if c >= _B14_NO_U)
    code = codes[k]

    def fail(k, u, v):
        if code == _B14_NO_U:
            raise nx.NetworkXAlgorithmError(f"No community information available for Node {u}")
        if code == _B14_NO_V:
            raise nx.NetworkXAlgorithmError(f"No community information available for Node {v}")
        if code == _B14_NO_W:
            node = G._nodes[w[k]]
            raise nx.NetworkXAlgorithmError(f"No community information available for Node {node}")
        # `1 / log(G.degree(w))` with a degree-1 neighbor: NetworkX's own error.
        1 / math.log(1)

    return k, fail


def _b14_sums(G, scorer):
    """Scores that are float sums, or the int 0 for an empty one."""
    def score(us, vs):
        codes, _, _, f, w = scorer.scores(us, vs)
        n, fail = _b14_split(G, codes, w)
        return [x if c == _B14_OK else 0 for c, x in zip(codes[:n], f)], fail

    return score


def resource_allocation_index(G, ebunch=None):
    pairs = _b14_pairs(G, ebunch)
    scorer = _b14_scorer(G, 2, _b14_degree_table(G, lambda d: 1 / d))
    return _b14_predictions(G, pairs, _b14_sums(G, scorer))


def adamic_adar_index(G, ebunch=None):
    pairs = _b14_pairs(G, ebunch)
    scorer = _b14_scorer(G, 2, _b14_degree_table(G, lambda d: 1 / math.log(d)))
    return _b14_predictions(G, pairs, _b14_sums(G, scorer))


def ra_index_soundarajan_hopcroft(G, ebunch=None, community="community"):
    pairs = _b14_pairs(G, ebunch)
    scorer = _b14_scorer(G, 2, _b14_degree_table(G, lambda d: 1 / d), community)
    return _b14_predictions(G, pairs, _b14_sums(G, scorer))


def jaccard_coefficient(G, ebunch=None):
    pairs = _b14_pairs(G, ebunch)
    scorer = _b14_scorer(G, 0)

    def score(us, vs):
        _, common, union, _, _ = scorer.scores(us, vs)
        return [c / s if s else 0 for c, s in zip(common, union)], None

    return _b14_predictions(G, pairs, score)


def preferential_attachment(G, ebunch=None):
    pairs = _b14_pairs(G, ebunch)
    degree = G._core.degrees()

    def score(us, vs):
        return [degree[u] * degree[v] for u, v in zip(us, vs)], None

    return _b14_predictions(G, pairs, score)


def cn_soundarajan_hopcroft(G, ebunch=None, community="community"):
    pairs = _b14_pairs(G, ebunch)
    scorer = _b14_scorer(G, 1, community=community)

    def score(us, vs):
        codes, common, same, _, w = scorer.scores(us, vs)
        n, fail = _b14_split(G, codes, w)
        # `len(cnbors) + neighbors`; the bonus is 0 across communities.
        return [c + b for c, b in zip(common[:n], same)], fail

    return _b14_predictions(G, pairs, score)


def within_inter_cluster(G, ebunch=None, delta=0.001, community="community"):
    if delta <= 0:
        raise nx.NetworkXAlgorithmError("Delta must be greater than zero")
    pairs = _b14_pairs(G, ebunch)
    scorer = _b14_scorer(G, 1, community=community)

    def score(us, vs):
        codes, common, within, together, w = scorer.scores(us, vs)
        n, fail = _b14_split(G, codes, w)
        values = [
            within_ / ((c - within_) + delta) if t else 0
            for c, within_, t in zip(common[:n], within, together)
        ]
        return values, fail

    return _b14_predictions(G, pairs, score)


def common_neighbor_centrality(G, ebunch=None, alpha=0.8):
    _undirected_only(G)
    simple = alpha == 1
    pairs = _b14_pairs(G, ebunch)
    scorer = _b14_scorer(G, 0)
    n = len(G)
    inf = float("inf")
    core = G._core

    def self_loop(k, u, v):
        raise nx.NetworkXAlgorithmError("Self loops are not supported")

    def score(us, vs):
        # NetworkX's `u == v` test, on positions (equal nodes share one).
        k = next((i for i, (u, v) in enumerate(zip(us, vs)) if u == v), None)
        if k is not None:
            us, vs = us[:k], vs[:k]
        common = scorer.scores(us, vs)[1]
        if simple:
            values = common
        else:
            dist = core.pair_distances(us, vs)
            values = [
                alpha * c + (1 - alpha) * n / (d if d >= 0 else inf)
                for c, d in zip(common, dist)
            ]
        return values, None if k is None else self_loop

    return _b14_predictions(G, pairs, score)


# Degree and attribute pairs, mixing and assortativity.

_B14_DEGREE_KINDS = {"out": 1, "in": 2}


def _b14_degree_kinds(G, x, y):
    """``node_degree_xy``'s degree kinds (0 ``G.degree``, 1 out, 2 in)."""
    if not G.is_directed():
        return 0, 0  # x and y are ignored
    if type(x) is not str or type(y) is not str or x not in _B14_DEGREE_KINDS or y not in _B14_DEGREE_KINDS:
        raise NotImplementedError("NetworkX raises its own error here")
    return _B14_DEGREE_KINDS[x], _B14_DEGREE_KINDS[y]


def _b14_xy_order(G, nodes):
    """NetworkX's ``set(G)`` or ``set(nodes)``: positions in its iteration
    order (nodes not in G skipped), and its membership per node (``None``
    for all nodes). The set is built the same way, so it iterates alike."""
    index = G._index
    S = set(G._nodes) if nodes is None else set(nodes)
    order = [index[v] for v in S if v in index]
    mask = None if nodes is None else [v in S for v in G._nodes]
    return order, mask


def _b14_degree_pairs(G, x, y, weight):
    """Validate a degree-pairs call; return a function of ``nodes`` giving
    ``(order, mask, xdeg, ydeg)``."""
    kx, ky = _b14_degree_kinds(G, x, y)
    weight = _b14_int_weight(G, weight)

    def prepare(nodes):
        order, mask = _b14_xy_order(G, nodes)
        xdeg = G._core.kind_degrees(kx, weight)
        ydeg = xdeg if ky == kx else G._core.kind_degrees(ky, weight)
        return order, mask, xdeg, ydeg

    return prepare


def node_degree_xy(G, x="out", y="in", weight=None, nodes=None):
    prepare = _b14_degree_pairs(G, x, y, weight)

    def compute():
        order, mask, xdeg, ydeg = prepare(nodes)
        us, vs = G._core.xy_pairs(order, mask)
        return zip(map(xdeg.__getitem__, us), map(ydeg.__getitem__, vs))

    def fallback(H):
        return _registered("node_degree_xy").orig_func(H, x=x, y=y, weight=weight, nodes=nodes)

    return _b14_lazy(G, compute, fallback)


def node_attribute_xy(G, attribute, nodes=None):
    values = _b14_node_values(G, attribute)

    def compute():
        if nodes is None:
            order = range(len(G))
        else:
            S = set(nodes)
            order = [i for i, v in enumerate(G._nodes) if v in S]
        us, vs = G._core.xy_pairs(list(order), None)
        return zip(map(values.__getitem__, us), map(values.__getitem__, vs))

    def fallback(H):
        return _registered("node_attribute_xy").orig_func(H, attribute, nodes=nodes)

    return _b14_lazy(G, compute, fallback)


def _b14_mixing_dict(G, order, mask, xcls, ycls, key, normalized):
    """NetworkX's ``mixing_dict`` over the pairs: the same keys (``key(class,
    node)`` gives the object NetworkX inserted) in the same order."""
    total, rows = G._core.mixing(order, xcls, ycls, mask)
    d = {key(c, node): {} for c, node, _ in rows}
    psum = float(total)
    for (_, _, inner), jdict in zip(rows, d.values()):
        if normalized:
            jdict.update((key(c, node), count / psum) for c, node, count in inner)
        else:
            jdict.update((key(c, node), count) for c, node, count in inner)
    return d


def degree_mixing_dict(G, x="out", y="in", weight=None, nodes=None, normalized=False):
    prepare = _b14_degree_pairs(G, x, y, weight)
    _b14_reusable(nodes)
    order, mask, xdeg, ydeg = prepare(nodes)
    return _b14_mixing_dict(G, order, mask, xdeg, ydeg, lambda c, node: c, normalized)


def _b14_matrix(d, mapping, normalized):
    a = nx.utils.dict_to_numpy_array(d, mapping=mapping)
    if normalized:
        a = a / a.sum()
    return a


def degree_mixing_matrix(G, x="out", y="in", weight=None, nodes=None, normalized=True, mapping=None):
    d = degree_mixing_dict(G, x=x, y=y, nodes=nodes, weight=weight)
    return _b14_matrix(d, mapping, normalized)


def degree_assortativity_coefficient(G, x="out", y="in", weight=None, nodes=None):
    from networkx.algorithms.assortativity.correlation import _numeric_ac

    _b14_degree_kinds(G, x, y)
    attr = _b14_int_weight(G, weight)
    single, positions = _nbunch_positions(G, nodes)
    if single:
        raise NotImplementedError("NetworkX raises its own error here")
    if positions is None:
        positions = range(len(G))
    core = G._core
    # The same insertions as NetworkX's set comprehensions, in node order.
    if G.is_directed():
        indeg = outdeg = set()
        if "in" in (x, y):
            ins = core.kind_degrees(2, attr)
            indeg = {ins[i] for i in positions}
        if "out" in (x, y):
            outs = core.kind_degrees(1, attr)
            outdeg = {outs[i] for i in positions}
        degrees = set.union(indeg, outdeg)
    else:
        deg = core.kind_degrees(0, attr)
        degrees = {deg[i] for i in positions}
    mapping = {d: i for i, d in enumerate(degrees)}
    M = degree_mixing_matrix(G, x=x, y=y, nodes=nodes, weight=weight, mapping=mapping)
    return _numeric_ac(M, mapping=mapping)


def degree_pearson_correlation_coefficient(G, x="out", y="in", weight=None, nodes=None):
    import scipy as sp

    prepare = _b14_degree_pairs(G, x, y, weight)
    _b14_reusable(nodes)
    order, mask, xdeg, ydeg = prepare(nodes)
    us, vs = G._core.xy_pairs(order, mask)
    if len(us) < 2:
        # 3.4 to 3.6 raise (from `zip(*xy)` or SciPy); 3.7 returns NaN.
        raise NotImplementedError("NetworkX versions differ here")
    xs = tuple(map(xdeg.__getitem__, us))
    ys = tuple(map(ydeg.__getitem__, vs))
    return float(sp.stats.pearsonr(xs, ys)[0])


def _b14_attribute_mixing(G, attribute, nodes, normalized):
    values = _b14_node_values(G, attribute)
    _b14_reusable(nodes)
    classes = {}
    try:
        cls = [classes.setdefault(value, len(classes)) for value in values]
    except TypeError:
        raise NotImplementedError("unhashable attribute values") from None
    if nodes is None:
        order = list(range(len(G)))
    else:
        S = set(nodes)
        order = [i for i, v in enumerate(G._nodes) if v in S]
    return _b14_mixing_dict(G, order, None, cls, cls, lambda c, node: values[node], normalized)


def attribute_mixing_dict(G, attribute, nodes=None, normalized=False):
    return _b14_attribute_mixing(G, attribute, nodes, normalized)


def attribute_mixing_matrix(G, attribute, nodes=None, mapping=None, normalized=True):
    d = _b14_attribute_mixing(G, attribute, nodes, False)
    return _b14_matrix(d, mapping, normalized)


def attribute_assortativity_coefficient(G, attribute, nodes=None):
    from networkx.algorithms.assortativity.correlation import attribute_ac

    M = attribute_mixing_matrix(G, attribute, nodes)
    return attribute_ac(M)


def numeric_assortativity_coefficient(G, attribute, nodes=None):
    from networkx.algorithms.assortativity.correlation import _numeric_ac

    if G._core.is_native():
        raise NotImplementedError("native graphs have no node attributes")
    _b14_reusable(nodes)
    base = _networkx_graph(G)
    # NetworkX's own expression on the NetworkX graph: same values, same errors.
    vals = {base.nodes[n][attribute] for n in (base.nodes if nodes is None else nodes)}
    mapping = {d: i for i, d in enumerate(vals)}
    M = attribute_mixing_matrix(G, attribute, nodes, mapping)
    return _numeric_ac(M, mapping)


def average_neighbor_degree(G, source="out", target="out", nodes=None, weight=None):
    kinds = {"in": 2, "out": 1, "in+out": 0}
    if G.is_directed():
        s = next((kinds[k] for k in kinds if source == k), None)
        if s is None:
            raise nx.NetworkXError(f"source argument {source} must be 'in', 'out' or 'in+out'")
        t = next((kinds[k] for k in kinds if target == k), None)
        if t is None:
            raise nx.NetworkXError(f"target argument {target} must be 'in', 'out' or 'in+out'")
    else:
        if source != "out" or target != "out":
            raise nx.NetworkXError(
                "source and target arguments are only supported for directed graphs"
            )
        s = t = 0
    attr = _b14_int_weight(G, weight)
    single, positions = _nbunch_positions(G, nodes)
    if single:
        raise NotImplementedError("NetworkX raises its own error here")
    if positions is None:
        positions = list(range(len(G)))
    terms = G._core.neighbor_degree_terms(positions, s, t, attr)
    nodes_ = G._nodes
    return {nodes_[i]: 0.0 if deg == 0 else total / deg for i, (total, deg) in zip(positions, terms)}


def average_degree_connectivity(G, source="in+out", target="in+out", nodes=None, weight=None):
    kinds = {"in": 2, "out": 1, "in+out": 0}
    if G.is_directed():
        if source not in ("in", "out", "in+out"):
            raise nx.NetworkXError('source must be one of "in", "out", or "in+out"')
        if target not in ("in", "out", "in+out"):
            raise nx.NetworkXError('target must be one of "in", "out", or "in+out"')
        s = next(kinds[k] for k in kinds if source == k)
        t = next(kinds[k] for k in kinds if target == k)
    else:
        if source != "in+out" or target != "in+out":
            raise nx.NetworkXError(
                "source and target arguments are only supported for directed graphs"
            )
        s = t = 0
    attr = _b14_int_weight(G, weight)
    if nodes is None:
        positions = list(range(len(G)))
    elif nodes in G:
        positions = [G._index[nodes]]
    else:
        positions = _b14_nbunch_list(G, nodes)
    keys, sums, norms = G._core.degree_connectivity(positions, s, t, attr)
    return {k: total if norm == 0 else total / norm for k, total, norm in zip(keys, sums, norms)}


# Reciprocity, rich club, s-metric and walks.


def overall_reciprocity(G):
    n_all_edge = G._core.number_of_edges()
    # Undirected graphs: `G.to_undirected()` keeps every edge.
    n_overlap_edge = G._core.reciprocated_edges() if G.is_directed() else 0
    if n_all_edge == 0:
        raise nx.NetworkXError("Not defined for empty graphs")
    return n_overlap_edge / n_all_edge


def reciprocity(G, nodes=None):
    if nodes is None:
        return overall_reciprocity(G)
    if not G.is_directed():
        raise NotImplementedError("NetworkX raises AttributeError here")
    if nodes in G:
        overlap, total = G._core.reciprocity_counts([G._index[nodes]])[0]
        if total == 0:
            raise nx.NetworkXError("Not defined for isolated nodes.")
        return 2 * overlap / total
    _, positions = _nbunch_positions(G, nodes)
    counts = G._core.reciprocity_counts(positions)
    nodes_ = G._nodes
    return {
        nodes_[i]: None if total == 0 else 2 * overlap / total
        for i, (overlap, total) in zip(positions, counts)
    }


def rich_club_coefficient(G, normalized=True, Q=100, seed=None):
    _undirected_only(G)
    if G._core.has_self_loops():
        raise Exception("rich_club_coefficient is not implemented for graphs with self loops.")
    if normalized:
        raise NotImplementedError("normalization uses random edge swaps")
    if G._core.number_of_edges() == 0:
        return {}
    return {d: 2 * ek / (nk * (nk - 1)) for d, (nk, ek) in enumerate(G._core.rich_club_counts())}


def s_metric(G):
    return float(G._core.s_metric_sum())


# NetworkX takes powers of a dense (3.4, 3.5) or sparse (3.6+) int64 matrix;
# rustnx walks the rows, which costs time linear in the walk length.
_B14_MAX_WALK = 100


def number_of_walks(G, walk_length):
    if type(walk_length) is not int:
        raise NotImplementedError("rustnx needs an int walk_length")
    if walk_length < 0:
        raise ValueError(f"`walk_length` cannot be negative: {walk_length}")
    if len(G) == 0:
        raise nx.NetworkXError("Graph has no nodes or edges")
    if G._core.number_of_edges() == 0:
        raise NotImplementedError("NetworkX returns floats for a graph without edges")
    if walk_length > _B14_MAX_WALK:
        raise NotImplementedError("long walks run in NetworkX")
    rows = G._core.walk_counts(walk_length)
    nodes = G._nodes
    return {u: dict(zip(nodes, row)) for u, row in zip(nodes, rows)}


# --- Batch 15: communities, efficiency and structural holes ----------------------
#
# Communities come back as sets built in NetworkX's insertion order (or from
# the same frozenset operations, replayed in Python), so they iterate alike.


def efficiency(G, u, v):
    _undirected_only(G)
    if u is None or v is None:
        raise NotImplementedError("NetworkX's shortest_path_length returns a dict here")
    try:
        eff = 1 / shortest_path_length(G, u, v)
    except nx.NetworkXNoPath:
        eff = 0
    return eff


def global_efficiency(G):
    _undirected_only(G)
    n = len(G)
    denom = n * (n - 1)
    if denom != 0:
        # NetworkX adds `1 / d` pair by pair, sources in order, each in BFS
        # order; the Rust total adds them in that order.
        return G._core.global_efficiency_total() / denom
    return 0


def local_efficiency(G):
    _undirected_only(G)
    n = len(G)
    core, nodes, index = G._core, G._nodes, G._index
    orders = []
    for v in range(n):
        nbrs = core.neighbors(v)
        if 2 * len(nbrs) < n:
            # `G.subgraph(G[v])` iterates its node set, a Python set, when
            # it holds fewer than half of G's nodes: build the same set.
            orders.append([index[x] for x in set([nodes[i] for i in nbrs])])
        else:
            orders.append(None)
    values = core.local_efficiencies(orders)
    return sum([0 if x is None else x for x in values]) / n


def _distance_index(G, weight, kind):
    """``gutman_index`` (kind 0), ``schultz_index`` (1) and
    ``hyper_wiener_index`` (2)."""
    _undirected_only(G)
    _check_not_null(G)
    if not is_connected(G):
        return float("inf")
    if "weight" in G:
        # NetworkX's `dict(G.degree, weight=weight)` adds a "weight" key.
        raise NotImplementedError("a node named 'weight' runs in NetworkX")
    float_mode = False
    if weight is not None:
        weight, all_int, has_hidden = _check_weight(G, weight)
        if has_hidden or G._core.weight_mixed(weight) or G._core.has_negative_weight(weight):
            raise NotImplementedError("rustnx needs non-negative numeric weights here")
        float_mode = not all_int
        if float_mode and kind == 2:
            raise NotImplementedError("float powers run in NetworkX")
    total = G._core.distance_index(kind, weight, float_mode, _COMPENSATED_SUM)
    if total is None:
        raise NotImplementedError("the sum is too large to match exactly")
    return total / 2


def gutman_index(G, weight=None):
    return _distance_index(G, weight, 0)


def schultz_index(G, weight=None):
    return _distance_index(G, weight, 1)


def hyper_wiener_index(G, weight=None):
    return _distance_index(G, weight, 2)


def closeness_vitality(G, node=None, weight=None, wiener_index=None):
    float_mode = False
    if weight is not None:
        weight, all_int, has_hidden = _check_weight(G, weight)
        if has_hidden or G._core.weight_mixed(weight) or G._core.has_negative_weight(weight):
            raise NotImplementedError("rustnx needs non-negative numeric weights here")
        float_mode = not all_int
    directed = G.is_directed()

    def value(total):
        if total is None:
            return float("inf")
        return total if directed else total / 2

    n = len(G)
    whole = wiener_index is None
    if whole:
        _check_not_null(G)  # nx.wiener_index raises on the null graph
    if node is not None:
        {node}  # raises TypeError for an unhashable node, as `set(G) - {node}` does
        i = G._index.get(node)
        if (n - (i is not None)) == 0:
            raise nx.NetworkXPointlessConcept("Connectivity is undefined for the null graph.")
        removals = [] if i is None else [i]
        total, after = G._core.vitality_totals(
            weight, float_mode, _COMPENSATED_SUM, removals, whole or i is None
        )
        if whole:
            wiener_index = value(total)
        return wiener_index - value(total if i is None else after[0])
    if n == 0:
        return {}
    if n == 1:
        raise nx.NetworkXPointlessConcept("Connectivity is undefined for the null graph.")
    total, after = G._core.vitality_totals(
        weight, float_mode, _COMPENSATED_SUM, list(range(n)), whole
    )
    if whole:
        wiener_index = value(total)
    return {v: wiener_index - value(t) for v, t in zip(G._nodes, after)}


def flow_hierarchy(G, weight=None):
    if G._core.number_of_edges() == 0:
        raise nx.NetworkXError("flow_hierarchy not applicable to empty graphs")
    if not G.is_directed():
        raise nx.NetworkXError("G must be a digraph in flow_hierarchy")
    if weight is not None:
        weight, all_int, has_hidden = _check_weight(G, weight)
        if has_hidden or not all_int:
            raise NotImplementedError("rustnx sums integer weights here")
    inside, total = G._core.scc_arc_weights(weight)
    if weight is None:
        return 1 - inside / total
    if max(abs(inside), abs(total)) >= _MAX_EXACT_INT:
        raise NotImplementedError("weights too large to sum exactly as floats")
    # `size(weight)` halves the degree sum with `/`: floats.
    return 1 - float(inside) / float(total)


def voronoi_cells(G, center_nodes, weight="weight"):
    positions = _multi_sources(G, center_nodes)
    weight, _, _ = _check_weight(G, weight)
    order, _, parents, seen, roots = G._core.dijkstra_forest(positions, weight, None)
    # The key order of `multi_source_dijkstra_path`'s result (see `_multi_paths`).
    distinct = list(dict.fromkeys(positions))
    k = len(distinct)
    if _dijkstra_paths_in_pop_order():
        if len(roots) != k or set(order[:k]) != set(distinct):
            raise NotImplementedError("negative weights reorder the sources")
        key_order = distinct + order[k:]
    else:
        key_order = seen
    root = {}
    for v, p in zip(order, parents):
        root[v] = v if p == _NO_PARENT else root[p]
    nodes = G._nodes
    nearest = {nodes[v]: nodes[root[v]] for v in key_order}
    cells = nx.utils.groups(nearest)
    unreachable = set(nodes) - set(nearest)
    if unreachable:
        cells["unreachable"] = unreachable
    return cells


# Communities (todo item 40).


def _present_nodes(G, communities):
    """``{n for c in communities for n in c if n in G}``."""
    index = G._index
    try:
        return {n for c in communities for n in c if n in index}
    except TypeError:
        return {n for c in communities for n in c if n in G}


def is_partition(G, communities):
    if not isinstance(communities, list):
        communities = list(communities)
    nodes = _present_nodes(G, communities)
    return len(G) == len(nodes) == sum(len(c) for c in communities)


def is_cover(G, communities):
    if not isinstance(communities, list):
        communities = list(communities)
    nodes = _present_nodes(G, communities)
    return len(nodes) == len(G)


@functools.cache
def _modularity_zero_check():
    """Whether NetworkX's ``modularity`` returns 0 for a graph without edge
    weight (3.5+); before, it divided by zero."""
    try:
        return "if m == 0" in inspect.getsource(nx.community.modularity.orig_func)
    except (AttributeError, OSError, TypeError):
        return False


def _modularity_weight(G, weight):
    """``(weight, float sums?)`` for NetworkX's weighted degree and edge
    sums; declines weights whose Python sums mix ints and floats."""
    if weight is None:
        return None, False
    weight, all_int, has_hidden = _check_weight(G, weight)
    if has_hidden or G._core.weight_mixed(weight):
        raise NotImplementedError("rustnx needs all-int or all-float weights here")
    return weight, not all_int


def _community_positions(G, communities):
    """Each community's nodes as positions, in ``set(community)`` order,
    flattened with each community's end."""
    index = G._index
    flat, ends = [], []
    for c in communities:
        flat.extend([index[n] for n in set(c)])
        ends.append(len(flat))
    return flat, ends


def modularity(G, communities, weight="weight", resolution=1):
    if not isinstance(communities, list):
        communities = list(communities)
    if not is_partition(G, communities):
        # NetworkX's NotAPartition names its own graph.
        raise NotImplementedError("NetworkX raises NotAPartition here")
    weight, float_mode = _modularity_weight(G, weight)
    if float_mode and G.is_directed():
        G._ensure_exact_pred()  # in-degree sums follow G._pred's order
    flat, ends = _community_positions(G, communities)
    total, per = G._core.modularity_stats(flat, ends, weight, float_mode, _COMPENSATED_SUM)
    if G.is_directed():
        m = total
        if m == 0:
            if not _modularity_zero_check():
                raise NotImplementedError("NetworkX divides by zero here")
            return 0
        norm = 1 / m**2
    else:
        deg_sum = total
        m = deg_sum / 2
        if m == 0:
            if not _modularity_zero_check():
                raise NotImplementedError("NetworkX divides by zero here")
            return 0
        norm = 1 / deg_sum**2
    return sum([l_c / m - resolution * out_sum * in_sum * norm for l_c, out_sum, in_sum in per])


def partition_quality(G, partition):
    node_community = {}
    for i, community in enumerate(partition):
        for node in community:
            node_community[node] = i
    possible_inter_community_edges = sum(
        len(p1) * len(p2) for p1, p2 in combinations(partition, 2)
    )
    if G.is_directed():
        possible_inter_community_edges *= 2
    n = len(G)
    total_pairs = n * (n - 1)
    if not G.is_directed():
        total_pairs //= 2
    index = G._index
    block = [-1] * n
    for node, i in node_community.items():
        j = index.get(node)
        if j is not None:
            block[j] = i
    counts = G._core.block_edge_counts(block, True)
    if counts is None:
        raise NotImplementedError("NetworkX raises KeyError for a node in no community")
    intra_community_edges, inter = counts
    inter_community_non_edges = possible_inter_community_edges - inter
    coverage = intra_community_edges / G._core.number_of_edges()
    performance = (intra_community_edges + inter_community_non_edges) / total_pairs
    return coverage, performance


def _subgraph_positions(G, nbunch):
    """The nodes ``G.subgraph(nbunch)`` keeps, as distinct positions."""
    index = G._index
    if nbunch in G:
        return [index[nbunch]]
    try:
        return list(dict.fromkeys([index[n] for n in nbunch if n in index]))
    except TypeError:
        raise NotImplementedError("NetworkX raises its nbunch errors here") from None


def intra_community_edges(G, partition):
    flat, ends = [], []
    for block in partition:
        flat.extend(_subgraph_positions(G, block))
        ends.append(len(flat))
    return G._core.edges_inside_blocks(flat, ends)


def _quotient_blocks(G, partition):
    """``quotient_graph``'s checks: each node's block (``-1``: not in the
    partition) and the block sizes."""
    if isinstance(partition, dict):
        partition = list(partition.values())
    if iter(partition) is partition:
        raise NotImplementedError("rustnx needs a reusable partition")
    partition_nodes = set().union(*partition)
    index = G._index
    if len(partition_nodes) != len(G):
        # NetworkX checks the partition on `G.subgraph(partition_nodes)`.
        kept = {n for n in partition_nodes if n in index}
    else:
        kept = None
    communities = partition if isinstance(partition, list) else list(partition)
    if kept is None:
        size = len(G)
        nodes = _present_nodes(G, communities)
    else:
        size = len(kept)
        nodes = {n for c in communities for n in c if n in kept}
    if not size == len(nodes) == sum(len(c) for c in communities):
        raise nx.NetworkXException("each node must be in exactly one part of `partition`")
    block = [-1] * len(G)
    sizes = []
    for i, c in enumerate(communities):
        for n in c:
            block[index[n]] = i
        sizes.append(len(c))
    return block, sizes


def inter_community_edges(G, partition):
    block, _ = _quotient_blocks(G, partition)
    return G._core.block_edge_counts(block, False)[1]


def inter_community_non_edges(G, partition):
    # NetworkX counts the edges of nx.complement(G) between blocks: the
    # pairs of nodes in different blocks, less G's edges between blocks.
    block, sizes = _quotient_blocks(G, partition)
    total = sum(sizes)
    pairs = total * total - sum(s * s for s in sizes)
    if not G.is_directed():
        pairs //= 2
    return pairs - G._core.block_edge_counts(block, False)[1]


def _pair_labels(G):
    """NetworkX's edge betweenness dict also holds the nodes as keys, so a
    node that is a 2-tuple could collide with an edge."""
    if any(type(v) is tuple and len(v) == 2 for v in G._nodes):
        raise NotImplementedError("2-tuple node labels run in NetworkX")


def _betweenness_scale(n):
    """``edge_betweenness_centrality``'s normalization (the same formula in
    ``_rescale_e`` and in 3.6's ``_rescale``)."""
    return 1 / (n * (n - 1)) if n >= 2 else None


def girvan_newman(G, most_valuable_edge=None):
    if most_valuable_edge is not None:
        raise NotImplementedError("rustnx does not support most_valuable_edge")
    _pair_labels(G)

    def compute():
        if G._core.number_of_edges() == 0:
            yield tuple(connected_components(G))
            return
        nodes = G._nodes
        state = G._core.girvan_newman(_betweenness_scale(len(G)))
        while True:
            level = state.next_level()
            if level is None:
                return
            yield tuple({nodes[i] for i in comp} for comp in level)

    return _computed_on_first_next(G, compute, lambda H: nx.community.girvan_newman(H, backend="networkx"))


def edge_betweenness_partition(G, number_of_sets, *, weight=None):
    if number_of_sets <= 0:
        raise nx.NetworkXError("number_of_sets must be >0")
    if number_of_sets == 1:
        return [set(G._nodes)]
    if number_of_sets == len(G):
        return [{n} for n in G._nodes]
    if number_of_sets > len(G):
        raise nx.NetworkXError("number_of_sets must be <= len(G)")
    _undirected_only(G)  # connected_components(G.copy())
    if type(number_of_sets) is not int:
        raise NotImplementedError("rustnx needs an int number_of_sets")
    _pair_labels(G)
    if weight is not None:
        weight, _, has_hidden = _check_weight(G, weight)
        if has_hidden:
            raise NotImplementedError("rustnx does not support None edge weights here")
    comps = G._core.edge_betweenness_partition(number_of_sets, weight, _betweenness_scale(len(G)))
    nodes = G._nodes
    return [{nodes[i] for i in comp} for comp in comps]


def _plain_number(value, name):
    if type(value) not in (int, float, bool) or not abs(value) < _MAX_EXACT_INT:
        raise NotImplementedError(f"rustnx needs a plain number for {name}")
    return float(value)


def greedy_modularity_communities(G, weight=None, resolution=1, cutoff=1, best_n=None):
    if not G._core.number_of_edges():
        return [{n} for n in G._nodes]
    if (cutoff < 1) or (cutoff > len(G)):
        raise ValueError(f"cutoff must be between 1 and {len(G)}. Got {cutoff}.")
    if best_n is not None:
        if (best_n < 1) or (best_n > len(G)):
            raise ValueError(f"best_n must be between 1 and {len(G)}. Got {best_n}.")
        if best_n < cutoff:
            raise ValueError(f"Must have best_n >= cutoff. Got {best_n} < {cutoff}")
        if best_n == 1:
            return [set(G._nodes)]
    else:
        best_n = len(G)
    numbers = [_plain_number(x, name) for x, name in
               [(resolution, "resolution"), (cutoff, "cutoff"), (best_n, "best_n")]]
    rank = _node_rank(G)  # ties compare the node tuples
    weight, float_mode = _modularity_weight(G, weight)
    if float_mode and G.is_directed():
        G._ensure_exact_pred()
    found = G._core.greedy_modularity(weight, float_mode, _COMPENSATED_SUM, numbers[0], rank, *numbers[1:])
    if found is None:
        raise NotImplementedError("NetworkX's merge loop fails or can't be followed here")
    merges, exhausted = found
    nodes = G._nodes
    communities = {n: frozenset([n]) for n in nodes}
    for u, v in merges:
        u, v = nodes[u], nodes[v]
        communities[v] = frozenset(communities[u] | communities[v])
        del communities[u]
    if exhausted:
        # The generator ran out: merge the largest communities, as NetworkX does.
        communities = sorted(communities.values(), key=len, reverse=True)
        while len(communities) > best_n:
            comm1, comm2, *rest = communities
            communities = [comm1 ^ comm2]
            communities.extend(rest)
        return communities
    return sorted(communities.values(), key=len, reverse=True)


def naive_greedy_modularity_communities(G, resolution=1, weight=None):
    _undirected_only(G)
    weight, float_mode = _modularity_weight(G, weight)
    if float_mode:
        raise NotImplementedError("float sums follow frozenset order: NetworkX runs these")
    if type(resolution) not in (int, float, bool) or (
        type(resolution) is not float and not abs(resolution) < 2**62
    ):
        raise NotImplementedError("rustnx needs a plain number for resolution")
    deg_sum, _ = G._core.modularity_stats([], [], weight, False, _COMPENSATED_SUM)
    nodes = G._nodes
    m = deg_sum / 2
    if m == 0:
        if not _modularity_zero_check():
            raise NotImplementedError("NetworkX divides by zero here")
        # Every modularity is 0: nothing is merged.
        return [frozenset([u]) for u in nodes]
    if deg_sum >= _MAX_EXACT_INT:
        raise NotImplementedError("weights too large to match exactly")
    norm = 1 / deg_sum**2
    if type(resolution) is float:
        res_int, res_float = None, resolution
    else:
        res_int, res_float = int(resolution), 0.0
    merges = G._core.naive_greedy_modularity(weight, m, norm, res_int, res_float, _COMPENSATED_SUM)
    if merges is None:
        raise NotImplementedError("weights too large to match exactly")
    communities = [frozenset([u]) for u in nodes]
    for i, j in merges:
        u, v = communities[i], communities[j]
        communities[j] = u | v
        communities[i] = frozenset([])
    return sorted((c for c in communities if len(c) > 0), key=len, reverse=True)


def asyn_lpa_communities(G, weight=None, seed=None):
    # `seed` is already a `random.Random` (py_random_state); rustnx replays
    # CPython's Mersenne Twister from its state and hands the state back.
    if type(seed) is not random.Random:
        raise NotImplementedError("rustnx replays random.Random generators only")
    if weight is not None:
        weight, _, has_hidden = _check_weight(G, weight)
        if has_hidden:
            raise NotImplementedError("rustnx does not support None edge weights here")

    def compute():
        version, internal, gauss = seed.getstate()
        found = G._core.asyn_lpa(weight, list(internal))
        if found is None:
            raise RuntimeError("unexpected random.Random state")
        labels, internal = found
        seed.setstate((version, tuple(internal), gauss))
        yield from nx.utils.groups(dict(zip(G._nodes, labels))).values()

    return _computed_on_first_next(
        G, compute, lambda H: nx.community.asyn_lpa_communities(H, weight, seed, backend="networkx")
    )


def fast_label_propagation_communities(G, *, weight=None, seed=None):
    if type(seed) is not random.Random:
        raise NotImplementedError("rustnx replays random.Random generators only")
    if weight is not None:
        weight, _, has_hidden = _check_weight(G, weight)
        if has_hidden:
            raise NotImplementedError("rustnx does not support None edge weights here")
    if G.is_directed():
        G._ensure_exact_pred()  # `all_neighbors` lists predecessors first

    def compute():
        version, internal, gauss = seed.getstate()
        found = G._core.fast_label_propagation(weight, list(internal))
        if found is None:
            raise RuntimeError("unexpected random.Random state")
        labels, internal = found
        seed.setstate((version, tuple(internal), gauss))
        yield from nx.utils.groups(dict(zip(G._nodes, labels))).values()

    return _computed_on_first_next(
        G,
        compute,
        lambda H: nx.community.fast_label_propagation_communities(
            H, weight=weight, seed=seed, backend="networkx"
        ),
    )


@functools.cache
def _fluidc_legacy():
    """Whether ``asyn_fluidc`` is NetworkX 3.5's or older: it loops while
    ``cont``, breaking once ``iter_count > max_iter``, and accepts any
    ``max_iter``. ``None`` if unrecognized."""
    try:
        source = inspect.getsource(nx.community.asyn_fluidc.orig_func)
    except (AttributeError, OSError, TypeError):
        return None
    if "while cont and iter_count < max_iter:" in source and "must be greater than 0" in source:
        return False
    if "if iter_count > max_iter:" in source:
        return True
    return None


def asyn_fluidc(G, k, max_iter=100, seed=None):
    _undirected_only(G)
    if not isinstance(k, int):
        raise nx.NetworkXError("k must be an integer.")
    if not k > 0:
        raise nx.NetworkXError("k must be greater than 0.")
    if not is_connected(G):
        raise nx.NetworkXError("Fluid Communities require connected Graphs.")
    if len(G) < k:
        raise nx.NetworkXError("k cannot be bigger than the number of nodes.")
    legacy = _fluidc_legacy()
    if legacy is None:
        raise NotImplementedError("unrecognized NetworkX asyn_fluidc")
    if not legacy and max_iter <= 0:
        msg = f"{max_iter=} must be greater than 0"
        raise ValueError(msg)
    if type(seed) is not random.Random or type(k) is not int or type(max_iter) is not int:
        raise NotImplementedError("rustnx needs int arguments and a random.Random seed")
    version, internal, gauss = seed.getstate()
    found = G._core.asyn_fluidc(k, max(min(max_iter, 2**62), -(2**62)), legacy, list(internal))
    if found is None:
        raise NotImplementedError("NetworkX's loop fails here")
    com, order, internal = found
    seed.setstate((version, tuple(internal), gauss))
    nodes = G._nodes
    return iter(nx.utils.groups({nodes[v]: com[v] for v in order}).values())


def overlapping_modularity(G, communities, *, weight="weight", resolution=1):
    _undirected_only(G)
    if not isinstance(communities, list):
        communities = list(communities)
    if not is_cover(G, communities):
        raise nx.NetworkXError("`communities` is not a valid cover of the nodes of G")
    index = G._index
    membership = [0] * len(G)
    for community in communities:
        for node in community:
            i = index.get(node) if node in G else None
            if i is None:
                # NetworkX raises KeyError when it reaches the node's degree.
                raise NotImplementedError("a community holds a node not in G")
            membership[i] += 1
    weight, float_mode = _modularity_weight(G, weight)
    flat, ends = _community_positions(G, communities)
    deg_sum, per = G._core.overlap_stats(flat, ends, membership, weight, float_mode, _COMPENSATED_SUM)
    if deg_sum == 0:
        return 0.0
    return sum([2 * l_c / deg_sum - resolution * (k_c / deg_sum) ** 2 for l_c, k_c in per])


# --- Batch 16: approximation algorithms and graph operations -----------------------


def _node_attr_values(G, attr, default=1):
    """``G.nodes[v].get(attr, default)`` for each node, in node order, or
    ``None`` for native graphs (they have no node data, so every node gets
    the default)."""
    if G._core.is_native():
        return None
    source = G._source
    if source is None or not G._source_unchanged():
        raise NotImplementedError("rustnx needs the NetworkX graph's node data")
    node_data = source._node
    return [node_data[v].get(attr, default) for v in G._nodes]


def min_weighted_vertex_cover(G, weight=None):
    order = G._core.local_ratio_cover(_node_attr_values(G, weight))
    if order is None:
        raise NotImplementedError("rustnx needs int or float node weights here")
    nodes = G._nodes
    # NetworkX adds the nodes to its set in this order.
    return {nodes[i] for i in order}


@functools.cache
def _dominating_counts_uncovered():
    """NetworkX 3.6+ prices a node by its uncovered closed neighbors; before,
    by its closed neighbors outside the dominating set."""
    return "neighborhood & uncovered_nodes" in _source_text(
        _registered("min_weighted_dominating_set")
    )


def _true_division_operands(values):
    """``values`` as floats that ``value / count`` divides exactly as Python
    does: ints that floats hold exactly, and floats other than NaN."""
    out = []
    for x in values:
        if type(x) in (int, bool):
            if abs(x) > _MAX_EXACT_INT:
                raise NotImplementedError("integer node weight is too large")
        elif not isinstance(x, float) or x != x:
            raise NotImplementedError("rustnx needs int or float node weights here")
        out.append(float(x))
    return out


def min_weighted_dominating_set(G, weight=None):
    _undirected_only(G)
    if len(G) == 0:
        return set()
    values = _node_attr_values(G, weight)
    if values is not None:
        values = _true_division_operands(values)
    order = G._core.min_weighted_dominating(values, _dominating_counts_uncovered())
    nodes = G._nodes
    return {nodes[i] for i in order}


def min_edge_dominating_set(G):
    if len(G) == 0:
        raise ValueError("Expected non-empty NetworkX graph!")
    return maximal_matching(G)


def min_maximal_matching(G):
    return maximal_matching(G)


def _tsp_weight(G, weight):
    """Validate a TSP weight; returns whether all values are ints."""
    if weight is None:
        raise NotImplementedError("rustnx needs a weight attribute here")
    weight, all_int, has_hidden = _check_weight(G, weight)
    if has_hidden:
        raise NotImplementedError("rustnx does not support None edge weights here")
    if G._core.weight_mixed(weight):
        raise NotImplementedError("edge weights mix ints and floats")
    return all_int


def greedy_tsp(G, weight="weight", source=None):
    if len(G) < 3:
        raise NotImplementedError("rustnx needs three or more nodes")
    _tsp_weight(G, weight)
    if not G._core.tsp_is_complete():
        raise nx.NetworkXError("G must be a complete graph.")
    if source is None:
        s = 0
    else:
        try:
            s = G._index[source]
        except (KeyError, TypeError):
            raise NotImplementedError("NetworkX raises for a missing source") from None
    status, cycle = G._core.greedy_tsp(weight, s)
    if status == 2:
        # NetworkX takes the first of the nearest nodes in set order.
        raise NotImplementedError("two nearest nodes tie")
    nodes = G._nodes
    first = nodes[s] if source is None else source
    return [first] + [nodes[i] for i in cycle[1:-1]] + [first]


def _tsp_tour(G, init_cycle, weight, source, move, message):
    """The checks ``simulated_annealing_tsp`` and ``threshold_accepting_tsp``
    make before their search: ``(cycle, move kind, tour state)``."""
    if len(G) < 3:
        raise NotImplementedError("rustnx needs three or more nodes")
    if move == "1-1":
        kind = 0
    elif move == "1-0":
        kind = 1
    else:
        raise NotImplementedError("rustnx does not support custom moves")
    all_int = _tsp_weight(G, weight)
    index = G._index
    if init_cycle == "greedy":
        cycle = greedy_tsp(G, weight=weight, source=source)
    else:
        cycle = list(init_cycle)
        if source is None:
            source = cycle[0]
        elif source != cycle[0]:
            raise nx.NetworkXError("source must be first node in init_cycle")
        if cycle[0] != cycle[-1]:
            raise nx.NetworkXError("init_cycle must be a cycle. (return to start)")
        if len(cycle) - 1 != len(G):
            raise nx.NetworkXError(message)
        try:
            found = [index.get(n) for n in cycle]
        except TypeError:
            raise NotImplementedError("init_cycle has an unhashable item") from None
        if len({i for i in found if i is not None}) != len(G):
            raise nx.NetworkXError(message)
        if not G._core.tsp_is_complete():
            raise nx.NetworkXError("G must be a complete graph.")
    positions = [index[n] for n in cycle]
    return cycle, kind, G._core.tsp_tour(weight, positions, all_int, _COMPENSATED_SUM)


def simulated_annealing_tsp(
    G,
    init_cycle,
    weight="weight",
    source=None,
    temp=100,
    move="1-1",
    max_iterations=10,
    N_inner=100,
    alpha=0.01,
    seed=None,
):
    cycle, kind, tour = _tsp_tour(
        G, init_cycle, weight, source, move, "init_cycle should be a cycle over all nodes in G."
    )
    # NetworkX's moves change the cycle in place, accepted or not; only the
    # cost it compares against waits for acceptance.
    cost = tour.cost()
    count = 0
    best_cost = cost
    while count <= max_iterations and temp > 0:
        count += 1
        for i in range(N_inner):
            a, b = seed.sample(range(1, len(cycle) - 1), k=2)
            adj_cost = tour.step(kind, a, b)
            delta = adj_cost - cost
            if delta <= 0:
                cost = adj_cost
                if cost < best_cost:
                    count = 0
                    tour.save_best()
                    best_cost = cost
            else:
                p = math.exp(-delta / temp)
                if p >= seed.random():
                    cost = adj_cost
        temp -= temp * alpha
    return [cycle[k] for k in tour.best()]


def threshold_accepting_tsp(
    G,
    init_cycle,
    weight="weight",
    source=None,
    threshold=1,
    move="1-1",
    max_iterations=10,
    N_inner=100,
    alpha=0.1,
    seed=None,
):
    cycle, kind, tour = _tsp_tour(
        G, init_cycle, weight, source, move, "init_cycle is not all and only nodes."
    )
    cost = tour.cost()
    count = 0
    best_cost = cost
    while count <= max_iterations:
        count += 1
        accepted = False
        for i in range(N_inner):
            a, b = seed.sample(range(1, len(cycle) - 1), k=2)
            adj_cost = tour.step(kind, a, b)
            delta = adj_cost - cost
            if delta <= threshold:
                accepted = True
                cost = adj_cost
                if cost < best_cost:
                    count = 0
                    tour.save_best()
                    best_cost = cost
        if accepted:
            threshold -= threshold * alpha
    return [cycle[k] for k in tour.best()]


def _treewidth_decomposition(G, order):
    """``treewidth_decomp``'s result for the elimination ``order`` (node
    positions). The bags are frozensets copied from sets whose iteration
    order depends on their history, so this replays NetworkX's set
    operations exactly; the heuristic, NetworkX's expensive part, ran in
    Rust."""
    nodes = G._nodes
    if G._core.is_native():
        core = G._core
        graph_dict = {n: set([nodes[j] for j in core.neighbors(i)]) - {n} for i, n in enumerate(nodes)}
    else:
        source = _networkx_graph(G)
        graph_dict = {n: set(source[n]) - {n} for n in source}
    node_stack = []
    for i in order:
        elim_node = nodes[i]
        nbrs = graph_dict[elim_node]
        for u, v in permutations(nbrs, 2):
            if v not in graph_dict[u]:
                graph_dict[u].add(v)
        node_stack.append((elim_node, nbrs))
        for u in graph_dict[elim_node]:
            graph_dict[u].remove(elim_node)
        del graph_dict[elim_node]

    decomp = nx.Graph()
    first_bag = frozenset(graph_dict.keys())
    decomp.add_node(first_bag)
    treewidth = len(first_bag) - 1
    while node_stack:
        curr_node, nbrs = node_stack.pop()
        old_bag = None
        for bag in decomp.nodes:
            if nbrs <= bag:
                old_bag = bag
                break
        if old_bag is None:
            old_bag = first_bag
        nbrs.add(curr_node)
        new_bag = frozenset(nbrs)
        treewidth = max(treewidth, len(new_bag) - 1)
        decomp.add_edge(old_bag, new_bag)
    return treewidth, decomp


def treewidth_decomp(G):
    if G.is_directed():
        raise NotImplementedError("rustnx supports treewidth_decomp on undirected graphs")
    return _treewidth_decomposition(G, G._core.min_fill_in_order())


def treewidth_min_fill_in(G):
    _undirected_only(G)
    return treewidth_decomp(G)


def approximate_diameter(G, seed=None):
    if len(G) == 0:
        raise nx.NetworkXError("Expected non-empty NetworkX graph!")
    if len(G) == 1:
        return 0
    source = seed.choice(list(G._nodes))
    result = G._core.two_sweep(G._index[source])
    if result is None:
        if G.is_directed():
            raise nx.NetworkXError("DiGraph not strongly connected.")
        raise nx.NetworkXError("Graph not connected.")
    return result


def _int_cut_weight(G, weight):
    if weight is not None:
        weight, all_int, has_hidden = _check_weight(G, weight)
        if not all_int or has_hidden:
            # Float sums follow the set order of the cut.
            raise NotImplementedError("rustnx needs int edge weights here")
    return weight


def _cut_partition(G, cut):
    # `G.nodes - cut`, as `Set.__sub__` builds it.
    return cut, set(v for v in G._nodes if v not in cut)


def one_exchange(G, initial_cut=None, seed=None, weight=None):
    _undirected_only(G)
    if len(G) == 0:
        raise NotImplementedError("rustnx needs a non-empty graph here")
    weight = _int_cut_weight(G, weight)
    if initial_cut is None:
        initial_cut = set()
    cut = set(initial_cut)
    nodes = G._nodes
    state = G._core.max_cut([v in cut for v in nodes], weight)
    current_cut_size = state.cut()
    n = len(nodes)
    while True:
        # Shuffling positions draws the same as shuffling `list(G.nodes())`.
        order = list(range(n))
        seed.shuffle(order)
        best, potential_cut_size = state.best(order)
        if potential_cut_size > current_cut_size:
            node = nodes[best]
            # NetworkX's `_swap_node_partition`; the new set's iteration
            # order depends on how it was built.
            cut = cut - {node} if node in cut else cut.union({node})
            current_cut_size = potential_cut_size
            state.switch(best)
        else:
            break
    return current_cut_size, _cut_partition(G, cut)


def randomized_partitioning(G, seed=None, p=0.5, weight=None):
    _undirected_only(G)
    weight = _int_cut_weight(G, weight)
    nodes = G._nodes
    cut = {node for node in nodes if seed.random() < p}
    cut_size = G._core.partition_cut_value([v in cut for v in nodes], weight)
    return cut_size, _cut_partition(G, cut)


def _kl_limit(G, l):
    """The path count at which NetworkX's ``cnt >= l`` first holds."""
    if type(l) not in (int, bool, float):
        raise NotImplementedError("rustnx needs a numeric l")
    if l <= 1:
        return 1
    if l != l or l == math.inf:
        if G._core.has_self_loops():
            raise NotImplementedError("NetworkX never finishes here")
        return 2**64 - 1
    limit = math.ceil(l)
    if limit >= 2**64:
        raise NotImplementedError("l is too large")
    return limit


def _kl_rejected(G, l, low_memory, first_only):
    if low_memory:
        # The searches then run on subgraphs whose order follows a set.
        raise NotImplementedError("rustnx does not support low_memory")
    limit = _kl_limit(G, l)
    if G.is_directed():
        G._ensure_exact_pred()
    return G._core.kl_rejected(limit, first_only)


def is_kl_connected(G, k, l, low_memory=False):
    us, _ = _kl_rejected(G, l, low_memory, True)
    return not us


def kl_connected_subgraph(G, k, l, low_memory=False, same_as_graph=False):
    us, vs = _kl_rejected(G, l, low_memory, False)
    # Each edge's searches run on a fresh copy of G, so the edges NetworkX's
    # later passes keep are exactly those its first pass keeps.
    H = copy.deepcopy(_networkx_graph(G))
    nodes = G._nodes
    for u, v in zip(us, vs):
        H.remove_edge(nodes[u], nodes[v])
    if same_as_graph:
        return (H, not us)
    return H



@functools.cache
def _steiner_by_distance():
    """NetworkX 3.6+'s Mehlhorn method finds each node's nearest terminal by
    the ``weight`` distances; before, by the ``"weight"`` attribute, counting
    the path's edges as the distance."""
    from networkx.algorithms.approximation import steinertree

    try:
        text = " ".join(inspect.getsource(steinertree._mehlhorn_steiner_tree).split())
    except (AttributeError, OSError, TypeError):
        text = ""
    if "nx.multi_source_dijkstra(G, terminal_nodes, weight=weight)" in text:
        return True
    if "nx.multi_source_dijkstra_path(G, terminal_nodes)" in text:
        return False
    return None


def steiner_tree(G, terminal_nodes, weight="weight", method=None):
    _undirected_only(G)
    if method is not None and method != "mehlhorn":
        # Kou's method starts from `set.pop()`.
        raise NotImplementedError("rustnx supports the mehlhorn method only")
    by_distance = _steiner_by_distance()
    if by_distance is None:
        raise NotImplementedError("unknown steiner_tree version")
    base = _networkx_graph(G)
    dist_weight = weight if by_distance else "weight"
    for attr in {dist_weight, weight} - {None}:
        attr, _, has_hidden = _check_weight(G, attr)
        if has_hidden or G._core.has_negative_weight(attr):
            raise NotImplementedError("rustnx needs non-negative, non-None weights here")
    positions = _multi_sources(G, terminal_nodes)
    missing, us, vs = G._core.mehlhorn_terminal_mst(positions, dist_weight, not by_distance, weight)
    nodes = G._nodes
    if missing is not None:
        raise KeyError(nodes[missing])
    # The rest works on small graphs: replay NetworkX's code.
    G_3 = nx.Graph()
    for u, v in zip(us, vs):
        if weight is None:
            path = bidirectional_shortest_path(G, nodes[u], nodes[v])
        else:
            path = _bidirectional_dijkstra(G, nodes[u], nodes[v], weight, False)[1]
        for n1, n2 in zip(path, path[1:]):
            if by_distance:
                G_3.add_edge(n1, n2, weight=base[n1][n2].get(weight, 1))
            else:
                G_3.add_edge(n1, n2)
    if by_distance:
        G_3_mst = list(nx.minimum_spanning_edges(G_3, data=False, weight=weight))
    else:
        G_3_mst = list(nx.minimum_spanning_edges(G_3, data=False))
    from networkx.algorithms.approximation import steinertree

    G_4 = base.edge_subgraph(G_3_mst).copy()
    steinertree._remove_nonterminal_leaves(G_4, terminal_nodes)
    return base.edge_subgraph(G_4.edges())


@functools.cache
def _peeling_heap_in_node_order():
    """NetworkX 3.7 fills ``_fractional_peeling``'s heap in node order. 3.5
    and 3.6 iterate a set of the nodes and index ``b`` by node, which only
    works for the nodes ``0..n-1`` (a set of those iterates in order)."""
    try:
        from networkx.algorithms.approximation import density

        text = " ".join(inspect.getsource(density._fractional_peeling).split())
    except (AttributeError, ImportError, OSError, TypeError):
        return None
    if "for idx, node in enumerate(G): heap.insert(node, b[idx])" in text:
        return True
    if "for idx in remaining_nodes: heap.insert(idx, b[idx])" in text:
        return False
    return None


def densest_subgraph(G, iterations=1, *, method="fista"):
    _undirected_only(G)
    if method not in ("fista", "greedy++") or type(method) is not str:
        raise NotImplementedError("NetworkX raises for this method")
    if G._core.number_of_edges() == 0:
        return 0.0, set()
    if type(iterations) not in (int, bool):
        raise NotImplementedError("rustnx needs an int number of iterations")
    if iterations < 1:
        raise ValueError(
            f"The number of iterations must be an integer >= 1. Provided: {iterations}"
        )
    heap_init = None
    if method == "fista":
        import numpy  # noqa: F401  (NetworkX's FISTA needs it)

        in_node_order = _peeling_heap_in_node_order()
        if in_node_order is None or G._core.has_self_loops():
            raise NotImplementedError("rustnx does not support this case")
        if not in_node_order:
            nodes = G._nodes
            if any(type(v) is not int for v in nodes) or set(nodes) != set(range(len(nodes))):
                raise NotImplementedError("NetworkX indexes by node here")
            index = G._index
            heap_init = [(index[v], v) for v in range(len(nodes))]
    density, removed, prefix = G._core.densest_peeling(iterations, method == "fista", heap_init)
    if prefix is None:
        return density, set()
    # `best_subgraph = set(remaining_nodes)` at the best point: replay the
    # removals so the set iterates in NetworkX's order.
    nodes = G._nodes
    remaining = set(nodes)
    for i in removed[:prefix]:
        remaining.remove(nodes[i])
    return density, set(remaining)

def _plain_result_class(G):
    """``G.__class__`` of the NetworkX graph G stands for (plain graphs only)."""
    if G._core.is_native():
        return nx.DiGraph if G.is_directed() else nx.Graph
    cls = type(G._source)
    if cls not in (nx.Graph, nx.DiGraph):
        raise NotImplementedError("rustnx builds plain Graph and DiGraph results only")
    return cls


def _graph_with_plain_edges(cls, nodes, us, vs):
    """A new ``cls`` with ``nodes``, then edges added as ``add_edges_from``
    would add the pairs of positions ``us``, ``vs``."""
    nodes = nodes if type(nodes) is list else list(nodes)
    R = cls()
    R.add_nodes_from(nodes)
    adj = R._adj
    pred = [R._pred[n] for n in nodes] if R.is_directed() else None
    _core._add_plain_edges([adj[n] for n in nodes], pred, nodes, us, vs)
    return R


def complement(G):
    cls = _plain_result_class(G)
    us, vs = G._core.complement_pairs()
    return _graph_with_plain_edges(cls, G._nodes, us, vs)


def power(G, k):
    _undirected_only(G)
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if type(k) in (int, bool):
        depth = min(k, len(G) + 1)
    elif type(k) is float:
        # The search stops after the first level `k <= level` holds for.
        depth = len(G) + 1 if k != k or k > len(G) else math.ceil(k)
    else:
        raise NotImplementedError("rustnx needs a numeric k")
    us, vs = G._core.power_pairs(depth)
    return _graph_with_plain_edges(nx.Graph, G._nodes, us, vs)


def _same_node_positions(G, H):
    if set(G._nodes) != set(H._nodes):
        raise nx.NetworkXError("Node sets of graphs not equal")
    index = H._index
    return [index[v] for v in G._nodes]


def difference(G, H):
    cls = _plain_result_class(G)
    mapping = _same_node_positions(G, H)
    us, vs = G._core.edges_missing_from(H._core, mapping)
    return _graph_with_plain_edges(cls, G._nodes, us, vs)


def symmetric_difference(G, H):
    cls = _plain_result_class(G)
    mapping = _same_node_positions(G, H)
    us, vs = G._core.edges_missing_from(H._core, mapping)
    index = G._index
    back = [index[v] for v in H._nodes]
    hus, hvs = H._core.edges_missing_from(G._core, back)
    us += [back[u] for u in hus]
    vs += [back[v] for v in hvs]
    return _graph_with_plain_edges(cls, G._nodes, us, vs)
