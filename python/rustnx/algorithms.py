"""NetworkX-compatible algorithms backed by the Rust core.

Each function has the same signature and return value as its NetworkX
counterpart. Inputs rustnx can't handle raise ``NotImplementedError``, which
makes NetworkX fall back to its own implementation.
"""

import functools
from collections import Counter, defaultdict
import inspect
from itertools import chain
import math
import operator
import random
import sys
import warnings

import networkx as nx
from networkx.algorithms.centrality import betweenness as _nx_betweenness

__all__ = [
    "all_pairs_all_shortest_paths",
    "all_pairs_bellman_ford_path",
    "all_pairs_bellman_ford_path_length",
    "all_pairs_dijkstra",
    "all_pairs_dijkstra_path",
    "all_pairs_dijkstra_path_length",
    "all_pairs_shortest_path",
    "all_pairs_shortest_path_length",
    "all_shortest_paths",
    "all_topological_sorts",
    "all_triangles",
    "ancestors",
    "articulation_points",
    "astar_path",
    "astar_path_length",
    "attracting_components",
    "average_clustering",
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
    "bridges",
    "center",
    "centroid",
    "chain_decomposition",
    "check_planarity",
    "check_planarity_recursive",
    "chordal_graph_treewidth",
    "closeness_centrality",
    "clustering",
    "complete_to_chordal_graph",
    "condensation",
    "connected_components",
    "core_number",
    "cycle_basis",
    "dag_longest_path",
    "dag_longest_path_length",
    "dag_to_branching",
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
    "diameter",
    "dijkstra_path",
    "dijkstra_path_length",
    "dijkstra_predecessor_and_distance",
    "dispersion",
    "dominance_frontiers",
    "eccentricity",
    "edge_betweenness_centrality",
    "edge_betweenness_centrality_subset",
    "edge_bfs",
    "edge_dfs",
    "edge_load_centrality",
    "eigenvector_centrality",
    "eulerian_circuit",
    "eulerian_path",
    "find_cycle",
    "find_negative_cycle",
    "generalized_degree",
    "generic_bfs_edges",
    "get_counterexample",
    "get_counterexample_recursive",
    "girth",
    "global_reaching_centrality",
    "greedy_color",
    "group_betweenness_centrality",
    "group_closeness_centrality",
    "group_degree_centrality",
    "group_in_degree_centrality",
    "group_out_degree_centrality",
    "harmonic_centrality",
    "harmonic_diameter",
    "has_bridges",
    "has_cycle",
    "has_eulerian_path",
    "has_path",
    "immediate_dominators",
    "in_degree_centrality",
    "intersection_array",
    "is_aperiodic",
    "is_arborescence",
    "is_at_free",
    "is_attracting_component",
    "is_biconnected",
    "is_bipartite",
    "is_branching",
    "is_chordal",
    "is_coloring",
    "is_connected",
    "is_directed_acyclic_graph",
    "is_distance_regular",
    "is_equitable",
    "is_eulerian",
    "is_forest",
    "is_k_regular",
    "is_planar",
    "is_reachable",
    "is_regular",
    "is_semiconnected",
    "is_semieulerian",
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
    "k_truss",
    "katz_centrality",
    "kosaraju_strongly_connected_components",
    "kruskal_mst_edges",
    "label_propagation_communities",
    "lexicographical_topological_sort",
    "local_bridges",
    "local_reaching_centrality",
    "maximum_spanning_edges",
    "maximum_spanning_tree",
    "minimum_spanning_edges",
    "minimum_spanning_tree",
    "multi_source_dijkstra",
    "multi_source_dijkstra_path",
    "multi_source_dijkstra_path_length",
    "negative_edge_cycle",
    "newman_betweenness_centrality",
    "node_connected_component",
    "number_attracting_components",
    "number_connected_components",
    "number_of_isolates",
    "number_strongly_connected_components",
    "number_weakly_connected_components",
    "onion_layers",
    "out_degree_centrality",
    "pagerank",
    "percolation_centrality",
    "periphery",
    "predecessor",
    "prominent_group",
    "radius",
    "root_to_leaf_paths",
    "score_sequence",
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
    "to_prufer_sequence",
    "topological_generations",
    "topological_sort",
    "tournament_is_strongly_connected",
    "transitive_closure",
    "transitive_closure_dag",
    "transitive_reduction",
    "transitivity",
    "triangles",
    "v_structures",
    "voterank",
    "weakly_connected_components",
    "wiener_index",
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
    values = G._core.closeness(distance, bool(wf_improved), sources)
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
    stats, total = G._core.dijkstra_stats(weight, None)
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


def centroid(G, weight=None):
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
    # NetworkX 3.4 and 3.5 name `centroid` this way.
    return centroid(G, weight)


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
