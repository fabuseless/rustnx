"""NetworkX-compatible algorithms backed by the Rust core.

Each function has the same signature and return value as its NetworkX
counterpart. Inputs rustnx can't handle raise ``NotImplementedError``, which
makes NetworkX fall back to its own implementation.
"""

import functools
from collections import defaultdict
import inspect
from itertools import chain
import math
import operator
import sys
import warnings

import networkx as nx
from networkx.algorithms.centrality import betweenness as _nx_betweenness

__all__ = [
    "all_pairs_dijkstra",
    "all_pairs_dijkstra_path",
    "all_pairs_dijkstra_path_length",
    "all_pairs_shortest_path",
    "all_pairs_shortest_path_length",
    "all_shortest_paths",
    "all_topological_sorts",
    "ancestors",
    "articulation_points",
    "attracting_components",
    "average_clustering",
    "average_shortest_path_length",
    "betweenness_centrality",
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
    "center",
    "closeness_centrality",
    "clustering",
    "condensation",
    "connected_components",
    "core_number",
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
    "eccentricity",
    "edge_betweenness_centrality",
    "edge_bfs",
    "edge_dfs",
    "eigenvector_centrality",
    "generic_bfs_edges",
    "greedy_color",
    "harmonic_centrality",
    "has_cycle",
    "has_path",
    "in_degree_centrality",
    "is_aperiodic",
    "is_attracting_component",
    "is_biconnected",
    "is_bipartite",
    "is_connected",
    "is_directed_acyclic_graph",
    "is_forest",
    "is_semiconnected",
    "is_strongly_connected",
    "is_tree",
    "is_weakly_connected",
    "k_core",
    "katz_centrality",
    "kosaraju_strongly_connected_components",
    "label_propagation_communities",
    "lexicographical_topological_sort",
    "maximum_spanning_edges",
    "maximum_spanning_tree",
    "minimum_spanning_edges",
    "minimum_spanning_tree",
    "node_connected_component",
    "number_attracting_components",
    "number_connected_components",
    "number_strongly_connected_components",
    "number_weakly_connected_components",
    "out_degree_centrality",
    "pagerank",
    "periphery",
    "radius",
    "root_to_leaf_paths",
    "shortest_path",
    "shortest_path_length",
    "single_source_dijkstra",
    "single_source_dijkstra_path",
    "single_source_dijkstra_path_length",
    "single_source_shortest_path",
    "single_source_shortest_path_length",
    "single_target_shortest_path",
    "strongly_connected_components",
    "topological_generations",
    "topological_sort",
    "transitive_closure",
    "transitive_closure_dag",
    "transitive_reduction",
    "transitivity",
    "triangles",
    "v_structures",
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


def _all_pairs(G, compute):
    guard = _MutationGuard(G)

    def generate():
        nodes = G._nodes
        try:
            for start in range(0, len(nodes), _ALL_PAIRS_BATCH):
                batch = list(range(start, min(start + _ALL_PAIRS_BATCH, len(nodes))))
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
    paths = nx.single_source_dijkstra_path(H, 0, backend="networkx")
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
    return len(list(nx.all_shortest_paths(H, 0, 3, weight="weight", backend="networkx"))) == 4


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
    if not isinstance(strategy, str) or strategy != "largest_first" or interchange:
        raise NotImplementedError("rustnx implements the largest_first strategy only")
    order, colors = G._core.greedy_color()
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
