"""NetworkX-compatible algorithms backed by the Rust core.

Each function has the same signature and return value as its NetworkX
counterpart. Inputs rustnx can't handle raise ``NotImplementedError``, which
makes NetworkX fall back to its own implementation.
"""

import functools
from collections import defaultdict
import inspect
import math
import operator
import sys

import networkx as nx
from networkx.algorithms.centrality import betweenness as _nx_betweenness

__all__ = [
    "all_pairs_dijkstra",
    "all_pairs_dijkstra_path",
    "all_pairs_dijkstra_path_length",
    "all_pairs_shortest_path",
    "all_pairs_shortest_path_length",
    "all_shortest_paths",
    "ancestors",
    "average_clustering",
    "average_shortest_path_length",
    "betweenness_centrality",
    "bfs_edges",
    "bfs_tree",
    "bidirectional_dijkstra",
    "bidirectional_shortest_path",
    "center",
    "closeness_centrality",
    "clustering",
    "connected_components",
    "core_number",
    "descendants",
    "dfs_edges",
    "dfs_preorder_nodes",
    "dfs_tree",
    "diameter",
    "dijkstra_path",
    "dijkstra_path_length",
    "eccentricity",
    "edge_betweenness_centrality",
    "eigenvector_centrality",
    "greedy_color",
    "harmonic_centrality",
    "has_path",
    "is_bipartite",
    "is_connected",
    "is_directed_acyclic_graph",
    "is_strongly_connected",
    "is_weakly_connected",
    "k_core",
    "katz_centrality",
    "label_propagation_communities",
    "maximum_spanning_edges",
    "maximum_spanning_tree",
    "minimum_spanning_edges",
    "minimum_spanning_tree",
    "number_connected_components",
    "number_strongly_connected_components",
    "number_weakly_connected_components",
    "pagerank",
    "periphery",
    "radius",
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
    "transitivity",
    "triangles",
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


def strongly_connected_components(G):
    _directed_only(G)
    return _components(G, G._core.strongly_connected_components())


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
