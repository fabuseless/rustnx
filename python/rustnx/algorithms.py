"""NetworkX-compatible algorithms backed by the Rust core.

Each function has the same signature and return value as its NetworkX
counterpart. Inputs rustnx can't handle raise ``NotImplementedError``, which
makes NetworkX fall back to its own implementation.
"""

import functools
import inspect
import math
import operator

import networkx as nx
from networkx.algorithms.centrality import betweenness as _nx_betweenness

__all__ = [
    "betweenness_centrality",
    "closeness_centrality",
    "connected_components",
    "is_connected",
    "is_directed_acyclic_graph",
    "is_strongly_connected",
    "is_weakly_connected",
    "number_connected_components",
    "number_strongly_connected_components",
    "number_weakly_connected_components",
    "pagerank",
    "single_source_dijkstra_path_length",
    "single_source_shortest_path_length",
    "strongly_connected_components",
    "topological_generations",
    "topological_sort",
    "weakly_connected_components",
]


def _index_of(G, node, message):
    try:
        return G._index[node]
    except (KeyError, TypeError):
        raise nx.NodeNotFound(message) from None


def _check_weight(G, attr):
    """Validate a weight argument; return ``(attr, all_int)``."""
    if callable(attr):
        raise NotImplementedError("rustnx does not support callable weights")
    all_int, has_hidden = G._core.weight_info(attr)
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
        cache = getattr(self.graph, "__networkx_cache__", None)
        return cache is not self._cache or self._key not in cache

    def release(self):
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
    weight, all_int, _ = _check_weight(G, weight)
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


def closeness_centrality(G, u=None, distance=None, wf_improved=True):
    distance, _, _ = _check_weight(G, distance)
    sources = None
    if u is not None:
        if distance is None:
            msg = f"Source {u} is not in G"
        else:
            msg = f"Node {u} not found in graph"
        sources = [_index_of(G, u, msg)]
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
