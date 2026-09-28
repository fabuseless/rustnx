"""The ``networkx.backends`` entry point for rustnx.

NetworkX looks up algorithms on this module by name, and calls
``convert_from_nx`` / ``convert_to_nx`` / ``can_run`` / ``should_run``.
"""

import inspect

from . import algorithms
from .algorithms import *  # noqa: F403
from .graph import RustnxGraph, from_networkx, to_networkx

_SIGNATURES = {
    name: inspect.signature(getattr(algorithms, name)) for name in algorithms.__all__
}

# Parameters that may hold an edge attribute name (or a callable).
_WEIGHT_PARAMS = ("weight", "distance")

# Linear-time algorithms where, on small graphs, converting to rustnx and
# dispatching costs more than NetworkX spends running the algorithm.
_LINEAR_TIME = {
    "connected_components",
    "is_connected",
    "number_connected_components",
    "single_source_dijkstra_path_length",
    "single_source_shortest_path_length",
}
SMALL_GRAPH_NODES = 500


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
    if preserve_edge_attrs is True:
        # Arbitrary edge data (e.g. for callable weights) isn't stored in Rust.
        raise NotImplementedError("rustnx only stores numeric edge attributes")
    if isinstance(preserve_edge_attrs, dict):
        edge_attrs = preserve_edge_attrs.get(graph_name) or edge_attrs
    return from_networkx(G, edge_attrs or {})


def convert_to_nx(obj, *, name=None):
    if isinstance(obj, RustnxGraph):
        return to_networkx(obj)
    return obj


def can_run(name, args, kwargs):
    sig = _SIGNATURES.get(name)
    if sig is None:
        return False
    try:
        bound = sig.bind(*args, **kwargs)
    except TypeError:
        return True  # let the call raise the usual error
    G = bound.arguments.get("G")
    if G is not None and G.is_multigraph():
        return "multigraphs are not supported yet"
    for param in _WEIGHT_PARAMS:
        if callable(bound.arguments.get(param)):
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
