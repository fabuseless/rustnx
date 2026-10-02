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
    "connected_components",
    "is_connected",
    "is_directed_acyclic_graph",
    "is_strongly_connected",
    "is_weakly_connected",
    "number_connected_components",
    "number_strongly_connected_components",
    "number_weakly_connected_components",
    "single_source_dijkstra_path_length",
    "single_source_shortest_path_length",
    "strongly_connected_components",
    "topological_generations",
    "topological_sort",
    "weakly_connected_components",
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


@functools.cache
def _nx_signature(name):
    return inspect.signature(getattr(nx, name))


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
    arguments = {}
    unsupported = []
    for key, value in bound.arguments.items():
        param = nx_sig.parameters[key]
        if key == "backend":
            continue
        if param.kind is param.VAR_KEYWORD:
            unsupported.extend(value)  # backend-specific keywords
        elif key in ours:
            arguments[key] = value
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
    if G is not None and G.is_multigraph():
        return "multigraphs are not supported yet"
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
