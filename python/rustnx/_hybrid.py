"""Run NetworkX's own code, with the matrices it builds supplied by rustnx.

Many NetworkX functions are NumPy or SciPy arithmetic on a matrix built from
the graph (``to_scipy_sparse_array``, ``laplacian_matrix``, ...), and
building that matrix in Python is often most of their time. A hybrid runs
the installed NetworkX's own function on the original NetworkX graph, with
those builders answered by rustnx's (which return identical arrays, much
faster). Every float operation is NetworkX's own NumPy/SciPy call on
identical inputs, so results match NetworkX bit for bit, and every
parameter of every NetworkX version is supported.

How: the function (and the NetworkX helper functions it calls) is copied
with ``nx`` in its globals replaced by a proxy module whose builders check
whether they were handed the graph being computed on; if so, and rustnx can
build that matrix, rustnx does, otherwise NetworkX's own builder runs.
"""

import contextvars
import functools
import inspect
import threading
import types

import networkx as nx

# Builders the proxy answers, by NetworkX name -> rustnx algorithms name.
BUILDERS = (
    "to_scipy_sparse_array",
    "to_numpy_array",
    "adjacency_matrix",
    "laplacian_matrix",
    "incidence_matrix",
    "biadjacency_matrix",
)

# NetworkX functions that hybrids call on graphs they build themselves (not
# the input graph): run in rustnx when it can, else in NetworkX.
NESTED = (
    "max_weight_matching",
    "is_perfect_matching",
    "complete_to_chordal_graph",
    "chordal_graph_cliques",
    "maximum_spanning_tree",
    "is_chordal",
)

# id(networkx graph) -> (networkx graph, rustnx graph) for the current call.
_graphs = contextvars.ContextVar("rustnx_hybrid_graphs", default={})


def _rustnx_builder(name):
    from . import algorithms

    ours = getattr(algorithms, name)
    theirs = nx.utils.backends._registered_algorithms[name]

    @functools.wraps(theirs)
    def build(G, *args, **kwargs):
        entry = _graphs.get().get(id(G))
        if entry is not None and entry[0] is G:
            try:
                return ours(entry[1], *args, **kwargs)
            except (NotImplementedError, TypeError):
                pass  # NetworkX's builder raises or handles it
        return theirs(G, *args, **kwargs)

    return build


@functools.cache
def _builder(name):
    return _rustnx_builder(name)


@functools.cache
def _nested(func):
    """``func`` (the public, decorated NetworkX function, so its input
    checks still run) called with ``backend="rustnx"``, or in NetworkX if
    rustnx declines."""

    @functools.wraps(func)
    def call(*args, **kwargs):
        try:
            return func(*args, backend="rustnx", **kwargs)
        except NotImplementedError:
            return func(*args, backend="networkx", **kwargs)

    return call


def _nested_name(value):
    """The NetworkX name of ``value`` if it is one of ``NESTED`` (a
    dispatchable, or a decorated function wrapping one), else None."""
    name = getattr(value, "name", None) or getattr(value, "__name__", None)
    if name not in NESTED:
        return None
    if isinstance(value, nx.utils.backends._dispatchable) or hasattr(value, "__wrapped__"):
        return name
    return None


@functools.cache
def _proxy():
    proxy = types.ModuleType("networkx")
    proxy.__dict__.update(nx.__dict__)
    bipartite = types.ModuleType("networkx.bipartite")
    bipartite.__dict__.update(nx.bipartite.__dict__)
    proxy.bipartite = bipartite
    for name in NESTED:
        if hasattr(nx, name):
            setattr(proxy, name, _nested(getattr(nx, name)))
    for name in BUILDERS:
        builder = _builder(name)
        if hasattr(nx, name):
            setattr(proxy, name, builder)
        if hasattr(nx.bipartite, name):
            setattr(bipartite, name, builder)
    return proxy


def _code_names(code):
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _code_names(const)
    return names


_copies = {}
# Copies are registered before their globals are patched (functions can
# call each other), so building them is serialized; free-threaded Python
# would otherwise let another thread pick up a half-patched copy.
_copy_lock = threading.RLock()


def _copy(func):
    """``func`` with ``nx`` replaced by the proxy, and the NetworkX
    functions it calls by name copied the same way."""
    with _copy_lock:
        return _copy_locked(func)


def _copy_locked(func):
    if func in _copies:
        return _copies[func]
    proxy = _proxy()
    scope = dict(func.__globals__)
    copy_ = types.FunctionType(
        func.__code__, scope, func.__name__, func.__defaults__, func.__closure__
    )
    copy_.__kwdefaults__ = func.__kwdefaults__
    _copies[func] = copy_
    for name in _code_names(func.__code__):
        value = scope.get(name)
        if value is nx:
            scope[name] = proxy
        elif _nested_name(value):
            scope[name] = _nested(value)
        elif isinstance(value, nx.utils.backends._dispatchable):
            if value.name in BUILDERS:
                scope[name] = _builder(value.name)
            elif value.name in HYBRIDS:
                scope[name] = _wrapper_for(value.name, value)
        elif (
            isinstance(value, types.FunctionType)
            and (getattr(value, "__module__", "") or "").startswith("networkx")
            # Decorated helpers (`np_random_state`, ...) are compiled
            # wrappers that convert their arguments; copying one would skip
            # that, so they run as they are.
            and not hasattr(value, "__wrapped__")
        ):
            scope[name] = _copy(value)
    return copy_


def _wrapper_for(name, dispatchable):
    """A nested call of another hybrid function: NetworkX's own code, copied,
    so its matrices come from rustnx too when it is handed the same graph."""
    inner = _copy(dispatchable.orig_func)

    @functools.wraps(dispatchable)
    def call(*args, **kwargs):
        return inner(*args, **kwargs)

    return call


# Filled in by algorithms.py: every NetworkX function run as a hybrid.
HYBRIDS = set()


@functools.cache
def _target(name):
    dispatchable = nx.utils.backends._registered_algorithms.get(name)
    if dispatchable is None:
        return None
    return dispatchable, _copy(dispatchable.orig_func)


def signature(name):
    """The installed NetworkX's signature for ``name`` (without
    ``backend``/backend keywords), or a catch-all if it doesn't have it."""
    target = nx.utils.backends._registered_algorithms.get(name)
    if target is None:
        return inspect.Signature(
            [inspect.Parameter("G", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        )
    params = [
        p
        for p in inspect.signature(target.orig_func).parameters.values()
        if p.kind is not p.VAR_KEYWORD and p.name != "backend"
    ]
    return inspect.Signature(params)


def run(name, arguments, base_graph):
    """Run NetworkX's ``name`` on ``arguments`` (rustnx graphs replaced by
    the NetworkX graphs they were converted from via ``base_graph``)."""
    target = _target(name)
    if target is None:
        raise NotImplementedError(f"this NetworkX has no {name}")
    dispatchable, func = target
    graphs = dict(_graphs.get())
    call_args = dict(arguments)
    for param in dispatchable.graphs:
        G = call_args.get(param)
        if G is not None and hasattr(G, "_core"):
            base = base_graph(G)
            graphs[id(base)] = (base, G)
            call_args[param] = base
    token = _graphs.set(graphs)
    try:
        return func(**call_args)
    finally:
        _graphs.reset(token)
