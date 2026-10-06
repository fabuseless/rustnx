"""rustnx: a Rust-accelerated backend for NetworkX.

Keep writing normal NetworkX code and opt in with either::

    NETWORKX_BACKEND_PRIORITY=rustnx NETWORKX_BACKEND_PRIORITY_GENERATORS=rustnx python my_script.py

or::

    import rustnx
    rustnx.enable()

Functions rustnx doesn't implement (or inputs it can't handle) keep running
in NetworkX, so enabling it never breaks working code.

A few functions (``betweenness_centrality``, ``edge_betweenness_centrality``
and ``pagerank``) use fast float arithmetic by default, so their floats can
differ from NetworkX's in the last bits (around 1e-16 to 1e-15 relative).
Turn on exact floats for every function or just some::

    nx.config.backends.rustnx.exact_floats = True
    nx.config.backends.rustnx.exact_floats_overrides["pagerank"] = True

Per-function overrides win over the global setting. ``rustnx.float_settings()``
and ``print(rustnx.explain_floats())`` show what applies now, and
``nx.config.backends.rustnx.verbose = True`` logs it as functions run. Every
other function always returns exactly what NetworkX returns.
"""

from ._config import explain_floats, float_settings
from ._info import FUNCTIONS
from .graph import DiGraph, Graph, RustnxGraph, from_networkx, to_networkx

__all__ = [
    "FUNCTIONS",
    "DiGraph",
    "Graph",
    "RustnxGraph",
    "enable",
    "explain_floats",
    "float_settings",
    "from_networkx",
    "to_networkx",
]


def enable(fallback=True, exact_floats=None, exact_floats_overrides=None, verbose=None):
    """Make NetworkX use rustnx for every function rustnx supports.

    Puts rustnx first in ``nx.config.backend_priority.algos`` (functions that
    take a graph) and ``nx.config.backend_priority.generators`` (functions
    that create one: generators, readers and graph builders, and on
    NetworkX 3.5+ also functions that return a new graph, such as
    ``union``). NetworkX graphs are converted to rustnx (and the conversion
    cached) for supported functions; everything else runs in NetworkX as
    usual, and so do small inputs, where NetworkX is faster.

    ``"networkx"`` is removed from the priority lists if present: NetworkX
    tries a backend that is both listed there and the input's own backend
    before any other, so listing it would stop rustnx from ever running on
    NetworkX graphs.

    With ``fallback=True`` (the default), rustnx graphs (``rustnx.Graph``)
    passed to functions rustnx doesn't implement are converted to NetworkX
    automatically instead of raising (``nx.config.fallback_to_nx``).
    Equivalent to ``NETWORKX_BACKEND_PRIORITY_ALGOS=rustnx`` and
    ``NETWORKX_BACKEND_PRIORITY_GENERATORS=rustnx`` plus
    ``NETWORKX_FALLBACK_TO_NX=1``.

    The float settings (see the README's "Exact floats" section), each left
    alone if not given:

    - ``exact_floats``: the global setting, ``nx.config.backends.rustnx.
      exact_floats`` (default False: fast floats, which can differ from
      NetworkX's in the last bits).
    - ``exact_floats_overrides``: ``{function name: bool}`` merged into
      ``nx.config.backends.rustnx.exact_floats_overrides``; these win over
      the global setting. Names without a float setting raise ``ValueError``.
    - ``verbose``: log which float setting each function runs with.
    """
    import networkx as nx

    priority = nx.config.backend_priority
    for key in ("algos", "generators"):
        others = [n for n in getattr(priority, key) if n not in ("rustnx", "networkx")]
        setattr(priority, key, ["rustnx", *others])
    if fallback:
        nx.config.fallback_to_nx = True
    settings = nx.config.backends.rustnx
    if exact_floats_overrides is not None:
        from ._config import FLOAT_FUNCTIONS

        unknown = sorted(set(exact_floats_overrides) - set(FLOAT_FUNCTIONS))
        if unknown:
            raise ValueError(
                f"no float setting for {', '.join(unknown)}; functions with one: "
                + ", ".join(sorted(FLOAT_FUNCTIONS))
            )
        settings.exact_floats_overrides = {
            **(settings.exact_floats_overrides or {}),
            **{name: bool(v) for name, v in exact_floats_overrides.items()},
        }
    if exact_floats is not None:
        settings.exact_floats = bool(exact_floats)
    if verbose is not None:
        settings.verbose = bool(verbose)


__version__ = "0.1.0a3"
