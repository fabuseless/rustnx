"""rustnx: a Rust-accelerated backend for NetworkX.

Keep writing normal NetworkX code and opt in with either::

    NETWORKX_BACKEND_PRIORITY=rustnx NETWORKX_BACKEND_PRIORITY_GENERATORS=rustnx python my_script.py

or::

    import rustnx
    rustnx.enable()

Functions rustnx doesn't implement (or inputs it can't handle) keep running
in NetworkX, so enabling it never breaks working code.
"""

from ._info import FUNCTIONS
from .graph import DiGraph, Graph, RustnxGraph, from_networkx, to_networkx

__all__ = [
    "FUNCTIONS",
    "DiGraph",
    "Graph",
    "RustnxGraph",
    "enable",
    "from_networkx",
    "to_networkx",
]


def enable(fallback=True):
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
    """
    import networkx as nx

    priority = nx.config.backend_priority
    for key in ("algos", "generators"):
        others = [n for n in getattr(priority, key) if n not in ("rustnx", "networkx")]
        setattr(priority, key, ["rustnx", *others])
    if fallback:
        nx.config.fallback_to_nx = True


__version__ = "0.1.0a3"
