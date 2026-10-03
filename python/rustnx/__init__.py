"""rustnx: a Rust-accelerated backend for NetworkX.

Keep writing normal NetworkX code and opt in with either::

    NETWORKX_BACKEND_PRIORITY=rustnx python my_script.py

or::

    import networkx as nx
    nx.config.backend_priority = ["rustnx"]

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

    Puts rustnx first in ``nx.config.backend_priority.algos``. NetworkX
    graphs are converted to rustnx (and the conversion cached) for supported
    functions; everything else runs in NetworkX as usual.

    ``"networkx"`` is removed from the priority list if present: NetworkX
    tries a backend that is both listed there and the input's own backend
    before any other, so listing it would stop rustnx from ever running on
    NetworkX graphs.

    With ``fallback=True`` (the default), rustnx graphs (``rustnx.Graph``)
    passed to functions rustnx doesn't implement are converted to NetworkX
    automatically instead of raising (``nx.config.fallback_to_nx``).
    Equivalent to ``NETWORKX_BACKEND_PRIORITY=rustnx`` plus
    ``NETWORKX_FALLBACK_TO_NX=1``.
    """
    import networkx as nx

    others = [n for n in nx.config.backend_priority.algos if n not in ("rustnx", "networkx")]
    nx.config.backend_priority.algos = ["rustnx", *others]
    if fallback:
        nx.config.fallback_to_nx = True


__version__ = "0.1.0a3"
