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

    With ``fallback=True`` (the default), rustnx graphs (``rustnx.Graph``)
    passed to functions rustnx doesn't implement are converted to NetworkX
    automatically, instead of raising. Equivalent to
    ``NETWORKX_BACKEND_PRIORITY=rustnx,networkx``.
    """
    import networkx as nx

    algos = [name for name in nx.config.backend_priority.algos if name != "rustnx"]
    algos.insert(0, "rustnx")
    if fallback and "networkx" not in algos:
        algos.append("networkx")
    nx.config.backend_priority.algos = algos
__version__ = "0.1.0a1"
