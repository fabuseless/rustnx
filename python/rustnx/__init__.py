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
from .graph import RustnxGraph, from_networkx, to_networkx

__all__ = ["FUNCTIONS", "RustnxGraph", "from_networkx", "to_networkx"]
__version__ = "0.1.0"
