"""rustnx's settings in ``nx.config.backends.rustnx``."""

import networkx as nx


def exact_floats():
    """Whether rustnx must return floats bit for bit equal to NetworkX's
    (``nx.config.backends.rustnx.exact_floats``, default True)."""
    try:
        return bool(nx.config.backends.rustnx.exact_floats)
    except AttributeError:
        return True


def inexact_message(what):
    """The reason rustnx gives for declining an input whose floats can
    differ from NetworkX's while ``exact_floats`` is on."""
    return (
        f"rustnx's {what} can differ from NetworkX in the last bits of a float; "
        "to allow it, set nx.config.backends.rustnx.exact_floats = False"
    )
