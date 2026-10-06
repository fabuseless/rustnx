"""rustnx's float settings in ``nx.config.backends.rustnx``.

A few functions have two ways to compute their floats:

- fast (the default): Rust arithmetic that adds numbers up in a different
  order from NetworkX, so the last bits of a float can differ (around 1e-16
  to 1e-15 relative);
- exact: the same additions in the same order as NetworkX (or NetworkX's own
  NumPy/SciPy code on identical inputs), so floats are bit for bit equal.

Which one runs is decided per function:

1. ``nx.config.backends.rustnx.exact_floats_overrides[name]``, if set;
2. otherwise ``nx.config.backends.rustnx.exact_floats`` (default False).

Every other rustnx function is always bit for bit identical to NetworkX.
"""

import logging

import networkx as nx

logger = logging.getLogger("rustnx")

# Functions with a fast and an exact way to compute floats: what each mode
# does, and how far the fast mode's floats can be from NetworkX's.
FLOAT_FUNCTIONS = {
    "betweenness_centrality": {
        "fast": "parallel blocks of sources, block sums added at the end",
        "exact": "parallel, but each source's share added in NetworkX's order",
        "differs_by": "about 1e-15 relative",
    },
    "edge_betweenness_centrality": {
        "fast": "parallel blocks of sources, block sums added at the end",
        "exact": "parallel, but each source's share added in NetworkX's order",
        "differs_by": "about 1e-15 relative",
    },
    "pagerank": {
        "fast": "power iteration in Rust; multigraphs run in Rust too",
        "exact": (
            "Rust builds the sparse matrix, NetworkX's own SciPy code does the "
            "arithmetic; multigraphs run in NetworkX"
        ),
        "differs_by": "about 1e-16 relative",
    },
}


def _settings():
    try:
        return nx.config.backends.rustnx
    except AttributeError:
        return None


def resolve(name):
    """``(exact, set_by)`` for ``name``: whether it computes exact floats,
    and which setting decided that."""
    cfg = _settings()
    if cfg is None:
        return False, "default"
    overrides = getattr(cfg, "exact_floats_overrides", None) or {}
    if name in overrides:
        return bool(overrides[name]), f"exact_floats_overrides[{name!r}]"
    return bool(getattr(cfg, "exact_floats", False)), "the global exact_floats setting"


def exact_floats(name):
    """Whether ``name`` must return floats bit for bit equal to NetworkX's."""
    return resolve(name)[0]


def _describe(name, exact, set_by):
    info = FLOAT_FUNCTIONS[name]
    if exact:
        return (
            f"rustnx {name}: exact floats (set by {set_by}): {info['exact']}. "
            "Results are bit for bit NetworkX's."
        )
    return (
        f"rustnx {name}: fast floats (set by {set_by}): {info['fast']}. Floats "
        f"may differ from NetworkX's in the last bits ({info['differs_by']}). "
        "For NetworkX's exact numbers set "
        f"nx.config.backends.rustnx.exact_floats_overrides[{name!r}] = True, "
        "or nx.config.backends.rustnx.exact_floats = True for every function."
    )


_announced = set()


def note(name):
    """Resolve ``name``'s setting, log it, and return whether it is exact.

    Every call is logged at DEBUG on the ``rustnx`` logger. With
    ``nx.config.backends.rustnx.verbose`` on, the first call of each function
    under each setting is also logged at INFO, with a handler printing to
    stderr if logging isn't configured.
    """
    exact, set_by = resolve(name)
    if getattr(_settings(), "verbose", False):
        key = (name, exact, set_by)
        if key not in _announced:
            _announced.add(key)
            _ensure_handler()
            logger.info(_describe(name, exact, set_by))
            return exact
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(_describe(name, exact, set_by))
    return exact


def _ensure_handler():
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


def float_settings():
    """The functions with a fast and an exact mode, and the setting that
    applies to each now: ``{name: {"exact": bool, "set_by": str, "fast": str,
    "exact_mode": str, "differs_by": str}}``. Overrides that name functions
    without a float setting (typos) are listed under ``"unknown_overrides"``.
    """
    out = {}
    for name, info in FLOAT_FUNCTIONS.items():
        exact, set_by = resolve(name)
        out[name] = {
            "exact": exact,
            "set_by": set_by,
            "fast": info["fast"],
            "exact_mode": info["exact"],
            "differs_by": info["differs_by"],
        }
    cfg = _settings()
    overrides = (getattr(cfg, "exact_floats_overrides", None) or {}) if cfg else {}
    unknown = sorted(n for n in overrides if n not in FLOAT_FUNCTIONS)
    if unknown:
        out["unknown_overrides"] = unknown
    return out


def explain_floats():
    """:func:`float_settings` as a readable report."""
    settings = float_settings()
    lines = ["rustnx float settings (every other function is always exact):"]
    for name in FLOAT_FUNCTIONS:
        s = settings[name]
        mode = "exact" if s["exact"] else f"fast (may differ by {s['differs_by']})"
        lines.append(f"  {name}: {mode}; set by {s['set_by']}")
    if "unknown_overrides" in settings:
        lines.append(
            "  ignored, no float setting: exact_floats_overrides for "
            + ", ".join(settings["unknown_overrides"])
        )
    return "\n".join(lines)


def inexact_message(name, what):
    """Why rustnx declines an input it can't compute exactly while ``name``
    is set to exact floats."""
    return (
        f"rustnx's {what} can differ from NetworkX in the last bits of a float, "
        f"and {name} is set to exact floats; to allow it, set "
        f"nx.config.backends.rustnx.exact_floats_overrides[{name!r}] = False"
    )
