"""Reporting signatures: one string naming every convention behind a reported score.

A score's name does not fix its value. HD95 depends on the surface representation, on how the
two directed distances are joined and on what an empty mask scores; NSD also depends on how
distances equal to the tolerance are compared. :func:`signature` writes all of these choices,
with Exacta's defaults filled in, plus the Exacta version, in a short string meant to be reported
next to the number. :func:`parse_signature` reads it back into options for
:func:`exacta.functional.evaluate`, so a reader can recompute the score under the same
definition::

    >>> signature("hd95")                                      # doctest: +SKIP
    'hd95[both_empty=nan;combine=max;one_empty=inf;surface=voxels]@exacta-0.1.1'
    >>> name, options, version = parse_signature(_)            # doctest: +SKIP
    >>> exacta.functional.evaluate(pred, target, metrics={name: options}, ...)  # doctest: +SKIP

Format: ``name[key=value;...]@exacta-<version>``, keys sorted. Values are ``true``/``false``,
``none``, integers, floats written as Python's ``repr`` (``nan``, ``inf`` and ``-inf``
included, so they read back bit for bit), bare words, or parenthesised comma-separated lists
(per-class tolerances). The optional ``aggregate`` key records dataset pooling
(``"cases"`` or ``"pooled"``, as in :class:`exacta.metrics.MetricCollection`).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from exacta.functional.collection import MetricSelection, _selection

_WORD = re.compile(r"^[A-Za-z_][A-Za-z0-9_.+-]*$")
_SIGNATURE = re.compile(r"^(?P<name>[a-z0-9_]+)\[(?P<body>[^\]]*)\]@exacta-(?P<version>\S+)$")
# Options without a default that evaluate() requires; a signature without them names no score.
_REQUIRED = {"nsd": "tolerance", "betti_error": "connectivity"}


def signature(metric: str, *, aggregate: str | None = None, **options: Any) -> str:
    """Signature of ``metric`` evaluated with ``options`` (Exacta's defaults fill the rest).

    ``metric`` and ``options`` are a name and its options as accepted by
    :func:`exacta.functional.evaluate`; unknown names or options raise as they do there.
    """
    from exacta import __version__

    resolved = _selection({metric: options})[metric]
    for name, option in _REQUIRED.items():
        if metric == name and resolved[option] is None:
            raise ValueError(f"{metric} requires {option}")
    if aggregate is not None:
        if aggregate not in ("cases", "pooled"):
            raise ValueError(f"aggregate must be 'cases' or 'pooled', got {aggregate!r}")
        resolved["aggregate"] = aggregate
    body = ";".join(f"{key}={_encode(key, resolved[key])}" for key in sorted(resolved))
    return f"{metric}[{body}]@exacta-{__version__}"


def signatures(metrics: MetricSelection, *, aggregate: str | None = None) -> dict[str, str]:
    """Signature of every metric of a selection, as accepted by :func:`exacta.functional.evaluate`."""
    selected = _selection(metrics)
    given = metrics if isinstance(metrics, Mapping) else {name: {} for name in selected}
    return {name: signature(name, aggregate=aggregate, **dict(given[name])) for name in selected}


def parse_signature(text: str) -> tuple[str, dict[str, Any], str]:
    """Read a signature back as ``(metric, options, version)``.

    ``options`` excludes ``aggregate``, which is not an option of the metric itself; it is
    returned under the key ``"aggregate"`` only when present, for the caller to pass on.
    """
    match = _SIGNATURE.match(text.strip())
    if match is None:
        raise ValueError(f"not an Exacta signature: {text!r}")
    options: dict[str, Any] = {}
    body = match["body"]
    for item in _split(body) if body else []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"malformed option {item!r} in signature {text!r}")
        options[key] = _decode(value)
    return match["name"], options, match["version"]


def _encode(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        if not _WORD.match(value) or value in ("true", "false", "none", "nan", "inf"):
            raise ValueError(f"option {key}={value!r} cannot be written in a signature")
        return value
    if isinstance(value, (list, tuple)):
        return "(" + ",".join(_encode(key, float(v) if isinstance(v, int) and not isinstance(v, bool) else v)
                              for v in value) + ")"
    try:  # 0-d arrays and tensors, e.g. a tolerance given as a tensor
        return _encode(key, value.tolist())
    except AttributeError:
        raise ValueError(f"option {key}={value!r} cannot be written in a signature") from None


def _decode(text: str) -> Any:
    if text.startswith("(") and text.endswith(")"):
        inner = text[1:-1]
        return tuple(_decode(v) for v in inner.split(",")) if inner else ()
    if text in ("true", "false"):
        return text == "true"
    if text == "none":
        return None
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    try:
        value = float(text)
    except ValueError:
        return text
    if math.isnan(value) and text != "nan":
        return text
    return value


def _split(body: str) -> list[str]:
    """Split ``a=1;b=(1,2)`` on semicolons outside parentheses."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(body):
        depth += ch == "("
        depth -= ch == ")"
        if ch == ";" and depth == 0:
            parts.append(body[start:i])
            start = i + 1
    parts.append(body[start:])
    return parts
