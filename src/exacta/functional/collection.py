"""Evaluate several segmentation metrics with shared volume passes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from exacta._inputs import InputFormat, like_input, to_torch, validate
from exacta.functional import overlap, surface as sf

MetricSelection = Sequence[str] | Mapping[str, Mapping[str, Any]]

_DEFAULTS: dict[str, dict[str, Any]] = {
    "dice": {"both_empty": math.nan},
    "iou": {"both_empty": math.nan},
    "sensitivity": {"zero_division": math.nan},
    "precision": {"zero_division": math.nan},
    "specificity": {"zero_division": math.nan},
    "volume_similarity": {"zero_division": math.nan, "convention": "taha"},
    "relative_volume_difference": {"zero_division": math.nan, "absolute": False},
    "hausdorff_distance": {"percentile": 100.0, "combine": "max", "surface": "voxels",
                           "both_empty": math.nan, "one_empty": math.inf},
    "hd95": {"combine": "max", "surface": "voxels", "both_empty": math.nan, "one_empty": math.inf},
    "assd": {"combine": "pooled", "surface": "voxels", "both_empty": math.nan, "one_empty": math.inf},
    "nsd": {"tolerance": None, "surface": "voxels", "both_empty": math.nan, "one_empty": 0.0},
}


def _selection(metrics: MetricSelection) -> dict[str, dict[str, Any]]:
    if isinstance(metrics, (str, bytes)) or not isinstance(metrics, (Sequence, Mapping)):
        raise TypeError("metrics must be a sequence of names or a mapping of names to options")
    result = {}
    for name in metrics:
        if name not in _DEFAULTS:
            raise ValueError(f"unknown metric {name!r}; choose from {tuple(_DEFAULTS)}")
        if name in result:
            raise ValueError(f"duplicate metric {name!r}")
        options = metrics[name] if isinstance(metrics, Mapping) else {}
        if not isinstance(options, Mapping):
            raise TypeError(f"options for {name!r} must be a mapping")
        unknown = options.keys() - _DEFAULTS[name].keys()
        if unknown:
            raise TypeError(f"unsupported options for {name!r}: {sorted(unknown)}")
        result[name] = dict(_DEFAULTS[name], **options)
    if not result:
        raise ValueError("metrics must not be empty")
    return result


def evaluate(
    pred: Any, target: Any, *, metrics: MetricSelection,
    spacing: Any = None, input_format: InputFormat = "index", num_classes: int | None = None,
    include_background: bool = True, backend: sf.Backend = "auto",
) -> dict[str, Any]:
    """Return named (B, K) arrays, sharing confusion counts and compatible surface distances.

    ``metrics`` is a sequence of functional names, or a mapping to each metric's options,
    e.g. ``{"dice": {}, "hd95": {"combine": "pooled"}, "nsd": {"tolerance": 1.0}}``.
    Options and defaults match the individual functions. Input layout, class selection and
    spacing are common; surface representations may differ. ``backend`` selects the distance
    transform, as for individual surface metrics. Spacing is required when requesting any
    surface metric. Results use the prediction's array library. No input or volume cache is
    retained after this call, so subsequent input mutation cannot return stale scores.
    """
    requests = _selection(metrics)
    counts_names = [n for n, options in requests.items() if "surface" not in options]
    if counts_names:
        counts, seg = overlap._confusion_counts_and_seg(
            pred, target, input_format, num_classes, include_background,
        )
        results = {n: overlap._score_counts(n, counts, **requests[n]) for n in counts_names}
    else:
        seg = validate(to_torch(pred), to_torch(target), input_format=input_format,
                       num_classes=num_classes, include_background=include_background)
        results = {}
    groups: dict[str, list[tuple[str, Any]]] = {}
    for name, options in requests.items():
        if name in counts_names:
            continue
        score = _surface_score(name, options, seg.classes)
        groups.setdefault(options["surface"], []).append((name, (score, options["both_empty"], options["one_empty"])))
    for representation, group in groups.items():
        values = sf._per_case_many(seg, spacing, [spec for _, spec in group],
                                   surface=representation, backend=backend)  # type: ignore[arg-type]
        for j, (name, _) in enumerate(group):
            results[name] = values[..., j]
    return {name: like_input(results[name], pred) for name in requests}


def _surface_score(name: str, options: dict[str, Any], classes: range):
    if name == "nsd":
        if options["tolerance"] is None:
            raise ValueError("nsd requires tolerance")
        tau = sf._tolerances(options["tolerance"], classes)

        def score(pt, tp, k):
            within_p, total_p = sf._within(pt, tau[k])
            within_t, total_t = sf._within(tp, tau[k])
            return (within_p + within_t) / (total_p + total_t)
        return score
    combine = options["combine"]
    sf._check_combine(combine)
    if name == "assd":
        return lambda pt, tp, k: sf._combine(pt, tp, combine, sf._mean)
    percentile = 95.0 if name == "hd95" else options["percentile"]
    if not 0 <= percentile <= 100:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    return lambda pt, tp, k: sf._combine(pt, tp, combine, lambda x: sf._percentile(x, percentile))
