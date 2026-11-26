import math

import numpy as np
import pytest
import torch

import exacta
import exacta.functional as EF
from _data import pair
from exacta.conventions import parse_signature, signature, signatures
from exacta.functional.collection import _DEFAULTS
from exacta.metrics import MetricCollection

# Options needed to evaluate each metric on the test inputs (nsd has no default tolerance).
REQUIRED = {"nsd": {"tolerance": 1.5}, "betti_error": {"connectivity": 1}}
AREA = {"predicted_area", "target_area", "area_error"}  # 2D inputs only


@pytest.mark.parametrize("metric", sorted(_DEFAULTS))
def test_signature_round_trips_and_recomputes_the_same_score(metric):
    options = REQUIRED.get(metric, {})
    text = signature(metric, **options)
    assert text.startswith(f"{metric}[") and text.endswith(f"@exacta-{exacta.__version__}")
    name, parsed, version = parse_signature(text)
    assert (name, version) == (metric, exacta.__version__)
    assert set(parsed) == set(_DEFAULTS[metric])  # every convention is written, defaults included
    shape, spacing = ((40, 36), (0.7, 0.9)) if metric in AREA else ((14, 16, 18), (1.5, 0.8, 0.8))
    pred, target = pair(shape, 3, seed=3)
    kw = dict(num_classes=3, spacing=spacing)
    expected = EF.evaluate(pred, target, metrics={metric: options}, **kw)[metric]
    got = EF.evaluate(pred, target, metrics={metric: parsed}, **kw)[metric]
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("value", [0.1, 1e-06, 1 / 3, math.inf, -math.inf, 5e-324, 1.7976931348623157e308])
def test_floats_round_trip_bit_for_bit(value):
    _, parsed, _ = parse_signature(signature("hd95", one_empty=value))
    assert parsed["one_empty"] == value
    _, parsed, _ = parse_signature(signature("dice", both_empty=math.nan))
    assert math.isnan(parsed["both_empty"])


def test_per_class_tolerances_and_strict_ties():
    text = signature("nsd", tolerance=[1.0, 2.5, torch.tensor(0.75).item()], tie_rtol=0)
    assert "tolerance=(1.0,2.5,0.75)" in text and "tie_rtol=0" in text
    _, parsed, _ = parse_signature(text)
    assert parsed["tolerance"] == (1.0, 2.5, 0.75) and parsed["tie_rtol"] == 0


def test_signature_rejects_what_evaluate_rejects():
    with pytest.raises(ValueError, match="unknown metric"):
        signature("dicee")
    with pytest.raises(TypeError, match="unsupported options"):
        signature("dice", tolerance=1.0)
    with pytest.raises(ValueError, match="requires tolerance"):
        signature("nsd")
    with pytest.raises(ValueError, match="requires connectivity"):
        signature("betti_error")
    with pytest.raises(ValueError, match="aggregate"):
        signature("dice", aggregate="mean")
    with pytest.raises(ValueError, match="cannot be written"):
        signature("hd95", surface="voxels;x=1")
    with pytest.raises(ValueError, match="not an Exacta signature"):
        parse_signature("hd95(combine=max)")


def test_collection_signatures_record_aggregation():
    pooled = MetricCollection(metrics={"dice": {}, "iou": {}}, num_classes=3, aggregate="pooled")
    assert pooled.signatures()["dice"] == signature("dice", aggregate="pooled")
    cases = MetricCollection(metrics=["hd95"], num_classes=3)
    assert "aggregate=cases" in cases.signatures()["hd95"]
    assert signatures(["dice", "hd95"]) == {"dice": signature("dice"), "hd95": signature("hd95")}
