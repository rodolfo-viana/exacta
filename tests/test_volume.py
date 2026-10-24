"""Sensitivity, precision, specificity, volume similarity and relative volume difference."""

import math
import warnings

import numpy as np
import pytest
import torch

import exacta.functional as EF
import reference
from _data import pair
from exacta.metrics import Precision, RelativeVolumeDifference, Sensitivity, Specificity, VolumeSimilarity


def _data(seed=40):
    """Label maps with 5 classes: class 3 missing from the prediction of case 0, class 4 from
    everything, so every zero-division path is exercised."""
    pred, target = pair((20, 22, 24), 4, seed=seed)
    pred[0][pred[0] == 3] = 0
    return pred, target


def _per_case(fn, pred, target, **kw):
    out = np.empty((pred.shape[0], 5))
    for b, c in np.ndindex(out.shape):
        out[b, c] = fn(pred[b] == c, target[b] == c, **kw)
    return out


CASES = [
    (EF.sensitivity, reference.sensitivity, {}),
    (EF.precision, reference.precision, {}),
    (EF.specificity, reference.specificity, {}),
    (EF.volume_similarity, reference.volume_similarity, {"convention": "taha"}),
    (EF.volume_similarity, reference.volume_similarity, {"convention": "itk"}),
    (EF.relative_volume_difference, reference.relative_volume_difference, {"absolute": False}),
    (EF.relative_volume_difference, reference.relative_volume_difference, {"absolute": True}),
]


@pytest.mark.parametrize(("fn", "ref", "kw"), CASES, ids=lambda x: getattr(x, "__name__", str(x)))
@pytest.mark.parametrize("zero_division", [math.nan, 0.0, 1.0])
def test_matches_reference(device, fn, ref, kw, zero_division):
    pred, target = _data()
    expected = _per_case(ref, pred, target, zero_division=zero_division, **kw)
    got = fn(
        torch.from_numpy(pred).to(device),
        torch.from_numpy(target).to(device),
        num_classes=5,
        zero_division=zero_division,
        **kw,
    )
    assert got.device.type == device.type and got.dtype == torch.float64
    np.testing.assert_allclose(got.cpu().numpy(), expected, rtol=1e-12)
    if fn is not EF.specificity:  # its zero division (a target covering the image) is tested below
        assert np.isnan(expected).any() == math.isnan(zero_division)  # the data hits a zero division


@pytest.mark.parametrize(("fn", "ref", "kw"), CASES, ids=lambda x: getattr(x, "__name__", str(x)))
def test_one_hot_and_background(fn, ref, kw):
    pred, target = (torch.from_numpy(a) for a in _data())
    one_hot = lambda x: torch.nn.functional.one_hot(x, 5).movedim(-1, 1)  # noqa: E731
    by_index = fn(pred, target, num_classes=5, **kw)
    by_one_hot = fn(one_hot(pred), one_hot(target), input_format="one-hot", **kw)
    torch.testing.assert_close(by_one_hot, by_index, equal_nan=True)
    without_background = fn(pred, target, num_classes=5, include_background=False, **kw)
    torch.testing.assert_close(without_background, by_index[:, 1:], equal_nan=True)


def test_specificity_zero_division():
    full = torch.ones(1, 4, 5, 6, dtype=torch.long)  # class 1 everywhere: no negatives
    assert EF.specificity(full, full, num_classes=2)[0, 1].isnan()
    assert EF.specificity(full, full, num_classes=2, zero_division=1.0)[0, 1] == 1
    assert EF.specificity(full, full, num_classes=2)[0, 0] == 1  # class 0: all true negatives


@pytest.mark.parametrize("name", ["sensitivity", "precision", "specificity"])
def test_matches_medpy(name):
    medpy = pytest.importorskip("medpy.metric.binary")
    pred, target = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # MedPy's 0/0 under NumPy 2
        expected = _per_case(lambda p, t: getattr(medpy, name)(p, t), pred, target)
    # MedPy returns 0 on zero division with NumPy 1.x and NaN with NumPy 2 (see overlap.py)
    zero_division = math.nan if int(np.__version__.split(".")[0]) >= 2 else 0.0
    got = getattr(EF, name)(pred, target, num_classes=5, zero_division=zero_division)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_relative_volume_difference_matches_medpy():
    medpy = pytest.importorskip("medpy.metric.binary")
    pred, target = _data()
    expected = np.full((2, 5), np.nan)
    for b, c in np.ndindex(expected.shape):
        if (target[b] == c).any():  # MedPy raises on an empty target
            expected[b, c] = medpy.ravd(pred[b] == c, target[b] == c)
    np.testing.assert_allclose(EF.relative_volume_difference(pred, target, num_classes=5), expected, rtol=1e-12)


@pytest.mark.parametrize("name", ["sensitivity", "precision", "specificity"])
def test_matches_monai(name):
    monai = pytest.importorskip("monai.metrics")
    pred, target = _data()
    one_hot = lambda x: torch.nn.functional.one_hot(torch.from_numpy(x), 5).movedim(-1, 1)  # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = monai.get_confusion_matrix(one_hot(pred), one_hot(target), include_background=True)
        expected = monai.compute_confusion_matrix_metric(name, matrix)
    got = getattr(EF, name)(pred, target, num_classes=5)  # NaN on zero division, as MONAI
    np.testing.assert_allclose(got, expected.double().numpy(), rtol=1e-6)


def test_itk_volume_similarity_matches_simpleitk():
    sitk = pytest.importorskip("SimpleITK")
    pred, target = _data()
    expected = np.full((2, 5), np.nan)
    for b, c in np.ndindex(expected.shape):
        p, t = (pred[b] == c).astype(np.uint8), (target[b] == c).astype(np.uint8)
        if p.any() or t.any():
            measures = sitk.LabelOverlapMeasuresImageFilter()
            measures.Execute(sitk.GetImageFromArray(p), sitk.GetImageFromArray(t))  # (source, target)
            expected[b, c] = measures.GetVolumeSimilarity()
    got = EF.volume_similarity(pred, target, num_classes=5, convention="itk")
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_volume_similarity_conventions_are_related():
    pred, target = _data()
    taha = EF.volume_similarity(pred, target, num_classes=5)
    itk = EF.volume_similarity(pred, target, num_classes=5, convention="itk")
    np.testing.assert_allclose(taha, 1 - np.abs(itk) / 2, rtol=1e-12)


@pytest.mark.parametrize(
    ("metric_cls", "fn", "kw"),
    [
        (Sensitivity, EF.sensitivity, {}),
        (Precision, EF.precision, {"zero_division": 0.0}),
        (Specificity, EF.specificity, {}),
        (VolumeSimilarity, EF.volume_similarity, {"convention": "itk"}),
        (RelativeVolumeDifference, EF.relative_volume_difference, {"absolute": True}),
    ],
)
def test_metric_classes_accumulate_like_functional(device, metric_cls, fn, kw):
    batches = [tuple(torch.from_numpy(a).to(device) for a in _data(seed)) for seed in (41, 42)]
    metric = metric_cls(num_classes=5, include_background=False, **kw)
    for pred, target in batches:
        metric.update(pred, target)
    expected = torch.cat([fn(p, t, num_classes=5, include_background=False, **kw) for p, t in batches])
    torch.testing.assert_close(metric.compute().per_case, expected, equal_nan=True)


def test_rejects_unknown_convention():
    x = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="convention must be"):
        EF.volume_similarity(x, x, num_classes=2, convention="sitk")
