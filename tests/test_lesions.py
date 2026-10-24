"""Lesion-wise evaluation against Panoptica and the BraTS 2023 evaluation code (fixtures/)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from exacta import lesions


@pytest.mark.parametrize("matching", ["greedy", "hungarian", "merge", "overlap"])
@pytest.mark.parametrize("case", ["perfect", "missed", "false_positive", "empty"])
def test_default_lesion_distance_penalty(device, matching, case):
    if matching == "hungarian":
        pytest.importorskip("scipy.optimize")
    target = torch.zeros((1, 1, 3, 3, 3), dtype=torch.bool, device=device)
    target[0, 0, 1, 1, 1] = True
    pred = target.clone()
    if case in ("missed", "empty"):
        pred.zero_()
    if case in ("false_positive", "empty"):
        target.zero_()
    got = lesions.evaluate(pred, target, matching=matching, connectivity=1,
                           input_format="one-hot", spacing=1.0).lesion_hd95.item()
    if case == "empty":
        assert math.isnan(got)
    else:
        assert got == (0.0 if case == "perfect" else math.inf)


@pytest.mark.parametrize("matching", ["greedy", "hungarian", "merge", "overlap"])
def test_ignored_lesions_do_not_change_empty_scores(device, matching):
    if matching == "hungarian":
        pytest.importorskip("scipy.optimize")
    target = torch.zeros((1, 1, 3, 3, 3), dtype=torch.bool, device=device)
    target[0, 0, 1, 1, 1] = True
    kw = dict(matching=matching, connectivity=1, input_format="one-hot", min_target_volume=1)
    for pred in (target, torch.zeros_like(target)):
        got = lesions.evaluate(pred, target, **kw)
        assert (got.tp.item(), got.fp.item(), got.fn.item()) == (0, 0, 0)
        for name in ("f1", "sq_iou", "sq_dice", "pq_iou", "pq_dice"):
            assert getattr(got, name).isnan().all(), name

FIXTURES = Path(__file__).parent / "fixtures"
CASES = np.load(FIXTURES / "lesion_cases.npz")
MATCHING = {"naive": "greedy", "bipartite": "hungarian", "merge": "merge"}
PANOPTICA_KEYS = {
    "tp": "tp",
    "fp": "fp",
    "fn": "fn",
    "prec": "precision",
    "rec": "recall",
    "rq": "f1",
    "sq": "sq_iou",
    "sq_dsc": "sq_dice",
    "pq": "pq_iou",
    "pq_dsc": "pq_dice",
}


def _panoptica_results():
    return json.loads((FIXTURES / "panoptica.json").read_text())["results"]


def _one_hot(mask):
    return torch.from_numpy(np.ascontiguousarray(mask))[None, None]


@pytest.mark.parametrize(
    "ref", _panoptica_results(), ids=lambda r: "-".join(str(v) for v in list(r.values())[:5])
)
def test_matches_panoptica(ref, device):
    pred, target = CASES[f"{ref['case']}_pred"], CASES[f"{ref['case']}_target"]
    # Panoptica labels 3-D images with cc3d (full connectivity) and 2-D ones with SciPy (faces)
    connectivity = pred.ndim if ref["backend"] == "cc3d" else 1
    got = lesions.evaluate(
        _one_hot(pred).to(device),
        _one_hot(target).to(device),
        matching=MATCHING[ref["matcher"]],
        connectivity=connectivity,
        score="iou" if ref["metric"] == "IOU" else "dice",
        threshold=ref["threshold"],
        input_format="one-hot",
    )
    for theirs, ours in PANOPTICA_KEYS.items():
        expected = math.nan if ref[theirs] is None else ref[theirs]
        value = float(getattr(got, ours)[0, 0])
        assert (math.isnan(value) and math.isnan(expected)) or value == pytest.approx(expected, rel=1e-12), ours


def test_fixture_covers_disagreeing_matchers():
    # The fixture is only useful if the strategies actually disagree on some cases.
    results = _panoptica_results()
    by_config = {}
    for r in results:
        key = (r["case"], r["metric"], r["threshold"], r["backend"])
        by_config.setdefault(key, {})[r["matcher"]] = (r["tp"], r["fp"], r["fn"], r["sq"])
    differing = [k for k, v in by_config.items() if len(set(v.values())) > 1]
    assert len(differing) >= 5


BRATS_KEYS = {
    "Num_TP": "tp",
    "Num_FP": "fp",
    "Num_FN": "fn",
    "LesionWise_Score_Dice": "lesion_dice",
    "LesionWise_Score_HD95": "lesion_hd95",
}


def _brats_results():
    return json.loads((FIXTURES / "brats2023.json").read_text())["results"]


@pytest.mark.parametrize(
    "ref", _brats_results(), ids=lambda r: f"{r['case']}-{r['challenge']}-{r['region']}-{r['spacing'][0]}"
)
def test_matches_brats2023(ref, device):
    pred, target = (
        lesions.brats_regions(torch.from_numpy(CASES[f"{ref['case']}_{role}"]).long()[None].to(device))
        for role in ("pred", "target")
    )
    # BraTS reads the voxel size from the NIfTI header, which stores it in float32 (1.2 comes back
    # as 1.2000000476837158); pass the same values.
    spacing = [float(np.float32(h)) for h in ref["spacing"]]
    got = lesions.brats2023(pred, target, challenge=ref["challenge"], spacing=spacing)
    region = ["WT", "TC", "ET"].index(ref["region"])
    for theirs, ours in BRATS_KEYS.items():
        assert float(getattr(got, ours)[0, region]) == pytest.approx(ref[theirs], rel=1e-12), ours


def test_label_maps_score_each_class_separately():
    p, t = CASES["binary4_pred"], CASES["binary4_target"]
    q, u = CASES["binary5_pred"], CASES["binary5_target"]
    pred = np.where(p, 1, np.where(q & ~p, 2, 0))[None]
    target = np.where(t, 1, np.where(u & ~t, 2, 0))[None]
    kw = dict(matching="greedy", connectivity=1)
    got = lesions.evaluate(pred, target, num_classes=3, include_background=False, **kw)
    for k, (a, b) in enumerate([(pred[0] == 1, target[0] == 1), (pred[0] == 2, target[0] == 2)]):
        alone = lesions.evaluate(a[None, None], b[None, None], input_format="one-hot", **kw)
        assert got.f1[0, k] == alone.f1[0, 0] and got.sq_iou[0, k] == alone.sq_iou[0, 0]


def test_small_target_lesions_are_ignored():
    target = np.zeros((1, 1, 20, 20), bool)
    target[0, 0, 2:4, 2:4] = True  # 4 voxels
    target[0, 0, 10:15, 10:15] = True  # 25 voxels
    pred = target.copy()
    pred[0, 0, 18:20, 18:20] = True  # a false positive
    kw = dict(matching="overlap", connectivity=2, input_format="one-hot")
    full = lesions.evaluate(pred, target, **kw)
    assert (full.tp[0, 0], full.fp[0, 0], full.fn[0, 0]) == (2, 1, 0)
    big = lesions.evaluate(pred, target, min_target_volume=4, **kw)  # "at most 4" is dropped
    assert (big.tp[0, 0], big.fp[0, 0], big.fn[0, 0]) == (1, 1, 0)
    assert big.lesion_dice[0, 0] == pytest.approx(1 / 2)  # one perfect lesion and one false positive
    assert np.isnan(big.lesion_hd95[0, 0])  # needs spacing


def test_lesion_hd95_counts_misses_and_false_positives():
    target = np.zeros((1, 1, 16, 16, 16), bool)
    target[0, 0, 2:6, 2:6, 2:6] = True
    target[0, 0, 10:13, 10:13, 10:13] = True  # missed
    pred = np.zeros_like(target)
    pred[0, 0, 2:6, 2:6, 2:6] = True  # perfect: HD95 0
    pred[0, 0, 14:16, 0:2, 14:16] = True  # false positive
    got = lesions.evaluate(
        pred, target, matching="overlap", connectivity=3, spacing=1.0, missed_distance=100.0,
        input_format="one-hot",
    )  # fmt: skip
    assert got.lesion_hd95[0, 0] == pytest.approx((0 + 100 + 100) / 3)


def test_rejects_bad_arguments():
    x = np.zeros((1, 1, 4, 4), bool)
    kw = dict(connectivity=1, input_format="one-hot")
    with pytest.raises(ValueError, match="matching must be"):
        lesions.evaluate(x, x, matching="panoptica", **kw)
    with pytest.raises(ValueError, match="score must be"):
        lesions.evaluate(x, x, matching="greedy", score="hd95", **kw)
    with pytest.raises(ValueError, match="only with matching='overlap'"):
        lesions.evaluate(x, x, matching="greedy", dilation=1, dilation_connectivity=1, **kw)
    with pytest.raises(ValueError, match="needs dilation_connectivity"):
        lesions.evaluate(x, x, matching="overlap", dilation=2, **kw)
    with pytest.raises(ValueError, match="challenge must be"):
        lesions.brats2023(x, x, challenge="GBM", spacing=1.0)
