"""A reporting signature recomputes its score bit for bit: score, signature, parse, recompute.

For every surface representation (voxels, surfels, mesh), every distance score (HD, HD95 and ASSD
with each way of joining the two directions, NSD with the default tie margin and with
``tie_rtol=0``), on 2-D and 3-D masks with dyadic, decimal and anisotropic spacings, the score is
computed with the public function, its signature is written, and the call is rebuilt from the
signature alone (``parse_signature`` and ``functional.evaluate``). The recomputed score must equal
the first in every bit. A signature names the conventions of a score, not its inputs, so the masks,
the spacing and the number of classes are passed to both calls.

The masks come from integer arithmetic, with no random numbers, and the cases are those of
``scripts/portability_digest.py``, which the Portability CI runs on every platform to check that the
signatures and the recomputed scores are identical across them.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

import exacta
import exacta.functional as F
from exacta.conventions import parse_signature, signature
from exacta.functional.collection import _DEFAULTS

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "portability_digest.py"
_spec = importlib.util.spec_from_file_location("portability_digest", SCRIPT)
digest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(digest)

CASES = digest.roundtrip_cases()
FUNCTIONS = {"hausdorff_distance": F.hausdorff_distance, "hd95": F.hd95, "assd": F.assd, "nsd": F.nsd}


def _hex(scores) -> list[str]:
    """Every score as the hexadecimal float of its bits; float.hex is exact and distinguishes -0.0."""
    return [value.hex() for value in scores.flatten().tolist()]


@pytest.mark.parametrize("case", CASES, ids=[case.label for case in CASES])
def test_signature_recomputes_the_score_bit_for_bit(case):
    score = FUNCTIONS[case.metric](case.pred, case.target, **case.inputs, **case.options)
    assert score.dtype.is_floating_point and score.isfinite().all()  # a NaN or inf pair would prove little

    text = signature(case.metric, **case.options)
    name, options, version = parse_signature(text)  # from here on, the conventions come from the text only
    assert (name, version) == (case.metric, exacta.__version__)
    assert set(options) == set(_DEFAULTS[name])
    assert options["surface"] == case.surface

    again = F.evaluate(case.pred, case.target, metrics={name: options}, **case.inputs)[name]
    assert again.dtype == score.dtype and again.shape == score.shape
    assert _hex(again) == _hex(score)
    assert again.numpy().tobytes() == score.numpy().tobytes()


def test_cases_cover_the_grid():
    # dimensions x spacings x surfaces x (3 distances x 3 joins + NSD with 2 tie margins)
    assert len(CASES) == 2 * 4 * 3 * 11
    assert {case.ndim for case in CASES} == {2, 3}
    assert {case.surface for case in CASES} == {"voxels", "surfels", "mesh"}
    assert len({case.label for case in CASES}) == len(CASES)
    for case in CASES:
        assert len(case.inputs["spacing"]) == case.ndim == case.pred.ndim - 1
        assert case.pred.any() and case.target.any() and not case.pred.all() and not case.target.all()


def test_nsd_cases_include_distances_that_tie_with_the_tolerance():
    """Where the default tie margin and ``tie_rtol=0`` give different scores, the round trip checks
    that the signature kept the margin; if no case differed, it would not."""
    key = lambda case: (case.ndim, case.surface, case.inputs["spacing"])  # noqa: E731
    nsd = [case for case in CASES if case.metric == "nsd"]
    default = {key(c): c for c in nsd if "tie_rtol" not in c.options}
    strict = {key(c): c for c in nsd if c.options.get("tie_rtol") == 0}
    assert default.keys() == strict.keys() and len(default) == len(nsd) // 2
    sensitive = set()
    for k, case in default.items():
        loose = F.nsd(case.pred, case.target, **case.inputs, **case.options)
        tight = F.nsd(case.pred, case.target, **case.inputs, **strict[k].options)
        if _hex(loose) != _hex(tight):
            sensitive.add(case.surface)
    assert sensitive == {"voxels", "surfels", "mesh"}


def test_signature_text_of_a_decimal_per_class_nsd_score():
    """Floats are written as Python's repr, a per-class tolerance as a parenthesised list, and
    ``tie_rtol=0`` as the integer it was given."""
    text = signature("nsd", surface="mesh", tolerance=(0.6, 0.9), tie_rtol=0)
    assert text == f"nsd[both_empty=nan;one_empty=0.0;surface=mesh;tie_rtol=0;tolerance=(0.6,0.9)]@exacta-{exacta.__version__}"
    _, options, _ = parse_signature(text)
    assert options["tolerance"] == (0.6, 0.9) and options["tie_rtol"] == 0
    assert math.isnan(options["both_empty"])
