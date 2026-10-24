"""Reference results of Panoptica on the binary cases of lesion_cases.npz (panoptica.json).

Panoptica 2.1.7 needs NumPy < 2.3, so this runs in an environment of its own; regenerate.sh
builds it and runs this script.

Every matcher x matching metric x threshold x connected-components backend is evaluated.
"""

from __future__ import annotations

import json
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import panoptica
from panoptica import (
    CCABackend,
    ConnectedComponentsInstanceApproximator,
    InputType,
    MaxBipartiteMatching,
    Metric,
    NaiveThresholdMatching,
    Panoptica_Evaluator,
)
from panoptica.instance_matcher import MaximizeMergeMatching

HERE = Path(__file__).parent
MATCHERS = {"naive": NaiveThresholdMatching, "bipartite": MaxBipartiteMatching, "merge": MaximizeMergeMatching}
KEYS = ("tp", "fp", "fn", "prec", "rec", "rq", "sq", "sq_dsc", "pq", "pq_dsc")


def evaluate(pred, target, matcher, backend):
    evaluator = Panoptica_Evaluator(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(cca_backend=backend),
        instance_matcher=matcher,
        verbose=False,
        log_times=False,
    )
    result = evaluator.evaluate(pred.astype(np.uint8), target.astype(np.uint8), verbose=False)["ungrouped"]
    values = result.to_dict()
    return {k: (None if values.get(k) is None else float(values[k])) for k in KEYS}


def main() -> None:
    warnings.simplefilter("ignore")
    cases = np.load(HERE / "lesion_cases.npz")
    results = []
    for name in sorted({k.rsplit("_", 1)[0] for k in cases.files if k.startswith("binary")}):
        pred, target = cases[f"{name}_pred"], cases[f"{name}_target"]
        for matcher in MATCHERS:
            for metric in ("IOU", "DSC"):
                for threshold in (0.5, 0.2):
                    for backend in ("cc3d", "scipy"):
                        algorithm = MATCHERS[matcher](getattr(Metric, metric), threshold)
                        values = evaluate(pred, target, algorithm, CCABackend[backend])
                        config = dict(
                            case=name, matcher=matcher, metric=metric, threshold=threshold, backend=backend
                        )
                        results.append({**config, **values})
    header = {"panoptica": version("panoptica")}
    (HERE / "panoptica.json").write_text(json.dumps({"source": header, "results": results}, indent=1))


if __name__ == "__main__":
    main()
