"""Reference results of the BraTS 2023 lesion-wise evaluation on the brats cases of
lesion_cases.npz (brats2023.json).

The reference is metrics.py of https://github.com/rachitsaluja/BraTS-2023-Metrics, with its
vendored surface_distance package. It uses DataFrame.append (removed in pandas 2) and np.Inf
(removed in NumPy 2), so it runs in an environment of its own:

    uv venv -p 3.10 /tmp/brats
    uv pip install -p /tmp/brats/bin/python "numpy<2" "scipy<1.14" "pandas<2" connected-components-3d nibabel
    /tmp/brats/bin/python tests/fixtures/make_brats_fixture.py /path/to/BraTS-2023-Metrics

Each case is written as NIfTI with an isotropic and an anisotropic voxel size, and evaluated for
the GLI settings (dilation 3, 50 mm^3 threshold) and the MET ones (dilation 1, 2 mm^3).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).parent
SPACINGS = {"iso": (1.0, 1.0, 1.0), "aniso": (1.2, 0.9, 1.1)}
KEYS = ("Num_TP", "Num_FP", "Num_FN", "LesionWise_Score_Dice", "LesionWise_Score_HD95")


def main() -> None:
    sys.path.insert(0, sys.argv[1])
    import metrics  # BraTS-2023-Metrics/metrics.py

    cases = np.load(HERE / "lesion_cases.npz")
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in sorted({k.rsplit("_", 1)[0] for k in cases.files if k.startswith("brats")}):
            for spacing_name, spacing in SPACINGS.items():
                files = {}
                for role in ("pred", "target"):
                    path = Path(tmp) / f"{name}_{spacing_name}_{role}.nii.gz"
                    nib.save(nib.Nifti1Image(cases[f"{name}_{role}"], np.diag([*spacing, 1.0])), path)
                    files[role] = str(path)
                for challenge in ("BraTS-GLI", "BraTS-MET"):
                    table = metrics.get_LesionWiseResults(files["pred"], files["target"], challenge)
                    for _, row in table.iterrows():
                        results.append(
                            dict(
                                case=name,
                                spacing=list(spacing),
                                challenge=challenge.split("-")[1],
                                region=row["Labels"],
                                **{k: float(row[k]) for k in KEYS},
                            )
                        )
    fixture = {"source": "BraTS-2023-Metrics", "results": results}
    (HERE / "brats2023.json").write_text(json.dumps(fixture, indent=1))


if __name__ == "__main__":
    main()
