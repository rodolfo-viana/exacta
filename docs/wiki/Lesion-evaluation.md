# Lesion evaluation and components

`exacta.lesions` treats connected components as lesions and matches predicted lesions to target
lesions. Matching conventions affect detection counts and segmentation quality, so connectivity
and matching are explicit required arguments.

| Matching | Behavior |
|---|---|
| `greedy` | One-to-one, best qualifying score first |
| `hungarian` | Maximum-total-score one-to-one assignment among qualifying pairs; requires SciPy |
| `merge` | Allows additional predicted pieces when their union improves a matched target's score |
| `overlap` | Many-to-many matching after target dilation, following BraTS-style evaluation |

For 3D data, connectivity 1 means faces (6 neighbors), 2 includes edges (18), and 3 includes
corners (26). Choose `score="iou"` or `"dice"` for thresholded matching; the default threshold
comparison is inclusive, with `strict=True` for an exclusive comparison.

```python
import numpy as np
from exacta import components, lesions

target = np.zeros((1, 12, 12, 12), dtype=np.int64)
target[:, 2:5, 2:5, 2:5] = 1
prediction = np.roll(target, 1, axis=1)
scores = lesions.evaluate(prediction, target, num_classes=2, include_background=False,
                          connectivity=3, matching="greedy", score="iou", threshold=0.5,
                          spacing=(1.0, 1.0, 1.0))
print(scores.tp, scores.fp, scores.fn, scores.f1)
print(scores.sq_iou, scores.pq_iou, scores.lesion_dice, scores.lesion_hd95)

labels, count = components.label(prediction[0] > 0, connectivity=3)
largest = components.largest(prediction[0] > 0, connectivity=3)
cleaned = components.remove_small(prediction[0] > 0, 5, connectivity=3)
```

`tp`, `fp`, `fn`, precision, recall, F1, SQ/PQ and lesion scores are per case and selected
class. `lesion_hd95` requires spacing; without it, that field is NaN. `missed_distance`
controls penalties for missed lesions and false positives. `min_target_volume` excludes
target lesions at or below the threshold, in physical volume units when spacing is supplied
(for example, mm³ in 3D), or voxel counts otherwise. Predictions matched to excluded lesions
are not counted as false positives; merely touching a lesion need not satisfy the chosen
matching rule.

Use `lesions.brats_regions` and `lesions.brats2023(..., challenge="GLI", spacing=...)` for the
implemented BraTS 2023 convention. Check the dataset's label encoding and challenge settings
against the [source documentation](https://github.com/rodolfo-viana/exacta/blob/main/src/exacta/lesions.py)
before applying it to another challenge year.
