# Instance evaluation

These APIs were introduced in `0.0.1a6`. `exacta.instances` evaluates
**instance label maps**: each case is a `(B, H, W)` or `(B, D, H, W)` array of nonnegative
integers in which 0 is background and every other value is one instance. Touching instances
such as nuclei, cells or glands stay apart, unlike the connected components of
[lesion evaluation](https://github.com/rodolfo-viana/exacta/wiki/Lesion-evaluation), and an
instance need not be connected. Values need not be contiguous; they are renumbered in
increasing order, which also fixes the order that breaks ties.

```python
from exacta import instances

scores = instances.evaluate(pred_instances, target_instances)   # IoU > 0.5
print(scores.pq, scores.aji, scores.aji_plus, scores.seg)       # each (B,)
dsb = instances.dsb2018(pred_instances, target_instances)       # (B,)
```

## Scores

With $I(p,g)$ the voxels shared by predicted instance $p$ and target instance $g$ and
$U(p,g)=\lvert p\rvert+\lvert g\rvert-I(p,g)$:

| Field | Definition |
|---|---|
| `tp`, `fp`, `fn` | Matched pairs, unmatched predictions, unmatched targets |
| `precision`, `recall` | $\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$, $\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$ |
| `f1` | $\mathrm{TP}/(\mathrm{TP}+\tfrac12\mathrm{FP}+\tfrac12\mathrm{FN})$: detection (recognition) quality |
| `accuracy` | $\mathrm{TP}/(\mathrm{TP}+\mathrm{FP}+\mathrm{FN})$: the 2018 Data Science Bowl's per-threshold score, StarDist's `accuracy` |
| `sq`, `pq` | Mean IoU of matched pairs, and $\mathrm{PQ}=\mathrm{SQ}\cdot\mathrm{F1}$ (Kirillov et al., CVPR 2019) |
| `aji` | $\sum I/\sum U$ over each target and its highest-IoU prediction, plus the areas of unpaired targets and untaken predictions in the union (Kumar et al., IEEE TMI 2017) |
| `aji_plus` | The same sums over the one-to-one pairing that maximises summed IoU (Graham et al., Medical Image Analysis 2019) |
| `seg` | Mean over target instances of $\mathrm{IoU}$ with the prediction covering more than half of it, or 0 (Cell Tracking Challenge, Maška et al., Bioinformatics 2014) |

Pairs match one-to-one when their IoU exceeds `threshold` (default 0.5; `strict=False` uses
$\ge$). With a strict threshold of at least 0.5, every instance has at most one candidate, so
the matching is unique. This is the rule of HoVer-Net's `get_fast_pq`, COCO panopticapi and the
Data Science Bowl. Below 0.5, `matching="greedy"` accepts pairs by decreasing IoU, while
`"hungarian"` maximises the summed IoU of accepted pairs, solved per group of overlapping
instances (SciPy). HoVer-Net assigns before thresholding and StarDist maximises the number of
matches first, so their low-threshold counts can differ. A sequence of thresholds adds a
trailing axis to the matching fields.

AJI follows HoVer-Net's `get_fast_aji`: a prediction taken by several targets enters each of
their unions, the over-penalisation AJI+ removes. Ties go to the larger intersection, then the
lower value, which is how HoVer-Net's 1e-6 term in the IoU denominator breaks exact ties. AJI+
pairings with equal summed IoU are resolved per overlapping group, so a tied optimum can
differ from a solve of the whole image. SEG ignores false positives.

A case without any instance gives `both_empty` (NaN by default) for every score. Precision
is NaN without predictions; recall and SEG are NaN without targets. With instances but no
match, F1, accuracy, SQ and PQ are 0, as in Panoptica, HoVer-Net and StarDist.

`dsb2018` averages `accuracy` over the thresholds $k/20$, $k=10,\dots,19$, with strict
matching, as the competition's evaluation page describes (Caicedo et al., Nature Methods 2019).
StarDist's `matching_dataset` uses $\mathrm{IoU}\ge t$ instead.

## Dataset aggregation and classes

`InstanceMetrics` accumulates statistics over batches and processes. With the default
`aggregate="cases"`, it averages each case's score (skipping NaN), as HoVer-Net and PanNuke
report image means. With `aggregate="pooled"`, it sums first:
$\mathrm{PQ}=\sum\mathrm{IoU}/\big(\sum\mathrm{TP}+\tfrac12\sum\mathrm{FP}+\tfrac12\sum\mathrm{FN}\big)$,
as COCO panopticapi and the CoNIC challenge pool. $\mathrm{AJI}=\sum I/\sum U$, and SEG becomes
the mean over all target instances, the challenge's dataset SEG. `tp`, `fp` and `fn` are
dataset totals in both modes.

```python
from exacta.instances import InstanceMetrics

pq_per_class = {c: InstanceMetrics(aggregate="pooled") for c in range(1, num_types)}
for pred_inst, pred_type, true_inst, true_type in loader:
    for c, metric in pq_per_class.items():
        metric.update(torch.where(pred_type == c, pred_inst, 0), torch.where(true_type == c, true_inst, 0))
mpq = torch.stack([m.compute().pq for m in pq_per_class.values()]).nanmean()
```

For panoptic (per-class) scores, keep one accumulator per class with that class's instances,
as above. A class without instances is NaN here; CoNIC's mPQ+ adds $10^{-6}$ to its
denominators, which turns such classes into 0 and keeps them in the mean. Every rank must call
`compute()` with identical settings, including ranks without updates, and samplers must not
repeat cases.

## Execution

Instances are renumbered with a lookup table, or by sorting when values are sparse. A
CPU (C++) or GPU (PyTorch) pass builds the sparse table of overlapping pairs. Matching, AJI and
SEG then work only on those pairs, so the cost follows the number of overlaps rather than the
product of instance counts. On a 1024 × 1024 image with 2000 touching nuclei, one evaluation
took 16.9 ms on a Ryzen 9 5900XT (two threads) and 9.7 ms on an RTX 5070; see
[Performance](https://github.com/rodolfo-viana/exacta/wiki/Performance). The values were
checked against independent dense NumPy implementations of HoVer-Net's PQ, AJI and AJI+
procedures and of the SEG rule, in 2D and 3D.
