# Exacta documentation

Exacta provides reference-validated segmentation metrics and differentiable losses for CPU
and GPU. Its current focus is 2D and 3D medical image segmentation, including MRI anatomy and lesions.
Definitions, empty-mask behavior and aggregation are explicit so results can be reproduced.

These pages document **Exacta 0.1.0**. Besides partially annotated evaluation, physical
area/volume measurements, probability channels and class-distance losses, it covers
dataset-level [classification](https://github.com/rodolfo-viana/exacta/wiki/Classification)
and [regression](https://github.com/rodolfo-viana/exacta/wiki/Regression),
[instance label maps](https://github.com/rodolfo-viana/exacta/wiki/Instance-evaluation),
dataset-pooled overlap, Betti errors, Skeleton Recall and soft-label overlap losses. Version 0.1.0
adds a relative tie margin for NSD, [reporting signatures](https://github.com/rodolfo-viana/exacta/wiki/Metrics)
that encode a score's full definition, and `zero_absent` Dice and Tversky losses.

Start with the pages that match your task:

| Task | Guide |
|---|---|
| Install Exacta | [Installation](https://github.com/rodolfo-viana/exacta/wiki/Installation) |
| Run a complete example | [Quickstart](https://github.com/rodolfo-viana/exacta/wiki/Quickstart) |
| Segment 2D medical images | [2D medical segmentation](https://github.com/rodolfo-viana/exacta/wiki/Medical-segmentation-2D) |
| Evaluate hard segmentations | [Metrics](https://github.com/rodolfo-viana/exacta/wiki/Metrics) |
| Train with segmentation losses | [Losses](https://github.com/rodolfo-viana/exacta/wiki/Losses) |
| Train/evaluate boundary, topology and probability objectives | [Advanced segmentation](https://github.com/rodolfo-viana/exacta/wiki/Advanced-segmentation) |
| Match another tool's definition | [Conventions](https://github.com/rodolfo-viana/exacta/wiki/Conventions) |
| Evaluate separate lesions | [Lesion evaluation](https://github.com/rodolfo-viana/exacta/wiki/Lesion-evaluation) |
| Evaluate touching instances | [Instance evaluation](https://github.com/rodolfo-viana/exacta/wiki/Instance-evaluation) |
| Choose and benchmark a backend | [Performance](https://github.com/rodolfo-viana/exacta/wiki/Performance) |
| Work on the library | [Development](https://github.com/rodolfo-viana/exacta/wiki/Development) |

The [PyPI project](https://pypi.org/project/exacta/0.1.0/),
[source repository](https://github.com/rodolfo-viana/exacta), and
[issue tracker](https://github.com/rodolfo-viana/exacta/issues) are maintained by Rodolfo Viana.
Exacta is licensed under Apache 2.0. Benchmark and validate the
conventions relevant to your experiment before changing an established evaluation pipeline.
