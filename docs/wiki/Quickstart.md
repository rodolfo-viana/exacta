# Quickstart

This synthetic example runs one training loss and evaluates hard labels. It fixes the random
seed and uses small 3D arrays; it is an API demonstration, not a segmentation experiment.

```python
import torch
import exacta.functional as EF
from exacta.losses import DiceCELoss
from exacta.metrics import Dice, HD95

generator = torch.Generator().manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logits = torch.randn(2, 3, 8, 8, 8, generator=generator).to(device).requires_grad_()
target = torch.randint(3, (2, 8, 8, 8), generator=generator).to(device)

loss = DiceCELoss(include_background=False)(logits, target)
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()

prediction = logits.detach().argmax(1)
spacing = (1.5, 0.8, 0.8)  # axis order of the array, in millimetres
per_case = EF.hd95(prediction, target, spacing=spacing, num_classes=3,
                   include_background=False)
assert per_case.shape == (2, 2)  # two cases and two foreground classes

dice = Dice(num_classes=3, include_background=False)
hd95 = HD95(num_classes=3, include_background=False)
dice.update(prediction, target)
hd95.update(prediction, target, spacing=spacing)
result = hd95.compute()
print("Loss:", loss.item())
print("HD95 per case:", result.per_case)
print("HD95 per class:", result.per_class)
print("HD95 mean:", result.mean)
```

Use `reset()` on a stateful metric at the start of a new evaluation. Metrics consume hard
labels; losses consume logits unless an overlap loss explicitly uses `activation="none"`.
Review [Conventions](https://github.com/rodolfo-viana/exacta/wiki/Conventions) before comparing
results against another package.
