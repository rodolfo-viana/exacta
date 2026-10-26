"""Independent checks for an anisotropic 2D medical-segmentation workflow."""
import math

import pytest
import torch
import torch.nn.functional as tf

from exacta import functional as f, losses


@pytest.mark.parametrize("spacing", [(0.5, 2.0), (1.3, 0.7)])
def test_single_pixel_geometry_and_areas(device, spacing):
    p = torch.zeros((1, 1, 7, 9), dtype=torch.bool, device=device)
    t = torch.zeros_like(p)
    p[0, 0, 1, 2], t[0, 0, 3, 5] = True, True
    distance = math.hypot(2 * spacing[0], 3 * spacing[1])
    values = f.evaluate(p, t, input_format="one-hot", spacing=spacing,
                        metrics={"dice": {}, "hd95": {}, "assd": {},
                                 "nsd": {"tolerance": distance * 1.001},
                                 "predicted_area": {}, "target_area": {}, "area_error": {}})
    expected = dict(dice=0, hd95=distance, assd=distance, nsd=1,
                    predicted_area=math.prod(spacing), target_area=math.prod(spacing), area_error=0)
    for name, value in expected.items():
        torch.testing.assert_close(values[name], torch.tensor([[value]], dtype=torch.float64, device=device))


@pytest.mark.parametrize("channels", [1, 3])
def test_2d_independent_label_training_matches_formula(device, channels):
    gen = torch.Generator().manual_seed(91)
    x = torch.randn((2, channels, 7, 9), generator=gen).to(device).requires_grad_()
    y = (torch.rand(x.shape, generator=gen) > .6).to(device)
    loss = losses.DiceCELoss(activation="sigmoid", target_format="one-hot", smooth=0)(x, y)
    prob = x.sigmoid()
    dice = 1 - (2 * (prob * y).sum((2, 3)) / (prob.sum((2, 3)) + y.sum((2, 3)))).mean()
    expected = dice + tf.binary_cross_entropy_with_logits(x, y.to(x.dtype))
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(torch.autograd.grad(loss, x)[0], torch.autograd.grad(expected, x)[0])
    hard = x.detach() >= 0
    areas = f.predicted_area(hard, y, input_format="one-hot", spacing=(.5, .8))
    torch.testing.assert_close(areas, hard.sum((2, 3)).double() * .4)
