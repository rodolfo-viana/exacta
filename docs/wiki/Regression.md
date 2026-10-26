# Regression

These APIs are **unreleased**, introduced in `0.0.1a6.dev0`. Functions in
`exacta.functional.regression` evaluate continuous predictions shaped `(N,)` or `(N,D)`.
Predictions come first; targets must have the identical shape. Functional outputs retain
the prediction's PyTorch, NumPy, CuPy or JAX array library. All metric statistics use
float64 and detach inputs from autograd.

## Errors and agreement

`evaluate` shares input passes across seven metrics. For one output, let $w_i$ denote
observed sample weights, $W=\sum_iw_i$, prediction $p_i$, target $t_i$, and error
$e_i=p_i-t_i$. Means, variances and covariance below use population normalization by $W$.

| Function | Definition |
|---|---|
| `mae` | $\sum_iw_i\lvert e_i\rvert/W$ |
| `mse` | $\sum_iw_ie_i^2/W$ |
| `rmse` | $\sqrt{\mathrm{MSE}}$ |
| `r2` | $1-\sum_iw_ie_i^2/\sum_iw_i(t_i-\bar t)^2$ |
| `explained_variance` | $1-\mathrm{Var}_w(e)/\mathrm{Var}_w(t)$ |
| `pearson` | $\mathrm{Cov}_w(p,t)/\sqrt{\mathrm{Var}_w(p)\mathrm{Var}_w(t)}$ |
| `concordance_correlation` | $2\mathrm{Cov}_w(p,t)/(\mathrm{Var}_w(p)+\mathrm{Var}_w(t)+(\bar p-\bar t)^2)$ |

Pearson measures linear association and ignores offsets/scaling that concordance penalizes.
Explained variance removes constant prediction bias; R² includes it. None of these scores
establishes causation. Predictions and targets must use the same physical units; MAE/RMSE
retain those units and MSE uses squared units.

`multioutput="raw_values"` returns `(D,)` (length one for scalar regression).
`uniform_average`, the default, takes the ordinary mean of output scores, including NaNs.
RMSE takes the square root **per output before averaging**. `variance_weighted` is supported
only for R² and explained variance; it weights by each output's weighted target squared
deviations. Constant outputs have zero weight; if every output is constant, use a uniform mean.
With different observation masks across outputs, this weighting also reflects observed mass.

## Missing values and degenerate inputs

Boolean `mask` selects individual outputs `(N,D)` or complete samples `(N,)` / `(N,1)`.
For one-dimensional inputs, the mask must also be `(N,)`. Finite, nonnegative
`sample_weight` has shape `(N,)`; zero-weight entries are excluded. Masked or zero-weight
values can be nonfinite, but every observed prediction and target must be finite.
Weights must be valid even on masked samples. Empty observed outputs return NaN.

R² requires at least two positive-weight observations. Constant targets yield NaN for
zero squared error or negative infinity otherwise. Explained variance yields NaN for
zero error variance or negative infinity otherwise. `force_finite=True` replaces those
constant-target cases with 1 or 0 respectively, while empty data and one-sample R² remain
NaN. Pearson is NaN if either input is constant. Concordance is NaN for identical constants
and zero for distinct constants. Very large finite values can still overflow float64
arithmetic; choose sensible units for the evaluation.

## Streaming and execution

`RegressionMetrics` pools weighted errors and centered moments in O(D) memory. It computes
scores after pooling, so changing batch size does not change the intended metric. CPU
float32/float64 input paths use C++/OpenMP with validation fused into accumulation; other
CPU dtypes are converted. GPU uses native PyTorch. Batch moments are anchored before
centering and merged using parallel centered-moment formulas, avoiding raw second-moment
subtraction at large offsets. Floating-point merge order can still produce small differences.

```python
import torch
from exacta.functional import regression as R
from exacta.metrics.regression import RegressionMetrics

pred = torch.tensor([[1., 5.], [2., 9.], [4., 10.]])
target = torch.tensor([[1., 4.], [3., 8.], [5., 12.]])
mask = torch.tensor([[True, True], [True, False], [True, True]])
print(R.evaluate(pred, target, mask=mask, multioutput="raw_values"))

metrics = RegressionMetrics(metrics=("mae", "rmse", "r2"), multioutput="raw_values")
metrics.update(pred[:1], target[:1], mask=mask[:1])
metrics.update(pred[1:], target[1:], mask=mask[1:])
print(metrics.compute())
metrics.reset()
```

Compute pools compact states across distributed ranks. All ranks must call it with
identical options, including ranks without updates. Set the current CUDA device for NCCL,
and avoid padded evaluation samplers that repeat observations. Compute is idempotent;
it does not overwrite local state. No rank having called update is an error. Multi-GPU
NCCL coverage remains experimental; the new accumulation is checked with CPU Gloo.

## Training objectives

`exacta.losses.regression` exports the original PyTorch `L1Loss`, `MSELoss`, `HuberLoss`,
`SmoothL1Loss`, `GaussianNLLLoss` and `PoissonNLLLoss` classes. Their semantics are exactly
PyTorch's. Custom `QuantileLoss` and `LogCoshLoss` require same-shaped floating predictions
and targets and accept a same-shaped boolean mask.

For residual $r=t-p$, pinball loss at quantile $0<q<1$ is

$$
L_q(p,t)=\max(qr,(q-1)r).
$$

At $q=0.5$ it equals half the absolute error. It fits one conditional quantile per
instance; it does not impose ordering between multiple predicted quantiles. At zero
residual the selected subgradient is the midpoint of the two one-sided slopes.
Log-cosh loss is $\log\cosh(p-t)$, quadratic near zero and approximately absolute error
minus $\log2$ for large errors. Its implementation avoids small-error cancellation and
large-error overflow, including in second derivatives.

```python
from exacta.losses.regression import QuantileLoss, LogCoshLoss

prediction = torch.tensor([1., 2., 3.], requires_grad=True)
target = torch.tensor([1.5, 1., 4.])
loss = QuantileLoss(quantile=0.9)(prediction, target)
loss.backward()
print(loss.item(), LogCoshLoss()(prediction.detach(), target).item())
```

Both custom losses support `reduction="none"`, `sum` and `mean`. Mean divides by observed
elements; fully masked or empty inputs give differentiable zero. Half/bfloat16 inputs
accumulate in float32; float64 and higher derivatives are preserved. Observed values are
expected finite, as in the standard PyTorch losses. `torch.compile` is supported; benchmark
eager and compiled execution for the actual workload rather than assuming compilation wins.

The [scikit-learn regression metric reference](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
documents common definitions and multioutput conventions. Native loss contracts are in
[PyTorch's loss documentation](https://docs.pytorch.org/docs/stable/nn.html#loss-functions).
