# Dim-R2

Code for "Yoo, Jaesung, et al. "A dimensional R2 regression metric." arXiv preprint arXiv:2605.01066 (2026)." ([Arxiv](https://arxiv.org/abs/2605.01066))

Dim-R2 extends the standard R2 score to multidimensional predictions by computing R2 selectively along user-specified axes, exposing per-channel or per-timestep performance instead of collapsing everything into a single scalar.

Please cite:

    @article{yoo2026dimensional,
    title={A dimensional R2 regression metric},
    author={Yoo, Jaesung and Lemke, Stefan and Guo, Jian Zhong and Rajan, Kanaka and Hantman, Adam},
    journal={arXiv preprint arXiv:2605.01066},
    year={2026}
    }

## File Overview

| File | Description |
|------|-------------|
| `utils.py` | Core implementation: `r2_score()`, `TSS_score()`, signal generation, and plotting utilities |
| `dim_r2_example.py` | Minimal working example demonstrating Dim-R2 on synthetic signals |
| `dimensional.py` | Experiments illustrating how Dim-R2 varies across dimensions (trials × time × channels) |
| `resilience.py` | Simulation comparing Dim-R2 vs. Mean-R2 robustness to noise channels |
| `VAE/vae_utils.py` | Dataset wrappers and VAE neural-network utilities |
| `VAE/MNIST.py` | VAE experiment on MNIST — evaluates reconstructions with Dim-R2 |
| `VAE/CelebA.py` | VAE experiment on CelebA — evaluates reconstructions with Dim-R2 |
| `VAE/cfg.yaml` | Configuration for VAE experiments (architecture, training, runtime) |

## Quick start

```python
import numpy as np
import utils as U

# y_true, y_pred: shape (time, channels)
t, y_true, y_pred = U.make_signals(time=100, C=5, stagger_phase=True)

# Scalar R2 (collapses all axes)
score = U.r2_score(y_true, y_pred)

# Per-channel R2 (collapses time axis only)
score_per_channel = U.r2_score(y_true, y_pred, axis=0)

# Per-timestep R2 (collapses channel axis only)
score_per_time = U.r2_score(y_true, y_pred, axis=1)
```

See `dim_r2_example.py` for a full example with visualizations.

## `r2_score` API

```python
utils.r2_score(y_true, y_pred, axis=None, axis_norm=None, axis_pool=None, force_finite=True)
```

| Parameter | Description |
|-----------|-------------|
| `y_true`, `y_pred` | `np.ndarray` of the same shape |
| `axis` | Axes to sum RSS/TSS over. `None` collapses all axes and returns a scalar. |
| `axis_norm` | Axes used to compute `y_true.mean()` for the TSS reference. Defaults to `axis`. Smaller = more localized normalization. |
| `axis_pool` | Axes to additionally average TSS across. Defaults to `axis_norm`. |
| `force_finite` | Replace `NaN` with 1 and `-Inf` with 0 (handles zero-variance channels). |

**Returns:** scalar if `axis=None`, otherwise an `np.ndarray` with the specified axes removed. The `axis` parameter and output shape follow the same conventions as NumPy reductions (e.g., `np.mean`, `np.sum`).

## VAE experiments

The `VAE/` directory contains experiments that apply Dim-R2 to evaluate image reconstruction quality of Variational Autoencoders on MNIST and CelebA. Both scripts read `VAE/cfg.yaml` for model and training configuration.

```bash
cd VAE
python MNIST.py
python CelebA.py
```

## Python library requirements

```bash
pip install -r requirements.txt
```
