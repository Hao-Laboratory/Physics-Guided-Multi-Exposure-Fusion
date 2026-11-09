# Physics-Guided Multi-Exposure Fusion

Official PyTorch implementation of a physics-guided network for fusing differently exposed nine-channel measurements into a linear HDR measurement.

The repository provides the training and evaluation code, deterministic exposure protocol, pretrained model, and results from the reported 500-epoch training run.

This release focuses on the learned HDR measurement fusion module and its synthetic-data experiments. Real-camera sensor characterization, LUT correction, reliability gating, downstream spectral reconstruction, and spectral-to-sRGB visualization are outside the scope of this repository.

## Method overview

For every exposure, a shared two-layer convolutional estimator predicts a
per-pixel, per-measurement reliability correction from the RAW-domain input.
It is combined with the analytical hat prior

```text
h(r) = 4 r (1 - r)
logit = log(h(r) + 0.01) + learned_logit
```

The logits are normalized with softmax over the exposure dimension and applied to the exposure-normalized measurements. The weighted estimate is then refined by a three-scale residual U-Net:

```text
output = clamp(weighted_fusion + residual, 0, 1)
```

The model has 1,290,786 trainable parameters. Its inputs are `raw` and
`normalized` tensors of shape `[B, E, 9, H, W]`, with EV values and a validity
mask of shape `[B, E]`. The same reliability estimator is applied to every
exposure.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```
Install a CUDA-enabled PyTorch build appropriate for your system when using a GPU.

## Dataset

The synthetic dataset used in this work is derived from the ICVL hyperspectral dataset. Due to the distribution terms of the original dataset, the prepared ICVL-derived measurements are not redistributed with this repository.

The simulation procedure is described in the paper and summarized in data/README.md. The reported experiments use 160 training scenes, 22 validation scenes, and 20 test scenes.

Place  prepared data at data/ICVL_dataset, or specify its location with --data-root.

The network preprocessing is:

```text
raw = RAW10 / 1023
normalized = raw / (2**EV) / HDR_SCALE
target = yhdr_9ch_gt / HDR_SCALE
```

All network outputs remain in the normalized linear domain. `evaluate.py`
multiplies the prediction by the HDR scale before saving the final linear HDR
measurement.

## Exposure protocol

Training uses three-exposure samples drawn from the EV pool

{-2, -1.5, -1, 0, 1, 1.5, 2}

with 24 predefined training combinations. The canonical (-2, 0, 2) combination is assigned to 64 of the 160 training scenes per epoch, while the remaining assignments are balanced across the other training combinations.

Scene order, crop positions, augmentation parameters, and exposure assignments are stored in

configs/exposure_schedule.npz

Validation and held-out exposure combinations are defined in

configs/exposure_protocol.json

Validation metrics are averaged over scenes within each exposure combination and then equally averaged across combinations.


## Training

The training objective is

```text
L = L1(mu-law(pred), mu-law(target)) + 0.05 * angle(pred, target)
mu-law(x) = log(1 + 5000*x) / log(1 + 5000)
```

The angular term is computed at each pixel between the two nine-dimensional
measurement vectors and averaged in radians.

The reported setup uses Adam with an initial learning rate of `2e-4`, batch
size 2, `128 x 128` crops, mixed precision, and gradient-norm clipping at 1.0.
The learning rate is halved every 100 epochs and training lasts 500 epochs.

Train with the default dataset location:

```bash
python train.py
```

Use an external prepared dataset:

```bash
python train.py --data-root /path/to/ICVL_dataset
```

Useful development smoke test:

```bash
python train.py --data-root /path/to/ICVL_dataset --epochs 1 \
  --limit-train 2 --limit-val 1 --no-amp --device cpu --run-name smoke_test
```

Resume a run:

```bash
python train.py --resume outputs/training/<run>/latest_checkpoint.pth
```

## Full-resolution HDR fusion

The released model is loaded by default:

```bash
python evaluate.py --data-root /path/to/ICVL_dataset \
  --combination=-2,0,2
```

Evaluate selected scenes or a held-out exposure combination:

```bash
python evaluate.py --data-root /path/to/ICVL_dataset \
  --scene hill_0325-1219 --scene bguCAMP_0514-1723 \
  --combination=-2,0,1
```

Each scene produces `fused_hdr_9ch.npy` in `H x W x 9` layout. The saved array
is a linear HDR measurement in the original target scale. The summary files
contain measurement-domain RMSE and angular error when ground truth is
available. Context-aware tiled inference is used to limit peak memory.

## Released checkpoint and results

The released model is provided at pretrained/model_best.pth and corresponds to the reported 500-epoch training run.

Training and validation records together with the training curve are provided under results/.
