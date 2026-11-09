## Dataset

The synthetic dataset used in this project is derived from the ICVL hyperspectral dataset. The original ICVL dataset should be obtained from its official distribution and used in accordance with its terms of use. The prepared ICVL-derived measurements are not distributed with this repository.

## Data generation

For each ICVL hyperspectral scene, the spectral data are projected onto the measured spectral responses of the nine-channel imaging system to obtain linear encoded measurements.

A global scaling factor is applied to construct the HDR target. Multi-exposure measurements are then synthesized by applying the corresponding exposure ratios, followed by noise simulation, clipping to the valid sensor range, and 10-bit quantization to generate the RAW10 inputs.

The detailed forward model, noise model, scaling procedure, and simulation settings are described in the accompanying paper.
## Expected layout

Training and evaluation expect the following directory structure:

ICVL_dataset/
├── train/
│   └── <scene>/
├── val/
│   └── <scene>/
└── test/
    └── <scene>/

Each scene directory contains:

yhdr_9ch_gt.npy: linear nine-channel HDR target in H x W x 9 layout;
the RAW10 multi-exposure measurements defined in configs/exposure_protocol.json.

The expected dataset contains 160 training scenes, 22 validation scenes, and 20 test scenes.

Exposure levels, file naming, and the HDR normalization scale are defined in configs/exposure_protocol.json.

Place the prepared dataset at

data/ICVL_dataset

or specify another location with

--data-root /path/to/ICVL_dataset