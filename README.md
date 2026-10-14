# PRISM

## Priority-Guided Scanning in the Wavelet Domain for UAV Maritime Small Object Detection

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-8A2BE2.svg)](https://neurips.cc/)
[![Paper](https://img.shields.io/badge/Paper-OpenReview-B31B1B.svg)](https://openreview.net/forum?id=k0w57wXF9Q)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg)](https://pytorch.org/)

> **Accepted to the NeurIPS 2026 Main Track as a poster.**

Official PyTorch implementation of **PRISM: Priority-Guided Scanning in the Wavelet Domain for UAV Maritime Small Object Detection**.

**Pengqi Gao, Fan Shi, Mianzhao Wang, Jiangpeng Zheng, Xu Cheng, Shengyong Chen**

[[Paper]](https://openreview.net/forum?id=k0w57wXF9Q)

## Overview

Detecting small maritime objects from UAV imagery is challenging because weak target evidence often occupies only a few pixels and is easily confused with structured background clutter such as waves and reflections. PRISM addresses this problem through frequency-separated representation learning and priority-conditioned feature interaction.

The framework decomposes each image into directional high-frequency evidence and low-frequency scene context, models the two streams with dedicated encoders, and learns a continuous, image-conditioned priority map. The learned scores induce an adaptive token ordering for local causal refinement. Multi-scale detection features are subsequently reconstructed through gated inverse wavelet synthesis and passed to a shared detection head.

PRISM was evaluated on SeaDronesSee, AFO, and VisDrone, demonstrating strong performance for small-object detection across maritime and aerial scenes.

## Highlights

- **Frequency-separated representation:** a four-level Haar wavelet pyramid maintains directional high-frequency evidence and low-frequency context as separate streams.
- **Dedicated frequency encoders:** HFE enhances target-sensitive directional evidence, while a Vision-RWKV-based LFE captures long-range scene context with linear token complexity.
- **Priority-guided interaction:** a continuous priority map induces an image-adaptive token ordering for local causal refinement.
- **Wavelet synthesis decoding:** gated inverse DWT progressively reconstructs multi-scale detection features while controlling clutter-sensitive detail injection.
- **Scene adaptation:** SSAN and the optional metadata-aware temperature adapt feature modulation and prioritization to flight conditions.
- **Metadata-free operation:** when flight metadata is unavailable, PRISM uses a learned default embedding and metadata-independent temperature.

## Architecture

The implementation contains the following principal components:

| Component | Description |
|---|---|
| WDP | Four-level Haar Wavelet Decomposition Pyramid |
| HFE | Multi-branch High-Frequency Encoder with cross-subband interaction |
| LFE | Low-Frequency Encoder with two Vision-RWKV blocks per scale |
| SSAN | Scene State Adaptive Normalization |
| Policy Network | Continuous image-conditioned priority estimation |
| PGF | Priority-Guided Fusion with kernel-7 causal depthwise refinement |
| WSD | Gated Wavelet Synthesis Decoder |
| Detection Head | Shared anchor-free multi-scale detection head |

The default six-class model contains approximately **27.2M trainable parameters**. Detailed manuscript-to-code correspondence is provided in [`METHOD_ALIGNMENT.md`](METHOD_ALIGNMENT.md).

## Installation

We recommend Python 3.10 and an NVIDIA GPU for full-resolution training.

```bash
# After cloning or downloading this repository:
cd PRISM

conda create -n prism python=3.10 -y
conda activate prism
pip install -r requirements.txt
```

Install the PyTorch build matching the CUDA version of the target machine when necessary.

## Dataset Preparation

The provided loader expects YOLO-format annotations. Each label line contains:

```text
class_id center_x center_y width height
```

All box coordinates are normalized to `[0, 1]`. The expected SeaDronesSee directory layout is:

```text
datasets/SeaDronesSee/
├── SeaDronesSee_train/
│   ├── images/
│   └── labels/
├── SeaDronesSee_val/
│   ├── images/
│   └── labels/
└── SeaDronesSee_test/
    ├── images/
    └── labels/
```

Set the dataset root in [`data/multispectral/SeaDroneSee.yaml`](data/multispectral/SeaDroneSee.yaml). Do not commit local datasets or machine-specific absolute paths.

### Optional flight metadata

PRISM accepts per-image altitude and viewing-angle metadata through CSV files:

```csv
image,altitude,viewing_angle
000001.jpg,42.5,31.0
```

Add the CSV paths to the dataset YAML:

```yaml
metadata:
  train: metadata/train.csv
  val: metadata/val.csv
  test: metadata/test.csv
```

Metadata is optional. Missing entries automatically use the learned default embedding. To force the strictly metadata-free setting, use `--metadata-free`.

## Training

### Direct execution

After configuring the dataset YAML, `train.py` can be launched directly from an IDE. Its default single-GPU settings use:

- 100 epochs;
- input resolution 640 × 640;
- physical batch size 1;
- automatic mixed precision;
- gradient accumulation over 8 iterations, giving an effective batch size of 8;
- AdamW with an initial learning rate of `1e-4` and weight decay of `5e-4`.

### Command-line execution

```bash
python train.py \
  --data data/multispectral/SeaDroneSee.yaml \
  --cfg models/prism.yaml \
  --hyp data/hyp.scratch.yaml \
  --device 0 \
  --name PRISM_SDS
```

For independent runs with different seeds:

```bash
python train.py --seed 0 --name PRISM_seed0
python train.py --seed 1 --name PRISM_seed1
python train.py --seed 2 --name PRISM_seed2
```

To resume training:

```bash
python train.py \
  --resume runs/train/PRISM_SDS/weights/last.pt \
  --data data/multispectral/SeaDroneSee.yaml \
  --name PRISM_SDS \
  --exist-ok
```

## Evaluation

```bash
python val.py \
  --weights runs/train/PRISM_SDS/weights/best.pt \
  --data data/multispectral/SeaDroneSee.yaml \
  --batch-size 1 \
  --device 0
```

The bundled evaluator is intended for development and sanity checks. Benchmark results should be reported with the official evaluation protocol of the corresponding dataset.

## Inference

```bash
python detect.py \
  --weights runs/train/PRISM_SDS/weights/best.pt \
  --source /path/to/test/images \
  --img-size 640 \
  --device 0 \
  --name PRISM_result
```

Optional flight metadata can be provided during standalone inference:

```bash
python detect.py \
  --weights runs/train/PRISM_SDS/weights/best.pt \
  --source /path/to/image.jpg \
  --height 50 \
  --angle 35 \
  --device 0
```

## Sanity Check

```bash
PYTHONPATH=. python tests/test_smoke.py
```

The check covers the Haar round trip, bidirectional WKV, four-scale forward path, metadata conditioning, loss computation, backward propagation, and inference.

## Citation

If this work is useful for your research, please cite:

```bibtex
@inproceedings{gao2026prism,
  title     = {PRISM: Priority-Guided Scanning in the Wavelet Domain for UAV Maritime Small Object Detection},
  author    = {Gao, Pengqi and Shi, Fan and Wang, Mianzhao and Zheng, Jiangpeng and Cheng, Xu and Chen, Shengyong},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgements

This repository includes a self-contained PyTorch implementation of bidirectional Vision-RWKV token mixing. We thank the authors of the related open-source projects and the maintainers of the public benchmark datasets used in this work.
