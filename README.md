# PRISM

PyTorch implementation of **PRISM: Priority-Guided Scanning in the Wavelet Domain for UAV Maritime Small Object Detection**.

PRISM separates directional high-frequency evidence from low-frequency scene context throughout encoding, learns a continuous image-conditioned priority map, applies local causal refinement in the induced order, and reconstructs multi-scale detection features through inverse wavelet synthesis.

## Method components

- four-level Haar wavelet decomposition pyramid;
- high-frequency encoder with multi-branch HFBlocks and cross-subband interaction;
- low-frequency encoder with two Vision-RWKV blocks per scale;
- Scene State Adaptive Normalization (SSAN) with optional flight metadata;
- continuous priority policy with metadata-aware temperature;
- priority-guided fusion with kernel-7 causal depthwise 1D refinement;
- gated wavelet synthesis decoder and shared anchor-free detection head;
- detection, wavelet, and policy-prior training objectives.

The default configuration follows the paper: channels `[96, 192, 384, 768]`, fused dimension `128`, three HFBlocks, two Vision-RWKV blocks per scale, approximately 27.2M parameters, AdamW, 100 epochs, batch size 8, initial learning rate `1e-4`, weight decay `5e-4`, and a 0.1 learning-rate decay after epoch 80.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Images and labels follow the YOLO layout. Each label line is
`class cx cy width height`, normalized to `[0, 1]`.

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

Edit `data/multispectral/SeaDroneSee.yaml` to point to the dataset root.

Flight metadata is optional. To enable it, provide one CSV per split and uncomment the `metadata` entries in the dataset YAML. Supported column names include:

```csv
image,altitude,viewing_angle
000001.jpg,42.5,31.0
```

If metadata is absent for an image, PRISM uses its learned default embedding, the metadata-independent temperature, and omits the policy-prior loss for that image.

## Training

```bash
python train.py \
  --data data/multispectral/SeaDroneSee.yaml \
  --cfg models/prism.yaml \
  --hyp data/hyp.scratch.yaml \
  --device 0 --amp --name PRISM_SDS
```

`python train.py` can also be launched directly from an IDE. The direct-run defaults use physical batch size 1, AMP, and gradient accumulation over 8 steps, preserving an effective batch size of 8 while fitting a 24 GB GPU. If memory permits, set a larger physical batch size explicitly.

Three-seed runs can be launched by adding `--seed 0`, `--seed 1`, or `--seed 2`.
Use `--metadata-free` to train or evaluate the strictly metadata-free variant.

```bash
python train.py --resume runs/train/PRISM_SDS/weights/last.pt \
  --data data/multispectral/SeaDroneSee.yaml \
  --name PRISM_SDS --exist-ok --amp
```

## Validation and inference

```bash
python val.py --weights runs/train/PRISM_SDS/weights/best.pt \
  --data data/multispectral/SeaDroneSee.yaml --device 0

python detect.py --weights runs/train/PRISM_SDS/weights/best.pt \
  --source datasets/SeaDronesSee/SeaDronesSee_val/images --device 0
```

Optional metadata can be supplied during standalone inference with
`--height` and `--angle`.

## Sanity check

```bash
python tests/test_smoke.py
```

The bundled evaluator is a self-contained COCO-style sanity evaluator. For reporting benchmark numbers, export predictions to the dataset's official COCO evaluation pipeline.
