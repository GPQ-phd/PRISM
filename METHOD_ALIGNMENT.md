# PRISM method alignment

This implementation follows the architecture and optimization details described in the accepted PRISM manuscript.

| Manuscript component | Implementation |
|---|---|
| Four-level Haar WDP | `haar` in `models/prism.py` |
| HFE and three repeated HFBlocks | `HFE`, `HFBlock` in `models/prism.py` |
| LFE with two Vision-RWKV blocks | `LFE` in `models/prism.py`; `VRWKVBlock` in `models/vrwkv.py` |
| Q-Shift and bidirectional WKV | `q_shift`, `BiWKV` in `models/vrwkv.py` |
| Scene State Adaptive Normalization | `Metadata`, `SSAN` in `models/prism.py` |
| Learned continuous priority map | `Policy` in `models/prism.py` |
| Deterministic non-differentiable ordering | `argsort` in `PGF.forward` |
| Continuous policy learning paths | policy inputs to PGF mixing weight and causal gate |
| Kernel-7 local causal refinement | `PGF.causal_refine` |
| Gated inverse wavelet synthesis | `WSDStep` in `models/prism.py` |
| Shared multi-scale detection head | `Head` in `models/prism.py` |
| Wavelet and policy-prior losses | `prism_loss` in `utils/loss.py` |
| Optional per-image metadata | `YOLODataset` in `utils/datasets.py` |

## Default configuration

- channel dimensions: 96, 192, 384, 768;
- three HFBlocks and two Vision-RWKV blocks per scale;
- fused feature dimension: 128;
- causal kernel size: 7;
- learnable temperature scalars initialized to 1.0 and 0.5;
- AdamW, learning rate `1e-4`, weight decay `5e-4`;
- 100 epochs, batch size 8, input size 640;
- learning rate multiplied by 0.1 from epoch 80;
- loss weights `lambda_wav=0.1` and `lambda_prior=0.01`.

With six detection classes, the default model contains 27,223,916 trainable parameters, which rounds to the 27.2M reported in the paper.

## Implementation note

Bi-WKV is implemented with chunked PyTorch prefix scans and float32 accumulation. This keeps the repository self-contained and preserves the bidirectional Vision-RWKV equation without requiring a custom CUDA extension. The implementation is method-equivalent, although the fused CUDA operator from the upstream Vision-RWKV project can be faster for full-resolution training.

The included evaluator is intended for development checks. Benchmark reporting should use the official evaluation pipeline of the target dataset.
