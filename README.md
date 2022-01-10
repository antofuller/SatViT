# SatViT (Satellite Vision Transformer)
Project directory for self-supervised training of multi-spectral optical and SAR vision transformers!

## Todo
1. Setup ViT models
2. Setup SSL objectives (CLIP + MAE)
3. Data loading utils

## Notes (to be moved later)
- I tried using a random patch projection (https://arxiv.org/pdf/2104.02057.pdf). It initially looked more stable than a learned patch projection, but it diverged around 5k steps (still during warmup). This is probably worth trying when using the compressed dataset.