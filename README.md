# SatViT (Satellite Vision Transformer)
Project directory for self-supervised training of multi-spectral optical and SAR vision transformers!

## Todo
- [x] ViT model
- [x] Masked Auto-Encoder (MAE) setup
- [ ] Better utils and logging
- [ ] Contrastive learning


## Notes (to be moved later)
- I tried using a random patch projection (https://arxiv.org/pdf/2104.02057.pdf). It initially looked more stable than a learned patch projection, but it diverged around 5k steps (still during warmup). This is probably worth trying when using the compressed dataset.
- The "Crop Type Mapping (Ghana and South Sudan)" task from SustainBench is poor quality. There are extra class labels, and many images are incomplete (over the edge of a scene). Even after taking the median across all images over the time period, it's almost impossible to classify the field boundaries let alone field types.
