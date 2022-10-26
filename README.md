# SatViT (Satellite Vision Transformer)
Project directory for self-supervised training of multispectral optical and SAR vision transformers!

# Pretrained model weights

SatViT-V2 is much better than SatViT-V1. See a draft of our SatViT-V2 paper: https://arxiv.org/abs/2209.14969

Both models were pretrained on an unlabelled dataset of 1.3 million images acquired from Sentinel-1 and Sentinel-2 (stacked along the channel dimension).

| Name 	 | Arch. 	 | Channels 	 | Image Size 	 | Patch Size 	 | Pretraining<br>epochs 	 | Weights<br>	     |
|---------|---------|-------------------------|------------------------------|------------------------------|--------------------------|-----------------------------|
| SatViT-V1 	 | ViT-Base 	 | optical-SAR (15 channels)                        	 | 256 by 256 pixels 	  |16 by 16 pixels                        	 | 100                  	 | [download](https://github.com/antofuller/SatViT/releases/download/pretrained_weights/SatViT_V1.pt)
| SatViT-V2 	 | ViT-Base 	 | optical-SAR (15 channels)                   	 | 256 by 256 pixels 	 | 8 by 8 pixels                        	 | 220                   	 | [download](https://github.com/antofuller/SatViT/releases/download/pretrained_weights/SatViT_V2.pt)

# Basic Usage
Copy directory, install PyTorch and einops

First, let's encode an image with SatViT-V1:
```python
from SatViT_model import SatViT
from einops import rearrange
import torch

patch_hw = 16  # pixels per patch (both height and width)
num_channels = 15  # total input bands
io_dim = int(patch_hw*patch_hw*num_channels)
model = SatViT(io_dim=io_dim,
               num_patches=256,
               encoder_dim=768,
               encoder_depth=12,
               encoder_num_heads=12,
               decoder_dim=384,
               decoder_depth=2,
               decoder_num_heads=6,
               ).cuda()

pretrained_checkpoint = torch.load('SatViT_V1.pt')
model.load_state_dict(pretrained_checkpoint)

random_image = torch.randn(1, 15, 256, 256).cuda()  # (BSZ, num_channels, height, width)

# Split image up into patches
image_patches = rearrange(random_image, 'b c (h i) (w j) -> b (h w) (c i j)', h=16, w=16)

# Encode with SatViT-V1 encoder
with torch.no_grad():
    patch_encodings = model.encode(images_patches=image_patches)  # (bsz, num_patches, encoder_dim)
```

Now, let's encode an image with SatViT-V2:
```python
from SatViT_model import SatViT
from einops import rearrange
import torch

patch_hw = 8  # pixels per patch (both height and width)
num_channels = 15  # total input bands
io_dim = int(patch_hw*patch_hw*num_channels)
model = SatViT(io_dim=io_dim,
               num_patches=1024,
               encoder_dim=768,
               encoder_depth=12,
               encoder_num_heads=12,
               decoder_dim=512,
               decoder_depth=1,
               decoder_num_heads=8,
               ).cuda()

pretrained_checkpoint = torch.load('SatViT_V2.pt')
model.load_state_dict(pretrained_checkpoint)

random_image = torch.randn(1, 15, 256, 256).cuda()  # (BSZ, num_channels, height, width)

# Split image up into patches
image_patches = rearrange(random_image, 'b c (h i) (w j) -> b (h w) (c i j)', h=8, w=8)

# Encode with SatViT-V2 encoder
with torch.no_grad():
    patch_encodings = model.encode(images_patches=image_patches)  # (bsz, num_patches, encoder_dim)
```

# Detailed Usage
[Please see here](https://github.com/antofuller/SatViT/tree/main/tutorials)

## Citation
SatViT-V1:
```bib
@ARTICLE{9866058,
  author={Fuller, Anthony and Millard, Koreen and Green, James R.},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={SatViT: Pretraining Transformers for Earth Observation}, 
  year={2022},
  volume={19},
  pages={1-5},
  doi={10.1109/LGRS.2022.3201489}}
```
SatViT-V2 (this is a draft manuscript that is under revision):
```bib
@article{fuller2022satvitv2,
  title={Transfer Learning with Pretrained Remote Sensing Transformers},
  author={Fuller, Anthony and Millard, Koreen and Green, James R.},
  journal={arXiv preprint arXiv:2209.14969},
  year={2022}
}
```

# Todo
- [x] Upload code
- [x] Upload pre-trained models
- [ ] Improve tutorials
