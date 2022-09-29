# SatViT (Satellite Vision Transformer)
Project directory for self-supervised training of multispectral optical and SAR vision transformers!

Edit from September 29th, 2022: <b>SatViT-V2 performs much better than SatViT-V1 and will be uploaded within 1 week!</b>

# Pre-trained model weights

| Arch. 	 | Inputs 	 | Pre-training<br>epochs 	 | Weights<br>	     |
|---------|------------------------------|--------------------------|-----------------------------|
| ViT-Base 	 | SAR (4 channels)                        	 | 15                  	 | [download](https://github.com/antofuller/SatViT/releases/download/models/SatViT_B_SAR_15.pt)
| ViT-Base 	 | multispectral optical (12 channels)                   	 | 15                   	 | [download](https://github.com/antofuller/SatViT/releases/download/models/SatViT_B_optical_15.pt)
| ViT-Base	 | optical-SAR (16 channels)                   	 | 15                   	 | [download](https://github.com/antofuller/SatViT/releases/download/models/SatViT_B_optical_SAR_15.pt)

# Basic Usage
Copy directory, install PyTorch and einops

Create model and load pre-trained weights
```python
from SatViT_model import SatViT
from einops import rearrange
import torch

patch_hw = 16  # pixels per patch (both height and width)
num_channels = 16  # total input bands
io_dim = int(patch_hw*patch_hw*num_channels)
model = SatViT(in_dim=io_dim, out_dim=io_dim).cuda()

pretrained_checkpoint = torch.load('SatViT_B_optical_SAR_15.pt')['mae_model']
model.load_state_dict(pretrained_checkpoint)
```
Encode an image
```python
random_image = torch.randn(1, 16, 256, 256).cuda()  # (BSZ, num_channels, height, width)

# Split image up into patches
image_patches = rearrange(random_image, 'b c (h i) (w j) -> b (h w) (c i j)', h=16, w=16)

# Encode with SatViT encoder
with torch.no_grad():
    patch_encodings = model.encode(images_patches=image_patches)  # (bsz, num_patches, encoder_dim)
```

# Detailed Usage
[Please see here](https://github.com/antofuller/SatViT/tree/main/tutorials)

# Todo
- [x] Upload code
- [x] Upload pre-trained models
- [ ] Improve comments in code
- [ ] Create tutorials
