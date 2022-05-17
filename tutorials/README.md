# Detailed Usage
For fine-tuning, we recommend using imagery and preprocessing steps as close as possible to our pre-training setup. Using other imagery (e.g., taken from Landsat, or no top-of-atmosphere corrections) may work better than initializing models from scratch, but this has not been demonstrated. As a general rule, the more you deviate form our preprocessing steps, the more fine-tuning data will likely be required to achieve satisfactory performance.

## Multispectral optical preprocessing
We use optical data taken from Sentinel-2 after top-of-atmosphere corrections, specifically the 'COPERNICUS/S2_SR' product from Google Earth  (GEE). For pre-training, each optical image was composed of averaging images taken over 1 month (only images with less than 20% cloud coverage). These preprocessing steps were done server-side via the GEE python API. However, if you want to use the median of images taken over some period of time, or a single image, we still expect our model to transfer relatively well. Importantly, our images are all 256x256 pixels (height and width). We recommend all of your fine-tuning images also be 256x256, you can deviate from this, but you will need to carefully select the appropriate position embeddings.

After querying from GEE you will have images of shape (256, 256, 12), were 12 is the number of optical bands. It is probably easiest to save your images as PyTorch tensors by stacking them such that your dataset has the shape (num_images, 256, 256, 12). At this point you can normalize all channels by subtracting the mean, and dividing by the standard deviation (for each channel). See below:

```python
import torch
from einops import rearrange

num_images = 500  # just for example
dummy_images = torch.randn(num_images, 256, 256, 12)
optical_mean_std = torch.Tensor([[ 884.7334, 1254.2722],
                                 [ 958.7927, 1269.4990],
                                 [1144.4814, 1201.9749],
                                 [1152.3252, 1262.4783],
                                 [1559.3584, 1241.1108],
                                 [2384.0615, 1175.8447],
                                 [2680.8501, 1203.6945],
                                 [2769.1274, 1223.6914],
                                 [2886.8269, 1193.5830],
                                 [3064.9661, 1439.5085],
                                 [2169.6279,  998.2980],
                                 [1491.0271,  907.8351]])

dummy_images = (dummy_images - optical_mean_std[:, 0]) / optical_mean_std[:, 1]
dummy_images = rearrange(dummy_images, 'b h w c -> b c h w')
```

This last step rearranges our data into the shape (num_images, 12, 256, 256). Yes, you can do this with PyTorch's rearrange function, but I've run into issues with it before.

## Synthetic Aperture Radar (SAR) preprocessing

We use SAR data taken from Sentinel-1, specifically the 'COPERNICUS/S1_GRD' GEE product in the 'IW' instrument mode and kept in decibels. Again, during pre-training each SAR image was composed of averaging images taken over 1 month. We also acquire both ascending and descending orbit data and stack them in the channel dimension, then normalize them. For each SAR image, the first band is the VV and the second is VH. See below:

```python
num_images = 500  # just for example

if both_orbits_available:
    # Here, both orbits are available
    ascending_data = torch.randn(num_images, 256, 256, 2)
    descending_data = torch.randn(num_images, 256, 256, 2)
    stacked_data = torch.cat([ascending_data, descending_data], dim=-1)
elif only_ascending:
    # Here, only ascending data are available so we need to re-use it
    ascending_data = torch.randn(num_images, 256, 256, 2)
    stacked_data = torch.cat([ascending_data, ascending_data], dim=-1)
elif only_descending:
    # Here, only descending data are available so we need to re-use it
    descending_data = torch.randn(num_images, 256, 256, 2)
    stacked_data = torch.cat([descending_data, descending_data], dim=-1)
    
SAR_mean_std = torch.Tensor([[ -10.9700,    3.6673],
                             [ -17.9148,    4.5137],
                             [ -10.9671,    3.6928],
                             [ -18.0231,    4.4928]])

stacked_data = (stacked_data - SAR_mean_std[:, 0]) / SAR_mean_std[:, 1]
stacked_data = rearrange(stacked_data, 'b h w c -> b c h w')
```

After these steps we are left with stacked SAR data in the shape (num_images, 4, 256, 256), where 4 is the total number of SAR channels.

## Fine-tuning

Welcome to the fun part! Fine-tuning our pre-trained models.