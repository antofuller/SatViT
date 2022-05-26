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

This last step reshape/view our data into the shape (num_images, 12, 256, 256). Yes, you can do this with PyTorch's rearrange function, but I've run into issues with it before.

## Synthetic Aperture Radar (SAR) preprocessing

We use SAR data taken from Sentinel-1, specifically the 'COPERNICUS/S1_GRD' GEE product in the 'IW' instrument mode and kept in decibels. Again, during pre-training each SAR image was composed of averaging images taken over 1 month. We also acquire both ascending and descending orbit data and stack them in the channel dimension, then normalize them. For each SAR image, the first band is the VV and the second is VH. See below:

```python
num_images = 500  # just for example

if both_orbits_available:
    # Here, both orbits are available
    ascending_data = torch.randn(num_images, 256, 256, 2)
    descending_data = torch.randn(num_images, 256, 256, 2)
    dummy_SAR = torch.cat([ascending_data, descending_data], dim=-1)
elif only_ascending:
    # Here, only ascending data are available so we need to re-use it
    ascending_data = torch.randn(num_images, 256, 256, 2)
    dummy_SAR = torch.cat([ascending_data, ascending_data], dim=-1)
elif only_descending:
    # Here, only descending data are available so we need to re-use it
    descending_data = torch.randn(num_images, 256, 256, 2)
    dummy_SAR = torch.cat([descending_data, descending_data], dim=-1)
    
SAR_mean_std = torch.Tensor([[ -10.9700,    3.6673],
                             [ -17.9148,    4.5137],
                             [ -10.9671,    3.6928],
                             [ -18.0231,    4.4928]])

dummy_SAR = (dummy_SAR - SAR_mean_std[:, 0]) / SAR_mean_std[:, 1]
dummy_SAR = rearrange(dummy_SAR, 'b h w c -> b c h w')
```

After these steps we are left with stacked SAR data in the shape (num_images, 4, 256, 256), where 4 is the total number of SAR channels.

## Fine-tuning

Welcome to the fun part, fine-tuning our pre-trained models! First, lets split our dataset (created above) into train/validation sets:
```python
from torch.utils.data import TensorDataset, DataLoader

dataset = torch.cat([dummy_images, dummy_SAR], dim=1)  # concatenate/stack imagery along the channel dimension
dummy_labels = torch.randint(low=0, high=3, size=(500,))  # one dummy label per image

batch_size = 32
train_loader = TensorDataset(dataset[:300], dummy_labels[:300])  # first 300 images in train
train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)

val_loader = TensorDataset(dataset[300:], dummy_labels[300:])  # last 200 images in validation
val_loader = DataLoader(val_loader, batch_size=256, shuffle=False)

```

Initialize model and load pre-trained weights:
```python
from SatViT_model import SatViT


patch_hw = 16  # pixels per patch (both height and width)
num_channels = 16  # total input bands
io_dim = int(patch_hw*patch_hw*num_channels)
mae_model = SatViT(in_dim=io_dim, out_dim=io_dim).cuda()

pretrained_checkpoint = torch.load('SatViT_B_optical_SAR_15.pt')['mae_model']
mae_model.load_state_dict(pretrained_checkpoint)
```

Define data augmentation and learning-rate functions:

```python
import math

def adjust_learning_rate(epoch, sched_config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < sched_config['warmup_epochs']:
        lr = sched_config['lr'] * epoch / sched_config['warmup_epochs'] 
    else:
        lr = sched_config['min_lr'] + (sched_config['lr'] - sched_config['min_lr']) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - sched_config['warmup_epochs']) / (sched_config['epochs'] - sched_config['warmup_epochs'])))
    return lr


def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])

def rot_img(x, theta, dtype=torch.FloatTensor):
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size(), align_corners=True).type(dtype)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def augment(_imgs):
    _imgs = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])(_imgs)
    _imgs = transforms.Compose([transforms.RandomVerticalFlip(p=0.5)])(_imgs)

    rand_num = torch.randint(low=0, high=4, size=(1,)).item()
    if rand_num == 0:
        return _imgs
    elif rand_num == 1:
        return rot_img(_imgs, np.pi/2)
    elif rand_num == 2:
        return rot_img(_imgs, -np.pi/2)
    else:
        return rot_img(_imgs, np.pi)
```

Prepare linear classifier, optimizer, LR scheduler, and loss function:
```python
linear = torch.nn.Linear(768, 4).cuda()  # linear projection from pooled representations to class logits
params = list(mae_model.parameters()) + list(linear.parameters())
opt = torch.optim.AdamW(params, lr=max_LR)

# Prep LR stepping
epochs = 50
sched_config = {'lr': max_LR,
                'warmup_epochs': 5,
                'min_lr': 0,
                'epochs': epochs}

record = {'val': 0}
loss_function = torch.nn.CrossEntropyLoss()
```

Prepare linear classifier, optimizer, LR scheduler, and loss function:
```python
for epoch in range(epochs):
    set_LR = adjust_learning_rate(epoch, sched_config)  # get LR for this epoch
    for g in opt.param_groups:
        g['lr'] = set_LR  # update

    
    train_losses = []
    mae_model = mae_model.train()
    for i_batch, batch in enumerate(train_loader):
        batch_ins, batch_labels = batch  # (bsz, 16, 256, 256), (bsz)
        batch_ins = augment(batch_ins).cuda()  # (bsz, 16, 256, 256)

        batch_ins = rearrange(batch_ins, 'b c (h i) (w j) -> b (h w) (c i j)', i=16, j=16)  # (bsz, 256, in_dim)

        x = mae_model.linear_input(batch_ins) + mae_model.pos_embed  # (bsz, 256, encoder_dim)
        encodings = mae_model.encoder(x).mean(dim=1)  # (bsz, encoder_dim)
        logits = linear(encodings)  # (bsz, 4)
        
        loss = loss_function(logits, batch_labels.cuda())
        train_losses.append(loss.cpu().item())
        loss.backward()
        opt.step()
        opt.zero_grad()

    train_losses = torch.Tensor(train_losses).mean().item()
    
    val_accs = []
    mae_model = mae_model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(val_loader):
            batch_ins, batch_labels = batch  # (bsz, 16, 256, 256), (bsz)
            batch_ins = rearrange(batch_ins, 'b c (h i) (w j) -> b (h w) (c i j)', i=16, j=16)  # (bsz, 256, 16*16*16)

            x = mae_model.linear_input(batch_ins) + mae_model.pos_embed  # (bsz, 256, encoder_dim)
            encodings = mae_model.encoder(x).mean(dim=1)  # (bsz, encoder_dim)
            logits = linear(encodings)  # (bsz, 4)
            
            decisions = logits.argmax(dim=-1)  # (bsz)
            equate = (decisions == batch_labels.cuda()).int()
            correct_num = equate.count_nonzero()
            total_num = batch_labels.shape[0]
            accuracy = correct_num/total_num
            val_accs.append(accuracy.cpu().item())
            
    val_accs = torch.Tensor(val_accs).mean().item()

    if val_accs > record['val']:
        record['val'] = val_accs

print(f"Our best validation accuracy is {record['val']}")
```