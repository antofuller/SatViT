from SSL_model import SatViT
# from convnets import ConvDecoder
# from einops import rearrange, repeat, reduce
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader


# Define Parameters
FLAGS = {}
FLAGS['batch_size'] = 64
FLAGS['num_workers'] = 4
FLAGS['learning_rate'] = 0.01
FLAGS['momentum'] = 0.5
FLAGS['num_epochs'] = 10
FLAGS['num_cores'] = 8
FLAGS['log_steps'] = 20
FLAGS['metrics_debug'] = False


SERIAL_EXEC = xmp.MpSerialExecutor()

in_channels = 12
in_patch_size = 16
in_dim = int(in_channels*in_patch_size*in_patch_size)

out_channels = 12
out_patch_size = 16
out_dim = int(out_channels*out_patch_size*out_patch_size)

# Only instantiate model weights once in memory.
WRAPPED_MODEL = xmp.MpModelWrapper(SatViT(in_dim=in_dim,
                                          out_dim=out_dim,
                                          num_patches=256,
                                          encoder_dim=768,
                                          encoder_depth=12,
                                          encoder_num_heads=16,
                                          decoder_dim=384,
                                          decoder_depth=2,
                                          decoder_num_heads=6,
                                          masking_ratio=0.75,
                                          )
                                   )


def train_mnist():
    torch.manual_seed(1)

    def get_dataset():
        N = 10000
        # random inputs
        all_data = torch.randn(N, 256, 3072)
        # print(f'all_data SHAPE: {all_data.shape}')
        labels = torch.ones(N).float()
        _dataset = TensorDataset(all_data, labels)
        return _dataset

    # Using the serial executor avoids multiple processes to
    # download the same data.
    train_dataset = SERIAL_EXEC.run(get_dataset)
    train_generator = DataLoader(train_dataset,
                                 batch_size=FLAGS['batch_size'],
                                 shuffle=True)

    # Scale learning rate to world size
    lr = FLAGS['learning_rate'] * xm.xrt_world_size()

    # Get loss function, optimizer, and model
    device = xm.xla_device()
    model = WRAPPED_MODEL.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=FLAGS['momentum'])

    def train_loop_fn(loader):
        tracker = xm.RateTracker()
        model.train()
        for x, (data, target) in enumerate(loader):
            optimizer.zero_grad()

            loss, _, _ = model(patch_encodings=data,
                               imgs=data)
            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(FLAGS['batch_size'])
            if x % FLAGS['log_steps'] == 0:
                print('[xla:{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}'.format(
                    xm.get_ordinal(), x, loss.item(), tracker.rate(),
                    tracker.global_rate(), time.asctime()), flush=True)

    # Train loops
    accuracy = 0.0
    data, pred, target = None, None, None
    for epoch in range(1, FLAGS['num_epochs'] + 1):
        para_loader = pl.ParallelLoader(train_generator, [device])
        train_loop_fn(para_loader.per_device_loader(device))
        xm.master_print("Finished training epoch {}".format(epoch))

    return accuracy, data, pred, target


# Start training processes
def _mp_fn(rank, flags):
    global FLAGS
    FLAGS = flags
    torch.set_default_tensor_type('torch.FloatTensor')
    try:
        accuracy, data, pred, target = train_mnist()
    except KeyboardInterrupt:
        exit()


if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS['num_cores'], start_method='fork')