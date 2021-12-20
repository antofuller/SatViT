import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from transformer_model import BaseTransformer


def exists(val):
    return val is not None


def log_spaced(_d):
    return torch.sign(_d) * torch.log(1 + torch.abs(_d))


def build_dx_dy():
    # All models will have 1024 patches, so 32x32
    position_ids = []
    for i in range(32):
        for j in range(32):
            position_ids.append([i, j])  # [x position, y position]

    # The dx_dy tensor will be of shape (1024, 1024, 2) or (num_patches, num_patches, num_coordinates)
    dx_dy = torch.zeros(1024, 1024, 2)
    for q_num, query_patch in enumerate(position_ids):
        for k_num, key_patch in enumerate(position_ids):
            # Get distance from query patch to key patch
            dx = torch.Tensor([key_patch[0] - query_patch[0]])
            dy = torch.Tensor([key_patch[1] - query_patch[1]])

            # Perform log-spaced operation from SwinV2 and fill in dx_dy
            dx_dy[q_num, k_num, 0] = log_spaced(dx)
            dx_dy[q_num, k_num, 1] = log_spaced(dy)

    return dx_dy


class RelativePosFFN(nn.Module):
    def __init__(self,
                 in_dim,
                 inner_dim,
                 out_dim,
                 dropout=0.,
                 ):
        """
        FFN (FeedForward Network) for a parametrized relative position bias
        :param in_dim: Number of input features - should be 2 (dx and dy)
        :param inner_dim: Inner/hidden size of FFN
        :param out_dim: Number of output features - equal to the number of attention heads
        :param dropout: Dropout between 0 and 1
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, inner_dim),  # (BSZ, num_patches, inner_dim)
            nn.GELU(),  # (BSZ, num_patches, inner_dim)
            nn.Dropout(dropout),  # (BSZ, num_patches, inner_dim)
            nn.Linear(inner_dim, out_dim)  # (BSZ, num_patches, out_dim)
        )

    def forward(self, x):
        return self.net(x)  # (num_patches, num_patches, out_dim)

