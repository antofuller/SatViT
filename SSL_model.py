import torch
from torch import nn
from einops import rearrange
from transformer_model import BaseTransformer, get_2d_sincos_pos_embed


class SatViT(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_patches=256,
                 encoder_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=16,
                 decoder_dim=384,
                 decoder_depth=2,
                 decoder_num_heads=6,
                 ):
        super().__init__()

        # Input data config
        self.encoder_dim = encoder_dim
        self.num_patches = num_patches
        self.encoder_dim = encoder_dim

        # The encoder is our main model. During inference the decoder won't likely even be used.
        self.encoder = BaseTransformer(dim=encoder_dim,
                                       depth=encoder_depth,
                                       num_heads=encoder_num_heads,
                                       )

        # Mask embeddings are used in the decoder (these are the locations the decoder will predict the input)
        # These are the grey blocks in Figure 1 (https://arxiv.org/pdf/2111.06377.pdf)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # If the encoder and decoder have different model widths (dim) we need to apply a linear projection from the
        # encoder to the decoder. If the models have equal width, no projection is needed.
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)

        self.decoder = BaseTransformer(dim=decoder_dim,
                                       depth=decoder_depth,
                                       num_heads=decoder_num_heads,
                                       )

        # Setup position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, encoder_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim), requires_grad=False)  # fixed sin-cos embedding

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Input and output maps
        self.linear_input = nn.Linear(in_dim, encoder_dim)
        self.linear_output = nn.Linear(decoder_dim, out_dim)

    def forward(self, x):

        x = self.linear_input(x) + self.pos_embed  # (bsz, seq, encoder_dim)

        # apply Transformer blocks
        x = self.encoder(x)

        return x
