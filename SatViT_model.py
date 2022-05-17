import torch
from torch import nn
from einops import rearrange
from transformer_model import BaseTransformer, get_2d_sincos_pos_embed


# --------------------------------------------------------
# Based on the MAE code base
# https://github.com/facebookresearch/mae
# --------------------------------------------------------

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

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # x should already come ready for the encoder, i.e. be of shape (bsz, seq, io_dim)
        # add pos embed
        x = self.linear_input(x) + self.pos_embed  # (bsz, seq, encoder_dim)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # apply Transformer blocks
        x = self.encoder(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.enc_to_dec(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        x = self.decoder(x)

        # predictor projection
        return self.linear_output(x)

    def forward_loss(self, imgs, pred, mask):
        loss = (pred - imgs) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def encode(self, images_patches):
        """
        We encode full images (i.e., no masking) by linearly projecting image patches, adding position embeddings,
        then encoding these inputs with our MAE encoder. This function will be used during fine-tuning and inference.
        """
        patch_encodings = self.linear_input(images_patches) + self.pos_embed  # (BSZ, num_patches, encoder_dim)
        return self.encoder(patch_encodings)

    def forward(self, patch_encodings, mask_ratio=0.75):
        """
        *** Masked Autoencoding Pre-training ***
        We encode a portion of image patches (1 - mask_ratio), then use the encoded representations of visible patches
        to predict all hidden patches.
        """
        latent, mask, ids_restore = self.forward_encoder(patch_encodings, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # (BSZ, num_patches, io_dim)
        loss = self.forward_loss(patch_encodings, pred, mask)
        return loss, pred, mask
