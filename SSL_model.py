import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange
from transformer_model import BaseTransformer


class SatViT(nn.Module):
    def __init__(self,
                 io_dim,
                 num_patches=256,
                 encoder_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=16,
                 decoder_dim=384,
                 decoder_depth=2,
                 decoder_num_heads=6,
                 masking_ratio=0.75,
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
        self.mask_emb = nn.Parameter(torch.randn(decoder_dim))

        # If the encoder and decoder have different model widths (dim) we need to apply a linear projection from the
        # encoder to the decoder. If the models have equal width, no projection is needed.
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.masking_ratio = masking_ratio

        self.decoder = BaseTransformer(dim=decoder_dim,
                                       depth=decoder_depth,
                                       num_heads=decoder_num_heads,
                                       )
        # Setup position embeddings
        enc_pos_scale = 1 / torch.sqrt(torch.Tensor([encoder_dim]))
        self.encoder_pos_emb = nn.Embedding(num_patches, encoder_dim)
        self.encoder_pos_emb.weight = nn.Parameter(self.encoder_pos_emb.weight * enc_pos_scale)

        dec_pos_scale = 1 / torch.sqrt(torch.Tensor([decoder_dim]))
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.decoder_pos_emb.weight = nn.Parameter(self.decoder_pos_emb.weight * dec_pos_scale)

        # Input and output maps
        self.linear_input = nn.Linear(io_dim, encoder_dim)
        self.linear_output = nn.Linear(decoder_dim, io_dim)

    def forward(self, x):
        # x should be shape (BSZ, num_patches, io_dim)
        bsz = x.shape[0]  # Batch size
        device = x.device  # Get the device of our input img, i.e. GPU:id or CPU

        # Get img patches, then add position embeddings
        patches = self.linear_input(x)  # (BSZ, num_patches, encoder_dim)
        encoder_position_embeds = self.encoder_pos_emb(torch.arange(256, device=device)).view(1, 256, self.encoder_dim)
        unmasked_patches = patches + encoder_position_embeds.repeat(bsz, 1, 1)  # (BSZ, num_patches, encoder_dim)

        # Calculate patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * self.num_patches)
        rand_indices = torch.rand(bsz, self.num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # Get the unmasked patches which will be encoded
        batch_range = torch.arange(bsz, device=device)[:, None]
        unmasked_patches = unmasked_patches[batch_range, unmasked_indices, :]

        # Get the masked patches which are used as labels (patches should not have position embeddings added)
        masked_patches = x[batch_range, masked_indices, :]

        # Encode the unmasked patches via the encoder
        encodings = self.encoder(unmasked_patches)

        # project encoder to decoder dimensions, if they are not equal, then add unmasked decoder position embeddings
        encodings = self.enc_to_dec(encodings) + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_embeds = repeat(self.mask_emb, 'd -> b n d', b=bsz, n=num_masked) + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_input = torch.cat([mask_embeds, encodings], dim=1)
        decoder_output = self.decoder(decoder_input)

        # splice out the mask tokens and project to pixel values
        pred_values = self.linear_output(decoder_output[:, :num_masked, :])

        # calculate reconstruction loss
        recon_loss = F.mse_loss(pred_values, masked_patches)
        return recon_loss

