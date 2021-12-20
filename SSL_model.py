import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from transformer_model import BaseTransformer
from relative_position_encoding import build_dx_dy, RelativePosFFN


def exists(val):
    return val is not None


class SSLModel(nn.Module):
    def __init__(self,
                 mae_config,
                 contrastive_config,
                 position_encoding='position_embeddings',
                 num_patches=1024,
                 encoder_dim=512,
                 encoder_depth=12,
                 encoder_num_heads=8,
                 ):
        super().__init__()

        if exists(mae_config):
            # Masked Auto-Encoding (MAE) is enabled
            # A decoder is only needed when we are training with MAE
            self.decoder = BaseTransformer(dim=mae_config['decoder_dim'],
                                           depth=mae_config['decoder_depth'],
                                           num_heads=mae_config['decoder_num_heads'],
                                           )

            # Mask embeddings are used in the decoder (these are the locations the decoder will predict the input)
            # These are the grey blocks in Figure 1 (https://arxiv.org/pdf/2111.06377.pdf)
            self.mask_emb = nn.Parameter(torch.randn(mae_config['decoder_dim']))

            # If the encoder and decoder have different model widths (dim) we need to apply a linear projection from the
            # encoder to the decoder. If the models have equal width, no projection is needed.
            self.enc_to_dec = nn.Linear(encoder_dim, mae_config['decoder_dim']) if encoder_dim != mae_config['decoder_dim'] else nn.Identity()

        # The encoder is our main model. During inference the decoder won't likely even be used.
        self.encoder = BaseTransformer(dim=encoder_dim,
                                       depth=encoder_depth,
                                       num_heads=encoder_num_heads,
                                       )

        # Setup our position encoding, it should be either standard position embeddings (which are added to the patch
        # embeddings), or a relative position bias (which are added to the self-attention scores)
        if position_encoding.lower() == 'position_embeddings':
            self.encoder_pos_emb = nn.Embedding(num_patches, encoder_dim)  # TODO: multiply by 1/sqrt(dim)
            if exists(mae_config):  # only if we have a decoder
                self.decoder_pos_emb = nn.Embedding(num_patches, mae_config['decoder_dim'])  # TODO: multiply by 1/sqrt(dim)
            self.relative_position_bias = None  # not used if using position embeddings

        elif position_encoding.lower() == 'relative_bias':
            self.encoder_pos_emb = None  # not used if using relative position bias
            self.dx_dy = build_dx_dy()  # tensor of shape (1024, 1024, 2)
            self.rpe_ffn = RelativePosFFN(in_dim=2,  # for dx and dy
                                          inner_dim=int(encoder_num_heads*2),  # arbitrarily chosen
                                          out_dim=encoder_num_heads)  # we need 1 bias per attention head
        else:
            raise f"position_encoding must be either 'position_embeddings' or 'relative_bias'"
    #     num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
    #     self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
    #     pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]
    #
    #     # decoder parameters
    #
    #     self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
    #     self.mask_token = nn.Parameter(torch.randn(decoder_dim))
    #     self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
    #     self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
    #     self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
    #
    # def forward(self, img):
    #     device = img.device
    #
    #     # get patches
    #
    #     patches = self.to_patch(img)
    #     batch, num_patches, *_ = patches.shape
    #
    #     # patch to encoder tokens and add positions
    #
    #     tokens = self.patch_to_emb(patches)
    #     tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]
    #
    #     # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
    #
    #     num_masked = int(self.masking_ratio * num_patches)
    #     rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
    #     masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
    #
    #     # get the unmasked tokens to be encoded
    #
    #     batch_range = torch.arange(batch, device = device)[:, None]
    #     tokens = tokens[batch_range, unmasked_indices]
    #
    #     # get the patches to be masked for the final reconstruction loss
    #
    #     masked_patches = patches[batch_range, masked_indices]
    #
    #     # attend with vision transformer
    #
    #     encoded_tokens = self.encoder.transformer(tokens)
    #
    #     # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
    #
    #     decoder_tokens = self.enc_to_dec(encoded_tokens)
    #
    #     # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
    #
    #     mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
    #     mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)
    #
    #     # concat the masked tokens to the decoder tokens and attend with decoder
    #
    #     decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim = 1)
    #     decoded_tokens = self.decoder(decoder_tokens)
    #
    #     # splice out the mask tokens and project to pixel values
    #
    #     mask_tokens = decoded_tokens[:, :num_masked]
    #     pred_pixel_values = self.to_pixels(mask_tokens)
    #
    #     # calculate reconstruction loss
    #
    #     recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
    #     return recon_loss