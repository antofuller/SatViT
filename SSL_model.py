import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange
from transformer_model import BaseTransformer
from relative_position_encoding import build_dx_dy, RelativePosFFN


def exists(val):
    return val is not None


class PatchEmbed(nn.Module):
    def __init__(self,
                 out_dim,
                 num_channels,
                 dim_channel=256,
                 patch_size=16,
                 ):
        """
        Given an input image (with all 16 channels) splits the image into patches, then creates a patch embedding via a
        simple linear projection. The patch embedding can then be feed into the vision transformer.
        :param out_dim: output dimension (number of features) equal to the encoder_dim
        :param num_channels: Number of channels/bands in our image
        :param dim_channel: input dimension size per channel (typically it will be 256, which is 16x16)
        :param patch_size: Width and height of each patch in pixels (typically 16)
        """
        super().__init__()
        self.patch_size = patch_size
        self.patch_proj = nn.Linear(int(num_channels*dim_channel), out_dim)

    def forward(self, img):
        # img is shape (BSZ, 512, 512, num_channels)
        img = rearrange(img, 'b (patches_w pix_w) (patches_h pix_h) c -> b (patches_w patches_h) (pix_w pix_h c)',
                        patches_w=32, pix_w=self.patch_size, patches_h=32, pix_h=self.patch_size)
        # img is now shape (BSZ, num_patches, num_channels*dim_channel)

        return self.patch_proj(img), img  # (BSZ, num_patches, out_dim)


class SatViT(nn.Module):
    def __init__(self,
                 mae_config,
                 contrastive_config=None,
                 position_encoding='position_embeddings',
                 num_patches=1024,
                 encoder_dim=512,
                 encoder_depth=12,
                 encoder_num_heads=8,
                 num_channels=16,
                 ):
        super().__init__()

        # Input data config
        self.encoder_dim = encoder_dim
        self.num_patches = 1024
        dim_channel = 256

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
            self.to_pixels = nn.Linear(mae_config['decoder_dim'], int(num_channels * dim_channel))
            self.masking_ratio = mae_config['masking_ratio']

        # The encoder is our main model. During inference the decoder won't likely even be used.
        self.encoder = BaseTransformer(dim=encoder_dim,
                                       depth=encoder_depth,
                                       num_heads=encoder_num_heads,
                                       )

        # Setup our position encoding, it should be either standard position embeddings (which are added to the patch
        # embeddings), or a relative position bias (which are added to the self-attention scores)
        if position_encoding.lower() == 'position_embeddings':
            enc_pos_scale = 1 / torch.sqrt(torch.Tensor([encoder_dim]))
            self.encoder_pos_emb = nn.Embedding(num_patches, encoder_dim)
            self.encoder_pos_emb.weight = nn.Parameter(self.encoder_pos_emb.weight * enc_pos_scale)

            if exists(mae_config):  # only if we have a decoder
                dec_pos_scale = 1 / torch.sqrt(torch.Tensor([mae_config['decoder_dim']]))
                self.decoder_pos_emb = nn.Embedding(num_patches, mae_config['decoder_dim'])
                self.decoder_pos_emb.weight = nn.Parameter(self.decoder_pos_emb.weight * dec_pos_scale)

            self.relative_position_bias = None  # not used if using position embeddings

        elif position_encoding.lower() == 'relative_bias':
            self.encoder_pos_emb = None  # not used if using relative position bias
            self.dx_dy = build_dx_dy()  # tensor of shape (1024, 1024, 2)
            self.rpe_ffn = RelativePosFFN(in_dim=2,  # for dx and dy
                                          inner_dim=int(encoder_num_heads*2),  # arbitrarily chosen
                                          out_dim=encoder_num_heads)  # we need 1 bias per attention head
        else:
            raise f"position_encoding must be either 'position_embeddings' or 'relative_bias'"

        self.to_patch_emb = PatchEmbed(out_dim=encoder_dim, num_channels=num_channels)

    def forward(self, img):
        bsz = img.shape[0]  # Batch size
        device = img.device  # Get the device of our input img, i.e. GPU:id or CPU

        # Get img patches, then add position embeddings
        patches, reshaped_inputs = self.to_patch_emb(img)  # (BSZ, num_patches, encoder_dim)
        position_embeds = self.encoder_pos_emb(torch.arange(1024, device=device)).view(1, 1024, self.encoder_dim)
        unmasked_patches = patches + position_embeds.repeat(bsz, 1, 1)  # (BSZ, num_patches, encoder_dim)

        # Calculate patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * self.num_patches)
        rand_indices = torch.rand(bsz, self.num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # Get the unmasked patches which are encoded
        batch_range = torch.arange(bsz, device=device)[:, None]
        unmasked_patches = unmasked_patches[batch_range, unmasked_indices, :]

        # Get the masked patches which are used as labels (patches should not have position embeddings added)
        masked_patches = reshaped_inputs[batch_range, masked_indices, :]

        # Encode the unmasked patches via the encoder
        encodings = self.encoder(unmasked_patches)

        # project encoder to decoder dimensions, if they are not equal
        decoder_input = self.enc_to_dec(encodings)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_embeds = repeat(self.mask_emb, 'd -> b n d', b=bsz, n=num_masked)
        mask_embeds = mask_embeds + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_input = torch.cat([mask_embeds, decoder_input], dim=1)
        decoder_output = self.decoder(decoder_input)

        # splice out the mask tokens and project to pixel values
        pred_pixel_values = self.to_pixels(decoder_output[:, :num_masked, :])

        # calculate reconstruction loss
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss

