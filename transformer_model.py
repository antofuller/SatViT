from torch import nn, einsum
from einops import rearrange


def exists(val):
    return val is not None


class FFN(nn.Module):
    def __init__(self,
                 dim,
                 mult=4,
                 dropout=0.,
                 ):
        """
        FFN (FeedForward Network)
        :param dim: model dimension (number of features)
        :param mult: multiply the model dimension by mult to get the FFN's inner dimension
        :param dropout: dropout between 0 and 1
        """
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),  # (BSZ, num_patches, inner_dim)
            nn.GELU(),  # (BSZ, num_patches, inner_dim)
            nn.Dropout(dropout),  # (BSZ, num_patches, inner_dim)
            nn.Linear(inner_dim, dim)  # (BSZ, num_patches, dim)
        )
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        return self.net(x)  # (BSZ, num_patches, dim)


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 dropout=0.,
                 ):
        """
        Self-Attention module without attention masks (not needed for this application). It includes an optional
        relative position bias for using AliBi or SwinV2 position encoding.
        :param dim: model dimension (number of features)
        :param num_heads: number of attention heads
        :param dropout: dropout between 0 and 1
        """
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, 'dim must be evenly divisible by num_heads'
        dim_head = int(dim / num_heads)
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, relative_position_bias=False):
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)  # (BSZ, num_patches, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))  # (BSZ, num_heads, num_patches, dim_head)

        attention_scores = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # (BSZ, num_heads, num_patches, num_patches)

        if exists(relative_position_bias):
            attention_scores = attention_scores + relative_position_bias  # (BSZ, num_heads, num_patches, num_patches)

        attn = attention_scores.softmax(dim=-1)  # (BSZ, num_heads, num_patches, num_patches)
        attn = self.dropout(attn)  # (BSZ, num_heads, num_patches, num_patches)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # (BSZ, num_heads, num_patches, dim_head)
        out = rearrange(out, 'b h n d -> b n (h d)')  # (BSZ, num_patches, dim)
        return self.to_out(out)  # (BSZ, num_patches, dim)


class BaseTransformer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads=8,
                 attn_dropout=0.,
                 ff_dropout=0.,
                 ff_mult=4,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, num_heads=num_heads, dropout=attn_dropout),
                FFN(dim=dim, mult=ff_mult, dropout=ff_dropout),
            ]))

        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, relative_position_bias):
        for self_attn, ffn in self.layers:
            x = self_attn(x, relative_position_bias) + x  # (BSZ, num_patches, dim)
            x = ffn(x) + x  # (BSZ, num_patches, dim)

        return self.norm_out(x)  # (BSZ, num_patches, dim)
