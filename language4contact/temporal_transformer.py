from typing import Optional, Sequence, Tuple
import torch
from torch import nn
import numpy as np

"""
Pytorch implementation of TemporalTransformer() in 
https://github.com/google-research/language-table/blob/770dade55237f31b7028dbff488e681a4da5385b/language_table/train/networks/lava.py#L336
"""

"""
  num_layers: int = 2

  sequence_length: int = 4
  temporal_transformer_num_layers: int = 2

  d_model: int = 128
  num_heads: int = 2
"""
class PrenormPixelLangEncoder(nn.Module): 
    def __init__(self, num_layers: int = 2, num_heads: int=2, dropout_rate: float = 0.1,
                 dff: int =128, mha_dropout_rate: float = 0.0, device: int = 'cpu'):
        super(PrenormPixelLangEncoder, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.mha_dropout_rate = mha_dropout_rate
        self.sequence_length = 15

        self.pixel_PosEmb = Add1DPositionEmbedding(max_len=self.sequence_length, device=self.device)
        self.lang_PosEmb = Add1DPositionEmbedding(max_len=self.sequence_length, device=self.device)

        self.multiheadattention = nn.MultiheadAttention(embed_dim = self.dff, num_heads = self.num_heads, dropout=self.mha_dropout_rate)
        self.Dropout = nn.Dropout(p = self.dropout_rate)
        self.LayerNorm1 = nn.LayerNorm([self.dff])
        self.LayerNorm2 = nn.LayerNorm([self.dff])
        self.LayerNorm3 = nn.LayerNorm([self.dff])

        self.l5 = nn.Linear(self.dff, self.dff)
        self.l6 = nn.Linear(self.dff, self.dff)
        self.relu = nn.ReLU()

        nn.init.uniform_(self.l5.weight, 0, 0.05)
        nn.init.uniform_(self.l5.bias, 0, 0.05)
        nn.init.uniform_(self.l6.weight, 0, 0.05)
        nn.init.uniform_(self.l6.bias, 0, 0.05)

    def forward(self, pixel_x, lang_x):
        # residual_lang = lang_x
        pixel_x = self.pixel_PosEmb(pixel_x)
        lang_x = self.lang_PosEmb(lang_x)

        for _ in range(self.num_layers):
            pixel_x_ = self.LayerNorm1(pixel_x).permute((1,0,2))
            lang_x_ = self.LayerNorm2(lang_x.to(pixel_x.dtype)).permute((1,0,2))

            x2, _ = self.multiheadattention(query = lang_x_, 
                                         key = pixel_x_, 
                                         value = pixel_x_)
            x2 = x2.permute((1,0,2))
            x2 = self.Dropout(x2)

            # Residual, only on the language path.
            x3 = lang_x + x2

            # layer norm just the ffn input.
            x4 = self.LayerNorm3(x3)

            # ffn.
            x5 = self.l5(x4)
            x5 = self.relu(x5)
            x5 = self.l6(x5)
            x5 = self.Dropout(x5)

            lang_x = x5 + x3

        return lang_x



class TemporalTransformer(nn.Module):
    """Transformer over time."""
    def __init__(self, num_layers: int=2, d_model: int=128, num_heads: int=2, dff: int =128, sequence_length: int=4, dim_in: int = None, device: int = 'cpu'):
        super(TemporalTransformer, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.dff = d_model
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        self.dim_in = dim_in
        self.device = device

        # Linear layers.
        self.l1 = nn.Linear(self.d_model, self.d_model)
        nn.init.uniform_(self.l1.weight, 0, 0.05)
        nn.init.uniform_(self.l1.bias, 0, 0.05)

        # Followed the original config.
        self.Dropout = nn.Dropout(p = 0.1)
        self.PosEmb = Add1DPositionEmbedding(max_len=self.sequence_length, device=self.device)
        self.PrenormEncoderLayer = [PrenormEncoderLayer(
                                    num_heads=self.num_heads,
                                    dropout_rate=0.1,
                                    mha_dropout_rate=0.0,
                                    dff=self.dff, 
                                    dim_in=self.d_model,
                                    ).to(self.device) for _ in range(self.num_layers)
                                    ]
        self.LayerNorm = nn.LayerNorm([self.d_model])



    def forward(self, x, padding_mask):
        x = self.l1(x)
        x *= np.sqrt(self.d_model) # L X B X ft

        x = self.PosEmb(x)
        x = self.Dropout(x)

        for enc_i in self.PrenormEncoderLayer:
            x = enc_i(x, padding_mask = padding_mask) # BxLxft - enc_i does permute inside their forward loop

        x = torch.mean(x, dim = 1)
        # x = torch.flatten(x, 1, 2)

        x = self.LayerNorm(x)
        return x


class PrenormEncoderLayer(nn.Module):
    """Prenorm MHA layer."""
    def __init__(self, num_heads: int, dropout_rate: float, mha_dropout_rate: float, dff: int, dim_in: int):
        super(PrenormEncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.mha_dropout_rate = mha_dropout_rate
        self.dff = dff
        self.dim_in = dim_in

        self.layernorm1 = nn.LayerNorm([self.dff]) # Need size 
        self.layernorm2 = nn.LayerNorm([self.dff]) # Need size 

        self.multiheadattention = nn.MultiheadAttention(embed_dim = self.dim_in, num_heads = self.num_heads, dropout=self.mha_dropout_rate)
        self.dropout = nn.Dropout(p = self.dropout_rate)

        self.l1 = nn.Linear(self.dim_in, self.dff)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(self.dff, self.dff)

        # linear layer initialization.
        nn.init.uniform_(self.l1.weight, 0, 0.05)
        nn.init.uniform_(self.l1.bias, 0, 0.05)

        nn.init.uniform_(self.l2.weight, 0, 0.05)
        nn.init.uniform_(self.l2.bias, 0, 0.05)


    def forward(self, x, padding_mask):
        """
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        
        """
        x1 = self.layernorm1(x)

        # pytorch transformer messed up with Batch order 
        x1 = x1.permute((1,0,2))
        x2, _ = self.multiheadattention(query = x1, key = x1, value = x1, key_padding_mask = padding_mask)
        x2 = x2.permute((1,0,2))

        x2 = self.dropout(x2)

        x3 = x + x2

        x4 = self.layernorm2(x3)

        x5 = self.l1(x4)
        x5 = self.relu(x5)
        x5 = self.l2(x5)

        x5 = self.dropout(x5)
        return x3 + x5


class Add1DPositionEmbedding(nn.Module):
    """Adds 1-dimensional positional embeddings to the inputs.
    Attributes:
    rescale_from: tuple; If not None, embeddings are rescaled from this shape.
    max_len: int; Maximum possible length for the input. If None, the max_len is
        set to the inputs sequence length.
    posemb_init: Positional embedding initializer.
    param_name: The name of the parameter that stores the positional embedding.
    """
    def __init__(self, rescale_from: Optional[Sequence[int]] = None, max_len: Optional[int] = None, param_name: str = "pos_embedding", device: str ='cpu'):
        super(Add1DPositionEmbedding, self).__init__()

        self.rescale_from = rescale_from
        self.max_len = max_len
        self.param_name = param_name
        self.device = device
    
    def forward(self, inputs):
        """Applies Add1DPositionEmbedding module.
        Args:
            inputs: nd-arrary; Input data.
        Returns:
            Output: `(bs, timesteps, in_dim)`.
        """
        assert inputs.ndim == 3, ("Number of dimensions should be 3,"
                                    " but it is: %d" % inputs.ndim)
        length = inputs.shape[1]
        max_len = self.max_len or length
        embedding_length = max_len

        if self.rescale_from:  # Shape: `[len, c]`.
            embedding_length = self.rescale_from[0]

        pos_emb_shape = (1, embedding_length, inputs.shape[-1])
        # Use a fixed (non-learned) sinusoidal position embedding.
        pos_embedding = sinusoidal_init(max_len=embedding_length)(None,
                                                                    pos_emb_shape,
                                                                    None)
        pe = pos_embedding[:, :length, :]
        return inputs + pe.to(self.device)

def sinusoidal_init(max_len, max_timescale = 1.0e4):
    """1D Sinusoidal Position Embedding Initializer.
    Args:
        max_len: maximum possible length for the input.
        max_timescale: Maximum time scale.
    Returns:
        output: init function returning `(1, max_len, d_feature)`
    """
    def init(key, shape, dtype = torch.float32):
        """Sinusoidal init.
        The defined API by JAX for a custom initializer is:
            `def init(key, shape, dtype)`
        Even though some of args might be not used, the signature should follow
        this API as JAX passes all the three arguments (key, shape, dtype)
        to the initializers.
        Args:
            key: JAXPRNG key.
            shape: Shape used for making the initialized values.
            dtype: JAX data type.
        Returns:
            Initialized values
        """
        del key, dtype
        d_feature = shape[-1]
        pos_emb = np.zeros((max_len, d_feature), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_feature, 2) * -(np.log(max_timescale) / d_feature))
        pos_emb[:, 0::2] = np.sin(position * div_term)
        pos_emb[:, 1::2] = np.cos(position * div_term)
        pe = pos_emb[np.newaxis, :, :]  # [1, max_len, d_feature]
        return torch.tensor(pe)
    return init