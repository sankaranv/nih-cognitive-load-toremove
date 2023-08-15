import math
import os
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from ..utils.data import make_dataset


class TransformerModel(nn.Module):
    """Create a transformer with continuous inputs and outputs.
       Token embeddings are replaced with a linear layer since we have real valued inputs.

    Args:
        d_model (int): Dimension of the model.
        n_heads (int): Number of attention heads.
        d_hidden (int): Dimension of the hidden layer.
        n_layers (int): Number of layers.
        dropout (float, optional): Dropout probability. Defaults to 0.5.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_hidden: int,
        n_layers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_hidden = d_hidden
        self.n_layers = n_layers

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_heads, d_hidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Linear(d_model, d_model)
        self.decoder = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self, init_range: float = 0.1):
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.weight.data.uniform_(-init_range, init_range)
        self.out.bias.data.zero_()
        self.out.weight.data.uniform_(-init_range, init_range)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """Takes in an input sequence and mask and returns an output sequence.

        Args:
            src (Tensor): Input tensor of shape (seq_len, batch_size, d_model).
            src_mask (Tensor): Input mask of shape (seq_len, seq_len).

        Returns:
            Tensor: Output tensor of shape (seq_len, batch_size, 1).
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        output = self.out(output)
        return output


class PositionalEncoding(nn.Module):
    """Encodes absolute or relative position of elements in the sequence of HRV measurements.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional embedding to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (seq_len, batch_size, d_model).

        Returns:
            Tensor: Output tensor of shape (seq_len, batch_size, d_model).
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
