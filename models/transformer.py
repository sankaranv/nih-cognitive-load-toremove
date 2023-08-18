import math
import os
from typing import Tuple
import sys
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ContinuousTransformer(nn.Module):
    """Create a transformer with continuous inputs and outputs.
       Token embeddings are replaced with a linear layer since we have real valued inputs.
       Input dimensions are (seq len, batch_size, n_features)
    Args:
        d_model (int): Dimension of the model.
        n_heads (int): Number of attention heads.
        d_hidden (int): Dimension of the hidden layer.
        n_layers (int): Number of layers.
        n_features (int): Number of features in the input for multi-dimensional time series.
        dropout (float, optional): Dropout probability. Defaults to 0.5.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_hidden: int,
        n_layers: int,
        n_in_features: int,
        n_out_features: int,
        n_phases: int,
        dropout: float = 0.5,
        max_len: int = 122,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.n_in_features = n_in_features
        self.n_out_features = n_out_features
        self.dropout = dropout

        self.encoder = nn.Linear(n_in_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.phase_encoder = PhaseEncoding(d_model, n_phases, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_heads, d_hidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.Linear(d_model, n_out_features)
        self.out = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self, init_range: float = 0.1):
        self.phase_encoder.embeddings.weight.data.uniform_(-init_range, init_range)
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.weight.data.uniform_(-init_range, init_range)
        self.out.bias.data.zero_()
        self.out.weight.data.uniform_(-init_range, init_range)

    def forward(
        self, src: Tensor, src_mask: Tensor, temporal_features: Tensor
    ) -> Tensor:
        """Takes in an input sequence and mask and returns an output sequence.
        The input mask specifies where there are NaNs in the data so they will be ignored in the attention mechanism.

        Args:
            src (Tensor): Input tensor of shape (seq_len, batch_size, n_in_features).
            src_mask (Tensor): Input mask of shape (seq_len, batch_size, n_in_features)

        Returns:
            Tensor: Output tensor of shape (seq_len, batch_size, n_out_features).
        """
        # TODO
        # Temporary fix for NaN masking
        src = torch.nan_to_num(src)
        # Continue with transformer
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src, temporal_features[:, :, -1])
        src = self.phase_encoder(src, temporal_features[:, :, :-1])
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):
    """Encodes absolute position of elements in the sequence of HRV measurements.
        Inputs have shape (seq_len, batch_size, d_model)
        This is given using sin and cos functions of the inputs so there is no need to learn the position embeddings.
    Args:
        d_model (int): Dimension of the model.
        max_len (int): Maximum length of the sequence.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
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

    def forward(self, x: Tensor, time_indices) -> Tensor:
        """Add positional embedding to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (seq_len, batch_size, d_model).
            time_indices (Tensor): Tensor of shape (seq_len, batch_size, 1) containing the time indices of the input.

        Returns:
            Tensor: Output tensor of shape (seq_len, batch_size, d_model).
        """
        # Figure out how to vectorize this later
        for i in range(x.shape[0]):
            x[i, :, :] += self.pe[time_indices[i].long(), 0, :]
        return self.dropout(x)


class PhaseEncoding(nn.Module):
    """Embeds phase ID as a vector of size d_model using a token-vocabulary lookup table.
    Embeddings are summed for each phase that is active
    An extra embedding is added to represent the absence of a phase.
    Imputs have shape (seq_len, batch_size, d_model)"""

    def __init__(self, d_model: int, n_phases: int, dropout: float = 0.1):
        super().__init__()
        self.n_phases = n_phases
        self.embeddings = nn.Embedding(n_phases + 1, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, phase_one_hot: Tensor) -> Tensor:
        """For each of the phases that are active, add the corresponding embedding to the input tensor."""

        for i in range(self.n_phases + 1):
            emb = torch.zeros(x.shape)
            emb += self.embeddings(torch.Tensor([i]).long())
            # Swapaxes is necessary for broadcasting
            emb.swapaxes_(0, 2).swapaxes_(1, 2)
            emb *= phase_one_hot[:, :, i]
            emb.swapaxes_(1, 2).swapaxes_(0, 2)
            x += emb
        return self.dropout(x)
