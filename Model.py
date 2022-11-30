import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


class Model(nn.Module):
    def __init__(self,
                 seq_length: int,
                 d_model: int,
                 nhead: int,
                 d_feed_forward: int,
                 n_encoders: int,
                 d_input: int,
                 num_conv_layers: int = 2,
                 kernel_size: int = 7,
                 encoder_dropout: float = 0.5,
                 max_pool_dim: int = 4,
                 d_mlp: int = 128,
                 n_mlp_layers: int = 2,
                 ):
        super().__init__()
        self.model_type = 'Transformer'

        self.conv = ConvBlock(d_input, d_model, num_conv_layers, kernel_size)

        self.pos_encoder = PositionalEncoding(d_model, encoder_dropout, seq_length=seq_length)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_feed_forward, encoder_dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_encoders)
        self.d_model = d_model

        self.max_pool = nn.MaxPool1d(max_pool_dim, stride=max_pool_dim)

        mlp_input_dimensions = d_model * seq_length // max_pool_dim  # the output of max pool will be flattened to this
        self.mlp = MLP(mlp_input_dimensions, 1, d_mlp, n_mlp_layers)

        self.sigmoid = nn.Sigmoid()

    def forward(self, data: Tensor,) -> Tensor:
        """
        :param data: Tensor, shape [batch_size, d_input, seq_length]
        :return: Tensor, shape [batch_size, 1]
        """
        # run data through the conv block
        data = self.conv(data)  # output shape [batch_size, d_model, seq_length]

        # swap the seq_length and d_model axes because of the expected shape for the transformer
        data = torch.swapaxes(data, 1, 2)  # output shape [batch_size, seq_length, d_model]

        # add the positional encodings
        data = self.pos_encoder(data)  # output shape [batch_size, seq_length, d_model]

        # run through the transformer layers
        data = self.transformer_encoder(data)  # output shape [batch_size, seq_length, d_model]

        # swap the seq_length and d_model axes because of the expected shape for max pool
        data = torch.swapaxes(data, 1, 2)  # output shape [batch_size, d_model, seq_length]

        # run through a max pool to reduce the dimensions
        data = self.max_pool(data)  # output shape [batch_size, d_model, seq_length//max_pool_dim]

        # rearrange the data to 1D to prep for the MLP
        data = torch.flatten(data, 1, 2)  # output shape [batch_size, d_model * seq_length//max_pool_dim]

        # run through the MLP
        data = self.mlp(data)  # output shape [batch_size, 1]

        return self.sigmoid(data)


class ConvBlock(nn.Module):
    def __init__(self, d_input: int, d_out: int, n_layers: int, kernel_size: int):
        super().__init__()
        first_layer = nn.Conv1d(d_input, d_out, kernel_size, padding='same')
        other_layers = [nn.Conv1d(d_out, d_out, kernel_size, padding='same') for _ in range(1, n_layers)]
        self.layers = [first_layer] + other_layers

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
            x = nn.ReLU()(x)
        return x


class MLP(nn.Module):
    def __init__(self, d_input: int, d_out: int, d_hidden: int, n_layers: int):
        super().__init__()
        self.layers = []
        self.layers.append(nn.Linear(d_input, d_hidden))
        for _ in range(2, n_layers):  # num hidden layers = num layers - 2
            self.layers.append(nn.Linear(d_hidden, d_hidden))
        self.layers.append(nn.Linear(d_hidden, d_out))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nn.ReLU()(x)
        output = self.layers[-1](x)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, seq_length: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos = nn.Parameter(torch.randn(seq_length, d_model))
        self.pos.requires_grad = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_length, d_model]
        """
        x = x + self.pos.unsqueeze(0)
        return self.dropout(x)


def test():
    seq_length = 512
    d_model = 32
    nhead = 4
    d_feed_forward = 512
    n_encoders = 3
    d_input = 33
    num_conv_layers = 3
    kernel_size = 8
    encoder_dropout = 0.5
    max_pool_dim = 4
    d_mlp = 128
    n_mlp_layers = 2

    model = Model(
        seq_length,
        d_model,
        nhead,
        d_feed_forward,
        n_encoders,
        d_input,
        num_conv_layers,
        kernel_size,
        encoder_dropout,
        max_pool_dim,
        d_mlp,
        n_mlp_layers,
    )

    batch_size = 16

    data = torch.randn(batch_size, d_input, seq_length)
    out = model(data)
    print(out)
    print(out.shape)


if __name__ == "__main__":
    test()
