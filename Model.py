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
                 d_emg: int,
                 d_ecg: int,
                 d_imu: int,
                 d_skin: int,
                 num_init_conv_layers: int = 3,
                 num_all_conv_layers: int = 2,
                 kernel_size: int = 7,
                 d_conv_layers: int = 8,
                 encoder_dropout: float = 0.5,
                 max_pool_dim: int = 4,
                 d_mlp: int = 128,
                 n_mlp_layers: int = 2,
                 ):
        super().__init__()
        self.model_type = 'Transformer'

        self.emg_conv = ConvBlock(d_emg, d_conv_layers, num_init_conv_layers, kernel_size)
        self.ecg_conv = ConvBlock(d_ecg, d_conv_layers, num_init_conv_layers, kernel_size)
        self.imu_conv = ConvBlock(d_imu, d_conv_layers, num_init_conv_layers, kernel_size)
        self.skin_conv = ConvBlock(d_skin, d_conv_layers, num_init_conv_layers, kernel_size)

        self.all_conv = ConvBlock(d_conv_layers*4, d_model, num_all_conv_layers, kernel_size)

        self.pos_encoder = PositionalEncoding(d_model, encoder_dropout, seq_length=seq_length)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_feed_forward, encoder_dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_encoders)
        self.d_model = d_model

        self.max_pool = nn.MaxPool1d(max_pool_dim, stride=max_pool_dim)

        mlp_input_dimensions = d_model * seq_length//max_pool_dim  # the output of max pool will be flattened to this
        self.mlp = MLP(mlp_input_dimensions, 1, d_mlp, n_mlp_layers)

        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,
                emg: Tensor,
                ecg: Tensor,
                imu: Tensor,
                skin: Tensor,) -> Tensor:
        """

        :param emg: Tensor, shape [batch_size, d_emg, seq_length]
        :param ecg: Tensor, shape [batch_size, d_ecg, seq_length]
        :param imu: Tensor, shape [batch_size, d_imu, seq_length]
        :param skin: Tensor, shape [batch_size, d_skin, seq_length]
        :return: Tensor, shape [batch_size, 1]
        """
        # run each input through a convolution block
        emg_out = self.emg_conv(emg)  # output shape [batch_size, d_conv_layers, seq_length]
        ecg_out = self.ecg_conv(ecg)  # output shape [batch_size, d_conv_layers, seq_length]
        imu_out = self.imu_conv(imu)  # output shape [batch_size, d_conv_layers, seq_length]
        skin_out = self.skin_conv(skin)  # output shape [batch_size, d_conv_layers, seq_length]

        # concatanate inputs. output shape [batch_size, 4*d_conv_layers, seq_length]
        data = torch.cat([emg_out, ecg_out, imu_out, skin_out], dim=1)

        # run data through the all conv block
        data = self.all_conv(data)  # output shape [batch_size, d_model, seq_length]

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


    # def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
    #     """
    #     Args:
    #         src: Tensor, shape [seq_len, batch_size]
    #         src_mask: Tensor, shape [seq_len, seq_len]
    #
    #     Returns:
    #         output Tensor of shape [seq_len, batch_size, ntoken]
    #     """
    #     src = self.encoder(src) * math.sqrt(self.d_model)
    #     src = self.pos_encoder(src)
    #     output = self.transformer_encoder(src, src_mask)
    #     output = self.decoder(output)
    #     return output


class ConvBlock(nn.Module):
    def __init__(self, d_input: int, d_out: int, n_layers: int, kernel_size: int):
        first_layer = nn.Conv1d(d_input, d_out, kernel_size)
        other_layers = [nn.Conv1d(d_out, d_out, kernel_size) for _ in range(1, n_layers)]
        self.layers = [first_layer] + other_layers

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
            x = nn.ReLU()(x)
        return x


class MLP(nn.Module):
    def __init__(self, d_input: int, d_out: int, d_hidden: int, n_layers: int):
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


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, seq_length: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos = nn.Parameter(torch.randn(d_model, seq_length))
        self.pos.requires_grad = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_length, d_model]
        """
        x = x + self.pos
        return self.dropout(x)
