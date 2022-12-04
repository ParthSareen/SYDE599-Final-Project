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
                 n_conv_layers_per_block: int = 2,
                 conv_max_pool_dim: int = 4,
                 kernel_size: int = 7,
                 encoder_dropout: float = 0.5,
                 end_max_pool_dim: int = 4,
                 d_mlp: int = 128,
                 n_mlp_layers: int = 2,
                 dropout: float = 0.5,
                 ):
        super().__init__()
        self.model_type = 'Transformer'

        self.batch_norm = nn.BatchNorm1d(d_input)
        self.conv1 = ConvBlock(d_input, d_model, n_conv_layers_per_block, kernel_size)
        self.conv_max_pool = nn.MaxPool1d(conv_max_pool_dim, stride=conv_max_pool_dim)
        self.dropout1d = nn.Dropout1d(p=dropout)

        # the input to the transformer will have length = transformer_seq_length
        self.transformer_seq_length = seq_length // conv_max_pool_dim

        self.pos_encoder = PositionalEncoding(d_model, encoder_dropout, seq_length=self.transformer_seq_length)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_feed_forward, encoder_dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_encoders)
        self.d_model = d_model

        self.max_pool = nn.MaxPool1d(end_max_pool_dim, stride=end_max_pool_dim)

        self.dropout = nn.Dropout(p=dropout)

        # the output of max pool will be flattened to mlp_input_dimensions
        mlp_input_dimensions = d_model * self.transformer_seq_length // end_max_pool_dim
        self.mlp = MLP(mlp_input_dimensions, 1, d_mlp, n_mlp_layers)

        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, data: Tensor,) -> Tensor:
        """
        :param data: Tensor, shape [batch_size, seq_length, d_input]
        :return: Tensor, shape [batch_size, 1]
        """
        # swap the seq_length and d_model axes because of the expected shape for the convolution
        data = torch.swapaxes(data, 1, 2)  # output shape [batch_size, d_input, seq_length]

        # run through a batch norm layer
        data = self.batch_norm(data)

        # run data through the conv1 block
        data = self.conv1(data)  # output shape [batch_size, d_model, seq_length]

        # run through a max pool
        data = self.conv_max_pool(data)  # output shape [batch_size, d_model, transformer_seq_length]

        # swap the seq_length and d_model axes because of the expected shape for the transformer
        data = torch.swapaxes(data, 1, 2)  # output shape [batch_size, transformer_seq_length, d_model]

        # # add the positional encodings
        # data = self.pos_encoder(data)  # output shape [batch_size, transformer_seq_length, d_model]

        # run through the transformer layers
        data = self.transformer_encoder(data)  # output shape [batch_size, transformer_seq_length, d_model]

        # swap the seq_length and d_model axes because of the expected shape for max pool
        data = torch.swapaxes(data, 1, 2)  # output shape [batch_size, d_model, transformer_seq_length]

        # run through a max pool to reduce the dimensions
        data = self.max_pool(data)  # output shape [batch_size, d_model, transformer_seq_length//max_pool_dim]

        # rearrange the data to 1D to prep for the MLP
        data = torch.flatten(data, 1, 2)  # output shape [batch_size, d_model * transformer_seq_length//max_pool_dim]

        # dropout to prevent overfitting
        data = self.dropout(data)

        # run through the MLP
        data = self.mlp(data)  # output shape [batch_size, 1]

        return self.sigmoid(data)


class ConvBlock(nn.Module):
    def __init__(self, d_input: int, d_out: int, n_layers: int, kernel_size: int):
        super().__init__()
        first_layer = nn.Conv1d(d_input, d_out, kernel_size, padding='same')
        other_layers = [nn.Conv1d(d_out, d_out, kernel_size, padding='same') for _ in range(1, n_layers)]
        self.layers = [first_layer] + other_layers
        self.layers = nn.ModuleList(self.layers)  # To easily store the parameters on the GPU, use a ModuleList

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            residual = x.clone()
            x = layer(x).clone()
            x = nn.ReLU(inplace=False)(x).clone()
            # x += residual
        return x


class MLP(nn.Module):
    def __init__(self, d_input: int, d_out: int, d_hidden: int, n_layers: int):
        super().__init__()
        self.layers = []
        self.layers.append(nn.Linear(d_input, d_hidden))
        for _ in range(2, n_layers):  # num hidden layers = num layers - 2
            self.layers.append(nn.Linear(d_hidden, d_hidden))
        self.layers.append(nn.Linear(d_hidden, d_out))
        self.layers = nn.ModuleList(self.layers)  # To easily store the parameters on the GPU, use a ModuleList

    def init_weights(self):
        init_range = 0.1
        for layer in self.layers:
            layer.bias.data.zero_()
            layer.weight.data.unifrom_(-init_range, init_range)

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
    seq_length = 2048
    d_model = 64
    nhead = 4
    d_feed_forward = 512
    n_encoders = 4
    d_input = 33
    num_conv_layers_per_block = 3
    max_pool_conv = 8
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
        num_conv_layers_per_block,
        max_pool_conv,
        kernel_size,
        encoder_dropout,
        max_pool_dim,
        d_mlp,
        n_mlp_layers,
    )

    batch_size = 16

    data = torch.randn(batch_size, seq_length, d_input)
    out = model(data)
    print(out)
    print(out.shape)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_no_train_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(n_params, n_no_train_params)


if __name__ == "__main__":
    test()
