import torch
from torch import nn, Tensor


class Model(nn.Module):
    def __init__(self,
                 seq_length: int,
                 d_input: int,
                 kernel_size: int,
                 pool_size: int,
                 dropout: float,
                 d_hidden: int,
                 n_mlp_layers: int,
                 ):
        super().__init__()
        self.conv1 = ConvBlock(d_input, 32, kernel_size)
        self.conv2 = ConvBlock(32, 64, kernel_size)
        self.conv3 = ConvBlock(64, 128, kernel_size)
        self.pool = nn.AvgPool1d(pool_size, pool_size)

        self.flatten = nn.Flatten(1, 2)
        self.dropout = nn.Dropout(p=dropout)

        mlp_input_size = seq_length // (pool_size**3) * 128
        self.mlp = MLP(mlp_input_size, 1, d_hidden, n_mlp_layers)

        self.sigmoid = nn.Sigmoid()

    def forward(self, data: Tensor) -> Tensor:
        """
        :param data: Tensor, shape [batch_size, seq_length, d_input]
        :return: Tensor, shape [batch_size, 1]
        """
        # swap the seq_length and d_model axes because of the expected shape for the convolution
        data = torch.swapaxes(data, 1, 2)  # output shape [batch_size, d_input, seq_length]

        data = self.conv1(data)
        data = self.pool(data)
        data = self.conv2(data)
        data = self.pool(data)
        data = self.conv3(data)
        data = self.pool(data)
        data = self.flatten(data)
        data = self.dropout(data)
        data = self.mlp(data)

        return self.sigmoid(data)


class ConvBlock(nn.Module):
    def __init__(self, d_input: int, d_out: int, kernel_size: int):
        super().__init__()
        self.layer1 = nn.Conv1d(d_input, d_out, kernel_size, padding='same')
        self.layer2 = nn.Conv1d(d_out, d_out, kernel_size, padding='same')
        self.layer3 = nn.Conv1d(d_out, d_out, kernel_size, padding='same')
        self.skip_layer = nn.Conv1d(d_input, d_out, 1, padding="same")

        self.batch_norm = nn.BatchNorm1d(d_out)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        skip = self.skip_layer(x)
        skip = self.batch_norm(skip)

        x = self.layer1(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        x = self.layer2(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        x = self.layer3(x)
        x = self.batch_norm(x)

        x = x + skip
        x = self.activation(x)
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


if __name__ == "__main__":
    seq_length = 4096
    d_input = 33
    kernel_size = 8
    pool_size = 8
    dropout = 0.5
    d_hidden = 128
    n_mlp_layers = 2

    model = Model(seq_length, d_input, kernel_size, pool_size, dropout, d_hidden, n_mlp_layers)

    data = torch.randn((16, seq_length, d_input))
    out = model(data)
    print(out)
