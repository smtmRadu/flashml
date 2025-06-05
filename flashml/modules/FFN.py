import torch.nn as nn


class FFN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim=4096,
        layers_num=1,
        hidden_activation=nn.SiLU,
    ):
        super(FFN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_dim))
        self.layers.append(hidden_activation())

        for i in range(layers_num - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(hidden_activation())
        self.layers.append(nn.Linear(hidden_dim, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
