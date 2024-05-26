import torch
import torch.nn as nn


class ZNet(nn.Module):
    def __init__(self, config):
        super(ZNet, self).__init__()
        dim_in = config.get('z_dim_in')
        dim_latent = config.get('generator_dim_latent')
        dim_out = config.get('z_dim_out')
        layer_num = config.get('z_layer_num')
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_in - (dim_in - dim_out) * i // layer_num,
                          dim_in - (dim_in - dim_out) * (i + 1) // layer_num),
                nn.BatchNorm1d(dim_in - (dim_in - dim_out) * (i + 1) // layer_num),
                nn.ReLU(),
            ) for i in range(layer_num)
        ])
        self.output_layer = nn.Linear(dim_out, dim_latent)
        self.output_label_layer = nn.Sigmoid()

    def forward(self, x):
        x_rep = x.to(torch.float32)
        for layer in self.mlp:
            x_rep = layer(x_rep)
        label = self.output_layer(x_rep)
        label = self.output_label_layer(label)
        return label

