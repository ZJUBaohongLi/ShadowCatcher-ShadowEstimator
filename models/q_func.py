import torch
import torch.nn as nn


class QFunc(nn.Module):
    def __init__(self, config):
        super(QFunc, self).__init__()
        dim_in = config.get('or_dim_in')
        dim_latent = 1
        self.output_label_layer = nn.Sigmoid()
        self.output_layer = nn.Linear(dim_in, dim_latent)

    def forward(self, x):
        x_rep = x.to(torch.float32)
        label = self.output_label_layer(self.output_layer(x_rep))
        return label

