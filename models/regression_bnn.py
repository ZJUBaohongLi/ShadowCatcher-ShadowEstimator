import torch
import torch.nn as nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class RegressionBNN(nn.Module):
    def __init__(self, config):
        super(RegressionBNN, self).__init__()
        dim_in = config.get('regression_dim_in')
        dim_latent = config.get('regression_dim_latent')
        dim_out = config.get('regression_dim_out')
        layer_num = config.get('regression_layer_num')
        self.rep_layer_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_in - (dim_in - dim_out) * i // layer_num,
                          dim_in - (dim_in - dim_out) * (i + 1) // layer_num),
                nn.BatchNorm1d(dim_in - (dim_in - dim_out) * (i + 1) // layer_num),
                nn.ReLU(),
            ) for i in range(layer_num)
        ])
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_out + 1, dim_out + 1),
                nn.BatchNorm1d(dim_out + 1),
                nn.ReLU(),
            ) for i in range(layer_num)
        ])
        self.output_layer = nn.Linear(dim_out + 1, dim_latent)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t, x):
        x_rep = x.to(torch.float32)
        for layer in self.rep_layer_list:
            x_rep = layer(x_rep)
        t = t.to(torch.float32)
        t_ind = torch.nonzero(t.reshape(-1) == 1).reshape(-1).to(device)
        c_ind = torch.nonzero(t.reshape(-1) == 0).reshape(-1).to(device)
        t_rep = x_rep[t_ind]
        c_rep = x_rep[c_ind]
        y_pre = torch.cat((x_rep, t), 1)
        for layer in self.mlp:
            y_pre = layer(y_pre)
        y_pre = self.output_layer(y_pre)
        res = y_pre, t_rep, c_rep
        return res
