

import torch
import torch.nn as nn
import numpy as np
from .attention import Seq_Transformer



class TC(nn.Module):
    def __init__(self, configs,args, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.TC.timesteps
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device

        self.projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4, heads=4, mlp_dim=64)

    def forward(self, features_aug1, features_aug2, x_k):
        z_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)  32,128,896
        seq_len = z_aug1.shape[2]#896
        z_aug1 = z_aug1.transpose(1, 2)#32,896,128

        z_aug2 = features_aug2
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]#32
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # randomly pick time stamps 16

        nce1 = 0
        nce2 = 0  # average over timestep and batch
        encode_samples1 = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        encode_samples2 = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)#1,32,128

        for i in np.arange(1, self.timestep + 1):
            encode_samples1[i - 1] = z_aug1[:, t_samples + i, :].view(batch, self.num_channels)
            encode_samples2[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)
        forward_seq = z_aug1[:, :t_samples + 1, :]#32,17,128

        x_k = x_k.transpose(1, 2)
        x_k = x_k[:, :t_samples + 1, :]


        c_t = self.seq_transformer(forward_seq, x_k)

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total1 = torch.mm(encode_samples1[i], torch.transpose(pred[i], 0, 1))
            nce1 += torch.sum(torch.diag(self.lsoftmax(total1)))
            total2 = torch.mm(encode_samples2[i], torch.transpose(pred[i], 0, 1))
            nce2 += torch.sum(torch.diag(self.lsoftmax(total2)))
        nce1 /= -1. * batch * self.timestep
        nce2 /= -1. * batch * self.timestep
        return nce1, nce2, self.projection_head(c_t)

