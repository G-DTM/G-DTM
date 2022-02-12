from __future__ import print_function

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, indim, outdim, dropout=0):
        super(MLP, self).__init__()

        def block(indim, outdim, active=True):
            layer = [nn.Linear(indim, outdim)]
            if active:
                layer.append(nn.LeakyReLU(0.2, inplace=True))
            return layer

        self.model = nn.Sequential(
            *block(indim, 512),
            nn.Dropout(p=dropout),
            *block(512, 256),
            nn.Dropout(p=dropout),
            *block(256,128),
            *block(128, outdim, active=False))

    def forward(self, x):
        return self.model(x)

class evolveLSTM(nn.Module):
    def __init__(self, in_dim, rnn_outdim, out_dim):
        super(evolveLSTM,self).__init__()
        # LSTM: num_layers=1
        self.lstm = nn.LSTM(in_dim, rnn_outdim)
        # self.mlp = MLP(rnn_outdim*2, out_dim)
        self.mlp = MLP(rnn_outdim+in_dim, out_dim)
    
    def forward(self, x):
        trx, content_info = x
        output, (hn,cn) = self.lstm(trx)
        x = torch.cat((output[-1], content_info), 1)
        out = self.mlp(x)
        return output, out
