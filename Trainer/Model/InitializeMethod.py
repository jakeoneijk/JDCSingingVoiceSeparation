import torch
from torch import nn
class InitializeMethod():
    def init_weights(self, network):
        if isinstance(network,nn.Linear):
            nn.init.kaiming_uniform_(network.weight)
            if network.bias is not None:
                nn.init.constant_(network.bias,0)
        elif isinstance(network, nn.Conv2d):
            nn.init.xavier_normal_(network.weight)
        elif isinstance(network, nn.LSTM) or isinstance(network, nn.LSTMCell):
            for p in network.parameters():
                if p.data is None:
                    continue
                if len(p.shape) >= 2:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.normal_(p.data)
