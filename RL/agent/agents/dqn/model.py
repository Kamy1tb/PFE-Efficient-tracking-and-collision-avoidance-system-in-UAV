__version__ = '1.0'
__author__ = 'Miloud Bagaa'
__author_emails__ = 'miloud.bagaa@uqtr.ca, bagmoul@gmail.com'

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class MODELDQN(nn.Module):
    def __init__(self, lr=1e-4, state_space_dim=0, action_space_dim=0, network_spec=[]):
        super(MODELDQN, self).__init__()

        __layers = []
        self.activations = []
        __layers.append(state_space_dim)
        for net_spec in network_spec:
            if net_spec["type"] == "dense":
                __layers.append(net_spec["size"])
                self.activations.append(net_spec["activation"])
        __layers.append(action_space_dim)

        self.layers = T.nn.ModuleList([])
        for value in zip(__layers, __layers[1:]):
            layer = nn.Linear(value[0], value[1])
            T.nn.init.xavier_uniform_(layer.weight)
            f = 1./np.sqrt(layer.weight.data.size()[0])
            T.nn.init.uniform_(layer.bias.data, -f, f)
            self.layers.append(layer)

        #self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, x):
        i = 0
        for layer in self.layers[:-1]:
            x = layer(x)
            # Other functions can be added later. 
            if self.activations[i] == "relu":
                x = F.relu(x)
            i += 1
        x = self.layers[-1:][0](x)
        return x