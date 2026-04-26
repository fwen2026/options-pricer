import torch.nn as nn
from src.constants import MODEL_LAYERS


class BSModel(nn.Module):
    def __init__(self, layers=MODEL_LAYERS):
        super(BSModel, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x