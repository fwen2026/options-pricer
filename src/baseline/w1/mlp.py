import torch.nn as nn

def create_mlp(input_dim, output_dim, hidden_layers):
    layers = []
    in_dim = input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)