import torch.nn as nn

# Just a very simple model. Two layers, 64 neurons to each.
class BSModel(nn.Module):
    def __init__(self):
        super(BSModel, self).__init__()
        self.fc_1 = nn.Linear(4, 64)
        self.fc_2 = nn.Linear(64, 64)
        self.fc_3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    # forward pass
    def forward(self, x):
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x