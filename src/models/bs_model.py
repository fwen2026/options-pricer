import torch.nn as nn


class BSModel(nn.Module):
    def __init__(self):
        super(BSModel, self).__init__()
        self.fc_1 = nn.Linear(4, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, 64)
        self.fc_4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        x = self.relu(self.fc_3(x))
        x = self.fc_4(x)
        return x