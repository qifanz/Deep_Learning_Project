import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, 35)
        self.fc2 = nn.Linear(35, 25)
        self.out = nn.Linear(25, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.out(x)
