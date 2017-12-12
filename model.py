import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, n_classes):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(5*5*32, 120)
        self.fc2 = nn.Linear(120, n_classes)
        # self.fc3 = nn.Linear(84, n_classes)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, 2)
        x = x.view(-1, 5*5*32)
        x = F.relu(self.fc1(x))
        x = self.fc2(F.dropout(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x
