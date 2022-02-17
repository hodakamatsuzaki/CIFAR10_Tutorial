import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

class Conv_Net(nn.Module):
    def __init__(self):
        super(Conv_Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(num_features=6)
        self.act = nn.ReLU()
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.mp(x)
        x = self.conv2(x)
        x = self.mp(x)
        x = x.view(-1, 16 * 5 * 5) #サイズ数を自動的に調整してくれる
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
        
class ResidualBlock(nn.Module):
    def __init__(self, in_units, out_units, kernel_size=3):