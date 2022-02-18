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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        padding = kernel_size//2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=False,)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, bias=False,)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()
        
        if in_channels!=out_channels or stride!=1:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        else:
            self.skip_conv = None
            
    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.skip_conv is not None:
            x += self.skip_conv(input_tensor)
        else:
            x += input_tensor
        x = self.act2(x)
        return x
    
class ResNet34(nn.Module):
    def __init__(self, in_channels=3, out_shape=10):
        super(ResNet34, self).__init__()
        
        self.conv_first = nn.Conv2d(in_channels, 64, kernel_size=7,
                                    padding=7//2, bias=False)
        self.bn_first = nn.BatchNorm2d(64)
        self.act_first = nn.ReLU()
        self.pool_first = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res_1 = nn.Sequential(OrderedDict([(f'res_1_{i+1}',
                                                 ResidualBlock(64, 64)) for i in range(3)]))
        self.res_2_1 = ResidualBlock(64, 128, stride=2)#
        self.res_2_2 = nn.Sequential(OrderedDict([(f'res_2_{i+1}',
                                                 ResidualBlock(128, 128)) for i in range(3)]))
        self.res_3_1 = ResidualBlock(128, 256, stride=2)#
        self.res_3_2 = nn.Sequential(OrderedDict([(f'res_3_{i+1}',
                                                 ResidualBlock(256, 256)) for i in range(5)]))
        self.res_4_1 = ResidualBlock(256, 512, stride=2)#
        self.res_4_2 = nn.Sequential(OrderedDict([(f'res_4_{i+1}',
                                                 ResidualBlock(512, 512)) for i in range(2)]))
        
        self.last_layer = nn.Linear(512, out_shape)
        
    def forward(self, x):
        
        x = self.conv_first(x)
        x = self.bn_first(x)
        x = self.act_first(x)
        x = self.pool_first(x)
        
        x = self.res_1(x)#(N, C, H, W)
        
        x = self.res_2_1(x)
        x = self.res_2_2(x)
        
        x = self.res_3_1(x)
        x = self.res_3_2(x)
        
        x = self.res_4_1(x)
        x = self.res_4_2(x)
        
        #GAP
        x = torch.mean(x, dim=(2, 3))#(N, C, H, W) => (N, C)
        x = self.last_layer(x)#(B, C)
        x = torch.log_softmax(x, dim=-1)
        
        return x