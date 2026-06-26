import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()
        
        
        #Building the residual sequence
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                            3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                            3, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        self.use_shortcut = stride != 1 or in_channels != out_channels
        
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = self.conv2(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        shortcut = self.shortcut(x) if self.use_shortcut else x
        out_add = out + shortcut
        out = F.relu(out_add)
        
        return out
    

class AudioCNN(nn.Module):
    def __init__(self, num_classes=50) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(64)
        )
        
        self.layer1 = nn.ModuleList([ResidualBlock(64,64) for i in range(3)])
        self.layer2 = nn.ModuleList(
            [ResidualBlock(64 if i == 0 else  128,128) for i in range(4)])
        self.layer3 = nn.ModuleList(
            [ResidualBlock(128 if i == 0 else 256,256) for i in range(6)])
        self.layer4 = nn.ModuleList(
            [ResidualBlock(256 if i == 0 else 512, 256) for i in range(3)])
        
        
        
        
