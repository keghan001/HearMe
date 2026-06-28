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
                            3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        self.use_shortcut = stride != 1 or in_channels != out_channels
        
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x, fmap_dict=None, prefix=""):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        shortcut = self.shortcut(x) if self.use_shortcut else x
        out_add = out + shortcut
        
        if fmap_dict is not None:
            fmap_dict[f"{prefix}.conv"] = out_add
            
        out = F.relu(out_add)
        if fmap_dict is not None:
            fmap_dict[f"{prefix}.relu"] = out
        
        return out
    

class AudioCNN(nn.Module):
    """Convolutional Neural Network model for audio classisfication"""
    
    def __init__(self, num_classes=50) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=3, padding=1)
        )
        
        self.layer1 = nn.ModuleList([ResidualBlock(64,64) for i in range(3)])
        self.layer2 = nn.ModuleList(
            [ResidualBlock(64 if i == 0 else  128,128, stride=2 if i==0 else 1) for i in range(4)])
        self.layer3 = nn.ModuleList(
            [ResidualBlock(128 if i == 0 else 256,256, stride=2 if i==0 else 1) for i in range(6)])
        self.layer4 = nn.ModuleList(
            [ResidualBlock(256 if i == 0 else 512, 512, stride=2 if i==0 else 1) for i in range(3)])
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x, return_feature_maps=False):
        if not return_feature_maps:
            out = self.conv1(x)
            
            for block in self.layer1:
                out = block(out)
            for block in self.layer2:
                out = block(out)
            for block in self.layer3:
                out = block(out)
            for block in self.layer4:
                out = block(out)
                
            
            out = self.classifier(out)
            
            return out
        else: 
            feature_maps = {}
            out = self.conv1(x)
            feature_maps["conv1"] = out
            
            for i, block in enumerate(self.layer1):
                out = block(out, feature_maps, prefix=f"layer1.block{i}")
            feature_maps["layer1"] = out
            
            for i, block in enumerate(self.layer2):
                out = block(out, feature_maps, prefix=f"layer2.block{i}")
            feature_maps["layer2"] = out
                
            for i, block in enumerate(self.layer3):
                out = block(out, feature_maps, prefix=f"layer3.block{i}")
            feature_maps["layer3"] = out
                
            for i, block in enumerate(self.layer4):
                out = block(out, feature_maps, prefix=f"layer4.block{i}")
            feature_maps["layer4"] = out
                
            
            out = self.classifier(out)
            
            return out, feature_maps
    


if __name__ == "__main__":
    aud = AudioCNN()
    print(aud.classifier[-1])
        
