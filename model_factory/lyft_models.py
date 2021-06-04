import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torchvision.models.resnet import *
from torchvision.models.mobilenet import mobilenet_v2
from torch.nn import functional as F
import timm


## ----------------------------------------------------------------------------------------------------------------------
## Resnets
## ----------------------------------------------------------------------------------------------------------------------

# Resnet single mode
class LyftResnet(nn.Module):
    
    def __init__(self, cfg, architecture="resnet18"):
        super().__init__()
        
        if architecture=="resnet18":
            self.model = resnet18(pretrained=True)
        elif architecture=="resnet34":
            self.model = resnet34(pretrained=True)
        elif architecture=="resnet50":
            self.model = resnet50(pretrained=True)

        # change input channels number to match the rasterizer's output
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        self.model.conv1 = nn.Conv2d(
            num_in_channels,
            self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            bias=False,
        )
        # change output size to (X, Y) * number of future states
        num_targets = 2 * cfg["model_params"]["future_num_frames"]
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=num_targets)
        
    def forward(self, img):
        return self.model(img)

# Resnet multi mode
class LyftResnetMulti(nn.Module):

    def __init__(self, cfg, num_modes=3, architecture="resnet18"):
        super().__init__()
        
        if architecture=="resnet18":
            self.model = resnet18(pretrained=True)
        elif architecture=="resnet34":
            self.model = resnet34(pretrained=True)
        elif architecture=="resnet50":
            self.model = resnet50(pretrained=True)

        # change input channels number to match the rasterizer's output
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        self.model.conv1 = nn.Conv2d(
            num_in_channels,
            self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            bias=False,
        )
        
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * cfg["model_params"]["future_num_frames"]
        
        self.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=2048)
        
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.logit = nn.Linear(2048, out_features=self.num_preds + num_modes)

    def forward(self, inputs):
        
        #####
        # Resnet blocks start
        #####
        x = self.model.conv1(inputs)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        
        #####
        # Resnet blocks end
        #####
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.logit(x)

        # pred (batch_size)x(modes)x(time)x(2D coords)
        # confidences (batch_size)x(modes)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes) 
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences
    
    
## ----------------------------------------------------------------------------------------------------------------------
## Mixnets
## ----------------------------------------------------------------------------------------------------------------------

# Mixnet single mode
class LyftMixnet(nn.Module):
    
    def __init__(self, cfg, pretrained=True, architecture="mixnet_xl"):
        super().__init__()
        
        # set pretrained=True while training
        self.backbone = timm.create_model(architecture, pretrained=pretrained) 
        
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        
        self.backbone.conv_stem = nn.Conv2d(
            num_in_channels,
            self.backbone.conv_stem.out_channels,
            kernel_size=self.backbone.conv_stem.kernel_size,
            stride=self.backbone.conv_stem.stride,
            padding=self.backbone.conv_stem.padding,
            bias=False,
        )
        
        # X, Y coords for the future positions (output shape: Bx50x2)
        num_targets = 2 * cfg["model_params"]["future_num_frames"]
        self.logit = nn.Linear(self.backbone.classifier.out_features, out_features=num_targets)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.logit(x)
        return x
    
    
# Mixnet multi mode
class LyftMixnetMulti(nn.Module):
    
    def __init__(self, cfg, num_modes=3, pretrained=True, architecture="mixnet_xl"):
        super().__init__()
        
        # set pretrained=True while training
        self.backbone = timm.create_model(architecture, pretrained=pretrained) 
        # get input size of last layer
        in_features = self.backbone.classifier.in_features
        
        # remove fc layer from the backbone
        self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))
        
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        
        self.backbone[0] = nn.Conv2d(
            num_in_channels,
            self.backbone[0].out_channels,
            kernel_size=self.backbone[0].kernel_size,
            stride=self.backbone[0].stride,
            padding=self.backbone[0].padding,
            bias=False,
        )
        
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * cfg["model_params"]["future_num_frames"]
        
        self.fc = nn.Linear(in_features=in_features, out_features=2048)
        
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.logit = nn.Linear(2048, out_features=self.num_preds + num_modes)
        
        
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.logit(x)
        
        # pred (batch_size)x(modes)x(time)x(2D coords)
        # confidences (batch_size)x(modes)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes) 
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences
