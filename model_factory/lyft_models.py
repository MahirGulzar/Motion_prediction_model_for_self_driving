import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18, resnet50, resnet34
from torchvision.models.mobilenet import mobilenet_v2
from torch.nn import functional as F

class LyftResnet50(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        
        # load pre-trained Conv2D model
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
        # This is 2048 for resnet50
        self.model.fc = nn.Linear(in_features=2048, out_features=num_targets)
        
    def forward(self, img):
        return self.model(img)


class LyftResnet34(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        
        # load pre-trained Conv2D model
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
        # This is 512 for resnet34
        self.model.fc = nn.Linear(in_features=512, out_features=num_targets)
        
    def forward(self, img):
        return self.model(img)
    
    
class LyftResnet34Multi(nn.Module):

    def __init__(self, cfg, num_modes=3):
        super().__init__()
        
        # load pre-trained Conv2D model
        self.model = resnet34(pretrained=True)

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
        
        # This is 2048 for resnet50
        self.fc = nn.Linear(in_features=512, out_features=2048)
        
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
