import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18, resnet50, resnet34
from torchvision.models.mobilenet import mobilenet_v2
from torch.nn import functional as F



# Baseline Lyft
class LyftResnet(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        # load pre-trained Conv2D model
        self.model = resnet50(pretrained=True)

        # change input channels number to match the rasterizer's output
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        self.model.conv1 = nn.Conv2d(
            num_in_channels,
            model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=False,
        )
        # change output size to (X, Y) * number of future states
        num_targets = 2 * cfg["model_params"]["future_num_frames"]
        self.model.fc = nn.Linear(in_features=2048, out_features=num_targets)


    def forward(self, img):
        return model(inputs)