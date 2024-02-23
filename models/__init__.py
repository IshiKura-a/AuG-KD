from typing import Callable

import torch
from torch import nn, Tensor
from torchvision.models import get_model, get_model_weights


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * torch.rsqrt((inputs ** 2).mean(dim=1) + 1e-8)


class View(nn.Module):
    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, inputs: Tensor):
        return inputs.view(inputs.shape[0], *self.shape)


class Lambda(nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, inputs: Tensor):
        return self.fn(inputs)


def one_hot(labels: Tensor, num_classes: int) -> Tensor:
    device = labels.device
    targets = torch.zeros(labels.size(0), num_classes).to(device)
    idx = torch.arange(labels.size(0)).to(device)
    targets[idx, labels] = 1
    return targets.float()


def get_pretrained_model(name: str, num_classes: int) -> nn.Module:
    model = get_model(name, weights=list(get_model_weights(name))[-1])
    if 'mobilenet' in name or 'efficientnet' in name:
        d = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(d, num_classes)
    elif 'resnet' in name or 'shufflenet':
        d = model.fc.in_features
        model.fc = nn.Linear(d, num_classes)
    else:
        raise NotImplementedError

    return model
