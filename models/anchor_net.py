from typing import Tuple

import torch

from torch import nn, Tensor

from models import Lambda, View, one_hot


class AnchorNet(nn.Module):
    """AnchorNet

    Args:
      latent_size (int): Latent dimensionality
      num_classes (int): Number of classes

    The AnchorNet module takes an input tensor and a label tensor as input.

    It embeds the class labels, generates a mask based on the embedding,
    masks the input, and passes it through a CNN module.

    The CNN module consists of 1D convolutional and linear layers.

    The weights are initialized from a uniform distribution in __init__.

    The forward pass:
      1. Embeds class labels
      2. Generates mask from label embedding
      3. Masks input tensor
      4. Passes masked input through CNN module
      5. Returns masked output and mask tensor

    """

    def __init__(self, latent_size: int, num_classes: int):
        super().__init__()

        self.num_classes = num_classes
        self.embed_class = nn.Linear(num_classes, latent_size)

        self.mask = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.Linear(latent_size, latent_size),
            nn.Linear(latent_size, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.Sigmoid(),
            Lambda(lambda x: x - 0.5),
            nn.Softsign(),
            nn.ReLU()
        )

        self.module = nn.Sequential(
            View(1, -1),
            nn.Conv1d(1, 4, 3, 1, 1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.Conv1d(4, 8, 3, 1, 1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.Conv1d(8, 4, 3, 1, 1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            View(-1),
            nn.Linear(4 * latent_size, latent_size),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, a=-0.08, b=0.08)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-0.08, b=0.08)
                nn.init.zeros_(m.bias)

    def forward(self, inputs: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        y = self.embed_class(one_hot(kwargs['labels'], self.num_classes))
        mask = self.mask(y)

        masked_inputs = inputs * mask
        z = self.module(masked_inputs)
        # return masked_inputs * z + (1 - masked_inputs) * inputs, mask
        return mask * z + (1 - mask) * inputs, mask
