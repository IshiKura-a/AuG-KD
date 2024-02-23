import logging
from abc import abstractmethod
from typing import List, Tuple, Callable, Optional, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from models import View, one_hot, Lambda
from utils.logger import logger


class BaseVAE(nn.Module):

    def __init__(self):
        super(BaseVAE, self).__init__()

    @abstractmethod
    def encode(self, inputs: Tensor) -> List[Tensor]:
        pass

    @abstractmethod
    def decode(self, inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass


class Base(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 latent_size: int,
                 hidden_dims: List = None,
                 img_size: int = 64,
                 **kwargs):
        super().__init__()
        self._kernel_size = kwargs['kernel_size'] if 'kernel_size' in kwargs else 5
        self._stride = kwargs['stride'] if 'stride' in kwargs else 3
        self._padding = kwargs['padding'] if 'padding' in kwargs else 2

        self.latent_size = latent_size
        self.num_classes = num_classes
        self.n = img_size

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()


class Encoder(Base):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 latent_size: int,
                 hidden_dims: List = None,
                 img_size: int = 64,
                 scale_strategy: List[float] = None,
                 **kwargs):
        super().__init__(in_channels, num_classes, latent_size, hidden_dims, img_size, **kwargs)

        self.embed_class = nn.Sequential(nn.Linear(num_classes, img_size * img_size),
                                         View(1, img_size, img_size))
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        in_channels += 1  # To account for the extra label channel
        modules = [nn.BatchNorm2d(in_channels)]

        for scale in scale_strategy:
            self.n = int(self.n * scale)

        # Build Encoder
        prev_dim = in_channels
        for idx, h in enumerate(self.hidden_dims):
            if idx < len(scale_strategy):
                scale = scale_strategy[idx]
                modules.append(Lambda(lambda x: F.interpolate(x, scale_factor=scale)))
            if idx < len(hidden_dims) - 1:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(prev_dim, out_channels=h,
                                  kernel_size=self._kernel_size, stride=self._stride, padding=self._padding,
                                  bias=False),
                        nn.BatchNorm2d(h),
                        nn.LeakyReLU(0.2, inplace=True))
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(prev_dim, out_channels=h,
                                  kernel_size=self._kernel_size, stride=self._stride, padding=self._padding),
                        nn.BatchNorm2d(h),
                        nn.Tanh())
                )
            prev_dim = h

        modules.append(nn.Flatten())
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.hidden_dims[-1] * self.n * self.n, latent_size)
        self.fc_var = nn.Linear(self.hidden_dims[-1] * self.n * self.n, latent_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, a=-0.08, b=0.08)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-0.08, b=0.08)
                nn.init.zeros_(m.bias)

    def forward(self, inputs: Tensor, **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        y = one_hot(kwargs['labels'], self.num_classes)
        embedded_class = self.embed_class(y)
        embedded_class = embedded_class
        embedded_input = self.embed_data(inputs)
        x = torch.cat([embedded_input, embedded_class], dim=1)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x).clip(min=-5, max=3)
        z = self.reparameterize(mu, log_var)

        if 'return_dist' in kwargs.keys() and kwargs['return_dist']:
            return z, mu, log_var
        else:
            return z

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param log_var: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std).clip(min=-10, max=10)
        return eps * std + mu


class Decoder(Base):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 latent_size: int,
                 hidden_dims: List = None,
                 img_size: int = 64,
                 **kwargs):
        super().__init__(in_channels, num_classes, latent_size, hidden_dims, img_size, **kwargs)

        output_padding = []
        for i in range(len(self.hidden_dims)):
            tmp = (self.n + 2 * self._padding - self._kernel_size + self._stride)
            self.n = tmp // self._stride
            output_padding.append(tmp - self._stride * self.n)
        output_padding.reverse()

        in_channels += 1  # To account for the extra label channel

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_size + num_classes, self.hidden_dims[-1] * self.n * self.n)

        self.hidden_dims.reverse()

        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=self._kernel_size,
                                       stride=self._stride,
                                       padding=self._padding,
                                       output_padding=output_padding[i]),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[-1],
                               self.hidden_dims[-1],
                               kernel_size=self._kernel_size,
                               stride=self._stride,
                               padding=self._padding,
                               output_padding=output_padding[-1]),
            nn.BatchNorm2d(self.hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_dims[-1], out_channels=in_channels - 1,
                      kernel_size=self._kernel_size, padding=(self._kernel_size - 1) // 2),
            nn.Tanh()
        )

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        y = one_hot(kwargs['labels'], self.num_classes)
        z = torch.cat([inputs, y], dim=1)
        z = self.decoder_input(z)
        z = z.view(-1, self.hidden_dims[-1], self.n, self.n)
        z = self.decoder(z)
        z = self.final_layer(z)
        return z


class ConditionalVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 latent_size: int,
                 hidden_dims: List = None,
                 img_size: int = 64,
                 **kwargs) -> None:
        super(ConditionalVAE, self).__init__()
        self.encoder = Encoder(in_channels, num_classes, latent_size, hidden_dims, img_size, **kwargs)
        self.decoder = Decoder(in_channels, num_classes, latent_size, hidden_dims, img_size, **kwargs)

    def encode(self, inputs: Tensor, **kwargs) -> List[Tensor]:
        return self.encoder(inputs, **kwargs, return_dist=True)

    def decode(self, z: Tensor, **kwargs) -> Tensor:
        return self.decoder(z, **kwargs)

    def forward(self, inputs: Tensor, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        z, mu, log_var = self.encode(inputs, **kwargs)
        return self.decode(z, **kwargs), mu, log_var


class Generator(Base):
    def __init__(self, in_channels: int,
                 num_classes: int,
                 latent_size: int,
                 hidden_dims: Optional[List[int]],
                 img_size: int = 64,
                 scale_strategy: List[int] = None,
                 **kwargs):
        super().__init__(in_channels, num_classes, latent_size, hidden_dims, img_size, **kwargs)

        tot_scale = 1
        for scale in scale_strategy:
            tot_scale *= scale

        self.init_size = img_size // tot_scale

        self.embed = nn.Sequential(nn.Linear(latent_size, hidden_dims[0] * self.init_size ** 2),
                                   View(hidden_dims[0], self.init_size, self.init_size),
                                   nn.BatchNorm2d(hidden_dims[0]))

        modules = []

        for idx, h in enumerate(hidden_dims):
            if idx < len(scale_strategy):
                modules.append(nn.Upsample(scale_factor=scale_strategy[idx]))

            if idx < len(hidden_dims) - 1:
                modules.append(nn.Conv2d(h, hidden_dims[idx + 1], kernel_size=self._kernel_size, stride=self._stride,
                                         padding=self._padding, bias=False))
                modules.append(nn.BatchNorm2d(hidden_dims[idx + 1]))
                modules.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                modules.append(nn.Conv2d(h, in_channels, kernel_size=self._kernel_size, stride=self._stride,
                                         padding=self._padding))
                modules.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*modules)

    def forward(self, z: Tensor, **kwargs):
        z = self.embed(z)
        z = self.decoder(z)
        return z


class CondGenerator(Base):
    def __init__(self, in_channels: int,
                 num_classes: int,
                 latent_size: int,
                 hidden_dims: Optional[List[int]],
                 img_size: int = 64,
                 scale_strategy: List[int] = None,
                 **kwargs):
        super().__init__(in_channels, num_classes, latent_size, hidden_dims, img_size, **kwargs)

        tot_scale = 1
        for scale in scale_strategy:
            tot_scale *= scale

        self.init_size = img_size // tot_scale

        self.embed = nn.Sequential(nn.Linear(latent_size + num_classes, hidden_dims[0] * self.init_size ** 2),
                                   View(hidden_dims[0], self.init_size, self.init_size),
                                   nn.BatchNorm2d(hidden_dims[0]))

        modules = []

        for idx, h in enumerate(hidden_dims):
            if idx < len(scale_strategy):
                modules.append(nn.Upsample(scale_factor=scale_strategy[idx]))

            if idx < len(hidden_dims) - 1:
                modules.append(nn.Conv2d(h, hidden_dims[idx + 1], kernel_size=self._kernel_size, stride=self._stride,
                                         padding=self._padding, bias=False))
                modules.append(nn.BatchNorm2d(hidden_dims[idx + 1]))
                modules.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                modules.append(nn.Conv2d(h, in_channels, kernel_size=self._kernel_size, stride=self._stride,
                                         padding=self._padding))
                modules.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*modules)

    def forward(self, z: Tensor, **kwargs):
        y = one_hot(kwargs['labels'], self.num_classes)
        z = torch.cat([z, y], dim=1)
        z = self.embed(z)
        z = self.decoder(z)
        return z
