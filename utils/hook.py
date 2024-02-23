from typing import List

import torch
from torch import nn, Tensor


class FeatureMeanVarHook:
    def __init__(self, module: nn.Module, on_input: bool = True, dim: List[int] = [0, 2, 3]):
        self.mean = None
        self.var = None
        self.output = None
        self.hook = module.register_forward_hook(self.hook_fn)
        self.on_input = on_input
        self.module = module
        self.dim = dim

    def hook_fn(self, module: nn.Module, inputs: Tensor, outputs: Tensor):
        if self.on_input:
            feature = inputs[0].clone()
        else:
            feature = outputs.clone()
        self.var, self.mean = torch.var_mean(feature, dim=self.dim, unbiased=True)

    def remove(self):
        self.hook.remove()
        self.output = None
