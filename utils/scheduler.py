import numpy as np

from abc import abstractmethod

import torch
from torch.nn import functional as F


class Scheduler:
    """
    Class: Scheduler
    Description: Given the total epoch and current epoch, output the uncertainty threshold s, which means data with
        uncertainty under s will be selected.
    """

    def __init__(self, epoch: int, a: float = 0, b: float = 1):
        self.epoch = epoch
        self.a = a
        self.b = b

    @abstractmethod
    def threshold(self, cur: int) -> float:
        raise NotImplementedError()


class LinearScheduler(Scheduler):
    """
    Class: LinearScheduler
    Description: TODO
    """

    def threshold(self, cur: int) -> float:
        return min((1 - self.b) * (cur / (self.a * self.epoch)) + self.b, 1)


class StepScheduler(Scheduler):
    """
    Class: LinearScheduler
    Description: TODO
    """

    def threshold(self, cur: int) -> float:
        return min(self.b + (1 - self.b) * np.trunc(10 * cur / (self.epoch * self.a)) / 10, 1)
