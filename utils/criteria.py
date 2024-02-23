import numpy as np
import torch

from torch import nn
from abc import abstractmethod, ABC
from typing import Mapping, Any, Dict, List, Union
from torch.nn import functional as F
from torch import Tensor

from utils.logger import logger


class Metric(ABC):
    @staticmethod
    def kl_div(predictions: Tensor, target: Tensor, t: float = 1.0, reduction: str = 'sum'):
        return F.kl_div(F.log_softmax(predictions / t, dim=-1), (target / t).softmax(dim=-1).clip(min=1e-8),
                        reduction=reduction) * (t * t)

    @staticmethod
    def js_div(predictions: Tensor, target: Tensor, t: float, reduction: str = 'sum'):
        soft_target = target / t
        mid = (predictions.softmax(dim=-1) + soft_target.softmax(dim=-1)) / 2
        return F.kl_div(F.log_softmax(predictions, dim=-1), mid, reduction=reduction) + \
            F.kl_div(F.log_softmax(soft_target, dim=-1), mid, reduction=reduction)

    def __init__(self):
        self._name = 'Metric'
        self._loss = 0
        self._cnt = 0

    @abstractmethod
    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        raise NotImplementedError()

    def get_results(self) -> Any:
        return self._loss / self._cnt if self._cnt != 0 else 0

    def reset(self):
        self._loss = 0
        self._cnt = 0

    def __str__(self):
        return f'{self.name} = {self.get_results():.3f}'

    @property
    def name(self):
        return self._name


class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self._name = 'Acc'
        self._correct = 0
        self._cnt = 0

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        predictions = predictions.softmax(dim=-1).argmax(dim=-1)
        self._cnt += target.shape[0]
        self._correct += (predictions.long() == target.long()).sum()
        return self

    def get_results(self) -> float:
        return (self._correct / self._cnt).item() if self._cnt != 0 else 0

    def reset(self):
        self._cnt = 0
        self._correct = 0


class BinaryAccuracy(Accuracy):
    def __init__(self):
        super().__init__()
        self._name = 'BAcc'
        self._correct = 0
        self._cnt = 0

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        predictions = (predictions >= 0.5)
        self._cnt += target.shape[0]
        self._correct += (predictions.long() == target.long()).sum()
        return self


class TopKAccuracy(Accuracy):
    def __init__(self, k: int):
        super().__init__()
        self._name = f'Acc@{k}'
        self._correct = 0
        self._cnt = 0
        self._k = k

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        predictions = predictions.softmax(dim=-1)
        predictions = torch.topk(predictions, k=self._k).indices
        self._cnt += target.shape[0]
        for i in range(self._k):
            self._correct += (predictions[:, i].long() == target.long()).sum()
        return self


class CrossEntropy(Metric):
    def __init__(self):
        super().__init__()
        self._name = 'CE'
        self._ce = nn.CrossEntropyLoss(reduction='sum')
        self._loss = 0
        self._cnt = 0

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        self._cnt += predictions.shape[0]
        self._loss += self._ce(predictions, target)
        return self


class KLDivLoss(Metric):
    def __init__(self, t: float = 1.0):
        super().__init__()
        self._name = 'KLDivLoss'
        self._fn = lambda x, y: self.kl_div(x, y, t)
        self._loss = 0
        self._cnt = 0

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        self._cnt += predictions.shape[0]
        self._loss += self._fn(predictions, target)
        return self


class JSDivLoss(KLDivLoss):
    def __init__(self, t: float = 1.0):
        super().__init__()
        self._name = 'JSDivLoss'
        self._fn = lambda x, y: self.js_div(x, y, t)

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        self._cnt += predictions.shape[0]
        self._loss += self._fn(predictions, target)
        return self


class DistillLoss(Metric):
    def __init__(self, t: float = 1.0):
        super().__init__()
        self._name = 'DistillLoss'
        self._ce = nn.CrossEntropyLoss(reduction='none')
        self._loss = 0
        self._t = t
        self._cnt = 0

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        self._cnt += predictions.shape[0]
        t_outputs = kwargs['t_outputs']
        if 'u' not in kwargs:
            u = 0.5
        else:
            u = kwargs['u']
        self._loss += (u * self._ce(predictions, target) +
                       (1 - u) * self.kl_div(predictions, t_outputs, self._t, 'none').sum(dim=-1)).sum()
        return self


class BCELoss(Metric):
    def __init__(self):
        super().__init__()
        self._name = 'BCE'
        self._bce = nn.BCELoss(reduction='sum')
        self._loss = 0
        self._cnt = 0

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        self._cnt += predictions.shape[0]
        self._loss += self._bce(predictions, target.float().unsqueeze(-1))
        return self


class MSELoss(Metric):
    def __init__(self):
        super().__init__()
        self._name = 'MSE'
        self._mse = nn.MSELoss(reduction='sum')
        self._loss = 0

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        self._loss += self._mse(predictions, target)
        self._cnt += predictions.shape[0]
        return self


class CVAELoss(MSELoss):
    def __init__(self):
        super().__init__()
        self._name = 'CVAELoss'

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        mu = kwargs['mu']
        log_var = kwargs['log_var']
        kl_div = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())
        self._loss += self._mse(predictions, target) + 0.00025 * kl_div
        self._cnt += predictions.shape[0]
        return self


class DCVAELoss(MSELoss):
    def __init__(self):
        super().__init__()
        self._name = 'DCVAELoss'
        self._bce = nn.BCELoss(reduction='sum')

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        mu = kwargs['mu']
        log_var = kwargs['log_var']
        d_outputs = kwargs['d_outputs']

        kl_div = 0.5 * torch.sum(-1 - log_var + torch.exp(log_var) + mu ** 2)
        # Try to fool discriminator, predict generated images to be real
        self._loss += self._mse(predictions, target) + \
                      0.00025 * kl_div + \
                      self._bce(d_outputs, torch.ones_like(d_outputs).float())

        self._cnt += predictions.shape[0]
        return self


class DecoderLoss(CrossEntropy):
    def __init__(self, t: float = 1.0):
        super().__init__()
        self._name = 'DecoderLoss'
        self._t = t

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        sp = predictions.softmax(dim=-1).mean(dim=0)
        self._loss += -1.0 * self.kl_div(target, predictions) + \
                      1.0 * self._ce(predictions, predictions.argmax(dim=1)) + \
                      20 * (sp * sp.log10()).sum() * predictions.shape[0] + \
                      1.0 * (kwargs['batch_mean'].norm(2) + kwargs['batch_var'].norm(2)) * predictions.shape[0]

        self._cnt += predictions.shape[0]
        return self


class CondDecoderLoss(CrossEntropy):
    def __init__(self, t: float = 1.0):
        super().__init__()
        self._name = 'CondDecoderLoss'
        self._t = t

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        sp = predictions.softmax(dim=-1)
        self._loss += -1.0 * self.kl_div(kwargs['s_outputs'], predictions) + \
                      1.0 * self._ce(predictions, target) + \
                      20 * (sp * sp.log10()).sum() + \
                      1.0 * (kwargs['batch_mean'].norm(2) + kwargs['batch_var'].norm(2))
        self._cnt += predictions.shape[0]
        return self


class Uncertainty(Metric):
    """
    - Class: Uncertainty
    - Description: Use Energy-OOD / 1 - Max-Softmax to model the uncertainty of outputs.
      # Since Energy-OOD is not scaled to [0,1], but is bounded theoretically in
      # [-t * log(num_classes) - ff, -ff] , where ff is minimum of the maximum of logits.
    - Ref-Arxiv: https://arxiv.org/pdf/2010.03759.pdf
    """

    MaxSoftmax = 'max_softmax'
    Energy = 'energy'

    fn_dict = {
        # Energy: lambda x, t, n, ff: (t * np.log(n) + ff[1] - t * torch.logsumexp(x / t, dim=1)) / (ff[1] - ff[0] + t * np.log(n)),
        Energy: lambda x, t: - t * torch.logsumexp(x / t, dim=1),
        MaxSoftmax: lambda x, t: 1 - x.softmax(dim=1).max(dim=1).values
    }
    reduction_dict = {
        'sum': lambda x: x.sum(),
        'none': lambda x: [x],
    }

    def __init__(self, mode: str = 'energy', t: float = 1.0, reduction: str = 'none', **kwargs):
        super().__init__()
        assert reduction in self.reduction_dict.keys()
        self._name = 'Uncertainty'
        self._loss = [] if reduction == 'none' else 0.
        self._t = t
        self.fn = lambda x: self.reduction_dict[reduction](self.fn_dict[mode](x, self._t))
        # (lambda x: self.fn_dict[mode](x, self._t, kwargs['num_classes'], kwargs['ff']))
        self._reduction = reduction

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        self._loss += self.fn(predictions)
        return self

    def get_results(self) -> Union[Tensor, float]:
        if self._reduction == 'none':
            return torch.cat(self._loss).to(self._loss[0].device)
        else:
            return self._loss

    def reset(self):
        self._loss = [] if self._reduction == 'none' else 0.


class AnchorLoss(Uncertainty):
    def __init__(self, mode: str = 'energy', t: float = 1.0, invariant: float = 0.25, **kwargs):
        super().__init__(mode, t, 'sum')
        self._inv = invariant
        self._name = 'AnchorLoss'
        self._ce = nn.CrossEntropyLoss(reduction='sum')
        self._loss = 0

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        mask = kwargs['mask']
        self._loss += self.fn(predictions) + (mask.sum() - mask.shape[-1] * (1 - self._inv)).abs() + \
                      0.1 * self._ce(predictions, target)
        self._cnt += predictions.shape[0]
        return self


class Compose(Metric):
    def __init__(self, metric: List[Metric] = []):
        super().__init__()
        self._list = metric
        self.reset()

    def __getitem__(self, index: int) -> Metric:
        return self._list[index]

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        for metric in self._list:
            metric.update(predictions, target, **kwargs)
        return self

    def get_results(self) -> Dict[str, float]:
        return {
            metric.name: metric.get_results() for metric in self._list
        }

    def reset(self):
        for metric in self._list:
            metric.reset()

    @staticmethod
    def compose(metrics: List[Metric]) -> 'Compose':
        _list = []
        for metric in metrics:
            if isinstance(metric, Compose):
                _list += metric._list
            else:
                _list.append(metric)
        return Compose(_list)

    def __str__(self):
        return ' '.join([metric.__str__() for metric in self._list])
