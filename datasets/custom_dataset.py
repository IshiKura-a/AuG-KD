import logging
import os
from argparse import Namespace
from typing import List, Tuple, Any, Union, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision.datasets import ImageFolder, Caltech256

from datasets.config import get_transform, dataset_config
from utils.logger import logger


class CustomBatch:
    def __init__(self, data: Any):
        transposed_data = list(zip(*data))
        self.data = [torch.stack(d, 0) for d in transposed_data]

    def pin_memory(self) -> 'CustomBatch':
        self.data = [
            d.pin_memory() for d in self.data
        ]
        return self


def collate_wrapper(batch: Any) -> CustomBatch:
    return CustomBatch(batch)


class CompositeDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        assert len(list(set([len(d) for d in datasets]))) == 1, "Datasets should be the same length"
        self.datasets = datasets

    def __getitem__(self, item):
        x = []
        for d in self.datasets:
            i = d.__getitem__(item)
            if isinstance(i, tuple):
                x += list(i)
            else:
                x.append(i)
        return tuple(x)

    def __len__(self):
        return len(self.datasets[0])


def parse_split(split: str) -> List[float]:
    split = np.array([float(i.strip()) for i in split.split(':')])
    split = (split / split.sum()).tolist()
    return split


def ConcatAndSplitDatasets(datasets: List[List[Dataset]], ratio: Union[str, List[float]]) -> Tuple[List[Dataset], List]:
    if isinstance(ratio, str):
        ratio = parse_split(ratio)

    assert len(ratio) >= len(datasets[0]), "#split must be no smaller than #datasets"
    if len(ratio) != len(datasets[0]):
        logger.warning('#split!=#datasets, use the last dataset for padding')

    res_dataset = [[] for _ in ratio]
    split = []
    for dataset in datasets:
        n_d = len(dataset)
        n = len(dataset[0])
        idx_perm = torch.randperm(n)
        acc_ratio = 0
        split.append([])
        for idx, r in enumerate(ratio):
            idx_chosen = idx_perm[acc_ratio: acc_ratio + int(r * n)] if idx != len(ratio) - 1 else idx_perm[acc_ratio:]
            split[-1].append(idx_chosen)
            res_dataset[idx].append(Subset(dataset[idx if idx < n_d - 1 else -1], idx_chosen))
            acc_ratio = acc_ratio + int(r * n)
    return [ConcatDataset(d) for d in res_dataset], split


def LoadImageFolders(dirs: List[str], args: Namespace, normalize: bool = True) -> List[List[ImageFolder]]:
    config = dataset_config[args.dataset]
    datasets = []
    for d in dirs:
        train_d = ImageFolder(root=d, transform=get_transform(config.target_size, True,
                                                              config.normalize if normalize else None))
        eval_d = ImageFolder(root=d, transform=get_transform(config.target_size, False,
                                                             config.normalize if normalize else None))
        datasets.append([train_d, eval_d])
    return datasets


class MCaltech256(Caltech256):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.open(
            os.path.join(
                self.root,
                "256_ObjectCategories",
                self.categories[self.y[index]],
                f"{self.y[index] + 1:03d}_{self.index[index]:04d}.jpg",
            )
        ).convert('RGB')

        target = self.y[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CustomDataset(Dataset):
    def __init__(self, root_dir: str, transform: transforms.Compose):
        self.root_dir = root_dir
        self.transform = transform

        attr_df = pd.read_csv(os.path.join(root_dir, 'image_list.txt'),
                              header=None, delimiter=' ', names=['filename', 'y'])
        self.filename_arr = attr_df.loc[:, 'filename']
        self.y_arr = attr_df.loc[:, 'y']

    def __len__(self):
        return len(self.filename_arr)

    def __getitem__(self, item):
        y = self.y_arr[item]
        img_name = os.path.join(self.root_dir, self.filename_arr[item])
        img = Image.open(img_name).convert('RGB')
        return self.transform(img), y
