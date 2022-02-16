from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torchvision
from pathlib import Path
from PIL import Image
import numpy as np


def is_image(path: str) -> bool:
    ext = Path(path).suffix[1:].lower()
    return ext in [
        'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'
    ]


CATEGORIES = [
    'bottle',
    'cable',
    'capsule',
    'carpet',
    'grid',
    'hazelnut',
    'leather',
    'metal_nut',
    'pill',
    'screw',
    'tile',
    'toothbrush',
    'transistor',
    'wood',
    'zipper'
]


transform = torchvision.transforms.Compose([
    # torchvision.transforms.Resize(256),
    # torchvision.transforms.CenterCrop(224),
    torchvision.transforms.Resize(74),
    torchvision.transforms.CenterCrop(64),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda t: (t * 2) - 1)
])

transform_gt = torchvision.transforms.Compose([
    # torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.NEAREST),
    # torchvision.transforms.CenterCrop(224),
    torchvision.transforms.Resize(74, interpolation=torchvision.transforms.InterpolationMode.NEAREST),
    torchvision.transforms.CenterCrop(64),
    torchvision.transforms.ToTensor(),
])


class TrainDataset(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index: Any) -> torch.Tensor:
        x = transform(self.x[index])

        return x


class TestDataset(Dataset):
    def __init__(self, x: np.ndarray,
                 y: np.ndarray,
                 gt: np.ndarray,
                 id_: List[str]):
        assert len(x) == len(y) == len(gt) == len(id_)
        self.x = x
        self.y = y
        self.gt = gt
        self.id_ = id_

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index: Any) -> torch.Tensor:
        x = transform(self.x[index])
        y = self.y[index]
        gt = transform_gt(self.gt[index])
        id_ = self.id_[index]
        return x, y, gt, id_


class MVTecDataModule(pl.LightningDataModule):
    def __init__(self, root_dir: str,
                 category: str,
                 batch_size: int = 32):
        super().__init__()
        assert category in CATEGORIES, \
            f'category must be in {CATEGORIES}'
        self.root_dir = root_dir
        self.category = category
        self.batch_size = batch_size

        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str]=None):
        if stage == 'fit' or stage is None:
            image_dir = Path(self.root_dir) / self.category / 'train' / 'good'
            x = [
                Image.open(p).convert('RGB') for p in image_dir.iterdir()
                if is_image(p)
            ]
            self.train_dataset = TrainDataset(x)

        if stage == 'test' or stage is None:
            image_dir = Path(self.root_dir) / self.category / 'test'
            gt_dir = Path(self.root_dir) / self.category / 'ground_truth'

            ng_categories = sorted(
                map(lambda p: p.name,
                    list(gt_dir.iterdir())
                )
            )
            images = []
            gts = []
            labels = []
            ids = []

            for cat in ['good'] + ng_categories:
                _images = [
                    Image.open(p).convert('RGB') for p in (image_dir/cat).iterdir()
                    if is_image(p)
                ]
                _ids = [f'{p.parent.name}/{p.name}' for p in (image_dir/cat).iterdir()]

                if cat == 'good':
                    _gts = np.zeros(np.array(_images).shape[:3], dtype=np.uint8)
                    _labels = [0] * len(_images)
                else:
                    _gts = [
                        Image.open(gt_dir/f'{cat}/{Path(_id).stem}_mask.png')
                        for _id in _ids if is_image(_id)
                    ]
                    assert len(_images) == len(_gts)
                    _labels = [1] * len(_images)
                images.extend(_images)
                gts.extend(_gts)
                labels.extend(_labels)
                ids.extend(_ids)
            self.test_dataset = TestDataset(
                images,
                np.array(labels),
                gts,
                ids,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_test, batch_size=self.batch_size)
