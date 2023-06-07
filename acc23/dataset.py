"""
Custom pytorch dataset class to read from the competition's data files.
"""
__docformat__ = "google"

from glob import glob
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger as logging
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from acc23.constants import IMAGE_SIZE, N_CHANNELS, TARGETS, TRUE_TARGETS
from acc23.mlsmote import mlsmote
from acc23.preprocessing import load_csv, load_image

ImageTransform_t = Callable[[torch.Tensor], torch.Tensor]


class ACCDataset(Dataset):
    """
    Random-access dataset that reads from a CSV file and a image directory,
    both assumed to conform to the ACC23 specs.
    """

    image_dir_path: Path
    data: pd.DataFrame
    image_transform: ImageTransform_t

    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame],
        image_dir_path: Union[str, Path],
        image_transform: Optional[ImageTransform_t] = None,
        load_csv_kwargs: Optional[dict] = None,
    ):
        """
        Args:
            data (Union[str, Path, pd.DataFrame]): the path to a csv file (e.g.
                `data/train.csv`) or a dataframe
            image_dir_path (Union[str, Path]): e.g. `data/images`
            image_transform (Optional[ImageTransform_t]): [torchvision
                transforms](https://pytorch.org/vision/stable/transforms.html)
                to apply to the images. Note that images are already resized to
                `constants.IMAGE_RESIZE_TO` and rescales to $[0, 1]$ before
                `image_transform`
            load_csv_kwargs (Optional[dict]): kwargs to pass to
                `acc23.preprocessing.load_csv` if `data` is a path (rather than
                a dataframe).
        """
        load_csv_kwargs = load_csv_kwargs or {}
        self.data = (
            data
            if isinstance(data, pd.DataFrame)
            else load_csv(data, **load_csv_kwargs)
        )
        self.image_dir_path = Path(image_dir_path)
        self.image_transform = image_transform or (lambda x: x)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[dict, dict, torch.Tensor]:
        row = self.data.iloc[idx]
        p, xy = row["Chip_Image_Name"], row.drop(["Chip_Image_Name"])
        if all(map(lambda c: c in xy, TARGETS)):
            # row has all target columns, so this is probably the training ds
            x, y = dict(xy.drop(TARGETS)), dict(xy[TRUE_TARGETS])
        else:
            x, y = dict(xy.drop(TARGETS, errors="ignore")), {}
        if isinstance(p, str) and (self.image_dir_path / p).is_file():
            img = load_image(self.image_dir_path / p)
        else:
            img = torch.zeros((N_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
        img = self.image_transform(img)
        return x, y, img

    def oversample(self, **kwargs) -> "ACCDataset":
        """
        MLSMOTE oversample

        See also:
            `acc23.mlsmote.mlsmote`

        Returns:
            `self`
        """
        self.data = mlsmote(self.data, TRUE_TARGETS, **kwargs)
        return self

    def train_test_split_dl(
        self,
        ratio: float = 0.8,
        dataloader_kwargs: Optional[dict] = None,
        oversample: bool = False,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Performs a random train/test split with the train dataset being roughly
        `ratio` of the size of the dataset. Returns two `DataLoader`s.

        Args:
            ratio (float): A number in $(0, 1]$. If $1$, then the current
                dataset is returned twice (into 2 different dataloaders).
            dataloader_kwargs (Optional[dict]): Defaults to
                {
                    "batch_size": 128,
                    "pin_memory": True,
                    "num_workers": 16,
                }
            oversample (bool): Whether to oversample the training set using
                MLSMOTE
        """
        kw = dataloader_kwargs or {
            "batch_size": 128,
            "pin_memory": True,
            "num_workers": 16,
        }
        if not 0.0 < ratio <= 1.0:
            raise ValueError("Train/test split ratio must be > 0 and <= 1")
        if ratio == 1.0:
            return DataLoader(self, **kw), DataLoader(self, **kw)
        idx, n = torch.randperm(len(self)), int(ratio * len(self))
        train = ACCDataset(
            data=self.data.iloc[idx[:n]].copy(),
            image_dir_path=self.image_dir_path,
            image_transform=self.image_transform,
        )
        test = ACCDataset(
            data=self.data.iloc[idx[n:]].copy(),
            image_dir_path=self.image_dir_path,
            image_transform=self.image_transform,
        )
        if oversample:
            train.oversample()
        return DataLoader(train, **kw), DataLoader(test, **kw)


class ImageFolderDataset(Dataset):
    """
    Simple random-access dataset that loads (unlabeled) images from a given
    directory. See `acc23.utils.load_image`. The images have shape `(C, W, H)`,
    values from `0` to `1`, and dtype `float32`.

    Image loading uses `acc23.utils.load_image`, so this dataset is not
    completely disconnected from the rest of the `acc23` package. In
    particular, it complies with the constants defined in `acc23.constants`.
    """

    image_transform: ImageTransform_t
    image_file_paths: List[Path]

    def __init__(
        self,
        image_dir_path: Union[str, Path],
        image_transform: Optional[ImageTransform_t] = None,
    ):
        """
        Args:
            image_dir_path (Union[str, Path]): e.g. `"data/images"`. The
                directory should only contain images.
            image_transform (Optional[ImageTransform_t]): [torchvision
                transforms](https://pytorch.org/vision/stable/transforms.html)
                to apply to the images. Note that images are already resized to
                `constants.IMAGE_RESIZE_TO` and rescales to $[0, 1]$ before
                `image_transform` can touch them.

        TODO: make it so that only image files are globbed (so that the image
        directory may also contain non-image files).
        """
        self.image_transform = image_transform or (lambda x: x)
        self.image_file_paths = list(
            map(Path, glob(str(Path(image_dir_path) / "*")))
        )

    def __len__(self) -> int:
        return len(self.image_file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        try:
            img = load_image(self.image_file_paths[idx])
        except OSError as err:
            logging.error(
                "OSError for file {} at index {}: {}",
                self.image_file_paths[idx],
                idx,
                err,
            )
            raise
        img = self.image_transform(img)
        return img

    def sample(self, n: int = 8) -> Tensor:
        """Samples `n` images and returns them in a batch"""
        return torch.stack([self[i] for i in np.random.choice(len(self), n)])

    def train_test_split_dl(
        self,
        ratio: float = 0.8,
        split_kwargs: Optional[dict] = None,
        dataloader_kwargs: Optional[dict] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Performs a random train/test split split using
        [`torch.utils.data.random_split`](https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split)
        with the train dataset being roughly `ratio` of the size of the
        dataset. Returns two `DataLoader`s.

        Args:
            ratio (float): A number in $(0, 1]$. If $1$, then the current
                dataset is returned twice (into 2 different dataloaders).
            dataloader_kwargs (Optional[dict]): Defaults to
                {
                    "batch_size": 128,
                    "pin_memory": True,
                    "num_workers": 16,
                }
        """
        split_kwargs = split_kwargs or {}
        kw = dataloader_kwargs or {
            "batch_size": 128,
            "pin_memory": True,
            "num_workers": 16,
        }
        if not 0.0 < ratio <= 1.0:
            raise ValueError("Train/test split ratio must be > 0 and <= 1")
        if ratio == 1.0:
            return DataLoader(self, **kw), DataLoader(self, **kw)
        test, train = torch.utils.data.random_split(
            self, lengths=[ratio, 1.0 - ratio], **split_kwargs
        )
        return DataLoader(test, **kw), DataLoader(train, **kw)
