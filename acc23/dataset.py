"""
Custom pytorch dataset class to read from the competition's data files.
"""
__docformat__ = "google"

from glob import glob
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
import torch
from loguru import logger as logging
from torch.utils.data import DataLoader, Dataset

from acc23.constants import (
    IMAGE_SIZE,
    N_CHANNELS,
    TARGETS,
    TRUE_TARGETS,
)
from acc23.preprocessing import load_csv, load_image

Transform_t = Callable[[torch.Tensor], torch.Tensor]


class ACCDataset(Dataset):
    """
    Random-access dataset that reads from a CSV file and a image directory,
    both assumed to conform to the ACC23 specs.
    """

    csv_file_path: Path
    image_dir_path: Path
    data: pd.DataFrame
    image_transform: Transform_t

    def __init__(
        self,
        csv_file_path: Union[str, Path],
        image_dir_path: Union[str, Path],
        image_transform: Optional[Transform_t] = None,
        load_csv_kwargs: Optional[dict] = None,
    ):
        """
        Args:
            csv_file_path (Union[str, Path]): e.g. `"data/train.csv"`
            image_dir_path (Union[str, Path]): e.g. `"data/images"`
            image_transform (Optional[Transform_t]): [torchvision
                transforms](https://pytorch.org/vision/stable/transforms.html)
                to apply to the images. Note that images are already resized to
                `constants.IMAGE_RESIZE_TO` and rescales to $[0, 1]$ before
                `image_transform` can touch them.
            load_csv_kwargs (Optional[dict]): kwargs to pass to
                `acc23.preprocessing.load_csv`.
        """
        load_csv_kwargs = load_csv_kwargs or {}
        self.csv_file_path = Path(csv_file_path)
        self.image_dir_path = Path(image_dir_path)
        self.data = load_csv(csv_file_path, **load_csv_kwargs)
        self.image_transform = image_transform or (lambda x: x)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[dict, dict, torch.Tensor]:
        row = self.data.loc[idx]
        p, xy = row["Chip_Image_Name"], row.drop(["Chip_Image_Name"])
        if all(map(lambda c: c in xy, TARGETS)):
            # row has all target columns, so this is probably the training ds
            x, y = dict(xy.drop(TARGETS)), dict(xy[TRUE_TARGETS])
        else:
            x, y = dict(xy.drop(TARGETS, errors="ignore")), {}
        try:
            img = load_image(self.image_dir_path / p)
        except:
            img = torch.zeros((N_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
        img = self.image_transform(img)
        return x, y, img

    def train_test_split_dl(
        self,
        ratio: float = 0.8,
        split_kwargs: Optional[dict] = None,
        dataloader_kwargs: Optional[dict] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Performs a train/test split using
        [`torch.utils.data.random_split`](https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split)
        with the train dataset being roughly `ratio %` of the size of the
        dataset. Returns two `DataLoader`s. The default dataloader parameters
        are

            {
                "batch_size": 128,
                "pin_memory": True,
                "num_workers": 16,
            }

        If `ratio = 1`, then the while dataset is returned twice (into 2
        different dataloaders).
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


class ImageFolderDataset(Dataset):
    """
    Simple random-access dataset that loads (unlabeled) images from a given
    directory. See `acc23.utils.load_image`. The images have shape `(C, W, H)`,
    values from `0` to `1`, and dtype `float32`.

    Image loading uses `acc23.utils.load_image`, so this dataset is not
    completely disconnected from the rest of the `acc23` package. In
    particular, it complies with the constants defined in `acc23.constants`.
    """

    image_transform: Transform_t
    image_file_paths: List[Path]

    def __init__(
        self,
        image_dir_path: Union[str, Path],
        image_transform: Optional[Transform_t] = None,
    ):
        """
        Args:
            image_dir_path (Union[str, Path]): e.g. `"data/images"`. The
                directory should only contain images.
            image_transform (Optional[Transform_t]): [torchvision
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

    def train_test_split_dl(
        self,
        ratio: float = 0.8,
        split_kwargs: Optional[dict] = None,
        dataloader_kwargs: Optional[dict] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Performs a train/test split using
        [`torch.utils.data.random_split`](https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split)
        with the train dataset being roughly `ratio %` of the size of the
        dataset. Returns two `DataLoader`s. The default dataloader parameters
        are

            {
                "batch_size": 128,
                "pin_memory": True,
                "num_workers": 16,
            }

        If `ratio = 1`, then the while dataset is returned twice (into 2
        different dataloaders).
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
