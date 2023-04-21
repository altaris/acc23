"""
Custom pytorch dataset class to read from the competition's data files.
"""
__docformat__ = "google"

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from acc23.constants import IMAGE_RESIZE_TO

from .preprocessing import TARGETS, load_csv, load_image

Transform_t = Callable[[torch.Tensor], torch.Tensor]


class ACCDataset(Dataset):
    """
    Simple random-access dataset that loads (unlabeled) images from a given
    directory using
    [`torchvision.io.read_image`](https://pytorch.org/vision/master/generated/torchvision.io.read_image.html).
    The images have shape `(C, H, W)`, values from `0` to `1`, and dtype
    `float32`.
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
    ):
        """
        The transform defaults to a `IMAGE_RESIZE_TO x IMAGE_RESIZE_TO` resize,
        see `constants.IMAGE_RESIZE_TO`.
        """
        self.csv_file_path = Path(csv_file_path)
        self.image_dir_path = Path(image_dir_path)
        self.data = load_csv(csv_file_path)
        self.image_transform = image_transform or transforms.Resize(
            (IMAGE_RESIZE_TO, IMAGE_RESIZE_TO), antialias=True
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[dict, dict, torch.Tensor]:
        row = self.data.loc[idx]
        p, xy = row["Chip_Image_Name"], row.drop(["Chip_Image_Name"])
        if all(map(lambda c: c in xy, TARGETS)):
            # row has all target columns, i.e. this is probably the training
            # dataset
            x, y = dict(xy.drop(TARGETS)), dict(xy[TARGETS])
        else:
            x, y = dict(xy), {}
        try:
            img = load_image(self.image_dir_path / p)
        except:
            img = torch.zeros((3, 8, 8))
        img = self.image_transform(img)
        return x, y, img

    def test_train_split_dl(
        self,
        ratio: float = 0.8,
        split_kwargs: Optional[dict] = None,
        dataloader_kwargs: Optional[dict] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Performs a train/test split using
        [`torch.utils.data.random_split`](https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split)
        with the train dataset being roughly `ratio %` of the size of the
        dataset. Returns two `DataLoader`s.
        """
        a = int(len(self) * ratio)
        split_kwargs = split_kwargs or {}
        dataloader_kwargs = dataloader_kwargs or {
            "batch_size": 32,
            "pin_memory": True,
            "num_workers": 8,
        }
        test, train = torch.utils.data.random_split(
            self, lengths=[a, len(self) - a], **split_kwargs
        )
        return DataLoader(test, **dataloader_kwargs), DataLoader(
            train, **dataloader_kwargs
        )
