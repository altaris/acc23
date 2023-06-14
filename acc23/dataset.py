"""
Custom pytorch dataset class to read from the competition's data files.
"""
__docformat__ = "google"

from genericpath import isfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from acc23.constants import IMAGE_SIZE, N_CHANNELS, TARGETS, TRUE_TARGETS
from acc23.preprocessing import load_csv, load_image

ImageTransform_t = Callable[[torch.Tensor], torch.Tensor]


DEFAULT_DATALOADER_KWARGS = {
    "batch_size": 128,
    "pin_memory": True,
    "num_workers": 16,
}


class ACCDataset(Dataset):
    """
    Random-access dataset that reads from a CSV file and a image directory,
    both assumed to conform to the ACC23 specs.
    """

    image_dir_path: Path
    data: pd.DataFrame

    def __init__(self, data: pd.DataFrame, image_dir_path: Union[str, Path]):
        """
        Args:
            data (pd.DataFrame):
            image_dir_path (Union[str, Path]):
        """
        self.data = data
        self.image_dir_path = Path(image_dir_path)

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
        # img = self.image_transform(img)  # TODO: Image transform
        return x, y, img


class ACCDataModule(pl.LightningDataModule):
    """Lightning datamodule to wrap `acc23.dataset.ACCDataset`"""

    train_csv_file_path: Path
    test_csv_file_path: Path
    image_dir_path: Path
    dataloader_kwargs: dict

    ds_train: ACCDataset
    ds_val: ACCDataset
    ds_pred: ACCDataset
    ds_test: ACCDataset

    def __init__(
        self,
        train_csv_file_path: Union[str, Path] = "data/train.csv",
        test_csv_file_path: Union[str, Path] = "data/test.csv",
        image_dir_path: Union[str, Path] = "data/images",
        dataloader_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.train_csv_file_path = Path(train_csv_file_path)
        self.test_csv_file_path = Path(test_csv_file_path)
        self.image_dir_path = Path(image_dir_path)
        self.dataloader_kwargs = dataloader_kwargs or DEFAULT_DATALOADER_KWARGS

    def prepare_data(self) -> None:
        path = Path("data/cache")  # TODO: Dehardcode
        path.mkdir(exist_ok=True)
        if (
            isfile(path / "train.csv")
            and isfile(path / "val.csv")
            and isfile(path / "test.csv")
            and isfile(path / "pred.csv")
        ):
            return
        df_tv = load_csv(  # TODO: Dehardcode
            path=self.train_csv_file_path,
            preprocess=True,
            drop_nan_targets=True,
            impute=True,
            impute_targets=False,
            oversample=True,
        )
        df_p = load_csv(  # TODO: Dehardcode
            path=self.test_csv_file_path,
            preprocess=True,
            drop_nan_targets=False,
            impute=True,
            impute_targets=False,
            oversample=False,
        )
        n, split_ratio = len(df_tv), 0.9  # TODO: Dehardcode
        idxs, m = torch.randperm(n), int(split_ratio * n)
        df_t, df_v = df_tv.iloc[idxs[:m]], df_tv.iloc[idxs[m:]]
        df_t.to_csv(path / "train.csv", index=False)
        df_v.to_csv(path / "val.csv", index=False)
        df_tv.to_csv(path / "test.csv", index=False)
        df_p.to_csv(path / "pred.csv", index=False)

    def setup(self, stage: str) -> None:
        path = Path("data/cache")  # TODO: Dehardcode
        if stage == "fit":
            df_train = pd.read_csv(path / "train.csv")
            df_val = pd.read_csv(path / "val.csv")
            self.ds_train = ACCDataset(df_train, self.image_dir_path)
            self.ds_val = ACCDataset(df_val, self.image_dir_path)
        elif stage == "test":
            df_test = pd.read_csv(path / "test.csv")
            self.ds_test = ACCDataset(df_test, self.image_dir_path)
        elif stage == "predict":
            df_pred = pd.read_csv(path / "pred.csv")
            self.ds_pred = ACCDataset(df_pred, self.image_dir_path)
        else:
            raise ValueError(f"Unsupported stage: '{stage}'")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.ds_train, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.ds_val, **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.ds_test, **self.dataloader_kwargs)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.ds_pred, **self.dataloader_kwargs)


# class ImageFolderDataset(Dataset):
#     """
#     Simple random-access dataset that loads (unlabeled) images from a given
#     directory. See `acc23.utils.load_image`. The images have shape `(C, W, H)`,
#     values from `0` to `1`, and dtype `float32`.

#     Image loading uses `acc23.utils.load_image`, so this dataset is not
#     completely disconnected from the rest of the `acc23` package. In
#     particular, it complies with the constants defined in `acc23.constants`.
#     """

#     image_transform: ImageTransform_t
#     image_file_paths: List[Path]

#     def __init__(
#         self,
#         image_dir_path: Union[str, Path],
#         image_transform: Optional[ImageTransform_t] = None,
#     ):
#         """
#         Args:
#             image_dir_path (Union[str, Path]): e.g. `"data/images"`. The
#                 directory should only contain images.
#             image_transform (Optional[ImageTransform_t]): [torchvision
#                 transforms](https://pytorch.org/vision/stable/transforms.html)
#                 to apply to the images. Note that images are already resized to
#                 `constants.IMAGE_RESIZE_TO` and rescales to $[0, 1]$ before
#                 `image_transform` can touch them.

#         TODO: make it so that only image files are globbed (so that the image
#         directory may also contain non-image files).
#         """
#         self.image_transform = image_transform or (lambda x: x)
#         self.image_file_paths = list(
#             map(Path, glob(str(Path(image_dir_path) / "*")))
#         )

#     def __len__(self) -> int:
#         return len(self.image_file_paths)

#     def __getitem__(self, idx: int) -> torch.Tensor:
#         try:
#             img = load_image(self.image_file_paths[idx])
#         except OSError as err:
#             logging.error(
#                 "OSError for file {} at index {}: {}",
#                 self.image_file_paths[idx],
#                 idx,
#                 err,
#             )
#             raise
#         img = self.image_transform(img)
#         return img

#     def sample(self, n: int = 8) -> Tensor:
#         """Samples `n` images and returns them in a batch"""
#         return torch.stack([self[i] for i in np.random.choice(len(self), n)])

#     def train_test_split_dl(
#         self,
#         ratio: float = 0.8,
#         split_kwargs: Optional[dict] = None,
#         dataloader_kwargs: Optional[dict] = None,
#     ) -> Tuple[DataLoader, DataLoader]:
#         """
#         Performs a random train/test split split using
#         [`torch.utils.data.random_split`](https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split)
#         with the train dataset being roughly `ratio` of the size of the
#         dataset. Returns two `DataLoader`s.

#         Args:
#             ratio (float): A number in $(0, 1]$. If $1$, then the current
#                 dataset is returned twice (into 2 different dataloaders).
#             dataloader_kwargs (Optional[dict]): Defaults to
#                 `acc23.dataset.DEFAULT_DATALOADER_KWARGS`
#         """
#         split_kwargs = split_kwargs or {}
#         kw = dataloader_kwargs or DEFAULT_DATALOADER_KWARGS
#         if not 0.0 < ratio <= 1.0:
#             raise ValueError("Train/test split ratio must be > 0 and <= 1")
#         if ratio == 1.0:
#             return DataLoader(self, **kw), DataLoader(self, **kw)
#         test, train = random_split(
#             self, lengths=[ratio, 1.0 - ratio], **split_kwargs
#         )
#         return DataLoader(test, **kw), DataLoader(train, **kw)
