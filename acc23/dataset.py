"""
Custom pytorch-lightning datamodule class to read from the competition's data
files.
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger as logging

from torch.utils.data import DataLoader, Dataset

from acc23.constants import IMAGE_SIZE, N_CHANNELS, TARGETS, TRUE_TARGETS
from acc23.preprocessing import load_csv, load_image

ImageTransform_t = Callable[[torch.Tensor], torch.Tensor]
"""
Convenience alias representing the type of an image transform. This is just for
type annotation.
"""


DEFAULT_DATALOADER_KWARGS = {
    "batch_size": 128,
    "pin_memory": True,
    "num_workers": 16,
}
"""
Default parameters for [pytorch
dataloaders](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader).
"""


class ACCDataset(Dataset):
    """
    Random-access [pytorch
    dataset](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset)
    that reads from a CSV file and a image directory, both assumed to conform
    to the competition specs.
    """

    image_dir_path: Path
    data: pd.DataFrame

    def __init__(self, data: pd.DataFrame, image_dir_path: Union[str, Path]):
        """
        Args:
            data (pd.DataFrame): **Preprocessed** (see
                `acc23.preprocessing.load_csv`) tabular data in the form of a
                pandas dataframe
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
    """
    Lightning datamodule to wrap `acc23.dataset.ACCDataset`. This module
    handles tabular data preprocessing. For efficiency, preprocess data are
    cached in a user-specified directory.
    """

    data_cache_path: Path
    dataloader_kwargs: dict
    image_dir_path: Path
    test_csv_file_path: Path
    train_csv_file_path: Path

    ds_train: Optional[ACCDataset] = None
    ds_val: Optional[ACCDataset] = None
    ds_pred: Optional[ACCDataset] = None
    ds_test: Optional[ACCDataset] = None

    _split_ratio: float

    def __init__(
        self,
        train_csv_file_path: Union[str, Path] = "data/train.csv",
        test_csv_file_path: Union[str, Path] = "data/test.csv",
        image_dir_path: Union[str, Path] = "data/images",
        dataloader_kwargs: Optional[dict] = None,
        data_cache_path: Union[str, Path] = "out/data.cache",
        split_ratio: float = 0.8,
    ) -> None:
        """
        Args:
            train_csv_file_path (Union[str, Path]): The path of `train.csv`
            test_csv_file_path (Union[str, Path]): The path of `test.csv`
            image_dir_path (Union[str, Path]): Path of the image folder
            dataloader_kwargs (dict, optional): Default to
                `acc23.dataset.DEFAULT_DATALOADER_KWARGS`.
            data_cache_path (Union[str, Path]): Path for data cache folder
            split_ratio (float): Split ratio for the train and validation
                dataset. A ratio of .8 (the default) means that 80% of samples
                are in the train dataset, whereas the other 20% are used in the
                validation dataset.
        """
        super().__init__()
        self.data_cache_path = Path(data_cache_path)
        self.dataloader_kwargs = dataloader_kwargs or DEFAULT_DATALOADER_KWARGS
        self.image_dir_path = Path(image_dir_path)
        self.test_csv_file_path = Path(test_csv_file_path)
        self.train_csv_file_path = Path(train_csv_file_path)
        self._split_ratio = split_ratio

    def prepare_data(self) -> None:
        """
        Overrides
        [pl.LightningDataModule.prepare_data](https://lightning.ai/docs/pytorch/stable/data/datamodule.html#prepare-data).
        This is automatically called so don't worry about it.

        TODO:
            Fix quick&dirty imputation
        """
        if self.data_cache_path.is_dir():
            logging.debug(
                "Data cache path '{}' exists, skipping "
                "`ACCDataLModule.prepare_data`",
                self.data_cache_path,
            )
            return
        self.data_cache_path.mkdir(parents=True, exist_ok=True)

        # Quick and dirty
        import turbo_broccoli as tb
        from sklearn.impute import IterativeImputer, KNNImputer
        from sklearn.experimental import (
            enable_iterative_imputer,
        )  # pylint: disable=unused-import

        a = load_csv(
            self.train_csv_file_path,
            impute=False,
            drop_nan_targets=False,
        )
        b = load_csv(
            self.test_csv_file_path,
            impute=False,
            drop_nan_targets=False,
        )
        c = pd.concat([a, b], axis=0)
        c = c.drop(columns=TARGETS + ["Chip_Image_Name"])
        imp = KNNImputer(n_neighbors=10).fit(c.to_numpy())
        # imp = IterativeImputer().fit(c.to_numpy())
        tb.set_artifact_path(self.data_cache_path)
        tb.save_json(imp, self.data_cache_path / "imputer.json")
        logging.warning(
            "Done quick & dirty imputer fit. Refactor that soon ok? <3"
        )

        dfs: Dict[str, pd.DataFrame] = {}
        dfs["test"] = load_csv(  # TODO: Dehardcode
            path=self.train_csv_file_path,
            impute=True,
            imputer_path=self.data_cache_path / "imputer.json",
            drop_nan_targets=False,
            oversample=False,
        )
        df_tv = load_csv(  # TODO: Dehardcode
            path=self.train_csv_file_path,
            impute=True,
            imputer_path=self.data_cache_path / "imputer.json",
            drop_nan_targets=False,
            oversample=False,
        )
        dfs["pred"] = load_csv(  # TODO: Dehardcode
            path=self.test_csv_file_path,
            impute=True,
            imputer_path=self.data_cache_path / "imputer.json",
            drop_nan_targets=False,  # There are no targets in pred ds =)
            oversample=False,
        )
        idxs = torch.randperm(len(df_tv))
        m = int(self._split_ratio * len(df_tv))
        dfs["train"], dfs["val"] = df_tv.iloc[idxs[:m]], df_tv.iloc[idxs[m:]]
        for k, df in dfs.items():
            df.to_csv(self.data_cache_path / f"{k}.csv", index=False)

    def setup(self, stage: str) -> None:
        """
        Overrides
        [pl.LightningDataModule.setup](https://lightning.ai/docs/pytorch/stable/data/datamodule.html#setup).
        This is automatically called so don't worry about it.
        """
        if stage == "fit":
            df_train = pd.read_csv(self.data_cache_path / "train.csv")
            df_val = pd.read_csv(self.data_cache_path / "val.csv")
            self.ds_train = ACCDataset(df_train, self.image_dir_path)
            self.ds_val = ACCDataset(df_val, self.image_dir_path)
        elif stage == "test":
            df_test = pd.read_csv(self.data_cache_path / "test.csv")
            self.ds_test = ACCDataset(df_test, self.image_dir_path)
        elif stage == "predict":
            df_pred = pd.read_csv(self.data_cache_path / "pred.csv")
            self.ds_pred = ACCDataset(df_pred, self.image_dir_path)
        else:
            raise ValueError(f"Unsupported stage: '{stage}'")

    def train_dataloader(self) -> DataLoader:
        """
        Returns a dataloader for the train dataset, which is a part of
        `train.csv`. Overrides
        [pl.LightningDataModule.train_dataloader](https://lightning.ai/docs/pytorch/stable/data/datamodule.html#train_dataloader).
        """
        if self.ds_train is None:
            raise RuntimeError(
                "Train dataset not loaded. Call "
                "`ACCDataModule.setup('fit')` before using this datamodule."
            )
        return DataLoader(dataset=self.ds_train, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        """
        Returns a dataloader for the validation dataset, which is a part of
        `train.csv`. Overrides
        [pl.LightningDataModule.val_dataloader](https://lightning.ai/docs/pytorch/stable/data/datamodule.html#val_dataloader).
        """
        if self.ds_val is None:
            raise RuntimeError(
                "Validation dataset not loaded. Call "
                "`ACCDataModule.setup('fit')` before using this datamodule."
            )
        return DataLoader(dataset=self.ds_val, **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        """
        Returns a dataloader for the test dataset. This is the full
        preprocessed `train.csv`, **not** the content of `test.csv`. Overrides
        [pl.LightningDataModule.test_dataloader](https://lightning.ai/docs/pytorch/stable/data/datamodule.html#test_dataloader).
        """
        if self.ds_test is None:
            raise RuntimeError(
                "Test dataset not loaded. Call "
                "`ACCDataModule.setup('test')` before using this datamodule."
            )
        return DataLoader(dataset=self.ds_test, **self.dataloader_kwargs)

    def predict_dataloader(self) -> DataLoader:
        """
        Returns a dataloader for the prediction dataset, which corresponds to
        `test.csv`. Overrides
        [pl.LightningDataModule.predict_dataloader](https://lightning.ai/docs/pytorch/stable/data/datamodule.html#predict_dataloader).
        """
        if self.ds_pred is None:
            raise RuntimeError(
                "Prediction dataset not loaded. Call "
                "`ACCDataModule.setup('predict')` before using this "
                "datamodule."
            )
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
#             image_transform (ImageTransform_t, optional): [torchvision
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
#             dataloader_kwargs (dict, optional): Defaults to
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
