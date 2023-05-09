# pylint: disable=missing-function-docstring
"""Script to train acc23's autoencoder"""

from glob import glob
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch
from loguru import logger as logging
from torch.utils.data import DataLoader, Dataset

from acc23.autoencoder import Autoencoder
from acc23.utils import train_model
from acc23.preprocessing import load_image

Transform_t = Callable[[torch.Tensor], torch.Tensor]


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
        dataset. Returns two `DataLoader`s. The default dataloader parameters
        are

            {
                "batch_size": 32,
                "pin_memory": True,
                "num_workers": 8,
            }
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


def main():
    ds = ImageFolderDataset("data/images")
    train, val = ds.test_train_split_dl()
    model = Autoencoder(
        out_channels=[
            8,  # IMAGE_RESIZE_TO = 512 -> 256
            8,  # -> 128
            16,  # -> 64
            16,  # -> 32
            32,  # -> 16
            32,  # -> 8
            64,  # -> 4
            64,  # -> 2
            128,  # -> 1
        ],
        n_blocks=1,
        latent_space_dim=128,
    )
    name = model.__class__.__name__.lower()
    train_model(model, train, val, root_dir="out", name=name)


if __name__ == "__main__":
    try:
        main()
    except:
        logging.exception("Oh no :(")
