"""All the constants"""
__docformat__ = "google"


IMAGE_RESIZE_TO = 512
"""
By default, images will be resized to `IMAGE_RESIZE_TO x IMAGE_RESIZE_TO`. See
also `ACCDataset.__getitem__` and
[`torchvision.transforms.Resize`](https://pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html).
"""

N_CHANNELS = 3
"""
Number of image channels after preprocessing, see `preprocessing.load_image`.
Even if this is 1, images will still be presented as `(C, H, W)` arrays.
"""

N_FEATURES = 474
"""
Number of feature columns AFTER preprocessing, see
`preprocessing.preprocess_dataframe`
"""

N_TARGETS = 29
"""Number of target columns"""
