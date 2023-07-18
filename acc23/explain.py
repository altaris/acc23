"""
Everything related to explainability
"""

from typing import Any, Callable, List, Literal, Tuple, Union

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage
from rich.progress import track
from torch import Tensor, nn, utils
from torchvision.transforms.functional import resize
from transformers.models.vit.modeling_vit import ViTModel, ViTSelfAttention


def batch_forward(
    f: Callable[[Tensor], Tensor], x: Tensor, batch_size: int = 128
):
    """
    Batched call to a callable that takes and returns a tensor (e.g. a model).
    Displays a [`rich` progress
    bar](https://rich.readthedocs.io/en/stable/progress.html).
    """
    if len(x) <= batch_size:
        return f(x)
    y = [f(b) for b in track(x.split(batch_size))]
    # y = list(map(module, x.split(batch_size)))
    return torch.cat(y)


def imshow(img: Union[Tensor, np.ndarray], **kwargs) -> AxesImage:
    """
    Convenience function to plot an image. Behaves similarly to
    [`matplotlib.pyplot.imshow`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html).

    Args:
        img (Union[Tensor, np.ndarray]): If it is a torch tensor, it must have
            shape `(W, H)` or `(3, W, H)`. If it is an numpy array, it must
            have shape `(W, H)` or `(W, H, 3)`.
        kwargs: Forwarded to `matplotlib.pyplot.imshow`
    """
    if img.ndim not in [2, 3]:
        raise ValueError(
            "imshow does not support batched images. The input tensor/array "
            "must have 2 or 3 dimensions"
        )
    if isinstance(img, Tensor):
        if img.ndim == 3:
            img = img.permute(1, 2, 0)
        img = img.numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-5)
    return plt.imshow(img, **kwargs)


def shap(
    module: Callable[[Tensor], Tensor],
    data: Tensor,
    n_samples: int = 10,
    batch_size: int = 128,
) -> Tensor:
    """
    Model-agnostic Monte-Carlo SHAP values. See

        Lundberg, Scott M., and Su-In Lee. "A unified approach to interpreting
        model predictions." Advances in neural information processing systems
        30 (2017).

    Args:
        module (Callable[[Tensor], Tensor]): A model that takes a `(N, F)`
            tensor, where `F` is the number of features, and outputs an `(N,
            T)` tensor where `T` is the number of targets/labels
        data (Tensor): A `(N, F)` tensor
        n_samples (int):
        batch_size (int): The model will be evaluated `2 * n_samples * F * N`
            times, so this needs to be batched. See
            `acc23.explain.batch_forward`.

    Returns:
        A `(N, F, T)` tensor
    """

    def _forward_4(x: Tensor):
        a, b, c, d = x.shape
        u = x.reshape(a * b * c, d)
        y = batch_forward(module, u, batch_size=batch_size)
        return y.reshape(a, b, c, -1)

    n_rows, n_features = data.shape
    means = data.mean(dim=0).repeat([n_samples, n_features, n_rows, 1])
    data = data.repeat([n_samples, n_features, 1, 1])
    mask = torch.randint(0, 2, (n_samples, 1, n_rows, n_features))
    mask = mask.repeat([1, n_features, 1, 1])

    mf = torch.eye(n_features)
    mf = mf.repeat([1, n_rows])
    mf = mf.reshape(1, n_features, n_rows, n_features)
    mf = mf.repeat([n_samples, 1, 1, 1])

    ma = (mask - mf).clamp(0, 1)
    xa = data * ma + means * (1 - ma)
    ya = _forward_4(xa)

    mb = (mask + mf).clamp(0, 1)
    xb = data * mb + means * (1 - mb)
    yb = _forward_4(xb)

    d = yb - ya
    d = d.mean(dim=0)
    d = d.permute(1, 0, 2)
    return d


# pylint: disable=no-member
def show_mask_on_image(
    img: Union[Tensor, np.ndarray],
    mask: Union[Tensor, np.ndarray],
    weight: float = 1,
    colormap: int = cv2.COLORMAP_JET,
):
    """
    Overlays an attention mask on top of an image

    Args:
        img (Union[Tensor, np.ndarray]):
        mask (Union[Tensor, np.ndarray]): The attention mask returned by
        `acc23.explain.VitExplainer.explain`, or any square array (which does
            not need to match the image size)
        weight (float, optional):
        colormap (int, optional):
    """

    def _norm(x: np.ndarray) -> np.ndarray:
        return (x - x.min()) / (x.max() - x.min() + 1e-5)

    img = _norm(
        img if isinstance(img, np.ndarray) else img.permute(1, 2, 0).numpy()
    )
    s = img.shape[1]
    mask = mask if isinstance(mask, Tensor) else torch.tensor(mask)
    mask = resize(mask.unsqueeze(0), (s, s), antialias=True)[0].numpy()
    hm = _norm(cv2.applyColorMap(np.uint8(255 * (1 - mask)), colormap))
    cam = _norm((weight * hm) + img)
    return torch.tensor(cam).permute(2, 0, 1)


class VitExplainer:
    """
    Class that facilitates producing attention masks from ViTs. This is
    specifically tailored for [Hugging Face Transformer's
    `ViTModel`s](https://huggingface.co/docs/transformers/v4.30.0/en/model_doc/vit#transformers.ViTModel).
    Code _heavily_ inspired from https://github.com/jacobgil/vit-explain.

    See also:
        https://jacobgil.github.io/deeplearning/vision-transformer-explainability
        https://arxiv.org/abs/2005.00928

    TODO:
        Batch the whole thing
    """

    _attentions: List[Tensor]
    _discard_ratio: float
    _fusion_mode: Literal["mean", "min", "max"]
    _hook_handles: List[utils.hooks.RemovableHandle]
    _mask_weight: float
    _vit: ViTModel

    def __init__(
        self,
        vit: ViTModel,
        fusion_mode: Literal["mean", "min", "max"] = "mean",
        discard_ratio: float = 0.99,
        mask_weight: float = 1.0,
    ) -> None:
        """
        Args:
            vit (ViTModel):
            fusion_mode (Literal["mean", "min", "max"], optional): The way
                attention maps of every self-attention modules are merged
            discard_ratio (float, optional): For visualization. Consolidated
                attention maps tend to very dense. Setting a high value makes
                clearer attention maps. This should be strictly between 0 and
                1.
            mask_weight (float, optional): For visualization.
        """
        if fusion_mode not in ["mean", "min", "max"]:
            raise ValueError(
                f"Attention fusion mode '{fusion_mode}' not "
                "supported. Available options are 'mean', 'min', and "
                "'max'."
            )
        self._discard_ratio, self._fusion_mode = discard_ratio, fusion_mode
        self._mask_weight, self._vit = mask_weight, vit
        self._attentions, self._hook_handles = [], []

    def _explain_one(self, img: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Only call this after `_register_hooks` so that the attention tensors
        are stored in `_attentions`.

        Args:
            img (Tensor): A single square image of shape `(C, W, W)`
        """
        self._attentions, s = [], self._vit.config.image_size
        with torch.no_grad():
            self._vit(
                resize(img, (s, s), antialias=True).unsqueeze(0),
                output_attentions=True,
            )
        m = self._mask()
        im = show_mask_on_image(img, m, self._mask_weight)
        return im, m

    def _mask(self) -> Tensor:
        """
        Computes the attention mask. Only call this after evaluating a sample
        so that `_attentions` is populated.
        """
        ca = torch.eye(self._attentions[0].shape[-1])
        with torch.no_grad():
            for a in self._attentions:
                if self._fusion_mode == "mean":
                    af = a.mean(dim=1)
                elif self._fusion_mode == "max":
                    af = a.max(dim=1)[0]
                else:  # self.fusion_mode == "min"
                    af = a.min(dim=1)[0]
                flat = af.view(af.shape[0], -1)
                _, idxs = flat.topk(
                    int(flat.shape[-1] * self._discard_ratio),
                    dim=-1,
                    largest=False,
                )
                idxs = idxs[idxs != 0]
                flat[0, idxs] = 0
                i = torch.eye(af.shape[-1], dtype=af.dtype)
                a = (af + i) / 2
                a = a / a.sum(dim=-1)
                ca = torch.matmul(a, ca)
        m = ca[0, 0, 1:]
        s = int(m.shape[-1] ** 0.5)
        m = m.reshape(s, s) / m.max()
        return m

    # pylint: disable=unused-argument
    def _module_forward_hook(
        self, module: nn.Module, args: Any, output: Tensor
    ) -> None:
        """
        The hook that will be called everytime a `ViTSelfAttention` is
        evaluated.
        """
        a = output[1].cpu()
        self._attentions.append(a)

    def _register_hooks(self) -> None:
        """
        Registers `_module_forward_hook` to every `ViTSelfAttention` submodule
        of `vit`
        """
        self._hook_handles = []
        for _, module in self._vit.named_modules():
            if isinstance(module, ViTSelfAttention):
                h = module.register_forward_hook(self._module_forward_hook)
                self._hook_handles.append(h)

    def _unregister_hooks(self) -> None:
        """Removes all the hooks that were added by `_register_hooks`"""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def explain(
        self,
        img: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            img (Tensor): A tensor of shape `(C, W, W)` or `(N, C, W, W)`

        Returns:
            1. A tensor with the same shape as `img` containing all the
               heatmaps (images superposed with the corresponding attention
               masks)
            2. The attention masks of shape `(W', W')` or `(N, W', W')`. The
               value of `W'` depends on the size of the original images and the
               patch size of the vision transformer.
        """
        try:
            self._register_hooks()
            if img.ndim == 3:
                im, m = self._explain_one(img)
            else:
                a = [self._explain_one(k) for k in img]
                b, c = zip(*a)
                im, m = torch.stack(b), torch.stack(c)
        finally:
            self._unregister_hooks()
        return im, m
