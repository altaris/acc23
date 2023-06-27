"""
Everything related to explainability

Code heavily inspired from https://github.com/jacobgil/vit-explain

See also:
    https://jacobgil.github.io/deeplearning/vision-transformer-explainability
"""
__docformat__ = "google"

from typing import Any, List, Literal, Tuple, Union

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage
from torch import Tensor, nn, utils
from torchvision.transforms.functional import resize
from transformers.models.vit.modeling_vit import ViTModel, ViTSelfAttention


def imshow(img: Union[Tensor, np.ndarray]) -> AxesImage:
    """
    Convenience function to plot an image

    Args:
        img (Union[Tensor, np.ndarray]): If it is a torch tensor, it must have
        shape `(W, H)` or `(3, W, H)`. If it is an numpy array, it must have
        shape `(W, H)` or `(W, H, 3)`
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
    return plt.imshow(img)


# pylint: disable=no-member
def show_mask_on_image(
    img: Union[Tensor, np.ndarray],
    mask: Union[Tensor, np.ndarray],
    weight: float = 1,
    colormap: int = cv2.COLORMAP_JET,
):
    """Overlays an attention mask on top of an image"""

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
    TODO: Batch the whole thing
    """

    _attentions: List[Tensor]
    _hook_handles: List[utils.hooks.RemovableHandle]
    discard_ratio: float
    fusion_mode: Literal["mean", "min", "max"]
    mask_weight: float
    vit: ViTModel

    def __init__(
        self,
        vit: ViTModel,
        fusion_mode: Literal["mean", "min", "max"] = "mean",
        discard_ratio: float = 0.99,
        mask_weight: float = 1.0,
    ) -> None:
        if fusion_mode not in ["mean", "min", "max"]:
            raise ValueError(
                f"Attention fusion mode '{fusion_mode}' not "
                "supported. Available options are 'mean', 'min', and "
                "'max'."
            )
        self.discard_ratio, self.fusion_mode = discard_ratio, fusion_mode
        self.mask_weight, self.vit = mask_weight, vit
        self._attentions, self._hook_handles = [], []

    def _explain_one(
        self,
        img: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Only call this after `_register_hooks` so that the attention tensors
        are stored in `_attentions`.

        Args:
            img (Tensor): A single square image of shape `(C, W, W)`
        """
        self._attentions, s = [], self.vit.config.image_size
        with torch.no_grad():
            self.vit(
                resize(img, (s, s), antialias=True).unsqueeze(0),
                output_attentions=True,
            )
        m = self._mask()
        im = show_mask_on_image(img, m, self.mask_weight)
        return im, m

    def _mask(self) -> Tensor:
        """
        Computes the attention mask. Only call this after evaluating a sample
        so that `_attentions` is populated.
        """
        ca = torch.eye(self._attentions[0].shape[-1])
        with torch.no_grad():
            for a in self._attentions:
                if self.fusion_mode == "mean":
                    af = a.mean(dim=1)
                elif self.fusion_mode == "max":
                    af = a.max(dim=1)[0]
                else:  # self.fusion_mode == "min"
                    af = a.min(dim=1)[0]
                flat = af.view(af.shape[0], -1)
                _, idxs = flat.topk(
                    int(flat.shape[-1] * self.discard_ratio),
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
        for _, module in self.vit.named_modules():
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
