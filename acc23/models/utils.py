"""Model utilities"""
__docformat__ = "google"

from typing import Dict, List, Optional, Tuple
import pytorch_lightning as pl

import torch
from torch import Tensor, nn
from transformers.models.resnet.modeling_resnet import ResNetConvLayer
from transformers.activations import get_activation
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

from acc23.constants import IMAGE_RESIZE_TO, N_CHANNELS


class BaseMultilabelClassifier(pl.LightningModule):
    """Base class for multilabel classifiers (duh)"""

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def evaluate(
        self,
        x: Dict[str, Tensor],
        y: Dict[str, Tensor],
        img: Tensor,
        stage: Optional[str] = None,
    ) -> Tuple[Tensor, float, float, float, float, float]:
        """
        Computes and returns various scores and losses. In order:
        1. Binary cross-entropy (which is the loss used during training),
        2. Accuracy score,
        3. Hamming loss,
        4. Precision score,
        5. Recall score,
        6. F1 score.

        Furthermore, if `stage` is given, these values are also logged
         to tensorboard under `<stage>/loss`, `<stage>/acc`, `<stage>/ham`
         (lol), `<stage>/prec`, `<stage>/rec`, and `<stage>/f1` respectively.
        """
        y_true = concat_tensor_dict(y).float().to(self.device)  # type: ignore
        y_pred = self(x, img)
        y_true_np = y_true.cpu().detach().numpy()
        y_pred_np = y_pred.cpu().detach().numpy() > 0.5
        # criterion = nn.functional.binary_cross_entropy
        criterion = nn.functional.mse_loss
        loss = criterion(y_pred, y_true)
        acc = accuracy_score(y_true_np, y_pred_np)
        ham = hamming_loss(y_true_np, y_pred_np)
        prec = precision_score(
            y_true_np,
            y_pred_np,
            average="samples",
            zero_division=0,
        )
        rec = recall_score(
            y_true_np,
            y_pred_np,
            average="samples",
            zero_division=0,
        )
        f1 = f1_score(
            y_true_np,
            y_pred_np,
            average="samples",
            zero_division=0,
        )
        if stage is not None:
            self.log_dict(
                {
                    f"{stage}/loss": loss,
                    f"{stage}/acc": acc,
                    f"{stage}/ham": ham,
                    f"{stage}/prec": prec,
                    f"{stage}/rec": rec,
                    f"{stage}/f1": f1,
                },
                sync_dist=True,
            )
        return loss, float(acc), float(ham), float(prec), float(rec), float(f1)

    def training_step(self, batch, *_, **__):
        x, y, img = batch
        return self.evaluate(x, y, img, "train")[0]

    def validation_step(self, batch, *_, **__):
        x, y, img = batch
        return self.evaluate(x, y, img, "val")[0]


def basic_encoder(
    blocks_channels: List[int],
    input_size: int = IMAGE_RESIZE_TO,
    input_channels: int = N_CHANNELS,
    activation: str = "relu",
) -> Tuple[nn.Sequential, int]:
    """
    Basic image encoder that is just a succession of (non skipped) downsampling
    blocks, followed by a final flatten layer. A block is made of
    1. a convolution layer which cuts the inputs' height and width by half,
    2. a batch normalization layer, except for the first block,
    3. a sigmoid activation.
    In particular, the output images' size will be `input_size / (2 **
    len(blocks_channels))`, and the output vector dimension will be

        blocks_channels[-1] * (input_size / (2 ** len(blocks_channels))) ** 2

    Args:
        blocks_channels (List[int]): Number of output channels of each
            block. Note that each blocks cuts the width and height of the input
            by half.
        input_size (int): The width (or equivalently the height) of an imput
            image.
        input_channels (int): Number of channels of the input image

    Return:
        The encoder and the (flat) dimension of the output vector.
    """
    module, c = nn.Sequential(), [input_channels] + blocks_channels
    module = nn.Sequential(
        *[
            ResNetConvLayer(c[i - 1], c[i], 5, 2, activation)
            for i in range(1, len(c))
        ]
    )
    # for i in range(1, len(c)):
    #     module.append(nn.Conv2d(c[i - 1], c[i], 4, 2, 1))
    #     if i > 1:
    #         module.append(nn.BatchNorm2d(c[i]))
    #     module.append(get_activation(activation))
    module.append(nn.Flatten())
    # Height (eq. width) of the encoded image before flatten
    es = int(input_size / (2 ** len(blocks_channels)))
    return (module, blocks_channels[-1] * (es**2))


def concat_tensor_dict(d: Dict[str, Tensor]) -> Tensor:
    """
    Converts a dict of tensors to a tensor. The tensors (of the dict) are
    concatenated along the last axis. If they are 1-dimensional, they are
    unsqueezed along a new axis at the end. I know it's not super clear, so
    here is an example

        # The foo tensor will be unsqueezed
        d = {"foo": torch.ones(3), "bar": torch.zeros((3, 1))}
        concat_tensor_dict(d)
        >>> tensor([[1., 0.],
        >>>         [1., 0.],
        >>>         [1., 0.]])

    """

    def _maybe_unsqueeze(t: Tensor) -> Tensor:
        return t if t.ndim > 1 else t.unsqueeze(-1)

    return torch.concatenate(list(map(_maybe_unsqueeze, d.values())), dim=-1)


def linear_chain(
    n_inputs: int, n_neurons: List[int], activation: str = "sigmoid"
) -> nn.Sequential:
    """
    A sequence of linear layers with sigmoid activation (including at the end).
    """
    n, module = [n_inputs] + n_neurons, nn.Sequential()
    for i in range(1, len(n)):
        module.append(nn.Linear(n[i - 1], n[i]))
        module.append(get_activation(activation))
    return module
