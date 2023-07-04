"""Base class for multilabel classifiers"""
__docformat__ = "google"

from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger as logging
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from torch import Tensor, nn, optim

from acc23.constants import TRUE_TARGETS_IRLBL, TRUE_TARGETS_PREVALENCE

from .layers import concat_tensor_dict


class BaseMultilabelClassifier(pl.LightningModule):
    """Base class for multilabel classifiers (duh)"""

    # pylint: disable=unused-argument
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        loss_function: Literal[
            "bce", "irlbl_bce", "focal", "db", "mse"
        ] = "db",
        swa_lr: Optional[float] = None,  # 1e-3,
        swa_epoch: Optional[Union[int, float]] = None,  # 10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )
        # scheduler = optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=1e-3,
        #     steps_per_epoch=28,  # TODO: do not hardcode
        #     epochs=200,  # TODO: do not hardcode
        # )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }

    def evaluate(
        self,
        x: Dict[str, Tensor],
        y: Dict[str, Tensor],
        img: Tensor,
        stage: Optional[str] = None,
    ) -> Tensor:
        """
        Computes various scores and losses:
        1. The loss used for training,
        2. Accuracy score,
        3. Hamming loss,
        4. Precision score,
        5. Recall score,
        6. F1 score,
        7. an optional extra loss reported by the model (e.g. regularization).

        Furthermore, if `stage` is given, these values are also logged to
        tensorboard under `<stage>/loss`, `<stage>/acc`, `<stage>/ham` (lol),
        `<stage>/prec`, `<stage>/rec`, `<stage>/f1`, and `<stage>/extra`
        respectively.
        """
        y_pred = self(x, img)
        y_true = concat_tensor_dict(y)
        y_true = torch.where(  # Replace NaN targets
            y_true.isnan(),
            y_pred > 0.0,  # Predicted targets ...
            # torch.randint_like(y_true, 2), # ... OR Random targets
            y_true,
        )
        y_true = y_true.float().to(y_pred.device)  # type: ignore

        # Loss
        lfn = self.hparams["loss_function"]
        if lfn == "bce":
            loss = nn.functional.binary_cross_entropy_with_logits(
                y_pred, y_true
            )
        elif lfn == "irlbl_bce":
            loss = nn.functional.binary_cross_entropy_with_logits(
                y_pred,
                y_true,
                weight=Tensor(TRUE_TARGETS_IRLBL).to(y_pred.device),
            )
        elif lfn == "focal":
            loss = focal_loss_with_logits(y_pred, y_true)
        elif lfn == "db":
            p = torch.tensor(TRUE_TARGETS_PREVALENCE, device=y_pred.device)
            loss = distribution_balanced_loss_with_logits(y_pred, y_true, p)
        elif lfn == "mse":
            loss = nn.functional.mse_loss(y_pred.sigmoid(), y_true)
        else:
            raise ValueError(f"Unsupported loss function '{lfn}'")
        if stage is not None:
            kw = {
                "y_true": y_true.cpu().detach().numpy().astype(int),
                "y_pred": (y_pred.cpu().detach().numpy() > 0).astype(int),
                "average": "macro",
                "zero_division": 0,
            }
            self.log_dict(
                {
                    f"{stage}/loss": loss,
                    f"{stage}/f1": f1_score(**kw),
                },
                sync_dist=True,
                prog_bar=True,
            )
            self.log_dict(
                {
                    f"{stage}/acc": accuracy_score(kw["y_true"], kw["y_pred"]),
                    f"{stage}/ham": hamming_loss(kw["y_true"], kw["y_pred"]),
                    f"{stage}/prec": precision_score(**kw),
                    f"{stage}/rec": recall_score(**kw),
                },
                sync_dist=True,
            )
        return loss

    # Uncomment this to find ununsed parameters
    # def on_after_backward(self) -> None:
    #     for n, p in self.named_parameters():
    #         if p.grad is None:
    #             logging.warning("Weight '{}' does not have a gradient!", n)

    def on_train_start(self):
        # https://lightning.ai/docs/pytorch/latest/extensions/logging.html#logging-hyperparameters
        keys = ["val/loss", "val/ham", "val/f1", "val/prec", "val/rec"]
        self.logger.log_hyperparams(self.hparams, {k: np.nan for k in keys})

    def training_step(self, batch, *_, **__):
        x, y, img = batch
        return self.evaluate(x, y, img, "train")

    def validation_step(self, batch, *_, **__):
        x, y, img = batch
        return self.evaluate(x, y, img, "val")

    def test_step(self, batch, *_, **__):
        x, y, img = batch
        return self.evaluate(x, y, img, "test")

    def predict_step(self, batch, *_, **__):
        x, _, img = batch
        return self(x, img)


class ModuleWeightsHistogram(pl.Callback):
    """Logs a histogram of the module's weights"""

    every_n_epochs: int
    key: str
    warned_wrong_logger: bool = False

    def __init__(
        self, every_n_epochs: int = 5, key: str = "train/weights"
    ) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.key = key

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        ps = [p.flatten() for p in pl_module.parameters()]
        if isinstance(trainer.logger, pl.loggers.TensorBoardLogger):
            trainer.logger.experiment.add_histogram(
                self.key,
                torch.concat(ps),
                bins="auto",
                global_step=trainer.global_step,
            )
        elif not self.warned_wrong_logger:
            logging.warning(
                "ModuleWeightsHistogram callback: Trainer's logger is has "
                f"type '{type(trainer.logger)}', but a tensorboard logger is "
                "required. This warning will only be logged once"
            )
            self.warned_wrong_logger = True


def bp_mll_loss(y_pred: Tensor, y_true: Tensor, *_, **__) -> Tensor:
    """
    BM-MLL loss introduced in

        Zhang, Min-Ling, and Zhi-Hua Zhou. "Multilabel neural networks with
        applications to functional genomics and text categorization." IEEE
        transactions on Knowledge and Data Engineering 18.10 (2006): 1338-1351.

    `y_pred` is expected to contain class probabilities.
    """
    y_bar = 1 - y_true
    k = 1 / (y_true.sum(dim=-1) * y_bar.sum(dim=-1) + 1e-10)
    s = y_true.unsqueeze(-1) * y_bar.unsqueeze(1)
    # s_ijk = y_true_ij and (not y_true_ik)
    a = y_pred.unsqueeze(-1) - y_pred.unsqueeze(1)
    a = a.sigmoid()
    # a_ijk = y_pred_ij - y_pred_ik
    b = s * torch.exp(-a)
    return (k * b.sum((1, 2))).mean()
    # Some tests:
    # y_true = torch.Tensor([[0, 1, 0], [1, 1, 0]])
    # y_pred = torch.Tensor([[.1, .8, .9], [.1, .9, .1]])
    # for i in range(2):
    #     for j in range(3):
    #         for k in range(3):
    #             assert s[i, j, k] == y_true[i, j] * (1 - y_true[i, k])
    #             assert a[i, j, k] == y_pred[i, j] - y_pred[i, k]


def class_balanced_focal_loss_with_logits(
    y_pred: Tensor,
    y_true: Tensor,
    count: Tensor,
    beta: float = 0.9,
    gamma: float = 2.0,
) -> Tensor:
    """
    Implementation of the class-balanced focal loss of

        Y. Cui, M. Jia, T. -Y. Lin, Y. Song and S. Belongie, "Class-Balanced
        Loss Based on Effective Number of Samples," 2019 IEEE/CVF Conference on
        Computer Vision and Pattern Recognition (CVPR), Long Beach, CA, USA,
        2019, pp. 9260-9269, doi: 10.1109/CVPR.2019.00949.
    """
    r = (1 - beta) / (1 - beta**count + 1e-5)
    p = y_pred.sigmoid()
    foc = (y_true * (1 - p) + (1 - y_true) * p) ** gamma
    bce = nn.functional.binary_cross_entropy_with_logits(
        y_pred, y_true, reduction="none", weight=r
    )
    return torch.mean(foc * bce)


def continuous_hamming_loss(
    y_pred: Tensor, y_true: Tensor, *_, **__
) -> Tensor:
    """
    Continuous (and differentiable) version of the Hamming loss. The (discrete)
    Hamming loss is the fraction of labels that are incorrectly predicted.

    `y_pred` is expected to contain class probabilities.
    """
    h = (1 - y_true) * y_pred + y_true * (1 - y_pred)
    return h.mean()


def continuous_f1_score(y_pred: Tensor, y_true: Tensor, *_, **__) -> Tensor:
    """
    Continuous macro-f1 score. `y_pred` is expected to contain class
    probabilities, not logits.
    """
    p, pp, tp = y_true.sum(0), y_pred.sum(0), (y_true * y_pred).sum(0)
    pr, re = tp / (pp + 1e-10), tp / (p + 1e-10)
    return torch.mean(2 * pr * re / (pr + re + 1e-10))


def distribution_balanced_loss_with_logits(
    y_pred: Tensor,
    y_true: Tensor,
    prevalence: Tensor,
    alpha: float = 0.1,
    beta: float = 10.0,
    mu: float = 0.3,
    gamma: float = 1.0,
    lambda_: float = 5.0,
    kappa: float = 5e-2,
) -> Tensor:
    """
    Distribution-balanced loss $\\mathcal{L}_{DB}$ from

        Wu, T., Huang, Q., Liu, Z., Wang, Y., Lin, D. (2020).
        Distribution-Balanced Loss for Multi-label Classification in
        Long-Tailed Datasets. In: Vedaldi, A., Bischof, H., Brox, T., Frahm,
        JM. (eds) Computer Vision - ECCV 2020. ECCV 2020. Lecture Notes in
        Computer Science(), vol 12349. Springer, Cham.
        https://doi.org/10.1007/978-3-030-58548-8_10

    It is essentially a mix between focal loss
    (`acc23.models.base_mlc.focal_loss_with_logits`) and rebalanced cross
    entropy
    (`acc23.models.base_mlc.rebalanced_binary_cross_entropy_with_logits`).

    See also:

        Huang, Yi, et al. "Balancing methods for multi-label text
        classification with long-tailed class distribution." arXiv preprint
        arXiv:2109.04712 (2021).

    """
    nu = kappa * torch.log(1 / (prevalence + 1e-5) - 1 + 1e-5)
    q_ns = torch.where(y_true == 1, 1, lambda_) * (y_pred - nu)
    q = q_ns.sigmoid()
    foc = torch.where(y_true == 1, 1 - q, q) ** gamma
    lmb = torch.where(y_true == 1, 1, 1 / (lambda_ + 1e-5))
    r_pi = torch.sum(y_true / (prevalence + 1e-5), dim=-1)
    r = 1 / (torch.outer(r_pi, prevalence) + 1e-5)
    r_hat = alpha + torch.sigmoid(beta * (r - mu))
    bce = nn.functional.binary_cross_entropy_with_logits(
        q_ns, y_true, reduction="none"
    )
    return (r_hat * lmb * foc * bce).sum(-1).mean()


def focal_loss_with_logits(
    y_pred: Tensor,
    y_true: Tensor,
    gamma: float = 2.0,
) -> Tensor:
    """
    Balanced-focal loss of

        T. -Y. Lin, P. Goyal, R. Girshick, K. He and P. Dollár, "Focal Loss for
        Dense Object Detection," 2017 IEEE International Conference on Computer
        Vision (ICCV), Venice, Italy, 2017, pp. 2999-3007, doi:
        10.1109/ICCV.2017.324.

    See also:

        Huang, Yi, et al. "Balancing methods for multi-label text
        classification with long-tailed class distribution." arXiv preprint
        arXiv:2109.04712 (2021).

    """
    p = y_pred.sigmoid()
    foc = torch.where(y_true == 1, 1 - p, p) ** gamma
    bce = nn.functional.binary_cross_entropy_with_logits(
        y_pred, y_true, reduction="none"
    )
    return (foc * bce).mean()


def rebalanced_bce_with_logits(
    y_pred: Tensor,
    y_true: Tensor,
    prevalence: Tensor,
    alpha: float = 0.1,
    beta: float = 10.0,
    mu: float = 0.3,
) -> Tensor:
    """
    Rebalanced binary cross-entropy loss $\\mathcal{L}_{R-BCE}$ from

        Wu, T., Huang, Q., Liu, Z., Wang, Y., Lin, D. (2020).
        Distribution-Balanced Loss for Multi-label Classification in
        Long-Tailed Datasets. In: Vedaldi, A., Bischof, H., Brox, T., Frahm,
        JM. (eds) Computer Vision - ECCV 2020. ECCV 2020. Lecture Notes in
        Computer Science(), vol 12349. Springer, Cham.
        https://doi.org/10.1007/978-3-030-58548-8_10

    """
    # prevalence: (C,), pc: (C,), pi: (N,) => r: (N, C)
    # n_targets = y_true.shape[1]
    # pc = 1 / (n_targets * prevalence + 1e-5)
    # pi = torch.sum(y_true / (n_targets * prevalence + 1e-5), dim=-1)
    # r = torch.outer(1 / (pi + 1e-5), pc)
    r_pi = torch.sum(y_true / (prevalence + 1e-5), dim=-1)
    r = 1 / (torch.outer(r_pi, prevalence) + 1e-5)
    r_hat = alpha + torch.sigmoid(beta * (r - mu))
    bce = nn.functional.binary_cross_entropy_with_logits(
        y_pred, y_true, reduction="none"
    )
    return (r_hat * bce).mean()
