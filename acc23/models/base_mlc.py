"""Base class for multilabel classifiers"""
__docformat__ = "google"

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from torch import Tensor, nn, optim

from acc23.constants import TRUE_TARGETS_PREVALENCE

from .layers import concat_tensor_dict


class BaseMultilabelClassifier(pl.LightningModule):
    """Base class for multilabel classifiers (duh)"""

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5
        )
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
        6. minimal F1 score (across targets).

        Furthermore, if `stage` is given, these values are also logged to
        tensorboard under `<stage>/loss`, `<stage>/acc`, `<stage>/ham` (lol),
        `<stage>/prec`, `<stage>/rec`, `<stage>/f1`, and `<stage>/f1_min`
        respectively.
        """
        out = self(x, img)  # out may be y_pred or (y_pred, extra loss term)
        y_pred, extra_loss = out if isinstance(out, tuple) else (out, 0.0)

        y_true = concat_tensor_dict(y)
        # Replace nan targets by predicted values
        y_true = torch.where(y_true.isnan(), y_pred.detach(), y_true)
        y_true = (y_true > 0.0).float().to(self.device)  # type: ignore

        y_true_np = y_true.cpu().detach().numpy()
        y_pred_np = y_pred.cpu().detach().numpy() > 0
        w = 1.0
        prevalence = Tensor(TRUE_TARGETS_PREVALENCE).to(y_pred.device)
        loss = (
            10 * nn.functional.mse_loss(y_pred.sigmoid(), y_true)
            # nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)
            # rebalanced_bce_with_logits(y_pred, y_true, prevalence)
            + focal_loss_with_logits(y_pred, y_true)
            # distribution_balanced_loss_with_logits(y_pred, y_true, prevalence)
            # bp_mll_loss(y_pred.sigmoid(), y_true)
            # - w * continuous_f1_score(y_pred.sigmoid(), y_true)
            + extra_loss
        )
        acc = accuracy_score(y_true_np, y_pred_np)
        ham = hamming_loss(y_true_np, y_pred_np)
        kw = {
            "y_true": y_true_np,
            "y_pred": y_pred_np,
            "average": "macro",
            "zero_division": 0,
        }
        prec = precision_score(**kw)
        rec = recall_score(**kw)
        f1 = f1_score(**kw)
        f1_min = f1_score(
            y_true=y_true_np, y_pred=y_pred_np, zero_division=0, average=None
        ).min()
        if stage is not None:
            self.log_dict(
                {
                    f"{stage}/loss": loss,
                    f"{stage}/f1": f1,
                },
                sync_dist=True,
                prog_bar=True,
            )
            self.log_dict(
                {
                    f"{stage}/acc": acc,
                    f"{stage}/extra": extra_loss,
                    f"{stage}/f1_min": f1_min,
                    f"{stage}/ham": ham,
                    f"{stage}/prec": prec,
                    f"{stage}/rec": rec,
                },
                sync_dist=True,
            )
        return loss

    def training_step(self, batch, *_, **__):
        x, y, img = batch
        return self.evaluate(x, y, img, "train")

    # Set self.automatic_optimization = False in __init__ to use this
    # def training_step(self, batch, *_, **__):
    #     x, y, img = batch
    #     opt = self.optimizers()
    #     opt.zero_grad()
    #     loss = self.evaluate(x, y, img, "train")
    #     self.manual_backward(loss)
    #     opt.step()
    #     a = []
    #     for p in self.parameters():
    #         if p.grad is not None:
    #             a.append(p.grad.mean())
    #     print("TRAIN", "loss", loss, "grads", Tensor(a).mean())

    def validation_step(self, batch, *_, **__):
        x, y, img = batch
        return self.evaluate(x, y, img, "val")


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


def class_balanced_loss(
    y_pred: Tensor,
    y_true: Tensor,
    prevalence: Tensor,
    beta: float = 10.0,
    gamma: float = 2.0,
) -> Tensor:
    """
    Implementation of the class-balanced loss of

        Y. Cui, M. Jia, T. -Y. Lin, Y. Song and S. Belongie, "Class-Balanced
        Loss Based on Effective Number of Samples," 2019 IEEE/CVF Conference on
        Computer Vision and Pattern Recognition (CVPR), Long Beach, CA, USA,
        2019, pp. 9260-9269, doi: 10.1109/CVPR.2019.00949.
    """
    r = (1 - beta) / (1 - beta**prevalence)
    pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    pt = (1 - pt) ** gamma
    bce = nn.functional.binary_cross_entropy(y_pred, y_true, reduction="none")
    return torch.mean(r * pt * bce)


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
    gamma: float = 2.0,
    lambda_: float = 5.0,
    kappa: float = 5e-2,
) -> Tensor:
    """
    A mix between focal loss (`acc23.models.base_mlc.focal_loss_with_logits`) and
    rebalanced cross entropy
    (`acc23.models.base_mlc.rebalanced_binary_cross_entropy_with_logits`).
    """
    nu = kappa * torch.log(1 / (prevalence + 1e-5) - 1 + 1e-5)
    q_ns = y_true * (y_pred - nu) + (1 - y_true) * lambda_ * (y_pred - nu)
    q = q_ns.sigmoid()
    foc = (y_true * (1 - q) + (1 - y_true) * q) ** gamma
    lmb = y_true + (1 - y_true) / (lambda_ + 1e-5)
    # n_targets = y_true.shape[1]
    # pc = 1 / (n_targets * prevalence + 1e-5)
    # pi = torch.sum(y_true / (n_targets * prevalence + 1e-5), dim=-1)
    # r = torch.outer(1 / (pi + 1e-5), pc)
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

        Focal Loss for Dense Object Detection
    """
    p = y_pred.sigmoid()
    foc = (y_true * (1 - p) + (1 - y_true) * p) ** gamma
    bce = nn.functional.binary_cross_entropy_with_logits(
        y_pred, y_true, reduction="none"
    )
    return (foc * bce).sum(-1).mean()


def rebalanced_bce_with_logits(
    y_pred: Tensor,
    y_true: Tensor,
    prevalence: Tensor,
    alpha: float = 0.1,
    beta: float = 10.0,
    mu: float = 0.3,
) -> Tensor:
    """
    Implementation of

        Distribution-Balanced Loss for Multi-Label Classification in
        Long-Tailed Datasets
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
    return (r_hat * bce).sum(-1).mean()
