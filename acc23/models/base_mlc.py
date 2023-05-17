"""Base class for multilabel classifiers"""

from typing import Any, Dict, Optional, Tuple
import pytorch_lightning as pl

import torch
from torch import Tensor, nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

from acc23.constants import (
    N_TARGETS,
    POSITIVE_TARGET_RATIOS,
    POSITIVE_TARGET_COUNTS,
)

from .layers import concat_tensor_dict


class BaseMultilabelClassifier(pl.LightningModule):
    """Base class for multilabel classifiers (duh)"""

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
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
        y_true = concat_tensor_dict(y).float().to(self.device)  # type: ignore
        y_pred = self(x, img)
        y_true_np = y_true.cpu().detach().numpy()
        y_pred_np = y_pred.cpu().detach().numpy() > 0
        w = 2
        positive_count = Tensor(POSITIVE_TARGET_COUNTS).to(y_pred.device)
        # positive_ratio = Tensor(POSITIVE_TARGET_RATIOS).to(y_pred.device)
        loss = (
            # nn.functional.mse_loss(y_pred.sigmoid(), y_true)
            # nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)
            # weighted_binary_cross_entropy_with_logits(
            #     y_pred, y_true, positive_ratio
            # )
            # rebalanced_binary_cross_entropy_with_logits(
            #     y_pred, y_true, positive_count
            # )
            focal_loss_with_logits(y_pred, y_true)
            # distribution_balanced_loss_with_logits(
            #     y_pred, y_true, positive_count
            # )
            # bp_mll_loss(y_pred.sigmoid(), y_true)
            - w * continuous_f1_score(y_pred.sigmoid(), y_true)
        )
        acc = accuracy_score(y_true_np, y_pred_np)
        ham = hamming_loss(y_true_np, y_pred_np)
        kw = {
            "y_true": y_true_np,
            "y_pred": y_pred_np,
            "average": "samples",
            "zero_division": 0,
        }
        prec = precision_score(**kw)
        rec = recall_score(**kw)
        f1 = f1_score(**kw)
        f1_min = f1_score(
            y_true_np, y_pred_np, zero_division=0, average=None
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


def class_balanced_loss(  # TODO
    y_pred: Tensor,
    y_true: Tensor,
    positive_count: Tensor,
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
    r = (1 - beta) / (1 - beta**positive_count)
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
    Implementation of the F1 score as specified by the challenge organizers:

        Let y_true (test labels) and y_pred (your predictions) be matrices with
        the same dimensions, where the columns represent the individual classes
        in a multi-label classification setting.

        1/ we calculate mean precision (P) and mean recall (R) for each column
        i:

            P(i) = mean(sum(y_true(i) * y_pred(i)) / (sum(y_pred(i)) + 1e-10))
            R(i) = mean(sum(y_true(i) * y_pred(i)) / (sum(y_true(i)) + 1e-10))

        2/ we calculate the F1 score for each column i using the formula:

            F1(i) = 2 * P(i) * R(i) / (P(i) + R(i) + 1e-10)

        3/ we compute the mean F1 score across all columns:

            mean_F1 = mean(F1(i)) for all i

        This mean_F1 value is the score.

    After experimentation, this seems to correspond to
    `sklearn.metrics.f1_score` with `average=macro`:

        import numpy as np
        from sklearn.metrics import f1_score
        from acc23.models.base_mlc import f1

        y_true = np.random.uniform(0, 1, [100, 20]) > .5
        y_pred = np.random.uniform(0, 1, [100, 20]) > .5

        for a in ["micro", "macro", "samples", "weighted"]:
            print(
                "sklearn.metrics.f1_score with average =", a,
                f1_score(y_true, y_pred, average=a)
            )
        print("Challenge implementation", continuous_f1_score(y_pred, y_true))

    `y_pred` is expected to contain class probabilities.
    """
    p, pp, tp = y_true.sum(0), y_pred.sum(0), (y_true * y_pred).sum(0)
    pr, re = tp / (pp + 1e-10), tp / (p + 1e-10)
    return torch.mean(2 * pr * re / (pr + re + 1e-10))


def distribution_balanced_loss_with_logits(
    y_pred: Tensor,
    y_true: Tensor,
    positive_count: Tensor,
    alpha: float = 0.1,
    beta: float = 10.0,
    mu: float = 0.3,
    gamma: float = 2.0,
    lambda_: float = 5.0,
    kappa: float = 5e-3,
) -> Tensor:
    """
    A mix between focal loss (`acc23.models.base_mlc.focal_loss_with_logits`) and
    rebalanced cross entropy
    (`acc23.models.base_mlc.rebalanced_binary_cross_entropy_with_logits`).
    """
    nu = -kappa * torch.log(1 / y_pred.sigmoid() - 1).clamp(-100, 0)
    z = y_true * (y_pred - nu) + (1 - y_true) * (-1) * lambda_ * (y_pred - nu)
    foc = (y_true * (1 - z.sigmoid()) + (1 - y_true) * z.sigmoid()) ** gamma
    lmb = y_true + (1 - y_true) / (lambda_ + 1e-10)
    pc = 1 / (N_TARGETS * positive_count + 1e-10)
    pi = torch.sum(y_true / (N_TARGETS * positive_count + 1e-10), dim=-1)
    r = torch.outer(1 / (pi + 1e-10), pc)
    r_hat = alpha + torch.sigmoid(beta * (r - mu))
    bce = nn.functional.binary_cross_entropy_with_logits(
        z, y_true, reduction="none"
    )
    return torch.mean(r_hat * lmb * foc * bce)


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
    return torch.mean(foc * bce)


def rebalanced_binary_cross_entropy_with_logits(
    y_pred: Tensor,
    y_true: Tensor,
    positive_count: Tensor,
    alpha: float = 0.1,
    beta: float = 10.0,
    mu: float = 0.3,
) -> Tensor:
    """
    Implementation of

        Distribution-Balanced Loss for Multi-Label Classification in
        Long-Tailed Datasets
    """
    # positive_count: (C,), pc: (C,), pi: (N,) => r: (N, C)
    pc = 1 / (N_TARGETS * (positive_count + 1e-10))
    pi = torch.sum(y_true / (N_TARGETS * positive_count + 1e-10), dim=-1)
    r = torch.outer(1 / (pi + 1e-10), pc)
    r_hat = alpha + torch.sigmoid(beta * (r - mu))
    bce = nn.functional.binary_cross_entropy_with_logits(
        y_pred, y_true, reduction="none"
    )
    return torch.mean(r_hat * bce)


def weighted_binary_cross_entropy_with_logits(
    y_pred: Tensor, y_true: Tensor, positive_ratio: Tensor
) -> Tensor:
    """
    Binary crossentropy loss that takes the positive class ratio into account.
    Use this in your fight against class imbalance. If `y_pred` and `y_true`
    have shape `(N, T)`, then `positive_ratio` must have shape `(T,)`, where
    `T` is the number of targets. The formula is as follows, where $r_p$ is
    `positive_ratio`:
    $$
        L = - \\frac{1}{2} \\left(
            \\frac{1}{r_p} y \\log \\sigma(x)
            + \\frac{1}{1-r_p} (1-y) \\log (1-\\sigma(x))
        \\right)
    $$
    The idea is that if $r_p$ is small (i.e. small positive class, large
    negative class), then false negatives get penalized more. Conversely, if
    $r_p$ is large (i.e. large positive class, small negative class), then
    false positives get penalized more. The factor of $1/2$ is so that if the
    classes are perfectly balanced (i.e. $r_p = 1/2$), then $L$ matches the
    usual binary cross entropy loss.

    See also:
        https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    """
    u, v = 1 / (positive_ratio + 1e-10), 1 / (1 - positive_ratio + 1e-10)
    w = y_true * u + (1 - y_true) * v
    bce = nn.functional.binary_cross_entropy_with_logits(
        y_pred, y_true, reduction="none"
    )
    return 0.5 * torch.mean(w * bce)
