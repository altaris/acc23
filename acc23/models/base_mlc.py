"""Base class for multilabel classifiers"""

from typing import Dict, Optional, Tuple
import numpy as np
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
from .utils import concat_tensor_dict


class BaseMultilabelClassifier(pl.LightningModule):
    """Base class for multilabel classifiers (duh)"""

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.2, patience=10, min_lr=5e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/f1",
        }

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

        Furthermore, if `stage` is given, these values are also logged to
        tensorboard under `<stage>/loss`, `<stage>/acc`, `<stage>/ham` (lol),
        `<stage>/prec`, `<stage>/rec`, and `<stage>/f1` respectively.
        """
        y_true = concat_tensor_dict(y).float().to(self.device)  # type: ignore
        y_pred = self(x, img)
        y_true_np = y_true.cpu().detach().numpy()
        y_pred_np = y_pred.cpu().detach().numpy() > 0.5
        w = 0.5
        loss = nn.functional.mse_loss(y_pred, y_true) - w * cf1(y_pred, y_true)
        # loss = nn.functional.binary_cross_entropy(y_pred, y_true) - w * cf1(
        #     y_pred, y_true
        # )
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
        f1s = f1_score(**kw)
        if stage is not None:
            self.log(f"{stage}/f1", f1s, sync_dist=True, prog_bar=True)
            self.log_dict(
                {
                    f"{stage}/loss": loss,
                    f"{stage}/acc": acc,
                    f"{stage}/ham": ham,
                    f"{stage}/prec": prec,
                    f"{stage}/rec": rec,
                },
                sync_dist=True,
            )
        return (
            loss,
            float(acc),
            float(ham),
            float(prec),
            float(rec),
            float(f1s),
        )

    def training_step(self, batch, *_, **__):
        x, y, img = batch
        return self.evaluate(x, y, img, "train")[0]

    def validation_step(self, batch, *_, **__):
        x, y, img = batch
        return self.evaluate(x, y, img, "val")[0]


def bp_mll_loss(y_pred: Tensor, y_true: Tensor, *_, **__) -> Tensor:
    """
    BM-MLL loss introduced in

        Zhang, Min-Ling, and Zhi-Hua Zhou. "Multilabel neural networks with
        applications to functional genomics and text categorization." IEEE
        transactions on Knowledge and Data Engineering 18.10 (2006): 1338-1351.
    """
    y_bar = 1 - y_true
    k = 1 / (y_true.sum(dim=-1) * y_bar.sum(dim=-1) + 1e-10)
    s = y_true.unsqueeze(-1) * y_bar.unsqueeze(1)
    # s_ijk = y_true_ij and (not y_true_ik)
    a = y_pred.unsqueeze(-1) - y_pred.unsqueeze(1)
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


def cf1(y_pred: Tensor, y_true: Tensor, *_, **__) -> Tensor:
    """
    Home-baked continuous (and differentiable) analogue to the mean (or
    macro-average) F1 score.
    """
    p, n = y_true, (1 - y_true)
    pp, pn = y_pred, (1 - y_pred)
    tp, fp, fn = p * pp, n * pp, p * pn
    tp, fp, fn = tp**2, fp**2, fn**2
    tp, fp, fn = tp.sum(dim=-1), fp.sum(dim=-1), fn.sum(dim=-1)
    nf1 = 2 * tp / (2 * tp + fp + fn)
    return nf1.mean()


def f1(y_pred: np.ndarray, y_true: np.ndarray, *_, **__) -> np.ndarray:
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
        print("Challenge implementation", f1(y_pred, y_true))
    """
    a, b, c = y_true.sum(0), y_pred.sum(0), (y_true * y_pred).sum(0)
    p, r = c / (b + 1e-10), c / (a + 1e-10)
    f = 2 * p * r / (p + r + 1e-10)
    return f.mean()
