"""Base class for multilabel classifiers"""

from typing import Dict, Optional, Tuple
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
        loss = nn.functional.mse_loss(y_pred, y_true)
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
            self.log(f"{stage}/f1", f1, sync_dist=True, prog_bar=True)
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
        return loss, float(acc), float(ham), float(prec), float(rec), float(f1)

    def training_step(self, batch, *_, **__):
        x, y, img = batch
        return self.evaluate(x, y, img, "train")[0]

    def validation_step(self, batch, *_, **__):
        x, y, img = batch
        return self.evaluate(x, y, img, "val")[0]
