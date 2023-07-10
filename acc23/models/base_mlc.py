"""
Base class for multilabel classifiers. This module also contains implementation
of various loss function.
"""

from typing import Any, Dict, Iterable, Literal, Optional, Union

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
    """
    Base class for multilabel classifiers (duh). All model prototypes listed
    in `acc23.models` inherit from this class.
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        loss_function: Literal[
            "bce", "irlbl_bce", "focal", "db", "mse", "mc"
        ] = "db",
        swa_lr: Optional[float] = None,  # 1e-3,
        swa_epoch: Optional[Union[int, float]] = None,  # 10,
        **kwargs,
    ) -> None:
        """
        Args:
            lr (float, optional): Learning rate
            weight_decay (float, optional): Weight decay a.k.a. regularization
                term
            loss_function (Literal["bce", "irlbl_bce", "focal", "db", "mse",
                "mc"], optional):
                - `bce`: Binary cross-entropy
                - `irlbl_bce`: Binary cross-entropy weighted by IRLbl scores,
                    see `acc23.mlsmote.irlbl`
                - `focal`: Focal loss, see
                  `acc23.models.base_mlc.focal_loss_with_logits`
                - `db`: Distribution-balanced loss, see
                  `acc23.models.base_mlc.distribution_balanced_loss_with_logits`
                - `mse`: Mean squared error
                - `mc`: Max constraint loss, see
                  `acc23.models.base_mlc.mc_loss`
            swa_lr (float, optional): Learning rate for stochastic
                weight averaging. If left to `None`, this method is not
                applied.
            swa_epoch (Union[int, float], optional): Number of epochs
                to skip before applying stochastic weight averaging.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def configure_optimizers(self) -> Any:
        """
        Configures an Adam optimizer with a scheduler that reduces the learning
        rate when the validation loss plateaus
        """
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
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
        elif lfn == "mc":
            loss = mc_loss(y_pred.sigmoid(), y_true)
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
        """
        Logs `NaN` to validation metrics (see
        `acc23.models.base_mlc.evaluate`). This allows to cleanly log
        hyperparameters to tensorboard.

        See also:
            https://lightning.ai/docs/pytorch/latest/extensions/logging.html#logging-hyperparameters
        """
        keys = ["val/loss", "val/ham", "val/f1", "val/prec", "val/rec"]
        self.logger.log_hyperparams(self.hparams, {k: np.nan for k in keys})

    def training_step(self, batch, *_, **__):
        """Override"""
        x, y, img = batch
        return self.evaluate(x, y, img, "train")

    def validation_step(self, batch, *_, **__):
        """Override"""
        x, y, img = batch
        return self.evaluate(x, y, img, "val")

    def test_step(self, batch, *_, **__):
        """Override"""
        x, y, img = batch
        return self.evaluate(x, y, img, "test")

    def predict_step(self, batch, *_, **__):
        """Override"""
        x, _, img = batch
        return self(x, img)


class ModuleWeightsHistogram(pl.Callback):
    """
    A callback to log a tensorboard histogram of the distribution of a module
    weights during training
    """

    _every_n_epochs: int
    _key: str
    _warned_wrong_logger: bool = False

    def __init__(
        self,
        every_n_epochs: int = 5,
        key: str = "train/weights",
    ) -> None:
        """
        Args:
            every_n_epochs (int, optional):
            key (str, optional): Key under which the histogram will be logged
                in tensorboard
        """
        super().__init__()
        self._every_n_epochs = every_n_epochs
        self._key = key

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Override"""
        if trainer.current_epoch % self._every_n_epochs != 0:
            return
        ps = [p.flatten() for p in pl_module.parameters()]
        if isinstance(trainer.logger, pl.loggers.TensorBoardLogger):
            trainer.logger.experiment.add_histogram(
                self._key,
                torch.concat(ps),
                bins="auto",
                global_step=trainer.global_step,
            )
        elif not self._warned_wrong_logger:
            logging.warning(
                "ModuleWeightsHistogram callback: Trainer's logger is has "
                f"type '{type(trainer.logger)}', but a tensorboard logger is "
                "required. This warning will only be logged once"
            )
            self._warned_wrong_logger = True


def bp_mll_loss(y_pred: Tensor, y_true: Tensor, *_, **__) -> Tensor:
    """
    BM-MLL loss introduced in

        Zhang, Min-Ling, and Zhi-Hua Zhou. "Multilabel neural networks with
        applications to functional genomics and text categorization." IEEE
        transactions on Knowledge and Data Engineering 18.10 (2006): 1338-1351.

    `y_pred` is expected to contain class probabilities, **not** logits.
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


def to_hierarchical_logits(
    y: Tensor, mode: Literal["min", "max", "prod"] = "max"
) -> Tensor:
    """
    Given a `(N, N_TRUE_TARGETS)` tensor of true target logits, corrects some
    of them based on the inherent hierarchy of the targets. The parent/child
    (aka. superclass/subclass) relationship is as follows:
    0. `Allergy_Present` parent of targets 1, 2, 3, 4
    1. `Severe_Allergy`
    2. `Respiratory_Allergy` parent of targets 5, 6, 7, 8, 9, 10, 11, 12,
        13
    3. `Food_Allergy` parent of targets 14, 15, 16, 17, 18, 19, 20, 21, 22,
        23, 24
    4. `Venom_Allergy` parent of targets 25, 26
    5. `Type_of_Respiratory_Allergy_ARIA`
    6. `Type_of_Respiratory_Allergy_CONJ`
    7. `Type_of_Respiratory_Allergy_GINA`
    8. `Type_of_Respiratory_Allergy_IGE_Pollen_Gram`
    9. `Type_of_Respiratory_Allergy_IGE_Pollen_Herb`
    10. `Type_of_Respiratory_Allergy_IGE_Pollen_Tree`
    11. `Type_of_Respiratory_Allergy_IGE_Dander_Animals`
    12. `Type_of_Respiratory_Allergy_IGE_Mite_Cockroach`
    13. `Type_of_Respiratory_Allergy_IGE_Molds_Yeast`
    14. `Type_of_Food_Allergy_Aromatics`
    15. `Type_of_Food_Allergy_Egg`
    16. `Type_of_Food_Allergy_Fish`
    17. `Type_of_Food_Allergy_Fruits_and_Vegetables`
    18. `Type_of_Food_Allergy_Mammalian_Milk`
    19. `Type_of_Food_Allergy_Oral_Syndrom`
    20. `Type_of_Food_Allergy_Other_Legumes`
    21. `Type_of_Food_Allergy_Peanut`
    22. `Type_of_Food_Allergy_Shellfish`
    23. `Type_of_Food_Allergy_TPO`
    24. `Type_of_Food_Allergy_Tree_Nuts`
    25. `Type_of_Venom_Allergy_ATCD_Venom`
    26. `Type_of_Venom_Allergy_IGE_Venom`

    The argument `mode` indicates how the logits are corrected.
    * If `mode` is `min`, child logits are capped by that of their parent:
        $$c' = \\mathrm{min} (c, p)$$ where where $p$ and $c$ are the logits
        of the parent and child target, and where $c'$ is the new child
        logit.
    * If `mode` is `max`, parent logits are corrected to be at least the
        max of their child: $$p' = \\mathrm{max} (p, c_1, c_2, \\ldots)$$
        where $p'$ is the new parent logit, and $c_1, c_2, \\ldots$ are the
        child logits.
    * If `mode` is `prod`, child logits are weighed by their parent using
        the following formula $$c' = \\sigma^{-1} ( \\sigma (p) \\sigma (c) )$$

    """

    def _mask(idxs: Iterable[int]) -> Tensor:
        """
        Produces a `(n_true_targets,)` binary tensor m, where `m_i` is 1 iff
        `i` is in `idx`. The tensor is cast to `float` and moved to the the
        same device as `y`
        """
        m = list(map(lambda i: int(i in idxs), range(n_true_targets)))
        return torch.tensor(m, dtype=torch.float32, device=y.device)

    # (idx of parent, idx of first child, idx of last child) For example (0, 1,
    # 4) means that target 0 (Allergy_Present) is the parent of targets 1 to 4
    # (Severe_Allergy, Respiratory_Allergy, Food_Allergy, and Venom_Allergy).
    # Children are always contiguous
    hierarchy = [  # topmost parent to bottommost children
        (0, 1, 4),
        (2, 5, 13),
        (3, 14, 24),
        (4, 25, 26),
    ]
    if mode == "max":  # need to start with the bottommost children
        hierarchy = hierarchy[::-1]
    n_true_targets = y.shape[-1]
    for p, c1, c2 in hierarchy:
        mc = _mask(range(c1, c2 + 1))  # mc_j = 1 if j child of p
        if mode == "min":
            a, b = y[:, [p]].repeat(1, n_true_targets), mc * y
            c = torch.minimum(a, b)
            y = c + (1 - mc) * y  # c is already masked
        elif mode == "max":
            mp = _mask([p])  # mp_j = 1 if j = p
            a, b = y[:, p], (mc * y).max(dim=1).values
            c = torch.maximum(a, b).unsqueeze(-1)
            y = mp * c + (1 - mp) * y
        elif mode == "prod":
            a, b = y[:, [p]].repeat(1, n_true_targets), mc * y
            c = -torch.log(torch.exp(-a) + torch.exp(-b) + torch.exp(-a - b))
            # c = \\sigma^{-1} (\\sigma (a) \\sigma (b))
            y = c + (1 - mc) * y  # c is already masked
        else:
            raise ValueError(
                f"Unsupported logit correction mode '{mode}'. Available modes "
                "are 'max', 'min', and 'prod'"
            )
    return y


def mc_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Max constraint loss of

        Giunchiglia, Eleonora, and Thomas Lukasiewicz. "Coherent hierarchical
        multi-label classification networks." Advances in neural information
        processing systems 33 (2020): 9662-9673.

    `y_pred` is expected to contain class probabilities, **not** logits.
    """

    def _mask(idxs: Iterable[int]) -> Tensor:
        """
        Produces a `(n_true_targets,)` binary tensor m, where `m_i` is 1 iff
        `i` is in `idx`. The tensor is cast to `float` and moved to the the
        same device as `y_pred`
        """
        m = list(map(lambda i: int(i in idxs), range(n_true_targets)))
        return torch.tensor(m, dtype=torch.float32, device=y_pred.device)

    n_true_targets = y_pred.shape[-1]
    hierarchy = [
        (0, 1, 4),
        (2, 5, 13),
        (3, 14, 24),
        (4, 25, 26),
    ]
    mcm = torch.zeros_like(y_pred, device=y_pred.device)
    mhy = torch.zeros_like(y_pred, device=y_pred.device)
    for p, c1, c2 in hierarchy:
        mp = _mask([p])  # mp_j = 1 if j = p
        mc = _mask(range(c1, c2 + 1))  # mc_j = 1 if j child of p
        a = (y_pred * mc).max(dim=1).values
        # a_i = max (y_pred_ij) for j child of p
        b = (y_true * y_pred * mc).max(dim=1).values
        # b_i = max (y_true_ij * y_pred_ij) for j child of p
        mcm = mcm + mp * a.unsqueeze(-1)
        mhy = mhy + mp * b.unsqueeze(-1)
    return -torch.where(
        y_true == 1, torch.log(mhy + 1e-5), torch.log(1 - mcm + 1e-5)
    ).mean()
