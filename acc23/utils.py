"""General utilities"""

import os
from pathlib import Path
from typing import Any, List, Optional, Type, Union

import pytorch_lightning as pl
import torch
from loguru import logger as logging
from pytorch_lightning.strategies import Strategy
from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def _load_from_checkpoint(
    cls: Type[pl.LightningModule], path: Union[str, Path], **kwargs: Any
) -> pl.LightningModule:
    """
    Convenience function to load a module checkpoint only in rank 0. In
    other ranks, the wrapped function returns `None`
    """
    logging.debug("Loading checkpoint '{}'", path)
    return cls.load_from_checkpoint(path, **kwargs)


def train_model(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    root_dir: Union[str, Path],
    name: Optional[str] = None,
    max_epochs: int = 512,
    additional_callbacks: Optional[List[pl.Callback]] = None,
    early_stopping_kwargs: Optional[dict] = None,
    strategy: Union[str, Strategy] = "ddp",
    **kwargs,
) -> pl.LightningModule:
    """
    Convenience function to invoke Pytorch Lightning's trainer fit. Returns the
    best checkpoint. The accelerator is set to `auto` unless otherwise directed
    by the `PL_ACCELERATOR` environment variable. If the accelerator ends up
    using a CUDA `gpu`, the trainer uses a
    [`DDPStrategy`](https://pytorch-lightning.readthedocs.io/en/latest/api/lightning.pytorch.strategies.DDPStrategy.html).

    Args:
        model (pl.LightningModule): The model to train. In its
            `validation_step`, the model must log the `val/loss` metric.
        datamodule (pl.LightningDataModule):
        root_dir (Union[str, Path]): The root dir of the trainer. The
            tensorboard logs will be stored under `root_dir/tb_logs/name` and
            the CSV logs under `root_dir/csv_logs/name`.
        name (str, optional): The name of the model. The
            tensorboard logs will be stored under `root_dir/tb_logs/name`.
        max_epochs (int): The maximum number of epochs. Note that an early
            stopping callbacks with a patience of 10 monitors the `val/loss`
            metric by default.
        additional_callbacks (List[pl.Callback], optional): Additional
            trainer callbacks. Note that the following callbacks are
            automatically set:
            ```py
            pl.callbacks.EarlyStopping(monitor="val/loss", patience=10),
            pl.callbacks.LearningRateMonitor("epoch"),
            pl.callbacks.ModelCheckpoint(save_weights_only=True),
            pl.callbacks.RichProgressBar(),
            ```
        early_stopping_kwargs (dict, optional): kwargs for the [`pl.callbacks.EarlyStopping`](https://pytorch-lightning.readthedocs.io/en/latest/api/lightning.pytorch.callbacks.EarlyStopping.html)
            callback. By default, it is
            ```py
            {
                monitor="val/loss",
                patience=10,
            }
            ```
        strategy (Union[str, Strategy]): Strategy to use (duh).
        **kwargs: Forwarded to the [`pl.Trainer`
            constructor](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#init).
    """

    if not isinstance(root_dir, Path):
        root_dir = Path(root_dir)

    name = name or model.__class__.__name__.lower()
    logging.info("Training model '{}' in '{}'", name, root_dir)

    accelerator = os.getenv("PL_ACCELERATOR", "auto").lower()
    logging.debug("Set accelerator to '{}'", accelerator)
    if accelerator in ["auto", "gpu"] and torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")  # hehe matmul go brrrr

    # https://stackoverflow.com/questions/48250053/pytorchs-dataloader-too-many-open-files-error-when-no-files-should-be-open
    # https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy("file_system")

    if not additional_callbacks:
        additional_callbacks = []
    if not early_stopping_kwargs:
        early_stopping_kwargs = {
            "check_finite": True,
            "mode": "min",
            "monitor": "val/loss",
            "patience": 10,
        }

    tb_logger = pl.loggers.TensorBoardLogger(
        str(root_dir / "tb_logs"),
        name=name,
        default_hp_metric=False,
        log_graph=True,
    )
    csv_logger = pl.loggers.CSVLogger(
        str(root_dir / "csv_logs"),
        name=name,
    )

    if (
        model.hparams.get("swa_lr") is not None
        and model.hparams.get("swa_epoch") is not None
    ):
        additional_callbacks.append(
            pl.callbacks.StochasticWeightAveraging(
                swa_lrs=model.hparams["swa_lr"],
                swa_epoch_start=model.hparams["swa_epoch"],
            ),
        )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[
            pl.callbacks.EarlyStopping(**early_stopping_kwargs),
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
            pl.callbacks.ModelCheckpoint(
                mode=early_stopping_kwargs.get("mode", "min"),
                monitor=early_stopping_kwargs["monitor"],
                save_weights_only=True,
            ),
            pl.callbacks.RichProgressBar(),
            *additional_callbacks,
        ],
        default_root_dir=str(root_dir),
        logger=[tb_logger, csv_logger],
        accelerator=accelerator,
        strategy=strategy,
        **kwargs,
    )

    trainer.fit(model, datamodule=datamodule)

    ckpt = str(trainer.checkpoint_callback.best_model_path)  # type: ignore
    return _load_from_checkpoint(type(model), ckpt) or model
