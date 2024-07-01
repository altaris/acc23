# pylint: disable=missing-function-docstring
# pylint: disable=wildcard-import
# pylint: disable=unused-wildcard-import
"""Hyperparameter tuning for prototype Norway"""


import pytorch_lightning as pl
from loguru import logger as logging
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from acc23 import *
from acc23.utils import train_model


def _train(hparams: dict):
    model = Orchid(**hparams)
    datamodule = ACCDataModule(
        train_csv_file_path="/home/cedric/repositories/acc23/data/train.csv",
        test_csv_file_path="/home/cedric/repositories/acc23/data/test.csv",
        image_dir_path="/home/cedric/repositories/acc23/data/images",
        data_cache_path="/home/cedric/repositories/acc23/out/data.cache",
    )
    model = train_model(
        model,
        datamodule,
        root_dir="out",
        early_stopping_kwargs={
            "check_finite": True,
            "mode": "min",
            "monitor": "val/loss",
            "patience": 10,
        },
        max_epochs=200,
    )
    _, metrics = eval_on_train_dataset(model, datamodule, "out/eval")
    return {"val/f1": metrics.describe().loc["mean"]["f1"]}


def main():
    metric, metric_mode = "val/f1", "max"
    max_epochs, num_samples = 50, 10
    hparams_config = {
        # "embed_dim": tune.choice([32, 128, 512]),
        # "patch_size": tune.choice([8]),
        # "n_transformers": tune.choice([8, 16, 32]),
        # "n_heads": tune.choice([8, 16]),
        "dropout": tune.choice([0.1, 0.2, 0.5]),
        # "mlp_dim": tune.choice([32, 512, 2048, 4096]),
        "freeze_vit": tune.choice([True, False]),
        "weight_decay": tune.choice([0.0, 1e-3, 5e-3]),
        "lr": tune.choice([1e-4, 5e-4, 1e-3]),
        "swa_lr": tune.choice([1e-4, 5e-4, 1e-3]),
        "swa_epoch": tune.choice([5, 10, 20])
    }
    scheduler = ASHAScheduler(
        max_t=max_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    reporter = CLIReporter(
        parameter_columns=list(hparams_config.keys()),
        metric_columns=[metric],
    )
    tuner = tune.Tuner(
        tune.with_resources(
            _train,
            resources={"cpu": 16, "gpu": 1},
        ),
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=metric_mode,
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            progress_reporter=reporter,
        ),
        param_space=hparams_config,
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric=metric, mode=metric_mode)
    print(best_result)


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    try:
        main()
    except:
        logging.exception("Oh no :(")
