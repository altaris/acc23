# [Allergy Chip Challenge 2023](https://app.trustii.io/datasets/1439)

![Python 3](https://img.shields.io/badge/python-3-blue?logo=python)
[![License](https://img.shields.io/badge/license-MIT-green)](https://choosealicense.com/licenses/mit/)
[![Code style](https://img.shields.io/badge/style-black-black)](https://pypi.org/project/black)

## Minimal example

```py
from acc23 import models, ACCDataModule
from acc23.utils import train_model

# 1. Choose a model
model = models.Orchid()

# 2. Construct a datamodule from the competition data
datamodule = ACCDataModule("data/train.csv", "data/test.csv", "data/images")

# 3. Train the model. This can of course be done directly with pytorch
# lightning's API, or even a classic pytorch training loop
model = train_model(model, datamodule, root_dir="out")

# 4. Evaluate the model on the test dataset. The output file can readily
# be submitted to trustii.io!
df = eval_on_test_dataset(model, datamodule, root_dir="out/eval/test")
df.to_csv(f"out/predictions.csv", index=False)
```

## Package organization

The important user-facing modules are:

- `acc23.models`: Subpackage that contains all model definitions
- `acc23.dataset`: Submodule that defines `acc23.dataset.ACCDataModule`, which
  is a [pytorch lightning
  datamodule](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningDataModule.html)
  that takes care of importing and preprocessing the challenge data.
- `acc23.postprocessing`: Contains everything pertaining to postprocessing,
  i.e. going from raw model outputs to clean prediction CSV files. Important
  methods are `acc23.postprocessing.eval_on_test_dataset` and
  `acc23.postprocessing.eval_on_train_dataset`.
- `acc23.explain`: Everything related to explainability of the model's
  predictions. Important members are `acc23.explain.VitExplainer` which
  produces attention maps for vision transformers, and `acc23.explain.shap`,
  which approximates SHAP values in a model-agnostic way.
- `acc23.utils`: `acc23.utils.train_model` and other miscellaneous stuff.

The following modules also exist but the user shouldn't need to use them
directly

- `acc23.preprocessing`: Contains everything pertaining to preprocessing. Used
  by `acc23.dataset.ACCDataModule`.
- `acc23.constants`: Constants about the dataset, e.g. the number of features
  or the name of the target columns.
- `acc23.mlsmote`: Implementation of the MLSMOTE dataset augmentation
  algorithm. Part of the preprocessing pipeline.

## Submitting via `acc23` CLI:

```sh
. ./secret.env && python3 -m acc23 submit -t "$TOKEN" out.csv dummy.ipynb
```

## Troubleshooting

### 2023-04-26 Image corruption

`data/images/CY60527_4_190006236104_2022_12_22_12_15_22.bmp` is corrupted? PIL
raises an `OSError` when loading...
`data/images/CY60527_4_190006236104_2022_12_22_12_11_20.bmp` appears similar,
so i just

```sh
cp data/images/CY60527_4_190006236104_2022_12_22_12_15_22.bmp data/images/CY60527_4_190006236104_2022_12_22_12_11_20.bmp
```

and called it a day

## Contributing

### Dependencies

- `python3.10` or newer;
- `requirements.txt` for runtime dependencies;
- `requirements.dev.txt` for development dependencies.
- `make` (optional);

Simply run

```sh
virtualenv venv -p python3.10
. ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements.dev.txt
```

### Documentation

Simply run

```sh
make docs
```

This will generate the HTML doc of the project, and the index file should be at
`docs/index.html`. To have it directly in your browser, run

```sh
make docs-browser
```

### Code quality

Don't forget to run

```sh
make
```

to format the code following [black](https://pypi.org/project/black/),
typecheck it using [mypy](http://mypy-lang.org/), and check it against coding
standards using [pylint](https://pylint.org/).
