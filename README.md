# [Allergy Chip Challenge 2023](https://app.trustii.io/datasets/1439)

![Python 3](https://img.shields.io/badge/python-3-blue?logo=python)
[![License](https://img.shields.io/badge/license-MIT-green)](https://choosealicense.com/licenses/mit/)
[![Code style](https://img.shields.io/badge/style-black-black)](https://pypi.org/project/black)

# Submitting via `acc23` CLI:

```sh
. ./secret.env && python3 -m acc23 submit -t "$TOKEN" out.csv dummy.ipynb
```

# Troubleshooting

## 2023-04-26 Image corruption

`data/images/CY60527_4_190006236104_2022_12_22_12_15_22.bmp` is corrupted? PIL
raises an `OSError` when loading...
`data/images/CY60527_4_190006236104_2022_12_22_12_11_20.bmp` appears similar,
so i just

```sh
cp data/images/CY60527_4_190006236104_2022_12_22_12_15_22.bmp data/images/CY60527_4_190006236104_2022_12_22_12_11_20.bmp
```

and called it a day


# Contributing

## Dependencies

* `python3.10` or newer;
* `requirements.txt` for runtime dependencies;
* `requirements.dev.txt` for development dependencies.
* `make` (optional);

Simply run
```sh
virtualenv venv -p python3.10
. ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements.dev.txt
```

## Documentation

Simply run
```sh
make docs
```
This will generate the HTML doc of the project, and the index file should be at
`docs/index.html`. To have it directly in your browser, run
```sh
make docs-browser
```

## Code quality

Don't forget to run
```sh
make
```
to format the code following [black](https://pypi.org/project/black/),
typecheck it using [mypy](http://mypy-lang.org/), and check it against coding
standards using [pylint](https://pylint.org/).
