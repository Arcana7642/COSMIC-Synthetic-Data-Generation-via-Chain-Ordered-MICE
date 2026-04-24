<<<<<<< HEAD
# COSMIC-Synthetic-Data-Generation-via-Chain-Ordered-MICE
AAAI paper
=======
# COSMIC

AAAI submission codebase scaffold for tabular synthesis experiments.

This project follows a single-entry CLI:

```bash
python main.py --dataname [NAME] --method [METHOD] --mode [train|sample]
```

Supported methods:

- `cosmic`
- `smote`
- `ctgan`
- `tvae`
- `stasy`
- `codi`
- `tabddpm`
- `tabsyn`
- `vae`

## COSMIC

`COSMIC` is scoped as a three-stage method:

1. Train a masked autoencoder on tabular training data.
2. Estimate column-wise SHAP values from the trained representation model or a surrogate.
3. Generate synthetic data with an order-aware MICE procedure that follows descending SHAP importance.

The current COSMIC scaffold lives in:

- `COSMIC.py`
- `cosmic/config.py`
- `cosmic/pipeline.py`

The actual MAE, SHAP scoring, and order-aware generator are still placeholders.

## Baselines

The following methods are treated as baselines:

- `SMOTE`
- `CTGAN`
- `TVAE`
- `STaSy`
- `CoDi`
- `TabDDPM`
- `TabSyn`

`CTGAN` and `TVAE` run directly inside this repository.

`SMOTE`, `STaSy`, `CoDi`, `TabDDPM`, `TabSyn`, and `VAE` pretraining code were copied from `tabsyn-main` and wired to the same CLI.

## RTX 5080 environment

For RTX 5080, use a PyTorch build with CUDA 12.8 or newer.

Recommended setup:

```bash
conda env create -f environment-cu128.yml
conda activate cosmic-cu128
```

Or with `pip` in a Python 3.11 environment:

```bash
pip install -r requirements-cu128.txt
```

Quick verification:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

## Dataset preprocessing

The baseline pipeline expects the same processed dataset layout as TabSyn.

Included utilities:

- `download_dataset.py`
- `process_dataset.py`
- `data/Info`

Expected layout after preprocessing:

```text
COSMIC/
  data/
    Info/
      mydata.json
    mydata/
      info.json
      train.csv
      test.csv
      X_num_train.npy
      X_num_test.npy
      X_cat_train.npy
      X_cat_test.npy
      y_train.npy
      y_test.npy
```

## Example commands

COSMIC placeholder:

```bash
python main.py --dataname adult --method cosmic --mode train
python main.py --dataname adult --method cosmic --mode sample
```

CTGAN / TVAE:

```bash
python main.py --dataname adult --method ctgan --mode train --epochs 300
python main.py --dataname adult --method ctgan --mode sample
python main.py --dataname adult --method tvae --mode train --epochs 300
python main.py --dataname adult --method tvae --mode sample
```

Other baselines:

```bash
python main.py --dataname adult --method smote --mode train
python main.py --dataname adult --method stasy --mode train
python main.py --dataname adult --method codi --mode train
python main.py --dataname adult --method tabddpm --mode train
python main.py --dataname adult --method vae --mode train
python main.py --dataname adult --method tabsyn --mode train
```

```bash
python main.py --dataname adult --method smote --mode sample
python main.py --dataname adult --method stasy --mode sample
python main.py --dataname adult --method codi --mode sample
python main.py --dataname adult --method tabddpm --mode sample --ddim --steps 50
python main.py --dataname adult --method tabsyn --mode sample
```

Preprocess a dataset:

```bash
python process_dataset.py --dataname adult
```
>>>>>>> 558e96d (initial commit)
