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

## Workflow

Use the commands in this order:

1. `download_dataset.py` downloads the raw dataset into `data/[NAME]`.
2. `process_dataset.py` converts the raw files into the processed NumPy/CSV layout expected by the baselines.
3. `--mode train` fits the selected method and writes model artifacts under `ckpt/[NAME]/[METHOD]` or the method-specific checkpoint directory.
4. `--mode sample` loads the trained artifacts and writes synthetic data under `synthetic/[NAME]/[METHOD].csv`, unless `--save_path` is provided.

`smote` does not train a neural checkpoint; its `train`/`sample` path directly produces synthetic data from the processed training split.
`vae` is the TabSyn pretraining step and supports `train` only; run `tabsyn --mode sample` to generate data after the VAE and TabSyn model are trained.

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

All baselines run inside this repository through the same CLI, dataset paths, checkpoints, and CUDA/CPU device selection.

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

## Docker

Build the image:

```bash
docker build -t cosmic-cu128 .
```

Run a quick CUDA check:

```bash
docker run --rm --gpus all -it cosmic-cu128 python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

Run commands with the repository mounted so datasets, checkpoints, and synthetic outputs stay on the host:

```bash
docker run --rm --gpus all -it -v "%cd%:/workspace" cosmic-cu128 python download_dataset.py --dataname adult
docker run --rm --gpus all -it -v "%cd%:/workspace" cosmic-cu128 python process_dataset.py --dataname adult
docker run --rm --gpus all -it -v "%cd%:/workspace" cosmic-cu128 python main.py --dataname adult --method ctgan --mode train --epochs 300
docker run --rm --gpus all -it -v "%cd%:/workspace" cosmic-cu128 python main.py --dataname adult --method ctgan --mode sample
```

For PowerShell, use `${PWD}` instead of `%cd%`:

```powershell
docker run --rm --gpus all -it -v "${PWD}:/workspace" cosmic-cu128 python download_dataset.py --dataname adult
docker run --rm --gpus all -it -v "${PWD}:/workspace" cosmic-cu128 python process_dataset.py --dataname adult
docker run --rm --gpus all -it -v "${PWD}:/workspace" cosmic-cu128 python main.py --dataname adult --method tabsyn --mode train
```

CPU-only runs can omit `--gpus all` and set `--gpu -1`:

```bash
docker run --rm -it -v "%cd%:/workspace" cosmic-cu128 python main.py --dataname adult --method smote --mode train --gpu -1
```

## Dataset preprocessing

The baseline pipeline expects the same processed dataset layout as TabSyn.
Download the raw dataset first, then run preprocessing.

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

Download and preprocess a dataset:

```bash
python download_dataset.py --dataname adult
python process_dataset.py --dataname adult
```

Omit `--dataname` to download or preprocess every supported dataset.
