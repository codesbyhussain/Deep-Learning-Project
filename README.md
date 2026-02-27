# MultiROCKET ECG Classification Ablation Study

Reproducible notebook-driven ablation study comparing autoencoder vs structured pooling reduction, and MLP vs FT-Transformer classifiers on MultiROCKET-extracted ECG features.

**Collaborative project** — this repo is set up for 3 people. Use Poetry for a consistent environment; reports are gitignored (share via your preferred channel). Track progress with **[ANALYSIS_ROADMAP.md](ANALYSIS_ROADMAP.md)** (2×2 ablation checklist).

## Setup (Poetry)

Install [Poetry](https://python-poetry.org/docs/#installation), then from the project root:

```bash
poetry install
poetry shell
```

To add the optional MultiROCKET backend (sktime):

```bash
poetry add sktime
# or: poetry install --extras sktime  (if sktime is listed under [tool.poetry.extras])
```

**Without Poetry:** export a lockfile-based requirements file and use pip:

```bash
poetry export -f requirements.txt --without-hashes -o requirements.txt
pip install -r requirements.txt
```

## Raw Data

Place your raw ECG dataset in:

```
data/raw/
```

Do **not** commit large raw data files. Implement the dataset loader in one place:

- **`src/data/io.py`** → `load_raw_dataset()`  
  Replace the placeholder (which raises `NotImplementedError`) with your logic to load ECG time series and labels. The docstring in that function describes the expected return format.

## Running an Experiment

Each ablation condition has a YAML config under `experiments/`:

- **A1**: autoencoder + MLP  
- **A2**: autoencoder + FT-Transformer  
- **B1**: structured pooling + MLP  
- **B2**: structured pooling + FT-Transformer  

Run training (full pipeline: features → reduction → classifier). From the project root, with the Poetry env active:

```bash
python -m src.training.train --config experiments/A1_autoencoder_mlp/config.yaml
```

Other examples:

```bash
python -m src.training.train --config experiments/B1_pooling_mlp/config.yaml
python -m src.training.train --config experiments/B2_pooling_ft/config.yaml
```

The script will fail at `load_raw_dataset()` until you implement it. Artifacts (splits, MultiROCKET transformer, scaler, memmaps, checkpoints, metrics, confusion matrices) are written under `experiments/<condition>/` and shared model dirs under `models/`.

## Notebook Workflow

1. **00_data_inspection.ipynb** – Inspect raw data (call `load_raw_dataset()` after implementing it).
2. **01_preprocessing.ipynb** – Preprocessing and split creation.
3. **02_multirocket.ipynb** – MultiROCKET feature extraction and persistence.
4. **03_autoencoder.ipynb** – Autoencoder training and reduced features.
5. **04_structured_pooling.ipynb** – Structured pooling reduction.
6. **05_mlp_models.ipynb** – MLP classifier training/evaluation.
7. **06_ft_transformer.ipynb** – FT-Transformer training/evaluation.

Notebooks import from `src`; keep core logic in `src/` and use notebooks for exploration and visualization. Use the same Poetry environment for Jupyter (`poetry run jupyter notebook` or select the Poetry venv as the kernel).

## Reports

The `reports/` directory is in `.gitignore`. Share proposal, preliminary, and final reports (e.g. `reports/proposal/`, `reports/preliminary/`, `reports/final/`) via your team’s chosen channel (Drive, OneDrive, etc.).
