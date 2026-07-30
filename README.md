# Multilayer Perceptron

MLP classifier built from scratch in NumPy, trained on the Wisconsin Breast Cancer dataset to predict malignant vs. benign tumors.

## Architecture overview

```
split_data.py
    └─> datasets/data_train.csv, data_val.csv

train.py
    ├── MLPDataLoader   (network.py)   — loads CSVs, normalizes, shuffles per epoch
    ├── MLP             (network.py)   — builds layers, runs training loop
    │     ├── Layer     (network.py)   — weight + bias + activation tag
    │     ├── Value     (autograd.py)  — wraps numpy arrays, records ops, runs .backward()
    │     ├── softmax / ReLU (ft_function.py)
    │     ├── loss      (ft_function.py) — binary or categorical cross entropy
    │     ├── optimizer (optimizer.py) — Adam or NesterovMomentum (optional)
    │     └── EarlyStopping (early_stopping.py)
    └─> artefacts.npz  (weights, biases, topology, normalization stats)

predict.py
    └── loads artefacts.npz, applies same normalization, runs feed_forward_no_grad
```

Data flows one way: `split_data.py` → `train.py` → `artefacts.npz` → `predict.py`.

The autograd engine (`Value`) wraps NumPy arrays (not scalars), so all operations — add, mul, matmul, pow, exp, log, sum, mean — work on full batches. Gradients accumulate via a topological sort of the computation graph on each `.backward()` call.

## Install

```bash
uv pip install -r requirements.txt
```

Python 3.12 assumed (that's what the `__pycache__` bytecode targets).

## Usage

### 1. Prepare the dataset

```bash
uv run split_data.py
```

Reads `datasets/data.csv` (Wisconsin Breast Cancer, no header), drops the ID column, one-hot encodes the M/B label, shuffles with seed 42, and writes an 80/20 train/val split to `datasets/data_train.csv` and `datasets/data_val.csv`.

### 2. Train

```bash
uv run train.py --layer <neurons...> --epochs <n> --loss <fn> --batch_size <n> --learning_rate <lr> [--optimizer <opt>] [--early_stopping <patience>]
```

**Required arguments**

| Argument | Values | Description |
|---|---|---|
| `--layer` | one or more ints | Neurons in each hidden layer |
| `--epochs` | int | Max training epochs |
| `--loss` | `binaryCrossentropy` \| `categoricalCrossentropy` | Loss function |
| `--batch_size` | int | Mini-batch size |
| `--learning_rate` | float | Learning rate |

**Optional arguments**

| Argument | Values | Description |
|---|---|---|
| `--optimizer` | `Adam` \| `Nesterov` | Optimizer (default: vanilla SGD) |
| `--early_stopping` | int | Stop after this many epochs without val_loss improvement |

**Examples**

```bash
uv run train.py --layer 24 24 24 --epochs 84 --loss categoricalCrossentropy --batch_size 8 --learning_rate 0.0314

uv run train.py --layer 16 8 8 --epochs 130 --loss binaryCrossentropy --batch_size 8 --learning_rate 0.01 --early_stopping 100
```

Training prints per-epoch metrics and saves:
- `artefacts.npz` — serialized model (weights, biases, topology, normalization stats, loss function name)
- `graphs/graph.png` — training vs. validation loss
- `graphs/graph2.png` — training vs. validation accuracy

### 3. Predict

```bash
uv run predict.py <csv_file>
```

If the CSV has fewer than 31 columns (raw features only), it prints the softmax probability vector for each sample.

If it has 31+ columns (features + one-hot labels), it prints the loss over the full dataset.

The normalization (mean/std computed from the training set) is read from `artefacts.npz` and applied automatically.

## Project structure

```
autograd.py        — Value class: numpy-backed autograd engine with full op support
network.py         — Layer, MLP (training loop, feed_forward, export), MLPDataLoader
ft_function.py     — Loss functions and activations (grad-tracked and plain numpy variants)
optimizer.py       — Adam and NesterovMomentum
early_stopping.py  — EarlyStopping with patience, min_delta, min/max mode
train.py           — CLI entry point for training
predict.py         — CLI entry point for inference
split_data.py      — Preprocesses raw data.csv into train/val CSVs
datasets/          — data.csv (raw), data_train.csv, data_val.csv
graphs/            — Loss and accuracy plots saved after training
artefacts.npz      — Serialized model produced by train.py, consumed by predict.py
requirements.txt   — numpy, pandas, matplotlib
```
