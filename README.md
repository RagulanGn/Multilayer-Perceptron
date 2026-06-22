# Multilayer Perceptron

MLP classifier built from scratch in NumPy, trained on the Wisconsin Breast Cancer dataset to predict malignant vs. benign tumors.

## Why it exists

No PyTorch, no TensorFlow — the goal is to implement the full forward/backward pass, including a custom autograd engine, optimizers, and training loop, using only NumPy. The dataset is a fixed binary classification problem (30 features → 2 classes) used to validate correctness.

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
pip install -r requirements.txt
```

Python 3.12 assumed (that's what the `__pycache__` bytecode targets).

## Usage

### 1. Prepare the dataset

```bash
python split_data.py
```

Reads `datasets/data.csv` (Wisconsin Breast Cancer, no header), drops the ID column, one-hot encodes the M/B label, shuffles with seed 42, and writes an 80/20 train/val split to `datasets/data_train.csv` and `datasets/data_val.csv`.

### 2. Train

```bash
python train.py --layer <neurons...> --epochs <n> --loss <fn> --batch_size <n> --learning_rate <lr> [--optimizer <opt>] [--early_stopping <patience>]
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

**Examples from code comments**

```bash
python train.py --layer 24 24 24 --epochs 84 --loss categoricalCrossentropy --batch_size 8 --learning_rate 0.0314

python train.py --layer 16 8 8 --epochs 130 --loss binaryCrossentropy --batch_size 8 --learning_rate 0.01 --early_stopping 100
```

Training prints per-epoch metrics and saves:
- `artefacts.npz` — serialized model (weights, biases, topology, normalization stats, loss function name)
- `graphs/graph.png` — training vs. validation loss
- `graphs/graph2.png` — training vs. validation accuracy

### 3. Predict

```bash
python predict.py <csv_file>
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

## Key design decisions

**Batch-native autograd.** Unlike scalar autograd engines (e.g. micrograd), `Value` wraps full NumPy arrays. This means a single matrix multiply in feed_forward covers the entire batch, and `_unbroadcast` handles gradient shape correction when broadcasting occurred during the forward pass.

**Architecture is fixed at 2 outputs.** `train.py` always appends `2` as the last layer size (`[features_len, *hidden_layers, 2]`), with softmax activation. The hidden layers use ReLU.

**Weight initialization.** Weights are drawn from `Normal(0, sqrt(2 / (in + out)))` — a Xavier/Glorot variant. Biases are zero-initialized.

**Normalization stored in artefacts.** The training set mean and std are saved into `artefacts.npz` so `predict.py` applies the identical transform without access to training data.

**Two forward paths.** `feed_forward` uses `Value` objects (grad tracking, used during training and validation loss). `feed_forward_no_grad` uses plain NumPy (used in `predict.py` — no graph built, no memory overhead).

**Loss function serialized by name.** `artefacts.npz` stores `loss_function` as the Python function object via `allow_pickle=True`. `predict.py` checks the string name (`'binaryCrossentropy'`) to dispatch to the correct no-grad variant.
