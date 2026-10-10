"""Every setting in one place. SHARED is used identically by backprop and predictive coding.

The optimiser, output activation and loss are fixed in models.py; they are listed here so that they are
saved with the results.
"""

DATASETS = {
    "adult": {"hidden": [64, 32], "epochs": 20, "lr": 0.2},
    "mnist": {"hidden": [256, 128], "epochs": 20, "lr": 0.5},
}

SHARED = {
    "batch_size": 64,
    "optimizer": "SGD",
    "hidden_activation": "relu",
    "output_activation": "linear",
    "loss": "1/2 squared error",
    "seeds": [0, 1, 2],
}

PC = {"eta": 0.1, "T": 20, "mode": "strict"}

LR_GRID = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
TUNE_EPOCHS = 5
