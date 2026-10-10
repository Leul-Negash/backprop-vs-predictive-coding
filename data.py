"""Datasets: UCI Adult (tabular, like the loan example in the slides) and MNIST.

Both end up as float tensors in [0, 1] (min-max scaling, as in the slides) with one-hot targets,
because in both datasets the classes are nominal: they have no order.
"""
import gzip
import os
import urllib.request

import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
ADULT_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
                 "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week",
                 "native_country", "income"]
NUMERIC = ["age", "capital_gain", "capital_loss", "hours_per_week"]
SKEWED = ["capital_gain", "capital_loss"]
ORDINAL = {"education": ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad",
                         "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters", "Prof-school",
                         "Doctorate"]}
NOMINAL = ["workclass", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {"train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
               "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz"}


def _download(url, name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, path)
    return path


def _read_adult(name):
    rows = []
    with open(_download(ADULT_URL + name, name)) as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(ADULT_COLUMNS):
                continue
            row = dict(zip(ADULT_COLUMNS, parts))
            row["income"] = row["income"].rstrip(".")
            rows.append(row)
    return rows


def _encode_adult(rows, stats):
    """Turns one table into numbers with the encodings fitted on the training set."""
    columns = []
    for c in NUMERIC:
        v = torch.tensor([float(r[c]) for r in rows])
        if c in SKEWED:
            v = torch.log1p(v)
        lo, hi = stats[c]
        columns.append(((v - lo) / (hi - lo)).clamp(0, 1).unsqueeze(1))
    for c, order in ORDINAL.items():
        rank = torch.tensor([float(order.index(r[c])) for r in rows])
        columns.append((rank / (len(order) - 1)).unsqueeze(1))
    for c in NOMINAL:
        cats = stats[c]
        one_hot = torch.zeros(len(rows), len(cats))
        for i, r in enumerate(rows):
            value = r[c] if r[c] != "?" else "Unknown"
            if value in cats:
                one_hot[i, cats.index(value)] = 1.0
        columns.append(one_hot)
    x = torch.cat(columns, dim=1)
    y = torch.tensor([1 if r["income"] == ">50K" else 0 for r in rows])
    return x, y


def load_adult():
    train_rows, test_rows = _read_adult("adult.data"), _read_adult("adult.test")
    stats = {}
    for c in NUMERIC:
        v = torch.tensor([float(r[c]) for r in train_rows])
        if c in SKEWED:
            v = torch.log1p(v)
        stats[c] = (float(v.min()), float(v.max()))
    for c in NOMINAL:
        stats[c] = sorted({r[c] if r[c] != "?" else "Unknown" for r in train_rows})
    x_tr, y_tr = _encode_adult(train_rows, stats)
    x_te, y_te = _encode_adult(test_rows, stats)
    return x_tr, y_tr, x_te, y_te, ["<=50K", ">50K"]


def _read_mnist(key):
    with gzip.open(_download(MNIST_URL + MNIST_FILES[key], MNIST_FILES[key]), "rb") as f:
        raw = f.read()
    if key.endswith("_x"):
        return torch.frombuffer(bytearray(raw[16:]), dtype=torch.uint8).reshape(-1, 784).float() / 255.0
    return torch.frombuffer(bytearray(raw[8:]), dtype=torch.uint8).long()


def load_mnist():
    return _read_mnist("train_x"), _read_mnist("train_y"), _read_mnist("test_x"), _read_mnist("test_y"), \
        [str(d) for d in range(10)]


def load(name, val_fraction=0.1, seed=0):
    """Returns a dict with train / validation / test splits and one-hot targets."""
    x_tr, y_tr, x_te, y_te, classes = load_adult() if name == "adult" else load_mnist()
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(x_tr), generator=g)
    n_val = int(val_fraction * len(x_tr))
    val, train = order[:n_val], order[n_val:]
    k = len(classes)
    return {
        "name": name, "classes": classes, "n_in": x_tr.shape[1], "n_out": k,
        "train": (x_tr[train], y_tr[train]), "val": (x_tr[val], y_tr[val]), "test": (x_te, y_te),
        "one_hot": lambda y: torch.nn.functional.one_hot(y, k).float(),
    }


def batches(x, y, batch_size, generator):
    """Shuffled mini-batches; the same generator seed gives BP and PC the same batch order."""
    order = torch.randperm(len(x), generator=generator)
    for i in range(0, len(x), batch_size):
        idx = order[i:i + batch_size]
        yield x[idx], y[idx]


if __name__ == "__main__":
    for name in ["adult", "mnist"]:
        d = load(name)
        x, y = d["train"]
        print(name, "inputs", d["n_in"], "classes", d["n_out"], "train", len(x), "val", len(d["val"][0]),
              "test", len(d["test"][0]), "range", float(x.min()), float(x.max()),
              "class counts", torch.bincount(y).tolist())
