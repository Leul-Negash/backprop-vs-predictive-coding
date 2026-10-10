"""Chooses the learning rate on the validation split, with the same grid and budget for both learners."""
import json
import os
import sys

import torch

from config import DATASETS, LR_GRID, PC, SHARED, TUNE_EPOCHS
from data import batches, load
from models import BPNet, PCNet, make_layers

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def accuracy(net, x, y):
    with torch.no_grad():
        return (net(x).argmax(dim=1) == y).float().mean().item()


def main(name):
    d = load(name)
    sizes = [d["n_in"], *DATASETS[name]["hidden"], d["n_out"]]
    print(f"{name}: network {' -> '.join(map(str, sizes))}, {TUNE_EPOCHS} epochs per run, seed 0, "
          f"validation set of {len(d['val'][0])}")
    print(f"{'learning rate':>14}{'BP val acc':>14}{'PC val acc':>14}")
    table = {}
    for lr in LR_GRID:
        layers = make_layers(sizes, seed=0)
        accs = {}
        for method in ["BP", "PC"]:
            net = BPNet(layers) if method == "BP" else PCNet(layers, PC["eta"], PC["T"], mode=PC["mode"])
            optimizer = torch.optim.SGD(net.parameters(), lr=lr)
            g = torch.Generator().manual_seed(0)
            for _ in range(TUNE_EPOCHS):
                for x, y in batches(*d["train"], SHARED["batch_size"], g):
                    net.train_step(x, d["one_hot"](y), optimizer)
            accs[method] = accuracy(net, *d["val"])
        table[lr] = accs
        print(f"{lr:>14}{accs['BP']:>14.4f}{accs['PC']:>14.4f}", flush=True)
    best = {m: max(LR_GRID, key=lambda lr: table[lr][m]) for m in ["BP", "PC"]}
    print(f"best learning rate: BP {best['BP']}, PC {best['PC']}; both use BP's best, {best['BP']}")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"tuning_{name}.json"), "w") as f:
        json.dump({"grid": table, "best": best}, f, indent=2)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["adult", "mnist"]:
        main(name)
