"""How the PC inference phase behaves, and why eta = 0.1, T = 20 and forward initialisation were chosen."""
import json
import os
import sys
import time

import torch

from config import DATASETS, PC, SHARED
from data import batches, load
from models import PCNet, make_layers

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def trained_net(d, sizes, epochs, T, lr):
    pc = PCNet(make_layers(sizes, 0), PC["eta"], T, mode=PC["mode"])
    optimizer = torch.optim.SGD(pc.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    start = time.perf_counter()
    for _ in range(epochs):
        for x, y in batches(*d["train"], SHARED["batch_size"], g):
            pc.train_step(x, d["one_hot"](y), optimizer)
    return pc, (time.perf_counter() - start) / epochs


def main(name):
    d = load(name)
    sizes = [d["n_in"], *DATASETS[name]["hidden"], d["n_out"]]
    x, y = d["train"][0][:256], d["one_hot"](d["train"][1][:256])
    out = {}

    print("=" * 78)
    print(f"A. Energy of each layer during inference ({name}, untrained network, batch of 256)")
    print("=" * 78)
    pc = PCNet(make_layers(sizes, 0), PC["eta"], 50)
    for label, state in [("forward initialisation (z = mu, then clamp z_out = y)", pc.infer(x, y, record=True)),
                         ("zero initialisation (z = 0, then clamp z_out = y)", pc.infer(x, y, record=True, init="zero"))]:
        trace = state["trace"]
        out[label] = trace
        print(f"{label}")
        print(f"  {'step(T)':<9}{'HL-1':>11}{'HL-2':>11}{'output':>11}{'total':>11}")
        for t in [0, 1, 2, 5, 10, 20, 50]:
            print(f"  {t:<9}" + "".join(f"{e:>11.4f}" for e in trace[t]) + f"{sum(trace[t]):>11.4f}")
        print()

    print("=" * 78)
    print("B. Inference rate eta: total energy after 20 steps (lower is better)")
    print("=" * 78)
    out["eta"] = {}
    print(f"  {'eta':<9}{'start':>11}{'fixed eta':>13}{'capped eta':>13}")
    for eta in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        row = {}
        for label, cap in [("fixed", False), ("capped", True)]:
            trace = PCNet(make_layers(sizes, 0), eta, 20, cap_eta=cap).infer(x, y, record=True)["trace"]
            row[label] = sum(trace[-1])
        out["eta"][eta] = row
        shown = [f"{v:.4f}" if v < 1e3 else "diverged" for v in (row["fixed"], row["capped"])]
        print(f"  {eta:<9}{sum(trace[0]):>11.4f}{shown[0]:>13}{shown[1]:>13}")

    print("\n" + "=" * 78)
    print("C. Number of inference steps T: validation accuracy after 2 epochs and training time")
    print("=" * 78)
    out["T"] = {}
    print(f"  {'T':<9}{'val acc':>11}{'s / epoch':>11}")
    for T in [1, 2, 5, 10, 20, 50]:
        pc, seconds = trained_net(d, sizes, 2, T, DATASETS[name]["lr"])
        with torch.no_grad():
            acc = (pc(d["val"][0]).argmax(dim=1) == d["val"][1]).float().mean().item()
        out["T"][T] = {"val_acc": acc, "seconds_per_epoch": seconds}
        print(f"  {T:<9}{acc:>11.4f}{seconds:>11.2f}", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"inference_{name}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "adult")
