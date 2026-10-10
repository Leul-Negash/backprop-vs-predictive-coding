"""Extra: how close is the PC weight update to the backprop gradient?

strict = the slides' PC: predictions are recomputed from the updated states at every inference step.
fixed  = the Fixed Prediction Assumption (Millidge et al., 2020): predictions stay at their forward values.
"""
import json
import os

import torch

from config import PC
from data import load
from models import BPNet, PCNet, make_layers

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def cosine(pc, x, y, T):
    state = pc.infer(x, y, T=T)
    pc_grads = torch.cat([g.flatten() for pair in pc.weight_gradients(state) for g in pair])
    bp = BPNet(pc.layers)
    for p in bp.parameters():
        p.requires_grad_(True)
    bp.loss(x, y).backward()
    bp_grads = torch.cat([p.grad.flatten() for p in bp.parameters()])
    return torch.nn.functional.cosine_similarity(pc_grads, bp_grads, dim=0).item()


def main():
    d = load("mnist")
    x, y = d["train"][0][:256], d["one_hot"](d["train"][1][:256])
    out = {}

    print("=" * 78)
    print("A. cosine(PC update, BP gradient) vs number of inference steps (MNIST 784-256-128-10)")
    print("=" * 78)
    sizes = [784, 256, 128, 10]
    print(f"  {'T':<8}{'strict':>10}{'fixed':>10}")
    out["steps"] = {}
    for T in [1, 5, 20, 100, 500]:
        row = {mode: cosine(PCNet(make_layers(sizes, 0), PC["eta"], mode=mode), x, y, T)
               for mode in ["strict", "fixed"]}
        out["steps"][T] = row
        print(f"  {T:<8}{row['strict']:>10.4f}{row['fixed']:>10.4f}", flush=True)

    print("\n" + "=" * 78)
    print("B. The same after 500 steps, for deeper networks (hidden layers of 128, mean of 3 seeds)")
    print("=" * 78)
    print(f"  {'hidden layers':<16}{'strict':>10}{'fixed':>10}")
    out["depth"] = {}
    for depth in [1, 2, 4, 6, 8]:
        sizes = [784] + [128] * depth + [10]
        row = {mode: sum(cosine(PCNet(make_layers(sizes, seed), PC["eta"], mode=mode), x, y, 500)
                         for seed in range(3)) / 3
               for mode in ["strict", "fixed"]}
        out["depth"][depth] = row
        print(f"  {depth:<16}{row['strict']:>10.4f}{row['fixed']:>10.4f}", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "alignment.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
