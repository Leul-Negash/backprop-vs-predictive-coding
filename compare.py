"""Backprop vs predictive coding with the same data, architecture, initial weights, batches and optimiser."""
import argparse
import json
import os
import time

import torch

from config import DATASETS, PC, SHARED
from data import batches, load
from models import BPNet, PCNet, count_parameters, make_layers

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
NODE_NAMES = ["HL-1", "HL-2", "output"]


def accuracy(net, x, y):
    with torch.no_grad():
        return (net(x).argmax(dim=1) == y).float().mean().item()


def balanced_accuracy(net, x, y, k):
    with torch.no_grad():
        pred = net(x).argmax(dim=1)
    return sum((pred[y == c] == c).float().mean().item() for c in range(k)) / k


def output_loss(net, x, y_onehot):
    with torch.no_grad():
        return (0.5 * ((net(x) - y_onehot) ** 2).sum(dim=1)).mean().item()


def gradient_cosine(pc, x, y):
    """Cosine between the PC weight update and the backprop gradient, both at PC's current weights."""
    pc_grads = torch.cat([g.flatten() for pair in pc.weight_gradients(pc.infer(x, y)) for g in pair])
    bp = BPNet(pc.layers)
    for p in bp.parameters():
        p.requires_grad_(True)
    bp.loss(x, y).backward()
    bp_grads = torch.cat([p.grad.flatten() for p in bp.parameters()])
    return torch.nn.functional.cosine_similarity(pc_grads, bp_grads, dim=0).item()


def inference_time(net, x, repeats=5):
    """Median time to predict the whole test set (prediction = one forward pass for both learners)."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        with torch.no_grad():
            net(x).argmax(dim=1)
        times.append(time.perf_counter() - start)
    return sorted(times)[len(times) // 2]


def multiply_adds(sizes, T):
    """Multiply-adds per training example: BP = forward + backward; PC = forward init + T inference steps + update."""
    per_layer = [a * b for a, b in zip(sizes[:-1], sizes[1:])]
    forward = sum(per_layer)
    bp = forward + forward + sum(per_layer[1:])
    pc = forward + T * (forward + sum(per_layer[1:])) + forward + forward
    return forward, bp, pc


def run_seed(d, sizes, seed, epochs, lr):
    layers = make_layers(sizes, seed)
    nets = {"BP": BPNet(layers, SHARED["hidden_activation"]),
            "PC": PCNet(layers, PC["eta"], PC["T"], SHARED["hidden_activation"], PC["mode"])}
    optimizers = {m: torch.optim.SGD(net.parameters(), lr=lr) for m, net in nets.items()}
    one_hot = d["one_hot"]
    x_te, y_te = d["test"]
    x_val, y_val = d["val"]
    probe_x, probe_y = d["train"][0][:256], one_hot(d["train"][1][:256])
    history = {m: [] for m in nets}
    for epoch in range(1, epochs + 1):
        for method, net in nets.items():
            g = torch.Generator().manual_seed(1000 * seed + epoch)
            start = time.perf_counter()
            objective, energies, n = 0.0, [0.0] * (len(sizes) - 1), 0
            for x, y in batches(*d["train"], SHARED["batch_size"], g):
                objective += net.train_step(x, one_hot(y), optimizers[method])
                if method == "PC":
                    energies = [a + b for a, b in zip(energies, net.last_energies)]
                n += 1
            seconds = time.perf_counter() - start
            row = {"epoch": epoch, "seconds": seconds, "objective": objective / n,
                   "train_output_loss": output_loss(net, probe_x, probe_y),
                   "val_acc": accuracy(net, x_val, y_val), "test_acc": accuracy(net, x_te, y_te)}
            if method == "PC":
                row["layer_energy"] = [e / n for e in energies]
                row["cos_bp"] = gradient_cosine(net, probe_x, probe_y)
            history[method].append(row)
        bp_row, pc_row = history["BP"][-1], history["PC"][-1]
        print(f"  seed {seed} epoch {epoch:2d}  BP test {bp_row['test_acc']:.4f} ({bp_row['seconds']:.1f}s)  "
              f"PC test {pc_row['test_acc']:.4f} ({pc_row['seconds']:.1f}s)  PC energy "
              + " ".join(f"{name} {e:.4f}" for name, e in zip(NODE_NAMES, pc_row["layer_energy"]))
              + f"  cos(PC,BP) {pc_row['cos_bp']:.3f}", flush=True)
    final = {}
    for method, net in nets.items():
        final[method] = {"test_acc": accuracy(net, x_te, y_te),
                         "balanced_acc": balanced_accuracy(net, x_te, y_te, d["n_out"]),
                         "inference_seconds": inference_time(net, x_te)}
    return history, final


def summarise(name, d, sizes, runs):
    weights, biases, hidden_states, output_states = count_parameters(make_layers(sizes, 0))
    forward, bp_ops, pc_ops = multiply_adds(sizes, PC["T"])
    print(f"\n{name}: {' -> '.join(map(str, sizes))}, {len(runs)} seeds, mean (std)")
    print(f"{'':28}{'BP':>22}{'PC':>22}")

    def stat(method, key, scale=1.0, fmt=".2f"):
        values = torch.tensor([r["final"][method][key] * scale for r in runs])
        return f"{values.mean():{fmt}} ({values.std(unbiased=False):{fmt}})"

    def per_epoch(method):
        values = torch.tensor([sum(h["seconds"] for h in r["history"][method]) / len(r["history"][method])
                               for r in runs])
        return f"{values.mean():.2f} ({values.std(unbiased=False):.2f})"

    print(f"{'test accuracy (%)':<28}{stat('BP', 'test_acc', 100):>22}{stat('PC', 'test_acc', 100):>22}")
    print(f"{'balanced accuracy (%)':<28}{stat('BP', 'balanced_acc', 100):>22}"
          f"{stat('PC', 'balanced_acc', 100):>22}")
    print(f"{'training time / epoch (s)':<28}{per_epoch('BP'):>22}{per_epoch('PC'):>22}")
    print(f"{'test-set prediction (ms)':<28}{stat('BP', 'inference_seconds', 1000):>22}"
          f"{stat('PC', 'inference_seconds', 1000):>22}")
    print(f"{'learned parameters':<28}{weights + biases:>22,}{weights + biases:>22,}")
    states = f"{hidden_states} + {output_states}"
    print(f"{'latent states (hidden+out)':<28}{'0':>22}{states:>22}")
    print(f"{'multiply-adds per example':<28}{bp_ops:>22,}{pc_ops:>22,}")
    print(f"{'  (one forward pass)':<28}{forward:>22,}{forward:>22,}")
    last = [r["history"]["PC"][-1] for r in runs]
    print("PC layer-wise energy after inference, last epoch: "
          + "  ".join(f"{node} {sum(row['layer_energy'][i] for row in last) / len(last):.4f}"
                      for i, node in enumerate(NODE_NAMES))
          + f"  total {sum(sum(row['layer_energy']) for row in last) / len(last):.4f}")
    print(f"cosine(PC update, BP gradient), last epoch: {sum(row['cos_bp'] for row in last) / len(last):.4f}")
    print(f"test accuracy, both trained on {len(d['train'][0])} examples, tested on {len(d['test'][0])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=["adult", "mnist"])
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    for name in args.datasets:
        d = load(name)
        sizes = [d["n_in"], *DATASETS[name]["hidden"], d["n_out"]]
        path = os.path.join(RESULTS, f"compare_{name}.json")
        if os.path.exists(path) and not args.retrain:
            with open(path) as f:
                runs = json.load(f)["runs"]
            print(f"{name}: loaded cached runs from {os.path.relpath(path)}")
        else:
            print(f"{name}: shared settings {SHARED}, lr {DATASETS[name]['lr']}, PC {PC}")
            runs = []
            for seed in SHARED["seeds"]:
                history, final = run_seed(d, sizes, seed, DATASETS[name]["epochs"], DATASETS[name]["lr"])
                runs.append({"seed": seed, "history": history, "final": final})
            with open(path, "w") as f:
                json.dump({"sizes": sizes, "shared": SHARED, "pc": PC, "runs": runs}, f)
        summarise(name, d, sizes, runs)


if __name__ == "__main__":
    main()
