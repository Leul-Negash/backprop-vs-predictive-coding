"""Draws the figures in results/ from the saved JSON files."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
COLORS = {"BP": "#1f4e79", "PC": "#c0504d"}
NODES = ["HL-1", "HL-2", "output"]


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, name), dpi=140)
    plt.close(fig)
    print("saved", os.path.join("results", name))


def accuracy_and_energy():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for col, name in enumerate(["adult", "mnist"]):
        runs = load(f"compare_{name}.json")["runs"]
        ax = axes[0, col]
        for method in ["BP", "PC"]:
            acc = np.array([[100 * h["test_acc"] for h in r["history"][method]] for r in runs])
            epochs = np.arange(1, acc.shape[1] + 1)
            ax.plot(epochs, acc.mean(0), color=COLORS[method], lw=2, label=method)
            ax.fill_between(epochs, acc.mean(0) - acc.std(0), acc.mean(0) + acc.std(0), color=COLORS[method],
                            alpha=0.2)
        ax.set_title(f"{name}: test accuracy (mean of {len(runs)} seeds, band = std)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("test accuracy (%)")
        ax.grid(alpha=0.3)
        ax.legend()
        ax = axes[1, col]
        energy = np.array([[h["layer_energy"] for h in r["history"]["PC"]] for r in runs]).mean(0)
        for i, node in enumerate(NODES):
            ax.plot(epochs, energy[:, i], lw=2, label=f"E {node}")
        ax.plot(epochs, energy.sum(1), "k--", lw=1.5, label="E total")
        ax.set_yscale("log")
        ax.set_title(f"{name}: PC energy per layer after inference (training mean)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("energy per example")
        ax.grid(alpha=0.3)
        ax.legend()
    save(fig, "accuracy_and_energy.png")


def inference_energy():
    data = load("inference_adult.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (label, short) in zip(axes, [("forward initialisation (z = mu, then clamp z_out = y)", "forward initialisation"),
                                         ("zero initialisation (z = 0, then clamp z_out = y)", "zero initialisation")]):
        trace = np.array(data[label])
        for i, node in enumerate(NODES):
            ax.plot(trace[:, i], lw=2, label=f"E {node}")
        ax.plot(trace.sum(1), "k--", lw=1.5, label="E total")
        ax.axvline(20, color="#999", lw=1)
        ax.set_title(f"adult, untrained network: {short}")
        ax.set_xlabel("inference step T")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("energy per example")
    axes[0].legend()
    save(fig, "inference_energy.png")


def alignment():
    data = load("alignment.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key, xlabel in [(axes[0], "steps", "inference steps T"), (axes[1], "depth", "hidden layers")]:
        xs = [int(k) for k in data[key]]
        for mode, color in [("strict", COLORS["PC"]), ("fixed", "#548235")]:
            ax.plot(xs, [data[key][str(x)][mode] for x in xs], "o-", color=color, lw=2,
                    label="strict PC (slides)" if mode == "strict" else "fixed prediction PC")
        if key == "steps":
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("cosine(PC update, BP gradient)")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_title("MNIST 784-256-128-10")
    axes[1].set_title("after 500 steps, 128 units per layer")
    save(fig, "alignment.png")


if __name__ == "__main__":
    inference_energy()
    alignment()
    if all(os.path.exists(os.path.join(RESULTS, f"compare_{n}.json")) for n in ["adult", "mnist"]):
        accuracy_and_energy()
