"""The two learners. Both use the same torch.nn.Linear layers, so they share architecture and initial weights.

Notation (slides "Predictive Coding Architecture"):
    z(0) = x                              input node, clamped to the data
    S(l) = W(l) z(l-1) + b(l)             weighted sum entering node l
    mu(l) = A(S(l))                       prediction (projection) of node l
    eps(l) = z(l) - mu(l)                 prediction error of node l
    E(l) = 1/2 ||eps(l)||^2               energy of node l
    E_total = sum_l E(l)

Backprop: forward pass, L = 1/2 ||y_hat - y||^2, gradient by the chain rule (autograd), W <- W - alpha dL/dW.
PC: z(L) is clamped to the target, the latent states z(1..L-1) relax for T steps
    z(l) <- z(l) - eta (dE(l)/dz(l) + dE(l+1)/dz(l)),   eta capped at 1 / (1 + ||W(l+1)||^2) for stability,
then W <- W - alpha dE(l)/dW(l), computed from local quantities only.
"""
import copy

import torch
from torch import nn


def relu_derivative(s):
    return (s > 0).to(s.dtype)


ACTIVATIONS = {
    "relu": (torch.relu, relu_derivative),
    "tanh": (torch.tanh, lambda s: 1 - torch.tanh(s) ** 2),
    "linear": (lambda s: s, torch.ones_like),
}


def make_layers(sizes, seed):
    """One stack of Linear layers; BP and PC each get an identical deep copy."""
    torch.manual_seed(seed)
    return nn.ModuleList(nn.Linear(n_in, n_out) for n_in, n_out in zip(sizes[:-1], sizes[1:]))


def activation_names(n_layers, hidden="relu"):
    """Hidden nodes use ReLU and the output node is linear, as in the slides."""
    return [hidden] * (n_layers - 1) + ["linear"]


class BPNet(nn.Module):
    def __init__(self, layers, hidden="relu"):
        super().__init__()
        self.layers = copy.deepcopy(layers)
        self.acts = activation_names(len(self.layers), hidden)

    def forward(self, x):
        for layer, name in zip(self.layers, self.acts):
            x = ACTIVATIONS[name][0](layer(x))
        return x

    def loss(self, x, y):
        return 0.5 * ((self(x) - y) ** 2).sum(dim=1).mean()

    def train_step(self, x, y, optimizer):
        optimizer.zero_grad()
        loss = self.loss(x, y)
        loss.backward()
        optimizer.step()
        return loss.item()


class PCNet:
    def __init__(self, layers, eta=0.1, T=20, hidden="relu", mode="strict", cap_eta=True):
        self.layers = copy.deepcopy(layers)
        for p in self.layers.parameters():
            p.requires_grad_(False)
        self.acts = activation_names(len(self.layers), hidden)
        self.eta, self.T, self.mode, self.cap_eta = eta, T, mode, cap_eta
        self.L = len(self.layers)

    def parameters(self):
        return self.layers.parameters()

    def project(self, l, z_prev):
        """S(l) and mu(l) = A(S(l)) for node l (1..L) from the node below."""
        layer = self.layers[l - 1]
        S = torch.nn.functional.linear(z_prev, layer.weight, layer.bias)
        return S, ACTIVATIONS[self.acts[l - 1]][0](S)

    def dA(self, l, S):
        return ACTIVATIONS[self.acts[l - 1]][1](S)

    @torch.no_grad()
    def forward(self, x):
        """Plain forward pass: the forward initialisation of training, and the whole of testing."""
        z = [x]
        for l in range(1, self.L + 1):
            z.append(self.project(l, z[-1])[1])
        return z

    def __call__(self, x):
        return self.forward(x)[-1]

    @torch.no_grad()
    def errors(self, z, fixed=None):
        """Predictions, weighted sums, errors and energies of nodes 1..L (index 0 is unused)."""
        S, mu, eps, E = [None], [None], [None], [None]
        for l in range(1, self.L + 1):
            if fixed is None:
                S_l, mu_l = self.project(l, z[l - 1])
            else:
                S_l, mu_l = fixed[0][l], fixed[1][l]
            S.append(S_l)
            mu.append(mu_l)
            eps.append(z[l] - mu_l)
            E.append(0.5 * (eps[l] ** 2).sum(dim=1))
        return S, mu, eps, E

    @torch.no_grad()
    def stable_eta(self):
        """eta for each hidden node, capped at 1 / (1 + ||W(l+1)||^2) so the state update cannot overshoot."""
        if not self.cap_eta:
            return [None] + [self.eta] * (self.L - 1)
        return [None] + [min(self.eta, 1 / (1 + torch.linalg.matrix_norm(self.layers[l].weight, 2).item() ** 2))
                         for l in range(1, self.L)]

    @torch.no_grad()
    def infer(self, x, y, T=None, record=False, init="forward"):
        """Forward initialisation (or zeros), clamp the output to y, then T synchronous state updates."""
        T = self.T if T is None else T
        z = self.forward(x)
        if init == "zero":
            z = [x] + [torch.zeros_like(t) for t in z[1:]]
        z[-1] = y.clone()
        z_ff = [t.clone() for t in z[:-1]]
        fixed = None
        if self.mode == "fixed":
            S0, mu0, _, _ = self.errors(self.forward(x))
            fixed = (S0, mu0)
        eta = self.stable_eta()
        trace = []
        for _ in range(T):
            S, mu, eps, E = self.errors(z, fixed)
            if record:
                trace.append([float(E[l].mean()) for l in range(1, self.L + 1)])
            for l in range(1, self.L):
                dE_self = eps[l]
                dE_next = -(eps[l + 1] * self.dA(l + 1, S[l + 1])) @ self.layers[l].weight
                z[l] = z[l] - eta[l] * (dE_self + dE_next)
        S, mu, eps, E = self.errors(z, fixed)
        if record:
            trace.append([float(E[l].mean()) for l in range(1, self.L + 1)])
        return {"z": z, "z_ff": z_ff, "S": S, "mu": mu, "eps": eps, "E": E, "trace": trace}

    @torch.no_grad()
    def weight_gradients(self, state):
        """dE(l)/dW(l) = -(eps(l) * A'(S(l)))^T z(l-1): error of node l times activity of node l-1."""
        z = state["z_ff"] if self.mode == "fixed" else state["z"]
        S, eps = state["S"], state["eps"]
        grads = []
        for l in range(1, self.L + 1):
            delta = eps[l] * self.dA(l, S[l])
            grads.append((-(delta.T @ z[l - 1]) / len(delta), -delta.mean(dim=0)))
        return grads

    def train_step(self, x, y, optimizer):
        state = self.infer(x, y)
        for layer, (dW, db) in zip(self.layers, self.weight_gradients(state)):
            layer.weight.grad = dW
            layer.bias.grad = db
        optimizer.step()
        self.last_energies = self.layer_energies(state)
        return sum(self.last_energies)

    def layer_energies(self, state):
        return [float(state["E"][l].mean()) for l in range(1, self.L + 1)]


def count_parameters(layers):
    """Learned weights and biases, and the latent states PC holds per example (hidden nodes, output node)."""
    weights = sum(layer.weight.numel() for layer in layers)
    biases = sum(layer.bias.numel() for layer in layers)
    hidden_states = sum(layer.out_features for layer in layers[:-1])
    return weights, biases, hidden_states, layers[-1].out_features
