"""The worked example of the slides, run through the same BPNet and PCNet used for the experiments.

The slides give every weight only for the part of the network from HL-1 to the output, so the example
starts from the HL-1 activities [0.5, 0.433, 0.0, 0.2] and uses HL-1 -> HL-2 (ReLU) -> output (linear).
The slides round intermediate values (e.g. 0.1566 -> 0.156), so the last digit can differ.
"""
import torch

from models import BPNet, PCNet, make_layers

torch.set_printoptions(precision=5)


def slide_layers():
    layers = make_layers([4, 2, 1], seed=0).double()
    with torch.no_grad():
        layers[0].weight.copy_(torch.tensor([[-0.1, 0.2, 0.3, 0.1], [0.0, 0.0, 0.0, 0.0]]))
        layers[0].bias.copy_(torch.tensor([0.1, 0.1]))
        layers[1].weight.copy_(torch.tensor([[0.5, 0.5]]))
        layers[1].bias.copy_(torch.tensor([0.1]))
    return layers


def row(name, ours, slide):
    print(f"  {name:<44}{ours:>12.5f}{slide:>12}")


def main():
    x = torch.tensor([[0.5, 0.433, 0.0, 0.2]], dtype=torch.float64)
    y = torch.tensor([[1.0]], dtype=torch.float64)
    print("slides' HL-2 neuron 2 shows only mu = 0.1, so its incoming weights are set to 0 with b = 0.1\n")

    print("=" * 68)
    print("Backprop: forward pass, loss and one SGD step (alpha = 0.01)")
    print("=" * 68)
    print(f"  {'quantity':<44}{'this code':>12}{'slides':>12}")
    bp = BPNet(slide_layers())
    h = torch.relu(bp.layers[0](x))
    y_hat = bp(x)
    loss = bp.loss(x, y)
    row("HL-2 neuron 1: A = ReLU(Z)", h[0, 0].item(), "0.1566")
    row("output: y_hat = Linear(Z)", y_hat.item(), "0.2283")
    row("loss L = 1/2 (y_hat - y)^2", loss.item(), "0.2977")
    optimizer = torch.optim.SGD(bp.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    row("output: dW (from HL-2 neuron 1)", bp.layers[1].weight.grad[0, 0].item(), "-0.1208")
    row("output: db", bp.layers[1].bias.grad[0].item(), "-0.7717")
    row("HL-2 neuron 1: dW (from HL-1 activity 0.433)", bp.layers[0].weight.grad[0, 1].item(), "-0.16707")
    row("HL-2 neuron 1: db", bp.layers[0].bias.grad[0].item(), "-0.38585")
    optimizer.step()
    row("output: W_new = W - alpha dW", bp.layers[1].weight[0, 0].item(), "0.50121")
    row("output: b_new = b - alpha db", bp.layers[1].bias[0].item(), "0.10771")
    row("HL-2 neuron 1: W_new", bp.layers[0].weight[0, 1].item(), "0.20167")
    row("HL-2 neuron 1: b_new", bp.layers[0].bias[0].item(), "0.10386")

    print("\n" + "=" * 68)
    print("Predictive coding: forward initialisation, output clamped to y, eta = 0.1")
    print("=" * 68)
    pc = PCNet(slide_layers(), eta=0.1, T=2)
    print(f"  {'quantity':<44}{'this code':>12}{'slides':>12}")
    state = pc.infer(x, y, T=0)
    row("T=0: z = mu for HL-2 neuron 1", state["z"][1][0, 0].item(), "0.156")
    row("T=0: output error eps = z - mu", state["eps"][2][0, 0].item(), "0.7717")
    row("T=0: output energy E = 1/2 eps^2", state["E"][2][0].item(), "0.2977")
    state = pc.infer(x, y, T=1)
    row("T=1: HL-2 neuron 1 z_new", state["z"][1][0, 0].item(), "0.1945")
    row("T=1: HL-2 neuron 2 z_new", state["z"][1][0, 1].item(), "0.1385")
    S, mu, eps, E = pc.errors(state["z"])
    row("T=2: output prediction mu_new", mu[2][0, 0].item(), "0.2665")
    row("T=2: output error eps_new", eps[2][0, 0].item(), "0.7335")
    row("T=2: output energy E_new", E[2][0].item(), "0.269")
    row("T=2: HL-2 neuron errors eps_1 = eps_2", eps[1][0, 0].item(), "0.0385")
    row("T=2: HL-2 energy E_HL2 = E_1 + E_2", E[1][0].item(), "0.0015")

    print("\nslide table: energy of each layer at each inference step (ours)")
    print(f"  {'step(T)':<10}{'HL-2':>12}{'Output':>12}{'total':>12}")
    for t, energies in enumerate(pc.infer(x, y, T=4, record=True)["trace"]):
        print(f"  {t + 1:<10}{energies[0]:>12.5f}{energies[1]:>12.5f}{sum(energies):>12.5f}")
    print("  (slides: step 1 -> HL-2 0, output 0.2977; step 2 -> HL-2 0.0015, output 0.269)")


if __name__ == "__main__":
    main()
