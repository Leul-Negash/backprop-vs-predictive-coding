import copy

import pytest
import torch

from models import ACTIVATIONS, BPNet, PCNet, make_layers
from slide_example import slide_layers

torch.manual_seed(0)


def energies(pc, layers, z):
    """E(l) for every node, computed with autograd switched on (only the tests use this)."""
    out = []
    for l, layer in enumerate(layers, start=1):
        mu = ACTIVATIONS[pc.acts[l - 1]][0](z[l - 1] @ layer.weight.T + layer.bias)
        out.append(0.5 * ((z[l] - mu) ** 2).sum(dim=1))
    return out


def small_net(mode="strict", sizes=(12, 8, 6, 3), seed=0):
    return PCNet(make_layers(list(sizes), seed).double(), eta=0.1, T=20, mode=mode)


def batch(n=16, d=12, k=3):
    x = torch.rand(n, d, dtype=torch.float64)
    y = torch.nn.functional.one_hot(torch.randint(0, k, (n,)), k).double()
    return x, y


def test_backprop_reproduces_the_slides():
    x = torch.tensor([[0.5, 0.433, 0.0, 0.2]], dtype=torch.float64)
    y = torch.tensor([[1.0]], dtype=torch.float64)
    bp = BPNet(slide_layers())
    loss = bp.loss(x, y)
    loss.backward()
    assert bp(x).item() == pytest.approx(0.2283, abs=1e-4)
    assert loss.item() == pytest.approx(0.2977, abs=1e-4)
    assert bp.layers[1].weight.grad[0, 0].item() == pytest.approx(-0.1208, abs=1e-4)
    assert bp.layers[0].weight.grad[0, 1].item() == pytest.approx(-0.16707, abs=1e-5)


def test_predictive_coding_reproduces_the_slides():
    x = torch.tensor([[0.5, 0.433, 0.0, 0.2]], dtype=torch.float64)
    y = torch.tensor([[1.0]], dtype=torch.float64)
    pc = PCNet(slide_layers(), eta=0.1)
    start = pc.infer(x, y, T=0)
    assert start["E"][2].item() == pytest.approx(0.2977, abs=1e-4)
    assert start["E"][1].item() == 0.0
    one = pc.infer(x, y, T=1)
    assert one["z"][1][0, 0].item() == pytest.approx(0.1566 + 0.1 * 0.5 * 0.7717, abs=1e-4)
    assert one["z"][1][0, 1].item() == pytest.approx(0.1385, abs=1e-3)
    assert one["E"][1].item() == pytest.approx(0.0015, abs=1e-4)
    assert one["eps"][2].item() == pytest.approx(0.7335, abs=1e-3)


def test_testing_is_the_same_forward_pass_for_both():
    layers = make_layers([12, 8, 6, 3], 0)
    x = torch.rand(5, 12)
    assert torch.equal(BPNet(layers)(x), PCNet(layers)(x))


def test_state_update_is_the_gradient_of_the_total_energy():
    pc = small_net()
    x, y = batch()
    state = pc.infer(x, y, T=3)
    z = [t.clone().requires_grad_(True) for t in state["z"]]
    energy = sum(e.sum() for e in energies(pc, pc.layers, z))
    energy.backward()
    S, mu, eps, E = pc.errors(state["z"])
    for l in range(1, pc.L):
        local = eps[l] - (eps[l + 1] * pc.dA(l + 1, S[l + 1])) @ pc.layers[l].weight
        assert torch.allclose(local, z[l].grad, atol=1e-10)


def test_weight_update_is_the_gradient_of_the_total_energy():
    pc = small_net()
    x, y = batch()
    state = pc.infer(x, y, T=3)
    grads = pc.weight_gradients(state)
    layers = copy.deepcopy(pc.layers)
    for p in layers.parameters():
        p.requires_grad_(True)
    energy = sum(e.mean() for e in energies(pc, layers, state["z"]))
    energy.backward()
    for layer, (dW, db) in zip(layers, grads):
        assert torch.allclose(dW, layer.weight.grad, atol=1e-10)
        assert torch.allclose(db, layer.bias.grad, atol=1e-10)


def test_weight_update_uses_only_its_own_layer():
    pc = small_net()
    x, y = batch()
    state = pc.infer(x, y)
    reference = pc.weight_gradients(state)[0][0]
    tampered = dict(state)
    tampered["eps"] = [state["eps"][0], state["eps"][1]] + [torch.randn_like(e) * 10 for e in state["eps"][2:]]
    tampered["z"] = state["z"][:2] + [torch.randn_like(t) * 10 for t in state["z"][2:]]
    assert torch.equal(reference, pc.weight_gradients(tampered)[0][0])


@pytest.mark.parametrize("mode", ["strict", "fixed"])
def test_predictive_coding_trains_with_autograd_switched_off(monkeypatch, mode):
    def forbidden(name):
        def fail(*args, **kwargs):
            raise AssertionError(f"autograd was used: {name}")
        return fail

    monkeypatch.setattr(torch.autograd, "grad", forbidden("torch.autograd.grad"))
    monkeypatch.setattr(torch.autograd, "backward", forbidden("torch.autograd.backward"))
    monkeypatch.setattr(torch.Tensor, "backward", forbidden("Tensor.backward"))
    pc = small_net(mode)
    x, y = batch()
    optimizer = torch.optim.SGD(pc.parameters(), lr=0.2)
    before = 0.5 * ((pc(x) - y) ** 2).sum(dim=1).mean()
    for _ in range(100):
        pc.train_step(x, y, optimizer)
    after = 0.5 * ((pc(x) - y) ** 2).sum(dim=1).mean()
    assert after < 0.5 * before


def test_no_tensor_in_predictive_coding_carries_a_graph():
    pc = small_net()
    x, y = batch()
    state = pc.infer(x, y)
    for key in ["z", "mu", "eps"]:
        for t in state[key][1:]:
            assert not t.requires_grad and t.grad_fn is None
    assert not any(p.requires_grad for p in pc.parameters())


def test_energy_never_increases_during_inference_with_a_smooth_activation():
    pc = PCNet(make_layers([12, 8, 6, 3], 0).double(), eta=0.1, hidden="tanh")
    x, y = batch()
    totals = [sum(step) for step in pc.infer(x, y, T=50, record=True)["trace"]]
    assert all(b <= a + 1e-12 for a, b in zip(totals, totals[1:]))


def test_energy_falls_during_inference_with_relu():
    pc = small_net()
    x, y = batch()
    totals = [sum(step) for step in pc.infer(x, y, T=50, record=True)["trace"]]
    assert totals[-1] < 0.8 * totals[0]
    assert all(b <= a + 1e-4 for a, b in zip(totals, totals[1:]))


def test_fixed_prediction_recovers_the_backprop_gradient():
    pc = small_net("fixed")
    x, y = batch()
    pc_grads = torch.cat([g.flatten() for pair in pc.weight_gradients(pc.infer(x, y, T=500)) for g in pair])
    bp = BPNet(pc.layers)
    for p in bp.parameters():
        p.requires_grad_(True)
    bp.loss(x, y).backward()
    bp_grads = torch.cat([p.grad.flatten() for p in bp.parameters()])
    assert torch.nn.functional.cosine_similarity(pc_grads, bp_grads, dim=0).item() > 0.9999


def test_inference_stays_stable_when_the_weights_grow():
    pc = PCNet(make_layers([12, 8, 6, 3], 0).double(), eta=0.1, hidden="tanh")
    with torch.no_grad():
        for layer in pc.layers:
            layer.weight.mul_(8)
    x, y = batch()
    totals = [sum(step) for step in pc.infer(x, y, T=50, record=True)["trace"]]
    assert all(b <= a + 1e-12 for a, b in zip(totals, totals[1:]))
