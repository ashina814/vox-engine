import torch
from vox.data.features.uv import compute_uv


def test_voiced_when_f0_positive():
    f0 = torch.tensor([0.0, 100.0, 200.0, 0.0])
    uv = compute_uv(f0)
    assert uv.tolist() == [False, True, True, False]


def test_unvoiced_when_energy_low():
    f0 = torch.tensor([100.0, 200.0, 300.0])
    energy = torch.tensor([1e-5, 1.0, 1e-5])
    uv = compute_uv(f0, energy=energy, energy_threshold=1e-4)
    assert uv.tolist() == [False, True, False]


def test_without_energy():
    f0 = torch.tensor([0.0, 50.0, 0.0])
    uv = compute_uv(f0)
    assert uv.tolist() == [False, True, False]
