import pytest
import torch

from vox.inference.style_blend import slerp, slerp_barycentric


def test_slerp_alpha_zero_returns_v0():
    v0, v1 = torch.randn(8), torch.randn(8)
    assert torch.equal(slerp(v0, v1, 0.0), v0)


def test_slerp_alpha_one_returns_v1():
    v0, v1 = torch.randn(8), torch.randn(8)
    assert torch.equal(slerp(v0, v1, 1.0), v1)


def test_slerp_identity_when_v0_equals_v1():
    v = torch.randn(16)
    assert torch.allclose(slerp(v, v, 0.5), v, atol=1e-5)


def test_slerp_preserves_unit_norm():
    """SLERP between two unit vectors should yield (approximately) a unit vector."""
    torch.manual_seed(0)
    v0 = torch.randn(32)
    v0 = v0 / v0.norm()
    v1 = torch.randn(32)
    v1 = v1 / v1.norm()
    out = slerp(v0, v1, 0.5)
    assert abs(out.norm().item() - 1.0) < 1e-4


def test_barycentric_one_hot():
    a, b, c = torch.randn(8), torch.randn(8), torch.randn(8)
    assert torch.equal(slerp_barycentric([a, b, c], [1, 0, 0]), a)
    assert torch.equal(slerp_barycentric([a, b, c], [0, 1, 0]), b)
    assert torch.equal(slerp_barycentric([a, b, c], [0, 0, 1]), c)


def test_barycentric_two_way_matches_slerp():
    a, b = torch.randn(8), torch.randn(8)
    expected = slerp(a, b, 0.3)
    out = slerp_barycentric([a, b], [0.7, 0.3])
    assert torch.allclose(out, expected, atol=1e-5)


def test_barycentric_normalises_weights():
    a, b = torch.randn(4), torch.randn(4)
    out1 = slerp_barycentric([a, b], [3.0, 7.0])  # un-normalized
    out2 = slerp_barycentric([a, b], [0.3, 0.7])
    assert torch.allclose(out1, out2, atol=1e-5)


def test_barycentric_zero_total_raises():
    with pytest.raises(ValueError):
        slerp_barycentric([torch.randn(4), torch.randn(4)], [0.0, 0.0])


def test_barycentric_three_vectors_shape():
    vecs = [torch.randn(16) for _ in range(3)]
    out = slerp_barycentric(vecs, [0.5, 0.3, 0.2])
    assert out.shape == (16,)
    assert torch.isfinite(out).all()
