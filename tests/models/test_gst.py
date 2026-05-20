import torch
import pytest

from vox.models.conditioning.gst import GlobalStyleTokens, ReferenceEncoder


def test_reference_encoder_shape():
    enc = ReferenceEncoder(n_mels=128, out_dim=256)
    mel = torch.randn(2, 128, 200)
    out = enc(mel)
    assert out.shape == (2, 256)


def test_gst_forward_shape():
    gst = GlobalStyleTokens(num_tokens=10, hidden=256, num_heads=8, n_mels=128)
    ref = torch.randn(3, 128, 250)
    style = gst(ref)
    assert style.shape == (3, 256)


def test_gst_backward():
    gst = GlobalStyleTokens(num_tokens=4, hidden=64, num_heads=4, n_mels=128)
    ref = torch.randn(2, 128, 100)
    style = gst(ref)
    loss = style.pow(2).mean()
    loss.backward()
    # Token bank must accumulate gradient.
    assert gst.tokens.grad is not None
    assert gst.tokens.grad.abs().sum() > 0


@pytest.mark.parametrize("B,T", [(1, 80), (4, 300)])
def test_gst_varying_input(B, T):
    gst = GlobalStyleTokens(num_tokens=6, hidden=128, num_heads=4, n_mels=128)
    style = gst(torch.randn(B, 128, T))
    assert style.shape == (B, 128)
