import torch

from vox.models.conditioning.whisper_branch import WHISPER_ID, WhisperAwareConditioning


def test_whisper_branch_shape():
    branch = WhisperAwareConditioning(hidden=64)
    base = torch.randn(2, 64, 50)
    uv = torch.ones(2, 50)
    style = torch.tensor([0, WHISPER_ID])
    out = branch(base, uv, style)
    assert out.shape == base.shape


def test_whisper_branch_skips_when_no_whisper():
    """If no sample in the batch is whisper, output must equal input exactly."""
    branch = WhisperAwareConditioning(hidden=32)
    base = torch.randn(3, 32, 40)
    uv = torch.rand(3, 40)
    style = torch.zeros(3, dtype=torch.long)  # all normal
    out = branch(base, uv, style)
    assert torch.equal(out, base)


def test_whisper_branch_only_modifies_whisper_samples():
    branch = WhisperAwareConditioning(hidden=32)
    base = torch.randn(2, 32, 40)
    uv = torch.zeros(2, 40)  # fully unvoiced → max residual
    style = torch.tensor([0, WHISPER_ID])
    out = branch(base, uv, style)
    # Non-whisper sample (index 0) unchanged
    assert torch.equal(out[0], base[0])
    # Whisper sample (index 1) modified
    assert not torch.equal(out[1], base[1])


def test_whisper_branch_voiced_frames_unchanged():
    """uv == 1 frames are gated to 0 → those columns should stay the same."""
    branch = WhisperAwareConditioning(hidden=32)
    base = torch.randn(1, 32, 40)
    uv = torch.zeros(1, 40)
    uv[:, :20] = 1.0  # first half voiced
    style = torch.tensor([WHISPER_ID])
    out = branch(base, uv, style)
    assert torch.allclose(out[..., :20], base[..., :20])
    # latter half must differ
    assert not torch.allclose(out[..., 20:], base[..., 20:])
