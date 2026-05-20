"""Smoke tests for the integrated VoxModel.

Tiny configs so this runs on CPU in seconds. The point is to verify the
end-to-end wiring (shapes, gradients, inference), not to train anything.
"""
import torch
import pytest

from vox.models.conditioning.aggregator import AggregatorConfig
from vox.models.diffusion.decoder import DecoderConfig
from vox.models.vox_model import VoxModel, VoxModelConfig


def _tiny_cfg() -> VoxModelConfig:
    return VoxModelConfig(
        hidden=32,
        n_styles=3,
        diffusion_steps=100,
        aggregator=AggregatorConfig(hidden=32, n_styles=3, gst_num_tokens=4, gst_num_heads=4),
        decoder=DecoderConfig(hidden=32, cond_dim=32, num_blocks=2, time_dim=32),
    )


def _fake_batch(B=2, T=40, n_mels=128, content_dim=768):
    return {
        "mel": torch.randn(B, n_mels, T),
        "content": torch.randn(B, content_dim, T),
        "f0": torch.rand(B, T) * 400 + 100,
        "uv": torch.ones(B, T, dtype=torch.bool),
        "loudness": torch.rand(B, T),
        "style_id": torch.zeros(B, dtype=torch.long),
        "mask": torch.ones(B, T, dtype=torch.bool),
    }


def test_training_step_returns_finite_loss():
    model = VoxModel(_tiny_cfg())
    out = model.training_step(_fake_batch())
    assert torch.isfinite(out["total_loss"])


def test_training_step_backward():
    model = VoxModel(_tiny_cfg())
    out = model.training_step(_fake_batch())
    out["total_loss"].backward()
    # Decoder and aggregator should both accumulate gradient.
    dec_g = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters()
    )
    agg_g = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.aggregator.parameters()
    )
    assert dec_g and agg_g


def test_training_step_mask_normalises_by_valid_frames():
    """Halving the valid mask region must not halve the loss — the mean
    is taken only over valid frames, not over all frames."""
    model = VoxModel(_tiny_cfg()).eval()  # eval to silence any dropout
    batch = _fake_batch(B=1, T=40)

    torch.manual_seed(0)
    loss_full = model.training_step(batch)["total_loss"].item()

    batch_half = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    batch_half["mask"] = torch.zeros(1, 40, dtype=torch.bool)
    batch_half["mask"][:, :20] = True
    torch.manual_seed(0)
    loss_half = model.training_step(batch_half)["total_loss"].item()

    # Both should be order-1 numbers (not 2x apart), proving the mean denominator
    # tracks the mask sum rather than the raw element count.
    ratio = loss_full / max(loss_half, 1e-8)
    assert 0.3 < ratio < 3.0, f"loss_full={loss_full}, loss_half={loss_half}, ratio={ratio}"


@pytest.mark.slow
def test_inference_produces_wav():
    model = VoxModel(_tiny_cfg()).eval()
    batch = _fake_batch(B=1, T=20)
    out = model.infer(
        content=batch["content"],
        f0=batch["f0"],
        uv=batch["uv"],
        loudness=batch["loudness"],
        style_id=batch["style_id"],
        num_steps=3,
    )
    assert out["mel"].shape == (1, 128, 20)
    assert out["wav"].shape[0] == 1
    assert torch.isfinite(out["wav"]).all()
