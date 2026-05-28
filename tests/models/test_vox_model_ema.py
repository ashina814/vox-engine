"""EMA integration into VoxModel.infer.

These tests guard against the failure mode where EMA shadow weights are
accumulated but never used at inference (which would make EMA a no-op
despite costing memory and doubling ckpt size).
"""

import torch

from vox.models.conditioning.aggregator import AggregatorConfig
from vox.models.diffusion.decoder import DecoderConfig
from vox.models.vox_model import VoxModel, VoxModelConfig
from vox.training.ema import ExponentialMovingAverage


def _tiny_model() -> VoxModel:
    cfg = VoxModelConfig(
        hidden=32,
        n_styles=3,
        diffusion_steps=100,
        aggregator=AggregatorConfig(hidden=32, n_styles=3, gst_num_tokens=4, gst_num_heads=4),
        decoder=DecoderConfig(hidden=32, cond_dim=32, num_blocks=2, time_dim=32),
    )
    return VoxModel(cfg).eval()


def _fake_batch(B=1, T=20):
    return {
        "content": torch.randn(B, 768, T),
        "f0": torch.rand(B, T) * 400 + 100,
        "uv": torch.ones(B, T, dtype=torch.bool),
        "loudness": torch.rand(B, T),
        "style_id": torch.zeros(B, dtype=torch.long),
    }


def _infer(model: VoxModel, batch: dict, **kw) -> torch.Tensor:
    torch.manual_seed(0)
    out = model.infer(num_steps=2, **batch, **kw)
    return out["mel"]


def test_infer_uses_ema_when_attached():
    """Snapshot pre-training weights into EMA, mutate the model, confirm
    infer falls back to the EMA snapshot."""
    model = _tiny_model()
    ema = ExponentialMovingAverage(model, decay=0.5)
    batch = _fake_batch()

    baseline = _infer(model, batch)

    # Push live weights far away from the EMA snapshot.
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(5.0)

    # Without EMA hook: infer reflects the perturbed weights.
    perturbed = _infer(model, batch)
    assert not torch.allclose(
        baseline, perturbed, atol=1e-4
    ), "Sanity check failed: perturbation didn't change inference output"

    # Attach EMA → infer should now match the EMA snapshot (the pre-perturbation state).
    model.ema = ema
    with_ema = _infer(model, batch)
    assert torch.allclose(
        with_ema, baseline, atol=1e-4
    ), "infer() ignored ema.swap_in — shadow weights are dead code"


def test_infer_use_ema_false_bypasses_shadow():
    """``use_ema=False`` must return live-weight output even when EMA attached."""
    model = _tiny_model()
    ema = ExponentialMovingAverage(model, decay=0.5)
    model.ema = ema
    batch = _fake_batch()

    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(3.0)

    live = _infer(model, batch, use_ema=False)
    shadow = _infer(model, batch, use_ema=True)
    assert not torch.allclose(live, shadow, atol=1e-4)


def test_infer_restores_weights_after_ema_swap():
    """After infer with EMA, the model's live parameters must be untouched."""
    model = _tiny_model()
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(4.0)
    model.ema = ema

    pre = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    _infer(model, _fake_batch())
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert torch.allclose(
                p, pre[n], atol=1e-6
            ), f"infer with EMA left live parameter {n!r} corrupted"
