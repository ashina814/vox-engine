import torch

from vox.models.conditioning.aggregator import AggregatorConfig, ConditionAggregator


def _fake_inputs(B=2, T=100, content_dim=768, n_mels=128):
    return dict(
        content=torch.randn(B, content_dim, T),
        f0=torch.rand(B, T) * 400 + 100,
        uv=torch.ones(B, T, dtype=torch.bool),
        loudness=torch.rand(B, T),
        style_id=torch.zeros(B, dtype=torch.long),
        ref_mel=torch.randn(B, n_mels, 200),
    )


def test_aggregator_output_shape():
    agg = ConditionAggregator()
    out = agg(**_fake_inputs())
    assert out.shape == (2, 256, 100)


def test_aggregator_no_ref_mel():
    """ref_mel is optional; GST contribution should be skipped."""
    agg = ConditionAggregator()
    inputs = _fake_inputs()
    inputs["ref_mel"] = None
    out = agg(**inputs)
    assert out.shape == (2, 256, 100)


def test_aggregator_no_gst():
    cfg = AggregatorConfig(use_gst=False, hidden=128)
    agg = ConditionAggregator(cfg)
    inputs = _fake_inputs(B=1, T=50)
    out = agg(**inputs)
    assert out.shape == (1, 128, 50)


def test_aggregator_style_id_routing():
    """Different style_ids should produce different outputs (sanity)."""
    torch.manual_seed(0)
    cfg = AggregatorConfig(n_styles=3, hidden=64)
    agg = ConditionAggregator(cfg).eval()
    inputs = _fake_inputs(B=1, T=40)

    inputs["style_id"] = torch.tensor([0])
    out_a = agg(**inputs)
    inputs["style_id"] = torch.tensor([2])
    out_b = agg(**inputs)
    assert not torch.allclose(out_a, out_b)


def test_aggregator_backward():
    agg = ConditionAggregator(AggregatorConfig(hidden=64))
    out = agg(**_fake_inputs(B=1, T=20))
    out.pow(2).mean().backward()
    # At least one parameter should accumulate grad.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agg.parameters())
