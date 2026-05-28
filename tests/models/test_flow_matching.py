import torch

from vox.models.diffusion.decoder import DecoderConfig, DiffusionDecoder
from vox.models.diffusion.flow_matching import FlowMatchingSampler, FlowMatchingSchedule


def test_flow_matching_add_noise_shapes():
    sched = FlowMatchingSchedule(num_steps=1000)
    x0 = torch.randn(2, 128, 50)
    t = torch.tensor([100, 500])
    x_t, noise, v = sched.add_noise(x0, t)
    assert x_t.shape == x0.shape == noise.shape == v.shape


def test_flow_matching_v_target_is_noise_minus_x0():
    """Core invariant of Lipman et al.'s linear flow: v == noise - x0."""
    sched = FlowMatchingSchedule(num_steps=1000)
    x0 = torch.randn(1, 4, 10)
    noise = torch.randn_like(x0)
    _, _, v = sched.add_noise(x0, torch.tensor([500]), noise=noise)
    assert torch.allclose(v, noise - x0, atol=1e-6)


def test_flow_matching_t1_is_noise():
    """At t = num_steps (the noise endpoint), x_t == noise exactly."""
    sched = FlowMatchingSchedule(num_steps=1000)
    x0 = torch.randn(1, 4, 10)
    noise = torch.randn_like(x0)
    x_t, _, _ = sched.add_noise(x0, torch.tensor([1000]), noise=noise)
    assert torch.allclose(x_t, noise, atol=1e-6)


def test_flow_matching_endpoint_blending():
    """At t/N near 0.5, x_t lies on the midpoint of the x0 -> noise line."""
    sched = FlowMatchingSchedule(num_steps=1000)
    x0 = torch.zeros(1, 4, 10)
    noise = torch.ones_like(x0)
    x_t, _, _ = sched.add_noise(x0, torch.tensor([500]), noise=noise)
    assert torch.allclose(x_t, torch.full_like(x_t, 0.5), atol=1e-3)


def test_flow_matching_decoder_one_step_backward():
    cfg = DecoderConfig(n_mels=128, hidden=32, cond_dim=32, num_blocks=2, time_dim=32)
    dec = DiffusionDecoder(cfg)
    sched = FlowMatchingSchedule(num_steps=500)
    x0 = torch.randn(2, 128, 30)
    cond = torch.randn(2, 32, 30)
    t = sched.sample_random_t(2, device=torch.device("cpu"))
    x_t, _, v_target = sched.add_noise(x0, t)
    v_pred = dec(x_t, t, cond)
    loss = (v_pred - v_target).pow(2).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in dec.parameters())


def test_flow_matching_sampler_shape():
    cfg = DecoderConfig(n_mels=128, hidden=32, cond_dim=32, num_blocks=2, time_dim=32)
    dec = DiffusionDecoder(cfg).eval()
    sched = FlowMatchingSchedule(num_steps=200)
    sampler = FlowMatchingSampler(sched)
    cond = torch.randn(1, 32, 20)
    out = sampler.sample(dec, cond, shape=(1, 128, 20), num_steps=4)
    assert out.shape == (1, 128, 20)
    assert torch.isfinite(out).all()


def test_flow_matching_sampler_k_steps_robust():
    """Output must remain finite across K = 1, 4, 16."""
    cfg = DecoderConfig(n_mels=128, hidden=32, cond_dim=32, num_blocks=2, time_dim=32)
    dec = DiffusionDecoder(cfg).eval()
    sched = FlowMatchingSchedule(num_steps=200)
    sampler = FlowMatchingSampler(sched)
    cond = torch.randn(1, 32, 15)
    for k in (1, 4, 16):
        out = sampler.sample(dec, cond, shape=(1, 128, 15), num_steps=k)
        assert torch.isfinite(out).all(), f"NaN/Inf with K={k}"
