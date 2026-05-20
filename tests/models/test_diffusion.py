import torch

from vox.models.diffusion.decoder import DecoderConfig, DiffusionDecoder
from vox.models.diffusion.noise_schedule import NoiseSchedule, cosine_alpha_bar
from vox.models.diffusion.sampler import DDIMSampler


def test_cosine_alpha_bar_monotonic():
    ab = cosine_alpha_bar(100)
    assert ab.shape == (101,)
    assert ab[0].item() == 1.0
    assert ab[-1].item() < 0.05
    # monotonically non-increasing
    diffs = ab[1:] - ab[:-1]
    assert (diffs <= 1e-6).all()


def test_noise_schedule_add_noise_shapes():
    sched = NoiseSchedule(num_steps=1000)
    x0 = torch.randn(2, 128, 50)
    t = torch.tensor([100, 500])
    x_t, noise, v = sched.add_noise(x0, t)
    assert x_t.shape == x0.shape == noise.shape == v.shape


def test_noise_schedule_t0_is_identity():
    """At t=0, alpha_bar=1 → x_t == x0 and v == noise (v-prediction)."""
    sched = NoiseSchedule(num_steps=1000)
    x0 = torch.randn(1, 4, 10)
    noise = torch.randn_like(x0)
    x_t, _, v = sched.add_noise(x0, torch.tensor([0]), noise=noise)
    assert torch.allclose(x_t, x0, atol=1e-5)
    assert torch.allclose(v, noise, atol=1e-5)


def test_decoder_forward_shape():
    cfg = DecoderConfig(n_mels=128, hidden=64, cond_dim=64, num_blocks=2, time_dim=64)
    dec = DiffusionDecoder(cfg)
    mel = torch.randn(2, 128, 40)
    t = torch.tensor([10, 200])
    cond = torch.randn(2, 64, 40)
    out = dec(mel, t, cond)
    assert out.shape == mel.shape


def test_decoder_resizes_cond_to_mel_T():
    cfg = DecoderConfig(n_mels=128, hidden=64, cond_dim=64, num_blocks=2, time_dim=64)
    dec = DiffusionDecoder(cfg)
    mel = torch.randn(1, 128, 40)
    t = torch.tensor([5])
    cond = torch.randn(1, 64, 30)  # mismatched T
    out = dec(mel, t, cond)
    assert out.shape == mel.shape


def test_one_step_backward():
    cfg = DecoderConfig(n_mels=128, hidden=32, cond_dim=32, num_blocks=2, time_dim=32)
    dec = DiffusionDecoder(cfg)
    sched = NoiseSchedule(num_steps=500)
    x0 = torch.randn(2, 128, 30)
    cond = torch.randn(2, 32, 30)
    t = sched.sample_random_t(2, device=torch.device("cpu"))
    x_t, _, v_target = sched.add_noise(x0, t)
    v_pred = dec(x_t, t, cond)
    loss = (v_pred - v_target).pow(2).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in dec.parameters())


def test_ddim_sample_shape():
    cfg = DecoderConfig(n_mels=128, hidden=32, cond_dim=32, num_blocks=2, time_dim=32)
    dec = DiffusionDecoder(cfg).eval()
    sched = NoiseSchedule(num_steps=200)
    sampler = DDIMSampler(sched)
    cond = torch.randn(1, 32, 20)
    out = sampler.sample(dec, cond, shape=(1, 128, 20), num_steps=5)
    assert out.shape == (1, 128, 20)
    assert torch.isfinite(out).all()
