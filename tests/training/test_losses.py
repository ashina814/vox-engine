import torch

from vox.training.losses import diffusion_loss, f0_consistency_loss, mel_l1_loss


def test_diffusion_loss_matches_mean_without_mask():
    a = torch.randn(2, 4, 10)
    b = torch.randn(2, 4, 10)
    expected = (a - b).pow(2).mean()
    assert torch.allclose(diffusion_loss(a, b, mask=None), expected)


def test_diffusion_loss_mask_zero_outside():
    """When mask=full, masked loss must equal unmasked loss."""
    a = torch.randn(1, 4, 10)
    b = torch.randn(1, 4, 10)
    mask = torch.ones(1, 10, dtype=torch.bool)
    assert torch.allclose(diffusion_loss(a, b), diffusion_loss(a, b, mask), atol=1e-6)


def test_diffusion_loss_mask_excludes_garbage_tail():
    a = torch.zeros(1, 4, 10)
    b = torch.zeros(1, 4, 10)
    b[..., 5:] = 1e6  # huge error in masked-out region
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[:, :5] = True
    loss = diffusion_loss(a, b, mask)
    assert loss.item() < 1e-3


def test_mel_l1_zero_for_identical():
    x = torch.randn(2, 8, 30)
    assert mel_l1_loss(x, x).item() == 0.0


def test_f0_consistency_no_extractor_returns_zero():
    mel = torch.randn(1, 128, 20)
    f0 = torch.rand(1, 20)
    loss = f0_consistency_loss(mel, f0, f0_extractor=None)
    assert loss.item() == 0.0


def test_f0_consistency_with_stub_extractor():
    mel = torch.randn(1, 128, 20)
    f0_target = torch.full((1, 20), 220.0)

    def stub(mel):
        return torch.full((mel.shape[0], mel.shape[-1]), 220.0)

    assert f0_consistency_loss(mel, f0_target, stub).item() == 0.0
