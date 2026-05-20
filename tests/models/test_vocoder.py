import pytest
import torch

from vox.models.vocoder.nsf_hifigan import NSFHifiGANWrapper


def test_placeholder_when_no_ckpt():
    voc = NSFHifiGANWrapper(ckpt_path=None)
    assert voc.is_placeholder


def test_vocoder_forward_shape():
    voc = NSFHifiGANWrapper(ckpt_path=None, freeze=False)
    mel = torch.randn(2, 128, 40)
    f0 = torch.rand(2, 40) * 400 + 100
    wav = voc(mel, f0)
    assert wav.shape[0] == 2
    # T_wav is roughly T_mel * hop with the placeholder upsampler.
    assert wav.shape[1] >= 40 * 512 - 1024
    assert wav.shape[1] <= 40 * 512 + 1024


def test_vocoder_freeze():
    voc = NSFHifiGANWrapper(ckpt_path=None, freeze=True)
    assert all(not p.requires_grad for p in voc.parameters())
    voc.set_frozen(False)
    assert all(p.requires_grad for p in voc.parameters())


def test_vocoder_rejects_mel_mismatch():
    voc = NSFHifiGANWrapper(ckpt_path=None, freeze=False)
    bad_mel = torch.randn(1, 64, 40)  # wrong n_mels
    f0 = torch.rand(1, 40)
    with pytest.raises(ValueError):
        voc(bad_mel, f0)


def test_vocoder_rejects_T_mismatch():
    voc = NSFHifiGANWrapper(ckpt_path=None, freeze=False)
    mel = torch.randn(1, 128, 40)
    f0 = torch.rand(1, 30)
    with pytest.raises(ValueError):
        voc(mel, f0)
