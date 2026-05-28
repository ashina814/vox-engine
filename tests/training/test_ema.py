import torch
from torch import nn

from vox.training.ema import ExponentialMovingAverage


def _tiny():
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))


def test_ema_initial_shadow_equals_params():
    model = _tiny()
    ema = ExponentialMovingAverage(model, decay=0.99)
    for name, p in model.named_parameters():
        assert torch.equal(ema.shadow[name], p.detach())


def test_ema_update_moves_shadow_toward_current():
    model = _tiny()
    ema = ExponentialMovingAverage(model, decay=0.5)
    # Push params far from shadow.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(10.0)
    ema.update(model)
    # decay=0.5 with old_shadow == p - 10 (we just added 10):
    # shadow = 0.5*(p-10) + 0.5*p = p - 5
    for name, p in model.named_parameters():
        assert torch.allclose(ema.shadow[name], p - 5.0, atol=1e-6)


def test_ema_swap_in_then_out_restores_original():
    model = _tiny()
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(5.0)
    originals = {n: p.detach().clone() for n, p in model.named_parameters()}
    with ema.swap_in(model):
        # Inside the context: model has the OLD shadow values, not the +5 ones.
        for n, p in model.named_parameters():
            assert not torch.allclose(p, originals[n])
    # After: params restored.
    for n, p in model.named_parameters():
        assert torch.allclose(p, originals[n], atol=1e-6)


def test_ema_state_dict_roundtrip():
    model = _tiny()
    ema = ExponentialMovingAverage(model, decay=0.9)
    state = ema.state_dict()
    ema2 = ExponentialMovingAverage(_tiny(), decay=0.9)
    ema2.load_state_dict(state)
    for k in ema.shadow:
        assert torch.allclose(ema.shadow[k], ema2.shadow[k])


def test_ema_clone_model_returns_independent_copy():
    model = _tiny()
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(7.0)
    ema_model = ema.clone_model(model)
    # The clone holds the shadow (old) values.
    for (na, pa), (nb, pb) in zip(model.named_parameters(), ema_model.named_parameters()):
        assert not torch.allclose(pa, pb)
    # Mutating the clone must not touch the live model.
    with torch.no_grad():
        for p in ema_model.parameters():
            p.zero_()
    for p in model.parameters():
        assert p.abs().sum() > 0


def test_ema_invalid_decay_raises():
    import pytest

    model = _tiny()
    with pytest.raises(ValueError):
        ExponentialMovingAverage(model, decay=1.5)
    with pytest.raises(ValueError):
        ExponentialMovingAverage(model, decay=0.0)


# ----- decay warmup ---------------------------------------------------


def test_ema_warmup_ramps_decay_linearly():
    """current_decay() must rise from 0 toward target over warmup_steps."""
    model = _tiny()
    ema = ExponentialMovingAverage(model, decay=0.999, warmup_steps=100)
    # Before any update: about 1 / (warmup+1) of target
    initial = ema.current_decay()
    assert initial < 0.05  # well below target
    # Simulate updates and check monotonic approach to target.
    decays = []
    for _ in range(120):
        decays.append(ema.current_decay())
        ema.update(model)
    # After warmup, decay must clip to target.
    assert abs(decays[-1] - 0.999) < 1e-6
    # Monotone non-decreasing within warmup window.
    assert all(decays[i] <= decays[i + 1] + 1e-9 for i in range(100))


def test_ema_warmup_disabled_by_default():
    """warmup_steps=0 → decay is constant from step 1, same behaviour as before."""
    model = _tiny()
    ema = ExponentialMovingAverage(model, decay=0.7, warmup_steps=0)
    assert ema.current_decay() == 0.7
    ema.update(model)
    assert ema.current_decay() == 0.7


def test_ema_warmup_negative_raises():
    import pytest

    with pytest.raises(ValueError):
        ExponentialMovingAverage(_tiny(), decay=0.99, warmup_steps=-1)


def test_ema_state_dict_persists_step_counter():
    """When warmup is in progress, the step counter must round-trip."""
    model = _tiny()
    ema = ExponentialMovingAverage(model, decay=0.999, warmup_steps=200)
    for _ in range(50):
        ema.update(model)
    mid_decay = ema.current_decay()

    state = ema.state_dict()
    ema2 = ExponentialMovingAverage(_tiny(), decay=0.999, warmup_steps=200)
    ema2.load_state_dict(state)
    assert ema2._step == ema._step
    assert abs(ema2.current_decay() - mid_decay) < 1e-9


def test_ema_load_legacy_flat_state_dict():
    """Old checkpoints saved tensors under the flat ``{name: tensor}`` layout.
    The new loader must still accept those."""
    model = _tiny()
    legacy_state = {
        name: p.detach().clone() + 1.0 for name, p in model.named_parameters() if p.requires_grad
    }
    ema = ExponentialMovingAverage(model, decay=0.99)
    ema.load_state_dict(legacy_state)
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert torch.allclose(ema.shadow[n], p + 1.0)
