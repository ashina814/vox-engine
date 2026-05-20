import torch

from vox.training.optim import OptimConfig, build_optimizer, build_scheduler, cosine_with_warmup_lambda


def test_warmup_linear_ramp():
    fn = cosine_with_warmup_lambda(warmup_steps=10, max_steps=100, min_lr_ratio=0.0)
    assert fn(0) == 0.0
    assert abs(fn(5) - 0.5) < 1e-6
    assert abs(fn(10) - 1.0) < 1e-6  # peak right after warmup


def test_cosine_decay_after_warmup():
    fn = cosine_with_warmup_lambda(warmup_steps=10, max_steps=100, min_lr_ratio=0.0)
    early = fn(20)
    mid = fn(55)
    late = fn(95)
    assert early > mid > late
    assert fn(150) == 0.0  # clamped to min_lr_ratio after max_steps


def test_min_lr_ratio_floor():
    fn = cosine_with_warmup_lambda(warmup_steps=0, max_steps=10, min_lr_ratio=0.1)
    assert abs(fn(10) - 0.1) < 1e-6
    assert abs(fn(1000) - 0.1) < 1e-6


def test_scheduler_steps_optimizer_lr():
    p = torch.nn.Linear(4, 4).parameters()
    cfg = OptimConfig(lr=1e-3, warmup_steps=5, max_steps=20)
    opt = build_optimizer(p, cfg)
    sch = build_scheduler(opt, cfg)
    initial_lr = opt.param_groups[0]["lr"]
    assert initial_lr == 0.0  # step=0 in warmup
    for _ in range(5):
        opt.step()
        sch.step()
    after_warmup = opt.param_groups[0]["lr"]
    assert abs(after_warmup - 1e-3) < 1e-9
