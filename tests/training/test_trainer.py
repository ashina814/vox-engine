"""Smoke test: tiny VoxModel + synthetic batches → loss decreases over a handful of steps."""
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from vox.models.conditioning.aggregator import AggregatorConfig
from vox.models.diffusion.decoder import DecoderConfig
from vox.models.vox_model import VoxModel, VoxModelConfig
from vox.training.logging import StdoutLogger
from vox.training.optim import OptimConfig
from vox.training.trainer import TrainConfig, VoxTrainer


class _FakeBatchDataset(Dataset):
    """Returns the same fake batch every time — overfitting target = noise floor."""

    def __init__(self, length: int = 100, T: int = 30, n_mels: int = 128, B: int = 1):
        torch.manual_seed(0)
        self.B, self.T, self.n_mels = B, T, n_mels
        self.length = length
        self.cached = {
            "mel": torch.randn(n_mels, T),
            "content": torch.randn(768, T),
            "f0": torch.rand(T) * 400 + 100,
            "uv": torch.ones(T, dtype=torch.bool),
            "loudness": torch.rand(T),
            "style_id": torch.tensor(0, dtype=torch.long),
        }

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.cached.items()}


def _collate(batch):
    out = {}
    for k in batch[0]:
        v = batch[0][k]
        if isinstance(v, torch.Tensor) and v.dim() > 0:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    B, T = out["mel"].shape[0], out["mel"].shape[-1]
    out["mask"] = torch.ones(B, T, dtype=torch.bool)
    return out


def _tiny_cfg() -> VoxModelConfig:
    return VoxModelConfig(
        hidden=32,
        n_styles=3,
        diffusion_steps=200,
        aggregator=AggregatorConfig(hidden=32, n_styles=3, gst_num_tokens=4, gst_num_heads=4),
        decoder=DecoderConfig(hidden=32, cond_dim=32, num_blocks=2, time_dim=32),
    )


def test_trainer_single_step():
    model = VoxModel(_tiny_cfg())
    loader = DataLoader(_FakeBatchDataset(length=4, B=1), batch_size=2, collate_fn=_collate)
    trainer = VoxTrainer(
        TrainConfig(max_steps=1, log_interval=1, val_interval=1000, ckpt_interval=1000,
                    optim=OptimConfig(lr=1e-3, warmup_steps=0, max_steps=10)),
        model=model,
        train_loader=loader,
    )
    batch = next(iter(loader))
    losses = trainer.training_step(batch)
    assert torch.isfinite(losses["total_loss"])


def test_trainer_runs_full_loop(tmp_path):
    model = VoxModel(_tiny_cfg())
    loader = DataLoader(_FakeBatchDataset(length=20, B=1), batch_size=2, collate_fn=_collate)
    trainer = VoxTrainer(
        TrainConfig(
            max_steps=5,
            log_interval=2,
            val_interval=1000,
            ckpt_interval=5,
            ckpt_dir=tmp_path,
            optim=OptimConfig(lr=1e-3, warmup_steps=0, max_steps=10),
        ),
        model=model,
        train_loader=loader,
    )
    trainer.train()
    assert trainer.global_step == 5
    # ckpt at step 5 was written
    ckpts = list(tmp_path.glob("*.pt"))
    assert len(ckpts) == 1


def test_trainer_checkpoint_roundtrip(tmp_path):
    model = VoxModel(_tiny_cfg())
    loader = DataLoader(_FakeBatchDataset(length=4, B=1), batch_size=1, collate_fn=_collate)
    trainer = VoxTrainer(
        TrainConfig(max_steps=2, log_interval=10, val_interval=10, ckpt_interval=10,
                    ckpt_dir=tmp_path,
                    optim=OptimConfig(lr=1e-3, warmup_steps=0, max_steps=10)),
        model=model,
        train_loader=loader,
    )
    trainer.train()
    path = trainer.save_checkpoint(42)

    model2 = VoxModel(_tiny_cfg())
    trainer2 = VoxTrainer(
        TrainConfig(max_steps=2, log_interval=10, val_interval=10, ckpt_interval=10,
                    ckpt_dir=tmp_path,
                    optim=OptimConfig(lr=1e-3, warmup_steps=0, max_steps=10)),
        model=model2,
        train_loader=loader,
    )
    step = trainer2.load_checkpoint(path)
    assert step == 42
    # Parameters must match exactly.
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


@pytest.mark.slow
def test_trainer_loss_decreases():
    """Overfit on a single fixed batch → loss should drop noticeably in ~50 steps."""
    torch.manual_seed(0)
    model = VoxModel(_tiny_cfg())
    loader = DataLoader(_FakeBatchDataset(length=4, B=1), batch_size=2, collate_fn=_collate)
    trainer = VoxTrainer(
        TrainConfig(max_steps=50, log_interval=200, val_interval=200, ckpt_interval=200,
                    optim=OptimConfig(lr=3e-3, warmup_steps=0, max_steps=50)),
        model=model,
        train_loader=loader,
        logger=StdoutLogger(),
    )
    # measure loss at step 0
    batch = next(iter(loader))
    initial = 0.0
    for _ in range(5):
        initial += float(model.training_step(batch)["total_loss"])
    initial /= 5

    trainer.train()

    final = 0.0
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            final += float(model.training_step(batch)["total_loss"])
        final /= 5

    assert final < initial * 0.95, f"loss didn't drop: initial={initial:.4f}, final={final:.4f}"
