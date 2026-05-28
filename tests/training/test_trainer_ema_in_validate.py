"""Trainer.validate() must score the EMA weights, not the training weights.

Without this guard, val metrics track the noisier raw weights and the EMA
contribution is unmeasurable.
"""

import torch
from torch.utils.data import DataLoader, Dataset

from vox.models.conditioning.aggregator import AggregatorConfig
from vox.models.diffusion.decoder import DecoderConfig
from vox.models.vox_model import VoxModel, VoxModelConfig
from vox.training.optim import OptimConfig
from vox.training.trainer import TrainConfig, VoxTrainer


class _OneBatchDataset(Dataset):
    def __init__(self, T: int = 20):
        torch.manual_seed(0)
        self.item = {
            "mel": torch.randn(128, T),
            "content": torch.randn(768, T),
            "f0": torch.rand(T) * 400 + 100,
            "uv": torch.ones(T, dtype=torch.bool),
            "loudness": torch.rand(T),
            "style_id": torch.tensor(0, dtype=torch.long),
        }

    def __len__(self):
        return 4

    def __getitem__(self, i):
        return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in self.item.items()}


def _collate(batch):
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}
    out["mask"] = torch.ones(out["mel"].shape[0], out["mel"].shape[-1], dtype=torch.bool)
    return out


def _tiny_cfg() -> VoxModelConfig:
    return VoxModelConfig(
        hidden=32,
        n_styles=3,
        diffusion_steps=100,
        aggregator=AggregatorConfig(hidden=32, n_styles=3, gst_num_tokens=4, gst_num_heads=4),
        decoder=DecoderConfig(hidden=32, cond_dim=32, num_blocks=2, time_dim=32),
    )


def test_validate_uses_ema_weights():
    """If we mutate live weights far from the EMA snapshot, validate() must
    score against the EMA snapshot, not the mutated live weights."""
    torch.manual_seed(0)
    model = VoxModel(_tiny_cfg())
    loader = DataLoader(_OneBatchDataset(), batch_size=2, collate_fn=_collate)

    trainer = VoxTrainer(
        TrainConfig(
            max_steps=1,
            log_interval=10,
            val_interval=10,
            ckpt_interval=999,
            ema_decay=0.5,
            optim=OptimConfig(lr=1e-3, warmup_steps=0, max_steps=10),
        ),
        model=model,
        train_loader=loader,
        val_loader=loader,
    )

    # Baseline validation with EMA shadow == live weights.
    torch.manual_seed(0)
    val_baseline = trainer.validate()

    # Mutate live weights far away from the EMA snapshot.
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(5.0)

    # validate() must still produce the baseline number (because it swaps
    # EMA in). If it didn't swap, the mutated weights would be scored and
    # the loss would differ.
    torch.manual_seed(0)
    val_after = trainer.validate()
    assert (
        abs(val_after["val/total_loss"] - val_baseline["val/total_loss"]) < 1e-4
    ), f"validate() did not use EMA weights: baseline={val_baseline}, after={val_after}"


def test_validate_works_without_ema():
    """ema_decay=0 disables EMA; validate() should still run."""
    model = VoxModel(_tiny_cfg())
    loader = DataLoader(_OneBatchDataset(), batch_size=2, collate_fn=_collate)
    trainer = VoxTrainer(
        TrainConfig(
            max_steps=1,
            log_interval=10,
            val_interval=10,
            ckpt_interval=999,
            ema_decay=0.0,
            optim=OptimConfig(lr=1e-3, warmup_steps=0, max_steps=10),
        ),
        model=model,
        train_loader=loader,
        val_loader=loader,
    )
    out = trainer.validate()
    assert "val/total_loss" in out
    assert trainer.ema is None
