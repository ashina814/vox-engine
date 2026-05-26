"""Hydra training CLI.

Resolves a config under ``configs/train/{debug,smoke,production}.yaml`` plus a
``configs/data/*.yaml`` profile, builds the dataloaders, the VoxModel and
VoxTrainer, then runs ``trainer.train()``.

Usage:
    uv run python scripts/train.py train=debug data=synthetic
    uv run python scripts/train.py train=smoke data=opensinger
    uv run python scripts/train.py train=production data=self \
        +ckpt_path=ckpts/last.pt

CLI overrides (Hydra):
    train.max_steps=2000  train.batch_size=8  +train.optim.lr=1e-4
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from vox.data.dataset import VoxDataset, collate_fn
from vox.models.conditioning.aggregator import AggregatorConfig
from vox.models.diffusion.decoder import DecoderConfig
from vox.models.vox_model import VoxModel, VoxModelConfig
from vox.training.logging import build_logger
from vox.training.optim import OptimConfig
from vox.training.trainer import TrainConfig, VoxTrainer


def _build_model(cfg: DictConfig) -> VoxModel:
    """Resolve the model config from train+audio sections."""
    model_cfg = cfg.train.get("model", {})
    hidden = int(model_cfg.get("hidden", 256))
    num_blocks = int(model_cfg.get("num_blocks", 8))
    n_mels = int(cfg.audio.n_mels)
    n_styles = int(cfg.data.get("n_styles", 3))

    vox_cfg = VoxModelConfig(
        n_mels=n_mels,
        hop=int(cfg.audio.hop),
        hidden=hidden,
        n_styles=n_styles,
        aggregator=AggregatorConfig(hidden=hidden, n_styles=n_styles, n_mels=n_mels),
        decoder=DecoderConfig(n_mels=n_mels, hidden=hidden, cond_dim=hidden, num_blocks=num_blocks),
        vocoder_ckpt=cfg.get("vocoder_ckpt"),
    )
    return VoxModel(vox_cfg)


def _build_loaders(cfg: DictConfig, batch_size: int) -> tuple[DataLoader, DataLoader | None]:
    """Build train + val DataLoaders from the data profile's index.parquet."""
    index_path = Path(cfg.data.index_path)
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found — run `scripts/preprocess.py data={cfg.data.name}` first."
        )

    train_ds = VoxDataset(index_path, split="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=int(cfg.train.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    val_loader: DataLoader | None = None
    try:
        val_ds = VoxDataset(index_path, split="val")
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )
    except Exception:
        pass

    return train_loader, val_loader


def _build_optim_cfg(cfg: DictConfig, max_steps: int) -> OptimConfig:
    opt = cfg.train.get("optimizer", {}) or {}
    sched = cfg.train.get("scheduler", {}) or {}
    return OptimConfig(
        lr=float(opt.get("lr", 2.0e-4)),
        betas=tuple(opt.get("betas", (0.9, 0.98))),
        weight_decay=float(opt.get("weight_decay", 0.01)),
        warmup_steps=int(sched.get("warmup_steps", 0)),
        max_steps=max_steps,
        min_lr_ratio=float(sched.get("min_lr_ratio", 0.01)),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}, cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[train] gpu={torch.cuda.get_device_name(0)}")

    print(f"[train] config:\n{OmegaConf.to_yaml(cfg.train)}")

    model = _build_model(cfg)
    batch_size = int(cfg.train.get("batch_size", 4))
    train_loader, val_loader = _build_loaders(cfg, batch_size)
    max_steps = int(cfg.train.max_steps)

    train_cfg = TrainConfig(
        max_steps=max_steps,
        log_interval=int(cfg.train.log_interval),
        val_interval=int(cfg.train.val_interval),
        ckpt_interval=int(cfg.train.ckpt_interval),
        grad_clip=float(cfg.train.get("grad_clip", 1.0)),
        val_batches=int(cfg.train.get("val_batches", 10)),
        ckpt_dir=Path(cfg.train.get("ckpt_dir", "checkpoints")),
        optim=_build_optim_cfg(cfg, max_steps),
    )

    logger_kind = cfg.train.get("logger", "stdout")
    logger = build_logger(logger_kind)

    trainer = VoxTrainer(
        cfg=train_cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        device=device,
    )

    if cfg.get("ckpt_path"):
        step = trainer.load_checkpoint(cfg.ckpt_path)
        print(f"[train] resumed from step {step}")

    trainer.train()
    print(f"[train] done, final step={trainer.global_step}")


if __name__ == "__main__":
    main()
