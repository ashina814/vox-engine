# Training

## Step semantics

`VoxModel.training_step(batch)` performs:

1. Build conditioning via `ConditionAggregator` + `WhisperAwareConditioning`.
2. Sample `t ~ Uniform[1, num_steps]` per batch item.
3. `x_t, _, v_target = NoiseSchedule.add_noise(mel, t)`.
4. `v_pred = DiffusionDecoder(x_t, t, cond)`.
5. Masked L2 between `v_pred` and `v_target` over valid frames.

The mask divisor is `mask.sum() * n_mels` so loss magnitudes are comparable
to an unmasked `.mean()` reduction (validated by `tests/training/test_losses.py`).

## Losses (`vox.training.losses`)

| Loss | When to use |
|---|---|
| `diffusion_loss(v_pred, v_target, mask)` | primary objective |
| `mel_l1_loss(mel_pred, mel_target, mask)` | optional auxiliary (design spec weight 0.5) |
| `f0_consistency_loss(mel_pred, f0_target, f0_extractor)` | dependency-injected — pass `None` to disable |

## Optimizer

`vox.training.optim.OptimConfig` (design defaults):

```python
OptimConfig(
    lr=2.0e-4,
    betas=(0.9, 0.98),
    weight_decay=0.01,
    warmup_steps=2000,
    max_steps=100_000,
    min_lr_ratio=0.01,
)
```

`build_scheduler(opt, cfg)` returns a `LambdaLR` that does **linear warmup →
cosine decay → min_lr_ratio floor**.

## Trainer (`vox.training.trainer.VoxTrainer`)

Minimal main loop:

```python
trainer = VoxTrainer(
    cfg=TrainConfig(max_steps=100_000, log_interval=100, val_interval=1000,
                    ckpt_interval=5000, grad_clip=1.0, ckpt_dir="ckpts",
                    optim=OptimConfig()),
    model=VoxModel(cfg),
    train_loader=DataLoader(...),
    val_loader=DataLoader(...),
    logger=StdoutLogger(),
)
trainer.train()
```

Per step:

- Moves batch to device.
- Runs `model.training_step`.
- `optimizer.zero_grad(set_to_none=True)`, `loss.backward()`,
  `clip_grad_norm_(grad_clip)`, `optimizer.step()`, `scheduler.step()`.
- Logs every `log_interval` (loss + current LR).
- Validates every `val_interval` (averages `model.training_step` losses across
  `val_batches` items).
- Saves every `ckpt_interval` to `ckpt_dir/step_########.pt`.

## Checkpoints

```
{
  "step": int,
  "model": state_dict,
  "optimizer": state_dict,
  "scheduler": state_dict,
}
```

Round-trip is tested (`test_trainer_checkpoint_roundtrip`). Reload with
`VoxTrainer.load_checkpoint(path)`.

## Logging backends

The `Logger` Protocol is implemented by three classes; pick via
`build_logger(kind)`:

| kind | Class | Extra deps |
|---|---|---|
| `"stdout"` | `StdoutLogger` | — (always available) |
| `"tensorboard"` | `TensorBoardLogger` | `tensorboard` |
| `"wandb"` | `WandbLogger` | `wandb` |

Imports are lazy so missing optional deps don't block CI.

## CLI (not yet written — placeholder)

A future `scripts/train.py` will resolve `configs/train/{debug,smoke,production}.yaml`
into the dataclasses above and call `trainer.train()`. The trainer itself is
fully usable from a notebook today.
