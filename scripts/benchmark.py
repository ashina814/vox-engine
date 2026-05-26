"""Hydra benchmark CLI.

Renders every chunk in the eval split through ``InferencePipeline``, then
scores it against the ground truth with the metrics in ``vox.evaluation``.
Outputs a JSON next to the index.

Usage:
    uv run python scripts/benchmark.py data=opensinger benchmark.max_samples=20
    uv run python scripts/benchmark.py ckpt_path=ckpts/step_00010000.pt
"""

from __future__ import annotations

from pathlib import Path

import hydra
import soundfile as sf
import torch
from omegaconf import DictConfig

from vox.data.dataset import VoxDataset
from vox.data.features.f0 import F0Extractor
from vox.data.features.loudness import a_weighted_rms
from vox.data.features.uv import compute_uv
from vox.evaluation.benchmark import BenchmarkConfig, BenchmarkRunner
from vox.inference.pipeline import InferencePipeline, InferenceRequest
from vox.models.vox_model import VoxModel, VoxModelConfig


def _make_render_fn(pipeline: InferencePipeline, f0_ext, sr: int, hop: int):
    """Render fn that reads a Dataset row, runs inference, and aligns features."""

    def render(item: dict) -> tuple:
        # item comes from VoxDataset.__getitem__; original wav is implied via source_wav field.
        source_wav = item.get("source_wav")
        if source_wav is None or not Path(source_wav).exists():
            raise FileNotFoundError(f"source_wav missing for chunk {item.get('chunk_id')}")
        ref, _sr_in = sf.read(source_wav, dtype="float32", always_2d=False)
        if ref.ndim > 1:
            ref = ref.mean(axis=-1)

        req = InferenceRequest(
            input_wav=ref,
            sr=sr,
            target_key=None,  # evaluation = no auto-tune
            style_weights=(1.0, 0.0, 0.0),  # neutral
        )
        result = pipeline(req)
        pred = result.output_wav

        # Score-side F0 / UV recomputed from each waveform (consistent extractor).
        f0_ref_t, _ = f0_ext(torch.from_numpy(ref).float())
        f0_pred_t, _ = f0_ext(torch.from_numpy(pred).float())
        loud_ref = a_weighted_rms(torch.from_numpy(ref).float(), hop=hop, sr=sr)
        loud_pred = a_weighted_rms(torch.from_numpy(pred).float(), hop=hop, sr=sr)
        uv_ref = compute_uv(f0_ref_t, loud_ref).numpy().astype(bool)
        uv_pred = compute_uv(f0_pred_t, loud_pred).numpy().astype(bool)

        return (
            ref,
            pred,
            f0_ref_t.numpy(),
            f0_pred_t.numpy(),
            uv_ref,
            uv_pred,
            int(item["style_id"]),
        )

    return render


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    sr = int(cfg.audio.sr)
    hop = int(cfg.audio.hop)

    model = VoxModel(VoxModelConfig())
    ckpt = cfg.get("ckpt_path")
    if ckpt:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)

    f0 = F0Extractor(backend="torchcrepe", hop=hop, sr=sr)
    # Content extractor is heavy; skip if the user asks for a smoke run.
    skip_content = bool(cfg.get("benchmark", {}).get("skip_content", False))
    if skip_content:

        def content_fn(wav, sr):
            T = wav.shape[-1] // hop + 1
            return torch.randn(768, T)

    else:
        from vox.data.features.content import ContentVecExtractor

        cv = ContentVecExtractor()
        content_fn = lambda wav, sr: cv(wav, src_sr=sr)  # noqa: E731

    pipeline = InferencePipeline(model=model, f0_fn=f0, content_fn=content_fn, sr=sr, hop=hop)
    runner = BenchmarkRunner(
        BenchmarkConfig(
            sr=sr,
            hop=hop,
            max_samples=cfg.get("benchmark", {}).get("max_samples", None),
            n_styles=int(cfg.data.get("n_styles", 3)),
        )
    )

    dataset = VoxDataset(cfg.data.index_path, split="val")
    items = [
        {**dataset[i], "source_wav": dataset.df.iloc[i].get("source_wav")}
        for i in range(len(dataset))
    ]
    res = runner.run(items, _make_render_fn(pipeline, f0, sr, hop))

    out_json = Path(cfg.get("benchmark", {}).get("out_json", "benchmark.json"))
    res.to_json(out_json)
    print(f"Wrote {out_json}")
    print(res.metrics)


if __name__ == "__main__":
    main()
