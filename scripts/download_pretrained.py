"""Download external pretrained checkpoints into ``models/pretrained/``.

Components fetched on demand:

  * **ContentVec-768** (``lengyue233/content-vec-best``) — pulled via the
    HuggingFace transformers cache. Used by ``ContentVecExtractor`` and the
    inference pipeline.
  * **NSF-HiFiGAN** vocoder checkpoint — Phase B uses a real ckpt; for Phase A
    the placeholder generator in ``NSFHifiGANWrapper`` already works without a
    download. This script writes a tiny stub note when no real ckpt is given,
    so the directory layout is correct from day one.

Usage:
    uv run python scripts/download_pretrained.py
    uv run python scripts/download_pretrained.py --skip-vocoder
    uv run python scripts/download_pretrained.py --vocoder-url https://example/nsf_hifigan.zip
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_CONTENTVEC = "lengyue233/content-vec-best"
DEFAULT_PRETRAINED_DIR = Path("models/pretrained")


def download_contentvec(model_id: str = DEFAULT_CONTENTVEC, target: Path | None = None) -> Path:
    """Materialise the ContentVec model into the HF cache (or a local snapshot)."""
    print(f"[contentvec] loading {model_id} ...")
    from transformers import HubertModel, Wav2Vec2FeatureExtractor

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    model = HubertModel.from_pretrained(model_id, output_hidden_states=True)

    if target is not None:
        target.mkdir(parents=True, exist_ok=True)
        extractor.save_pretrained(target)
        model.save_pretrained(target)
        print(f"[contentvec] snapshot saved at {target}")
        return target

    cache_dir = Path(model.config._name_or_path).parent  # type: ignore[attr-defined]
    print(f"[contentvec] cached. params={sum(p.numel() for p in model.parameters()):,}")
    return cache_dir


def download_vocoder(url: str, dest: Path) -> Path:
    """Download an NSF-HiFiGAN bundle (zip or single file) and extract."""
    dest.mkdir(parents=True, exist_ok=True)
    fname = url.split("/")[-1].split("?")[0] or "nsf_hifigan.bin"
    archive = dest / fname
    print(f"[vocoder] downloading {url} -> {archive}")
    urllib.request.urlretrieve(url, archive)

    if archive.suffix == ".zip":
        print(f"[vocoder] extracting {archive}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        archive.unlink()

    print(f"[vocoder] done -> {dest}")
    return dest


def write_vocoder_note(dest: Path) -> None:
    """Drop a README explaining how to swap in a real NSF-HiFiGAN ckpt later."""
    dest.mkdir(parents=True, exist_ok=True)
    note = dest / "README.md"
    note.write_text(
        "# NSF-HiFiGAN checkpoint\n\n"
        "Phase A uses the placeholder generator in "
        "`src/vox/models/vocoder/nsf_hifigan.py`. Phase B drops a real NSF-HiFiGAN\n"
        "checkpoint here (e.g. `nsf_hifigan.bin` or `generator.pt`).\n\n"
        "## How to populate\n\n"
        "```bash\n"
        "uv run python scripts/download_pretrained.py \\\n"
        "    --vocoder-url <official_zip_url>\n"
        "```\n\n"
        "Then pass the path to `VoxModelConfig.vocoder_ckpt` or\n"
        "`scripts/infer.py +vocoder_ckpt=models/pretrained/nsf_hifigan/generator.pt`.\n"
    )
    print(f"[vocoder] wrote note -> {note} (no real ckpt yet)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contentvec-id", default=DEFAULT_CONTENTVEC, help="HF model id for ContentVec"
    )
    parser.add_argument(
        "--contentvec-target",
        type=Path,
        default=None,
        help="Optional local snapshot directory (defaults to HF cache only)",
    )
    parser.add_argument("--skip-contentvec", action="store_true")
    parser.add_argument(
        "--vocoder-url",
        default=None,
        help="HTTPS URL to an NSF-HiFiGAN zip / single-file ckpt",
    )
    parser.add_argument("--skip-vocoder", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_PRETRAINED_DIR,
        help="Root directory for downloaded artefacts",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_contentvec:
        target = args.contentvec_target or (args.out_dir / "contentvec")
        download_contentvec(model_id=args.contentvec_id, target=target)
    else:
        print("[contentvec] skipped")

    voc_dir = args.out_dir / "nsf_hifigan"
    if args.skip_vocoder:
        print("[vocoder] skipped")
    elif args.vocoder_url:
        download_vocoder(args.vocoder_url, voc_dir)
    else:
        write_vocoder_note(voc_dir)

    print("\n[done] pretrained artefacts under", args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
