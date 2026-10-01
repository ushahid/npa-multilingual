#!/usr/bin/env python3
"""End-to-end automation: build indexes + embedding caches (all models) for CC12M + XM3600,
then generate similarity plots.

This script is the orchestration layer. It calls the existing scripts we already have:
  - cc12m_dataset.py  (index/text/image)
  - xm3600_dataset.py (index/text/image)
  - cc12m_xm3600_sims.py (plots)

Defaults are set for our plan:
- CC12M: build embeddings for the first 1.6M samples
- XM3600: build embeddings for the full dataset
- Images resized via processors with --image_size 300
- Similarity plots for anchors [0,100,200,300,400] and languages [en,fr,hi,te,mi] + "all" (ANY)

Re-run friendly:
- The dataset scripts will skip a model cache if its DONE marker exists.
- The sims script will overwrite plots if run again.

Example:
  python run_all_embeddings_and_sims.py \
    --cc_wds_dir /path/to/cc12m/shards \
    --cc_cache_dir /path/to/cc_cache \
    --xm_captions /path/to/xm3600/captions.jsonl \
    --xm_images_dir /path/to/xm3600/unpackedImages \
    --xm_cache_dir /path/to/xm_cache \
    --plots_out_root /path/to/plots \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# --- default model matrix (keys supported by our dataset scripts) ---
TEXT_MODELS_DEFAULT = [
    "nllb-200-3.3B",
    "bge-m3",
    "qwen3-embed-8b",
]

IMAGE_MODELS_DEFAULT = [
    "dinov3-vit7b16",
    "clip-vit-bigg14",
    "c-radio-v4-h",
]

# We can override globally with --txt_batch_override / --img_batch_override.
TEXT_BATCH_DEFAULT: Dict[str, int] = {
    "nllb-200-3.3B": 16,
    "bge-m3": 64,
    "qwen3-embed-8b": 4,
}

IMAGE_BATCH_DEFAULT: Dict[str, int] = {
    "dinov3-vit7b16": 4,
    "clip-vit-bigg14": 32,
    "c-radio-v4-h": 16,
}


def _stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.=+" else "_" for c in s).strip("_") or "model"


def _flatten_list(args_list: Optional[List[str]], default: List[str]) -> List[str]:
    """Accept either: --x a b c  OR  --x a,b,c"""
    if not args_list:
        return list(default)
    out: List[str] = []
    for x in args_list:
        parts = [p.strip() for p in x.split(",") if p.strip()]
        out.extend(parts)
    return out


def run_cmd(cmd: List[str], *, log_path: Path, env: Dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print("[CMD]", shlex.join(cmd))
    print("[LOG]", str(log_path))
    print("=" * 100)

    with log_path.open("w", encoding="utf-8") as f:
        f.write("# " + shlex.join(cmd) + "\n")
        f.write(f"# started: {_dt.datetime.now().isoformat()}\n\n")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        assert proc.stdout is not None

        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)

        ret = proc.wait()
        f.write(f"\n# exit_code: {ret}\n")

    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}: {shlex.join(cmd)}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("run_all_embeddings_and_sims")

    ap.add_argument("--cc_wds_dir", required=True, help="CC12M WebDataset shard dir (cc12m-train-*.tar)")
    ap.add_argument("--cc_cache_dir", required=True, help="CC12M cache_dir root")

    ap.add_argument("--xm_captions", required=True, help="XM3600 captions.jsonl")
    ap.add_argument("--xm_images_dir", required=True, help="XM3600 images root dir")
    ap.add_argument("--xm_cache_dir", required=True, help="XM3600 cache_dir root")

    ap.add_argument("--plots_out_root", required=True, help="Where to write plot PNGs / logs")

    ap.add_argument("--device", default="cuda:0", help="Device for embedding generation")

    ap.add_argument("--infer_dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="Inference dtype")
    ap.add_argument("--save_dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="Saved embedding dtype")

    ap.add_argument("--cc_max_images", type=int, default=1_600_000, help="Limit CC12M to first N samples")
    ap.add_argument("--cc_pattern", default="cc12m-train-*.tar", help="Glob pattern inside cc_wds_dir")

    ap.add_argument("--image_size", type=int, default=304, help="Fixed resize size passed to dataset scripts")
    ap.add_argument("--max_length", type=int, default=256, help="Max token length for text encoders")

    ap.add_argument("--text_models", nargs="*", default=None, help="Text models (preset key or HF repo); default = all presets")
    ap.add_argument("--image_models", nargs="*", default=None, help="Image models (preset key or HF repo); default = all presets")

    ap.add_argument("--txt_batch_override", type=int, default=None, help="Override ALL text batch sizes")
    ap.add_argument("--img_batch_override", type=int, default=None, help="Override ALL image batch sizes")

    ap.add_argument("--anchors", type=int, nargs="*", default=[0, 100, 200, 300, 400], help="XM3600 anchor image indices")
    ap.add_argument(
        "--languages",
        type=str,
        nargs="*",
        default=["en", "fr", "hi", "te", "mi", "all"],
        help="Preferred XM3600 caption languages. Include 'all' to also run ANY-caption plots.",
    )

    ap.add_argument("--top_k", type=int, default=1000, help="Top-k CC images for text-sim subset")
    ap.add_argument("--bins", type=int, default=80, help="Histogram bins")

    ap.add_argument("--plot_device", default=None, help="Device for similarity plots (defaults to --device)")
    ap.add_argument("--plot_dtype", default="float32", choices=["float16", "bfloat16", "float32"], help="Compute dtype for plots")

    ap.add_argument("--skip_missing", action="store_true", help="For plots: skip (model,modality) combos missing in caches")

    ap.add_argument("--skip_cc", action="store_true", help="Skip all CC12M work")
    ap.add_argument("--skip_xm", action="store_true", help="Skip all XM3600 work")
    ap.add_argument("--skip_index", action="store_true", help="Skip index building (both datasets)")
    ap.add_argument("--skip_embeddings", action="store_true", help="Skip embedding generation (both datasets)")
    ap.add_argument("--skip_plots", action="store_true", help="Skip similarity plots")

    ap.add_argument("--cc12m_script", default="/mnt/data/shared/npa-multilingual/scripts/cc12m_dataset.py", help="Path to cc12m_dataset.py")
    ap.add_argument("--xm3600_script", default="/mnt/data/shared/npa-multilingual/scripts/xm3600_dataset.py", help="Path to xm3600_dataset.py")
    ap.add_argument("--sims_script", default="/mnt/data/shared/npa-multilingual/scripts/cc12m_xm3600_sims.py", help="Path to cc12m_xm3600_sims.py")

    ap.add_argument(
        "--log_dir",
        default=None,
        help="Where to write orchestration logs (default: <plots_out_root>/logs/<timestamp>/)",
    )
    ap.add_argument(
        "--cuda_alloc_conf",
        default="expandable_segments:True,max_split_size_mb:128",
        help="Value for PYTORCH_CUDA_ALLOC_CONF (set empty string to disable)",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    py = sys.executable

    # environment passed to all subprocesses
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.cuda_alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf

    log_root = Path(args.log_dir or os.path.join(args.plots_out_root, "logs", _stamp()))
    log_root.mkdir(parents=True, exist_ok=True)

    # validate scripts exist
    for p in [args.cc12m_script, args.xm3600_script, args.sims_script]:
        if not Path(p).is_file():
            raise FileNotFoundError(f"Script not found: {p}")

    text_models = _flatten_list(args.text_models, TEXT_MODELS_DEFAULT)
    image_models = _flatten_list(args.image_models, IMAGE_MODELS_DEFAULT)

    # ---------------- CC12M ----------------
    if not args.skip_cc:
        if not args.skip_index:
            run_cmd(
                [
                    py,
                    args.cc12m_script,
                    "index",
                    "--cache_dir",
                    args.cc_cache_dir,
                    "--wds_dir",
                    args.cc_wds_dir,
                    "--pattern",
                    args.cc_pattern,
                    "--max_images",
                    str(args.cc_max_images),
                ],
                log_path=log_root / "01_cc12m_index.log",
                env=env,
            )

        if not args.skip_embeddings:
            # text
            for m in text_models:
                bs = args.txt_batch_override if args.txt_batch_override is not None else TEXT_BATCH_DEFAULT.get(m, 32)
                run_cmd(
                    [
                        py,
                        args.cc12m_script,
                        "text",
                        "--cache_dir",
                        args.cc_cache_dir,
                        "--text_model",
                        m,
                        "--device",
                        args.device,
                        "--infer_dtype",
                        args.infer_dtype,
                        "--save_dtype",
                        args.save_dtype,
                        "--max_length",
                        str(args.max_length),
                        "--txt_batch",
                        str(bs),
                    ],
                    log_path=log_root / f"02_cc12m_text_{_sanitize(m)}.log",
                    env=env,
                )

            # image
            for m in image_models:
                bs = args.img_batch_override if args.img_batch_override is not None else IMAGE_BATCH_DEFAULT.get(m, 16)
                run_cmd(
                    [
                        py,
                        args.cc12m_script,
                        "image",
                        "--cache_dir",
                        args.cc_cache_dir,
                        "--wds_dir",
                        args.cc_wds_dir,
                        "--pattern",
                        args.cc_pattern,
                        "--max_images",
                        str(args.cc_max_images),
                        "--image_model",
                        m,
                        "--device",
                        args.device,
                        "--infer_dtype",
                        args.infer_dtype,
                        "--save_dtype",
                        args.save_dtype,
                        "--img_batch",
                        str(bs),
                        "--image_size",
                        str(args.image_size),
                    ],
                    log_path=log_root / f"03_cc12m_image_{_sanitize(m)}.log",
                    env=env,
                )

    # ---------------- XM3600 ----------------
    if not args.skip_xm:
        if not args.skip_index:
            run_cmd(
                [
                    py,
                    args.xm3600_script,
                    "index",
                    "--captions",
                    args.xm_captions,
                    "--images",
                    args.xm_images_dir,
                    "--cache_dir",
                    args.xm_cache_dir,
                ],
                log_path=log_root / "04_xm3600_index.log",
                env=env,
            )

        if not args.skip_embeddings:
            # text
            for m in text_models:
                bs = args.txt_batch_override if args.txt_batch_override is not None else TEXT_BATCH_DEFAULT.get(m, 32)
                run_cmd(
                    [
                        py,
                        args.xm3600_script,
                        "text",
                        "--cache_dir",
                        args.xm_cache_dir,
                        "--text_model",
                        m,
                        "--device",
                        args.device,
                        "--infer_dtype",
                        args.infer_dtype,
                        "--save_dtype",
                        args.save_dtype,
                        "--max_length",
                        str(args.max_length),
                        "--txt_batch",
                        str(bs),
                    ],
                    log_path=log_root / f"05_xm3600_text_{_sanitize(m)}.log",
                    env=env,
                )

            # image
            for m in image_models:
                bs = args.img_batch_override if args.img_batch_override is not None else IMAGE_BATCH_DEFAULT.get(m, 16)
                run_cmd(
                    [
                        py,
                        args.xm3600_script,
                        "image",
                        "--cache_dir",
                        args.xm_cache_dir,
                        "--image_model",
                        m,
                        "--device",
                        args.device,
                        "--infer_dtype",
                        args.infer_dtype,
                        "--save_dtype",
                        args.save_dtype,
                        "--img_batch",
                        str(bs),
                        "--image_size",
                        str(args.image_size),
                    ],
                    log_path=log_root / f"06_xm3600_image_{_sanitize(m)}.log",
                    env=env,
                )

    # ---------------- Similarity plots ----------------
    if not args.skip_plots:
        plot_device = args.plot_device or args.device

        # treat "all" as ANY-caption run (i.e., omit --anchor_language)
        langs_raw = [l.strip() for l in (args.languages or []) if l.strip()]
        include_all = any(l.lower() == "all" for l in langs_raw)
        langs = [l for l in langs_raw if l.lower() != "all"]

        # Build base plot command
        base_cmd: List[str] = [
            py,
            args.sims_script,
            "--xm_cache_dir",
            args.xm_cache_dir,
            "--cc_cache_dir",
            args.cc_cache_dir,
            "--text_model",
            *text_models,
            "--image_model",
            *image_models,
            "--xm_image_idx",
            *[str(x) for x in args.anchors],
            "--top_k",
            str(args.top_k),
            "--bins",
            str(args.bins),
            "--device",
            plot_device,
            "--dtype",
            args.plot_dtype,
            "--out_root",
            args.plots_out_root,
        ]

        if args.skip_missing:
            base_cmd.append("--skip_missing")

        if langs:
            run_cmd(
                base_cmd + ["--anchor_language", *langs, "--output", "sim_langs"],
                log_path=log_root / "07_sims_langs.log",
                env=env,
            )

        if include_all:
            run_cmd(
                base_cmd + ["--output", "sim_all"],
                log_path=log_root / "08_sims_all.log",
                env=env,
            )

    print("\n[OK] Pipeline finished.")
    print(f"[OK] Logs: {log_root}")


if __name__ == "__main__":
    main()
