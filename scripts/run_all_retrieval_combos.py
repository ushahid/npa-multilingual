#!/usr/bin/env python3
"""Run XM3600 RR retrieval for all model combinations and (k,p) grids.

This script wraps your retrieval runner:
  scripts/xm3600_rr_retrieval.py

It does NOT modify retrieval logic. It only iterates over:
  - text_model ∈ {nllb-200-3.3B, bge-m3, qwen3-embed-8b}
  - image_model ∈ {dinov3-vit7b16, clip-vit-bigg14, c-radio-v4-h}
  - k ∈ {200, 400, 800, 1200, 1600}
  - p ∈ {2, 4, 8}

Output folder naming matches your example:
  <imageTag>_<textTag>_k<K>_p<P>

Per-run logs are written under:
  <out_root>/logs/<imageTag>_<textTag>_k<K>_p<P>.log
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from datetime import datetime

DEFAULT_TEXT_MODELS = ["nllb-200-3.3B", "bge-m3", "qwen3-embed-8b"]
DEFAULT_IMAGE_MODELS = ["dinov3-vit7b16", "clip-vit-bigg14", "c-radio-v4-h"]
DEFAULT_KS = [200, 400, 800, 1200, 1600]
DEFAULT_PS = [2, 4, 8]


def _run_cmd(cmd: list[str], log_path: Path, env: dict[str, str] | None, dry_run: bool) -> int:
    """Run a command, streaming stdout/stderr to a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[DRY RUN] {shlex.join(cmd)}")
        print(f"          log -> {log_path}")
        return 0

    with log_path.open("w") as f:
        f.write(f"[CMD] {shlex.join(cmd)}\n")
        f.write(f"[TIME] {datetime.now().isoformat()}\n\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        return proc.returncode


def _should_skip(out_dir: Path) -> bool:
    """Skip if output dir exists and is non-empty."""
    if not out_dir.exists():
        return False
    # Non-empty dir => consider done/started; don't overwrite unless user wants.
    try:
        return any(out_dir.iterdir())
    except Exception:
        return True


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--retrieval_script",
        type=str,
        default="scripts/xm3600_rr_retrieval.py",
        help="Path to xm3600_rr_retrieval.py",
    )

    ap.add_argument("--xm_cache", type=str, required=True)
    ap.add_argument("--cc_cache", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)

    ap.add_argument("--dtype", type=str, default="float32")
    ap.add_argument("--proc", type=str, default="asif")

    # Optional: only pass through if provided (to keep behavior aligned with your current command)
    ap.add_argument("--device", type=str, default=None, help="If set, passes --device to retrieval script")

    ap.add_argument(
        "--text_models",
        nargs="+",
        default=DEFAULT_TEXT_MODELS,
        choices=DEFAULT_TEXT_MODELS,
        help="Text embedding model tags to sweep",
    )
    ap.add_argument(
        "--image_models",
        nargs="+",
        default=DEFAULT_IMAGE_MODELS,
        choices=DEFAULT_IMAGE_MODELS,
        help="Image embedding model tags to sweep",
    )

    ap.add_argument("--ks", nargs="+", type=int, default=DEFAULT_KS)
    ap.add_argument("--ps", nargs="+", type=int, default=DEFAULT_PS)

    ap.add_argument("--inner_progress", action="store_true", default=True)
    ap.add_argument("--no_inner_progress", action="store_true", help="Do not pass --inner_progress")

    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", action="store_true", help="Re-run even if out_dir is non-empty")

    ap.add_argument("--continue_on_error", action="store_true", default=True)
    ap.add_argument("--stop_on_error", action="store_true", help="Stop immediately on first failure")

    ap.add_argument("--dry_run", action="store_true")

    args = ap.parse_args()

    retrieval_script = Path(args.retrieval_script)
    xm_cache = Path(args.xm_cache)
    cc_cache = Path(args.cc_cache)
    out_root = Path(args.out_root)

    if args.no_inner_progress:
        inner_progress = False
    else:
        inner_progress = True

    if args.no_skip_existing:
        skip_existing = False
    else:
        skip_existing = True

    if args.stop_on_error:
        continue_on_error = False
    else:
        continue_on_error = True

    logs_dir = out_root / "logs"
    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()

    total = 0
    skipped = 0
    failed = 0

    for image_model in args.image_models:
        for text_model in args.text_models:
            for k in args.ks:
                for p in args.ps:
                    total += 1

                    run_name = f"{image_model}_{text_model}_k{k}_p{p}"
                    out_dir = out_root / run_name
                    log_path = logs_dir / f"{run_name}.log"

                    if skip_existing and _should_skip(out_dir):
                        skipped += 1
                        print(f"[SKIP] {run_name} (out_dir non-empty: {out_dir})")
                        continue

                    out_dir.mkdir(parents=True, exist_ok=True)

                    cmd = [
                        "python",
                        str(retrieval_script),
                        "--xm_cache",
                        str(xm_cache),
                        "--cc_cache",
                        str(cc_cache),
                        "--out_dir",
                        str(out_dir),
                        "--dtype",
                        str(args.dtype),
                        "--k",
                        str(k),
                        "--p",
                        str(p),
                        "--proc",
                        str(args.proc),
                        "--text_model",
                        str(text_model),
                        "--image_model",
                        str(image_model),
                    ]

                    if inner_progress:
                        cmd.append("--inner_progress")

                    if args.device:
                        cmd.extend(["--device", args.device])

                    print(f"[RUN ] {run_name}")
                    ret = _run_cmd(cmd, log_path=log_path, env=env, dry_run=args.dry_run)

                    if ret != 0:
                        failed += 1
                        print(f"[FAIL] {run_name} (exit={ret})  log={log_path}")
                        if not continue_on_error and not args.dry_run:
                            raise SystemExit(ret)

    print("\n==== Summary ====")
    print(f"total configs : {total}")
    print(f"skipped       : {skipped}")
    print(f"failed        : {failed}")


if __name__ == "__main__":
    main()