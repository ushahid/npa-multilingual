#!/usr/bin/env python3
"""Sweep anchor-bank size using the first N CC12M pairs at each percentage.
Calls the RR retrieval script and writes recall summaries and a plot."""

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


DEFAULT_TEXT_MODEL = "qwen3-embed-8b"
DEFAULT_IMAGE_MODEL = "c-radio-v4-h"
DEFAULT_K = 800
DEFAULT_P = 4
DEFAULT_PROC = "asif"
DEFAULT_DTYPE = "float32"
DEFAULT_BASE_ANCHOR_N = 1_600_000
DEFAULT_PERCENTAGES = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0]
DEFAULT_KS = [1, 5, 10, 25, 50]


@dataclass
class RunSummary:
    percentage: float
    percentage_label: str
    anchor_n: int
    status: str
    run_dir: str
    log_path: str
    results_path: str
    text_model: str
    image_model: str
    k: int
    p: int
    proc: str
    dtype: str
    n_queries: Optional[int] = None
    r_at_1: Optional[float] = None
    r_at_5: Optional[float] = None
    r_at_10: Optional[float] = None
    r_at_25: Optional[float] = None
    r_at_50: Optional[float] = None
    error: str = ""


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_float_list(items: Optional[Sequence[str]], default: Sequence[float]) -> List[float]:
    if not items:
        return list(default)
    out: List[float] = []
    for item in items:
        for piece in str(item).split(","):
            piece = piece.strip()
            if not piece:
                continue
            out.append(float(piece))
    return out


def _fmt_pct_label(pct: float) -> str:
    if float(pct).is_integer():
        return str(int(pct))
    return str(pct).replace(".", "p")


def _pct_to_anchor_n(base_anchor_n: int, pct: float, min_anchor_n: int) -> int:
    raw = int(round(base_anchor_n * (pct / 100.0)))
    return max(min_anchor_n, raw)


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _write_csv(path: Path, rows: List[RunSummary]) -> None:
    fieldnames = [
        "percentage",
        "percentage_label",
        "anchor_n",
        "status",
        "run_dir",
        "log_path",
        "results_path",
        "text_model",
        "image_model",
        "k",
        "p",
        "proc",
        "dtype",
        "n_queries",
        "r_at_1",
        "r_at_5",
        "r_at_10",
        "r_at_25",
        "r_at_50",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _run_cmd(cmd: List[str], log_path: Path, env: Dict[str, str], dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"[DRY RUN] {shlex.join(cmd)}")
        print(f"          log -> {log_path}")
        return 0

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[CMD] {shlex.join(cmd)}\n")
        f.write(f"[TIME] {datetime.now().isoformat()}\n\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        return proc.returncode


def _load_summary_from_results(
    *,
    percentage: float,
    anchor_n: int,
    run_dir: Path,
    log_path: Path,
    results_path: Path,
    text_model: str,
    image_model: str,
    k: int,
    p: int,
    proc: str,
    dtype: str,
) -> RunSummary:
    obj = _read_json(results_path)
    overall = obj.get("overall", {})
    return RunSummary(
        percentage=percentage,
        percentage_label=_fmt_pct_label(percentage),
        anchor_n=anchor_n,
        status="ok",
        run_dir=str(run_dir),
        log_path=str(log_path),
        results_path=str(results_path),
        text_model=text_model,
        image_model=image_model,
        k=k,
        p=p,
        proc=proc,
        dtype=dtype,
        n_queries=obj.get("n_queries"),
        r_at_1=overall.get("R@1"),
        r_at_5=overall.get("R@5"),
        r_at_10=overall.get("R@10"),
        r_at_25=overall.get("R@25"),
        r_at_50=overall.get("R@50"),
    )


def _plot_results(path: Path, rows: List[RunSummary]) -> None:
    ok_rows = [r for r in rows if r.status == "ok"]
    if not ok_rows:
        return

    ok_rows = sorted(ok_rows, key=lambda r: r.percentage)
    x = [r.percentage for r in ok_rows]

    metric_map = {
        "R@1": [r.r_at_1 for r in ok_rows],
        "R@5": [r.r_at_5 for r in ok_rows],
        "R@10": [r.r_at_10 for r in ok_rows],
        "R@25": [r.r_at_25 for r in ok_rows],
        "R@50": [r.r_at_50 for r in ok_rows],
    }

    plt.figure(figsize=(9, 5.5))
    for label, ys in metric_map.items():
        plt.plot(x, ys, marker="o", linewidth=2, label=label)

    plt.xlabel("Anchor percentage used")
    plt.ylabel("Recall")
    plt.title("Anchor percentage ablation")
    plt.xticks(x, [f"{int(v)}%" if float(v).is_integer() else f"{v}%" for v in x])
    plt.ylim(bottom=0.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run anchor-percentage ablations for XM3600 RR retrieval")

    ap.add_argument("--retrieval_script", default=str(Path(__file__).resolve().with_name("xm3600_rr_retrieval.py")), help="Path to xm3600_rr_retrieval.py")
    ap.add_argument("--xm_cache", required=True, help="XM3600 cache dir")
    ap.add_argument("--cc_cache", required=True, help="CC12M cache dir")
    ap.add_argument("--out_root", required=True, help="Output root for runs, logs, summaries, and plot")

    ap.add_argument("--text_model", default=DEFAULT_TEXT_MODEL)
    ap.add_argument("--image_model", default=DEFAULT_IMAGE_MODEL)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--p", type=int, default=DEFAULT_P)
    ap.add_argument("--proc", choices=["none", "norm", "asif"], default=DEFAULT_PROC)
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=DEFAULT_DTYPE)

    ap.add_argument("--base_anchor_n", type=int, default=DEFAULT_BASE_ANCHOR_N, help="Anchor count corresponding to 100%%")
    ap.add_argument(
        "--percentages",
        nargs="*",
        default=None,
        help="Anchor percentages to evaluate. Accepts either space-separated or comma-separated values. Default: 100 90 80 70",
    )

    ap.add_argument("--device", default=None, help="Optional device forwarded to retrieval script, e.g. cuda:0")
    ap.add_argument("--anchor_chunk", type=int, default=100_000)
    ap.add_argument("--image_batch", type=int, default=256)
    ap.add_argument("--query_batch", type=int, default=2048)
    ap.add_argument("--Ks", default="1,5,10,25,50")
    ap.add_argument("--languages", default=None)
    ap.add_argument("--max_queries", type=int, default=None)
    ap.add_argument("--clamp_min", type=float, default=None)
    ap.add_argument("--empty_cache_every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--reuse_cache", action="store_true", help="Pass --reuse_cache to retrieval runs")
    ap.add_argument("--inner_progress", action="store_true", help="Pass --inner_progress to retrieval runs")
    ap.add_argument("--one_per_image_per_lang", action="store_true")

    ap.add_argument("--skip_existing", action="store_true", default=True, help="Skip runs with an existing results.json")
    ap.add_argument("--no_skip_existing", action="store_true", help="Force rerun even if results.json exists")
    ap.add_argument("--continue_on_error", action="store_true", default=True)
    ap.add_argument("--stop_on_error", action="store_true", help="Stop immediately if a run fails")
    ap.add_argument("--summarize_only", action="store_true", help="Do not launch retrieval; only summarize/plot existing runs")
    ap.add_argument("--dry_run", action="store_true")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_skip_existing:
        skip_existing = False
    else:
        skip_existing = True

    if args.stop_on_error:
        continue_on_error = False
    else:
        continue_on_error = True

    retrieval_script = Path(args.retrieval_script)
    if not retrieval_script.is_file():
        raise SystemExit(f"retrieval script not found: {retrieval_script}")

    percentages = _parse_float_list(args.percentages, DEFAULT_PERCENTAGES)
    if not percentages:
        raise SystemExit("No percentages provided.")

    for pct in percentages:
        if pct <= 0 or pct > 100:
            raise SystemExit(f"Invalid percentage: {pct}. Expected 0 < pct <= 100.")

    ks_from_arg = [int(x) for x in args.Ks.split(",") if x.strip()]
    required_ks = DEFAULT_KS
    if ks_from_arg != required_ks:
        print(f"[WARN] Using Ks={ks_from_arg}; paper/default reporting uses {required_ks}.")

    out_root = Path(args.out_root)
    runs_root = out_root / "runs"
    logs_root = out_root / "logs"
    out_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    rows: List[RunSummary] = []

    for pct in percentages:
        pct_label = _fmt_pct_label(pct)
        anchor_n = _pct_to_anchor_n(args.base_anchor_n, pct, min_anchor_n=args.k)
        run_name = f"pct{pct_label.zfill(3) if pct_label.isdigit() else pct_label}_n{anchor_n}"
        run_dir = runs_root / run_name
        log_path = logs_root / f"{run_name}.log"
        results_path = run_dir / "results.json"

        if args.summarize_only:
            if results_path.is_file():
                rows.append(
                    _load_summary_from_results(
                        percentage=pct,
                        anchor_n=anchor_n,
                        run_dir=run_dir,
                        log_path=log_path,
                        results_path=results_path,
                        text_model=args.text_model,
                        image_model=args.image_model,
                        k=args.k,
                        p=args.p,
                        proc=args.proc,
                        dtype=args.dtype,
                    )
                )
            else:
                rows.append(
                    RunSummary(
                        percentage=pct,
                        percentage_label=pct_label,
                        anchor_n=anchor_n,
                        status="missing_results",
                        run_dir=str(run_dir),
                        log_path=str(log_path),
                        results_path=str(results_path),
                        text_model=args.text_model,
                        image_model=args.image_model,
                        k=args.k,
                        p=args.p,
                        proc=args.proc,
                        dtype=args.dtype,
                        error="results.json not found",
                    )
                )
            continue

        if skip_existing and results_path.is_file():
            print(f"[SKIP] {run_name} (results already exist)")
            rows.append(
                _load_summary_from_results(
                    percentage=pct,
                    anchor_n=anchor_n,
                    run_dir=run_dir,
                    log_path=log_path,
                    results_path=results_path,
                    text_model=args.text_model,
                    image_model=args.image_model,
                    k=args.k,
                    p=args.p,
                    proc=args.proc,
                    dtype=args.dtype,
                )
            )
            continue

        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(retrieval_script),
            "--xm_cache", str(args.xm_cache),
            "--cc_cache", str(args.cc_cache),
            "--out_dir", str(run_dir),
            "--text_model", str(args.text_model),
            "--image_model", str(args.image_model),
            "--dtype", str(args.dtype),
            "--anchor_n", str(anchor_n),
            "--k", str(args.k),
            "--anchor_chunk", str(args.anchor_chunk),
            "--proc", str(args.proc),
            "--p", str(args.p),
            "--image_batch", str(args.image_batch),
            "--query_batch", str(args.query_batch),
            "--Ks", str(args.Ks),
            "--empty_cache_every", str(args.empty_cache_every),
            "--seed", str(args.seed),
        ]

        if args.device:
            cmd.extend(["--device", str(args.device)])
        if args.languages:
            cmd.extend(["--languages", str(args.languages)])
        if args.max_queries is not None:
            cmd.extend(["--max_queries", str(args.max_queries)])
        if args.clamp_min is not None:
            cmd.extend(["--clamp_min", str(args.clamp_min)])
        if args.reuse_cache:
            cmd.append("--reuse_cache")
        if args.inner_progress:
            cmd.append("--inner_progress")
        if args.one_per_image_per_lang:
            cmd.append("--one_per_image_per_lang")

        print(f"[RUN ] {run_name}  percentage={pct}  anchor_n={anchor_n}")
        ret = _run_cmd(cmd, log_path=log_path, env=env, dry_run=args.dry_run)

        if ret != 0:
            row = RunSummary(
                percentage=pct,
                percentage_label=pct_label,
                anchor_n=anchor_n,
                status="failed",
                run_dir=str(run_dir),
                log_path=str(log_path),
                results_path=str(results_path),
                text_model=args.text_model,
                image_model=args.image_model,
                k=args.k,
                p=args.p,
                proc=args.proc,
                dtype=args.dtype,
                error=f"retrieval exit code {ret}",
            )
            rows.append(row)
            print(f"[FAIL] {run_name} (exit={ret})  log={log_path}")
            if not continue_on_error and not args.dry_run:
                break
            continue

        if results_path.is_file():
            rows.append(
                _load_summary_from_results(
                    percentage=pct,
                    anchor_n=anchor_n,
                    run_dir=run_dir,
                    log_path=log_path,
                    results_path=results_path,
                    text_model=args.text_model,
                    image_model=args.image_model,
                    k=args.k,
                    p=args.p,
                    proc=args.proc,
                    dtype=args.dtype,
                )
            )
        else:
            rows.append(
                RunSummary(
                    percentage=pct,
                    percentage_label=pct_label,
                    anchor_n=anchor_n,
                    status="missing_results",
                    run_dir=str(run_dir),
                    log_path=str(log_path),
                    results_path=str(results_path),
                    text_model=args.text_model,
                    image_model=args.image_model,
                    k=args.k,
                    p=args.p,
                    proc=args.proc,
                    dtype=args.dtype,
                    error="results.json not found after successful run",
                )
            )

    summary_json = out_root / "anchor_percentage_summary.json"
    summary_csv = out_root / "anchor_percentage_summary.csv"
    plot_png = out_root / "anchor_percentage_plot.png"
    run_meta = out_root / "anchor_percentage_run_config.json"

    _write_json(
        run_meta,
        {
            "created_at": datetime.now().isoformat(),
            "retrieval_script": str(retrieval_script),
            "xm_cache": os.path.abspath(args.xm_cache),
            "cc_cache": os.path.abspath(args.cc_cache),
            "out_root": os.path.abspath(str(out_root)),
            "text_model": args.text_model,
            "image_model": args.image_model,
            "k": args.k,
            "p": args.p,
            "proc": args.proc,
            "dtype": args.dtype,
            "base_anchor_n": args.base_anchor_n,
            "percentages": percentages,
            "Ks": ks_from_arg,
            "device": args.device,
            "anchor_chunk": args.anchor_chunk,
            "image_batch": args.image_batch,
            "query_batch": args.query_batch,
            "languages": args.languages,
            "max_queries": args.max_queries,
            "reuse_cache": bool(args.reuse_cache),
            "inner_progress": bool(args.inner_progress),
            "one_per_image_per_lang": bool(args.one_per_image_per_lang),
        },
    )

    _write_json(summary_json, {"rows": [asdict(r) for r in rows]})
    _write_csv(summary_csv, rows)
    _plot_results(plot_png, rows)

    print("\n==== Anchor percentage ablation summary ====")
    for row in rows:
        if row.status == "ok":
            print(
                f"pct={row.percentage:g}%  anchor_n={row.anchor_n}  "
                f"R@1={row.r_at_1:.4f}  R@5={row.r_at_5:.4f}  "
                f"R@10={row.r_at_10:.4f}  R@25={row.r_at_25:.4f}  R@50={row.r_at_50:.4f}"
            )
        else:
            print(f"pct={row.percentage:g}%  anchor_n={row.anchor_n}  status={row.status}  error={row.error}")

    print(f"\nWrote {summary_json}")
    print(f"Wrote {summary_csv}")
    if plot_png.is_file():
        print(f"Wrote {plot_png}")


if __name__ == "__main__":
    main()