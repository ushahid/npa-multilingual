#!/usr/bin/env python3
"""Evaluate repeated random subsets of the CC12M anchor bank.
Writes sampled indices, per-run recall, aggregate means/standard deviations, and a plot."""


import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch


DEFAULT_TEXT_MODEL = "qwen3-embed-8b"
DEFAULT_IMAGE_MODEL = "c-radio-v4-h"
DEFAULT_K = 800
DEFAULT_P = 4
DEFAULT_PROC = "asif"
DEFAULT_DTYPE = "float32"
DEFAULT_BASE_ANCHOR_N = 1_600_000
DEFAULT_PERCENTAGES = [1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 75.0, 100.0]
DEFAULT_REPEATS = 10
DEFAULT_KS = [1, 5, 10, 25, 50]


@dataclass
class PerRunSummary:
    percentage: float
    percentage_label: str
    repeat_idx: int
    seed: int
    anchor_n: int
    subset_path: str
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


@dataclass
class AggregateSummary:
    percentage: float
    percentage_label: str
    anchor_n: int
    requested_repeats: int
    executed_runs: int
    ok_runs: int
    failed_runs: int
    mean_r_at_1: Optional[float] = None
    std_r_at_1: Optional[float] = None
    mean_r_at_5: Optional[float] = None
    std_r_at_5: Optional[float] = None
    mean_r_at_10: Optional[float] = None
    std_r_at_10: Optional[float] = None
    mean_r_at_25: Optional[float] = None
    std_r_at_25: Optional[float] = None
    mean_r_at_50: Optional[float] = None
    std_r_at_50: Optional[float] = None
    note: str = ""


def _parse_float_list(items: Optional[Sequence[str]], default: Sequence[float]) -> List[float]:
    if not items:
        return list(default)
    out: List[float] = []
    for item in items:
        for piece in str(item).split(","):
            piece = piece.strip()
            if piece:
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


def _write_csv(path: Path, rows: Iterable[Dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _make_subset_indices(base_anchor_n: int, anchor_n: int, seed: int) -> torch.Tensor:
    if anchor_n > base_anchor_n:
        raise ValueError(f"anchor_n ({anchor_n}) cannot exceed base_anchor_n ({base_anchor_n})")
    if anchor_n == base_anchor_n:
        return torch.arange(base_anchor_n, dtype=torch.long)

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    idx = torch.randperm(base_anchor_n, generator=g, dtype=torch.long)[:anchor_n]
    # Sort for slightly better storage/load locality. The subset itself remains random.
    idx, _ = torch.sort(idx)
    return idx


def _ensure_subset_file(path: Path, base_anchor_n: int, anchor_n: int, seed: int, dry_run: bool) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return
    idx = _make_subset_indices(base_anchor_n=base_anchor_n, anchor_n=anchor_n, seed=seed)
    torch.save(idx, path)


def _load_run_summary(
    *,
    percentage: float,
    repeat_idx: int,
    seed: int,
    anchor_n: int,
    subset_path: Path,
    run_dir: Path,
    log_path: Path,
    results_path: Path,
    text_model: str,
    image_model: str,
    k: int,
    p: int,
    proc: str,
    dtype: str,
) -> PerRunSummary:
    obj = _read_json(results_path)
    overall = obj.get("overall", {})
    return PerRunSummary(
        percentage=percentage,
        percentage_label=_fmt_pct_label(percentage),
        repeat_idx=repeat_idx,
        seed=seed,
        anchor_n=anchor_n,
        subset_path=str(subset_path),
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


def _mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return float(mean), math.sqrt(var)


def _aggregate_rows(rows: List[PerRunSummary], requested_repeats: int) -> List[AggregateSummary]:
    groups: Dict[float, List[PerRunSummary]] = defaultdict(list)
    for row in rows:
        groups[row.percentage].append(row)

    out: List[AggregateSummary] = []
    for pct in sorted(groups.keys()):
        group = sorted(groups[pct], key=lambda r: (r.repeat_idx, r.seed))
        ok = [r for r in group if r.status == "ok"]
        failed = [r for r in group if r.status != "ok"]

        note = ""
        if pct == 100.0 and requested_repeats > 1 and len(group) == 1:
            note = "100% uses all anchors; one run is sufficient because repeats are identical."

        m1, s1 = _mean_std([r.r_at_1 for r in ok if r.r_at_1 is not None])
        m5, s5 = _mean_std([r.r_at_5 for r in ok if r.r_at_5 is not None])
        m10, s10 = _mean_std([r.r_at_10 for r in ok if r.r_at_10 is not None])
        m25, s25 = _mean_std([r.r_at_25 for r in ok if r.r_at_25 is not None])
        m50, s50 = _mean_std([r.r_at_50 for r in ok if r.r_at_50 is not None])

        out.append(
            AggregateSummary(
                percentage=pct,
                percentage_label=_fmt_pct_label(pct),
                anchor_n=group[0].anchor_n,
                requested_repeats=requested_repeats,
                executed_runs=len(group),
                ok_runs=len(ok),
                failed_runs=len(failed),
                mean_r_at_1=m1,
                std_r_at_1=s1,
                mean_r_at_5=m5,
                std_r_at_5=s5,
                mean_r_at_10=m10,
                std_r_at_10=s10,
                mean_r_at_25=m25,
                std_r_at_25=s25,
                mean_r_at_50=m50,
                std_r_at_50=s50,
                note=note,
            )
        )
    return out


# def _plot_aggregates(path: Path, rows: List[AggregateSummary], *, k: int, p: int) -> None:
#     if not rows:
#         return
#     rows = sorted(rows, key=lambda r: r.percentage)
#     x = [r.percentage for r in rows]
#     y = [r.mean_r_at_10 for r in rows]
#     e = [r.std_r_at_10 or 0.0 for r in rows]

#     plt.figure(figsize=(9.5, 6))
#     plt.errorbar(x, y, yerr=e, marker="o", linewidth=2, capsize=4)
#     plt.xlabel("Anchor percentage used")
#     plt.ylabel("R@10")
#     plt.title(f"Random anchor subset ablation: R@10 vs. anchor percentage (k={k}, p={p})")
#     plt.xticks(x, [f"{int(v)}%" if float(v).is_integer() else f"{v:g}%" for v in x], rotation=45, ha="center")
#     plt.ylim(bottom=0.0)
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(path, dpi=220, bbox_inches="tight")
#     plt.close()

def _plot_aggregates(path: Path, rows: List[AggregateSummary], *, k: int, p: int) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r.percentage)
    x = [r.percentage for r in rows]
    y = [r.mean_r_at_10 for r in rows]
    e = [r.std_r_at_10 or 0.0 for r in rows]

    plt.figure(figsize=(11, 6.5))

    plt.xlabel("Anchor percentage used", fontsize=12)
    plt.ylabel("R@10", fontsize=12)
    plt.title(f"Random anchor subset ablation: R@10 vs. anchor percentage (k={k}, p={p})", 
              fontsize=13, pad=15)


    labels = [f"{int(v)}%" if float(v).is_integer() else f"{v:g}%" for v in x]
    plt.xticks(x, labels, rotation=45, ha='center', fontsize=10)

    # Improved Y-axis: tighter range to remove empty space
    ymin = max(0.0, min(y) - 0.03) 
    ymax = min(0.65, max(y) + 0.02) 
    plt.ylim(ymin, ymax)

    plt.yticks(
        [round(i * 0.01, 2) for i in range(int(ymin*100), int(ymax*100) + 1, 2)],
        fontsize=10
    )

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)

    # # Plot with error bars
    # plt.errorbar(
    #     x, y, 
    #     yerr=e, 
    #     marker="o", 
    #     linewidth=2.5, 
    #     capsize=4,
    #     capthick=1.5,
    #     elinewidth=1.5,
    #     label="R@10 ± std",
    #     ecolor="red",
    # )

    plt.plot(
        x, y,
        marker="o",
        linewidth=2.5,
        label="R@10",
        zorder=2
    )

    plt.errorbar(
        x, y,
        yerr=e,
        fmt="none",
        ecolor="red",
        capsize=4,
        capthick=1.5,
        elinewidth=1.5,
        zorder=5
    )

    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run independent random-anchor ablations for XM3600 RR retrieval")

    ap.add_argument(
        "--retrieval_script",
        default=str(Path(__file__).resolve().with_name("xm3600_rr_ablation.py")),
        help="Path to xm3600_rr_ablation.py",
    )
    ap.add_argument("--xm_cache", required=True)
    ap.add_argument("--cc_cache", required=True)
    ap.add_argument("--out_root", required=True)

    ap.add_argument("--text_model", default=DEFAULT_TEXT_MODEL)
    ap.add_argument("--image_model", default=DEFAULT_IMAGE_MODEL)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--p", type=int, default=DEFAULT_P)
    ap.add_argument("--proc", choices=["none", "norm", "asif"], default=DEFAULT_PROC)
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=DEFAULT_DTYPE)

    ap.add_argument("--base_anchor_n", type=int, default=DEFAULT_BASE_ANCHOR_N)
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument(
        "--percentages",
        nargs="*",
        default=None,
        help="Percentages to evaluate. Accepts space-separated or comma-separated values.",
    )
    ap.add_argument(
        "--repeat_full_percentage",
        action="store_true",
        help="Force all repeats at 100%% even though they are deterministic duplicates.",
    )

    ap.add_argument("--device", default=None)
    ap.add_argument("--anchor_chunk", type=int, default=100_000)
    ap.add_argument("--image_batch", type=int, default=256)
    ap.add_argument("--query_batch", type=int, default=2048)
    ap.add_argument("--Ks", default="1,5,10,25,50")
    ap.add_argument("--languages", default=None)
    ap.add_argument("--max_queries", type=int, default=None)
    ap.add_argument("--clamp_min", type=float, default=None)
    ap.add_argument("--empty_cache_every", type=int, default=0)

    ap.add_argument("--inner_progress", action="store_true")
    ap.add_argument("--reuse_cache", action="store_true")
    ap.add_argument("--one_per_image_per_lang", action="store_true")

    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", action="store_true")
    ap.add_argument("--continue_on_error", action="store_true", default=True)
    ap.add_argument("--stop_on_error", action="store_true")
    ap.add_argument("--summarize_only", action="store_true")
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
    if ks_from_arg != DEFAULT_KS:
        print(f"[WARN] Using Ks={ks_from_arg}; default reporting uses {DEFAULT_KS}.")

    out_root = Path(args.out_root)
    indices_root = out_root / "indices"
    runs_root = out_root / "runs"
    logs_root = out_root / "logs"
    out_root.mkdir(parents=True, exist_ok=True)
    indices_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    per_run_rows: List[PerRunSummary] = []

    for pct in percentages:
        pct_label = _fmt_pct_label(pct)
        anchor_n = _pct_to_anchor_n(args.base_anchor_n, pct, min_anchor_n=args.k)

        repeats_for_pct = args.repeats
        if anchor_n == args.base_anchor_n and not args.repeat_full_percentage:
            repeats_for_pct = 1

        for repeat_idx in range(repeats_for_pct):
            seed = args.seed_start + repeat_idx
            subset_name = f"pct{pct_label.zfill(3) if pct_label.isdigit() else pct_label}_seed{repeat_idx:03d}_n{anchor_n}"
            subset_path = indices_root / f"{subset_name}.pt"
            run_dir = runs_root / subset_name
            log_path = logs_root / f"{subset_name}.log"
            results_path = run_dir / "results.json"

            if args.summarize_only:
                if results_path.is_file():
                    per_run_rows.append(
                        _load_run_summary(
                            percentage=pct,
                            repeat_idx=repeat_idx,
                            seed=seed,
                            anchor_n=anchor_n,
                            subset_path=subset_path,
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
                    per_run_rows.append(
                        PerRunSummary(
                            percentage=pct,
                            percentage_label=pct_label,
                            repeat_idx=repeat_idx,
                            seed=seed,
                            anchor_n=anchor_n,
                            subset_path=str(subset_path),
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

            _ensure_subset_file(subset_path, args.base_anchor_n, anchor_n, seed, dry_run=args.dry_run)

            if skip_existing and results_path.is_file():
                print(f"[SKIP] {subset_name} (results already exist)")
                per_run_rows.append(
                    _load_run_summary(
                        percentage=pct,
                        repeat_idx=repeat_idx,
                        seed=seed,
                        anchor_n=anchor_n,
                        subset_path=subset_path,
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
                "--anchor_indices_path", str(subset_path),
                "--k", str(args.k),
                "--anchor_chunk", str(args.anchor_chunk),
                "--proc", str(args.proc),
                "--p", str(args.p),
                "--image_batch", str(args.image_batch),
                "--query_batch", str(args.query_batch),
                "--Ks", str(args.Ks),
                "--empty_cache_every", str(args.empty_cache_every),
                "--seed", str(seed),
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

            print(
                f"[RUN ] {subset_name}  percentage={pct}  anchor_n={anchor_n}  repeat={repeat_idx + 1}/{repeats_for_pct}"
            )
            ret = _run_cmd(cmd, log_path=log_path, env=env, dry_run=args.dry_run)

            if ret != 0:
                row = PerRunSummary(
                    percentage=pct,
                    percentage_label=pct_label,
                    repeat_idx=repeat_idx,
                    seed=seed,
                    anchor_n=anchor_n,
                    subset_path=str(subset_path),
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
                per_run_rows.append(row)
                print(f"[FAIL] {subset_name} (exit={ret}) log={log_path}")
                if not continue_on_error and not args.dry_run:
                    break
                continue

            if results_path.is_file():
                per_run_rows.append(
                    _load_run_summary(
                        percentage=pct,
                        repeat_idx=repeat_idx,
                        seed=seed,
                        anchor_n=anchor_n,
                        subset_path=subset_path,
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
                per_run_rows.append(
                    PerRunSummary(
                        percentage=pct,
                        percentage_label=pct_label,
                        repeat_idx=repeat_idx,
                        seed=seed,
                        anchor_n=anchor_n,
                        subset_path=str(subset_path),
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

    aggregate_rows = _aggregate_rows(per_run_rows, requested_repeats=args.repeats)

    per_run_json = out_root / "random_anchor_per_run.json"
    per_run_csv = out_root / "random_anchor_per_run.csv"
    agg_json = out_root / "random_anchor_aggregate.json"
    agg_csv = out_root / "random_anchor_aggregate.csv"
    plot_png = out_root / "random_anchor_plot.png"
    run_meta = out_root / "random_anchor_run_config.json"

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
            "repeats": args.repeats,
            "seed_start": args.seed_start,
            "repeat_full_percentage": bool(args.repeat_full_percentage),
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
    _write_json(per_run_json, {"rows": [asdict(r) for r in per_run_rows]})
    _write_json(agg_json, {"rows": [asdict(r) for r in aggregate_rows]})

    _write_csv(per_run_csv, [asdict(r) for r in per_run_rows], fieldnames=list(asdict(per_run_rows[0]).keys()) if per_run_rows else list(PerRunSummary.__dataclass_fields__.keys()))
    _write_csv(agg_csv, [asdict(r) for r in aggregate_rows], fieldnames=list(asdict(aggregate_rows[0]).keys()) if aggregate_rows else list(AggregateSummary.__dataclass_fields__.keys()))
    _plot_aggregates(plot_png, aggregate_rows, k=args.k, p=args.p)

    print("\nPer-run summary:")
    print(per_run_json)
    print(per_run_csv)
    print("\nAggregate summary:")
    print(agg_json)
    print(agg_csv)
    print("\nPlot:")
    print(plot_png)


if __name__ == "__main__":
    main()