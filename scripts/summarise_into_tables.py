#!/usr/bin/env python3
"""
Summarize XM3600 RR retrieval sweep outputs into tables.

It scans a root directory for run folders containing:
  - results.json
  - run_config.json

and produces:
  1) rr_sweep_summary.csv / .xlsx  (one row per run; overall metrics)
  2) rr_sweep_per_language.csv     (one row per run per language)

Usage:
  python summarize_rr_runs.py --root /path/to/outputs/rr_sweep

Notes:
- Robust to partial runs (missing results.json).
- Backward compatible with older runs where config keys differ.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _find_runs(root: Path) -> List[Path]:
    # Prefer run_config.json as an anchor (exists even if results failed late)
    run_cfgs = list(root.rglob("run_config.json"))
    if run_cfgs:
        return sorted({p.parent for p in run_cfgs})
    # fallback: results.json
    results = list(root.rglob("results.json"))
    return sorted({p.parent for p in results})


def _flatten_dict(d: Dict[str, Any], prefix: str = "", out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if out is None:
        out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else str(k)
        if isinstance(v, dict):
            _flatten_dict(v, prefix=key + ".", out=out)
        else:
            out[key] = v
    return out


def _extract_overall_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    overall = results.get("overall") or {}
    if isinstance(overall, dict):
        for k, v in overall.items():
            out[f"overall.{k}"] = v
    out["n_queries"] = results.get("n_queries", results.get("n_query", None))
    return out


def _extract_per_language_rows(run_row: Dict[str, Any], results: Dict[str, Any]) -> List[Dict[str, Any]]:
    pl = results.get("per_language") or results.get("per_lang") or {}
    rows: List[Dict[str, Any]] = []
    if not isinstance(pl, dict):
        return rows

    for lang, metrics in pl.items():
        r = dict(run_row)
        r["language"] = lang
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                r[f"R.{k}"] = v
        rows.append(r)
    return rows


def _safe_get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in cfg:
            return cfg[k]
    return default


def build_tables(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    runs = _find_runs(root)
    summary_rows: List[Dict[str, Any]] = []
    per_lang_rows: List[Dict[str, Any]] = []

    for run_dir in runs:
        run_name = run_dir.name
        cfg = _read_json(run_dir / "run_config.json") or {}
        res = _read_json(run_dir / "results.json") or {}

        # Base row
        row: Dict[str, Any] = {
            "run_dir": str(run_dir),
            "run_name": run_name,
            "status": "ok" if res else "missing_results",
        }

        # Prefer run_config.json; fallback to results["run_config"]
        if not cfg and isinstance(res.get("run_config"), dict):
            cfg = res["run_config"]

        # Pull key config fields (keep names stable)
        row["text_model"] = _safe_get(cfg, "text_model", "txt_model", default=None)
        row["text_tag"] = _safe_get(cfg, "text_tag", default=None)
        row["image_model"] = _safe_get(cfg, "image_model", "img_model", default=None)
        row["image_tag"] = _safe_get(cfg, "image_tag", default=None)

        row["k"] = _safe_get(cfg, "k", default=None)
        row["p"] = _safe_get(cfg, "p", default=None)
        row["proc"] = _safe_get(cfg, "proc", default=None)
        row["dtype"] = _safe_get(cfg, "dtype", default=None)
        row["anchor_n"] = _safe_get(cfg, "anchor_n", default=None)
        row["anchor_chunk"] = _safe_get(cfg, "anchor_chunk", default=None)
        row["Ks"] = ",".join(map(str, _safe_get(cfg, "Ks", default=[]))) if _safe_get(cfg, "Ks", default=None) is not None else None
        row["languages"] = ",".join(_safe_get(cfg, "languages", default=[])) if isinstance(_safe_get(cfg, "languages", default=None), list) else _safe_get(cfg, "languages", default=None)
        row["max_queries"] = _safe_get(cfg, "max_queries", default=None)
        row["one_per_image_per_lang"] = _safe_get(cfg, "one_per_image_per_lang", default=None)
        row["seed"] = _safe_get(cfg, "seed", default=None)

        # Add overall metrics
        if res:
            row.update(_extract_overall_metrics(res))

            # per-language rows (long table)
            per_lang_rows.extend(_extract_per_language_rows(row, res))

        # Optional: include other cfg fields without exploding the table too much
        # We'll flatten but only keep a small prefix set.
        for k in ("xm_cache", "cc_cache", "text_repo", "image_repo"):
            if k in cfg:
                row[k] = cfg[k]

        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)

    # Sort nicely if those columns exist
    sort_cols = [c for c in ["text_model", "image_model", "k", "p", "proc"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable")

    df_lang = pd.DataFrame(per_lang_rows)
    if not df_lang.empty:
        sort_cols2 = [c for c in ["text_model", "image_model", "k", "p", "language"] if c in df_lang.columns]
        df_lang = df_lang.sort_values(sort_cols2, kind="stable")

    return df, df_lang


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing run folders (each with run_config.json / results.json).",
    )
    ap.add_argument(
        "--out_prefix",
        type=str,
        default="rr_sweep",
        help="Output filename prefix (default: rr_sweep). Files are written in --root.",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    df, df_lang = build_tables(root)

    # Write outputs
    out_csv = root / f"{args.out_prefix}_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} runs)")

    # Excel (nice for quick filtering)
    out_xlsx = root / f"{args.out_prefix}_summary.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="summary", index=False)
        if not df_lang.empty:
            df_lang.to_excel(w, sheet_name="per_language", index=False)
    print(f"Wrote {out_xlsx}")

    if not df_lang.empty:
        out_lang_csv = root / f"{args.out_prefix}_per_language.csv"
        df_lang.to_csv(out_lang_csv, index=False)
        print(f"Wrote {out_lang_csv} ({len(df_lang)} rows)")

    # Quick console preview
    cols_preview = [c for c in df.columns if c.startswith("overall.R@")]
    preview = ["run_name", "text_model", "image_model", "k", "p", "proc"] + cols_preview[:5]
    preview = [c for c in preview if c in df.columns]
    print("\nPreview:")
    print(df[preview].head(10).to_string(index=False))


if __name__ == "__main__":
    main()