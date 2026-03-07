#!/usr/bin/env python3
"""
Generate RR sweep plots from rr_sweep_per_language.csv

Plots created:
1) KP sensitivity per (text_model, image_model) for R@1 across ALL languages:
   - line = mean across languages
   - band = IQR (25–75%)
   - dashed = min/max
   (optionally overlays overall.R@1 if present)

2) Scatter plot for ALL runs:
   - x = overall R@1
   - y = overall R@50
   - points for all (text,image,k,p) runs

3) Bar chart: best overall R@1 per model pair (max over k,p), annotated with (k,p)

No seaborn. No explicit colors (matplotlib default cycle only).
"""

import argparse
import os
import re
import zipfile
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _safe_name(s: str) -> str:
    s = s.replace("/", "_")
    s = re.sub(r"[^a-zA-Z0-9_.\-+=]+", "_", s)
    return s


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _scatter_points(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return points sorted by x ascending. Maximizing both x and y."""
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return np.zeros((0, 2), dtype=float)

    # Sort by x desc, keep points with increasing y
    order = np.argsort(-x)
    x_s = x[order]
    y_s = y[order]
    front = []
    best_y = -1e18
    for xx, yy in zip(x_s, y_s):
        if yy > best_y + 1e-12:
            front.append((xx, yy))
            best_y = yy

    front = np.array(sorted(front, key=lambda v: v[0]))
    return front


def make_kp_language_distribution_plots(
    df_long: pd.DataFrame,
    out_dir: str,
    show_overall_line: bool = True,
) -> List[str]:
    """
    One PNG per (text_model,image_model):
    - y is per-language R@1 distribution across languages for each (k,p)
    """
    paths = []

    # Column detection (robust to slight naming changes)
    text_col = _pick_col(df_long, ["text_model", "text"])
    img_col = _pick_col(df_long, ["image_model", "image"])
    lang_col = _pick_col(df_long, ["language", "lang"])
    k_col = _pick_col(df_long, ["k"])
    p_col = _pick_col(df_long, ["p"])

    # Per-language metric column for R@1
    rr1_col = _pick_col(df_long, ["R.R@1", "R@1", "rr_r1", "lang.R@1"])
    # Overall metric (optional overlay)
    overall_r1_col = _pick_col(df_long, ["overall.R@1", "overall_r1", "overall_R@1"])

    required = [text_col, img_col, lang_col, k_col, p_col, rr1_col]
    if any(c is None for c in required):
        missing = [n for n, c in zip(
            ["text_model", "image_model", "language", "k", "p", "R@1(per-language)"], required
        ) if c is None]
        raise RuntimeError(f"Missing required columns in CSV: {missing}")

    _to_numeric(df_long, [k_col, p_col, rr1_col])
    if overall_r1_col:
        _to_numeric(df_long, [overall_r1_col])

    text_models = sorted(df_long[text_col].dropna().unique())
    image_models = sorted(df_long[img_col].dropna().unique())

    for t in text_models:
        for im in image_models:
            sub = df_long[(df_long[text_col] == t) & (df_long[img_col] == im)].copy()
            sub = sub.dropna(subset=[k_col, p_col, rr1_col])
            if sub.empty:
                continue

            fig = plt.figure(figsize=(8.2, 5.2))
            ax = fig.add_subplot(111)

            for pval in sorted(sub[p_col].dropna().unique()):
                sP = sub[sub[p_col] == pval]
                if sP.empty:
                    continue

                g = sP.groupby(k_col)[rr1_col]
                mean = g.mean()
                q25 = g.quantile(0.25)
                q75 = g.quantile(0.75)
                mn = g.min()
                mx = g.max()

                ks = mean.index.to_numpy()
                ys = mean.to_numpy()

                # mean line (default color cycle)
                line, = ax.plot(ks, ys, marker="o", label=f"p={int(pval)} mean")

                # IQR band & min/max envelope using the same line color
                ax.fill_between(ks, q25.to_numpy(), q75.to_numpy(),
                                alpha=0.15, color=line.get_color())
                ax.plot(ks, mn.to_numpy(), linestyle="--", linewidth=1,
                        alpha=0.5, color=line.get_color())
                ax.plot(ks, mx.to_numpy(), linestyle="--", linewidth=1,
                        alpha=0.5, color=line.get_color())

                # Optional overall.R@1 overlay (thin dotted) if present and consistent
                if show_overall_line and overall_r1_col:
                    g_overall = sP.groupby(k_col)[overall_r1_col].mean()
                    if len(g_overall) == len(mean):
                        ax.plot(ks, g_overall.loc[ks].to_numpy(),
                                linestyle=":", linewidth=1.5, alpha=0.8,
                                color=line.get_color(),
                                label=f"p={int(pval)} overall")

            ax.set_xlabel("k (candidates)")
            ax.set_ylabel("R@1")
            ax.set_title(
                "KP sensitivity for R@1 across ALL languages\n"
                f"text={t} | image={im}\n"
                "(solid=mean, band=IQR, dashed=min/max)"
                + (" + dotted=overall" if (show_overall_line and overall_r1_col) else "")
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=1)

            fname = _safe_name(f"kp_R1_langdist__txt-{t}__img-{im}.png")
            path = os.path.join(out_dir, fname)
            fig.tight_layout()
            fig.savefig(path, dpi=200)
            plt.close(fig)
            paths.append(path)

    return paths

def make_per_language_bar_charts(df_long: pd.DataFrame, out_dir: str) -> List[str]:
    """
    For each (text_model, image_model) pair:
      - Find the (k, p) that gives the highest overall R@1
      - Create ONE horizontal grouped bar chart showing BOTH R@1 and R@50
        for EVERY language (sorted by R@1 descending)
    """
    paths: List[str] = []

    # Column detection
    text_col = _pick_col(df_long, ["text_model", "text"])
    img_col = _pick_col(df_long, ["image_model", "image"])
    k_col = _pick_col(df_long, ["k"])
    p_col = _pick_col(df_long, ["p"])
    lang_col = _pick_col(df_long, ["language", "lang"])
    r1_col = _pick_col(df_long, ["R.R@1", "R@1", "rr_r1", "lang.R@1"])
    r50_col = _pick_col(df_long, ["R.R@50", "R@50", "rr_r50", "lang.R@50"])
    overall_r1_col = _pick_col(df_long, ["overall.R@1", "overall_r1", "overall_R@1"])
    overall_r50_col = _pick_col(df_long, ["overall.R@50", "overall_r50", "overall_R@50"])

    required = [text_col, img_col, k_col, p_col, lang_col, r1_col, r50_col, overall_r1_col, overall_r50_col]
    if any(c is None for c in required):
        missing = [n for n, c in zip(
            ["text", "image", "k", "p", "language", "per-lang R@1", "per-lang R@50",
             "overall R@1", "overall R@50"], required
        ) if c is None]
        raise RuntimeError(f"Missing columns for language bar charts: {missing}")

    _to_numeric(df_long, [k_col, p_col, r1_col, r50_col, overall_r1_col, overall_r50_col])

    # 1. Find the best (k, p) for each model pair (highest overall R@1)
    group_cols = [text_col, img_col, k_col, p_col]
    best_idx = df_long.groupby(group_cols)[overall_r1_col].idxmax()
    best_configs = df_long.loc[best_idx].copy()

    # 2. For each model pair, create one grouped bar chart
    for (t, im), best_row in best_configs.groupby([text_col, img_col]):
        best_k = int(best_row[k_col].iloc[0])
        best_p = int(best_row[p_col].iloc[0])
        best_overall_r1 = best_row[overall_r1_col].iloc[0]
        best_overall_r50 = best_row[overall_r50_col].iloc[0]

        # Get ALL language rows for this exact config
        config_mask = (
            (df_long[text_col] == t) &
            (df_long[img_col] == im) &
            (df_long[k_col] == best_k) &
            (df_long[p_col] == best_p)
        )
        config_df = df_long[config_mask].copy()

        if config_df.empty:
            continue

        # Aggregate per language
        lang_perf = config_df.groupby(lang_col).agg({
            r1_col: 'mean',
            r50_col: 'mean'
        }).sort_values(r1_col, ascending=False)   # sort by R@1

        languages = lang_perf.index.tolist()
        r1_scores = lang_perf[r1_col].values
        r50_scores = lang_perf[r50_col].values

        # --- Plot: grouped horizontal bars ---
        fig, ax = plt.subplots(figsize=(14, max(8, len(languages) * 0.38)))

        y_pos = np.arange(len(languages))
        width = 0.35

        bars1 = ax.barh(y_pos - width/2, r1_scores, width, label='R@1', color='tab:blue', alpha=0.9)
        bars2 = ax.barh(y_pos + width/2, r50_scores, width, label='R@50', color='tab:orange', alpha=0.9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(languages)
        ax.set_xlabel("Recall")
        ax.set_xlim(0, 1.05)
        ax.grid(axis="x", alpha=0.3)
        ax.legend(loc='upper right')

        ax.set_title(
            f"Per-language performance — {t} + {im}\n"
            f"Best config: k={best_k}, p={best_p}   |   "
            f"Overall R@1 = {best_overall_r1:.4f}   |   "
            f"Overall R@50 = {best_overall_r50:.4f}"
        )

        # Value labels
        for i, (v1, v2) in enumerate(zip(r1_scores, r50_scores)):
            ax.text(v1 + 0.008, i - width/2, f"{v1:.3f}", va="center", fontsize=8, fontweight="bold", color="tab:blue")
            ax.text(v2 + 0.008, i + width/2, f"{v2:.3f}", va="center", fontsize=8, fontweight="bold", color="tab:orange")

        fname = _safe_name(f"langbars_R1_R50__txt-{t}__img-{im}__k{best_k}_p{best_p}.png")
        path = os.path.join(out_dir, fname)
        fig.tight_layout()
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    print(f"[OK] Created {len(paths)} per-language grouped bar charts (R@1 + R@50)")
    return paths


def make_scatter_plot(
    df_long: pd.DataFrame,
    out_dir: str,
    annotate: str = "frontier",     # "none" | "frontier" | "all"
    annotate_topn: int = 0,         # if >0, annotate top-N by overall.R@1
) -> str:
    # Use one row per run for Scatter plot (dedupe by run_name if available)
    run_col = _pick_col(df_long, ["run_name", "run", "name"])
    text_col = _pick_col(df_long, ["text_model", "text"])
    img_col = _pick_col(df_long, ["image_model", "image"])
    k_col = _pick_col(df_long, ["k"])
    p_col = _pick_col(df_long, ["p"])
    r1_col = _pick_col(df_long, ["overall.R@1", "overall_r1", "overall_R@1"])
    r50_col = _pick_col(df_long, ["overall.R@50", "overall_r50", "overall_R@50"])

    required = [text_col, img_col, r1_col, r50_col, k_col, p_col]
    if any(c is None for c in required):
        missing = [n for n, c in zip(
            ["text_model", "image_model", "overall.R@1", "overall.R@50", "k", "p"], required
        ) if c is None]
        raise RuntimeError(f"Missing required columns for Scatter plot: {missing}")

    runs = df_long.copy()
    if run_col:
        runs = runs.drop_duplicates(subset=[run_col]).copy()
    else:
        runs = runs.drop_duplicates(subset=[text_col, img_col, k_col, p_col]).copy()

    _to_numeric(runs, [r1_col, r50_col, k_col, p_col])
    runs = runs.dropna(subset=[r1_col, r50_col, k_col, p_col])

    text_models = sorted(runs[text_col].dropna().unique())
    image_models = sorted(runs[img_col].dropna().unique())

    markers = ["o", "s", "^", "D", "P", "X", "v", ">", "<"]
    im_to_marker = {im: markers[i % len(markers)] for i, im in enumerate(image_models)}

    fig = plt.figure(figsize=(8.4, 5.8))
    ax = fig.add_subplot(111)

    # Plot each (text,image) combo as a series (9 legend entries total)
    for t in text_models:
        sub_t = runs[runs[text_col] == t]
        for im in image_models:
            sub = sub_t[sub_t[img_col] == im]
            if sub.empty:
                continue
            ax.scatter(
                sub[r1_col].to_numpy(),
                sub[r50_col].to_numpy(),
                marker=im_to_marker[im],
                alpha=0.7,
                label=f"{t} | {im}",
            )

    tmp = runs.sort_values(r1_col, ascending=False).reset_index()
    frontier_idx = []
    best_y = -1e18
    for _, row in tmp.iterrows():
        yy = float(row[r50_col])
        if yy > best_y + 1e-12:
            frontier_idx.append(int(row["index"]))
            best_y = yy

    # ---- Annotations ----
    def _label_for_row(r) -> str:
        return f'k={int(r[k_col])},p={int(r[p_col])}'

    if annotate == "frontier":
        for _, r in runs.loc[frontier_idx].iterrows():
            ax.annotate(
                _label_for_row(r),
                (float(r[r1_col]), float(r[r50_col])),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )

    elif annotate == "all":
        for _, r in runs.iterrows():
            ax.annotate(
                _label_for_row(r),
                (float(r[r1_col]), float(r[r50_col])),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=6,
                alpha=0.8,
            )

    if annotate_topn and annotate_topn > 0:
        top = runs.sort_values(r1_col, ascending=False).head(int(annotate_topn))
        for _, r in top.iterrows():
            ax.annotate(
                _label_for_row(r),
                (float(r[r1_col]), float(r[r50_col])),
                textcoords="offset points",
                xytext=(4, -10),
                fontsize=7,
            )

    ax.set_xlabel("Overall R@1")
    ax.set_ylabel("Overall R@50")
    ax.set_title("Scatter plot across ALL model pairs and (k,p)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, frameon=True)

    out_path = os.path.join(out_dir, "scatter_plot_all_pairs_R1_vs_R50.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def make_best_bar_chart(df_long: pd.DataFrame, out_dir: str) -> str:
    run_col = _pick_col(df_long, ["run_name", "run", "name"])
    text_col = _pick_col(df_long, ["text_model", "text"])
    img_col = _pick_col(df_long, ["image_model", "image"])
    k_col = _pick_col(df_long, ["k"])
    p_col = _pick_col(df_long, ["p"])
    r1_col = _pick_col(df_long, ["overall.R@1", "overall_r1", "overall_R@1"])

    required = [text_col, img_col, k_col, p_col, r1_col]
    if any(c is None for c in required):
        missing = [n for n, c in zip(
            ["text_model", "image_model", "k", "p", "overall.R@1"], required
        ) if c is None]
        raise RuntimeError(f"Missing required columns for bar chart: {missing}")

    runs = df_long.copy()
    if run_col:
        runs = runs.drop_duplicates(subset=[run_col]).copy()
    else:
        runs = runs.drop_duplicates(subset=[text_col, img_col, k_col, p_col]).copy()

    _to_numeric(runs, [k_col, p_col, r1_col])
    runs = runs.dropna(subset=[k_col, p_col, r1_col])

    best_rows = []
    for (t, im), g in runs.groupby([text_col, img_col]):
        if g.empty:
            continue
        idx = g[r1_col].idxmax()
        best_rows.append(g.loc[idx])

    best = pd.DataFrame(best_rows)
    best["pair"] = best[img_col].astype(str) + " + " + best[text_col].astype(str)
    best = best.sort_values(r1_col, ascending=False)

    fig = plt.figure(figsize=(11.6, 5.2))
    ax = fig.add_subplot(111)
    ax.bar(best["pair"], best[r1_col])
    ax.set_ylabel("Best Overall R@1 (max over k,p)")
    ax.set_title("Best top-1 recall per model pair (best k,p)")
    ax.set_xticklabels(best["pair"], rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    # annotate k,p
    for i, (_, r) in enumerate(best.iterrows()):
        ax.text(
            i,
            float(r[r1_col]),
            f'k={int(r[k_col])}, p={int(r[p_col])}',
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    out_path = os.path.join(out_dir, "bar_best_R1_per_model_pair.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        required=True,
        help="Path to rr_sweep_per_language.csv (long-form per-language results).",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Directory to write plots into.",
    )
    ap.add_argument(
        "--no_overall_line",
        action="store_true",
        help="Do NOT overlay overall.R@1 dotted lines on KP plots.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)

    created = []
    created += make_kp_language_distribution_plots(
        df, args.out_dir, show_overall_line=(not args.no_overall_line)
    )
    created.append(make_scatter_plot(df, args.out_dir, annotate="frontier"))
    created.append(make_best_bar_chart(df, args.out_dir))
    created.append(make_per_language_bar_charts(df, args.out_dir))

    print(f"[OK] Wrote {len(created)} plot files to: {args.out_dir}")


if __name__ == "__main__":
    main()