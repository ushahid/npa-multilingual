#!/usr/bin/env python3
import os
import json
import time
import argparse
from typing import Dict, List, Optional, Sequence, Tuple

import torch


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


@torch.no_grad()
def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = torch.linalg.norm(x.float(), dim=1, keepdim=True).clamp_min(eps)
    return (x.float() / denom).to(dtype=x.dtype)


def load_cache(cache_dir: str, device: str = "cpu") -> Dict[str, object]:
    image_index_path = os.path.join(cache_dir, "image_index.json")
    caption_index_path = os.path.join(cache_dir, "caption_index.json")
    image_embs_path = os.path.join(cache_dir, "image_embs.pt")
    caption_embs_path = os.path.join(cache_dir, "caption_embs.pt")
    meta_path = os.path.join(cache_dir, "meta.json")

    with open(image_index_path, "r", encoding="utf-8") as f:
        image_index = json.load(f)
    with open(caption_index_path, "r", encoding="utf-8") as f:
        caption_index = json.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    image_embs = torch.load(image_embs_path, map_location=device)
    caption_embs = torch.load(caption_embs_path, map_location=device)

    return {
        "image_index": image_index,
        "caption_index": caption_index,
        "image_embs": image_embs,
        "caption_embs": caption_embs,
        "meta": meta,
    }


def select_caption_indices(
    caption_index: List[Dict],
    languages: Optional[Sequence[str]],
    one_per_image_per_lang: bool,
    max_queries: Optional[int],
) -> Tuple[List[int], List[int], List[str]]:
    langs_set = set(languages) if languages else None

    selected: List[int] = []
    gt: List[int] = []
    lang_codes: List[str] = []
    seen = set() if one_per_image_per_lang else None

    for ci, meta in enumerate(caption_index):
        lang = str(meta.get("language"))
        if langs_set is not None and lang not in langs_set:
            continue

        img_idx = int(meta.get("image_idx"))
        if seen is not None:
            key = (img_idx, lang)
            if key in seen:
                continue
            seen.add(key)

        selected.append(ci)
        gt.append(img_idx)
        lang_codes.append(lang)

        if max_queries is not None and len(selected) >= max_queries:
            break

    return selected, gt, lang_codes


def init_gpu_counters(Ks: List[int], n_langs: int, device: torch.device):
    Ks = sorted(Ks)
    return {
        "Ks": Ks,
        "total": torch.zeros((), dtype=torch.long, device=device),
        "hits": torch.zeros(len(Ks), dtype=torch.long, device=device),
        "mrr_sum": torch.zeros((), dtype=torch.float64, device=device),
        "total_by_lang": torch.zeros(n_langs, dtype=torch.long, device=device),
        "hits_by_lang": torch.zeros(len(Ks), n_langs, dtype=torch.long, device=device),
        "mrr_by_lang": torch.zeros(n_langs, dtype=torch.float64, device=device),
    }


@torch.no_grad()
def update_counters_gpu(
    counters,
    top_idx: torch.Tensor,
    gt_img: torch.Tensor,
    lang_ids: torch.Tensor,
):
    """
    top_idx: (B, maxK) on GPU
    gt_img:  (B,) on GPU
    lang_ids:(B,) on GPU
    """
    B = gt_img.shape[0]
    n_langs = counters["total_by_lang"].shape[0]

    match = (top_idx == gt_img.unsqueeze(1))  # (B, maxK), bool

    counters["total"] += B
    counters["total_by_lang"] += torch.bincount(lang_ids, minlength=n_langs)

    any_match = match.any(dim=1)
    first_rank = torch.argmax(match.to(torch.int32), dim=1) + 1
    rr = torch.where(
        any_match,
        1.0 / first_rank.to(torch.float32),
        torch.zeros(B, device=gt_img.device, dtype=torch.float32),
    )
    counters["mrr_sum"] += rr.sum(dtype=torch.float64)
    counters["mrr_by_lang"] += torch.bincount(
        lang_ids,
        weights=rr.to(torch.float64),
        minlength=n_langs,
    )

    for ki, K in enumerate(counters["Ks"]):
        hit_k = match[:, :K].any(dim=1)
        counters["hits"][ki] += hit_k.sum()
        if hit_k.any():
            counters["hits_by_lang"][ki] += torch.bincount(
                lang_ids[hit_k],
                minlength=n_langs,
            )


def counters_to_results(counters, unique_langs: List[str]) -> Dict[str, object]:
    total = int(counters["total"].item())
    total_safe = max(1, total)

    hits = counters["hits"].detach().cpu()
    total_by_lang = counters["total_by_lang"].detach().cpu()
    hits_by_lang = counters["hits_by_lang"].detach().cpu()
    mrr_sum = float(counters["mrr_sum"].item())
    mrr_by_lang = counters["mrr_by_lang"].detach().cpu()

    overall = {
        f"R@{K}": float(hits[i].item()) / total_safe
        for i, K in enumerate(counters["Ks"])
    }
    overall["MRR"] = mrr_sum / total_safe

    per_lang = {}
    for li, lc in enumerate(unique_langs):
        denom = int(total_by_lang[li].item())
        if denom == 0:
            continue

        rec = {
            f"R@{K}": float(hits_by_lang[ki, li].item()) / denom
            for ki, K in enumerate(counters["Ks"])
        }
        rec["MRR"] = float(mrr_by_lang[li].item()) / denom
        per_lang[lc] = rec

    return {
        "overall": overall,
        "per_language": per_lang,
        "n_queries": total,
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Dense caption->image retrieval evaluation for cached XM3600 baselines")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--query_batch", type=int, default=2048)
    ap.add_argument("--Ks", default="1,5,10,25,50")
    ap.add_argument("--languages", default=None, help="Comma-separated languages to evaluate (default: all present in caption_index)")
    ap.add_argument("--one_per_image_per_lang", action="store_true")
    ap.add_argument("--max_queries", type=int, default=None)
    return ap.parse_args()


def _dtype_from_str(s: str) -> torch.dtype:
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32
    raise ValueError(s)


@torch.inference_mode()
def main():
    args = parse_args()
    _ensure_dir(args.out_dir)

    device = torch.device(args.device)
    dtype = _dtype_from_str(args.dtype)

    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    Ks = [int(x) for x in args.Ks.split(",") if x.strip()]
    maxK = max(Ks)
    languages = None
    if args.languages:
        languages = [x.strip() for x in args.languages.split(",") if x.strip()]

    print(f"[{_now()}] Loading cache from {args.cache_dir}")
    bundle = load_cache(args.cache_dir, device="cpu")
    image_index = bundle["image_index"]
    caption_index = bundle["caption_index"]
    image_embs = bundle["image_embs"]
    caption_embs = bundle["caption_embs"]
    meta = bundle["meta"]

    n_images = int(image_embs.shape[0])
    n_captions = int(caption_embs.shape[0])
    print(f"[{_now()}] Loaded {n_images} image embeddings and {n_captions} caption embeddings")

    # Move everything to GPU once.
    image_embs = image_embs.to(device=device, dtype=dtype, non_blocking=True)
    caption_embs = caption_embs.to(device=device, dtype=dtype, non_blocking=True)

    image_embs = l2_normalize(image_embs)
    caption_embs = l2_normalize(caption_embs)

    sel_rows, gt_img, lang_codes = select_caption_indices(
        caption_index,
        languages=languages,
        one_per_image_per_lang=args.one_per_image_per_lang,
        max_queries=args.max_queries,
    )
    if len(sel_rows) == 0:
        raise RuntimeError("No captions selected. Check --languages filter.")

    unique_langs = sorted(set(lang_codes))
    lang_to_id = {lc: i for i, lc in enumerate(unique_langs)}
    lang_ids = [lang_to_id[lc] for lc in lang_codes]

    counters = init_gpu_counters(Ks, len(unique_langs), device)

    sel_rows_t = torch.tensor(sel_rows, dtype=torch.long, device=device)
    gt_img_t = torch.tensor(gt_img, dtype=torch.long, device=device)
    lang_ids_t = torch.tensor(lang_ids, dtype=torch.long, device=device)

    print(f"[{_now()}] Selected {len(sel_rows)} queries across {len(unique_langs)} languages")
    print(f"[{_now()}] Embeddings now on {device}; evaluating in batches of {args.query_batch}")

    for start in range(0, len(sel_rows), args.query_batch):
        end = min(len(sel_rows), start + args.query_batch)

        batch_sel = sel_rows_t[start:end]
        batch_gt = gt_img_t[start:end]
        batch_lang_ids = lang_ids_t[start:end]

        q = caption_embs.index_select(0, batch_sel)   # all on GPU
        scores = q @ image_embs.T                     # all on GPU
        _, top_idx = torch.topk(scores, k=maxK, dim=1, largest=True, sorted=True)

        update_counters_gpu(counters, top_idx, batch_gt, batch_lang_ids)

        del q, scores, top_idx

    results = counters_to_results(counters, unique_langs)
    run_cfg = {
        "cache_dir": os.path.abspath(args.cache_dir),
        "dtype": args.dtype,
        "Ks": Ks,
        "languages": languages,
        "one_per_image_per_lang": bool(args.one_per_image_per_lang),
        "max_queries": args.max_queries,
        "query_batch": args.query_batch,
        "meta": meta,
    }
    results.update({"run_config": run_cfg, "timestamp": _now()})

    results_path = os.path.join(args.out_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[{_now()}] Done. Wrote {results_path}")
    print("Overall:")
    for K in Ks:
        print(f"  R@{K}: {results['overall'][f'R@{K}']:.4f}")
    print(f"  MRR: {results['overall']['MRR']:.4f}")


if __name__ == "__main__":
    main()