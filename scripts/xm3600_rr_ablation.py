#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch


def _load_base_module() -> Any:
    candidates = [
        os.environ.get("XM3600_RR_BASE_SCRIPT", ""),
        "/mnt/data/shared/npa-multilingual/scripts/xm3600_rr_retrieval.py",
        str(Path(__file__).with_name("xm3600_rr_retrieval.py")),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("xm3600_rr_retrieval_base", path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        "Could not locate xm3600_rr_retrieval.py. "
        "Set XM3600_RR_BASE_SCRIPT or place the base script beside this file."
    )


BASE = _load_base_module()


@dataclass
class AnchorSubset:
    img_anchors: torch.Tensor
    txt_anchors: torch.Tensor
    n: int
    indices: torch.Tensor


def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_anchor_indices(path: str) -> torch.Tensor:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"anchor indices file not found: {path}")

    if p.suffix == ".pt":
        idx = torch.load(p, map_location="cpu")
        if not torch.is_tensor(idx):
            idx = torch.tensor(idx, dtype=torch.long)
        idx = idx.to(dtype=torch.long).flatten().cpu()
    elif p.suffix == ".npy":
        idx = torch.from_numpy(np.load(p)).to(dtype=torch.long).flatten().cpu()
    elif p.suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "indices" in obj:
            obj = obj["indices"]
        idx = torch.tensor(obj, dtype=torch.long).flatten().cpu()
    else:
        with p.open("r", encoding="utf-8") as f:
            vals = [int(line.strip()) for line in f if line.strip()]
        idx = torch.tensor(vals, dtype=torch.long).flatten().cpu()

    if idx.numel() == 0:
        raise ValueError("anchor indices file is empty")
    if (idx < 0).any():
        raise ValueError("anchor indices must be non-negative")
    if torch.unique(idx).numel() != idx.numel():
        raise ValueError("anchor indices contain duplicates; expected unique anchors within a run")
    return idx


def _load_embedding_subset(
    *,
    cache_dir: str,
    modality: str,
    model_tag: str,
    indices: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    normalize: bool,
) -> torch.Tensor:
    if modality == "image":
        shard_patterns = ("image_embs_*.pt", "image_embs.pt")
        legacy_single = ("image_embs.pt",)
        prefix = "image"
    elif modality == "text":
        shard_patterns = ("caption_embs_*.pt", "text_embs_*.pt", "caption_embs.pt", "text_embs.pt")
        legacy_single = ("caption_embs.pt",)
        prefix = "text"
    else:
        raise ValueError(modality)

    shard_paths = BASE._find_embedding_files(
        cache_dir,
        modality=modality,
        model_tag=model_tag,
        shard_patterns=shard_patterns,
        legacy_single_files=legacy_single,
    )
    if not shard_paths:
        raise FileNotFoundError(
            f"[CC12M] Could not find {modality} embeddings for tag='{model_tag}' under {cache_dir}."
        )

    # Fast path: single legacy file.
    if len(shard_paths) == 1 and os.path.basename(shard_paths[0]) in legacy_single:
        t = torch.load(shard_paths[0], map_location="cpu")
        if t.ndim != 2:
            raise ValueError(f"Expected 2D tensor in {shard_paths[0]}, got {tuple(t.shape)}")
        max_idx = int(indices.max().item())
        if max_idx >= t.shape[0]:
            raise IndexError(f"Anchor index {max_idx} out of range for {prefix} tensor with {t.shape[0]} rows")
        sel = t.index_select(0, indices)
        sel = sel.to(dtype=dtype).to(device, non_blocking=True)
        if normalize:
            sel = BASE.l2_normalize(sel)
        return sel

    meta_path = os.path.join(cache_dir, "embeddings", modality, model_tag, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"Expected metadata for subset loading at {meta_path}. "
            "Multi-shard random subset loading requires emb_chunk_size metadata."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    emb_chunk_size = int(meta.get("emb_chunk_size", 0))
    n_samples = int(meta.get("n_samples", 0))
    if emb_chunk_size <= 0 or n_samples <= 0:
        raise ValueError(f"Invalid emb_chunk_size/n_samples in {meta_path}")

    max_idx = int(indices.max().item())
    if max_idx >= n_samples:
        raise IndexError(f"Anchor index {max_idx} out of range for {prefix} cache with {n_samples} rows")

    idx_np = indices.numpy().astype(np.int64, copy=False)
    positions = np.arange(idx_np.shape[0], dtype=np.int64)
    shard_ids = idx_np // emb_chunk_size
    local_offsets = idx_np % emb_chunk_size

    out: Optional[torch.Tensor] = None
    unique_shards = sorted(set(int(x) for x in shard_ids.tolist()))

    for shard_id in unique_shards:
        shard_path = shard_paths[shard_id]
        mask = shard_ids == shard_id
        shard_positions = positions[mask]
        shard_local_offsets = local_offsets[mask]

        t = torch.load(shard_path, map_location="cpu")
        if t.ndim != 2:
            raise ValueError(f"Expected 2D tensor in {shard_path}, got {tuple(t.shape)}")

        off_t = torch.from_numpy(shard_local_offsets).to(dtype=torch.long)
        if off_t.numel() > 0 and int(off_t.max().item()) >= t.shape[0]:
            raise IndexError(
                f"Shard-local anchor index {int(off_t.max().item())} out of range for {shard_path} "
                f"with {t.shape[0]} rows"
            )

        chunk = t.index_select(0, off_t).to(dtype=dtype).to(device, non_blocking=True)
        if normalize:
            chunk = BASE.l2_normalize(chunk)

        if out is None:
            out = torch.empty((indices.numel(), chunk.shape[1]), device=device, dtype=chunk.dtype)

        pos_t = torch.from_numpy(shard_positions).to(device=device, dtype=torch.long)
        out.index_copy_(0, pos_t, chunk)

        del t, off_t, pos_t, chunk

    if out is None:
        raise RuntimeError(f"No {prefix} anchors loaded from subset indices")
    return out


@torch.no_grad()
def load_cc12m_anchor_subset(
    cc_cache_dir: str,
    anchor_indices: torch.Tensor,
    text_model_tag: str,
    image_model_tag: str,
    device: torch.device,
    dtype: torch.dtype,
    normalize: bool = True,
) -> AnchorSubset:
    img_anchors = _load_embedding_subset(
        cache_dir=cc_cache_dir,
        modality="image",
        model_tag=image_model_tag,
        indices=anchor_indices,
        device=device,
        dtype=dtype,
        normalize=normalize,
    )
    txt_anchors = _load_embedding_subset(
        cache_dir=cc_cache_dir,
        modality="text",
        model_tag=text_model_tag,
        indices=anchor_indices,
        device=device,
        dtype=dtype,
        normalize=normalize,
    )

    n = int(anchor_indices.numel())
    if img_anchors.shape[0] != n or txt_anchors.shape[0] != n:
        raise RuntimeError(
            f"Anchor subset mismatch: requested {n}, loaded img={img_anchors.shape[0]}, txt={txt_anchors.shape[0]}"
        )
    return AnchorSubset(
        img_anchors=img_anchors,
        txt_anchors=txt_anchors,
        n=n,
        indices=anchor_indices,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="XM3600 caption→image retrieval using CC12M-based relative representations with explicit anchor subsets"
    )
    ap.add_argument("--xm_cache", required=True, help="XM3600 cache dir")
    ap.add_argument("--cc_cache", required=True, help="CC12M cache dir")
    ap.add_argument("--out_dir", required=True, help="Output directory (caches + results.json)")

    ap.add_argument(
        "--text_model",
        default="nllb-200-3.3B",
        nargs=1,
        help="Text model tag (cache folder name).",
    )
    ap.add_argument(
        "--image_model",
        default="dinov3-vit7b16",
        nargs=1,
        help="Image model tag (cache folder name).",
    )

    ap.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0 or cpu")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="Compute dtype for embeddings")

    ap.add_argument("--anchor_n", type=int, default=1_600_000, help="Number of CC12M anchors (used only if --anchor_indices_path is not provided)")
    ap.add_argument("--anchor_indices_path", default=None, help="Optional path to a tensor/list of explicit CC12M anchor indices")
    ap.add_argument("--k", type=int, default=800, help="Non-zeros per RR (top-k)")
    ap.add_argument("--anchor_chunk", type=int, default=100_000, help="Chunk size over anchors when computing RR")

    ap.add_argument("--proc", choices=["none", "norm", "asif"], default="asif", help="RR value processing")
    ap.add_argument("--p", type=int, default=8, help="Exponent p for proc=asif")
    ap.add_argument("--clamp_min", type=float, default=None, help="Optional clamp min for similarities before processing")

    ap.add_argument("--image_batch", type=int, default=256, help="Batch size to compute RR for XM3600 images")
    ap.add_argument("--query_batch", type=int, default=2048, help="Caption query batch size")
    ap.add_argument("--inner_progress", action="store_true")
    ap.add_argument("--empty_cache_every", type=int, default=0)

    ap.add_argument("--Ks", default="1,5,10,25,50", help="Recall@K list")
    ap.add_argument("--languages", default=None, help="Comma-separated language codes to evaluate (default: all)")
    ap.add_argument("--one_per_image_per_lang", action="store_true")
    ap.add_argument("--max_queries", type=int, default=None)

    ap.add_argument("--reuse_cache", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    if isinstance(args.text_model, list):
        args.text_model = args.text_model[0]
    if isinstance(args.image_model, list):
        args.image_model = args.image_model[0]
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = BASE._dtype_from_str(args.dtype)

    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    BASE._ensure_dir(args.out_dir)

    Ks = [int(x) for x in args.Ks.split(",") if x.strip()]
    maxK = max(Ks)

    languages = None
    if args.languages:
        languages = [x.strip() for x in args.languages.split(",") if x.strip()]

    text_tag, text_repo = BASE._resolve_model_tag(args.text_model, BASE.TEXT_MODEL_PRESETS)
    img_tag, img_repo = BASE._resolve_model_tag(args.image_model, BASE.IMAGE_MODEL_PRESETS)

    anchor_indices = None
    anchor_indices_sha1 = None
    actual_anchor_n = int(args.anchor_n)
    if args.anchor_indices_path:
        anchor_indices = _load_anchor_indices(args.anchor_indices_path)
        actual_anchor_n = int(anchor_indices.numel())
        anchor_indices_sha1 = _sha1_file(args.anchor_indices_path)

    run_cfg = {
        "xm_cache": os.path.abspath(args.xm_cache),
        "cc_cache": os.path.abspath(args.cc_cache),
        "text_model": args.text_model,
        "text_tag": text_tag,
        "text_repo": text_repo,
        "image_model": args.image_model,
        "image_tag": img_tag,
        "image_repo": img_repo,
        "anchor_n": actual_anchor_n,
        "anchor_n_requested": int(args.anchor_n),
        "anchor_indices_path": None if not args.anchor_indices_path else os.path.abspath(args.anchor_indices_path),
        "anchor_indices_sha1": anchor_indices_sha1,
        "k": int(args.k),
        "anchor_chunk": int(args.anchor_chunk),
        "proc": args.proc,
        "p": int(args.p),
        "clamp_min": args.clamp_min,
        "dtype": args.dtype,
        "Ks": Ks,
        "languages": languages,
        "one_per_image_per_lang": bool(args.one_per_image_per_lang),
        "max_queries": args.max_queries,
        "seed": int(args.seed),
    }

    cached_cfg = BASE.load_cached_config(args.out_dir) if args.reuse_cache else None
    cfg_matches = (cached_cfg == run_cfg)

    print(
        f"[{BASE._now()}] device={device} dtype={dtype} proc={args.proc} k={args.k} "
        f"anchor_n={actual_anchor_n}"
    )
    print(
        f"[{BASE._now()}] text_model={args.text_model} (tag={text_tag}) | "
        f"image_model={args.image_model} (tag={img_tag})"
    )
    if args.anchor_indices_path:
        print(f"[{BASE._now()}] anchor_indices_path={args.anchor_indices_path}")
    print(f"[{BASE._now()}] out_dir={args.out_dir}")

    print(f"[{BASE._now()}] Loading XM3600 cache from {args.xm_cache}")
    xm = BASE.load_xm3600_cache(
        args.xm_cache,
        text_model_tag=text_tag,
        image_model_tag=img_tag,
        device=device,
        dtype=dtype,
        normalize=True,
    )

    xm_image_embs = xm["image_embs"]
    xm_caption_embs = xm["caption_embs"]
    xm_caption_index = xm["caption_index"]
    n_images = int(xm_image_embs.shape[0])

    if anchor_indices is not None:
        anchors = load_cc12m_anchor_subset(
            args.cc_cache,
            anchor_indices=anchor_indices,
            text_model_tag=text_tag,
            image_model_tag=img_tag,
            device=device,
            dtype=dtype,
            normalize=True,
        )
    else:
        anchors = BASE.load_cc12m_anchors(
            args.cc_cache,
            anchor_n=args.anchor_n,
            text_model_tag=text_tag,
            image_model_tag=img_tag,
            device=device,
            dtype=dtype,
            normalize=True,
        )
    n_anchors = anchors.n

    if xm_image_embs.shape[1] != anchors.img_anchors.shape[1]:
        raise ValueError(f"Image dim mismatch: XM {tuple(xm_image_embs.shape)} vs CC anchors {tuple(anchors.img_anchors.shape)}")
    if xm_caption_embs.shape[1] != anchors.txt_anchors.shape[1]:
        raise ValueError(f"Text dim mismatch: XM {tuple(xm_caption_embs.shape)} vs CC anchors {tuple(anchors.txt_anchors.shape)}")

    rr_img_idxs_path = os.path.join(args.out_dir, "xm_image_rr_idxs.pt")
    rr_img_vals_path = os.path.join(args.out_dir, "xm_image_rr_vals.pt")
    post_indptr_path = os.path.join(args.out_dir, "post_indptr.pt")
    post_imgids_path = os.path.join(args.out_dir, "post_imgids_sorted.pt")
    post_vals_path = os.path.join(args.out_dir, "post_vals_sorted.pt")

    if args.reuse_cache and cfg_matches and all(
        os.path.isfile(p)
        for p in [rr_img_idxs_path, rr_img_vals_path, post_indptr_path, post_imgids_path, post_vals_path]
    ):
        print(f"[{BASE._now()}] Reusing cached XM image RR + postings (config matches)")
        idxs_img = torch.load(rr_img_idxs_path, map_location=device)
        vals_img = torch.load(rr_img_vals_path, map_location=device)
        postings = BASE.ImagePostingIndex(
            indptr=torch.load(post_indptr_path, map_location=device),
            img_ids_sorted=torch.load(post_imgids_path, map_location=device),
            img_vals_sorted=torch.load(post_vals_path, map_location=device),
            n_images=n_images,
            n_anchors=n_anchors,
        )
    else:
        print(f"[{BASE._now()}] Computing XM3600 image RR...")
        idxs_img_list: List[torch.Tensor] = []
        vals_img_list: List[torch.Tensor] = []

        for start in BASE.tqdm(range(0, n_images, args.image_batch), desc="[XM] Image RR"):
            end = min(n_images, start + args.image_batch)
            q = xm_image_embs[start:end]
            idxs, vals = BASE.compute_rr_topk(
                q,
                anchors.img_anchors,
                k=args.k,
                anchor_chunk=args.anchor_chunk,
                desc=f"[RR-img] anchors chunking ({start}:{end})",
                show_progress=args.inner_progress,
            )
            vals = BASE.process_rr_values(vals, proc=args.proc, p=args.p, clamp_min=args.clamp_min)
            idxs_img_list.append(idxs)
            vals_img_list.append(vals)

        idxs_img = torch.cat(idxs_img_list, dim=0)
        vals_img = torch.cat(vals_img_list, dim=0)

        torch.save(idxs_img.detach().cpu(), rr_img_idxs_path)
        torch.save(vals_img.detach().cpu(), rr_img_vals_path)
        print(f"[{BASE._now()}] Saved image RR to {rr_img_idxs_path} / {rr_img_vals_path}")

        print(f"[{BASE._now()}] Building postings...")
        postings = BASE.build_image_posting_index(
            idxs_img=idxs_img,
            vals_img=vals_img,
            n_anchors=n_anchors,
            device=device,
        )

        torch.save(postings.indptr.detach().cpu(), post_indptr_path)
        torch.save(postings.img_ids_sorted.detach().cpu(), post_imgids_path)
        torch.save(postings.img_vals_sorted.detach().cpu(), post_vals_path)
        print(f"[{BASE._now()}] Saved postings to {post_indptr_path} / {post_imgids_path} / {post_vals_path}")

        BASE.save_config(args.out_dir, run_cfg)

    sel_rows, gt_img, lang_codes = BASE.select_caption_indices(
        xm_caption_index,
        languages=languages,
        one_per_image_per_lang=args.one_per_image_per_lang,
        max_queries=args.max_queries,
    )

    if len(sel_rows) == 0:
        raise RuntimeError("No captions selected. Check --languages filter.")

    unique_langs = sorted(set(lang_codes))
    lang_to_id = {lc: i for i, lc in enumerate(unique_langs)}
    lang_ids_all = torch.tensor([lang_to_id[lc] for lc in lang_codes], dtype=torch.long)

    sel_rows_t = torch.tensor(sel_rows, dtype=torch.long, device=device)
    gt_img_t = torch.tensor(gt_img, dtype=torch.long, device=device)
    lang_ids_t = lang_ids_all.to(device=device)

    print(f"[{BASE._now()}] Selected {len(sel_rows)} caption queries across {len(unique_langs)} languages")

    counters = BASE.init_counters(Ks, unique_langs)

    postings = BASE.ImagePostingIndex(
        indptr=postings.indptr.to(device=device),
        img_ids_sorted=postings.img_ids_sorted.to(device=device),
        img_vals_sorted=postings.img_vals_sorted.to(device=device),
        n_images=postings.n_images,
        n_anchors=postings.n_anchors,
    )

    nQ = sel_rows_t.numel()
    print(f"[{BASE._now()}] Starting retrieval evaluation (batches of {args.query_batch})")

    for bi, start in enumerate(BASE.tqdm(range(0, nQ, args.query_batch), desc="[EVAL] Caption batches")):
        end = min(nQ, start + args.query_batch)
        batch_sel = sel_rows_t[start:end]
        batch_gt = gt_img_t[start:end]
        batch_lang = lang_ids_t[start:end]

        abs_caps = xm_caption_embs.index_select(0, batch_sel)

        idxs_q, vals_q = BASE.compute_rr_topk(
            abs_caps,
            anchors.txt_anchors,
            k=args.k,
            anchor_chunk=args.anchor_chunk,
            desc="[RR-txt] anchors chunking",
            show_progress=args.inner_progress,
        )
        vals_q = BASE.process_rr_values(vals_q, proc=args.proc, p=args.p, clamp_min=args.clamp_min)

        scores = BASE.score_caption_batch(idxs_q, vals_q, postings)
        _, top_idx = torch.topk(scores, k=maxK, dim=1)

        BASE.update_counters(counters, top_idx, batch_gt, batch_lang)

        del abs_caps, idxs_q, vals_q, scores, top_idx
        if args.empty_cache_every > 0 and device.type == "cuda":
            if (bi + 1) % args.empty_cache_every == 0:
                torch.cuda.empty_cache()

    results = BASE.counters_to_results(counters)
    results.update({"run_config": run_cfg, "timestamp": BASE._now()})

    results_path = os.path.join(args.out_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[{BASE._now()}] Done. Wrote {results_path}")
    print("Overall:")
    for K in Ks:
        print(f"  R@{K}: {results['overall'][f'R@{K}']:.4f}")


if __name__ == "__main__":
    main()