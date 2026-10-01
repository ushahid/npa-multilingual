#!/usr/bin/env python3
"""Generate similarity distribution plots between XM3600 anchors and CC12M candidates (caches).

This is a drop-in evolution of the original cc12m_xm3600_sims.py, updated to support the
new cache layout with *model-tagged* / decoupled embeddings:

cache_dir/
  index/
    (XM3600) images.jsonl, captions.jsonl, meta.json
    (CC12M)  samples.jsonl (newer) OR images.jsonl + captions.jsonl (if unified)
  embeddings/
    text/<model_tag>/...
    image/<model_tag>/...

New capabilities:
- Choose which TEXT model and which IMAGE model to use for plots.
- Accept model inputs as either preset keys (e.g. bge-m3) or HF repo ids
  (e.g. BAAI/bge-m3), and auto-resolve to an existing cached model_tag.

All previous functionality is preserved:
- Multiple anchors per run (--xm_image_idx ...)
- Multiple preferred languages per run (--anchor_language ...)
- Top-k retrieval on CC12M using IMAGE sims; then inspect TEXT sims of those top-k images.
- Histogram plots saved as PNG + info log TXT + top-captions CSV.

Notes:
- The *compute dtype* (--dtype) is independent of how saved embeddings.
- For CC12M, this script assumes 1 caption per image (as before).
"""

import os
import json
import argparse
import random
import csv
from typing import Optional, Dict, Any, List, Tuple, Iterable

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image


# ----------------------------
# Model presets (same as the cache builders)
# ----------------------------

TEXT_MODEL_PRESETS: Dict[str, str] = {
    "nllb-200-3.3B": "facebook/nllb-200-3.3B",
    "bge-m3": "BAAI/bge-m3",
    "qwen3-embed-8b": "Qwen/Qwen3-Embedding-8B",
}

IMAGE_MODEL_PRESETS: Dict[str, str] = {
    "dinov3-vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    # largest common CLIP vision checkpoint on HF
    "clip-vit-bigg14": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    # largest RADIO family checkpoint on HF
    "c-radio-v4-h": "nvidia/C-RADIOv4-H",
}


def normalize_model_input(model: str, modality: str) -> str:
    """Accept a few human-friendly aliases and map them to preset keys.

    This is helpful when orchestration scripts pass values like "CLIPImageEncoder".
    If the input is a real preset key or HF repo id, it is returned unchanged.
    """
    m = (model or "").strip()
    ml = m.lower().replace(" ", "")

    if modality == "image":
        if ml in {"clipimageencoder", "clipimageencoder()", "clip"}:
            return "clip-vit-bigg14"
        if ml in {"radioimageencoder", "radioimageencoder()", "radio"}:
            return "c-radio-v4-h"
        if ml in {"dinov3", "dinov3imageencoder", "dino"}:
            return "dinov3-vit7b16"

    if modality == "text":
        if ml in {"nllb", "nllbtextencoder"}:
            return "nllb-200-3.3B"
        if ml in {"bge", "bgeembedding", "bgetextencoder"}:
            return "bge-m3"
        if ml in {"qwen", "qwen3", "qwenembedding", "qwentextencoder"}:
            return "qwen3-embed-8b"

    return m


# ----------------------------
# Small helpers
# ----------------------------

def sanitize_for_fs(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.=+" else "_" for c in s).strip("_") or "model"


def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def _resolve_xm_image_path(
    rec: Dict[str, Any],
    *,
    xm_cache_dir: str,
    xm_images_dir: Optional[str] = None,
) -> Optional[str]:
    """Resolve an XM3600 image file path for visualization.

    Newer cache layout keeps *raw* images outside the cache (e.g. downloads/xm3600/unpackedImages)
    and stores only an index with `image_relpath` + `meta.json` containing `images_dir`.

    We try, in order:
      1) an explicit `image_path` inside the record (if present)
      2) join(xm_images_dir, image_relpath/relpath)
      3) legacy join(xm_cache_dir, "images", image_relpath) (older layouts)
      4) heuristic join(dirname(xm_cache_dir), "unpackedImages", image_relpath)
      5) try common extensions by `image_id` stem
    """
    # 1) explicit absolute path
    p = rec.get("image_path")
    if isinstance(p, str) and p and os.path.isfile(p):
        return p

    rel = rec.get("image_relpath") or rec.get("relpath")
    img_id = rec.get("image_id")

    # Build a small list of candidate base dirs
    base_dirs: List[str] = []
    if xm_images_dir:
        base_dirs.append(xm_images_dir)
    base_dirs.append(os.path.join(xm_cache_dir, "images"))  # legacy
    base_dirs.append(os.path.join(os.path.dirname(xm_cache_dir), "unpackedImages"))  # common tree

    exts = [".jpg", ".jpeg", ".png", ".webp"]

    # 2-4) check relpath against candidate dirs
    if isinstance(rel, str) and rel:
        if os.path.isabs(rel) and os.path.isfile(rel):
            return rel

        stem, ext = os.path.splitext(rel)
        for d in base_dirs:
            cand = os.path.join(d, rel)
            if os.path.isfile(cand):
                return cand

            # If rel already has an extension but it doesn't exist, try swapping it
            if ext:
                for e in exts:
                    cand2 = os.path.join(d, stem + e)
                    if os.path.isfile(cand2):
                        return cand2
            else:
                for e in exts:
                    cand2 = os.path.join(d, rel + e)
                    if os.path.isfile(cand2):
                        return cand2

    # 5) try by image_id stem
    if isinstance(img_id, str) and img_id:
        for d in base_dirs:
            for e in exts:
                cand = os.path.join(d, img_id + e)
                if os.path.isfile(cand):
                    return cand

    return None



def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _list_pt_shards(out_dir: str, prefix: str) -> List[str]:
    files: List[str] = []
    for fn in os.listdir(out_dir):
        if fn.startswith(prefix) and fn.endswith(".pt"):
            files.append(os.path.join(out_dir, fn))
    return sorted(files)


def _load_concat_shards(paths: List[str], device: str = "cpu") -> torch.Tensor:
    tensors = [torch.load(p, map_location=device) for p in paths]
    return torch.cat(tensors, dim=0)


def _find_cached_model_tag(
    cache_dir: str,
    modality: str,
    model_input: str,
    presets: Dict[str, str],
) -> Tuple[str, str, Dict[str, Any]]:
    """Resolve a user-provided model identifier to an *existing* cached model_tag.

    Returns (resolved_tag, resolved_repo_or_name, meta_json).

    Resolution strategy:
      1) Try direct folder matches via a few candidate tags.
      2) If not found, scan meta.json under embeddings/<modality>/* for a model_repo/model_name match.

    This makes the script robust to whether built caches using preset keys
    (e.g. --text_model bge-m3) or full HF repo ids (e.g. --text_model BAAI/bge-m3).
    """

    base = os.path.join(cache_dir, "embeddings", modality)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Missing embeddings/{modality} in {cache_dir}")

    # Candidate tags to try (in order)
    cand_tags: List[str] = []

    # If user passes a preset repo id, the cache might have been built under preset key tag
    for k, v in presets.items():
        if model_input == v:
            cand_tags.append(sanitize_for_fs(k))
            break

    # If user passes a preset key directly
    if model_input in presets:
        cand_tags.append(sanitize_for_fs(model_input))

    # If the cache was built using the raw user string
    cand_tags.append(sanitize_for_fs(model_input))

    # Try folder existence
    for tag in cand_tags:
        out_dir = os.path.join(base, tag)
        meta_path = os.path.join(out_dir, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            repo = meta.get("model_name") or meta.get("model_repo") or meta.get("model") or ""
            return tag, repo, meta

    # Fallback: scan all meta.json files for matching repo/name
    for tag in sorted(os.listdir(base)):
        out_dir = os.path.join(base, tag)
        meta_path = os.path.join(out_dir, "meta.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        repo = meta.get("model_name") or meta.get("model_repo") or meta.get("model") or ""
        if repo == model_input:
            return tag, repo, meta

        # Also accept preset key / preset repo equivalence
        if model_input in presets and repo == presets[model_input]:
            return tag, repo, meta
        for k, v in presets.items():
            if model_input == v and repo == v:
                return tag, repo, meta

    raise FileNotFoundError(
        f"No cached {modality} embeddings found for model='{model_input}' in {base}. "
        f"Available tags: {sorted(os.listdir(base))[:20]}{' ...' if len(os.listdir(base)) > 20 else ''}"
    )


# ---------------- XM3600 helpers (many captions per image) ---------------- #

def build_caption_lists_by_image(
    caption_index: List[Dict[str, Any]],
    n_images: int,
) -> List[List[int]]:
    caps_by_img: List[List[int]] = [[] for _ in range(n_images)]
    for j, rec in enumerate(caption_index):
        i = int(rec["image_idx"])
        if 0 <= i < n_images:
            caps_by_img[i].append(j)
    return caps_by_img


def pick_one_caption_for_image(
    image_idx: int,
    caption_index: List[Dict[str, Any]],
    captions_by_image: List[List[int]],
    preferred_lang: Optional[str] = None,
) -> Optional[int]:
    """Pick a single caption for the anchor image.

    - If preferred_lang exists for that image, pick it.
    - Else fall back to any caption for that image.
    """
    indices = captions_by_image[image_idx]
    if not indices:
        return None

    if preferred_lang is None:
        return indices[0]

    for j in indices:
        if caption_index[j].get("language") == preferred_lang:
            return j

    return indices[0]


# ---------------- CC12M helpers (1 caption per image) ---------------- #

def build_cc12m_caption_id_by_image(
    caption_index: List[Dict[str, Any]],
    n_images: int,
) -> List[int]:
    """Build cap_id_by_img[i] = caption_idx for that image, or -1 if missing."""
    cap_id_by_img = [-1] * n_images
    for j, rec in enumerate(caption_index):
        i = int(rec.get("image_idx", -1))
        if 0 <= i < n_images and cap_id_by_img[i] == -1:
            cap_id_by_img[i] = j
    return cap_id_by_img


# ---------------- Histogram helper (FAST: compute counts on GPU, plot counts only) ---------------- #

def compute_hist_counts(
    values: torch.Tensor,
    bins: int,
    vmin: float,
    vmax: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (bin_edges [bins+1], counts [bins]).

    Works on GPU and returns tensors on CPU for plotting.
    """
    # torch.histc on CUDA does NOT support float16, so cast just for hist.
    if values.dtype in (torch.float16, torch.bfloat16):
        values = values.float()
    counts = torch.histc(values, bins=bins, min=vmin, max=vmax)
    edges = torch.linspace(vmin, vmax, steps=bins + 1, device=values.device, dtype=torch.float32)
    return edges.detach().cpu(), counts.detach().cpu()


def plot_hist_from_counts(ax, edges_cpu: torch.Tensor, counts_cpu: torch.Tensor, label: str, alpha: float = 0.7):
    edges = edges_cpu.numpy()
    counts = counts_cpu.numpy()
    width = float(edges[1] - edges[0])
    ax.bar(edges[:-1], counts, width=width, align="edge", alpha=alpha, label=label)


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Similarity plots: anchors from XM3600 (multi-language captions), "
            "candidates = CC12M (English, 1 caption/image)."
        )
    )

    ap.add_argument("--xm_cache_dir", required=True)
    ap.add_argument(
        "--xm_images_dir",
        type=str,
        default=None,
        help=(
            "Path to the *raw* XM3600 images directory (e.g. downloads/xm3600/unpackedImages). "
            "If omitted, this is read from xm_cache_dir/index/meta.json (images_dir) or guessed."
        ),
    )
    ap.add_argument("--cc_cache_dir", required=True)

    # New: choose models (can pass multiple to generate multiple sets in one run)
    ap.add_argument(
        "--text_model",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Text model(s) for BOTH XM3600+CC12M. Accepts preset key or HF repo. "
            f"Preset keys: {sorted(TEXT_MODEL_PRESETS.keys())}. "
            "Examples: bge-m3 OR BAAI/bge-m3"
        ),
    )
    ap.add_argument(
        "--image_model",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Image model(s) for BOTH XM3600+CC12M. Accepts preset key or HF repo. "
            f"Preset keys: {sorted(IMAGE_MODEL_PRESETS.keys())}. "
            "Examples: dinov3-vit7b16 OR facebook/dinov3-vit7b16-pretrain-lvd1689m"
        ),
    )

    # Multiple anchors + multiple preferred languages in one run
    ap.add_argument(
        "--xm_image_idx",
        type=int,
        nargs="*",
        default=None,
        help="One or more XM3600 image indices. If omitted, one random image is used.",
    )
    ap.add_argument(
        "--anchor_language",
        type=str,
        nargs="*",
        default=None,
        help="One or more preferred XM3600 language codes (e.g. en hi fr). If omitted, uses any caption.",
    )

    ap.add_argument("--top_k", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=80)

    ap.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Where to run similarity computations (cuda:0 recommended for speed).",
    )
    ap.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Compute dtype for embeddings on device.",
    )

    ap.add_argument(
        "--out_root",
        type=str,
        default="/mnt/data/shared/npa-multilingual/plots/cc12m_sim_dist_from_xm3600_query",
    )
    ap.add_argument("--output", type=str, default="similarity")

    ap.add_argument(
        "--skip_missing",
        action="store_true",
        help="If a requested (model,modality) cache is missing, skip that combo instead of failing.",
    )

    return ap.parse_args()


def _load_xm_index(xm_cache_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    idx_dir = os.path.join(xm_cache_dir, "index")
    images_path = os.path.join(idx_dir, "images.jsonl")
    captions_path = os.path.join(idx_dir, "captions.jsonl")
    meta_path = os.path.join(idx_dir, "meta.json")
    if not (os.path.isfile(images_path) and os.path.isfile(captions_path) and os.path.isfile(meta_path)):
        raise FileNotFoundError(f"XM3600 index not found in {idx_dir}")
    images = _read_jsonl(images_path)
    captions = _read_jsonl(captions_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return images, captions, meta


def _load_cc_index(cc_cache_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Load CC12M index.

    Supports either:
      - official: index/samples.jsonl + index/meta.json
      - (old) unified:     index/images.jsonl + index/captions.jsonl + index/meta.json
    """
    idx_dir = os.path.join(cc_cache_dir, "index")
    meta_path = os.path.join(idx_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"CC12M meta.json not found in {idx_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    images_path = os.path.join(idx_dir, "images.jsonl")
    captions_path = os.path.join(idx_dir, "captions.jsonl")
    samples_path = os.path.join(idx_dir, "samples.jsonl")

    if os.path.isfile(images_path) and os.path.isfile(captions_path):
        images = _read_jsonl(images_path)
        captions = _read_jsonl(captions_path)
        return images, captions, meta

    if not os.path.isfile(samples_path):
        raise FileNotFoundError(f"CC12M index not found. Expected samples.jsonl or images+captions.jsonl in {idx_dir}")

    # Derive image_index + caption_index from samples.jsonl (1 caption per image)
    images: List[Dict[str, Any]] = []
    captions: List[Dict[str, Any]] = []

    # Streaming parse (keep behavior similar to the original script by holding indices in RAM)
    with open(samples_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # For CC12M, treat sample_id == image_idx == caption_idx
            img_idx = i
            images.append(
                {
                    "image_idx": img_idx,
                    "key": rec.get("key"),
                    "shard": rec.get("shard"),
                    "sample_id": rec.get("sample_id", img_idx),
                }
            )
            captions.append(
                {
                    "image_idx": img_idx,
                    "caption_idx": img_idx,
                    "language": rec.get("language", "en"),
                    "caption": rec.get("caption", ""),
                    "key": rec.get("key"),
                    "shard": rec.get("shard"),
                }
            )

    return images, captions, meta


def _load_cached_embeddings(
    cache_dir: str,
    modality: str,
    model_tag: str,
    prefix_candidates: List[str],
    device: str = "cpu",
) -> Tuple[torch.Tensor, Dict[str, Any], List[str]]:
    """Load and concat embedding shards from cache_dir/embeddings/<modality>/<model_tag>/.

    prefix_candidates: list like ["caption_embs"] or ["text_embs", "caption_embs"].
    """
    out_dir = os.path.join(cache_dir, "embeddings", modality, model_tag)
    meta_path = os.path.join(out_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing embedding meta.json: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    shard_paths: List[str] = []
    for prefix in prefix_candidates:
        shard_paths = _list_pt_shards(out_dir, prefix)
        if shard_paths:
            break

    if not shard_paths:
        raise FileNotFoundError(
            f"No embedding shards found in {out_dir} for prefixes {prefix_candidates}"
        )

    emb = _load_concat_shards(shard_paths, device=device)
    return emb, meta, shard_paths


def run_single(
    *,
    anchor_xm_img_idx: int,
    preferred_lang: Optional[str],
    xm_image_index: List[Dict[str, Any]],
    xm_caption_index: List[Dict[str, Any]],
    xm_caps_by_img: List[List[int]],
    xm_image_embs: torch.Tensor,      # normalized on device
    xm_caption_embs: torch.Tensor,    # normalized on device
    cc_image_index: List[Dict[str, Any]],
    cc_caption_index: List[Dict[str, Any]],
    cap_id_by_img: List[int],
    cc_image_embs: torch.Tensor,      # normalized on device
    cc_text_embs: torch.Tensor,       # normalized on device
    args: argparse.Namespace,
    model_prefix: str,
):
    info_lines: List[str] = []

    def log(msg: str):
        print(msg)
        info_lines.append(msg)

    # ----- pick anchor caption from XM3600 -----
    anchor_xm_cap_idx = pick_one_caption_for_image(
        anchor_xm_img_idx,
        xm_caption_index,
        xm_caps_by_img,
        preferred_lang=preferred_lang,
    )
    if anchor_xm_cap_idx is None:
        raise ValueError(f"XM3600 anchor image {anchor_xm_img_idx} has no captions.")

    anchor_cap_lang = xm_caption_index[anchor_xm_cap_idx]["language"]
    anchor_cap_text = xm_caption_index[anchor_xm_cap_idx]["caption"]
    # Resolve anchor image path (for visualization only; embeddings already loaded).
    rec = xm_image_index[anchor_xm_img_idx]
    anchor_img_path = _resolve_xm_image_path(
        rec,
        xm_cache_dir=args.xm_cache_dir,
        xm_images_dir=getattr(args, "xm_images_dir", None),
    )
    if anchor_img_path is None:
        rel = rec.get("image_relpath") or rec.get("relpath") or ""
        raise FileNotFoundError(
            "Could not resolve anchor image path for XM3600 image record. "
            f"image_id={rec.get('image_id')} relpath={rel!r}. "
            "Pass --xm_images_dir to point to the raw images directory (e.g. .../downloads/xm3600/unpackedImages)."
        )

    xm_img_id = xm_image_index[anchor_xm_img_idx].get("image_id", f"xm{anchor_xm_img_idx}")

    log(f"[INFO] Models: {model_prefix}")
    log(f"[INFO] Anchor XM image_idx={anchor_xm_img_idx}, image_id={xm_img_id}")
    log(f"[INFO] Preferred lang={preferred_lang if preferred_lang is not None else 'ANY'}")
    log(f"[INFO] Chosen anchor caption_idx={anchor_xm_cap_idx}, actual_lang={anchor_cap_lang}")
    log(f"[INFO] Anchor caption: {anchor_cap_text}")

    anchor_img_vec = xm_image_embs[anchor_xm_img_idx]     # [D_img]
    anchor_cap_vec = xm_caption_embs[anchor_xm_cap_idx]   # [D_txt]

    # ----- compute similarities on device (FAST) -----
    img_sims = cc_image_embs @ anchor_img_vec            # [N_cc]
    txt_sims = cc_text_embs @ anchor_cap_vec             # [N_cc_caps] (≈ N_cc)

    cc_n_images = cc_image_embs.shape[0]
    k = min(args.top_k, cc_n_images)

    top_vals, top_idx = torch.topk(img_sims, k=k, largest=True, sorted=True)
    topk_img_indices = top_idx
    actual_k = int(topk_img_indices.shape[0])

    # ----- hist counts (avoid moving N floats to CPU) -----
    bins = args.bins
    hist_range = (-0.2, 1.0)
    vmin, vmax = hist_range

    edges_img_all, counts_img_all = compute_hist_counts(img_sims, bins=bins, vmin=vmin, vmax=vmax)
    edges_img_top, counts_img_top = compute_hist_counts(top_vals, bins=bins, vmin=vmin, vmax=vmax)

    edges_txt_all, counts_txt_all = compute_hist_counts(txt_sims, bins=bins, vmin=vmin, vmax=vmax)

    # top-k captions are exactly the captions of those top-k images (1 caption per image)
    top_caption_records: List[Dict[str, Any]] = []
    top_txt_vals: List[float] = []

    topk_img_list = topk_img_indices.tolist()
    for rank, img_i in enumerate(topk_img_list, start=1):
        cid = cap_id_by_img[img_i]
        if cid == -1:
            continue
        sim = float(txt_sims[cid].item())
        rec = cc_caption_index[cid]
        top_txt_vals.append(sim)
        top_caption_records.append(
            {
                "rank": rank,
                "cc_image_idx": img_i,
                "cc_caption_idx": cid,
                "cc_key": rec.get("key"),
                "cc_shard": rec.get("shard"),
                "language": rec.get("language", "en"),
                "similarity_to_anchor_caption": sim,
                "caption": rec.get("caption", ""),
            }
        )

    if len(top_txt_vals) == 0:
        raise ValueError("No CC12M captions found for top-k images; check caption_index mapping.")

    top_txt_tensor = torch.tensor(top_txt_vals, device="cpu", dtype=torch.float32)
    # hist on CPU (only ~k values)
    top_txt_edges = torch.linspace(vmin, vmax, steps=bins + 1)
    top_txt_counts = torch.histc(top_txt_tensor, bins=bins, min=vmin, max=vmax)

    # ----- stats -----
    mean_all_img = float(img_sims.mean().item())
    mean_top_img = float(top_vals.mean().item())
    mean_all_txt = float(txt_sims.mean().item())
    mean_top_txt = float(top_txt_tensor.mean().item())

    log(f"[STATS] CC12M image sims: mean(all)={mean_all_img:.3f}, mean(top-{actual_k})={mean_top_img:.3f}")
    log(f"[STATS] CC12M text sims:  mean(all)={mean_all_txt:.3f}, mean(top-{len(top_txt_vals)})={mean_top_txt:.3f}")

    # ----- output paths -----
    out_root = os.path.expanduser(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(args.output))[0] if args.output else "similarity"
    xm_img_id_safe = sanitize_for_fs(str(xm_img_id))
    pref_safe = sanitize_for_fs(preferred_lang) if preferred_lang is not None else "any"
    act_safe = sanitize_for_fs(anchor_cap_lang)

    prefix = (
        f"{base_name}"
        f"__{sanitize_for_fs(model_prefix)}"
        f"__xmidx-{anchor_xm_img_idx}"
        f"__xmid-{xm_img_id_safe}"
        f"__pref-{pref_safe}"
        f"__act-{act_safe}"
        f"__topk-{actual_k}"
    )

    plot_path = os.path.join(out_root, f"{prefix}.png")
    meta_path_txt = os.path.join(out_root, f"{prefix}.txt")
    meta_path_csv = os.path.join(out_root, f"{prefix}__topcaps.csv")

    # ----- plot -----
    img = None
    try:
        img = Image.open(anchor_img_path).convert("RGB")
    except Exception as e:
        log(f"[WARN] Could not open anchor image for visualization: {anchor_img_path} ({e})")

    fig, axes = plt.subplots(3, 1, figsize=(5, 8), gridspec_kw={"height_ratios": [2, 3, 3]})
    ax0, ax1, ax2 = axes

    if img is not None:
        ax0.imshow(img)
    else:
        ax0.set_xlim(0, 1)
        ax0.set_ylim(0, 1)
        ax0.text(0.5, 0.5, "anchor image not available", ha="center", va="center", fontsize=10)
    ax0.axis("off")
    title_cap = anchor_cap_text
    if len(title_cap) > 120:
        title_cap = title_cap[:117] + "..."
    ax0.set_title(title_cap, fontsize=9)

    # Image similarities
    plot_hist_from_counts(ax1, edges_img_all, counts_img_all, label="All CC12M images", alpha=0.7)
    plot_hist_from_counts(ax1, edges_img_top, counts_img_top, label=f"Top {actual_k} CC12M images", alpha=0.7)
    ax1.set_yscale("log")
    ax1.set_xlabel("Image similarities (cosine)")
    ax1.set_ylabel("Count (log)")
    ax1.set_title(f"CC12M image similarities ({model_prefix})")
    ax1.axvline(mean_all_img, linestyle="--", color="C0")
    ax1.axvline(mean_top_img, linestyle="--", color="C1")
    y1_max = ax1.get_ylim()[1]
    ax1.text(mean_all_img, y1_max * 0.15, f"{mean_all_img:.2f}", color="C0", ha="center")
    ax1.text(mean_top_img, y1_max * 0.35, f"{mean_top_img:.2f}", color="C1", ha="center")
    ax1.legend()

    # Text similarities
    plot_hist_from_counts(ax2, edges_txt_all, counts_txt_all, label=f"All CC12M captions (en) vs anchor ({anchor_cap_lang})", alpha=0.7)
    plot_hist_from_counts(ax2, top_txt_edges, top_txt_counts, label=f"Captions of top {len(top_txt_vals)} images vs ({anchor_cap_lang})", alpha=0.7)
    ax2.set_yscale("log")
    ax2.set_xlabel("Text similarities (cosine)")
    ax2.set_ylabel("Count (log)")
    ax2.set_title(f"CC12M text similarities ({model_prefix})")
    ax2.axvline(mean_all_txt, linestyle="--", color="C0")
    ax2.axvline(mean_top_txt, linestyle="--", color="C1")
    y2_max = ax2.get_ylim()[1]
    ax2.text(mean_all_txt, y2_max * 0.15, f"{mean_all_txt:.2f}", color="C0", ha="center")
    ax2.text(mean_top_txt, y2_max * 0.35, f"{mean_top_txt:.2f}", color="C1", ha="center")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    log(f"[INFO] Saved plot: {plot_path}")
    log(f"[INFO] XM cache_dir={args.xm_cache_dir}")
    log(f"[INFO] CC cache_dir={args.cc_cache_dir}")
    log(f"[INFO] device={args.device}, dtype={args.dtype}")

    with open(meta_path_txt, "w", encoding="utf-8") as f:
        for ln in info_lines:
            f.write(ln + "\n")
    print(f"[DONE] Saved info log: {meta_path_txt}")

    with open(meta_path_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank_in_top_images",
                "cc_image_idx",
                "cc_caption_idx",
                "cc_key",
                "cc_shard",
                "language",
                "similarity_to_anchor_caption",
                "caption",
            ]
        )
        for rec in top_caption_records:
            caption_flat = rec["caption"].replace("\n", " ").strip()
            writer.writerow(
                [
                    rec["rank"],
                    rec["cc_image_idx"],
                    rec["cc_caption_idx"],
                    rec["cc_key"],
                    rec["cc_shard"],
                    rec["language"],
                    f"{rec['similarity_to_anchor_caption']:.6f}",
                    caption_flat,
                ]
            )
    print(f"[DONE] Saved top-captions CSV: {meta_path_csv}")


def main(args):
    device = torch.device(args.device)
    dtype = _dtype_from_str(args.dtype)
    rng = random.Random(args.seed)

    # ---------------- Load indices once ----------------
    xm_image_index, xm_caption_index, _xm_meta = _load_xm_index(args.xm_cache_dir)
    # Resolve where raw XM3600 images live (needed only to render the anchor image in the plot).
    # Newer pipeline keeps images outside cache_dir; index/meta.json stores images_dir.
    if getattr(args, "xm_images_dir", None) is None:
        guess = os.path.join(os.path.dirname(args.xm_cache_dir), "unpackedImages")
        args.xm_images_dir = _xm_meta.get("images_dir") or (guess if os.path.isdir(guess) else None)
    cc_image_index, cc_caption_index, _cc_meta = _load_cc_index(args.cc_cache_dir)

    xm_n_images = len(xm_image_index)
    xm_caps_by_img = build_caption_lists_by_image(xm_caption_index, xm_n_images)

    # CC12M: 1 caption per image mapping
    # (works even if some images have missing captions)
    cap_id_by_img = build_cc12m_caption_id_by_image(cc_caption_index, len(cc_image_index))

    # ---------------- Expand anchors + languages ----------------
    if args.xm_image_idx is None or len(args.xm_image_idx) == 0:
        anchor_indices = [rng.randrange(xm_n_images)]
    else:
        anchor_indices = []
        for idx in args.xm_image_idx:
            if not (0 <= idx < xm_n_images):
                raise ValueError(f"--xm_image_idx must be in [0, {xm_n_images-1}], got {idx}")
            anchor_indices.append(idx)

    if args.anchor_language is None or len(args.anchor_language) == 0:
        lang_list: List[Optional[str]] = [None]  # ANY
    else:
        lang_list = list(args.anchor_language)

    # ---------------- Model combos ----------------
    text_models = [normalize_model_input(x, "text") for x in args.text_model]
    image_models = [normalize_model_input(x, "image") for x in args.image_model]

    combos = [(t, i) for t in text_models for i in image_models]
    print(f"[INFO] Will generate plots for {len(combos)} model combos (text × image)")

    for text_model_in, image_model_in in combos:
        try:
            xm_txt_tag, xm_txt_repo, xm_txt_meta = _find_cached_model_tag(
                args.xm_cache_dir, "text", text_model_in, TEXT_MODEL_PRESETS
            )
            xm_img_tag, xm_img_repo, xm_img_meta = _find_cached_model_tag(
                args.xm_cache_dir, "image", image_model_in, IMAGE_MODEL_PRESETS
            )
            cc_txt_tag, cc_txt_repo, cc_txt_meta = _find_cached_model_tag(
                args.cc_cache_dir, "text", text_model_in, TEXT_MODEL_PRESETS
            )
            cc_img_tag, cc_img_repo, cc_img_meta = _find_cached_model_tag(
                args.cc_cache_dir, "image", image_model_in, IMAGE_MODEL_PRESETS
            )
        except FileNotFoundError as e:
            if args.skip_missing:
                print(f"[WARN] {e}")
                print("[WARN] Skipping this (text_model,image_model) combo.")
                continue
            raise

        # Sanity: embedding dims must match within modality
        # (We don't require matching across datasets here, but they SHOULD if same model.)
        print(
            f"[INFO] Using text_model='{text_model_in}' -> XM tag={xm_txt_tag}, CC tag={cc_txt_tag} | repo={xm_txt_repo or cc_txt_repo}"
        )
        print(
            f"[INFO] Using image_model='{image_model_in}' -> XM tag={xm_img_tag}, CC tag={cc_img_tag} | repo={xm_img_repo or cc_img_repo}"
        )

        # ---------------- Load embeddings to CPU ----------------
        # XM3600
        xm_image_embs_cpu, _, _ = _load_cached_embeddings(
            args.xm_cache_dir, "image", xm_img_tag, prefix_candidates=["image_embs"], device="cpu"
        )
        xm_caption_embs_cpu, _, _ = _load_cached_embeddings(
            args.xm_cache_dir, "text", xm_txt_tag, prefix_candidates=["caption_embs", "text_embs"], device="cpu"
        )

        # CC12M
        cc_image_embs_cpu, _, _ = _load_cached_embeddings(
            args.cc_cache_dir, "image", cc_img_tag, prefix_candidates=["image_embs"], device="cpu"
        )
        cc_text_embs_cpu, _, _ = _load_cached_embeddings(
            args.cc_cache_dir, "text", cc_txt_tag, prefix_candidates=["text_embs", "caption_embs"], device="cpu"
        )

        # ---------------- Basic shape checks ----------------
        if xm_image_embs_cpu.shape[0] != len(xm_image_index):
            raise ValueError("XM3600 image_index length != image_embs rows")
        if xm_caption_embs_cpu.shape[0] != len(xm_caption_index):
            raise ValueError("XM3600 caption_index length != caption_embs rows")

        if cc_image_embs_cpu.shape[0] != len(cc_image_index):
            raise ValueError("CC12M image_index length != image_embs rows")
        if cc_text_embs_cpu.shape[0] != len(cc_caption_index):
            raise ValueError("CC12M caption_index length != text_embs rows")

        # ---------------- Move embeddings to device ONCE (per model combo) ----------------
        print(f"[INFO] Moving embeddings to {device} as dtype={dtype} (one-time cost for this combo)")

        xm_image_embs = xm_image_embs_cpu.to(device=device, dtype=dtype)
        xm_caption_embs = xm_caption_embs_cpu.to(device=device, dtype=dtype)

        cc_image_embs = cc_image_embs_cpu.to(device=device, dtype=dtype)
        cc_text_embs = cc_text_embs_cpu.to(device=device, dtype=dtype)

        # Normalize once for cosine
        xm_image_embs = F.normalize(xm_image_embs, p=2, dim=1)
        xm_caption_embs = F.normalize(xm_caption_embs, p=2, dim=1)
        cc_image_embs = F.normalize(cc_image_embs, p=2, dim=1)
        cc_text_embs = F.normalize(cc_text_embs, p=2, dim=1)

        # Model prefix that shows up in file names
        model_prefix = f"txt={text_model_in}__img={image_model_in}"

        total_runs = len(anchor_indices) * len(lang_list)
        print(f"[INFO] Will generate {total_runs} plots for this combo = {len(anchor_indices)} anchors × {len(lang_list)} languages")

        for aidx in anchor_indices:
            for lang in lang_list:
                run_single(
                    anchor_xm_img_idx=aidx,
                    preferred_lang=lang,
                    xm_image_index=xm_image_index,
                    xm_caption_index=xm_caption_index,
                    xm_caps_by_img=xm_caps_by_img,
                    xm_image_embs=xm_image_embs,
                    xm_caption_embs=xm_caption_embs,
                    cc_image_index=cc_image_index,
                    cc_caption_index=cc_caption_index,
                    cap_id_by_img=cap_id_by_img,
                    cc_image_embs=cc_image_embs,
                    cc_text_embs=cc_text_embs,
                    args=args,
                    model_prefix=model_prefix,
                )


if __name__ == "__main__":
    main(parse_args())