#!/usr/bin/env python3
"""XM3600 multilingual caption → image retrieval using AS-IF Relative Representations.

This evaluates multilingual caption→image retrieval on XM3600 using CC12M as the
reference/anchor set for relative representations (RR).

RR(y) is the sparse vector of top-k cosine similarities between y and the anchor
set (one coordinate per anchor).

Scoring for retrieval is a dot product in RR space:
  score(img | caption) = < RR_text(caption), RR_img(img) >

We support 3 processing modes for RR values:
  - none: keep raw top-k cosine similarities
  - norm: L2-normalize the sparse vector (no exponentiation)
  - asif: exponentiate similarities by p and L2-normalize (paper-style)

This script supports the *new cache layout* used by the updated dataset builders:

cache_dir/
  index/
    images.jsonl
    captions.jsonl
    meta.json
  embeddings/
    text/<model_tag>/
      meta.json
      caption_embs_0000.pt ... (or text_embs_0000.pt)
      DONE
    image/<model_tag>/
      meta.json
      image_embs_0000.pt ...
      DONE

It also remains backward-compatible with older flat caches that had:
  - image_embs.pt / caption_embs.pt
  - image_index.json / caption_index.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


# -----------------------------
# Model presets (keys = model tags used in cache_dir/embeddings/*/<model_tag>/)
# -----------------------------

TEXT_MODEL_PRESETS: Dict[str, str] = {
    "nllb-200-3.3B": "facebook/nllb-200-3.3B",
    "bge-m3": "BAAI/bge-m3",
    "qwen3-embed-8b": "Qwen/Qwen3-Embedding-8B",
}

IMAGE_MODEL_PRESETS: Dict[str, str] = {
    "dinov3-vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "clip-vit-bigg14": "CLIPImageEncoder(largest/latest)",   # cache tag; repo may be resolved elsewhere
    "c-radio-v4-h": "nvidia/C-RADIOv4-H",
}


# -----------------------------
# Utilities
# -----------------------------

@torch.no_grad()
def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise L2 normalization."""
    denom = torch.linalg.norm(x.float(), dim=1, keepdim=True).clamp_min(eps)
    return (x.float() / denom).to(dtype=x.dtype)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _dtype_from_str(s: str) -> torch.dtype:
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32
    raise ValueError(s)


def _is_jsonl(path: str) -> bool:
    return os.path.isfile(path) and path.endswith(".jsonl")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _glob_sorted(dir_path: str, patterns: Sequence[str]) -> List[str]:
    """Return lexicographically sorted filepaths for any of the patterns."""
    p = Path(dir_path)
    out: List[str] = []
    for pat in patterns:
        out.extend([str(x) for x in p.glob(pat)])
    # De-dup + stable sort
    out = sorted(set(out))
    return out


def _resolve_model_tag(user_value: str, presets: Dict[str, str]) -> Tuple[str, str]:
    """Return (model_tag, model_repo_or_label)."""
    if user_value in presets:
        return user_value, presets[user_value]
    # If user passes a HF repo directly, the cache tag is ambiguous. We assume they used
    # the repo name sanitized as a folder; fall back to replacing '/' with '__'.
    tag = user_value.replace("/", "__").replace(":", "_")
    return tag, user_value


# -----------------------------
# Cache loading (new + legacy)
# -----------------------------

def _find_embedding_files(
    cache_dir: str,
    modality: str,        # "text" or "image"
    model_tag: str,
    shard_patterns: Sequence[str],
    legacy_single_files: Sequence[str],
) -> List[str]:
    """Find embedding tensor shards for a given (cache_dir, modality, model_tag).

    Prefers the new layout under cache_dir/embeddings/<modality>/<model_tag>/.
    Falls back to legacy single files in cache_dir/ if present.
    """
    # New layout
    base = os.path.join(cache_dir, "embeddings", modality, model_tag)
    if os.path.isdir(base):
        files = _glob_sorted(base, shard_patterns)
        if files:
            return files
        # allow single-file inside the model directory too
        for lf in legacy_single_files:
            p = os.path.join(base, lf)
            if os.path.isfile(p):
                return [p]

    # Legacy layout: single files at root
    for lf in legacy_single_files:
        p = os.path.join(cache_dir, lf)
        if os.path.isfile(p):
            return [p]

    return []


def _load_concat_shards(paths: Sequence[str], device: torch.device, dtype: torch.dtype, normalize: bool) -> torch.Tensor:
    if not paths:
        raise RuntimeError("No embedding shard paths provided.")
    parts: List[torch.Tensor] = []
    for p in paths:
        t = torch.load(p, map_location="cpu")
        if t.ndim != 2:
            raise ValueError(f"Expected 2D tensor in {p}, got {tuple(t.shape)}")
        t = t.to(dtype=dtype).to(device, non_blocking=True)
        if normalize:
            t = l2_normalize(t)
        parts.append(t)
    return torch.cat(parts, dim=0)


def _load_first_n_from_tensor_file(
    path: str,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
    normalize: bool,
    desc: str,
) -> torch.Tensor:
    """Load first n rows from a single .pt tensor file."""
    t = torch.load(path, map_location="cpu")
    if t.ndim != 2:
        raise ValueError(f"{desc}: expected 2D tensor, got {tuple(t.shape)} in {path}")
    take = min(n, t.shape[0])
    t = t[:take].to(dtype=dtype).to(device, non_blocking=True)
    if normalize:
        t = l2_normalize(t)
    if take < n:
        print(f"[WARN] Requested {n} anchors for {desc}, but only loaded {take}.")
    return t


@torch.no_grad()
def _load_first_n_from_shards(
    shard_paths: Sequence[str],
    n: int,
    device: torch.device,
    dtype: torch.dtype,
    normalize: bool,
    desc: str,
) -> torch.Tensor:
    """Load the first n rows across multiple shard .pt files (in order)."""
    remaining = n
    parts: List[torch.Tensor] = []

    for p in tqdm(shard_paths, desc=desc):
        if remaining <= 0:
            break
        t = torch.load(p, map_location="cpu")
        if t.ndim != 2:
            raise ValueError(f"Shard {p} expected 2D tensor, got {tuple(t.shape)}")

        take = min(remaining, t.shape[0])
        t = t[:take]
        t = t.to(dtype=dtype)
        t = t.to(device, non_blocking=True)
        if normalize:
            t = l2_normalize(t)
        parts.append(t)
        remaining -= take

    if not parts:
        raise RuntimeError(f"No anchors loaded for {desc}. Check shard paths.")

    out = torch.cat(parts, dim=0)
    if out.shape[0] < n:
        print(f"[WARN] Requested {n} anchors for {desc}, but only loaded {out.shape[0]}.")
    return out


def load_xm3600_cache(
    xm_cache_dir: str,
    text_model_tag: str,
    image_model_tag: str,
    device: torch.device,
    dtype: torch.dtype,
    normalize: bool = True,
) -> Dict[str, Any]:
    """Load XM3600 embeddings + caption index from new or legacy cache layouts."""
    # New layout detection
    idx_dir = os.path.join(xm_cache_dir, "index")
    images_jsonl = os.path.join(idx_dir, "images.jsonl")
    captions_jsonl = os.path.join(idx_dir, "captions.jsonl")
    meta_json = os.path.join(idx_dir, "meta.json")

    if _is_jsonl(images_jsonl) and _is_jsonl(captions_jsonl):
        image_index = _read_jsonl(images_jsonl)
        caption_index_raw = _read_jsonl(captions_jsonl)
        meta = {}
        if os.path.isfile(meta_json):
            with open(meta_json, "r", encoding="utf-8") as f:
                meta = json.load(f)

        # Map image_id -> image_idx (if needed)
        id_to_idx: Dict[str, int] = {}
        for i, rec in enumerate(image_index):
            iid = rec.get("image_id") or rec.get("id") or rec.get("key")
            if iid is not None:
                id_to_idx[str(iid)] = i

        caption_index: List[Dict[str, Any]] = []
        for ci, rec in enumerate(caption_index_raw):
            out = dict(rec)
            # normalize expected keys used by select_caption_indices()
            if "language" not in out and "lang" in out:
                out["language"] = out["lang"]

            if "caption" not in out:
                if "text" in out:
                    out["caption"] = out["text"]

            if "image_idx" not in out:
                iid = out.get("image_id") or out.get("id") or out.get("key")
                if iid is None:
                    raise KeyError(f"[XM] captions.jsonl row {ci} missing image_idx and image_id/id/key")
                iid = str(iid)
                if iid not in id_to_idx:
                    raise KeyError(f"[XM] captions.jsonl row {ci} has image_id={iid} not found in images.jsonl")
                out["image_idx"] = int(id_to_idx[iid])

            caption_index.append(out)

        # Embeddings
        img_paths = _find_embedding_files(
            xm_cache_dir,
            modality="image",
            model_tag=image_model_tag,
            shard_patterns=("image_embs_*.pt", "image_embs.pt"),
            legacy_single_files=("image_embs.pt",),
        )
        txt_paths = _find_embedding_files(
            xm_cache_dir,
            modality="text",
            model_tag=text_model_tag,
            shard_patterns=("caption_embs_*.pt", "text_embs_*.pt", "caption_embs.pt", "text_embs.pt"),
            legacy_single_files=("caption_embs.pt",),
        )
        if not img_paths:
            raise FileNotFoundError(
                f"[XM] Could not find image embeddings for tag='{image_model_tag}' under {xm_cache_dir}/embeddings/image/"
            )
        if not txt_paths:
            raise FileNotFoundError(
                f"[XM] Could not find text embeddings for tag='{text_model_tag}' under {xm_cache_dir}/embeddings/text/"
            )

        image_embs = _load_concat_shards(img_paths, device=device, dtype=dtype, normalize=normalize)
        caption_embs = _load_concat_shards(txt_paths, device=device, dtype=dtype, normalize=normalize)

        return {
            "image_index": image_index,
            "caption_index": caption_index,
            "image_embs": image_embs,
            "caption_embs": caption_embs,
            "meta": meta,
        }

    # Legacy layout
    image_index_path = os.path.join(xm_cache_dir, "image_index.json")
    caption_index_path = os.path.join(xm_cache_dir, "caption_index.json")
    image_embs_path = os.path.join(xm_cache_dir, "image_embs.pt")
    caption_embs_path = os.path.join(xm_cache_dir, "caption_embs.pt")
    meta_path = os.path.join(xm_cache_dir, "meta.json")

    if not (os.path.isfile(image_index_path) and os.path.isfile(caption_index_path) and os.path.isfile(image_embs_path) and os.path.isfile(caption_embs_path)):
        raise FileNotFoundError(
            f"[XM] Could not detect a supported cache layout under {xm_cache_dir}. "
            "Expected either index/*.jsonl + embeddings/*/<tag>/... or legacy *_index.json + *_embs.pt."
        )

    with open(image_index_path, "r", encoding="utf-8") as f:
        image_index = json.load(f)
    with open(caption_index_path, "r", encoding="utf-8") as f:
        caption_index = json.load(f)

    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    image_embs = torch.load(image_embs_path, map_location="cpu").to(dtype=dtype).to(device, non_blocking=True)
    caption_embs = torch.load(caption_embs_path, map_location="cpu").to(dtype=dtype).to(device, non_blocking=True)
    if normalize:
        image_embs = l2_normalize(image_embs)
        caption_embs = l2_normalize(caption_embs)

    return {
        "image_index": image_index,
        "caption_index": caption_index,
        "image_embs": image_embs,
        "caption_embs": caption_embs,
        "meta": meta,
    }


# -----------------------------
# Anchor loading (CC12M)
# -----------------------------

@dataclass
class AnchorSet:
    img_anchors: torch.Tensor  # [N, D_img]
    txt_anchors: torch.Tensor  # [N, D_txt]
    n: int


@torch.no_grad()
def load_cc12m_anchors(
    cc_cache_dir: str,
    anchor_n: int,
    text_model_tag: str,
    image_model_tag: str,
    device: torch.device,
    dtype: torch.dtype,
    normalize: bool = True,
) -> AnchorSet:
    """Load first anchor_n CC12M image+text embeddings (aligned).

    Supports:
      - New layout: cache_dir/embeddings/{image|text}/<tag>/*.pt
      - Legacy flat: cache_dir/image_embs.pt and cache_dir/caption_embs.pt
    """
    # Prefer new layout
    img_paths = _find_embedding_files(
        cc_cache_dir,
        modality="image",
        model_tag=image_model_tag,
        shard_patterns=("image_embs_*.pt", "image_embs.pt"),
        legacy_single_files=("image_embs.pt",),
    )
    txt_paths = _find_embedding_files(
        cc_cache_dir,
        modality="text",
        model_tag=text_model_tag,
        shard_patterns=("caption_embs_*.pt", "text_embs_*.pt", "caption_embs.pt", "text_embs.pt"),
        legacy_single_files=("caption_embs.pt",),
    )

    if not img_paths or not txt_paths:
        # Backward-compat: some older CC caches used root sharded paths referenced by helper loaders.
        # Here we attempt to find shards in cache_dir directly.
        if not img_paths:
            img_paths = _glob_sorted(cc_cache_dir, ("image_embs_*.pt", "image_embs.pt"))
        if not txt_paths:
            txt_paths = _glob_sorted(cc_cache_dir, ("caption_embs_*.pt", "text_embs_*.pt", "caption_embs.pt", "text_embs.pt"))

    if not img_paths or not txt_paths:
        raise FileNotFoundError(
            f"[CC12M] Could not find embeddings for tags text='{text_model_tag}', image='{image_model_tag}' under {cc_cache_dir}."
        )

    # If single file, use the optimized path; else read shards in order.
    if len(img_paths) == 1 and len(txt_paths) == 1 and (img_paths[0].endswith(".pt") and txt_paths[0].endswith(".pt")):
        img_anchors = _load_first_n_from_tensor_file(
            img_paths[0],
            anchor_n,
            device=device,
            dtype=dtype,
            normalize=normalize,
            desc=f"[CC12M] Loading image anchors (first {anchor_n})",
        )
        txt_anchors = _load_first_n_from_tensor_file(
            txt_paths[0],
            anchor_n,
            device=device,
            dtype=dtype,
            normalize=normalize,
            desc=f"[CC12M] Loading text anchors (first {anchor_n})",
        )
    else:
        img_anchors = _load_first_n_from_shards(
            img_paths,
            anchor_n,
            device=device,
            dtype=dtype,
            normalize=normalize,
            desc=f"[CC12M] Loading image anchors (first {anchor_n})",
        )
        txt_anchors = _load_first_n_from_shards(
            txt_paths,
            anchor_n,
            device=device,
            dtype=dtype,
            normalize=normalize,
            desc=f"[CC12M] Loading text anchors (first {anchor_n})",
        )

    n = min(int(img_anchors.shape[0]), int(txt_anchors.shape[0]))
    img_anchors = img_anchors[:n]
    txt_anchors = txt_anchors[:n]

    if img_anchors.shape[0] != txt_anchors.shape[0]:
        raise RuntimeError(f"Anchor alignment mismatch: img={img_anchors.shape}, txt={txt_anchors.shape}")

    print(f"[{_now()}] Loaded CC12M anchors: N={n}, img_dim={img_anchors.shape[1]}, txt_dim={txt_anchors.shape[1]}")
    return AnchorSet(img_anchors=img_anchors, txt_anchors=txt_anchors, n=n)


# -----------------------------
# RR computation
# -----------------------------

@torch.no_grad()
def compute_rr_topk(
    queries: torch.Tensor,           # [B, D]
    anchors: torch.Tensor,           # [N, D]
    k: int,
    anchor_chunk: int,
    desc: str = "RR",
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute top-k cosine similarities between queries and anchors by chunking anchors.

    Returns:
      idxs: [B, k] int64 (anchor indices)
      vals: [B, k] float32
    """
    if anchor_chunk < k:
        raise ValueError(f"anchor_chunk ({anchor_chunk}) must be >= k ({k})")

    device = queries.device
    B, D = queries.shape
    N, D2 = anchors.shape
    if D != D2:
        raise ValueError(f"Dim mismatch: queries {D}, anchors {D2}")

    cur_vals = torch.full((B, k), -1e9, device=device, dtype=torch.float32)
    cur_idxs = torch.full((B, k), -1, device=device, dtype=torch.int64)

    it: Iterable[int] = range(0, N, anchor_chunk)
    if show_progress:
        it = tqdm(it, desc=desc)

    for start in it:
        end = min(N, start + anchor_chunk)
        a = anchors[start:end]  # [C, D]

        sims = (queries @ a.transpose(0, 1)).to(torch.float32)  # [B, C]
        kk = min(k, sims.shape[1])
        v, i = torch.topk(sims, kk, dim=1)
        i = i.to(torch.int64) + start

        merged_vals = torch.cat([cur_vals, v], dim=1)
        merged_idxs = torch.cat([cur_idxs, i], dim=1)
        new_vals, pos = torch.topk(merged_vals, k, dim=1)
        new_idxs = torch.gather(merged_idxs, 1, pos)

        cur_vals = new_vals
        cur_idxs = new_idxs

        del sims, v, i, merged_vals, merged_idxs, new_vals, new_idxs, pos

    return cur_idxs, cur_vals


@torch.no_grad()
def process_rr_values(
    vals: torch.Tensor,  # [B, k], float32
    proc: str,
    p: int,
    clamp_min: Optional[float] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Apply one of: none, norm, asif."""
    if clamp_min is not None:
        vals = vals.clamp_min(clamp_min)

    if proc == "none":
        return vals

    if proc == "asif":
        vals = vals ** p

    denom = torch.linalg.norm(vals, dim=1, keepdim=True).clamp_min(eps)
    return vals / denom


# -----------------------------
# Scoring (vectorized inverted index)
# -----------------------------

@dataclass
class ImagePostingIndex:
    """Inverted index over image RRs.

    For each anchor coordinate j, postings are stored in a single flat array:
      img_ids_sorted[indptr[j]:indptr[j+1]]
      img_vals_sorted[indptr[j]:indptr[j+1]]
    """
    indptr: torch.Tensor
    img_ids_sorted: torch.Tensor
    img_vals_sorted: torch.Tensor
    n_images: int
    n_anchors: int


@torch.no_grad()
def build_image_posting_index(
    idxs_img: torch.Tensor,   # [I, k]
    vals_img: torch.Tensor,   # [I, k]
    n_anchors: int,
    device: torch.device,
) -> ImagePostingIndex:
    I, k = idxs_img.shape
    flat_anchor = idxs_img.reshape(-1).to(device=device, dtype=torch.int64)
    flat_vals = vals_img.reshape(-1).to(device=device, dtype=torch.float32)
    flat_img_ids = torch.arange(I, device=device, dtype=torch.int64).repeat_interleave(k)

    perm = torch.argsort(flat_anchor)
    flat_anchor_sorted = flat_anchor[perm]
    img_ids_sorted = flat_img_ids[perm]
    img_vals_sorted = flat_vals[perm]

    counts = torch.bincount(flat_anchor_sorted, minlength=n_anchors)
    indptr = torch.zeros(n_anchors + 1, device=device, dtype=torch.int64)
    indptr[1:] = torch.cumsum(counts, dim=0)

    return ImagePostingIndex(
        indptr=indptr,
        img_ids_sorted=img_ids_sorted,
        img_vals_sorted=img_vals_sorted,
        n_images=I,
        n_anchors=n_anchors,
    )


@torch.no_grad()
def score_caption_batch(
    idxs_q: torch.Tensor,     # [B, k]
    vals_q: torch.Tensor,     # [B, k]
    postings: ImagePostingIndex,
) -> torch.Tensor:
    """Compute dense scores [B, n_images] using vectorized join via postings."""
    device = vals_q.device
    B, k = idxs_q.shape
    n_images = postings.n_images

    q_cols = idxs_q.reshape(-1).to(torch.int64)
    q_vals = vals_q.reshape(-1).to(torch.float32)
    q_rows = torch.arange(B, device=device, dtype=torch.int64).repeat_interleave(k)

    indptr = postings.indptr
    starts = indptr[q_cols]
    ends = indptr[q_cols + 1]
    counts = (ends - starts).to(torch.int64)

    total = int(counts.sum().item())
    if total == 0:
        return torch.zeros((B, n_images), device=device, dtype=torch.float32)

    rep_starts = torch.repeat_interleave(starts, counts)
    rep_qvals = torch.repeat_interleave(q_vals, counts)
    rep_qrows = torch.repeat_interleave(q_rows, counts)

    prefix = torch.cumsum(counts, dim=0) - counts
    rep_prefix = torch.repeat_interleave(prefix, counts)
    intra = torch.arange(total, device=device, dtype=torch.int64) - rep_prefix
    pos = rep_starts + intra

    img_ids = postings.img_ids_sorted[pos].to(torch.int64)
    img_vals = postings.img_vals_sorted[pos].to(torch.float32)

    contrib = rep_qvals * img_vals

    scores_flat = torch.zeros(B * n_images, device=device, dtype=torch.float32)
    flat_idx = rep_qrows * n_images + img_ids
    scores_flat.scatter_add_(0, flat_idx, contrib)

    return scores_flat.view(B, n_images)


# -----------------------------
# Evaluation
# -----------------------------

@dataclass
class RecallCounters:
    Ks: List[int]
    total: int
    hits: List[int]
    total_by_lang: torch.Tensor  # [L]
    hits_by_lang: torch.Tensor   # [len(Ks), L]
    lang_id_to_code: List[str]


def init_counters(Ks: List[int], lang_codes: List[str]) -> RecallCounters:
    Ks = sorted(Ks)
    L = len(lang_codes)
    return RecallCounters(
        Ks=Ks,
        total=0,
        hits=[0 for _ in Ks],
        total_by_lang=torch.zeros(L, dtype=torch.long),
        hits_by_lang=torch.zeros((len(Ks), L), dtype=torch.long),
        lang_id_to_code=lang_codes,
    )


@torch.no_grad()
def update_counters(
    counters: RecallCounters,
    top_pred: torch.Tensor,       # [B, maxK]
    gt_img: torch.Tensor,         # [B]
    lang_ids: torch.Tensor,       # [B]
) -> None:
    B, maxK = top_pred.shape
    match = (top_pred == gt_img.unsqueeze(1))  # [B, maxK]

    counters.total += int(B)

    for ki, K in enumerate(counters.Ks):
        hit_k = match[:, :K].any(dim=1)
        counters.hits[ki] += int(hit_k.sum().item())

        hit_w = hit_k.float()
        counters.hits_by_lang[ki] += torch.bincount(
            lang_ids.to(torch.int64).cpu(),
            weights=hit_w.cpu(),
            minlength=counters.total_by_lang.numel(),
        ).to(torch.long)

    counters.total_by_lang += torch.bincount(
        lang_ids.to(torch.int64).cpu(),
        minlength=counters.total_by_lang.numel(),
    ).to(torch.long)


def counters_to_results(counters: RecallCounters) -> Dict[str, object]:
    overall = {f"R@{K}": (counters.hits[i] / max(1, counters.total)) for i, K in enumerate(counters.Ks)}

    per_lang: Dict[str, Dict[str, float]] = {}
    for li, code in enumerate(counters.lang_id_to_code):
        denom = int(counters.total_by_lang[li].item())
        if denom == 0:
            continue
        per_lang[code] = {
            f"R@{K}": (float(counters.hits_by_lang[i, li].item()) / denom)
            for i, K in enumerate(counters.Ks)
        }

    return {
        "overall": overall,
        "per_language": per_lang,
        "n_queries": counters.total,
    }


# -----------------------------
# Query selection
# -----------------------------

def select_caption_indices(
    caption_index: List[Dict[str, object]],
    languages: Optional[Sequence[str]],
    one_per_image_per_lang: bool,
    max_queries: Optional[int],
) -> Tuple[List[int], List[int], List[str]]:
    """Return (selected_caption_rows, gt_image_idx, lang_code)."""
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


# -----------------------------
# Caching helpers
# -----------------------------

def _config_path(out_dir: str) -> str:
    return os.path.join(out_dir, "run_config.json")


def load_cached_config(out_dir: str) -> Optional[Dict[str, object]]:
    p = _config_path(out_dir)
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(out_dir: str, cfg: Dict[str, object]) -> None:
    with open(_config_path(out_dir), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="XM3600 caption→image retrieval using CC12M-based relative representations"
    )
    ap.add_argument("--xm_cache", required=True, help="XM3600 cache dir")
    ap.add_argument("--cc_cache", required=True, help="CC12M cache dir")
    ap.add_argument("--out_dir", required=True, help="Output directory (caches + results.json)")

    ap.add_argument(
        "--text_model",
        default="nllb-200-3.3B",
        nargs=1,
        help="Text model tag (cache folder name). Presets: " + ", ".join(TEXT_MODEL_PRESETS.keys()),
    )
    ap.add_argument(
        "--image_model",
        default="dinov3-vit7b16",
        nargs=1,
        help="Image model tag (cache folder name). Presets: " + ", ".join(IMAGE_MODEL_PRESETS.keys()),
    )

    ap.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0 or cpu")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="Compute dtype for embeddings")

    ap.add_argument("--anchor_n", type=int, default=1_600_000, help="Number of CC12M anchors (first N)")
    ap.add_argument("--k", type=int, default=800, help="Non-zeros per RR (top-k)")
    ap.add_argument("--anchor_chunk", type=int, default=100_000, help="Chunk size over anchors when computing RR")

    ap.add_argument("--proc", choices=["none", "norm", "asif"], default="asif", help="RR value processing")
    ap.add_argument("--p", type=int, default=8, help="Exponent p for proc=asif")
    ap.add_argument("--clamp_min", type=float, default=None, help="Optional clamp min for similarities before processing")

    ap.add_argument("--image_batch", type=int, default=256, help="Batch size to compute RR for XM3600 images")
    ap.add_argument("--query_batch", type=int, default=2048, help="Caption query batch size")

    ap.add_argument(
        "--inner_progress",
        action="store_true",
        help=(
            "Show progress bars for the *inner* loop over anchor chunks when computing RR. "
            "This is very verbose for captions (many batches)."
        ),
    )
    ap.add_argument(
        "--empty_cache_every",
        type=int,
        default=0,
        help=(
            "If >0 and running on CUDA, call torch.cuda.empty_cache() every N caption batches. "
            "Default 0 = never (recommended on big GPUs)."
        ),
    )

    ap.add_argument("--Ks", default="1,5,10,25,50", help="Recall@K list")
    ap.add_argument("--languages", default=None, help="Comma-separated language codes to evaluate (default: all)")
    ap.add_argument("--one_per_image_per_lang", action="store_true", help="Evaluate only first caption for each (image,lang)")
    ap.add_argument("--max_queries", type=int, default=None, help="Evaluate at most this many captions")

    ap.add_argument("--reuse_cache", action="store_true", help="Reuse cached image RR / posting index if config matches")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    # normalize nargs=1 lists to scalars (keeps compatibility if someone passes just one token)
    if isinstance(args.text_model, list):
        args.text_model = args.text_model[0]
    if isinstance(args.image_model, list):
        args.image_model = args.image_model[0]
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = _dtype_from_str(args.dtype)

    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    _ensure_dir(args.out_dir)

    Ks = [int(x) for x in args.Ks.split(",") if x.strip()]
    maxK = max(Ks)

    languages = None
    if args.languages:
        languages = [x.strip() for x in args.languages.split(",") if x.strip()]

    text_tag, text_repo = _resolve_model_tag(args.text_model, TEXT_MODEL_PRESETS)
    img_tag, img_repo = _resolve_model_tag(args.image_model, IMAGE_MODEL_PRESETS)

    run_cfg = {
        "xm_cache": os.path.abspath(args.xm_cache),
        "cc_cache": os.path.abspath(args.cc_cache),
        "text_model": args.text_model,
        "text_tag": text_tag,
        "text_repo": text_repo,
        "image_model": args.image_model,
        "image_tag": img_tag,
        "image_repo": img_repo,
        "anchor_n": int(args.anchor_n),
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

    cached_cfg = load_cached_config(args.out_dir) if args.reuse_cache else None
    cfg_matches = (cached_cfg == run_cfg)

    print(f"[{_now()}] device={device} dtype={dtype} proc={args.proc} k={args.k} anchor_n={args.anchor_n}")
    print(f"[{_now()}] text_model={args.text_model} (tag={text_tag}) | image_model={args.image_model} (tag={img_tag})")
    print(f"[{_now()}] out_dir={args.out_dir}")

    # 1) Load XM3600 absolute embeddings
    print(f"[{_now()}] Loading XM3600 cache from {args.xm_cache}")
    xm = load_xm3600_cache(
        args.xm_cache,
        text_model_tag=text_tag,
        image_model_tag=img_tag,
        device=device,
        dtype=dtype,
        normalize=True,
    )

    xm_image_embs = xm["image_embs"]        # [3600, D_img]
    xm_caption_embs = xm["caption_embs"]    # [N_caps, D_txt]
    xm_caption_index = xm["caption_index"]

    n_images = int(xm_image_embs.shape[0])

    # 2) Load CC12M anchors
    anchors = load_cc12m_anchors(
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

    # 3) Compute (or load) image RR and posting index
    rr_img_idxs_path = os.path.join(args.out_dir, "xm_image_rr_idxs.pt")
    rr_img_vals_path = os.path.join(args.out_dir, "xm_image_rr_vals.pt")
    post_indptr_path = os.path.join(args.out_dir, "post_indptr.pt")
    post_imgids_path = os.path.join(args.out_dir, "post_imgids_sorted.pt")
    post_vals_path = os.path.join(args.out_dir, "post_vals_sorted.pt")

    if args.reuse_cache and cfg_matches and all(
        os.path.isfile(p)
        for p in [rr_img_idxs_path, rr_img_vals_path, post_indptr_path, post_imgids_path, post_vals_path]
    ):
        print(f"[{_now()}] Reusing cached XM image RR + postings (config matches)")
        idxs_img = torch.load(rr_img_idxs_path, map_location=device)
        vals_img = torch.load(rr_img_vals_path, map_location=device)
        postings = ImagePostingIndex(
            indptr=torch.load(post_indptr_path, map_location=device),
            img_ids_sorted=torch.load(post_imgids_path, map_location=device),
            img_vals_sorted=torch.load(post_vals_path, map_location=device),
            n_images=n_images,
            n_anchors=n_anchors,
        )
    else:
        print(f"[{_now()}] Computing XM3600 image RR...")
        idxs_img_list: List[torch.Tensor] = []
        vals_img_list: List[torch.Tensor] = []

        for start in tqdm(range(0, n_images, args.image_batch), desc="[XM] Image RR"):
            end = min(n_images, start + args.image_batch)
            q = xm_image_embs[start:end]
            idxs, vals = compute_rr_topk(
                q,
                anchors.img_anchors,
                k=args.k,
                anchor_chunk=args.anchor_chunk,
                desc=f"[RR-img] anchors chunking ({start}:{end})",
                show_progress=args.inner_progress,
            )
            vals = process_rr_values(vals, proc=args.proc, p=args.p, clamp_min=args.clamp_min)
            idxs_img_list.append(idxs)
            vals_img_list.append(vals)

        idxs_img = torch.cat(idxs_img_list, dim=0)
        vals_img = torch.cat(vals_img_list, dim=0)

        torch.save(idxs_img.detach().cpu(), rr_img_idxs_path)
        torch.save(vals_img.detach().cpu(), rr_img_vals_path)
        print(f"[{_now()}] Saved image RR to {rr_img_idxs_path} / {rr_img_vals_path}")

        print(f"[{_now()}] Building postings...")
        postings = build_image_posting_index(
            idxs_img=idxs_img,
            vals_img=vals_img,
            n_anchors=n_anchors,
            device=device,
        )

        torch.save(postings.indptr.detach().cpu(), post_indptr_path)
        torch.save(postings.img_ids_sorted.detach().cpu(), post_imgids_path)
        torch.save(postings.img_vals_sorted.detach().cpu(), post_vals_path)
        print(f"[{_now()}] Saved postings to {post_indptr_path} / {post_imgids_path} / {post_vals_path}")

        save_config(args.out_dir, run_cfg)

    # 4) Select caption queries
    sel_rows, gt_img, lang_codes = select_caption_indices(
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

    print(f"[{_now()}] Selected {len(sel_rows)} caption queries across {len(unique_langs)} languages")

    counters = init_counters(Ks, unique_langs)

    postings = ImagePostingIndex(
        indptr=postings.indptr.to(device=device),
        img_ids_sorted=postings.img_ids_sorted.to(device=device),
        img_vals_sorted=postings.img_vals_sorted.to(device=device),
        n_images=postings.n_images,
        n_anchors=postings.n_anchors,
    )

    # 5) Evaluate in batches
    nQ = sel_rows_t.numel()
    print(f"[{_now()}] Starting retrieval evaluation (batches of {args.query_batch})")

    for bi, start in enumerate(tqdm(range(0, nQ, args.query_batch), desc="[EVAL] Caption batches")):
        end = min(nQ, start + args.query_batch)
        batch_sel = sel_rows_t[start:end]
        batch_gt = gt_img_t[start:end]
        batch_lang = lang_ids_t[start:end]

        abs_caps = xm_caption_embs.index_select(0, batch_sel)

        idxs_q, vals_q = compute_rr_topk(
            abs_caps,
            anchors.txt_anchors,
            k=args.k,
            anchor_chunk=args.anchor_chunk,
            desc="[RR-txt] anchors chunking",
            show_progress=args.inner_progress,
        )
        vals_q = process_rr_values(vals_q, proc=args.proc, p=args.p, clamp_min=args.clamp_min)

        scores = score_caption_batch(idxs_q, vals_q, postings)  # [B, 3600]
        _, top_idx = torch.topk(scores, k=maxK, dim=1)

        update_counters(counters, top_idx, batch_gt, batch_lang)

        del abs_caps, idxs_q, vals_q, scores, top_idx
        if args.empty_cache_every > 0 and device.type == "cuda":
            if (bi + 1) % args.empty_cache_every == 0:
                torch.cuda.empty_cache()

    # 6) Save results
    results = counters_to_results(counters)
    results.update({"run_config": run_cfg, "timestamp": _now()})

    results_path = os.path.join(args.out_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[{_now()}] Done. Wrote {results_path}")
    print("Overall:")
    for K in Ks:
        print(f"  R@{K}: {results['overall'][f'R@{K}']:.4f}")


if __name__ == "__main__":
    main()