#!/usr/bin/env python3
"""XM3600: index + embeddings cache (text/image decoupled).

Requested upgrades:
- Default save dtype is float32 (no fp16 truncation unless requested).
- Store text and image embeddings in separate folders with independent loaders.
- Support choosing which model is used to save embeddings.
- Skip building when a cache for (modality, model_tag) already exists (DONE marker).

Cache layout under --cache_dir:

  cache_dir/
    index/
      images.jsonl
      captions.jsonl
      meta.json
    embeddings/
      image/<model_tag>/
        meta.json
        image_embs_0000.pt ...
        DONE
      text/<model_tag>/
        meta.json
        caption_embs_0000.pt ...
        DONE

Invariants:
- Image embeddings are aligned to index/images.jsonl by image_idx (line order).
- Caption embeddings are aligned to index/captions.jsonl by caption_id (line order).
- Can be consolidated back in memory by joining on image_idx/caption_id.

Models (presets):
Text:
  - nllb-200-3.3B  -> facebook/nllb-200-3.3B
  - bge-m3         -> BAAI/bge-m3
  - qwen3-embed-8b -> Qwen/Qwen3-Embedding-8B
Image:
  - dinov3-vit7b16 -> facebook/dinov3-vit7b16-pretrain-lvd1689m
  - clip-vit-bigg14 -> laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
  - c-radio-v4-h   -> nvidia/C-RADIOv4-H

Notes:
- RADIO/Qwen embedding models may require trust_remote_code=True.
- DINOv3 requires recent transformers.
"""

from __future__ import annotations

import os
import re
import json
import time
import shutil
import hashlib
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Iterable

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

import numpy as np
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoImageProcessor,
)

# Optional imports for CLIP
try:
    from transformers import CLIPModel, CLIPProcessor
except Exception:
    CLIPModel = None
    CLIPProcessor = None



# ----------------------------
# Workarounds
# ----------------------------

def _install_httpx_compat_shim() -> None:
    """Compatibility shim for httpx kwarg API changes.

    The HF stack (transformers / huggingface_hub) has used httpx over time and, depending
    on the installed versions, may call httpx methods with kwargs that were renamed/removed.

    Common breakages seen in the wild:
      - allow_redirects -> follow_redirects
      - proxies -> proxy   (best-effort; dict-style proxies are dropped)

    Additionally, this shim drops any unexpected kwargs for the installed httpx method
    signature (when the method does not accept **kwargs), preventing hard crashes.
    """
    try:
        import inspect
        import httpx
    except Exception:
        return

    # Don't patch twice
    if getattr(httpx, "_npa_httpx_compat_shim_installed", False):
        return

    def _wrap_method(orig):
        try:
            sig = inspect.signature(orig)
            params = sig.parameters
        except Exception:
            # If we can't introspect, just return original
            return orig

        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        allowed = set(params.keys())
        supports_follow = "follow_redirects" in allowed
        supports_proxy = "proxy" in allowed

        def wrapped(self, *args, **kwargs):
            # Map allow_redirects -> follow_redirects (or drop if unsupported)
            if "allow_redirects" in kwargs and "follow_redirects" not in kwargs:
                if supports_follow:
                    kwargs["follow_redirects"] = kwargs.pop("allow_redirects")
                else:
                    kwargs.pop("allow_redirects", None)

            # Map proxies -> proxy (best effort). If proxies is a dict, drop it.
            if "proxies" in kwargs and "proxy" not in kwargs:
                pv = kwargs.pop("proxies")
                if supports_proxy and not isinstance(pv, dict):
                    kwargs["proxy"] = pv
                # else: drop silently

            # Drop any other unexpected kwargs if the method doesn't accept **kwargs.
            if not has_varkw:
                for k in list(kwargs.keys()):
                    if k not in allowed:
                        kwargs.pop(k, None)

            return orig(self, *args, **kwargs)

        return wrapped

    def _patch_client(cls) -> None:
        for name in ("request", "head", "get", "post", "put", "delete", "patch", "options"):
            if not hasattr(cls, name):
                continue
            try:
                orig = getattr(cls, name)
                setattr(cls, name, _wrap_method(orig))
            except Exception:
                continue

    try:
        _patch_client(httpx.Client)
    except Exception:
        pass
    try:
        _patch_client(httpx.AsyncClient)
    except Exception:
        pass

    httpx._npa_httpx_compat_shim_installed = True


_install_httpx_compat_shim()

def _install_hf_additional_chat_templates_404_workaround() -> None:
    """Ignore optional `additional_chat_templates/` 404s during tokenizer loading.

    Some Transformers + huggingface_hub version combinations probe the Hub for an
    optional directory named `additional_chat_templates/` when loading tokenizers.
    Many repos (e.g., facebook/nllb-200-3.3B) don't have that directory, and the Hub
    correctly returns 404. In affected versions, that 404 can be treated as fatal.

    This monkey-patches the reference used by `transformers.tokenization_utils_base`
    to treat that specific 404 as "no templates" and continue.
    """
    try:
        import transformers.tokenization_utils_base as tub
        from transformers.utils.hub import list_repo_templates as _orig_list_repo_templates
    except Exception:
        return

    if getattr(tub, "_ignore_additional_chat_templates_404", False):
        return

    try:
        from huggingface_hub.errors import RemoteEntryNotFoundError  # type: ignore
    except Exception:  # pragma: no cover
        RemoteEntryNotFoundError = None  # type: ignore

    def _safe_list_repo_templates(*args, **kwargs):
        try:
            yield from _orig_list_repo_templates(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            if "additional_chat_templates" in msg:
                # Only ignore the "missing optional dir" failure mode
                if RemoteEntryNotFoundError is not None and isinstance(e, RemoteEntryNotFoundError):
                    return
                if ("404" in msg) or ("Entry Not Found" in msg) or ("does not exist" in msg):
                    return
            raise

    # Patch the reference used inside tokenization_utils_base.from_pretrained(...)
    tub.list_repo_templates = _safe_list_repo_templates  # type: ignore[attr-defined]
    tub._ignore_additional_chat_templates_404 = True


_install_hf_additional_chat_templates_404_workaround()
# ----------------------------
# Presets
# ----------------------------

TEXT_MODEL_PRESETS: Dict[str, str] = {
    "nllb-200-3.3B": "facebook/nllb-200-3.3B",
    "bge-m3": "BAAI/bge-m3",
    "qwen3-embed-8b": "Qwen/Qwen3-Embedding-8B",
}

IMAGE_MODEL_PRESETS: Dict[str, str] = {
    "dinov3-vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "clip-vit-bigg14": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "c-radio-v4-h": "nvidia/C-RADIOv4-H",
}


# ----------------------------
# Helpers
# ----------------------------

def _torch_dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("fp32", "float32"):
        return torch.float32
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {s}")


def _sanitize_tag(s: str) -> str:
    s = s.strip()
    # Replace path-y / punctuation with underscores
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("_")
    return s or "model"


def resolve_text_model(model: str) -> Tuple[str, str]:
    """Return (model_name, model_tag)."""
    if model in TEXT_MODEL_PRESETS:
        return TEXT_MODEL_PRESETS[model], _sanitize_tag(model)
    return model, _sanitize_tag(model)


def resolve_image_model(model: str) -> Tuple[str, str]:
    """Return (model_name, model_tag)."""
    if model in IMAGE_MODEL_PRESETS:
        return IMAGE_MODEL_PRESETS[model], _sanitize_tag(model)
    return model, _sanitize_tag(model)


def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_clean_outdir(out_dir: str, overwrite: bool) -> None:
    if os.path.isdir(out_dir):
        done = os.path.join(out_dir, "DONE")
        running = os.path.join(out_dir, "RUNNING")
        if os.path.exists(done) and not overwrite:
            # caller will skip
            return
        if os.path.exists(running) and not overwrite:
            raise RuntimeError(
                f"Found RUNNING marker in {out_dir}. "
                "A previous run likely crashed. Use --overwrite to rebuild "
                "or delete this folder."
            )
        if overwrite:
            shutil.rmtree(out_dir)


def _mark_running(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "RUNNING"), "w", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))


def _mark_done(out_dir: str) -> None:
    running = os.path.join(out_dir, "RUNNING")
    if os.path.exists(running):
        os.remove(running)
    with open(os.path.join(out_dir, "DONE"), "w", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))


def _list_pt_shards(out_dir: str, prefix: str) -> List[str]:
    files = []
    for fn in os.listdir(out_dir):
        if fn.startswith(prefix) and fn.endswith(".pt"):
            files.append(os.path.join(out_dir, fn))
    return sorted(files)


# ----------------------------
# 1) Dataset parsing + index
# ----------------------------

@dataclass
class XM3600ImageEntry:
    """One XM3600 image and all captions in all languages."""

    image_id: str
    captions: Dict[str, List[str]] = field(default_factory=dict)


class XM3600MultiLangDataset:
    """Parse XM3600 captions.jsonl and collect all languages per image."""

    def __init__(
        self,
        captions_path: str,
        images_dir: str,
        max_images: Optional[int] = None,
    ):
        self.captions_path = captions_path
        self.images_dir = images_dir
        self.max_images = max_images

        self.entries: List[XM3600ImageEntry] = []
        self.languages: List[str] = []

        self._load()

    def _parse_one_record(self, j: dict) -> Optional[XM3600ImageEntry]:
        image_id = (
            j.get("image/key")
            or j.get("image_id")
            or j.get("id")
            or j.get("image")
        )
        if image_id is None:
            return None

        img_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        if not os.path.isfile(img_path):
            return None

        captions_by_lang: Dict[str, List[str]] = {}
        for key, value in j.items():
            if key.startswith("image/"):
                continue
            if not isinstance(value, dict):
                continue

            caps = value.get("caption") or value.get("captions")
            if not caps:
                continue

            if isinstance(caps, str):
                texts = [caps]
            elif isinstance(caps, list):
                texts = [c for c in caps if isinstance(c, str)]
            else:
                continue

            if texts:
                captions_by_lang[key] = texts

        if not captions_by_lang:
            return None

        return XM3600ImageEntry(image_id=str(image_id), captions=captions_by_lang)

    def _load(self) -> None:
        print(f"[XM3600] Reading captions from: {self.captions_path}")
        n_kept = 0
        all_langs = set()

        with open(self.captions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                j = json.loads(line)

                entry = self._parse_one_record(j)
                if entry is None:
                    continue

                self.entries.append(entry)
                n_kept += 1

                for lang in entry.captions.keys():
                    all_langs.add(lang)

                if self.max_images is not None and n_kept >= self.max_images:
                    break

                if n_kept % 1000 == 0:
                    print(f"[XM3600] Parsed {n_kept} images so far...")

        self.languages = sorted(all_langs)
        total_caps = sum(sum(len(caps) for caps in e.captions.values()) for e in self.entries)
        print(
            f"[XM3600] Loaded {len(self.entries)} images "
            f"with {total_caps} captions total "
            f"across {len(self.languages)} languages."
        )
        if not self.entries:
            raise ValueError(
                "XM3600 dataset is empty. Check captions_path, images_dir and JSON schema."
            )


def build_xm3600_index(
    captions_path: str,
    images_dir: str,
    cache_dir: str,
    max_images: Optional[int] = None,
) -> None:
    """Build index files under cache_dir/index."""
    index_dir = os.path.join(cache_dir, "index")
    os.makedirs(index_dir, exist_ok=True)

    images_path = os.path.join(index_dir, "images.jsonl")
    captions_path_out = os.path.join(index_dir, "captions.jsonl")
    meta_path = os.path.join(index_dir, "meta.json")

    # If index already exists, don't rebuild.
    if os.path.isfile(images_path) and os.path.isfile(captions_path_out) and os.path.isfile(meta_path):
        print(f"[INDEX] Already exists at {index_dir}. Skipping.")
        return

    ds = XM3600MultiLangDataset(captions_path=captions_path, images_dir=images_dir, max_images=max_images)

    # Write images.jsonl and captions.jsonl
    n_images = 0
    n_captions = 0

    with open(images_path, "w", encoding="utf-8") as f_img, open(
        captions_path_out, "w", encoding="utf-8"
    ) as f_cap:
        for image_idx, entry in enumerate(ds.entries):
            image_relpath = f"{entry.image_id}.jpg"
            img_row = {
                "image_idx": image_idx,
                "image_id": entry.image_id,
                "image_relpath": image_relpath,
                "languages": sorted(list(entry.captions.keys())),
                "num_captions_per_lang": {k: len(v) for k, v in entry.captions.items()},
            }
            f_img.write(json.dumps(img_row, ensure_ascii=False) + "\n")
            n_images += 1

            for lang, caps in entry.captions.items():
                for cap in caps:
                    cap_row = {
                        "caption_id": n_captions,
                        "image_idx": image_idx,
                        "image_id": entry.image_id,
                        "language": lang,
                        "caption": cap,
                    }
                    f_cap.write(json.dumps(cap_row, ensure_ascii=False) + "\n")
                    n_captions += 1

    meta = {
        "schema_version": 2,
        "captions_jsonl": os.path.abspath(captions_path),
        "images_dir": os.path.abspath(images_dir),
        "max_images": max_images,
        "n_images": n_images,
        "n_captions": n_captions,
        "languages": ds.languages,
    }

    # Fingerprint helps ensure we’re aligning to the same index.
    meta["images_sha1"] = _sha1_file(images_path)
    meta["captions_sha1"] = _sha1_file(captions_path_out)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[INDEX] Wrote: {images_path}")
    print(f"[INDEX] Wrote: {captions_path_out}")
    print(f"[INDEX] Wrote: {meta_path}")


def load_xm3600_index(cache_dir: str) -> Dict[str, Any]:
    index_dir = os.path.join(cache_dir, "index")
    images_path = os.path.join(index_dir, "images.jsonl")
    captions_path = os.path.join(index_dir, "captions.jsonl")
    meta_path = os.path.join(index_dir, "meta.json")

    if not (os.path.isfile(images_path) and os.path.isfile(captions_path) and os.path.isfile(meta_path)):
        raise FileNotFoundError(
            f"XM3600 index not found in {index_dir}. Run build_index first."
        )

    images: List[dict] = []
    with open(images_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                images.append(json.loads(line))

    captions: List[dict] = []
    with open(captions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                captions.append(json.loads(line))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "index_dir": index_dir,
        "images_path": images_path,
        "captions_path": captions_path,
        "meta_path": meta_path,
        "images": images,
        "captions": captions,
        "meta": meta,
    }


# ----------------------------
# 2) Pooling helpers
# ----------------------------

@torch.no_grad()
def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean pool excluding padding."""
    if attention_mask is None:
        return last_hidden_state.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)  # [B,T,1]
    denom = mask.sum(dim=1).clamp_min(1e-6)  # [B,1]
    return (last_hidden_state * mask).sum(dim=1) / denom


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Pool the last non-padding token (common for causal embeddings)."""
    if attention_mask is None:
        return last_hidden_state[:, -1, :]
    idx = attention_mask.sum(dim=1) - 1
    idx = idx.clamp(min=0)
    b = last_hidden_state.shape[0]
    return last_hidden_state[torch.arange(b, device=last_hidden_state.device), idx]


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


# ----------------------------
# 3) NLLB language mapping
# ----------------------------

def resolve_nllb_src_lang(tokenizer, xm_lang: str) -> Tuple[str, bool]:
    """Map XM3600 language code to NLLB/FLORES-200 code when possible."""
    available = getattr(tokenizer, "lang_code_to_id", None)
    available_keys = set(available.keys()) if isinstance(available, dict) else set()

    if xm_lang in available_keys:
        return xm_lang, False

    candidates: Dict[str, List[str]] = {
        "en": ["eng_Latn"],
        "de": ["deu_Latn"],
        "fr": ["fra_Latn"],
        "es": ["spa_Latn"],
        "it": ["ita_Latn"],
        "pt": ["por_Latn"],
        "nl": ["nld_Latn"],
        "sv": ["swe_Latn"],
        "no": ["nob_Latn", "nor_Latn"],
        "fi": ["fin_Latn"],
        "pl": ["pol_Latn"],
        "cs": ["ces_Latn", "cze_Latn"],
        "ro": ["ron_Latn", "rum_Latn"],
        "ru": ["rus_Cyrl"],
        "uk": ["ukr_Cyrl"],
        "el": ["ell_Grek"],
        "tr": ["tur_Latn"],
        "ar": ["arb_Arab", "ara_Arab"],
        "he": ["heb_Hebr"],
        "fa": ["pes_Arab", "fas_Arab"],
        "hi": ["hin_Deva"],
        "bn": ["ben_Beng"],
        "te": ["tel_Telu"],
        "ta": ["tam_Taml"],
        "ml": ["mal_Mlym"],
        "kn": ["kan_Knda"],
        "gu": ["guj_Gujr"],
        "mr": ["mar_Deva"],
        "pa": ["pan_Guru"],
        "ur": ["urd_Arab"],
        "th": ["tha_Thai"],
        "vi": ["vie_Latn"],
        "id": ["ind_Latn"],
        "ja": ["jpn_Jpan"],
        "ko": ["kor_Hang"],
        "zh": ["zho_Hans", "zho_Hant"],
        "mi": ["mri_Latn"],
        "quz": ["quz_Latn"],
    }

    for cand in candidates.get(xm_lang, []):
        if cand in available_keys:
            return cand, False

    fallback = "eng_Latn"
    if fallback in available_keys:
        return fallback, True

    if available_keys:
        return sorted(list(available_keys))[0], True

    return "eng_Latn", True


# ----------------------------
# 4) Encoders
# ----------------------------

class DinoV3ImageEncoder:
    """DINOv3 image encoder wrapper."""

    def __init__(
        self,
        model_name: str,
        device: str,
        infer_dtype: str,
        image_size: Optional[int] = None,
        use_pooler_output: bool = True,
    ):
        self.device = torch.device(device)
        self.model_name = model_name
        self.infer_dtype = _torch_dtype_from_str(infer_dtype)
        self.image_size = image_size
        self.use_pooler_output = use_pooler_output

        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=(self.infer_dtype if self.device.type == "cuda" else None),
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_paths(self, image_paths: List[str], batch_size: int = 16) -> torch.Tensor:
        all_embs = []
        desc = f"[DINOv3] Encoding images ({os.path.basename(self.model_name)})"

        for i in tqdm(range(0, len(image_paths), batch_size), desc=desc):
            batch_paths = image_paths[i: i + batch_size]
            images = []
            for p in batch_paths:
                with Image.open(p) as img:
                    images.append(img.convert("RGB"))

            proc_kwargs = {"images": images, "return_tensors": "pt"}
            if self.image_size is not None:
                proc_kwargs["size"] = {"height": int(self.image_size), "width": int(self.image_size)}

            inputs = self.image_processor(**proc_kwargs)
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

            if self.device.type == "cuda" and self.infer_dtype in (torch.float16, torch.bfloat16):
                with torch.autocast(device_type="cuda", dtype=self.infer_dtype):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

            if self.use_pooler_output and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            else:
                emb = outputs.last_hidden_state[:, 0, :]

            all_embs.append(emb.detach().float().cpu())

        return torch.cat(all_embs, dim=0)


class CLIPImageEncoder:
    """CLIP image encoder (uses CLIPModel.get_image_features)."""

    def __init__(
        self,
        model_name: str,
        device: str,
        infer_dtype: str,
        image_size: Optional[int] = None,
        normalize: bool = True,
    ):
        if CLIPModel is None or CLIPProcessor is None:
            raise ImportError("transformers.CLIPModel/CLIPProcessor not available in this environment")

        self.device = torch.device(device)
        self.model_name = model_name
        self.infer_dtype = _torch_dtype_from_str(infer_dtype)
        self.image_size = image_size
        self.normalize = normalize

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(
            model_name,
            torch_dtype=(self.infer_dtype if self.device.type == "cuda" else None),
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_paths(self, image_paths: List[str], batch_size: int = 16) -> torch.Tensor:
        all_embs = []
        desc = f"[CLIP] Encoding images ({os.path.basename(self.model_name)})"

        for i in tqdm(range(0, len(image_paths), batch_size), desc=desc):
            batch_paths = image_paths[i: i + batch_size]
            images = []
            for p in batch_paths:
                with Image.open(p) as img:
                    images.append(img.convert("RGB"))

            proc_kwargs = {"images": images, "return_tensors": "pt"}
            if self.image_size is not None:
                # CLIPProcessor supports size via image_processor; this is best-effort.
                proc_kwargs["size"] = {"height": int(self.image_size), "width": int(self.image_size)}

            inputs = self.processor(**proc_kwargs)
            pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)

            if self.device.type == "cuda" and self.infer_dtype in (torch.float16, torch.bfloat16):
                with torch.autocast(device_type="cuda", dtype=self.infer_dtype):
                    feats = self.model.get_image_features(pixel_values=pixel_values)
            else:
                feats = self.model.get_image_features(pixel_values=pixel_values)

            feats = feats.detach().float()
            if self.normalize:
                feats = l2_normalize(feats)
            all_embs.append(feats.cpu())

        return torch.cat(all_embs, dim=0)


class RADIOImageEncoder:
    """RADIO image encoder (best-effort handling of different output formats)."""

    def __init__(
        self,
        model_name: str,
        device: str,
        infer_dtype: str,
        image_size: Optional[int] = None,
        normalize: bool = True,
    ):
        self.device = torch.device(device)
        self.model_name = model_name
        self.infer_dtype = _torch_dtype_from_str(infer_dtype)
        self.image_size = image_size
        self.normalize = normalize

        self.image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=(self.infer_dtype if self.device.type == "cuda" else None),
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def _extract_embedding(self, outputs: Any) -> torch.Tensor:
        # outputs may be a ModelOutput, dict, or tuple
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, "summary") and outputs.summary is not None:
            return outputs.summary
        if isinstance(outputs, dict):
            if "pooler_output" in outputs and outputs["pooler_output"] is not None:
                return outputs["pooler_output"]
            if "summary" in outputs and outputs["summary"] is not None:
                return outputs["summary"]
            if "embeddings" in outputs and outputs["embeddings"] is not None:
                return outputs["embeddings"]
            if "last_hidden_state" in outputs:
                return outputs["last_hidden_state"][:, 0, :]
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[:, 0, :]
        # tuple fallback
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
            # assume first tensor is [B, T, D]
            t = outputs[0]
            if t.dim() == 3:
                return t[:, 0, :]
            return t
        raise RuntimeError("Could not extract RADIO embedding from model outputs")

    @torch.no_grad()
    def encode_paths(self, image_paths: List[str], batch_size: int = 16) -> torch.Tensor:
        all_embs = []
        desc = f"[RADIO] Encoding images ({os.path.basename(self.model_name)})"

        def _safe_exif_transpose(im: Image.Image) -> Image.Image:
            # Some images have corrupted EXIF that can raise (e.g., "not a TIFF file").
            try:
                return ImageOps.exif_transpose(im)
            except Exception:
                return im

        def _prep_letterbox(im: Image.Image, sz: int) -> Image.Image:
            im = _safe_exif_transpose(im)
            im = im.convert("RGB")
            # Resize to fit inside sz×sz without cropping, then pad to exactly sz×sz.
            im = ImageOps.contain(im, (sz, sz), method=Image.BICUBIC)
            canvas = Image.new("RGB", (sz, sz), (0, 0, 0))  # black padding
            x = (sz - im.width) // 2
            y = (sz - im.height) // 2
            canvas.paste(im, (x, y))
            return canvas

        # RADIO enforces certain input resolutions (multiple-of-step). Snap if needed.
        def _snap_radio_size(sz: int) -> int:
            try:
                fn = getattr(self.model, "get_nearest_supported_resolution", None)
                if fn is None:
                    return sz
                res = fn(int(sz), int(sz))
                # res can be a tuple or an object with .height/.width
                if isinstance(res, (tuple, list)) and len(res) >= 2:
                    return int(res[0])
                h = getattr(res, "height", None)
                if h is not None:
                    return int(h)
            except Exception:
                pass
            return sz

        def _forward_model(inputs: Dict[str, torch.Tensor]) -> Any:
            """RADIO remote-code models differ: some expect x as positional arg, not pixel_values kw."""
            try:
                return self.model(**inputs)
            except TypeError as e:
                msg = str(e)
                # Most common: forward(x) but we passed pixel_values=...
                if "pixel_values" in inputs and "unexpected keyword argument" in msg:
                    pv = inputs["pixel_values"]
                    # Try positional first
                    try:
                        return self.model(pv)
                    except Exception:
                        # Try a few common alternative kw names
                        for k in ("x", "images", "inputs", "input"):
                            try:
                                return self.model(**{k: pv})
                            except Exception:
                                continue
                        raise
                raise

        for i in tqdm(range(0, len(image_paths), batch_size), desc=desc):
            batch_paths = image_paths[i: i + batch_size]

            # Decide target size for this batch
            if self.image_size is None:
                # If no image_size, pick a consistent size per batch (largest side).
                tmp = []
                for p in batch_paths:
                    with Image.open(p) as img:
                        tmp.append(_safe_exif_transpose(img).convert("RGB"))
                sz = max(max(im.size) for im in tmp)
                sz = _snap_radio_size(sz)
                images = [_prep_letterbox(im, sz) for im in tmp]
            else:
                sz = _snap_radio_size(int(self.image_size))
                images = []
                for p in batch_paths:
                    with Image.open(p) as img:
                        images.append(_prep_letterbox(img, sz))

            # Run processor. Some processors don't accept do_resize/do_center_crop kwargs.
            try:
                inputs = self.image_processor(
                    images=images,
                    return_tensors="pt",
                    do_resize=False,
                    do_center_crop=False,
                )
            except TypeError:
                inputs = self.image_processor(images=images, return_tensors="pt")

            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

            if self.device.type == "cuda" and self.infer_dtype in (torch.float16, torch.bfloat16):
                with torch.autocast(device_type="cuda", dtype=self.infer_dtype):
                    outputs = _forward_model(inputs)
            else:
                outputs = _forward_model(inputs)

            emb = self._extract_embedding(outputs).detach().float()
            if self.normalize:
                emb = l2_normalize(emb)
            all_embs.append(emb.cpu())

        return torch.cat(all_embs, dim=0)
class NLLBTextEncoder:
    """NLLB encoder embeddings using the encoder side of the seq2seq model."""

    def __init__(
        self,
        model_name: str,
        device: str,
        infer_dtype: str,
        max_length: int,
        normalize: bool = True,
    ):
        self.device = torch.device(device)
        self.model_name = model_name
        self.infer_dtype = _torch_dtype_from_str(infer_dtype)
        self.max_length = int(max_length)
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=(self.infer_dtype if self.device.type == "cuda" else None),
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.encoder = self.model.get_encoder()
        self.encoder.to(self.device)
        self.encoder.eval()

    @torch.no_grad()
    def encode_grouped_by_lang_into_memmap(
        self,
        captions: List[dict],
        out_memmap: np.memmap,
        batch_size: int,
    ) -> Dict[str, Any]:
        """Fill out_memmap[caption_id] with embeddings. Returns language mapping info."""
        # Build language grouping
        lang_to_rows: Dict[str, List[int]] = {}
        lang_to_texts: Dict[str, List[str]] = {}
        for row in captions:
            lang = row["language"]
            lang_to_rows.setdefault(lang, []).append(int(row["caption_id"]))
            lang_to_texts.setdefault(lang, []).append(row["caption"])

        lang_map: Dict[str, str] = {}
        fallback_langs: List[str] = []

        for xm_lang in tqdm(sorted(lang_to_rows.keys()), desc="[NLLB] Encoding captions by lang"):
            rows = lang_to_rows[xm_lang]
            texts = lang_to_texts[xm_lang]
            resolved, is_fallback = resolve_nllb_src_lang(self.tokenizer, xm_lang)
            lang_map[xm_lang] = resolved
            if is_fallback:
                fallback_langs.append(xm_lang)

            try:
                self.tokenizer.src_lang = resolved
            except Exception:
                pass

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i: i + batch_size]
                batch_rows = rows[i: i + batch_size]

                enc = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}

                if self.device.type == "cuda" and self.infer_dtype in (torch.float16, torch.bfloat16):
                    with torch.autocast(device_type="cuda", dtype=self.infer_dtype):
                        out = self.encoder(
                            input_ids=enc["input_ids"],
                            attention_mask=enc.get("attention_mask", None),
                        )
                else:
                    out = self.encoder(
                        input_ids=enc["input_ids"],
                        attention_mask=enc.get("attention_mask", None),
                    )

                emb = masked_mean_pool(out.last_hidden_state, enc.get("attention_mask", None))
                emb = emb.detach().float()
                if self.normalize:
                    emb = l2_normalize(emb)

                out_memmap[np.asarray(batch_rows, dtype=np.int64)] = emb.cpu().numpy()

        return {"xm_to_nllb_lang_map": lang_map, "fallback_langs": sorted(set(fallback_langs))}


class BGEM3TextEncoder:
    """BGE-M3 encoder (CLS pooling by default)."""

    def __init__(
        self,
        model_name: str,
        device: str,
        infer_dtype: str,
        max_length: int,
        normalize: bool = True,
    ):
        self.device = torch.device(device)
        self.model_name = model_name
        self.infer_dtype = _torch_dtype_from_str(infer_dtype)
        self.max_length = int(max_length)
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=(self.infer_dtype if self.device.type == "cuda" else None),
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int) -> torch.Tensor:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}
            if self.device.type == "cuda" and self.infer_dtype in (torch.float16, torch.bfloat16):
                with torch.autocast(device_type="cuda", dtype=self.infer_dtype):
                    out = self.model(**enc)
            else:
                out = self.model(**enc)

            emb = out.last_hidden_state[:, 0, :].detach().float()
            if self.normalize:
                emb = l2_normalize(emb)
            all_embs.append(emb.cpu())

        return torch.cat(all_embs, dim=0)


class Qwen3EmbeddingTextEncoder:
    """Qwen3 embedding encoder (best-effort; last-token pooling fallback)."""

    def __init__(
        self,
        model_name: str,
        device: str,
        infer_dtype: str,
        max_length: int,
        normalize: bool = True,
    ):
        self.device = torch.device(device)
        self.model_name = model_name
        self.infer_dtype = _torch_dtype_from_str(infer_dtype)
        self.max_length = int(max_length)
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=(self.infer_dtype if self.device.type == "cuda" else None),
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int) -> torch.Tensor:
        # If model provides an encode method, try to use it first.
        if hasattr(self.model, "encode"):
            try:
                vecs = self.model.encode(texts, batch_size=batch_size)  # type: ignore
                if isinstance(vecs, np.ndarray):
                    t = torch.from_numpy(vecs).float()
                elif torch.is_tensor(vecs):
                    t = vecs.detach().float().cpu()
                else:
                    t = torch.tensor(vecs, dtype=torch.float32)
                if self.normalize:
                    t = l2_normalize(t)
                return t
            except Exception:
                # fall back to forward pass
                pass

        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}

            if self.device.type == "cuda" and self.infer_dtype in (torch.float16, torch.bfloat16):
                with torch.autocast(device_type="cuda", dtype=self.infer_dtype):
                    out = self.model(**enc)
            else:
                out = self.model(**enc)

            hs = out.last_hidden_state
            emb = last_token_pool(hs, enc.get("attention_mask", None)).detach().float()
            if self.normalize:
                emb = l2_normalize(emb)
            all_embs.append(emb.cpu())

        return torch.cat(all_embs, dim=0)


# ----------------------------
# 5) Embedding writers
# ----------------------------

def _save_pt_shards_from_memmap(
    memmap_arr: np.memmap,
    out_dir: str,
    prefix: str,
    save_dtype: torch.dtype,
    chunk_size: int,
) -> List[str]:
    paths = []
    n = int(memmap_arr.shape[0])
    for start in tqdm(range(0, n, chunk_size), desc=f"[SAVE] {prefix} shards"):
        end = min(n, start + chunk_size)
        chunk = torch.from_numpy(np.asarray(memmap_arr[start:end])).to(save_dtype)
        fn = f"{prefix}_{start // chunk_size:04d}.pt"
        p = os.path.join(out_dir, fn)
        torch.save(chunk, p)
        paths.append(p)
    return paths


def _save_pt_shards_streaming(
    embs: Iterable[torch.Tensor],
    out_dir: str,
    prefix: str,
    save_dtype: torch.dtype,
    chunk_size: int,
) -> List[str]:
    """Save shards from an iterator of [B,D] cpu float tensors."""
    paths: List[str] = []
    buf: List[torch.Tensor] = []
    count = 0
    shard_idx = 0

    def flush() -> None:
        nonlocal buf, count, shard_idx
        if not buf:
            return
        chunk = torch.cat(buf, dim=0).to(save_dtype)
        fn = f"{prefix}_{shard_idx:04d}.pt"
        p = os.path.join(out_dir, fn)
        torch.save(chunk, p)
        paths.append(p)
        shard_idx += 1
        buf = []

    for t in embs:
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32)
        t = t.detach().cpu().float()
        buf.append(t)
        count += int(t.shape[0])
        if count >= chunk_size:
            flush()
            count = 0

    flush()
    return paths


# ----------------------------
# 6) Build embeddings
# ----------------------------

def build_xm3600_image_embeddings(
    cache_dir: str,
    image_model: str,
    device: str = "cpu",
    infer_dtype: str = "float32",
    save_dtype: str = "float32",
    img_batch: int = 16,
    image_size: Optional[int] = None,
    emb_chunk_size: int = 50000,
    overwrite: bool = False,
    skip_existing: bool = True,
    images_dir: Optional[str] = None,
    normalize: bool = True,
) -> str:
    """Build image embeddings for one model. Returns output folder."""
    idx = load_xm3600_index(cache_dir)
    meta = idx["meta"]

    model_name, model_tag = resolve_image_model(image_model)

    out_dir = os.path.join(cache_dir, "embeddings", "image", model_tag)
    done_path = os.path.join(out_dir, "DONE")
    if skip_existing and os.path.exists(done_path) and not overwrite:
        print(f"[SKIP] Image embeddings already cached: {out_dir}")
        return out_dir

    _ensure_clean_outdir(out_dir, overwrite=overwrite)
    if skip_existing and os.path.exists(done_path) and not overwrite:
        print(f"[SKIP] Image embeddings already cached: {out_dir}")
        return out_dir

    _mark_running(out_dir)

    save_torch_dtype = _torch_dtype_from_str(save_dtype)

    base_images_dir = images_dir or meta.get("images_dir")
    if not base_images_dir:
        raise ValueError("images_dir not found. Provide --images_dir or rebuild index.")

    image_paths = [os.path.join(base_images_dir, row["image_relpath"]) for row in idx["images"]]

    # Select encoder
    if model_tag == "dinov3-vit7b16" or model_name.startswith("facebook/dinov3"):
        encoder = DinoV3ImageEncoder(
            model_name=model_name,
            device=device,
            infer_dtype=infer_dtype,
            image_size=image_size,
            use_pooler_output=True,
        )
    elif model_tag == "clip-vit-bigg14" or "CLIP" in model_name.upper():
        encoder = CLIPImageEncoder(
            model_name=model_name,
            device=device,
            infer_dtype=infer_dtype,
            image_size=image_size,
            normalize=normalize,
        )
    else:
        encoder = RADIOImageEncoder(
            model_name=model_name,
            device=device,
            infer_dtype=infer_dtype,
            image_size=image_size,
            normalize=normalize,
        )

    # Encode in batches, stream to shard writer
    def emb_iter() -> Iterable[torch.Tensor]:
        for i in range(0, len(image_paths), img_batch):
            batch_paths = image_paths[i: i + img_batch]
            yield encoder.encode_paths(batch_paths, batch_size=len(batch_paths))

    shard_paths = _save_pt_shards_streaming(
        embs=emb_iter(),
        out_dir=out_dir,
        prefix="image_embs",
        save_dtype=save_torch_dtype,
        chunk_size=emb_chunk_size,
    )

    # Write meta
    # Determine embedding dim from first shard
    first = torch.load(shard_paths[0], map_location="cpu")
    emb_dim = int(first.shape[1])

    emb_meta = {
        "schema_version": 2,
        "modality": "image",
        "model_name": model_name,
        "model_tag": model_tag,
        "infer_dtype": infer_dtype,
        "save_dtype": save_dtype,
        "normalize": normalize,
        "image_size": image_size,
        "img_batch": img_batch,
        "n_images": len(image_paths),
        "embedding_dim": emb_dim,
        "index_images_sha1": meta.get("images_sha1"),
        "index_captions_sha1": meta.get("captions_sha1"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "shards": [os.path.basename(p) for p in shard_paths],
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(emb_meta, f, ensure_ascii=False, indent=2)

    _mark_done(out_dir)
    print(f"[OK] Wrote image cache: {out_dir}")
    return out_dir


def build_xm3600_text_embeddings(
    cache_dir: str,
    text_model: str,
    device: str = "cpu",
    infer_dtype: str = "float32",
    save_dtype: str = "float32",
    txt_batch: int = 64,
    max_length: int = 256,
    emb_chunk_size: int = 200000,
    overwrite: bool = False,
    skip_existing: bool = True,
    normalize: bool = True,
) -> str:
    """Build text embeddings for one model. Returns output folder."""
    idx = load_xm3600_index(cache_dir)
    meta = idx["meta"]

    model_name, model_tag = resolve_text_model(text_model)

    out_dir = os.path.join(cache_dir, "embeddings", "text", model_tag)
    done_path = os.path.join(out_dir, "DONE")
    if skip_existing and os.path.exists(done_path) and not overwrite:
        print(f"[SKIP] Text embeddings already cached: {out_dir}")
        return out_dir

    _ensure_clean_outdir(out_dir, overwrite=overwrite)
    if skip_existing and os.path.exists(done_path) and not overwrite:
        print(f"[SKIP] Text embeddings already cached: {out_dir}")
        return out_dir

    _mark_running(out_dir)

    save_torch_dtype = _torch_dtype_from_str(save_dtype)

    captions = idx["captions"]
    texts = [row["caption"] for row in captions]

    # Choose encoder
    is_nllb = (model_tag == "nllb-200-3.3B") or ("nllb" in model_name.lower())
    is_bge = (model_tag == "bge-m3") or ("bge-m3" in model_name.lower())
    is_qwen = (model_tag == "qwen3-embed-8b") or ("qwen" in model_name.lower())

    shard_paths: List[str] = []
    extra_meta: Dict[str, Any] = {}

    if is_nllb:
        # NLLB requires per-language src_lang; fill via memmap to avoid huge RAM.
        enc = NLLBTextEncoder(
            model_name=model_name,
            device=device,
            infer_dtype=infer_dtype,
            max_length=max_length,
            normalize=normalize,
        )

        # Determine embedding dim from a single forward pass on a tiny batch.
        sample_lang = captions[0]["language"]
        resolved, _ = resolve_nllb_src_lang(enc.tokenizer, sample_lang)
        try:
            enc.tokenizer.src_lang = resolved
        except Exception:
            pass
        tiny = enc.tokenizer([texts[0]], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        tiny = {k: v.to(enc.device) for k, v in tiny.items()}
        with torch.no_grad():
            if enc.device.type == "cuda" and enc.infer_dtype in (torch.float16, torch.bfloat16):
                with torch.autocast(device_type="cuda", dtype=enc.infer_dtype):
                    out = enc.encoder(input_ids=tiny["input_ids"], attention_mask=tiny.get("attention_mask", None))
            else:
                out = enc.encoder(input_ids=tiny["input_ids"], attention_mask=tiny.get("attention_mask", None))
        d = int(out.last_hidden_state.shape[-1])

        tmp_path = os.path.join(out_dir, "_tmp_caption_embs.f32.dat")
        mm = np.memmap(tmp_path, dtype="float32", mode="w+", shape=(len(captions), d))
        info = enc.encode_grouped_by_lang_into_memmap(captions=captions, out_memmap=mm, batch_size=txt_batch)
        mm.flush()

        shard_paths = _save_pt_shards_from_memmap(
            memmap_arr=mm,
            out_dir=out_dir,
            prefix="caption_embs",
            save_dtype=save_torch_dtype,
            chunk_size=emb_chunk_size,
        )

        # Cleanup tmp file
        try:
            del mm
            os.remove(tmp_path)
        except Exception:
            pass

        extra_meta.update(info)

    else:
        if is_bge:
            enc = BGEM3TextEncoder(
                model_name=model_name,
                device=device,
                infer_dtype=infer_dtype,
                max_length=max_length,
                normalize=normalize,
            )
        else:
            enc = Qwen3EmbeddingTextEncoder(
                model_name=model_name,
                device=device,
                infer_dtype=infer_dtype,
                max_length=max_length,
                normalize=normalize,
            )

        def emb_iter() -> Iterable[torch.Tensor]:
            desc = f"[TEXT] Encoding captions ({os.path.basename(model_name)})"
            for i in tqdm(range(0, len(texts), txt_batch), desc=desc):
                batch = texts[i: i + txt_batch]
                out = enc.encode_texts(batch, batch_size=len(batch))
                yield out

        shard_paths = _save_pt_shards_streaming(
            embs=emb_iter(),
            out_dir=out_dir,
            prefix="caption_embs",
            save_dtype=save_torch_dtype,
            chunk_size=emb_chunk_size,
        )

    first = torch.load(shard_paths[0], map_location="cpu")
    emb_dim = int(first.shape[1])

    emb_meta = {
        "schema_version": 2,
        "modality": "text",
        "model_name": model_name,
        "model_tag": model_tag,
        "infer_dtype": infer_dtype,
        "save_dtype": save_dtype,
        "normalize": normalize,
        "max_length": max_length,
        "txt_batch": txt_batch,
        "n_captions": len(captions),
        "embedding_dim": emb_dim,
        "index_images_sha1": meta.get("images_sha1"),
        "index_captions_sha1": meta.get("captions_sha1"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "shards": [os.path.basename(p) for p in shard_paths],
        **extra_meta,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(emb_meta, f, ensure_ascii=False, indent=2)

    _mark_done(out_dir)
    print(f"[OK] Wrote text cache: {out_dir}")
    return out_dir


# ----------------------------
# 7) Load embeddings
# ----------------------------

def load_xm3600_image_embeddings(
    cache_dir: str,
    image_model: str,
    device: str = "cpu",
    load_shards: bool = True,
) -> Dict[str, Any]:
    _, model_tag = resolve_image_model(image_model)
    out_dir = os.path.join(cache_dir, "embeddings", "image", model_tag)
    meta_path = os.path.join(out_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Image embedding cache not found: {out_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    shard_paths = _list_pt_shards(out_dir, "image_embs")
    if not shard_paths:
        raise FileNotFoundError(f"No image_embs shards found in: {out_dir}")

    out = {"dir": out_dir, "meta": meta, "shard_paths": shard_paths}
    if load_shards:
        tensors = [torch.load(p, map_location=device) for p in shard_paths]
        out["image_embs"] = torch.cat(tensors, dim=0)
    return out


def load_xm3600_text_embeddings(
    cache_dir: str,
    text_model: str,
    device: str = "cpu",
    load_shards: bool = True,
) -> Dict[str, Any]:
    _, model_tag = resolve_text_model(text_model)
    out_dir = os.path.join(cache_dir, "embeddings", "text", model_tag)
    meta_path = os.path.join(out_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Text embedding cache not found: {out_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    shard_paths = _list_pt_shards(out_dir, "caption_embs")
    if not shard_paths:
        raise FileNotFoundError(f"No caption_embs shards found in: {out_dir}")

    out = {"dir": out_dir, "meta": meta, "shard_paths": shard_paths}
    if load_shards:
        tensors = [torch.load(p, map_location=device) for p in shard_paths]
        out["caption_embs"] = torch.cat(tensors, dim=0)
    return out


def load_xm3600_embeddings(
    cache_dir: str,
    text_model: str,
    image_model: str,
    device: str = "cpu",
    load_shards: bool = True,
) -> Dict[str, Any]:
    idx = load_xm3600_index(cache_dir)
    txt = load_xm3600_text_embeddings(cache_dir, text_model=text_model, device=device, load_shards=load_shards)
    img = load_xm3600_image_embeddings(cache_dir, image_model=image_model, device=device, load_shards=load_shards)

    out = {"index": idx, "text": txt, "image": img}

    # Helpful consolidation metadata
    if load_shards:
        out["n_images"] = len(idx["images"])
        out["n_captions"] = len(idx["captions"])

    return out


def build_image_to_caption_ids(index: Dict[str, Any]) -> List[List[int]]:
    """Return mapping image_idx -> list of caption_ids."""
    n_images = len(index["images"])
    mapping: List[List[int]] = [[] for _ in range(n_images)]
    for row in index["captions"]:
        mapping[int(row["image_idx"])].append(int(row["caption_id"]))
    return mapping


# ----------------------------
# 8) CLI (subcommands)
# ----------------------------

def _add_common_cache_flags(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--cache_dir", required=True, help="Cache root directory")
    ap.add_argument("--device", default="cpu", help="cpu or cuda:0")
    ap.add_argument(
        "--infer_dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Inference dtype (autocast on CUDA)",
    )
    ap.add_argument(
        "--save_dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Saved embedding dtype (default float32)",
    )
    ap.add_argument("--overwrite", action="store_true", help="Rebuild even if cache exists")
    ap.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Do not skip when DONE exists",
    )


def main():
    ap = argparse.ArgumentParser(description="XM3600 index + embeddings cache")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_index = sub.add_parser("index", help="Build XM3600 index")
    ap_index.add_argument("--captions", required=True, help="Path to XM3600 captions.jsonl")
    ap_index.add_argument("--images", required=True, help="Directory with unpacked XM3600 images")
    ap_index.add_argument("--cache_dir", required=True, help="Cache root directory")
    ap_index.add_argument("--max_images", type=int, default=None, help="Optional: limit to first N images")

    ap_txt = sub.add_parser("text", help="Build text embeddings")
    _add_common_cache_flags(ap_txt)
    ap_txt.add_argument(
        "--text_model",
        default="nllb-200-3.3B",
        help="Preset key or HF repo id (e.g., bge-m3, qwen3-embed-8b)",
    )
    ap_txt.add_argument("--txt_batch", type=int, default=64)
    ap_txt.add_argument("--max_length", type=int, default=256)
    ap_txt.add_argument("--emb_chunk_size", type=int, default=200000)
    ap_txt.add_argument("--no_normalize", action="store_true", help="Do not L2-normalize embeddings")

    ap_img = sub.add_parser("image", help="Build image embeddings")
    _add_common_cache_flags(ap_img)
    ap_img.add_argument(
        "--image_model",
        default="dinov3-vit7b16",
        help="Preset key or HF repo id (e.g., clip-vit-bigg14, c-radio-v4-h)",
    )
    ap_img.add_argument("--img_batch", type=int, default=16)
    ap_img.add_argument("--image_size", type=int, default=None)
    ap_img.add_argument("--emb_chunk_size", type=int, default=50000)
    ap_img.add_argument("--images_dir", type=str, default=None, help="Override images_dir from index meta")
    ap_img.add_argument("--no_normalize", action="store_true", help="Do not L2-normalize embeddings")

    args = ap.parse_args()

    if args.cmd == "index":
        build_xm3600_index(
            captions_path=args.captions,
            images_dir=args.images,
            cache_dir=args.cache_dir,
            max_images=args.max_images,
        )
        return

    skip_existing = not getattr(args, "no_skip_existing")

    if args.cmd == "text":
        build_xm3600_text_embeddings(
            cache_dir=args.cache_dir,
            text_model=args.text_model,
            device=args.device,
            infer_dtype=args.infer_dtype,
            save_dtype=args.save_dtype,
            txt_batch=args.txt_batch,
            max_length=args.max_length,
            emb_chunk_size=args.emb_chunk_size,
            overwrite=args.overwrite,
            skip_existing=skip_existing,
            normalize=not args.no_normalize,
        )
        return

    if args.cmd == "image":
        build_xm3600_image_embeddings(
            cache_dir=args.cache_dir,
            image_model=args.image_model,
            device=args.device,
            infer_dtype=args.infer_dtype,
            save_dtype=args.save_dtype,
            img_batch=args.img_batch,
            image_size=args.image_size,
            emb_chunk_size=args.emb_chunk_size,
            overwrite=args.overwrite,
            skip_existing=skip_existing,
            images_dir=args.images_dir,
            normalize=not args.no_normalize,
        )
        return


if __name__ == "__main__":
    main()