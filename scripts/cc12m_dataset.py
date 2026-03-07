#!/usr/bin/env python3
"""
CC12M (pixparse WebDataset) indexing + embedding caching pipeline (v2).

Goals (v2):
  1) Default save_dtype=float32 (no truncation unless explicitly requested).
  2) Decouple image/text embeddings into separate folders with separate loaders.
  3) Allow choosing among multiple text/image encoder models; skip if already cached.

Folder layout (cache_dir):
  index/
    samples.jsonl          # one record per sample_id (caption + key + shard + meta)
    meta.json
  embeddings/
    text/<model_tag>/
      meta.json
      DONE
      text_embs_0000.pt, text_embs_0001.pt, ...
    image/<model_tag>/
      meta.json
      DONE
      image_embs_0000.pt, image_embs_0001.pt, ...

The ordering invariant is: embeddings are stored in the same order as index/samples.jsonl.
Thus sample_id i corresponds to:
  shard_id = i // emb_chunk_size
  offset   = i %  emb_chunk_size
for a given embedding cache (model_tag + modality).
"""
import glob
import os
import io
import json
import argparse
from typing import Dict, List, Any, Optional, Iterator, Tuple

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from PIL import Image, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# ----------------------------
# 0. Helpers (dtype + pooling)
# ----------------------------

CACHE_VERSION = 2


def sanitize_for_fs(s: str) -> str:
    """Make a string safe for filenames / folder names."""
    return "".join(c if c.isalnum() or c in "-_.=+" else "_" for c in s)


def _torch_dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s == "float32":
        return torch.float32
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {s}")


@torch.no_grad()
def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    last_hidden_state: [B, T, D]
    attention_mask:    [B, T] with 1 for real tokens, 0 for padding
    """
    if attention_mask is None:
        return last_hidden_state.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)   # [B,T,1]
    denom = mask.sum(dim=1).clamp_min(1e-6)                                 # [B,1]
    return (last_hidden_state * mask).sum(dim=1) / denom


@torch.no_grad()
def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Qwen-style last-token pooling that works with either left-padding or right-padding.

    last_hidden_states: [B, T, D]
    attention_mask:     [B, T]
    """
    # If left padded, all rows have last attention token == 1
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def _atomic_write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _is_cache_done(out_dir: str) -> bool:
    return os.path.isfile(os.path.join(out_dir, "DONE")) and os.path.isfile(os.path.join(out_dir, "meta.json"))


def _write_done(out_dir: str) -> None:
    with open(os.path.join(out_dir, "DONE"), "w", encoding="utf-8") as f:
        f.write("ok\n")


# ----------------------------
# 1. Model presets
# ----------------------------

TEXT_MODEL_PRESETS: Dict[str, str] = {
    "nllb-200-3.3B": "facebook/nllb-200-3.3B",
    "bge-m3": "BAAI/bge-m3",
    "qwen3-embed-8b": "Qwen/Qwen3-Embedding-8B",
}

IMAGE_MODEL_PRESETS: Dict[str, str] = {
    "dinov3-vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    # "largest and latest" OpenCLIP variant (ViT-bigG/14)
    "clip-vit-bigg14": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    # "latest" C-RADIO line (v4), largest v4 checkpoint on HF at time of writing
    "c-radio-v4-h": "nvidia/C-RADIOv4-H",
}


def resolve_model_name(user_value: str, presets: Dict[str, str]) -> Tuple[str, str]:
    """
    Returns (model_repo, model_tag).

    - If user_value matches a preset key: use preset repo.
    - Else treat user_value as a HF repo id directly.
    """
    if user_value in presets:
        repo = presets[user_value]
        tag = sanitize_for_fs(user_value)
        return repo, tag
    # treat as direct repo id
    repo = user_value
    tag = sanitize_for_fs(user_value)
    return repo, tag


# ----------------------------
# 2. Streaming CC12M pixparse samples (WDS)
# ----------------------------

def iter_cc12m_pixparse(
    wds_dir: str,
    pattern: str = "cc12m-train-*.tar",
) -> Iterator[Dict[str, Any]]:
    """
    Stream samples from pixparse Conceptual 12M WebDataset shards.

    Each sample yields:
        {
          "image":  PIL.Image (RGB),
          "caption": str,
          "meta": dict,
          "key": str,
          "shard": str,
          "shard_url": str,
        }
    """
    try:
        import webdataset as wds
    except ImportError as e:
        raise ImportError(
            "webdataset is required for CC12M streaming.\n"
            "Install it with: pip install webdataset"
        ) from e

    pattern_path = os.path.join(wds_dir, pattern)
    shard_paths = sorted(glob.glob(pattern_path))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found matching pattern: {pattern_path}")

    print(f"[CC12M] Found {len(shard_paths)} shard tars")

    dataset = (
        wds.WebDataset(
            shard_paths,
            shardshuffle=False,
            handler=wds.warn_and_continue,
        )
        .decode("pil", handler=wds.warn_and_continue)
        .to_tuple("jpg", "txt", "json", "__key__", "__url__")
    )

    for img, txt, meta, key, url in dataset:
        # caption
        if isinstance(txt, bytes):
            caption = txt.decode("utf-8", errors="ignore").strip()
        else:
            caption = str(txt).strip()
        if not caption:
            continue

        # meta
        if not isinstance(meta, dict):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        # image
        if isinstance(img, Image.Image):
            pil_img = img.convert("RGB")
        else:
            pil_img = Image.open(io.BytesIO(img)).convert("RGB")

        shard = os.path.basename(url)

        yield {
            "image": pil_img,
            "caption": caption,
            "meta": meta,
            "key": key,
            "shard": shard,
            "shard_url": url,
        }


# ----------------------------
# 3. Encoders (Image: DINOv3/CLIP/RADIO; Text: NLLB/BGE/Qwen)
# ----------------------------

class DinoV3ImageEncoder:
    """
    Wrapper around a DINOv3 model from HuggingFace Transformers.

    Default:
      facebook/dinov3-vit7b16-pretrain-lvd1689m
    """

    def __init__(
        self,
        model_name: str = IMAGE_MODEL_PRESETS["dinov3-vit7b16"],
        device: str = "cpu",
        infer_dtype: str = "float32",
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
    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        if not images:
            return torch.empty(0, 0)

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
            emb = outputs.last_hidden_state[:, 0, :]  # CLS token

        return emb.detach().cpu()


class CLIPImageEncoder:
    """
    CLIP / OpenCLIP-style image encoder.

    Default: laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
    """

    def __init__(
        self,
        model_name: str = IMAGE_MODEL_PRESETS["clip-vit-bigg14"],
        device: str = "cpu",
        infer_dtype: str = "float32",
        image_size: Optional[int] = None,
    ):
        from transformers import CLIPImageProcessor  # avoid import if unused

        self.device = torch.device(device)
        self.model_name = model_name
        self.infer_dtype = _torch_dtype_from_str(infer_dtype)
        self.image_size = image_size

        # Many CLIP checkpoints use CLIPImageProcessor
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=(self.infer_dtype if self.device.type == "cuda" else None),
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        if not images:
            return torch.empty(0, 0)

        proc_kwargs = {"images": images, "return_tensors": "pt"}
        if self.image_size is not None:
            proc_kwargs["size"] = {"shortest_edge": int(self.image_size)}

        inputs = self.image_processor(**proc_kwargs)
        pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)

        if self.device.type == "cuda" and self.infer_dtype in (torch.float16, torch.bfloat16):
            with torch.autocast(device_type="cuda", dtype=self.infer_dtype):
                emb = self._forward(pixel_values)
        else:
            emb = self._forward(pixel_values)

        return emb.detach().cpu()

    def _forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Preferred: model.get_image_features
        if hasattr(self.model, "get_image_features"):
            return self.model.get_image_features(pixel_values=pixel_values)
        # Next: CLIPModel-style .vision_model output
        if hasattr(self.model, "vision_model"):
            out = self.model.vision_model(pixel_values=pixel_values)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                return out.pooler_output
            return out.last_hidden_state[:, 0, :]
        # Fallback: generic transformer output
        out = self.model(pixel_values=pixel_values)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0, :]


class RADIOImageEncoder:
    """
    NVIDIA C-RADIO encoder via HF custom code.

    Default: nvidia/C-RADIOv4-H

    Note: This requires trust_remote_code=True when loading AutoModel.
    """

    def __init__(
        self,
        model_name: str = IMAGE_MODEL_PRESETS["c-radio-v4-h"],
        device: str = "cpu",
        infer_dtype: str = "float32",
        image_size: Optional[int] = None,
    ):
        from transformers import CLIPImageProcessor  # C-RADIO docs commonly reference this processor

        self.device = torch.device(device)
        self.model_name = model_name
        self.infer_dtype = _torch_dtype_from_str(infer_dtype)
        self.image_size = image_size

        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=(self.infer_dtype if self.device.type == "cuda" else None),
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        if not images:
            return torch.empty(0, 0)
        
        def _prep_letterbox(im: Image.Image, sz: int) -> Image.Image:
            # EXIF can be corrupt; if it fails, keep image as-is (no blank)
            try:
                im = ImageOps.exif_transpose(im)
            except Exception:
                pass

            # Ensure RGB (also safe if already RGB)
            try:
                im = im.convert("RGB")
            except Exception:
                # If convert fails (rare), try a more defensive fallback without blanking:
                im = im.copy().convert("RGB")

            # Preserve aspect ratio, no crop: resize to fit inside sz×sz
            im = ImageOps.contain(im, (sz, sz), method=Image.BICUBIC)

            # Pad to exactly sz×sz
            canvas = Image.new("RGB", (sz, sz), (0, 0, 0))
            x = (sz - im.width) // 2
            y = (sz - im.height) // 2
            canvas.paste(im, (x, y))
            return canvas

        # Choose target size
        if self.image_size is None:
            # Fallback: pad to max dimension within this batch (keeps batching valid)
            sz = max(max(im.size) for im in images)
        else:
            sz = int(self.image_size)

        images = [_prep_letterbox(im, sz) for im in images]

        # IMPORTANT: now all images are the same size => processor can stack tensors
        inputs = self.image_processor(
            images=images,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)

        if self.device.type == "cuda" and self.infer_dtype in (torch.float16, torch.bfloat16):
            with torch.autocast(device_type="cuda", dtype=self.infer_dtype):
                emb = self._forward(pixel_values)
        else:
            emb = self._forward(pixel_values)

        return emb.detach().cpu()

    def _forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values)
        # C-RADIO typically returns (summary, spatial_features)
        if isinstance(out, tuple) and len(out) >= 1:
            return out[0]
        if hasattr(out, "summary"):
            return out.summary
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0, :]
        raise RuntimeError("Unexpected output format from RADIO model.")


class NLLBTextEncoder:
    """
    NLLB encoder embeddings via the encoder side of the seq2seq model.

    Default:
      facebook/nllb-200-3.3B

    Uses masked mean pooling (does not average padding tokens).
    """

    def __init__(
        self,
        model_name: str = TEXT_MODEL_PRESETS["nllb-200-3.3B"],
        device: str = "cpu",
        infer_dtype: str = "float32",
        max_length: int = 256,
        src_lang: str = "eng_Latn",
    ):
        self.device = torch.device(device)
        self.model_name = model_name
        self.infer_dtype = _torch_dtype_from_str(infer_dtype)
        self.max_length = max_length
        self.src_lang = src_lang

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        try:
            self.tokenizer.src_lang = src_lang
        except Exception:
            pass

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
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        n = len(texts)
        if n == 0:
            return torch.empty(0, 0)

        all_embs = []
        for i in range(0, n, batch_size):
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
            all_embs.append(emb.detach().cpu())

        return torch.cat(all_embs, dim=0)


class BGETextEncoder:
    """
    BGE-family encoder (e.g., BAAI/bge-m3).

    NOTE: BGE is trained to use CLS embedding (last_hidden_state[:, 0]),
    not mean pooling.
    """

    def __init__(
        self,
        model_name: str = TEXT_MODEL_PRESETS["bge-m3"],
        device: str = "cpu",
        infer_dtype: str = "float32",
        max_length: int = 512,
    ):
        self.device = torch.device(device)
        self.model_name = model_name
        self.infer_dtype = _torch_dtype_from_str(infer_dtype)
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=(self.infer_dtype if self.device.type == "cuda" else None),
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        n = len(texts)
        if n == 0:
            return torch.empty(0, 0)

        all_embs = []
        for i in range(0, n, batch_size):
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

            emb = out.last_hidden_state[:, 0, :]  # CLS
            all_embs.append(emb.detach().cpu())

        return torch.cat(all_embs, dim=0)


class Qwen3EmbeddingTextEncoder:
    """
    Qwen3 Embedding model encoder (e.g., Qwen/Qwen3-Embedding-8B).

    Uses last-token pooling (Qwen model card "Transformers Usage").
    """

    def __init__(
        self,
        model_name: str = TEXT_MODEL_PRESETS["qwen3-embed-8b"],
        device: str = "cpu",
        infer_dtype: str = "float32",
        max_length: int = 8192,
        padding_side: str = "left",
    ):
        self.device = torch.device(device)
        self.model_name = model_name
        self.infer_dtype = _torch_dtype_from_str(infer_dtype)
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side, use_fast=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=(self.infer_dtype if self.device.type == "cuda" else None),
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 16) -> torch.Tensor:
        n = len(texts)
        if n == 0:
            return torch.empty(0, 0)

        all_embs = []
        for i in range(0, n, batch_size):
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

            emb = last_token_pool(out.last_hidden_state, enc["attention_mask"])
            all_embs.append(emb.detach().cpu())

        return torch.cat(all_embs, dim=0)


def make_text_encoder(
    model_repo: str,
    device: str,
    infer_dtype: str,
    max_length: int,
) -> Any:
    # pick by repo or by known preset value
    if model_repo == TEXT_MODEL_PRESETS["nllb-200-3.3B"]:
        return NLLBTextEncoder(model_name=model_repo, device=device, infer_dtype=infer_dtype, max_length=max_length)
    if model_repo == TEXT_MODEL_PRESETS["bge-m3"]:
        return BGETextEncoder(model_name=model_repo, device=device, infer_dtype=infer_dtype, max_length=max_length)
    if model_repo == TEXT_MODEL_PRESETS["qwen3-embed-8b"]:
        # Qwen tends to be heavy; default batch_size differs inside encoder
        return Qwen3EmbeddingTextEncoder(model_name=model_repo, device=device, infer_dtype=infer_dtype, max_length=max_length)
    # Fallback: AutoModel + mean pooling
    return BGETextEncoder(model_name=model_repo, device=device, infer_dtype=infer_dtype, max_length=max_length)


def make_image_encoder(
    model_repo: str,
    device: str,
    infer_dtype: str,
    image_size: Optional[int],
) -> Any:
    if model_repo == IMAGE_MODEL_PRESETS["dinov3-vit7b16"]:
        return DinoV3ImageEncoder(model_name=model_repo, device=device, infer_dtype=infer_dtype, image_size=image_size)
    if model_repo == IMAGE_MODEL_PRESETS["clip-vit-bigg14"]:
        return CLIPImageEncoder(model_name=model_repo, device=device, infer_dtype=infer_dtype, image_size=image_size)
    if model_repo == IMAGE_MODEL_PRESETS["c-radio-v4-h"]:
        return RADIOImageEncoder(model_name=model_repo, device=device, infer_dtype=infer_dtype, image_size=image_size)
    # Fallback to DINO-like behavior
    return DinoV3ImageEncoder(model_name=model_repo, device=device, infer_dtype=infer_dtype, image_size=image_size)


# ----------------------------
# 4. Index builder (one-time)
# ----------------------------

def prepare_conceptual12m_index(
    wds_dir: str,
    cache_dir: str,
    pattern: str = "cc12m-train-*.tar",
    max_images: Optional[int] = None,
) -> str:
    """
    Streams WDS and writes:
      cache_dir/index/samples.jsonl
      cache_dir/index/meta.json

    Returns the samples.jsonl path.
    """
    index_dir = os.path.join(cache_dir, "index")
    _ensure_dir(index_dir)

    samples_path = os.path.join(index_dir, "samples.jsonl")
    meta_path = os.path.join(index_dir, "meta.json")

    if os.path.isfile(samples_path) and os.path.isfile(meta_path):
        print(f"[INDEX] Index already exists: {samples_path}")
        return samples_path

    print(f"[INDEX] Building CC12M index @ {index_dir}")
    print(f"[INDEX] wds_dir={wds_dir}, pattern={pattern}, max_images={max_images}")

    n = 0
    with open(samples_path, "w", encoding="utf-8") as f:
        for sample in tqdm(iter_cc12m_pixparse(wds_dir=wds_dir, pattern=pattern), desc="[INDEX] streaming", total=max_images):
            if max_images is not None and n >= max_images:
                break
            rec = {
                "sample_id": n,
                "key": sample["key"],
                "shard": sample["shard"],
                "shard_url": sample["shard_url"],
                "caption": sample["caption"],
                "language": "en",
                "meta": sample.get("meta", {}),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    meta = {
        "cache_version": CACHE_VERSION,
        "created_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "wds_dir": wds_dir,
        "pattern": pattern,
        "max_images": max_images,
        "n_samples": n,
        "caption_language": "en",
        "notes": "samples.jsonl is the authoritative ordering for all embedding caches.",
    }
    _atomic_write_json(meta_path, meta)
    print(f"[INDEX] Wrote {n} samples to {samples_path}")
    return samples_path


def _iter_samples_jsonl(samples_path: str) -> Iterator[Dict[str, Any]]:
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ----------------------------
# 5. Embedding builders (text / image)
# ----------------------------

def build_conceptual12m_text_embeddings(
    cache_dir: str,
    text_model: str,
    device: str = "cpu",
    infer_dtype: str = "float32",
    save_dtype: str = "float32",
    max_length: int = 256,
    txt_batch: int = 64,
    emb_chunk_size: int = 1_000_000,
    overwrite: bool = False,
) -> None:
    """
    Encodes captions from index/samples.jsonl with the chosen text model.
    """
    samples_path = os.path.join(cache_dir, "index", "samples.jsonl")
    if not os.path.isfile(samples_path):
        raise FileNotFoundError(
            f"Missing index file: {samples_path}\n"
            "Run cc12m_build_index.py first."
        )

    model_repo, model_tag = resolve_model_name(text_model, TEXT_MODEL_PRESETS)
    out_dir = os.path.join(cache_dir, "embeddings", "text", model_tag)
    _ensure_dir(out_dir)

    if _is_cache_done(out_dir) and not overwrite:
        print(f"[TEXT] Cache exists for model_tag={model_tag} @ {out_dir} — skipping (use --overwrite to rebuild)")
        return

    if os.path.exists(os.path.join(out_dir, "RUNNING")) and not overwrite:
        raise RuntimeError(f"[TEXT] Found RUNNING marker in {out_dir}. Use --overwrite or delete the folder.")

    with open(os.path.join(out_dir, "RUNNING"), "w", encoding="utf-8") as f:
        f.write("running\n")

    save_torch_dtype = _torch_dtype_from_str(save_dtype)
    text_encoder = make_text_encoder(model_repo, device=device, infer_dtype=infer_dtype, max_length=max_length)

    print(f"[TEXT] model={model_repo} (tag={model_tag})")
    print(f"[TEXT] infer_dtype={infer_dtype}, save_dtype={save_dtype}, max_length={max_length}")
    print(f"[TEXT] txt_batch={txt_batch}, emb_chunk_size={emb_chunk_size}")
    print(f"[TEXT] out_dir={out_dir}")

    chunk_tensors: List[torch.Tensor] = []
    chunk_count = 0
    chunk_id = 0
    batch_texts: List[str] = []
    total = 0

    def flush_chunk():
        nonlocal chunk_tensors, chunk_count, chunk_id
        if chunk_count == 0:
            return
        embs = torch.cat(chunk_tensors, dim=0)
        if embs.shape[0] != chunk_count:
            raise RuntimeError(f"[TEXT] Chunk mismatch: embs={embs.shape}, chunk_count={chunk_count}")
        p = os.path.join(out_dir, f"text_embs_{chunk_id:04d}.pt")
        torch.save(embs.to(save_torch_dtype), p)
        print(f"[TEXT][CACHE] Saved chunk {chunk_id:04d} with {chunk_count} embeddings -> {p}")
        chunk_tensors = []
        chunk_count = 0
        chunk_id += 1

    for rec in tqdm(_iter_samples_jsonl(samples_path), desc="[TEXT] encoding"):
        batch_texts.append(rec["caption"])
        if len(batch_texts) >= txt_batch:
            embs = text_encoder.encode_texts(batch_texts, batch_size=txt_batch).cpu()
            if embs.shape[0] != len(batch_texts):
                raise RuntimeError("[TEXT] Batch mismatch.")
            chunk_tensors.append(embs)
            chunk_count += len(batch_texts)
            total += len(batch_texts)
            batch_texts = []
            if chunk_count >= emb_chunk_size:
                flush_chunk()

    if batch_texts:
        embs = text_encoder.encode_texts(batch_texts, batch_size=txt_batch).cpu()
        chunk_tensors.append(embs)
        chunk_count += len(batch_texts)
        total += len(batch_texts)

    flush_chunk()

    shards = sorted(glob.glob(os.path.join(out_dir, "text_embs_*.pt")))
    meta = {
        "cache_version": CACHE_VERSION,
        "modality": "text",
        "model_repo": model_repo,
        "model_tag": model_tag,
        "device_used": str(device),
        "infer_dtype": infer_dtype,
        "save_dtype": save_dtype,
        "max_length": max_length,
        "txt_batch": txt_batch,
        "emb_chunk_size": int(emb_chunk_size),
        "n_samples": int(total),
        "shards": [os.path.basename(p) for p in shards],
        "index_samples_path": os.path.relpath(samples_path, cache_dir),
        "ordering": "matches index/samples.jsonl by sample_id",
    }
    _atomic_write_json(os.path.join(out_dir, "meta.json"), meta)
    _write_done(out_dir)
    os.remove(os.path.join(out_dir, "RUNNING"))
    print("[TEXT] Done.")


def build_conceptual12m_image_embeddings(
    wds_dir: str,
    cache_dir: str,
    image_model: str,
    device: str = "cpu",
    infer_dtype: str = "float32",
    save_dtype: str = "float32",
    img_batch: int = 32,
    pattern: str = "cc12m-train-*.tar",
    emb_chunk_size: int = 1_000_000,
    image_size: Optional[int] = None,
    max_images: Optional[int] = None,
    overwrite: bool = False,
) -> None:
    """
    Encodes images from WDS, enforcing the same ordering as index/samples.jsonl.
    """
    samples_path = os.path.join(cache_dir, "index", "samples.jsonl")
    if not os.path.isfile(samples_path):
        raise FileNotFoundError(
            f"Missing index file: {samples_path}\n"
            "Run cc12m_build_index.py first."
        )

    model_repo, model_tag = resolve_model_name(image_model, IMAGE_MODEL_PRESETS)
    out_dir = os.path.join(cache_dir, "embeddings", "image", model_tag)
    _ensure_dir(out_dir)

    if _is_cache_done(out_dir) and not overwrite:
        print(f"[IMAGE] Cache exists for model_tag={model_tag} @ {out_dir} — skipping (use --overwrite to rebuild)")
        return

    if os.path.exists(os.path.join(out_dir, "RUNNING")) and not overwrite:
        raise RuntimeError(f"[IMAGE] Found RUNNING marker in {out_dir}. Use --overwrite or delete the folder.")

    with open(os.path.join(out_dir, "RUNNING"), "w", encoding="utf-8") as f:
        f.write("running\n")

    save_torch_dtype = _torch_dtype_from_str(save_dtype)
    image_encoder = make_image_encoder(model_repo, device=device, infer_dtype=infer_dtype, image_size=image_size)

    print(f"[IMAGE] model={model_repo} (tag={model_tag})")
    print(f"[IMAGE] infer_dtype={infer_dtype}, save_dtype={save_dtype}, image_size={image_size}")
    print(f"[IMAGE] img_batch={img_batch}, emb_chunk_size={emb_chunk_size}")
    print(f"[IMAGE] wds_dir={wds_dir}, pattern={pattern}, max_images={max_images}")
    print(f"[IMAGE] out_dir={out_dir}")

    # iterate WDS + index in lockstep to guarantee identical sample_id ordering
    idx_iter = _iter_samples_jsonl(samples_path)
    wds_iter = iter_cc12m_pixparse(wds_dir=wds_dir, pattern=pattern)

    chunk_tensors: List[torch.Tensor] = []
    chunk_count = 0
    chunk_id = 0
    batch_images: List[Image.Image] = []
    total = 0

    def flush_chunk():
        nonlocal chunk_tensors, chunk_count, chunk_id
        if chunk_count == 0:
            return
        embs = torch.cat(chunk_tensors, dim=0)
        if embs.shape[0] != chunk_count:
            raise RuntimeError(f"[IMAGE] Chunk mismatch: embs={embs.shape}, chunk_count={chunk_count}")
        p = os.path.join(out_dir, f"image_embs_{chunk_id:04d}.pt")
        torch.save(embs.to(save_torch_dtype), p)
        print(f"[IMAGE][CACHE] Saved chunk {chunk_id:04d} with {chunk_count} embeddings -> {p}")
        chunk_tensors = []
        chunk_count = 0
        chunk_id += 1

    # We keep reading until either index runs out or max_images reached.
    for wds_sample in tqdm(wds_iter, desc="[IMAGE] streaming/encoding", total=max_images):
        if max_images is not None and total >= max_images:
            break

        try:
            idx_rec = next(idx_iter)
        except StopIteration:
            break

        # Sanity-check order
        if str(wds_sample["key"]) != str(idx_rec["key"]):
            raise RuntimeError(
                "[IMAGE] WDS stream order does not match index/samples.jsonl.\n"
                f"  got key={wds_sample['key']} from WDS, expected key={idx_rec['key']} from index.\n"
                "This usually means the index was built with a different wds_dir/pattern or filtering."
            )

        batch_images.append(wds_sample["image"])

        if len(batch_images) >= img_batch:
            embs = image_encoder.encode_images(batch_images).cpu()
            if embs.shape[0] != len(batch_images):
                raise RuntimeError("[IMAGE] Batch mismatch.")
            chunk_tensors.append(embs)
            chunk_count += len(batch_images)
            total += len(batch_images)
            batch_images = []
            if chunk_count >= emb_chunk_size:
                flush_chunk()

    # remaining
    if batch_images:
        embs = image_encoder.encode_images(batch_images).cpu()
        chunk_tensors.append(embs)
        chunk_count += len(batch_images)
        total += len(batch_images)

    flush_chunk()

    shards = sorted(glob.glob(os.path.join(out_dir, "image_embs_*.pt")))
    meta = {
        "cache_version": CACHE_VERSION,
        "modality": "image",
        "model_repo": model_repo,
        "model_tag": model_tag,
        "device_used": str(device),
        "infer_dtype": infer_dtype,
        "save_dtype": save_dtype,
        "image_size": image_size,
        "img_batch": img_batch,
        "emb_chunk_size": int(emb_chunk_size),
        "n_samples": int(total),
        "shards": [os.path.basename(p) for p in shards],
        "wds_dir": wds_dir,
        "pattern": pattern,
        "max_images": max_images,
        "index_samples_path": os.path.relpath(samples_path, cache_dir),
        "ordering": "matches index/samples.jsonl by sample_id",
    }
    _atomic_write_json(os.path.join(out_dir, "meta.json"), meta)
    _write_done(out_dir)
    os.remove(os.path.join(out_dir, "RUNNING"))
    print("[IMAGE] Done.")


# ----------------------------
# 6. Loaders (index + modality-specific + consolidated)
# ----------------------------

def load_conceptual12m_index(cache_dir: str) -> Dict[str, Any]:
    index_dir = os.path.join(cache_dir, "index")
    samples_path = os.path.join(index_dir, "samples.jsonl")
    meta_path = os.path.join(index_dir, "meta.json")
    if not os.path.isfile(samples_path) or not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing index files in {index_dir}. Run cc12m_build_index.py first.")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {"samples_path": samples_path, "meta": meta}


def _load_emb_meta(cache_dir: str, modality: str, model_tag: str) -> Dict[str, Any]:
    meta_path = os.path.join(cache_dir, "embeddings", modality, model_tag, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing embedding meta: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_emb_shards(cache_dir: str, modality: str, model_tag: str, prefix: str) -> List[str]:
    d = os.path.join(cache_dir, "embeddings", modality, model_tag)
    return sorted(glob.glob(os.path.join(d, f"{prefix}_*.pt")))


def load_conceptual12m_text_embeddings(
    cache_dir: str,
    text_model: str,
    device: str = "cpu",
    load_shards: bool = False,
) -> Dict[str, Any]:
    _, model_tag = resolve_model_name(text_model, TEXT_MODEL_PRESETS)
    meta = _load_emb_meta(cache_dir, "text", model_tag)
    shard_paths = _list_emb_shards(cache_dir, "text", model_tag, "text_embs")
    if not shard_paths:
        raise RuntimeError("No text embedding shards found.")
    shards = [torch.load(p, map_location=device) for p in shard_paths] if load_shards else None
    return {
        "model_tag": model_tag,
        "meta": meta,
        "shard_paths": shard_paths,
        "shards": shards,
    }


def load_conceptual12m_image_embeddings(
    cache_dir: str,
    image_model: str,
    device: str = "cpu",
    load_shards: bool = False,
) -> Dict[str, Any]:
    _, model_tag = resolve_model_name(image_model, IMAGE_MODEL_PRESETS)
    meta = _load_emb_meta(cache_dir, "image", model_tag)
    shard_paths = _list_emb_shards(cache_dir, "image", model_tag, "image_embs")
    if not shard_paths:
        raise RuntimeError("No image embedding shards found.")
    shards = [torch.load(p, map_location=device) for p in shard_paths] if load_shards else None
    return {
        "model_tag": model_tag,
        "meta": meta,
        "shard_paths": shard_paths,
        "shards": shards,
    }


def load_conceptual12m_embeddings(
    cache_dir: str,
    text_model: str,
    image_model: str,
    device: str = "cpu",
    load_shards: bool = False,
) -> Dict[str, Any]:
    """
    Convenience: load index + both modalities.
    """
    idx = load_conceptual12m_index(cache_dir)
    txt = load_conceptual12m_text_embeddings(cache_dir, text_model=text_model, device=device, load_shards=load_shards)
    img = load_conceptual12m_image_embeddings(cache_dir, image_model=image_model, device=device, load_shards=load_shards)
    return {
        "index": idx,
        "text": txt,
        "image": img,
    }


# ----------------------------
# 7. CLI (subcommands)
# ----------------------------

def _add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--cache_dir", required=True, help="Root cache directory")


def _add_device_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--device", default="cpu", help="cpu or cuda (e.g. cuda, cuda:0)")
    ap.add_argument("--infer_dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="Inference dtype")
    ap.add_argument("--save_dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="Save dtype for cached tensors (default=float32)")
    ap.add_argument("--emb_chunk_size", type=int, default=1_000_000, help="Max number of samples per embedding shard (.pt file)")
    ap.add_argument("--overwrite", action="store_true", help="Rebuild even if cache exists")


def parse_args():
    ap = argparse.ArgumentParser(description="CC12M (pixparse) indexing + embedding cache builder (v2)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_index = sub.add_parser("index", help="Build index/samples.jsonl (one-time)")
    _add_common_args(ap_index)
    ap_index.add_argument("--wds_dir", required=True, help="Directory containing cc12m-train-*.tar WebDataset shards")
    ap_index.add_argument("--pattern", type=str, default="cc12m-train-*.tar", help="Glob pattern inside wds_dir for shards")
    ap_index.add_argument("--max_images", type=int, default=None, help="Optional: limit to first N images")

    ap_text = sub.add_parser("text", help="Build text embeddings from index")
    _add_common_args(ap_text)
    _add_device_args(ap_text)
    ap_text.add_argument("--text_model", required=True, help=f"Preset key or HF repo. Presets: {sorted(TEXT_MODEL_PRESETS.keys())}")
    ap_text.add_argument("--max_length", type=int, default=256, help="Max token length (model-dependent)")
    ap_text.add_argument("--txt_batch", type=int, default=64, help="Batch size for text encoding")

    ap_img = sub.add_parser("image", help="Build image embeddings from WDS (checked against index order)")
    _add_common_args(ap_img)
    _add_device_args(ap_img)
    ap_img.add_argument("--wds_dir", required=True, help="Directory containing cc12m-train-*.tar WebDataset shards")
    ap_img.add_argument("--pattern", type=str, default="cc12m-train-*.tar", help="Glob pattern inside wds_dir for shards")
    ap_img.add_argument("--max_images", type=int, default=None, help="Optional: limit to first N images")
    ap_img.add_argument("--image_model", required=True, help=f"Preset key or HF repo. Presets: {sorted(IMAGE_MODEL_PRESETS.keys())}")
    ap_img.add_argument("--img_batch", type=int, default=32, help="Batch size for image encoding")
    ap_img.add_argument("--image_size", type=int, default=None, help="Optional fixed image size (resize shortest edge)")

    return ap.parse_args()


def main():
    args = parse_args()
    if args.cmd == "index":
        prepare_conceptual12m_index(
            wds_dir=args.wds_dir,
            cache_dir=args.cache_dir,
            pattern=args.pattern,
            max_images=args.max_images,
        )
        return

    if args.cmd == "text":
        build_conceptual12m_text_embeddings(
            cache_dir=args.cache_dir,
            text_model=args.text_model,
            device=args.device,
            infer_dtype=args.infer_dtype,
            save_dtype=args.save_dtype,
            max_length=args.max_length,
            txt_batch=args.txt_batch,
            emb_chunk_size=args.emb_chunk_size,
            overwrite=args.overwrite,
        )
        return

    if args.cmd == "image":
        build_conceptual12m_image_embeddings(
            wds_dir=args.wds_dir,
            cache_dir=args.cache_dir,
            image_model=args.image_model,
            device=args.device,
            infer_dtype=args.infer_dtype,
            save_dtype=args.save_dtype,
            img_batch=args.img_batch,
            pattern=args.pattern,
            emb_chunk_size=args.emb_chunk_size,
            image_size=args.image_size,
            max_images=args.max_images,
            overwrite=args.overwrite,
        )
        return


if __name__ == "__main__":
    main()
