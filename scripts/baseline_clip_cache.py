#!/usr/bin/env python3
import os
import json
import shutil
import argparse
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import open_clip

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


def load_translated_records(
    translated_jsonl: str,
    images_dir: str,
    text_source: str,
    max_images: Optional[int] = None,
) -> Tuple[List[Dict], List[str]]:
    """
    Read translated_captions.jsonl and build:
      items = [
        {
          "image_id": ...,
          "image_path": ...,
          "captions": {lang: [caption1, caption2, ...]}
        },
        ...
      ]

    text_source must be one of:
      - translation_nmt
      - translation_tllm
    """
    if text_source not in {"translation_nmt", "translation_tllm"}:
        raise ValueError(f"Unsupported text_source={text_source}")

    items = []
    langs = set()

    with open(translated_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            j = json.loads(line)
            image_id = j.get("image/key")
            if image_id is None:
                continue

            image_path = os.path.join(images_dir, f"{image_id}.jpg")
            if not os.path.isfile(image_path):
                continue

            captions = {}
            for key, value in j.items():
                if key == "image/key":
                    continue
                if not isinstance(value, dict):
                    continue

                texts = value.get(text_source, [])
                if isinstance(texts, str):
                    texts = [texts]
                elif isinstance(texts, list):
                    texts = [x for x in texts if isinstance(x, str) and x.strip()]
                else:
                    texts = []

                if len(texts) > 0:
                    captions[key] = texts
                    langs.add(key)

            if captions:
                items.append(
                    {
                        "image_id": image_id,
                        "image_path": image_path,
                        "captions": captions,
                    }
                )

            if max_images is not None and len(items) >= max_images:
                break

    return items, sorted(langs)


@torch.no_grad()
def encode_images(model, preprocess, image_paths, device, batch_size):
    outs = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="[CLIP-bigG] images"):
        batch = []
        for p in image_paths[i:i + batch_size]:
            with Image.open(p) as img:
                batch.append(preprocess(img.convert("RGB")))
        x = torch.stack(batch).to(device, non_blocking=True)
        z = model.encode_image(x)
        outs.append(z.float().cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size):
    outs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="[CLIP-bigG] texts"):
        toks = tokenizer(texts[i:i + batch_size]).to(device)
        z = model.encode_text(toks)
        outs.append(z.float().cpu())
    return torch.cat(outs, dim=0)


def maybe_reuse_image_cache(reuse_from: Optional[str], cache_dir: str) -> bool:
    if not reuse_from:
        return False

    src_image_embs = os.path.join(reuse_from, "image_embs.pt")
    src_image_index = os.path.join(reuse_from, "image_index.json")

    dst_image_embs = os.path.join(cache_dir, "image_embs.pt")
    dst_image_index = os.path.join(cache_dir, "image_index.json")

    if not (os.path.isfile(src_image_embs) and os.path.isfile(src_image_index)):
        raise FileNotFoundError(
            f"--reuse_image_cache_from={reuse_from} does not contain image_embs.pt and image_index.json"
        )

    shutil.copy2(src_image_embs, dst_image_embs)
    shutil.copy2(src_image_index, dst_image_index)
    print(f"[CACHE] Reused image cache from {reuse_from}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--translated_jsonl", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--cache_dir", required=True)

    ap.add_argument(
        "--text_source",
        required=True,
        choices=["translation_nmt", "translation_tllm"],
        help="Which translated caption field to cache",
    )

    ap.add_argument(
        "--model_name",
        default="hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--img_batch", type=int, default=32)
    ap.add_argument("--txt_batch", type=int, default=256)
    ap.add_argument("--save_dtype", default="float32", choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--max_images", type=int, default=None)

    ap.add_argument(
        "--reuse_image_cache_from",
        default=None,
        help="Optional existing cache dir to reuse image_embs.pt + image_index.json from",
    )

    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    device = torch.device(args.device)

    if args.save_dtype == "float16":
        save_dtype = torch.float16
    elif args.save_dtype == "bfloat16":
        save_dtype = torch.bfloat16
    else:
        save_dtype = torch.float32

    items, languages = load_translated_records(
        translated_jsonl=args.translated_jsonl,
        images_dir=args.images_dir,
        text_source=args.text_source,
        max_images=args.max_images,
    )

    if not items:
        raise ValueError("No items loaded for this text_source. Check translated_jsonl and images_dir.")

    print(f"[CACHE] Loaded {len(items)} images for text_source={args.text_source}")
    print(f"[CACHE] Languages present in this cache: {languages}")

    # ---------------- Build indices ----------------
    image_index = []
    caption_index = []
    all_texts = []

    for img_idx, it in enumerate(items):
        image_index.append(
            {
                "image_id": it["image_id"],
                "image_path": it["image_path"],
                "languages": list(it["captions"].keys()),
                "num_captions_per_lang": {k: len(v) for k, v in it["captions"].items()},
            }
        )

        for lang, caps in it["captions"].items():
            for cap in caps:
                caption_index.append(
                    {
                        "image_idx": img_idx,
                        "image_id": it["image_id"],
                        "language": lang,
                        "caption": cap,
                    }
                )
                all_texts.append(cap)

    n_images = len(image_index)
    n_captions = len(caption_index)

    print(f"[CACHE] Will encode/save {n_images} images and {n_captions} captions")

    # caption_index is always specific to the chosen text source
    with open(os.path.join(args.cache_dir, "caption_index.json"), "w", encoding="utf-8") as f:
        json.dump(caption_index, f, ensure_ascii=False, indent=2)

    # image_index is identical across sources if the same image set is loaded
    reused_images = maybe_reuse_image_cache(args.reuse_image_cache_from, args.cache_dir)
    if not reused_images:
        with open(os.path.join(args.cache_dir, "image_index.json"), "w", encoding="utf-8") as f:
            json.dump(image_index, f, ensure_ascii=False, indent=2)

    # ---------------- Load model ----------------
    model, preprocess = open_clip.create_model_from_pretrained(args.model_name)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model = model.to(device).eval()

    # ---------------- Encode or reuse images ----------------
    if reused_images:
        image_embs = torch.load(os.path.join(args.cache_dir, "image_embs.pt"), map_location="cpu").float()
        print(f"[CACHE] Loaded reused image_embs.pt with shape {tuple(image_embs.shape)}")
    else:
        image_embs = encode_images(
            model=model,
            preprocess=preprocess,
            image_paths=[x["image_path"] for x in items],
            device=device,
            batch_size=args.img_batch,
        )
        image_embs = l2_normalize(image_embs)
        torch.save(image_embs.to(save_dtype), os.path.join(args.cache_dir, "image_embs.pt"))
        print(f"[CACHE] Saved image_embs.pt with shape {tuple(image_embs.shape)}")

    # ---------------- Encode texts ----------------
    caption_embs = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=all_texts,
        device=device,
        batch_size=args.txt_batch,
    )
    caption_embs = l2_normalize(caption_embs)
    torch.save(caption_embs.to(save_dtype), os.path.join(args.cache_dir, "caption_embs.pt"))
    print(f"[CACHE] Saved caption_embs.pt with shape {tuple(caption_embs.shape)}")

    # ---------------- Meta ----------------
    meta = {
        "model_name": args.model_name,
        "text_source": args.text_source,
        "translated_jsonl": os.path.abspath(args.translated_jsonl),
        "images_dir": os.path.abspath(args.images_dir),
        "reuse_image_cache_from": (
            os.path.abspath(args.reuse_image_cache_from)
            if args.reuse_image_cache_from else None
        ),
        "n_images": n_images,
        "n_captions": n_captions,
        "languages": languages,
        "save_dtype": args.save_dtype,
    }

    with open(os.path.join(args.cache_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[CACHE] Wrote meta.json to {args.cache_dir}")
    print("[CACHE] Done.")


if __name__ == "__main__":
    main()