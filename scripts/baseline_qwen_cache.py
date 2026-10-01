#!/usr/bin/env python3
import os
import json
import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from vllm import LLM, EngineArgs


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


def load_xm3600_records(
    captions_jsonl: str,
    images_dir: str,
    max_images: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load XM3600 into:
      [
        {
          "image_id": str,
          "image_path": str,
          "captions": {lang: [caption1, caption2, ...]}
        },
        ...
      ]
    """
    items = []

    with open(captions_jsonl, "r", encoding="utf-8") as f:
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
                if key.startswith("image/"):
                    continue
                if not isinstance(value, dict):
                    continue

                caps = value.get("caption") or value.get("captions")
                if isinstance(caps, str):
                    caps = [caps]
                elif isinstance(caps, list):
                    caps = [x for x in caps if isinstance(x, str) and x.strip()]
                else:
                    caps = []

                if caps:
                    captions[key] = caps

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

    return items


def format_input_to_conversation(
    text: Optional[str] = None,
    image_path: Optional[str] = None,
    instruction: str = "Represent the user's input for multilingual image-text retrieval.",
) -> List[Dict[str, Any]]:
    """
    Matches the model-card style:
      - system instruction in English
      - user content with text and/or image
    """
    content = []

    if image_path is not None:
        abs_image_path = os.path.abspath(image_path)
        content.append(
            {
                "type": "image",
                "image": "file://" + abs_image_path,
            }
        )

    if text is not None and len(text.strip()) > 0:
        content.append(
            {
                "type": "text",
                "text": text,
            }
        )

    if not content:
        content.append({"type": "text", "text": ""})

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]
    return conversation


def prepare_vllm_input(
    llm: LLM,
    text: Optional[str] = None,
    image_path: Optional[str] = None,
    instruction: str = "Represent the user's input for multilingual image-text retrieval.",
) -> Dict[str, Any]:
    conversation = format_input_to_conversation(
        text=text,
        image_path=image_path,
        instruction=instruction,
    )

    prompt_text = llm.llm_engine.tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    multi_modal_data = None
    if image_path is not None:
        abs_image_path = os.path.abspath(image_path)
        with Image.open(abs_image_path) as img:
            multi_modal_data = {"image": img.convert("RGB")}

    return {
        "prompt": prompt_text,
        "multi_modal_data": multi_modal_data,
    }


def build_indices(items: List[Dict[str, Any]]):
    image_index = []
    caption_index = []

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

    return image_index, caption_index


def embed_inputs(llm: LLM, inputs: List[Dict[str, Any]], desc: str) -> torch.Tensor:
    outputs = []
    for inp in tqdm(inputs, desc=desc):
        out = llm.embed([inp])[0]
        emb = out.outputs.embedding
        outputs.append(emb)
    return torch.tensor(np.array(outputs), dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--captions_jsonl",
        default="/mnt/data/shared/npa-multilingual/downloads/xm3600/captions.jsonl",
    )
    ap.add_argument(
        "--images_dir",
        default="/mnt/data/shared/npa-multilingual/downloads/xm3600/unpackedImages",
    )
    ap.add_argument(
        "--cache_dir",
        required=True,
    )
    ap.add_argument(
        "--model_path",
        default="Qwen/Qwen3-VL-Embedding-8B",
    )
    ap.add_argument(
        "--dtype",
        default="float32",
        choices=["bfloat16", "float16", "float32"],
    )
    ap.add_argument(
        "--instruction",
        default="Represent the user's input for multilingual image-text retrieval.",
    )
    ap.add_argument(
        "--max_images",
        type=int,
        default=None,
    )
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    print("[INFO] Loading XM3600...")
    items = load_xm3600_records(
        captions_jsonl=args.captions_jsonl,
        images_dir=args.images_dir,
        max_images=args.max_images,
    )
    print(f"[INFO] Loaded {len(items)} images")

    image_index, caption_index = build_indices(items)
    print(f"[INFO] Built {len(image_index)} image rows and {len(caption_index)} caption rows")

    with open(os.path.join(args.cache_dir, "image_index.json"), "w", encoding="utf-8") as f:
        json.dump(image_index, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.cache_dir, "caption_index.json"), "w", encoding="utf-8") as f:
        json.dump(caption_index, f, ensure_ascii=False, indent=2)

    print("[INFO] Loading Qwen3-VL-Embedding-8B with vLLM pooling runner...")
    engine_args = EngineArgs(
        model=args.model_path,
        runner="pooling",
        dtype=args.dtype,
        trust_remote_code=True,
    )
    llm = LLM(**vars(engine_args))

    print("[INFO] Preparing image inputs...")
    image_inputs = [
        prepare_vllm_input(
            llm=llm,
            text=None,
            image_path=it["image_path"],
            instruction=args.instruction,
        )
        for it in image_index
    ]

    print("[INFO] Preparing caption inputs...")
    caption_inputs = [
        prepare_vllm_input(
            llm=llm,
            text=rec["caption"],
            image_path=None,
            instruction=args.instruction,
        )
        for rec in caption_index
    ]

    print("[INFO] Embedding images...")
    image_embs = embed_inputs(llm, image_inputs, desc="[Qwen3-VL-Embedding] images")
    image_embs = l2_normalize(image_embs)

    print("[INFO] Embedding captions...")
    caption_embs = embed_inputs(llm, caption_inputs, desc="[Qwen3-VL-Embedding] captions")
    caption_embs = l2_normalize(caption_embs)

    torch.save(image_embs.half(), os.path.join(args.cache_dir, "image_embs.pt"))
    torch.save(caption_embs.half(), os.path.join(args.cache_dir, "caption_embs.pt"))

    meta = {
        "model_name": args.model_path,
        "model_type": "multimodal_embedding",
        "backend": "vllm_pooling_runner",
        "captions_source": "xm3600_source_caption",
        "instruction": args.instruction,
        "n_images": len(image_index),
        "n_captions": len(caption_index),
    }
    with open(os.path.join(args.cache_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[DONE] Cache written to:", args.cache_dir)


if __name__ == "__main__":
    main()