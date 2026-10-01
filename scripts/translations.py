#!/usr/bin/env python3
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

from google.cloud import translate_v3
from google.api_core.exceptions import InvalidArgument

def get_image_id(record: dict) -> Optional[str]:
    return record.get("image/key")


def chunk_list(items: List[str], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def load_processed_ids(output_path: Path) -> set:
    processed = set()
    if not output_path.exists():
        return processed

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                image_id = rec.get("image/key")
                if image_id:
                    processed.add(image_id)
            except Exception:
                continue
    return processed


def build_request(
    parent: str,
    model: str,
    texts: List[str],
    target_language: str,
    source_language: Optional[str],
) -> dict:
    req = {
        "parent": parent,
        "contents": texts,
        "mime_type": "text/plain",
        "target_language_code": target_language,
        "model": model,
    }
    if source_language:
        req["source_language_code"] = source_language
    return req


def translate_with_model(
    client: translate_v3.TranslationServiceClient,
    parent: str,
    model: str,
    texts: List[str],
    target_language: str = "en",
    source_language: Optional[str] = None,
    max_retries: int = 5,
    sleep_seconds: float = 2.0,
) -> List[str]:
    if not texts:
        return []

    req = build_request(
        parent=parent,
        model=model,
        texts=texts,
        target_language=target_language,
        source_language=source_language,
    )

    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.translate_text(request=req)
            return [t.translated_text for t in response.translations]
        except InvalidArgument:
            # Non-transient: surface immediately to caller.
            raise
        except Exception as e:
            last_err = e
            wait = sleep_seconds * (2 ** attempt)
            print(
                f"[WARN] transient failure for model={model}, source={source_language}, "
                f"batch_size={len(texts)}, attempt={attempt+1}/{max_retries}: {e}"
            )
            time.sleep(wait)

    raise RuntimeError(f"Translation failed after {max_retries} retries: {last_err}")


def translate_nmt_all(
    client: translate_v3.TranslationServiceClient,
    parent: str,
    texts: List[str],
    source_language: str,
    target_language: str,
    batch_size: int,
) -> List[str]:
    nmt_model = f"{parent}/models/general/nmt"
    out = []
    for batch in chunk_list(texts, batch_size):
        out.extend(
            translate_with_model(
                client=client,
                parent=parent,
                model=nmt_model,
                texts=batch,
                source_language=source_language,
                target_language=target_language,
            )
        )
    return out


def translate_tllm_if_supported(
    client: translate_v3.TranslationServiceClient,
    parent: str,
    texts: List[str],
    source_language: str,
    target_language: str,
    batch_size: int,
    tllm_support_cache: Dict[str, bool],
) -> List[str]:
    # If we already know this source language is unsupported, skip immediately.
    if source_language in tllm_support_cache and not tllm_support_cache[source_language]:
        return []

    tllm_model = f"{parent}/models/general/translation-llm"
    out = []

    try:
        for batch in chunk_list(texts, batch_size):
            out.extend(
                translate_with_model(
                    client=client,
                    parent=parent,
                    model=tllm_model,
                    texts=batch,
                    source_language=source_language,
                    target_language=target_language,
                )
            )
        tllm_support_cache[source_language] = True
        return out

    except InvalidArgument as e:
        msg = str(e)
        if "Unsupported language pair" in msg or "unsupported" in msg.lower():
            print(f"[INFO] TLLM unsupported for source={source_language}; writing empty array.")
            tllm_support_cache[source_language] = False
            return []
        raise


def process_record(
    record: dict,
    client: translate_v3.TranslationServiceClient,
    parent: str,
    batch_size: int,
    target_language: str,
    tllm_support_cache: Dict[str, bool],
) -> dict:
    image_id = record["image/key"]
    out = {"image/key": image_id}

    for lang, value in record.items():
        if lang.startswith("image/"):
            continue
        if not isinstance(value, dict):
            continue

        caps = value.get("caption") or value.get("captions")
        if not caps:
            continue

        if isinstance(caps, str):
            source_caps = [caps]
        else:
            source_caps = [c for c in caps if isinstance(c, str)]

        google_lang = lang

        # English -> English: just copy through for both.
        if google_lang == target_language:
            out[lang] = {
                "source_caption": source_caps,
                "translation_nmt": list(source_caps),
                "translation_tllm": list(source_caps),
            }
            continue

        translation_nmt = translate_nmt_all(
            client=client,
            parent=parent,
            texts=source_caps,
            source_language=google_lang,
            target_language=target_language,
            batch_size=batch_size,
        )

        translation_tllm = translate_tllm_if_supported(
            client=client,
            parent=parent,
            texts=source_caps,
            source_language=google_lang,
            target_language=target_language,
            batch_size=batch_size,
            tllm_support_cache=tllm_support_cache,
        )

        out[lang] = {
            "source_caption": source_caps,
            "translation_nmt": translation_nmt,
            "translation_tllm": translation_tllm,
        }

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--captions",
        default="/mnt/data/shared/npa-multilingual/downloads/xm3600/captions.jsonl",
        help="Path to XM3600 captions.jsonl",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output translated_captions.jsonl; defaults beside captions.jsonl",
    )
    parser.add_argument(
        "--project-id",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="GCP project id; defaults to GOOGLE_CLOUD_PROJECT",
    )
    parser.add_argument(
        "--location",
        default="global",
        help="Cloud Translation location",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Captions per translation request",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Debug limit",
    )
    parser.add_argument(
        "--target-language",
        default="en",
        help="Target language code",
    )
    args = parser.parse_args()

    if not args.project_id:
        raise ValueError("Set GOOGLE_CLOUD_PROJECT or pass --project-id")

    captions_path = Path(args.captions)
    if not captions_path.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_path}")

    output_path = (
        Path(args.output)
        if args.output
        else captions_path.parent / "translated_captions.jsonl"
    )

    client = translate_v3.TranslationServiceClient()
    parent = f"projects/{args.project_id}/locations/{args.location}"

    processed_ids = load_processed_ids(output_path)
    tllm_support_cache: Dict[str, bool] = {}

    print(f"[INFO] Input : {captions_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Parent: {parent}")
    print(f"[INFO] NMT  : general/nmt")
    print(f"[INFO] TLLM : general/translation-llm")
    print(f"[INFO] Batch: {args.batch_size}")

    num_seen = 0
    num_written = 0
    num_skipped = 0

    with captions_path.open("r", encoding="utf-8") as fin, \
         output_path.open("a", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            image_id = get_image_id(record)
            if not image_id:
                continue

            num_seen += 1

            if image_id in processed_ids:
                num_skipped += 1
                continue

            translated_record = process_record(
                record=record,
                client=client,
                parent=parent,
                batch_size=args.batch_size,
                target_language=args.target_language,
                tllm_support_cache=tllm_support_cache,
            )

            fout.write(json.dumps(translated_record, ensure_ascii=False) + "\n")
            fout.flush()

            num_written += 1
            if num_written % 25 == 0:
                print(
                    f"[INFO] written={num_written}, seen={num_seen}, skipped={num_skipped}"
                )

            if args.max_images is not None and num_written >= args.max_images:
                print("[INFO] Reached --max-images limit")
                break

    print(
        f"[DONE] seen={num_seen}, written={num_written}, skipped={num_skipped}, "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()