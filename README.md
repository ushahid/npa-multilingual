Training free Cross-Modal Alignment for Multilingual Image Retrieval, Shahid et al, AACL-IJCNLP 2026

## Setup

Run the commands below in Bash from the repository root, with a CUDA-compatible PyTorch/torchvision installation.

```bash
python -m pip install transformers huggingface_hub accelerate webdataset pillow numpy tqdm pandas matplotlib sentencepiece protobuf timm einops
# Optional: CLIP translation baselines, Qwen-VL baseline, Excel export (respectively).
python -m pip install open_clip_torch google-cloud-translate
python -m pip install vllm
python -m pip install openpyxl
```

Original package versions were not saved; this is not a pinned environment.

- **Hugging Face:** authenticate with `hf auth login` or `HF_TOKEN` for [DINOv3 access](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m). `HF_HOME` optionally changes the model-cache location.
- **Google Cloud, translation baselines only:** enable Cloud Translation and configure [Application Default Credentials](https://docs.cloud.google.com/translate/docs/authentication), using `gcloud auth application-default login` or `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json`. Set `GOOGLE_CLOUD_PROJECT` or pass `--project-id`. This client uses credentials, not a plain API key.
- Cached retrieval and local CLIP/Qwen inference need no hosted inference API keys. Do not commit tokens or service-account files.

## Data

| Required input | Location used below |
| --- | --- |
| [XM3600 captions](https://google.github.io/crossmodal-3600/) | `downloads/xm3600/captions.jsonl` |
| XM3600 images, extracted with original filenames | `downloads/xm3600/unpackedImages/` |
| [CC12M WebDataset shards](https://huggingface.co/datasets/pixparse/cc12m-wds), downloaded by the script below | `downloads/conceptual12m/cc12m_pixparse_wds/cc12m-train-*.tar` |

Datasets, model weights, translations, and embedding caches are not included. `top2.jsonl` contains two example records, not the full evaluation set. Change the paths below if your data lives elsewhere.

```bash
XM=downloads/xm3600
CC=downloads/conceptual12m
python scripts/download_hf_pixparse.py --local_dir "$CC/cc12m_pixparse_wds"
```

## Run

**1. Build indexes and embeddings.** This calls `cc12m_dataset.py` and `xm3600_dataset.py`, each in `index`, `text`, `image` order. Defaults build all three text and three image models on the first 1.6M CC12M pairs and all XM3600 captions/images.

```bash
python scripts/run_all_embs_and_sims.py \
  --cc_wds_dir "$CC/cc12m_pixparse_wds" --cc_cache_dir "$CC/caches" \
  --xm_captions "$XM/captions.jsonl" --xm_images_dir "$XM/unpackedImages" \
  --xm_cache_dir "$XM/caches" --plots_out_root plots/similarities \
  --device cuda:0 --infer_dtype float32 --save_dtype float32 --skip_plots
```

For one pair, add `--text_models qwen3-embed-8b --image_models c-radio-v4-h`. Other text choices are `bge-m3` and `nllb-200-3.3B`; other image choices are `clip-vit-bigg14` and `dinov3-vit7b16`. Batch sizes can be reduced with `--txt_batch_override` and `--img_batch_override`; `--image_size` defaults to 304.

Caches contain `index/` and `embeddings/{text,image}/<model_tag>/` with metadata, tensor shards, and `DONE` markers. Keep the same CC12M index order for both modalities. Completed embeddings are reused; rebuild them if preprocessing or model weights change.

**2. Retrieve.** Example for one encoder pair:

```bash
python scripts/xm3600_rr_retrieval.py \
  --xm_cache "$XM/caches" --cc_cache "$CC/caches" --out_dir outputs/rr_example \
  --text_model qwen3-embed-8b --image_model c-radio-v4-h \
  --anchor_n 1600000 --k 800 --p 4 --proc asif --dtype float32 --device cuda:0
```

For the full 135-run sweep, build all six encoder caches first:

```bash
python scripts/run_all_retrieval_combos.py \
  --xm_cache "$XM/caches" --cc_cache "$CC/caches" --out_root outputs/rr_sweep \
  --dtype float32 --proc asif --device cuda:0 --stop_on_error
```

The grid is nine encoder pairs, k = 200/400/800/1200/1600, and p = 2/4/8. All captions are evaluated unless `--max_queries` or `--one_per_image_per_lang` is set on the single-run script. Use a fresh output directory when changing a run's settings; the sweep skips nonempty run directories by default.

**3. Export tables and plot.**

```bash
python scripts/summarise_into_tables.py --mode rr --root outputs/rr_sweep
python scripts/plots.py --csv outputs/rr_sweep/rr_sweep_per_language.csv --out_dir plots/retrieval
```

The exporter writes `rr_sweep_summary.csv` and `rr_sweep_per_language.csv` into the run directory; add `--xlsx` for Excel. For optional query/anchor similarity plots, use `cc12m_xm3600_sims.py --help`; `check_dims.py` checks the C-RADIO output dimension with a dummy image.

## Ablations

Run after building the Qwen3-Embed-8B and C-RADIO caches:

```bash
# Random subsets: calls xm3600_rr_ablation.py, which loads xm3600_rr_retrieval.py.
python scripts/sampling_ablations.py \
  --xm_cache "$XM/caches" --cc_cache "$CC/caches" --out_root outputs/ablations/sampling \
  --device cuda:0 --repeats 10 --seed_start 0 --stop_on_error
# Separate first-N experiment: calls xm3600_rr_retrieval.py.
python scripts/anchor_percentage_ablations.py \
  --xm_cache "$XM/caches" --cc_cache "$CC/caches" --out_root outputs/ablations/first_n \
  --device cuda:0 --percentages 100 90 80 70 60 50 --stop_on_error
```

Both use k=800, p=4, float32, and a 1.6M base bank. Random sampling defaults to 1/2.5/5/7.5/10/15/20/25/30/40/50/75/100%. It generates `indices/*.pt` and summaries under `--out_root`. The 100% bank runs once by default; `--repeat_full_percentage` repeats it. Direct use of `xm3600_rr_ablation.py --anchor_indices_path` requires a supplied indices file.

## Baselines

Translation credentials are needed only for the first command. NMT and TLLM translations share one output file; CLIP image embeddings can be reused.

```bash
B="$XM/baselines"
python scripts/translations.py --captions "$XM/captions.jsonl" --output "$XM/translated_captions.jsonl"
python scripts/baseline_clip_cache.py \
  --translated_jsonl "$XM/translated_captions.jsonl" --images_dir "$XM/unpackedImages" \
  --text_source translation_nmt --cache_dir "$B/clip_nmt"
python scripts/baseline_clip_cache.py \
  --translated_jsonl "$XM/translated_captions.jsonl" --images_dir "$XM/unpackedImages" \
  --text_source translation_tllm --cache_dir "$B/clip_tllm" --reuse_image_cache_from "$B/clip_nmt"
python scripts/baseline_qwen_cache.py \
  --captions_jsonl "$XM/captions.jsonl" --images_dir "$XM/unpackedImages" --cache_dir "$B/qwen_vl"
for model in clip_nmt clip_tllm qwen_vl; do
  python scripts/baseline_rr.py --cache_dir "$B/$model" --out_dir "outputs/baselines/$model" --dtype float32
done
python scripts/summarise_into_tables.py --out_csv outputs/baselines/baseline_per_language.csv
```

`baseline_rr.py` performs dense cosine retrieval. Qwen-VL uses original multilingual captions and does not need translations.
