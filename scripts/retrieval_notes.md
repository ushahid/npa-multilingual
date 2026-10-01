# XM3600 RR Retrieval Procedure — Notes

**Purpose**: Document exactly how the caption → image retrieval works in `xm3600_rr_retrieval.py`.

**Script**: `xm3600_rr_retrieval.py`

### Goal
Evaluate how well XM3600 multilingual captions retrieve their matching image using ASIF-style relative representations, with CC12M (first N pairs) as the fixed anchor set. No training involved.

### High-Level Flow
1. Load XM3600 embeddings (images + captions) and CC12M anchor embeddings.
2. Compute relative representations (RR) for all XM3600 **images** w.r.t. CC12M image anchors → save sparse top-k vectors + build inverted posting index (for fast lookup).
3. For each selected XM3600 **caption**:
   - Compute its RR w.r.t. CC12M text anchors (same top-k + processing).
   - Score every XM image using sparse dot product in RR space.
   - Rank images → compute Recall@K.
4. Output: `results.json` with overall + per-language R@K.

### Key Design Choices & Reasoning

**Anchor set (`--anchor_n`, default 1.6M)**  
- We take the first N English image-caption pairs from CC12M.  

**Sparsity (`--k`, default 800)**  
- Keep only the top-800 most similar anchors per RR vector.  
- Why? Matches ASIF paper. Removes weak/noisy similarities that hurt when summing/dot-producting. Makes everything sparse, interpretable (each dimension = one specific CC12M pair), and much faster.

**RR processing (`--proc` + `--p`)**  
- **asif** (default): raise top-k values to power p (default 8), then L2-normalize.  
- **norm**: just L2-normalize (no exponent).  
- **none**: raw top-k values.  
- Why asif/p=8? Exact match to paper (exponentiation + sparsification). p=8 emphasizes strongest matches and worked well in paper's tuning. We default to this for main results.

**Scoring**  
- Sparse dot product between caption RR and image RR.  
- Implemented via inverted posting index (anchor → list of images + their values) for speed.  
- This is the efficient way ASIF recommends for large anchor sets.

**Query selection options**  
- `--one_per_image_per_lang`: use only first caption per (image, lang) → avoids bias from images with many captions in one language.  
- `--languages`: filter to specific langs (e.g. en,hi,fr).  
- `--max_queries`: cap total captions (for fast debugging).  
- Why? Makes results fairer and easier to interpret across languages.

**Efficiency & Reproducibility**  
- `--reuse_cache`: re-uses precomputed image RR + posting index when config matches. Recommended after first run.  
- Image RR is computed once and cached (expensive part).  
- Caption RR is computed on-the-fly per batch (cheap).  
- All random seeds are set; config is saved in `run_config.json`.

### Verification
- Check that image RR cache matches current `--anchor_n`, `--k`, `--proc`, `--p`.
- Verify `results.json` has both "overall" and "per_language" sections.
- Spot-check a few queries: does the correct image rank high for its own captions?
- Compare R@1 for English vs non-English — expect English higher but non-English still decent (cross-lingual transfer test).
- If you change anchors or k/p, delete the cached `.pt` files so it recomputes.

**Default run for main results**:
```bash
python ./scripts/xm3600_rr_retrieval.py --xm_cache /mnt/data/shared/npa-multilingual/downloads/xm3600/cache_dinov3_nllb --cc_cache /mnt/data/shared/npa-multilingual/downloads/conceptual12m/cache_dinov3_nllb_first_1_6mil --out_dir /mnt/data/shared/npa-multilingual/outputs/xm3600_rr_dinov3_nllb --proc asif --p 8 --k 800 --anchor_n 1600000 --device cuda:0 --dtype float16
```

**For Raw results**:
```bash
python ./scripts/xm3600_rr_retrieval.py --xm_cache /mnt/data/shared/npa-multilingual/downloads/xm3600/cache_dinov3_nllb --cc_cache /mnt/data/shared/npa-multilingual/downloads/conceptual12m/cache_dinov3_nllb_first_1_6mil --out_dir /mnt/data/shared/npa-multilingual/outputs/xm3600_rr_dinov3_nllb_raw --proc none --k 800 --anchor_n 1600000 --device cuda:0 --dtype float16
```

**Pipeline**
```bash
python /mnt/data/shared/npa-multilingual/scripts/run_models.py \
  --scripts_dir /mnt/data/shared/npa-multilingual/scripts \
  --xm_captions /mnt/data/shared/npa-multilingual/downloads/xm3600/captions.jsonl \
  --xm_images /mnt/data/shared/npa-multilingual/downloads/xm3600/unpackedImages \
  --cc_wds_dir /mnt/data/shared/npa-multilingual/downloads/conceptual12m/webdataset \
  --xm_cache_root /mnt/data/shared/npa-multilingual/downloads/xm3600/caches \
  --cc_cache_root /mnt/data/shared/npa-multilingual/downloads/conceptual12m/caches \
  --out_root /mnt/data/shared/npa-multilingual/outputs \
  --hf_home /mnt/data/shared/hf_cache \
  --device cuda:0 \
  --dtype float32 \
  --save_dtype float32 \
  --anchor_n 1600000 \
  --proc norm \
  --k 1600 \
  --p 2 \
  --do_sims \
  --xm_image_idx 0 100 200 300 400 500 \
  --anchor_language en hi fr \
  --bins 100 \
  --skip_existing
```

**Tmux session**
```bash
tmux attach -t cc12m
```