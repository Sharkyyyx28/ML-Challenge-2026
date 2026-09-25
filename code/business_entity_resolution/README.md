# Business Entity Resolution — Amazon ML Challenge 2026

Blocking + gradient-boosted pair classifier + one-owner assignment, tuned for macro F0.5.

## Pipeline

| Stage | Script | Output (in `work/` or `output/`) |
|---|---|---|
| 1. Normalize every record (transliterate Indic scripts, split domain names, legal-form classes, address numbers/abbreviations) | `src/prepare.py train test` | `norm_{split}_s{1,2,3}.parquet` |
| 2. Candidate generation: per-country TF-IDF over name words, name char 4-grams, address words, numbers, number×street; union of three sparse top-n searches per source — forward (top 10 per S1), name-only (top 5 per S1), reverse (top 3 S1 per S2/S3 record) | `src/blocking.py {train,test}` | `cand_{split}.parquet` |
| 2b. Blocking diagnostics (pair recall and oracle F0.5 of the union and of each channel) | `src/blocking_eval.py` | stdout |
| 3. Pair features (name/address similarity, legal-form conflict, number-set deltas, record-quality flags, blocking context incl. reverse rank) + LightGBM (CPU) or XGBoost (GPU); threshold tuned for macro F0.5 on a held-out S1 fold | `src/train.py --train-folds 0,2 --val-fold 1 --model lgb` | `model.txt`/`model.ubj`, `model_meta.json`, `val_preds.parquet` |
| 3b. Error analysis: cost of each error category, per-country F0.5, examples | `src/error_analysis.py` | stdout |
| 4. Score test, assign each S2/S3 record to at most one S1, write submission | `src/predict.py` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |

`candidate_pairs.tsv` is exactly the set of pairs the model scores; matches are always a subset.

## Running

```bash
pip install -r requirements.txt
cd src
python prepare.py train test
python blocking.py train
python blocking.py test
python train.py --train-folds 0,2 --val-fold 1 --model lgb     # or --model xgb (GPU)
python error_analysis.py
python predict.py
python ../../../student_resource/utils/validate_submission.py --matching ../../../output/matching_results.tsv --candidate ../../../output/candidate_pairs.tsv --test-dir ../../../student_resource/dataset/test
```

Paths default to `student_resource/dataset`, `work/`, `output/` at the project root; override with
`ER_DATA_DIR`, `ER_WORK_DIR`, `ER_OUT_DIR`, `ER_N_JOBS`. On Kaggle the data is found under `/kaggle/input` automatically.

**Memory:** full-size training needs ~20–30 GB RAM, so use Kaggle (CPU session, 30 GB). A 16 GB laptop
crashed at the training stage. Regenerate the single-file Kaggle notebook after any code change:
`python make_kaggle_notebook.py` → `kaggle/er_pipeline.ipynb`.

## Status (2026-09-25)

- Done: EDA (`eda/`, scripts + outputs), normalization, blocking, features, training, prediction; full
  pipeline verified end-to-end on a 20k-record slice (official validator: PASS).
- Blocking on full train (K per source → pair recall / oracle macro F0.5):
  K=5 0.901/0.963, **K=10 0.931/0.974**, K=20 0.949/0.981, K=30 0.957/0.984.
- v1 (forward blocking K=10, LightGBM on fold 0): validation macro F0.5 **0.9577**, public leaderboard
  **0.953** (first submission, rank ~250).
- v2 (this code): three-channel blocking union, LightGBM lr 0.1 with early stopping on folds 0+2,
  optional XGBoost-GPU backend, saved validation predictions + error analysis, parquet cache reuse on
  Kaggle. Verified end-to-end on the 20k slice (both backends, validator PASS with `--check-ids`);
  Full run: validation macro F0.5 **0.9675** (India 0.960, US 0.973), threshold 0.725.
  Error analysis (F0.5 if the category alone were fixed): true pair missing from candidates +0.0174,
  true pair below threshold +0.0101 (mostly bare-name S2/S3 records without address), wrong merge
  with a distractor +0.0037, wrong merge with a record owned by another S1 +0.0021.
- v3: Indic back-transliteration, forward K=20 + reverse K=5, XGBoost GPU.
- v4 (this code, same normalization/blocking cache as v3): record rarity counts (how many S1s share
  this exact name / address, rarest-name-token frequency) and similarity-aware competition features
  (name/address similarity of this pair vs. every other S1 competing for the same record, and vs. the
  S1's other candidates) — targets the bare-name misses and the same-address wrong merges.

## Key data findings (details in `eda/*_out.txt`)

- Every S2/S3 record belongs to at most one S1 → one-owner assignment. 26% of S2/S3 are distractors;
  only 5.6% of S1 are singletons; mean 3.5 matches per S1.
- Deliberate "twin" distractors: same name/street/city, different legal form and/or a house/unit number
  shifted by 1–20. Same-name pairs: no legal conflict + same number → 99.9% match; legal conflict +
  number shifted 1–20 → 0.1–0.3%. True matches have digit typos (dropped/inserted digits → large delta).
- ~24% of Indian S2/S3 names are in Devanagari/Tamil/Telugu/Gujarati/Gurmukhi; S1 is always Latin.
- Test adds France (15%, unseen in train; SARL/SAS/EURL/SCI legal forms, French street abbreviations)
  and shifts the country mix (train 60% US/40% India; test 38% US/47% India/15% France).

## Next steps

1. Run v4 on the v3 cache → compare validation F0.5 against v2 (0.9675) / v3.
2. Blocking misses (largest cost): acronym key (`jemie snow generating` ↔ `jsg`), phonetic/skeleton
   keys for heavily misspelled names, house-number prefix/suffix truncation (`1661` ↔ `661`).
3. Second-stage model on the S1 side (its matches in the other source, how many strong matches it
   already has), trained with cross-fitting.
4. Fine-tuned multilingual bi-encoder as an extra blocking channel + feature (the hybrid step).
