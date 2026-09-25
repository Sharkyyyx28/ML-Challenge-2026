# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary

We resolve Source 2/3 records against the Source 1 reference list with a three-stage pipeline:
IDF-weighted sparse candidate generation per country, a LightGBM pair classifier over 48
country-agnostic similarity and "twin-detector" features, and a one-owner assignment that gives each
Source 2/3 record to at most one Source 1 entity. The key insight from EDA is that the ground truth is a
partition (no S2/S3 record belongs to two S1 entities) and that many distractors are deliberately
constructed near-duplicates of real businesses, separable mainly by legal form and small shifts in
house/unit numbers. Validation macro F0.5: **0.9577** (held-out S1 fold, train distribution).

---

## 2. Methodology

### 2.1 Problem Analysis

Scale: train 2.21M S1 / 5.03M S2 / 5.29M S3 records; test 1.73M / 4.89M / 5.08M.

- **Match structure.** Every S2/S3 record is matched to at most one S1 entity (checked over all 7.64M
  positive pairs). 26% of S2/S3 records match nothing (distractors). Only 5.6% of S1 entities are
  singletons; the mean is 3.5 matches per S1 (0–11). Statistics are near-identical for US and India.
- **Twin distractors.** Many unmatched records copy a real S1 business's name, street and city but
  change the legal form and/or shift one house or unit number by a small amount
  (e.g. `Orthopedic Interstate Partners LLC, 800 Park Street` vs `... Ltd, 807 PARK ST`). On pairs
  sharing a normalized name and street: no legal-form conflict + identical number → 99.9% true match;
  legal-form conflict + number shifted by 1–20 → 0.1–0.3%. Twins sometimes shift the *second* number
  (`475 6` vs `475 4`), so all numbers must be compared. True matches also carry number typos, but these
  usually drop/insert digits (`970` vs `9709`), giving large numeric deltas.
- **Name noise.** ~24% of Indian S2/S3 names (and addresses) are written in Devanagari, Tamil, Telugu,
  Gujarati or Gurmukhi while S1 is always Latin; names appear as domains (`gallocation.com`), with
  DBA/trade names, reordered words, duplicated words, injected diacritics and ID junk (`(ID: 55428)`).
  30% of S1 names are shared with another S1 entity (e.g. "meridian" ×560), so names alone are ambiguous.
- **Address noise.** Abbreviations, reordered components, state names in native script, zero-padded
  numbers (`T-192` vs `T-00192`), injected house numbers, district swaps; ~3% empty and ~3% literal
  "null"/"None" addresses in S2/S3.
- **Test shift.** Test adds France (15% of S1, unseen in training; legal forms SARL/SAS/SASU/EURL/SCI/SNC,
  abbreviations `R.`/`Pl.`/`Q.`/`N°`) and changes the mix (train 60% US / 40% India; test 38% US /
  47% India / 15% France). Test also has more S2/S3 records per S1 (5.75 vs 4.67).
- No leakage: row order is uncorrelated between sources (|r| < 0.001).

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier + constrained (one-owner) assignment  
**Core Innovation:** Treating the task as assigning each S2/S3 record to one owner, with explicit
twin-detector features (legal-form agreement, number-set deltas and digit edit distance) and
"reverse-view" blocking-context features that compare a candidate S1 against every other S1 that
retrieved the same record.

---

## 3. Candidate Generation (Blocking)

Each record is normalized (transliteration of Indic scripts to Latin with `anyascii`, domain names split
to words, legal forms mapped to classes and removed from the name core, repeated tokens dropped,
country-specific address abbreviation tables, US state names to codes, numbers stripped of leading zeros).
Records are then hashed into a sparse bag of features:

- **Blocking keys used:** name-core words; character 4-grams of the space-free name core (robust to typos
  and domain-style names); address words; address numbers; number×street-word combinations (rare,
  precise); numbers inside the name. Features are IDF-weighted per country over S1+S2+S3, features in
  more than 3,000 records or in only one record are dropped, and rows are L2-normalized.
- **Search:** per country label (open set; an unseen label searches the full pool), multithreaded sparse
  top-n matrix product (`sparse_dot_topn`) keeps the top K=10 S2 and top K=10 S3 records per S1.
- **Candidate pairs generated:** 44.1M on train, 34.6M on test (20 per S1 entity).
- **How you ensured true matches were not lost:** recall measured on the full train set at several K:

| K per source | Pair recall | Oracle macro F0.5 (perfect matcher) | Candidates per S1 |
|---|---|---|---|
| 3 | 0.834 | 0.946 | 6 |
| 5 | 0.901 | 0.963 | 10 |
| **10 (used)** | **0.931** | **0.974** | 20 |
| 20 | 0.949 | 0.981 | 40 |
| 30 | 0.957 | 0.984 | 60 |

Remaining misses are dominated by same-name records with a missing or very different address, which
lose to twins/lookalikes that share both name and address (a name-only channel is planned).

---

## 4. Matching Model

**Features used (48, none encode the country identity):**
- Name features: rapidfuzz ratio, token-set, token-sort and partial ratios on the name core; Jaro-Winkler
  and partial ratio on the space-free core; ratio on the full normalized name; token Jaccard; first-token
  equality; name lengths.
- Legal-form features: same class, conflicting classes, one side missing, both missing, exact match.
- Address features: ratio, token-set and partial ratios, token Jaccard, token counts, number-string ratio.
- Number-set features: set sizes, intersection, unmatched numbers on each side, first-number equality
  and delta, minimum numeric delta and digit edit distance between unmatched numbers, name-number conflict.
- Record-quality flags: non-Latin name, domain-style name, missing address (both sides).
- Blocking context: similarity score, rank, per-S1 best score and gap, and the reverse view per S2/S3
  record (number of S1 entities that retrieved it, its rank among them, best score, gap and margin).

**Model type:** LightGBM binary classifier (MIT licence), 127 leaves, learning rate 0.05, 2,000 rounds,
feature/bagging fraction 0.8. Trained on the candidate pairs of one S1 fold (8.83M pairs, 1.42M
positive); S1 entities are split into 5 folds and validation uses a disjoint fold (8.83M pairs).

**Threshold selection method:** each S2/S3 record is assigned to its highest-probability S1 candidate
only if that probability clears a threshold; the threshold is swept on the validation fold to maximise
macro F0.5 exactly as defined by the challenge (singletons included). Selected threshold: 0.70.

Most important features by gain: `o_margin` (reverse-view margin), `num_min_delta` (twin detector),
`o_gap`, address token-set ratio, `o_rank`, name ratios, unmatched-number counts, digit edit distance.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.9577 on the validation fold (train distribution: US + India).
  Blocking ceiling at K=10 is 0.974, so ~0.016 is lost in matching and ~0.026 to blocking recall.
  Public leaderboard (first submission, 2026-09-25): **0.953**. The small gap to validation
  (0.005) indicates the country-agnostic features transfer to test, including unseen France.
  Test output: 5.54M matches (3.2 per S1, 6.2% empty lists), consistent with train
  (3.46 true matches per S1 × 93% blocking recall).
- **Common false positives (wrong merges):** [pending — per-type analysis of validation predictions]
- **Common false negatives (missed matches):** blocking misses are mainly same-name records whose address
  is missing or differs strongly (e.g. `patriot alliance | 5101 retreat hill way` vs `patriot aaince |`
  with no address), fully transliterated names with short addresses, and same-name records whose
  number is shifted like a twin. [matching-stage analysis pending]

---

## 6. Conclusion

A partition-aware assignment on top of a feature-rich pair classifier reaches 0.958 macro F0.5 on
validation, with the largest remaining loss in candidate recall. Next steps: a name-only and a reverse
(S2/S3 → S1) blocking channel, leave-one-country-out validation as a proxy for France, and a fine-tuned
multilingual bi-encoder as an additional blocking channel and feature for transliterated names.

**Fair play.** Only the provided data is used. Unlabeled records (train and test) are used to compute
IDF weights for candidate search; transliteration uses the static `anyascii` character table; all
abbreviation, legal-form and US-state tables are hand-written in `src/normalize.py`. No external
databases, APIs or geocoding are used.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `src/config.py` — paths (env-overridable; auto-detects Kaggle), seeds
- `src/normalize.py` — record normalization
- `src/prepare.py` — stage 1: normalize all records → parquet
- `src/blocking.py`, `src/blocking_eval.py` — stage 2: candidate generation and recall diagnostics
- `src/features.py` — stage 3: pair features
- `src/train.py`, `src/metrics.py`, `src/decide.py` — stage 4: training, macro F0.5, one-owner assignment
- `src/predict.py` — stage 5: test scoring, writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`
- `make_kaggle_notebook.py` — builds the single-file Kaggle notebook used for full-size runs (30 GB RAM, 4 CPU)
- `README.md`, `requirements.txt` — exact run instructions and pinned versions

Entry point order: `prepare.py train test` → `blocking.py train 10` → `blocking.py test 10` →
`train.py --k 10 --train-folds 0 --val-fold 1` → `predict.py`.

### B. Additional Results

EDA scripts and their outputs are in `eda/` (file profiles, ground-truth structure, pair similarity,
twin signals, France and blocking feasibility).
