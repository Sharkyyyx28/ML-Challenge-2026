"""Stage 4: train the pair classifier on train candidates and tune the decision threshold.

S1 entities are split into 5 folds by row index mod 5 (source files are shuffled). The model
trains on `--train-folds`; the threshold is tuned for macro F0.5 on `--val-fold` after the
one-owner assignment. Validation predictions are saved for error_analysis.py.

    python train.py --train-folds 0,2 --val-fold 1 --model lgb
"""
import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import polars as pl

import model as M
from blocking_eval import truth_index
from config import N_JOBS, SEED, WORK_DIR, cand_path, norm_path
from decide import assign
from features import add_record_stats, context_features, feature_columns, pair_features, sim_context
from metrics import macro_f05

NORM_COLS = ["core", "compact", "name_n", "addr_n", "anums", "name_nums", "legal",
             "name_nonascii", "is_domain", "addr_missing"]
CHUNK = 2_000_000


def load_norm(split):
    n1 = pl.read_parquet(norm_path(split, 1), columns=NORM_COLS)
    n2 = pl.read_parquet(norm_path(split, 2), columns=NORM_COLS)
    n3 = pl.read_parquet(norm_path(split, 3), columns=NORM_COLS)
    n1, n2, n3 = add_record_stats(n1, n2, n3)
    return n1, pl.concat([n2, n3]), n2.height


def load_candidates(split, n1, no_all, n2_len):
    t = time.time()
    cand = sim_context(context_features(pl.read_parquet(cand_path(split))), n1, no_all, n2_len)
    print(f"{split} candidates {cand.height:,} with context features ({time.time() - t:.0f}s)", flush=True)
    return cand


def featurize(pairs, n1, no_all, n2_len, pool):
    parts = []
    for i in range(0, pairs.height, CHUNK):
        parts.append(pair_features(pairs[i:i + CHUNK], n1, no_all, n2_len, pool))
        print(f"    features {min(i + CHUNK, pairs.height):,}/{pairs.height:,}", flush=True)
    return pl.concat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-folds", default="0,2")
    ap.add_argument("--val-fold", type=int, default=1)
    ap.add_argument("--sample", type=float, default=0.75, help="fraction of train-fold S1s to use (memory)")
    ap.add_argument("--val-sample", type=float, default=0.5, help="fraction of val-fold S1s to use (memory)")
    ap.add_argument("--model", choices=["lgb", "xgb"], default="lgb")
    args = ap.parse_args()
    t0 = time.time()

    truth, n_s1 = truth_index()
    n1, no_all, n2_len = load_norm("train")
    cand = load_candidates("train", n1, no_all, n2_len).with_columns(fold=(pl.col("i1") % 5).cast(pl.Int8))
    cand = cand.join(truth.with_columns(label=pl.lit(1, pl.Int8)), on=["i1", "io", "src"], how="left") \
               .with_columns(pl.col("label").fill_null(0))
    train_folds = [int(f) for f in args.train_folds.split(",")]
    tr = cand.filter(pl.col("fold").is_in(train_folds))
    if args.sample < 1:
        keep = tr.select("i1").unique().sample(fraction=args.sample, seed=SEED)
        tr = tr.join(keep, on="i1", how="semi")
    val_s1 = pl.Series("i1", np.arange(args.val_fold, n_s1, 5, dtype=np.int32))
    if args.val_sample < 1:
        val_s1 = val_s1.sample(fraction=args.val_sample, seed=SEED)
    va = cand.filter(pl.col("i1").is_in(val_s1.implode()))
    del cand
    print(f"train pairs {tr.height:,} (pos {tr['label'].sum():,}) | val pairs {va.height:,} ({time.time()-t0:.0f}s)")

    with Pool(N_JOBS) as pool:
        tr = featurize(tr, n1, no_all, n2_len, pool)
        feats = [c for c in feature_columns(tr) if c != "label"]
        Xtr, ytr = M.as_matrix(tr, feats), tr["label"].to_numpy()
        del tr
        va = featurize(va, n1, no_all, n2_len, pool)
    del n1, no_all
    Xva, yva = M.as_matrix(va, feats), va["label"].to_numpy()
    print(f"{len(feats)} features, train matrix {Xtr.nbytes / 1e9:.1f} GB ({time.time()-t0:.0f}s)", flush=True)

    model = M.fit(args.model, Xtr, ytr, Xva, yva, feats)
    del Xtr, ytr
    model_file = M.save(args.model, model)
    print(f"trained ({time.time()-t0:.0f}s)", flush=True)

    va = va.with_columns(prob=pl.Series(M.predict(args.model, model, Xva)))
    del Xva
    val_truth = truth.filter(pl.col("i1").is_in(val_s1.implode()))
    best = (0, 0.5)
    for t in np.arange(0.3, 0.951, 0.025):
        f = macro_f05(assign(va, t), val_truth, val_s1)
        print(f"  thr {t:.3f}: macro F0.5 {f:.5f}")
        best = max(best, (f, float(t)))
    print(f"BEST val macro F0.5 {best[0]:.5f} at thr {best[1]:.3f} ({time.time()-t0:.0f}s)")

    va.write_parquet(WORK_DIR / "val_preds.parquet")
    pl.DataFrame({"i1": val_s1}).write_parquet(WORK_DIR / "val_s1.parquet")
    imp = M.importance(args.model, model, feats)
    print("top features:", [(f, round(g / 1e3)) for g, f in imp[:25]])
    json.dump({"backend": args.model, "model_file": model_file, "threshold": best[1], "val_f05": best[0],
               "features": feats, "best_iteration": M.best_iteration(args.model, model),
               "train_folds": train_folds, "val_fold": args.val_fold},
              open(WORK_DIR / "model_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
