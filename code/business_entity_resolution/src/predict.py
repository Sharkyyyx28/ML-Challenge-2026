"""Stage 5: score test candidates, apply the one-owner assignment, write both output files.

    python predict.py            # uses the model + threshold recorded in work/model_meta.json
"""
import json
import time
from multiprocessing import Pool

import numpy as np
import polars as pl

import model as M
from config import N_JOBS, OUT_DIR, WORK_DIR, norm_path
from decide import assign
from train import featurize, load_candidates, load_norm


def write_lists(path, col, s1_ids, pairs, ids2, ids3):
    """pairs: (i1, io, src) -> one row per S1 entity with comma-joined S2/S3 ids."""
    named = pl.concat([
        pairs.filter(pl.col("src") == s).join(ids.rename({"entity_id": "oid"}), on="io").select("i1", "oid")
        for s, ids in ((2, ids2), (3, ids3))
    ])
    lists = named.unique().sort("oid").group_by("i1").agg(pl.col("oid").str.join(","))
    out = (s1_ids.join(lists, on="i1", how="left")
           .select(pl.col("entity_id").alias("source1_entity_id"), pl.col("oid").fill_null("").alias(col)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never")
    return out


def main():
    t0 = time.time()
    meta = json.load(open(WORK_DIR / "model_meta.json"))
    backend = meta["backend"]
    model = M.load(backend)
    n1, no_all, n2_len = load_norm("test")
    cand = load_candidates("test", n1, no_all, n2_len)
    print(f"test candidate pairs {cand.height:,} ({time.time()-t0:.0f}s)")
    probs = []
    with Pool(N_JOBS) as pool:
        for i in range(0, cand.height, 4_000_000):
            part = featurize(cand[i:i + 4_000_000], n1, no_all, n2_len, pool)
            probs.append(M.predict(backend, model, M.as_matrix(part, meta["features"])))
            print(f"  scored {min(i + 4_000_000, cand.height):,}/{cand.height:,} ({time.time()-t0:.0f}s)", flush=True)
    del n1, no_all
    cand = cand.select("i1", "io", "src").with_columns(prob=pl.Series(np.concatenate(probs)))
    kept = assign(cand, meta["threshold"])
    print(f"kept {kept.height:,} matches at threshold {meta['threshold']:.3f} ({time.time()-t0:.0f}s)")

    s1_ids = pl.read_parquet(norm_path("test", 1), columns=["entity_id"]).with_row_index("i1") \
               .with_columns(pl.col("i1").cast(pl.Int32))
    ids2 = pl.read_parquet(norm_path("test", 2), columns=["entity_id"]).with_row_index("io").with_columns(pl.col("io").cast(pl.Int32))
    ids3 = pl.read_parquet(norm_path("test", 3), columns=["entity_id"]).with_row_index("io").with_columns(pl.col("io").cast(pl.Int32))
    write_lists(OUT_DIR / "candidate_pairs.tsv", "candidate_entity_ids", s1_ids, cand, ids2, ids3)
    m = write_lists(OUT_DIR / "matching_results.tsv", "matched_entity_ids", s1_ids, kept, ids2, ids3)
    print(f"wrote {m.height:,} rows; entities with >=1 match: {(m['matched_entity_ids'] != '').sum():,} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
