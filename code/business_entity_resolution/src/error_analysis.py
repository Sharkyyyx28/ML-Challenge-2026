"""Where does validation macro F0.5 go? Splits errors into categories, measures how much each
category costs (F0.5 gained if that category alone were fixed), scores each country separately
and prints examples.

    python error_analysis.py          # needs work/val_preds.parquet, val_s1.parquet, model_meta.json
"""
import json

import polars as pl

from blocking_eval import truth_index
from config import WORK_DIR, norm_path
from decide import assign
from metrics import macro_f05

KEY = ["i1", "io", "src"]


def main():
    meta = json.load(open(WORK_DIR / "model_meta.json"))
    thr = meta["threshold"]
    va = pl.read_parquet(WORK_DIR / "val_preds.parquet")
    val_s1 = pl.read_parquet(WORK_DIR / "val_s1.parquet")["i1"]
    truth_all, _ = truth_index()
    truth = truth_all.filter(pl.col("i1").is_in(val_s1.implode()))
    kept = assign(va, thr)
    base = macro_f05(kept, truth, val_s1)
    print(f"val S1 {val_s1.len():,} | threshold {thr:.3f} | macro F0.5 {base:.5f}\n")

    # ---- false negatives
    fn = truth.join(kept, on=KEY, how="anti")
    in_cand = fn.join(va.select(KEY), on=KEY, how="semi")
    fn_block = fn.join(va.select(KEY), on=KEY, how="anti")
    owner = kept.select("io", "src", pl.col("i1").alias("owner"))
    fn_c = in_cand.join(owner, on=["io", "src"], how="left")
    fn_other = fn_c.filter(pl.col("owner").is_not_null()).select(KEY)
    fn_low = fn_c.filter(pl.col("owner").is_null()).select(KEY)
    # ---- false positives
    fp = kept.join(truth, on=KEY, how="anti")
    has_owner = truth_all.select("io", "src").unique()
    fp_distr = fp.join(has_owner, on=["io", "src"], how="anti")
    fp_wrong = fp.join(has_owner, on=["io", "src"], how="semi")

    cats = {
        "FN not in candidates (blocking)": ("fn", fn_block),
        "FN below threshold": ("fn", fn_low),
        "FN assigned to another S1": ("fn", fn_other),
        "FP record is a distractor": ("fp", fp_distr),
        "FP record belongs to another S1": ("fp", fp_wrong),
    }
    print(f"{'category':<34}{'pairs':>10}{'F0.5 if fixed':>16}{'gain':>9}")
    for name, (kind, df) in cats.items():
        fixed = pl.concat([kept, df]) if kind == "fn" else kept.join(df, on=KEY, how="anti")
        f = macro_f05(fixed, truth, val_s1)
        print(f"{name:<34}{df.height:>10,}{f:>16.5f}{f - base:>+9.5f}")

    # ---- what the wrong merges look like
    fpf = fp.join(va, on=KEY, how="left")
    print("\nFP profile (share of wrong merges):")
    print(fpf.select(
        legal_conflict=(pl.col("legal_conflict") == 1).mean(),
        number_shift_1_20=pl.col("num_min_delta").is_between(1, 20).mean(),
        same_first_number=(pl.col("num_first_eq") == 1).mean(),
        o_nonascii=(pl.col("o_nonascii") == 1).mean(),
        o_addr_missing=(pl.col("o_addr_missing") == 1).mean(),
        name_tset_ge_95=(pl.col("n_tset") >= 95).mean(),
        prob_median=pl.col("prob").median()))

    # ---- per country
    country = pl.read_parquet(norm_path("train", 1), columns=["country"]).with_row_index("i1") \
                .with_columns(pl.col("i1").cast(pl.Int32))
    print("\nper-country macro F0.5:")
    for c in country["country"].unique().sort().to_list():
        ids = country.filter(pl.col("country") == c).join(val_s1.to_frame(), on="i1", how="semi")["i1"]
        sub = lambda d: d.filter(pl.col("i1").is_in(ids.implode()))
        print(f"  {c:<10} n={ids.len():>8,}  F0.5 {macro_f05(sub(kept), sub(truth), ids):.5f}")

    # ---- examples
    n1 = pl.read_parquet(norm_path("train", 1), columns=["name_n", "addr_n", "anums"])
    no = {s: pl.read_parquet(norm_path("train", s), columns=["name_n", "addr_n", "anums"]) for s in (2, 3)}
    probs = va.select(*KEY, "prob")
    for name, (_, df) in cats.items():
        ex = df.join(probs, on=KEY, how="left").sample(n=min(8, df.height), seed=0)
        print(f"\n--- {name}")
        for i1, io, s, p in ex.select(*KEY, "prob").iter_rows():
            a, b = n1.row(i1), no[s].row(io)
            ptxt = "  -  " if p is None else f"{p:.3f}"
            print(f"  p={ptxt} S1: {a[0]} | {a[2]} {a[1]}\n          S{s}: {b[0]} | {b[2]} {b[1]}")


if __name__ == "__main__":
    main()
