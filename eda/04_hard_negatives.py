"""How ambiguous is the space? Name-key collisions within S1, distractor lookalikes,
row-order leakage check."""
import re
import unicodedata
import polars as pl

D = "C:/Users/yashv/Desktop/ML-challenge-2026/student_resource/dataset/train"
E = "C:/Users/yashv/Desktop/ML-challenge-2026/eda"
rd = lambda p: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False)
LEGAL = r"\b(private|pvt|limited|ltd|llc|llp|inc|incorporated|corp|corporation|co|company|m s|the|and|pc|pllc|lp|plc|services)\b"


def key_expr(c):
    return (pl.col(c).str.to_lowercase().str.replace_all(r"[^a-z0-9 ]", " ")
            .str.replace_all(LEGAL, " ").str.replace_all(r"\s+", " ").str.strip_chars())


s1 = rd(f"{D}/train_source1.tsv").with_row_index("pos1")
s1 = s1.with_columns(k=key_expr("business_name"))
kc = s1.group_by("k").len()
print("S1 distinct name keys:", kc.height, "of", s1.height)
print("S1 rows whose name key is shared with another S1:", kc.filter(pl.col("len") > 1)["len"].sum())
print(kc.sort("len", descending=True).head(15))
same = s1.filter(pl.col("k") == kc.filter(pl.col("len").is_between(3, 6)).sort("len")["k"][0])
print("\nexample same-key S1 group:\n", same.select("business_name", "business_address", "country"))

# same key AND same city-ish? use last 2 address tokens as crude locality
pairs = pl.read_parquet(f"{E}/train_pairs.parquet")
for s in (2, 3):
    o = rd(f"{D}/train_source{s}.tsv").with_row_index("pos").with_columns(k=key_expr("business_name"))
    o = o.join(pairs, left_on="entity_id", right_on="ids", how="left")
    o = o.with_columns(is_pos=pl.col("source1_entity_id").is_not_null())
    # does the record's name key hit any S1 key?  and how many S1s?
    o = o.join(kc.rename({"len": "s1_hits"}), on="k", how="left").with_columns(pl.col("s1_hits").fill_null(0))
    print(f"\nS{s}: name-key exact hit rate into S1 & mean #S1 sharing key")
    print(o.group_by("is_pos").agg(hit=(pl.col("s1_hits") > 0).mean(),
                                   hits_ge2=(pl.col("s1_hits") > 1).mean(),
                                   mean_hits=pl.col("s1_hits").mean()))
    print(f"S{s} distractor examples:")
    for r in o.filter(~pl.col("is_pos")).sample(12, seed=3).select("business_name", "business_address", "country", "s1_hits").iter_rows():
        print("  ", r)
    # distractors whose key hits exactly 1 S1: show them next to that S1
    dd = (o.filter(~pl.col("is_pos") & (pl.col("s1_hits") == 1)).sample(8, seed=4)
          .join(s1.select("k", "business_name", "business_address"), on="k", suffix="_s1"))
    print(f"S{s} distractors vs same-key S1:")
    for r in dd.select("business_name", "business_address", "business_name_s1", "business_address_s1").iter_rows():
        print(f"   D : {r[0]} | {r[1]}\n   S1: {r[2]} | {r[3]}")
    # leakage check: row-position correlation for positives
    lk = o.filter(pl.col("is_pos")).join(s1.select("entity_id", "pos1"), left_on="source1_entity_id", right_on="entity_id")
    print(f"S{s} row-position corr with S1 (leak check):", lk.select(pl.corr("pos", "pos1")).item())
    del o
