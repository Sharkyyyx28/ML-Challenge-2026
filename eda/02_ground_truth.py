"""Ground-truth structure: match counts, singletons, coverage of S2/S3, many-to-one."""
import polars as pl

D = "C:/Users/yashv/Desktop/ML-challenge-2026/student_resource/dataset/train"
rd = lambda p: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False)

gt = rd(f"{D}/train_ground_truth.tsv").with_columns(pl.col("matched_entity_ids").fill_null(""))
s1 = rd(f"{D}/train_source1.tsv").select("entity_id", "country")
gt = gt.join(s1, left_on="source1_entity_id", right_on="entity_id", how="left")
print("gt rows", gt.height, "| S1 ids missing from gt:", s1.height - gt.height)

gt = gt.with_columns(ids=pl.col("matched_entity_ids").str.split(",").list.eval(
    pl.element().filter(pl.element() != "")))
gt = gt.with_columns(n=pl.col("ids").list.len(),
                     n2=pl.col("ids").list.eval(pl.element().str.starts_with("S2")).list.sum(),
                     n3=pl.col("ids").list.eval(pl.element().str.starts_with("S3")).list.sum())

print("\nmatches per S1 (all):")
print(gt["n"].value_counts().sort("n").with_columns((pl.col("count") / gt.height).round(4).alias("frac")))
print("\nby country: singleton rate, mean matches, mean S2, mean S3")
print(gt.group_by("country").agg(singleton=(pl.col("n") == 0).mean(), mean_n=pl.col("n").mean(),
                                 mean_s2=pl.col("n2").mean(), mean_s3=pl.col("n3").mean(),
                                 max_n=pl.col("n").max(), p99=pl.col("n").quantile(0.99)))
print("\n(n2,n3) combos top:")
print(gt.group_by(["n2", "n3"]).len().sort("len", descending=True).head(15))

pairs = gt.select("source1_entity_id", "ids").explode("ids").drop_nulls("ids")
print("\ntotal positive pairs:", pairs.height)
multi = pairs.group_by("ids").len().filter(pl.col("len") > 1)
print("S2/S3 ids matched to >1 S1:", multi.height)

for s in (2, 3):
    ids = rd(f"{D}/train_source{s}.tsv").select("entity_id", "country")
    matched = ids.join(pairs, left_on="entity_id", right_on="ids", how="semi")
    print(f"\nS{s}: {ids.height:,} records, matched {matched.height:,} ({matched.height/ids.height:.3f}); "
          f"unmatched (distractors) {ids.height-matched.height:,}")
    print(ids.with_columns(m=pl.col("entity_id").is_in(pairs["ids"].implode()))
          .group_by("country").agg(pl.col("m").mean()))

pairs.write_parquet("C:/Users/yashv/Desktop/ML-challenge-2026/eda/train_pairs.parquet")
