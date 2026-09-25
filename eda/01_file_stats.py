"""Per-file profile: countries, empties, 'null' strings, scripts, lengths, duplicates."""
import sys
import polars as pl

D = "C:/Users/yashv/Desktop/ML-challenge-2026/student_resource/dataset"


def read(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       missing_utf8_is_empty_string=True)


def script_expr(col):
    return (pl.when(pl.col(col).str.contains(r"[\u0900-\u097F]")).then(pl.lit("devanagari"))
            .when(pl.col(col).str.contains(r"[\u0980-\u0DFF]")).then(pl.lit("other_indic"))
            .when(pl.col(col).str.contains(r"[^\x00-\x7F]")).then(pl.lit("latin_ext/other"))
            .otherwise(pl.lit("ascii")))


for split in ("train", "test"):
    for s in (1, 2, 3):
        df = read(f"{D}/{split}/{split}_source{s}.tsv")
        print(f"\n===== {split} source{s}: {df.height:,} rows, cols={df.columns}")
        print("dup entity_id:", df.height - df["entity_id"].n_unique())
        print(df["country"].value_counts(sort=True).with_columns(
            (pl.col("count") / df.height).round(3).alias("frac")))
        for c in ("business_name", "business_address"):
            col = pl.col(c)
            st = df.select(
                empty=(col.str.strip_chars() == "").mean(),
                null_str=col.str.to_lowercase().str.contains(r"\bnull\b|\bnan\b|\bn/?a\b").mean(),
                len_med=col.str.len_chars().median(),
                len_p95=col.str.len_chars().quantile(0.95),
                upper=(col == col.str.to_uppercase()).mean(),
            )
            print(c, st.to_dicts()[0])
            print(df.with_columns(script=script_expr(c)).group_by(["country", "script"]).len()
                  .sort(["country", "len"], descending=[False, True]))
        # exact duplicate (name,address) within file
        print("dup (name,addr) rows:", df.height - df.select(["business_name", "business_address"]).n_unique())
        print("dup name rows (lower):", df.height - df["business_name"].str.to_lowercase().n_unique())
        del df
        sys.stdout.flush()
