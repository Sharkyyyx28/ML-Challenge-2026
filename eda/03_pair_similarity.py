"""How similar are true pairs? Sample S1 entities, join their matches, measure fuzzy sims,
and print examples across the similarity spectrum. Also check country agreement."""
import random
import polars as pl
from rapidfuzz import fuzz

D = "C:/Users/yashv/Desktop/ML-challenge-2026/student_resource/dataset/train"
E = "C:/Users/yashv/Desktop/ML-challenge-2026/eda"
rd = lambda p: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False)
random.seed(0)

pairs = pl.read_parquet(f"{E}/train_pairs.parquet")
s1 = rd(f"{D}/train_source1.tsv")
samp_ids = s1.sample(n=60000, seed=0)["entity_id"]
s1 = s1.filter(pl.col("entity_id").is_in(samp_ids.implode()))
pairs = pairs.filter(pl.col("source1_entity_id").is_in(samp_ids.implode()))
others = []
for s in (2, 3):
    o = rd(f"{D}/train_source{s}.tsv")
    others.append(o.filter(pl.col("entity_id").is_in(pairs["ids"].implode())))
    del o
oth = pl.concat(others)
df = (pairs.join(s1, left_on="source1_entity_id", right_on="entity_id")
      .join(oth, left_on="ids", right_on="entity_id", suffix="_o"))
df.write_parquet(f"{E}/sample_pos_pairs.parquet")
print("sample positive pairs:", df.height)
print("country agree:", (df["country"] == df["country_o"]).mean())
print(df.filter(pl.col("country") != pl.col("country_o")).head(5))

rows = df.to_dicts()
low = lambda s: (s or "").lower()
for r in rows:
    r["name_tsr"] = fuzz.token_set_ratio(low(r["business_name"]), low(r["business_name_o"]))
    r["addr_tsr"] = fuzz.token_set_ratio(low(r["business_address"]), low(r["business_address_o"]))
    r["name_exact"] = low(r["business_name"]) == low(r["business_name_o"])
sim = pl.DataFrame(rows)
sim = sim.with_columns(src=pl.col("ids").str.slice(0, 2))
print("\nname token_set_ratio quantiles by src/country:")
print(sim.group_by(["src", "country"]).agg(
    [pl.col("name_tsr").quantile(q).alias(f"n_q{int(q*100)}") for q in (0.05, 0.25, 0.5)]
    + [pl.col("addr_tsr").quantile(q).alias(f"a_q{int(q*100)}") for q in (0.05, 0.25, 0.5)]
    + [pl.col("name_exact").mean().alias("name_exact")]))
print("\nfrac name_tsr<50:", (sim["name_tsr"] < 50).mean(), " addr_tsr<50:", (sim["addr_tsr"] < 50).mean(),
      " both<50:", ((sim["name_tsr"] < 50) & (sim["addr_tsr"] < 50)).mean())

cols = ["business_name", "business_address", "business_name_o", "business_address_o", "country", "name_tsr", "addr_tsr"]
def show(title, frame, k=12):
    print(f"\n--- {title} ({frame.height})")
    for r in frame.sample(n=min(k, frame.height), seed=1).select(cols).iter_rows():
        print(f"[{r[4]}] n={r[5]:.0f} a={r[6]:.0f}\n   S1: {r[0]} | {r[1]}\n   S*: {r[2]} | {r[3]}")

show("random positives", sim, 25)
show("low NAME sim positives", sim.filter(pl.col("name_tsr") < 50), 20)
show("low ADDRESS sim positives", sim.filter(pl.col("addr_tsr") < 40), 15)
