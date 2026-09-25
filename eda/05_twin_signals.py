"""Quantify the discriminating signals between true matches and 'twin' distractors:
legal-suffix conflict and house-number delta, on pairs that share name key + street tokens."""
import re
import polars as pl

D = "C:/Users/yashv/Desktop/ML-challenge-2026/student_resource/dataset/train"
E = "C:/Users/yashv/Desktop/ML-challenge-2026/eda"
rd = lambda p: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False)
LEGAL = r"\b(private|pvt|limited|ltd|llc|llp|inc|incorporated|corp|corporation|co|company|the|pc|pllc|lp|plc|services)\b"
SUF = {"llc": "llc", "l l c": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
       "co": "co", "company": "co", "ltd": "ltd", "limited": "ltd", "llp": "llp", "pllc": "pllc", "pc": "pc",
       "lp": "lp", "plc": "plc", "pvt": "pvt", "private": "pvt"}


def key_expr(c):
    return (pl.col(c).str.to_lowercase().str.replace_all(r"[^a-z0-9 ]", " ")
            .str.replace_all(LEGAL, " ").str.replace_all(r"\s+", " ").str.strip_chars())


def suffixes(name):
    toks = re.sub(r"[^a-z ]", " ", (name or "").lower()).split()
    return {SUF[t] for t in toks if t in SUF}


def first_num(addr):
    m = re.search(r"\d+", addr or "")
    return int(m.group()) if m else None


def words(addr):
    return set(t for t in re.sub(r"[^a-z ]", " ", (addr or "").lower()).split() if len(t) > 2)


s1 = rd(f"{D}/train_source1.tsv").with_columns(k=key_expr("business_name"))
kc = s1.group_by("k").len().filter(pl.col("len") <= 20)
s1 = s1.join(kc.select("k"), on="k", how="semi")
pairs = pl.read_parquet(f"{E}/train_pairs.parquet")
out = []
for s in (2, 3):
    o = rd(f"{D}/train_source{s}.tsv").filter(pl.col("country") == "US").sample(fraction=0.15, seed=s)
    o = o.with_columns(k=key_expr("business_name")).filter(pl.col("k") != "")
    j = o.join(s1, on="k", suffix="_1").join(pairs, left_on="entity_id", right_on="ids", how="left")
    j = j.with_columns(label=(pl.col("source1_entity_id") == pl.col("entity_id_1")).fill_null(False))
    out.append(j.select("business_name", "business_address", "business_name_1", "business_address_1", "label"))
    del o
df = pl.concat(out)
rows = []
for n, a, n1, a1, lab in df.iter_rows():
    w, w1 = words(a), words(a1)
    jac = len(w & w1) / len(w | w1) if w and w1 else 0
    if jac < 0.5:
        continue
    s, s_1 = suffixes(n), suffixes(n1)
    x, x1 = first_num(a), first_num(a1)
    rows.append(dict(label=lab, suf_conflict=bool(s and s_1 and not (s & s_1)),
                     suf_same=bool(s and s_1 and (s & s_1)), suf_one_missing=bool(bool(s) != bool(s_1)),
                     num_delta=None if x is None or x1 is None else abs(x - x1), jac=jac,
                     ex=f"{n} | {a}  <->  {n1} | {a1}"))
r = pl.DataFrame(rows)
print("same-namekey & street-overlap pairs:", r.height, " positives:", r["label"].sum())
print(r.group_by("label").agg(pl.col("suf_conflict").mean(), pl.col("suf_same").mean(),
                              pl.col("suf_one_missing").mean(),
                              num_eq=(pl.col("num_delta") == 0).mean(),
                              num_le2=(pl.col("num_delta").is_between(1, 2)).mean(),
                              num_3_20=(pl.col("num_delta").is_between(3, 20)).mean(),
                              num_gt20=(pl.col("num_delta") > 20).mean(),
                              num_missing=pl.col("num_delta").is_null().mean()))
print("\nP(match | suffix-conflict, num delta bucket):")
print(r.with_columns(b=pl.when(pl.col("num_delta").is_null()).then(pl.lit("na"))
                     .when(pl.col("num_delta") == 0).then(pl.lit("0"))
                     .when(pl.col("num_delta") <= 2).then(pl.lit("1-2"))
                     .when(pl.col("num_delta") <= 20).then(pl.lit("3-20")).otherwise(pl.lit(">20")))
      .group_by(["suf_conflict", "b"]).agg(n=pl.len(), p_match=pl.col("label").mean()).sort(["suf_conflict", "b"]))
print("\nexamples: positives with suffix conflict")
for e in r.filter(pl.col("label") & pl.col("suf_conflict")).sample(6, seed=1)["ex"]:
    print("  ", e)
print("examples: negatives with same number & no suffix conflict")
neg = r.filter(~pl.col("label") & (pl.col("num_delta") == 0) & ~pl.col("suf_conflict"))
for e in neg.sample(min(8, neg.height), seed=1)["ex"]:
    print("  ", e)
