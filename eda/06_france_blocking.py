"""(a) France in test: what do records look like? (b) Blocking feasibility on true pairs:
what cheap keys do positives share?"""
import re
import polars as pl

T = "C:/Users/yashv/Desktop/ML-challenge-2026/student_resource/dataset/test"
E = "C:/Users/yashv/Desktop/ML-challenge-2026/eda"
rd = lambda p: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False)

for s in (1, 2, 3):
    df = rd(f"{T}/test_source{s}.tsv").filter(pl.col("country") == "France")
    print(f"\n--- test S{s} France samples")
    for r in df.sample(10, seed=2).select("business_name", "business_address").iter_rows():
        print("  ", r)
    toks = (df["business_name"].str.to_lowercase().str.replace_all(r"[^\w ]", " ").str.split(" ")
            .explode())
    vc = toks.value_counts(sort=True).head(25)
    print("top name tokens:", [(a, b) for a, b in vc.iter_rows() if a])
    del df

# blocking feasibility on sampled positive pairs (train)
p = pl.read_parquet(f"{E}/sample_pos_pairs.parquet")
STOP = set("private limited pvt ltd llc inc corp co company the and of services group llp".split())


def ntoks(s):
    return {t for t in re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split() if t not in STOP and len(t) > 1}


def nums(s):
    return set(re.findall(r"\d+", s or ""))


def atoks(s):
    return {t for t in re.sub(r"[^a-z ]", " ", (s or "").lower()).split() if len(t) > 3}


stats = dict(name_tok=0, addr_num=0, addr_word=0, any_=0, none_=0)
none_ex = []
for n1, a1, n2, a2 in p.select("business_name", "business_address", "business_name_o", "business_address_o").iter_rows():
    a = bool(ntoks(n1) & ntoks(n2)); b = bool(nums(a1) & nums(a2)); c = len(atoks(a1) & atoks(a2)) >= 2
    stats["name_tok"] += a; stats["addr_num"] += b; stats["addr_word"] += c
    if a or b or c:
        stats["any_"] += 1
    else:
        stats["none_"] += 1
        if len(none_ex) < 12:
            none_ex.append((n1, a1, n2, a2))
print("\nblocking feasibility over", p.height, "positive pairs:",
      {k: round(v / p.height, 4) for k, v in stats.items()})
print("positives sharing NO cheap key:")
for e in none_ex:
    print(f"   S1: {e[0]} | {e[1]}\n   S*: {e[2]} | {e[3]}")
