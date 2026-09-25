"""Blocking diagnostics on train: pair recall and oracle macro-F0.5 of the candidate union and
of each channel, plus examples of true pairs that no channel retrieved.

    python blocking_eval.py
"""
import polars as pl

from config import DATA_DIR, cand_path, norm_path

NOT_FOUND = 99


def load_truth():
    gt = pl.read_csv(DATA_DIR / "train" / "train_ground_truth.tsv", separator="\t", quote_char=None,
                     infer_schema=False).with_columns(pl.col("matched_entity_ids").fill_null(""))
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
            .rename({"source1_entity_id": "s1", "matched_entity_ids": "o"}))


def truth_index(split="train"):
    """True pairs as (i1, io, src) row indices into the normalized parquet files."""
    ids1 = pl.read_parquet(norm_path(split, 1), columns=["entity_id"]).with_row_index("i1")
    t = load_truth().join(ids1.rename({"entity_id": "s1"}), on="s1")
    out = []
    for s in (2, 3):
        ids = pl.read_parquet(norm_path(split, s), columns=["entity_id"]).with_row_index("io")
        out.append(t.join(ids.rename({"entity_id": "o"}), on="o").with_columns(src=pl.lit(s, pl.Int8)))
    truth = pl.concat(out).select(pl.col("i1").cast(pl.Int32), pl.col("io").cast(pl.Int32), "src")
    return truth, ids1.height


def oracle(found, truth, n_s1):
    """Pair recall and macro F0.5 of a perfect matcher restricted to `found` true pairs."""
    per_true = truth.group_by("i1").len().rename({"len": "nt"})
    per_found = found.group_by("i1").len().rename({"len": "nf"})
    d = per_true.join(per_found, on="i1", how="left").with_columns(pl.col("nf").fill_null(0))
    r = d["nf"] / d["nt"]
    f = (1.25 * r / (0.25 + r)).fill_nan(0)
    n_singletons = n_s1 - d.height                     # no true matches -> empty prediction scores 1
    return found.height / truth.height, (f.sum() + n_singletons) / n_s1


def main():
    truth, n_s1 = truth_index()
    cand = pl.read_parquet(cand_path("train"))
    hit = truth.join(cand, on=["i1", "io", "src"], how="left")
    found = hit.filter(pl.col("score").is_not_null())
    print(f"true pairs {truth.height:,}; candidates {cand.height:,} ({cand.height / n_s1:.1f} per S1)")
    for name, cond in [
        ("union (all channels)", pl.lit(True)),
        ("forward K=5", pl.col("rank_c") < 5),
        ("forward K=10", pl.col("rank_c") < 10),
        ("forward + name", (pl.col("rank_c") < NOT_FOUND) | (pl.col("rank_n") < NOT_FOUND)),
        ("forward + reverse", (pl.col("rank_c") < NOT_FOUND) | (pl.col("rank_r") < NOT_FOUND)),
    ]:
        rec, f = oracle(found.filter(cond), truth, n_s1)
        n = cand.filter(cond).height
        print(f"  {name:<22} pair recall {rec:.4f} | oracle macro F0.5 {f:.4f} | cands/S1 {n / n_s1:.1f}")

    miss = hit.filter(pl.col("score").is_null()).sample(n=min(15, hit.height - found.height), seed=0)
    n1 = pl.read_parquet(norm_path("train", 1), columns=["core", "addr_n", "anums", "country"])
    no = {s: pl.read_parquet(norm_path("train", s), columns=["core", "addr_n", "anums"]) for s in (2, 3)}
    print("\ntrue pairs missed by every channel (normalized):")
    for i1, io, s in miss.select("i1", "io", "src").iter_rows():
        a, b = n1.row(i1), no[s].row(io)
        print(f"  [{a[3]}] S1: {a[0]} | {a[2]} {a[1]}\n       S{s}: {b[0]} | {b[2]} {b[1]}")


if __name__ == "__main__":
    main()
