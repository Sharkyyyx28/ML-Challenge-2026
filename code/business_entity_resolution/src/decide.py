"""Decision rule: each S2/S3 record goes to at most one S1 entity (its highest-probability
S1 among its candidates), and only if that probability clears the threshold."""
import polars as pl


def assign(scored, threshold):
    """scored: DataFrame with i1, io, src, prob. Returns kept (i1, io, src)."""
    best = scored.with_columns(is_best=pl.col("prob") == pl.col("prob").max().over(["io", "src"]))
    return (best.filter(pl.col("is_best") & (pl.col("prob") >= threshold))
            .unique(subset=["io", "src"], keep="first").select("i1", "io", "src"))
