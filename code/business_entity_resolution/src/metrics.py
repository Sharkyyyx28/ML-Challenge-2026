"""Macro F0.5 exactly as the challenge defines it (per S1 entity, singletons included)."""
import numpy as np
import polars as pl


def macro_f05(pred, truth, s1_ids):
    """pred/truth: DataFrames (i1, io, src) of predicted / true pairs. s1_ids: Series of all evaluated i1."""
    base = pl.DataFrame({"i1": s1_ids})
    tp = pred.join(truth, on=["i1", "io", "src"], how="semi").group_by("i1").len().rename({"len": "tp"})
    npred = pred.group_by("i1").len().rename({"len": "np"})
    ntrue = truth.group_by("i1").len().rename({"len": "nt"})
    d = base.join(tp, on="i1", how="left").join(npred, on="i1", how="left").join(ntrue, on="i1", how="left").fill_null(0)
    tp_, np_, nt_ = (d[c].to_numpy().astype(np.float64) for c in ("tp", "np", "nt"))
    p = np.divide(tp_, np_, out=np.zeros_like(tp_), where=np_ > 0)
    r = np.divide(tp_, nt_, out=np.zeros_like(tp_), where=nt_ > 0)
    denom = 0.25 * p + r
    f = np.divide(1.25 * p * r, denom, out=np.zeros_like(p), where=denom > 0)
    f[(nt_ == 0) & (np_ == 0)] = 1.0
    return float(f.mean())
