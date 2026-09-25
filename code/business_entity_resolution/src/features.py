"""Stage 3: pairwise features for candidate pairs.

Feature groups (all country-agnostic, no country identity used as a feature):
  * name string similarity (rapidfuzz, vectorised C++ via cpdist)
  * legal-form agreement: same / conflicting / one side missing   <- twin detector
  * address string similarity + number-set comparison              <- twin detector
    (a twin shifts one house/unit number by a small delta; typos in true matches
     usually drop/insert digits, so both the numeric delta and digit edit distance matter)
  * record-quality flags: non-latin script, domain-style name, missing address
  * blocking context: score, rank, gap to best candidate, and the reverse view
    (how this S2/S3 record ranks among all S1 records that retrieved it)
"""
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from config import N_JOBS

NUM_FEATS = ["num_n1", "num_n2", "num_inter", "num_s1_unmatched", "num_o_unmatched", "num_first_eq",
             "num_min_delta", "num_min_digit_ed", "num_first_delta", "name_num_conflict"]


def _num_feats(args):
    a_str, b_str, na_str, nb_str = args
    A = a_str.split()[:6] if a_str else []
    B = b_str.split()[:6] if b_str else []
    sa, sb = set(A), set(B)
    inter = sa & sb
    ua, ub = [x for x in sa - inter], [x for x in sb - inter]
    min_delta, min_ed = -1.0, -1.0
    if ua and ub:
        best = None
        for x in ua:
            for y in ub:
                d = abs(int(x) - int(y)) if len(x) < 12 and len(y) < 12 else 1e9
                if best is None or d < best[0]:
                    best = (d, Levenshtein.distance(x, y))
        min_delta, min_ed = float(min(best[0], 1e6)), float(best[1])
    first_eq = -1.0 if not A or not B else float(A[0] == B[0])
    first_delta = -1.0 if not A or not B or len(A[0]) > 11 or len(B[0]) > 11 else float(min(abs(int(A[0]) - int(B[0])), 1e6))
    nna, nnb = set(na_str.split()) if na_str else set(), set(nb_str.split()) if nb_str else set()
    name_num_conflict = float(bool(nna) and bool(nnb) and not (nna & nnb))
    return (len(sa), len(sb), len(inter), len(ua), len(ub), first_eq, min_delta, min_ed, first_delta,
            name_num_conflict)


def _num_chunk(rows):
    return [_num_feats(r) for r in rows]


def _legal_feats(a, b):
    """a, b: polars Series of '|' joined legal classes."""
    la = a.str.split("|").list.eval(pl.element().filter(pl.element() != ""))
    lb = b.str.split("|").list.eval(pl.element().filter(pl.element() != ""))
    na, nb = la.list.len(), lb.list.len()
    inter = la.list.set_intersection(lb).list.len()
    both = (na > 0) & (nb > 0)
    return {
        "legal_same": (both & (inter > 0)).cast(pl.Float32),
        "legal_conflict": (both & (inter == 0)).cast(pl.Float32),
        "legal_one_missing": ((na > 0) != (nb > 0)).cast(pl.Float32),
        "legal_both_missing": ((na == 0) & (nb == 0)).cast(pl.Float32),
        "legal_exact": (a == b).cast(pl.Float32),
    }


def _sim(q, c, scorer, **kw):
    return cpdist(q, c, scorer=scorer, workers=N_JOBS, dtype=np.float32, **kw)


def _jaccard(a, b):
    ta = a.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    tb = b.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    inter = ta.list.set_intersection(tb).list.len()
    union = ta.list.set_union(tb).list.len()
    return (inter / union).fill_nan(0).fill_null(0).cast(pl.Float32)


def context_features(cand):
    """Blocking-context features computed over the full candidate table.
    cand: i1, io, src, score, rank."""
    return cand.with_columns(
        s1_best=pl.col("score").max().over(["i1", "src"]),
        s1_n=pl.len().over(["i1", "src"]),
        o_best=pl.col("score").max().over(["io", "src"]),
        o_n=pl.len().over(["io", "src"]),
        o_rank=pl.col("score").rank("ordinal", descending=True).over(["io", "src"]) - 1,
    ).with_columns(
        s1_gap=pl.col("s1_best") - pl.col("score"),
        o_gap=pl.col("o_best") - pl.col("score"),
        s1_second=pl.col("score").sort(descending=True).slice(1, 1).first().over(["i1", "src"]).fill_null(0),
        o_second=pl.col("score").sort(descending=True).slice(1, 1).first().over(["io", "src"]).fill_null(0),
    ).with_columns(
        o_margin=pl.when(pl.col("o_rank") == 0).then(pl.col("score") - pl.col("o_second"))
        .otherwise(pl.col("score") - pl.col("o_best")),
    ).drop("s1_second", "o_second")


def pair_features(pairs, n1, no_all, n2_len, pool=None):
    """pairs: DataFrame with i1, io, src (+ context cols). n1: normalized S1; no_all: normalized S2
    and S3 stacked (S3 rows offset by n2_len). Returns pairs with feature columns appended."""
    a = n1[pairs["i1"].to_numpy()]
    b = no_all[pairs["io"].to_numpy() + np.where(pairs["src"].to_numpy() == 3, n2_len, 0)]

    ac, bc = a["core"].to_list(), b["core"].to_list()
    acp, bcp = a["compact"].to_list(), b["compact"].to_list()
    aa, ba = a["addr_n"].to_list(), b["addr_n"].to_list()
    feats = {
        "n_ratio": _sim(ac, bc, fuzz.ratio),
        "n_tset": _sim(ac, bc, fuzz.token_set_ratio),
        "n_tsort": _sim(ac, bc, fuzz.token_sort_ratio),
        "n_partial": _sim(ac, bc, fuzz.partial_ratio),
        "n_compact_jw": _sim(acp, bcp, JaroWinkler.normalized_similarity),
        "n_compact_partial": _sim(acp, bcp, fuzz.partial_ratio),
        "n_full_ratio": _sim(a["name_n"].to_list(), b["name_n"].to_list(), fuzz.ratio),
        "a_ratio": _sim(aa, ba, fuzz.ratio),
        "a_tset": _sim(aa, ba, fuzz.token_set_ratio),
        "a_partial": _sim(aa, ba, fuzz.partial_ratio),
        "anum_ratio": _sim(a["anums"].to_list(), b["anums"].to_list(), fuzz.ratio),
    }
    out = pairs.with_columns(**{k: pl.Series(v) for k, v in feats.items()})
    out = out.with_columns(
        n_jacc=_jaccard(a["core"], b["core"]),
        a_jacc=_jaccard(a["addr_n"], b["addr_n"]),
        n_first_eq=(a["core"].str.split(" ").list.first() == b["core"].str.split(" ").list.first()).cast(pl.Float32),
        n_len1=a["core"].str.len_chars().cast(pl.Float32),
        n_len2=b["core"].str.len_chars().cast(pl.Float32),
        a_ntok1=a["addr_n"].str.count_matches(" ").cast(pl.Float32),
        a_ntok2=b["addr_n"].str.count_matches(" ").cast(pl.Float32),
        o_nonascii=b["name_nonascii"].cast(pl.Float32),
        o_domain=b["is_domain"].cast(pl.Float32),
        o_addr_missing=b["addr_missing"].cast(pl.Float32),
        s1_addr_missing=a["addr_missing"].cast(pl.Float32),
        **_legal_feats(a["legal"], b["legal"]),
    )
    rows = list(zip(a["anums"].to_list(), b["anums"].to_list(), a["name_nums"].to_list(), b["name_nums"].to_list()))
    chunks = [rows[i:i + 50000] for i in range(0, len(rows), 50000)]
    own = pool is None
    pool = pool or Pool(N_JOBS)
    num = [r for part in pool.map(_num_chunk, chunks) for r in part]
    if own:
        pool.close()
    num = np.asarray(num, dtype=np.float32).reshape(-1, len(NUM_FEATS))
    return out.with_columns(**{k: pl.Series(num[:, j]) for j, k in enumerate(NUM_FEATS)})


def feature_columns(df):
    skip = {"i1", "io", "label", "fold"}
    return [c for c in df.columns if c not in skip]
