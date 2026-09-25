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
  * record rarity: how many S1s share this exact name / address, rarest name-token frequency
  * similarity competition: this pair's name/address similarity vs. the other S1s competing for
    the same record and vs. the S1's other candidates
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


RS_COLS = ["rs_nm_s1", "rs_ad_s1", "rs_nm_other", "rs_tokdf"]


def _addr_key(df):
    return pl.when(df["addr_missing"]).then(pl.lit("")).otherwise(df["anums"] + "|" + df["addr_n"])


def add_record_stats(n1, n2, n3):
    """Per-record rarity counts over the whole split (no labels involved):
      rs_nm_s1    S1 records with this exact core name   (is the name ambiguous among owners?)
      rs_ad_s1    S1 records at this exact address       (shared building / office block?)
      rs_nm_other S1 row: S2+S3 records with this name; S2/S3 row: same-source records with it
      rs_tokdf    records (all sources) containing this name's rarest core token
    A bare name with no address can only be linked safely when the name is rare; these let the
    model tell 'becerra montessori school' (unique) from 'new delhi services' (many owners)."""
    frames = [n1.select("core", ak=_addr_key(n1)), n2.select("core", ak=_addr_key(n2)),
              n3.select("core", ak=_addr_key(n3))]
    cnt = lambda s, name: (s.filter(s != "").value_counts(name="c").rename({s.name: "k"})
                           .with_columns(pl.col("c").cast(pl.Float32)).rename({"c": name}))
    nm1 = cnt(frames[0]["core"], "rs_nm_s1")
    ad1 = cnt(frames[0]["ak"], "rs_ad_s1")
    nm2, nm3 = cnt(frames[1]["core"], "n"), cnt(frames[2]["core"], "n")
    nm23 = pl.concat([nm2, nm3]).group_by("k").agg(pl.col("n").sum().alias("rs_nm_other"))
    tok = (pl.concat([f.select(pl.col("core").str.split(" ").list.unique().alias("t")) for f in frames])
           .explode("t").filter(pl.col("t") != "")["t"].value_counts(name="d").rename({"t": "k"}))

    def attach(f, other):
        f = f.with_row_index("r")
        mind = (f.select("r", t=pl.col("core").str.split(" ")).explode("t").filter(pl.col("t") != "")
                .join(tok, left_on="t", right_on="k").group_by("r").agg(rs_tokdf=pl.col("d").min().cast(pl.Float32)))
        out = (f.join(nm1, left_on="core", right_on="k", how="left")
               .join(ad1, left_on="ak", right_on="k", how="left")
               .join(other, left_on="core", right_on="k", how="left")
               .join(mind, on="r", how="left").sort("r"))
        return out.select(pl.col(RS_COLS).fill_null(0))

    return (n1.hstack(attach(frames[0], nm23)),
            n2.hstack(attach(frames[1], nm2.rename({"n": "rs_nm_other"}))),
            n3.hstack(attach(frames[2], nm3.rename({"n": "rs_nm_other"}))))


def sim_context(cand, n1, no_all, n2_len, chunk=4_000_000):
    """Similarity-aware version of the blocking-context features, computed over the FULL candidate
    table (every S1 that retrieved a record competes, whatever its fold). Blocking cosine says little
    about *which* of several same-address or same-name S1s owns a record; name/address similarity
    relative to the competing candidates does:
      o_*  : over the S1s competing for this S2/S3 record   s1_* : over this S1's own candidates
      *_js_best / *_js_gap / o_js_rank on js = name token-sort + 0.5 * full-address token-set
      *_ns_best / *_ns_gap on name token-sort; *_n90 = competitors with name token-sort >= 90."""
    fa1 = (n1["anums"] + " " + n1["addr_n"]).to_list()
    fao = (no_all["anums"] + " " + no_all["addr_n"]).to_list()
    c1, co = n1["core"].to_list(), no_all["core"].to_list()
    i1 = cand["i1"].to_numpy()
    io = cand["io"].to_numpy() + np.where(cand["src"].to_numpy() == 3, n2_len, 0)
    ns, as_ = np.empty(len(i1), np.float32), np.empty(len(i1), np.float32)
    for s in range(0, len(i1), chunk):
        a, b = i1[s:s + chunk], io[s:s + chunk]
        ns[s:s + chunk] = _sim([c1[j] for j in a], [co[j] for j in b], fuzz.token_sort_ratio)
        as_[s:s + chunk] = _sim([fa1[j] for j in a], [fao[j] for j in b], fuzz.token_set_ratio)
    del fa1, fao, c1, co
    o, s1 = ["io", "src"], ["i1", "src"]
    return cand.with_columns(c_ns=pl.Series(ns), c_fa_tset=pl.Series(as_)).with_columns(
        c_js=pl.col("c_ns") + 0.5 * pl.col("c_fa_tset"),
    ).with_columns(
        o_js_best=pl.col("c_js").max().over(o),
        o_ns_best=pl.col("c_ns").max().over(o),
        o_n90=(pl.col("c_ns") >= 90).sum().over(o).cast(pl.Float32),
        o_js_rank=(pl.col("c_js").rank("min", descending=True).over(o) - 1).cast(pl.Float32),
        s1_js_best=pl.col("c_js").max().over(s1),
        s1_ns_best=pl.col("c_ns").max().over(s1),
        s1_n90=(pl.col("c_ns") >= 90).sum().over(s1).cast(pl.Float32),
    ).with_columns(
        o_js_gap=pl.col("o_js_best") - pl.col("c_js"),
        o_ns_gap=pl.col("o_ns_best") - pl.col("c_ns"),
        s1_js_gap=pl.col("s1_js_best") - pl.col("c_js"),
        s1_ns_gap=pl.col("s1_ns_best") - pl.col("c_ns"),
    ).drop("c_ns")


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
        **{f"a_{c}": a[c] for c in RS_COLS},
        **{f"b_{c}": b[c] for c in RS_COLS},
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
