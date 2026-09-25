"""Stage 2: candidate generation (three channels, unioned).

For every country label (open set) each record is hashed into a sparse bag of blocking
features, IDF-weighted on that country's S1+S2+S3 records (features that are too common or
appear only once are dropped) and L2-normalized. Three searches then run per source:

  * forward  : top-K_FWD S2/S3 records per S1 on the full name+address vector
  * name     : top-K_NAME S2/S3 records per S1 on the name-only vector
               (recovers same-name records whose address is missing or very different)
  * reverse  : top-K_REV S1 records per S2/S3 record on the full vector
               (recovers true owners pushed out of an S1's forward list by lookalikes)

The union is written with both cosine scores and the rank in each channel (99 = not found).

    python blocking.py train
"""
import sys
import time
from multiprocessing import Pool

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sparse_dot_topn import sp_matmul_topn

from config import N_JOBS, cand_path, norm_path

N_FEATURES = 2 ** 23
MAX_DF = 3000          # features present in more records than this (per country) are dropped
K_FWD, K_NAME, K_REV = 20, 0, 5   # K_NAME = 0 disables the name-only channel (v2: +0.001 oracle F0.5 for +5 cands/S1)
NOT_FOUND = 99
COLS = ["core", "compact", "addr_n", "anums", "name_nums"]


def name_feats(core, compact, name_nums):
    feats = ["w:" + t for t in core.split()]
    c = f"^{compact}$"
    feats += ["c:" + c[i:i + 4] for i in range(max(1, len(c) - 3))]
    feats += ["m:" + n for n in name_nums.split()]
    return feats


def analyzer(doc):
    core, compact, addr, anums, name_nums = doc
    feats = name_feats(core, compact, name_nums)
    words = [t for t in addr.split() if len(t) >= 3]
    feats += ["a:" + t for t in words]
    nums = anums.split()[:4]
    feats += ["n:" + n for n in nums if len(n) >= 2]
    feats += ["x:" + n + "_" + w for n in nums for w in words if len(w) >= 4]
    return feats


def name_analyzer(doc):
    core, compact, _addr, _anums, name_nums = doc
    return name_feats(core, compact, name_nums)


# Same feature strings hash to the same columns in both vectorizers, so one IDF serves both.
_HV = HashingVectorizer(analyzer=analyzer, n_features=N_FEATURES, alternate_sign=False,
                        norm=None, binary=True, dtype=np.float32)
_HVN = HashingVectorizer(analyzer=name_analyzer, n_features=N_FEATURES, alternate_sign=False,
                         norm=None, binary=True, dtype=np.float32)


def _hash(docs):
    return _HV.transform(docs)


def _hash_name(docs):
    return _HVN.transform(docs)


def hash_docs(df, pool, name_only=False):
    docs = list(zip(*[df[c].fill_null("").to_list() for c in COLS]))
    parts = [docs[i:i + 50000] for i in range(0, len(docs), 50000)]
    if not parts:
        return sp.csr_matrix((0, N_FEATURES), dtype=np.float32)
    return sp.vstack(pool.map(_hash_name if name_only else _hash, parts)).tocsr()


def doc_freq(mats):
    df = np.zeros(N_FEATURES, dtype=np.int64)
    for m in mats:
        df += np.bincount(m.indices, minlength=N_FEATURES)
    return df


def make_idf(df, n):
    """IDF weights; over-common (> MAX_DF) and singleton features get weight 0."""
    idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
    idf[(df > MAX_DF) | (df < 2)] = 0
    return idf


def weight_(m, idf):
    """In-place IDF weighting, pruning and row L2-normalization (no matrix copies)."""
    m.data *= idf[m.indices]
    m.eliminate_zeros()
    sq = (np.add.reduceat(m.data ** 2, np.minimum(m.indptr[:-1], m.nnz - 1)) if m.nnz
          else np.zeros(m.shape[0], np.float32))
    sq[np.diff(m.indptr) == 0] = 0
    norms = np.sqrt(sq).astype(np.float32)
    norms[norms == 0] = 1
    m.data /= np.repeat(norms, np.diff(m.indptr))
    return m


def topn(A, BT, k):
    """Top-k columns of A @ BT per row -> (row, col, rank) arrays."""
    R = sp_matmul_topn(A, BT, top_n=k, threshold=0.05, sort=True, n_threads=N_JOBS).tocsr()
    rows = np.repeat(np.arange(R.shape[0]), np.diff(R.indptr))
    return rows, R.indices.astype(np.int64), (np.arange(R.nnz) - R.indptr[rows]).astype(np.int16)


def rowdot(A, B, ia, ib, chunk=1_000_000):
    """Cosine of row ia[j] of A with row ib[j] of B (rows already L2-normalized)."""
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        a, b = A[ia[s:s + chunk]], B[ib[s:s + chunk]]
        out[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def search_source(X1, X1n, X1T, Xo, Xon, k_fwd, k_name, k_rev):
    """All three channels for one (country, source) block -> DataFrame over local indices."""
    XoT = Xo.T.tocsr()
    r, c, k = topn(X1, XoT, k_fwd)
    del XoT
    fwd = pl.DataFrame({"a": r, "b": c, "rank_c": k})
    if k_name > 0:
        XonT = Xon.T.tocsr()
        r, c, k = topn(X1n, XonT, k_name)
        del XonT
        nam = pl.DataFrame({"a": r, "b": c, "rank_n": k})
    else:
        nam = pl.DataFrame({"a": [], "b": [], "rank_n": []},
                           schema={"a": pl.Int64, "b": pl.Int64, "rank_n": pl.Int16})
    r, c, k = topn(Xo, X1T, k_rev)          # rows are S2/S3 records, cols are S1 records
    rev = pl.DataFrame({"a": c, "b": r, "rank_r": k})
    u = (pl.concat([fwd.select("a", "b"), nam.select("a", "b"), rev.select("a", "b")]).unique()
         .join(fwd, on=["a", "b"], how="left").join(nam, on=["a", "b"], how="left")
         .join(rev, on=["a", "b"], how="left")
         .with_columns(pl.col("rank_c", "rank_n", "rank_r").fill_null(NOT_FOUND).cast(pl.Int16)))
    ia, ib = u["a"].to_numpy(), u["b"].to_numpy()
    return u.with_columns(score=pl.Series(rowdot(X1, Xo, ia, ib)),
                          score_n=pl.Series(rowdot(X1n, Xon, ia, ib)))


def block(split, k_fwd=K_FWD, k_name=K_NAME, k_rev=K_REV):
    t0 = time.time()
    s1 = pl.read_parquet(norm_path(split, 1), columns=["country"] + COLS).with_row_index("i1")
    results = []
    with Pool(N_JOBS) as pool:
        for country in s1["country"].unique().to_list():
            g1 = s1.filter(pl.col("country") == country)
            X1 = hash_docs(g1, pool)
            df, n = doc_freq([X1]), X1.shape[0]
            go = {}
            for s in (2, 3):   # pass 1: document frequencies only
                o = pl.read_parquet(norm_path(split, s), columns=["country"] + COLS).with_row_index("io")
                g = o.filter(pl.col("country") == country)
                go[s] = g if g.height else o   # unseen / mismatched label: search the full pool
                del o
                m = hash_docs(go[s], pool)
                df += doc_freq([m]); n += m.shape[0]
                del m
            idf = make_idf(df, n)
            del df
            X1 = weight_(X1, idf)
            X1n = weight_(hash_docs(g1, pool, name_only=True), idf)
            X1T = X1.T.tocsr()
            print(f"  [{country}] S1 {X1.shape[0]:,} nnz/row {X1.nnz / max(1, X1.shape[0]):.1f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            for s in (2, 3):   # pass 2: re-hash, weight, search, free
                Xo = weight_(hash_docs(go[s], pool), idf)
                Xon = weight_(hash_docs(go[s], pool, name_only=True), idf)
                u = search_source(X1, X1n, X1T, Xo, Xon, k_fwd, k_name, k_rev)
                del Xo, Xon
                results.append(u.select(
                    pl.Series("i1", g1["i1"].to_numpy()[u["a"].to_numpy()].astype(np.int32)),
                    pl.Series("io", go[s]["io"].to_numpy()[u["b"].to_numpy()].astype(np.int32)),
                    pl.lit(s, pl.Int8).alias("src"),
                    "score", "score_n", "rank_c", "rank_n", "rank_r"))
                print(f"    S{s} {go[s].height:,}: {u.height:,} candidates "
                      f"(fwd {int((u['rank_c'] < NOT_FOUND).sum()):,} | name-only new "
                      f"{int(((u['rank_c'] == NOT_FOUND) & (u['rank_n'] < NOT_FOUND)).sum()):,} | "
                      f"reverse-only new {int(((u['rank_c'] == NOT_FOUND) & (u['rank_n'] == NOT_FOUND)).sum()):,}) "
                      f"({time.time() - t0:.0f}s)", flush=True)
                del u
            del X1, X1n, X1T, go
    cand = pl.concat(results)
    cand.write_parquet(cand_path(split))
    print(f"total candidates {cand.height:,} ({cand.height / s1.height:.1f} per S1) in {time.time() - t0:.0f}s")
    return cand


if __name__ == "__main__":
    block(sys.argv[1])
