"""Stage 1: learn the Indic back-transliteration table from train, then read raw TSVs,
normalize every record in parallel and cache as parquet.

    python prepare.py train test
"""
import collections
import sys
import time
from multiprocessing import Pool

import polars as pl

import normalize
import translit
from config import DATA_DIR, N_JOBS, norm_path, src_path, translit_path
from normalize import normalize_record

CHUNK = 20000


def read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def _tokens(name):
    return normalize.norm_name(name)["name_n"].split()   # called with transliteration disabled


def build_translit():
    """Aligned (S1 words, romanized S2/S3 words) from train true pairs whose S2/S3 name is in an
    Indic script, plus the S1 name vocabulary of train and test (unlabeled) for the fallback."""
    t = time.time()
    gt = (read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv")
          .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
          .explode("matched_entity_ids").rename({"source1_entity_id": "s1", "matched_entity_ids": "o"}))
    s1 = read_tsv(src_path("train", 1)).select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"))
    pairs = []
    for s in (2, 3):
        o = (read_tsv(src_path("train", s)).select("entity_id", "business_name")
             .filter(pl.col("business_name").str.contains(r"[ऀ-෿]")))
        pairs.append(o.join(gt, left_on="entity_id", right_on="o").join(s1, on="s1")
                     .select("n1", pl.col("business_name").alias("no")))
    pairs = pl.concat(pairs)
    vocab = collections.Counter()
    for split in ("train", "test"):
        names = read_tsv(src_path(split, 1)).filter(pl.col("country") == "India")["business_name"]
        vocab.update(tok for n in names.to_list() for tok in _tokens(n))
    with Pool(N_JOBS) as pool:
        a = pool.map(_tokens, pairs["n1"].to_list(), chunksize=5000)
        b = pool.map(_tokens, pairs["no"].to_list(), chunksize=5000)
    table = translit.build(zip(a, b), vocab)
    translit.save(table, translit_path())
    print(f"translit: {pairs.height:,} Indic pairs -> {len(table['learned']):,} learned words, "
          f"{len(table['fallback']):,} fallback skeletons ({time.time() - t:.0f}s)", flush=True)


def _init_worker(path):
    normalize.set_translit(translit.load(path))


def _work(rows):
    return [normalize_record(*r) for r in rows]


def prepare(split):
    for s in (1, 2, 3):
        t = time.time()
        df = read_tsv(src_path(split, s))
        rows = list(zip(df["entity_id"], df["business_name"], df["business_address"], df["country"]))
        del df
        chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
        out = []
        with Pool(N_JOBS, initializer=_init_worker, initargs=(str(translit_path()),)) as pool:
            for part in pool.imap(_work, chunks, chunksize=1):
                out.append(pl.DataFrame(part))
        del rows, chunks
        res = pl.concat(out).with_columns(pl.col("country").fill_null(""))
        res.write_parquet(norm_path(split, s))
        print(f"{split} s{s}: {res.height:,} rows in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    if not translit_path().exists():
        build_translit()
    for split in sys.argv[1:] or ["train", "test"]:
        prepare(split)
