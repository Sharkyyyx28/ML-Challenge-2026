"""Day-1 baseline for the Business Entity Resolution challenge.

Pipeline: normalize -> TF-IDF char n-gram blocking (per country, top-K from S2 and S3)
-> pairwise similarity features -> gradient boosting classifier -> threshold tuned for
macro F0.5 on a held-out split of train -> writes output/matching_results.tsv and
output/candidate_pairs.tsv.

Run from the student_resource/ directory:
    python baseline.py
"""
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = Path("dataset")
OUT = Path("output")
TOP_K = 10          # candidates per source (S2 and S3 each) per S1 entity
SEED = 42

ABBR = {
    "corp": "corporation", "co": "company", "inc": "incorporated", "ltd": "limited",
    "pvt": "private", "llc": "llc", "intl": "international", "mfg": "manufacturing",
    "svc": "services", "svcs": "services", "assoc": "associates", "bros": "brothers",
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard", "ln": "lane",
    "dr": "drive", "hwy": "highway", "nr": "near", "opp": "opposite", "apt": "apartment",
    "fl": "floor", "ste": "suite", "bldg": "building", "sect": "sector", "sec": "sector",
    "n": "north", "s": "south", "e": "east", "w": "west",
}
LEGAL = {"corporation", "company", "incorporated", "limited", "private", "llc", "llp",
         "plc", "the", "sarl", "sas", "sa", "gmbh"}


def norm(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(ABBR.get(t, t) for t in text.split())


def core_name(n):
    return " ".join(t for t in n.split() if t not in LEGAL)


def load(split):
    dfs = []
    for i in (1, 2, 3):
        df = pd.read_csv(DATA / split / f"{split}_source{i}.tsv", sep="\t", dtype=str,
                         keep_default_na=False)
        dfs.append(df)
    for df in dfs:
        df["name_n"] = df["business_name"].map(norm)
        df["addr_n"] = df["business_address"].map(norm)
        df["core"] = df["name_n"].map(core_name)
        df["country_n"] = df["country"].str.strip().str.lower()
        df["text"] = df["core"] + " | " + df["addr_n"]
    return dfs


def block(s1, others, top_k=TOP_K):
    """Return {s1_id: [candidate ids]} using TF-IDF cosine within the same country."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True)
    vec.fit(pd.concat([s1["text"]] + [o["text"] for o in others]))
    cands = {i: [] for i in s1["entity_id"]}
    for country, g1 in s1.groupby("country_n"):
        X1 = vec.transform(g1["text"])
        ids1 = g1["entity_id"].to_numpy()
        for o in others:
            go = o[o["country_n"] == country]
            if go.empty:           # unseen / mismatched country label: fall back to everything
                go = o
            Xo = vec.transform(go["text"])
            idso = go["entity_id"].to_numpy()
            k = min(top_k, len(idso))
            for start in range(0, X1.shape[0], 2000):
                sims = (X1[start:start + 2000] @ Xo.T).toarray()
                top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
                for r, row in enumerate(top):
                    cands[ids1[start + r]].extend(idso[row])
    return cands


def jacc(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b) if a and b else 0.0


def nums(s):
    return set(re.findall(r"\d+", s))


def features(pairs, lut):
    rows = []
    for s1_id, c_id in pairs:
        a, b = lut[s1_id], lut[c_id]
        na, nb = nums(a["addr_n"]), nums(b["addr_n"])
        rows.append([
            fuzz.ratio(a["name_n"], b["name_n"]),
            fuzz.token_sort_ratio(a["name_n"], b["name_n"]),
            fuzz.token_set_ratio(a["name_n"], b["name_n"]),
            fuzz.partial_ratio(a["core"], b["core"]),
            fuzz.ratio(a["core"], b["core"]),
            jacc(a["core"], b["core"]),
            fuzz.token_set_ratio(a["addr_n"], b["addr_n"]),
            fuzz.partial_ratio(a["addr_n"], b["addr_n"]),
            jacc(a["addr_n"], b["addr_n"]),
            len(na & nb) / len(na | nb) if na and nb else -1,
            float(a["country_n"] == b["country_n"]),
            float(c_id.startswith("S3")),
            len(a["core"]), len(b["core"]),
        ])
    return np.array(rows, dtype=float)


def make_pairs(cands):
    return [(s, c) for s, cs in cands.items() for c in dict.fromkeys(cs)]


def f05(pred, true):
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def decide(pairs, probs, thr):
    out = {}
    for (s, c), p in zip(pairs, probs):
        out.setdefault(s, set())
        if p >= thr:
            out[s].add(c)
    return out


def write(path, col, ids, mapping):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s in ids:
            f.write(f"{s}\t{','.join(sorted(mapping.get(s, [])))}\n")


def main():
    rng = np.random.default_rng(SEED)
    tr1, tr2, tr3 = load("train")
    gt = pd.read_csv(DATA / "train" / "train_ground_truth.tsv", sep="\t", dtype=str,
                     keep_default_na=False)
    truth = {r.source1_entity_id: set(x for x in r.matched_entity_ids.split(",") if x)
             for r in gt.itertuples()}

    lut = {r["entity_id"]: r for df in (tr1, tr2, tr3) for r in df.to_dict("records")}
    cands = block(tr1, [tr2, tr3])
    pairs = make_pairs(cands)
    total = sum(len(v) for v in truth.values())
    hit = sum(1 for s, c in pairs if c in truth.get(s, ()))
    print(f"train blocking: {len(pairs)} pairs, recall ceiling {hit / max(total, 1):.4f}")

    X = features(pairs, lut)
    y = np.array([c in truth.get(s, ()) for s, c in pairs], dtype=int)

    s1_ids = tr1["entity_id"].to_numpy()
    val_ids = set(rng.choice(s1_ids, size=len(s1_ids) // 5, replace=False))
    is_val = np.array([s in val_ids for s, _ in pairs])

    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=SEED)
    model.fit(X[~is_val], y[~is_val])
    val_pairs = [p for p, v in zip(pairs, is_val) if v]
    val_probs = model.predict_proba(X[is_val])[:, 1]

    best_thr, best = 0.5, -1
    for thr in np.arange(0.3, 0.96, 0.02):
        pred = decide(val_pairs, val_probs, thr)
        score = np.mean([f05(pred.get(s, set()), truth.get(s, set())) for s in val_ids])
        if score > best:
            best_thr, best = thr, score
    print(f"validation macro F0.5 = {best:.4f} at threshold {best_thr:.2f}")

    model.fit(X, y)  # refit on all training pairs

    te1, te2, te3 = load("test")
    lut = {r["entity_id"]: r for df in (te1, te2, te3) for r in df.to_dict("records")}
    cands = block(te1, [te2, te3])
    pairs = make_pairs(cands)
    probs = model.predict_proba(features(pairs, lut))[:, 1]
    pred = decide(pairs, probs, best_thr)

    OUT.mkdir(exist_ok=True)
    ids = te1["entity_id"].tolist()
    write(OUT / "candidate_pairs.tsv", "candidate_entity_ids", ids,
          {s: list(dict.fromkeys(c)) for s, c in cands.items()})
    write(OUT / "matching_results.tsv", "matched_entity_ids", ids, pred)
    print(f"wrote {len(ids)} rows; {sum(bool(v) for v in pred.values())} entities with matches")


if __name__ == "__main__":
    main()
