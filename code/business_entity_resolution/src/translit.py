"""Indic-script name back-transliteration, learned from the training data only.

S2/S3 names for Indian businesses are often the S1 name transliterated word by word into
Devanagari, Tamil, Telugu, Gujarati, Gurmukhi, Kannada or Bengali. After `anyascii`
romanization they read like `yunik srvisej praivet limited` ("Unique Services Private Limited").

  * learned table: align romanized words with S1 words by position on training true pairs whose
    word counts agree (98% do), keep a mapping seen >= 2 times with >= 60% agreement.
  * fallback: for words the table has not seen, match a consonant skeleton against the vocabulary
    of S1 names (unlabeled records), only for skeletons of 3+ consonants to avoid short collisions.

Built once from train (ground truth + S1 names of train and test) by prepare.py and saved to
work/translit.json; applied only to records that contain Indic characters.
"""
import collections
import json
import re

INDIC_RE = re.compile(r"[ऀ-෿]")
_SKEL_RULES = [("ph", "f"), ("sh", "s"), ("ch", "c"), ("th", "t"), ("kh", "k"), ("gh", "g"), ("bh", "b"),
               ("dh", "d"), ("x", "ks"), ("q", "k"), ("c", "k"), ("w", "v"), ("z", "j")]
MIN_COUNT, MIN_SHARE, MIN_SKEL = 2, 0.6, 3


def skel(word):
    w = word.lower()
    for a, b in _SKEL_RULES:
        w = w.replace(a, b)
    w = re.sub(r"[aeiouy]", "", w)
    return re.sub(r"(.)\1+", r"\1", w)


def build(aligned_pairs, s1_vocab_counts):
    """aligned_pairs: iterable of (s1_tokens, romanized_tokens); s1_vocab_counts: Counter of S1 words."""
    cnt = collections.defaultdict(collections.Counter)
    for ta, tb in aligned_pairs:
        if len(ta) == len(tb):
            for x, y in zip(tb, ta):
                cnt[x][y] += 1
    learned = {}
    for k, c in cnt.items():
        (best, n), total = c.most_common(1)[0], sum(c.values())
        if total >= MIN_COUNT and n / total >= MIN_SHARE:
            learned[k] = best   # identity entries kept on purpose: they stop the fallback remapping them
    fallback = {}
    for w, _ in s1_vocab_counts.most_common():
        s = skel(w)
        if len(s) >= MIN_SKEL and w.isalpha():
            fallback.setdefault(s, w)
    return {"learned": learned, "fallback": fallback}


class Translit:
    def __init__(self, table):
        self.learned = table["learned"]
        self.fallback = table["fallback"]

    def map_token(self, t):
        if t in self.learned:
            return self.learned[t]
        s = skel(t)
        if len(s) >= MIN_SKEL:
            return self.fallback.get(s, t)
        return t

    def map_tokens(self, toks):
        return [self.map_token(t) for t in toks]


def save(table, path):
    json.dump(table, open(path, "w", encoding="utf-8"))


def load(path):
    return Translit(json.load(open(path, encoding="utf-8")))
