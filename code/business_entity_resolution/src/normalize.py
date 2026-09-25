"""Record normalization shared by blocking and feature extraction.

Everything here is rule-based and country-agnostic by default; country-specific
abbreviation tables are applied only when the country label matches, and unknown
countries fall back to the generic table.
"""
import re

from anyascii import anyascii

from translit import INDIC_RE

# Indic back-transliteration (translit.Translit), installed by prepare.py; None = disabled.
_TRANSLIT = None


def set_translit(tr):
    global _TRANSLIT
    _TRANSLIT = tr

# ---------------------------------------------------------------- legal suffixes
# Token -> legal-form class. Transliterated Indic forms (via anyascii) included.
LEGAL = {
    "llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cie": "co", "ltd": "ltd", "limited": "ltd", "limitet": "ltd",
    "pvt": "pvt", "private": "pvt", "praivet": "pvt", "piraivet": "pvt", "pra": "pvt", "li": "ltd",
    "llp": "llp", "elelpi": "llp", "pc": "pc", "pllc": "pllc", "lp": "lp", "plc": "plc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci",
    "snc": "snc", "scp": "scp", "selarl": "selarl", "scop": "scop", "gmbh": "gmbh",
}
# Tokens dropped from the "core" name (legal forms + function words + web noise).
STOP = set(LEGAL) | {
    "the", "and", "of", "ms", "m", "s", "www", "com", "net", "org", "in", "fr", "de", "du", "des",
    "la", "le", "les", "et", "a", "an", "l", "d", "li",
}

DOMAIN_RE = re.compile(r"\b(?:https?://)?(?:www\.)?([a-z0-9\-]{3,})\.(?:com|net|org|co\.in|in|fr|biz|info|us)\b")
ID_JUNK_RE = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)|#\s*\d+|\bid\s*:\s*\d+")
DOT_IN_WORD_RE = re.compile(r"(?<=[a-z])\.(?=[a-z])")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
DIGITS_RE = re.compile(r"\d+")

# ---------------------------------------------------------------- addresses
ADDR_GENERIC = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "cir": "circle", "pl": "place", "pkwy": "parkway",
    "hwy": "highway", "trl": "trail", "ter": "terrace", "sq": "square", "mt": "mount", "ft": "fort",
    "apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor", "bldg": "building", "bldng": "building",
    "nr": "near", "opp": "opposite", "mkt": "market", "sec": "sector", "ph": "phase", "dist": "district",
    "n": "north", "s": "south", "e": "east", "w": "west", "hno": "", "no": "", "unit": "", "po": "",
    "null": "", "none": "", "na": "", "nan": "",
}
ADDR_BY_COUNTRY = {
    "france": {**ADDR_GENERIC, "r": "rue", "bd": "boulevard", "q": "quai", "imp": "impasse",
               "all": "allee", "che": "chemin", "chem": "chemin", "rte": "route", "fbg": "faubourg",
               "st": "saint", "ste": "sainte", "pl": "place", "av": "avenue", "bis": "bis", "ter": "ter",
               "n": "", "e": "", "s": ""},
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
_US_STATE_RE = re.compile(r"\b(" + "|".join(sorted(US_STATES, key=len, reverse=True)) + r")\s*$")


def _ascii_lower(s):
    if not s:
        return ""
    return anyascii(s).lower()


def norm_name(raw):
    """Returns dict with normalized name pieces."""
    s = _ascii_lower(raw)
    domains = DOMAIN_RE.findall(s)
    s = DOMAIN_RE.sub(lambda m: " " + m.group(1).replace("-", " ") + " ", s)
    s = ID_JUNK_RE.sub(" ", s).replace("&", " and ").replace("m/s", " ")
    s = DOT_IN_WORD_RE.sub("", s)  # l.l.c -> llc, s.a.s -> sas
    toks = NON_ALNUM_RE.sub(" ", s).split()
    if _TRANSLIT is not None and raw and INDIC_RE.search(raw):
        toks = _TRANSLIT.map_tokens(toks)
    legal = sorted({LEGAL[t] for t in toks if t in LEGAL})
    core, prev = [], None
    for t in toks:
        if t in STOP or t == prev:
            continue
        core.append(t)
        prev = t
    return {
        "name_n": " ".join(toks),
        "core": " ".join(core),
        "compact": "".join(core),
        "legal": "|".join(legal),
        "is_domain": bool(domains),
        "name_nonascii": bool(raw) and not raw.isascii(),
        "name_nums": " ".join(str(int(x)) for x in DIGITS_RE.findall(" ".join(core))),
    }


def norm_addr(raw, country):
    s = _ascii_lower(raw)
    c = (country or "").strip().lower()
    table = ADDR_BY_COUNTRY.get(c, ADDR_GENERIC)
    if c in ("us", "usa", "united states"):
        s = _US_STATE_RE.sub(lambda m: US_STATES[m.group(1)], s.strip())
    nums = [str(int(x)) for x in DIGITS_RE.findall(s)]
    toks = []
    for t in NON_ALNUM_RE.sub(" ", s.replace("n°", " ")).split():
        if t.isdigit():
            continue
        m = re.fullmatch(r"(\d+)(st|nd|rd|th|bis|ter)?", t)
        if m:
            continue
        t = re.sub(r"\d+", "", t) if not t.isalpha() else t   # 25th / 3rt -> th / rt noise
        t = table.get(t, t)
        if len(t) >= 2:
            toks.append(t)
    missing = not toks and not nums
    return {"addr_n": " ".join(toks), "anums": " ".join(nums), "addr_missing": missing}


def normalize_record(eid, name, addr, country):
    out = {"entity_id": eid, "country": country}
    out.update(norm_name(name))
    out.update(norm_addr(addr, country))
    return out
