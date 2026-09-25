"""Paths and knobs. Override paths with env vars so the same code runs locally or on Kaggle."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ON_KAGGLE = Path("/kaggle/input").exists()


def _find_kaggle_data():
    hits = [p for p in Path("/kaggle/input").rglob("train_source1.tsv") if "__MACOSX" not in str(p)]
    if not hits:
        raise FileNotFoundError("train_source1.tsv not found under /kaggle/input - add the dataset as input")
    return hits[0].parent.parent


DATA_DIR = Path(os.environ.get("ER_DATA_DIR") or (_find_kaggle_data() if ON_KAGGLE else ROOT / "student_resource" / "dataset"))
WORK_DIR = Path(os.environ.get("ER_WORK_DIR") or ("/kaggle/working/work" if ON_KAGGLE else ROOT / "work"))
OUT_DIR = Path(os.environ.get("ER_OUT_DIR") or ("/kaggle/working/output" if ON_KAGGLE else ROOT / "output"))
N_JOBS = int(os.environ.get("ER_N_JOBS", os.cpu_count() or 4))
SEED = 42

WORK_DIR.mkdir(parents=True, exist_ok=True)


def src_path(split, s):
    return DATA_DIR / split / f"{split}_source{s}.tsv"


# Bump when normalization changes: cached norm/cand files from older versions are then ignored.
NORM_VERSION = 3


def norm_path(split, s):
    return WORK_DIR / f"norm_v{NORM_VERSION}_{split}_s{s}.parquet"


def cand_path(split):
    return WORK_DIR / f"cand_v{NORM_VERSION}_{split}.parquet"


def translit_path():
    return WORK_DIR / f"translit_v{NORM_VERSION}.json"
