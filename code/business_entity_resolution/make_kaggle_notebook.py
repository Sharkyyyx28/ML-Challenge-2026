"""Builds kaggle/er_pipeline.ipynb: one self-contained notebook that writes src/*.py and runs
the full pipeline (prepare -> blocking -> train -> error analysis -> predict -> validate) on Kaggle.

    python make_kaggle_notebook.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
OUT = HERE.parents[1] / "kaggle" / "er_pipeline.ipynb"
# Pipeline-specific libraries are installed at the exact versions pinned in requirements.txt.
# Kaggle's preinstalled numpy/scipy/scikit-learn/pyarrow are kept (pinning them can break the image);
# their versions are recorded to output/environment.txt on every run.
PINNED = ["polars", "rapidfuzz", "anyascii", "sparse_dot_topn", "lightgbm", "xgboost"]
REQS = dict(l.strip().split("==") for l in (HERE / "requirements.txt").read_text().splitlines()
            if "==" in l and not l.startswith("#"))
PIP = " ".join(f"{p}=={REQS[p]}" for p in PINNED)
NORM_VERSION = int(re.search(r"NORM_VERSION = (\d+)", (SRC / "config.py").read_text()).group(1))
MODULES = ["config.py", "normalize.py", "prepare.py", "blocking.py", "blocking_eval.py", "features.py",
           "metrics.py", "decide.py", "model.py", "train.py", "error_analysis.py", "predict.py", "translit.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": "\n".join(lines)}


cells = [
    md("# Business Entity Resolution — full pipeline\n"
       "Needs: the challenge data added as an input dataset, **Internet ON** (for pip). "
       "CPU session for `MODEL = 'lgb'`, GPU session for `MODEL = 'xgb'`.\n"
       "Outputs land in `/kaggle/working/output/`."),
    code(f"!pip install -q {PIP}",
         "import os, platform, multiprocessing, importlib.metadata as md",
         "os.makedirs('/kaggle/working/src', exist_ok=True); os.makedirs('/kaggle/working/output', exist_ok=True)",
         f"pkgs = {sorted(REQS)!r}",
         "env = [f'python=={platform.python_version()}', f'cpus={multiprocessing.cpu_count()}']",
         "env += [f'{p}=={md.version(p)}' for p in pkgs]",
         "open('/kaggle/working/output/environment.txt', 'w').write('\\n'.join(env) + '\\n')",
         "print('\\n'.join(env))",
         "%cd /kaggle/working"),
]
for m in MODULES:
    cells.append(code(f"%%writefile src/{m}", (SRC / m).read_text(encoding="utf-8")))
cells += [
    md("## 0. Run settings and cache\n"
       "`MODEL = 'xgb'` trains and predicts with XGBoost on the GPU (pick a GPU accelerator); "
       "`'lgb'` uses LightGBM on CPU.\n\n"
       "If a previous run's `norm_*.parquet` / `cand_*.parquet` files are attached as an input dataset they "
       "are reused, skipping normalization and blocking. `KEEP_CACHE = True` leaves them in this run's output "
       "so you can create that dataset from it (Output tab → New Dataset)."),
    code("MODEL = 'xgb'   # 'xgb' needs a GPU accelerator; use 'lgb' on a CPU session",
         "KEEP_CACHE = True",
         "import glob, os",
         f"V = {NORM_VERSION}   # cache version: files from other normalization versions are ignored",
         "WORK = '/kaggle/working/work'; os.makedirs(WORK, exist_ok=True)",
         "cached = [f for pat in (f'norm_v{V}_*.parquet', f'cand_v{V}_*.parquet', f'translit_v{V}.json')",
         "          for f in glob.glob(f'/kaggle/input/**/{pat}', recursive=True)]",
         "for f in cached:",
         "    dst = os.path.join(WORK, os.path.basename(f))",
         "    if not os.path.exists(dst):",
         "        os.symlink(f, dst)",
         "have = lambda n: os.path.exists(os.path.join(WORK, n))",
         "print('cached:', sorted(os.listdir(WORK)))"),
    md("## 1. Learn Indic back-transliteration from train, normalize all records"),
    code("if not all(have(f'norm_v{V}_{sp}_s{i}.parquet') for sp in ('train', 'test') for i in (1, 2, 3)):",
         "    !python src/prepare.py train test"),
    md("## 2. Candidate generation (forward top-20 + reverse top-5 channels)"),
    code("if not have(f'cand_v{V}_train.parquet'):",
         "    !python src/blocking.py train",
         "if not have(f'cand_v{V}_test.parquet'):",
         "    !python src/blocking.py test"),
    code("!python src/blocking_eval.py"),
    md("## 3. Train pair classifier, tune threshold for macro F0.5, analyse errors\n"
       "Rarity and similarity-competition features are computed here, so a v3 cache (same `V`) is reused."),
    code("!python src/train.py --train-folds 0,2 --sample 0.75 --val-fold 1 --val-sample 0.5 --model {MODEL}"),
    code("!python src/error_analysis.py"),
    md("## 4. Predict test + write outputs"),
    code("!python src/predict.py"),
    md("## 5. Validate with the official script"),
    code("import sys; sys.path.insert(0, 'src')",
         "from config import DATA_DIR, OUT_DIR",
         "v = DATA_DIR.parent / 'utils' / 'validate_submission.py'",
         "!python {v} --matching {OUT_DIR}/matching_results.tsv --candidate {OUT_DIR}/candidate_pairs.tsv "
         "--test-dir {DATA_DIR}/test --check-ids"),
    code("!head -5 /kaggle/working/output/matching_results.tsv",
         "# drop symlinked inputs and validation predictions; keep norm/cand files only if KEEP_CACHE",
         "!find /kaggle/working/work -type l -delete; rm -f /kaggle/working/work/val_*.parquet",
         "if not KEEP_CACHE:",
         "    !rm -f /kaggle/working/work/*.parquet",
         "!ls -la /kaggle/working/work /kaggle/working/output"),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", OUT)
