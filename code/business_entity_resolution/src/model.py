"""Gradient-boosted pair classifier: LightGBM (CPU) or XGBoost (GPU when available).
Both are Apache-2.0/MIT licensed. The backend is recorded in model_meta.json."""
import numpy as np

from config import N_JOBS, SEED, WORK_DIR

LGB_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=N_JOBS, verbose=-1, seed=SEED)
XGB_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", eta=0.1, max_depth=0,
                  max_leaves=127, grow_policy="lossguide", min_child_weight=5, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", max_bin=256, seed=SEED)
MAX_ROUNDS = {"lgb": 700, "xgb": 2000}   # CPU LightGBM capped for run time; GPU XGBoost can afford more
EARLY_STOP = 50


def _cuda_available():
    try:
        import subprocess
        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except OSError:
        return False


def fit(backend, train_data, Xva, yva, feats):
    """train_data: [Xtr, ytr] list; cleared as soon as the library has built its own binned copy,
    so the raw float matrix is freed before boosting starts."""
    Xtr, ytr = train_data
    if backend == "lgb":
        import lightgbm as lgb
        dtr = lgb.Dataset(Xtr, ytr, feature_name=feats, free_raw_data=True).construct()
        dva = lgb.Dataset(Xva, yva, reference=dtr).construct()
        train_data.clear()
        del Xtr, ytr
        return lgb.train(LGB_PARAMS, dtr, num_boost_round=MAX_ROUNDS["lgb"], valid_sets=[dva],
                         callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(100)])
    import xgboost as xgb
    params = dict(XGB_PARAMS, device="cuda" if _cuda_available() else "cpu", nthread=N_JOBS)
    print("xgboost device:", params["device"], flush=True)
    dtr = xgb.QuantileDMatrix(Xtr, ytr, feature_names=feats, max_bin=params["max_bin"])
    dva = xgb.QuantileDMatrix(Xva, yva, feature_names=feats, ref=dtr)
    train_data.clear()
    del Xtr, ytr
    return xgb.train(params, dtr, num_boost_round=MAX_ROUNDS["xgb"], evals=[(dva, "valid")],
                     early_stopping_rounds=EARLY_STOP, verbose_eval=100)


def predict(backend, model, X):
    if backend == "lgb":
        return model.predict(X, num_threads=N_JOBS)
    import xgboost as xgb
    return model.predict(xgb.DMatrix(X, feature_names=model.feature_names),
                         iteration_range=(0, model.best_iteration + 1))


def save(backend, model):
    path = WORK_DIR / ("model.txt" if backend == "lgb" else "model.ubj")
    model.save_model(str(path))
    return path.name


def load(backend):
    if backend == "lgb":
        import lightgbm as lgb
        return lgb.Booster(model_file=str(WORK_DIR / "model.txt"))
    import xgboost as xgb
    m = xgb.Booster()
    m.load_model(str(WORK_DIR / "model.ubj"))
    m.set_param({"device": "cuda" if _cuda_available() else "cpu", "nthread": N_JOBS})
    return m


def importance(backend, model, feats):
    if backend == "lgb":
        return sorted(zip(model.feature_importance("gain"), feats), reverse=True)
    g = model.get_score(importance_type="total_gain")
    return sorted(((g.get(f, 0.0), f) for f in feats), reverse=True)


def best_iteration(backend, model):
    return int(model.best_iteration)


def as_matrix(df, feats):
    return np.ascontiguousarray(df.select(feats).to_numpy().astype(np.float32, copy=False))
