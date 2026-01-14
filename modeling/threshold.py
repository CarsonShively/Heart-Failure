import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.base import clone
from sklearn.metrics import f1_score, average_precision_score
from lightgbm import LGBMClassifier
import optuna
from huggingface_hub import hf_hub_download

train_data = hf_hub_download(
        repo_id="Carson-Shively/heart-failure",
        filename="data/gold/gold_hf.parquet",
        repo_type="dataset",
        revision="main",
    )

df = pd.read_parquet(train_data)

X = df.drop(columns=["DEATH_EVENT"])
y = df["DEATH_EVENT"].astype(int)

def X_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.where(np.isfinite(out), np.nan) 
    return out.astype(np.float32)

ensure_float32 = FunctionTransformer(X_to_float32, validate=False, feature_names_out="one-to-one")

lgbm = LGBMClassifier(
    objective="binary",
    n_estimators=572,
    learning_rate=0.0065330908418549626,
    num_leaves=255,
    max_depth=1,
    min_child_samples=14,
    subsample=0.5746476312662783,
    colsample_bytree=0.8011920396952499,
    reg_alpha=1.0479155453124996e-05,
    reg_lambda=3.672437853380864e-07,
    class_weight="balanced",
    random_state=42,
    verbosity=-1,
    n_jobs=-1,
)

pipe = Pipeline([
    ("ensure_float32", ensure_float32),
    ("lgbm", lgbm),
]).set_output(transform="pandas")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_validate(
    pipe, X, y, cv=cv, n_jobs=-1,
    scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "accuracy": "accuracy", "f1": "f1"},
    return_train_score=False
)

oof_proba = np.zeros(len(y), dtype=float)
for tr, va in cv.split(X, y):
    fold_pipe = clone(pipe)
    fold_pipe.fit(X.iloc[tr], y.iloc[tr])
    oof_proba[va] = fold_pipe.predict_proba(X.iloc[va])[:, 1]

print(f"OOF PR AUC: {average_precision_score(y, oof_proba):.4f}")

def objective_thr(trial: optuna.Trial) -> float:
    thr = trial.suggest_float("threshold", 0.01, 0.99)
    pred = (oof_proba >= thr).astype(int)
    return f1_score(y, pred)

study = optuna.create_study(direction="maximize", study_name="threshold_oof_f1")
study.optimize(objective_thr, n_trials=200, show_progress_bar=True)

best_thr = study.best_params["threshold"]
print(f"Best threshold (F1, OOF): {best_thr:.4f}")

