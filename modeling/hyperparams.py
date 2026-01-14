import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial: optuna.Trial) -> float:
    params = {
        "objective": "binary",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }

    clf = LGBMClassifier(**params)

    pipe = Pipeline([
        ("ensure_float32", ensure_float32),
        ("lgbm", clf),
    ]).set_output(transform="pandas")

    scores = cross_validate(
        pipe, X, y, cv=cv, n_jobs=-1,
        scoring={"pr_auc": "average_precision"},
        return_train_score=False
    )
    return float(scores["test_pr_auc"].mean())

study = optuna.create_study(direction="maximize", study_name="lgbm_pr_auc")
study.optimize(objective, n_trials=200, show_progress_bar=True)

best = study.best_trial
print(f"Best PR AUC: {best.value:.4f}")
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
print(f"Trials: {len(study.trials)}; Best trial #: {best.number}")
