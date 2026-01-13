import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import json, joblib
from huggingface_hub import hf_hub_download

train_data = hf_hub_download(
        repo_id="Carson-Shively/heart-failure",
        filename="data/gold/gold_uf.parquet",
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

BEST_THR = 0.4891

pipe.fit(X, y)

joblib.dump(pipe, "heart_failure.pkl") 

with open("feature_columns_hf.json", "w") as f:
    json.dump(list(X.columns), f)