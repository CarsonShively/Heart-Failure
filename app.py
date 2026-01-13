import sys
import types
import json
import threading
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import duckdb
import joblib
import gradio as gr
from pydantic import BaseModel, ConfigDict, ValidationError
import threading, types, duckdb
from importlib.resources import files
import os

state = types.SimpleNamespace()
_duck_lock = threading.Lock()

def X_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.where(np.isfinite(out), np.nan)
    return out.astype(np.float32)

m = sys.modules.get('__main__') or types.ModuleType('__main__')
sys.modules['__main__'] = m
for name, fn in {
    'X_to_float32': X_to_float32,
}.items():
    setattr(m, name, fn)

REPO_ID = "Carson-Shively/heart-failure"
REV = "main" 

ROOT = Path(__file__).resolve().parent

MODEL_PKL_PATH  = ROOT / "artifacts" / "heart_failure.pkl"
FEATS_JSON_PATH = ROOT / "artifacts" / "feature_columns.json"
SQL_PKG = "heart_failure.data_layers.gold"
MACROS_SQL_FILE = "macros.sql"
ONLINE_SQL_FILE = "online.sql"

def _read_pkg_sql(pkg: str, filename: str) -> str:
    return (files(pkg) / filename).read_text(encoding="utf-8")

def init_connection(con) -> None:
    con.execute(_read_pkg_sql(SQL_PKG, MACROS_SQL_FILE))


def load_model_and_schema():
    try:
        model = joblib.load(MODEL_PKL_PATH)
        feature_columns = json.loads(FEATS_JSON_PATH.read_text(encoding="utf-8"))
        return model, feature_columns
    except Exception as e:
        raise RuntimeError("Failed model/schema load") from e


def init_app():
    state.con = duckdb.connect()
    init_connection(state.con)
    state.MODEL, state.FEATURE_COLUMNS = load_model_and_schema()
    state.FEATURE_COLUMNS = tuple(state.FEATURE_COLUMNS)
    with _duck_lock:
        state.con.execute(_read_pkg_sql(SQL_PKG, ONLINE_SQL_FILE))

def collect_raw_inputs(
    age, anaemia, diabetes, high_blood_pressure, sex, smoking,
    creatinine_phosphokinase, ejection_fraction, platelets,
    serum_creatinine, serum_sodium
):
    raw = {
        "age": age,
        "anaemia": anaemia,
        "diabetes": diabetes,
        "high_blood_pressure": high_blood_pressure,
        "sex": sex,
        "smoking": smoking,
        "creatinine_phosphokinase": creatinine_phosphokinase,
        "ejection_fraction": ejection_fraction,
        "platelets": platelets,
        "serum_creatinine": serum_creatinine,
        "serum_sodium": serum_sodium,
    }
    return raw, "Collected raw inputs. (Not validated yet.)"

class OnlineRequired(BaseModel):
    model_config = ConfigDict(strict=True)  
    anaemia: Literal[0, 1]
    diabetes: Literal[0, 1]
    high_blood_pressure: Literal[0, 1]
    sex: Literal[0, 1]
    smoking: Literal[0, 1]
    age: float
    creatinine_phosphokinase: float
    ejection_fraction: float
    platelets: float
    serum_creatinine: float
    serum_sodium: float

def run_gold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a single-row DataFrame with online inputs and returns the Gold features
    by invoking gold_online_row(age, anaemia, diabetes, hbp, smoking, sex, cpk, ef, plts, scr, na).
    """
    row = df.iloc[0]

    age   = float(row["age"])
    ana   = int(row["anaemia"])
    diab  = int(row["diabetes"])
    hbp   = int(row["high_blood_pressure"])
    smok  = int(row["smoking"])
    sex   = int(row["sex"])

    cpk   = int(row["creatinine_phosphokinase"])
    ef    = int(row["ejection_fraction"])
    plts  = float(row["platelets"])
    scr   = float(row["serum_creatinine"])
    na    = int(row["serum_sodium"])

    sql = """
    SELECT *
    FROM gold_online_row(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    with _duck_lock:
        return state.con.execute(sql, [
            age, ana, diab, hbp, smok, sex,
            cpk, ef, plts, scr, na
        ]).fetchdf()

def make_one_row_df(payload) -> pd.DataFrame:
    return pd.DataFrame([payload.model_dump()])

def _prepare_X_for_model(X_gold: pd.DataFrame) -> pd.DataFrame:
    cols = list(state.FEATURE_COLUMNS)  
    missing = [c for c in cols if c not in X_gold.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return X_gold.loc[:, cols]

BEST_THR = 0.4891  

def format_risk(proba: float, thr: float = BEST_THR) -> str:
    pct = proba * 100.0
    label = "At risk" if proba >= thr else "Low risk"
    return f"{pct:.1f}% — {label}"

def predict_from_raw(
    age, anaemia, diabetes, high_blood_pressure, sex, smoking,
    creatinine_phosphokinase, ejection_fraction,
    platelets, serum_creatinine, serum_sodium
) -> str:
    raw, _ = collect_raw_inputs(
        age, anaemia, diabetes, high_blood_pressure, sex, smoking,
        creatinine_phosphokinase, ejection_fraction,
        platelets, serum_creatinine, serum_sodium
    )
    try:
        payload = OnlineRequired.model_validate(raw)
    except ValidationError as e:
        raise gr.Error(e.errors()[0]["msg"])
    X_gold = run_gold(make_one_row_df(payload))
    X = _prepare_X_for_model(X_gold)
    proba = float(state.MODEL.predict_proba(X)[:, 1][0])
    return format_risk(proba)

def predict(
    age, anaemia, diabetes, high_blood_pressure, sex, smoking,
    creatinine_phosphokinase, ejection_fraction,
    platelets, serum_creatinine, serum_sodium
):
    return predict_from_raw(
        age, anaemia, diabetes, high_blood_pressure, sex, smoking,
        creatinine_phosphokinase, ejection_fraction,
        platelets, serum_creatinine, serum_sodium
    )


with gr.Blocks() as demo:
    gr.Markdown("## Heart Failure Inputs")

    with gr.Row():
        with gr.Column():
            age = gr.Slider(18, 120, step=1, value=60, label="Age")
            sex = gr.Radio([0, 1], value=1, label="Sex (0=female, 1=male)")
            smoking = gr.Radio([0, 1], value=0, label="Smoking")
            anaemia = gr.Radio([0, 1], value=0, label="Anaemia")
            diabetes = gr.Radio([0, 1], value=0, label="Diabetes")
            high_blood_pressure = gr.Radio([0, 1], value=0, label="High Blood Pressure")

        with gr.Column():
            ejection_fraction = gr.Slider(5, 85, step=1, value=35, label="Ejection Fraction")
            creatinine_phosphokinase = gr.Slider(10, 20000, step=1, value=250, label="Creatinine Phosphokinase")
            platelets = gr.Slider(30000, 1_000_000, step=1000, value=250_000, label="Platelets")
            serum_creatinine = gr.Slider(0.2, 15.0, step=0.01, value=1.1, label="Serum Creatinine")
            serum_sodium = gr.Slider(110, 170, step=1, value=138, label="Serum Sodium")

    submit = gr.Button("Predict")
    outputs = gr.Textbox(label="Risk")

    submit.click(
        fn=predict,
        inputs=[
            age, anaemia, diabetes, high_blood_pressure, sex, smoking,
            creatinine_phosphokinase, ejection_fraction, platelets,
            serum_creatinine, serum_sodium
        ],
        outputs = outputs
    )

if __name__ == "__main__":
    init_app()

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
    )
