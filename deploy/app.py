from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
import numpy as np
import gradio as gr
import joblib
import pandas as pd

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._transform_output = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['ef_to_creatinine'] = X['ejection_fraction'] / X['serum_creatinine']
        X['ef_drop_per_age'] = (100 - X['ejection_fraction']) / X['age']
        X['ef_per_time'] = X['ejection_fraction'] / X['time']
        X['time_x_ef'] = X['time'] * X['ejection_fraction']
        X['creatinine_x_ef'] = X['serum_creatinine'] * X['ejection_fraction']
        X['time_x_creatinine'] = X['time'] * X['serum_creatinine']
        X.drop(columns=['diabetes', 'anaemia', 'smoking', 'sex', 'high_blood_pressure'], errors='ignore', inplace=True)

        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        if self._transform_output == "pandas":
            return pd.DataFrame(X, columns=X.columns, index=X.index)
        else:
            return X

    def set_output(self, transform=None):
        self._transform_output = transform
        return self

class CoerceToFloat(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].astype(float)
        return X

class NumericImputer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        X = X.copy()
        self.medians_ = {
            col: X[col].median(skipna=True)
            for col in self.columns
        }
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].fillna(self.medians_[col])
        return X

pipeline = joblib.load("model_pipeline.pkl")
threshold = 0.1027

def predict_from_input(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                       high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex,
                       smoking, time):
    try:
        input_dict = {
            "age": age,
            "anaemia": anaemia,
            "creatinine_phosphokinase": creatinine_phosphokinase,
            "diabetes": diabetes,
            "ejection_fraction": ejection_fraction,
            "high_blood_pressure": high_blood_pressure,
            "platelets": platelets,
            "serum_creatinine": serum_creatinine,
            "serum_sodium": serum_sodium,
            "sex": sex,
            "smoking": smoking,
            "time": time
        }

        df = pd.DataFrame([input_dict])
        proba = pipeline.predict_proba(df)[0][1]
        prediction = int(proba >= threshold)
        result = "(Not at Risk)" if prediction == 0 else "(At Risk)"
        return f"{result} (Probability: {proba:.2%})"
    
    except Exception as e:
        return f"Error during prediction: {str(e)}"

iface = gr.Interface(
    fn=predict_from_input,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio([0, 1], label="Anaemia"),
        gr.Number(label="Creatinine Phosphokinase"),
        gr.Radio([0, 1], label="Diabetes"),
        gr.Number(label="Ejection Fraction"),
        gr.Radio([0, 1], label="High Blood Pressure"),
        gr.Number(label="Platelets"),
        gr.Number(label="Serum Creatinine"),
        gr.Number(label="Serum Sodium"),
        gr.Radio([0, 1], label="Sex"),
        gr.Radio([0, 1], label="Smoking"),
        gr.Number(label="Time (Follow-up in days)")
    ],
    outputs="text",
    title="Heart Failure Risk Prediction",
    description="Predict whether a patient is at risk of death using clinical features. Model trained with a custom probability threshold optimized for recall."
)

if __name__ == "__main__":
    iface.launch()
