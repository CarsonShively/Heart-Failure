import gradio as gr
import joblib
import pandas as pd

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
        result = "No (Low Risk)" if prediction == 0 else "Yes (High Risk)"
        return f"{result} (Probability of Death: {proba:.2%})"
    
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
