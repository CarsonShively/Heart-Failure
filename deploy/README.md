# Heart Failure Risk Prediction

This app predicts whether a patient is at high risk of death due to heart failure using clinical data.

## Dataset  
Davide Chicco, Giuseppe Jurman:  
*Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone.*  
BMC Medical Informatics and Decision Making 20, 16 (2020)  
[Read the paper](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5)

## Inputs  
Age, Anaemia, Creatinine Phosphokinase, Diabetes, Ejection Fraction, High Blood Pressure, Platelets, Serum Creatinine, Serum Sodium, Sex, Smoking, Time  
Includes engineered features to enhance performance.

## Outputs  
- High/Low risk classification  
- Probability of death (custom threshold applied)

## Model  
- Trained with `RandomForestClassifier`  
- Hyperparameters tuned with `Optuna`  
- Deployed using Hugging Face Spaces

---

Built by **Carson Shively** to demonstrate real-world ML deployment.
