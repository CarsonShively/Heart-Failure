# Heart Failure Classification

Primary Goal:
Minimize false negatives, as missing an at-risk patient can result in death. In contrast, false positives, while not ideal, only lead to additional screening or short-term stress.

Approach:
Designed and optimized the model specifically for high recall, using domain-informed preprocessing, threshold tuning, and validation to ensure at-risk patients are not missed.

Hugging face deployment demo: https://huggingface.co/spaces/Carson-Shively/deploy-heart-failure

## Dataset:
Source: UCI Heart Failure Clinical Records

Target: DEATH_EVENT (binary)

Imbalance: Present

Citation:
Davide Chicco, Giuseppe Jurman: Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making 20, 16 (2020). (https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5

## Modeling Overview:

### Baseline Model: Logistic Regression

Cross-Validation Performance:

  accuracy: 0.8074 ± 0.0406
  
  precision: 0.6755 ± 0.0787
  
  recall: 0.8050 ± 0.0447
  
  f1: 0.7314 ± 0.0394
  
  roc_auc: 0.8926 ± 0.0413

Solid baseline metrics and gained valuable insights into the data for further modeling.

### Models Tested: Random Forest, XGBoost, LightGBM

Random Forest provided the strongest performance while aligning with the characteristics of the dataset:

-The dataset is relatively small, making simpler models less prone to overfitting.

### Final model tuned for recall:

 #### Cross-Validated Metrics with Threshold = 0.1027:
 
  Accuracy : 0.7113
  
  Precision: 0.5286
  
  Recall   : 0.9610
  
  F1 Score : 0.6820
  
  ROC AUC  : 0.9178
  
#### Evaluation on Hold-out Validation Set:

  Accuracy : 0.7333
  
  Precision: 0.5455
  
  Recall   : 0.9474
  
  F1 Score : 0.6923
  
  ROC AUC  : 0.8691
  
Confusion Matrix:

(TN/FP)     26     15

(FN/TP)      1     18

Final Model: Random Forest

## Feature Analysis Overview
The two most influential features in the model’s predictions were:

-Time × Ejection Fraction

-Ejection Fraction × Serum Creatinine

These engineered interaction terms provided a clear performance edge over all other features, indicating strong predictive value and alignment with clinical intuition. Their impact was consistently validated through SHAP analysis and cross-validation performance.

## The model successfully achieves its primary objective:
-Minimizing false negatives, which is critical in a clinical setting where missing high-risk patients could lead to death.

-By prioritizing recall, the model helps ensure at-risk individuals are identified, supporting timely intervention and potentially saving lives.
