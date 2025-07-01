# Heart Failure Classification

Primary Goal:
Minimize false negatives, as missing an at-risk patient can result in death. In contrast, false positives, while not ideal, only lead to additional screening or short-term stress.

Approach:
Designed and optimized the model specifically for high recall, using domain-informed preprocessing, threshold tuning, and validation to ensure at-risk patients are not missed.

Hugging face deployment demo: https://huggingface.co/spaces/Carson-Shively/deploy-heart-failure

# Dataset:
Source: UCI Heart Failure Clinical Records

Target: DEATH_EVENT (binary)

Imbalance: Present — handled via threshold tuning

Citation:
Davide Chicco, Giuseppe Jurman: Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making 20, 16 (2020). (https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5

# Modeling Overview:

Baseline Model: Logistic Regression

Baseline metrics:

Cross-Validation Performance:
  accuracy: 0.8074 ± 0.0406
  precision: 0.6755 ± 0.0787
  recall: 0.8050 ± 0.0447
  f1: 0.7314 ± 0.0394
  roc_auc: 0.8926 ± 0.0413

Models Tested: Random Forest, XGBoost, LightGBM

Random Forest provided strong performance while aligning with the characteristics of the dataset:
The dataset is relatively small, making simpler models less prone to overfitting

Random Forest achieved comparable or better generalization than more complex models like XGBoost and LightGBM

It offers robustness, interpretability, and consistent performance across validation folds

Simpler models are often better suited to limited data, and Random Forest struck the right balance between performance and generalization.

Model  accuracy Mean  precision Mean  recall Mean   f1 Mean  \
0  Random Forest       0.853901        0.833846     0.701667  0.758867   
1        XGBoost       0.841312        0.781822     0.715000  0.745710   
2       LightGBM       0.832890        0.785761     0.701667  0.732190   

   roc_auc Mean  accuracy Std  precision Std  recall Std    f1 Std  \
0      0.906395      0.053920       0.139600    0.050042  0.081171   
1      0.905821      0.046204       0.096192    0.047770  0.066913   
2      0.902520      0.052184       0.146104    0.088530  0.073878   

   roc_auc Std  
0     0.032229  
1     0.027031  
2     0.027179 

Final metrics tuned for recall:
 Cross-Validated Metrics with Threshold = 0.1027:
  Accuracy : 0.7113
  Precision: 0.5286
  Recall   : 0.9610
  F1 Score : 0.6820
  ROC AUC  : 0.9178
Evaluation on Hold-out Validation Set:
  Accuracy : 0.7333
  Precision: 0.5455
  Recall   : 0.9474
  F1 Score : 0.6923
  ROC AUC  : 0.8691
Confusion Matrix:
(TN/FP)     26     15
(FN/TP)      1     18
Final Model: Random Forest

the two most influential features in the model’s predictions were:
Time × Ejection Fraction
Ejection Fraction × Serum Creatinine

These engineered interaction terms provided a clear performance edge over all other features, indicating strong predictive value and alignment with clinical intuition.
Their impact was consistently validated through SHAP analysis and cross-validation performance.

Summary
The model successfully achieves its primary objective:
 minimizing false negatives, which is critical in a clinical setting where missing high-risk patients could lead to death.
By prioritizing recall, the model helps ensure at-risk individuals are identified, supporting timely intervention and potentially saving lives.
