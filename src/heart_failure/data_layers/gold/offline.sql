CREATE SCHEMA IF NOT EXISTS gold;
CREATE OR REPLACE VIEW gold.heart_failure AS
SELECT
  age, anaemia, diabetes, high_blood_pressure, smoking, sex,
  creatinine_phosphokinase, ejection_fraction,
  platelets, serum_creatinine, serum_sodium,
  DEATH_EVENT,
  ef_deficit(ejection_fraction)                         AS ef_deficit,
  sodium_dev(serum_sodium)                              AS sodium_dev,
  hyponatremia(serum_sodium)                            AS hyponatremia,
  hypernatremia(serum_sodium)                           AS hypernatremia,
  creatinine_high(serum_creatinine)                     AS creatinine_high,
  cpk_very_high(creatinine_phosphokinase)               AS cpk_very_high,
  thrombocytopenia(platelets)                           AS thrombocytopenia,
  thrombocytosis(platelets)                             AS thrombocytosis,
  ef_hfref(ejection_fraction)                           AS ef_hfref,
  ef_hfmrEF(ejection_fraction)                          AS ef_hfmrEF,
  ef_hfpef(ejection_fraction)                           AS ef_hfpef,
  comorbidity_count(anaemia, diabetes, high_blood_pressure, smoking) AS comorbidity_count,
  metabolic_risk(diabetes, high_blood_pressure)         AS metabolic_risk,
  anaemia_or_ckd(anaemia, serum_creatinine)             AS anaemia_or_ckd
FROM silver.heart_failure;
