CREATE SCHEMA IF NOT EXISTS silver;

CREATE OR REPLACE VIEW silver.heart_failure AS
WITH
typed AS (
  SELECT
    CAST(age                       AS DOUBLE)  AS age,
    CAST(anaemia                   AS INTEGER) AS anaemia,
    CAST(creatinine_phosphokinase  AS INTEGER) AS creatinine_phosphokinase,
    CAST(diabetes                  AS INTEGER) AS diabetes,
    CAST(ejection_fraction         AS INTEGER) AS ejection_fraction,
    CAST(high_blood_pressure       AS INTEGER) AS high_blood_pressure,
    CAST(platelets                 AS DOUBLE)  AS platelets,
    CAST(serum_creatinine          AS DOUBLE)  AS serum_creatinine,
    CAST(serum_sodium              AS INTEGER) AS serum_sodium,
    CAST(sex                       AS INTEGER) AS sex,
    CAST(smoking                   AS INTEGER) AS smoking,
    CAST("time"                    AS INTEGER) AS "time",
    CAST(DEATH_EVENT               AS INTEGER) AS DEATH_EVENT
  FROM bronze.heart_failure
),
valid AS (
  SELECT *
  FROM typed
 WHERE
    age IS NOT NULL AND isfinite(age) AND
    anaemia IS NOT NULL AND
    creatinine_phosphokinase IS NOT NULL AND
    diabetes IS NOT NULL AND
    ejection_fraction IS NOT NULL AND
    high_blood_pressure IS NOT NULL AND
    platelets IS NOT NULL AND isfinite(platelets) AND
    serum_creatinine IS NOT NULL AND isfinite(serum_creatinine) AND
    serum_sodium IS NOT NULL AND
    sex IS NOT NULL AND
    smoking IS NOT NULL AND
    DEATH_EVENT IS NOT NULL AND
    "time" IS NOT NULL AND
  
    anaemia IN (0,1)
    AND diabetes IN (0,1)
    AND high_blood_pressure IN (0,1)
    AND sex IN (0,1)
    AND smoking IN (0,1)
    AND DEATH_EVENT IN (0,1)
    AND age BETWEEN 18 AND 120
    AND creatinine_phosphokinase BETWEEN 10 AND 20000
    AND ejection_fraction BETWEEN 5 AND 85
    AND platelets BETWEEN 30000 AND 1000000
    AND serum_creatinine BETWEEN 0.2 AND 15.0
    AND serum_sodium BETWEEN 110 AND 170
    AND "time" BETWEEN 0 AND 400
),
deduped AS (
  SELECT DISTINCT
    age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
    high_blood_pressure, platelets, serum_creatinine, serum_sodium,
    sex, smoking, "time", DEATH_EVENT
  FROM valid
)
SELECT *
FROM deduped;