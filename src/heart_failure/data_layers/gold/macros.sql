CREATE OR REPLACE MACRO flag(expr) AS CASE WHEN (expr) THEN 1 ELSE 0 END;

CREATE OR REPLACE MACRO ef_deficit(ef) AS (100 - ef);
CREATE OR REPLACE MACRO sodium_dev(na) AS (na - 140);

CREATE OR REPLACE MACRO hyponatremia(na)       AS flag(na < 135);
CREATE OR REPLACE MACRO hypernatremia(na)      AS flag(na > 145);
CREATE OR REPLACE MACRO creatinine_high(scr)   AS flag(scr >= 1.3);
CREATE OR REPLACE MACRO cpk_very_high(cpk)     AS flag(cpk >= 1000);
CREATE OR REPLACE MACRO thrombocytopenia(plts) AS flag(plts < 150000);
CREATE OR REPLACE MACRO thrombocytosis(plts)   AS flag(plts > 450000);

CREATE OR REPLACE MACRO ef_hfref(ef)  AS flag(ef < 40);
CREATE OR REPLACE MACRO ef_hfmrEF(ef) AS flag(ef BETWEEN 40 AND 49);
CREATE OR REPLACE MACRO ef_hfpef(ef)  AS flag(ef >= 50);

CREATE OR REPLACE MACRO comorbidity_count(anaemia, diabetes, hbp, smoking)
AS (COALESCE(anaemia,0) + COALESCE(diabetes,0) + COALESCE(hbp,0) + COALESCE(smoking,0));

CREATE OR REPLACE MACRO metabolic_risk(diabetes, hbp)
AS flag(COALESCE(diabetes,0) = 1 OR COALESCE(hbp,0) = 1);

CREATE OR REPLACE MACRO anaemia_or_ckd(anaemia, scr)
AS flag(COALESCE(anaemia,0) = 1 OR (scr >= 1.3));