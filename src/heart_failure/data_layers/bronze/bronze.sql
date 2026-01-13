INSTALL httpfs;
LOAD httpfs;

CREATE SCHEMA IF NOT EXISTS bronze;

CREATE OR REPLACE VIEW bronze.heart_failure AS
SELECT *
FROM read_parquet(
  'https://huggingface.co/datasets/Carson-Shively/heart-failure/resolve/main/data/bronze/bronze_hf.parquet'
);