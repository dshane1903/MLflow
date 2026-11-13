# Databricks notebook source
# Run this from the repository's Git folder on serverless compute.

# COMMAND ----------

import sys
from pathlib import Path

source_dir = next(
    (path / "src" for path in (Path.cwd(), Path.cwd().parent) if (path / "src" / "mle101").exists()),
    None,
)
if source_dir is None:
    raise RuntimeError("Open this notebook from the repository Git folder, or set source_dir to its src path")
sys.path.insert(0, str(source_dir))

from mle101.data import make_claims  # noqa: E402
from mle101.spark_data import build_delta_tables  # noqa: E402

# COMMAND ----------

catalog = spark.sql("SELECT current_catalog()").first()[0]
if catalog == "hive_metastore":
    raise RuntimeError("Select your workspace's Unity Catalog catalog before running this notebook")

schema = "default"
volume = "mle101"
spark.sql(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{schema}`.`{volume}`")
volume_path = Path(f"/Volumes/{catalog}/{schema}/{volume}")
source_csv = volume_path / "synthetic_claims.csv"
make_claims(2500, seed=42).to_csv(source_csv, index=False)

# COMMAND ----------

result = build_delta_tables(
    spark,
    str(source_csv),
    f"{catalog}.{schema}.claims_bronze",
    f"{catalog}.{schema}.claims_silver",
)
print(result)
display(spark.table(result["silver_table"]))
display(spark.sql(f"DESCRIBE HISTORY {result['silver_table']}"))
