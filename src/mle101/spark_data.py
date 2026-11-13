"""PySpark validation and Delta tables for the synthetic claims feed.

The local CSV lesson remains available. This module uses the same contract on
Databricks or in a local Spark session with the optional ``spark`` dependencies.
"""

import os
import re

import pandas as pd

from .data import CATEGORICAL, FEATURES, INCIDENT_TYPES, NUMERIC, REGIONS, TARGET

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*\Z")
_INTEGER_COLUMNS = {"policy_tenure_months", "prior_claims", "reported_delay_days"}


def _checked_table_name(name: str) -> str:
    parts = [part[1:-1] if part.startswith("`") and part.endswith("`") else part for part in name.split(".")]
    if not 1 <= len(parts) <= 3 or not all(_IDENTIFIER.fullmatch(part) for part in parts):
        raise ValueError("table name must be one to three simple SQL identifiers")
    return ".".join(f"`{part}`" for part in parts)


def spark_session():
    """Use Databricks' session or make a small Delta-enabled local session."""
    from pyspark.sql import SparkSession

    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    builder = SparkSession.builder.appName("mle101-claims")
    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        return builder.getOrCreate()
    from delta import configure_spark_with_delta_pip

    builder = (
        builder.master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def validate_claims_spark(frame, require_target: bool = True) -> None:
    """Reject invalid rows before publishing the curated Delta table."""
    from pyspark.sql import functions as F

    required = {"claim_id", *FEATURES}
    if require_target:
        required.add(TARGET)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    if frame.limit(1).count() == 0:
        raise ValueError("claim feed must not be empty")
    if frame.where(F.col("claim_id").isNull() | (F.trim(F.col("claim_id")) == "")).limit(1).count():
        raise ValueError("claim_id must be present and nonempty")
    if frame.groupBy("claim_id").count().where(F.col("count") > 1).limit(1).count():
        raise ValueError("claim_id must be unique")
    for column in NUMERIC:
        value = F.expr(f"try_cast(`{column}` as double)")
        invalid = value.isNull() | F.isnan(value) | (F.abs(value) == float("inf")) | (value < 0)
        if column in _INTEGER_COLUMNS:
            invalid = invalid | (value != F.floor(value))
        if frame.where(invalid).limit(1).count():
            raise ValueError(f"{column} must contain finite nonnegative numbers")
    allowed = {"incident_type": INCIDENT_TYPES, "region": REGIONS}
    for column in CATEGORICAL:
        if frame.where(F.col(column).isNull() | ~F.col(column).isin(*allowed[column])).limit(1).count():
            raise ValueError(f"{column} contains an unknown or missing category")
    if require_target:
        value = F.expr(f"try_cast(`{TARGET}` as double)")
        if frame.where(value.isNull() | ~value.isin(0.0, 1.0)).limit(1).count():
            raise ValueError("high_cost must contain only 0 or 1")


def curated_claims(frame):
    """Select stable, typed columns after distributed contract checks."""
    from pyspark.sql import functions as F

    validate_claims_spark(frame, require_target=True)
    columns = [F.col("claim_id").cast("string").alias("claim_id")]
    columns += [F.col(name).cast("double").alias(name) for name in NUMERIC if name not in _INTEGER_COLUMNS]
    columns += [F.col(name).cast("long").alias(name) for name in NUMERIC if name in _INTEGER_COLUMNS]
    columns += [F.col(name).cast("string").alias(name) for name in CATEGORICAL]
    columns += [F.col(TARGET).cast("int").alias(TARGET)]
    return frame.select(*columns)


def build_delta_tables(spark, input_path: str, bronze_table: str, silver_table: str) -> dict:
    """Persist raw rows, then publish validated model-ready rows."""
    bronze_table = _checked_table_name(bronze_table)
    silver_table = _checked_table_name(silver_table)
    if bronze_table == silver_table:
        raise ValueError("bronze and silver tables must differ")
    raw = spark.read.option("header", True).csv(input_path)
    raw.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(bronze_table)
    curated = curated_claims(raw)
    curated.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(silver_table)
    return {"bronze_table": bronze_table, "silver_table": silver_table, "rows": curated.count()}


def training_frame_from_delta(spark, table_name: str, max_rows: int = 100_000) -> pd.DataFrame:
    """Collect only this bounded teaching dataset for the sklearn estimator."""
    table_name = _checked_table_name(table_name)
    frame = spark.table(table_name)
    validate_claims_spark(frame, require_target=True)
    rows = frame.limit(max_rows + 1).count()
    if rows > max_rows:
        raise ValueError(f"Delta table exceeds the {max_rows:,}-row local sklearn limit")
    return frame.select("claim_id", *FEATURES, TARGET).toPandas()


def train_from_delta(spark, table_name: str, output_dir=None) -> dict:
    from .training import train_frame

    table_name = _checked_table_name(table_name)
    frame = training_frame_from_delta(spark, table_name)
    version = spark.sql(f"DESCRIBE HISTORY {table_name}").select("version").first()[0]
    return train_frame(frame, f"delta:{table_name}@v{version}", output_dir)
