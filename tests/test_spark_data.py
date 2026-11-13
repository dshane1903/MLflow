import uuid

import pytest

from mle101.data import make_claims, validate_claims

pyspark = pytest.importorskip("pyspark")

from mle101.spark_data import build_delta_tables, spark_session, training_frame_from_delta  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    session = spark_session()
    yield session
    session.stop()


def test_pyspark_delta_pipeline_rejects_bad_data_before_curated_table(spark, tmp_path):
    suffix = uuid.uuid4().hex[:8]
    bronze, silver = f"bronze_{suffix}", f"silver_{suffix}"
    source = tmp_path / "claims.csv"
    make_claims(50, seed=12).to_csv(source, index=False)

    result = build_delta_tables(spark, str(source), bronze, silver)
    assert result["rows"] == 50
    assert spark.table(bronze).count() == 50
    frame = training_frame_from_delta(spark, silver)
    validate_claims(frame, require_target=True)
    assert len(frame) == 50

    bad = make_claims(50, seed=12)
    bad.loc[0, "initial_estimate"] = -1
    bad.to_csv(source, index=False)
    with pytest.raises(ValueError, match="initial_estimate"):
        build_delta_tables(spark, str(source), bronze, silver)
    assert spark.table(silver).count() == 50
