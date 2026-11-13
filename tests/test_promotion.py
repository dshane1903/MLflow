import json

from mlflow.tracking import MlflowClient

from mle101.config import ALIAS, MODEL_NAME
from mle101.data import make_claims
from mle101.promotion import compare_and_promote
from mle101.training import train_frame


def test_auto_promotion_bootstraps_once_and_preserves_equal_champion(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLE101_TRACKING_URI", uri)
    summary_dir = tmp_path / "artifacts"
    train_frame(make_claims(2500, seed=42), "synthetic:seed=42", summary_dir)
    summary_path = summary_dir / "training_summary.json"
    assert json.loads(summary_path.read_text())["recommended"] == "logistic"
    evaluation = make_claims(600, seed=77)

    first = compare_and_promote(summary_path, evaluation, "synthetic:seed=77")
    assert first["promoted"] is True
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    version = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version
    assert first["new_champion_version"] == version

    second = compare_and_promote(summary_path, evaluation, "synthetic:seed=77")
    assert second["promoted"] is False
    assert second["beat_champion"] is False
    assert second["new_champion_version"] == version
    assert client.get_model_version_by_alias(MODEL_NAME, ALIAS).version == version
