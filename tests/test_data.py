import pytest

from mle101.data import make_claims, validate_claims


def test_generated_data_is_reproducible_and_valid():
    first = make_claims(100, seed=7)
    second = make_claims(100, seed=7)
    assert first.equals(second)
    validate_claims(first, require_target=True)


def test_validation_catches_impossible_claim_amount():
    frame = make_claims(20)
    frame.loc[0, "initial_estimate"] = -10
    with pytest.raises(ValueError, match="initial_estimate"):
        validate_claims(frame)

