"""Generate and validate *synthetic* claim records."""

import numpy as np
import pandas as pd

NUMERIC = [
    "initial_estimate",
    "deductible",
    "policy_tenure_months",
    "prior_claims",
    "reported_delay_days",
]
CATEGORICAL = ["incident_type", "region"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "high_cost"
INCIDENT_TYPES = ("water", "theft", "fire", "wind")
REGIONS = ("northeast", "south", "midwest", "west")


def make_claims(n: int = 2500, seed: int = 42, shifted: bool = False) -> pd.DataFrame:
    """Simulate information available at first notice of loss.

    The target is a later outcome. It is generated from the features plus noise,
    and is deliberately excluded from the scoring inputs.
    """
    if n < 10:
        raise ValueError("n must be at least 10")
    rng = np.random.default_rng(seed)
    estimate = rng.lognormal(mean=8.4, sigma=0.75, size=n)
    if shifted:
        estimate *= 1.8  # A changed incoming population for the monitoring lesson.
    incident = rng.choice(INCIDENT_TYPES, n, p=[0.42, 0.23, 0.13, 0.22])
    region = rng.choice(REGIONS, n)
    deductible = rng.choice([500, 1000, 2500], n, p=[0.35, 0.45, 0.20])
    tenure = rng.integers(1, 181, n)
    prior = np.minimum(rng.poisson(0.55, n), 5)
    delay = rng.integers(0, 31, n)

    # A plausible, inspectable data-generating process; no real claims are used.
    log_odds = (
        -2.5
        + 0.00018 * estimate
        + 0.80 * (incident == "fire")
        + 0.26 * prior
        + 0.018 * delay
        - 0.002 * (deductible - 500) / 10
        - 0.003 * tenure
    )
    probability = 1 / (1 + np.exp(-np.clip(log_odds, -30, 30)))
    target = rng.binomial(1, probability)
    frame = pd.DataFrame(
        {
            "claim_id": [f"SYN-{seed}-{i:06d}" for i in range(n)],
            "initial_estimate": np.round(estimate, 2),
            "deductible": deductible,
            "policy_tenure_months": tenure,
            "prior_claims": prior,
            "reported_delay_days": delay,
            "incident_type": incident,
            "region": region,
            TARGET: target,
        }
    )
    validate_claims(frame, require_target=True)
    return frame


def validate_claims(frame: pd.DataFrame, require_target: bool = False) -> None:
    required = {"claim_id", *FEATURES}
    if require_target:
        required.add(TARGET)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    if (
        frame.empty
        or frame["claim_id"].isna().any()
        or frame["claim_id"].astype(str).str.strip().eq("").any()
        or frame["claim_id"].duplicated().any()
    ):
        raise ValueError("claim_id must be present, unique, and nonempty")
    for column in NUMERIC:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0).any():
            raise ValueError(f"{column} must contain finite nonnegative numbers")
    allowed = {"incident_type": INCIDENT_TYPES, "region": REGIONS}
    for column, choices in allowed.items():
        if not frame[column].isin(choices).all():
            raise ValueError(f"{column} contains an unknown or missing category")
    if require_target and not frame[TARGET].isin([0, 1]).all():
        raise ValueError("high_cost must contain only 0 or 1")

