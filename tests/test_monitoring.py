from mle101.data import make_claims
from mle101.monitoring import make_reference, psi


def test_psi_detects_deliberate_population_shift():
    regular = make_claims(2500, seed=42)
    shifted = make_claims(1000, seed=99, shifted=True)
    ref = make_reference(regular)
    item = ref["numeric"]["initial_estimate"]
    import numpy as np

    edges = np.r_[-np.inf, item["edges"], np.inf]
    counts = np.histogram(shifted["initial_estimate"], bins=edges)[0]
    value = psi(item["proportions"], (counts / len(shifted)).tolist())
    assert value > 0.2

