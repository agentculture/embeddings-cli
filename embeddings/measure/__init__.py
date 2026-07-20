"""embeddings.measure — the measurement apparatus every structural claim must pass through.

Modules here compute the evidence behind an Atlas entry, never the entry's
interpretation. :mod:`embeddings.measure.stability` is the first: it turns "we
clustered the space and found seven regions" into "we clustered it N times under
derived seeds and under bootstrap resampling, and here is how much the answer
agreed with itself" — or, when only one run exists, into the explicit label
``stability unverified``.

The rule these modules exist to make mechanical (design note §3): *a region that
only exists at one parameter setting is a finding about the parameters, not about
the model.*
"""

from __future__ import annotations

from embeddings.measure.stability import (
    MEASURE_ARI,
    VERDICT_MODERATE,
    VERDICT_STABLE,
    VERDICT_UNSTABLE,
    VERDICT_UNVERIFIED,
    AgreementDistribution,
    StabilityResult,
    adjusted_rand_index,
    bootstrap_agreement,
    measure_stability,
    pairwise_agreement,
    seed_reruns,
)

__all__ = [
    "MEASURE_ARI",
    "VERDICT_MODERATE",
    "VERDICT_STABLE",
    "VERDICT_UNSTABLE",
    "VERDICT_UNVERIFIED",
    "AgreementDistribution",
    "StabilityResult",
    "adjusted_rand_index",
    "bootstrap_agreement",
    "measure_stability",
    "pairwise_agreement",
    "seed_reruns",
]
