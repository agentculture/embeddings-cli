"""Tests for the shared cosine guards.

The case that matters is the tiny-but-nonzero norm: it slips past both
``norm == 0.0`` and ``norm > 0.0``, divides cleanly, and produces a finite
number with no information in it. A ``nan`` filter does not catch it, so it
sorts into a neighbor list looking like a real measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from embeddings._numeric import MIN_NORM, cosine_distances, is_degenerate, safe_norms

# --- the blind spot the old checks shared ---------------------------------


def test_a_tiny_nonzero_norm_is_degenerate() -> None:
    """The exact case `== 0.0` and `> 0.0` both let through."""
    tiny = MIN_NORM / 10
    assert is_degenerate(tiny)
    # ...and the checks it defeats, stated explicitly so the regression is legible.
    assert tiny != 0.0
    assert tiny > 0.0


def test_a_tiny_row_is_reported_as_maximally_distant_not_as_a_number() -> None:
    vectors = np.array([[1.0, 0.0], [1e-300, 1e-300]])
    distances = cosine_distances(vectors, np.array([1.0, 0.0]))
    assert distances[0] == pytest.approx(0.0)
    assert distances[1] == pytest.approx(2.0)
    assert np.all(np.isfinite(distances))


def test_no_nan_or_inf_survives_a_degenerate_row() -> None:
    vectors = np.array([[0.0, 0.0], [1e-320, 0.0], [3.0, 4.0]])
    distances = cosine_distances(vectors, np.array([3.0, 4.0]))
    assert np.all(np.isfinite(distances))
    assert distances[2] == pytest.approx(0.0)


def test_a_degenerate_query_does_not_raise_or_poison_the_result() -> None:
    """Callers own the query-level error; this helper must still stay finite."""
    distances = cosine_distances(np.array([[1.0, 0.0], [0.0, 1.0]]), np.zeros(2))
    assert np.all(np.isfinite(distances))


# --- ordinary behaviour ---------------------------------------------------


def test_distances_are_correct_and_bounded() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    distances = cosine_distances(vectors, np.array([1.0, 0.0]))
    assert distances[0] == pytest.approx(0.0)
    assert distances[1] == pytest.approx(1.0)
    assert distances[2] == pytest.approx(2.0)
    assert np.all(distances >= 0.0) and np.all(distances <= 2.0)


def test_an_identical_vector_never_yields_a_negative_distance() -> None:
    """Float drift can push similarity a few ulps past 1.0; clipping absorbs it."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        v = rng.normal(size=64)
        assert cosine_distances(v.reshape(1, -1), v)[0] >= 0.0


def test_magnitude_does_not_change_direction() -> None:
    """Cosine is scale-invariant — a rescaled row must not move."""
    vectors = np.array([[1.0, 2.0, 3.0]])
    base = cosine_distances(vectors, np.array([3.0, 2.0, 1.0]))
    scaled = cosine_distances(vectors * 1e6, np.array([3.0, 2.0, 1.0]))
    assert base == pytest.approx(scaled)


def test_safe_norms_are_always_safe_to_divide_by() -> None:
    norms, degenerate = safe_norms(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert np.all(norms >= MIN_NORM)
    assert degenerate.tolist() == [True, False]
    assert norms[1] == pytest.approx(5.0)


def test_the_norm_product_cannot_underflow() -> None:
    """The floor's justification: the denominator stays above float64's min normal."""
    assert MIN_NORM * MIN_NORM > np.finfo(np.float64).tiny
