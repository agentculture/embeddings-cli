"""Shared numeric guards for the cosine geometry every reader verb computes.

Why this module exists
----------------------
Four modules independently needed "cosine distance, but do not divide by a
degenerate norm", and each solved it differently:

- ``compare`` clamped with ``np.maximum(norms, 1e-12)`` — correct;
- ``neighbors`` tested ``norm == 0.0`` — exact float equality;
- ``directions`` tested ``norms > 0.0`` — strict inequality;
- ``gaps`` divided first and filtered non-finite results afterwards.

The last three share a blind spot that the equality check makes easiest to
see: a norm that is *tiny but not exactly zero* passes the guard and is then
divided by. The quotient is finite, so no ``nan`` filter catches it, but its
relative error is dominated by representation noise — the vector's direction
is not meaningfully determined at that magnitude. The result is a plausible
number with no information in it, sorted confidently into a neighbor list.

Testing floats for exact equality is the visible symptom; the defect is that
"is this vector usable" is a question about *magnitude*, which needs a
tolerance, not an identity test.

Choosing the tolerance
----------------------
:data:`MIN_NORM` is an absolute floor of ``1e-12`` on a vector's L2 norm.
Two properties motivate that value:

- **No underflow downstream.** Cosine divides by a *product* of two norms.
  With both operands at or above ``1e-12`` the product is at or above
  ``1e-24``, comfortably above float64's smallest normal (~2.2e-308), so the
  denominator can never silently flush to zero.
- **Real embeddings are nowhere near it.** Embedding vectors are unit-length
  or O(1); a norm below ``1e-12`` is not a small vector but a degenerate one
  (all-zero, or numerically indistinguishable from it).

The floor is absolute rather than relative because the inputs have a known
scale. If this tool ever ingests vectors at wildly different scales, the
right change is a scale-relative tolerance, not a smaller constant.
"""

from __future__ import annotations

import numpy as np

#: Smallest L2 norm treated as carrying a usable direction. See module
#: docstring for the derivation; do not lower it without revisiting the
#: underflow argument.
MIN_NORM = 1e-12


def is_degenerate(norm: float | np.ndarray) -> bool | np.ndarray:
    """True where an L2 ``norm`` is too small to define a direction.

    Use this instead of ``norm == 0.0`` or ``norm > 0.0``: it catches the
    tiny-but-nonzero case those miss, and it states the intent — the question
    is whether the vector is usable, not whether it is identically zero.
    """
    return np.asarray(norm) <= MIN_NORM if isinstance(norm, np.ndarray) else norm <= MIN_NORM


def safe_norms(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise L2 norms of ``vectors``, plus a mask of the degenerate rows.

    Returns ``(norms, degenerate)``. ``norms`` is clamped to :data:`MIN_NORM`
    so it is always safe to divide by; ``degenerate`` marks the rows whose
    result the caller must discard rather than trust.
    """
    raw = np.linalg.norm(np.asarray(vectors, dtype=np.float64), axis=1)
    degenerate = raw <= MIN_NORM
    return np.maximum(raw, MIN_NORM), degenerate


def cosine_distances(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine distance (``1 - similarity``, range ``[0, 2]``) from ``query`` to each row.

    A degenerate row is reported as maximally distant (``2.0``) rather than
    raising or yielding ``nan``: it is already a broken input, and poisoning
    the sort with a ``nan`` would make every neighbor list containing it
    unusable. The caller decides whether to surface it.

    The query itself is *not* checked here — a degenerate query makes the whole
    call meaningless rather than one row of it, so callers raise on that with
    their own message and remediation.
    """
    matrix = np.asarray(vectors, dtype=np.float64)
    vector = np.asarray(query, dtype=np.float64)
    norms, degenerate = safe_norms(matrix)
    query_norm = max(float(np.linalg.norm(vector)), MIN_NORM)

    similarity = (matrix @ vector) / (norms * query_norm)
    similarity = np.where(degenerate, -1.0, similarity)
    # Clip absorbs the float drift that can push an identical vector's
    # similarity a few ulps past 1.0, which would surface as a small negative
    # distance.
    return 1.0 - np.clip(similarity, -1.0, 1.0)
