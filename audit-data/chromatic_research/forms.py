"""Cholesky parametrization of normalized Gram forms.

Optimizers search over the lower triangle of a Cholesky factor; the resulting
form is normalized to unit determinant so that the objective is scale-free.
This replaces thirteen near-identical copies that had drifted apart in both
tolerance (1e-10 vs 1e-12) and dimension handling (hard-coded ``** 0.25``).
"""

from __future__ import annotations

import numpy as np


def norm_gram(basis: np.ndarray) -> np.ndarray:
    """Gram matrix of `basis`, scaled to determinant 1."""
    basis = np.asarray(basis, dtype=float)
    gram = basis @ basis.T
    dim = gram.shape[0]
    return gram / abs(np.linalg.det(gram)) ** (1.0 / dim)


def pack(form: np.ndarray) -> np.ndarray:
    """Lower triangle of the Cholesky factor of `form`, as a flat vector."""
    form = np.asarray(form, dtype=float)
    dim = form.shape[0]
    return np.linalg.cholesky(form)[np.tril_indices(dim)]


def unpack(vector: np.ndarray, dim: int, tol: float = 1e-12) -> np.ndarray | None:
    """Inverse of :func:`pack`; ``None`` when the resulting form is degenerate."""
    factor = np.zeros((dim, dim))
    factor[np.tril_indices(dim)] = np.asarray(vector, dtype=float)
    form = factor @ factor.T
    determinant = abs(np.linalg.det(form))
    if determinant <= tol:
        return None
    return form / determinant ** (1.0 / dim)
