"""The Coxeter--Todd lattice ``K12`` and its Eisenstein sections.

Construction (complex Construction A over ``F4 = Z[omega]/2``):

    K12 = { v in Z[omega]^6 : v mod 2 in H6 },

where ``H6`` is the hexacode, the ``[6,3,4]`` code over ``F4`` with codewords
``(a, b, c, f(1), f(omega), f(omega^2))`` for ``f(x) = a x^2 + b x + c``.

In this embedding ``lambda1^2 = 4`` (756 minimal vectors), ``det = 3^6`` and the
covering radius is ``R^2 = 8/3``, so

    rho = diam / lambda1 = 2 R / lambda1 = sqrt(8/3) = 1.63299...

Everything is validated numerically in :func:`build_k12` (minimum, kissing
number, determinant) and the covering radius is re-measured from below by the
vertex-ascent estimator in the campaign scripts.

Sections: for a minimal vector ``v`` the real hyperplane section
``K12 ∩ v^perp`` is an 11-dimensional lattice, and the *complex* hyperplane
section (orthogonal to both ``v`` and ``omega v``) is a rank-5 Eisenstein
lattice in dimension 10 -- a natural seed for the rank-5 Eisenstein search.
"""

from __future__ import annotations

import math

import numpy as np
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form


# ---------------------------------------------------------------- F4 arithmetic
# elements of F4 = Z[omega]/2 as pairs (x, y) mod 2 meaning x + y*omega,
# with omega^2 = 1 + omega in F4 (consistent with omega^2 = -1 - omega in Z[omega]).

def _f4_add(a, b):
    return ((a[0] + b[0]) % 2, (a[1] + b[1]) % 2)


def _f4_mul(a, b):
    x1, y1 = a
    x2, y2 = b
    return ((x1 * x2 + y1 * y2) % 2, (x1 * y2 + y1 * x2 + y1 * y2) % 2)


def hexacode() -> list[list[tuple[int, int]]]:
    """All 64 codewords of the hexacode ``[6,3,4]_4``."""
    elements = [(0, 0), (1, 0), (0, 1), (1, 1)]
    points = [(1, 0), (0, 1), (1, 1)]           # 1, omega, omega^2

    def f(a, b, c, x):
        return _f4_add(_f4_add(_f4_mul(a, _f4_mul(x, x)), _f4_mul(b, x)), c)

    words = []
    for a in elements:
        for b in elements:
            for c in elements:
                words.append([a, b, c] + [f(a, b, c, x) for x in points])
    return words


def _weight(word) -> int:
    return sum(1 for s in word if s != (0, 0))


# ------------------------------------------------------------------- embedding

def real_embedding() -> np.ndarray:
    """Row transform: coefficient rows ``(x1,y1,..,x6,y6)`` -> real 12-vectors,
    one 2x2 block ``[[1,0],[-1/2, sqrt(3)/2]]`` per complex coordinate."""
    block = np.array([[1.0, 0.0], [-0.5, math.sqrt(3) / 2]])
    T = np.zeros((12, 12))
    for j in range(6):
        T[2 * j:2 * j + 2, 2 * j:2 * j + 2] = block
    return T


def omega_action() -> np.ndarray:
    """Integer matrix of multiplication by ``omega`` on coefficient rows:
    ``(x + y w) w = -y + (x - y) w``."""
    block = np.array([[0, 1], [-1, -1]], dtype=np.int64)   # row (x,y) -> x*(0,1)+y*(-1,-1)
    A = np.zeros((12, 12), dtype=np.int64)
    for j in range(6):
        A[2 * j:2 * j + 2, 2 * j:2 * j + 2] = block
    return A


def build_k12() -> tuple[np.ndarray, np.ndarray]:
    """Integer coefficient basis (12x12, rows) of ``K12`` inside ``Z[omega]^6``
    and its real row basis (``lambda1^2 = 4``, ``det = 27``)."""
    words = hexacode()
    nonzero = [w for w in words if _weight(w)]
    assert len(nonzero) == 63 and min(_weight(w) for w in nonzero) == 4, "hexacode is wrong"

    generators: list[list[int]] = []
    for j in range(12):                       # 2 * (Z-basis of Z[omega]^6)
        row = [0] * 12
        row[j] = 2
        generators.append(row)
    omega = omega_action()
    for abc in (((1, 0), (0, 0), (0, 0)), ((0, 0), (1, 0), (0, 0)), ((0, 0), (0, 0), (1, 0))):
        a, b, c = abc
        points = [(1, 0), (0, 1), (1, 1)]
        word = [a, b, c] + [
            _f4_add(_f4_add(_f4_mul(a, _f4_mul(x, x)), _f4_mul(b, x)), c) for x in points
        ]
        lift = [int(s[k]) for s in word for k in range(2)]
        generators.append(lift)
        generators.append((np.asarray(lift, dtype=np.int64) @ omega).tolist())

    H = hermite_normal_form(Matrix(generators).T)          # columns = basis
    coeff = np.asarray(H.T.tolist(), dtype=np.int64)       # rows = basis
    assert coeff.shape == (12, 12)
    det = abs(int(Matrix(coeff.tolist()).det()))
    assert det == 64, f"index of K12 in Z[omega]^6 must be 64, got {det}"

    basis = coeff @ real_embedding()
    volume = abs(np.linalg.det(basis))
    assert abs(volume - 27.0) < 1e-9, f"covolume must be 27 = sqrt(3^6), got {volume}"
    return coeff, basis


def complex_section(coeff: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Rank-5 Eisenstein section: coefficient rows of ``{x in K12 : x perp_C v}``.

    ``direction`` is a coefficient row of ``v``; orthogonality to the complex
    line means real orthogonality to both ``v`` and ``omega v``.
    """
    T = real_embedding()
    gram2 = np.rint(2 * (T @ T.T)).astype(np.int64)        # integer 2*Gram
    omega = omega_action()
    eq1 = coeff @ gram2 @ np.asarray(direction, dtype=np.int64)
    eq2 = coeff @ gram2 @ (np.asarray(direction, dtype=np.int64) @ omega)
    system = Matrix([eq1.tolist(), eq2.tolist()])
    null = system.nullspace()
    rows = []
    for vec in null:
        denominators = [term.q for term in vec]
        scale = int(np.lcm.reduce([int(q) for q in denominators]))
        rows.append([int(term * scale) for term in vec])
    assert len(rows) == 10, f"complex section must have rank 10, got {len(rows)}"
    return np.asarray(rows, dtype=np.int64) @ coeff


def real_section(coeff: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Rank-11 real hyperplane section ``{x in K12 : <x, v> = 0}``."""
    T = real_embedding()
    gram2 = np.rint(2 * (T @ T.T)).astype(np.int64)
    eq = coeff @ gram2 @ np.asarray(direction, dtype=np.int64)
    null = Matrix([eq.tolist()]).nullspace()
    rows = []
    for vec in null:
        denominators = [term.q for term in vec]
        scale = int(np.lcm.reduce([int(q) for q in denominators]))
        rows.append([int(term * scale) for term in vec])
    assert len(rows) == 11, f"real section must have rank 11, got {len(rows)}"
    return np.asarray(rows, dtype=np.int64) @ coeff


def rotate_to_dimension(rows_real: np.ndarray) -> np.ndarray:
    """Rotate a rank-k row lattice living in R^12 down to R^k (QR of the row space)."""
    rows_real = np.asarray(rows_real, dtype=float)
    q, _ = np.linalg.qr(rows_real.T)                       # 12 x k orthonormal
    return rows_real @ q
