"""Symmetry-adapted A_n*: natural coordinates in R^{n+1} where S_{n+1} permutes
coordinates. Fundamental weights omega_i = (1^i 0^{n+1-i}) - (i/(n+1)) 1 span the
weight lattice = A_n*. Provides:
  - B_ambient (n x (n+1)): weight-basis rows in ambient coords
  - B_combigeo (n x n): a basis with the same Gram (for combigeo pipeline)
  - to_ambient(c): integer coords c (omega-basis) -> ambient vector (scaled *(n+1) = integer)
  - perm_matrix(sigma): the n x n integer matrix of a coordinate permutation in omega-basis
Verifies Gram vs lattices.Astar and that automorphism matrices are integral."""
import numpy as np, itertools, math
np.set_printoptions(suppress=True, linewidth=160)

def An_star_ambient(n):
    m = n + 1
    W = np.zeros((n, m))
    for i in range(1, n+1):
        W[i-1, :i] = 1.0
        W[i-1, :] -= i / m
    return W                       # rows omega_1..omega_n, each in hyperplane sum=0


def minimal_to_fundamental_transform(n):
    """Coordinate transform from the minimal-weight to fundamental-weight basis.

    Two coordinate conventions for ``A_n^*`` occur in the audit scripts:

    * the columns of ``M_Anstar`` are ``-(n+1) q_i``, where
      ``q_i=e_i-(1/(n+1))1`` are minimal weights;
    * :func:`An_star_ambient` and :func:`lattices.Astar` use the fundamental
      weights ``omega_i=q_1+...+q_i``.

    If a row coordinate ``x_q`` is written in the ``q`` basis, then

        x_omega = x_q U,   q_i = omega_i - omega_{i-1},

    where ``U`` is returned here.  Consequently a column-basis matrix for a
    sublattice transforms as ``C_omega = U.T @ C_q``.
    """
    if n < 1:
        raise ValueError("dimension must be positive")
    transform = np.eye(n, dtype=np.int64)
    for index in range(1, n):
        transform[index, index - 1] = -1
    return transform


def kernel_minimal_to_fundamental(kernel):
    """Convert an ``A_n^*`` sublattice column basis between the conventions."""
    matrix = np.asarray(kernel, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("kernel must be a square matrix")
    return minimal_to_fundamental_transform(len(matrix)).T @ matrix

def gram(W):
    return W @ W.T

def perm_matrix(W, sigma):
    """n x n integer matrix M with (permute columns of W by sigma) = M @ W."""
    Wg = W[:, sigma]
    G = W @ W.T
    M = Wg @ W.T @ np.linalg.inv(G)
    Mr = np.round(M)
    assert np.abs(M - Mr).max() < 1e-8, f"perm matrix not integral: err={np.abs(M-Mr).max()}"
    return Mr.astype(np.int64)

def to_ambient_int(c, W):
    """integer coords c (omega-basis) -> (n+1)-integer vector = (n+1)*ambient."""
    m = W.shape[1]
    v = np.asarray(c, float) @ W        # ambient (fractional, denom m)
    vi = np.round(v * m)
    assert np.abs(v*m - vi).max() < 1e-6
    return vi.astype(np.int64)

if __name__ == "__main__":
    import sys
    sys.path.insert(0,"/Users/mac/Documents/_My_code/Chromatic/audit-data/hd-2026-07")
    from lattices import Astar
    for n in [5, 7]:
        W = An_star_ambient(n)
        G = gram(W)
        # compare normalized Gram spectrum with lattices.Astar(n)
        B2 = Astar(n); G2 = B2 @ B2.T
        s1 = np.sort(np.linalg.eigvalsh(G / abs(np.linalg.det(G))**(1/n)))
        s2 = np.sort(np.linalg.eigvalsh(G2 / abs(np.linalg.det(G2))**(1/n)))
        cyc = list(range(1, n+1)) + [0]            # (0 1 2 ... n) cycle on n+1 coords
        Mg = perm_matrix(W, cyc)
        Mpow = np.linalg.matrix_power(Mg, n+1)
        print(f"n={n}: A_n* ambient built. Gram spectrum matches Astar: "
              f"{np.allclose(s1,s2)} (max dev {np.abs(s1-s2).max():.2e})")
        print(f"   cyclic automorphism M_g integral, order n+1={n+1}: "
              f"M_g^{n+1}==I: {np.array_equal(Mpow, np.eye(n, dtype=np.int64))}")
