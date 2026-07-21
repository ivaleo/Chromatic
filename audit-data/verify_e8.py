"""Verify chi(E^8) <= 2401 via Eisenstein E8 (tetracode construction) and alpha = 3+omega,
testing the conjectured law D(alpha*Lambda)^2 = (7/3)*lambda1^2  =>  d = sqrt(7/3)/ratio.

E8 = {x in E^4 : x mod theta in tetracode C4=[4,2,3]_3}; index 2401 sublattice alpha*E8.
Expected: 240 minimal vectors (120 pairs), 120 relevant pairs, ratio = sqrt(2), d^2 = 7/6.
"""
import math
import time
from fractions import Fraction

import numpy as np
from scipy.optimize import minimize

OMEGA = complex(-0.5, math.sqrt(3) / 2)
ALPHA = 3 + OMEGA

def gram_schmidt(B):
    n = len(B); Bs = B.astype(float).copy(); mu = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            mu[i, j] = B[i] @ Bs[j] / (Bs[j] @ Bs[j]); Bs[i] -= mu[i, j] * Bs[j]
    return Bs, mu

def lll(B, delta=0.75):
    B = B.astype(float).copy(); n = len(B); k = 1
    while k < n:
        Bs, mu = gram_schmidt(B)
        for j in range(k - 1, -1, -1):
            q = round(mu[k, j])
            if q: B[k] -= q * B[j]
        Bs, mu = gram_schmidt(B)
        if Bs[k] @ Bs[k] >= (delta - mu[k, k - 1] ** 2) * (Bs[k - 1] @ Bs[k - 1]): k += 1
        else:
            B[[k, k - 1]] = B[[k - 1, k]]; k = max(k - 1, 1)
    return B

def vectors_within(B, bound):
    B = lll(B); Bs, mu = gram_schmidt(B)
    bn2 = np.array([b @ b for b in Bs]); n = len(B); out = []; coeffs = [0] * n
    def descend(level, partial2):
        if level == 0:
            for c in coeffs:
                if c > 0: break
                if c < 0: return
            else: return
            v = np.array(coeffs, float) @ B
            if v @ v <= bound * bound + 1e-9: out.append(v)
            return
        j = level - 1
        center = sum(coeffs[i] * mu[i, j] for i in range(j + 1, n))
        rem = bound * bound - partial2
        if rem < -1e-9: return
        rad = math.sqrt(max(0.0, rem) / bn2[j])
        for c in range(math.ceil(-center - rad - 1e-9), math.floor(-center + rad + 1e-9) + 1):
            coeffs[j] = c
            descend(j, partial2 + (c + center) ** 2 * bn2[j])
        coeffs[j] = 0
    descend(n, 0.0)
    return out

def relevant_vectors(B, bound):
    n = len(B); Bl = lll(B); Binv = np.linalg.inv(Bl)
    vecs = vectors_within(Bl, bound)
    coset = {}
    for v in vecs:
        key = tuple(np.rint(v @ Binv).astype(int) % 2)
        if key == (0,) * n: continue
        coset.setdefault(key, []).append(v)
    assert len(coset) == 2 ** n - 1, f"only {len(coset)} of {2**n-1} cosets — enlarge bound"
    rel = []
    for vs in coset.values():
        m = min(np.linalg.norm(v) for v in vs)
        ties = [v for v in vs if np.linalg.norm(v) <= m + 1e-9]
        if len(ties) == 1: rel.append(ties[0])
    return rel

def dist_to_cell(p, A, b, tol=1e-12):
    if np.all(A @ p <= b + 1e-12): return 0.0
    cons = [{"type": "ineq", "fun": lambda x: b - A @ x, "jac": lambda x: -A}]
    res = minimize(lambda x: (x - p) @ (x - p), np.zeros(len(p)),
                   jac=lambda x: 2 * (x - p), constraints=cons,
                   method="SLSQP", options={"maxiter": 800, "ftol": tol})
    assert res.success, res.message
    return float(np.linalg.norm(res.x - p))

# ---- build E8 (tetracode) in integer (a,b)-coordinates, then realify ----
# z = a + b*omega per complex coordinate; real embedding 1->(1,0), omega->(-1/2, sqrt3/2)
gens_int = []
c1 = [1, 1, 1, 0]; c2 = [0, 1, 2, 1]                      # tetracode generators over F3
for c in (c1, c2):
    row_a = []; row_b = []
    for x in c:
        row_a += [x, 0]; row_b += [0, x]                   # c and omega*c
    gens_int.append(row_a); gens_int.append(row_b)
for j in range(4):                                          # theta*e_j and omega*theta*e_j
    row = [0] * 8; row[2 * j] = 1; row[2 * j + 1] = 2      # theta = 1 + 2omega
    gens_int.append(row)
    row2 = [0] * 8; row2[2 * j] = -2; row2[2 * j + 1] = -1 # omega*theta = -2 - omega
    gens_int.append(row2)

from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form
H = hermite_normal_form(Matrix(gens_int).T)                 # columns span -> HNF
Bint = np.array(H.T.tolist(), float)                        # rows = integer-coord basis
E1 = np.array([1.0, 0.0]); EW = np.array([-0.5, math.sqrt(3) / 2])
T = np.zeros((8, 8))
for j in range(4):
    T[2 * j, 2 * j:2 * j + 2] = E1
    T[2 * j + 1, 2 * j:2 * j + 2] = EW
B = Bint @ T                                                # real 8x8 row-basis of E8-scaled

det = abs(np.linalg.det(B)); det_E4 = (math.sqrt(3) / 2) ** 4
print(f"[E8] index in E^4 = {det / det_E4:.3f} (expect 9)", flush=True)
lam1 = min(np.linalg.norm(v) for v in vectors_within(B, 1.8))
nmin = 2 * sum(1 for v in vectors_within(B, lam1 + 1e-9))
print(f"[E8] lambda1^2 = {lam1**2:.6f} (expect 3), #min vectors = {nmin} (expect 240)", flush=True)

R = lam1 / math.sqrt(2)      # ratio sqrt(2) for E8 (unimodular, standard)
diam = 2 * R
rel = relevant_vectors(B, 2 * R * 1.02 + 1e-9)
print(f"[E8] relevant pairs: {len(rel)} (expect 120)", flush=True)
A = np.array([w for w in rel] + [-w for w in rel])
bvec = np.array([w @ w / 2 for w in rel] * 2)

# sublattice alpha*E8: real matrix of multiplication by alpha
Aalpha = np.kron(np.eye(4), np.array([[ALPHA.real, -ALPHA.imag], [ALPHA.imag, ALPHA.real]]))
sub = B @ Aalpha.T
print(f"[E8] index [L : alpha L] = {abs(np.linalg.det(sub))/det:.1f} (expect 2401)", flush=True)

t0 = time.time()
sub_l = lll(sub)
v0 = min(vectors_within(sub, min(np.linalg.norm(r) for r in sub_l) + 1e-9), key=np.linalg.norm)
current = 2 * dist_to_cell(v0 / 2, A, bvec)
cands = sorted(vectors_within(sub, current + diam), key=np.linalg.norm)
print(f"[E8] start D={current:.9f}, candidates: {len(cands)}", flush=True)
for v in cands:
    if np.linalg.norm(v) - diam >= current: break
    current = min(current, 2 * dist_to_cell(v / 2, A, bvec))
d = current / diam
print(f"[E8] D={current:.9f} diam={diam:.9f} d={d:.9f} d^2~{Fraction(d*d).limit_denominator(200000)} "
      f"feasible={d >= 1 - 1e-9}  law_check D^2/lam1^2={current**2/lam1**2:.6f} (expect 7/3={7/3:.6f}) "
      f"[{time.time()-t0:.1f}s]", flush=True)
print("DONE", flush=True)
