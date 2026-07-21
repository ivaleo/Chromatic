"""Verify chi(E^6) <= 343 via Eisenstein E6* and alpha = 3+omega (Arman et al. Thm 8),
and compute the exact normalized separation d = D/diam(V0) — a constant not printed
in the literature.

Pure numpy/scipy path (combigeo's vertex enumeration is infeasible for the E6* cell:
C(126,6) ~ 1.2e10). Distance point->cell as a QP over relevant-vector halfspaces.
Calibrated on D4 against the known d(D4,49) = sqrt(7/6).

Construction: complex E6 lattice L = {x in E^3 : x1=x2=x3 mod theta}, theta=sqrt(-3),
E = Z[omega]. Z-basis {u_i, omega*u_i}, u1=(theta,0,0), u2=(1,1,1), u3=(0,theta,0).
E6* = real dual of L. Sublattice alpha*E6*, index |alpha|^6 = 7^3 = 343.
"""
import math
import time
from fractions import Fraction

import numpy as np
from scipy.optimize import minimize

OMEGA = complex(-0.5, math.sqrt(3) / 2)
THETA = OMEGA - OMEGA.conjugate()          # sqrt(-3) = i*sqrt(3)
ALPHA = 3 + OMEGA                          # |alpha|^2 = 7


def realify(vecs_c):
    """complex 3-vectors -> real 6-vectors (re1, im1, re2, im2, re3, im3)"""
    out = []
    for v in vecs_c:
        r = []
        for z in v:
            r += [z.real, z.imag]
        out.append(r)
    return np.array(out)


def gram_schmidt(B):
    n = len(B)
    Bs = B.astype(float).copy()
    mu = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            mu[i, j] = B[i] @ Bs[j] / (Bs[j] @ Bs[j])
            Bs[i] -= mu[i, j] * Bs[j]
    return Bs, mu


def lll(B, delta=0.75):
    B = B.astype(float).copy()
    n = len(B)
    k = 1
    while k < n:
        Bs, mu = gram_schmidt(B)
        for j in range(k - 1, -1, -1):
            q = round(mu[k, j])
            if q:
                B[k] -= q * B[j]
        Bs, mu = gram_schmidt(B)
        if Bs[k] @ Bs[k] >= (delta - mu[k, k - 1] ** 2) * (Bs[k - 1] @ Bs[k - 1]):
            k += 1
        else:
            B[[k, k - 1]] = B[[k - 1, k]]
            k = max(k - 1, 1)
    return B


def vectors_within(B, bound):
    """All +-canonical nonzero lattice vectors with |v| <= bound (exact enumeration)."""
    B = lll(B)
    Bs, mu = gram_schmidt(B)
    bn2 = np.array([b @ b for b in Bs])
    n = len(B)
    out = []
    coeffs = [0] * n

    def descend(level, partial2):
        if level == 0:
            for c in coeffs:
                if c > 0:
                    break
                if c < 0:
                    return
            else:
                return
            v = np.array(coeffs) @ B
            if v @ v <= bound * bound + 1e-9:
                out.append(v)
            return
        j = level - 1
        center = sum(coeffs[i] * mu[i, j] for i in range(j + 1, n))
        rem = bound * bound - partial2
        if rem < -1e-9:
            return
        rad = math.sqrt(max(0.0, rem) / bn2[j])
        for c in range(math.ceil(-center - rad - 1e-9), math.floor(-center + rad + 1e-9) + 1):
            coeffs[j] = c
            add = (c + center) ** 2 * bn2[j]
            descend(j, partial2 + add)
        coeffs[j] = 0

    descend(n, 0.0)
    return out


def relevant_vectors(B):
    """Strict minima of nontrivial cosets of L/2L (Voronoi's theorem). B rows = basis."""
    n = len(B)
    lam1 = min(np.linalg.norm(v) for v in vectors_within(B, np.linalg.norm(lll(B)[0]) + 1e-9))
    # covering radius <= something; relevant vectors have |v| <= 2R <= 2*sqrt(n)/2*... use
    # safe bound: all coset minima are within |v| <= 2 * max_j |b*_j| * sqrt(n) heuristic;
    # instead enumerate up to a generous bound and take strict minima per coset mod 2.
    Bl = lll(B)
    bound = 2.05 * math.sqrt(sum(np.linalg.norm(b) ** 2 for b in Bl) / n) * math.sqrt(n)
    vecs = vectors_within(B, bound)
    # coset key: coefficients mod 2 w.r.t. Bl
    Binv = np.linalg.inv(Bl)
    coset = {}
    for v in vecs:
        c = np.rint(v @ Binv).astype(int)
        key = tuple(c % 2)
        if key == (0,) * n:
            continue
        coset.setdefault(key, []).append(v)
    rel = []
    for key, vs in coset.items():
        norms = sorted(np.linalg.norm(v) for v in vs)
        m = norms[0]
        ties = [v for v in vs if np.linalg.norm(v) <= m + 1e-9]
        if len(ties) == 1:  # strict minimum up to +-v (list is +-canonical)
            rel.append(ties[0])
    assert len(coset) == 2 ** n - 1, f"cosets covered: {len(coset)} of {2**n-1} — enlarge bound"
    return rel


def dist_to_cell(p, rel, tol=1e-12):
    """dist(p, V0), V0 = {x : x.w <= |w|^2/2 for all relevant w} — QP via SLSQP."""
    A = np.array([w for w in rel] + [-w for w in rel])
    b = np.array([w @ w / 2 for w in rel] * 2)
    if np.all(A @ p <= b + 1e-12):
        return 0.0
    cons = [{"type": "ineq", "fun": lambda x, A=A, b=b: b - A @ x,
             "jac": lambda x, A=A: -A}]
    res = minimize(lambda x: (x - p) @ (x - p), np.zeros(len(p)),
                   jac=lambda x: 2 * (x - p), constraints=cons,
                   method="SLSQP", options={"maxiter": 500, "ftol": tol})
    assert res.success, res.message
    return float(np.linalg.norm(res.x - p))


def min_color_distance(rel, diam, sub_B):
    """min over nonzero v in sublattice of 2*dist(v/2, V0)."""
    sub_l = lll(sub_B)
    v0 = min(vectors_within(sub_B, min(np.linalg.norm(r) for r in sub_l) + 1e-9),
             key=np.linalg.norm)
    current = 2 * dist_to_cell(v0 / 2, rel)
    cands = sorted(vectors_within(sub_B, current + diam), key=np.linalg.norm)
    for v in cands:
        if np.linalg.norm(v) - diam >= current:
            break
        d = 2 * dist_to_cell(v / 2, rel)
        current = min(current, d)
    return current


def report(name, d, D, diam):
    f = Fraction(d * d).limit_denominator(200000)
    print(f"{name}: D={D:.9f} diam={diam:.9f} d={d:.9f} d^2~{f} feasible={d >= 1 - 1e-9}",
          flush=True)


# ---- calibration: D4, sublattice of index 49 must give d = sqrt(7/6) = 1.080123 ----
D4 = np.array([[2, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]], float)
rel = relevant_vectors(D4)
print(f"[calib] D4 relevant vectors: {len(rel)} (expect 24)", flush=True)
diam = 2 * max(0.0, *(np.linalg.norm(w) for w in rel))  # placeholder, recomputed below

# diameter of D4 cell is known = 2; for E6* we will use 2R = sqrt(2)*lambda1 (see below).
# Here calibrate distance code with the known transition (HNF w.r.t. LLL(D4) from combigeo):
H = np.array([[1, 0, 0, 2], [0, 1, 2, 4], [0, 0, 7, 0], [0, 0, 0, 7]], float)
LLL_D4 = lll(D4)
subD4 = H @ LLL_D4
D = min_color_distance(rel, 2.0, subD4)
report("[calib] D4/49", D / 2.0, D, 2.0)

# ---- E6* ----
u1 = (THETA, 0, 0)
u2 = (1, 1, 1)
u3 = (0, THETA, 0)
gens_c = []
for u in (u1, u2, u3):
    gens_c.append(u)
    gens_c.append(tuple(OMEGA * z for z in u))
B_E6 = realify(gens_c)                      # 6x6, rows = Z-basis of complex E6
lam1_E6 = min(np.linalg.norm(v) for v in vectors_within(B_E6, 1.75))
n_roots = sum(1 for v in vectors_within(B_E6, lam1_E6 + 1e-9))
print(f"E6 check: lambda1^2 = {lam1_E6**2:.6f} (expect 3), "
      f"#min vectors = {2*n_roots} (expect 72)", flush=True)

B_dual = np.linalg.inv(B_E6).T              # rows = Z-basis of E6*
lam1 = min(np.linalg.norm(v) for v in vectors_within(B_dual, 1.2))
count_min = 2 * sum(1 for v in vectors_within(B_dual, lam1 + 1e-9))
print(f"E6*: lambda1^2 = {lam1**2:.6f}, #min vectors = {count_min} (expect 54 for E6*)",
      flush=True)

rel6 = relevant_vectors(B_dual)
print(f"E6* relevant vectors: {len(rel6)} (expect 63 canonical pairs = 126 total)", flush=True)

# diameter = 2R; covering/packing ratio of E6* = sqrt(2) (Arman et al. Table) => R = lam1/sqrt(2)
R = lam1 / math.sqrt(2)
diam6 = 2 * R
# independent numeric check of R: max over many random points of dist to lattice
rng = np.random.default_rng(7)
Binv = np.linalg.inv(B_dual)
best = 0.0
for _ in range(4000):
    x = rng.uniform(-0.5, 0.5, 6) @ B_dual
    c = np.rint(x @ Binv)
    y = x - c @ B_dual
    dmin2 = min((y - v) @ (y - v) for v in ([np.zeros(6)] + vectors_within(B_dual, np.linalg.norm(y) + lam1)))
    best = max(best, dmin2)
print(f"covering radius: analytic {R:.6f}; sampled lower bound {math.sqrt(best):.6f}", flush=True)

A_alpha = np.kron(np.eye(3), np.array([[ALPHA.real, -ALPHA.imag], [ALPHA.imag, ALPHA.real]]))
sub6 = B_dual @ A_alpha.T
idx = abs(np.linalg.det(sub6)) / abs(np.linalg.det(B_dual))
print(f"index [E6* : alpha E6*] = {idx:.3f} (expect 343)", flush=True)

t0 = time.time()
D6 = min_color_distance(rel6, diam6, sub6)
report("E6*/343 (alpha=3+omega)", D6 / diam6, D6, diam6)
print(f"guaranteed floor from their proof: 3/(2*sqrt(2)) = {3/(2*math.sqrt(2)):.6f} "
      f"[{time.time()-t0:.1f}s]", flush=True)
print("DONE", flush=True)
