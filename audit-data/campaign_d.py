"""Campaign D: verify Arman et al. constructions in dims 7 and 9 (python QP path).

E7* with C7 -> chi(R^7) <= 1372;  A9* with C9 -> chi(R^9) <= 17253.
Computes the exact normalized d = D/(2R) for each; R for A9* analytic
(R^2 = n(n+2)(n+1)/12 in the M-scaling), for E7* — numeric deep-hole search.
"""
import math, time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

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
        if Bs[k] @ Bs[k] >= (delta - mu[k, k-1]**2) * (Bs[k-1] @ Bs[k-1]): k += 1
        else:
            B[[k, k-1]] = B[[k-1, k]]; k = max(k - 1, 1)
    return B

def vectors_within(B, bound):
    B = lll(B); Bs, mu = gram_schmidt(B)
    bn2 = np.array([b @ b for b in Bs]); n = len(B); out = []; coeffs = [0]*n
    def descend(level, partial2):
        if level == 0:
            for c in coeffs:
                if c > 0: break
                if c < 0: return
            else: return
            v = np.array(coeffs, float) @ B
            if v @ v <= bound*bound + 1e-9: out.append(v)
            return
        j = level - 1
        center = sum(coeffs[i] * mu[i, j] for i in range(j+1, n))
        rem = bound*bound - partial2
        if rem < -1e-9: return
        rad = math.sqrt(max(0.0, rem) / bn2[j])
        for c in range(math.ceil(-center-rad-1e-9), math.floor(-center+rad+1e-9)+1):
            coeffs[j] = c
            descend(j, partial2 + (c+center)**2 * bn2[j])
        coeffs[j] = 0
    descend(n, 0.0)
    return out

def relevant_vectors(B, bound):
    n = len(B); Bl = lll(B); Binv = np.linalg.inv(Bl)
    coset = {}
    for v in vectors_within(Bl, bound):
        key = tuple(np.rint(v @ Binv).astype(int) % 2)
        if key == (0,)*n: continue
        coset.setdefault(key, []).append(v)
    assert len(coset) == 2**n - 1, f"{len(coset)} of {2**n-1} cosets — enlarge bound"
    rel = []
    for vs in coset.values():
        m = min(np.linalg.norm(v) for v in vs)
        ties = [v for v in vs if np.linalg.norm(v) <= m + 1e-9]
        if len(ties) == 1: rel.append(ties[0])
    return rel

def dist_to_cell(p, A, b, tol=1e-12):
    if np.all(A @ p <= b + 1e-12): return 0.0
    cons = [{"type": "ineq", "fun": lambda x: b - A @ x, "jac": lambda x: -A}]
    best = None
    rng = np.random.default_rng(1)
    for start in ([np.zeros(len(p))] +
                  [p * 0.5] +
                  [rng.normal(scale=0.1, size=len(p)) for _ in range(3)]):
        res = minimize(lambda x: (x-p) @ (x-p), start,
                       jac=lambda x: 2*(x-p), constraints=cons,
                       method="SLSQP", options={"maxiter": 2000, "ftol": tol})
        if res.success and np.all(A @ res.x <= b + 1e-9):
            d = float(np.linalg.norm(res.x - p))
            best = d if best is None else min(best, d)
    if best is not None:
        return best
    # фолбэк: медленный, но устойчивый trust-constr
    from scipy.optimize import LinearConstraint
    res = minimize(lambda x: (x-p) @ (x-p), np.zeros(len(p)),
                   jac=lambda x: 2*(x-p),
                   constraints=[LinearConstraint(A, -np.inf, b)],
                   method="trust-constr",
                   options={"maxiter": 5000, "gtol": 1e-12, "xtol": 1e-14})
    assert res.status in (1, 2), res.message
    return float(np.linalg.norm(res.x - p))

def min_color_distance(rel, diam, sub):
    A = np.array([w for w in rel] + [-w for w in rel])
    b = np.array([w @ w / 2 for w in rel] * 2)
    sub_l = lll(sub)
    v0 = min(vectors_within(sub, min(np.linalg.norm(r) for r in sub_l) + 1e-9),
             key=np.linalg.norm)
    cur = 2 * dist_to_cell(v0/2, A, b)
    cands = sorted(vectors_within(sub, cur + diam), key=np.linalg.norm)
    print(f"  candidates: {len(cands)}", flush=True)
    for v in cands:
        if np.linalg.norm(v) - diam >= cur: break
        cur = min(cur, 2 * dist_to_cell(v/2, A, b))
    return cur

def report(tag, D, diam):
    d = D / diam
    f = Fraction(d*d).limit_denominator(500000)
    print(f"{tag}: D={D:.9f} diam={diam:.9f} d={d:.9f} d^2~{f} feasible={d >= 1-1e-9}",
          flush=True)

# ---------------- E7* (n=7), index 1372 ----------------
t0 = time.time()
M7 = np.array([
    [-1,0,0,0,0,0,-0.75],[1,-1,0,0,0,0,-0.75],[0,1,-1,0,0,0,0.25],[0,0,1,-1,0,0,0.25],
    [0,0,0,1,-1,0,0.25],[0,0,0,0,1,-1,0.25],[0,0,0,0,0,1,0.25],[0,0,0,0,0,0,0.25]], float)
B7 = np.linalg.cholesky(M7.T @ M7)           # 7x7 row-basis of E7* (scaled)
lam1 = min(np.linalg.norm(v) for v in vectors_within(B7, min(np.linalg.norm(r) for r in lll(B7)) + 1e-9))
nmin = 2*sum(1 for v in vectors_within(B7, lam1 + 1e-9))
print(f"[E7*] lambda1^2={lam1**2:.6f} #min={nmin} (E7* expect 56)", flush=True)

# covering radius numerically: Babai-window deep-hole search + refine
Bl = lll(B7); Binv = np.linalg.inv(Bl)
grid = np.array(np.meshgrid(*[range(-2, 3)]*7)).reshape(7, -1).T @ Bl
rng = np.random.default_rng(3)
R2 = 0.0; argx = None
for _ in range(60):
    X = rng.uniform(-0.5, 0.5, (200, 7)) @ Bl
    Y = X - np.rint(X @ Binv) @ Bl
    d2 = ((Y[:, None, :] - grid[None, :, :])**2).sum(-1).min(1)
    i = int(d2.argmax())
    if d2[i] > R2: R2, argx = float(d2[i]), Y[i]
x = argx.copy(); step = 0.05
for _ in range(500):
    improved = False
    for j in range(7):
        for s in (step, -step):
            xx = x.copy(); xx[j] += s
            dd = (((xx[None, :] - grid)**2).sum(-1)).min()
            if dd > R2 + 1e-15: R2, x, improved = float(dd), xx, True
    if not improved:
        step *= 0.5
        if step < 1e-10: break
R7 = math.sqrt(R2)
print(f"[E7*] covering radius (numeric) = {R7:.9f}, ratio = {2*R7/lam1:.6f}", flush=True)

C7 = np.array([
    [0,-4,-5,-3,-4,-4,-1],[-1,-5,-10,-7,-5,-5,-4],[-2,-2,-9,-4,-5,-4,-4],
    [-3,-2,-5,-4,-4,-1,-3],[-1,-1,-4,-1,-3,0,-3],[-2,0,-1,0,0,0,0],[0,4,6,4,4,4,4]], float)
print(f"[E7*] |det C7| = {abs(np.linalg.det(C7)):.1f} (expect 1372)", flush=True)
sub7 = C7.T @ B7
rel7 = relevant_vectors(B7, 2*R7*1.02 + 1e-9)
print(f"[E7*] relevant pairs: {len(rel7)}", flush=True)
D7 = min_color_distance(rel7, 2*R7, sub7)
report("E7*/1372", D7, 2*R7)
print(f"[E7*] {time.time()-t0:.0f}s", flush=True)

# ---------------- A9* (n=9), index 17253 ----------------
t0 = time.time()
n = 9
M9 = np.ones((n+1, n), float)
for j in range(n):
    M9[j, j] = -n
B9 = np.linalg.cholesky(M9.T @ M9)
lam1_9 = math.sqrt(n*(n+1))
R9 = math.sqrt(n*(n+2)*(n+1)/12.0)          # M-scaling: R^2 = n(n+2)(n+1)/12
print(f"[A9*] lambda1={lam1_9:.6f} R={R9:.6f} diam={2*R9:.6f}", flush=True)
C9 = np.array([
    [0,0,-3,1,0,0,-1,1,0],[1,0,-3,1,1,0,-1,4,1],[0,0,-2,1,0,-1,-1,1,3],
    [0,0,-3,4,0,0,-1,1,0],[0,3,-3,1,0,0,-1,1,0],[3,0,-3,1,0,0,2,1,0],
    [0,0,-4,2,0,3,-1,2,0],[0,0,-3,1,3,0,-1,1,0],[-1,0,-3,1,-1,1,1,1,-1]], float)
print(f"[A9*] |det C9| = {abs(np.linalg.det(C9)):.1f} (expect 17253)", flush=True)
sub9 = C9.T @ B9
rel9 = relevant_vectors(B9, 2*R9*1.01 + 1e-9)
print(f"[A9*] relevant pairs: {len(rel9)} (permutohedron expect 2^9-1=511? strict subset)",
      flush=True)
D9 = min_color_distance(rel9, 2*R9, sub9)
report("A9*/17253", D9, 2*R9)
print(f"[A9*] {time.time()-t0:.0f}s", flush=True)
print("DONE", flush=True)
