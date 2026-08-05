"""R4: атака на открытый вопрос OQ1 Arman et al. — существует ли решёточная
раскраска R^6 меньше чем в 343 цвета?

Методика их же randomized-поиска: на E6* строится пул НЕзапрещённых векторов
(x не запрещён <=> D(x) = 2 dist(x/2, V0) >= 2R), случайные шестёрки пула
порождают подрешётки; кандидат валиден <=> min_color_distance >= 2R (все
векторы подрешётки не запрещены). Ищем валидные подрешётки с индексом < 343;
плюс жадный спуск от (3+omega)-решения (замены генераторов на более короткие).
Бюджет — по числу проб.
"""
import math, time
import numpy as np
from scipy.optimize import minimize


def main():
    OMEGA = complex(-0.5, math.sqrt(3) / 2)
    THETA = OMEGA - OMEGA.conjugate()
    ALPHA = 3 + OMEGA

    def realify(vecs):
        out = []
        for v in vecs:
            r = []
            for z in v:
                r += [z.real, z.imag]
            out.append(r)
        return np.array(out)

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
        assert len(coset) == 2**n - 1
        rel = []
        for vs in coset.values():
            m = min(np.linalg.norm(v) for v in vs)
            ties = [v for v in vs if np.linalg.norm(v) <= m + 1e-9]
            if len(ties) == 1: rel.append(ties[0])
        return rel

    def make_dist(rel):
        A = np.array([w for w in rel] + [-w for w in rel])
        b = np.array([w @ w / 2 for w in rel] * 2)
        def dist_to_cell(p):
            if np.all(A @ p <= b + 1e-12): return 0.0
            cons = [{"type": "ineq", "fun": lambda x: b - A @ x, "jac": lambda x: -A}]
            for start in (np.zeros(len(p)), p * 0.5):
                r = minimize(lambda x: (x-p) @ (x-p), start, jac=lambda x: 2*(x-p),
                             constraints=cons, method="SLSQP",
                             options={"maxiter": 1000, "ftol": 1e-12})
                if r.success and np.all(A @ r.x <= b + 1e-9):
                    return float(np.linalg.norm(r.x - p))
            return float("nan")
        return dist_to_cell

    # ---- E6* ----
    u1 = (THETA, 0, 0); u2 = (1, 1, 1); u3 = (0, THETA, 0)
    gens = []
    for u in (u1, u2, u3):
        gens.append(u); gens.append(tuple(OMEGA*z for z in u))
    B = np.linalg.inv(realify(gens)).T          # E6*, строки
    detL = abs(np.linalg.det(B))
    lam1 = min(np.linalg.norm(v) for v in vectors_within(B, 1.2))
    R = lam1 / math.sqrt(2)                     # ratio sqrt(2) подтверждён ранее
    diam = 2 * R
    rel = relevant_vectors(B, diam * 1.02 + 1e-9)
    dist_cell = make_dist(rel)
    print(f"E6*: lam1={lam1:.6f} R={R:.6f} |rel|={len(rel)} detL={detL:.6e}", flush=True)

    def D_of(v):
        d = dist_cell(np.asarray(v, float) / 2.0)
        return 2.0 * d if d == d else float("nan")

    # пул незапрещённых векторов (D(x) >= diam - eps), |x| <= pool_R
    pool_R = 1.20 * math.sqrt(7) * lam1
    t0 = time.time()
    cand = sorted(vectors_within(B, pool_R), key=np.linalg.norm)
    pool = []
    for v in cand:
        if np.linalg.norm(v) >= diam:           # необходимое условие D>=diam невозможно при |v|<diam
            Dv = D_of(v)
            if Dv == Dv and Dv >= diam - 1e-9:
                pool.append(v)
    print(f"pool: {len(pool)} незапрещённых из {len(cand)} коротких [{time.time()-t0:.0f}s]",
          flush=True)

    def min_D_sub(sub):
        """min D(v) по подрешётке (граница достаточности), NaN при сбое QP."""
        sub_l = lll(np.array(sub, float))
        v0 = min(vectors_within(sub_l, min(np.linalg.norm(r) for r in sub_l) + 1e-9),
                 key=np.linalg.norm)
        cur = D_of(v0)
        if cur != cur: return float("nan")
        for v in sorted(vectors_within(sub_l, cur + diam), key=np.linalg.norm):
            if np.linalg.norm(v) - diam >= cur: break
            Dv = D_of(v)
            if Dv != Dv: return float("nan")
            cur = min(cur, Dv)
        return cur

    # эталон: (3+omega)E6* — индекс 343
    Aalpha = np.kron(np.eye(3), np.array([[ALPHA.real, -ALPHA.imag], [ALPHA.imag, ALPHA.real]]))
    sub343 = B @ Aalpha.T
    print(f"reference 343: min D/diam = {min_D_sub(sub343)/diam:.6f}", flush=True)

    best_idx = 343
    rng = np.random.default_rng(20260722)
    found = []
    t0 = time.time()
    TRIES = 60000
    for trial in range(TRIES):
        idxs = rng.choice(len(pool), size=6, replace=False)
        S = np.array([pool[i] for i in idxs])
        det = abs(np.linalg.det(S))
        if det < 1e-9: continue
        index = det / detL
        index_r = round(index)
        if abs(index - index_r) > 1e-6 or index_r >= best_idx or index_r < 63:
            continue
        # быстрый реджект: кратчайший вектор подрешётки не должен быть запрещён
        md = min_D_sub(S)
        if md == md and md >= diam - 1e-9:
            print(f"*** VALID sublattice index {index_r} (d={md/diam:.6f}) at trial {trial}",
                  flush=True)
            found.append((index_r, md/diam, S.tolist()))
            best_idx = index_r
        if trial % 5000 == 4999:
            print(f"  [{trial+1}/{TRIES}] best={best_idx} [{time.time()-t0:.0f}s]", flush=True)

    print(f"random search done: best index = {best_idx} "
          f"({'улучшений НЕ найдено — 343 устояло' if best_idx == 343 else 'НАЙДЕНО ЛУЧШЕ!'})",
          flush=True)
    if found:
        import json
        json.dump(found, open("/Users/mac/Documents/_My_code/Chromatic/audit-data/r4_found.json", "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
