"""Кампания МСВ (мульти-решёточные косетные раскраски) в ℝ³.

Эксперименты:
  E1 «пол-14»: k = 14 — может ли псевдо-решётка (s ≥ 2 орбит) пробить порог
      d ≥ 1 там, где решёточный класс упирается в 0.901 (Кулсон: < 15 решёткой
      нельзя; для s ≥ 2 вопрос ОТКРЫТ, известен лишь пол 2³ = 8).
  E2 k = 15: решёточный Q-фронтир 1.026555 — бьётся ли он мульти-решёткой.
  E3 k = 16: решёточный фронтир 1.029720.

Метод: для каждого разбиения k = j_1 + … + j_s мультистарт Нелдера–Мида по
(Gram P, дробные сдвиги узлов); внутри каждой оценки — точный перебор
подрешёток Γ_i по HNF-спискам.  Оценка d сертифицирована снизу (см. msv_core).

Гарды: фиксированные сиды, потолок fev на старт, бюджет стенового времени на
эксперимент, хартбиты, ранняя остановка старта без прогресса.

Запуск:  python msv_campaign.py E1  (или E2, E3, all)
Артефакты: results/msv_{exp}.json
"""
from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import msv_core as mc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chromatic_research.forms import norm_gram, pack, unpack

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FW_ITERS_SEARCH = 36
FW_ITERS_FINAL = 800
D_CAP_SEARCH = 1.30
D_CAP_FINAL = 1.75
NM_MAXFEV = 260
NM_XATOL = 1e-6
NM_FATOL = 1e-9

_HNF_CACHE: dict[int, list] = {}


def hnfs_for(js: tuple[int, ...]) -> dict[int, list]:
    for j in js:
        if j not in _HNF_CACHE:
            _HNF_CACHE[j] = mc.prepare_hnfs(j)
    return {j: _HNF_CACHE[j] for j in js}


# --------------------------------------------------------------------------- #
# параметризация и цель                                                        #
# --------------------------------------------------------------------------- #

def unpack_x(x: np.ndarray, s: int):
    Q = unpack(x[:6], 3)
    if Q is None:
        return None, None
    try:
        B = np.linalg.cholesky(Q)
    except np.linalg.LinAlgError:
        return None, None
    frac = np.zeros((s, 3))
    if s > 1:
        frac[1:] = np.asarray(x[6:]).reshape(s - 1, 3)
    frac -= np.floor(frac)
    return B, frac


def objective(x: np.ndarray, j_list: list[int], d_cap: float, fw_iters: int) -> float:
    s = len(j_list)
    B, frac = unpack_x(np.asarray(x, float), s)
    if B is None:
        return 0.0
    # вырождение: слишком близкие узлы дают нулевые ячейки
    if s > 1:
        pts = frac @ B
        shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1)
                           for k in (-1, 0, 1)], float) @ B
        dmin = min(np.min(np.linalg.norm(pts[a] + shifts - pts[b], axis=1))
                   for a in range(s) for b in range(s)
                   if a != b) if s > 1 else 1.0
        if dmin < 0.05:
            return 0.0
    try:
        r = mc.evaluate(B, frac, j_list, hnfs_for(tuple(sorted(set(j_list)))),
                        d_cap=d_cap, fw_iters=fw_iters)
    except Exception:
        return 0.0
    return r.get("d", 0.0) if r.get("ok") else 0.0


def one_start(args):
    """Один NM-старт; выполняется в воркере пула."""
    x0, j_list, maxfev, tag = args
    from scipy.optimize import minimize
    t0 = time.time()
    f = lambda x: -objective(x, j_list, D_CAP_SEARCH, FW_ITERS_SEARCH)
    res = minimize(f, np.asarray(x0, float), method="Nelder-Mead",
                   options={"maxfev": maxfev, "xatol": NM_XATOL,
                            "fatol": NM_FATOL, "adaptive": True})
    return {"tag": tag, "j_list": j_list, "d": -float(res.fun),
            "x": [float(v) for v in res.x], "nfev": int(res.nfev),
            "wall_s": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------- #
# стартовые структуры                                                          #
# --------------------------------------------------------------------------- #

def gram_hex(c_over_a: float) -> np.ndarray:
    a = 1.0
    B = np.array([[a, 0, 0],
                  [a / 2, a * math.sqrt(3) / 2, 0],
                  [0, 0, c_over_a * a]])
    return norm_gram(B)


def structured_starts(s: int, rng: np.random.Generator, n_starts: int,
                      champion_Q: np.ndarray | None) -> list[np.ndarray]:
    """Портфель стартов: HCP, алмаз, чемпион+дырка, случайные возмущения."""
    BCC = norm_gram(np.array([[2., 0, 0], [0, 2, 0], [1, 1, 1]]))
    FCC = norm_gram(np.array([[1., 1, 0], [1, 0, 1], [0, 1, 1]]))
    base: list[np.ndarray] = []

    def mk(Q, fracs):
        return np.concatenate([pack(Q)] + [np.asarray(f, float) for f in fracs])

    extra = s - 1
    if s == 1:
        cands = [mk(BCC, []), mk(FCC, [])]
        if champion_Q is not None:
            cands.insert(0, mk(champion_Q, []))
    else:
        hcp_fracs = [[1 / 3, 1 / 3, 1 / 2], [2 / 3, 1 / 3, 1 / 4],
                     [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.5, 0.5, 0.25]]
        cands = []
        for c_a in (math.sqrt(8 / 3), 1.2, 1.6, 1.85):
            cands.append(mk(gram_hex(c_a), hcp_fracs[:extra]))
        cands.append(mk(FCC, [[0.25, 0.25, 0.25], [0.5, 0.5, 0.5],
                              [0.75, 0.75, 0.75]][:extra]))          # алмаз и далее
        cands.append(mk(BCC, [[0.5, 0.5, 0.5], [0.25, 0.25, 0.25],
                              [0.75, 0.75, 0.75]][:extra]))
        if champion_Q is not None:
            cands.insert(0, mk(champion_Q, [[0.5, 0.5, 0.5], [0.25, 0.5, 0.75],
                                            [0.75, 0.25, 0.5]][:extra]))
    base.extend(cands)
    while len(base) < n_starts:
        b = base[rng.integers(0, len(cands))].copy()
        scale = float(rng.choice([0.04, 0.10, 0.22]))
        b = b * (1 + rng.normal(scale=scale, size=len(b)))
        if s > 1:
            b[6:] += rng.normal(scale=0.5 * scale, size=len(b) - 6)
        base.append(b)
    return base[:n_starts]


# --------------------------------------------------------------------------- #
# каркас эксперимента                                                          #
# --------------------------------------------------------------------------- #

def run_experiment(name: str, k: int, splits: list[list[int]],
                   n_starts: int, wall_budget_s: float, seed: int,
                   workers: int = 8) -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    champ = None
    try:
        data = json.load(open(f"{ROOT}/audit-data/results/n1_r3_full.json"))
        if str(k) in data and data[str(k)].get("Q"):
            champ = norm_gram(np.linalg.cholesky(np.array(data[str(k)]["Q"]) /
                              abs(np.linalg.det(np.array(data[str(k)]["Q"]))) ** (1 / 3)))
    except Exception:
        pass

    all_jobs = []
    rng = np.random.default_rng(seed)
    for spl in splits:
        assert sum(spl) == k, (spl, k)
        starts = structured_starts(len(spl), rng, n_starts, champ)
        for si, x0 in enumerate(starts):
            all_jobs.append((x0.tolist(), list(spl), NM_MAXFEV,
                             f"{'+'.join(map(str, spl))}#s{si}"))

    print(f"[{name}] k={k}: {len(splits)} разбиений × {n_starts} стартов = "
          f"{len(all_jobs)} задач, бюджет {wall_budget_s/60:.0f} мин", flush=True)
    results, aborted = [], False
    with Pool(workers) as pool:
        for r in pool.imap_unordered(one_start, all_jobs):
            results.append(r)
            best = max(results, key=lambda t: t["d"])
            print(f"  [{time.time()-t0:7.1f}s] {r['tag']:>14s}: d={r['d']:.6f} "
                  f"(fev {r['nfev']}, {r['wall_s']}s) | лучший: {best['tag']} "
                  f"d={best['d']:.6f}", flush=True)
            if time.time() - t0 > wall_budget_s:
                aborted = True
                print(f"  [{name}] БЮДЖЕТ ИСЧЕРПАН — останов приёма", flush=True)
                pool.terminate()
                break

    results.sort(key=lambda t: -t["d"])
    # финальная пересдача топ-3 с полными итерациями и широким колпаком
    finals = []
    for r in results[:3]:
        B, frac = unpack_x(np.array(r["x"]), len(r["j_list"]))
        if B is None:
            continue
        rr = mc.evaluate(B, frac, r["j_list"], hnfs_for(tuple(sorted(set(r["j_list"])))),
                         d_cap=D_CAP_FINAL, fw_iters=FW_ITERS_FINAL)
        if rr.get("ok"):
            chk = mc.independent_check(B, frac, r["j_list"],
                                       [np.array(h) for h in rr["hnfs"]], box=2)
            finals.append({"tag": r["tag"], "j_list": r["j_list"],
                           "d_search": r["d"], "d_final": rr["d"],
                           "d_independent": chk.get("d_check"),
                           "gaps": rr["gaps"], "diams": rr["diams"],
                           "hnfs": rr["hnfs"], "x": r["x"],
                           "Q": (unpack(np.array(r["x"][:6]), 3)).tolist(),
                           "frac_sites": frac.tolist()})
            print(f"  ФИНАЛ {r['tag']}: поиск {r['d']:.6f} → пересдача "
                  f"{rr['d']:.6f} → независимая {chk.get('d_check'):.6f}", flush=True)

    out = {
        "experiment": name, "k": k, "splits": splits,
        "n_starts_per_split": n_starts, "seed": seed,
        "nm": {"maxfev": NM_MAXFEV, "xatol": NM_XATOL, "fatol": NM_FATOL,
               "adaptive": True},
        "eval": {"fw_iters_search": FW_ITERS_SEARCH, "fw_iters_final": FW_ITERS_FINAL,
                 "d_cap_search": D_CAP_SEARCH, "d_cap_final": D_CAP_FINAL},
        "wall_budget_s": wall_budget_s, "aborted_by_budget": aborted,
        "wall_s": round(time.time() - t0, 1),
        "host": {"platform": platform.platform(), "python": sys.version.split()[0],
                 "numpy": np.__version__},
        "top": finals,
        "all_starts": results,
    }
    path = os.path.join(RESULTS, f"msv_{name}.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"[{name}] готово за {out['wall_s']}s → {path}", flush=True)
    return out


EXPERIMENTS = {
    "E1_floor14": dict(k=14, splits=[[14], [7, 7], [6, 8], [5, 9], [5, 5, 4], [4, 4, 6]],
                       n_starts=10, wall_budget_s=3600, seed=20260807),
    "E2_k15": dict(k=15, splits=[[7, 8], [6, 9], [5, 10], [5, 5, 5], [4, 5, 6], [4, 4, 7]],
                   n_starts=10, wall_budget_s=3600, seed=20260815),
    "E3_k16": dict(k=16, splits=[[8, 8], [7, 9], [6, 10], [5, 5, 6], [4, 6, 6]],
                   n_starts=10, wall_budget_s=3000, seed=20260816),
    "E4_k18": dict(k=18, splits=[[9, 9], [8, 10], [6, 6, 6]],
                   n_starts=8, wall_budget_s=2400, seed=20260818),
    "E5_k21": dict(k=21, splits=[[10, 11], [7, 7, 7]],
                   n_starts=8, wall_budget_s=2400, seed=20260821),
}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(EXPERIMENTS) if which == "all" else [which]
    for nm_ in names:
        cfg = EXPERIMENTS[nm_]
        run_experiment(nm_, **cfg)
