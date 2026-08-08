r"""Независимая верификация финалистов ℝ¹¹-ламинирования.

Отличается от поисковой оценки (dim11_laminate.Probe11) по построению:
Γ₁₁ задаётся явным 11×11 базисом, ВСЕ потенциально связывающие векторы
перечисляются Финке–Похстом (никаких косетных конструкций), и каждый слоевой
вектор получает ДВЕ независимые нижние оценки D(v):
  (a) пофасетную: 2·max_n (⟨v/2,n⟩−|n|²/2)/|n| по узлам Λ₁₁;
  (b) SLSQP-проекцию v/2 на НАДмножество ячейки (полупространства соседей
      радиуса r_cell) — dist до надмножества ≤ dist до V₀, т.е. тоже снизу.
Берётся максимум (обе валидны). Горизонтальные векторы обязаны лежать в
Γ₁₀×{0} (проверяется) и закрыты базовым сертификатом D ≥ √7 [С].
Диаметр — (P1): √(diam₁₀²+t²) поверх [С]-диаметра базы.

Запуск: python dim11_verify.py '<json с t,m,c>'  или программно verify(t,m,c).
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chromatic_research.core.lamination import enumerate_upto
from dim11_laminate import build_base, DIAM10_CERT, SQRT7

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def verify(t: float, m: int, c: np.ndarray, r_neigh: float = 3.3,
           r_cell: float = 3.1, log=print) -> dict:
    lam10, gam10 = build_base()
    c = np.asarray(c, float)
    diam11 = math.sqrt(DIAM10_CERT ** 2 + t * t)
    lam11 = np.zeros((11, 11))
    lam11[:10, :10] = lam10
    lam11[10, :10] = c
    lam11[10, 10] = t
    gam11 = np.zeros((11, 11))
    gam11[:10, :10] = gam10
    gam11[10, :10] = m * c
    gam11[10, 10] = m * t
    index = round(abs(np.linalg.det(gam11)) / abs(np.linalg.det(lam11)))
    assert index == 28812 * m, index

    # все потенциально связывающие векторы Γ₁₁
    Vs = np.asarray(enumerate_upto(gam11, SQRT7 + diam11 - 1e-9), float)
    z = Vs[:, 10]
    horiz = Vs[np.abs(z) < 1e-9]
    layer = Vs[np.abs(z) >= 1e-9]
    log(f"  Γ₁₁: горизонтальных {len(horiz)}, слоевых {len(layer)} "
        f"(|v| < √7+diam = {SQRT7 + diam11:.4f})")
    # горизонтальные обязаны лежать в Γ₁₀ (сохранность (P2))
    if len(horiz):
        coef = horiz[:, :10] @ np.linalg.inv(gam10)
        assert np.allclose(coef, np.rint(coef), atol=1e-6), "горизонталь вне Γ₁₀!"

    # (a) пофасетные оценки по узлам Λ₁₁
    N = np.asarray(enumerate_upto(lam11, r_neigh), float)
    nn = np.einsum("ij,ij->i", N, N)
    sq = np.sqrt(nn)

    def facet_lb(p):
        return 2.0 * float(np.max((N @ p - nn / 2.0) / sq))

    # (b) SLSQP на надмножестве ячейки
    Nc = np.asarray(enumerate_upto(lam11, r_cell), float)
    nc = np.einsum("ij,ij->i", Nc, Nc)

    def slsqp_lb(p):
        viol = (Nc @ p - nc / 2.0) / np.sqrt(nc)
        if float(viol.max()) <= 1e-12:
            return 0.0
        cons = [{"type": "ineq", "fun": lambda y: nc / 2.0 - Nc @ y,
                 "jac": lambda y: -Nc}]
        res = minimize(lambda y: float(np.sum((y - p) ** 2)), 0.5 * p,
                       jac=lambda y: 2.0 * (y - p), constraints=cons,
                       method="SLSQP", options={"maxiter": 400, "ftol": 1e-16})
        if not res.success and res.fun <= 0:
            return 0.0
        return 2.0 * math.sqrt(max(float(res.fun), 0.0))

    rows = []
    Dmin = SQRT7
    for v in layer:
        p = v / 2.0
        a = facet_lb(p)
        b = slsqp_lb(p)
        D = max(a, b)
        rows.append({"v": [round(float(x), 6) for x in v],
                     "facet": round(a, 9), "slsqp": round(b, 9),
                     "D": round(D, 9)})
        Dmin = min(Dmin, D)
    d = min(Dmin, SQRT7) / diam11
    out = {"t": t, "m": m, "index": index, "c": [float(x) for x in c],
           "diam11_p1": diam11, "diam10_cert": DIAM10_CERT,
           "D_layer_min": float(Dmin), "d": float(d),
           "n_horiz": int(len(horiz)), "n_layer": int(len(layer)),
           "n_neighbors_facet": int(len(N)), "n_neighbors_cell": int(len(Nc)),
           "layer_rows": rows}
    log(f"  ВЕРИФИКАЦИЯ t={t} m={m}: D_layer_min = {Dmin:.6f}, "
        f"d = {d:.6f} {'≥ 1 ✓' if d >= 1 else '< 1'}")
    return out


if __name__ == "__main__":
    spec = json.loads(sys.argv[1])
    r = verify(spec["t"], spec["m"], np.array(spec["c"]))
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(r, open(os.path.join(
        RESULTS, f"dim11_verified_m{spec['m']}_t{spec['t']}.json"), "w"),
        indent=1, ensure_ascii=False)
