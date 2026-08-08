r"""ℝ¹¹: ламинирование сертифицированной базы 28812 (ℝ¹⁰) линейным слоем.

База: Λ₁₀ = E₈ + слой A₂ (α = 4+2ω, a = 0.95, офсет из dim10_layer_4+2w.json),
χ(ℝ¹⁰,[1,1.0432]) ≤ 28812, diam₁₀ = 2.5359520 [С], D_min = √7 (планарная
теорема + сертификат). Бюджет ширины: R_L ≤ (diam₀/2)√(d₀²−1) = 0.3736 —
хватает на линейный слой высоты t ≤ 0.747.

Конструкция: Λ₁₁ = ⟨Λ₁₀×{0}, (c,t)⟩, Γ₁₁ = ⟨Γ₁₀×{0}, m·(c,t)⟩, индекс
28812·m. При m = 5: **144060 < 3¹¹ = 177147** (в 1.23 раза).

Статусная арифметика:
  - diam₁₁ ≤ √(diam₁₀² + t²) — (P1), строго поверх [С]-диаметра базы;
  - горизонтальные v ∈ Γ₁₀: D₁₁(v) ≥ D₁₀(v) ≥ √7 — (P2), строго; нужно
    diam₁₁ ≤ √7, т.е. t ≤ 0.7745;
  - слоевые v (z ≠ 0): НИЖНЯЯ оценка D(v) = 2·dist(v/2, V₀(Λ₁₁)) ≥
    2·max_n (⟨v/2,n⟩−|n|²/2)/|n| по любому набору узлов n ∈ Λ₁₁ —
    пофасетный сертификат (V₀ ⊆ каждое полупространство).
Итоговое d = min(√7, min слоевых D-lb)/diam₁₁(P1) — консервативно.

Запуск: python dim11_laminate.py [search|probe] → results/dim11_laminate.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chromatic_research.core.layer_lamination import eisenstein_layer, eisenstein_map
from chromatic_research.core.lamination import enumerate_upto
from chromatic_research.campaigns.planar_theorem_check import e8_theta_basis

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

OMEGA = complex(-0.5, math.sqrt(3) / 2)
SQRT7 = math.sqrt(7.0)
DIAM10_CERT = 2.5359520164327183          # [С] dim10_certificate.json (28812, a=0.95)
OFFSET_28812 = [-0.15840831741164696, -0.17707367568765117, -0.02263669741537317,
                -0.035680156524421275, -0.35901916392206346, -0.3984506290492419,
                -0.2621978895861822, 0.08071755247199483]


def build_base() -> tuple[np.ndarray, np.ndarray]:
    """Λ₁₀, Γ₁₀ конструкции 28812 (a=0.95, α=4+2ω)."""
    e8 = e8_theta_basis()
    kernel = np.rint(np.linalg.solve(
        e8.T, (e8 @ eisenstein_map(3 + OMEGA, 4).T).T).T).astype(int)
    lay = eisenstein_layer(e8, kernel, np.array(OFFSET_28812), 0.95,
                           4 + 2 * OMEGA, 1.5)
    lam10, gam10 = lay.lattices()
    idx = round(abs(np.linalg.det(gam10)) / abs(np.linalg.det(lam10)))
    assert idx == 28812, idx
    return np.asarray(lam10, float), np.asarray(gam10, float)


class Probe11:
    def __init__(self, t: float, m: int, r_neigh: float = 3.3,
                 r_layer_pad: float = 0.6):
        self.lam10, self.gam10 = build_base()
        self.t, self.m = t, m
        self.diam11 = math.sqrt(DIAM10_CERT ** 2 + t * t)
        assert self.diam11 <= SQRT7 + 1e-12, "t слишком велик: рушится (P2)-потолок"
        # узлы Λ₁₀ для пофасетных сертификатов (радиус ограничен: дальние фасеты
        # не дают максимума, оценка остаётся валидной нижней при любом наборе)
        self.p10 = np.vstack([np.zeros((1, 10)),
                              np.asarray(enumerate_upto(self.lam10, r_neigh), float)])
        # точки Γ₁₀ для слоевых косетов (радиус: |v| ≤ D_cap+diam с запасом Бабая)
        rr = math.sqrt(max(1e-9, (SQRT7 + self.diam11 + r_layer_pad) ** 2
                           - (m * t) ** 2)) + 2.6
        self.g10 = np.vstack([np.zeros((1, 10)),
                              np.asarray(enumerate_upto(self.gam10, rr), float)])
        self.ginv = np.linalg.inv(self.gam10)

    def babai(self, x: np.ndarray) -> np.ndarray:
        """x − round(x·Γ⁻¹)·Γ — редукция в фундаментальную коробку Γ₁₀."""
        return x - np.rint(x @ self.ginv) @ self.gam10

    def layer_vectors(self, c: np.ndarray) -> np.ndarray:
        """Γ₁₁-векторы с z = j·m·t для всех уровней j ≥ 1, пока
        j·m·t < √7 + diam (выше — закрыто автоматически D ≥ |v| − diam)."""
        blocks = []
        j = 1
        while j * self.m * self.t < SQRT7 + self.diam11:
            delta = self.babai(j * self.m * c)
            horiz = self.g10 + delta                  # косет j·m·c + Γ₁₀
            z = np.full((len(horiz), 1), j * self.m * self.t)
            blocks.append(np.hstack([horiz, z]))
            j += 1
        return np.vstack(blocks)

    def neighbor_sites(self, c: np.ndarray, jmax: int | None = None) -> np.ndarray:
        if jmax is None:      # слои до высоты худшей проверяемой точки v/2
            jmax = int(math.ceil((SQRT7 + self.diam11) / 2.0 / self.t)) + 1
        """Узлы Λ₁₁ = (x + j·c + Бабай-своб., j·t) для пофасетных сертификатов.

        Внимание: сдвиг j·c редуцируется по Γ₁₀ ТОЛЬКО согласованно со слоевыми
        векторами быть не обязан — оценка валидна для любого набора УЗЛОВ Λ₁₁;
        (x + j·c) + Γ₁₀-сдвиг остаётся узлом слоя j. Держим точки вблизи нуля."""
        out = []
        for j in range(-jmax, jmax + 1):
            shift = self.babai(j * c) if abs(j) else np.zeros(10)
            pts = self.p10 + shift
            keep = np.einsum("ij,ij->i", pts, pts) <= 3.3 ** 2
            block = np.hstack([pts[keep],
                               np.full((int(keep.sum()), 1), j * self.t)])
            if j == 0:
                block = block[np.einsum("ij,ij->i", block, block) > 1e-12]
            out.append(block)
        return np.vstack(out)

    def d_value(self, c: np.ndarray) -> tuple[float, dict]:
        """Консервативное d конструкции при сдвиге c (батчевое)."""
        N = self.neighbor_sites(c)
        nn = np.einsum("ij,ij->i", N, N)
        sq = np.sqrt(nn)
        V = self.layer_vectors(c)
        # автоматически закрытые: |v| ≥ √7 + diam ⇒ D ≥ |v| − diam ≥ √7
        vnorm = np.linalg.norm(V, axis=1)
        need = V[vnorm < SQRT7 + self.diam11]
        if len(need) == 0:
            return SQRT7 / self.diam11, {"D_layer_min": SQRT7, "worst": None,
                                         "n_layer_checked": 0,
                                         "n_neighbors": int(len(N))}
        P = need / 2.0                                     # K × 11
        best = np.full(len(need), -np.inf)
        chunk = 60000
        for lo in range(0, len(N), chunk):
            Nb = N[lo:lo + chunk]
            viol = (Nb @ P.T - (nn[lo:lo + chunk, None]) / 2.0) / \
                sq[lo:lo + chunk, None]
            best = np.maximum(best, viol.max(axis=0))
        D_all = 2.0 * best
        i = int(np.argmin(D_all))
        Dmin = float(D_all[i])
        return min(Dmin, SQRT7) / self.diam11, {
            "D_layer_min": Dmin, "worst": need[i].tolist(),
            "n_layer_checked": int(len(need)), "n_neighbors": int(len(N))}


def search(t: float, m: int, n_starts: int, maxfev: int, seed: int,
           wall_s: float) -> dict:
    from scipy.optimize import minimize
    t0 = time.time()
    pr = Probe11(t, m)
    print(f"t={t}, m={m}: индекс {28812*m}, diam₁₁(P1) = {pr.diam11:.6f}, "
          f"|p10|={len(pr.p10)}, |g10|={len(pr.g10)}", flush=True)
    rng = np.random.default_rng(seed)
    best = {"d": -1.0}
    f = lambda c: -pr.d_value(np.asarray(c))[0]
    starts = [np.zeros(10)] + [rng.normal(scale=s, size=10)
                               for s in ([0.3] * (n_starts // 2) +
                                         [0.7] * (n_starts - 1 - n_starts // 2))]
    for si, c0 in enumerate(starts):
        if time.time() - t0 > wall_s:
            print("  бюджет исчерпан", flush=True)
            break
        res = minimize(f, c0, method="Nelder-Mead",
                       options={"maxfev": maxfev, "xatol": 1e-6, "fatol": 1e-10,
                                "adaptive": True})
        d = -float(res.fun)
        tag = f"s{si}"
        print(f"  [{time.time()-t0:6.1f}s] {tag}: d = {d:.6f} (fev {res.nfev})",
              flush=True)
        if d > best["d"]:
            dd, info = pr.d_value(res.x)
            best = {"d": dd, "c": [float(x) for x in res.x], "start": tag, **info}
    return {"t": t, "m": m, "index": 28812 * m, "diam11_p1": pr.diam11,
            "diam10_cert": DIAM10_CERT, "best": best,
            "wall_s": round(time.time() - t0, 1)}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "search"
    out = {"target": "χ(ℝ¹¹) ≤ 5·28812 = 144060 < 3¹¹ = 177147",
           "runs": []}
    if mode == "probe":
        for t in (0.65, 0.70, 0.747):
            pr = Probe11(t, 5)
            d, info = pr.d_value(np.zeros(10))
            print(f"t={t}: d(c=0) = {d:.6f} {info['D_layer_min']:.4f}", flush=True)
            out["runs"].append({"t": t, "m": 5, "c": "zero", "d": d, **info})
    else:
        for t in (0.70, 0.747, 0.65):
            r = search(t, 5, n_starts=7, maxfev=140, seed=20260808,
                       wall_s=1500.0)
            out["runs"].append(r)
            print(f"t={t}: ЛУЧШЕЕ d = {r['best']['d']:.6f} "
                  f"({'≥1 — КАНДИДАТ!' if r['best']['d'] >= 1 else 'ниже 1'})",
                  flush=True)
    json.dump(out, open(os.path.join(RESULTS, "dim11_laminate.json"), "w"),
              indent=1, ensure_ascii=False)
    print("→ results/dim11_laminate.json", flush=True)


if __name__ == "__main__":
    main()
