r"""ℝ¹²: окно между 7⁶ и 3¹² — подгруппы K₁₂ без минимальной оболочки.

Идея (синтез сессий 07–08.08): планарная теорема [Т] даёт per-vector
D((3+ω)w) ≥ √(7/3)·|w|. Если подгруппа Γ' ⊂ K₁₂ индекса m не содержит НИ
ОДНОГО из 756 минимальных векторов (|w|² = 4), то все её ненулевые векторы
имеют |w|² ≥ 6 (следующая оболочка K₁₂), и для Γ = (3+ω)Γ' (индекс m·7⁶):

    D_min(Γ) ≥ √(7/3)·√6 = √14 = 3.741657...
    diam(V₀(K₁₂)) ≤ 2R = 2√(8/3) = 3.265986...        (всегда diam ≤ 2R)
    ⇒  χ(ℝ¹², [1, ℓ]) ≤ m·7⁶  при ℓ < √14/(2√(8/3)) = √(21/16) = 1.145644

При m = 2: 235298, при m = 3: 352947 — против 3¹² = 531441 (в 2.26 / 1.51 раза).
7⁶ само по себе невозможно (D_min((3+ω)K₁₂) = 3.055 < diam — результат проекта),
поэтому вопрос ровно в существовании Γ'.

Скрин ИСЧЕРПЫВАЮЩИЙ: все подгруппы индекса 2 (4095 функционалов F₂¹²) и
индекса 3 (265720 прямых F₃¹²), при неудаче — индекса 4 (циклические ℤ/4 и
кляйновские (ℤ/2)²). Любой положительный кандидат верифицируется:
(1) целочисленный базис Γ', det = m, λ₁(Γ')² ≥ 6 Финке–Похстом;
(2) Γ = (3+ω)Γ', |det| = m·7⁶ точно;
(3) независимая числовая проверка D_min(Γ) перечислением коротких векторов
    Γ и проекцией v/2 на ячейку (по 756 полупространствам минимальных
    векторов — НАДмножество ячейки, т.е. проверка консервативна).

Запуск: python k12_shellfree.py  → results/k12_shellfree.json
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chromatic_research.core.k12 import build_k12, omega_action
from chromatic_research.core.lamination import enumerate_upto

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

SQRT14 = math.sqrt(14.0)
DIAM = 2.0 * math.sqrt(8.0 / 3.0)


def minimal_vectors(B: np.ndarray) -> np.ndarray:
    """756 минимальных векторов в ЦЕЛЫХ координатах базиса K₁₂
    (enumerate_upto возвращает вещественные векторы — переводим и проверяем)."""
    V = np.asarray(enumerate_upto(B, 2.0 + 1e-9), dtype=float)
    C = V @ np.linalg.inv(B)
    mins = np.rint(C).astype(np.int64)
    assert np.allclose(C, mins, atol=1e-8), "координаты не целые"
    norms = np.einsum("ij,ij->i", mins @ B, mins @ B)
    assert len(mins) == 756 and np.allclose(norms, 4.0, atol=1e-6), \
        (len(mins), norms.min(), norms.max())
    return mins


def screen_mod2(mins: np.ndarray) -> list[np.ndarray]:
    """φ ∈ F₂¹²\0 с φ(v) ≠ 0 для всех минимальных векторов."""
    M2 = (mins % 2).astype(np.int8)
    phis = np.array(list(itertools.product(range(2), repeat=12)), dtype=np.int8)[1:]
    prod = (M2 @ phis.T) % 2                          # 756 × 4095
    alive = np.all(prod != 0, axis=0)
    return [phis[i] for i in np.nonzero(alive)[0]]


def screen_mod3(mins: np.ndarray) -> list[np.ndarray]:
    """Прямые F₃¹² (канонические: первый ненулевой коэффициент = 1)."""
    M3 = (mins % 3).astype(np.int8)
    out = []
    all_phi = np.array(list(itertools.product(range(3), repeat=12)), dtype=np.int8)[1:]
    first_nz = np.argmax(all_phi != 0, axis=1)
    canon = all_phi[np.arange(len(all_phi)), first_nz] == 1
    phis = all_phi[canon]                             # 265720 прямых
    chunk = 20000
    for lo in range(0, len(phis), chunk):
        P = phis[lo:lo + chunk]
        prod = (M3.astype(np.int32) @ P.T.astype(np.int32)) % 3
        alive = np.all(prod != 0, axis=0)
        for i in np.nonzero(alive)[0]:
            out.append(P[i])
    return out


def screen_mod4_cyclic(mins: np.ndarray, log=print) -> list[np.ndarray]:
    """φ: ℤ¹² → ℤ/4 сюръективные (хотя бы один нечётный коэффициент);
    ядро индекса 4; условие: φ(v) ≢ 0 (mod 4) для всех минимальных."""
    M = (mins % 4).astype(np.int32)
    out = []
    # редукция: φ и -φ дают одно ядро; фиксируем первый нечётный коэффициент = 1
    total = 0
    t0 = time.time()
    for phi in itertools.product(range(4), repeat=12):
        p = np.array(phi, dtype=np.int32)
        odd = np.nonzero(p % 2 == 1)[0]
        if len(odd) == 0 or p[odd[0]] != 1:
            continue
        total += 1
        if total % 500000 == 0:
            log(f"    mod4 cyclic: {total} проверено, {time.time()-t0:.0f}s, "
                f"найдено {len(out)}")
        if np.all((M @ p) % 4 != 0):
            out.append(p)
    return out


def screen_mod4_klein(mins: np.ndarray, log=print) -> list[tuple[np.ndarray, np.ndarray]]:
    """Ядра пар независимых функционалов mod 2: v выживает, если
    (φ₁(v), φ₂(v)) ≠ (0,0). Перебор 2-мерных подпространств hom(F₂¹²)."""
    M2 = (mins % 2).astype(np.int8)
    phis = np.array(list(itertools.product(range(2), repeat=12)), dtype=np.int8)[1:]
    vals = (M2 @ phis.T) % 2                          # 756 × 4095, столбец = φ
    zero_sets = [frozenset(np.nonzero(vals[:, i] == 0)[0].tolist())
                 for i in range(len(phis))]
    out = []
    t0 = time.time()
    # подпространство {0, a, b, a+b}: минимал убивает его, если он в ядрах всех
    # трёх ненулевых элементов; т.е. нужно zero(a) ∩ zero(b) ∩ zero(a+b) = ∅
    idx_of = {tuple(p): i for i, p in enumerate(phis)}
    n = len(phis)
    for ia in range(n):
        za = zero_sets[ia]
        if len(za) == 0:
            continue                                  # уже покрыто mod2-скрином
        for ib in range(ia + 1, n):
            zb = zero_sets[ib]
            inter = za & zb
            if not inter:
                continue
            ic = idx_of[tuple((phis[ia] + phis[ib]) % 2)]
            if ic < ib:
                continue                              # каноничность тройки
            if inter & zero_sets[ic]:
                continue
            out.append((phis[ia], phis[ib]))
        if (ia + 1) % 400 == 0:
            log(f"    mod4 klein: {ia+1}/{n}, {time.time()-t0:.0f}s, "
                f"найдено {len(out)}")
    return out


# ------------------------------------------------------------------ верификация

def kernel_basis(phi: np.ndarray, m: int) -> np.ndarray:
    """Целочисленный базис {x ∈ ℤ¹² : φ·x ≡ 0 (mod m)}, det = m."""
    from sympy import Matrix
    from sympy.matrices.normalforms import hermite_normal_form
    rows = [list(np.eye(12, dtype=int)[i] * m) for i in range(12)]
    # генераторы: m·eᵢ и все eᵢ·φ_j-комбинации... проще: столбцовая решётка
    # {x : φx ≡ 0}: базис = HNF решётки, порождённой m·eᵢ и векторами
    # eᵢ·φ_k − eₖ·φ_i… Надёжный путь: перебрать стандартный базис по модулю.
    # Универсально: решётка K = {x: φx ≡ 0 (mod m)} содержит m·ℤ¹² и вектора
    # eᵢ − cᵢ·e_p, где p — координата с обратимым φ_p, cᵢ = φᵢ/φ_p mod m.
    p = None
    for i, c in enumerate(phi % m):
        if math.gcd(int(c), m) == 1:
            p = i
            break
    assert p is not None, "функционал не сюръективен"
    inv = pow(int(phi[p] % m), -1, m)
    gens = [np.eye(12, dtype=int)[p] * m]
    for i in range(12):
        if i == p:
            continue
        c = (int(phi[i] % m) * inv) % m
        gens.append(np.eye(12, dtype=int)[i] - c * np.eye(12, dtype=int)[p])
    H = hermite_normal_form(Matrix(np.array(gens, dtype=int).T))
    K = np.array(H.T.tolist(), dtype=np.int64)
    d = int(round(abs(np.linalg.det(K.astype(float)))))
    assert d == m, (d, m)
    # членство: φ·строки ≡ 0
    assert np.all((K @ phi) % m == 0), "базис не в ядре"
    return K


def kernel_basis_klein(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    from sympy import Matrix
    from sympy.matrices.normalforms import hermite_normal_form
    gens = [np.eye(12, dtype=int)[i] * 2 for i in range(12)]
    # x с φ₁x≡0, φ₂x≡0 (mod 2): пересечение двух ядер
    K1 = kernel_basis(phi1, 2)
    picked = [r for r in K1 if int(r @ phi2) % 2 == 0]
    rest = [r for r in K1 if int(r @ phi2) % 2 == 1]
    for a in range(len(rest)):
        for b in range(a + 1, len(rest)):
            picked.append(rest[a] + rest[b])
    picked += [2 * r for r in rest]
    H = hermite_normal_form(Matrix(np.array(picked, dtype=int).T))
    K = np.array(H.T.tolist(), dtype=np.int64)
    d = int(round(abs(np.linalg.det(K.astype(float)))))
    assert d == 4, d
    return K


def verify_candidate(K: np.ndarray, m: int, B: np.ndarray, U: np.ndarray,
                     mins: np.ndarray, log=print) -> dict:
    """Полная проверка Γ' = K·B: λ₁² ≥ 6; Γ = (3+ω)Γ'; D_min ≥ √14 численно."""
    from scipy.optimize import minimize
    Bp = K @ B                                        # базис Γ'
    short = enumerate_upto(Bp, math.sqrt(6.0) - 1e-6)
    ok_l1 = len(short) == 0
    log(f"    λ₁(Γ')² ≥ 6: {'OK' if ok_l1 else f'ПРОВАЛ ({len(short)} коротких)'}")
    T = 3 * np.eye(12, dtype=np.int64) + U            # коэффициентная матрица (3+ω)
    KG = K @ T                                        # базис Γ в координатах K₁₂
    detK = int(round(abs(np.linalg.det(KG.astype(float)))))
    ok_det = detK == m * 7 ** 6
    log(f"    det Γ = {detK} (= {m}·7⁶: {'OK' if ok_det else 'ПРОВАЛ'})")
    # независимая числовая D_min: короткие векторы Γ и проекция на ячейку
    BG = KG @ B
    cand = np.asarray(enumerate_upto(BG, SQRT14 + DIAM + 0.01), dtype=float)
    R = (mins @ B)                                    # 756 нормалей, |r|²=4
    Dmin, worst = math.inf, None
    for v in cand:
        x0 = v / 2.0
        # dist(x0, P), P = {y: ⟨y,r⟩ ≤ 2}: если x0 ∈ P — 0, иначе QP (SLSQP)
        viol = R @ x0 - 2.0
        if viol.max() <= 1e-12:
            D = 0.0
        else:
            act = np.argsort(-viol)[:80]
            cons = [{"type": "ineq", "fun": lambda y, Ra=R[act]: 2.0 - Ra @ y,
                     "jac": lambda y, Ra=R[act]: -Ra}]
            res = minimize(lambda y: float(np.sum((y - x0) ** 2)),
                           x0 - viol.max() * R[np.argmax(viol)] / 4.0,
                           jac=lambda y: 2.0 * (y - x0),
                           constraints=cons, method="SLSQP",
                           options={"maxiter": 300, "ftol": 1e-16})
            y = res.x
            # проверка на ПОЛНОМ наборе ограничений
            if (R @ y - 2.0).max() > 1e-9:
                cons_full = [{"type": "ineq", "fun": lambda y: 2.0 - R @ y,
                              "jac": lambda y: -R}]
                res = minimize(lambda y: float(np.sum((y - x0) ** 2)), y,
                               jac=lambda y: 2.0 * (y - x0),
                               constraints=cons_full, method="SLSQP",
                               options={"maxiter": 400, "ftol": 1e-16})
                y = res.x
            D = 2.0 * math.sqrt(max(res.fun, 0.0))
        if D < Dmin:
            Dmin, worst = D, v.tolist()
    ok_D = Dmin >= SQRT14 - 1e-6
    log(f"    D_min(Γ) числ. = {Dmin:.9f} (≥ √14 = {SQRT14:.9f}: "
        f"{'OK' if ok_D else 'ПРОВАЛ'}), кандидатов {len(cand)}")
    return {"ok": bool(ok_l1 and ok_det and ok_D),
            "lambda1_sq_ge6": bool(ok_l1), "det_ok": bool(ok_det),
            "Dmin_numeric": float(Dmin), "worst_vector": worst,
            "n_candidates": int(len(cand)),
            "K_gamma_prime": K.tolist()}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    coeff, B = build_k12()          # coeff: базис в ℤ[ω]⁶; B: вещественный базис
    B = np.asarray(B, dtype=float)
    coeff = np.asarray(coeff, dtype=np.int64)
    Om = np.asarray(omega_action(), dtype=np.int64)      # ω на координатах ℤ[ω]⁶
    # матрица умножения на (3+ω) в КООРДИНАТАХ БАЗИСА K₁₂ (целая, т.к. K₁₂ —
    # ℤ[ω]-модуль): U = coeff (3I+Ω) coeff⁻¹
    T6 = 3 * np.eye(12, dtype=np.int64) + Om
    U_f = coeff @ T6 @ np.linalg.inv(coeff.astype(float))
    U = np.rint(U_f).astype(np.int64)
    assert np.allclose(U_f, U, atol=1e-8), "(3+ω) не целочисленна в базисе K₁₂"
    assert int(round(abs(np.linalg.det(U.astype(float))))) == 7 ** 6
    mins = minimal_vectors(B)
    print(f"[{time.time()-t0:5.1f}s] K₁₂ построена, минимальных векторов: "
          f"{len(mins)}", flush=True)

    out = {"target": "χ(ℝ¹²,[1,√(21/16)]) ≤ m·7⁶ через Γ=(3+ω)Γ'",
           "width_if_found": math.sqrt(21.0 / 16.0),
           "diam": DIAM, "Dmin_bound": SQRT14}

    s2 = screen_mod2(mins)
    print(f"[{time.time()-t0:5.1f}s] mod 2: {len(s2)} выживших из 4095", flush=True)
    out["mod2_survivors"] = len(s2)

    s3 = screen_mod3(mins)
    print(f"[{time.time()-t0:5.1f}s] mod 3: {len(s3)} выживших из 265720", flush=True)
    out["mod3_survivors"] = len(s3)

    found = None
    for m, sv in ((2, s2), (3, s3)):
        if sv and found is None:
            phi = np.array(sv[0], dtype=np.int64)
            print(f"  верификация кандидата m={m}, φ={phi.tolist()}", flush=True)
            K = kernel_basis(phi, m)
            v = verify_candidate(K, m, B, U, mins)
            out[f"candidate_m{m}"] = {"phi": phi.tolist(), **v}
            if v["ok"]:
                found = (m, phi.tolist())
    if found is None and not s2 and not s3:
        print(f"[{time.time()-t0:5.1f}s] mod 4 (кляйновские)...", flush=True)
        s4k = screen_mod4_klein(mins)
        print(f"[{time.time()-t0:5.1f}s] mod 4 klein: {len(s4k)} выживших", flush=True)
        out["mod4_klein_survivors"] = len(s4k)
        if s4k:
            phi1, phi2 = s4k[0]
            K = kernel_basis_klein(np.array(phi1, np.int64), np.array(phi2, np.int64))
            v = verify_candidate(K, 4, B, U, mins)
            out["candidate_m4klein"] = {"phi1": np.array(phi1).tolist(),
                                        "phi2": np.array(phi2).tolist(), **v}
            if v["ok"]:
                found = (4, "klein")
        else:
            print(f"[{time.time()-t0:5.1f}s] mod 4 (циклические)...", flush=True)
            s4c = screen_mod4_cyclic(mins)
            print(f"[{time.time()-t0:5.1f}s] mod 4 cyclic: {len(s4c)} выживших",
                  flush=True)
            out["mod4_cyclic_survivors"] = len(s4c)
            if s4c:
                K = kernel_basis(np.array(s4c[0], np.int64), 4)
                v = verify_candidate(K, 4, B, U, mins)
                out["candidate_m4cyclic"] = {"phi": np.array(s4c[0]).tolist(), **v}
                if v["ok"]:
                    found = (4, "cyclic")

    out["found"] = found
    out["wall_s"] = round(time.time() - t0, 1)
    if found:
        m = found[0]
        print(f"\n*** НАЙДЕНО: χ(ℝ¹²,[1,ℓ]) ≤ {m}·7⁶ = {m*7**6} при "
              f"ℓ < √(21/16) = {math.sqrt(21/16):.6f} (против 3¹² = {3**12}) ***",
              flush=True)
    else:
        print("\nОтрицательно: подгрупп индекса 2/3/4 без минимальной оболочки нет",
              flush=True)
    json.dump(out, open(os.path.join(RESULTS, "k12_shellfree.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"→ results/k12_shellfree.json ({out['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
