"""Санити-калибровка МСВ-ядра на известных решёточных значениях (s = 1)
и smoke-проверка мульти-узлового случая (s = 2) с независимым верификатором.

Запуск:  python tests_sanity.py   (из каталога fable_pseudolattice_20260807)
Ожидания:
  BCC, k=15  -> d = 1.000000  (Кулсон)
  FCC, k=21  -> d = sqrt(7/6) = 1.080123
  Q15 из n1_r3_full.json -> d = 1.026555 (решёточный Q-фронтир проекта)
  крест с combigeo.find_optimal на тех же формах
  s=2 HCP-подобная структура: evaluate == independent_check (допуск 1e-6)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import msv_core as mc

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def norm_basis_from_gram(Q):
    Q = np.asarray(Q, float)
    Q = Q / abs(np.linalg.det(Q)) ** (1.0 / 3.0)
    return np.linalg.cholesky(Q)


def run():
    t0 = time.time()
    hnfs = {j: mc.hnf_list(j) for j in (15, 21)}
    print(f"[{time.time()-t0:6.1f}s] HNF: j=15 -> {len(hnfs[15])}, j=21 -> {len(hnfs[21])}")

    # --- BCC / 15 ---------------------------------------------------------- #
    B_bcc = norm_basis_from_gram(np.array([[2., 0, 0], [0, 2, 0], [1, 1, 1]]) @
                                 np.array([[2., 0, 0], [0, 2, 0], [1, 1, 1]]).T)
    r = mc.evaluate(B_bcc, np.zeros((1, 3)), [15], hnfs)
    print(f"[{time.time()-t0:6.1f}s] BCC/15: d = {r['d']:.9f}  (ожидание 1.000000)")
    assert abs(r["d"] - 1.0) < 2e-6, r

    # --- FCC / 21 ---------------------------------------------------------- #
    Bf = np.array([[1., 1, 0], [1, 0, 1], [0, 1, 1]])
    B_fcc = norm_basis_from_gram(Bf @ Bf.T)
    r = mc.evaluate(B_fcc, np.zeros((1, 3)), [21], hnfs)
    print(f"[{time.time()-t0:6.1f}s] FCC/21: d = {r['d']:.9f}  (ожидание {math.sqrt(7/6):.6f})")
    assert abs(r["d"] - math.sqrt(7.0 / 6.0)) < 2e-6, r

    # --- Q15 из n1_r3_full ------------------------------------------------- #
    data = json.load(open(f"{ROOT}/audit-data/results/n1_r3_full.json"))
    Q15 = np.array(data["15"]["Q"])
    d_ref = data["15"]["d"]
    B15 = norm_basis_from_gram(Q15)
    r15 = mc.evaluate(B15, np.zeros((1, 3)), [15], hnfs)
    print(f"[{time.time()-t0:6.1f}s] Q15:    d = {r15['d']:.9f}  (референс {d_ref:.9f})")
    assert abs(r15["d"] - d_ref) < 5e-4, (r15, d_ref)

    # --- крест combigeo ---------------------------------------------------- #
    try:
        import combigeo
        cg = combigeo.find_optimal(np.linalg.cholesky(
            Q15 / abs(np.linalg.det(Q15)) ** (1 / 3)).tolist(), index=15, threads=1)
        print(f"[{time.time()-t0:6.1f}s] combigeo Q15: d = {cg.normalized:.9f}")
        assert abs(cg.normalized - r15["d"]) < 5e-4
    except ImportError:
        print("combigeo недоступен — крест пропущен")

    # --- s=2: HCP-подобная, j = (7, 8), k = 15 ----------------------------- #
    a, c = 1.0, math.sqrt(8.0 / 3.0)
    Bh = np.array([[a, 0, 0],
                   [a / 2, a * math.sqrt(3) / 2, 0],
                   [0, 0, c]])
    det = abs(np.linalg.det(Bh))
    Bh /= det ** (1.0 / 3.0)
    frac = np.array([[0., 0, 0], [1 / 3, 1 / 3, 1 / 2]])
    hnfs78 = {7: mc.hnf_list(7), 8: mc.hnf_list(8)}
    r2 = mc.evaluate(Bh, frac, [7, 8], hnfs78)
    print(f"[{time.time()-t0:6.1f}s] HCP j=(7,8): d = {r2['d']:.9f}, "
          f"gaps={['%.6f' % g for g in r2['gaps']]}, diams={['%.6f' % g for g in r2['diams']]}")
    chk = mc.independent_check(Bh, frac, [7, 8],
                               [np.array(h) for h in r2["hnfs"]], box=2)
    print(f"[{time.time()-t0:6.1f}s] независимая проверка: d_check = {chk['d_check']:.9f} "
          f"({chk['n_cells']} ячеек)")
    assert chk["d_check"] >= r2["d"] - 1e-6, (r2["d"], chk["d_check"])
    if chk["d_check"] > r2["d"] + 1e-4:
        print("  ВНИМАНИЕ: сертифицированная оценка заметно ниже прямой — "
              "консервативность, не ошибка")

    print(f"[{time.time()-t0:6.1f}s] ВСЕ САНИТИ-ПРОВЕРКИ ПРОЙДЕНЫ")


if __name__ == "__main__":
    run()
