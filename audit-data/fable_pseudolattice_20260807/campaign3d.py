"""Кампания 3D: кластерные псевдо-решётчатые раскраски.

Подкоманды:
  control   — воспроизведение известных величин (A2/7, чемпион k=15, FCC/21)
  screen14  — попытка k=14 в R^3 кластерным классом (экран, если нет)
  ladder    — кластерные попытки расширить лестницу интервалов k=15..17

Запуск (из корня репозитория):
  .venv/bin/python audit-data/fable_pseudolattice_20260807/campaign3d.py control
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import combigeo
import cluster_coloring as cc

HERE = Path(__file__).parent
RESULTS = HERE / "results"
R3_FULL = HERE.parent / "results" / "n1_r3_full.json"


def runtime_info() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def champion_basis(k: str) -> np.ndarray:
    Q = np.array(json.load(open(R3_FULL))[k]["Q"])
    return np.linalg.cholesky(Q)


def bcc_basis() -> np.ndarray:
    # A3* (BCC), нормировка не важна — величины безразмерны
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.5]])


def fcc_basis() -> np.ndarray:
    return np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]) / math.sqrt(2)


def period_via_optimal(B: np.ndarray, N: int) -> np.ndarray:
    """Период = лучшая (max D_min) подрешётка индекса N родителя B."""
    r = combigeo.find_optimal(B.tolist(), index=N, threads=1)
    return np.array(r.best.transition, dtype=np.int64)


def period_inside_champion(B: np.ndarray, k: int, m: int) -> np.ndarray:
    """Период G ⊂ Γ_k (чемпионской подрешётки индекса k) индекса m в ней:
    гарантирует, что решёточная раскраска k содержится в пространстве поиска."""
    r = combigeo.find_optimal(B.tolist(), index=k, threads=1)
    T_gamma = np.array(r.best.transition, dtype=np.int64)
    gamma_basis = T_gamma.astype(float) @ B
    r2 = combigeo.find_optimal(gamma_basis.tolist(), index=m, threads=1)
    T_sub = np.array(r2.best.transition, dtype=np.int64)
    return T_sub @ T_gamma


def run_config(name: str, B: np.ndarray, H: np.ndarray, k: int, *,
               budget_s: float, cpsat_timeout: float = 10.0,
               a_cap_mult: float = 2.0, cutoff_mult: float = 3.2,
               verify: bool = True) -> dict:
    t0 = time.time()
    print(f"[{name}] N={abs(round(np.linalg.det(H.astype(float))))} k={k}",
          flush=True)
    conf = cc.build_conflicts(B, H, cutoff_mult=cutoff_mult)
    print(f"[{name}] конфликты собраны за {conf.build_seconds:.1f}s, "
          f"классов {conf.n_classes}, пар {len(conf.pairs)}", flush=True)
    res = cc.search_best(conf, k, budget_s=budget_s,
                         cpsat_timeout=cpsat_timeout, a_cap_mult=a_cap_mult)
    out = {
        "name": name, "k": k,
        "H": H.tolist(), "basis": B.tolist(),
        "n_classes": conf.n_classes,
        "cutoff": conf.cutoff, "claim_cap": conf.claim_cap,
        "best_ratio": res["best"]["ratio"],
        "best_gap": [res["best"]["a"], res["best"]["b"]],
        "phi": res["best"]["phi"],
        "cosets": conf.cosets.tolist(),
        "unknown_count": res["unknown_count"],
        "search_params": res["params"],
        "seconds": time.time() - t0,
    }
    if verify and res["best"]["phi"] is not None:
        ver = cc.verify_coloring(conf, res["best"]["phi"])
        out["verify"] = {
            "gjk_ratio": ver["gjk_gap"]["ratio"],
            "cert_ratio": ver["cert_gap"]["ratio"],
            "cert_gap": [ver["cert_gap"]["a"], ver["cert_gap"]["b"]],
            "max_procedure_deviation": ver["max_procedure_deviation"],
        }
        print(f"[{name}] итог: измерено {out['best_ratio']:.6f}, "
              f"сертиф. нижняя {ver['cert_gap']['ratio']:.6f}, "
              f"расхождение процедур {ver['max_procedure_deviation']:.2e}",
              flush=True)
    else:
        print(f"[{name}] итог: {out['best_ratio']:.6f}", flush=True)
    return out


def cmd_control(args) -> None:
    out = {"runtime": runtime_info(), "configs": []}
    # 2D классика
    B2 = np.array([[1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]])
    H2 = period_via_optimal(B2, 7)
    out["configs"].append(
        dict(run_config("A2/7", B2, H2, 7, budget_s=120),
             expected=math.sqrt(7.0) / 2.0))
    # чемпион k=15 (N=15 — принудительно решёточная)
    B15 = champion_basis("15")
    H15 = period_via_optimal(B15, 15)
    out["configs"].append(
        dict(run_config("champ15/N15", B15, H15, 15, budget_s=300),
             expected=1.0265551760632647))
    # FCC/21
    Bf = fcc_basis()
    Hf = period_via_optimal(Bf, 21)
    out["configs"].append(
        dict(run_config("FCC/21", Bf, Hf, 21, budget_s=300),
             expected=math.sqrt(7.0 / 6.0)))
    path = RESULTS / "control3d.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"-> {path}", flush=True)


def cmd_screen14(args) -> None:
    out = {"runtime": runtime_info(), "target": "k=14 в R^3", "configs": []}
    parents = {
        "champ14": champion_basis("14"),
        "champ15": champion_basis("15"),
        "BCC": bcc_basis(),
    }
    plans = [
        ("champ14", ("opt", 28)), ("champ14", ("opt", 42)),
        ("champ15", ("inside", 15, 2)), ("champ15", ("inside", 15, 3)),
        ("BCC", ("opt", 28)), ("BCC", ("opt", 42)),
    ]
    for pname, spec in plans:
        B = parents[pname]
        if spec[0] == "opt":
            H = period_via_optimal(B, spec[1])
            tag = f"{pname}/N{spec[1]}"
        else:
            H = period_inside_champion(B, spec[1], spec[2])
            tag = f"{pname}/G{spec[1]}x{spec[2]}"
        out["configs"].append(run_config(f"{tag}/k14", B, H, 14,
                                         budget_s=args.budget,
                                         cpsat_timeout=args.cpsat_timeout))
        json.dump(out, open(RESULTS / "screen14.json", "w"), indent=1)
    print(f"-> {RESULTS / 'screen14.json'}", flush=True)


def cmd_ladder(args) -> None:
    out = {"runtime": runtime_info(),
           "target": "кластерная лестница k=15..17 против решёточной",
           "lattice_ladder": {"15": 1.026593, "16": 1.029728, "17": 1.0648},
           "configs": []}
    for k, m in [(15, 2), (15, 3), (16, 2), (17, 2)]:
        B = champion_basis(str(k))
        H = period_inside_champion(B, k, m)
        out["configs"].append(run_config(f"champ{k}/x{m}/k{k}", B, H, k,
                                         budget_s=args.budget,
                                         cpsat_timeout=args.cpsat_timeout))
        json.dump(out, open(RESULTS / "ladder3d.json", "w"), indent=1)
    print(f"-> {RESULTS / 'ladder3d.json'}", flush=True)


def o2_champion_basis(k: int) -> np.ndarray:
    Q = np.array(json.load(open(HERE.parent / "results" / "o2_r3.json"))
                 [f"width_k{k}"]["Q"])
    return np.linalg.cholesky(Q)


def cmd_inversions(args) -> None:
    """Ремонт инверсий решёточной лестницы: k = база+1 на удвоенном периоде.

    Истинная лестница монотонна, решёточная — нет (22<21, 25<24, 28<27).
    Кластерный класс включает тривиальное решение (лишний цвет пуст), любое
    строгое превышение базы — новый measured-кандидат таблицы ширин R^3."""
    out = {"runtime": runtime_info(),
           "lattice_ladder": {"21": 1.262783, "22": 1.182875,
                              "24": 1.360735, "25": 1.281136,
                              "27": 1.549193, "28": 1.427532},
           "configs": []}
    for base, k in [(21, 22), (24, 25), (27, 28)]:
        B = o2_champion_basis(base)
        H = period_inside_champion(B, base, 2)
        out["configs"].append(dict(
            run_config(f"inv{base}->k{k}", B, H, k,
                       budget_s=args.budget, cpsat_timeout=args.cpsat_timeout),
            trivial_floor=out["lattice_ladder"][str(base)]))
        json.dump(out, open(RESULTS / "inversions3d.json", "w"), indent=1)
    print(f"-> {RESULTS / 'inversions3d.json'}", flush=True)


def cmd_screen14flat(args) -> None:
    """k=14 на плющеных (слоистых) ячейках: там склейка почти бесплатна.

    На круглых ячейках цена склейки H_min/diam = 1.40..1.67 запрещает кластеры;
    у плоской ячейки H(шаг слоя)/diam -> 1, и класс приближает произвольные
    воксельные тайлы. Стэкинг сдвинут (a3 = (s/2, s/2, h)) для шахматности."""
    out = {"runtime": runtime_info(), "target": "k=14, плоские ячейки",
           "configs": []}
    plans = [
        # (s, h, период p,p,r) — N = p*p*r вокселей
        (0.62, 0.31, (3, 3, 6), "flatA"),
        (0.55, 0.275, (4, 4, 4), "flatB"),
    ]
    for s, h, (p1, p2, r), tag in plans:
        B = np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [s / 2, s / 2, h]])
        H = np.diag([p1, p2, r]).astype(np.int64)
        cfg = run_config(f"{tag}/s{s}h{h}/N{p1 * p2 * r}/k14", B, H, 14,
                         budget_s=args.budget,
                         cpsat_timeout=args.cpsat_timeout,
                         a_cap_mult=2.6, cutoff_mult=4.0)
        cfg["cell_shape"] = {"s": s, "h": h}
        out["configs"].append(cfg)
        json.dump(out, open(RESULTS / "screen14flat.json", "w"), indent=1)
    print(f"-> {RESULTS / 'screen14flat.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("control")
    p14 = sub.add_parser("screen14")
    p14.add_argument("--budget", type=float, default=420.0)
    p14.add_argument("--cpsat-timeout", type=float, default=10.0)
    pl = sub.add_parser("ladder")
    pl.add_argument("--budget", type=float, default=420.0)
    pl.add_argument("--cpsat-timeout", type=float, default=10.0)
    pi = sub.add_parser("inversions")
    pi.add_argument("--budget", type=float, default=600.0)
    pi.add_argument("--cpsat-timeout", type=float, default=15.0)
    pf = sub.add_parser("screen14flat")
    pf.add_argument("--budget", type=float, default=900.0)
    pf.add_argument("--cpsat-timeout", type=float, default=20.0)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    {"control": cmd_control, "screen14": cmd_screen14,
     "ladder": cmd_ladder, "inversions": cmd_inversions,
     "screen14flat": cmd_screen14flat}[args.cmd](args)


if __name__ == "__main__":
    main()
