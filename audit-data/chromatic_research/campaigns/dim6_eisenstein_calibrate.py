"""Калибровка эйзенштейнова экрана: при каком индексе циклический подмодуль работает.

Зачем.  Отрицательный результат «при 337 не выжил ни один из 227 814
подмодулей» стоит ровно столько, сколько стоит уверенность, что экран вообще
СПОСОБЕН найти выживших.  Этот прогон отвечает на вопрос количественно: он
проходит по простым индексам ``p ≡ 1 (mod 3)`` и для каждого считает максимум
``d`` по всем подмодулям.

Что получается (посев — ``E₆*``, у которого ``2R/λ₁ = √2``):

    p:      307     331     337     379     409     487     541
    max d:  0.7625  0.7625  0.8938  0.8938  0.9350  0.9350  1.0379

то есть порог берётся между 487 и 541 — экран работает, а 337 не проходит по
существу.

Что из этого следует.  При ПРОСТОМ индексе фактор ``Λ/M`` циклический, то есть
``M`` — «гиперплоскостная», геометрически скошенная подрешётка.  Конструкция 343
устроена иначе: это ПОДОБНАЯ подрешётка ``(3+ω)Λ``, а подобные имеют индекс
``N(α)³``, и их лестница в ``ℤ[ω]³`` есть ``1, 27, 64, 343, 729, …`` — между 64
и 343 нет ничего.  Циклические платят за скошенность коэффициент ≈ 541/343 ≈
1.58 по числу цветов.  Значит целиться надо в индексы с РОВНЫМ разложением
нормы и в многоблочные подмодули, а не в индекс-простое.

Usage::

    python -m chromatic_research.campaigns.dim6_eisenstein_calibrate
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from chromatic_research.paths import results_path

DEFAULT_INDICES = (307, 313, 331, 337, 349, 367, 373, 379, 397, 409, 439,
                   487, 541, 601, 661)


def main(argv=None) -> int:
    import chromatic_research.campaigns.dim6_eisenstein_337 as screen_module

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", type=int, nargs="+",
                        default=list(DEFAULT_INDICES))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--directions", type=int, default=120)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    basis, automorphism = screen_module.seed_lattice()
    print("посев E₆*; max d считается по ВСЕМ подмодулям каждого индекса")
    rows = []
    threshold = None
    start = time.time()
    original = screen_module.INDEX
    try:
        for index in args.indices:
            if index % 3 != 1:
                print(f"  p={index}: пропуск, {index} ≢ 1 (mod 3)")
                continue
            screen_module.INDEX = index
            moment = time.time()
            ratio = screen_module.best_ratio(
                basis, automorphism, low=0.5, high=1.35,
                steps=args.steps, directions=args.directions)
            rows.append({"index": index, "max_d": round(float(ratio), 6),
                         "submodules": 2 * (index ** 2 + index + 1)})
            mark = "  <-- порог взят" if ratio >= 1.0 else ""
            if ratio >= 1.0 and threshold is None:
                threshold = index
            print(f"  p={index:5d} ({rows[-1]['submodules']:8d} подмодулей): "
                  f"max d = {ratio:.5f}{mark}  [{time.time()-moment:.0f}s]",
                  flush=True)
    finally:
        screen_module.INDEX = original

    out = args.output or results_path("dim6_eisenstein_337_calibration.json")
    out.write_text(json.dumps({
        "seed_lattice": "E6*", "table": rows,
        "first_index_reaching_one": threshold,
        "similar_sublattice_ladder": [1, 27, 64, 343, 729],
        "seconds": round(time.time() - start, 1),
    }, indent=1, ensure_ascii=False) + "\n")
    print(f"\nпорог впервые взят при индексе {threshold} "
          f"(подобная подрешётка (3+ω)E₆* берёт его при 343)")
    print(f"сохранено: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
