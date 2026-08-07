"""chi(R^10, [1, sqrt(217/214)]) <= 45619, verified on the explicit lattice.

The colouring is the orthogonal product of two blocks whose widths were already
theorems of this project and of plane geometry:

  * ``E8 / (3+omega)E8``  -- index ``7^4 = 2401``, ``diam^2 = 6``, ``D = sqrt 7``
    (the planar theorem gives ``D >= sqrt(7/3) lambda1`` and equality at minimal
    vectors), so ``d_1^2 = 7/6`` and it spends ``6/7`` of the product budget;
  * ``a A2 / (5+2omega) a A2`` -- index ``N(5+2w) = 19``.  In the plane the
    Voronoi cell *is* the hexagon, so ``D`` is exact: the nearest point of the
    hexagon to ``(5+2w)/2 = (2, sqrt3/2)`` is the vertex ``(1/2, 1/(2 sqrt3))``,
    at distance ``sqrt(31/12)``, whence ``d_2^2 = 3 * 31/12 = 31/4`` and a cost
    of ``4/31``.

``6/7 + 4/31 = 214/217 < 1``, so the product calculus applies and

    chi(R^10, [1, sqrt(217/214)]) <= 2401 * 19 = 45619        (3^10 = 59049).

Index 19 is not improvable inside this family: an exhaustive two-dimensional
search (:mod:`chromatic_research.campaigns.ladder2d`) shows no plane colouring of
index below 19 reaches the required ``d >= sqrt 7``, and the volumetric Minkowski
screen puts the floor at 16.

Usage::

    python -m chromatic_research.campaigns.dim10_product
"""

from __future__ import annotations

import json
import math

import numpy as np

import combigeo
from chromatic_research.core.lamination import enumerate_upto, unit_facets
from chromatic_research.campaigns.planar_theorem_check import a2_basis, e8_theta_basis
from chromatic_research.paths import results_path

OMEGA = complex(-0.5, math.sqrt(3) / 2)


def eisenstein_map(alpha: complex, n_complex: int) -> np.ndarray:
    block = np.array([[alpha.real, -alpha.imag], [alpha.imag, alpha.real]])
    return np.kron(np.eye(n_complex), block)


def min_separation_wide(sub: np.ndarray, diam: float,
                        facets: list[tuple[list[float], float]]) -> float:
    """``min D(v)`` over nonzero ``v`` in ``sub``, valid for any width.

    ``core.lamination.min_separation`` stops the enumeration at ``2 diam``, which
    is only enough when ``D_min <= diam``.  Here ``D_min`` may be several times
    the diameter, so the radius is grown until the tail bound ``D(v) >= |v| -
    diam`` certifies that nothing outside can win.
    """
    lam1_sub = float(np.linalg.norm(combigeo.shortest_vector(sub.tolist())))
    radius, best, done = lam1_sub * 1.0001, math.inf, 0.0
    for _ in range(12):
        vectors = enumerate_upto(sub, radius)
        norms = np.linalg.norm(vectors, axis=1)
        for index in np.argsort(norms):
            if norms[index] <= done:
                continue
            if norms[index] - diam >= best:
                break
            best = min(best, 2.0 * float(
                combigeo.dist_to_halfspaces((vectors[index] / 2).tolist(), facets)))
        done = radius
        if best <= radius - diam:
            return best
        radius *= 1.25
    raise RuntimeError("separation sweep did not close")


def build() -> dict:
    e8 = e8_theta_basis()                                   # lambda1^2 = 3
    a2 = a2_basis()                                         # lambda1 = 1
    gamma8 = e8 @ eisenstein_map(3 + OMEGA, 4).T
    unit2 = a2 @ eisenstein_map(5 + 2 * OMEGA, 1).T

    diam8, diam2_unit = math.sqrt(6.0), 2.0 / math.sqrt(3.0)
    d8 = min_separation_wide(gamma8, diam8, unit_facets(e8))
    d2_unit = min_separation_wide(unit2, diam2_unit, unit_facets(a2))
    scale = d8 / d2_unit                                    # equalise the two blocks

    lattice, sub = np.zeros((10, 10)), np.zeros((10, 10))
    lattice[:8, :8], lattice[8:, 8:] = e8, scale * a2
    sub[:8, :8], sub[8:, 8:] = gamma8, scale * unit2

    index = abs(round(np.linalg.det(sub) / np.linalg.det(lattice)))
    diam = math.hypot(diam8, scale * diam2_unit)
    separation = min_separation_wide(sub, diam, unit_facets(lattice))
    lam1 = float(np.linalg.norm(combigeo.shortest_vector(lattice.tolist())))

    return dict(
        index=index, expected_index=2401 * 19,
        lambda1=lam1, diam=diam, D_min=separation, d=separation / diam,
        closed_form_d=math.sqrt(217 / 214),
        block_widths=[d8 / diam8, d2_unit / diam2_unit],
        block_costs=[6 / 7, 4 / 31], budget=6 / 7 + 4 / 31,
        layer_scale=scale, D_e8=d8, D_a2_unit=d2_unit,
    )


def main() -> None:
    record = build()
    for key in ("index", "expected_index", "lambda1", "diam", "D_min", "d",
                "closed_form_d", "budget", "layer_scale"):
        print(f"  {key:>16} = {record[key]}")
    assert record["index"] == record["expected_index"], record
    assert abs(record["d"] - record["closed_form_d"]) < 1e-9, record
    assert abs(record["D_e8"] - math.sqrt(7.0)) < 1e-9, record
    assert record["d"] > 1.0, record
    print(f"\n  chi(R^10, [1, {record['d']:.9f}]) <= {record['index']}"
          f"   (3^10 = {3**10})")
    path = results_path("dim10_product.json")
    path.write_text(json.dumps(record, indent=2))
    print(f"  written {path}")


if __name__ == "__main__":
    main()
