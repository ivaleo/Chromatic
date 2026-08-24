"""Контроли точного слоёного сертификата.

Положительные: замкнутая форма при нулевом сдвиге слоя и сверка с полностью
независимым численным вычислением того же максимума.
Отрицательный: заведомо недостижимая цель обязана быть отвергнута.
"""

from fractions import Fraction as Fr

import numpy as np
import pytest

from chromatic_research.core.exact_layer_cert import (
    LayeredGeometry,
    certify_covering_radius,
    geometry_from_gram,
)

A2 = ((Fr(2), Fr(-1)), (Fr(-1), Fr(2)))          # lambda_1^2 = 2, R^2 = 2/3
A2_R2 = Fr(2, 3)
Z2 = ((Fr(1), Fr(0)), (Fr(0), Fr(1)))            # R^2 = 1/2
Z2_R2 = Fr(1, 2)


def _geometry(gram, offset, height2, base_r2):
    return LayeredGeometry(gram, tuple(offset), height2, base_r2).with_relevant()


# ---------------------------------------------------------------- замкнутая форма


@pytest.mark.parametrize("height2", [Fr(1, 2), Fr(1), Fr(9, 4)])
def test_aligned_layers_closed_form(height2):
    """Сдвиг 0: решётка распадается в ортогональную сумму, R^2 = R_base^2 + t^2/4."""
    geometry = _geometry(A2, (Fr(0), Fr(0)), height2, A2_R2)
    report = certify_covering_radius(geometry, None, verbose=False)
    assert report["max_phi"] == A2_R2 + height2 / 4


def test_aligned_layers_square_base():
    geometry = _geometry(Z2, (Fr(0), Fr(0)), Fr(1), Z2_R2)
    report = certify_covering_radius(geometry, None, verbose=False)
    assert report["max_phi"] == Z2_R2 + Fr(1, 4)


# ------------------------------------------------- сверка с независимым float


def _float_max_phi(gram, offset, height2, base_r2, samples=240):
    """Независимое численное вычисление ``max_x U(x)`` прямым перебором сетки.

    Считает ``min`` по многим слоям и точный максимум по ``z`` — то есть
    ЛЕВУЮ часть неравенства, которую мажорирует сертификат.  Возвращает
    нижнюю оценку истинного ``R^2``.
    """
    gram_f = np.array([[float(v) for v in row] for row in gram])
    basis = np.linalg.cholesky(gram_f)
    offset_cart = np.array([float(v) for v in offset]) @ basis
    height = float(height2) ** 0.5
    points = []
    span = range(-4, 5)
    for i in span:
        for j in span:
            points.append(np.array([i, j], float) @ basis)
    points = np.array(points)

    best = 0.0
    grid = np.linspace(-0.5, 0.5, samples)
    for a in grid:
        for b in grid:
            x = np.array([a, b]) @ basis
            values = []
            for layer in (-1, 0, 1, 2):
                delta = x - layer * offset_cart - points
                values.append(np.min(np.einsum("ij,ij->i", delta, delta))
                              + 0.0)
            # max_z min_i (A_i + (z - i t)^2) на [0, t]
            candidates = [0.0, height]
            for p in range(len(values)):
                for q in range(p + 1, len(values)):
                    i, j = (-1, 0, 1, 2)[p], (-1, 0, 1, 2)[q]
                    z = ((values[q] - values[p] + (j * j - i * i) * height ** 2)
                         / (2 * (j - i) * height))
                    candidates.append(min(max(z, 0.0), height))
            for z in candidates:
                value = min(values[k] + (z - (-1, 0, 1, 2)[k] * height) ** 2
                            for k in range(4))
                best = max(best, value)
    return best


@pytest.mark.parametrize("offset,height2", [
    ((Fr(1, 3), Fr(2, 3)), Fr(1)),         # глубокая дыра A2
    ((Fr(1, 5), Fr(2, 5)), Fr(3, 4)),
    ((Fr(3, 7), Fr(1, 7)), Fr(5, 4)),
])
def test_exact_bound_dominates_float_truth(offset, height2):
    """Сертификат обязан быть верхней оценкой честного численного максимума."""
    geometry = _geometry(A2, offset, height2, A2_R2)
    report = certify_covering_radius(geometry, None, verbose=False)
    truth = _float_max_phi(A2, offset, height2, A2_R2)
    exact = float(report["max_phi"])
    assert exact >= truth - 1e-9
    # и не слишком грубой: двухслойное ограничение и мажоранта phi >= U
    # дают запас, но не порядок величины
    assert exact <= truth * 1.35 + 0.05


def test_deep_hole_shift_is_tight():
    """Сдвиг в глубокую дыру: точка на равном расстоянии от обоих слоёв.

    Тогда существует ``x`` с ``A_0 = A_1 = R_base^2``, и максимум мажоранты
    равен ``R_base^2 + t^2/4`` — тот же ответ, что у выровненных слоёв.
    """
    geometry = _geometry(A2, (Fr(1, 3), Fr(2, 3)), Fr(1), A2_R2)
    report = certify_covering_radius(geometry, None, verbose=False)
    assert report["max_phi"] == A2_R2 + Fr(1, 4)


# ------------------------------------------------------------- отрицательный


def test_impossible_target_rejected():
    geometry = _geometry(A2, (Fr(1, 3), Fr(2, 3)), Fr(1), A2_R2)
    report = certify_covering_radius(geometry, Fr(1, 2), verbose=False)
    assert not report["certified"]
    assert report["failed_pieces"]


def test_easy_target_accepted():
    geometry = _geometry(A2, (Fr(1, 3), Fr(2, 3)), Fr(1), A2_R2)
    report = certify_covering_radius(geometry, Fr(2), verbose=False)
    assert report["certified"]
    assert report["max_phi"] < Fr(2)


# ---------------------------------------------------------- разбор грамиана


def test_geometry_from_gram_recovers_offset_and_height():
    """3x3 грамиан: сдвиг и высота восстанавливаются точно."""
    # база A2, сдвиг (1/3, 2/3), высота^2 = 1
    offset = (Fr(1, 3), Fr(2, 3))
    height2 = Fr(1)
    row = [sum(A2[i][j] * offset[j] for j in range(2)) for i in range(2)]
    last = sum(row[i] * offset[i] for i in range(2)) + height2
    gram = [[A2[0][0], A2[0][1], row[0]],
            [A2[1][0], A2[1][1], row[1]],
            [row[0], row[1], last]]
    scale = 9
    integer_gram = [[int(v * scale) for v in r] for r in gram]
    geometry = geometry_from_gram(integer_gram, scale, A2_R2)
    assert geometry.offset == offset
    assert geometry.height2 == height2
