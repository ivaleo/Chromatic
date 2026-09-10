"""Точное перечисление вершин рационального многогранника (double description).

Задача: для ``P = {x in Q^n : A x <= b}``, о котором заранее известно, что он
лежит в кубе ``|x_i| <= M``, получить ПОЛНЫЙ список вершин в точной арифметике.

Метод — алгоритм двойного описания (Моцкин и др.).  Он требует заострённого
стартового конуса, поэтому начинаем не со всего пространства, а с СИМПЛЕКСА

    S = { x : x_i >= -M (i = 1..n),  sum_i x_i <= n*M },

который содержит куб (а значит и ``P``) и имеет ровно ``n+1`` известную
вершину.  Дальше неравенства ``P`` добавляются по одному: каждый шаг делит
текущие вершины на нарушающие/лежащие на гиперплоскости/допустимые и порождает
новые вершины из СМЕЖНЫХ пар (нарушающая, допустимая).

Смежность проверяется комбинаторно: вершины ``u`` и ``v`` смежны тогда и только
тогда, когда ни одна другая вершина не обнуляет надмножество
``Z(u) ∩ Z(v)`` (``Z`` — множество активных неравенств).  Тест точен, поэтому
вырожденные вершины (активных неравенств больше ``n``) обрабатываются штатно —
именно они у этой задачи и встречаются.

Всё считается в ``Fraction``: ни одного округления.  Активные множества
хранятся битовыми масками ``int``.
"""

from __future__ import annotations

from fractions import Fraction as Fr
from typing import Sequence


class DDLimit(Exception):
    """Превышен лимит вершин: вызывающий должен упростить систему."""


def _simplex_start(dim: int, bound: Fr):
    """Вершины и активные маски симплекса ``x_i >= -M``, ``sum x_i <= n*M``.

    Его ``dim + 1`` неравенств нумеруются 0..dim-1 (``-x_i <= M``) и dim
    (``sum x <= n*M``); сами строки нигде дальше не нужны — вершины симплекса
    известны в замкнутом виде, а неравенства задачи получают номера от dim+1.
    """
    verts: list[tuple[Fr, ...]] = []
    masks: list[int] = []
    # все нижние границы активны, сумма — нет
    verts.append(tuple(-bound for _ in range(dim)))
    masks.append((1 << dim) - 1)
    # снята i-я нижняя граница, активна сумма
    for i in range(dim):
        point = [-bound] * dim
        point[i] = bound * dim + bound * (dim - 1)
        verts.append(tuple(point))
        masks.append((((1 << dim) - 1) ^ (1 << i)) | (1 << dim))
    return verts, masks


def dd_vertices(
    rows: Sequence[Sequence[Fr]],
    rhs: Sequence[Fr],
    dim: int,
    bound: Fr,
    *,
    max_vertices: int = 60_000,
) -> list[tuple[Fr, ...]]:
    """Полный точный список вершин ``{x : rows . x <= rhs}``.

    ``bound`` — рациональное ``M`` с гарантией ``|x_i| <= M`` на многограннике
    (корректность результата опирается на это включение).  Пустой список
    означает, что многогранник пуст.
    """
    verts, masks = _simplex_start(dim, Fr(bound))
    offset = dim + 1                      # номера битов симплекса: 0..dim

    for index, (row, value) in enumerate(zip(rows, rhs)):
        row = tuple(Fr(c) for c in row)
        value = Fr(value)
        bit = 1 << (offset + index)
        vals = [
            sum(row[k] * v[k] for k in range(dim)) - value
            for v in verts
        ]
        pos = [i for i, x in enumerate(vals) if x > 0]
        if not pos:
            for i, x in enumerate(vals):
                if x == 0:
                    masks[i] |= bit
            continue
        neg = [i for i, x in enumerate(vals) if x < 0]
        nul = [i for i, x in enumerate(vals) if x == 0]

        new_verts: list[tuple[Fr, ...]] = []
        new_masks: list[int] = []
        for i in nul:
            new_verts.append(verts[i])
            new_masks.append(masks[i] | bit)
        for i in neg:
            new_verts.append(verts[i])
            new_masks.append(masks[i])

        total = len(verts)
        for i in pos:
            mi, vi = masks[i], vals[i]
            for j in neg:
                common = mi & masks[j]
                # необходимое условие: ребро лежит минимум на dim-1
                # гиперплоскостях; дешёвый отсев большинства пар
                if common.bit_count() < dim - 1:
                    continue
                adjacent = True
                for k in range(total):
                    if k == i or k == j:
                        continue
                    if common & masks[k] == common:
                        adjacent = False
                        break
                if not adjacent:
                    continue
                # точка пересечения ребра с гиперплоскостью
                lam = vi / (vi - vals[j])
                point = tuple(
                    verts[i][k] + lam * (verts[j][k] - verts[i][k])
                    for k in range(dim)
                )
                new_verts.append(point)
                new_masks.append(common | bit)

        # дедупликация (у вырожденных вершин одна точка приходит много раз)
        seen: dict[tuple[Fr, ...], int] = {}
        verts, masks = [], []
        for point, mask in zip(new_verts, new_masks):
            found = seen.get(point)
            if found is not None:
                masks[found] |= mask
                continue
            seen[point] = len(verts)
            verts.append(point)
            masks.append(mask)
        if not verts:
            return []
        if len(verts) > max_vertices:
            raise DDLimit(f"{len(verts)} вершин на неравенстве {index}")

    return verts


def recompute_masks(
    verts: Sequence[Sequence[Fr]],
    rows: Sequence[Sequence[Fr]],
    rhs: Sequence[Fr],
) -> list[int]:
    """Пересчитывает точные активные маски (диагностика/проверка)."""
    out = []
    for point in verts:
        mask = 0
        for i, (row, value) in enumerate(zip(rows, rhs)):
            total = sum(Fr(row[k]) * point[k] for k in range(len(point)))
            if total == Fr(value):
                mask |= 1 << i
        out.append(mask)
    return out
