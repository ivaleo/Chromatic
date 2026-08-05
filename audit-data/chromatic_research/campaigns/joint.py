"""Совместная оптимизация (форма Q, подрешётка Γ) БЕЗ перебора всех подрешёток.

Ключевая идея (снимает барьер больших размерностей): вместо
    d(Q,k) = max по ВСЕМ подрешёткам индекса k  (миллиарды в 5D)
оптимизируем d(Q, Γ) НАПРЯМУЮ по паре (Q, Γ), где Γ — ЦИКЛИЧЕСКАЯ подрешётка
индекса k, заданная последним столбцом HNF c=(c_1,...,c_{n-1}, k). Вычисление d
для КОНКРЕТНОЙ (Q, Γ) дёшево (ячейка строится один раз, дальше — расстояние до
одной подрешётки в границе достаточности). Стоимость одного вызова в 5D падает
с ~50 мин до ~секунд.

Параметры оптимизации: нижнетреугольные элементы Холецкого Q (непрерывные) +
c_1..c_{n-1} (целые в [0,k), релаксируются и округляются).
"""
import numpy as np
import combigeo
from voronoi4d import lattice_points_within, lll_reduce, shortest_vector
from chromatic_research.paths import results_path


def cyclic_transition(c_int, k, dim):
    """HNF циклической подрешётки индекса k: I, последний столбец = (c..., k)."""
    T = np.eye(dim)
    T[:dim - 1, dim - 1] = c_int
    T[dim - 1, dim - 1] = k
    return T


def cholesky_unpack(x, dim):
    L = np.zeros((dim, dim))
    L[np.tril_indices(dim)] = x
    Q = L @ L.T
    d = abs(np.linalg.det(Q))
    return Q / d ** (1.0 / dim) if d > 1e-12 else None


def d_of_pair(B, cell, diam, c_int, k, dim):
    """d = D(Γ)/diam для циклической Γ, с ГОТОВОЙ ячейкой (без перестроения)."""
    T = cyclic_transition(c_int, k, dim)
    sub = T @ B
    sub_l = lll_reduce(sub)
    v0 = shortest_vector(sub_l)
    cur = 2.0 * combigeo.distance_to_cell((0.5 * np.asarray(v0)).tolist(), cell)
    for v in sorted(lattice_points_within(sub_l, cur + diam), key=lambda w: float(w @ w)):
        if float(np.linalg.norm(v)) - diam >= cur:
            break
        cur = min(cur, 2.0 * combigeo.distance_to_cell((0.5 * np.asarray(v)).tolist(), cell))
    return cur / diam


# кэш ячейки по округлённой форме (оптимизатор часто пробует близкие Q)
_cell_cache = {}

def eval_joint(params, dim, k, ncyc=None):
    """params = [Cholesky(Q) (dim(dim+1)/2 шт)] + [c_1..c_{ncyc} релакс.].
    Возвращает d (или 0 при вырождении). ncyc по умолчанию dim-1."""
    if ncyc is None:
        ncyc = dim - 1
    m = dim * (dim + 1) // 2
    Q = cholesky_unpack(np.asarray(params[:m]), dim)
    if Q is None:
        return 0.0
    c_int = [int(round(x)) % k for x in params[m:m + ncyc]]
    try:
        B = np.linalg.cholesky(Q + 1e-12 * np.eye(dim))
    except Exception:
        return 0.0
    key = tuple(np.round(B.ravel(), 5))
    cell = _cell_cache.get(key)
    if cell is None:
        try:
            cell = combigeo.voronoi_cell(B.tolist())
        except Exception:
            return 0.0
        if len(_cell_cache) > 200:
            _cell_cache.clear()
        _cell_cache[key] = cell
    try:
        return d_of_pair(B, cell, cell.diameter, c_int, k, dim)
    except Exception:
        return 0.0


if __name__ == "__main__":
    # ВАЛИДАЦИЯ на известных циклических подрешётках
    import math
    def gram_of(B): return (np.array(B) @ np.array(B).T)
    def chol_params(Q, dim):
        return np.linalg.cholesky(Q / abs(np.linalg.det(Q))**(1/dim))[np.tril_indices(dim)]

    # D4/49: наша HNF была [[1,0,0,2],[0,1,2,4],[0,0,7,0],[0,0,0,7]] — НЕ циклическая.
    # но voronoi4d/combigeo находили и циклический эквивалент. Проверим напрямую
    # нашу 4D-запись k=45 (циклическая c=(15,41,37)):
    import json
    Q45 = np.array([[float(__import__("fractions").Fraction(s)) for s in row]
                    for row in json.load(open(results_path("n6_k45_rational.json")))["Q_fractions"]])
    p = list(chol_params(Q45, 4)) + [15, 41, 37]
    print("4D k=45 циклическая (15,41,37): d =", round(eval_joint(p, 4, 45), 6),
          " (ожидалось 1.016339)")

    # D4/49 классическая: найдём лучшую циклическую c перебором (мал. индекс, 4D)
    D4 = np.array([[2,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]], float)
    pD4 = list(chol_params(gram_of(D4), 4))
    best = 0; bc = None
    B = np.linalg.cholesky(gram_of(D4)/abs(np.linalg.det(gram_of(D4)))**0.25)
    cell = combigeo.voronoi_cell(B.tolist())
    import itertools
    # грубый скан циклических c для D4/49 (49^3 велик — берём разреженную сетку)
    rng = np.random.default_rng(0)
    for _ in range(3000):
        c = rng.integers(0, 49, size=3)
        d = d_of_pair(B, cell, cell.diameter, c, 49, 4)
        if d > best: best, bc = d, c.tolist()
    print(f"D4/49 лучшая циклическая (скан 3000): d = {best:.6f} c={bc} "
          f"(полный оптимум D4/49 = 1.080123, циклич. может быть ниже)")
    print("DONE")
