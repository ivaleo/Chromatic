"""Обобщённый CSP-поиск подрешётки через ПРОИЗВЕДЕНИЕ циклических факторгрупп.

Открытие: валидные подрешётки симметричных решёток имеют НЕциклическую
факторгруппу Λ/Λ' = Z/e_1 × ... × Z/e_m (A5*/140: Z/2×Z/70; D4/49: Z/7×Z/7).
Подрешётка = ядро гомоморфизма φ=(φ_1,...,φ_m), φ_j(x)=Σ a_ji x_i mod e_j.
Она ИЗБЕГАЕТ запрещённого f  ⟺  φ(f)≠0  ⟺  ∃j: φ_j(f) ≢ 0 (mod e_j).
Несколько форм ГИБЧЕ одной (циклической) → выполнимо при меньшем индексе.

Индекс = |образ φ| = ∏ e_j (при сюръективности). Ищем φ (коэффициенты форм),
избегающий всё F. Дёшево: |F| проверок на пробу.
"""
import numpy as np
from sympy import Matrix
from sympy.matrices.normalforms import smith_normal_form


def invariant_factor_structures(k):
    """Все структуры Z/e_1×...×Z/e_m с e_1|e_2|...|e_m, ∏e_i=k, e_i>1 (m>=1)."""
    res = []
    def rec(rem, prev, chain):
        if rem == 1:
            if chain:
                res.append(chain[:])
            return
        # следующий множитель e >= max(prev,2), e | rem, prev | e
        for e in range(max(prev, 2), rem + 1):
            if rem % e == 0 and e % prev == 0:
                rec(rem // e, e, chain + [e])
    rec(k, 1, [])
    return res


def sublattice_index(a_forms, e_list, n):
    """Индекс ядра φ = |образ| = ∏ e_j при сюръективности (иначе делитель).
    Возвращает точный индекс через SNF матрицы отображения."""
    # матрица отображения Z^n -> ⊕ Z/e_j: строки — формы, но модули разные.
    # Проще: строим генераторы ядра и берём |det|. Ядро = {x: a_j·x≡0 mod e_j ∀j}.
    # Индекс = |Z^n / ker| = |образ|. Образ ⊆ ∏Z/e_j; сюръективность проверяем рангом.
    # Быстрая проверка: собрать матрицу [diag(e_j) | a_forms] и SNF.
    m = len(e_list)
    # Решётка ker: порождается {e_j·(координата)} и соотношениями. Используем:
    # ker = преобразование HNF системы сравнений. Строим базис ker численно.
    # Матрица A (m x n) со строками a_j; ker в Z^n от системы A x ≡ 0 (покомпонентно mod e_j).
    # Эквивалентно: решётка L = {x∈Z^n : a_j·x ≡ 0 mod e_j}. Её индекс = ∏e_j / |коядро|.
    # Точно через SNF расширенной матрицы:
    rows = []
    for j in range(m):
        rows.append(list(a_forms[j]) + [0] * m)
    for j in range(m):
        r = [0] * n + [0] * m
        r[n + j] = e_list[j]
        rows.append(r)
    # это задаёт отображение; индекс ker в Z^n = произведение нетривиальных инв.факторов
    # проекции. Надёжнее: прямой перебор образа малого размера.
    return None  # используем прямой метод ниже


def index_and_check(a_forms, e_list, F, n):
    """Возвращает (index, avoids_F). index — точный индекс ядра; avoids_F — избегает ли F."""
    m = len(e_list)
    Fa = np.asarray(F, dtype=np.int64)
    # avoids: для каждого f — НЕ (все φ_j(f)≡0)
    killed = np.ones(len(F), dtype=bool)
    for j in range(m):
        res = (Fa @ np.asarray(a_forms[j], dtype=np.int64)) % e_list[j]
        killed &= (res == 0)
    avoids = not killed.any()
    # индекс ядра = |образ φ|. Образ порождён столбцами (a_1i mod e_1,...,a_mi mod e_m).
    # |образ| = |группа|/|коядро|. Считаем образ прямо: подгруппа ∏Z/e_j, порождённая n
    # элементами g_i=(a_1i,...,a_mi). Индекс ker = |образ|.
    G = [tuple(int(a_forms[j][i]) % e_list[j] for j in range(m)) for i in range(n)]
    # порождаем подгруппу перебором (|∏e_j|=k мало)
    from itertools import product as iproduct
    gen = set()
    gen.add(tuple([0] * m))
    frontier = [tuple([0] * m)]
    while frontier:
        x = frontier.pop()
        for g in G:
            y = tuple((x[j] + g[j]) % e_list[j] for j in range(m))
            if y not in gen:
                gen.add(y); frontier.append(y)
    index = len(gen)
    return index, avoids


def search_structure(F, e_list, n, k, ntry=40000, seed=0):
    """Ищет φ с ядром индекса k, избегающим F, для структуры e_list. None если нет.
    Оптимизация: сначала дешёвая проверка avoids (|F| операций), индекс — только
    для избегающих кандидатов (их ~1/k, поэтому редко)."""
    rng = np.random.default_rng(seed)
    m = len(e_list)
    Fa = np.asarray(F, dtype=np.int64)
    for _ in range(ntry):
        a_forms = [rng.integers(0, e_list[j], size=n).astype(np.int64) for j in range(m)]
        killed = np.ones(len(F), dtype=bool)
        for j in range(m):
            killed &= ((Fa @ a_forms[j]) % e_list[j] == 0)
        if killed.any():
            continue                       # не избегает F — дешёвый отказ
        idx, _ = index_and_check(a_forms, e_list, F, n)
        if idx == k:
            return [list(int(x) for x in a) for a in a_forms]
    return None


def search_structure_nested(F, e_list, n, k, outer=4000, inner=20000, seed=0):
    """Вложенный поиск: фиксируем формы малых факторов, фокусно ищем последнюю
    (наибольший модуль) как циклический CSP на остаточном F. Гораздо эффективнее
    для структур с большим последним фактором (напр. Z/2×Z/70)."""
    rng = np.random.default_rng(seed)
    m = len(e_list)
    if m == 1:
        return search_structure(F, e_list, n, k, ntry=inner, seed=seed)
    Fa = np.asarray(F, dtype=np.int64)
    e_last = e_list[-1]
    for _ in range(outer):
        # случайные формы для факторов 0..m-2
        head = [rng.integers(0, e_list[j], size=n) for j in range(m - 1)]
        # остаточное F: те f, что «убиты» всеми головными формами (для них нужна last-форма)
        killed_head = np.ones(len(F), dtype=bool)
        for j in range(m - 1):
            killed_head &= ((Fa @ head[j]) % e_list[j] == 0)
        Fres = Fa[killed_head]                 # для этих f нужна last: φ_last(f) ≢ 0 mod e_last
        if len(Fres) > 6 * e_last:
            continue                            # слишком много — почти наверняка невыполнимо
        # циклический CSP mod e_last для φ_last, избегающий Fres
        fhead = Fres[:, :n] % e_last
        found = None
        # структурные пробы + случайные
        cand = [np.array([pow(t, i + 1, e_last) for i in range(n)], np.int64)
                for t in range(1, min(e_last, 300))]
        for _ in range(inner - len(cand)):
            cand.append(rng.integers(0, e_last, size=n).astype(np.int64))
        for last in cand:
            if len(Fres) == 0 or np.all((fhead @ last) % e_last != 0):
                a_forms = head + [last]
                idx, avoids = index_and_check(a_forms, e_list, F, n)
                if idx == k and avoids:
                    found = [list(int(x) for x in a) for a in a_forms]
                    break
        if found:
            return found
    return None


def find_sublattice_any_structure(F, k, n, ntry=40000, seed=0):
    """Пробует ВСЕ структуры факторгруппы индекса k; возвращает (структура, формы) или None."""
    for e_list in invariant_factor_structures(k):
        r = search_structure(F, e_list, n, k, ntry=ntry, seed=seed)
        if r is not None:
            return e_list, r
    return None, None


if __name__ == "__main__":
    import sys, time
    import combigeo
    sys.path.insert(0, "/Users/mac/Documents/_My_code/Chromatic/audit-data")
    from cyclic_csp import forbidden_coords

    n = 5; k = int(sys.argv[1]) if len(sys.argv) > 1 else 140
    M = np.ones((n + 1, n))
    for j in range(n):
        M[j, j] = -n
    B = np.linalg.cholesky(M.T @ M); B /= abs(np.linalg.det(B)) ** (1.0 / n)
    cell = combigeo.voronoi_cell(B.tolist())
    F, diam = forbidden_coords(B, cell, 1.0, n)
    print(f"A5* |F_1|={len(F)}, структуры индекса {k}: {invariant_factor_structures(k)}", flush=True)
    t = time.time()
    e_list, forms = find_sublattice_any_structure(F, k, n, ntry=30000)
    if forms:
        grp = " × ".join(f"Z/{e}" for e in e_list)
        print(f"НАЙДЕНА подрешётка индекса {k} с факторгруппой {grp}, избегающая F! "
              f"[{time.time()-t:.0f}s]", flush=True)
        print(f"  формы φ: {forms}")
    else:
        print(f"при индексе {k} валидной подрешётки не найдено ни в одной структуре "
              f"[{time.time()-t:.0f}s]", flush=True)
    print("DONE")
