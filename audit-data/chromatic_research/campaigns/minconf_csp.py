"""Min-conflicts (WalkSAT-подобный) локальный поиск валидной подрешётки.

Случайный поиск проваливается: валидные φ вжато-редки из-за коррелированности
запрещённых векторов. Локальный поиск — правильный инструмент для трудных CSP:
стартуем со случайной φ, повторно берём «убитый» запрещённый вектор и ФЛИПАЕМ
коэффициент формы так, чтобы un-kill его, минимизируя суммарное число убитых.

Ищем φ=(φ_1..φ_m) над Z/e_1×..×Z/e_m с φ(f)≠0 для всех f∈F (тогда d>=1 при индексе k).
"""
import numpy as np


def killed_mask(Fa, a_forms, e_list):
    killed = np.ones(len(Fa), dtype=bool)
    for j in range(len(e_list)):
        killed &= ((Fa @ a_forms[j]) % e_list[j] == 0)
    return killed


def min_conflicts(F, e_list, n, max_steps=200000, restarts=40, seed=0):
    """Возвращает a_forms (список форм) с 0 убитых, либо None."""
    rng = np.random.default_rng(seed)
    Fa = np.asarray(F, dtype=np.int64)
    m = len(e_list)
    best_forms, best_killed = None, len(F) + 1
    for _ in range(restarts):
        a = [rng.integers(0, e_list[j], size=n).astype(np.int64) for j in range(m)]
        killed = killed_mask(Fa, a, e_list)
        nk = int(killed.sum())
        for _ in range(max_steps):
            if nk == 0:
                return [[int(x) for x in aj] for aj in a]
            # берём случайный убитый вектор
            ki = np.flatnonzero(killed)
            f_idx = ki[rng.integers(len(ki))]
            # чтобы un-kill: изменить хотя бы одну форму так, чтобы φ_j(f)≢0.
            # перебираем (форма j, координата i, новое значение) — выбираем ход,
            # минимизирующий общее число убитых (с шумом для выхода из локальных ям)
            best_move, best_after = None, nk + 1
            trials = []
            for j in range(m):
                for i in range(n):
                    # достаточно проверить значения, меняющие φ_j(f) mod e_j
                    for val in range(e_list[j]):
                        if val == a[j][i]:
                            continue
                        trials.append((j, i, val))
            rng.shuffle(trials)
            for (j, i, val) in trials[:60]:      # ограничиваем перебор ходов
                old = a[j][i]; a[j][i] = val
                after = int(killed_mask(Fa, a, e_list).sum())
                a[j][i] = old
                if after < best_after:
                    best_after, best_move = after, (j, i, val)
                    if after == 0:
                        break
            if best_move is None or (rng.random() < 0.2):  # шум: случайный ход
                j = rng.integers(m); i = rng.integers(n); val = rng.integers(e_list[j])
                a[j][i] = val
            else:
                j, i, val = best_move; a[j][i] = val
            killed = killed_mask(Fa, a, e_list)
            nk = int(killed.sum())
            if nk < best_killed:
                best_killed = nk
                best_forms = [[int(x) for x in aj] for aj in a]
    return None if best_killed > 0 else best_forms


if __name__ == "__main__":
    import sys, time
    import combigeo
    from chromatic_research.core.cyclic_csp import forbidden_coords
    from chromatic_research.core.general_csp import invariant_factor_structures, index_and_check

    n = 5; k = int(sys.argv[1]) if len(sys.argv) > 1 else 140
    M = np.ones((n + 1, n))
    for j in range(n):
        M[j, j] = -n
    B = np.linalg.cholesky(M.T @ M); B /= abs(np.linalg.det(B)) ** (1.0 / n)
    cell = combigeo.voronoi_cell(B.tolist())
    F, diam = forbidden_coords(B, cell, 1.0, n)
    print(f"A5* |F_1|={len(F)}, k={k}, структуры {invariant_factor_structures(k)}", flush=True)
    for e_list in invariant_factor_structures(k):
        t = time.time()
        r = min_conflicts(F, e_list, n, max_steps=3000, restarts=30, seed=0)
        if r is not None:
            idx, avoids = index_and_check(r, e_list, F, n)
            grp = " × ".join(f"Z/{e}" for e in e_list)
            print(f"  {grp}: НАЙДЕНА (индекс {idx}, avoids={avoids}) [{time.time()-t:.0f}s] φ={r}",
                  flush=True)
        else:
            print(f"  {e_list}: не найдена [{time.time()-t:.0f}s]", flush=True)
    print("DONE", flush=True)
