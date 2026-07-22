"""Q-поиск (q2poisk) — глобальный безградиентный оптимизатор методом
многократного «безответственного» сжатия допустимого бруса с перезапусками.

Точный порт алгоритма Л.Л. Иванова из articles/Qpoisk.c (минимизация),
адаптированный для максимизации нормированной ширины d(Q,k) в наших задачах.

Ключевое отличие от Нелдера–Мида: периодический СБРОС бруса к полному домену
(сохраняя лучшую точку) делает поиск ГЛОБАЛЬНЫМ — он выходит из локальных
бассейнов, где NM застревает.
"""
import numpy as np


def qsearch(f, xl, xg, max_prob=3000, eps_brus=1e-4, seed=0, callback=None):
    """Минимизирует f(x) на брусе [xl, xg] методом Q-поиска.

    :param f: целевая функция (принимает np.ndarray длины n).
    :param xl, xg: нижние/верхние границы (np.ndarray длины n).
    :param max_prob: бюджет — максимум вычислений f.
    :param eps_brus: доля размера бруса, при которой происходит сброс к домену.
    :param seed: инициализация ГСЧ.
    :param callback: необяз. func(best_x, best_f, nprob) при каждом улучшении.
    :return: (best_x, best_f).
    """
    rng = np.random.default_rng(seed)
    xl = np.asarray(xl, float); xg = np.asarray(xg, float)
    n = len(xl)
    dvbrus0 = float(np.sum(xg - xl)) / n

    x = 0.5 * (xl + xg)                 # старт — центр бруса
    ftek = f(x)
    ifsum = 1
    best_x, best_f = x.copy(), ftek

    while ifsum <= max_prob:
        r1, r2 = xl.copy(), xg.copy()   # СБРОС бруса к полному домену
        while ifsum <= max_prob:
            # случайная пробная точка в текущем брусе
            r3 = r1 + rng.random(n) * (r2 - r1)
            fnew = f(r3); ifsum += 1

            if fnew < ftek:
                ftek = fnew
                # сжать брус к СТОРОНЕ улучшения относительно старого x
                below = r3 < x
                r2 = np.where(below, x, r2)
                r1 = np.where(below, r1, x)
                x = r3.copy()
                if ftek < best_f:
                    best_f, best_x = ftek, x.copy()
                    if callback:
                        callback(best_x, best_f, ifsum)
            else:
                # исключить область за неудачной пробой (x остаётся в брусе)
                below = r3 < x
                r1 = np.where(below, r3, r1)
                r2 = np.where(below, r2, r3)

            dvbrus = float(np.sum(r2 - r1)) / n
            if dvbrus / dvbrus0 < eps_brus:
                break                    # брус схлопнулся — на внешний сброс
    return best_x, best_f


# ------------------- применение к нашей задаче d(Q, k) -------------------

def cholesky_unpack(x, dim):
    """Вектор нижнетреугольных параметров -> нормированная (det=1) форма Грама."""
    L = np.zeros((dim, dim))
    L[np.tril_indices(dim)] = x
    Q = L @ L.T
    d = abs(np.linalg.det(Q))
    return Q / d ** (1.0 / dim) if d > 1e-12 else None


def make_objective(dim, k, threads=1):
    """Возвращает f(x) = -d(Q(x), k) для минимизации Q-поиском."""
    import combigeo
    def f(x):
        Q = cholesky_unpack(x, dim)
        if Q is None:
            return 0.0
        try:
            B = np.linalg.cholesky(Q + 1e-12 * np.eye(dim))
            return -combigeo.find_optimal(B.tolist(), index=k, threads=threads).normalized
        except Exception:
            return 0.0
    return f


def default_bounds(dim):
    """Границы параметров Холецкого: диагональ > 0, наддиагональ любого знака."""
    m = dim * (dim + 1) // 2
    xl = np.full(m, -3.0); xg = np.full(m, 3.0)
    # диагональные позиции в tril-порядке
    idx = 0
    for i in range(dim):
        for j in range(i + 1):
            if i == j:
                xl[idx] = 0.2; xg[idx] = 3.0
            idx += 1
    return xl, xg


if __name__ == "__main__":
    import sys, time
    dim = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 45
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    f = make_objective(dim, k)
    xl, xg = default_bounds(dim)
    t0 = time.time()
    def cb(x, fv, npr):
        print(f"  [{npr}] d = {-fv:.6f}", flush=True)
    bx, bf = qsearch(f, xl, xg, max_prob=budget, seed=seed, callback=cb)
    print(f"dim={dim} k={k}: Q-поиск max d = {-bf:.7f}  "
          f"({budget} проб, {time.time()-t0:.0f}s)", flush=True)
