"""R5: дожим прорыва chi(R^4) <= 48.

(a) продолжение Нелдера-Мида на d(48) от победителя (полировка максимума);
(b) попытки k=47 и k=46 со стартов: победитель-48 и его возмущения;
(c) поиск малознаменательной рациональной формы с d(48) >= 1.02.
Оптимизация — по ПРОИЗВОЛЬНОЙ PD-форме (параметризация Холецкого, 10 параметров,
масштаб фиксируем нормировкой det=1) — конус больше не ограничивает.
"""
import json, math
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize
import combigeo
from chromatic_research.paths import results_path

cand = json.load(open(results_path("r2_k48_candidate.json")))
Q0 = np.array(cand["Q"])
Q0 /= abs(np.linalg.det(Q0)) ** 0.25          # нормировка масштаба

def pack(Q):
    L = np.linalg.cholesky(Q)
    return L[np.tril_indices(4)]

def unpack(x):
    L = np.zeros((4, 4))
    L[np.tril_indices(4)] = x
    Q = L @ L.T
    d = abs(np.linalg.det(Q))
    return Q / d ** 0.25 if d > 1e-12 else None

def d_of(Q, k):
    if Q is None:
        return 0.0
    try:
        B = np.linalg.cholesky(Q + 1e-12 * np.eye(4))
        return combigeo.find_optimal(B.tolist(), index=k).normalized
    except Exception:
        return 0.0

def climb(k, x0, maxfev, tag):
    hist = {"best": 0.0, "x": None}
    def obj(x):
        d = d_of(unpack(x), k)
        if d > hist["best"] + 1e-7:
            hist["best"], hist["x"] = d, x.copy()
            print(f"  [{tag}] d({k}) = {d:.7f}", flush=True)
        return -d
    r = minimize(obj, x0, method="Nelder-Mead",
                 options={"maxfev": maxfev, "xatol": 1e-7, "fatol": 1e-10})
    return (-r.fun, r.x if -r.fun >= hist["best"] else hist["x"])


def main():
    x0 = pack(Q0)
    out = {}

    # (a) полировка k=48
    d48, x48 = climb(48, x0, 500, "polish48")
    print(f"POLISH k=48: d = {d48:.7f}", flush=True)
    Q48 = unpack(x48)
    out["k48"] = {"d": d48, "Q": Q48.tolist()}

    # (b) k=47 и k=46 от победителя и возмущений
    rng = np.random.default_rng(7)
    for k in (47, 46):
        best = (0.0, None)
        for i in range(3):
            s = x48 if i == 0 else x48 * (1 + rng.normal(scale=0.06, size=len(x48)))
            d, x = climb(k, s, 300, f"k{k}s{i}")
            if d > best[0]:
                best = (d, x)
        print(f"BEST k={k}: d = {best[0]:.7f}  {'>=1 !!!' if best[0] >= 1 else '< 1'}", flush=True)
        out[f"k{k}"] = {"d": best[0], "Q": None if best[1] is None else unpack(best[1]).tolist()}

    # (c) малознаменательные рациональные формы вблизи победителя-48
    best_rat = None
    for den in (12, 16, 20, 24, 30, 36, 40, 48, 60, 80, 100, 120):
        s = 48 / abs(np.linalg.det(Q48)) ** 0.25          # удобный масштаб перед округлением
        Qs = Q48 * (den / np.max(np.abs(Q48)))            # нормируем крупнейший элемент к den
        Qr = np.array([[float(Fraction(x).limit_denominator(den)) for x in row] for row in Qs])
        try:
            d = d_of(Qr / abs(np.linalg.det(Qr)) ** 0.25, 48)
        except Exception:
            d = 0.0
        print(f"  [rational den<={den}] d(48) = {d:.7f}", flush=True)
        if d >= 1.0 and (best_rat is None or den < best_rat[0]):
            best_rat = (den, d, Qr.tolist())
    if best_rat:
        out["rational48"] = {"den": best_rat[0], "d": best_rat[1], "Q": best_rat[2]}
        print(f"RATIONAL: den<={best_rat[0]} даёт d = {best_rat[1]:.7f}", flush=True)

    json.dump(out, open(results_path("r5_push48.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
