"""Q-поиск на границе 4D: параллельно 8 инстансов Q-поиска (разные семена) на
каждый индекс k, плюс «тёплые» старты от известных победителей. Глобальный
сброс бруса должен пробить барьеры, где Нелдер–Мид застревает (k=44: NM дал 0.966).

Использование: python q_frontier.py <k1> [k2 ...]
"""
import json
import sys
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, "/Users/mac/Documents/_My_code/Chromatic/audit-data")
from qsearch import qsearch, cholesky_unpack, make_objective, default_bounds


def one_instance(args):
    k, seed, budget, warm = args
    import numpy as np
    from qsearch import qsearch, make_objective, default_bounds, cholesky_unpack
    f = make_objective(4, k, threads=1)
    xl, xg = default_bounds(4)
    # тёплый старт: если дан warm (вектор параметров), сузим брус вокруг него
    if warm is not None:
        w = np.asarray(warm)
        # брус = пересечение домена и окрестности warm (даём Q-поиску и глобальность,
        # и локальную зацепку — первый inner-цикл стартует из центра ~ warm)
        xl = np.maximum(xl, w - 1.0)
        xg = np.minimum(xg, w + 1.0)
    bx, bf = qsearch(f, xl, xg, max_prob=budget, eps_brus=1e-4, seed=seed)
    return (-bf, bx.tolist())


def pack_lower(Q, dim=4):
    return np.linalg.cholesky(Q)[np.tril_indices(dim)].tolist()


if __name__ == "__main__":
    ks = [int(a) for a in sys.argv[1:]] or [44]
    # известные победители как тёплые старты
    warms = []
    for fn in ("n6_push45.json", "n4_push46.json"):
        try:
            Q = np.array(json.load(open(f"/Users/mac/Documents/_My_code/Chromatic/audit-data/{fn}"))["Q"])
            warms.append(pack_lower(Q))
        except Exception:
            pass

    out = {}
    with Pool(8) as pool:
        for k in ks:
            # 6 «холодных» глобальных инстансов (из центра, полный домен) + 2 тёплых
            jobs = [(k, s, 2500, None) for s in range(6)]
            jobs += [(k, 100 + i, 2000, w) for i, w in enumerate(warms[:2])]
            res = pool.map(one_instance, jobs)
            bd, bx = max(res, key=lambda t: t[0])
            alld = sorted(round(d, 5) for d, _ in res)
            print(f"k={k}: Q-поиск max d = {bd:.7f}  {'>=1 ПРОБОЙ!' if bd >= 1 else '< 1'}  "
                  f"(инстансы: {alld})", flush=True)
            out[f"k{k}"] = {"d": bd, "Q": cholesky_unpack(np.asarray(bx), 4).tolist()}
    json.dump(out, open("/Users/mac/Documents/_My_code/Chromatic/audit-data/q_frontier.json", "w"),
              indent=1)
    print("DONE", flush=True)
