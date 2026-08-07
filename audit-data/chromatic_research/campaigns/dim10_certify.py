"""Сертификат диаметра для кандидатов ℝ¹⁰ (см. core/layer_certificate).

Usage::

    python -m chromatic_research.campaigns.dim10_certify
"""
import json, math, numpy as np
from chromatic_research.core.layer_certificate import certify_hex_layer_strips
from chromatic_research.campaigns.planar_theorem_check import e8_theta_basis
from chromatic_research.paths import results_path
E8 = e8_theta_basis(); R0 = math.sqrt(1.5); S7 = math.sqrt(7.0)
out = []
for tag in ("3", "4+2w"):
    for r in json.loads(results_path(f"dim10_layer_{tag}.json").read_text())["records"]:
        a, g = r["scale"], np.array(r["offset"])
        need = min(r["D_min"], S7)
        c = certify_hex_layer_strips(E8, g, a, base_covering_radius=R0, strips=8)
        ok = c["diam_upper"] < need
        print(f"index={r['index']} a={a}: diam <= {c['diam_upper']:.9f}  порог {need:.9f}"
              f"  d >= {need/c['diam_upper']:.6f}  {'ПРОХОДИТ' if ok else 'нет'}"
              f"  sound={c['sound']}", flush=True)
        for s in c["strips"]:
            print(f"    h<={s['h_hi']:.4f} окно {s['window']:.4f} -> {s['value']:.7f}", flush=True)
        out.append(dict(index=r["index"], scale=a, D_min=r["D_min"], threshold=need,
                        diam_certified=c["diam_upper"], R2=c["R2_upper"],
                        d_certified=need / c["diam_upper"], admissible=bool(ok),
                        sound=c["sound"], strips=c["strips"]))
results_path("dim10_certificate.json").write_text(json.dumps(out, indent=2))
print("written dim10_certificate.json")
