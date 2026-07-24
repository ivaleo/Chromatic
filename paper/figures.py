"""Генерация иллюстраций для статьи (векторные PDF).

Запуск:  python figures.py   (из каталога paper/ или любого другого — пути
относительны расположению этого файла). Читает json из ../audit-data,
пишет fig_*.pdf рядом с собой.
"""
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 140, "savefig.bbox": "tight", "axes.axisbelow": True,
})
_HERE = Path(__file__).resolve().parent
DATA = str(_HERE.parent / "audit-data")   # ../audit-data
OUT = str(_HERE)                          # каталог этого файла (paper/)
BLUE, RED, GREEN, ORANGE, PURPLE = "#2456a6", "#c0392b", "#1e8449", "#e08a00", "#7d3c98"


def running_max(rows):
    ks, ds, best = [], [], 0.0
    for r in sorted(rows, key=lambda r: r["k"]):
        best = max(best, r["d"])
        ks.append(r["k"]); ds.append(best)
    return ks, ds


# --- Fig 3: лестницы d(k) для R^2, R^3, R^4 ---
ca = json.load(open(f"{DATA}/campaign_a.json"))
cc = json.load(open(f"{DATA}/campaign_c.json"))

def merged_rows(dicts):
    """бегущий максимум d по нескольким решёткам одной размерности."""
    bykey = {}
    for d in dicts:
        for r in d:
            bykey[r["k"]] = max(bykey.get(r["k"], 0.0), r["d"])
    return [{"k": k, "d": v} for k, v in bykey.items()]

fig, ax = plt.subplots(figsize=(7.2, 4.3))
r2 = running_max(merged_rows([ca["Z2"], ca["A2"]]))
r3 = running_max(merged_rows([ca["Z3"], ca["FCC"], ca["BCC"]]))
r4 = running_max(merged_rows([cc["D4"], cc["A4s"], cc["K3,3"], cc["111-"]]))
ax.step(r2[0], r2[1], where="post", color=BLUE, lw=1.4, label=r"$\mathbb{R}^2$  ($\mathbb{Z}^2,A_2$)")
ax.step(r3[0], r3[1], where="post", color=GREEN, lw=1.4, label=r"$\mathbb{R}^3$  ($\mathbb{Z}^3,\mathrm{FCC},\mathrm{BCC}$)")
ax.step(r4[0], r4[1], where="post", color=RED, lw=1.4, label=r"$\mathbb{R}^4$  ($D_4,A_4^*,K_{3,3},111^-$)")
ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.7)
ax.text(2, 1.02, r"порог $d=1$ (пригодная раскраска)", fontsize=9)
ax.set_xlabel(r"число цветов $k$"); ax.set_ylabel(r"максимальная ширина $d=D/\mathrm{diam}\,V_0$")
ax.set_xlim(2, 82); ax.set_ylim(0, 3.2); ax.legend(loc="upper left", framealpha=0.95)
ax.annotate(r"$k=7$", (7, 1.32), (10, 1.7), fontsize=9, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE))
ax.annotate(r"$k=15$", (15, 1.0), (20, 0.55), fontsize=9, color=GREEN,
            arrowprops=dict(arrowstyle="->", color=GREEN))
ax.annotate(r"$k=49$ ($D_4$)", (49, 1.08), (52, 0.5), fontsize=9, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED))
fig.savefig(f"{OUT}/fig_staircase.pdf")
plt.close(fig)
print("fig_staircase.pdf")

# --- Fig 4: спуск к порогу в R^4 (max d при точном индексе k) ---
n2 = json.load(open(f"{DATA}/n2_4d_frontier.json"))
try:
    n6 = json.load(open(f"{DATA}/n6_push45.json"))
    d45 = n6["d"]
except Exception:
    d45 = json.load(open(f"{DATA}/n5_cascade.json"))["k45"]["d"]
# лучшее найденное d при точном индексе k из всех кампаний (только реальные данные)
n5 = json.load(open(f"{DATA}/n5_cascade.json"))
best_at = {
    45: d45,
    46: json.load(open(f"{DATA}/n4_push46.json"))["d"],
    47: max(n2["k47"]["d"], n5.get("k47", {}).get("d", 0.0)),
    48: json.load(open(f"{DATA}/r5_push48.json"))["k48"]["d"],
}
for k in range(49, 57):
    best_at[k] = n2[f"width{k}"]["d"]
ks = sorted(best_at)
ds = [best_at[k] for k in ks]
fig, ax = plt.subplots(figsize=(7.2, 4.3))
above = [k >= 1.0 for k in ds]
ax.axhline(1.0, color="k", ls="--", lw=1.2)
ax.axhspan(1.0, 1.25, color=GREEN, alpha=0.06)
ax.axhspan(0.9, 1.0, color=RED, alpha=0.06)
for k, d in zip(ks, ds):
    col = GREEN if d >= 1.0 else RED
    ax.plot([k], [d], "o", color=col, ms=5.5, zorder=5)
ax.plot(ks, ds, "-", color="gray", lw=0.8, alpha=0.6, zorder=1)
ax.annotate(r"$k=45$: $d=%.4f$ (рекорд, сертификат)" % d45, (45, d45),
            (45.4, 1.15), fontsize=9, color=GREEN,
            arrowprops=dict(arrowstyle="->", color=GREEN))
ax.annotate(r"$k=48$: $d=1.0433$", (48, best_at[48]), (49.5, 1.17), fontsize=9, color=GREEN,
            arrowprops=dict(arrowstyle="->", color=GREEN))
ax.annotate(r"$k=47$: $d=%.4f<1$" % best_at[47], (47, best_at[47]), (47, 0.93),
            fontsize=9, color=RED, ha="center",
            arrowprops=dict(arrowstyle="->", color=RED))
ax.text(53.3, 1.02, "область пригодных\nраскрасок ($d\\geq1$)", fontsize=8.5, color=GREEN)
ax.set_xlabel(r"число цветов $k$ (точный индекс подрешётки)")
ax.set_ylabel(r"наилучшая найденная ширина $d(k)$")
ax.set_xlim(44.3, 56.5); ax.set_ylim(0.9, 1.22)
fig.savefig(f"{OUT}/fig_descent.pdf")
plt.close(fig)
print("fig_descent.pdf, d(45) =", d45)

# --- Fig 5: закон (3+w): d vs отношение покрытие/упаковка ---
fig, ax = plt.subplots(figsize=(6.6, 4.2))
x = np.linspace(1.1, 1.75, 200)
ax.plot(x, math.sqrt(7/3) / x, color=PURPLE, lw=1.5,
        label=r"$d=\sqrt{7/3}\,/\,(2R/\lambda_1)$")
pts = [(2/math.sqrt(3), math.sqrt(7)/2, r"$A_2$"),
       (math.sqrt(2), math.sqrt(7/6), r"$D_4,E_6^*,E_8,\Lambda_{24}$"),
       (math.sqrt(8/3), math.sqrt(7/3)/math.sqrt(8/3), r"$K_{12}$")]
for rr, dd, lab in pts:
    col = GREEN if dd >= 1 else RED
    ax.plot([rr], [dd], "o", color=col, ms=5.5, zorder=5)
    ax.annotate(lab, (rr, dd), (rr - 0.02, dd + 0.06), fontsize=9,
                ha="center", color=col)
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.text(1.5, 1.02, r"порог $d=1$", fontsize=9)
ax.set_xlabel(r"отношение покрытие/упаковка $2R/\lambda_1$")
ax.set_ylabel(r"ширина запрещённого интервала $d$")
ax.set_xlim(1.1, 1.72); ax.set_ylim(0.8, 1.5); ax.legend(loc="upper right")
fig.savefig(f"{OUT}/fig_eisenstein.pdf")
plt.close(fig)
print("fig_eisenstein.pdf")

# --- Fig 1: схема метода — 7-раскраска шестиугольной решётки A2 ---
from scipy.spatial import Voronoi
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection

v1 = np.array([1.0, 0.0]); v2 = np.array([0.5, math.sqrt(3) / 2])
pts, coords = [], []
for a in range(-6, 7):
    for b in range(-6, 7):
        pts.append(a * v1 + b * v2); coords.append((a, b))
pts = np.array(pts)
vor = Voronoi(pts)
# палитра 7 цветов (мягкая)
pal = ["#e8eef7", "#f5d9d0", "#d6ead9", "#f7f0cf", "#ddd5ea", "#d0e3ea", "#f2dbe8"]
fig, ax = plt.subplots(figsize=(6.6, 5.4))
patches, colors = [], []
for (a, b), pr in zip(coords, vor.point_region):
    reg = vor.regions[pr]
    if not reg or -1 in reg:
        continue
    poly = vor.vertices[reg]
    c = (a + 3 * b) % 7                       # классическая 7-раскраска (Исбелл)
    patches.append(MplPoly(poly, closed=True))
    colors.append(pal[c])
pc = PatchCollection(patches, facecolor=colors, edgecolor="#888", lw=0.6)
ax.add_collection(pc)
# выделяем центральную ячейку (цвет 0) и ближайшую одноцветную
c0 = np.array([0.0, 0.0])
same = [p for p, (a, b) in zip(pts, coords) if (a + 3 * b) % 7 == 0 and np.linalg.norm(p) > 1e-6]
c1 = min(same, key=np.linalg.norm)
ax.plot(*c0, "o", color="#c0392b", ms=5, zorder=6)
ax.plot(*c1, "o", color="#c0392b", ms=5, zorder=6)
# отрезок D между ближайшими точками ячеек (вдоль линии центров, минус по «радиусу» ячейки)
inr = math.sqrt(3) / 2 * (1 / math.sqrt(3))   # инрадиус ячейки = 1/2
u = (c1 - c0) / np.linalg.norm(c1 - c0)
ax.annotate("", (c1 - u * 0.5), (c0 + u * 0.5),
            arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.8))
mid = (c0 + c1) / 2
ax.text(mid[0] + 0.15, mid[1] + 0.15, r"$D(\Gamma)$", color="#c0392b", fontsize=13)
# диаметр центральной ячейки
reg0 = vor.regions[vor.point_region[coords.index((0, 0))]]
vv = vor.vertices[reg0]
i0 = int(np.argmax(vv[:, 0])); i1 = int(np.argmin(vv[:, 0]))
ax.annotate("", vv[i1], vv[i0], arrowprops=dict(arrowstyle="<->", color="#2456a6", lw=1.6))
ax.text(-0.15, -0.42, r"$\mathrm{diam}\,V_0$", color="#2456a6", fontsize=12, ha="center")
ax.set_xlim(-3.4, 3.4); ax.set_ylim(-3.0, 3.0); ax.set_aspect("equal")
ax.axis("off")
ax.set_title(r"7-раскраска $\mathbb{R}^2$ решёткой $A_2$: одноцветные ячейки на расстоянии $D(\Gamma)$",
             fontsize=10.5)
fig.savefig(f"{OUT}/fig_method.pdf")
plt.close(fig)
print("fig_method.pdf")

# --- Fig 2: числовая ось запрещённого интервала ---
fig, ax = plt.subplots(figsize=(7.2, 1.9))
diam, D = 1.0, 1.9
ax.axhline(0, color="k", lw=1)
# реализуемые расстояния (внутри ячейки и между ячейками) — зелёным
ax.plot([0, diam], [0, 0], color=GREEN, lw=5.5, solid_capstyle="butt")
ax.plot([D, 3.0], [0, 0], color=GREEN, lw=5.5, solid_capstyle="butt")
# запрещённый (свободный) интервал — красная штриховка
ax.plot([diam, D], [0, 0], color=RED, lw=5.5, alpha=0.25, solid_capstyle="butt")
ax.plot([diam, D], [0, 0], color=RED, lw=1.5, ls=(0, (2, 2)))
for x, lab, dy in [(0, "0", -0.42), (diam, r"$\mathrm{diam}\,V_0$", 0.28),
                   (D, r"$D(\Gamma)$", 0.28)]:
    ax.plot([x, x], [-0.08, 0.08], color="k", lw=1)
    ax.text(x, dy, lab, ha="center", fontsize=11)
# отрезок [1, l] внутри свободного интервала (после нормировки diam=1)
ax.annotate("", (1.7, -0.32), (1.02, -0.32),
            arrowprops=dict(arrowstyle="<->", color="#333", lw=1.4))
ax.text(1.36, -0.62, r"запрещаемый отрезок $[1,\ell]$", ha="center", fontsize=10)
ax.text(0.5, 0.42, "внутри\nодной ячейки", ha="center", fontsize=8.5, color=GREEN)
ax.text(2.45, 0.42, "между одноцветными\nячейками", ha="center", fontsize=8.5, color=GREEN)
ax.text((diam + D) / 2, 0.55, "не реализуется\n(свободно)", ha="center", fontsize=8.5, color=RED)
ax.set_xlim(-0.25, 3.05); ax.set_ylim(-0.9, 0.9); ax.axis("off")
ax.set_title(r"Расстояния между одноцветными точками: пригодность при $d=D/\mathrm{diam}\,V_0>1$",
             fontsize=10.5)
fig.savefig(f"{OUT}/fig_interval.pdf")
plt.close(fig)
print("fig_interval.pdf")
print("DONE")
