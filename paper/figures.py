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
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPoly
from scipy.spatial import Voronoi

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 140, "savefig.bbox": "tight", "axes.axisbelow": True,
})
_HERE = Path(__file__).resolve().parent
DATA = _HERE.parent / "audit-data" / "results"   # ../audit-data/results
OUT = _HERE                                      # каталог этого файла (paper/)
BLUE, RED, GREEN, ORANGE, PURPLE = "#2456a6", "#c0392b", "#1e8449", "#e08a00", "#7d3c98"


def load_json(name):
    """JSON-артефакт кампании из каталога результатов аудита."""
    return json.loads((DATA / name).read_text())


def load_json_opt(name):
    """То же, что load_json, но None, если артефакта нет (он не обязателен)."""
    try:
        return load_json(name)
    except FileNotFoundError:
        return None


def num(x, nd=4):
    """Число как формула с русской десятичной запятой: 1.6073 -> $1{,}6073$."""
    return "$" + f"{x:.{nd}f}".replace(".", "{,}") + "$"


def running_max(rows):
    """Бегущий максимум d по возрастанию k: (список k, список максимумов)."""
    ks, ds, best = [], [], 0.0
    for r in sorted(rows, key=lambda r: r["k"]):
        best = max(best, r["d"])
        ks.append(r["k"])
        ds.append(best)
    return ks, ds


# --- Fig 3: лестницы d(k) для R^2, R^3, R^4 ---
ca = load_json("campaign_a.json")
cc = load_json("campaign_c.json")


def merged_rows(tables):
    """Максимум d по каждому k среди нескольких решёток одной размерности."""
    best = {}
    for table in tables:
        for r in table:
            best[r["k"]] = max(best.get(r["k"], 0.0), r["d"])
    return [{"k": k, "d": d} for k, d in best.items()]

fig, ax = plt.subplots(figsize=(7.2, 4.3))
for tables, colour, label in [
        ([ca["Z2"], ca["A2"]], BLUE,
         r"$\mathbb{R}^2$  ($\mathbb{Z}^2,A_2$)"),
        ([ca["Z3"], ca["FCC"], ca["BCC"]], GREEN,
         r"$\mathbb{R}^3$  ($\mathbb{Z}^3,\mathrm{FCC},\mathrm{BCC}$)"),
        ([cc["D4"], cc["A4s"], cc["K3,3"], cc["111-"]], RED,
         r"$\mathbb{R}^4$  ($D_4,A_4^*,K_{3,3},111^-$)")]:
    ks, ds = running_max(merged_rows(tables))
    ax.step(ks, ds, where="post", color=colour, lw=1.4, label=label)
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
fig.savefig(OUT / "fig_staircase.pdf")
plt.close(fig)
print("fig_staircase.pdf")

# --- Fig 5: закон (3+w): d vs отношение покрытие/упаковка ---
fig, ax = plt.subplots(figsize=(6.6, 4.2))
x = np.linspace(1.1, 1.75, 200)
ax.plot(x, math.sqrt(7/3) / x, color=PURPLE, lw=1.5,
        label=r"$d=\sqrt{7/3}\,/\,(2R/\lambda_1)$")
lattice_marks = [(2/math.sqrt(3), math.sqrt(7)/2, r"$A_2$"),
                 (math.sqrt(2), math.sqrt(7/6), r"$D_4,E_6^*,E_8,\Lambda_{24}$"),
                 (math.sqrt(8/3), math.sqrt(7/3)/math.sqrt(8/3), r"$K_{12}$")]
for rr, dd, lab in lattice_marks:
    col = GREEN if dd >= 1 else RED
    ax.plot([rr], [dd], "o", color=col, ms=3.6, zorder=5)
    ax.annotate(lab, (rr, dd), (rr - 0.02, dd + 0.06), fontsize=9,
                ha="center", color=col)
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.text(1.5, 1.02, r"порог $d=1$", fontsize=9)
ax.set_xlabel(r"отношение покрытие/упаковка $2R/\lambda_1$")
ax.set_ylabel(r"ширина запрещённого интервала $d$")
ax.set_xlim(1.1, 1.72); ax.set_ylim(0.8, 1.5); ax.legend(loc="upper right")
fig.savefig(OUT / "fig_eisenstein.pdf")
plt.close(fig)
print("fig_eisenstein.pdf")

# --- Fig 1: схема метода — 7-раскраска шестиугольной решётки A2 ---
v1 = np.array([1.0, 0.0]); v2 = np.array([0.5, math.sqrt(3) / 2])
coords = [(a, b) for a in range(-6, 7) for b in range(-6, 7)]
pts = np.array([a * v1 + b * v2 for a, b in coords])
vor = Voronoi(pts)
# палитра 7 цветов (мягкая)
pal = ["#e8eef7", "#f5d9d0", "#d6ead9", "#f7f0cf", "#ddd5ea", "#d0e3ea", "#f2dbe8"]
fig, ax = plt.subplots(figsize=(6.6, 5.4))
patches, colors = [], []
for (a, b), region_idx in zip(coords, vor.point_region):
    reg = vor.regions[region_idx]
    if not reg or -1 in reg:
        continue
    patches.append(MplPoly(vor.vertices[reg], closed=True))
    colors.append(pal[(a + 3 * b) % 7])       # классическая 7-раскраска (Исбелл)
pc = PatchCollection(patches, facecolor=colors, edgecolor="#888", lw=0.6)
ax.add_collection(pc)
# выделяем центральную ячейку (цвет 0) и ближайшую одноцветную
c0 = np.array([0.0, 0.0])
same = [p for p, (a, b) in zip(pts, coords) if (a + 3 * b) % 7 == 0 and np.linalg.norm(p) > 1e-6]
c1 = min(same, key=np.linalg.norm)
ax.plot(*c0, "o", color="#c0392b", ms=3.4, zorder=6)
ax.plot(*c1, "o", color="#c0392b", ms=3.4, zorder=6)
# отрезок D между ближайшими точками ячеек: вдоль линии центров, укороченный
# с обоих концов на инрадиус ячейки (он равен 1/2)
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
fig.savefig(OUT / "fig_method.pdf")
plt.close(fig)
print("fig_method.pdf")

# --------------------------------------------------------------------------
# fig_budget: продуктовое исчисление -- ширина как расходуемый ресурс
# --------------------------------------------------------------------------
lad2 = load_json("ladder2d.json")
fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 3.4))

left = 0.0
for label, block_cost, colour in [("$E_8/2401$", 6 / 7, BLUE), ("", 4 / 31, GREEN)]:
    axL.barh(0, block_cost, left=left, height=0.40, color=colour, edgecolor="white")
    if label:
        axL.text(left + block_cost / 2, 0, label, ha="center", va="center",
                 color="white", fontsize=10.5)
    left += block_cost
axL.annotate(r"$aA_2/19$:  $4/31$", xy=(6 / 7 + 2 / 31, 0.20), xytext=(0.46, 0.62),
             fontsize=9.5, color=GREEN,
             arrowprops=dict(arrowstyle="->", lw=0.9, color=GREEN))
axL.barh(0, 1 - left, left=left, height=0.40, color="#cfd6e0", edgecolor="white")
axL.annotate(r"запас $3/217$", xy=(left + (1 - left) / 2, -0.20), xytext=(1.02, -0.52),
             fontsize=9.5, color="0.30",
             arrowprops=dict(arrowstyle="->", lw=0.9, color="0.45"))
axL.text(1.02, -0.78, r"$\ell=\sqrt{217/214}$", fontsize=9.5, color="0.30")
axL.barh(-1.05, 12 / 7, height=0.40, color=BLUE, alpha=0.40, edgecolor="white")
axL.text(0.85, -1.05, r"два блока $E_8/2401$:  $12/7>1$", ha="center", va="center",
         fontsize=9.5, color="#16305e")
axL.axvline(1.0, color=RED, lw=1.5)
axL.text(1.01, 0.62, "бюджет $=1$", color=RED, fontsize=9.5)
axL.set_xlim(0, 1.80); axL.set_ylim(-1.45, 0.90)
axL.set_yticks([]); axL.set_xlabel(r"израсходовано $\sum_i 1/d_i^2$")
axL.set_title(r"$\sum_i 1/d_i^2 \leq 1$:  "
              r"$\chi(\mathbb{R}^{10}) \leq 2401\cdot 19 = 45619$", fontsize=10.5)
axL.grid(axis="x", alpha=0.3)

flat_k = [r["index"] for r in lad2["records"]]
flat_cost = [1.0 / r["d"] ** 2 for r in lad2["records"]]
axR.axvspan(2, 16.5, color="0.86", alpha=0.7, zorder=0)
axR.plot(flat_k, flat_cost, "o-", color=BLUE, ms=2.8, lw=1.0, zorder=3)
axR.axhline(1 / 7, color=RED, lw=1.3, ls="--", zorder=2)
axR.text(2.6, 1 / 7 * 1.07, r"остаток бюджета $1/7$,  т.е. $d\geq\sqrt{7}$",
         color=RED, fontsize=9, va="bottom")
axR.text(8.6, 1.9, "запрещено экранами\nМинковского:  $k\\leq 16$",
         ha="center", fontsize=9, color="0.25")
for k in (16, 17, 18):
    axR.plot([k], [flat_cost[flat_k.index(k)]], "o", color=ORANGE, ms=4.2, zorder=5)
axR.plot([19], [flat_cost[flat_k.index(19)]], "*", color=GREEN, ms=9.5, zorder=6)
axR.annotate(r"$k=19$", xy=(19, flat_cost[flat_k.index(19)]), xytext=(19.6, 0.34),
             fontsize=10.5, color=GREEN,
             arrowprops=dict(arrowstyle="->", lw=0.9, color=GREEN))
axR.set_yscale("log"); axR.set_xlim(2, 23)
axR.set_xlabel("индекс плоского блока $k$"); axR.set_ylabel(r"цена $1/d^2$")
axR.set_title("плоская лестница: где она пробивает остаток", fontsize=10.5)
fig.tight_layout()
fig.savefig(OUT / "fig_budget.pdf")
plt.close(fig)
print("fig_budget.pdf")


# --------------------------------------------------------------------------
# fig_spacer: почему плоский проставок стоит ровно 19
# --------------------------------------------------------------------------
OM = complex(-0.5, math.sqrt(3) / 2)
HEXV = np.array([[math.cos(math.pi / 6 + k * math.pi / 3),
                  math.sin(math.pi / 6 + k * math.pi / 3)]
                 for k in range(6)]) / math.sqrt(3)
HEXN = np.array([[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)]
                 for k in range(6)])


def hex_nearest(p):
    """Ближайшая точка шестиугольной ячейки Вороного A2 и расстояние до неё."""
    p = np.asarray(p, float)
    if np.all(HEXN @ p <= 0.5 + 1e-12):
        return p, 0.0
    best_q, best_d = None, math.inf
    for k in range(6):
        a, b = HEXV[k], HEXV[(k + 1) % 6]
        t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
        q = a + t * (b - a)                   # ближайшая точка k-го ребра
        d = float(np.linalg.norm(p - q))
        if d < best_d:
            best_q, best_d = q, d
    return best_q, best_d


THR = math.sqrt(7 / 3)
fig, ax = plt.subplots(figsize=(6.4, 5.2))
gx, gy = np.meshgrid(np.linspace(-3.0, 3.4, 460), np.linspace(-2.6, 2.8, 400))
Z = np.array([hex_nearest(q)[1] for q in np.column_stack([gx.ravel(), gy.ravel()])])
Z = Z.reshape(gx.shape)
ax.contourf(gx, gy, Z, levels=[-1, THR], colors=["#f7dcd8"])
ax.contour(gx, gy, Z, levels=[THR], colors=[RED], linewidths=1.5)
ax.add_patch(plt.Polygon(HEXV, closed=True, fc="#dae3f3", ec=BLUE, lw=1.8, zorder=3))
ax.text(0, 0, r"$V_0$", ha="center", va="center", color=BLUE, fontsize=11, zorder=4)

marks = [((3, 1), r"$3+\omega$", 7, (-2.35, 1.72)),
         ((3, 0), r"$3$", 9, (0.15, -1.42)),
         ((4, 0), r"$4$", 16, (-0.35, -2.42)),
         ((5, 2), r"$5+2\omega$", 19, (0.62, 2.05))]
for (a, b), tag, k, tpos in marks:
    z = (a + b * OM) / 2
    P = np.array([z.real, z.imag])
    q, d = hex_nearest(P)
    ok = d >= THR - 1e-12
    col = GREEN if ok else RED
    ax.plot([P[0], q[0]], [P[1], q[1]], "-", color=col, lw=0.9, alpha=0.8, zorder=4)
    ax.plot(*P, "o", color=col, ms=4.6, zorder=6, mec="white", mew=0.6)
    ax.annotate(f"$\\alpha={tag[1:-1]}$,  $k={k}$\n"
                r"$\mathrm{dist}=$" + num(d),
                xy=P, xytext=tpos, fontsize=9, color=col, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=col, alpha=0.6))
ax.set_aspect("equal"); ax.set_xlim(-3.0, 3.4); ax.set_ylim(-2.6, 2.8)
ax.set_title(r"порог $\mathrm{dist}(\alpha/2,\,V_0)\geq\sqrt{7/3}=1{,}5275$"
             "\nрозовое запрещено; $k=16$ не дотягивает $1{,}8\\%$", fontsize=10.5)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "fig_spacer.pdf")
plt.close(fig)
print("fig_spacer.pdf")


# --------------------------------------------------------------------------
# fig_shells: почему слой ранга 2 не проходит
# --------------------------------------------------------------------------
sh = load_json("layer_shells.json")
lo = min(x for key in ("rank1", "rank2") for x, _ in sh[key]["shells"]) - 0.25
hi = 6.35
fig, axes = plt.subplots(2, 1, figsize=(8.2, 4.4), sharex=True)
for ax, key, colour, title in [
        (axes[0], "rank1", GREEN, r"$\mathbb{R}^{10}$, слой ранга 1 ($A_2$): индекс 21609"),
        (axes[1], "rank2", RED, r"$\mathbb{R}^{12}$, слой ранга 2: индекс 345744")]:
    rec = sh[key]
    w = rec["window"]
    inside = [(x, c) for x, c in rec["shells"] if x <= w]
    ax.axvspan(lo, w, color=colour, alpha=0.10, zorder=0)
    ax.axvline(w, color="0.30", ls="--", lw=1.2, zorder=2)
    for x, c in rec["shells"]:
        good = x <= w
        ax.vlines(x, 0, c, color=colour if good else "0.72", lw=2.0 if good else 1.3,
                  zorder=3)
        ax.plot(x, c, "o", color=colour if good else "0.72", ms=3.6, zorder=4)
    if key == "rank2":
        ax.annotate("три орбиты почти вырождены\n"
                    r"($3{,}2909\ /\ 3{,}2913\ /\ 3{,}2916$)",
                    xy=(3.2913, 6), xytext=(4.35, 7.6), fontsize=8.5, color=colour,
                    arrowprops=dict(arrowstyle="->", lw=0.8, color=colour))
    else:
        ax.annotate("следующая оболочка далеко\nза окном", xy=(5.9756, 6),
                    xytext=(4.55, 8.9), fontsize=8.5, color="0.35",
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="0.55"))
    ax.text(w - 0.06, 8.9, "граница окна", fontsize=8.5, color="0.30",
            ha="right")
    ax.set_title(f"{title}   —   в окне орбит: {len(inside)}, "
                 f"векторов: {sum(c for _, c in inside)}",
                 fontsize=10, loc="left")
    ax.set_ylabel("кратность"); ax.set_ylim(0, 10.5); ax.set_xlim(lo, hi)
axes[1].set_xlabel(r"длина слоевого вектора $|c|$")
fig.suptitle("поправки обязаны вытолкнуть за диаметр каждый класс из окна", fontsize=10.5)
fig.tight_layout()
fig.savefig(OUT / "fig_shells.pdf")
plt.close(fig)
print("fig_shells.pdf")
print("DONE")
