#include "combigeo/polytope.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "combigeo/lll.hpp"

namespace combigeo {

namespace {

// Геометрический допуск построения (фильтр неравенств, дедупликация вершин,
// инцидентность вершина-фасета). Грубее kEps: погрешности накапливаются
// при решении СЛАУ на пересечениях фасет.
constexpr double kGeomEps = 1e-7;

// Ранг матрицы (Гаусс с частичным выбором ведущего элемента).
// Матрица передаётся по значению и разрушается.
int matrix_rank(Mat a, double tol) {
    if (a.empty()) return 0;
    const std::size_t rows = a.size();
    const std::size_t cols = a[0].size();
    std::size_t rank = 0;
    for (std::size_t col = 0; col < cols && rank < rows; ++col) {
        std::size_t pivot = rank;
        for (std::size_t r = rank + 1; r < rows; ++r)
            if (std::abs(a[r][col]) > std::abs(a[pivot][col])) pivot = r;
        if (std::abs(a[pivot][col]) < tol) continue;
        std::swap(a[pivot], a[rank]);
        for (std::size_t r = rank + 1; r < rows; ++r) {
            const double f = a[r][col] / a[rank][col];
            for (std::size_t c = col; c < cols; ++c) a[r][c] -= f * a[rank][c];
        }
        ++rank;
    }
    return static_cast<int>(rank);
}

// Релевантные векторы решётки, по одному представителю каждой пары ±v —
// СЕРТИФИЦИРОВАННЫЙ перебор (без эвристических окон).
// Теорема Вороного: v релевантен ⟺ ±v — единственные кратчайшие векторы своего
// ненулевого класса смежности Λ/2Λ. Для класса с представителем x минимумы
// класса x + 2Λ находятся точным CVP-перебором по решётке 2Λ вокруг цели -x
// (стартовый радиус — расстояние до точки Бабаи, заведомо достижимое); классы
// с неединственным (с точностью до знака) минимумом фасет не несут и
// пропускаются — их биссекторы содержат ячейку целиком (та же теорема).
std::vector<Vec> relevant_vectors(const Mat& reduced) {
    const std::size_t n = reduced.size();

    // решётка 2Λ в том же (приведённом) кадре
    Mat doubled;
    doubled.reserve(n);
    for (const Vec& row : reduced) doubled.push_back(scaled(row, 2.0));
    const Lattice lat2(doubled);

    std::vector<Vec> out;
    for (unsigned mask = 1; mask < (1u << n); ++mask) {
        std::vector<long> c(n);
        for (std::size_t i = 0; i < n; ++i) c[i] = static_cast<long>((mask >> i) & 1u);
        const Vec x = combination(c, reduced);
        const Vec target = scaled(x, -1.0);

        // все u ∈ 2Λ с |x + u| <= r0 (r0 достижим по Бабаи) ⇒ все минимумы класса
        const double r0 = lat2.babai_distance(target, /*assume_reduced=*/true) + kEps;
        double best2 = std::numeric_limits<double>::infinity();
        std::vector<Vec> minima;
        for (const Vec& u : lat2.vectors_near(target, r0, /*assume_reduced=*/true)) {
            Vec w = add(x, u);
            const double n2 = norm2(w);
            // первый вектор инициализирует best2 (инвариант: best2 конечен ниже)
            if (minima.empty() || n2 < best2 - kEps * (1.0 + best2)) {
                best2 = n2;
                minima.clear();
                minima.push_back(std::move(w));
            } else if (n2 <= best2 + kEps * (1.0 + best2)) {
                minima.push_back(std::move(w));
            }
        }
        // минимумы приходят парами ±w; релевантен ⟺ ровно одна пара
        if (minima.size() == 2) out.push_back(std::move(minima.front()));
    }
    return out;
}

// Масштабная нормировка: при экстремальном масштабе базиса абсолютные допуски
// (kGeomEps и т.п.) теряют смысл — считаем в масштабе det^(1/n) ~ 1 и в конце
// возвращаем результат в исходных единицах. Возвращает 1.0, если нормировка
// не нужна.
double normalization_scale(const Lattice& lat) {
    const double s = std::pow(lat.det(), 1.0 / lat.dim());
    return (s > 1e2 || s < 1e-2) ? s : 1.0;
}

// Базис решётки, поделённый на масштаб нормировки.
Mat rescaled_basis(const Lattice& lat, double s) {
    Mat base = lat.basis();
    if (s != 1.0)
        for (Vec& row : base)
            for (double& x : row) x /= s;
    return base;
}

// Фасеты-биссекторы релевантных векторов: x·v <= |v|^2/2, т.е. x·(v/|v|) <= |v|/2.
// Ячейка центрально-симметрична: каждая пара ±v даёт ДВЕ фасеты.
std::vector<Halfspace> bisector_facets(const Mat& reduced) {
    std::vector<Halfspace> facets;
    for (const Vec& v : relevant_vectors(reduced)) {
        const double len = norm(v);
        Halfspace h;
        h.normal = scaled(v, 1.0 / len);
        h.offset = 0.5 * len;
        h.lattice_vector = v;
        facets.push_back(h);
        h.normal = scaled(v, -1.0 / len);
        h.lattice_vector = scaled(v, -1.0);
        facets.push_back(std::move(h));
    }
    return facets;
}

// Возврат фасет из рабочего масштаба в исходные единицы (нормаль единичная —
// от масштаба не зависит).
void rescale_facets(std::vector<Halfspace>& facets, double s) {
    for (Halfspace& h : facets) {
        h.offset *= s;
        for (double& x : h.lattice_vector) x *= s;
    }
}

}  // namespace

bool VoronoiCell::contains(const Vec& p, double tol) const {
    for (const Halfspace& f : facets)
        if (dot(p, f.normal) > f.offset + tol) return false;
    return true;
}

VoronoiCell build_voronoi_cell(const Lattice& lat, int window) {
    (void)window;  // устарел: релевантные векторы ищутся сертифицированным CVP-перебором

    const int n = lat.dim();
    const std::size_t un = static_cast<std::size_t>(n);

    // шаг 0: масштабная нормировка (см. normalization_scale)
    const double s = normalization_scale(lat);

    // шаг 1: LLL-приведение — узкие границы CVP-перебора классов смежности
    const Mat reduced = lll_reduce(rescaled_basis(lat, s));

    VoronoiCell cell;
    cell.dim = n;

    // шаг 2: кандидаты в фасеты — биссекторы релевантных векторов
    const std::vector<Halfspace> cand = bisector_facets(reduced);

    // шаг 3: вершины — пересечения всех сочетаний n фасет, прошедшие
    // все неравенства (с допуском) и дедупликацию
    std::vector<Vec> verts;
    const std::size_t m = cand.size();
    if (m >= un) {
        std::vector<std::size_t> idx(un);
        std::iota(idx.begin(), idx.end(), std::size_t{0});

        // следующее сочетание n индексов из m (лексикографически)
        const auto next_combination = [&idx, m, un]() {
            std::size_t i = un;
            while (i > 0) {
                --i;
                if (idx[i] != m - un + i) {
                    ++idx[i];
                    for (std::size_t j = i + 1; j < un; ++j) idx[j] = idx[j - 1] + 1;
                    return true;
                }
            }
            return false;
        };

        do {
            Mat a;
            Vec b;
            a.reserve(un);
            b.reserve(un);
            for (const std::size_t fi : idx) {
                a.push_back(cand[fi].normal);
                b.push_back(cand[fi].offset);
            }
            Vec x;
            if (!solve_linear(std::move(a), std::move(b), x)) continue;

            // точка обязана удовлетворять ВСЕМ неравенствам
            const bool inside = std::none_of(cand.begin(), cand.end(), [&x](const Halfspace& f) {
                return dot(x, f.normal) > f.offset + kGeomEps;
            });
            if (!inside) continue;

            // дедупликация по расстоянию
            const bool dup = std::any_of(verts.begin(), verts.end(), [&x](const Vec& w) {
                return dist2(w, x) < kGeomEps * kGeomEps;
            });
            if (!dup) verts.push_back(std::move(x));
        } while (next_combination());
    }

    // шаг 4: отсев кандидатов, не несущих (n-1)-мерной грани.
    // Просто «есть инцидентная вершина» недостаточно: биссектор нерелевантного
    // вектора может касаться ячейки в грани меньшей размерности (пример Z²:
    // x+y <= 1 проходит через угол (0.5, 0.5) квадрата). Настоящая фасета —
    // та, чьи инцидентные вершины аффинно порождают всю гиперплоскость
    // (аффинный ранг n-1).
    std::vector<Halfspace> kept;
    for (const Halfspace& f : cand) {
        std::vector<const Vec*> inc;  // инцидентные вершины
        for (const Vec& w : verts)
            if (std::abs(dot(w, f.normal) - f.offset) < kGeomEps) inc.push_back(&w);
        if (inc.size() < un) continue;  // меньше n точек не порождают гиперплоскость

        Mat diffs;  // разности с первой инцидентной вершиной
        diffs.reserve(inc.size() - 1);
        for (std::size_t i = 1; i < inc.size(); ++i) diffs.push_back(sub(*inc[i], *inc[0]));
        if (matrix_rank(std::move(diffs), kGeomEps) == n - 1) kept.push_back(f);
    }
    cell.facets = std::move(kept);
    cell.vertices = std::move(verts);

    // инцидентность вершина-фасета (индексы возрастают — уже отсортированы)
    cell.vertex_facets.assign(cell.vertices.size(), {});
    for (std::size_t vi = 0; vi < cell.vertices.size(); ++vi)
        for (std::size_t fi = 0; fi < cell.facets.size(); ++fi)
            if (std::abs(dot(cell.vertices[vi], cell.facets[fi].normal) -
                         cell.facets[fi].offset) < kGeomEps)
                cell.vertex_facets[vi].push_back(static_cast<int>(fi));

    // шаг 5: диаметр = 2 * max|вершина| (ячейка центрально-симметрична)
    double max_n2 = 0.0;
    for (const Vec& w : cell.vertices) max_n2 = std::max(max_n2, norm2(w));
    cell.diameter = 2.0 * std::sqrt(max_n2);

    // шаг 6: f-вектор комбинаторно по инцидентности вершина-фасета.
    // Грань ↔ её канонический набор фасет G = пересечение vf(v) по вершинам
    // грани; семейство всех канонических наборов — замыкание порождающих
    // наборов vf(v) относительно пересечения (итерируем до неподвижности).
    // Пустое пересечение — весь многогранник, не учитывается. Размерность
    // грани = n - rank(нормали её фасет): аффинная оболочка грани есть
    // пересечение гиперплоскостей всех содержащих её фасет.
    std::set<std::vector<int>> faces(cell.vertex_facets.begin(), cell.vertex_facets.end());
    std::vector<std::vector<int>> queue(faces.begin(), faces.end());
    while (!queue.empty()) {
        const std::vector<int> g = std::move(queue.back());
        queue.pop_back();
        for (const std::vector<int>& vf : cell.vertex_facets) {
            std::vector<int> h;
            std::set_intersection(g.begin(), g.end(), vf.begin(), vf.end(),
                                  std::back_inserter(h));
            if (h.empty()) continue;
            if (faces.insert(h).second) queue.push_back(std::move(h));
        }
    }

    cell.f_vector.assign(un, 0);
    for (const std::vector<int>& g : faces) {
        Mat normals;
        normals.reserve(g.size());
        for (const int fi : g) normals.push_back(cell.facets[static_cast<std::size_t>(fi)].normal);
        const int k = n - matrix_rank(std::move(normals), kGeomEps);
        if (k >= 0 && k < n) ++cell.f_vector[static_cast<std::size_t>(k)];
    }

    // шаг 7: self-check построенной ячейки (страховка от численных сбоев;
    // выполняется в нормированном масштабе, где допуски осмысленны).
    // (a) центральная симметрия множества вершин: для каждой w есть -w;
    if (cell.vertices.empty())
        throw std::runtime_error("build_voronoi_cell: не найдено ни одной вершины");
    for (const Vec& w : cell.vertices) {
        const Vec neg = scaled(w, -1.0);
        const bool has_antipode =
            std::any_of(cell.vertices.begin(), cell.vertices.end(), [&neg](const Vec& u) {
                return dist2(u, neg) < kGeomEps * kGeomEps;
            });
        if (!has_antipode)
            throw std::runtime_error(
                "build_voronoi_cell: множество вершин не центрально-симметрично "
                "(численный сбой построения)");
    }
    // (b) соотношение Эйлера: f_0 - f_1 + ... + (-1)^{n-1} f_{n-1} = 1 - (-1)^n
    long euler = 0;
    for (std::size_t i = 0; i < un; ++i) euler += (i % 2 == 0 ? 1 : -1) * cell.f_vector[i];
    const long euler_expected = 1 - (n % 2 == 0 ? 1 : -1);
    if (euler != euler_expected)
        throw std::runtime_error("build_voronoi_cell: нарушено соотношение Эйлера (f-вектор "
                                 "построен неверно — численный сбой)");

    // шаг 8: возврат в исходный масштаб
    if (s != 1.0) {
        for (Vec& w : cell.vertices)
            for (double& x : w) x *= s;
        rescale_facets(cell.facets, s);
        cell.diameter *= s;
    }

    return cell;
}

std::vector<Halfspace> relevant_facets(const Lattice& lat) {
    // те же шаги 0-2, что и в build_voronoi_cell, но без перечисления вершин
    const double s = normalization_scale(lat);
    std::vector<Halfspace> facets = bisector_facets(lll_reduce(rescaled_basis(lat, s)));
    if (s != 1.0) rescale_facets(facets, s);
    return facets;
}

}  // namespace combigeo
