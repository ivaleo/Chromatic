#include "combigeo/gjk.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace combigeo {

namespace {

// Результат проекции точки на симплекс: сама точка, квадрат расстояния до p
// и барицентрические координаты по точкам simplex (нулевые у точек,
// не вошедших в оптимальное подмножество).
struct SimplexProjection {
    Vec point;
    double dist2 = std::numeric_limits<double>::infinity();
    std::vector<double> bary;
};

// Проекция p на conv(simplex) перебором всех непустых подмножеств.
// Для каждого подмножества {q_0..q_{k-1}} решаются нормальные уравнения
// проекции p на его аффинную оболочку (система Грама):
//   G a = b,  G_ij = (q_i - q_0)·(q_j - q_0),  b_i = (p - q_0)·(q_i - q_0),
// аффинно зависимые подмножества пропускаются: их выпуклая оболочка покрыта
// невырожденными подмножествами (теорема Каратеодори). Кандидат допустим,
// если все барицентрические координаты >= -kEps (точка внутри подсимплекса);
// среди допустимых берётся ближайший к p. Надёжный перебор вместо знаковых
// объёмов: подмножеств не больше 2^(dim+2), для малых размерностей дёшево.
SimplexProjection project_on_simplex(const Vec& p, const std::vector<Vec>& simplex) {
    const std::size_t m = simplex.size();
    assert(m >= 1);

    SimplexProjection best;
    best.bary.assign(m, 0.0);

    std::vector<std::size_t> idx;   // индексы точек текущего подмножества
    std::vector<Vec> edges;         // q_i - q_0, i = 1..k-1
    // 64-битная маска: размер симплекса в GJK ограничен dim+2 логикой редукции,
    // но сдвиг считаем в 64 битах независимо от assert (вырезается в release)
    for (unsigned long long mask = 1; mask < (1ull << m); ++mask) {
        idx.clear();
        for (std::size_t i = 0; i < m; ++i)
            if (mask & (1ull << i)) idx.push_back(i);
        const std::size_t k = idx.size();
        const Vec& q0 = simplex[idx[0]];

        edges.clear();
        for (std::size_t i = 1; i < k; ++i) edges.push_back(sub(simplex[idx[i]], q0));

        // coeffs = a_1..a_{k-1}; a_0 = 1 - сумма остальных
        Vec coeffs;
        if (k > 1) {
            Mat g(k - 1, Vec(k - 1, 0.0));
            Vec b(k - 1, 0.0);
            const Vec pq = sub(p, q0);
            for (std::size_t i = 0; i + 1 < k; ++i) {
                for (std::size_t j = 0; j + 1 < k; ++j) g[i][j] = dot(edges[i], edges[j]);
                b[i] = dot(pq, edges[i]);
            }
            if (!solve_linear(g, b, coeffs)) continue;  // вырожденное подмножество
        }

        // проверка допустимости барицентрических координат
        double a0 = 1.0;
        bool feasible = true;
        for (double c : coeffs) {
            a0 -= c;
            if (c < -kEps) {
                feasible = false;
                break;
            }
        }
        if (!feasible || a0 < -kEps) continue;

        // точка-кандидат: x = q_0 + sum a_i * (q_i - q_0)
        Vec x = q0;
        for (std::size_t i = 0; i + 1 < k; ++i) axpy_inplace(x, coeffs[i], edges[i]);

        const double d2 = dist2(p, x);
        if (d2 < best.dist2) {
            best.dist2 = d2;
            best.point = std::move(x);
            std::fill(best.bary.begin(), best.bary.end(), 0.0);
            best.bary[idx[0]] = a0;
            for (std::size_t i = 1; i < k; ++i) best.bary[idx[i]] = coeffs[i - 1];
        }
    }
    return best;
}

}  // namespace

Vec closest_point_on_simplex(const Vec& p, const std::vector<Vec>& simplex) {
    return project_on_simplex(p, simplex).point;
}

GjkResult distance_to_polytope(const Vec& p, const std::vector<Vec>& vertices,
                               double tol, int max_iter) {
    // валидация на границе: пустой многогранник и несовпадение размерностей
    // иначе в release-сборке (assert вырезан) дают чтение за границей буфера
    if (vertices.empty())
        throw std::invalid_argument("distance_to_polytope: пустой список вершин");
    const int dim = static_cast<int>(p.size());
    for (const Vec& v : vertices)
        if (static_cast<int>(v.size()) != dim)
            throw std::invalid_argument(
                "distance_to_polytope: размерность точки и вершин не совпадает");

    // старт: ближайшая к p вершина многогранника
    std::size_t start = 0;
    double start_d2 = dist2(p, vertices[0]);
    for (std::size_t i = 1; i < vertices.size(); ++i) {
        const double d2 = dist2(p, vertices[i]);
        if (d2 < start_d2) {
            start_d2 = d2;
            start = i;
        }
    }

    std::vector<std::size_t> active = {start};   // индексы вершин текущего симплекса
    std::vector<Vec> simplex = {vertices[start]};

    GjkResult res;
    res.closest = vertices[start];
    res.distance = std::sqrt(start_d2);
    double prev_dist = std::numeric_limits<double>::max();

    for (int iter = 0; iter < max_iter; ++iter) {
        res.iterations = iter + 1;

        const SimplexProjection proj = project_on_simplex(p, simplex);
        const double dist = std::sqrt(proj.dist2);
        res.closest = proj.point;
        res.distance = dist;

        // |p - w| -> 0: точка внутри (или на границе) многогранника
        if (dist <= tol) {
            res.distance = 0.0;
            res.closest = p;
            return res;
        }

        // улучшение исчерпано — стоп (защита от топтания на месте)
        if (prev_dist - dist <= tol * std::max(1.0, prev_dist)) break;
        prev_dist = dist;

        // редукция симплекса: выбрасываем точки с нулевой барицентрической
        // координатой (критично для сходимости — симплекс не разрастается
        // сверх dim+1 точки и не накапливает «мёртвых» вершин)
        std::vector<std::size_t> kept_active;
        std::vector<Vec> kept_simplex;
        for (std::size_t i = 0; i < simplex.size(); ++i) {
            if (proj.bary[i] > kEps) {
                kept_active.push_back(active[i]);
                kept_simplex.push_back(simplex[i]);
            }
        }
        if (!kept_active.empty()) {
            active = std::move(kept_active);
            simplex = std::move(kept_simplex);
        }

        // вершина-носитель в направлении d = p - w
        const Vec d = sub(p, proj.point);
        std::size_t support = 0;
        double support_dot = -std::numeric_limits<double>::max();
        for (std::size_t i = 0; i < vertices.size(); ++i) {
            const double s = dot(vertices[i], d);
            if (s > support_dot) {
                support_dot = s;
                support = i;
            }
        }

        // носитель не строго дальше w вдоль d — улучшение невозможно, w оптимальна
        if (support_dot <= dot(proj.point, d) + tol * dist) break;

        // защита от зацикливания: не добавляем уже присутствующую вершину
        if (std::find(active.begin(), active.end(), support) != active.end()) break;

        active.push_back(support);
        simplex.push_back(vertices[support]);
    }

    // Сертификат на выходе (любой путь): по двойственности для d = p - w
    //   dist(p, P) >= (<p,d> - max_x <x,d>)/|d| = |d| - (support_dot - <w,d>)/|d|,
    // т.е. ошибка res.distance - dist(p, P) <= (support_dot - <w,d>)/|d|.
    if (res.distance > tol) {
        const Vec d = sub(p, res.closest);
        const double dn = norm(d);
        if (dn > tol) {
            double support_dot = -std::numeric_limits<double>::max();
            for (const Vec& v : vertices) support_dot = std::max(support_dot, dot(v, d));
            res.error = std::max(0.0, (support_dot - dot(res.closest, d)) / dn);
        }
    }
    return res;
}

}  // namespace combigeo
