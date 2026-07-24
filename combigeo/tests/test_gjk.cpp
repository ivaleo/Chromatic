// Тесты GJK: расстояние от точки до выпуклого многогранника по вершинам.
// Эталоны — гиперкуб [-0.5,0.5]^n с точной аналитикой:
//   dist(p) = sqrt(sum_i max(|p_i| - 0.5, 0)^2).
#include "combigeo/gjk.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "test_framework.hpp"

using namespace combigeo;

namespace {

// Вершины гиперкуба [-half, half]^n (2^n штук).
std::vector<Vec> hypercube(int n, double half) {
    std::vector<Vec> vs;
    vs.reserve(std::size_t{1} << n);
    for (int mask = 0; mask < (1 << n); ++mask) {
        Vec v(n);
        for (int i = 0; i < n; ++i) v[i] = ((mask >> i) & 1) ? half : -half;
        vs.push_back(std::move(v));
    }
    return vs;
}

// Аналитическое расстояние до гиперкуба [-half, half]^n.
double cube_distance(const Vec& p, double half) {
    double s = 0.0;
    for (double c : p) {
        const double d = std::fabs(c) - half;
        if (d > 0.0) s += d * d;
    }
    return std::sqrt(s);
}

// Проверка согласованности результата: |p - closest| == distance.
void check_consistent(const Vec& p, const GjkResult& r) {
    CHECK_NEAR(norm(sub(p, r.closest)), r.distance, 1e-9);
}

// Простой детерминированный LCG (без <random>), равномерно в [0, 1).
struct Lcg {
    std::uint64_t state;
    explicit Lcg(std::uint64_t seed) : state(seed) {}
    double next_unit() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(state >> 11) / 9007199254740992.0;  // / 2^53
    }
    double next(double lo, double hi) { return lo + (hi - lo) * next_unit(); }
};

}  // namespace

TEST(cube3_outside_face) {
    const auto cube = hypercube(3, 0.5);
    const Vec p{1, 0, 0};
    const GjkResult r = distance_to_polytope(p, cube);
    CHECK_NEAR(r.distance, 0.5, 1e-9);
    check_consistent(p, r);
    // ближайшая точка — центр грани x = 0.5
    CHECK_NEAR(r.closest[0], 0.5, 1e-9);
    CHECK_NEAR(r.closest[1], 0.0, 1e-9);
    CHECK_NEAR(r.closest[2], 0.0, 1e-9);
}

TEST(cube3_corner) {
    const auto cube = hypercube(3, 0.5);
    const Vec p{1, 1, 1};
    const GjkResult r = distance_to_polytope(p, cube);
    CHECK_NEAR(r.distance, std::sqrt(0.75), 1e-9);
    check_consistent(p, r);
    // ближайшая точка — вершина (0.5, 0.5, 0.5)
    for (int i = 0; i < 3; ++i) CHECK_NEAR(r.closest[i], 0.5, 1e-9);
}

TEST(cube3_inside) {
    const auto cube = hypercube(3, 0.5);
    const Vec p{0.2, -0.1, 0.3};
    const GjkResult r = distance_to_polytope(p, cube);
    CHECK_NEAR(r.distance, 0.0, 1e-9);
    check_consistent(p, r);
}

TEST(cube3_boundary) {
    const auto cube = hypercube(3, 0.5);
    const Vec p{0.5, 0, 0};
    const GjkResult r = distance_to_polytope(p, cube);
    CHECK_NEAR(r.distance, 0.0, 1e-9);
    check_consistent(p, r);
}

TEST(cube3_far_face) {
    const auto cube = hypercube(3, 0.5);
    const Vec p{1.7, 0.1, 0.1};
    const GjkResult r = distance_to_polytope(p, cube);
    CHECK_NEAR(r.distance, 1.2, 1e-9);
    check_consistent(p, r);
    // ближайшая точка — (0.5, 0.1, 0.1) внутри грани x = 0.5
    CHECK_NEAR(r.closest[0], 0.5, 1e-9);
    CHECK_NEAR(r.closest[1], 0.1, 1e-9);
    CHECK_NEAR(r.closest[2], 0.1, 1e-9);
}

TEST(tesseract_known_points) {
    const auto tess = hypercube(4, 0.5);

    const Vec p1{1, 1, 1, 1};
    const GjkResult r1 = distance_to_polytope(p1, tess);
    CHECK_NEAR(r1.distance, 1.0, 1e-9);
    check_consistent(p1, r1);

    const Vec p2{2, 0, 0, 0};
    const GjkResult r2 = distance_to_polytope(p2, tess);
    CHECK_NEAR(r2.distance, 1.5, 1e-9);
    check_consistent(p2, r2);

    const Vec p3{0.6, 0.6, 0, 0};
    const GjkResult r3 = distance_to_polytope(p3, tess);
    CHECK_NEAR(r3.distance, std::sqrt(0.02), 1e-9);
    check_consistent(p3, r3);
}

TEST(tesseract_random_vs_analytic) {
    const auto tess = hypercube(4, 0.5);
    Lcg rng(20260607ULL);
    for (int t = 0; t < 200; ++t) {
        Vec p(4);
        for (int i = 0; i < 4; ++i) p[i] = rng.next(-2.0, 2.0);
        const GjkResult r = distance_to_polytope(p, tess);
        CHECK_NEAR(r.distance, cube_distance(p, 0.5), 1e-9);
        check_consistent(p, r);
        CHECK(r.iterations <= 200);
    }
}

TEST(segment_2d) {
    const std::vector<Vec> seg{{0, 0}, {1, 0}};

    const Vec p1{2, 1};
    const GjkResult r1 = distance_to_polytope(p1, seg);
    CHECK_NEAR(r1.distance, std::sqrt(2.0), 1e-9);
    check_consistent(p1, r1);
    // ближайшая точка — конец отрезка (1, 0)
    CHECK_NEAR(r1.closest[0], 1.0, 1e-9);
    CHECK_NEAR(r1.closest[1], 0.0, 1e-9);

    const Vec p2{0.5, 1};
    const GjkResult r2 = distance_to_polytope(p2, seg);
    CHECK_NEAR(r2.distance, 1.0, 1e-9);
    check_consistent(p2, r2);
    // ближайшая точка — середина отрезка (0.5, 0)
    CHECK_NEAR(r2.closest[0], 0.5, 1e-9);
    CHECK_NEAR(r2.closest[1], 0.0, 1e-9);
}

TEST(triangle_4d) {
    // Треугольник на e1, e2, e3 в R^4; p = e4.
    // Ближайшая точка — центр (1/3, 1/3, 1/3, 0), dist^2 = 3/9 + 1 = 4/3.
    const std::vector<Vec> tri{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
    const Vec p{0, 0, 0, 1};
    const GjkResult r = distance_to_polytope(p, tri);
    CHECK_NEAR(r.distance, std::sqrt(4.0 / 3.0), 1e-9);
    check_consistent(p, r);
    for (int i = 0; i < 3; ++i) CHECK_NEAR(r.closest[i], 1.0 / 3.0, 1e-9);
    CHECK_NEAR(r.closest[3], 0.0, 1e-9);
}

TEST(closest_point_basic) {
    // проекция начала координат на отрезок {(1,-1), (1,1)} — точка (1, 0)
    const Vec w = closest_point_on_simplex(Vec{0, 0}, {{1, -1}, {1, 1}});
    CHECK_NEAR(w[0], 1.0, 1e-9);
    CHECK_NEAR(w[1], 0.0, 1e-9);

    // одна точка: проекция совпадает с ней
    const Vec w1 = closest_point_on_simplex(Vec{3, 4}, {{1, 1}});
    CHECK_NEAR(w1[0], 1.0, 1e-12);
    CHECK_NEAR(w1[1], 1.0, 1e-12);
}

TEST(closest_point_degenerate_simplex) {
    // три коллинеарные точки: вырожденные подмножества отбрасываются,
    // ответ берётся с невырожденных подотрезков
    const Vec w = closest_point_on_simplex(Vec{3, 1}, {{0, 0}, {1, 0}, {2, 0}});
    CHECK_NEAR(w[0], 2.0, 1e-9);
    CHECK_NEAR(w[1], 0.0, 1e-9);

    // точка внутри проекционного диапазона
    const Vec w2 = closest_point_on_simplex(Vec{0.7, -2}, {{0, 0}, {1, 0}, {2, 0}});
    CHECK_NEAR(w2[0], 0.7, 1e-9);
    CHECK_NEAR(w2[1], 0.0, 1e-9);
}

TEST(closest_point_inside_triangle) {
    // p над центром треугольника: проекция — сам центр
    const std::vector<Vec> tri{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    const Vec p{0.25, 0.25, 5.0};
    const Vec w = closest_point_on_simplex(p, tri);
    CHECK_NEAR(w[0], 0.25, 1e-9);
    CHECK_NEAR(w[1], 0.25, 1e-9);
    CHECK_NEAR(w[2], 0.0, 1e-9);
}

int main() { return run_all_tests(); }
