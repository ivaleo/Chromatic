// Тесты build_voronoi_cell: эталонные ячейки Вороного классических решёток
// (квадрат, куб, ромбододекаэдр FCC, усечённый октаэдр BCC, тессеракт,
// 24-ячейка D4) — f-векторы, диаметры, принадлежность, симметрия.

#include "combigeo/polytope.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "combigeo/lattice.hpp"
#include "combigeo/linalg.hpp"
#include "test_framework.hpp"

using combigeo::build_voronoi_cell;
using combigeo::Lattice;
using combigeo::Vec;
using combigeo::VoronoiCell;

namespace {

// Структурная проверка ячейки: f-вектор, согласованность списков, единичные
// нормали, принадлежность вершин, центральная симметрия множества вершин.
void check_cell(const VoronoiCell& cell, const std::vector<long>& f_expected,
                double diam_expected, double tol = 1e-9) {
    CHECK(cell.f_vector == f_expected);
    CHECK(static_cast<long>(cell.vertices.size()) == f_expected.front());
    CHECK(static_cast<long>(cell.facets.size()) == f_expected.back());
    CHECK_NEAR(cell.diameter, diam_expected, tol);

    // вершины лежат в ячейке и инцидентны хотя бы n фасетам (списки отсортированы)
    CHECK(cell.vertex_facets.size() == cell.vertices.size());
    for (std::size_t i = 0; i < cell.vertices.size(); ++i) {
        CHECK(cell.contains(cell.vertices[i], 1e-7));
        const std::vector<int>& vf = cell.vertex_facets[i];
        CHECK(vf.size() >= static_cast<std::size_t>(cell.dim));
        CHECK(std::is_sorted(vf.begin(), vf.end()));
    }

    // фасеты: единичная нормаль, положительный offset, |lattice_vector| = 2*offset
    // (последнее — с относительным допуском: offset масштабируется с решёткой)
    for (const auto& f : cell.facets) {
        CHECK_NEAR(combigeo::norm(f.normal), 1.0, 1e-9);
        CHECK(f.offset > 0.0);
        CHECK_NEAR(combigeo::norm(f.lattice_vector), 2.0 * f.offset,
                   1e-9 * std::max(1.0, 2.0 * f.offset));
    }

    // центральная симметрия: для каждой вершины v есть вершина -v
    for (const Vec& v : cell.vertices) {
        bool found = false;
        for (const Vec& w : cell.vertices)
            if (combigeo::dist2(combigeo::scaled(v, -1.0), w) < 1e-12) {
                found = true;
                break;
            }
        CHECK(found);
    }
}

}  // namespace

TEST(z1_segment) {
    // решётка 2Z: ячейка — отрезок [-1, 1]
    const VoronoiCell cell = build_voronoi_cell(Lattice({{2.0}}));
    CHECK(cell.dim == 1);
    check_cell(cell, {2}, 2.0);
}

TEST(z2_square) {
    const VoronoiCell cell = build_voronoi_cell(Lattice({{1.0, 0.0}, {0.0, 1.0}}));
    CHECK(cell.dim == 2);
    check_cell(cell, {4, 4}, std::sqrt(2.0));
    CHECK(cell.contains({0.0, 0.0}));
    CHECK(!cell.contains({1.0, 1.0}));
    CHECK(cell.contains({0.5, 0.0}, 1e-9));      // середина фасеты — на границе
    CHECK(!cell.contains({0.5 + 1e-6, 0.0}));    // чуть снаружи
}

TEST(z2_skew_basis_same_lattice) {
    // косой базис той же решётки Z^2 — LLL приводит к единичному, ячейка та же
    const VoronoiCell cell = build_voronoi_cell(Lattice({{1.0, 0.0}, {3.0, 1.0}}));
    check_cell(cell, {4, 4}, std::sqrt(2.0));
}

TEST(z2_scaled) {
    const VoronoiCell cell = build_voronoi_cell(Lattice({{2.0, 0.0}, {0.0, 2.0}}));
    check_cell(cell, {4, 4}, 2.0 * std::sqrt(2.0));
    CHECK(cell.contains({0.9, 0.9}));
    CHECK(!cell.contains({1.1, 0.0}));
}

TEST(hexagonal_cell) {
    // гексагональная решётка: ячейка — правильный шестиугольник,
    // вписанный радиус 1/2, описанный 1/sqrt(3)
    const VoronoiCell cell =
        build_voronoi_cell(Lattice({{1.0, 0.0}, {0.5, std::sqrt(3.0) / 2.0}}));
    check_cell(cell, {6, 6}, 2.0 / std::sqrt(3.0), 1e-7);
}

TEST(z3_cube) {
    const VoronoiCell cell =
        build_voronoi_cell(Lattice({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}));
    check_cell(cell, {8, 12, 6}, std::sqrt(3.0));
    // куб — простой многогранник: каждая вершина инцидентна ровно трём фасетам
    for (const auto& vf : cell.vertex_facets) CHECK(vf.size() == 3);
}

TEST(fcc_rhombic_dodecahedron) {
    const VoronoiCell cell =
        build_voronoi_cell(Lattice({{1.0, 1.0, 0.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}}));
    check_cell(cell, {14, 24, 12}, 2.0);
}

TEST(fcc_window_2_enough) {
    // по контракту polytope.hpp после LLL достаточно окна 2
    const VoronoiCell cell =
        build_voronoi_cell(Lattice({{1.0, 1.0, 0.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}}), 2);
    check_cell(cell, {14, 24, 12}, 2.0);
}

TEST(bcc_truncated_octahedron) {
    const VoronoiCell cell = build_voronoi_cell(
        Lattice({{1.0, -1.0, -1.0}, {-1.0, 1.0, -1.0}, {-1.0, -1.0, 1.0}}));
    check_cell(cell, {24, 36, 14}, std::sqrt(5.0));
}

TEST(z4_tesseract) {
    const VoronoiCell cell = build_voronoi_cell(Lattice({{1.0, 0.0, 0.0, 0.0},
                                                         {0.0, 1.0, 0.0, 0.0},
                                                         {0.0, 0.0, 1.0, 0.0},
                                                         {0.0, 0.0, 0.0, 1.0}}));
    check_cell(cell, {16, 32, 24, 8}, 2.0);
}

TEST(d4_24_cell) {
    const VoronoiCell cell = build_voronoi_cell(Lattice({{2.0, 0.0, 0.0, 0.0},
                                                         {1.0, 1.0, 0.0, 0.0},
                                                         {1.0, 0.0, 1.0, 0.0},
                                                         {1.0, 0.0, 0.0, 1.0}}));
    check_cell(cell, {24, 96, 96, 24}, 2.0);
}

TEST(d4_vertex_symmetry) {
    // для каждой вершины v ячейки D4 вершина -v тоже в списке
    const VoronoiCell cell = build_voronoi_cell(Lattice({{2.0, 0.0, 0.0, 0.0},
                                                         {1.0, 1.0, 0.0, 0.0},
                                                         {1.0, 0.0, 1.0, 0.0},
                                                         {1.0, 0.0, 0.0, 1.0}}));
    CHECK(cell.vertices.size() == 24);
    for (const Vec& v : cell.vertices) {
        bool found = false;
        for (const Vec& w : cell.vertices)
            if (combigeo::dist2(combigeo::scaled(v, -1.0), w) < 1e-12) {
                found = true;
                break;
            }
        CHECK(found);
    }
}

TEST(extreme_scale_hexagonal) {
    // абсолютные допуски раньше давали молча неверную ячейку при масштабе 1e10
    // и ложную «вырожденность» при 1e-6; теперь масштаб нормируется внутри
    const double s = 1e8;
    const VoronoiCell big =
        build_voronoi_cell(Lattice({{2.0 * s, 0.0}, {1.0 * s, 1.7320508075688772 * s}}));
    check_cell(big, {6, 6}, s * 2.0 * 2.0 / 1.7320508075688772, s * 1e-9);

    const double t = 1e-6;
    const VoronoiCell small =
        build_voronoi_cell(Lattice({{2.0 * t, 0.0}, {1.0 * t, 1.7320508075688772 * t}}));
    check_cell(small, {6, 6}, t * 2.0 * 2.0 / 1.7320508075688772, t * 1e-9);
}

TEST(z5_smoke) {
    // n = 5: гиперкуб — сертифицированный перебор релевантных векторов обязан
    // отбросить все классы с неединственным минимумом (их биссекторы лишние)
    const VoronoiCell cell = build_voronoi_cell(Lattice({{1, 0, 0, 0, 0},
                                                         {0, 1, 0, 0, 0},
                                                         {0, 0, 1, 0, 0},
                                                         {0, 0, 0, 1, 0},
                                                         {0, 0, 0, 0, 1}}));
    check_cell(cell, {32, 80, 80, 40, 10}, std::sqrt(5.0));
}

int main() { return run_all_tests(); }
