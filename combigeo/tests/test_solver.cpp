// Интеграционные тесты решателя: эталоны с аналитически известными расстояниями.
#include "combigeo/solver.hpp"

#include <cmath>

#include "test_framework.hpp"

using namespace combigeo;

namespace {
const Mat kZ2{{1, 0}, {0, 1}};
const Mat kZ3{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
const Mat kD4{{2, 0, 0, 0}, {1, 1, 0, 0}, {1, 0, 1, 0}, {1, 0, 0, 1}};
}  // namespace

TEST(min_color_distance_z3_doubled) {
    // подрешётка 2*Z^3: одноцветные единичные кубы в 2Z^3; ближайшая пара
    // граней x=0.5 и x=1.5 -> D = 1
    const Lattice z3(kZ3);
    const VoronoiCell cell = build_voronoi_cell(z3);
    const Lattice sub(Mat{{2, 0, 0}, {0, 2, 0}, {0, 0, 2}});
    CHECK_NEAR(min_color_distance(cell, sub), 1.0, 1e-9);
}

TEST(min_color_distance_z2_rotated5) {
    // подрешётка (2,1),(-1,2) индекса 5 в Z^2: одноцветные единичные квадраты
    // с центрами на расстоянии sqrt(5); ближайшие точки (0.5,y) и (1.5,y) -> D = 1
    const Lattice z2(kZ2);
    const VoronoiCell cell = build_voronoi_cell(z2);
    const Lattice sub(Mat{{2, 1}, {-1, 2}});
    CHECK_NEAR(min_color_distance(cell, sub), 1.0, 1e-9);
}

TEST(min_color_distance_d4_doubled) {
    // 2*D4: ячейка — 24-ячейка диаметра 2; для v = 2*(1,1,0,0) точка v/2=(1,1,0,0)
    // проецируется в (0.5,0.5,0,0) на фасете-биссекторе -> dist = sqrt(0.5), D = sqrt(2)
    const Lattice d4(kD4);
    const VoronoiCell cell = build_voronoi_cell(d4);
    const Lattice sub(Mat{{4, 0, 0, 0}, {2, 2, 0, 0}, {2, 0, 2, 0}, {2, 0, 0, 2}});
    CHECK_NEAR(min_color_distance(cell, sub), std::sqrt(2.0), 1e-9);
}

TEST(find_optimal_z2_index2_touching) {
    // обе подрешётки индекса 2 в Z^2 дают касающиеся ячейки: best D = 0
    SolveResult r = find_optimal(kZ2, 2);
    CHECK(r.examined == SublatticeIterator::count(2, 2));
    CHECK_NEAR(r.best.min_distance, 0.0, 1e-6);
    CHECK_NEAR(r.diameter, std::sqrt(2.0), 1e-9);
}

TEST(find_optimal_z3_index8) {
    // кандидат diag(2,2,2) даёт D=1 -> оптимум не меньше 1
    SolveResult r = find_optimal(kZ3, 8);
    CHECK(r.examined == SublatticeIterator::count(3, 8));
    CHECK(r.best.min_distance >= 1.0 - 1e-9);
    CHECK_NEAR(r.diameter, std::sqrt(3.0), 1e-9);
    // нормировка согласована
    CHECK_NEAR(r.normalized, r.best.min_distance / r.diameter, 1e-12);
    // матрица перехода верхнетреугольная с det = 8
    Mat t;
    for (const auto& row : r.best.transition) {
        Vec v(row.begin(), row.end());
        t.push_back(v);
    }
    CHECK_NEAR(std::abs(det(t)), 8.0, 1e-9);
}

TEST(cache_consistency) {
    // с кэшем и без — одинаковый результат
    SolveOptions with, without;
    without.use_cache = false;
    SolveResult a = find_optimal(kZ3, 4, with);
    SolveResult b = find_optimal(kZ3, 4, without);
    CHECK_NEAR(a.best.min_distance, b.best.min_distance, 1e-12);
    CHECK(a.examined == b.examined);
}

TEST(witness_vector_realizes_minimum) {
    SolveResult r = find_optimal(kZ3, 8);
    // свидетель — ненулевой вектор лучшей подрешётки
    CHECK(!r.best.witness.empty());
    CHECK(norm(r.best.witness) > 1e-9);
}

TEST(find_optimal_d4_index49_valid_coloring) {
    // D4, index = 49 (49 цветов): существует подрешётка с d = D/diam >= 1 —
    // раскраска пригодна. Эталонное значение d ≈ 1.080123.
    // Перебор ~140050 подрешёток (порядка полсекунды) — единственный «тяжёлый»
    // тест, проверяющий именно случай успешной раскраски.
    SolveResult r = find_optimal(kD4, 49);
    CHECK(r.examined == SublatticeIterator::count(4, 49));
    CHECK(r.normalized >= 1.0);                     // раскраска пригодна
    CHECK_NEAR(r.normalized, 1.080123, 1e-4);
    CHECK_NEAR(r.diameter, 2.0, 1e-9);             // 24-ячейка D4
    // матрица перехода верхнетреугольная с det = 49
    Mat t;
    for (const auto& row : r.best.transition) {
        Vec v(row.begin(), row.end());
        t.push_back(v);
    }
    CHECK_NEAR(std::abs(det(t)), 49.0, 1e-9);
}

namespace {

// подрешётки L(transition * basis) и L(sub_basis) совпадают: взаимные
// целочисленные координаты (через transition_in_basis) и равные определители
bool same_sublattice(const HnfMatrix& transition, const Mat& basis, const Mat& sub_basis) {
    const Mat from_transition = apply_hnf(transition, basis);
    const HnfMatrix h1 = transition_in_basis(sub_basis, from_transition);      // бросит, если не подрешётка
    const HnfMatrix h2 = transition_in_basis(from_transition, sub_basis);
    for (std::size_t i = 0; i < h1.size(); ++i)
        if (h1[i][i] != 1 || h2[i][i] != 1) return false;  // индекс 1 в обе стороны = равенство
    return true;
}

}  // namespace

TEST(transition_in_user_basis_skew) {
    // критический случай из аудита: для скошенного (не LLL-приведённого) базиса
    // transition обязан быть в базисе ПОЛЬЗОВАТЕЛЯ, а не внутреннем LLL-кадре
    const Mat skew{{3, 1}, {2, 1}};  // унимодулярная перестройка Z^2
    SolveResult r = find_optimal(skew, 5);
    CHECK(!r.best.transition.empty());
    CHECK(same_sublattice(r.best.transition, skew, r.best.sub_basis));

    const Mat skew13{{13, 5}, {8, 3}};
    SolveResult r13 = find_optimal(skew13, 13);
    CHECK(same_sublattice(r13.best.transition, skew13, r13.best.sub_basis));

    // и для D4 в «неудобном» базисе пресета
    SolveResult rd4 = find_optimal(kD4, 16);
    CHECK(same_sublattice(rd4.best.transition, kD4, rd4.best.sub_basis));
}

TEST(hnf_of_normalization) {
    // hnf_of: верхнетреугольная, диагональ > 0, наддиагональ в [0, d_j)
    const HnfMatrix h = hnf_of({{2, 4}, {3, 1}});
    CHECK(h.size() == 2);
    CHECK(h[1][0] == 0);
    CHECK(h[0][0] > 0 && h[1][1] > 0);
    CHECK(h[0][0] * h[1][1] == 10);  // |det| сохраняется
    CHECK(h[0][1] >= 0 && h[0][1] < h[1][1]);
}

int main() { return run_all_tests(); }
