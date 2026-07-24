// Тесты решётки: определитель, перебор коротких векторов, кратчайший вектор.
#include "combigeo/lattice.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "test_framework.hpp"

using namespace combigeo;

TEST(det_and_validation) {
    Lattice z3(Mat{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    CHECK_NEAR(z3.det(), 1.0, 1e-12);

    Lattice d4(Mat{{2, 0, 0, 0}, {1, 1, 0, 0}, {1, 0, 1, 0}, {1, 0, 0, 1}});
    CHECK_NEAR(d4.det(), 2.0, 1e-9);

    bool threw = false;
    try {
        Lattice bad(Mat{{1, 2}, {2, 4}});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(rejects_nan_inf_basis) {
    // NaN/Inf в базисе: det даёт NaN, а NaN < kEps == false — без явной
    // проверки конечности такой базис прошёл бы (регрессия)
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (double bad : {nan, inf}) {
        bool threw = false;
        try {
            Lattice l(Mat{{1, 0}, {bad, 1}});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw);
    }
}

TEST(rejects_nonsquare_basis) {
    bool threw = false;
    try {
        Lattice l(Mat{{1, 0, 0}, {0, 1, 0}});  // 2 строки по 3
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(vectors_within_z2) {
    // Z^2, |v| <= 1: канонические представители пар +-v: (1,0), (0,1) — ровно 2
    Lattice z2(Mat{{1, 0}, {0, 1}});
    auto vs = z2.vectors_within(1.0);
    CHECK(vs.size() == 2);
    for (const auto& v : vs) CHECK_NEAR(norm(v), 1.0, 1e-9);
}

TEST(vectors_within_z2_radius2) {
    // |v| <= 2: пары (1,0),(0,1),(1,1),(1,-1),(2,0),(0,2) — 6 канонических
    Lattice z2(Mat{{1, 0}, {0, 1}});
    auto vs = z2.vectors_within(2.0);
    CHECK(vs.size() == 6);
    for (const auto& v : vs) CHECK(norm(v) <= 2.0 + 1e-9);
}

TEST(vectors_within_skewed_complete) {
    // тот же Z^2, но скошенный базис: набор векторов обязан совпасть
    Lattice skew(Mat{{1, 0}, {1000, 1}});
    auto vs = skew.vectors_within(2.0);
    CHECK(vs.size() == 6);
}

TEST(shortest_vector_d4) {
    // кратчайший вектор D4 имеет длину sqrt(2)
    Lattice d4(Mat{{2, 0, 0, 0}, {1, 1, 0, 0}, {1, 0, 1, 0}, {1, 0, 0, 1}});
    CHECK_NEAR(norm(d4.shortest_vector()), std::sqrt(2.0), 1e-9);
}

TEST(shortest_vector_scaled) {
    Lattice z3x2(Mat{{2, 0, 0}, {0, 2, 0}, {0, 0, 2}});
    CHECK_NEAR(norm(z3x2.shortest_vector()), 2.0, 1e-9);
}

TEST(vectors_within_d4_kissing) {
    // у D4 кратчайших векторов 24 (kissing number), канонических пар — 12
    Lattice d4(Mat{{2, 0, 0, 0}, {1, 1, 0, 0}, {1, 0, 1, 0}, {1, 0, 0, 1}});
    auto vs = d4.vectors_within(std::sqrt(2.0) + 1e-9);
    int shortest = 0;
    for (const auto& v : vs)
        if (std::abs(norm(v) - std::sqrt(2.0)) < 1e-6) ++shortest;
    CHECK(shortest == 12);
}

int main() { return run_all_tests(); }
