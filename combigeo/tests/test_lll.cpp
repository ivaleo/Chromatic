// Тесты LLL-приведения.
#include "combigeo/lll.hpp"

#include <algorithm>
#include <cmath>

#include "test_framework.hpp"

using namespace combigeo;

namespace {
double max_row_norm(const Mat& m) {
    double r = 0;
    for (const auto& row : m) r = std::max(r, norm(row));
    return r;
}
}  // namespace

TEST(preserves_determinant) {
    Mat basis{{2, 0, 0, 0}, {1, 1, 0, 0}, {1, 0, 1, 0}, {1, 0, 0, 1}};
    Mat reduced = lll_reduce(basis);
    CHECK_NEAR(std::abs(det(reduced)), std::abs(det(basis)), 1e-9);
}

TEST(does_not_lengthen) {
    Mat basis{{2, 0, 0, 0}, {1, 1, 0, 0}, {1, 0, 1, 0}, {1, 0, 0, 1}};
    Mat reduced = lll_reduce(basis);
    CHECK(max_row_norm(reduced) <= max_row_norm(basis) + 1e-9);
}

TEST(identity_unchanged) {
    Mat eye{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    Mat reduced = lll_reduce(eye);
    CHECK_NEAR(std::abs(det(reduced)), 1.0, 1e-12);
    CHECK_NEAR(max_row_norm(reduced), 1.0, 1e-12);
}

TEST(reduces_skewed_basis) {
    // сильно скошенный базис Z^2: должен привестись к коротким векторам
    Mat basis{{1, 0}, {1000, 1}};
    Mat reduced = lll_reduce(basis);
    CHECK_NEAR(std::abs(det(reduced)), 1.0, 1e-9);
    CHECK(max_row_norm(reduced) < 2.0);
}

TEST(d4_stays_short) {
    // базис D4 (det=2): после LLL все векторы длины sqrt(2)..2
    Mat basis{{2, 0, 0, 0}, {1, 1, 0, 0}, {1, 0, 1, 0}, {1, 0, 0, 1}};
    Mat reduced = lll_reduce(basis);
    for (const auto& row : reduced) CHECK(norm(row) <= 2.0 + 1e-9);
}

TEST(invalid_delta_does_not_hang) {
    // delta >= 1 нарушает условие завершения Ловаса — реализация приводит
    // delta к безопасному значению и не зависает (регрессия)
    Mat basis{{2, 0, 0, 0}, {1, 1, 0, 0}, {1, 0, 1, 0}, {1, 0, 0, 1}};
    Mat r1 = lll_reduce(basis, 1.5);   // вне диапазона -> 0.75
    Mat r2 = lll_reduce(basis, -3.0);  // вне диапазона -> 0.75
    CHECK_NEAR(std::abs(det(r1)), 2.0, 1e-9);
    CHECK_NEAR(std::abs(det(r2)), 2.0, 1e-9);
}

int main() { return run_all_tests(); }
