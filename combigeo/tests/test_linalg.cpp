// Тесты линейной алгебры.
#include "combigeo/linalg.hpp"

#include "test_framework.hpp"

using namespace combigeo;

TEST(dot_norm) {
    Vec a{1, 2, 3, 4}, b{4, 3, 2, 1};
    CHECK_NEAR(dot(a, b), 20.0, 1e-12);
    CHECK_NEAR(norm2(a), 30.0, 1e-12);
    CHECK_NEAR(norm(Vec{3, 4}), 5.0, 1e-12);
}

TEST(add_sub_scale) {
    Vec a{1, 2}, b{3, 5};
    CHECK_NEAR(add(a, b)[1], 7.0, 1e-12);
    CHECK_NEAR(sub(b, a)[0], 2.0, 1e-12);
    CHECK_NEAR(scaled(a, 2.5)[1], 5.0, 1e-12);
    Vec y{1, 1};
    axpy_inplace(y, 2.0, a);
    CHECK_NEAR(y[0], 3.0, 1e-12);
    CHECK_NEAR(dist2(a, b), 13.0, 1e-12);
}

TEST(determinant) {
    CHECK_NEAR(det({{1, 0}, {0, 1}}), 1.0, 1e-12);
    CHECK_NEAR(det({{2, 0, 0, 0}, {1, 1, 0, 0}, {1, 0, 1, 0}, {1, 0, 0, 1}}), 2.0, 1e-9);
    CHECK_NEAR(det({{1, 2}, {2, 4}}), 0.0, 1e-12);  // вырожденная
    // антисимметрия перестановки строк
    CHECK_NEAR(det({{0, 1}, {1, 0}}), -1.0, 1e-12);
}

TEST(solve) {
    Vec x;
    CHECK(solve_linear({{2, 0}, {0, 4}}, {6, 8}, x));
    CHECK_NEAR(x[0], 3.0, 1e-12);
    CHECK_NEAR(x[1], 2.0, 1e-12);

    // вырожденная система
    CHECK(!solve_linear({{1, 2}, {2, 4}}, {1, 2}, x));

    // система с выбором ведущего элемента (нулевой угловой)
    CHECK(solve_linear({{0, 1}, {1, 0}}, {5, 7}, x));
    CHECK_NEAR(x[0], 7.0, 1e-12);
    CHECK_NEAR(x[1], 5.0, 1e-12);
}

TEST(combination_long) {
    Mat basis{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    Vec v = combination(std::vector<long>{2, -1, 3}, basis);
    CHECK_NEAR(v[0], 2.0, 1e-12);
    CHECK_NEAR(v[1], -1.0, 1e-12);
    CHECK_NEAR(v[2], 3.0, 1e-12);
}

TEST(gram_schmidt_orthogonal) {
    Mat basis{{1, 1, 0}, {1, 0, 1}, {0, 1, 1}};
    Mat b_star, mu;
    gram_schmidt(basis, b_star, mu);

    // попарная ортогональность
    CHECK_NEAR(dot(b_star[0], b_star[1]), 0.0, 1e-9);
    CHECK_NEAR(dot(b_star[0], b_star[2]), 0.0, 1e-9);
    CHECK_NEAR(dot(b_star[1], b_star[2]), 0.0, 1e-9);

    // диагональ mu = |b*_i|^2
    CHECK_NEAR(mu[0][0], norm2(b_star[0]), 1e-9);
    CHECK_NEAR(mu[2][2], norm2(b_star[2]), 1e-9);
}

int main() { return run_all_tests(); }
