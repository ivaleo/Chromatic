// Тесты перебора подрешёток (HNF-матрицы заданного индекса).
#include "combigeo/sublattice.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "combigeo/linalg.hpp"
#include "test_framework.hpp"

using combigeo::HnfMatrix;
using combigeo::Mat;
using combigeo::SublatticeIterator;
using combigeo::Vec;

namespace {

// единичный базис размерности n
Mat identity(int n) {
    Mat e(static_cast<std::size_t>(n), Vec(static_cast<std::size_t>(n), 0.0));
    for (int i = 0; i < n; ++i)
        e[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = 1.0;
    return e;
}

// собрать все матрицы итератора в список
std::vector<HnfMatrix> collect(int dim, long index) {
    SublatticeIterator it(dim, index);
    std::vector<HnfMatrix> out;
    HnfMatrix h;
    while (it.next(h)) out.push_back(h);
    return out;
}

// сплющить матрицу в один вектор (для проверки уникальности)
std::vector<long> flatten(const HnfMatrix& h) {
    std::vector<long> flat;
    for (const auto& row : h) flat.insert(flat.end(), row.begin(), row.end());
    return flat;
}

const std::pair<int, long> kCases[] = {{2, 4}, {3, 2}, {4, 2}, {4, 3}, {4, 6}};

}  // namespace

TEST(ordered_factorizations_basic) {
    auto f = combigeo::ordered_factorizations(4, 2);
    std::sort(f.begin(), f.end());
    const std::vector<std::vector<long>> expected42 = {{1, 4}, {2, 2}, {4, 1}};
    CHECK(f == expected42);

    CHECK(combigeo::ordered_factorizations(12, 1) ==
          (std::vector<std::vector<long>>{{12}}));
    CHECK(combigeo::ordered_factorizations(1, 3) ==
          (std::vector<std::vector<long>>{{1, 1, 1}}));
}

TEST(count_primes_z2) {
    // число подрешёток простого индекса p в Z^2 равно p + 1
    for (long p : {2L, 3L, 5L, 7L}) CHECK(SublatticeIterator::count(2, p) == p + 1);
}

TEST(count_known_values) {
    CHECK(SublatticeIterator::count(2, 4) == 7);
    CHECK(SublatticeIterator::count(3, 2) == 7);
    CHECK(SublatticeIterator::count(4, 2) == 15);
    CHECK(SublatticeIterator::count(4, 3) == 40);
}

TEST(iterator_matches_count) {
    for (auto [dim, index] : kCases) {
        const auto all = collect(dim, index);
        CHECK(static_cast<long>(all.size()) == SublatticeIterator::count(dim, index));
    }
}

TEST(iterator_matrices_valid) {
    for (auto [dim, index] : kCases) {
        const Mat e = identity(dim);
        for (const HnfMatrix& h : collect(dim, index)) {
            CHECK(static_cast<int>(h.size()) == dim);
            for (int i = 0; i < dim; ++i) {
                const auto& row = h[static_cast<std::size_t>(i)];
                CHECK(static_cast<int>(row.size()) == dim);
                // диагональ положительна
                CHECK(row[static_cast<std::size_t>(i)] >= 1);
                for (int j = 0; j < dim; ++j) {
                    const long v = row[static_cast<std::size_t>(j)];
                    // ниже диагонали нули
                    if (j < i) CHECK(v == 0);
                    // наддиагональ столбца j в [0, d_j)
                    if (j > i) {
                        CHECK(v >= 0);
                        CHECK(v < h[static_cast<std::size_t>(j)][static_cast<std::size_t>(j)]);
                    }
                }
            }
            // det(h * E) == index
            const Mat m = combigeo::apply_hnf(h, e);
            CHECK_NEAR(combigeo::det(m), static_cast<double>(index), 1e-9);
        }
    }
}

TEST(iterator_matrices_unique) {
    for (auto [dim, index] : kCases) {
        const auto all = collect(dim, index);
        std::set<std::vector<long>> seen;
        for (const HnfMatrix& h : all) seen.insert(flatten(h));
        CHECK(seen.size() == all.size());
    }
}

TEST(apply_hnf_identity_basis) {
    const HnfMatrix h = {{2, 1}, {0, 3}};
    const Mat m = combigeo::apply_hnf(h, identity(2));
    CHECK(m.size() == 2);
    CHECK_NEAR(m[0][0], 2.0, 1e-12);
    CHECK_NEAR(m[0][1], 1.0, 1e-12);
    CHECK_NEAR(m[1][0], 0.0, 1e-12);
    CHECK_NEAR(m[1][1], 3.0, 1e-12);
}

TEST(apply_hnf_general_basis) {
    // строки результата — линейные комбинации строк базиса:
    // (2,0)*B = 2*(1,1) = (2,2); (0,1)*B = (0,2)
    const HnfMatrix h = {{2, 0}, {0, 1}};
    const Mat basis = {{1.0, 1.0}, {0.0, 2.0}};
    const Mat m = combigeo::apply_hnf(h, basis);
    CHECK_NEAR(m[0][0], 2.0, 1e-12);
    CHECK_NEAR(m[0][1], 2.0, 1e-12);
    CHECK_NEAR(m[1][0], 0.0, 1e-12);
    CHECK_NEAR(m[1][1], 2.0, 1e-12);
}

TEST(count_overflow_throws) {
    // число подрешёток может превысить long — count должен бросить, а не тихо
    // вернуть мусор. Большое простое p даёт всего dim разложений (p в одной
    // позиции, единицы в остальных), но prod d_j^j = p^(dim-1) переполняет long:
    // 10007^5 ≈ 1.0e20 > LONG_MAX ≈ 9.2e18. Перечисление при этом тривиально.
    using combigeo::SublatticeIterator;
    bool threw = false;
    try {
        SublatticeIterator::count(6, 10007);
    } catch (const std::overflow_error&) {
        threw = true;
    }
    CHECK(threw);
}

int main() { return run_all_tests(); }
