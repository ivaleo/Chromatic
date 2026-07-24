#include "combigeo/sublattice.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace combigeo {

namespace {

// Рекурсивный перебор упорядоченных разложений: первый множитель — любой
// делитель d | n (по возрастанию), хвост — разложение n/d на k-1 множителей.
// Порядок выдачи детерминированный (лексикографический по множителям).
void factorizations_rec(long n, int k, std::vector<long>& current,
                        std::vector<std::vector<long>>& out) {
    if (k == 1) {
        current.push_back(n);
        out.push_back(current);
        current.pop_back();
        return;
    }
    for (long d = 1; d <= n; ++d) {
        if (n % d != 0) continue;
        current.push_back(d);
        factorizations_rec(n / d, k - 1, current, out);
        current.pop_back();
    }
}

}  // namespace

std::vector<std::vector<long>> ordered_factorizations(long n, int k) {
    std::vector<std::vector<long>> out;
    if (n <= 0 || k <= 0) return out;
    std::vector<long> current;
    current.reserve(static_cast<std::size_t>(k));
    factorizations_rec(n, k, current, out);
    return out;
}

SublatticeIterator::SublatticeIterator(int dim, long index)
    // dim_ и index_ инициализируются раньше diagonals_ (порядок объявления в классе)
    : dim_(dim), index_(index), diagonals_(ordered_factorizations(index_, dim_)) {}

bool SublatticeIterator::next(HnfMatrix& h) {
    if (diag_pos_ >= diagonals_.size()) return false;

    const std::vector<long>& d = diagonals_[diag_pos_];
    const std::size_t n = static_cast<std::size_t>(dim_);
    const std::size_t m = n * (n - 1) / 2;  // число наддиагональных позиций

    if (fresh_diag_) {
        offdiag_.assign(m, 0);
        fresh_diag_ = false;
    }

    // собрать матрицу из текущего состояния счётчика:
    // диагональ — текущее разложение, наддиагональ — offdiag_ по позициям
    // (i,j), i<j, в порядке обхода по строкам; ниже диагонали — нули
    h.assign(n, std::vector<long>(n, 0));
    std::size_t pos = 0;
    for (std::size_t i = 0; i < n; ++i) {
        h[i][i] = d[i];
        for (std::size_t j = i + 1; j < n; ++j) h[i][j] = offdiag_[pos++];
    }

    // лимиты счётчика: элемент позиции (i,j) лежит в [0, d_j);
    // при d_j == 1 позиция всегда нулевая
    std::vector<long> limits(m);
    pos = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) limits[pos++] = d[j];

    // продвинуть одометр: инкремент младшей (последней) позиции с переносом
    for (std::size_t p = m; p-- > 0;) {
        if (offdiag_[p] + 1 < limits[p]) {
            ++offdiag_[p];
            std::fill(offdiag_.begin() + static_cast<std::ptrdiff_t>(p) + 1,
                      offdiag_.end(), 0L);
            return true;
        }
    }

    // переполнение счётчика — переходим к следующей диагонали
    ++diag_pos_;
    fresh_diag_ = true;
    return true;
}

long SublatticeIterator::count(int dim, long index) {
    // для диагонали (d_0..d_{dim-1}) наддиагональ столбца j имеет d_j^j
    // вариантов (j элементов, каждый из [0, d_j)); итого prod_j d_j^j.
    // Накопление в __int128 с проверкой переполнения long — для больших
    // индексов число подрешёток быстро превышает диапазон (защита от тихого UB).
    //
    // __int128 — расширение GCC/Clang, в MSVC отсутствует. Проект таргетит
    // POSIX и собирается clang/gcc (см. CMakeLists.txt: -Wall -Wextra), где тип
    // доступен всегда; на иной платформе здесь понадобился бы fallback
    // (например, проверка переполнения через деление). Оставляем __int128.
    __int128 total = 0;
    for (const std::vector<long>& d : ordered_factorizations(index, dim)) {
        __int128 variants = 1;
        for (int j = 0; j < dim; ++j)
            for (int t = 0; t < j; ++t) variants *= d[static_cast<std::size_t>(j)];
        total += variants;
    }
    if (total > static_cast<__int128>(std::numeric_limits<long>::max()))
        throw std::overflow_error("SublatticeIterator::count: число подрешёток превышает long");
    return static_cast<long>(total);
}

Mat apply_hnf(const HnfMatrix& h, const Mat& basis) {
    assert(h.size() == basis.size());
    Mat result;
    result.reserve(h.size());
    // строка i результата = линейная комбинация строк basis с коэффициентами h[i]
    for (const std::vector<long>& row : h) result.push_back(combination(row, basis));
    return result;
}

HnfMatrix hnf_of(HnfMatrix c) {
    const std::size_t n = c.size();
    for (const auto& row : c)
        if (row.size() != n) throw std::invalid_argument("hnf_of: матрица не квадратная");

    for (std::size_t col = 0; col < n; ++col) {
        // евклидово исключение: обнулить элементы ниже (col, col)
        for (std::size_t row = col + 1; row < n; ++row) {
            while (c[row][col] != 0) {
                if (c[col][col] == 0) {
                    std::swap(c[col], c[row]);
                    continue;
                }
                const long q = c[row][col] / c[col][col];
                for (std::size_t j = 0; j < n; ++j) c[row][j] -= q * c[col][j];
                if (c[row][col] != 0) std::swap(c[col], c[row]);
            }
        }
        if (c[col][col] == 0)
            throw std::invalid_argument("hnf_of: вырожденная матрица");
        // нормализация знака диагонали
        if (c[col][col] < 0)
            for (std::size_t j = 0; j < n; ++j) c[col][j] = -c[col][j];
        // приведение элементов выше диагонали в [0, d): floor-деление
        const long d = c[col][col];
        for (std::size_t row = 0; row < col; ++row) {
            long q = c[row][col] / d;
            if (c[row][col] % d < 0) --q;
            if (q != 0)
                for (std::size_t j = 0; j < n; ++j) c[row][j] -= q * c[col][j];
        }
    }
    return c;
}

HnfMatrix transition_in_basis(const Mat& sub_basis, const Mat& basis, double tol) {
    const std::size_t n = basis.size();
    // B^T x = s^T построчно: коэффициенты разложения строки sub_basis по строкам basis
    Mat bt(n, Vec(n, 0.0));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) bt[i][j] = basis[j][i];

    HnfMatrix coeffs;
    coeffs.reserve(n);
    for (const Vec& s : sub_basis) {
        Vec x;
        if (!solve_linear(bt, s, x))
            throw std::invalid_argument("transition_in_basis: вырожденный базис");
        std::vector<long> row(n);
        for (std::size_t j = 0; j < n; ++j) {
            const double r = std::round(x[j]);
            if (std::abs(x[j] - r) > tol)
                throw std::invalid_argument(
                    "transition_in_basis: sub_basis не является подрешёткой basis "
                    "(нецелые коэффициенты разложения)");
            row[j] = static_cast<long>(r);
        }
        coeffs.push_back(std::move(row));
    }
    return hnf_of(std::move(coeffs));
}

}  // namespace combigeo
