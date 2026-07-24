#include "combigeo/linalg.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace combigeo {

int validated_dim(const Mat& matrix, const char* who) {
    if (matrix.empty())
        throw std::invalid_argument(std::string(who) + ": пустая матрица");
    const std::size_t n = matrix.size();
    for (const Vec& row : matrix) {
        if (row.size() != n)
            throw std::invalid_argument(std::string(who) + ": матрица не квадратная (строка длины " +
                                        std::to_string(row.size()) + " при размерности " +
                                        std::to_string(n) + ")");
        for (double x : row)
            if (!std::isfinite(x))
                throw std::invalid_argument(std::string(who) + ": нечисловой элемент (NaN/Inf)");
    }
    return static_cast<int>(n);
}

void validate_vector(const Vec& v, int dim, const char* who) {
    if (static_cast<int>(v.size()) != dim)
        throw std::invalid_argument(std::string(who) + ": размерность " +
                                    std::to_string(v.size()) + " вместо ожидаемой " +
                                    std::to_string(dim));
    for (double x : v)
        if (!std::isfinite(x))
            throw std::invalid_argument(std::string(who) + ": нечисловой элемент (NaN/Inf)");
}

double dot(const Vec& a, const Vec& b) {
    assert(a.size() == b.size());
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

double norm2(const Vec& a) { return dot(a, a); }

double norm(const Vec& a) { return std::sqrt(norm2(a)); }

Vec add(const Vec& a, const Vec& b) {
    assert(a.size() == b.size());
    Vec r(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) r[i] = a[i] + b[i];
    return r;
}

Vec sub(const Vec& a, const Vec& b) {
    assert(a.size() == b.size());
    Vec r(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) r[i] = a[i] - b[i];
    return r;
}

Vec scaled(const Vec& a, double k) {
    Vec r(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) r[i] = a[i] * k;
    return r;
}

void axpy_inplace(Vec& y, double k, const Vec& x) {
    assert(y.size() == x.size());
    for (std::size_t i = 0; i < y.size(); ++i) y[i] += k * x[i];
}

double dist2(const Vec& a, const Vec& b) {
    assert(a.size() == b.size());
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

Vec zeros(std::size_t n) { return Vec(n, 0.0); }

// --- матрицы ---

// Гаусс с частичным выбором ведущего элемента; возвращает определитель.
// Матрица передаётся по значению и разрушается.
double det(Mat a) {
    const std::size_t n = a.size();
    double d = 1.0;

    for (std::size_t col = 0; col < n; ++col) {
        // выбор ведущего элемента
        std::size_t pivot = col;
        for (std::size_t r = col + 1; r < n; ++r)
            if (std::abs(a[r][col]) > std::abs(a[pivot][col])) pivot = r;

        if (std::abs(a[pivot][col]) < kEpsDegenerate) return 0.0;

        if (pivot != col) {
            std::swap(a[pivot], a[col]);
            d = -d;
        }

        d *= a[col][col];

        for (std::size_t r = col + 1; r < n; ++r) {
            const double f = a[r][col] / a[col][col];
            for (std::size_t c = col; c < n; ++c) a[r][c] -= f * a[col][c];
        }
    }
    return d;
}

bool solve_linear(Mat a, Vec b, Vec& x) {
    const std::size_t n = a.size();
    assert(b.size() == n);

    // прямой ход с частичным выбором
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        for (std::size_t r = col + 1; r < n; ++r)
            if (std::abs(a[r][col]) > std::abs(a[pivot][col])) pivot = r;

        if (std::abs(a[pivot][col]) < kEpsDegenerate) return false;

        if (pivot != col) {
            std::swap(a[pivot], a[col]);
            std::swap(b[pivot], b[col]);
        }

        for (std::size_t r = col + 1; r < n; ++r) {
            const double f = a[r][col] / a[col][col];
            for (std::size_t c = col; c < n; ++c) a[r][c] -= f * a[col][c];
            b[r] -= f * b[col];
        }
    }

    // обратный ход
    x.assign(n, 0.0);
    for (std::size_t i = n; i-- > 0;) {
        double s = b[i];
        for (std::size_t c = i + 1; c < n; ++c) s -= a[i][c] * x[c];
        x[i] = s / a[i][i];
    }
    return true;
}

Vec combination(const std::vector<long>& c, const Mat& basis) {
    assert(c.size() == basis.size());
    Vec r = zeros(basis.empty() ? 0 : basis[0].size());
    for (std::size_t i = 0; i < c.size(); ++i)
        if (c[i] != 0) axpy_inplace(r, static_cast<double>(c[i]), basis[i]);
    return r;
}

Vec combination(const Vec& c, const Mat& basis) {
    assert(c.size() == basis.size());
    Vec r = zeros(basis.empty() ? 0 : basis[0].size());
    for (std::size_t i = 0; i < c.size(); ++i)
        if (c[i] != 0.0) axpy_inplace(r, c[i], basis[i]);
    return r;
}

void gram_schmidt(const Mat& basis, Mat& b_star, Mat& mu) {
    const std::size_t n = basis.size();
    b_star = basis;  // копия, будет ортогонализована
    mu.assign(n, Vec(n, 0.0));

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            const double denom = dot(b_star[j], b_star[j]);
            mu[i][j] = (denom > kEpsDegenerate) ? dot(basis[i], b_star[j]) / denom : 0.0;
            axpy_inplace(b_star[i], -mu[i][j], b_star[j]);
        }
        mu[i][i] = dot(b_star[i], b_star[i]);  // beta_i = |b*_i|^2
    }
}

}  // namespace combigeo
