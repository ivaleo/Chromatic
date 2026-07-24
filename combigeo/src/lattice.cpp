#include "combigeo/lattice.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "combigeo/lll.hpp"

namespace combigeo {

Lattice::Lattice(Mat basis) : basis_(std::move(basis)) {
    // проверяет квадратность, размерность и конечность элементов (NaN/Inf)
    validated_dim(basis_, "Lattice: базис");
    // проверка вырожденности ОТНОСИТЕЛЬНАЯ (масштабно-инвариантная):
    // |det| сравнивается с произведением норм строк (неравенство Адамара),
    // иначе валидная решётка в мелком масштабе ложно бракуется, а нулевые
    // строки в крупном — проходят
    double hadamard = 1.0;
    for (const Vec& row : basis_) hadamard *= norm(row);
    if (!(hadamard > 0.0) || std::abs(combigeo::det(basis_)) < kEpsDegenerate * hadamard)
        throw std::invalid_argument("Lattice: вырожденный базис");
}

double Lattice::det() const { return std::abs(combigeo::det(basis_)); }

Lattice Lattice::lll_reduced(double delta) const { return Lattice(lll_reduce(basis_, delta)); }

namespace {

// Рекурсивный перебор коэффициентов с отсечением по частичной норме
// (sphere decoding в координатах Грама-Шмидта), с необязательным сдвигом (CVP).
//
// Вектор v = sum_i c_i b_i. В ортогональном базисе b*_j координата (v - target):
//   (c_j + sum_{i>j} c_i mu[i][j] - t_j) * |b*_j|,  t_j = <target, b*_j>/|b*_j|^2.
// Перебираем коэффициенты с конца (j = n-1 .. 0); частичная сумма квадратов
// уже зафиксированных координат — нижняя оценка |v - target|^2, что даёт точные
// границы для c_j: |c_j + center_j| <= sqrt(remaining)/|b*_j|.
//
// Режимы: target == nullptr — перебор вокруг нуля, нулевой вектор исключается,
// из пары (v, -v) выдаётся канонический представитель; target != nullptr —
// CVP-режим, выдаются ВСЕ векторы с |v - target| <= bound, включая нулевой.
struct SphereEnum {
    const Mat& basis;
    const Mat& b_star;
    const Mat& mu;
    double bound2;                    // |v - target|^2 <= bound2
    std::vector<double> bstar_norm2;  // |b*_j|^2
    std::vector<double> tcoord;      // t_j (пусто в режиме target == nullptr)
    std::vector<long> coeffs;
    std::vector<Vec>* out;
    bool cvp_mode = false;

    void run() {
        const std::size_t n = basis.size();
        coeffs.assign(n, 0);
        descend(n, 0.0);
    }

    // level: индекс j+1 (идём от n к 0); partial2 — сумма квадратов координат j+1..n-1
    void descend(std::size_t level, double partial2) {
        if (level == 0) {
            if (!cvp_mode) {
                // не нулевой вектор и канонический представитель пары (v, -v):
                // первый ненулевой коэффициент положителен
                for (std::size_t i = 0; i < coeffs.size(); ++i) {
                    if (coeffs[i] > 0) break;
                    if (coeffs[i] < 0) return;
                    if (i + 1 == coeffs.size()) return;  // все нули
                }
            }
            Vec v = combination(coeffs, basis);
            // финальная проверка нормы по фактическим координатам вектора
            if (cvp_mode) {
                double d2 = 0.0;
                for (std::size_t j = 0; j < b_star.size(); ++j) {
                    double c = static_cast<double>(coeffs[j]) - tcoord[j];
                    for (std::size_t i = j + 1; i < coeffs.size(); ++i)
                        c += static_cast<double>(coeffs[i]) * mu[i][j];
                    d2 += c * c * bstar_norm2[j];
                }
                if (d2 <= bound2 + kEps) out->push_back(std::move(v));
            } else {
                if (norm2(v) <= bound2 + kEps) out->push_back(std::move(v));
            }
            return;
        }

        const std::size_t j = level - 1;

        // центр интервала: c_j + sum_{i>j} c_i mu[i][j] - t_j
        // в пределах +-sqrt(rem)/|b*_j|
        double center = cvp_mode ? -tcoord[j] : 0.0;
        for (std::size_t i = j + 1; i < coeffs.size(); ++i)
            center += static_cast<double>(coeffs[i]) * mu[i][j];

        const double remaining2 = bound2 - partial2;
        if (remaining2 < -kEps) return;
        const double radius =
            std::sqrt(std::max(0.0, remaining2) / std::max(bstar_norm2[j], kEpsDegenerate));

        const long lo = static_cast<long>(std::ceil(-center - radius - kEps));
        const long hi = static_cast<long>(std::floor(-center + radius + kEps));

        for (long c = lo; c <= hi; ++c) {
            coeffs[j] = c;
            const double coord = (static_cast<double>(c) + center);
            const double add2 = coord * coord * bstar_norm2[j];
            descend(j, partial2 + add2);
        }
        coeffs[j] = 0;
    }
};

// координаты цели в ортогонализованном базисе: t_j = <target, b*_j>/|b*_j|^2
std::vector<double> target_coords(const Vec& target, const Mat& b_star,
                                  const std::vector<double>& bn2) {
    std::vector<double> t(b_star.size());
    for (std::size_t j = 0; j < b_star.size(); ++j)
        t[j] = dot(target, b_star[j]) / std::max(bn2[j], kEpsDegenerate);
    return t;
}

}  // namespace

std::vector<Vec> Lattice::vectors_within(double bound, bool assume_reduced) const {
    // работаем в LLL-приведённом базисе: узкие границы перебора
    const Mat reduced = assume_reduced ? basis_ : lll_reduce(basis_);

    Mat b_star, mu;
    gram_schmidt(reduced, b_star, mu);

    std::vector<double> bn2(reduced.size());
    for (std::size_t j = 0; j < reduced.size(); ++j) bn2[j] = norm2(b_star[j]);

    std::vector<Vec> out;
    SphereEnum e{reduced, b_star, mu, bound * bound, std::move(bn2), {}, {}, &out, false};
    e.run();
    return out;
}

Vec Lattice::shortest_vector(bool assume_reduced) const {
    // стартовая граница — минимальная длина вектора LLL-базиса
    const Mat reduced = assume_reduced ? basis_ : lll_reduce(basis_);
    double best = std::numeric_limits<double>::infinity();
    for (const auto& row : reduced) best = std::min(best, norm(row));

    const Lattice red(reduced);
    std::vector<Vec> cand = red.vectors_within(best + kEps, /*assume_reduced=*/true);
    Vec result;
    double best2 = std::numeric_limits<double>::infinity();
    for (auto& v : cand) {
        const double n2 = norm2(v);
        if (n2 > kEpsDegenerate && n2 < best2) {
            best2 = n2;
            result = std::move(v);
        }
    }
    return result;
}

double Lattice::babai_distance(const Vec& target, bool assume_reduced) const {
    const Mat reduced = assume_reduced ? basis_ : lll_reduce(basis_);
    Mat b_star, mu;
    gram_schmidt(reduced, b_star, mu);
    std::vector<double> bn2(reduced.size());
    for (std::size_t j = 0; j < reduced.size(); ++j) bn2[j] = norm2(b_star[j]);

    // ближайшая плоскость Бабаи: округление коэффициентов с конца
    const std::size_t n = reduced.size();
    std::vector<double> t = target_coords(target, b_star, bn2);
    std::vector<long> c(n, 0);
    for (std::size_t jj = n; jj-- > 0;) {
        double center = -t[jj];
        for (std::size_t i = jj + 1; i < n; ++i)
            center += static_cast<double>(c[i]) * mu[i][jj];
        c[jj] = std::lround(-center);
    }
    Vec b = combination(c, reduced);
    return std::sqrt(dist2(b, target));
}

std::vector<Vec> Lattice::vectors_near(const Vec& target, double bound,
                                       bool assume_reduced) const {
    const Mat reduced = assume_reduced ? basis_ : lll_reduce(basis_);
    Mat b_star, mu;
    gram_schmidt(reduced, b_star, mu);
    std::vector<double> bn2(reduced.size());
    for (std::size_t j = 0; j < reduced.size(); ++j) bn2[j] = norm2(b_star[j]);

    std::vector<Vec> out;
    SphereEnum e{reduced,  b_star, mu, bound * bound, bn2, target_coords(target, b_star, bn2),
                 {},       &out,   true};
    e.run();
    return out;
}

}  // namespace combigeo
