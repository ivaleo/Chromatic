#include "combigeo/lattice.hpp"

#include <algorithm>
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

// Приведённый базис вместе с его ортогонализацией Грама-Шмидта — общая
// подготовка всех переборов (они работают в координатах b*).
struct ReducedFrame {
    Mat basis;
    Mat b_star;
    Mat mu;
    std::vector<double> bstar_norm2;  // |b*_j|^2
};

ReducedFrame make_frame(const Mat& basis, bool assume_reduced) {
    ReducedFrame frame;
    frame.basis = assume_reduced ? basis : lll_reduce(basis);
    gram_schmidt(frame.basis, frame.b_star, frame.mu);
    frame.bstar_norm2.resize(frame.b_star.size());
    for (std::size_t j = 0; j < frame.b_star.size(); ++j)
        frame.bstar_norm2[j] = norm2(frame.b_star[j]);
    return frame;
}

// координаты цели в ортогонализованном базисе: t_j = <target, b*_j>/|b*_j|^2
std::vector<double> target_coords(const Vec& target, const ReducedFrame& frame) {
    std::vector<double> t(frame.b_star.size());
    for (std::size_t j = 0; j < t.size(); ++j)
        t[j] = dot(target, frame.b_star[j]) / std::max(frame.bstar_norm2[j], kEpsDegenerate);
    return t;
}

// Рекурсивный перебор коэффициентов с отсечением по частичной норме
// (sphere decoding в координатах Грама-Шмидта), с необязательным сдвигом (CVP).
//
// Вектор v = sum_i c_i b_i. В ортогональном базисе b*_j координата (v - target):
//   (c_j + sum_{i>j} c_i mu[i][j] - t_j) * |b*_j|,  t_j = <target, b*_j>/|b*_j|^2.
// Перебираем коэффициенты с конца (j = n-1 .. 0); частичная сумма квадратов
// уже зафиксированных координат — нижняя оценка |v - target|^2, что даёт точные
// границы для c_j: |c_j + center_j| <= sqrt(remaining)/|b*_j|.
//
// Режимы: cvp_mode == false — перебор вокруг нуля, нулевой вектор исключается,
// из пары (v, -v) выдаётся канонический представитель; cvp_mode == true —
// выдаются ВСЕ векторы с |v - target| <= bound, включая нулевой (tcoord = t_j).
struct SphereEnum {
    const ReducedFrame& frame;
    double bound2;                 // |v - target|^2 <= bound2
    std::vector<double> tcoord;    // t_j (пусто вне CVP-режима)
    bool cvp_mode;
    std::vector<Vec>* out;
    std::vector<long> coeffs{};    // рабочее состояние перебора, задаётся в run()

    void run() {
        coeffs.assign(frame.basis.size(), 0);
        descend(frame.basis.size(), 0.0);
    }

    // Смещение координаты j: c_j + sum_{i>j} c_i mu[i][j] - t_j.
    double offset_at(std::size_t j) const {
        double c = cvp_mode ? -tcoord[j] : 0.0;
        for (std::size_t i = j + 1; i < coeffs.size(); ++i)
            c += static_cast<double>(coeffs[i]) * frame.mu[i][j];
        return c;
    }

    // |v - target|^2 по всем координатам (финальная проверка листа в CVP-режиме)
    double shifted_norm2() const {
        double d2 = 0.0;
        for (std::size_t j = 0; j < frame.b_star.size(); ++j) {
            const double c = static_cast<double>(coeffs[j]) + offset_at(j);
            d2 += c * c * frame.bstar_norm2[j];
        }
        return d2;
    }

    // level: индекс j+1 (идём от n к 0); partial2 — сумма квадратов координат j+1..n-1
    void descend(std::size_t level, double partial2) {
        if (level == 0) {
            if (cvp_mode) {
                if (shifted_norm2() <= bound2 + kEps)
                    out->push_back(combination(coeffs, frame.basis));
                return;
            }
            // канонический представитель пары (v, -v): первый ненулевой
            // коэффициент положителен; нулевой вектор отбрасывается
            const auto first_nz =
                std::find_if(coeffs.begin(), coeffs.end(), [](long c) { return c != 0; });
            if (first_nz == coeffs.end() || *first_nz < 0) return;
            Vec v = combination(coeffs, frame.basis);
            if (norm2(v) <= bound2 + kEps) out->push_back(std::move(v));
            return;
        }

        const std::size_t j = level - 1;

        // c_j лежит в интервале -center +- sqrt(remaining2)/|b*_j|
        const double center = offset_at(j);
        const double remaining2 = bound2 - partial2;
        if (remaining2 < -kEps) return;
        const double radius =
            std::sqrt(std::max(0.0, remaining2) / std::max(frame.bstar_norm2[j], kEpsDegenerate));

        const long lo = static_cast<long>(std::ceil(-center - radius - kEps));
        const long hi = static_cast<long>(std::floor(-center + radius + kEps));

        for (long c = lo; c <= hi; ++c) {
            coeffs[j] = c;
            const double coord = static_cast<double>(c) + center;
            descend(j, partial2 + coord * coord * frame.bstar_norm2[j]);
        }
        coeffs[j] = 0;
    }
};

}  // namespace

std::vector<Vec> Lattice::vectors_within(double bound, bool assume_reduced) const {
    // работаем в LLL-приведённом базисе: узкие границы перебора
    const ReducedFrame frame = make_frame(basis_, assume_reduced);

    std::vector<Vec> out;
    SphereEnum e{frame, bound * bound, {}, /*cvp_mode=*/false, &out};
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
    const ReducedFrame frame = make_frame(basis_, assume_reduced);
    const std::vector<double> t = target_coords(target, frame);

    // ближайшая точка Бабаи: округление коэффициентов с конца
    const std::size_t n = frame.basis.size();
    std::vector<long> c(n, 0);
    for (std::size_t j = n; j-- > 0;) {
        double center = -t[j];
        for (std::size_t i = j + 1; i < n; ++i)
            center += static_cast<double>(c[i]) * frame.mu[i][j];
        c[j] = std::lround(-center);
    }
    return std::sqrt(dist2(combination(c, frame.basis), target));
}

std::vector<Vec> Lattice::vectors_near(const Vec& target, double bound,
                                       bool assume_reduced) const {
    const ReducedFrame frame = make_frame(basis_, assume_reduced);

    std::vector<Vec> out;
    SphereEnum e{frame, bound * bound, target_coords(target, frame), /*cvp_mode=*/true, &out};
    e.run();
    return out;
}

}  // namespace combigeo
