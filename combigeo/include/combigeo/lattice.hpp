#pragma once
// Решётка: базис, определитель, LLL-приведение, перебор коротких векторов.

#include <vector>

#include "combigeo/linalg.hpp"

namespace combigeo {

class Lattice {
public:
    // basis: строки — векторы базиса; должна быть квадратной и невырожденной
    // (иначе std::invalid_argument).
    explicit Lattice(Mat basis);

    int dim() const { return static_cast<int>(basis_.size()); }
    const Mat& basis() const { return basis_; }
    double det() const;                 // |определитель| (объём фундаментальной области)
    Lattice lll_reduced(double delta = 0.75) const;

    // Все векторы решётки v = c*basis с |v| <= bound (v != 0), по одному из пары (v, -v):
    // первый ненулевой коэффициент положителен. Гарантированно полный перебор:
    // рекурсивное перечисление коэффициентов с отсечением по частичной норме
    // в ортогонализованном базисе (sphere decoding по Грам-Шмидту LLL-базиса).
    // assume_reduced = true: базис уже LLL-приведён, повторное приведение пропускается
    // (границы перебора остаются корректными при любом базисе, но без приведения
    // могут быть катастрофически широкими — используйте только для приведённых).
    std::vector<Vec> vectors_within(double bound, bool assume_reduced = false) const;

    // Кратчайший ненулевой вектор решётки (через vectors_within с адаптивной границей).
    Vec shortest_vector(bool assume_reduced = false) const;

    // Все векторы решётки u с |u - target| <= bound (включая u = 0) — точный
    // CVP-перебор (сдвинутый sphere decoding). Полнота гарантирована тем же
    // отсечением по частичной норме; bound должен быть достижим (например,
    // расстояние до точки Бабаи, см. babai_distance).
    std::vector<Vec> vectors_near(const Vec& target, double bound,
                                  bool assume_reduced = false) const;

    // Расстояние |target - b|, где b — точка Бабаи (округление коэффициентов
    // в ортогонализованном базисе): верхняя оценка dist(target, Λ), пригодная
    // как стартовый радиус для vectors_near.
    double babai_distance(const Vec& target, bool assume_reduced = false) const;

private:
    Mat basis_;
};

}  // namespace combigeo
