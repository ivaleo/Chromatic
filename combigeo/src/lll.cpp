#include "combigeo/lll.hpp"

#include <cmath>

namespace combigeo {

// Порт проверенной реализации из python-пакета voronoi4d (lll_reduce_python):
// size reduction + условие Ловаса, с полным пересчётом Грама-Шмидта после
// каждого изменения базиса. Для размерностей <= 6 затраты несущественны,
// зато реализация прозрачна и надёжна.
Mat lll_reduce(Mat basis, double delta) {
    // delta вне (0.25, 1] нарушает условие завершения алгоритма Ловаса —
    // приводим к безопасному диапазону (классическое значение 0.75)
    if (!(delta > 0.25 && delta <= 1.0)) delta = 0.75;

    const std::size_t n = basis.size();
    if (n < 2) return basis;

    Mat b_star, mu;
    gram_schmidt(basis, b_star, mu);

    auto beta = [&](std::size_t i) { return mu[i][i]; };  // |b*_i|^2 хранится на диагонали

    std::size_t k = 1;
    while (k < n) {
        // size reduction
        for (std::size_t l = k; l-- > 0;) {
            if (std::abs(mu[k][l]) > 0.5) {
                const double q = std::round(mu[k][l]);
                axpy_inplace(basis[k], -q, basis[l]);
                gram_schmidt(basis, b_star, mu);  // пересчёт после изменения базиса
            }
        }

        // условие Ловаса
        if (beta(k) < (delta - mu[k][k - 1] * mu[k][k - 1]) * beta(k - 1)) {
            std::swap(basis[k], basis[k - 1]);
            gram_schmidt(basis, b_star, mu);
            k = (k > 1) ? k - 1 : 1;
        } else {
            ++k;
        }
    }
    return basis;
}

}  // namespace combigeo
