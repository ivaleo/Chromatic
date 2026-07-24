#pragma once
// Перебор подрешёток заданного индекса: верхнетреугольные целочисленные матрицы
// в эрмитовой нормальной форме (HNF).
//
// Матрица перехода H (dim x dim): диагональ d_0..d_{dim-1} — упорядоченное
// разложение index = d_0 * d_1 * ... * d_{dim-1} (включая единицы);
// наддиагональные элементы столбца j лежат в [0, d_j).
// Каждая подрешётка индекса index встречается ровно один раз.
// Базис подрешётки: строки H * basis (как в эталонном python-пакете voronoi4d).

#include <cstdint>
#include <vector>

#include "combigeo/linalg.hpp"

namespace combigeo {

using HnfMatrix = std::vector<std::vector<long>>;

class SublatticeIterator {
public:
    SublatticeIterator(int dim, long index);

    // Следующая HNF-матрица; false — перебор закончен.
    bool next(HnfMatrix& h);

    // Общее количество подрешёток индекса index в решётке ранга dim
    // (= числу HNF-матриц; для проверки и прогресса).
    static long count(int dim, long index);

private:
    int dim_;
    long index_;
    std::vector<std::vector<long>> diagonals_;  // все упорядоченные разложения index на dim множителей
    std::size_t diag_pos_ = 0;
    std::vector<long> offdiag_;   // текущие наддиагональные элементы (лексикографический счётчик)
    bool fresh_diag_ = true;      // начали новую диагональ — offdiag нулевой ещё не выдан
};

// Все упорядоченные разложения n на k множителей >= 1 (включая перестановки):
// например, n=4, k=2 -> {1,4},{2,2},{4,1}.
std::vector<std::vector<long>> ordered_factorizations(long n, int k);

// Применяет HNF-матрицу к базису: строки результата = h * basis.
Mat apply_hnf(const HnfMatrix& h, const Mat& basis);

// Эрмитова нормальная форма ЗАДАННОЙ целочисленной матрицы (унимодулярные
// операции над строками): верхнетреугольная, диагональ > 0, элементы выше
// диагонали приведены в [0, d_j) по модулю диагонали своего столбца — та же
// нормализация, что у SublatticeIterator. Бросает std::invalid_argument
// на вырожденной матрице.
HnfMatrix hnf_of(HnfMatrix c);

// Матрица перехода подрешётки в координатах ПРОИЗВОЛЬНОГО базиса basis:
// строки sub_basis раскладываются по строкам basis, коэффициенты обязаны быть
// целыми (допуск tol, иначе std::invalid_argument), результат приводится к HNF.
HnfMatrix transition_in_basis(const Mat& sub_basis, const Mat& basis, double tol = 1e-6);

}  // namespace combigeo
