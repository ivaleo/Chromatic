#pragma once
// Линейная алгебра малых размерностей (n <= 8) на чистом STL.
// Vec — вектор, Mat — матрица как список строк (строки = векторы базиса).

#include <cstddef>
#include <vector>

namespace combigeo {

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;

// допуски
inline constexpr double kEps = 1e-9;        // общий допуск сравнения
inline constexpr double kEpsDegenerate = 1e-12;  // порог вырожденности

// --- валидация на границе публичного API ---
// (внутренние горячие функции полагаются на assert; эти проверки работают
//  и в release-сборке, где assert вырезан -DNDEBUG)

// Проверяет, что matrix — непустая квадратная матрица из конечных чисел;
// возвращает её размерность. Бросает std::invalid_argument с понятным текстом.
int validated_dim(const Mat& matrix, const char* who = "матрица");

// Проверяет, что вектор имеет размерность dim и состоит из конечных чисел.
void validate_vector(const Vec& v, int dim, const char* who = "вектор");

// --- базовые операции над векторами ---
//
// Горячие функции, работающие над парой векторов (dot/add/sub/axpy_inplace/
// dist2 и т.п.), ТРЕБУЮТ совпадения размерностей аргументов. Это контролируется
// только assert: в release-сборке (-DNDEBUG) рассогласование длин — UB
// (выход за границы буфера). Валидируйте размерности на границе публичного API
// через validated_dim / validate_vector (см. выше), а не в этих функциях.
// norm2/norm принимают вектор любой длины.

double dot(const Vec& a, const Vec& b);      // требует a.size() == b.size()
double norm2(const Vec& a);                  // квадрат длины
double norm(const Vec& a);
Vec add(const Vec& a, const Vec& b);         // требует равной размерности
Vec sub(const Vec& a, const Vec& b);         // требует равной размерности
Vec scaled(const Vec& a, double k);
void axpy_inplace(Vec& y, double k, const Vec& x);  // y += k*x; требует равной размерности
double dist2(const Vec& a, const Vec& b);    // квадрат расстояния; требует равной размерности
Vec zeros(std::size_t n);

// --- операции над матрицами ---

// определитель (Гаусс с частичным выбором ведущего элемента); матрица копируется
double det(Mat a);

// решает A x = b (Гаусс с частичным выбором). Возвращает false, если
// матрица вырождена (|ведущий элемент| < kEpsDegenerate).
bool solve_linear(Mat a, Vec b, Vec& x);

// умножение вектора-строки коэффициентов на матрицу базиса: result = c * basis
// (линейная комбинация строк basis с коэффициентами c)
Vec combination(const std::vector<long>& c, const Mat& basis);
Vec combination(const Vec& c, const Mat& basis);

// ортогонализация Грама-Шмидта БЕЗ нормировки.
// b_star — ортогонализованные векторы, mu[i][j] — коэффициенты (j < i).
void gram_schmidt(const Mat& basis, Mat& b_star, Mat& mu);

}  // namespace combigeo
