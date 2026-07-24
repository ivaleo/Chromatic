#pragma once
// Основной конвейер: поиск оптимальной подрешётки для периодической раскраски.
//
// Теория (методология Л.Л. Иванова, см. статью в python-проекте voronoi):
// det матрицы перехода = индекс подрешётки = число цветов k.
// Расстояние между одноцветными ячейками Вороного с центрами 0 и v
// (по лемме Иванова, для центрально-симметричных ячеек):
//   D(v) = 2 * dist(v/2, V0).
// Нормированное запрещённое расстояние d = D / diam(V0); раскраска пригодна при d >= 1
// (множитель 2 уже содержится в D = 2*dist(v/2, V0)).
// Для каждого индекса ищем подрешётку с максимальным min_v D(v).

#include <cstdint>
#include <functional>
#include <map>
#include <vector>

#include "combigeo/lattice.hpp"
#include "combigeo/polytope.hpp"
#include "combigeo/sublattice.hpp"

namespace combigeo {

struct SolveOptions {
    double lll_delta = 0.75;
    bool use_cache = true;       // кэш расстояний по вектору v (у каждого потока свой)
    int progress_every = 0;      // печатать прогресс каждые N подрешёток (0 = никогда)
    std::function<void(long examined, long total, double best)> on_progress;  // или коллбек
    // Число потоков перебора подрешёток: 0 = авто (число ядер минус два, но не
    // меньше 1), 1 = однопоточно. Результат (max d) от числа потоков не зависит;
    // при нескольких оптимумах с равным d выбранная подрешётка может отличаться.
    int threads = 0;
};
// Ранняя отбраковка встроена: как только промежуточный минимум подрешётки
// опускается ниже текущего лучшего результата, её обсчёт прерывается
// (branch-and-bound для задачи max-min).

struct SublatticeResult {
    HnfMatrix transition;        // матрица перехода (HNF) В БАЗИСЕ ПОЛЬЗОВАТЕЛЯ:
                                 // строки transition * входной_базис порождают подрешётку
    Mat sub_basis;               // базис подрешётки (после LLL, объемлющие координаты)
    double min_distance = 0.0;   // min_v D(v) — расстояние между одноцветными ячейками
    Vec witness;                 // вектор v, на котором достигается минимум
};

struct SolveResult {
    long index = 0;              // число цветов k
    double diameter = 0.0;       // diam(V0)
    SublatticeResult best;       // подрешётка с максимальным min_distance
    double normalized = 0.0;     // d = D / diam(V0), D = best.min_distance (межъячеечное);
                                 // эквивалентно 2*dist(s, V0)/diam из статьи; пригодно при d >= 1
    long examined = 0;           // сколько подрешёток обсчитано
};

// Минимальное расстояние между одноцветными ячейками для подрешётки sub:
//   min по ненулевым v из sub от D(v) = 2*dist(v/2, cell).
// Гарантия полноты перебора: достаточно кандидатов v с |v| < current + cell.diameter
// (т.к. D(v) >= |v| - diameter, см. реализацию); граница фиксируется по
// НАЧАЛЬНОМУ значению current и остаётся корректной, т.к. current только убывает.
// cache: optional, ключ — округлённый вектор v (см. реализацию), может быть nullptr.
// early_stop: если текущий минимум стал < early_stop — вернуть его немедленно
// (подрешётка уже не интересна; 0 = считать до конца).
// witness_out: если не nullptr — вектор, на котором достигнут возвращённый минимум
// (при раннем выходе — лучший из просмотренных).
// ВНИМАНИЕ: cell и sub должны быть в согласованном масштабе (публичные обёртки
// find_optimal/биндинги нормируют масштаб сами).
double min_color_distance(const VoronoiCell& cell, const Lattice& sub,
                          std::map<std::vector<long>, double>* cache = nullptr,
                          double early_stop = 0.0, Vec* witness_out = nullptr);

// Полный поиск для одного индекса.
// Бросает std::invalid_argument при index < 1, а также при некорректном базисе
// (неквадратная/вырожденная матрица, нечисловые элементы — проверка validated_dim).
// Бросает std::overflow_error, если число подрешёток (SublatticeIterator::count)
// не помещается в long (большой index в размерности >= 3).
SolveResult find_optimal(const Mat& basis, long index, const SolveOptions& opts = {});

// Диапазон индексов [from, to]; ключ результата — индекс.
// Те же исключения, что и find_optimal (вызывается для каждого индекса диапазона):
// std::invalid_argument (index < 1 / плохой базис), std::overflow_error (из count).
std::map<long, SolveResult> find_optimal_range(const Mat& basis, long from, long to,
                                               const SolveOptions& opts = {});

}  // namespace combigeo
