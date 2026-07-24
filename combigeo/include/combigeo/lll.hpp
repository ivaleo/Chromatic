#pragma once
// LLL-приведение базиса решётки (порт проверенной реализации из python-пакета
// voronoi4d: алгоритм Ленстры–Ленстры–Ловаса с полным пересчётом Грама-Шмидта).

#include "combigeo/linalg.hpp"

namespace combigeo {

// LLL-приведение. basis — строки = векторы базиса; delta — параметр Ловаса.
// Возвращает приведённый базис (определитель решётки сохраняется с точностью до знака).
Mat lll_reduce(Mat basis, double delta = 0.75);

}  // namespace combigeo
