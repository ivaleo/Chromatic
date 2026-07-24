#pragma once
// Инструменты для больших размерностей (n >= 5): поиск пригодных подрешёток без
// перебора всех подрешёток. Ячейка задаётся опорными полупространствами
// (relevant_facets), расстояния — проекцией (Dykstra), поиск подрешётки —
// CSP-переформулировкой + локальный поиск min-conflicts.

#include <vector>

#include "combigeo/lattice.hpp"
#include "combigeo/linalg.hpp"
#include "combigeo/polytope.hpp"

namespace combigeo {

// Расстояние от точки p до H-многогранника {x : normal_i·x <= offset_i}
// (пересечение полупространств facets). 0, если p внутри. Метод Дейкстры
// (циклическая проекция с коррекциями) — сходится к проекции на пересечение
// выпуклых множеств; не требует вершин. Работает в любой размерности.
double dist_to_halfspaces(const Vec& p, const std::vector<Halfspace>& facets,
                          double tol = 1e-10, int max_iter = 2000);

// Запрещённые векторы решётки: v с D(v) < ell*diam, D(v) = 2*dist(v/2, ячейка).
// Возвращает ЦЕЛОЧИСЛЕННЫЕ координаты v в базисе lat (для CSP). diam передаётся
// (радиус покрытия; для известных решёток точный, иначе оценка сверху).
// facets — опорные полупространства (relevant_facets).
std::vector<std::vector<long>> forbidden_coords(const Lattice& lat,
                                                const std::vector<Halfspace>& facets,
                                                double diam, double ell = 1.0);

// Результат min-conflicts: формы phi (m строк по n коэффициентов) и индекс
// (размер образа = |Z^n / ker phi|). found=false, если не найдено.
struct McResult {
    bool found = false;
    std::vector<std::vector<long>> phi;   // m форм по n коэффициентов
    long index = 0;                       // |образ| = индекс подрешётки ker phi
    long best_killed = -1;                // мин. число «убитых» f по всем рестартам
                                          // (0 ⟺ found); гладкий сигнал для оптимизатора
};

// Локальный поиск min-conflicts валидной подрешётки: ищет гомоморфизм
// phi = (phi_1..phi_m) на группу Z/e_1 x .. x Z/e_m (e_list), у которого
// phi(f) != 0 для ВСЕХ f из F (тогда d >= ell на индексе = |образ|).
// Стартуем со случайного phi, повторно меняем коэффициент, минимизируя число
// "убитых" f (инкрементальные остатки). max_steps шагов на рестарт, restarts
// рестартов. seed — ГСЧ. Возвращает найденный phi и его индекс.
McResult min_conflicts_csp(const std::vector<std::vector<long>>& F,
                           const std::vector<long>& e_list, int n,
                           int max_steps = 3000, int restarts = 20, unsigned seed = 0);

}  // namespace combigeo
