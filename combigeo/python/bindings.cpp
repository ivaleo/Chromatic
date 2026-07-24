// Python-модуль combigeo (pybind11).
//
// Экспортирует полный конвейер: построение ячейки Вороного, перебор подрешёток,
// расстояния, поиск оптимума. Все результаты возвращаются данными (не печатью).

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>

#include "combigeo/bigdim.hpp"
#include "combigeo/gjk.hpp"
#include "combigeo/lll.hpp"
#include "combigeo/solver.hpp"

namespace py = pybind11;
using namespace combigeo;

namespace {

// ограничение на materialized-список подрешёток (защита от случайного OOM)
constexpr long kMaxSublatticesList = 2'000'000;

std::vector<HnfMatrix> list_sublattices(int dim, long index) {
    const long total = SublatticeIterator::count(dim, index);
    if (total > kMaxSublatticesList)
        throw std::runtime_error("слишком много подрешёток (" + std::to_string(total) +
                                 "); используйте find_optimal");
    std::vector<HnfMatrix> out;
    out.reserve(static_cast<std::size_t>(total));
    SublatticeIterator it(dim, index);
    HnfMatrix h;
    while (it.next(h)) out.push_back(h);
    return out;
}

double min_color_distance_py(const Mat& basis, const Mat& sub_basis) {
    // обе решётки одной размерности (validated_dim проверит квадратность/конечность)
    const int dim = validated_dim(basis, "basis");
    if (validated_dim(sub_basis, "sub_basis") != dim)
        throw std::invalid_argument("min_color_distance: размерности basis и sub_basis различны");

    // масштабная нормировка по родительскому базису (согласованно для обеих решёток);
    // результат возвращается в исходных единицах
    const double s_raw = std::pow(Lattice(basis).det(), 1.0 / dim);
    const double s = (s_raw > 1e2 || s_raw < 1e-2) ? s_raw : 1.0;
    Mat b = basis;
    Mat sb = sub_basis;
    if (s != 1.0) {
        for (auto& row : b)
            for (double& x : row) x /= s;
        for (auto& row : sb)
            for (double& x : row) x /= s;
    }
    const Lattice lat = Lattice(b).lll_reduced();
    const VoronoiCell cell = build_voronoi_cell(lat);
    return s * min_color_distance(cell, Lattice(sb));
}

Mat validated_basis_for_lll(const Mat& basis) {
    validated_dim(basis, "lll_reduce: базис");
    return basis;
}

}  // namespace

PYBIND11_MODULE(combigeo, m) {
    m.doc() =
        "Периодические раскраски R^n по решётчатым разбиениям Вороного "
        "(методология Л.Л. Иванова). Все функции принимают базис решётки "
        "как список строк-векторов.";

    py::class_<Halfspace>(m, "Halfspace")
        .def_readonly("normal", &Halfspace::normal)
        .def_readonly("offset", &Halfspace::offset)
        .def_readonly("lattice_vector", &Halfspace::lattice_vector);

    py::class_<VoronoiCell>(m, "VoronoiCell")
        .def_readonly("dim", &VoronoiCell::dim)
        .def_readonly("vertices", &VoronoiCell::vertices)
        .def_readonly("facets", &VoronoiCell::facets)
        .def_readonly("diameter", &VoronoiCell::diameter)
        .def_readonly("f_vector", &VoronoiCell::f_vector)
        .def_readonly("vertex_facets", &VoronoiCell::vertex_facets)
        .def("contains", &VoronoiCell::contains, py::arg("p"), py::arg("tol") = 1e-9)
        .def("__repr__", [](const VoronoiCell& c) {
            std::string f;
            for (std::size_t i = 0; i < c.f_vector.size(); ++i)
                f += (i ? ", " : "") + std::to_string(c.f_vector[i]);
            return "<VoronoiCell dim=" + std::to_string(c.dim) + " f=[" + f +
                   "] diam=" + std::to_string(c.diameter) + ">";
        });

    py::class_<SublatticeResult>(m, "SublatticeResult")
        .def_readonly("transition", &SublatticeResult::transition)
        .def_readonly("sub_basis", &SublatticeResult::sub_basis)
        .def_readonly("min_distance", &SublatticeResult::min_distance)
        .def_readonly("witness", &SublatticeResult::witness);

    py::class_<SolveResult>(m, "SolveResult")
        .def_readonly("index", &SolveResult::index)
        .def_readonly("diameter", &SolveResult::diameter)
        .def_readonly("best", &SolveResult::best)
        .def_readonly("normalized", &SolveResult::normalized)
        .def_readonly("examined", &SolveResult::examined)
        .def("__repr__", [](const SolveResult& r) {
            return "<SolveResult index=" + std::to_string(r.index) +
                   " d=" + std::to_string(r.normalized) +
                   " D=" + std::to_string(r.best.min_distance) + ">";
        });

    m.attr("__version__") = "1.1.0";

    m.def(
        "lll_reduce",
        [](const Mat& basis, double delta) { return lll_reduce(validated_basis_for_lll(basis), delta); },
        py::arg("basis"), py::arg("delta") = 0.75,
        "LLL-приведение базиса решётки (вход валидируется: квадратность, конечность).");

    m.def(
        "shortest_vector",
        [](const Mat& basis) { return Lattice(basis).shortest_vector(); }, py::arg("basis"),
        "Кратчайший ненулевой вектор решётки.");

    m.def(
        "voronoi_cell",
        [](const Mat& basis, int window) {
            // тяжёлое построение (для Z4 — >1.5M решений СЛАУ) — отпускаем GIL
            py::gil_scoped_release release;
            return build_voronoi_cell(Lattice(basis).lll_reduced(), window);
        },
        py::arg("basis"), py::arg("window") = 3,
        "Ячейка Вороного центральной точки решётки: вершины, фасеты, f-вектор, диаметр.");

    m.def(
        "distance_to_cell",
        [](const Vec& point, const VoronoiCell& cell) {
            validate_vector(point, cell.dim, "point");
            // масштабная нормировка по диаметру ячейки: допуски GJK абсолютные
            const double s = (cell.diameter > 1e2 || cell.diameter < 1e-2) && cell.diameter > 0.0
                                 ? cell.diameter
                                 : 1.0;
            if (s == 1.0) return distance_to_polytope(point, cell.vertices).distance;
            Vec p = point;
            for (double& x : p) x /= s;
            std::vector<Vec> verts = cell.vertices;
            for (Vec& w : verts)
                for (double& x : w) x /= s;
            return s * distance_to_polytope(p, verts).distance;
        },
        py::arg("point"), py::arg("cell"), "Расстояние от точки до ячейки (GJK).");

    m.def(
        "_vectors_near",
        [](const Mat& basis, const Vec& target, double bound, bool assume_reduced) {
            return Lattice(basis).vectors_near(target, bound, assume_reduced);
        },
        py::arg("basis"), py::arg("target"), py::arg("bound"),
        py::arg("assume_reduced") = false,
        "(отладочное) все векторы решётки в шаре радиуса bound вокруг target.");

    m.def(
        "_babai_distance",
        [](const Mat& basis, const Vec& target) { return Lattice(basis).babai_distance(target); },
        py::arg("basis"), py::arg("target"),
        "(отладочное) расстояние до точки Бабаи.");

    m.def("count_sublattices", &SublatticeIterator::count, py::arg("dim"), py::arg("index"),
          "Количество подрешёток данного индекса (HNF-матриц).");

    m.def("sublattices", &list_sublattices, py::arg("dim"), py::arg("index"),
          "Все HNF-матрицы перехода данного индекса (список).");

    m.def("min_color_distance", &min_color_distance_py, py::arg("basis"), py::arg("sub_basis"),
          py::call_guard<py::gil_scoped_release>(),
          "Минимальное расстояние между одноцветными ячейками для подрешётки.");

    m.def(
        "find_optimal",
        [](const Mat& basis, long index, int progress_every, bool use_cache, int threads) {
            SolveOptions opts;
            opts.progress_every = progress_every;
            opts.use_cache = use_cache;
            opts.threads = threads;
            return find_optimal(basis, index, opts);
        },
        py::arg("basis"), py::arg("index"), py::arg("progress_every") = 0,
        py::arg("use_cache") = true, py::arg("threads") = 0,
        py::call_guard<py::gil_scoped_release>(),
        "Поиск оптимальной подрешётки индекса index (максимум min-расстояния). "
        "threads: 0 = авто (ядра-2), 1 = однопоточно. "
        "Пригодная раскраска: result.normalized >= 1.");

    m.def(
        "find_optimal_range",
        [](const Mat& basis, long frm, long to, int progress_every, bool use_cache,
           int threads) {
            SolveOptions opts;
            opts.progress_every = progress_every;
            opts.use_cache = use_cache;
            opts.threads = threads;
            return find_optimal_range(basis, frm, to, opts);
        },
        py::arg("basis"), py::arg("frm"), py::arg("to"), py::arg("progress_every") = 0,
        py::arg("use_cache") = true, py::arg("threads") = 0,
        py::call_guard<py::gil_scoped_release>(),
        "Поиск по диапазону индексов [frm, to]; ячейка общая на весь диапазон, "
        "кэш расстояний — по одному на поток; threads: 0 = авто. "
        "Возвращает {index: SolveResult}.");

    // ---- инструменты для больших размерностей (n >= 5) ----

    m.def(
        "relevant_facets",
        [](const Mat& basis) {
            py::gil_scoped_release rel;
            std::vector<std::pair<Vec, double>> out;
            for (const Halfspace& h : relevant_facets(Lattice(basis)))
                out.emplace_back(h.lattice_vector, h.offset);
            return out;   // (вектор решётки v, offset=|v|^2/2 нормировки нет — offset=|v|/2)
        },
        py::arg("basis"),
        "Релевантные фасеты ячейки (опорные полупространства) БЕЗ перечисления "
        "вершин — для больших размерностей. Возвращает список (lattice_vector, offset).");

    m.def(
        "forbidden_coords",
        [](const Mat& basis, double diam, double ell) {
            py::gil_scoped_release rel;
            const Lattice lat(basis);
            const auto facets = relevant_facets(lat);
            return forbidden_coords(lat, facets, diam, ell);
        },
        py::arg("basis"), py::arg("diam"), py::arg("ell") = 1.0,
        "Целочисленные координаты запрещённых векторов (D(v) < ell*diam) через "
        "опорные полупространства (быстро, любая размерность).");

    m.def(
        "min_conflicts",
        [](const std::vector<std::vector<long>>& F, const std::vector<long>& e_list, int n,
           int max_steps, int restarts, unsigned seed) {
            McResult r;
            {
                // GIL отпускаем ТОЛЬКО на счёт; make_tuple ниже создаёт python-объекты
                // и требует удержания GIL (иначе segfault).
                py::gil_scoped_release rel;
                r = min_conflicts_csp(F, e_list, n, max_steps, restarts, seed);
            }
            return py::make_tuple(r.found, r.phi, r.index);
        },
        py::arg("F"), py::arg("e_list"), py::arg("n"), py::arg("max_steps") = 3000,
        py::arg("restarts") = 20, py::arg("seed") = 0u,
        "Локальный поиск min-conflicts валидной подрешётки (CSP): формы phi над "
        "Z/e_1 x..x Z/e_m, phi(f)!=0 для всех f. Возвращает (found, phi, index).");

    m.def(
        "min_conflicts_cost",
        [](const std::vector<std::vector<long>>& F, const std::vector<long>& e_list, int n,
           int max_steps, int restarts, unsigned seed) {
            McResult r;
            {
                py::gil_scoped_release rel;
                r = min_conflicts_csp(F, e_list, n, max_steps, restarts, seed);
            }
            return py::make_tuple(r.found, r.best_killed, r.index);
        },
        py::arg("F"), py::arg("e_list"), py::arg("n"), py::arg("max_steps") = 3000,
        py::arg("restarts") = 20, py::arg("seed") = 0u,
        "Как min_conflicts, но возвращает (found, best_killed, index): best_killed "
        "— минимум числа неотделённых f по всем рестартам (0 ⟺ found). Гладкий сигнал "
        "для оптимизации по формам Грама.");

    m.def(
        "dist_to_halfspaces",
        [](const Vec& p, const std::vector<std::pair<Vec, double>>& facets) {
            std::vector<Halfspace> hs;
            for (const auto& [v, off] : facets) {
                Halfspace h;
                const double len = norm(v);
                h.normal = scaled(v, 1.0 / len);
                h.offset = off;
                hs.push_back(std::move(h));
            }
            return dist_to_halfspaces(p, hs);
        },
        py::arg("point"), py::arg("facets"),
        "Расстояние от точки до H-многогранника (проекция Дейкстры).");
}
