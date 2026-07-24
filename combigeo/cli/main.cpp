// CLI: поиск оптимальных подрешёток для периодической раскраски.
//
// Использование:
//   combigeo <решётка> <index_from> [index_to]
//   решётка: Z2 | A2 | Z3 | FCC | BCC | Z4 | D4
//
// Пример: combigeo D4 49 54

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <string>

#include "combigeo/solver.hpp"

using namespace combigeo;

namespace {

Mat preset_basis(const std::string& name) {
    static const std::map<std::string, Mat> presets = {
        {"Z2", {{1, 0}, {0, 1}}},
        {"A2", {{1.0, 0.0}, {0.5, 0.8660254037844386}}},  // гексагональная, sqrt(3)/2
        {"Z3", {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
        {"FCC", {{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}},
        {"BCC", {{1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}}},
        {"Z4", {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}},
        {"D4", {{2, 0, 0, 0}, {1, 1, 0, 0}, {1, 0, 1, 0}, {1, 0, 0, 1}}},
    };
    auto it = presets.find(name);
    if (it == presets.end()) {
        std::fprintf(stderr, "Неизвестная решётка '%s' (доступны: Z2 A2 Z3 FCC BCC Z4 D4)\n",
                     name.c_str());
        std::exit(2);
    }
    return it->second;
}

void print_result(const SolveResult& r) {
    std::printf("index=%ld  diam=%.6f  examined=%ld\n", r.index, r.diameter, r.examined);
    if (r.best.min_distance <= 0) {
        std::printf("  оптимум D=0 (одноцветные ячейки касаются при любой подрешётке)\n");
        return;
    }
    std::printf("  D=%.6f  d=D/diam=%.6f %s\n", r.best.min_distance, r.normalized,
                r.normalized >= 1.0 ? "(раскраска пригодна)" : "");
    std::printf("  матрица перехода (HNF, в базисе пресета):\n");
    for (const auto& row : r.best.transition) {
        std::printf("   ");
        for (long x : row) std::printf(" %3ld", x);
        std::printf("\n");
    }
    std::printf("  базис подрешётки (объемлющие координаты, после LLL):\n");
    for (const auto& row : r.best.sub_basis) {
        std::printf("   ");
        for (double x : row) std::printf(" %10.6f", x);
        std::printf("\n");
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "Использование: %s <Z2|A2|Z3|FCC|BCC|Z4|D4> <index_from> [index_to]\n",
                     argv[0]);
        return 2;
    }

    const Mat basis = preset_basis(argv[1]);

    auto parse_index = [](const char* s) -> long {
        char* end = nullptr;
        const long v = std::strtol(s, &end, 10);
        if (end == s || *end != '\0' || v < 1) {
            std::fprintf(stderr, "Некорректный индекс '%s' (ожидается целое >= 1)\n", s);
            std::exit(2);
        }
        return v;
    };

    const long from = parse_index(argv[2]);
    const long to = (argc >= 4) ? parse_index(argv[3]) : from;
    if (to < from) {
        std::fprintf(stderr, "index_to (%ld) меньше index_from (%ld)\n", to, from);
        return 2;
    }

    SolveOptions opts;
    opts.progress_every = 500;

    // find_optimal_range может бросить std::exception: при большом индексе
    // SublatticeIterator::count бросает std::overflow_error (число подрешёток
    // не помещается в long), а вырожденный/некорректный базис или index < 1 —
    // std::invalid_argument. Печатаем понятную ошибку и выходим с кодом 1
    // (без try/catch это привело бы к std::terminate / SIGABRT).
    try {
        auto results = find_optimal_range(basis, from, to, opts);
        for (const auto& [index, r] : results) print_result(r);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Ошибка: %s\n", e.what());
        return 1;
    }
    return 0;
}
