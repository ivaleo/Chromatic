#include "combigeo/solver.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "combigeo/gjk.hpp"
#include "combigeo/lll.hpp"

namespace combigeo {

namespace {

// Ключ кэша: координаты вектора v, округлённые с фиксированным масштабом.
// Точки v/2 повторяются между подрешётками одной решётки — кэш убирает
// до ~95% повторных вычислений (замерено на старой реализации).
std::vector<long> cache_key(const Vec& v) {
    constexpr double kScale = 1 << 20;
    std::vector<long> key(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) key[i] = std::lround(v[i] * kScale);
    return key;
}

// D(v) = 2 * dist(v/2, V0) — расстояние между ячейками с центрами 0 и v
// (лемма Иванова для центрально-симметричных ячеек).
// Сертификат GJK: при недоказанной оптимальности (error выше допуска) — исключение,
// а не тихо неточное значение (все погрешности смещают d вверх, к ложной пригодности).
double cell_distance(const VoronoiCell& cell, const Vec& v,
                     std::map<std::vector<long>, double>* cache) {
    if (cache) {
        auto it = cache->find(cache_key(v));
        if (it != cache->end()) return it->second;
    }
    const GjkResult r = distance_to_polytope(scaled(v, 0.5), cell.vertices);
    if (r.error > 1e-6 * std::max(1.0, r.distance))
        throw std::runtime_error("cell_distance: GJK не достиг сертификата оптимальности");
    const double d = 2.0 * r.distance;
    if (cache) (*cache)[cache_key(v)] = d;
    return d;
}

// Всё, что зависит только от родительского базиса, вычисляется один раз
// на весь диапазон индексов (ячейка — самая дорогая часть).
struct Prepared {
    double scale = 1.0;   // масштабная нормировка: вход = scale * рабочий базис
    Mat input_scaled;     // входной базис в рабочем масштабе (для transition)
    Lattice lat;          // LLL-приведённая решётка в рабочем масштабе
    VoronoiCell cell;     // ячейка в рабочем масштабе
};

Prepared prepare(const Mat& basis, double lll_delta) {
    const Lattice raw(basis);
    const int n = raw.dim();

    // масштабная нормировка (см. build_voronoi_cell): при экстремальном
    // масштабе базиса абсолютные допуски GJK/LLL/кэша теряют смысл
    const double s_raw = std::pow(raw.det(), 1.0 / n);
    const double s = (s_raw > 1e2 || s_raw < 1e-2) ? s_raw : 1.0;
    Mat scaled_basis = basis;
    if (s != 1.0)
        for (Vec& row : scaled_basis)
            for (double& x : row) x /= s;

    Lattice lat = Lattice(scaled_basis).lll_reduced(lll_delta);
    VoronoiCell cell = build_voronoi_cell(lat);
    return Prepared{s, std::move(scaled_basis), std::move(lat), std::move(cell)};
}

// Поиск по одному индексу на подготовленной решётке; результат в РАБОЧЕМ масштабе
// (пересчёт в исходные единицы — у публичных обёрток).
//
// Параллельность: T потоков, поток t обрабатывает HNF-матрицы с порядковыми
// номерами ≡ t (mod T) (страйд — итератор дёшев по сравнению с обсчётом
// подрешётки). Текущий best разделяется атомарно (стейл-значение лишь ослабляет
// отсечение, не корректность); обновления победителя — под мьютексом; кэш
// расстояний у каждого потока свой.
SolveResult solve_one(const Prepared& prep, long index, const SolveOptions& opts) {
    if (index < 1)
        throw std::invalid_argument("find_optimal: index (число цветов) должен быть >= 1");

    SolveResult result;
    result.index = index;
    result.diameter = prep.cell.diameter;

    const int dim = prep.lat.dim();
    const long total = SublatticeIterator::count(dim, index);

    int T = opts.threads;
    if (T <= 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        T = hw > 2 ? static_cast<int>(hw) - 2 : 1;
    }
    if (static_cast<long>(T) > total) T = static_cast<int>(total > 0 ? total : 1);

    std::atomic<long> examined{0};
    std::atomic<double> best_shared{-1.0};
    std::mutex winner_mu;
    Mat winner_sub_basis;
    Vec winner_witness;
    double winner_d = -1.0;

    auto worker = [&](int tid) {
        std::map<std::vector<long>, double> cache;
        std::map<std::vector<long>, double>* cache_ptr = opts.use_cache ? &cache : nullptr;

        SublatticeIterator it(dim, index);
        HnfMatrix h;
        long ordinal = 0;
        while (it.next(h)) {
            const long my = ordinal++;
            if (my % T != tid) continue;

            const long done = examined.fetch_add(1, std::memory_order_relaxed) + 1;
            const Mat sub_basis = apply_hnf(h, prep.lat.basis());
            const Lattice sub(sub_basis);

            const double snapshot = best_shared.load(std::memory_order_relaxed);
            Vec witness;
            const double d = min_color_distance(prep.cell, sub, cache_ptr, snapshot, &witness);

            if (d > best_shared.load(std::memory_order_relaxed)) {
                std::lock_guard<std::mutex> lk(winner_mu);
                if (d > winner_d) {
                    winner_d = d;
                    best_shared.store(d, std::memory_order_relaxed);
                    winner_sub_basis = sub_basis;
                    winner_witness = std::move(witness);
                }
            }

            if (tid == 0 && opts.progress_every > 0 && done % opts.progress_every == 0) {
                std::fprintf(stderr, "\r[index %ld] %ld/%ld  best D=%.6f", index, done, total,
                             best_shared.load(std::memory_order_relaxed));
                std::fflush(stderr);
            }
            if (opts.on_progress)
                opts.on_progress(done, total, best_shared.load(std::memory_order_relaxed));
        }
    };

    if (T == 1) {
        worker(0);
    } else {
        std::vector<std::thread> pool;
        pool.reserve(static_cast<std::size_t>(T));
        for (int t = 0; t < T; ++t) pool.emplace_back(worker, t);
        for (std::thread& th : pool) th.join();
    }
    result.examined = examined.load();
    if (opts.progress_every > 0) std::fprintf(stderr, "\n");

    if (winner_d >= 0.0) {
        result.best.min_distance = winner_d;
        result.normalized = winner_d / prep.cell.diameter;
        result.best.witness = std::move(winner_witness);
        result.best.sub_basis = lll_reduce(winner_sub_basis, opts.lll_delta);
        // матрица перехода — в координатах базиса ПОЛЬЗОВАТЕЛЯ (входного),
        // а не внутреннего LLL-приведённого кадра
        result.best.transition = transition_in_basis(result.best.sub_basis, prep.input_scaled);
    }
    return result;
}

// пересчёт полей результата из рабочего масштаба в исходные единицы
void rescale_result(SolveResult& r, double s) {
    if (s == 1.0) return;
    r.diameter *= s;
    r.best.min_distance *= s;
    for (Vec& row : r.best.sub_basis)
        for (double& x : row) x *= s;
    for (double& x : r.best.witness) x *= s;
    // r.normalized и r.best.transition масштабно-инвариантны
}

}  // namespace

double min_color_distance(const VoronoiCell& cell, const Lattice& sub,
                          std::map<std::vector<long>, double>* cache, double early_stop,
                          Vec* witness_out) {
    // одно LLL-приведение на подрешётку; все последующие переборы работают
    // с уже приведённым базисом (assume_reduced)
    const Lattice reduced = sub.lll_reduced();

    // стартовая оценка — расстояние для кратчайшего вектора подрешётки
    const Vec v0 = reduced.shortest_vector(/*assume_reduced=*/true);
    double current = cell_distance(cell, v0, cache);
    if (witness_out) *witness_out = v0;
    if (early_stop > 0.0 && current < early_stop) return current;

    // Гарантия полноты: D(v) >= |v| - diam (ячейки лежат в шарах радиуса diam/2),
    // поэтому минимум могут улучшить только v с |v| < current + diam.
    // Граница берётся по НАЧАЛЬНОМУ current и не пересматривается (current
    // дальше только уменьшается, сужая фактическую проверку ниже).
    std::vector<Vec> candidates =
        reduced.vectors_within(current + cell.diameter, /*assume_reduced=*/true);

    // сортировка по длине: короткие чаще дают минимум — раньше срабатывают отсечения
    std::sort(candidates.begin(), candidates.end(),
              [](const Vec& a, const Vec& b) { return norm2(a) < norm2(b); });

    for (const Vec& v : candidates) {
        if (norm(v) - cell.diameter >= current) break;  // дальше только длиннее — не улучшат

        const double d = cell_distance(cell, v, cache);
        if (d < current) {
            current = d;
            if (witness_out) *witness_out = v;
            if (early_stop > 0.0 && current < early_stop) return current;  // branch-and-bound
        }
    }
    return current;
}

SolveResult find_optimal(const Mat& basis, long index, const SolveOptions& opts) {
    const Prepared prep = prepare(basis, opts.lll_delta);
    SolveResult r = solve_one(prep, index, opts);
    rescale_result(r, prep.scale);
    return r;
}

std::map<long, SolveResult> find_optimal_range(const Mat& basis, long from, long to,
                                               const SolveOptions& opts) {
    // ячейка зависит только от родительского базиса — готовится один раз
    // на весь диапазон (кэши расстояний — внутри solve_one, по одному на поток)
    const Prepared prep = prepare(basis, opts.lll_delta);
    std::map<long, SolveResult> results;
    for (long index = from; index <= to; ++index) {
        SolveResult r = solve_one(prep, index, opts);
        rescale_result(r, prep.scale);
        results[index] = std::move(r);
    }
    return results;
}

}  // namespace combigeo
