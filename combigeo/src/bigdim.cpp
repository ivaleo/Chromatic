#include "combigeo/bigdim.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <set>

namespace combigeo {

// ---------- расстояние до пересечения полупространств (Дейкстра) ----------

double dist_to_halfspaces(const Vec& p, const std::vector<Halfspace>& facets, double tol,
                          int max_iter) {
    // внутри?
    bool inside = true;
    for (const Halfspace& h : facets)
        if (dot(p, h.normal) > h.offset + tol) { inside = false; break; }
    if (inside) return 0.0;

    const std::size_t m = facets.size();
    Vec x = p;
    std::vector<Vec> corr(m, zeros(p.size()));   // коррекции Дейкстры
    for (int it = 0; it < max_iter; ++it) {
        double max_move = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            // z = x + corr[i]; проекция z на полупространство i
            Vec z = add(x, corr[i]);
            const double slack = dot(z, facets[i].normal) - facets[i].offset;
            Vec proj = z;
            if (slack > 0.0)                       // нормаль единичная => проекция проста
                proj = sub(z, scaled(facets[i].normal, slack));
            // новая коррекция и сдвиг x
            Vec new_corr = sub(z, proj);
            double move = 0.0;
            for (std::size_t d = 0; d < x.size(); ++d) move += std::abs(proj[d] - x[d]);
            max_move = std::max(max_move, move);
            corr[i] = std::move(new_corr);
            x = std::move(proj);
        }
        if (max_move < tol) break;
    }
    return std::sqrt(dist2(p, x));
}

// ---------- запрещённое множество ----------

std::vector<std::vector<long>> forbidden_coords(const Lattice& lat,
                                                const std::vector<Halfspace>& facets,
                                                double diam, double ell) {
    const int n = lat.dim();
    // короткие векторы: D(v) >= |v| - diam, поэтому D < ell*diam => |v| < (ell+1)diam
    const double R = (ell + 1.0) * diam + 1e-9;
    const std::vector<Vec> shorts = lat.vectors_within(R);

    // обратный базис для перевода в целочисленные координаты
    // v = c * basis  =>  c = v * basis^{-1}; решаем basis^T c^T = v^T покомпонентно
    const Mat& B = lat.basis();
    Mat Bt(n, Vec(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) Bt[i][j] = B[j][i];

    std::vector<std::vector<long>> F;
    for (const Vec& v : shorts) {
        const Vec half = scaled(v, 0.5);
        const double D = 2.0 * dist_to_halfspaces(half, facets);
        if (D < ell * diam - 1e-9) {
            Vec c;
            if (!solve_linear(Bt, v, c)) continue;
            std::vector<long> ci(n);
            bool ok = true;
            for (int i = 0; i < n; ++i) {
                const double r = std::round(c[i]);
                if (std::abs(c[i] - r) > 1e-6) { ok = false; break; }
                ci[i] = static_cast<long>(r);
            }
            if (ok) F.push_back(std::move(ci));
        }
    }
    return F;
}

// ---------- min-conflicts CSP ----------

namespace {

// приведение к неотрицательному остатку
inline long mod(long a, long e) { long r = a % e; return r < 0 ? r + e : r; }

// индекс = размер образа phi в Z/e_1 x .. x Z/e_m (перебор подгруппы, |G| <= k)
long image_size(const std::vector<std::vector<long>>& phi, const std::vector<long>& e_list,
                int n) {
    const int m = static_cast<int>(e_list.size());
    // образ порождён столбцами g_i = (phi_0[i]..phi_{m-1}[i])
    std::vector<std::vector<long>> gens(n, std::vector<long>(m));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) gens[i][j] = mod(phi[j][i], e_list[j]);
    std::set<std::vector<long>> grp;
    std::vector<long> zero(m, 0);
    grp.insert(zero);
    std::vector<std::vector<long>> frontier{zero};
    while (!frontier.empty()) {
        std::vector<long> x = std::move(frontier.back());
        frontier.pop_back();
        for (const std::vector<long>& g : gens) {
            std::vector<long> y(m);
            for (int j = 0; j < m; ++j) y[j] = mod(x[j] + g[j], e_list[j]);
            if (grp.insert(y).second) frontier.push_back(y);
        }
    }
    return static_cast<long>(grp.size());
}

}  // namespace

McResult min_conflicts_csp(const std::vector<std::vector<long>>& F,
                           const std::vector<long>& e_list, int n, int max_steps, int restarts,
                           unsigned seed) {
    const int m = static_cast<int>(e_list.size());
    const std::size_t nf = F.size();
    std::mt19937_64 rng(seed);
    long best_killed = static_cast<long>(nf) + 1;   // глобальный минимум по рестартам

    // предвычислим F по столбцам как long
    // res[j][f] = (phi_j . F[f]) mod e_j  — держим инкрементально
    for (int restart = 0; restart < restarts; ++restart) {
        std::vector<std::vector<long>> phi(m, std::vector<long>(n));
        for (int j = 0; j < m; ++j)
            for (int i = 0; i < n; ++i)
                phi[j][i] = static_cast<long>(rng() % static_cast<unsigned long long>(e_list[j]));

        // остатки и число убитых
        std::vector<std::vector<long>> res(m, std::vector<long>(nf, 0));
        for (int j = 0; j < m; ++j)
            for (std::size_t f = 0; f < nf; ++f) {
                long s = 0;
                for (int i = 0; i < n; ++i) s += phi[j][i] * F[f][i];
                res[j][f] = mod(s, e_list[j]);
            }
        auto killed_count = [&]() {
            long cnt = 0;
            for (std::size_t f = 0; f < nf; ++f) {
                bool k = true;
                for (int j = 0; j < m; ++j)
                    if (res[j][f] != 0) { k = false; break; }
                if (k) ++cnt;
            }
            return cnt;
        };
        long nk = killed_count();
        if (nk < best_killed) best_killed = nk;

        for (int step = 0; step < max_steps; ++step) {
            if (nk == 0) {
                McResult r;
                r.found = true;
                r.phi = phi;
                r.index = image_size(phi, e_list, n);
                r.best_killed = 0;
                return r;
            }
            // выбираем случайный убитый вектор
            std::vector<std::size_t> killed_idx;
            for (std::size_t f = 0; f < nf; ++f) {
                bool k = true;
                for (int j = 0; j < m; ++j)
                    if (res[j][f] != 0) { k = false; break; }
                if (k) killed_idx.push_back(f);
            }
            const std::size_t fsel = killed_idx[rng() % killed_idx.size()];

            // ход должен ОЖИВИТЬ fsel: сменить phi[j][i] (F[fsel][i] != 0) так,
            // чтобы res_j[fsel] стал != 0; среди таких выбираем минимизирующий
            // общее число убитых.
            int best_j = -1, best_i = -1;
            long best_val = 0, best_after = nk + 1;
            const int tries = 40;
            for (int t = 0; t < tries; ++t) {
                const int j = static_cast<int>(rng() % static_cast<unsigned>(m));
                const int i = static_cast<int>(rng() % static_cast<unsigned>(n));
                const long e = e_list[j];
                if (mod(F[fsel][i], e) == 0) continue;   // не влияет на res_j[fsel]
                const long val = static_cast<long>(rng() % static_cast<unsigned long long>(e));
                const long delta = val - phi[j][i];
                if (delta == 0) continue;
                if (mod(res[j][fsel] + delta * F[fsel][i], e) == 0) continue;  // fsel всё ещё убит по j
                // считаем новых убитых при этом ходе (инкрементально, с откатом)
                long after = 0;
                for (std::size_t f = 0; f < nf; ++f) {
                    long rj = mod(res[j][f] + delta * F[f][i], e);
                    bool k = (rj == 0);
                    if (k)
                        for (int jj = 0; jj < m && k; ++jj)
                            if (jj != j && res[jj][f] != 0) k = false;
                    if (k) ++after;
                }
                if (after < best_after) { best_after = after; best_j = j; best_i = i; best_val = val; }
                if (after == 0) break;
            }

            // шум: с вероятностью 0.2 случайный ход
            bool noise = (rng() % 5 == 0);
            if (best_j < 0 || noise) {
                best_j = static_cast<int>(rng() % static_cast<unsigned>(m));
                best_i = static_cast<int>(rng() % static_cast<unsigned>(n));
                best_val = static_cast<long>(rng() % static_cast<unsigned long long>(e_list[best_j]));
            }
            // применяем ход
            const long e = e_list[best_j];
            const long delta = best_val - phi[best_j][best_i];
            if (delta != 0) {
                phi[best_j][best_i] = best_val;
                for (std::size_t f = 0; f < nf; ++f)
                    res[best_j][f] = mod(res[best_j][f] + delta * F[f][best_i], e);
                nk = killed_count();
                if (nk < best_killed) best_killed = nk;
            }
        }
    }
    McResult r;
    r.best_killed = best_killed;
    return r;
}

}  // namespace combigeo
