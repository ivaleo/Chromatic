// Exact projective finite-field threshold enumerator.
//
// Input on stdin:
//   prime dimension constraint_count
//   c_11 ... c_1n
//   ...
//
// Every projective row is visited once in the charts
//   a_0=...=a_{j-1}=0, a_j=1.
// A row is feasible iff its dot product with every input constraint is
// nonzero modulo prime.  The output is one compact JSON object.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using Row = std::vector<int>;

struct Search {
  int prime = 0;
  int dimension = 0;
  std::vector<Row> constraints;
  std::atomic<bool> found{false};
  std::atomic<std::uint64_t> tested{0};
  std::mutex result_mutex;
  Row result;

  bool feasible(const Row& row) const {
    for (const Row& constraint : constraints) {
      int dot = 0;
      for (int coordinate = 0; coordinate < dimension; ++coordinate) {
        dot += constraint[coordinate] * row[coordinate];
      }
      if (dot % prime == 0) {
        return false;
      }
    }
    return true;
  }

  void visit_leaf(const Row& row, std::uint64_t& local_tested) {
    if (found.load(std::memory_order_relaxed)) {
      return;
    }
    ++local_tested;
    if (!feasible(row)) {
      return;
    }
    bool expected = false;
    if (found.compare_exchange_strong(
            expected, true, std::memory_order_relaxed)) {
      std::lock_guard<std::mutex> lock(result_mutex);
      result = row;
    }
  }

  void enumerate_tail(
      int coordinate, Row& row, std::uint64_t& local_tested) {
    if (found.load(std::memory_order_relaxed)) {
      return;
    }
    if (coordinate == dimension) {
      visit_leaf(row, local_tested);
      return;
    }
    for (int value = 0; value < prime; ++value) {
      row[coordinate] = value;
      enumerate_tail(coordinate + 1, row, local_tested);
      if (found.load(std::memory_order_relaxed)) {
        return;
      }
    }
  }

  bool run_chart(int pivot, int requested_threads) {
    Row base(dimension, 0);
    base[pivot] = 1;
    const int first_free = pivot + 1;
    if (first_free == dimension) {
      std::uint64_t local_tested = 0;
      visit_leaf(base, local_tested);
      tested.fetch_add(local_tested, std::memory_order_relaxed);
      return found.load(std::memory_order_relaxed);
    }

    std::atomic<int> next_first_value{0};
    const int thread_count = std::max(
        1, std::min({requested_threads, prime,
                     static_cast<int>(std::thread::hardware_concurrency())}));
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (int thread_index = 0; thread_index < thread_count; ++thread_index) {
      threads.emplace_back([&, base]() mutable {
        std::uint64_t local_tested = 0;
        while (!found.load(std::memory_order_relaxed)) {
          const int value = next_first_value.fetch_add(
              1, std::memory_order_relaxed);
          if (value >= prime) {
            break;
          }
          base[first_free] = value;
          enumerate_tail(first_free + 1, base, local_tested);
        }
        tested.fetch_add(local_tested, std::memory_order_relaxed);
      });
    }
    for (std::thread& worker : threads) {
      worker.join();
    }
    return found.load(std::memory_order_relaxed);
  }
};

}  // namespace

int main(int argc, char** argv) {
  int requested_threads = 1;
  if (argc >= 2) {
    requested_threads = std::max(1, std::atoi(argv[1]));
  }

  Search search;
  int constraint_count = 0;
  if (!(std::cin >> search.prime >> search.dimension >> constraint_count)) {
    std::cerr << "failed to read problem header\n";
    return 2;
  }
  if (search.prime < 2 || search.dimension < 1 || constraint_count < 0) {
    std::cerr << "invalid problem dimensions\n";
    return 2;
  }
  search.constraints.assign(
      constraint_count, Row(search.dimension, 0));
  for (Row& constraint : search.constraints) {
    bool nonzero = false;
    for (int& coefficient : constraint) {
      if (!(std::cin >> coefficient)) {
        std::cerr << "failed to read constraint matrix\n";
        return 2;
      }
      coefficient %= search.prime;
      if (coefficient < 0) {
        coefficient += search.prime;
      }
      nonzero = nonzero || coefficient != 0;
    }
    if (!nonzero) {
      std::cout << "{\"status\":\"INFEASIBLE\","
                   "\"reason\":\"zero_constraint\",\"tested\":0}\n";
      return 0;
    }
  }

  const auto started = std::chrono::steady_clock::now();
  for (int pivot = 0; pivot < search.dimension; ++pivot) {
    if (search.run_chart(pivot, requested_threads)) {
      const double elapsed = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started).count();
      std::cout << "{\"status\":\"FEASIBLE\",\"pivot\":" << pivot
                << ",\"tested\":" << search.tested.load()
                << ",\"elapsed_seconds\":" << elapsed << ",\"row\":[";
      for (int coordinate = 0; coordinate < search.dimension; ++coordinate) {
        if (coordinate) {
          std::cout << ',';
        }
        std::cout << search.result[coordinate];
      }
      std::cout << "]}\n";
      return 0;
    }
  }

  const double elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  std::cout << "{\"status\":\"INFEASIBLE\",\"tested\":"
            << search.tested.load()
            << ",\"elapsed_seconds\":" << elapsed << "}\n";
  return 0;
}
