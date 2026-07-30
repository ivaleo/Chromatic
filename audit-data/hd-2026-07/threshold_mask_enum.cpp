// Exact tuple search over precomputed conflict bitmasks.
//
// Input on stdin:
//   block_count word_count
//   option_count(block 0)
//   mask_0_word_0 ... mask_0_word_(word_count-1)
//   ...
//   option_count(block 1)
//   ...
//
// An option mask marks the bad constraints killed by that block.  A tuple is
// feasible exactly when the intersection of its masks is empty.  Every tuple
// is examined once unless a feasible tuple permits early termination.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using Mask = std::vector<std::uint64_t>;

struct Search {
  std::vector<std::vector<Mask>> blocks;
  int word_count = 0;
  std::atomic<bool> found{false};
  std::atomic<std::uint64_t> tested{0};
  std::mutex result_mutex;
  std::vector<int> result;

  bool intersect(
      const Mask& left, const Mask& right, Mask& output) const {
    bool any = false;
    for (int word = 0; word < word_count; ++word) {
      output[word] = left[word] & right[word];
      any = any || output[word] != 0;
    }
    return any;
  }

  void save_result(const std::vector<int>& choices) {
    bool expected = false;
    if (found.compare_exchange_strong(
            expected, true, std::memory_order_relaxed)) {
      std::lock_guard<std::mutex> lock(result_mutex);
      result = choices;
    }
  }

  void recurse(
      int block,
      const Mask& active,
      std::vector<int>& choices,
      std::uint64_t& local_tested) {
    if (found.load(std::memory_order_relaxed)) {
      return;
    }
    if (block == static_cast<int>(blocks.size()) - 1) {
      Mask intersection(word_count, 0);
      for (int option = 0;
           option < static_cast<int>(blocks[block].size());
           ++option) {
        ++local_tested;
        choices[block] = option;
        if (!intersect(active, blocks[block][option], intersection)) {
          save_result(choices);
          return;
        }
      }
      return;
    }

    Mask intersection(word_count, 0);
    for (int option = 0;
         option < static_cast<int>(blocks[block].size());
         ++option) {
      choices[block] = option;
      if (!intersect(active, blocks[block][option], intersection)) {
        for (int remaining = block + 1;
             remaining < static_cast<int>(blocks.size());
             ++remaining) {
          choices[remaining] = 0;
        }
        save_result(choices);
        return;
      }
      recurse(block + 1, intersection, choices, local_tested);
      if (found.load(std::memory_order_relaxed)) {
        return;
      }
    }
  }
};

}  // namespace

int main(int argc, char** argv) {
  int requested_threads = 1;
  if (argc >= 2) {
    requested_threads = std::max(1, std::atoi(argv[1]));
  }
  const bool binary_input = argc >= 3 && std::string(argv[2]) == "binary";

  Search search;
  int block_count = 0;
  if (binary_input) {
    std::uint64_t header[2] = {0, 0};
    if (!std::cin.read(
            reinterpret_cast<char*>(header), sizeof(header))) {
      std::cerr << "failed to read binary problem header\n";
      return 2;
    }
    block_count = static_cast<int>(header[0]);
    search.word_count = static_cast<int>(header[1]);
  } else {
    if (!(std::cin >> block_count >> search.word_count)) {
      std::cerr << "failed to read problem header\n";
      return 2;
    }
  }
  if (block_count < 2 || search.word_count < 1) {
    std::cerr << "invalid block or mask size\n";
    return 2;
  }
  search.blocks.resize(block_count);
  std::uint64_t total_tuples = 1;
  for (auto& block : search.blocks) {
    int option_count = 0;
    if (binary_input) {
      std::uint64_t encoded_count = 0;
      if (!std::cin.read(
              reinterpret_cast<char*>(&encoded_count),
              sizeof(encoded_count))) {
        std::cerr << "failed to read binary option count\n";
        return 2;
      }
      option_count = static_cast<int>(encoded_count);
    } else if (!(std::cin >> option_count)) {
      std::cerr << "failed to read option count\n";
      return 2;
    }
    if (option_count < 1) {
      std::cerr << "invalid option count\n";
      return 2;
    }
    if (total_tuples > UINT64_MAX / static_cast<std::uint64_t>(option_count)) {
      std::cerr << "tuple count overflows uint64\n";
      return 2;
    }
    total_tuples *= static_cast<std::uint64_t>(option_count);
    block.assign(option_count, Mask(search.word_count, 0));
    for (Mask& mask : block) {
      if (binary_input) {
        if (!std::cin.read(
                reinterpret_cast<char*>(mask.data()),
                static_cast<std::streamsize>(
                    sizeof(std::uint64_t) * mask.size()))) {
          std::cerr << "failed to read binary option masks\n";
          return 2;
        }
      } else {
        for (std::uint64_t& word : mask) {
          if (!(std::cin >> word)) {
            std::cerr << "failed to read option masks\n";
            return 2;
          }
        }
      }
    }
  }

  const auto started = std::chrono::steady_clock::now();
  std::atomic<int> next_first{0};
  const int thread_count = std::max(
      1, std::min({requested_threads,
                   static_cast<int>(search.blocks[0].size()),
                   static_cast<int>(std::thread::hardware_concurrency())}));
  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (int thread_index = 0; thread_index < thread_count; ++thread_index) {
    threads.emplace_back([&]() {
      std::uint64_t local_tested = 0;
      std::vector<int> choices(block_count, 0);
      while (!search.found.load(std::memory_order_relaxed)) {
        const int first = next_first.fetch_add(1, std::memory_order_relaxed);
        if (first >= static_cast<int>(search.blocks[0].size())) {
          break;
        }
        choices[0] = first;
        search.recurse(
            1, search.blocks[0][first], choices, local_tested);
      }
      search.tested.fetch_add(local_tested, std::memory_order_relaxed);
    });
  }
  for (std::thread& worker : threads) {
    worker.join();
  }

  const double elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  if (search.found.load(std::memory_order_relaxed)) {
    std::cout << "{\"status\":\"FEASIBLE\",\"tested\":"
              << search.tested.load()
              << ",\"total_tuples\":" << total_tuples
              << ",\"elapsed_seconds\":" << elapsed
              << ",\"choices\":[";
    for (int block = 0; block < block_count; ++block) {
      if (block) {
        std::cout << ',';
      }
      std::cout << search.result[block];
    }
    std::cout << "]}\n";
    return 0;
  }

  if (search.tested.load() != total_tuples) {
    std::cerr << "internal error: incomplete UNSAT enumeration\n";
    return 3;
  }
  std::cout << "{\"status\":\"INFEASIBLE\",\"tested\":"
            << search.tested.load()
            << ",\"total_tuples\":" << total_tuples
            << ",\"elapsed_seconds\":" << elapsed << "}\n";
  return 0;
}
