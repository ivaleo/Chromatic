#pragma once
// Минимальный тестовый каркас (без внешних зависимостей).
// Использование:
//   TEST(имя) { CHECK(условие); CHECK_NEAR(a, b, 1e-9); }
//   int main() { return run_all_tests(); }

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace testfw {

struct Registry {
    static Registry& instance() {
        static Registry r;
        return r;
    }
    std::vector<std::pair<std::string, std::function<void()>>> tests;
    int failures = 0;
    std::string current;
};

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) {
        Registry::instance().tests.emplace_back(name, std::move(fn));
    }
};

inline void report_failure(const char* expr, const char* file, int line) {
    ++Registry::instance().failures;
    std::printf("  FAIL [%s] %s (%s:%d)\n", Registry::instance().current.c_str(), expr, file,
                line);
}

}  // namespace testfw

#define TEST(name)                                                       \
    static void test_##name();                                           \
    static testfw::Registrar registrar_##name(#name, test_##name);       \
    static void test_##name()

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) testfw::report_failure(#cond, __FILE__, __LINE__);  \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                          \
    do {                                                                               \
        const double va_ = (a), vb_ = (b);                                             \
        if (!(std::fabs(va_ - vb_) <= (tol))) {                                        \
            char buf_[256];                                                            \
            std::snprintf(buf_, sizeof buf_, "%s ~ %s (%.12g vs %.12g)", #a, #b, va_,  \
                          vb_);                                                        \
            testfw::report_failure(buf_, __FILE__, __LINE__);                          \
        }                                                                              \
    } while (0)

inline int run_all_tests() {
    auto& reg = testfw::Registry::instance();
    for (auto& [name, fn] : reg.tests) {
        reg.current = name;
        const int before = reg.failures;
        fn();
        std::printf("%s %s\n", reg.failures == before ? "PASS" : "FAIL", name.c_str());
    }
    std::printf("%zu tests, %d failures\n", reg.tests.size(), reg.failures);
    return reg.failures == 0 ? 0 : 1;
}
