cmake_minimum_required(VERSION 3.20)
project(combigeo VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# предупреждения добавляем, не перетирая пользовательские флаги
add_compile_options(-Wall -Wextra)

# --- ядро ---------------------------------------------------------------

add_library(combigeo_core
  src/linalg.cpp
  src/lll.cpp
  src/lattice.cpp
  src/polytope.cpp
  src/gjk.cpp
  src/sublattice.cpp
  src/solver.cpp
  src/bigdim.cpp
)
target_include_directories(combigeo_core PUBLIC include)
# ядро может линковаться в shared python-модуль (combigeo_py): на Linux
# статическая либа без -fPIC не слинкуется в .so — включаем PIC всегда.
set_target_properties(combigeo_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- CLI ------------------------------------------------------------------

add_executable(combigeo_cli cli/main.cpp)
target_link_libraries(combigeo_cli PRIVATE combigeo_core)
set_target_properties(combigeo_cli PROPERTIES OUTPUT_NAME combigeo)

# --- python-модуль (опционально) -------------------------------------------

option(COMBIGEO_BUILD_PYTHON "Собирать python-модуль combigeo" OFF)
if(COMBIGEO_BUILD_PYTHON)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.13.6)
    FetchContent_MakeAvailable(pybind11)
  endif()
  pybind11_add_module(combigeo_py python/bindings.cpp)
  target_link_libraries(combigeo_py PRIVATE combigeo_core)
  set_target_properties(combigeo_py PROPERTIES OUTPUT_NAME combigeo)
endif()

# --- тесты -----------------------------------------------------------------

enable_testing()
foreach(t linalg lll lattice polytope gjk sublattice solver)
  add_executable(test_${t} tests/test_${t}.cpp)
  target_link_libraries(test_${t} PRIVATE combigeo_core)
  add_test(NAME ${t} COMMAND test_${t})
endforeach()
