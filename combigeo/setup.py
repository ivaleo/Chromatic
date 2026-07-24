from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# в модуль компилируется ВСЁ ядро + обёртка (прежняя сборка включала только
# wrapper.cpp и давала .so без реализации — см. legacy/)
sources = [
    "src/linalg.cpp",
    "src/lll.cpp",
    "src/lattice.cpp",
    "src/polytope.cpp",
    "src/gjk.cpp",
    "src/sublattice.cpp",
    "src/solver.cpp",
    "src/bigdim.cpp",
    "python/bindings.cpp",
]

ext_modules = [
    Pybind11Extension(
        "combigeo",
        sources,
        include_dirs=["include"],
        cxx_std=17,
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
