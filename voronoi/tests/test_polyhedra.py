"""Тесты построения геометрии Вороного (polyhedra.py)."""

import numpy as np

from voronoi4d import Face3D


def test_voronoi_build_integrity(vor):
    assert len(vor.coords4) > 0, "Координаты не сгенерированы"
    assert vor.central_region_index is not None, "Центральный регион не найден"
    assert len(vor.faces_3d) > 0, "3D грани не найдены"
    assert len(vor.polyhedrons) > 0, "Объекты полиэдров не созданы"


def test_polyhedron_geometry(vor):
    poly = vor.polyhedrons[0]
    assert len(poly.normal) == 4
    assert np.isclose(np.linalg.norm(poly.normal), 1.0, atol=1e-5), "Нормаль не единичная"
    assert len(poly.faces) > 0
    assert isinstance(poly, Face3D)


def test_max_len_calculation(vor):
    assert vor.max_len > 0


def test_triangulation(vor):
    assert vor.delaunay is not None


def test_faces_and_polyhedrons(vor):
    assert len(vor.faces_3d) > 0
    assert len(vor.polyhedrons) > 0

    # для 4D гиперкуба должно быть 8 кубических граней (ячеек)
    assert len(vor.polyhedrons) == 8


def test_central_region_found(vor):
    assert vor.central_region_index is not None
    assert len(vor.central) > 0
    # для кубической решётки центральная точка (0,0,0,0) должна быть в coords4
    assert [0.0, 0.0, 0.0, 0.0] in vor.coords4
