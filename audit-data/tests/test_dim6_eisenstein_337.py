"""Контроли эйзенштейнова экрана ℝ⁶ / 337.

Главное, что здесь проверяется, — не сходимость счёта, а ПОЛНОТА перебора:
что подмодули, которые экран вычёркивает прямыми в ℙ², — это в точности
ℤ[ω]-подмодули индекса 337, и что их ровно 227 814.
"""

import numpy as np
import pytest

from chromatic_research.campaigns.dim6_eisenstein_337 import (
    DIM,
    INDEX,
    RANK,
    canonical,
    cube_roots_of_unity,
    eigenspace,
    hyperplane_points,
    invariant_form_basis,
    screen,
    seed_lattice,
)


@pytest.fixture(scope="module")
def lattice():
    return seed_lattice()


def test_337_splits_in_eisenstein_integers():
    roots = cube_roots_of_unity()
    assert len(roots) == 2
    for root in roots:
        assert (root * root + root + 1) % INDEX == 0
    assert INDEX % 3 == 1


def test_automorphism_is_an_eisenstein_structure(lattice):
    basis, automorphism = lattice
    identity = np.eye(DIM, dtype=int)
    assert np.array_equal(automorphism @ automorphism @ automorphism, identity)
    assert not np.array_equal(automorphism, identity)
    # характеристический многочлен (x²+x+1)³ ⟺ S² + S + I = 0
    assert not np.any(automorphism @ automorphism + automorphism + identity)


def test_automorphism_is_an_isometry(lattice):
    basis, automorphism = lattice
    gram = basis @ basis.T
    assert np.allclose(automorphism @ gram @ automorphism.T, gram)


def test_seed_is_e6_star(lattice):
    """Посев обязан быть именно E₆*: 2R/λ₁ = √2 и 126 фасет."""
    import combigeo

    from chromatic_research.campaigns.dim6_eisenstein_337 import covering_radius

    basis, _ = lattice
    shortest = float(np.linalg.norm(combigeo.shortest_vector(basis.tolist())))
    facets = combigeo.relevant_facets(basis.tolist())
    radius = covering_radius(facets, 400, 0)
    assert len(facets) == 126
    assert 2 * radius / shortest == pytest.approx(np.sqrt(2), abs=1e-5)


def test_invariant_cone_is_nine_dimensional(lattice):
    _, automorphism = lattice
    forms = invariant_form_basis(automorphism)
    assert len(forms) == 9
    for form in forms:
        assert np.allclose(automorphism @ form @ automorphism.T, form)
        assert np.allclose(form, form.T)


def test_eigenspaces_are_three_dimensional(lattice):
    _, automorphism = lattice
    for root in cube_roots_of_unity():
        space = eigenspace(automorphism, root)
        assert len(space) == RANK
        for vector in space:
            image = [sum(int(automorphism[i][j]) * vector[j] for j in range(DIM))
                     for i in range(DIM)]
            assert all((image[i] - root * vector[i]) % INDEX == 0
                       for i in range(DIM))


def _submodule_basis(functional):
    """ℤ-базис ``M_c = {x : x·c ≡ 0 (mod 337)}`` через HNF."""
    from chromatic_research.campaigns.dim6_cyclotomic7_337 import (
        _hermite_normal_form,
    )

    rows = []
    for i in range(DIM):
        row = [0] * DIM
        row[i] = INDEX
        rows.append(row)
    pivot = next(i for i in range(DIM) if functional[i] % INDEX)
    inverse = pow(int(functional[pivot]) % INDEX, INDEX - 2, INDEX)
    for i in range(DIM):
        if i == pivot:
            continue
        row = [0] * DIM
        row[i] = 1
        row[pivot] = (-functional[i] * inverse) % INDEX
        rows.append(row)
    return np.array(_hermite_normal_form(rows, DIM))


def test_submodules_have_index_337_and_are_invariant(lattice):
    """Каждая точка ℙ² даёт ℤ[ω]-подмодуль ровно индекса 337."""
    _, automorphism = lattice
    for root in cube_roots_of_unity():
        space = eigenspace(automorphism, root)
        for functional in space:
            basis = _submodule_basis(functional)
            assert round(abs(np.linalg.det(basis.astype(float)))) == INDEX
            image = basis @ automorphism
            coordinates = np.linalg.solve(basis.T.astype(float),
                                          image.T.astype(float)).T
            assert np.allclose(coordinates, np.rint(coordinates), atol=1e-6)


def test_hyperplane_has_p_plus_one_points():
    points = hyperplane_points((3, 5, 7))
    assert len(points) == INDEX + 1
    assert len(set(points)) == INDEX + 1
    for point in points:
        assert sum(a * b for a, b in zip(point, (3, 5, 7))) % INDEX == 0
        assert canonical(point) == point


def test_screen_counts_all_submodules(lattice):
    """При нулевом пороге запретов нет и выживают ВСЕ подмодули."""
    basis, automorphism = lattice
    report = screen(basis, automorphism, ell=0.0, directions=40)
    assert report["ok"]
    assert report["forbidden"] == 0
    assert report["survivors"] == [INDEX ** 2 + INDEX + 1] * 2
    assert sum(report["survivors"]) == 227_814


def test_screen_is_monotone_in_threshold(lattice):
    """Чем выше порог, тем меньше выживших — на этом стоит двоичный поиск."""
    basis, automorphism = lattice
    counts = []
    for ell in (0.2, 0.5, 0.8, 1.0):
        report = screen(basis, automorphism, ell=ell, directions=40)
        counts.append(sum(report["survivors"]))
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == 0


def test_e6_star_does_not_reach_the_threshold(lattice):
    """У самого E₆* при k = 337 ни один подмодуль не даёт d ≥ 1."""
    basis, automorphism = lattice
    report = screen(basis, automorphism, ell=1.0, directions=120)
    assert report["survivors"] == [0, 0]
