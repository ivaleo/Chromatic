"""Параметризация форм Грама: одна реализация вместо тринадцати копий."""

import numpy as np
import pytest

from chromatic_research.forms import norm_gram, pack, unpack


@pytest.mark.parametrize("dim", [2, 3, 4, 5])
def test_norm_gram_has_unit_determinant(dim):
    rng = np.random.default_rng(dim)
    basis = rng.normal(size=(dim, dim)) + dim * np.eye(dim)
    assert np.linalg.det(norm_gram(basis)) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("dim", [2, 3, 4, 5])
def test_pack_unpack_roundtrip(dim):
    rng = np.random.default_rng(100 + dim)
    basis = rng.normal(size=(dim, dim)) + dim * np.eye(dim)
    form = norm_gram(basis)
    assert unpack(pack(form), dim) == pytest.approx(form, abs=1e-12)


def test_pack_length_is_triangular_number():
    form = norm_gram(np.eye(4))
    assert pack(form).shape == (10,)     # 4*5/2


def test_unpack_rejects_degenerate_form():
    assert unpack(np.zeros(10), 4) is None


def test_d4_gram_matches_known_value():
    d4 = np.array([[2, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]], float)
    form = norm_gram(d4)
    assert np.linalg.det(form) == pytest.approx(1.0, rel=1e-12)
    assert form == pytest.approx(form.T, abs=1e-15)
