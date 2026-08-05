"""Prime-primary search for high-dimensional lattice colorings.

The coloring condition for a fixed lattice is

    A f != 0 in G for every forbidden coordinate vector f,

where A is a homomorphism Z^n -> G and the coloring sublattice is ker(A).
This module searches the primary components of A a whole row at a time.

For a prime p, every possible row is a projective point of F_p^n.  The number
of still-killed vectors for *all* rows is the finite Radon transform of the
histogram of forbidden residues.  It is evaluated with one n-dimensional FFT:

    sum_{a.x=0} h(x) = (1/p) sum_t Fourier(h)(t a).

This differs from min-conflicts in two important ways:

* a move is globally best among every index-p child, rather than one random
  coefficient flip;
* repeated prime rows and prime-power rows recover arbitrary primary quotient
  structures without enumerating HNF matrices of a prescribed total index.

The module is deliberately kept in audit-data while the method is experimental.
It uses exact integer arithmetic for every final coloring check; FFT rounding is
only a candidate-ranking oracle and every returned score is checked directly.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sympy import Matrix, ZZ
from sympy.matrices.normalforms import hermite_normal_form, smith_normal_form
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.normalforms import smith_normal_decomp
from chromatic_research.paths import results_path




_FORM_CACHE: dict[tuple[int, int], np.ndarray] = {}


def weighted_improves(candidate: float, incumbent: float) -> bool:
    """Scale-aware strict comparison for nonnegative weighted losses.

    Geometry objectives often use high powers of a small distance deficit.
    Their meaningful values can be far below 1e-15, so an absolute tolerance
    silently freezes the search.  This comparison keeps a relative tolerance
    plus a few ULPs at the actual scale.
    """
    candidate = float(candidate)
    incumbent = float(incumbent)
    if not math.isfinite(incumbent):
        return math.isfinite(candidate)
    scale = max(abs(candidate), abs(incumbent))
    tolerance = max(
        1e-12 * scale,
        16.0 * float(np.spacing(scale)),
    )
    return candidate < incumbent - tolerance


def weighted_close(first: float, second: float) -> bool:
    """Scale-aware equality companion to :func:`weighted_improves`."""
    first = float(first)
    second = float(second)
    scale = max(abs(first), abs(second))
    tolerance = max(
        1e-10 * scale,
        64.0 * float(np.spacing(scale)),
    )
    return abs(first - second) <= tolerance


def _prime_power(q: int) -> tuple[int, int]:
    """Return (p, exponent) for a prime power q; raise for other q."""
    if q < 2:
        raise ValueError("modulus must be >= 2")
    for p in range(2, int(math.isqrt(q)) + 2):
        if q % p:
            continue
        x = q
        exponent = 0
        while x % p == 0:
            x //= p
            exponent += 1
        if x == 1 and all(p % d for d in range(2, int(math.isqrt(p)) + 1)):
            return p, exponent
        break
    # q itself may be prime.
    if all(q % d for d in range(2, int(math.isqrt(q)) + 1)):
        return q, 1
    raise ValueError(f"{q} is not a prime power; split it by CRT")


def is_prime(q: int) -> bool:
    try:
        p, exponent = _prime_power(q)
    except ValueError:
        return False
    return p == q and exponent == 1


def projective_forms(n: int, q: int) -> np.ndarray:
    """Primitive rows modulo a prime power q, one representative per unit orbit.

    The first coordinate not divisible by the underlying prime is normalized to
    one.  Earlier coordinates are divisible by p; later coordinates are free.
    For prime q this is the usual projective space P^(n-1)(F_q).
    """
    key = (n, q)
    cached = _FORM_CACHE.get(key)
    if cached is not None:
        return cached

    p, _ = _prime_power(q)
    chunks: list[np.ndarray] = []
    small = q // p
    for pivot in range(n):
        count = (small**pivot) * (q ** (n - pivot - 1))
        values = np.arange(count, dtype=np.int64)
        rows = np.zeros((count, n), dtype=np.int16)
        rows[:, pivot] = 1
        for col in range(n - 1, pivot, -1):
            rows[:, col] = values % q
            values //= q
        for col in range(pivot - 1, -1, -1):
            rows[:, col] = p * (values % small)
            values //= small
        chunks.append(rows)
    result = np.vstack(chunks).astype(np.int64, copy=False)
    _FORM_CACHE[key] = result
    return result


def rank_mod(matrix: Sequence[Sequence[int]] | np.ndarray, p: int) -> int:
    """Rank over F_p."""
    a = np.asarray(matrix, dtype=np.int64).copy() % p
    if a.ndim != 2:
        raise ValueError("rank_mod expects a matrix")
    rows, cols = a.shape
    rank = 0
    for col in range(cols):
        pivot = next((r for r in range(rank, rows) if a[r, col] % p), None)
        if pivot is None:
            continue
        a[[rank, pivot]] = a[[pivot, rank]]
        a[rank] = a[rank] * pow(int(a[rank, col]), -1, p) % p
        for row in range(rows):
            if row != rank and a[row, col] % p:
                a[row] = (a[row] - a[row, col] * a[rank]) % p
        rank += 1
        if rank == rows:
            break
    return rank


def nullspace_mod(matrix: Sequence[Sequence[int]] | np.ndarray, p: int) -> np.ndarray:
    """A row basis for the right nullspace over F_p."""
    a = np.asarray(matrix, dtype=np.int64).copy() % p
    rows, cols = a.shape
    rank = 0
    pivots: list[int] = []
    for col in range(cols):
        pivot = next((r for r in range(rank, rows) if a[r, col] % p), None)
        if pivot is None:
            continue
        a[[rank, pivot]] = a[[pivot, rank]]
        a[rank] = a[rank] * pow(int(a[rank, col]), -1, p) % p
        for row in range(rows):
            if row != rank and a[row, col] % p:
                a[row] = (a[row] - a[row, col] * a[rank]) % p
        pivots.append(col)
        rank += 1
        if rank == rows:
            break
    free = [col for col in range(cols) if col not in set(pivots)]
    basis: list[np.ndarray] = []
    for free_col in free:
        vector = np.zeros(cols, dtype=np.int64)
        vector[free_col] = 1
        for row, pivot_col in enumerate(pivots):
            vector[pivot_col] = -a[row, free_col] % p
        basis.append(vector)
    return np.asarray(basis, dtype=np.int64)


def killed_mask(
    forbidden: np.ndarray,
    rows: Sequence[np.ndarray],
    moduli: Sequence[int],
    skip: int | None = None,
) -> np.ndarray:
    """Vectors mapped to zero by every row, optionally excluding one row."""
    mask = np.ones(len(forbidden), dtype=bool)
    for index, (row, modulus) in enumerate(zip(rows, moduli)):
        if index == skip:
            continue
        mask &= (forbidden @ row) % modulus == 0
    return mask


def radon_scores(
    residues: np.ndarray,
    p: int,
    forms: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Score every projective form over F_p using a finite Radon transform.

    Scores are recomputed directly for the best candidates by callers; rounded
    FFT output is never used as a final validity decision.
    """
    if not is_prime(p):
        raise ValueError("radon_scores requires a prime modulus")
    if residues.ndim != 2:
        raise ValueError("residues must be a two-dimensional array")
    n = residues.shape[1]
    forms = projective_forms(n, p) if forms is None else forms
    histogram = np.zeros((p,) * n, dtype=np.float64)
    if len(residues):
        index = tuple((residues % p).T)
        if weights is None:
            np.add.at(histogram, index, 1.0)
        else:
            np.add.at(histogram, index, np.asarray(weights, dtype=np.float64))
    spectrum = np.fft.fftn(histogram)
    sums = np.zeros(len(forms), dtype=np.complex128)
    for scalar in range(p):
        index = tuple(((scalar * forms) % p).T)
        sums += spectrum[index]
    values = sums.real / p
    if weights is None:
        return np.rint(values).astype(np.int64)
    return values


def direct_scores(
    vectors: np.ndarray,
    forms: np.ndarray,
    modulus: int,
    weights: np.ndarray | None = None,
    chunk: int = 1024,
) -> np.ndarray:
    """Direct scores for a prime-power pool, bounded in memory."""
    dtype = np.float64 if weights is not None else np.int64
    scores = np.empty(len(forms), dtype=dtype)
    for lo in range(0, len(forms), chunk):
        hi = min(lo + chunk, len(forms))
        zero = (vectors @ forms[lo:hi].T) % modulus == 0
        if weights is None:
            scores[lo:hi] = np.count_nonzero(zero, axis=0)
        else:
            scores[lo:hi] = np.asarray(weights, dtype=np.float64) @ zero
    return scores


def score_forms(
    vectors: np.ndarray,
    modulus: int,
    forms: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    forms = projective_forms(vectors.shape[1], modulus) if forms is None else forms
    if is_prime(modulus) and modulus ** vectors.shape[1] <= 2_000_000:
        return radon_scores(vectors, modulus, forms, weights)
    return direct_scores(vectors, forms, modulus, weights)


def image_size(rows: Sequence[np.ndarray], moduli: Sequence[int], n: int) -> int:
    """Exact size of the image in the direct product of cyclic components."""
    generators = [
        tuple(int(rows[j][i]) % moduli[j] for j in range(len(rows)))
        for i in range(n)
    ]
    zero = (0,) * len(rows)
    group = {zero}
    frontier = [zero]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            nxt = tuple(
                (current[j] + generator[j]) % moduli[j]
                for j in range(len(rows))
            )
            if nxt not in group:
                group.add(nxt)
                frontier.append(nxt)
    return len(group)


def kernel_basis(rows: Sequence[np.ndarray], moduli: Sequence[int], n: int) -> np.ndarray:
    """Integer column basis of the common modular kernel.

    This constructive routine is sufficient for the primary structures used in
    the campaigns here: independent rows over prime fields and primitive cyclic
    prime-power rows whose modulus is coprime to preceding blocks.
    """
    basis = np.eye(n, dtype=np.int64)
    for row, modulus in zip(rows, moduli):
        restricted = np.asarray(row, dtype=np.int64) @ basis
        restricted %= modulus
        common = modulus
        for value in restricted:
            common = math.gcd(common, int(value))
        effective = modulus // common
        if effective == 1:
            continue
        reduced = (restricted // common) % effective
        pivot = next(
            (i for i, value in enumerate(reduced) if math.gcd(int(value), effective) == 1),
            None,
        )
        if pivot is None:
            # A primitive row over a composite cyclic group need not contain
            # an individually invertible coordinate (for example [2, 3]
            # modulo 6).  Compute a unimodular Smith transformation of
            # [reduced, -effective].  Its last n columns span all integer
            # solutions of reduced*x - effective*q = 0; dropping q gives a
            # full column basis of the modular kernel.
            augmented_values = [
                *[int(value) for value in reduced],
                -int(effective),
            ]
            augmented = DomainMatrix(
                [[ZZ(value) for value in augmented_values]],
                (1, n + 1),
                ZZ,
            )
            smith, _, right = smith_normal_decomp(augmented)
            if abs(int(smith.to_Matrix()[0, 0])) != 1:
                raise ValueError(
                    f"restricted row is not primitive modulo {effective}: "
                    f"{reduced.tolist()}"
                )
            transform = right.to_Matrix()
            raw_child = transform[:n, 1:]
            child = np.asarray(
                hermite_normal_form(raw_child).tolist(),
                dtype=np.int64,
            )
            if (
                child.shape != (n, n)
                or abs(int(Matrix(child.tolist()).det())) != effective
                or np.any((reduced @ child) % effective)
            ):
                raise AssertionError(
                    "Smith fallback returned an invalid cyclic kernel"
                )
        else:
            normalized = (
                reduced
                * pow(int(reduced[pivot]), -1, effective)
                % effective
            )
            child = np.eye(n, dtype=np.int64)
            child[pivot, :] = -normalized
            child[pivot, pivot] = effective
        basis = basis @ child
    return basis


def hnf_columns(basis: np.ndarray) -> np.ndarray:
    """Canonical HNF for a lattice represented by columns."""
    return np.asarray(hermite_normal_form(Matrix(basis.tolist())).tolist(), dtype=np.int64)


def smith_diagonal(basis: np.ndarray) -> list[int]:
    diagonal = smith_normal_form(Matrix(basis.tolist())).diagonal()
    return [abs(int(value)) for value in diagonal]


def rref_subspaces(
    n: int, p: int, rank: int
) -> Iterable[np.ndarray]:
    """Yield one RREF row basis for every rank-dimensional subspace of F_p^n."""
    if not (0 <= rank <= n):
        raise ValueError("rank must lie between zero and n")
    if rank == 0:
        yield np.zeros((0, n), dtype=np.int64)
        return
    for pivots in itertools.combinations(range(n), rank):
        pivot_set = set(pivots)
        free_positions = [
            (row, col)
            for row, pivot in enumerate(pivots)
            for col in range(pivot + 1, n)
            if col not in pivot_set
        ]
        count = p ** len(free_positions)
        for code in range(count):
            matrix = np.zeros((rank, n), dtype=np.int64)
            for row, pivot in enumerate(pivots):
                matrix[row, pivot] = 1
            value = code
            for row, col in free_positions:
                matrix[row, col] = value % p
                value //= p
            yield matrix


def gaussian_binomial(n: int, rank: int, p: int) -> int:
    """Number of rank-dimensional subspaces of F_p^n."""
    rank = min(rank, n - rank)
    numerator = 1
    denominator = 1
    for index in range(rank):
        numerator *= p ** (n - index) - 1
        denominator *= p ** (rank - index) - 1
    return numerator // denominator


def _mask_to_int(mask: np.ndarray) -> int:
    packed = np.packbits(np.asarray(mask, dtype=np.uint8), bitorder="little")
    return int.from_bytes(packed.tobytes(), byteorder="little")


def exact_field_subspace(
    vectors: np.ndarray,
    p: int,
    rank: int,
    *,
    progress_every: int = 100_000,
) -> tuple[int, np.ndarray]:
    """Exact minimum killed set over every rank-rowspace in F_p^n.

    Zero masks of projective rows are represented by Python integers, making
    each subspace evaluation only ``rank-1`` bitwise ANDs plus ``bit_count``.
    This is practical for, e.g., all [7 choose 3]_3 = 925,771 subspaces.
    """
    if not is_prime(p):
        raise ValueError("exact_field_subspace requires a prime")
    vectors = np.asarray(vectors, dtype=np.int64)
    n = vectors.shape[1]
    forms = projective_forms(n, p)
    lookup = {tuple(row.tolist()): index for index, row in enumerate(forms)}
    zero = (vectors @ forms.T) % p == 0
    bitsets = [_mask_to_int(zero[:, index]) for index in range(len(forms))]
    del zero

    total = gaussian_binomial(n, rank, p)
    best = len(vectors) + 1
    best_basis: np.ndarray | None = None
    start = time.perf_counter()
    for number, basis in enumerate(rref_subspaces(n, p, rank), start=1):
        indices = [lookup[tuple(row.tolist())] for row in basis]
        killed = bitsets[indices[0]]
        for index in indices[1:]:
            killed &= bitsets[index]
        count = killed.bit_count()
        if count < best:
            best = count
            best_basis = basis.copy()
            print(
                f"    exact F_{p} rank={rank} best={best} "
                f"after {number}/{total}",
                flush=True,
            )
        if best == 0:
            break
        if progress_every and number % progress_every == 0:
            print(
                f"    exact F_{p} rank={rank}: {number}/{total} "
                f"best={best} elapsed={time.perf_counter()-start:.1f}s",
                flush=True,
            )
    assert best_basis is not None
    # Final direct check.
    exact = int(
        np.count_nonzero(
            np.all((vectors @ best_basis.T) % p == 0, axis=1)
        )
    )
    if exact != best:
        raise AssertionError(f"bitset score mismatch: {best} != {exact}")
    return best, best_basis


def solve_field_subspace_cpsat(
    vectors: np.ndarray,
    p: int,
    rank: int,
    *,
    seconds_per_pivot: float = 5.0,
    workers: int = 8,
    verbose: bool = True,
) -> tuple[str, np.ndarray | None]:
    """Find a rank-rowspace separating every vector, with RREF symmetry breaking.

    The generic modular CP-SAT model used elsewhere in the repository has a
    large product-group symmetry.  Here each possible RREF pivot pattern is
    handled separately, removing GL(rank, p) exactly and leaving only the free
    entries of a canonical field subspace.

    Returns ("SAT", basis), ("UNSAT", None), or ("UNKNOWN", None).  UNSAT is
    reported only when every pivot pattern was proved infeasible.
    """
    if not is_prime(p):
        raise ValueError("CP-SAT field solver requires a prime")
    from ortools.sat.python import cp_model

    vectors = np.asarray(vectors, dtype=np.int64) % p
    n = vectors.shape[1]
    if np.any(np.all(vectors == 0, axis=1)):
        return "UNSAT", None
    unknown = False
    patterns = list(itertools.combinations(range(n), rank))
    for pattern_number, pivots in enumerate(patterns, start=1):
        pivot_set = set(pivots)
        model = cp_model.CpModel()
        entries: list[list[int | cp_model.IntVar]] = [
            [0 for _ in range(n)] for _ in range(rank)
        ]
        variables: dict[tuple[int, int], cp_model.IntVar] = {}
        for row, pivot in enumerate(pivots):
            entries[row][pivot] = 1
            for col in range(pivot + 1, n):
                if col in pivot_set:
                    continue
                variable = model.NewIntVar(0, p - 1, f"a_{row}_{col}")
                entries[row][col] = variable
                variables[row, col] = variable

        for vector_number, vector in enumerate(vectors):
            nonzero = []
            constraint_already_satisfied = False
            for row in range(rank):
                constant = 0
                terms = []
                maximum = 0
                for col, coefficient in enumerate(vector):
                    coefficient = int(coefficient)
                    if coefficient == 0:
                        continue
                    entry = entries[row][col]
                    if isinstance(entry, int):
                        constant += coefficient * entry
                    else:
                        terms.append(coefficient * entry)
                        maximum += coefficient * (p - 1)
                if not terms:
                    if constant % p:
                        constraint_already_satisfied = True
                        break
                    continue
                total = model.NewIntVar(
                    constant, constant + maximum, f"s_{row}_{vector_number}"
                )
                model.Add(total == constant + sum(terms))
                remainder = model.NewIntVar(0, p - 1, f"r_{row}_{vector_number}")
                model.AddModuloEquality(remainder, total, p)
                is_nonzero = model.NewBoolVar(f"nz_{row}_{vector_number}")
                model.Add(remainder >= 1).OnlyEnforceIf(is_nonzero)
                model.Add(remainder == 0).OnlyEnforceIf(is_nonzero.Not())
                nonzero.append(is_nonzero)
            if constraint_already_satisfied:
                continue
            if not nonzero:
                # This RREF pivot pattern kills the vector identically.
                model.AddBoolOr([])
            else:
                model.AddBoolOr(nonzero)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = seconds_per_pivot
        solver.parameters.num_search_workers = workers
        status = solver.Solve(model)
        if verbose:
            label = solver.StatusName(status)
            print(
                f"    CP-SAT F_{p} rank={rank} pivots={pivots} "
                f"{pattern_number}/{len(patterns)}: {label} "
                f"{solver.WallTime():.2f}s",
                flush=True,
            )
        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            basis = np.zeros((rank, n), dtype=np.int64)
            for row, pivot in enumerate(pivots):
                basis[row, pivot] = 1
            for key, variable in variables.items():
                basis[key] = int(solver.Value(variable))
            # Each vector needs at least one nonzero row, not every row.
            exact = np.any((vectors @ basis.T) % p != 0, axis=1)
            if not bool(np.all(exact)):
                raise AssertionError("CP-SAT returned an invalid field subspace")
            return "SAT", basis
        if status != cp_model.INFEASIBLE:
            unknown = True
    return ("UNKNOWN" if unknown else "UNSAT"), None


@dataclass
class SearchResult:
    killed: int
    rows: list[np.ndarray]
    moduli: list[int]
    image_index: int
    restarts: int
    sweeps: int
    seconds: float

    @property
    def found(self) -> bool:
        return self.killed == 0

    def as_json(self) -> dict:
        payload = {
            "found": self.found,
            "killed": self.killed,
            "moduli": self.moduli,
            "rows": [row.astype(int).tolist() for row in self.rows],
            "image_index": self.image_index,
            "restarts": self.restarts,
            "sweeps": self.sweeps,
            "seconds": round(self.seconds, 6),
        }
        if self.found:
            basis = kernel_basis(self.rows, self.moduli, len(self.rows[0]))
            canonical = hnf_columns(basis)
            payload["kernel_basis_columns"] = canonical.astype(int).tolist()
            payload["det"] = abs(int(Matrix(canonical.tolist()).det()))
            payload["smith"] = smith_diagonal(canonical)
        return payload


@dataclass
class WeightedSearchResult:
    weighted_loss: float
    killed: int
    rows: list[np.ndarray]
    moduli: list[int]
    image_index: int
    restarts: int
    sweeps: int
    seconds: float

    @property
    def found(self) -> bool:
        return self.killed == 0

    def as_json(self) -> dict:
        payload = {
            "found": self.found,
            "weighted_loss": round(self.weighted_loss, 12),
            "killed": self.killed,
            "moduli": self.moduli,
            "rows": [row.astype(int).tolist() for row in self.rows],
            "image_index": self.image_index,
            "restarts": self.restarts,
            "sweeps": self.sweeps,
            "seconds": round(self.seconds, 6),
        }
        if self.found:
            basis = kernel_basis(self.rows, self.moduli, len(self.rows[0]))
            canonical = hnf_columns(basis)
            payload["kernel_basis_columns"] = canonical.astype(int).tolist()
            payload["det"] = abs(int(Matrix(canonical.tolist()).det()))
            payload["smith"] = smith_diagonal(canonical)
        return payload


class PrimarySearch:
    """Block-coordinate search over complete primary character pools."""

    def __init__(
        self,
        forbidden: Sequence[Sequence[int]] | np.ndarray,
        moduli: Sequence[int],
        *,
        seed: int = 0,
    ) -> None:
        self.forbidden = np.asarray(forbidden, dtype=np.int64)
        if self.forbidden.ndim != 2:
            raise ValueError("forbidden must be a two-dimensional array")
        self.n = self.forbidden.shape[1]
        self.moduli = [int(q) for q in moduli]
        self.pools = [projective_forms(self.n, q) for q in self.moduli]
        self.rng = np.random.default_rng(seed)

    def _independent(
        self, candidate: np.ndarray, rows: Sequence[np.ndarray], row_index: int
    ) -> bool:
        modulus = self.moduli[row_index]
        prime, _ = _prime_power(modulus)
        peers = [
            np.asarray(rows[index], dtype=np.int64) % prime
            for index, other_modulus in enumerate(self.moduli)
            if index != row_index and _prime_power(other_modulus)[0] == prime
        ]
        if not peers:
            return True
        before = rank_mod(np.asarray(peers), prime)
        after = rank_mod(np.vstack([peers, candidate % prime]), prime)
        return after == before + 1

    def random_rows(self) -> list[np.ndarray]:
        rows: list[np.ndarray] = []
        for index, pool in enumerate(self.pools):
            for _ in range(10_000):
                candidate = pool[int(self.rng.integers(len(pool)))].copy()
                provisional = rows + [candidate]
                # _independent expects all row slots; check directly for the
                # already initialized peers of this modulus.
                modulus = self.moduli[index]
                prime, _ = _prime_power(modulus)
                peers = [
                    rows[j] % prime
                    for j in range(len(rows))
                    if _prime_power(self.moduli[j])[0] == prime
                ]
                if (
                    not peers
                    or rank_mod(np.vstack(peers + [candidate % prime]), prime)
                    == len(peers) + 1
                ):
                    rows.append(candidate)
                    break
            else:
                raise RuntimeError("could not initialize independent rows")
        return rows

    def best_candidates(
        self,
        rows: Sequence[np.ndarray],
        row_index: int,
        *,
        top: int = 8,
        weights: np.ndarray | None = None,
    ) -> list[tuple[float, np.ndarray]]:
        active = killed_mask(self.forbidden, rows, self.moduli, skip=row_index)
        vectors = self.forbidden[active]
        active_weights = None if weights is None else np.asarray(weights)[active]
        pool = self.pools[row_index]
        scores = score_forms(vectors, self.moduli[row_index], pool, active_weights)
        request = min(len(pool), max(top * 8, 64))
        ids = np.argpartition(scores, request - 1)[:request]
        ids = ids[np.argsort(scores[ids], kind="stable")]
        result: list[tuple[float, np.ndarray]] = []
        for candidate_index in ids:
            candidate = pool[int(candidate_index)]
            if not self._independent(candidate, rows, row_index):
                continue
            # Direct integer recount guards against any FFT rounding issue.
            zero = (vectors @ candidate) % self.moduli[row_index] == 0
            score = (
                float(np.asarray(active_weights, dtype=np.float64)[zero].sum())
                if active_weights is not None
                else float(np.count_nonzero(zero))
            )
            result.append((score, candidate.copy()))
            if len(result) >= top:
                break
        return result

    def _independent_against(
        self,
        candidate: np.ndarray,
        row_index: int,
        rows: Sequence[np.ndarray],
        ignored: set[int],
        extra: Sequence[np.ndarray] = (),
    ) -> bool:
        """Independence from fixed rows of the same prime modulus."""
        modulus = self.moduli[row_index]
        prime, _ = _prime_power(modulus)
        peers = [
            np.asarray(rows[index], dtype=np.int64) % prime
            for index, other_modulus in enumerate(self.moduli)
            if index not in ignored and _prime_power(other_modulus)[0] == prime
        ]
        peers.extend(np.asarray(row, dtype=np.int64) % prime for row in extra)
        if not peers:
            return True
        before = rank_mod(np.asarray(peers), prime)
        after = rank_mod(np.vstack([peers, candidate % prime]), prime)
        return after == before + 1

    def pair_improve(
        self,
        rows: Sequence[np.ndarray],
        first_index: int,
        second_index: int,
        *,
        first_top: int = 512,
        progress_every: int = 100,
    ) -> tuple[int, list[np.ndarray]]:
        """Best two-row replacement after retaining top first-row candidates.

        A one-row local minimum is common in the record-sized problems.  Given
        all other rows, this routine enumerates promising first rows and solves
        the second-row best response *globally* for each of them.
        """
        if first_index == second_index:
            raise ValueError("pair indices must differ")
        rows = [np.asarray(row, dtype=np.int64).copy() for row in rows]
        ignored = {first_index, second_index}
        active = np.ones(len(self.forbidden), dtype=bool)
        for index, (row, modulus) in enumerate(zip(rows, self.moduli)):
            if index in ignored:
                continue
            active &= (self.forbidden @ row) % modulus == 0
        vectors = self.forbidden[active]
        first_pool = self.pools[first_index]
        first_scores = score_forms(
            vectors, self.moduli[first_index], first_pool
        )
        request = min(len(first_pool), max(first_top * 4, 64))
        ids = np.argpartition(first_scores, request - 1)[:request]
        ids = ids[np.argsort(first_scores[ids], kind="stable")]
        first_candidates: list[np.ndarray] = []
        for candidate_id in ids:
            candidate = first_pool[int(candidate_id)]
            if self._independent_against(
                candidate, first_index, rows, ignored
            ):
                first_candidates.append(candidate.copy())
            if len(first_candidates) >= first_top:
                break

        current = int(killed_mask(self.forbidden, rows, self.moduli).sum())
        best = current
        best_rows = [row.copy() for row in rows]
        second_pool = self.pools[second_index]
        same_prime = (
            _prime_power(self.moduli[first_index])[0]
            == _prime_power(self.moduli[second_index])[0]
        )
        for number, first in enumerate(first_candidates, start=1):
            residual = vectors[
                (vectors @ first) % self.moduli[first_index] == 0
            ]
            second_scores = score_forms(
                residual, self.moduli[second_index], second_pool
            )
            second_ids = np.argsort(second_scores, kind="stable")
            for second_id in second_ids:
                second = second_pool[int(second_id)]
                extra = [first] if same_prime else []
                if not self._independent_against(
                    second, second_index, rows, ignored, extra
                ):
                    continue
                exact = int(
                    np.count_nonzero(
                        (residual @ second) % self.moduli[second_index] == 0
                    )
                )
                if exact < best:
                    best = exact
                    best_rows = [row.copy() for row in rows]
                    best_rows[first_index] = first.copy()
                    best_rows[second_index] = second.copy()
                    print(
                        f"    pair ({first_index},{second_index}) best={best} "
                        f"after {number}/{len(first_candidates)} first rows",
                        flush=True,
                    )
                break
            if best == 0:
                return best, best_rows
            if progress_every and number % progress_every == 0:
                print(
                    f"    pair ({first_index},{second_index}) "
                    f"{number}/{len(first_candidates)} best={best}",
                    flush=True,
                )
        return best, best_rows

    def pair_polish(
        self,
        rows: Sequence[np.ndarray],
        *,
        first_top: int = 512,
    ) -> tuple[int, list[np.ndarray]]:
        """Try every ordered pair, prioritizing the larger candidate pool."""
        best_rows = [np.asarray(row, dtype=np.int64).copy() for row in rows]
        best = int(killed_mask(self.forbidden, best_rows, self.moduli).sum())
        pairs = list(itertools.combinations(range(len(rows)), 2))
        pairs.sort(
            key=lambda pair: -max(len(self.pools[pair[0]]), len(self.pools[pair[1]]))
        )
        for left, right in pairs:
            # Enumerate the larger pool only up to first_top, then solve the
            # smaller/global response exactly.
            if len(self.pools[left]) < len(self.pools[right]):
                left, right = right, left
            killed, candidate_rows = self.pair_improve(
                best_rows, left, right, first_top=first_top
            )
            if killed < best:
                best, best_rows = killed, candidate_rows
            if best == 0:
                break
        return best, best_rows

    def pair_improve_weighted(
        self,
        rows: Sequence[np.ndarray],
        first_index: int,
        second_index: int,
        weights: Sequence[float] | np.ndarray,
        *,
        first_top: int = 512,
        progress_every: int = 100,
    ) -> tuple[float, list[np.ndarray]]:
        """Geometry-weighted two-row look-ahead.

        ``descend`` computes an exact best response for one block while all
        others remain fixed.  That can be trapped when two blocks must change
        together.  Here the most promising ``first_top`` rows of the first
        block are retained and the second block is solved globally for each.
        All candidate rankings use floating weights, but every accepted loss
        is recomputed directly on the exact modular zero mask.
        """
        if first_index == second_index:
            raise ValueError("pair indices must differ")
        weights_array = np.asarray(weights, dtype=np.float64)
        if weights_array.shape != (len(self.forbidden),):
            raise ValueError(
                f"weights must have shape {(len(self.forbidden),)}, "
                f"got {weights_array.shape}"
            )
        if not np.all(np.isfinite(weights_array)) or np.any(
            weights_array < 0
        ):
            raise ValueError("weights must be finite and nonnegative")

        rows = [np.asarray(row, dtype=np.int64).copy() for row in rows]
        ignored = {first_index, second_index}
        active = np.ones(len(self.forbidden), dtype=bool)
        for index, (row, modulus) in enumerate(zip(rows, self.moduli)):
            if index in ignored:
                continue
            active &= (self.forbidden @ row) % modulus == 0
        vectors = self.forbidden[active]
        active_weights = weights_array[active]

        first_pool = self.pools[first_index]
        first_scores = score_forms(
            vectors,
            self.moduli[first_index],
            first_pool,
            active_weights,
        )
        request = min(len(first_pool), max(first_top * 4, 64))
        ids = np.argpartition(first_scores, request - 1)[:request]
        ids = ids[np.argsort(first_scores[ids], kind="stable")]
        first_candidates: list[np.ndarray] = []
        for candidate_id in ids:
            candidate = first_pool[int(candidate_id)]
            if self._independent_against(
                candidate, first_index, rows, ignored
            ):
                first_candidates.append(candidate.copy())
            if len(first_candidates) >= first_top:
                break

        current_mask = killed_mask(self.forbidden, rows, self.moduli)
        best = float(weights_array[current_mask].sum())
        best_rows = [row.copy() for row in rows]
        second_pool = self.pools[second_index]
        same_prime = (
            _prime_power(self.moduli[first_index])[0]
            == _prime_power(self.moduli[second_index])[0]
        )
        for number, first in enumerate(first_candidates, start=1):
            first_zero = (
                (vectors @ first) % self.moduli[first_index] == 0
            )
            residual = vectors[first_zero]
            residual_weights = active_weights[first_zero]
            second_scores = score_forms(
                residual,
                self.moduli[second_index],
                second_pool,
                residual_weights,
            )
            second_ids = np.argsort(second_scores, kind="stable")
            for second_id in second_ids:
                second = second_pool[int(second_id)]
                extra = [first] if same_prime else []
                if not self._independent_against(
                    second, second_index, rows, ignored, extra
                ):
                    continue
                zero = (
                    (residual @ second)
                    % self.moduli[second_index]
                    == 0
                )
                exact = float(residual_weights[zero].sum())
                if weighted_improves(exact, best):
                    best = exact
                    best_rows = [row.copy() for row in rows]
                    best_rows[first_index] = first.copy()
                    best_rows[second_index] = second.copy()
                    print(
                        f"    weighted pair ({first_index},{second_index}) "
                        f"best={best:.9g} after "
                        f"{number}/{len(first_candidates)} first rows",
                        flush=True,
                    )
                break
            if best == 0.0:
                return best, best_rows
            if progress_every and number % progress_every == 0:
                print(
                    f"    weighted pair ({first_index},{second_index}) "
                    f"{number}/{len(first_candidates)} best={best:.9g}",
                    flush=True,
                )
        return best, best_rows

    def pair_polish_weighted(
        self,
        rows: Sequence[np.ndarray],
        weights: Sequence[float] | np.ndarray,
        *,
        first_top: int = 512,
    ) -> tuple[float, list[np.ndarray]]:
        """Try weighted two-block look-ahead for every ordered block pair."""
        weights_array = np.asarray(weights, dtype=np.float64)
        best_rows = [
            np.asarray(row, dtype=np.int64).copy() for row in rows
        ]
        best = float(
            weights_array[
                killed_mask(self.forbidden, best_rows, self.moduli)
            ].sum()
        )
        pairs = list(itertools.combinations(range(len(rows)), 2))
        pairs.sort(
            key=lambda pair: -max(
                len(self.pools[pair[0]]), len(self.pools[pair[1]])
            )
        )
        for left, right in pairs:
            if len(self.pools[left]) < len(self.pools[right]):
                left, right = right, left
            loss, candidate_rows = self.pair_improve_weighted(
                best_rows,
                left,
                right,
                weights_array,
                first_top=first_top,
            )
            if weighted_improves(loss, best):
                best, best_rows = loss, candidate_rows
            if best == 0.0:
                break
        return best, best_rows

    def descend(
        self,
        initial: Sequence[np.ndarray] | None = None,
        *,
        max_sweeps: int = 20,
        top: int = 8,
        kick_probability: float = 0.15,
        temperature: float = 0.35,
        weights: np.ndarray | None = None,
    ) -> tuple[float, list[np.ndarray], int]:
        rows = (
            [np.asarray(row, dtype=np.int64).copy() for row in initial]
            if initial is not None
            else self.random_rows()
        )
        mask = killed_mask(self.forbidden, rows, self.moduli)
        current = (
            float(mask.sum())
            if weights is None
            else float(np.asarray(weights, dtype=np.float64)[mask].sum())
        )
        best = current
        best_rows = [row.copy() for row in rows]
        stale = 0
        sweeps_done = 0
        for sweep in range(max_sweeps):
            sweeps_done = sweep + 1
            before_sweep = current
            order = self.rng.permutation(len(rows))
            for row_index in order:
                candidates = self.best_candidates(
                    rows, int(row_index), top=top, weights=weights
                )
                if not candidates:
                    continue
                choice = 0
                if (
                    len(candidates) > 1
                    and self.rng.random() < kick_probability
                    and stale > 0
                ):
                    ranks = np.arange(len(candidates), dtype=np.float64)
                    probabilities = np.exp(-ranks / max(temperature, 1e-9))
                    probabilities /= probabilities.sum()
                    choice = int(self.rng.choice(len(candidates), p=probabilities))
                score, candidate = candidates[choice]
                # Greedy choices never worsen. A kick may worsen temporarily,
                # but only on a plateau and the global incumbent is retained.
                if score <= current or choice != 0:
                    rows[int(row_index)] = candidate
                    current = score
                if current < best:
                    best = current
                    best_rows = [row.copy() for row in rows]
                if best == 0:
                    return best, best_rows, sweeps_done
            if current < before_sweep:
                stale = 0
            else:
                stale += 1
            if stale >= 3:
                break
        return best, best_rows, sweeps_done

    def run(
        self,
        *,
        restarts: int = 100,
        max_sweeps: int = 20,
        top: int = 8,
        progress_every: int = 10,
        initial_rows: Sequence[Sequence[int]] | None = None,
    ) -> SearchResult:
        start = time.perf_counter()
        best = len(self.forbidden) + 1
        best_rows: list[np.ndarray] | None = None
        total_sweeps = 0
        starts: list[Sequence[np.ndarray] | None] = []
        if initial_rows is not None:
            starts.append([np.asarray(row, dtype=np.int64) for row in initial_rows])
        starts.extend([None] * restarts)
        for restart, initial in enumerate(starts):
            killed, rows, sweeps = self.descend(
                initial,
                max_sweeps=max_sweeps,
                top=top,
            )
            total_sweeps += sweeps
            if killed < best:
                best = killed
                best_rows = [row.copy() for row in rows]
                elapsed = time.perf_counter() - start
                print(
                    f"  primary best={best} restart={restart} "
                    f"sweeps={sweeps} elapsed={elapsed:.2f}s",
                    flush=True,
                )
            if best == 0:
                break
            if progress_every and (restart + 1) % progress_every == 0:
                print(
                    f"  primary progress {restart + 1}/{len(starts)} "
                    f"best={best}",
                    flush=True,
                )
        assert best_rows is not None
        exact_killed = int(
            killed_mask(self.forbidden, best_rows, self.moduli).sum()
        )
        if exact_killed != best:
            raise AssertionError(f"score mismatch: search={best}, exact={exact_killed}")
        index = image_size(best_rows, self.moduli, self.n)
        return SearchResult(
            killed=exact_killed,
            rows=best_rows,
            moduli=self.moduli.copy(),
            image_index=index,
            restarts=len(starts),
            sweeps=total_sweeps,
            seconds=time.perf_counter() - start,
        )

    def run_weighted(
        self,
        weights: Sequence[float] | np.ndarray,
        *,
        restarts: int = 100,
        max_sweeps: int = 20,
        top: int = 8,
        progress_every: int = 10,
        initial_rows: Sequence[Sequence[int]] | None = None,
    ) -> WeightedSearchResult:
        """Run block descent for a nonnegative geometric conflict loss.

        The final killed count is still checked exactly.  Weights only guide
        the search toward conflicts that are easier to remove by a subsequent
        lattice-metric deformation.
        """
        weights_array = np.asarray(weights, dtype=np.float64)
        if weights_array.shape != (len(self.forbidden),):
            raise ValueError(
                f"weights must have shape {(len(self.forbidden),)}, "
                f"got {weights_array.shape}"
            )
        if not np.all(np.isfinite(weights_array)) or np.any(weights_array < 0):
            raise ValueError("weights must be finite and nonnegative")

        start = time.perf_counter()
        best_loss = float("inf")
        best_rows: list[np.ndarray] | None = None
        total_sweeps = 0
        starts: list[Sequence[np.ndarray] | None] = []
        if initial_rows is not None:
            starts.append([np.asarray(row, dtype=np.int64) for row in initial_rows])
        starts.extend([None] * restarts)
        for restart, initial in enumerate(starts):
            loss, rows, sweeps = self.descend(
                initial,
                max_sweeps=max_sweeps,
                top=top,
                weights=weights_array,
            )
            total_sweeps += sweeps
            if weighted_improves(loss, best_loss):
                best_loss = loss
                best_rows = [row.copy() for row in rows]
                killed = int(
                    killed_mask(self.forbidden, best_rows, self.moduli).sum()
                )
                print(
                    f"  weighted best={best_loss:.9g} killed={killed} "
                    f"restart={restart} sweeps={sweeps} "
                    f"elapsed={time.perf_counter()-start:.2f}s",
                    flush=True,
                )
            if best_loss == 0.0:
                break
            if progress_every and (restart + 1) % progress_every == 0:
                print(
                    f"  weighted progress {restart + 1}/{len(starts)} "
                    f"best={best_loss:.9g}",
                    flush=True,
                )
        assert best_rows is not None
        final_mask = killed_mask(self.forbidden, best_rows, self.moduli)
        exact_loss = float(weights_array[final_mask].sum())
        if not weighted_close(exact_loss, best_loss):
            raise AssertionError(
                f"weighted score mismatch: search={best_loss}, exact={exact_loss}"
            )
        index = image_size(best_rows, self.moduli, self.n)
        return WeightedSearchResult(
            weighted_loss=exact_loss,
            killed=int(final_mask.sum()),
            rows=best_rows,
            moduli=self.moduli.copy(),
            image_index=index,
            restarts=len(starts),
            sweeps=total_sweeps,
            seconds=time.perf_counter() - start,
        )

    def run_weighted_archive(
        self,
        weights: Sequence[float] | np.ndarray,
        *,
        archive_size: int = 16,
        restarts: int = 100,
        max_sweeps: int = 20,
        top: int = 8,
        progress_every: int = 10,
        initial_rows: Sequence[Sequence[int]] | None = None,
    ) -> list[WeightedSearchResult]:
        """Return HNF-distinct local optima instead of only the best restart."""
        if archive_size < 1 or restarts < 0:
            raise ValueError("archive size must be positive")
        weights_array = np.asarray(weights, dtype=np.float64)
        if weights_array.shape != (len(self.forbidden),):
            raise ValueError(
                f"weights must have shape {(len(self.forbidden),)}, "
                f"got {weights_array.shape}"
            )
        if not np.all(np.isfinite(weights_array)) or np.any(weights_array < 0):
            raise ValueError("weights must be finite and nonnegative")

        start = time.perf_counter()
        starts: list[Sequence[np.ndarray] | None] = []
        if initial_rows is not None:
            starts.append(
                [np.asarray(row, dtype=np.int64) for row in initial_rows]
            )
        starts.extend([None] * restarts)
        archive: dict[
            tuple[int, ...], tuple[float, int, list[np.ndarray]]
        ] = {}
        total_sweeps = 0
        best_loss = float("inf")
        for restart, initial in enumerate(starts):
            _, rows, sweeps = self.descend(
                initial,
                max_sweeps=max_sweeps,
                top=top,
                weights=weights_array,
            )
            total_sweeps += sweeps
            rows = [np.asarray(row, dtype=np.int64).copy() for row in rows]
            mask = killed_mask(self.forbidden, rows, self.moduli)
            exact_loss = float(weights_array[mask].sum())
            kernel = hnf_columns(
                kernel_basis(rows, self.moduli, self.n)
            )
            key = tuple(int(value) for value in kernel.flat)
            incumbent = archive.get(key)
            if incumbent is None or weighted_improves(
                exact_loss, incumbent[0]
            ):
                archive[key] = (
                    exact_loss,
                    int(mask.sum()),
                    [row.copy() for row in rows],
                )
            if weighted_improves(exact_loss, best_loss):
                best_loss = exact_loss
                print(
                    f"  weighted archive best={best_loss:.9g} "
                    f"killed={int(mask.sum())} restart={restart} "
                    f"unique={len(archive)}",
                    flush=True,
                )
            if progress_every and (restart + 1) % progress_every == 0:
                print(
                    f"  weighted archive progress "
                    f"{restart + 1}/{len(starts)} "
                    f"best={best_loss:.9g} unique={len(archive)}",
                    flush=True,
                )

        ranked = sorted(
            archive.values(),
            key=lambda item: (item[0], item[1]),
        )[:archive_size]
        elapsed = time.perf_counter() - start
        results: list[WeightedSearchResult] = []
        for exact_loss, killed, rows in ranked:
            index = image_size(rows, self.moduli, self.n)
            results.append(
                WeightedSearchResult(
                    weighted_loss=exact_loss,
                    killed=killed,
                    rows=[row.copy() for row in rows],
                    moduli=self.moduli.copy(),
                    image_index=index,
                    restarts=len(starts),
                    sweeps=total_sweeps,
                    seconds=elapsed,
                )
            )
        return results


def load_forbidden(name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (basis, F, diameter) for a catalog lattice or ABPR E7 coordinates."""
    if name == "E7*-ABPR":
        from chromatic_research.campaigns.beat_e7 import build

        _, _, basis, diameter, forbidden = build()
        return np.asarray(basis), np.asarray(forbidden, dtype=np.int64), float(diameter)

    # In dimensions 8 and 9, recomputing F dominates a campaign (minutes rather
    # than seconds).  Use the exact covering-radius ratios for the two record
    # parent lattices and cache only the reproducible enumeration.  This also
    # avoids treating the lower estimate returned by random-direction LPs as
    # an exact diameter.
    exact_diameter_ratios = {
        "E8": math.sqrt(2.0),
        "A9*": math.sqrt(11.0 / 3.0),
    }
    if name in exact_diameter_ratios:
        cache_dir = results_path(".search-cache")
        cache_path = cache_dir / f"forbidden-{name.replace('*', 'star')}-v1.npz"
        if cache_path.exists():
            with np.load(cache_path) as payload:
                basis = np.asarray(payload["basis"], dtype=np.float64)
                forbidden = np.asarray(payload["forbidden"], dtype=np.int64)
                diameter = float(payload["diameter"])
            print(
                f"forbidden cache: {name} |F|={len(forbidden)} "
                f"from {cache_path}",
                flush=True,
            )
            return basis, forbidden, diameter

        import combigeo
        from chromatic_research.core.lattices import CATALOG

        basis = np.asarray(CATALOG[name](), dtype=np.float64)
        shortest = float(
            np.linalg.norm(combigeo.shortest_vector(basis.tolist()))
        )
        diameter = exact_diameter_ratios[name] * shortest
        forbidden = np.asarray(
            combigeo.forbidden_coords(basis.tolist(), diameter, 1.0),
            dtype=np.int64,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            basis=basis,
            forbidden=forbidden.astype(np.int16),
            diameter=np.asarray(diameter),
        )
        print(
            f"forbidden cache: built {name} |F|={len(forbidden)} "
            f"at {cache_path}",
            flush=True,
        )
        return basis, forbidden, diameter

    from chromatic_research.core.campaign_hd import prep

    basis, diameter, forbidden = prep(name)
    return (
        np.asarray(basis, dtype=np.float64),
        np.asarray(forbidden, dtype=np.int64),
        float(diameter),
    )


def known_e7_rows() -> tuple[list[np.ndarray], list[int]]:
    """Primary characters annihilating the published C7 witness."""
    from chromatic_research.core.e7_abpr import C7

    rows7 = nullspace_mod(np.asarray(C7, dtype=np.int64).T, 7)
    forms4 = projective_forms(7, 4)
    annihilators4 = forms4[
        np.all((forms4 @ np.asarray(C7, dtype=np.int64)) % 4 == 0, axis=1)
    ]
    if len(rows7) != 3 or not len(annihilators4):
        raise AssertionError("could not reconstruct primary characters of C7")
    rows = [row.copy() for row in rows7] + [annihilators4[0].copy()]
    return rows, [7, 7, 7, 4]


def parse_moduli(text: str) -> list[int]:
    values = json.loads(text)
    if not isinstance(values, list) or not values:
        raise argparse.ArgumentTypeError("moduli must be a non-empty JSON list")
    return [int(value) for value in values]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lattice", help="catalog name, or E7*-ABPR")
    parser.add_argument("moduli", type=parse_moduli, help='row moduli, e.g. "[7,7,7,4]"')
    parser.add_argument("--restarts", type=int, default=100)
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--known-e7-seed", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    _, forbidden, diameter = load_forbidden(args.lattice)
    print(
        f"{args.lattice}: n={forbidden.shape[1]} |F|={len(forbidden)} "
        f"diam={diameter:.12g} moduli={args.moduli}",
        flush=True,
    )
    initial = None
    if args.known_e7_seed:
        initial, known_moduli = known_e7_rows()
        if known_moduli != args.moduli:
            parser.error(f"known E7 seed requires moduli {known_moduli}")
    search = PrimarySearch(forbidden, args.moduli, seed=args.seed)
    result = search.run(
        restarts=args.restarts,
        max_sweeps=args.sweeps,
        top=args.top,
        initial_rows=initial,
    )
    payload = {
        "lattice": args.lattice,
        "n": forbidden.shape[1],
        "n_forbidden": len(forbidden),
        "diameter": diameter,
        **result.as_json(),
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"saved {args.output}", flush=True)
    return 0 if result.found else 2


if __name__ == "__main__":
    raise SystemExit(main())
