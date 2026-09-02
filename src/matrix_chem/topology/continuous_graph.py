"""
continuous_graph.py
===================

Continuous geometry-based molecular descriptors.

Provides:
- Continuous Coordination Number (CNA)
- Coordination-dependent covalent radii (Pyykkö)
- Coordination-normalized topological edge orders (BO)
- Fixed-radius Pauling/Mayer multiplicity and sigma/pi/pi-pi indices

All quantities are continuous and suitable for differentiation.
"""

from dataclasses import dataclass
import math
import numpy as np

from .descriptor_parameters import (
    CNA_ALPHA,
    BO_LAMBDA_STRONG,
    BO_LAMBDA_WEAK,
    ALPHA_LAMBDA,
    BO_COMPONENT_RAMP_WIDTH,
    BO_PAULING_DECAY_ANGSTROM,
)
from .pykko_radii import PYYKKO
from .covalent_radii import covalent_radius as standard_rcov


class ContinuousGraph:
    """
    Minimal continuous graph used by the pipeline.

    Provides:
    - coords
    - Z
    - natoms
    - BO (topological edge-order matrix)
    - BO_MULTIPLICITY and its ordinal component matrices
    """

    def __init__(self, coords, Z, *, bond_order_overrides=None):
        self.coords = np.array(coords, dtype=float)
        self.Z = np.array(Z, dtype=int)
        self.natoms = len(self.Z)
        self.bond_order_overrides = bond_order_overrides or {}

        neighbors = [list(range(self.natoms)) for _ in range(self.natoms)]
        for i in range(self.natoms):
            neighbors[i].remove(i)

        self.BO = np.zeros((self.natoms, self.natoms))
        self.BO_MULTIPLICITY = np.zeros_like(self.BO)
        self.BO_SIGMA = np.zeros_like(self.BO)
        self.BO_PI = np.zeros_like(self.BO)
        self.BO_PI_PI = np.zeros_like(self.BO)
        cache = {}
        for i in range(self.natoms):
            for j in range(i + 1, self.natoms):
                key = (i, j) if i < j else (j, i)
                bo = self.bond_order_overrides.get(key)
                if bo is None:
                    bo = bond_order(i, j, self.Z, self.coords, neighbors, cache)
                self.BO[i, j] = self.BO[j, i] = bo
                multiplicity = self.bond_order_overrides.get(key)
                if multiplicity is None:
                    multiplicity = pauling_bond_order(i, j, self.Z, self.coords)
                self.BO_MULTIPLICITY[i, j] = self.BO_MULTIPLICITY[j, i] = multiplicity
                components = bond_order_components(multiplicity)
                self.BO_SIGMA[i, j] = self.BO_SIGMA[j, i] = components.sigma
                self.BO_PI[i, j] = self.BO_PI[j, i] = components.pi
                self.BO_PI_PI[i, j] = self.BO_PI_PI[j, i] = components.pi_pi


@dataclass(frozen=True)
class BondOrderComponents:
    """Ordinal sigma/first-pi/second-pi resolution of a multiplicity order."""

    sigma: float
    pi: float
    pi_pi: float

    @property
    def total(self) -> float:
        return float(self.sigma + self.pi + self.pi_pi)

    @property
    def total_pi(self) -> float:
        return float(self.pi + self.pi_pi)


def _c2_positive_excess(value: float, width: float) -> float:
    """Return a one-sided C2 approximation to ``max(value, 0)``.

    The polynomial joins zero at the integer bond-order boundary to the
    exact positive excess at ``width``.  It has matching first and second
    derivatives at both ends and introduces no pre-boundary pi leakage.
    """

    x = float(value)
    w = float(width)
    if not math.isfinite(x):
        raise ValueError("bond order must be finite")
    if not math.isfinite(w) or w <= 0.0:
        raise ValueError("bond-order component ramp width must be positive")
    if x <= 0.0:
        return 0.0
    if x >= w:
        return x
    t = x / w
    return w * (6.0 * t**3 - 8.0 * t**4 + 3.0 * t**5)


def bond_order_components(
    total_bond_order: float,
    *,
    ramp_width: float = BO_COMPONENT_RAMP_WIDTH,
) -> BondOrderComponents:
    """Resolve a non-negative multiplicity order into sigma, pi and pi-pi indices.

    ``pi`` is the first pi-bond occupancy and ``pi_pi`` is the second.  The
    three returned components are non-negative and sum to the input total to
    floating-point precision.
    """

    total = float(total_bond_order)
    if not math.isfinite(total):
        raise ValueError("bond order must be finite")
    total = max(total, 0.0)
    excess_one = _c2_positive_excess(total - 1.0, ramp_width)
    excess_two = _c2_positive_excess(total - 2.0, ramp_width)
    sigma = total - excess_one
    pi = excess_one - excess_two
    pi_pi = excess_two
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, total)
    values = [0.0 if abs(value) <= tolerance else float(value) for value in (sigma, pi, pi_pi)]
    if any(value < 0.0 for value in values):
        raise ArithmeticError("invalid negative bond-order component")
    return BondOrderComponents(*values)


def pauling_bond_order(
    i,
    j,
    Z,
    coords,
    *,
    decay_length: float = BO_PAULING_DECAY_ANGSTROM,
):
    """Return the radial Pauling bond-multiplicity index used by ORACLE.

    The fixed Mantina--Truhlar covalent radii intentionally differ from the
    coordination-dependent Pyykko radii used by :func:`bond_order`: the latter
    stabilize continuous topology, whereas this index must retain radial
    multiple-bond character.  External Mayer orders supersede this geometric
    estimate at the pipeline level.
    """

    decay = float(decay_length)
    if not math.isfinite(decay) or decay <= 0.0:
        raise ValueError("Pauling bond-order decay length must be positive")
    rcov_i = standard_rcov(int(Z[i]))
    rcov_j = standard_rcov(int(Z[j]))
    if rcov_i is None or rcov_j is None:
        return 0.0
    distance = float(np.linalg.norm(np.asarray(coords[i]) - np.asarray(coords[j])))
    return math.exp((float(rcov_i) + float(rcov_j) - distance) / decay)


# ============================================================
# Principal quantum number
# ============================================================


def principal_quantum_number(Z):
    if Z <= 2:
        return 1
    elif Z <= 10:
        return 2
    elif Z <= 18:
        return 3
    elif Z <= 36:
        return 4
    elif Z <= 54:
        return 5
    elif Z <= 86:
        return 6
    else:
        return 7


# ============================================================
# Continuous Coordination Number (CNA)
# ============================================================


def continuous_coordination_number(i, Z, coords, neighbors):
    Zi = Z[i]
    Ri = coords[i]
    cna = 0.0

    for j in neighbors[i]:
        Zj = Z[j]
        Rj = coords[j]
        Rij = np.linalg.norm(Ri - Rj)

        rcov_i = standard_rcov(Zi)
        rcov_j = standard_rcov(Zj)
        if rcov_i is None or rcov_j is None:
            continue

        R0 = rcov_i + rcov_j
        x = CNA_ALPHA * (R0 - Rij)
        cna += 0.5 * (1.0 + math.erf(x))

    return cna


# ============================================================
# Effective covalent radius (Pyykkö, C¹ interpolation)
# ============================================================


def _hermite(R0, R1, m, t):
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * R0 + h10 * m + h01 * R1 + h11 * m


def effective_covalent_radius(Zi, cna):
    table = PYYKKO.get(Zi, {})
    if not table:
        return standard_rcov(Zi)

    CNs = sorted(table.keys())
    Rs = [table[cn] for cn in CNs]

    if cna <= CNs[0]:
        return Rs[0]
    if cna >= CNs[-1]:
        return Rs[-1]

    for k in range(len(CNs) - 1):
        if CNs[k] <= cna <= CNs[k + 1]:
            CN0, CN1 = CNs[k], CNs[k + 1]
            R0, R1 = Rs[k], Rs[k + 1]
            break

    t = (cna - CN0) / (CN1 - CN0)
    m = (R1 - R0) / (CN1 - CN0)
    return _hermite(R0, R1, m * (CN1 - CN0), t)


# ============================================================
# Bond order
# ============================================================


def _bond_order_switched(Rij, R0):
    """
    Smoothly blend strong- and weak-decay exponentials.
    Short bonds use the strong decay, long distances use the weak decay.
    """
    if R0 <= 1.0e-12:
        return 0.0

    x = (Rij - R0) / R0
    w_strong = 0.5 * (1.0 - math.tanh(ALPHA_LAMBDA * x))
    bo_strong = math.exp((R0 - Rij) / BO_LAMBDA_STRONG)
    bo_weak = math.exp((R0 - Rij) / BO_LAMBDA_WEAK)
    return w_strong * bo_strong + (1.0 - w_strong) * bo_weak


def bond_order(i, j, Z, coords, neighbors, cache=None):
    if cache is not None:
        key = (i, j) if i < j else (j, i)
        if key in cache:
            return cache[key]

    Ri = coords[i]
    Rj = coords[j]
    Rij = np.linalg.norm(Ri - Rj)

    cna_i = continuous_coordination_number(i, Z, coords, neighbors)
    cna_j = continuous_coordination_number(j, Z, coords, neighbors)

    rcov_i = effective_covalent_radius(Z[i], cna_i)
    rcov_j = effective_covalent_radius(Z[j], cna_j)

    if rcov_i is None or rcov_j is None:
        bo = 0.0
    else:
        bo = _bond_order_switched(Rij, rcov_i + rcov_j)

    if cache is not None:
        cache[key] = bo
    return bo
