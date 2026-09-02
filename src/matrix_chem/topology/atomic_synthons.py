import math
import numpy as np

from .descriptor_parameters import (
    REF_ANGLE_SUM,
    BO_MIN_DESC,
    EAN_SIGMA,
)
from .pykko_radii import PYYKKO
from .covalent_radii import covalent_radius
from .vdw_radii import vdw_radius
from .continuous_graph import (
    BondOrderComponents,
    bond_order_components as resolve_bond_order_components,
    continuous_coordination_number,
    bond_order as continuous_bond_order,
    pauling_bond_order,
)


# ============================================================
# Utility
# ============================================================


def angle_between(v1, v2):
    num = np.dot(v1, v2)
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    if den < 1e-12:
        return 0.0
    c = np.clip(num / den, -1.0, 1.0)
    return math.degrees(math.acos(c))


def hermite_c1(t, y0, y1, m0, m1):
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1


def _hermite_slope(table, keys, D):
    """Return a finite-difference slope using the closest available tabulated keys."""
    if D <= keys[0]:
        return table[keys[1]] - table[keys[0]]
    if D >= keys[-1]:
        return table[keys[-1]] - table[keys[-2]]

    lower = max(k for k in keys if k < D and k in table)
    upper = min(k for k in keys if k > D and k in table)
    if upper == lower:
        return 0.0
    if lower == keys[0]:
        left = lower
    else:
        left = max(k for k in keys if k < lower and k in table)
    if upper == keys[-1]:
        right = upper
    else:
        right = min(k for k in keys if k > upper and k in table)
    return (
        0.5 * (table[upper] - table[lower]) if right == left else 0.5 * (table[right] - table[left])
    )


def principal_n(Z):
    if Z <= 2:
        return 1
    if Z <= 10:
        return 2
    if Z <= 18:
        return 3
    if Z <= 36:
        return 4
    if Z <= 54:
        return 5
    if Z <= 86:
        return 6
    return 7


def nval_main_group(Z):
    if Z == 1:
        return 1
    if 3 <= Z <= 10:
        return {3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8}[Z]
    if 11 <= Z <= 18:
        return {11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8}[Z]
    if 31 <= Z <= 36:
        return {31: 3, 32: 4, 33: 5, 34: 6, 35: 7, 36: 8}[Z]
    if 49 <= Z <= 54:
        return {49: 3, 50: 4, 51: 5, 52: 6, 53: 7, 54: 8}[Z]
    return 0


# ============================================================
# Atomic Synthons
# ============================================================


class AtomicSynthons:
    def __init__(self, Z, coords, neighbors):
        self.Z = Z
        self.coords = coords
        self.neighbors = neighbors
        self.natoms = len(Z)
        self._theta_bar = None
        self._all_neighbors = [list(range(self.natoms)) for _ in range(self.natoms)]
        for i in range(self.natoms):
            self._all_neighbors[i].remove(i)

    # --------------------------------------------------------
    # Continuous coordination number (CNA)
    # --------------------------------------------------------

    def cna(self, i):
        return continuous_coordination_number(i, self.Z, self.coords, self._all_neighbors)

    # --------------------------------------------------------
    # Effective covalent radius
    # --------------------------------------------------------

    def covalent_radius_eff(self, i):
        Zi = int(self.Z[i])
        cna = self.cna(i)

        table = PYYKKO[Zi]
        if not table:
            fallback = covalent_radius(Zi)
            if fallback is None:
                raise ValueError(f"no covalent radius is available for atomic number {Zi}")
            return float(fallback)
        Ds = sorted(table.keys())

        if cna <= Ds[0]:
            return table[Ds[0]]
        if cna >= Ds[-1]:
            return table[Ds[-1]]

        D0 = int(math.floor(cna))
        if D0 not in table:
            D0 = max(d for d in Ds if d <= D0)
        D1 = min(d for d in Ds if d > D0)

        t = (cna - D0) / (D1 - D0)
        y0 = table[D0]
        y1 = table[D1]

        def slope(D):
            return _hermite_slope(table, Ds, D)

        return hermite_c1(t, y0, y1, slope(D0), slope(D1))

    # --------------------------------------------------------
    # Bond order
    # --------------------------------------------------------

    def bond_order(self, i, j):
        ext = getattr(self, "_external_bond_orders", None)
        if ext:
            key = (i, j) if i < j else (j, i)
            if key in ext:
                return float(ext[key])
        return continuous_bond_order(i, j, self.Z, self.coords, self._all_neighbors)

    def bond_order_desc(self, i, j):
        return max(self.bond_order(i, j), BO_MIN_DESC)

    def bond_order_multiplicity(self, i, j):
        """Return Mayer BO when supplied, otherwise ORACLE's Pauling index."""
        ext = getattr(self, "_external_bond_orders", None)
        if ext:
            key = (i, j) if i < j else (j, i)
            if key in ext:
                return float(ext[key])
        return pauling_bond_order(i, j, self.Z, self.coords)

    def bond_order_components(self, i, j) -> BondOrderComponents:
        return resolve_bond_order_components(self.bond_order_multiplicity(i, j))

    def bond_order_sigma(self, i, j):
        return self.bond_order_components(i, j).sigma

    def bond_order_pi(self, i, j):
        """First pi-bond occupancy (kept as the historical public name)."""
        return self.bond_order_components(i, j).pi

    def bond_order_pi_pi(self, i, j):
        """Second pi-bond occupancy, non-zero for triple-bond character."""
        return self.bond_order_components(i, j).pi_pi

    def bond_order_total_pi(self, i, j):
        return self.bond_order_components(i, j).total_pi

    def bond_order_indices(self, i, j):
        """Return the canonical ``(sigma, pi, pi_pi)`` bond indices."""
        components = self.bond_order_components(i, j)
        return components.sigma, components.pi, components.pi_pi

    def sigma_index(self, i):
        return sum(self.bond_order_sigma(i, j) for j in self.neighbors[i])

    def pi_index(self, i):
        return sum(self.bond_order_pi(i, j) for j in self.neighbors[i])

    def pi_pi_index(self, i):
        return sum(self.bond_order_pi_pi(i, j) for j in self.neighbors[i])

    # --------------------------------------------------------
    # Electronic domains
    # --------------------------------------------------------

    def nlp_nos(self, i):
        Z = int(self.Z[i])
        nval = nval_main_group(Z)
        if nval == 0:
            return 0.0, 0.0

        CNA = self.cna(i)
        Npi = self.pi_index(i) + self.pi_pi_index(i)
        Nres = nval - CNA - Npi

        NOS = float(int(round(Nres)) % 2)
        NLP = 0.5 * (Nres - NOS)
        if NLP < 0.0:
            NLP = 0.0
        return NLP, NOS

    def electron_domains(self, i):
        CNA = self.cna(i)
        NLP, NOS = self.nlp_nos(i)
        return CNA + NLP + NOS

    def _electron_domains(self, i):
        return self.electron_domains(i)

    # --------------------------------------------------------
    # Reference angles
    # --------------------------------------------------------

    def theta_ref(self, N):
        if self._theta_bar is None:
            self._theta_bar = {
                int(k): REF_ANGLE_SUM[k] / (k * (k - 1) / 2) for k in REF_ANGLE_SUM if k >= 2
            }

        keys = sorted(self._theta_bar.keys())
        if N <= keys[0]:
            return self._theta_bar[keys[0]]
        if N >= keys[-1]:
            return self._theta_bar[keys[-1]]

        k0 = int(math.floor(N))
        if k0 not in self._theta_bar:
            k0 = max(k for k in keys if k <= k0)
        k1 = min(k for k in keys if k > k0)

        t = (N - k0) / (k1 - k0)
        y0 = self._theta_bar[k0]
        y1 = self._theta_bar[k1]

        def slope(k):
            return _hermite_slope(self._theta_bar, keys, k)

        return hermite_c1(t, y0, y1, slope(k0), slope(k1))

    # --------------------------------------------------------
    # Strain
    # --------------------------------------------------------

    def strain(self, i):
        neigh = self.neighbors[i]
        if len(neigh) < 2:
            return 0.0

        Ndom = self.electron_domains(i)
        theta0 = self.theta_ref(Ndom)
        c0 = math.cos(math.radians(theta0))

        Ri = self.coords[i]
        S = 0.0
        for a in range(len(neigh)):
            for b in range(a + 1, len(neigh)):
                v1 = self.coords[neigh[a]] - Ri
                v2 = self.coords[neigh[b]] - Ri
                c = math.cos(math.radians(angle_between(v1, v2)))
                S += (c - c0) ** 2

        return math.sqrt(S / (len(neigh) * (len(neigh) - 1) / 2))

    # --------------------------------------------------------
    # Charge / Polarizability / Hindrance
    # --------------------------------------------------------

    def charge(self, i):
        ext = getattr(self, "_external_charges", None)
        if ext and i in ext:
            return float(ext[i])
        from .electronegativity import electronegativity

        Zi = int(self.Z[i])
        chi_i = electronegativity(Zi)
        ni = principal_n(Zi)

        q = 0.0
        for j in self.neighbors[i]:
            Zj = int(self.Z[j])
            chi_j = electronegativity(Zj)
            nj = principal_n(Zj)
            bo = self.bond_order_desc(i, j)
            q += (chi_j - chi_i) / ((ni + nj) * bo)
        return q

    def polarizability(self, i):
        from .polarizability import polarizability

        Zi = int(self.Z[i])
        ai = polarizability(Zi)
        ni = principal_n(Zi)

        a = ai
        for j in self.neighbors[i]:
            Zj = int(self.Z[j])
            aj = polarizability(Zj)
            nj = principal_n(Zj)
            bo = self.bond_order_desc(i, j)
            a += (aj - ai) / ((ni + nj) * bo)
        return a

    def hindrance(self, i):
        Zi = int(self.Z[i])
        hi = vdw_radius(Zi)
        ni = principal_n(Zi)

        H = hi
        for j in self.neighbors[i]:
            Zj = int(self.Z[j])
            hj = vdw_radius(Zj)
            nj = principal_n(Zj)
            bo = self.bond_order_desc(i, j)
            H += (hj - hi) / ((ni + nj) * bo)
        return H

    # --------------------------------------------------------
    # Covalency / Delocalization
    # --------------------------------------------------------

    def covalency(self, i):
        neigh = self.neighbors[i]
        if not neigh:
            return 0.0
        ri = self.covalent_radius_eff(i)
        Ri = self.coords[i]
        C = 0.0
        for j in neigh:
            rj = self.covalent_radius_eff(j)
            d = np.linalg.norm(self.coords[j] - Ri)
            C += (ri + rj) / (d + ri + rj)
        return C / len(neigh)

    def delocalization(self, i):
        neigh = self.neighbors[i]
        if len(neigh) <= 1:
            return 0.0
        Ri = self.coords[i]
        dists = [np.linalg.norm(self.coords[j] - Ri) for j in neigh]
        dmean = sum(dists) / len(dists)
        if dmean <= 0.0:
            return 0.0
        return sum(abs(d - dmean) for d in dists) / (len(dists) * dmean)

    # --------------------------------------------------------
    # Spin / EAN / Zeff
    # --------------------------------------------------------

    def spin_density(self, i):
        _, NOS = self.nlp_nos(i)
        return NOS

    def EAN(self, i):
        vals = [
            self.charge(i),
            self.covalency(i),
            self.delocalization(i),
            self.strain(i),
        ]
        norm = [math.erf(v / EAN_SIGMA) for v in vals]
        r = math.sqrt(sum(v * v for v in norm))
        return r / (1.0 + r)

    def Zeff(self, i):
        return self.Z[i] - 0.5 + self.EAN(i)

    # --------------------------------------------------------
    # Discrete signature
    # --------------------------------------------------------

    def canonical_signature(self, i):
        D = int(round(self.cna(i)))
        NED = int(round(self.electron_domains(i)))
        return (self.Z[i], False, NED, D)

    def canonical_signature_str(self, i):
        Z, A, NED, D = self.canonical_signature(i)
        return f"{Z}-{int(A)}-{NED}-{D}"
