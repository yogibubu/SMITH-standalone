"""
Topology construction pipeline for ORACLE.

This module centralizes the geometry-first construction of all
topological objects and descriptor/synthon objects, without performing any I/O
or workflow decisions.

Frozen contracts:
- Does NOT write files.
- Does NOT call RDKit.
- Does NOT alter topology semantics.
"""

from .continuous_graph import ContinuousGraph
from .discrete_graph import DiscreteGraph
from .ringset import RingSet
from .atomic_synthons import AtomicSynthons
from .aromaticity import Aromaticity

# ============================================================
# Public API
# ============================================================


def build_topology_objects(
    coords,
    Z,
    *,
    bond_order_overrides=None,
    external_charges=None,
    charge_source="Synthons electronegativity model",
    bond_order_source="ORACLE continuous topology model",
    force_aromatic=False,
):
    """
    Build all topology-related objects from Cartesian coordinates.

    Parameters
    ----------
    coords : array-like, shape (N,3)
        Cartesian coordinates.
    Z : array-like, shape (N,)
        Atomic numbers.
    force_aromatic : bool, optional
        Passed to Aromaticity (no topology effect).

    Returns
    -------
    cg : ContinuousGraph
    dg : DiscreteGraph
    ringset : RingSet
    synthons : AtomicSynthons
    aromaticity : Aromaticity
    """

    # --------------------------------------------------------
    # Continuous topology (geometry-first)
    # --------------------------------------------------------
    cg = ContinuousGraph(coords, Z, bond_order_overrides=bond_order_overrides)

    # --------------------------------------------------------
    # Discrete topology (H-robust)
    # --------------------------------------------------------
    dg = DiscreteGraph(cg)

    # --------------------------------------------------------
    # Ring detection
    # --------------------------------------------------------
    ringset = RingSet(dg, coords=cg.coords)

    # --------------------------------------------------------
    # Atomic synthons (continuous descriptors)
    # --------------------------------------------------------
    neighbors = [list(dg.adjacency[i]) for i in range(dg.natoms)]
    synthons = AtomicSynthons(
        Z=cg.Z,
        coords=cg.coords,
        neighbors=neighbors,
    )
    synthons._external_charges = external_charges or None
    synthons._external_bond_orders = bond_order_overrides or None
    synthons._charge_source = charge_source
    synthons._bond_order_source = bond_order_source
    synthons._bond_multiplicity_source = (
        bond_order_source
        if bond_order_overrides
        else "ORACLE Pauling multiplicity model (Mantina-Truhlar radii; decay 0.30 A)"
    )

    # --------------------------------------------------------
    # Aromaticity (ring-driven, discrete + geometry)
    # --------------------------------------------------------
    aromaticity = Aromaticity(
        graph=cg,
        discrete_graph=dg,
        ring_set=ringset,
        synthons=synthons,
        force_aromatic=force_aromatic,
    )

    return cg, dg, ringset, synthons, aromaticity
