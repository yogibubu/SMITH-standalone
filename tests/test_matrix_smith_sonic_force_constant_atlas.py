import math
from types import SimpleNamespace

import numpy as np
import pytest


def _atom(index, atomic_number, effective, charge):
    from matrix_smith.sonic_force_constant_atlas import SonicAtomDescriptor

    return SonicAtomDescriptor(index, atomic_number, effective, charge)


def test_atlas_receives_complete_atom_typing_and_bond_order():
    from matrix_smith.sonic_force_constant_atlas import (
        DEFAULT_SONIC_FORCE_CONSTANT_ATLAS,
        SonicBondDescriptor,
        SonicCoordinateEnvironment,
    )

    carbon = _atom(1, 6, 5.72, -0.12)
    oxygen = _atom(2, 8, 7.35, 0.12)
    single = SonicCoordinateEnvironment(
        atoms=(carbon, oxygen),
        bonds=(SonicBondDescriptor(1, 2, 1.0),),
    )
    double = SonicCoordinateEnvironment(
        atoms=(carbon, oxygen),
        bonds=(SonicBondDescriptor(1, 2, 2.0),),
    )

    single_q0 = DEFAULT_SONIC_FORCE_CONSTANT_ATLAS.equilibria(("STRETCH",), (single,))[0]
    double_q0 = DEFAULT_SONIC_FORCE_CONSTANT_ATLAS.equilibria(("STRETCH",), (double,))[0]

    assert single_q0 == pytest.approx(0.75 + 0.64)
    assert double_q0 < single_q0
    assert single.atoms[0].effective_atomic_number == pytest.approx(5.72)
    assert single.atoms[0].partial_charge == pytest.approx(-0.12)


def test_periodic_torsion_curvature_includes_declared_periodicity():
    from matrix_smith.sonic_force_constant_atlas import (
        DEFAULT_SONIC_FORCE_CONSTANT_ATLAS,
        PERIODIC_COSINE_POTENTIAL,
        SonicCoordinateEnvironment,
    )

    environment = SonicCoordinateEnvironment(
        atoms=(
            _atom(1, 6, 5.9, -0.1),
            _atom(2, 6, 5.9, 0.0),
            _atom(3, 6, 5.9, 0.0),
            _atom(4, 6, 5.9, 0.1),
        )
    )
    entry = DEFAULT_SONIC_FORCE_CONSTANT_ATLAS.entry("TORSION")
    evaluation = DEFAULT_SONIC_FORCE_CONSTANT_ATLAS.evaluate_force_field(
        (1.0e-5,), ("TORSION",), (environment,)
    )

    assert entry.potential_model == PERIODIC_COSINE_POTENTIAL
    assert entry.periodicity == 3
    assert evaluation.hessian_diagonal[0] == pytest.approx(entry.force_constant * math.cos(3.0e-5))
    assert evaluation.gradient[0] / 1.0e-5 == pytest.approx(entry.force_constant, rel=1.0e-8)


def test_minimum_atlas_covers_every_nonreactive_smith_family():
    from matrix_smith.policy import PRIMITIVE_FAMILY_POLICIES
    from matrix_smith.sonic_force_constant_atlas import (
        DEFAULT_SONIC_FORCE_CONSTANT_ATLAS,
        MINIMUM_REGIME,
    )

    expected = {
        policy.family
        for policy in PRIMITIVE_FAMILY_POLICIES
        if policy.family not in {"TS_REACTION_DISTANCE", "TS_REACTIVE_MODE"}
    }
    actual = {
        entry.family
        for entry in DEFAULT_SONIC_FORCE_CONSTANT_ATLAS.records_for(regime=MINIMUM_REGIME)
    }

    assert expected <= actual
    assert {"RING_PUCKER_COMPONENT", "HBOND_DISTANCE", "FRAG_ORIENTATION"} <= actual


def test_sonic_seed_is_direct_family_diagonal_after_homogeneous_symmetrization():
    from matrix_link.optimizer import (
        OptimizerCoordinateModel,
        OptimizerSettings,
        _initial_optimizer_hessian_selection,
    )

    definition = SimpleNamespace(
        gics=(
            SimpleNamespace(identifier="g1", name="s1", family="STRETCH"),
            SimpleNamespace(identifier="g2", name="s2", family="STRETCH"),
        )
    )
    model = OptimizerCoordinateModel(
        kind="sonic",
        labels=("symmetric_stretch",),
        directions_angstrom=np.ones((1, 6)),
        metric_diagonal=np.ones(1),
        sonic_labels=("s1", "s2"),
        sonic_from_coordinates=np.asarray([[2.0**-0.5], [2.0**-0.5]]),
        sonic_definition=definition,
    )

    hessian, source = _initial_optimizer_hessian_selection(
        model,
        OptimizerSettings(initial_hessian_model="lindh_swart_special"),
        initial_hessian=None,
    )

    np.testing.assert_allclose(hessian, [[1.0]])
    assert source == "sonic-hessian-sh-diagonal-no-transform"


def test_sonic_seed_estimates_mixed_family_curvature_from_local_support():
    from matrix_link.optimizer import (
        OptimizerCoordinateModel,
        OptimizerSettings,
        _initial_optimizer_hessian_selection,
    )

    definition = SimpleNamespace(
        gics=(
            SimpleNamespace(identifier="g1", name="stretch", family="STRETCH"),
            SimpleNamespace(identifier="g2", name="bend", family="BEND"),
        )
    )
    model = OptimizerCoordinateModel(
        kind="sonic",
        labels=("mixed",),
        directions_angstrom=np.ones((1, 6)),
        metric_diagonal=np.ones(1),
        sonic_labels=("stretch", "bend"),
        sonic_from_coordinates=np.asarray([[2.0**-0.5], [2.0**-0.5]]),
        sonic_definition=definition,
    )

    hessian, source = _initial_optimizer_hessian_selection(
        model,
        OptimizerSettings(),
        initial_hessian=None,
    )

    np.testing.assert_allclose(hessian, [[0.625]])
    assert source == "sonic-hessian-sh-diagonal-no-transform"


def test_caller_qm_hessian_is_not_replaced_by_the_estimated_sonic_atlas():
    from matrix_link.optimizer import (
        OptimizerCoordinateModel,
        OptimizerSettings,
        _initial_optimizer_hessian_selection,
    )

    definition = SimpleNamespace(
        gics=(
            SimpleNamespace(identifier="g1", name="s", family="STRETCH"),
            SimpleNamespace(identifier="g2", name="b", family="BEND"),
        )
    )
    model = OptimizerCoordinateModel(
        kind="sonic",
        labels=("s", "b"),
        directions_angstrom=np.eye(2),
        metric_diagonal=np.ones(2),
        sonic_definition=definition,
    )
    supplied = np.asarray([[0.7, 0.2], [0.2, 0.4]])

    hessian, source = _initial_optimizer_hessian_selection(
        model,
        OptimizerSettings(),
        initial_hessian=supplied,
    )

    np.testing.assert_allclose(hessian, supplied)
    assert source == "caller-provided-initial-hessian"


def test_reactive_sonic_family_has_a_provisional_negative_ts_curvature():
    from matrix_link.optimizer import (
        OptimizerCoordinateModel,
        OptimizerSettings,
        _initial_optimizer_hessian_selection,
    )

    model = OptimizerCoordinateModel(
        kind="sonic",
        labels=("reaction",),
        directions_angstrom=np.ones((1, 6)),
        metric_diagonal=np.ones(1),
        sonic_definition=SimpleNamespace(
            gics=(
                SimpleNamespace(
                    identifier="reaction",
                    name="reaction",
                    family="TS_REACTION_DISTANCE",
                ),
            )
        ),
    )

    hessian, source = _initial_optimizer_hessian_selection(
        model,
        OptimizerSettings(stationary_point="transition_state"),
        initial_hessian=None,
    )

    np.testing.assert_allclose(hessian, [[-0.1]])
    assert source == "sonic-hessian-sh-diagonal-no-transform"
