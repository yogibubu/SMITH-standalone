"""Native quadratic atlas for local SONIC coordinate families.

The atlas is evaluated on local SONIC primitives before point-group
symmetrization.  Each minimum family currently carries a scalar curvature, so
orthogonal symmetrization within a homogeneous family leaves the Hessian block
equal to ``k I``.  Estimated seeds must never be obtained by a primitive,
Cartesian, metric, or pseudoinverse congruence transformation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .policy import PRIMITIVE_FAMILY_POLICIES, TS_REACTIVE_COORDINATE_CLASS


SONIC_FORCE_CONSTANT_ATLAS_SCHEMA = "matrix.smith.sonic_hessian_atlas.v1"
MINIMUM_REGIME = "MINIMUM"
TRANSITION_STATE_REGIME = "TRANSITION_STATE"
SONIC_FORCE_CONSTANT_REGIMES = frozenset({MINIMUM_REGIME, TRANSITION_STATE_REGIME})
STABLE_MODE = "STABLE"
REACTION_MODE = "REACTION"
SONIC_CURVATURE_ROLES = frozenset({STABLE_MODE, REACTION_MODE})
FAMILY_CONSTANT_EQUILIBRIUM = "FAMILY_CONSTANT"
COVALENT_RADII_BOND_ORDER_EQUILIBRIUM = "COVALENT_RADII_BOND_ORDER"
SONIC_EQUILIBRIUM_MODELS = frozenset(
    {FAMILY_CONSTANT_EQUILIBRIUM, COVALENT_RADII_BOND_ORDER_EQUILIBRIUM}
)
STRETCH_BOND_ORDER_EXPONENT = -0.16
HARMONIC_POTENTIAL = "HARMONIC"
PERIODIC_COSINE_POTENTIAL = "PERIODIC_COSINE"
SONIC_POTENTIAL_MODELS = frozenset({HARMONIC_POTENTIAL, PERIODIC_COSINE_POTENTIAL})


@dataclass(frozen=True)
class SonicAtomDescriptor:
    """Frozen chemical descriptors delivered to the atlas for one atom."""

    atom_index: int
    atomic_number: int
    effective_atomic_number: float
    partial_charge: float

    def __post_init__(self) -> None:
        index = int(self.atom_index)
        number = int(self.atomic_number)
        effective = float(self.effective_atomic_number)
        charge = float(self.partial_charge)
        if index < 1 or number < 1:
            raise ValueError("SONIC atom indexes and atomic numbers must be positive")
        if not math.isfinite(effective) or effective <= 0.0:
            raise ValueError("SONIC effective atomic numbers must be finite and positive")
        if not math.isfinite(charge):
            raise ValueError("SONIC atomic charges must be finite")
        object.__setattr__(self, "atom_index", index)
        object.__setattr__(self, "atomic_number", number)
        object.__setattr__(self, "effective_atomic_number", effective)
        object.__setattr__(self, "partial_charge", charge)


@dataclass(frozen=True)
class SonicBondDescriptor:
    """Frozen bond-order descriptor for atoms in one local SONIC support."""

    atom_index_a: int
    atom_index_b: int
    order: float

    def __post_init__(self) -> None:
        left, right = sorted((int(self.atom_index_a), int(self.atom_index_b)))
        order = float(self.order)
        if left < 1 or left == right or not math.isfinite(order) or order <= 0.0:
            raise ValueError("SONIC bonds require distinct atoms and positive finite order")
        object.__setattr__(self, "atom_index_a", left)
        object.__setattr__(self, "atom_index_b", right)
        object.__setattr__(self, "order", order)


@dataclass(frozen=True)
class SonicCoordinateEnvironment:
    """Typed atoms and local bonded graph supplied for one SONIC primitive."""

    atoms: tuple[SonicAtomDescriptor, ...]
    bonds: tuple[SonicBondDescriptor, ...] = ()

    def __post_init__(self) -> None:
        atoms = tuple(self.atoms)
        bonds = tuple(self.bonds)
        indexes = tuple(atom.atom_index for atom in atoms)
        if not atoms or len(set(indexes)) != len(indexes):
            raise ValueError("a SONIC environment needs unique typed atoms")
        allowed = set(indexes)
        pairs: set[tuple[int, int]] = set()
        for bond in bonds:
            pair = (bond.atom_index_a, bond.atom_index_b)
            if not set(pair).issubset(allowed) or pair in pairs:
                raise ValueError("SONIC environment bonds must be unique and locally supported")
            pairs.add(pair)
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "bonds", bonds)

    def bond_order(self, atom_index_a: int, atom_index_b: int) -> float:
        pair = tuple(sorted((int(atom_index_a), int(atom_index_b))))
        matches = tuple(
            bond.order for bond in self.bonds if (bond.atom_index_a, bond.atom_index_b) == pair
        )
        if len(matches) != 1:
            raise ValueError(f"SONIC environment has no unique bond order for {pair}")
        return matches[0]


@dataclass(frozen=True)
class SonicFamilyForceConstant:
    """Native diagonal curvature and equilibrium model for one SONIC family."""

    family: str
    force_constant: float
    unit: str
    equilibrium_value: float | None
    equilibrium_unit: str
    equilibrium_model: str
    regimes: frozenset[str]
    model: str
    rationale: str
    curvature_role: str = STABLE_MODE
    potential_model: str = HARMONIC_POTENTIAL
    periodicity: int = 0

    def __post_init__(self) -> None:
        family = str(self.family).strip().upper()
        regimes = frozenset(str(item).strip().upper() for item in self.regimes)
        force_constant = float(self.force_constant)
        equilibrium_model = str(self.equilibrium_model).strip().upper()
        equilibrium_value = (
            None if self.equilibrium_value is None else float(self.equilibrium_value)
        )
        role = str(self.curvature_role).strip().upper()
        potential_model = str(self.potential_model).strip().upper()
        periodicity = int(self.periodicity)
        if not family:
            raise ValueError("SONIC force-constant family must be nonempty")
        if not regimes or not regimes.issubset(SONIC_FORCE_CONSTANT_REGIMES):
            raise ValueError(f"invalid SONIC force-constant regimes for {family}")
        if role not in SONIC_CURVATURE_ROLES:
            raise ValueError(f"invalid SONIC curvature role for {family}")
        if not math.isfinite(force_constant) or force_constant == 0.0:
            raise ValueError(f"SONIC force constant for {family} must be finite and nonzero")
        if role == STABLE_MODE and force_constant <= 0.0:
            raise ValueError(f"stable SONIC force constant for {family} must be positive")
        if role == REACTION_MODE and (
            force_constant >= 0.0 or regimes != frozenset({TRANSITION_STATE_REGIME})
        ):
            raise ValueError(f"reactive SONIC curvature for {family} must be negative and TS-only")
        if equilibrium_model not in SONIC_EQUILIBRIUM_MODELS:
            raise ValueError(f"invalid SONIC equilibrium model for {family}")
        if equilibrium_model == FAMILY_CONSTANT_EQUILIBRIUM:
            if equilibrium_value is None or not math.isfinite(equilibrium_value):
                raise ValueError(f"SONIC equilibrium value for {family} must be finite")
        elif equilibrium_value is not None:
            raise ValueError(f"typed SONIC equilibrium model for {family} cannot store one scalar")
        if not str(self.unit).strip() or not str(self.equilibrium_unit).strip():
            raise ValueError(f"SONIC quadratic model for {family} lacks units")
        if not str(self.model).strip() or not str(self.rationale).strip():
            raise ValueError(f"SONIC quadratic model for {family} lacks provenance")
        if potential_model not in SONIC_POTENTIAL_MODELS:
            raise ValueError(f"invalid SONIC potential model for {family}")
        if potential_model == HARMONIC_POTENTIAL and periodicity != 0:
            raise ValueError(f"harmonic SONIC family {family} cannot have periodicity")
        if potential_model == PERIODIC_COSINE_POTENTIAL and periodicity < 1:
            raise ValueError(f"periodic SONIC family {family} needs positive periodicity")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "force_constant", force_constant)
        object.__setattr__(self, "equilibrium_value", equilibrium_value)
        object.__setattr__(self, "equilibrium_model", equilibrium_model)
        object.__setattr__(self, "regimes", regimes)
        object.__setattr__(self, "curvature_role", role)
        object.__setattr__(self, "potential_model", potential_model)
        object.__setattr__(self, "periodicity", periodicity)

    def resolve_equilibrium(self, environment: SonicCoordinateEnvironment) -> float:
        """Resolve q0 from the complete frozen local chemical environment."""

        if not isinstance(environment, SonicCoordinateEnvironment):
            raise TypeError("the SONIC atlas requires a typed coordinate environment")
        if self.equilibrium_model == FAMILY_CONSTANT_EQUILIBRIUM:
            assert self.equilibrium_value is not None
            return self.equilibrium_value
        if self.equilibrium_model == COVALENT_RADII_BOND_ORDER_EQUILIBRIUM:
            if len(environment.atoms) != 2:
                raise ValueError(f"{self.family} equilibrium requires exactly two atoms")
            from matrix_chem.topology.covalent_radii import covalent_radius

            left, right = environment.atoms
            radii = (covalent_radius(left.atomic_number), covalent_radius(right.atomic_number))
            if any(radius is None or float(radius) <= 0.0 for radius in radii):
                raise ValueError(f"{self.family} equilibrium lacks covalent-radius data")
            order = environment.bond_order(left.atom_index, right.atom_index)
            return float(sum(float(radius) for radius in radii)) * order ** (
                STRETCH_BOND_ORDER_EXPONENT
            )
        raise AssertionError(f"unhandled SONIC equilibrium model {self.equilibrium_model}")


@dataclass(frozen=True)
class SonicDiagonalQuadraticEvaluation:
    """Calculator-independent evaluation of the native diagonal SONIC model."""

    energy_hartree: float
    gradient: tuple[float, ...]
    hessian_diagonal: tuple[float, ...]
    equilibrium_values: tuple[float, ...]
    displacements: tuple[float, ...]


@dataclass(frozen=True)
class SonicForceConstantAtlas:
    """Immutable registry extendible with new families or chemical typings."""

    entries: tuple[SonicFamilyForceConstant, ...]
    schema: str = SONIC_FORCE_CONSTANT_ATLAS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SONIC_FORCE_CONSTANT_ATLAS_SCHEMA:
            raise ValueError(f"unsupported SONIC force-constant atlas schema: {self.schema}")
        entries = tuple(self.entries)
        seen: set[tuple[str, str]] = set()
        for entry in entries:
            if not isinstance(entry, SonicFamilyForceConstant):
                raise TypeError("SONIC force-constant atlas entries must be typed records")
            for regime in entry.regimes:
                key = (entry.family, regime)
                if key in seen:
                    raise ValueError(f"duplicate SONIC atlas entry for {entry.family}/{regime}")
                seen.add(key)
        object.__setattr__(self, "entries", entries)

    def entry(self, family: str, *, regime: str = MINIMUM_REGIME) -> SonicFamilyForceConstant:
        normalized_family = str(family).strip().upper()
        normalized_regime = str(regime).strip().upper()
        matches = tuple(
            item
            for item in self.entries
            if item.family == normalized_family and normalized_regime in item.regimes
        )
        if len(matches) != 1:
            raise KeyError(
                f"SONIC atlas has no unique {normalized_family}/{normalized_regime} entry"
            )
        return matches[0]

    def diagonal(
        self,
        families: Sequence[str],
        *,
        regime: str = MINIMUM_REGIME,
    ) -> tuple[float, ...]:
        """Return the local-family diagonal with no basis transformation."""

        return tuple(self.entry(family, regime=regime).force_constant for family in families)

    def equilibria(
        self,
        families: Sequence[str],
        environments: Sequence[SonicCoordinateEnvironment],
        *,
        regime: str = MINIMUM_REGIME,
    ) -> tuple[float, ...]:
        """Resolve q0 after receiving complete atom and bond descriptors."""

        contexts = tuple(environments)
        if len(contexts) != len(families):
            raise ValueError("SONIC family and environment lists must have equal length")
        return tuple(
            self.entry(family, regime=regime).resolve_equilibrium(environment)
            for family, environment in zip(families, contexts)
        )

    def evaluate_quadratic(
        self,
        coordinate_values: Sequence[float],
        families: Sequence[str],
        environments: Sequence[SonicCoordinateEnvironment],
        *,
        regime: str = MINIMUM_REGIME,
    ) -> SonicDiagonalQuadraticEvaluation:
        """Evaluate E=1/2 sum_i k_i(q_i-q0_i)^2 directly in local SONICs."""

        values = tuple(float(value) for value in coordinate_values)
        diagonal = self.diagonal(families, regime=regime)
        equilibria = self.equilibria(families, environments, regime=regime)
        if len(values) != len(diagonal) or not all(math.isfinite(value) for value in values):
            raise ValueError("SONIC coordinate and family lists must be finite and equal length")
        delta = tuple(value - equilibrium for value, equilibrium in zip(values, equilibria))
        gradient = tuple(k * value for k, value in zip(diagonal, delta))
        energy = 0.5 * sum(value * derivative for value, derivative in zip(delta, gradient))
        return SonicDiagonalQuadraticEvaluation(
            energy_hartree=float(energy),
            gradient=gradient,
            hessian_diagonal=diagonal,
            equilibrium_values=equilibria,
            displacements=delta,
        )

    def evaluate_force_field(
        self,
        coordinate_values: Sequence[float],
        families: Sequence[str],
        environments: Sequence[SonicCoordinateEnvironment],
        *,
        regime: str = MINIMUM_REGIME,
    ) -> SonicDiagonalQuadraticEvaluation:
        """Evaluate harmonic and periodic family terms directly in local SONICs.

        For a periodic family the convention is
        ``E = (k/n^2) [1-cos(n(q-q0))]``.  Thus the stored ``k`` is the
        equilibrium Hessian curvature and can also seed the optimizer without
        an extra periodicity-dependent conversion.
        """

        values = tuple(float(value) for value in coordinate_values)
        entries = tuple(self.entry(family, regime=regime) for family in families)
        equilibria = self.equilibria(families, environments, regime=regime)
        if len(values) != len(entries) or not all(math.isfinite(value) for value in values):
            raise ValueError("SONIC coordinate and family lists must be finite and equal length")
        displacements = tuple(value - q0 for value, q0 in zip(values, equilibria))
        energy = 0.0
        gradient: list[float] = []
        hessian: list[float] = []
        for entry, displacement in zip(entries, displacements):
            k = entry.force_constant
            if entry.potential_model == HARMONIC_POTENTIAL:
                energy += 0.5 * k * displacement * displacement
                gradient.append(k * displacement)
                hessian.append(k)
            else:
                n = entry.periodicity
                phase = n * displacement
                energy += k / (n * n) * (1.0 - math.cos(phase))
                gradient.append(k / n * math.sin(phase))
                hessian.append(k * math.cos(phase))
        return SonicDiagonalQuadraticEvaluation(
            energy_hartree=float(energy),
            gradient=tuple(gradient),
            hessian_diagonal=tuple(hessian),
            equilibrium_values=equilibria,
            displacements=displacements,
        )

    def extended(self, entries: Iterable[SonicFamilyForceConstant]) -> "SonicForceConstantAtlas":
        """Return a new atlas with additional non-conflicting typed records."""

        return SonicForceConstantAtlas((*self.entries, *tuple(entries)))

    def records_for(self, *, regime: str) -> tuple[SonicFamilyForceConstant, ...]:
        normalized = str(regime).strip().upper()
        return tuple(item for item in self.entries if normalized in item.regimes)


def sonic_atom_descriptors_from_xyzin(
    path: Path | str,
    atoms: Sequence[str],
) -> tuple[SonicAtomDescriptor, ...]:
    """Read charge and Z_eff from the frozen ORACLE state for atlas typing."""

    from matrix_chem import preferred_atomic_charges_from_xyzin
    from matrix_chem.topology.elements import atomic_number
    from matrix_core import read_sectioned_lines, section_content

    symbols = tuple(str(atom) for atom in atoms)
    charges, _source = preferred_atomic_charges_from_xyzin(path, len(symbols))
    content = section_content(read_sectioned_lines(Path(path)), "SYNTHONS")
    columns: tuple[str, ...] = ()
    for raw in content:
        fields = raw.split()
        if fields and fields[0].upper() == "COLUMNS":
            columns = tuple(field.upper() for field in fields[1:])
            break
    if "ATOM" not in columns or "ZEFF" not in columns:
        raise ValueError("#SYNTHONS must provide complete ATOM and ZEFF columns to the atlas")
    atom_column, zeff_column = columns.index("ATOM"), columns.index("ZEFF")
    effective: dict[int, float] = {}
    for raw in content:
        fields = raw.split()
        if not fields or not fields[0].isdigit() or len(fields) < len(columns):
            continue
        index = int(fields[atom_column])
        effective[index] = float(fields[zeff_column])
    if set(effective) != set(range(1, len(symbols) + 1)):
        raise ValueError("#SYNTHONS must provide one ZEFF value for every atom")
    return tuple(
        SonicAtomDescriptor(
            atom_index=index,
            atomic_number=atomic_number(symbol),
            effective_atomic_number=effective[index],
            partial_charge=float(charges[index - 1]),
        )
        for index, symbol in enumerate(symbols, start=1)
    )


def sonic_coordinate_environment(
    atom_indices: Sequence[int],
    atom_descriptors: Sequence[SonicAtomDescriptor],
    bond_orders: Mapping[tuple[int, int], float],
) -> SonicCoordinateEnvironment:
    """Select the complete local typing context for one primitive support."""

    indexes = tuple(dict.fromkeys(int(index) for index in atom_indices))
    by_index = {atom.atom_index: atom for atom in atom_descriptors}
    try:
        local_atoms = tuple(by_index[index] for index in indexes)
    except KeyError as exc:
        raise ValueError(f"SONIC coordinate references untyped atom {exc.args[0]}") from exc
    local_set = set(indexes)
    bonds = tuple(
        SonicBondDescriptor(left, right, order)
        for (left, right), order in sorted(bond_orders.items())
        if left in local_set and right in local_set
    )
    return SonicCoordinateEnvironment(atoms=local_atoms, bonds=bonds)


def sonic_primitive_environments_from_xyzin(
    path: Path | str,
    atoms: Sequence[str],
    definition: object,
) -> tuple[SonicCoordinateEnvironment, ...]:
    """Build atlas inputs for every frozen local primitive in definition order."""

    from matrix_core import read_sectioned_lines

    from .topology_io import topology_bond_orders_from_lines

    primitives = tuple(definition.primitives)
    by_identifier = {primitive.identifier: primitive for primitive in primitives}
    if len(by_identifier) != len(primitives):
        raise ValueError("SONIC primitive identifiers must be unique for SH typing")
    atom_descriptors = sonic_atom_descriptors_from_xyzin(path, atoms)
    bond_orders = topology_bond_orders_from_lines(
        read_sectioned_lines(Path(path)),
        natoms=len(tuple(atoms)),
    )
    cache: dict[str, tuple[int, ...]] = {}

    def support(identifier: str, visiting: frozenset[str] = frozenset()) -> tuple[int, ...]:
        if identifier in cache:
            return cache[identifier]
        if identifier in visiting:
            raise ValueError(f"cyclic SONIC primitive reference in {identifier}")
        try:
            primitive = by_identifier[identifier]
        except KeyError as exc:
            raise ValueError(f"unknown SONIC primitive reference {identifier}") from exc
        indexes = list(
            dict.fromkeys(
                (
                    *primitive.atoms,
                    *primitive.ref_atoms,
                    *primitive.frame_atoms,
                    *primitive.ref_frame_atoms,
                )
            )
        )
        for reference in primitive.refs:
            if reference in by_identifier:
                indexes.extend(support(reference, visiting | {identifier}))
        result = tuple(dict.fromkeys(int(index) for index in indexes))
        if not result:
            raise ValueError(f"SONIC primitive {identifier} has no atom support for SH typing")
        cache[identifier] = result
        return result

    return tuple(
        sonic_coordinate_environment(
            support(primitive.identifier),
            atom_descriptors,
            bond_orders,
        )
        for primitive in primitives
    )


def _minimum(
    family: str,
    force_constant: float,
    force_unit: str,
    equilibrium_value: float | None,
    equilibrium_unit: str,
    rationale: str,
    *,
    equilibrium_model: str = FAMILY_CONSTANT_EQUILIBRIUM,
    periodicity: int = 0,
) -> SonicFamilyForceConstant:
    return SonicFamilyForceConstant(
        family=family,
        force_constant=force_constant,
        unit=force_unit,
        equilibrium_value=equilibrium_value,
        equilibrium_unit=equilibrium_unit,
        equilibrium_model=equilibrium_model,
        regimes=frozenset({MINIMUM_REGIME, TRANSITION_STATE_REGIME}),
        model="BERNY_STYLE_LOCAL_SONIC_DIAGONAL_V1",
        rationale=rationale,
        potential_model=(PERIODIC_COSINE_POTENTIAL if periodicity else HARMONIC_POTENTIAL),
        periodicity=periodicity,
    )


def _transition_state_reaction(
    family: str,
    force_constant: float,
    force_unit: str,
    equilibrium_unit: str,
    rationale: str,
) -> SonicFamilyForceConstant:
    return SonicFamilyForceConstant(
        family=family,
        force_constant=force_constant,
        unit=force_unit,
        equilibrium_value=0.0,
        equilibrium_unit=equilibrium_unit,
        equilibrium_model=FAMILY_CONSTANT_EQUILIBRIUM,
        regimes=frozenset({TRANSITION_STATE_REGIME}),
        model="PROVISIONAL_REACTIVE_LOCAL_SONIC_DIAGONAL_V1",
        rationale=rationale,
        curvature_role=REACTION_MODE,
    )


_RADIAN = "radian"
_ANGSTROM = "angstrom"
_ANGULAR_FORCE = "hartree/radian^2"
_DISTANCE_FORCE = "hartree/angstrom^2"
_TYPED_STRETCH = COVALENT_RADII_BOND_ORDER_EQUILIBRIUM


DEFAULT_SONIC_FORCE_CONSTANT_ATLAS = SonicForceConstantAtlas(
    entries=(
        _minimum(
            "STRETCH",
            1.000,
            _DISTANCE_FORCE,
            None,
            _ANGSTROM,
            "covalent stretch",
            equilibrium_model=_TYPED_STRETCH,
        ),
        _minimum(
            "LOCAL_XH_STRETCH",
            1.200,
            _DISTANCE_FORCE,
            None,
            _ANGSTROM,
            "local X--H stretch",
            equilibrium_model=_TYPED_STRETCH,
        ),
        _minimum(
            "BEND", 0.250, _ANGULAR_FORCE, 1.9106332362490186, _RADIAN, "ordinary valence bend"
        ),
        _minimum(
            "CYCLIC_BEND", 0.300, _ANGULAR_FORCE, 2.0943951023931953, _RADIAN, "ring valence bend"
        ),
        _minimum(
            "SPIRO_BEND", 0.220, _ANGULAR_FORCE, 1.9106332362490186, _RADIAN, "spiro junction bend"
        ),
        _minimum("LINEAR_BEND", 0.100, _ANGULAR_FORCE, 0.0, _RADIAN, "displacement from linearity"),
        _minimum("COUPLED_ANGULAR", 0.200, _ANGULAR_FORCE, 0.0, _RADIAN, "coupled angular mode"),
        _minimum(
            "TORSION", 0.020, _ANGULAR_FORCE, 0.0, _RADIAN, "acyclic proper torsion", periodicity=3
        ),
        _minimum(
            "PSEUDO_CYCLE_BEND", 0.180, _ANGULAR_FORCE, 0.0, _RADIAN, "pseudo-cycle angular closure"
        ),
        _minimum(
            "PSEUDO_CYCLE_TORSION",
            0.030,
            _ANGULAR_FORCE,
            0.0,
            _RADIAN,
            "pseudo-cycle torsional closure",
            periodicity=2,
        ),
        _minimum(
            "CYCLIC_TORSION", 0.040, _ANGULAR_FORCE, 0.0, _RADIAN, "cyclic torsion", periodicity=2
        ),
        _minimum(
            "RING_PUCKER_COMPONENT",
            0.050,
            _ANGULAR_FORCE,
            0.0,
            _RADIAN,
            "CHARM ring-puckering component",
        ),
        _minimum(
            "CONDENSED_RING_TORSION",
            0.060,
            _ANGULAR_FORCE,
            0.0,
            _RADIAN,
            "condensed-ring torsion",
            periodicity=2,
        ),
        _minimum("BUTTERFLY", 0.050, _ANGULAR_FORCE, 0.0, _RADIAN, "butterfly ring mode"),
        _minimum("OUT_OF_PLANE", 0.120, _ANGULAR_FORCE, 0.0, _RADIAN, "out-of-plane bend"),
        _minimum(
            "IMPROPER_DIHEDRAL", 0.120, _ANGULAR_FORCE, 0.0, _RADIAN, "improper planarity mode"
        ),
        _minimum(
            "FRAG_DISTANCE", 0.100, _DISTANCE_FORCE, 3.0, _ANGSTROM, "interfragment center distance"
        ),
        _minimum(
            "FRAG_CENTER_ATOM_DISTANCE",
            0.100,
            _DISTANCE_FORCE,
            2.5,
            _ANGSTROM,
            "fragment-center to atom distance",
        ),
        _minimum(
            "FRAG_TRANSLATION",
            0.080,
            _DISTANCE_FORCE,
            0.0,
            _ANGSTROM,
            "relative fragment translation",
        ),
        _minimum(
            "FRAG_ORIENTATION",
            0.025,
            _ANGULAR_FORCE,
            0.0,
            _RADIAN,
            "quaternion exponential-map orientation",
        ),
        _minimum(
            "CENTER_ATOM_DISTANCE",
            0.100,
            _DISTANCE_FORCE,
            2.5,
            _ANGSTROM,
            "multicenter to atom distance",
        ),
        _minimum(
            "HBOND_DISTANCE", 0.120, _DISTANCE_FORCE, 1.9, _ANGSTROM, "hydrogen-bond distance"
        ),
        _minimum(
            "PSEUDO_BOND_DISTANCE",
            0.080,
            _DISTANCE_FORCE,
            2.5,
            _ANGSTROM,
            "weak fragment-link distance",
        ),
        _minimum(
            "PSEUDO_BOND_BEND",
            0.080,
            _ANGULAR_FORCE,
            1.9106332362490186,
            _RADIAN,
            "weak fragment-link bend",
        ),
        _minimum(
            "PSEUDO_BOND_TORSION",
            0.015,
            _ANGULAR_FORCE,
            0.0,
            _RADIAN,
            "weak fragment-link torsion",
            periodicity=3,
        ),
        _minimum(
            "CARTESIAN_DISPLACEMENT",
            0.500,
            _DISTANCE_FORCE,
            0.0,
            _ANGSTROM,
            "typed-ONIC substrate displacement",
        ),
        _minimum(
            "INVERSE_DISTANCE_PAIR",
            0.100,
            "hartree*angstrom^2",
            1.0 / 3.0,
            "angstrom^-1",
            "inverse-distance projector component",
        ),
        _transition_state_reaction(
            "TS_REACTION_DISTANCE",
            -0.100,
            _DISTANCE_FORCE,
            _ANGSTROM,
            "antisymmetric breaking-minus-forming distance coordinate",
        ),
        _transition_state_reaction(
            TS_REACTIVE_COORDINATE_CLASS,
            -0.050,
            "hartree/native_coordinate^2",
            "native_coordinate",
            "special mixed-family reactive SONIC coordinate",
        ),
    )
)
DEFAULT_SH_ATLAS = DEFAULT_SONIC_FORCE_CONSTANT_ATLAS


_TS_ONLY_FAMILIES = frozenset({"TS_REACTION_DISTANCE", TS_REACTIVE_COORDINATE_CLASS})
_POLICY_MINIMUM_FAMILIES = frozenset(
    policy.family for policy in PRIMITIVE_FAMILY_POLICIES if policy.family not in _TS_ONLY_FAMILIES
)
_ATLAS_MINIMUM_FAMILIES = frozenset(
    item.family for item in DEFAULT_SONIC_FORCE_CONSTANT_ATLAS.records_for(regime=MINIMUM_REGIME)
)
_MISSING_MINIMUM_FAMILIES = _POLICY_MINIMUM_FAMILIES - _ATLAS_MINIMUM_FAMILIES
if _MISSING_MINIMUM_FAMILIES:
    raise RuntimeError(
        "default SONIC force-constant atlas is missing minimum families: "
        + ", ".join(sorted(_MISSING_MINIMUM_FAMILIES))
    )


__all__ = [
    "COVALENT_RADII_BOND_ORDER_EQUILIBRIUM",
    "DEFAULT_SONIC_FORCE_CONSTANT_ATLAS",
    "DEFAULT_SH_ATLAS",
    "FAMILY_CONSTANT_EQUILIBRIUM",
    "HARMONIC_POTENTIAL",
    "MINIMUM_REGIME",
    "REACTION_MODE",
    "PERIODIC_COSINE_POTENTIAL",
    "SONIC_CURVATURE_ROLES",
    "SONIC_EQUILIBRIUM_MODELS",
    "SONIC_FORCE_CONSTANT_ATLAS_SCHEMA",
    "SONIC_FORCE_CONSTANT_REGIMES",
    "SONIC_POTENTIAL_MODELS",
    "STABLE_MODE",
    "STRETCH_BOND_ORDER_EXPONENT",
    "SonicAtomDescriptor",
    "SonicBondDescriptor",
    "SonicCoordinateEnvironment",
    "SonicDiagonalQuadraticEvaluation",
    "SonicFamilyForceConstant",
    "SonicForceConstantAtlas",
    "TRANSITION_STATE_REGIME",
    "sonic_atom_descriptors_from_xyzin",
    "sonic_coordinate_environment",
    "sonic_primitive_environments_from_xyzin",
]
