from __future__ import annotations

import argparse
from importlib.resources import as_file, files
from pathlib import Path
import tempfile

import numpy as np

from matrix_chem import (
    build_primitive_contract,
    preprocess_to_enriched_xyz,
    read_geometry_with_kind,
    write_primitive_contract,
    write_validation_section,
)
from matrix_chem.topology.elements import atomic_number
from matrix_core import read_sectioned_lines, replace_section, section_content
from matrix_fragments import (
    build_fragment_definition_from_xyzin,
    write_fragment_build_section,
    write_interaction_center_section,
)
from matrix_smith import (
    write_gic_report,
    write_gicforge_gaussian_input,
    write_sonic_build_sections_from_cartesian,
)
from matrix_smith.standalone import (
    _normalized_source_kind,
    _optional_bool,
    _optional_string,
    _pairs,
    _read_smith_input,
    _strings,
)

from . import __version__


REQUIRED_FROZEN_SECTIONS = (
    "VALIDATION",
    "TOPOLOGY",
    "SYNTHONS",
    "SYMMETRY",
    "PRIMITIVES",
)
PRESERVED_STANDALONE_SECTIONS = (
    "TOPOLOGY",
    "SYNTHONS",
    "SYMMETRY",
    "BASIC",
    "PRIMITIVES",
    "FRAGMENTS",
    "INTERACTION_CENTERS",
)
PROVENANCE_SCHEMA = "smith.sonic.standalone.v1"
IMPLEMENTATION_REVISION = "bc6d62140aab5fbcef8dd6fa7bc6c468debba69a"
DEFAULT_G16_ROUTE = "#p hf/sto-3g opt=(readallgic,calcfc,maxcycle=80)"
EXAMPLES = {
    "water": ("water.smith.xyz", False),
    "norbornane": ("norbornane.smith.xyz", False),
    "formic-acid-water": ("formic_acid_water.smith.xyz", False),
    "water-dimer": ("water_dimer.smith.xyz", False),
    "benzene-water": ("benzene_water.smith.xyz", False),
    "saccharin": ("saccharin.smith.xyz", False),
    "eta3-allyl-palladium": ("eta3_allyl_palladium.frozen.xyzin", True),
}


def _has_complete_frozen_state(path: Path) -> bool:
    lines = read_sectioned_lines(path)
    return all(section_content(lines, name) for name in REQUIRED_FROZEN_SECTIONS)


def _build(args: argparse.Namespace) -> int:
    source = Path(args.input)
    target = Path(args.output) if args.output is not None else source.with_suffix(".xyzin")
    has_frozen_state = _has_complete_frozen_state(source)
    if args.require_frozen_state and not has_frozen_state:
        missing = [
            name
            for name in REQUIRED_FROZEN_SECTIONS
            if not section_content(read_sectioned_lines(source), name)
        ]
        raise SystemExit(
            "SMITH was asked to require a complete frozen state; missing sections: "
            + ", ".join(missing)
        )

    if has_frozen_state:
        definition = write_sonic_build_sections_from_cartesian(
            source,
            target,
            improper_dihedrals=True,
        )
        profile = "FROZEN_STATE"
        fragment_profile = (
            "INPUT_SUPPLIED"
            if section_content(read_sectioned_lines(source), "FRAGMENTS")
            else "NONE"
        )
    else:
        definition, fragment_count, profile = _build_from_standalone_input(source, target)
        fragment_profile = "AUTO_CONNECTED_COMPONENTS" if fragment_count > 1 else "NONE"

    replace_section(
        target,
        "SMITH_PROVENANCE",
        [
            f"SCHEMA {PROVENANCE_SCHEMA}",
            f"SMITH_VERSION {__version__}",
            f"IMPLEMENTATION_REVISION {IMPLEMENTATION_REVISION}",
            f"PERCEPTION_PROFILE {profile}",
            f"FRAGMENT_PROFILE {fragment_profile}",
            "STATE_INTERFACE TOPOLOGY_OR_PRIMITIVES_OR_CARTESIAN",
        ],
    )
    report_path, gaussian_path = _write_sidecars(target)
    print(
        f"Wrote {target} (profile={profile}, GICs={len(definition.gics)}, "
        f"rank={definition.rank})"
    )
    print(f"Wrote coordinate report {report_path}")
    print(f"Wrote Gaussian 16 input {gaussian_path}")
    if profile == "STANDALONE_MINIMAL":
        print(
            "Note: the Cartesian input used the bundled, revision-pinned "
            "topology/symmetry/primitive perception frontend. "
            "Supply #TOPOLOGY or #PRIMITIVES to control that boundary, or use "
            "--require-frozen-state when a complete externally validated state is required."
        )
    return 0


def _build_from_standalone_input(source: Path, target: Path):
    xyz_lines, options = _read_standalone_geometry_and_options(source)
    source_lines = read_sectioned_lines(source)
    supplied = {
        name: section_content(source_lines, name)
        for name in PRESERVED_STANDALONE_SECTIONS
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smith-sonic-") as scratch:
        geometry_source = Path(scratch) / "geometry.xyz"
        geometry_source.write_text("\n".join(xyz_lines) + "\n", encoding="utf-8")
        preprocess_to_enriched_xyz(
            geometry_source,
            target,
            source_kind=_normalized_source_kind(options.get("source_kind", "auto")),
        )
    for name, content in supplied.items():
        if content:
            replace_section(target, name, content)
    if supplied["TOPOLOGY"] and not supplied["PRIMITIVES"]:
        _write_primitives_from_supplied_topology(target)
    write_validation_section(target)

    fragments = build_fragment_definition_from_xyzin(target)
    if len(fragments.fragments) > 1:
        write_fragment_build_section(target)
        write_interaction_center_section(target)

    g16_compatibility = bool(options.get("g16", True))
    improper_dihedrals = _optional_bool(options.get("improper_dihedrals"))
    if g16_compatibility:
        improper_dihedrals = True
    fragment_mode = _optional_string(options.get("fragment_mode"))
    if len(fragments.fragments) > 1 and fragment_mode is None:
        fragment_mode = "special-coordinates"
    definition = write_sonic_build_sections_from_cartesian(
        target,
        target,
        symmetrize=bool(options.get("symmetrize", False)),
        sycart=bool(options.get("sycart", False)),
        symmetry_group=_optional_string(options.get("symmetry_group")),
        improper_dihedrals=improper_dihedrals,
        fragment_mode=fragment_mode,
        xh_stretch_policy=_optional_string(options.get("xh_stretch_policy")),
        local_xh_bonds=_pairs(options.get("local_xh_bonds")),
        local_xh_classes=_strings(options.get("local_xh_classes")),
    )
    if supplied["TOPOLOGY"] and supplied["PRIMITIVES"]:
        profile = "STANDALONE_PRIMITIVES"
    elif supplied["TOPOLOGY"]:
        profile = "STANDALONE_TOPOLOGY"
    elif supplied["PRIMITIVES"]:
        profile = "STANDALONE_PRIMITIVES"
    else:
        profile = "STANDALONE_MINIMAL"
    return definition, len(fragments.fragments), profile


def _read_standalone_geometry_and_options(source: Path) -> tuple[list[str], dict[str, object]]:
    """Read SMITH directives without treating appended contract sections as options."""
    raw_lines = source.read_text(encoding="utf-8").splitlines()
    if len(raw_lines) < 2:
        raise ValueError("SMITH input must start with a standard XYZ block")
    try:
        atom_count = int(raw_lines[0].split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError("SMITH input first line must be an XYZ atom count") from exc
    xyz_end = atom_count + 2
    if len(raw_lines) < xyz_end:
        raise ValueError("SMITH input XYZ block is shorter than the atom count")

    directive_lines: list[str] = []
    in_smith_block = False
    seen_section = False
    for raw in raw_lines[xyz_end:]:
        marker = raw.strip().upper()
        if marker in {"#SMITH", "#SONIC", "$SMITH", "$SONIC"}:
            in_smith_block = True
            seen_section = True
            directive_lines.append(raw)
            continue
        if raw.strip().startswith("#"):
            in_smith_block = False
            seen_section = True
            continue
        if in_smith_block or not seen_section:
            directive_lines.append(raw)

    with tempfile.TemporaryDirectory(prefix="smith-input-") as scratch:
        filtered = Path(scratch) / "input.xyz"
        filtered.write_text(
            "\n".join(raw_lines[:xyz_end] + directive_lines) + "\n",
            encoding="utf-8",
        )
        return _read_smith_input(filtered)


class _SuppliedTopologyGraph:
    """Minimal graph adapter used only to generate redundant primitives."""

    def __init__(
        self,
        symbols: tuple[str, ...],
        coordinates_angstrom: np.ndarray,
        bonds: tuple[tuple[int, int], ...],
    ) -> None:
        numbers = [atomic_number(symbol) for symbol in symbols]
        if any(number is None for number in numbers):
            unknown = [
                symbol
                for symbol, number in zip(symbols, numbers, strict=True)
                if number is None
            ]
            raise SystemExit(
                "Unknown element labels in standalone topology input: "
                + ", ".join(unknown)
            )
        self.Z = np.asarray(numbers, dtype=int)
        self.coords = np.asarray(coordinates_angstrom, dtype=float)
        self.natoms = len(symbols)
        self.bonds = list(bonds)
        self.adjacency = [set() for _ in range(self.natoms)]
        for left, right in self.bonds:
            self.adjacency[left].add(right)
            self.adjacency[right].add(left)


def _write_primitives_from_supplied_topology(path: Path) -> None:
    lines = read_sectioned_lines(path)
    topology = section_content(lines, "TOPOLOGY")
    bonds: list[tuple[int, int]] = []
    in_bonds = False
    for raw in topology:
        text = raw.strip()
        if text.startswith("[") and text.endswith("]"):
            in_bonds = text.upper() == "[BONDS]"
            continue
        if not in_bonds or not text or text.upper() == "NONE":
            continue
        fields = text.split()
        if len(fields) >= 2 and fields[0].isdigit() and fields[1].isdigit():
            bonds.append((int(fields[0]) - 1, int(fields[1]) - 1))
    geometry = read_geometry_with_kind(path, "enriched_xyz")
    natoms = len(geometry.atoms)
    normalized = tuple(sorted({tuple(sorted(pair)) for pair in bonds}))
    if any(left < 0 or right >= natoms or left == right for left, right in normalized):
        raise SystemExit("#TOPOLOGY contains an invalid atom index")
    graph = _SuppliedTopologyGraph(
        tuple(geometry.atoms),
        np.asarray(geometry.coordinates_angstrom, dtype=float),
        normalized,
    )
    write_primitive_contract(path, build_primitive_contract(graph, graph.coords))


def _write_sidecars(target: Path) -> tuple[Path, Path]:
    report_path = target.with_suffix(".smith.out")
    gaussian_path = target.with_suffix(".g16.gjf")
    write_gic_report(target, report_path)
    write_gicforge_gaussian_input(
        target,
        gaussian_path,
        route=DEFAULT_G16_ROUTE,
        title=f"{target.stem} SMITH/SONIC Gaussian 16 optimization",
        g16_compatibility=True,
    )
    return report_path, gaussian_path


def _inspect(args: argparse.Namespace) -> int:
    path = Path(args.xyzin)
    lines = read_sectioned_lines(path)
    provenance = section_content(lines, "SMITH_PROVENANCE")
    gics = section_content(lines, "GIC")
    print(f"path: {path}")
    print(f"frozen_state: {'complete' if _has_complete_frozen_state(path) else 'incomplete'}")
    print(f"gic_section: {'present' if gics else 'missing'}")
    if provenance:
        print("provenance:")
        for line in provenance:
            print(f"  {line}")
    return 0


def _example(args: argparse.Namespace) -> int:
    filename, require_frozen_state = EXAMPLES[args.name]
    resource = files("smith_sonic.examples").joinpath(filename)
    target = Path(args.output) if args.output is not None else Path(f"{args.name}.xyzin")
    with as_file(resource) as source:
        build_args = argparse.Namespace(
            input=source,
            output=target,
            require_frozen_state=require_frozen_state,
        )
        return _build(build_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smith-sonic",
        description="Build a frozen SONIC coordinate contract with standalone SMITH.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build a SONIC contract")
    build.add_argument(
        "input",
        type=Path,
        help="SMITH XYZ, topology/primitive input, or complete frozen xyzin state",
    )
    build.add_argument("output", type=Path, nargs="?", help="Output xyzin path")
    build.add_argument(
        "--require-frozen-state",
        dest="require_frozen_state",
        action="store_true",
        help=(
            "Refuse partial/plain input and require validation/topology/synthon/"
            "symmetry/primitive sections"
        ),
    )
    build.set_defaults(func=_build)

    inspect = subparsers.add_parser("inspect", help="Inspect state and standalone provenance")
    inspect.add_argument("xyzin", type=Path)
    inspect.set_defaults(func=_inspect)

    example = subparsers.add_parser("example", help="Run an example shipped in the package")
    example.add_argument("name", choices=tuple(EXAMPLES))
    example.add_argument("output", type=Path, nargs="?", help="Output xyzin path")
    example.set_defaults(func=_example)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
