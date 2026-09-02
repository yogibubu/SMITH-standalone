from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from matrix_chem import preprocess_to_enriched_xyz, read_enriched_xyz, write_validation_section
from matrix_core import replace_section
from matrix_fragments import (
    AtomCenterInteractionRecord,
    InteractionCenterDefinition,
    InteractionCenterRecord,
    interaction_center_section_lines,
)


def build_fixture(source: Path, output: Path) -> Path:
    """Create a complete frozen eta3 fixture from an idealized Cartesian geometry."""
    preprocess_to_enriched_xyz(source, output, source_kind="xyz")
    geometry = read_enriched_xyz(output)
    ligand_atoms = (3, 4, 5)
    coordinates = np.asarray(geometry.coordinates_angstrom, dtype=float)
    center = tuple(float(value) for value in coordinates[[idx - 1 for idx in ligand_atoms]].mean(axis=0))
    definition = InteractionCenterDefinition(
        strategy="SUPPLIED_EXPLICIT_ETA_CENTER",
        centers=(
            InteractionCenterRecord(
                identifier="C001",
                kind="ETA3_CENTER",
                label="allyl_eta3",
                atoms=ligand_atoms,
                center=center,
                source="SUPPLIED_EXPLICIT_TEST_FIXTURE",
            ),
        ),
        interactions=(
            AtomCenterInteractionRecord(
                identifier="I001",
                kind="ATOM_ETA3_CENTER",
                atom=1,
                center_id="C001",
                score=1.0,
                source="SUPPLIED_EXPLICIT_TEST_FIXTURE",
            ),
        ),
    )
    replace_section(output, "INTERACTION_CENTERS", interaction_center_section_lines(definition))
    write_validation_section(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_fixture(args.source, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
