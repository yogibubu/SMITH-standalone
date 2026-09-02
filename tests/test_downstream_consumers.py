from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile
import unittest

MATRIX_ROOT = os.environ.get("MATRIX_ROOT")
if MATRIX_ROOT:
    packages = Path(MATRIX_ROOT).expanduser().resolve() / "packages"
    for source_root in sorted(packages.glob("*/src")):
        sys.path.append(str(source_root))

try:
    from matrix_morpheus import prepare_semiexperimental_xyzin, read_geometry_input
    from matrix_trinity import coordinate_model_from_xyzin
except ImportError:
    prepare_semiexperimental_xyzin = None
    read_geometry_input = None
    coordinate_model_from_xyzin = None

from smith_sonic.cli import main

REQUIRE_DOWNSTREAM = os.environ.get("SMITH_REQUIRE_DOWNSTREAM") == "1"
DOWNSTREAM_AVAILABLE = coordinate_model_from_xyzin is not None


@unittest.skipIf(
    not DOWNSTREAM_AVAILABLE and not REQUIRE_DOWNSTREAM,
    "full MATRIX MORPHEUS/LINK packages are not installed",
)
class DownstreamConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DOWNSTREAM_AVAILABLE:
            raise RuntimeError(
                "SMITH downstream verification requires MATRIX_ROOT to name "
                "a compatible clean MATRIX checkout"
            )

    def test_advanced_examples_are_consumed_without_translation(self) -> None:
        cases = (
            ("formic-acid-water", 8, 18),
            ("eta3-allyl-palladium", 10, 24),
        )
        with tempfile.TemporaryDirectory(prefix="smith-downstream-") as scratch:
            for name, atom_count, rank in cases:
                xyzin = Path(scratch) / f"{name}.xyzin"
                self.assertEqual(main(["example", name, str(xyzin)]), 0)

                geometry = read_geometry_input(xyzin)
                prepared = prepare_semiexperimental_xyzin(xyzin)
                link_model = coordinate_model_from_xyzin(xyzin, kind="sonic")

                self.assertEqual(geometry.source_format, "xyzin")
                self.assertEqual(len(geometry.atoms), atom_count)
                self.assertEqual(prepared.xyzin, xyzin)
                self.assertEqual(prepared.observations, ())
                self.assertEqual(
                    link_model.directions_angstrom.shape,
                    (rank, 3 * atom_count),
                )


if __name__ == "__main__":
    unittest.main()
