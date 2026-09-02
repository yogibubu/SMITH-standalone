from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from matrix_core import read_sectioned_lines, replace_section, section_content
from matrix_smith import build_gic_b_matrix_from_xyzin, read_gic_definition_from_xyzin
from smith_sonic.cli import main


class PackagedExampleTests(unittest.TestCase):
    def _run_example(self, name: str) -> tuple[Path, list[str], tempfile.TemporaryDirectory]:
        scratch = tempfile.TemporaryDirectory(prefix=f"smith-{name}-")
        output = Path(scratch.name) / f"{name}.xyzin"
        self.assertEqual(main(["example", name, str(output)]), 0)
        return output, read_sectioned_lines(output), scratch

    def test_plain_xyz_uses_embedded_point_group_perception(self) -> None:
        _, lines, scratch = self._run_example("water")
        self.addCleanup(scratch.cleanup)
        symmetry = section_content(lines, "SYMMETRY")
        provenance = section_content(lines, "SMITH_PROVENANCE")

        self.assertIn("POINT_GROUP C2v", symmetry)
        self.assertIn("OPERATION_COUNT 4", symmetry)
        self.assertIn("PERCEPTION_PROFILE STANDALONE_MINIMAL", provenance)

    def test_formic_acid_water_has_six_interfragment_coordinates(self) -> None:
        output, lines, scratch = self._run_example("formic-acid-water")
        self.addCleanup(scratch.cleanup)
        contract = section_content(lines, "GIC")
        provenance = section_content(lines, "SMITH_PROVENANCE")

        self.assertTrue(output.is_file())
        self.assertIn("TARGET_RANK 18", contract)
        self.assertIn("RANK 18", contract)
        self.assertIn("FRAGMENT_MODE SPECIAL_COORDINATES", contract)
        self.assertEqual(
            sum(row.startswith("P") and "FAMILY=FRAG_TRANSLATION" in row for row in contract),
            3,
        )
        self.assertEqual(
            sum(row.startswith("P") and "FAMILY=FRAG_ORIENTATION" in row for row in contract),
            3,
        )
        self.assertIn("FRAGMENT_PROFILE AUTO_CONNECTED_COMPONENTS", provenance)

    def test_additional_non_covalent_examples_have_six_fragment_coordinates(self) -> None:
        for name, target_rank in (("water-dimer", 12), ("benzene-water", 39)):
            with self.subTest(example=name):
                output, lines, scratch = self._run_example(name)
                self.addCleanup(scratch.cleanup)
                contract = section_content(lines, "GIC")
                provenance = section_content(lines, "SMITH_PROVENANCE")

                self.assertTrue(output.is_file())
                self.assertIn(f"TARGET_RANK {target_rank}", contract)
                self.assertIn(f"RANK {target_rank}", contract)
                self.assertIn("FRAGMENT_MODE SPECIAL_COORDINATES", contract)
                self.assertEqual(
                    sum(
                        row.startswith("P") and "FAMILY=FRAG_TRANSLATION" in row
                        for row in contract
                    ),
                    3,
                )
                self.assertEqual(
                    sum(
                        row.startswith("P") and "FAMILY=FRAG_ORIENTATION" in row
                        for row in contract
                    ),
                    3,
                )
                self.assertIn("FRAGMENT_PROFILE AUTO_CONNECTED_COMPONENTS", provenance)

    def test_eta3_example_preserves_supplied_center_as_protected_coordinate(self) -> None:
        _, lines, scratch = self._run_example("eta3-allyl-palladium")
        self.addCleanup(scratch.cleanup)
        contract = section_content(lines, "GIC")
        centers = section_content(lines, "INTERACTION_CENTERS")
        primitives = section_content(lines, "PRIMITIVES")
        provenance = section_content(lines, "SMITH_PROVENANCE")

        self.assertIn("TARGET_RANK 24", contract)
        self.assertTrue(primitives)
        self.assertIn("RANK 24", contract)
        self.assertTrue(any("KIND=ETA3_CENTER" in row and "ATOMS=3,4,5" in row for row in centers))
        self.assertTrue(
            any(
                "FAMILY=CENTER_ATOM_DISTANCE" in row
                and "CLASS=SPECIAL_PROTECTED" in row
                and "REFS=C001,A1" in row
                for row in contract
            )
        )
        self.assertIn("PERCEPTION_PROFILE FROZEN_STATE", provenance)

    def test_saccharin_preserves_all_fused_ring_rows_and_global_rank(self) -> None:
        output, _, scratch = self._run_example("saccharin")
        self.addCleanup(scratch.cleanup)
        definition = read_gic_definition_from_xyzin(output)
        rows = np.asarray(build_gic_b_matrix_from_xyzin(output).rows, dtype=float)
        butterfly = [gic for gic in definition.gics if gic.family == "BUTTERFLY"]
        puckering = [
            gic for gic in definition.gics if gic.family == "RING_PUCKER_COMPONENT"
        ]

        self.assertEqual(definition.point_group, "Cs")
        self.assertEqual((len(definition.gics), definition.rank, definition.target_rank), (45, 45, 45))
        self.assertEqual(np.linalg.matrix_rank(rows, tol=1.0e-8), 45)
        self.assertEqual(len(butterfly), 1)
        self.assertEqual(len(puckering), 5)
        self.assertLess(butterfly[0].coefficients[0][1] * butterfly[0].coefficients[1][1], 0.0)

    def test_standalone_accepts_supplied_topology_or_primitives(self) -> None:
        _seed, seed_lines, scratch = self._run_example("water")
        self.addCleanup(scratch.cleanup)
        standalone = Path(__file__).resolve().parents[1]

        for section_name, profile in (
            ("TOPOLOGY", "STANDALONE_TOPOLOGY"),
            ("PRIMITIVES", "STANDALONE_PRIMITIVES"),
        ):
            with self.subTest(section=section_name):
                source = Path(scratch.name) / f"water-{section_name.lower()}.smith.xyz"
                target = Path(scratch.name) / f"water-{section_name.lower()}.xyzin"
                shutil.copyfile(standalone / "examples" / "water.smith.xyz", source)
                replace_section(source, section_name, section_content(seed_lines, section_name))
                self.assertEqual(main(["build", str(source), str(target)]), 0)
                built = read_sectioned_lines(target)
                provenance = section_content(built, "SMITH_PROVENANCE")
                self.assertIn(f"PERCEPTION_PROFILE {profile}", provenance)
                self.assertTrue(section_content(built, "TOPOLOGY"))
                self.assertTrue(section_content(built, "PRIMITIVES"))

    def test_eta3_fixture_generator_is_reproducible(self) -> None:
        standalone = Path(__file__).resolve().parents[1]
        source = standalone / "examples" / "eta3_allyl_palladium.source.xyz"
        packaged = standalone / "examples" / "eta3_allyl_palladium.frozen.xyzin"
        self.assertTrue(source.is_file())
        self.assertTrue(packaged.is_file())
        self.assertIn("SOURCE=SUPPLIED_EXPLICIT_TEST_FIXTURE", packaged.read_text(encoding="utf-8"))

    def test_build_writes_merlino_style_report_and_g16_input_by_default(self) -> None:
        output, _, scratch = self._run_example("water")
        self.addCleanup(scratch.cleanup)
        report = output.with_suffix(".smith.out")
        gaussian = output.with_suffix(".g16.gjf")

        report_text = report.read_text(encoding="utf-8")
        gaussian_text = gaussian.read_text(encoding="utf-8")
        self.assertIn("Coordinate Definitions", report_text)
        self.assertIn("state=ACTIVE", report_text)
        self.assertIn("state=FROZEN", report_text)
        self.assertIn("value=", report_text)
        self.assertIn("components=", report_text)
        self.assertIn("opt=(readallgic,calcfc,maxcycle=80)", gaussian_text.lower())
        self.assertIn("(Frozen) =", gaussian_text)


if __name__ == "__main__":
    unittest.main()
