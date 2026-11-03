"""Oracle self-score: oracle.py writes the truth back in extractor format, and score.py must rate it perfect."""

from __future__ import annotations

import json
import runpy
import shutil
import sys

import common
import pytest
import score

CASE = ["syn-AgCu-06He-v0", "syn-BiSn-08Li-v0"]
RENDERINGS = ["vector", "clean_raster", "degraded_raster", "image_only_pdf"]


@pytest.fixture(scope="module")
def scored(tmp_path_factory, bench_dir):
    """Run oracle.py and score.py on two truth records in a temporary tree and return the score file."""
    pytest.importorskip("scipy")
    root = tmp_path_factory.mktemp("oracle")
    truth = root / "truth"
    truth.mkdir()
    for doc_id in CASE:
        shutil.copy(bench_dir / "truth" / f"{doc_id}.json", truth / f"{doc_id}.json")
    out = root / "scores.json"
    with pytest.MonkeyPatch.context() as mp:
        # oracle.py and score.py bind these names at import; patch them so nothing is written under bench/.
        mp.setattr(common, "bench", root)
        mp.setattr(common, "truth_dir", truth)
        mp.setattr(score, "truth_dir", truth)
        mp.setattr(sys, "path", list(sys.path))  # oracle.py prepends bench/ to sys.path
        runpy.run_path(str(bench_dir / "baselines" / "oracle.py"))
        score.main(["--outputs", str(root / "outputs" / "oracle"), "--out", str(out)])
    return _load(out)


def _load(path):
    """Read a score file as JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


class TestOracleCase:
    """The self-score covers exactly the two truth records it was given."""

    def test_every_rendering_is_scored_on_both_documents(self, scored):
        """Each of the four renderings is scored on both truth records and no other document."""
        assert sorted(scored["per_document"]) == CASE
        assert {r: scored["aggregate"][r]["documents"] for r in RENDERINGS} == dict.fromkeys(RENDERINGS, len(CASE))


@pytest.mark.parametrize("rendering", RENDERINGS)
class TestOracleSelfScore:
    """Scoring the oracle's own output against its truth gives perfect scores for every rendering."""

    def test_table_cells_and_numbers_are_exact(self, scored, rendering):
        """Every table cell and every numeric table value matches the truth exactly."""
        tables = scored["aggregate"][rendering]["tables"]
        assert tables["cell_exact"] == pytest.approx(1.0)
        assert tables["numeric_exact"] == pytest.approx(1.0)
        assert tables["numeric_rel_err_median"] == pytest.approx(0.0, abs=1e-12)

    def test_property_panels_series_and_points_are_recovered(self, scored, rendering):
        """Every truth panel is found, with every series and every marked point recovered."""
        xy = scored["aggregate"][rendering]["xy"]
        assert xy["panels_found"] == xy["panels_true"] > 0
        assert xy["series_recall"] == pytest.approx(1.0)
        assert xy["series_precision"] == pytest.approx(1.0)
        assert xy["point_recall@0.01"] == pytest.approx(1.0)
        assert xy["point_err_median"] == pytest.approx(0.0, abs=1e-9)

    def test_phase_diagram_fields_boundaries_and_invariants_match(self, scored, rendering):
        """Fields, boundary polylines and invariant points of the phase diagram match the truth."""
        pd = scored["aggregate"][rendering]["phase_diagram"]
        for key in (
            "field_precision",
            "field_recall",
            "boundary_precision@0.01",
            "boundary_recall@0.01",
            "invariant_precision",
            "invariant_recall",
            "invariant_type_accuracy",
        ):
            assert pd[key] == pytest.approx(1.0), key
        assert pd["invariant_T_err_K"] == pytest.approx(0.0, abs=1e-9)
        assert pd["boundary_hausdorff95"] == pytest.approx(0.0, abs=1e-3)

    def test_running_text_values_are_recovered(self, scored, rendering):
        """Every number printed in running text and figure captions is recovered, with no extras."""
        text = scored["aggregate"][rendering]["text"]
        assert text["values_pred"] == text["values_true"] > 0
        assert text["precision"] == pytest.approx(1.0)
        assert text["recall"] == pytest.approx(1.0)

    def test_invariant_and_phase_rows_are_recovered(self, scored, rendering):
        """Invariant-table rows and phase-stability rows match the truth on every checked field."""
        agg = scored["aggregate"][rendering]
        for key in ("row_recall", "row_precision", "cell_exact", "type_accuracy", "evidence_accuracy", "citation_accuracy"):
            assert agg["invariant_table"][key] == pytest.approx(1.0), key
        for key in ("row_recall", "row_precision", "cell_exact", "structure_accuracy", "prototype_accuracy", "pearson_accuracy"):
            assert agg["phase_stability"][key] == pytest.approx(1.0), key
        assert agg["phase_stability"]["rows_pred"] == agg["phase_stability"]["rows_true"] > 0

    def test_thermodynamic_records_are_recovered(self, scored, rendering):
        """Every property record matches its truth value, unit and basis."""
        rec = scored["aggregate"][rendering]["thermo_tables"]
        assert rec["record_recall"] == pytest.approx(1.0)
        assert rec["record_precision"] == pytest.approx(1.0)
        for key in ("value_exact", "within_1pct", "unit_accuracy", "basis_accuracy"):
            assert rec[key] == pytest.approx(1.0), key
