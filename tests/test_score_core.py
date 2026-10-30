"""Unit tests for bench/score_core.py: text, number, box, polyline and count helpers used by score.py."""

from __future__ import annotations

import math

import numpy as np
import pytest
import score_core as sc


class TestNormText:
    """norm_text folds dashes and whitespace so that printed text compares equal across renderings."""

    def test_folds_unicode_minus_and_en_dash_to_hyphen(self):
        """The minus sign and the en dash both become an ASCII hyphen."""
        assert sc.norm_text("a−b–c") == "a-b-c"

    def test_collapses_runs_of_whitespace_including_no_break_space(self):
        """Runs of ordinary and no-break spaces collapse to one space and are stripped at the ends."""
        assert sc.norm_text("  a  b\t\tc  ") == "a b c"

    def test_none_becomes_empty_string(self):
        """A missing value normalises to the empty string."""
        assert sc.norm_text(None) == ""


class TestNumOf:
    """num_of pulls the first number out of a printed string, or returns None."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("T = −12.5 K", -12.5),
            ("3.2e-4 eV", 3.2e-4),
            ("x = +7", 7.0),
            ("1 234", 1234.0),
        ],
    )
    def test_extracts_first_number(self, text, expected):
        """Signs, unicode minus, exponents and embedded spaces are read as the printed value."""
        assert sc.num_of(text) == pytest.approx(expected)

    def test_returns_none_without_a_number(self):
        """A string with no digits gives None."""
        assert sc.num_of("none") is None


class TestAabb:
    """aabb gives the axis-aligned bounding box of a set of corner points."""

    def test_box_spans_minimum_and_maximum_of_each_axis(self):
        """The result is [x_min, y_min, x_max, y_max] over all corners."""
        assert sc.aabb([(1, 5), (3, 2), (0, 4)]) == [0, 2, 3, 5]


class TestIou:
    """iou is intersection over union for axis-aligned boxes, zero when they cannot overlap."""

    def test_identical_boxes_give_one(self):
        """A box compared with itself has IoU 1."""
        assert sc.iou([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)

    def test_disjoint_boxes_give_zero(self):
        """Boxes that do not touch have IoU 0."""
        assert sc.iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0

    def test_half_overlapping_boxes_give_one_third(self):
        """Two 2x2 boxes shifted by one unit share 2 of 6 area units."""
        assert sc.iou([0, 0, 2, 2], [1, 0, 3, 2]) == pytest.approx(1.0 / 3.0)

    def test_zero_area_boxes_give_zero(self):
        """Degenerate boxes with no union area return 0 rather than dividing by zero."""
        assert sc.iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0


class TestDensify:
    """densify resamples polyline edges so that consecutive points are at most one step apart."""

    def test_inserts_points_every_step_and_keeps_the_end_point(self):
        """A unit segment at step 0.25 gives points at 0, 0.25, 0.5, 0.75 and the end point 1."""
        pts = sc.densify([(0, 0), (1, 0)], 0.25)
        assert pts.shape == (5, 2)
        assert pts[:, 0].tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
        assert np.all(pts[:, 1] == 0.0)

    def test_empty_polyline_gives_empty_point_array(self):
        """An empty polyline gives a (0, 2) array rather than an error."""
        assert sc.densify([], 0.25).shape == (0, 2)


class TestNnDist:
    """nn_dist returns, for each point, the distance to its nearest neighbour in the other set."""

    @pytest.fixture(autouse=True)
    def _needs_scipy(self):
        """Skip the nearest-neighbour tests when scipy is not installed."""
        pytest.importorskip("scipy")

    def test_distances_to_nearest_point(self):
        """(0,0) is 1 from (0,1) and (3,4) is sqrt(18) from it."""
        d = sc.nn_dist(np.array([[0.0, 0.0], [3.0, 4.0]]), np.array([[0.0, 1.0]]))
        assert d.tolist() == pytest.approx([1.0, math.sqrt(18.0)])

    def test_empty_query_gives_empty_result(self):
        """With no points to place there are no distances."""
        assert sc.nn_dist(np.zeros((0, 2)), np.array([[0.0, 1.0]])).shape == (0,)

    def test_empty_reference_gives_infinite_distance(self):
        """With nothing to match against every distance is infinite."""
        assert sc.nn_dist(np.array([[0.0, 0.0]]), np.zeros((0, 2))).tolist() == [math.inf]


class TestStats:
    """stats summarises a list of values and skips missing or non-finite entries."""

    def test_skips_none_and_nan(self):
        """Only the finite numbers 1, 2 and 3 are counted and summarised."""
        out = sc.stats([1, 2, None, float("nan"), 3])
        assert out["n"] == 3
        assert out["mean"] == pytest.approx(2.0)
        assert out["median"] == pytest.approx(2.0)

    def test_empty_input_gives_none_summaries(self):
        """With no usable values the count is 0 and mean and median are None."""
        assert sc.stats([]) == {"n": 0, "mean": None, "median": None}


class TestPrf:
    """prf turns matched and total counts into precision, recall and F1."""

    def test_precision_recall_and_f1(self):
        """3 matches among 4 predictions and 6 truths give precision 0.75, recall 0.5 and F1 0.6."""
        out = sc.prf(3, 4, 3, 6)
        assert out["precision"] == pytest.approx(0.75)
        assert out["recall"] == pytest.approx(0.5)
        assert out["f1"] == pytest.approx(0.6)
        assert (out["n_pred"], out["n_true"]) == (4, 6)

    def test_no_predictions_and_no_truth_give_none(self):
        """With empty denominators precision and recall are undefined, so they are None."""
        out = sc.prf(0, 0, 0, 0)
        assert out["precision"] is None
        assert out["recall"] is None
        assert out["f1"] is None

    def test_zero_matches_give_zero_f1(self):
        """Predictions and truths that never match give precision, recall and F1 of zero."""
        out = sc.prf(0, 2, 0, 2)
        assert (out["precision"], out["recall"], out["f1"]) == (0.0, 0.0, 0.0)


class TestPhaseLabels:
    """_canon and _loose map printed phase labels to one comparable form."""

    @pytest.mark.parametrize(
        ("label", "expected"),
        [("liquid#1 ", "L"), ("LIQUID", "L"), ("fcc_a1", "FCC_A1"), ("Al 3 Ni", "AL3NI")],
    )
    def test_canon_folds_case_spaces_liquid_and_sublattice_suffix(self, label, expected):
        """Upper-cases, drops spaces and sublattice suffixes, and maps LIQUID to L."""
        assert sc._canon(label) == expected

    @pytest.mark.parametrize(("label", "expected"), [("L2", "L"), ("FCC)2", "FCC)"), ("L", "L")])
    def test_loose_drops_liquid_index_and_closing_bracket_index(self, label, expected):
        """Loose matching also drops a trailing digit after L or after a closing bracket."""
        assert sc._loose(label) == expected


class TestUnitHelpers:
    """_toK, _tox and _close convert units and compare with a relative tolerance."""

    def test_celsius_is_shifted_to_kelvin(self):
        """0 degC is 273.15 K, and a value in kelvin is returned unchanged."""
        assert sc._toK(0, "°C") == pytest.approx(273.15)
        assert sc._toK(500, "K") == pytest.approx(500.0)
        assert sc._toK(None, "K") is None

    def test_atomic_percent_is_divided_by_one_hundred(self):
        """25 at.% is the mole fraction 0.25; a mole fraction is returned unchanged."""
        assert sc._tox(25, "at.%") == pytest.approx(0.25)
        assert sc._tox(0.3, "mole_fraction") == pytest.approx(0.3)
        assert sc._tox(None, "at.%") is None

    def test_close_uses_relative_tolerance_and_rejects_none(self):
        """Values within 1e-6 relative of each other are close, and a missing value never is."""
        assert sc._close(1.0, 1.0 + 1e-9)
        assert not sc._close(1.0, 1.1)
        assert not sc._close(None, 1.0)
