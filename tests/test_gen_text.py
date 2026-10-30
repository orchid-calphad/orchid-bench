"""Unit tests for bench/gen_text.py: running-text sentences and table layouts for the synthetic papers."""

from __future__ import annotations

import gen_text
import numpy as np
import pytest

pytest.importorskip("pymupdf")
import generate  # noqa: E402

INVARIANT = {
    "reaction": "L = α + β",
    "type": "eutectic",
    "T": 1000.0,
    "reactants": ["LIQUID"],
    "products": ["ALPHA", "BETA"],
    "points": [
        {"phase": "ALPHA", "label": "α", "x": 0.1},
        {"phase": "LIQUID", "label": "L", "x": 0.5},
        {"phase": "BETA", "label": "β", "x": 0.9},
    ],
}
PLAIN = {"merged_header": False, "footnotes": False, "uncertainty": False}
FOOTED = {"merged_header": False, "footnotes": True, "uncertainty": False}
MERGED = {"merged_header": True, "footnotes": False, "uncertainty": False}
CURVES = {
    "curves": [
        {"phase": "LIQUID", "x": [0.0, 1.0], "T": [1000.0, 1200.0]},
        {"phase": "FCC_A1", "x": [0.0, 1.0], "T": [900.0, 1100.0]},
    ]
}


class TestTypeArticle:
    """type_article gives the indefinite article to place before each invariant type name."""

    def test_unknown_type_takes_an(self):
        """The fallback type 'unknown' takes 'an', and the named reaction types take 'a'."""
        assert gen_text.type_article["unknown"] == "an"
        assert gen_text.type_article["eutectic"] == "a"
        assert gen_text.type_article["monotectoid"] == "a"


class TestInvariantSentences:
    """invariant_sentences writes one sentence per invariant that quotes the stored printed values."""

    def test_sentence_quotes_the_recorded_temperature_and_composition(self):
        """Each sentence contains the printed temperature and composition that were recorded for it."""
        vals = generate.Values("Cu")
        out = gen_text.invariant_sentences(np.random.default_rng(3), {"invariants": [INVARIANT]}, vals, "K", "at.%", 0)
        assert len(out) == 1
        sentence, (ti, xi) = out[0]
        assert vals.items[ti]["printed"] == "1000 K"
        assert vals.items[ti]["quantity"] == "invariant_temperature"
        assert vals.items[xi]["quantity"] == "invariant_composition:L"
        assert vals.items[ti]["exact_SI"] == pytest.approx(1000.0)
        assert vals.items[xi]["exact_SI"] == pytest.approx(0.5)
        assert vals.items[xi]["value"] == pytest.approx(50.0)
        assert vals.items[ti]["printed"] in sentence
        assert vals.items[xi]["printed"] in sentence
        assert "L = α + β" in sentence

    def test_same_seed_gives_same_sentence(self):
        """The wording is drawn from the generator, so one seed always gives one sentence."""
        first = gen_text.invariant_sentences(
            np.random.default_rng(9), {"invariants": [INVARIANT]}, generate.Values("Cu"), "K", "at.%", 0
        )
        second = gen_text.invariant_sentences(
            np.random.default_rng(9), {"invariants": [INVARIANT]}, generate.Values("Cu"), "K", "at.%", 0
        )
        assert first[0][0] == second[0][0]


class TestTmSentences:
    """tm_sentences writes one melting-point sentence for each pure component that has a melting point."""

    def test_skips_component_without_melting_point(self):
        """Only the component with a melting point is described, and its printed value is recorded."""
        vals = generate.Values("Cu")
        out = gen_text.tm_sentences(
            np.random.default_rng(0), {"Tm_A": None, "Tm_B": 1234.5, "elements": ["Ag", "Cu"]}, vals, "K", 1
        )
        assert len(out) == 1
        sentence, (ti,) = out[0]
        assert vals.items[ti]["printed"] == "1234.5 K"
        assert sentence in {
            "Pure Cu melts at 1234.5 K.",
            "The melting point of Cu reproduced by the dataset is 1234.5 K.",
            "For Cu, the liquidus meets the axis at 1234.5 K.",
        }

    def test_no_melting_points_gives_no_sentences(self):
        """With neither melting point given, nothing is written."""
        out = gen_text.tm_sentences(
            np.random.default_rng(0), {"Tm_A": None, "Tm_B": None, "elements": ["Ag", "Cu"]}, generate.Values("Cu"), "K", 1
        )
        assert out == []


class TestInvariantTable:
    """invariant_table lays out one row per invariant, with its temperature and phase compositions."""

    def test_row_values_follow_the_requested_units(self):
        """Temperature is printed in kelvin to one decimal and compositions in at.% to one decimal."""
        table = gen_text.invariant_table({"elements": ["Ag", "Cu"], "invariants": [INVARIANT]}, PLAIN, "at.%", "K")
        assert table["id"] == "invariants"
        assert table["rows"] == [["L = α + β", "eutectic", "1000.0", "50.0", "10.0", "90.0"]]
        assert table["column_units"] == [None, None, "K", "at.%", "at.%", "at.%"]

    def test_mole_fraction_units_use_three_decimals_and_element_label(self):
        """Mole fractions are printed to three decimals under an x(Cu) column header."""
        table = gen_text.invariant_table({"elements": ["Ag", "Cu"], "invariants": [INVARIANT]}, PLAIN, "mole_fraction", "K")
        assert table["rows"][0][3:] == ["0.500", "0.100", "0.900"]
        assert table["header"][0][3]["text"] == "x(Cu), 1"

    def test_merged_header_has_two_rows_with_a_spanning_composition_cell(self):
        """The merged layout puts the composition heading over three phase columns."""
        table = gen_text.invariant_table({"elements": ["Ag", "Cu"], "invariants": [INVARIANT]}, MERGED, "at.%", "K")
        assert len(table["header"]) == 2
        assert table["header"][0][3] == {"text": "Compositions (at.% Cu)", "colspan": 3}

    def test_footnote_is_added_only_when_the_style_asks_for_it(self):
        """A footnote is written only when the style has footnotes switched on."""
        data = {"elements": ["Ag", "Cu"], "invariants": [INVARIANT]}
        assert gen_text.invariant_table(data, PLAIN, "at.%", "K")["footnotes"] == []
        assert len(gen_text.invariant_table(data, FOOTED, "at.%", "K")["footnotes"]) == 1


class TestLiquidusTable:
    """liquidus_table reads liquidus and solidus temperatures by linear interpolation at fixed compositions."""

    def test_interpolates_liquidus_and_solidus_at_fixed_compositions(self):
        """With a straight liquidus and solidus, four compositions give four interpolated rows."""
        rows = gen_text.liquidus_table({"elements": ["Ag", "Cu"], "boundaries": [CURVES]}, PLAIN, "at.%", "K")["rows"]
        assert rows == [
            ["20", "1040.0", "940.0"],
            ["40", "1080.0", "980.0"],
            ["60", "1120.0", "1020.0"],
            ["80", "1160.0", "1060.0"],
        ]

    def test_takes_the_highest_liquidus_when_several_liquid_curves_cross(self):
        """With two liquid curves crossing the same composition, the reported liquidus is the higher one."""
        boundary = {
            "curves": [
                {"phase": "LIQUID", "x": [0.0, 1.0], "T": [1000.0, 1200.0]},
                {"phase": "LIQUID", "x": [0.0, 1.0], "T": [1050.0, 1250.0]},
                {"phase": "FCC_A1", "x": [0.0, 1.0], "T": [900.0, 1100.0]},
            ]
        }
        rows = gen_text.liquidus_table({"elements": ["Ag", "Cu"], "boundaries": [boundary]}, PLAIN, "at.%", "K")["rows"]
        assert rows[0] == ["20", "1090.0", "940.0"]

    def test_no_liquid_boundary_gives_no_rows(self):
        """A boundary without a liquid phase gives an empty table."""
        boundary = {"curves": [{"phase": "FCC_A1", "x": [0.0, 1.0], "T": [900.0, 1100.0]}]}
        assert gen_text.liquidus_table({"elements": ["Ag", "Cu"], "boundaries": [boundary]}, PLAIN, "at.%", "K")["rows"] == []


class TestThermoTable:
    """thermo_table builds the formation or Redlich-Kister table, and returns None when neither is present."""

    def test_formation_row_converts_units_and_uses_the_minus_glyph(self):
        """Enthalpy is shown in kJ/mol and entropy in J/(mol K), with the requested minus sign."""
        data = {
            "elements": ["Ag", "Cu"],
            "properties": {"formation": [{"label": "AgCu", "x": 0.5, "HM_298": -5000.0, "SM_298": -1.5}]},
        }
        table = gen_text.thermo_table(data, PLAIN, np.random.default_rng(0), "−")
        assert table["id"] == "formation"
        assert table["rows"] == [["AgCu", "50.0", "−5.00", "−1.50"]]
        assert table["column_units"] == [None, "at.%", "kJ/mol", "J/(mol K)"]

    def test_uncertainty_is_appended_to_the_enthalpy_value(self):
        """With uncertainty on, the enthalpy cell carries a plus-minus value and the footnote says so."""
        data = {
            "elements": ["Ag", "Cu"],
            "properties": {"formation": [{"label": "AgCu", "x": 0.5, "HM_298": -5000.0, "SM_298": -1.5}]},
        }
        style = {"merged_header": False, "footnotes": True, "uncertainty": True}
        table = gen_text.thermo_table(data, style, np.random.default_rng(0), "−")
        assert table["rows"][0][2].startswith("−5.00 ± ")
        assert "synthetic illustration uncertainties" in table["footnotes"][0]

    def test_redlich_kister_rows_print_python_floats(self):
        """Redlich-Kister coefficients are printed with repr and the minus glyph, one row per order."""
        data = {
            "elements": ["Ag", "Cu"],
            "properties": {
                "rk_liquid": [{"order": 0, "a": -1000.5, "b": 2.0, "c": 0.0}, {"order": 1, "a": 3.25, "b": -0.5, "c": 1e-3}]
            },
        }
        table = gen_text.thermo_table(data, PLAIN, np.random.default_rng(0), "−")
        assert table["id"] == "rk_liquid"
        assert table["rows"] == [["L0", "−1000.5", "2.0", "0.0"], ["L1", "3.25", "−0.5", "0.001"]]

    def test_no_properties_gives_no_table(self):
        """With neither formation nor Redlich-Kister data there is no table."""
        assert gen_text.thermo_table({"elements": ["Ag", "Cu"], "properties": {}}, PLAIN, np.random.default_rng(0), "−") is None
