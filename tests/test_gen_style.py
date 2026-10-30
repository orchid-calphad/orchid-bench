"""Unit tests for bench/gen_style.py: random but reproducible presentation styles for the synthetic papers."""

from __future__ import annotations

import gen_style
import numpy as np

ALL_PROPERTIES = {"properties": {"mixing": True, "activity": True, "cp": True, "gibbs": True}}
NO_PROPERTIES = {"properties": {}}
SEEDS = range(50)


def _style(seed, data=ALL_PROPERTIES):
    """Draw one style from a generator seeded with seed."""
    return gen_style.choose_style(np.random.default_rng(seed), data)


class TestChooseStyle:
    """choose_style draws every presentation choice from its declared set, reproducibly for a seed."""

    def test_same_seed_gives_identical_style(self):
        """Two generators with the same seed draw the same style."""
        assert _style(7) == _style(7)

    def test_choices_come_from_the_declared_sets(self):
        """Units, line widths, figure sizes and legend positions are always drawn from their allowed values."""
        for seed in SEEDS:
            style = _style(seed)
            pd = style["phase_diagram"]
            assert pd["T_unit"] in {"K", "°C"}
            assert pd["x_unit"] in {"at.%", "mole_fraction"}
            assert pd["linewidth"] in {0.6, 0.8, 1.0, 1.3, 1.6}
            assert pd["fig_size"] in {(4.6, 3.6), (5.0, 3.8), (4.2, 3.6)}
            assert pd["legend_loc"] in {"upper center", "lower center", "upper left", "upper right"}
            assert style["table"]["rules"] in {"booktabs", "grid", "plain"}

    def test_legend_is_shown_exactly_when_markers_or_overlay_are_drawn(self):
        """The legend flag is the logical or of the markers and overlay flags."""
        for seed in SEEDS:
            pd = _style(seed)["phase_diagram"]
            assert pd["legend"] == (pd["markers"] or pd["overlay"])

    def test_inset_forces_the_legend_to_the_lower_centre(self):
        """When an inset is drawn the legend moves to the lower centre, where the inset does not sit."""
        insets = [_style(seed)["phase_diagram"] for seed in SEEDS]
        insets = [pd for pd in insets if pd["inset"]]
        assert insets, "no seed in the sample drew an inset"
        assert all(pd["legend_loc"] == "lower center" for pd in insets)

    def test_property_kinds_come_from_the_available_properties(self):
        """Only properties that the data has are chosen, at most two, and never mixing with two_panel together."""
        for seed in SEEDS:
            kinds = _style(seed)["property_kinds"]
            assert len(kinds) <= 2
            assert set(kinds) <= {"mixing", "activity", "cp", "gibbs", "two_panel"}
            assert not {"mixing", "two_panel"} <= set(kinds)

    def test_no_available_properties_gives_no_property_kinds(self):
        """With no property data the style asks for no property figure kinds."""
        assert _style(1, NO_PROPERTIES)["property_kinds"] == []

    def test_overlay_offset_is_a_pair_of_numbers(self):
        """The overlay offset is a two-element tuple of floats."""
        offset = _style(3)["phase_diagram"]["overlay_offset"]
        assert len(offset) == 2
        assert all(isinstance(v, float) for v in offset)

    def test_both_temperature_units_occur_across_seeds(self):
        """Over the sample both Kelvin and Celsius are drawn, so the generator covers both."""
        units = {_style(seed)["phase_diagram"]["T_unit"] for seed in SEEDS}
        assert units == {"K", "°C"}


class TestChooseStyleV2:
    """choose_style_v2 extends a style with the formation, mixing and phase-fraction choices."""

    @staticmethod
    def _ext(formula_atoms=2, n_phases=6):
        """Build a minimal extension record with formation rows and phase-stability rows."""
        return {
            "formation": {"rows": [{"formula_atoms": formula_atoms}, {"formula_atoms": formula_atoms}], "method": "x"},
            "phase_stability": [{}] * n_phases,
        }

    def test_returns_the_six_expected_sections(self):
        """The v2 style has the invariant, phase, formation, mixing, phase-fraction and text sections."""
        base = _style(0)
        out = gen_style.choose_style_v2(np.random.default_rng(0), ALL_PROPERTIES, self._ext(), base)
        assert set(out) == {"invariant_table", "phase_table", "formation_table", "mixing_table", "phase_fraction", "text_v2"}

    def test_formation_uncertainty_and_phase_fraction_style_are_inherited(self):
        """Uncertainty, colour, line width and x unit are copied from the base style."""
        base = _style(0)
        out = gen_style.choose_style_v2(np.random.default_rng(0), ALL_PROPERTIES, self._ext(), base)
        assert out["formation_table"]["uncertainty"] == base["table"]["uncertainty"]
        assert out["phase_fraction"]["color"] == base["property"]["color"]
        assert out["phase_fraction"]["linewidth"] == base["property"]["linewidth"]
        assert out["phase_fraction"]["x_unit"] == base["property"]["x_unit"]

    def test_formula_unit_basis_needs_formula_atoms_on_every_row(self):
        """Without formula-atom counts the formation basis is always per mole of atoms."""
        base = _style(0)
        for seed in SEEDS:
            out = gen_style.choose_style_v2(np.random.default_rng(seed), ALL_PROPERTIES, self._ext(formula_atoms=None), base)
            assert out["formation_table"]["basis"] == "mol_atoms"

    def test_phase_split_needs_at_least_five_phase_rows(self):
        """The phase table is never split for documents with fewer than five phase-stability rows."""
        base = _style(0)
        for seed in SEEDS:
            out = gen_style.choose_style_v2(np.random.default_rng(seed), ALL_PROPERTIES, self._ext(n_phases=3), base)
            assert out["phase_table"]["split"] is False
