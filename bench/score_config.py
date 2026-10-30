"""Tolerances, renderings and unit-conversion tables for the scoring harness, split from score.py."""

renderings = ["vector", "clean_raster", "degraded_raster", "image_only_pdf"]
hit_tol = 0.01  # normalised axis distance counted as a hit
series_tol = 0.02  # median distance for a predicted series to match a truth series
min_field_cells = 6
inv_t_tol = 15.0  # K


# ------------------------------------------------------------------------------------------------ text values
unit_aliases = {
    "k": "K",
    "°c": "°C",
    "c": "°C",
    "degc": "°C",
    "at.%": "at.%",
    "at%": "at.%",
    "mole_fraction": "mole_fraction",
    "x": "mole_fraction",
    "kj/mol": "kJ/mol",
    "j/mol": "J/mol",
    "atm": "atm",
    "cal/mol": "cal/mol",
    "j/(mol k)": "J/(mol K)",
    "j/mol/k": "J/(mol K)",
    "j": "J",
}


# ------------------------------------------------------------------------------------------------ semantic tables (v2)
inv_t_row_tol = 15.0
e_factor = {
    "J/mol": 1.0,
    "kJ/mol": 1000.0,
    "cal/mol": 4.184,
    "kcal/mol": 4184.0,
    "J/(mol K)": 1.0,
    "cal/(mol K)": 4.184,
    "1": 1.0,
}
unit_norm = {
    "j/mol": "J/mol",
    "kj/mol": "kJ/mol",
    "cal/mol": "cal/mol",
    "kcal/mol": "kcal/mol",
    "j/(mol k)": "J/(mol K)",
    "j/mol/k": "J/(mol K)",
    "cal/(mol k)": "cal/(mol K)",
    "1": "1",
    "": "1",
    "none": "1",
}
basis_norm = {
    "mol_atoms": "mol_atoms",
    "atom": "mol_atoms",
    "mol-atom": "mol_atoms",
    "per mole of atoms": "mol_atoms",
    "formula_unit": "formula_unit",
    "f.u.": "formula_unit",
    "per formula unit": "formula_unit",
}
