# ORCHID synthetic-papers benchmark (pilot)

Purpose: a ground-truthed test set for ORCHID's extraction tools. Twenty short synthetic "papers" (4–5 pages each,
86 pages in total), one per binary system, generated from pycalphad calculations on SGTE datasets, each rendered four ways and
accompanied by an exact truth key. Nothing here is a real publication; every page says so.

## Licence and attribution

- Generated documents, renderings and truth keys: **CC BY 4.0**.
- Derived from the SGTE binary dataset collection, B. Hallstedt, *Calphad* 89 (2025) 102833 (CC BY 4.0), as
  distributed in `open-tdb/databases/sgte-binary/`, and from the original assessment named in each TDB (cited in
  each document and in `truth/<doc>.json:original_assessment`). Every truth key records the TDB file and its SHA-256
  plus the ORCHID calculated-diagram JSON it reused (`orchid-data/phase-diagrams/sgte/`, CC BY 4.0) and its SHA-256.
- Fonts embedded in the PDFs: DejaVu (Bitstream Vera licence), shipped with matplotlib.
- Code: MIT.

## Layout

```
bench/
  generate.py        documents + renderings + truth (deterministic: --seed, --systems, --n, --jobs)
  pd_data.py         per-system data: reused ORCHID boundaries/invariants; own pycalphad field grid,
                     invariant typing, liquid mixing/activity, Cp, G curves, formation enthalpies, RK parameters
  pd_ext.py          truth v2 data (cache/<stem>.ext.json): phase stability ranges, homogeneity ranges, congruent
                     points, structure labels, phase fractions vs T, formation H/S/Cp at 298.15 K, liquid mixing table
  paper_tables.py    v2 table builders (invariant, phase, formation, mixing) with per-cell semantic truth
  systems.py         the 20 systems and their conventional phase labels
  figures.py         matplotlib figure generators (explicit axes, exact geometry)
  document.py        PyMuPDF composition (real text, vector figures, ruled tables)
  score.py           scoring harness
  baselines/         deterministic baselines (no model calls) + oracle self-test + run_all.py
  docs/<doc>/        vector.pdf, clean/page-N.png (300 dpi), degraded/page-N.jpg (150 dpi scan simulation),
                     image_only.pdf, figs/*.pdf (the figure PDFs as placed)
  truth/<doc>.json   canonical JSON truth key (schema orchid-synthetic-paper-truth/v2)
  outputs/<extractor>/<rendering>/<doc>.json   extractor outputs; scores.json per extractor
  results/           RESULTS.md, results.json
  showcase/index.html  self-contained page with three examples
  cache/, work/      intermediate data (cache: prepared system data; work: OCR TSV, tool inventories, crops)
```

## How to run

```
PY=~/orchid-py313/bin/python      # pycalphad 0.11.2, matplotlib, numpy, scipy, Pillow, PyMuPDF, pdfplumber
$PY bench/generate.py --jobs 8                     # ~10 s once bench/cache is filled (cache: ~3 min + ~1.5 min v2)
$PY bench/baselines/run_all.py                     # OCR (tesseract), baselines, scores, results
$PY bench/showcase/build.py
$PY bench/score.py --outputs bench/outputs/<your-extractor>
```

Environment knobs: `ORCHID_TDB_DIR`, `ORCHID_PD_DIR`, `OPEN_TDB`, `ASSESSED_PD_SRC`.

## Documents

Each document has: title, "ORCHID synthetic benchmark" byline, licence notice, abstract, five sections and a reference
list. Running text states values (invariant temperatures and compositions, melting points, congruent melting, stability
limits, maximum solubilities, mixing-enthalpy extremum, formation enthalpy per mole of atoms and per formula unit, Cp,
mixing Gibbs energy) with varied phrasing and units (K / °C, at.% / mole fraction, kJ/mol / J/mol / kJ/mol-atom, and
cal/mol in one document with the conversion stated).

Figures:

- Fig. 1 phase diagram, styles drawn per document from the seed: line width, black or coloured, labelled or
  unlabelled fields (conventional labels or TDB names), K or °C, at.% or mole fraction, invariant temperature labels,
  optional inset zoom, optional experimental-style markers (two series with legend), optional dashed second
  "assessment" shifted by a known (dx, dT).
- Fig. 2 equilibrium phase fractions vs temperature (pycalphad step, 5 K) for one or two alloys (one panel each),
  one series per phase, drawn only where the phase is present; K or °C.
- Figs. 3–4 property figures from: liquid mixing enthalpy (one or two temperatures), activities (linear or log,
  Raoult lines), Cp(T) of a compound, Gibbs-energy curves with common tangent, two-panel ΔmixH/ΔmixG; lines,
  markers or both.

Tables (every document):

1. Invariant reactions and congruent points: reaction in standard notation with the paper's phase names, type
   (eutectic, peritectic, monotectic, syntectic, eutectoid, peritectoid, congruent melting, congruent
   transformation; pure-component melting for Cu–Ni, which has no invariant), T (K or °C), composition of every
   participating phase (at.% or mole fraction). In 12 documents a "T exp." column (inline "1059.8 [3]" or a separate
   "Ref." column) holds the calculated value shifted by a recorded offset and attributed to one of three fictitious
   references [3]–[5], listed as fictitious in the reference list.
2. Solid phases: Pearson symbol, Strukturbericht designation and prototype where derivable from the TDB phase name
   (explicit mapping in `pd_ext.py`; "–" otherwise), stability range (from / to; "< 300 K" when stable at the lowest
   calculated temperature), homogeneity range at a stated temperature.
3. Formation properties at 298.15 K of the intermediate phases (ΔfH, ΔfS, Cp; kJ/mol, J/mol or cal/mol; per mole
   of atoms or, in 4 documents, per formula unit with the basis stated in the header; optional illustrative ±);
   systems without intermediate phases print the liquid Redlich–Kister parameters instead.
4. Liquid mixing properties at one temperature (ΔmixH, ΔmixG, activities of both components, optionally partial
   molar enthalpies) at 5 or 9 compositions.
5. (Cu–Ni only) liquidus and solidus temperatures.

Table styles vary per table: booktabs, fully ruled grid, zebra; merged or single headers; footnotes; footnote
markers (ᵃ ᵇ); ± values; "Table N (continued)" with the header repeated on a new page (4 documents).

Renderings: (a) `vector.pdf` born-digital, text layer present; (b) `clean/` 300 dpi PNG; (c) `degraded/` 150 dpi
JPEG with contrast loss, blur, rotation within ±1°, salt-and-pepper noise, JPEG quality 30–55, half in greyscale;
(d) `image_only.pdf` = (c) wrapped full-page, no text layer. Generation is byte-deterministic for a given seed.

## Truth key (`truth/<doc>.json`)

- `source_tdb` {file, sha256}, `diagram_source`, `original_assessment`, `calculation` settings, `style`.
- `renderings`: files + SHA-256 and, per page, a 2×3 affine `transforms` matrix from vector-page points (origin
  top-left, y down) to that rendering's coordinates (pt for PDFs, px for rasters); degraded pages record their
  rotation, blur/noise/JPEG settings.
- `figures[]`: page, bbox, `corners` per rendering, `axes[]` (ranges, units, scale, label, bbox, corners); phase
  diagram: `boundaries` (polylines x, T_K; `on_frame` marks curves lying on the frame), `invariant_lines`,
  `field_labels` (printed or not), `text_in_figure`, `marker_series`, `overlay` (offset), `inset`, `occluders`
  (legend/inset rectangles excluded from scoring); property plots: `series` with exact plotted `line`/`markers`.
- `phase_diagram`: `fields` (assemblage as TDB composition sets and conventional labels, interior point, size),
  `field_map` (field id per 0.01 × 10 K cell), `invariants` (T, type, phases and compositions, reaction),
  `congruent`, `melting_points_K`.
- `tables[]`: every cell {row, col, colspan, header, text (= printed), value, uncertainty, unit, bbox, corners,
  segment, seg_row, repeated_header, meta}; `meta` holds the semantic field, exact SI value, printed unit, basis,
  evidence class, citation and footnote marker. `segments[]` gives page/bbox of each part of a continued table.
- `semantic` (v2): `invariant_rows` (reaction, type, exact and printed T, every phase's label, TDB phase, exact and
  printed composition, `evidence` = [calculated_this_work, experimental_cited {citation, offset_K, fictitious}]),
  `phase_rows` (structure labels or null with the derivation, T_low/T_high exact, printed and kind, homogeneity
  range), `property_records` (formation/mixing records: quantity, printed value, unit, basis, atoms per formula unit,
  exact SI per mole of atoms), `statements` (stability sentences), `phase_fraction` (series per alloy),
  `phase_stability_full` (all limits with their source: invariant / melting / composition search).
- `citations[]`: the reference list ([1] SGTE collection, [2] original assessment, [3]–[5] fictitious).
- `text`: `sentences` (text, rects) and `values` (printed string, printed numeric value, unit, quantity, exact SI
  value, rects).

Printed values are exact: `value` equals the printed number; `exact_SI` keeps the computed value.

## Extractor output format

`outputs/<extractor>/<rendering>/<doc_id>.json`, rendering ∈ {vector, clean_raster, degraded_raster, image_only_pdf}:

```json
{"doc_id": "...", "rendering": "...", "extractor": "...", "status": {"task": "ran: ... | not run: reason"},
 "tables": [{"page": 0, "cells": [{"row": 0, "col": 0, "text": "..."}]}],
 "xy_figures": [{"page": 1, "panels": [{"bbox": [x0, y0, x1, y1], "series": [{"name": "...", "points": [[x, y]]}]}]}],
 "phase_diagrams": [{"page": 0, "bbox": [...], "x_unit": "mole_fraction|at.%", "T_unit": "K|°C",
                     "fields": [{"point": [x, T], "phases": ["L", "(Cu)"]}],
                     "boundaries": [{"points": [[x, T]], "polyline": false}],
                     "invariants": [{"T": 1055.8, "type": "eutectic", "points": [{"phase": "L", "x": 0.397}]}]}],
 "text_values": [{"value": 1055.8, "unit": "K|°C|at.%|mole_fraction|kJ/mol|J/mol|atm"}]}
```

v2 adds (all optional):

```json
{"invariant_rows": [{"reaction": "L → (Ag) + (Cu)", "type": "eutectic", "T": 782.7, "T_unit": "°C", "x_unit": "at.%",
                     "compositions": [{"phase": "L", "x": 39.9}],
                     "evidence": [{"class": "experimental", "T": 785.7, "citation": 3}]}],
 "phase_rows": [{"phase": "Ni3Al", "pearson": "cP4", "strukturbericht": "L1₂", "prototype": "Cu3Au", "T_low": null,
                 "T_high": 1369.6, "T_unit": "°C",
                 "homogeneity": {"x_min": 0.724, "x_max": 0.771, "x_unit": "mole_fraction", "T": 1047, "T_unit": "°C"}}],
 "property_records": [{"table": "formation", "phase": "NiAl", "quantity": "dfH|dfS|Cp", "value": -62.24,
                       "unit": "kJ/mol", "basis": "mol_atoms|formula_unit"},
                      {"table": "mixing", "x": 0.5, "x_unit": "mole_fraction",
                       "quantity": "HM_MIX|GM_MIX|a_A|a_B|HM_PART_A|HM_PART_B", "value": -49.27, "unit": "kJ/mol"}]}
```

`bbox` is in the rendering's own coordinates; panels and diagrams are matched to the truth by bbox overlap. Field
phases may be conventional labels (`L`, `(Cu)`, `Ni3Al`) or TDB names (`LIQUID`, `FCC_A1`).

## Metrics (score.py)

- Tables: best-aligned cell exact match per table segment (strict and whitespace/minus-normalised), numeric exact and
  relative error; also reported per v2 task (invariant / phase / thermodynamic tables).
- Invariant table: rows matched by phase set (labels or TDB names) + T within 15 K; row recall/precision, T error and
  printed-T exactness, composition error and exactness, type accuracy, evidence accuracy (experimental value and
  citation, or correctly no experimental value) and citation accuracy.
- Phase stability: rows matched by phase name; Pearson / Strukturbericht / prototype accuracy (blank = null), stability
  limit T errors, homogeneity x and T errors; phase-fraction curves via the XY scorer restricted to `quantity = NP`.
- Thermodynamic tables: records matched by (table, phase or composition, quantity); record recall/precision, printed
  value exact, relative error after unit and basis conversion, within 1 %, unit and basis accuracy.
- Errors of the v2 semantic tasks are measured against the printed value (converted to K, mole fraction, J per mole
  of atoms), so printing precision does not count as error; exact values stay in the truth.
- XY figures: per truth point, distance to the nearest extracted point in axis-normalised units (log axes in log10);
  recall@0.01, marker-only recall, median error; series recall/precision (median distance ≤ 0.02).
- Phase diagrams: boundary precision/recall@0.01 and 95th-percentile Hausdorff (x and T normalised to the axes),
  median |Δx| and |ΔT|; field precision (extracted label point lies in a truth field with the same assemblage)
  and recall (truth fields of ≥ 6 grid cells with a correct point); invariant recall/precision (matched within
  15 K), T and composition errors, type accuracy. Occluded areas (legend, inset) are excluded.
- Running text: multiset precision/recall of (value, unit) against values printed in text, captions, figure labels,
  legend entries and table headers.

`baselines/oracle.py` writes the truth in this format; it scores 1.0 everywhere (self-test).

## Notes and limitations

- Field assemblages come from our own pycalphad grid (the ORCHID diagram grid merges same-phase miscibility gaps);
  boundaries and invariants are reused from ORCHID data after removing duplicates and degenerate frame-edge regions.
- Thin fields can split into several grid components; only fields of ≥ 6 cells are scored.
- Invariant types follow the dataset's geometry (e.g. Pb–Zn gives a syntectic L1 + L2 → (Zn)).
- Formation-enthalpy "±" values are illustrative and labelled synthetic in the document.
- Order/disorder: composition sets of a phase with equivalent sublattices reported in the disordered state (L12_FCC in
  Al–Ni) are renamed to the disordered lattice phase (FCC_A1) in the field grid and v2 data (cache v12).
- Stability limits: the extremum over composition of the temperature at which a phase disappears (bisection 0.01 K,
  golden-section in x), snapped to an ORCHID invariant or element melting point within 0.05 K, or within 2 K when the
  neighbouring state is not single-phase (ORCHID invariant temperatures carry ~0.5 K mapping error). Congruent points
  come from this search (one single-phase neighbour), not from the ORCHID boundaries.
- Homogeneity ranges are min/max equilibrium compositions on the 0.01 grid row (exact at two-phase cells); "< 300 K"
  lower limits are bounds and are not scored as temperatures.
- No metatectic or monotectoid occurs in the 20 systems; congruent transformation occurs once (σ in Cr–Fe). Five
  phases carry no structure label (not derivable from their TDB names: Al3Ni5, NiTi2, γ in Al–Cu, β and ε in Al–Mg).
  Cp(298.15 K) was finite for every intermediate phase. Cu–Pb contains a tiny HCP_A3 field near Pb from the dataset;
  it is listed under its TDB name.
