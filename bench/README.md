# ORCHID synthetic-papers benchmark (pilot)

Purpose: a ground-truthed test set for ORCHID's extraction tools. Twenty short synthetic "papers" (3 pages each), one
per binary system, generated from pycalphad calculations on SGTE datasets, each rendered four ways and
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
  systems.py         the 20 systems and their conventional phase labels
  figures.py         matplotlib figure generators (explicit axes, exact geometry)
  document.py        PyMuPDF composition (real text, vector figures, ruled tables)
  score.py           scoring harness
  baselines/         deterministic baselines (no model calls) + oracle self-test + run_all.py
  docs/<doc>/        vector.pdf, clean/page-N.png (300 dpi), degraded/page-N.jpg (150 dpi scan simulation),
                     image_only.pdf, figs/*.pdf (the figure PDFs as placed)
  truth/<doc>.json   canonical JSON truth key (schema orchid-synthetic-paper-truth/v1)
  outputs/<extractor>/<rendering>/<doc>.json   extractor outputs; scores.json per extractor
  results/           RESULTS.md, results.json
  showcase/index.html  self-contained page with three examples
  cache/, work/      intermediate data (cache: prepared system data; work: OCR TSV, tool inventories, crops)
```

## How to run

```
PY=~/orchid-py313/bin/python      # pycalphad 0.11.2, matplotlib, numpy, scipy, Pillow, PyMuPDF, pdfplumber
$PY bench/generate.py --jobs 8                     # ~10 s once bench/cache is filled (cache: ~3 min)
$PY bench/baselines/run_all.py                     # OCR (tesseract), baselines, scores, results
$PY bench/showcase/build.py
$PY bench/score.py --outputs bench/outputs/<your-extractor>
```

Environment knobs: `ORCHID_TDB_DIR`, `ORCHID_PD_DIR`, `OPEN_TDB`, `ASSESSED_PD_SRC`.

## Documents

Each document has: title, "ORCHID synthetic benchmark" byline, licence notice, abstract, sections with running text
that states values (invariant temperatures and compositions, melting points, congruent melting, mixing-enthalpy
extremum, formation enthalpy) with varied phrasing and units (K / °C, at.% / mole fraction, kJ/mol / J/mol), and:

- Fig. 1 phase diagram, styles drawn per document from the seed: line width, black or coloured, labelled or
  unlabelled fields (conventional labels or TDB names), K or °C, at.% or mole fraction, invariant temperature labels,
  optional inset zoom, optional experimental-style markers (two series with legend), optional dashed second
  "assessment" shifted by a known (dx, dT).
- Two property figures from: liquid mixing enthalpy (one or two temperatures), activities (linear or log, Raoult
  lines), Cp(T) of a compound, Gibbs-energy curves with common tangent, two-panel ΔmixH/ΔmixG; lines, markers or both.
- Tables: invariant reactions (type, T, phase compositions; merged header or not) or liquidus/solidus temperatures
  (systems without invariants); formation enthalpy/entropy of intermediate phases (optional ± values, explicitly
  labelled synthetic) or liquid Redlich–Kister parameters as printed in the TDB. Styles: booktabs, full grid,
  zebra/plain; optional footnotes.

Renderings: (a) `vector.pdf` born-digital, text layer present; (b) `clean/` 300 dpi PNG; (c) `degraded/` 150 dpi
JPEG with contrast loss, blur, rotation within ±1°, salt-and-pepper noise, JPEG quality 30–55, half in greyscale;
(d) `image_only.pdf` = (c) wrapped full-page, no text layer.

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
- `tables[]`: every cell {row, col, colspan, header, text, value, uncertainty, unit, bbox, corners}.
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

`bbox` is in the rendering's own coordinates; panels and diagrams are matched to the truth by bbox overlap. Field
phases may be conventional labels (`L`, `(Cu)`, `Ni3Al`) or TDB names (`LIQUID`, `FCC_A1`).

## Metrics (score.py)

- Tables: best-aligned cell exact match (strict and whitespace/minus-normalised), numeric exact and relative error.
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
