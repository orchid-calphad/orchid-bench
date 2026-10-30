#!/usr/bin/env python3
"""Camelot table routes on the born-digital PDF (camelot-py 0.11.0 in .venv-vision):

  * `camelot_stream`, `camelot_network`: the thermo_extractor route itself (thermo_extractor.tables.extraction,
    `extract_assessment_table`): one hash-bound manifest per PDF page with a whole-page stream or network profile
    (no table_areas, i.e. no human-drawn region); build_report() runs Camelot and returns the raw matrices.
    The tool contract allows only stream and network.
  * `camelot_lattice`: Camelot lattice called directly (the tool does not allow lattice; glue, recorded).

Raster renderings: Camelot reads PDF text objects only; image-only PDFs have none (not run, see tables_ocr.py).
Semantic glue: the same header-keyword parser as tables_pdfplumber.py (no truth used)."""
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BENCH, DOCS, ORCHID_PIPELINE, TRUTH, load_json, sha256_file, write_json  # noqa: E402
from baselines.tables_pdfplumber import semantic  # noqa: E402

sys.path.insert(0, str(ORCHID_PIPELINE / "src"))
from thermo_extractor.tables import extraction as TX  # noqa: E402

WORK = BENCH / "work" / "camelot"


def matrix_cells(m):
    return [{"row": i, "col": j, "text": (c or "").replace("\n", " ").strip()} for i, r in enumerate(m) for j, c in enumerate(r)]


def tool_tables(doc, pdf, page, flavor):
    import camelot
    w = WORK / doc
    w.mkdir(parents=True, exist_ok=True)
    man = {"schema_version": 1, "id": f"{doc}-p{page}-{flavor}",
           "source": {"locator": f"page {page}", "page": page, "pdf_sha256": sha256_file(pdf), "table_label": f"all tables on page {page}"},
           "tool": {"package": "camelot-py", "version": camelot.__version__},
           "review": {"notes": "ORCHID synthetic benchmark; whole-page profile, no table areas", "origin": "calculated",
                      "selected_profile": None, "status": "needs_human_review"},
           "profiles": [{"id": flavor, "flavor": flavor}]}
    mp = w / f"p{page}-{flavor}.manifest.json"
    mp.write_text(json.dumps(man), encoding="utf-8")
    rep = TX.build_report(mp, pdf)
    (w / f"p{page}-{flavor}.report.json").write_text(TX.canonical_json(rep), encoding="utf-8")
    return [t["matrix"] for t in rep["profiles"][0]["tables"]]


class PdftoppmBackend:
    """Camelot lattice needs a PDF->PNG converter (Ghostscript or the pdftopng wheel, neither installed); this glue
    backend calls Poppler's pdftoppm at Camelot's default 300 dpi."""

    def convert(self, pdf_path, png_path, resolution=300):
        import subprocess
        stem = str(png_path)[:-4] if str(png_path).endswith(".png") else str(png_path)
        subprocess.run(["pdftoppm", "-r", str(resolution), "-png", "-singlefile", str(pdf_path), stem], check=True,
                       capture_output=True)


def lattice_tables(pdf, page):
    import camelot
    return [TX.dataframe_matrix(t.df) for t in camelot.read_pdf(str(pdf), pages=str(page), flavor="lattice", backend=PdftoppmBackend())]


def main():
    import camelot
    for tf in sorted(TRUTH.glob("*.json")):
        t = load_json(tf)
        doc = t["doc_id"]
        pdf = DOCS / doc / "vector.pdf"
        n = t["renderings"]["vector"]["pages"]
        for flavor in ("stream", "network", "lattice"):
            tabs, errs = [], []
            for page in range(1, n + 1):
                try:
                    ms = lattice_tables(pdf, page) if flavor == "lattice" else tool_tables(doc, pdf, page, flavor)
                except Exception as e:  # noqa: BLE001
                    errs.append(f"page {page}: {type(e).__name__}: {str(e)[:120]}")
                    continue
                for m in ms:
                    if len(m) >= 2:
                        tabs.append({"page": page - 1, "route": f"camelot {flavor}", "cells": matrix_cells(m)})
            inv, ph, recs = semantic(tabs)
            how = ("thermo_extractor.tables.extraction.build_report (whole-page profile)" if flavor != "lattice"
                   else "camelot.read_pdf flavor=lattice with a pdftoppm backend (outside the tool contract)")
            write_json(BENCH / "outputs" / f"camelot_{flavor}" / "vector" / f"{doc}.json",
                       {"doc_id": doc, "rendering": "vector", "extractor": f"camelot_{flavor}",
                        "status": {"tables": f"ran: camelot-py {camelot.__version__} {flavor} via {how}" + (f"; errors: {errs}" if errs else ""),
                                   "semantic": "ran: header-keyword glue (tables_pdfplumber.semantic)"},
                        "tables": tabs, "invariant_rows": inv, "phase_rows": ph, "property_records": recs})
        print(doc, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
