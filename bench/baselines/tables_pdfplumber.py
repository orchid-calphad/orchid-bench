#!/usr/bin/env python3
"""Table baseline on the vector PDF: pdfplumber table finder (Camelot, the thermo_extractor table route, is not
installed). Strategy 1: ruling lines both ways; strategy 2 (pages where 1 finds nothing): horizontal rules +
text-aligned columns. Raster renderings: not run (thermo_extractor.tables.ocr needs OpenCV)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BENCH, DOCS, TRUTH, load_json, write_json  # noqa: E402

OUT = BENCH / "outputs" / "tables_pdfplumber"


def tables_of(page):
    found = page.find_tables()
    route = "lines/lines"
    if not found:
        found = page.find_tables({"vertical_strategy": "text", "horizontal_strategy": "lines"})
        route = "text/lines"
    out = []
    for tb in found:
        rows = tb.extract()
        if not rows or len(rows) < 2:
            continue
        cells = [{"row": i, "col": j, "text": (c or "").replace("\n", " ")} for i, r in enumerate(rows) for j, c in enumerate(r)]
        out.append({"page": page.page_number - 1, "bbox": list(tb.bbox), "route": route, "cells": cells})
    return out


def main():
    import pdfplumber
    for tf in sorted(TRUTH.glob("*.json")):
        t = load_json(tf)
        doc = t["doc_id"]
        with pdfplumber.open(DOCS / doc / "vector.pdf") as pdf:
            tabs = [x for p in pdf.pages for x in tables_of(p)]
        write_json(OUT / "vector" / f"{doc}.json", {"doc_id": doc, "rendering": "vector", "extractor": "tables_pdfplumber",
                                                     "status": {"tables": "ran: pdfplumber 0.11.9"}, "tables": tabs})


if __name__ == "__main__":
    main()
