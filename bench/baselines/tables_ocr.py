#!/usr/bin/env python3
"""OCR table route: thermo_extractor `ocr_table_cells` (orchid-pipeline scripts/ocr_table_cells.py, Tesseract on a
page render, words grouped into lines and placed into declared columns) on every rendering.

The tool's contract requires the table region and the column boundaries to be declared by a person. Here both come
from the truth key (oracle, recorded): region = the aabb of the segment's cells in that rendering (+ 0.4 % margin),
column boundaries = midpoints between the x-extents of adjacent single-span columns. So this measures what
Tesseract reads in a perfectly declared layout, not table finding. Rendering inputs: vector.pdf (rendered),
the Tesseract searchable PDFs of clean / degraded (work/ocrpdf), image_only.pdf. dpi 400, psm 6 (tool defaults
for a table block). Semantic records: tables_pdfplumber.semantic. Continued tables are separate segments."""
import json
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BENCH, DOCS, ORCHID_PIPELINE, TRUTH, load_json, write_json  # noqa: E402
from baselines.tables_pdfplumber import semantic  # noqa: E402

PIPE = ORCHID_PIPELINE
PY = sys.executable
OUT = BENCH / "outputs" / "ocr_table_cells"
WORK = BENCH / "work" / "ocr_table_cells"
RENDS = ["vector", "clean_raster", "degraded_raster", "image_only_pdf"]


def pdf_for(doc, r):
    return {"vector": DOCS / doc / "vector.pdf", "image_only_pdf": DOCS / doc / "image_only.pdf"}.get(
        r, BENCH / "work" / "ocrpdf" / doc / f"{r}.pdf")


def page_size(t, r, pdf):
    if r in ("clean_raster", "degraded_raster"):
        return t["renderings"][r]["files"][0]["size_px"]
    import pymupdf
    rect = pymupdf.open(pdf)[0].rect
    return [rect.width, rect.height]


def layout(segment_cells, r, W, H):
    xs0 = [min(p[0] for p in c["corners"][r]) for c in segment_cells]
    xs1 = [max(p[0] for p in c["corners"][r]) for c in segment_cells]
    ys0 = [min(p[1] for p in c["corners"][r]) for c in segment_cells]
    ys1 = [max(p[1] for p in c["corners"][r]) for c in segment_cells]
    mx, my = 0.004 * W, 0.004 * H
    L, R_, T, B = max(0, min(xs0) - mx), min(W, max(xs1) + mx), max(0, min(ys0) - my), min(H, max(ys1) + my)
    cols = {}
    for c, a, b in zip(segment_cells, xs0, xs1):
        if c.get("colspan", 1) == 1:
            lo, hi = cols.get(c["col"], (a, b))
            cols[c["col"]] = (min(lo, a), max(hi, b))
    ks = sorted(cols)
    bounds = []
    for j, k in zip(ks, ks[1:]):
        m = 0.5 * (cols[j][1] + cols[k][0])
        f = (m - L) / (R_ - L)
        if bounds and f <= bounds[-1] + 1e-4:
            f = bounds[-1] + 1e-3
        bounds.append(round(min(max(f, 1e-3), 0.999), 4))
    region = [round(T / H, 4), round(B / H, 4), round(L / W, 4), round(R_ / W, 4)]
    return region, bounds


def main():
    from multiprocessing import Pool
    jobs = []
    for tf in sorted(TRUTH.glob("*.json")):
        for r in RENDS:
            jobs.append((str(tf), r))
    with Pool(8) as p:
        for line in p.starmap(one, jobs):
            print(line, flush=True)


def one(tf, r):
    t = load_json(tf)
    doc = t["doc_id"]
    pdf = pdf_for(doc, r)
    W, H = page_size(t, r, pdf)
    tabs, status = [], []
    for tb in t["tables"]:
        segs = {}
        for c in tb["cells"]:
            segs.setdefault(c.get("segment", 0), []).append(c)
        for si, cells in sorted(segs.items()):
            page = cells[0]["page"]
            region, bounds = layout(cells, r, W, H)
            w = WORK / doc / r
            w.mkdir(parents=True, exist_ok=True)
            out = w / f"{tb['id']}-{si}.json"
            cmd = [PY, str(PIPE / "scripts/ocr_table_cells.py"), str(pdf), "--page", str(page + 1),
                   "--region", ",".join(str(v) for v in region), "--columns", ",".join(str(v) for v in bounds),
                   "--dpi", "400", "--psm", "6", "--output", str(out)]
            rr = subprocess.run(cmd, capture_output=True)
            if rr.returncode != 0 or not out.exists():
                status.append(f"{tb['id']}[{si}]: failed {rr.stderr.decode('utf-8', 'replace')[-160:]}")
                continue
            rep = load_json(out)
            tabs.append({"page": page, "route": "ocr_table_cells (declared region/columns from truth)", "table_id": tb["id"],
                         "cells": [{"row": i, "col": j, "text": v} for i, row in enumerate(rep["cells"]) for j, v in enumerate(row)]})
            status.append(f"{tb['id']}[{si}]: {len(rep['cells'])} rows, median conf {rep.get('median_confidence')}")
    inv, ph, recs = semantic(tabs)
    write_json(OUT / r / f"{doc}.json", {"doc_id": doc, "rendering": r, "extractor": "ocr_table_cells",
                                         "status": {"tables": "; ".join(status), "region_columns": "truth (oracle)"},
                                         "tables": tabs, "invariant_rows": inv, "phase_rows": ph, "property_records": recs})
    return f"{doc} {r} {len(tabs)} tables"


if __name__ == "__main__":
    main()
