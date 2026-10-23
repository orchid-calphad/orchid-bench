#!/usr/bin/env python3
"""Searchable-PDF wrappers for the raster renderings (Tesseract `pdf` output: page image + invisible OCR text layer),
so the PDF-native thermo_extractor tools (extract_pdf_figure_panels needs captions from a text layer) can run on
scans. work/ocrpdf/<doc>/<rendering>.pdf, rendering in clean_raster (300 dpi PNG), degraded_raster (150 dpi JPEG),
image_only_pdf (the page images embedded in image_only.pdf, 150 dpi). Deterministic (single thread)."""
import subprocess, sys, tempfile
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BENCH, DOCS, TRUTH, load_json  # noqa: E402
OUT = BENCH / "work" / "ocrpdf"
OCR = BENCH / "work" / "ocr"


def pages(doc, rendering, n):
    if rendering == "clean_raster":
        return [(DOCS / doc / "clean" / f"page-{i}.png", 300) for i in range(1, n + 1)]
    if rendering == "degraded_raster":
        return [(DOCS / doc / "degraded" / f"page-{i}.jpg", 150) for i in range(1, n + 1)]
    return [(next((OCR / doc / "image_only_pdf").glob(f"page-{i}.*[gG]")), 150) for i in range(1, n + 1)]


def one(doc, rendering, n):
    import pymupdf
    out = OUT / doc / f"{rendering}.pdf"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    merged = pymupdf.open()
    with tempfile.TemporaryDirectory() as tmp:
        for k, (img, dpi) in enumerate(pages(doc, rendering, n)):
            base = Path(tmp) / f"p{k}"
            subprocess.run(["tesseract", str(img), str(base), "--dpi", str(dpi), "--psm", "3", "pdf"], check=True,
                           capture_output=True, env={"OMP_THREAD_LIMIT": "1", "PATH": "/opt/homebrew/bin:/usr/bin:/bin"})
            merged.insert_pdf(pymupdf.open(str(base) + ".pdf"))
    merged.save(out, deflate=True)


if __name__ == "__main__":
    jobs = []
    for tf in sorted(TRUTH.glob("*.json")):
        t = load_json(tf)
        for r in ("clean_raster", "degraded_raster", "image_only_pdf"):
            jobs.append((t["doc_id"], r, t["renderings"]["vector"]["pages"]))
    with Pool(8) as p:
        p.starmap(one, jobs)
    print(len(jobs), "searchable PDFs")
