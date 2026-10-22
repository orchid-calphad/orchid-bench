#!/usr/bin/env python3
"""Running-text baseline: regular expressions for value + unit over the PDF text layer (vector, image-only PDF)
or Tesseract OCR text (rasters; image-only PDF falls back to OCR when the text layer is empty)."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BENCH, TRUTH, DOCS, load_json, write_json  # noqa: E402
from baselines.ocr_cache import WORK, tsv_text  # noqa: E402

OUT = BENCH / "outputs" / "text_regex"
NUM = r"([-−–]?\d+(?:\.\d+)?)"
PATTERNS = [
    (re.compile(NUM + r"\s?°\s?C\b"), "°C"),
    (re.compile(r"(?<![\w.])" + NUM + r"\s?K\b"), "K"),
    (re.compile(NUM + r"\s?at\.?\s?%"), "at.%"),
    (re.compile(r"x\(\s?[A-Z][a-z]?\s?\)\s?=\s?(\d?\.\d+)"), "mole_fraction"),
    (re.compile(r"mole fraction of\s+(\d?\.\d+)"), "mole_fraction"),
    (re.compile(NUM + r"\s?kJ\s?/\s?mol"), "kJ/mol"),
    (re.compile(r"(?<![k\w])" + NUM + r"\s?J\s?/\s?mol(?!\w)"), "J/mol"),
    (re.compile(NUM + r"\s?atm\b"), "atm"),
]


def extract(text):
    text = text.replace("\n", " ")
    vals = []
    for rx, unit in PATTERNS:
        for m in rx.finditer(text):
            s = m.group(1).replace("−", "-").replace("–", "-")
            try:
                vals.append({"value": float(s), "unit": unit, "text": m.group(0)})
            except ValueError:
                pass
    return vals


def main():
    import pdfplumber
    for tf in sorted(TRUTH.glob("*.json")):
        t = load_json(tf)
        doc = t["doc_id"]
        n = t["renderings"]["vector"]["pages"]
        texts = {}
        with pdfplumber.open(DOCS / doc / "vector.pdf") as pdf:
            texts["vector"] = ("pdf_text_layer", "\n".join(p.extract_text() or "" for p in pdf.pages))
        with pdfplumber.open(DOCS / doc / "image_only.pdf") as pdf:
            layer = "\n".join(p.extract_text() or "" for p in pdf.pages)
        if layer.strip():
            texts["image_only_pdf"] = ("pdf_text_layer", layer)
        else:
            texts["image_only_pdf"] = ("text layer empty -> tesseract OCR of embedded page images",
                                       "\n".join(tsv_text(WORK / doc / "image_only_pdf" / f"page-{i}.tsv") for i in range(1, n + 1)))
        for r in ("clean_raster", "degraded_raster"):
            texts[r] = ("tesseract OCR", "\n".join(tsv_text(WORK / doc / r / f"page-{i}.tsv") for i in range(1, n + 1)))
        for r, (route, txt) in texts.items():
            write_json(OUT / r / f"{doc}.json", {"doc_id": doc, "rendering": r, "extractor": "text_regex",
                                                  "status": {"text": f"ran: {route}"}, "text_values": extract(txt)})


if __name__ == "__main__":
    main()
