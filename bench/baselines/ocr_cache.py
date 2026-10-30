#!/usr/bin/env python3
"""Run Tesseract (local binary, deterministic) on every raster page and cache word-level TSV.
work/ocr/<doc>/<rendering>/page-N.tsv ; rendering in clean_raster, degraded_raster, image_only_pdf
(image_only_pdf: the page images are extracted from the PDF itself, not taken from the degraded folder).
"""
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BENCH, DOCS, TRUTH, load_json  # noqa: E402

WORK = BENCH / "work" / "ocr"


def ocr(img, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return
    r = subprocess.run(["tesseract", str(img), "stdout", "--psm", "3", "tsv"], capture_output=True, text=True,
                       env={"OMP_THREAD_LIMIT": "1", "PATH": "/opt/homebrew/bin:/usr/bin:/bin"})
    out.write_text(r.stdout, encoding="utf-8")


def jobs():
    import pymupdf
    for tf in sorted(TRUTH.glob("*.json")):
        t = load_json(tf)
        d = DOCS / t["doc_id"]
        n = t["renderings"]["vector"]["pages"]
        for i in range(1, n + 1):
            yield d / "clean" / f"page-{i}.png", WORK / t["doc_id"] / "clean_raster" / f"page-{i}.tsv"
            yield d / "degraded" / f"page-{i}.jpg", WORK / t["doc_id"] / "degraded_raster" / f"page-{i}.tsv"
        # image-only PDF: extract the embedded image of each page
        ex = WORK / t["doc_id"] / "image_only_pdf"
        ex.mkdir(parents=True, exist_ok=True)
        doc = pymupdf.open(d / "image_only.pdf")
        for i, page in enumerate(doc, 1):
            xref = page.get_images()[0][0]
            info = doc.extract_image(xref)
            img = ex / f"page-{i}.{info['ext']}"
            if not img.exists():
                img.write_bytes(info["image"])
            yield img, ex / f"page-{i}.tsv"


def read_tsv(path):
    """Words with boxes and line keys."""
    words = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for ln in lines[1:]:
        f = ln.split("\t")
        if len(f) < 12 or f[0] != "5" or not f[11].strip():
            continue
        words.append({"text": f[11], "conf": float(f[10]), "bbox": [int(f[6]), int(f[7]), int(f[6]) + int(f[8]), int(f[7]) + int(f[9])],
                      "line": (int(f[2]), int(f[3]), int(f[4]))})
    return words


def tsv_text(path):
    words = read_tsv(path)
    out, cur, key = [], [], None
    for w in words:
        if w["line"] != key and cur:
            out.append(" ".join(cur))
            cur = []
        key = w["line"]
        cur.append(w["text"])
    if cur:
        out.append(" ".join(cur))
    return "\n".join(out)


if __name__ == "__main__":
    js = list(jobs())
    with Pool(8) as p:
        p.starmap(ocr, js)
    print(len(js), "pages OCR'd")
