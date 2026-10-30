#!/usr/bin/env python3
"""Raster marker route: thermo_extractor.proposals.multipanel_markers (`propose_multipanel_markers`, OpenCV template
matching against the legend marker) on the clean, degraded and image-only renderings.

The detector needs a calibrated multipanel manifest (panel bounds, two calibration ticks per axis, legend bounds and
the legend marker pixel). No raster manifest proposer exists for XY plots (propose_multipanel_manifest reads PDF
geometry), so the manifests are those vector_figures.py proposed from the born-digital twin's page geometry:
  * clean_raster: used as proposed (they are written in the 300 dpi clean-raster pixel frame) -> inputs from the
    vector twin, recorded;
  * degraded_raster / image_only_pdf: the same manifest mapped into the degraded page with the truth page transform
    (oracle: the rotation / scale of the scan simulation), rebound to the degraded image and image_only.pdf.
Only legend entries the manifest proposer classed as points are detected; curves are out of scope for this tool.
Panels without a vector manifest (calibration or legend failure on the vector route) are not attempted.
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import bench, docs_dir, load_json, sha256_file, truth_dir, write_json  # noqa: E402

sys.path.insert(0, str(Path.home() / "projects/orchid-pipeline/src"))
from thermo_extractor.proposals.multipanel_markers import propose_multipanel_markers  # noqa: E402

OUT = bench / "outputs" / "raster_markers"
WORK = bench / "work" / "raster_xy"
RENDS = ["clean_raster", "degraded_raster", "image_only_pdf"]


def affine(m):
    """Convert a mapping description into an affine matrix."""
    return np.array([[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [0, 0, 1.0]])


def mapper(t, page, r):
    """clean-raster px -> rendering px (identity for clean)."""
    if r == "clean_raster":
        return lambda x, y: (x, y)
    C = affine(t["renderings"]["clean_raster"]["transforms"][page])
    D = affine(t["renderings"]["degraded_raster"]["transforms"][page])
    M = D @ np.linalg.inv(C)
    return lambda x, y: tuple((M @ np.array([x, y, 1.0]))[:2])


def remap(man, f, img, pdf):
    """Map manifest coordinates of one figure onto the given rendering."""
    m = json.loads(json.dumps(man))
    for p in m["panels"]:
        b = p["bounds"]
        xm, ym = (b["left"] + b["right"]) / 2, (b["top"] + b["bottom"]) / 2
        for c in p["axes"]["x"]["calibration"]:
            c["pixel"] = round(f(c["pixel"], b["bottom"])[0], 3)
        for c in p["axes"]["y"]["calibration"]:
            c["pixel"] = round(f(b["left"], c["pixel"])[1], 3)
        for key in ("bounds",):
            pts = [f(b["left"], b["top"]), f(b["right"], b["top"]), f(b["right"], b["bottom"]), f(b["left"], b["bottom"])]
            p[key] = {
                "left": round(min(q[0] for q in pts), 3),
                "right": round(max(q[0] for q in pts), 3),
                "top": round(min(q[1] for q in pts), 3),
                "bottom": round(max(q[1] for q in pts), 3),
            }
        lg = p["legend"]
        if lg.get("bounds"):
            lb = lg["bounds"]
            pts = [f(lb["left"], lb["top"]), f(lb["right"], lb["bottom"])]
            lg["bounds"] = {
                "left": round(min(q[0] for q in pts), 3),
                "right": round(max(q[0] for q in pts), 3),
                "top": round(min(q[1] for q in pts), 3),
                "bottom": round(max(q[1] for q in pts), 3),
            }
        for e in lg.get("entries", []):
            mp = e["source"]["marker_pixel"]
            x, y = f(mp["x"], mp["y"])
            e["source"]["marker_pixel"] = {"x": round(x, 3), "y": round(y, 3)}
        del xm, ym
    m["source"]["image_sha256"] = sha256_file(img)
    m["source"]["pdf_sha256"] = sha256_file(pdf)
    return m


def one(tf, r):
    """Process one truth document for one rendering."""
    t = load_json(tf)
    doc = t["doc_id"]
    panels, status = [], []
    for mp in sorted((bench / "work" / "vector" / doc).glob("*/manifest.json")):
        man = load_json(mp)
        page = man["source"]["pdf_page"] - 1
        try:
            if r == "clean_raster":
                img, pdf, use = docs_dir / doc / "clean" / f"page-{page + 1}.png", docs_dir / doc / "vector.pdf", mp
            else:
                img = (
                    docs_dir / doc / "degraded" / f"page-{page + 1}.jpg"
                    if r == "degraded_raster"
                    else next((bench / "work" / "ocr" / doc / "image_only_pdf").glob(f"page-{page + 1}.*[gG]"))
                )
                pdf = docs_dir / doc / "image_only.pdf"
                png = WORK / doc / r / f"page-{page + 1}.png"  # the tool accepts PNG only: decoded JPEG pixels, lossless
                if not png.exists():
                    from PIL import Image

                    png.parent.mkdir(parents=True, exist_ok=True)
                    Image.open(img).save(png)
                img = png
                use = WORK / doc / r / mp.parent.name / "manifest.json"
                use.parent.mkdir(parents=True, exist_ok=True)
                use.write_text(json.dumps(remap(man, mapper(t, page, r), img, pdf)), encoding="utf-8")
            data = propose_multipanel_markers(use, pdf, img)
        except Exception as e:  # noqa: BLE001
            status.append({"panel": mp.parent.name, "markers": f"failed: {type(e).__name__}: {str(e)[:160]}"})
            continue
        grp = {}
        for q in data.get("points", []):
            if q.get("status") == "rejected":
                continue
            grp.setdefault(str(q.get("legend_entry_id") or q.get("series_id") or "all"), []).append(
                [q["coordinates"]["x"], q["coordinates"]["y"]]
            )
        pm = load_json(use)["panels"][0]["bounds"]
        bbox = [pm["left"], pm["top"], pm["right"], pm["bottom"]]
        if r == "image_only_pdf":  # image_only coordinates are PDF points: 150 dpi image on the page
            s = 72.0 / 150.0
            bbox = [v * s for v in bbox]
        panels.append({"page": page, "bbox": bbox, "series": [{"name": k, "points": v} for k, v in grp.items()]})
        auto = sum(1 for q in data.get("points", []) if q.get("status") == "accepted")
        status.append({"panel": mp.parent.name, "markers": f"ok ({len(data.get('points', []))} proposals, {auto} auto-accepted)"})
    src = (
        "manifests from the vector twin (clean-raster pixel frame)"
        if r == "clean_raster"
        else "vector-twin manifests mapped with the truth page transform (oracle)"
    )
    write_json(
        OUT / r / f"{doc}.json",
        {
            "doc_id": doc,
            "rendering": r,
            "extractor": "raster_markers",
            "status": {"xy": status, "manifest": src},
            "xy_figures": [{"panels": panels}] if panels else [],
        },
    )
    return doc, r, len(panels), sum(1 for s in status if s["markers"].startswith("failed"))


if __name__ == "__main__":
    jobs = [(tf, r) for tf in sorted(truth_dir.glob("*.json")) for r in RENDS]
    with Pool(8) as p:
        for res in p.starmap(one, jobs):
            print(*res, flush=True)
