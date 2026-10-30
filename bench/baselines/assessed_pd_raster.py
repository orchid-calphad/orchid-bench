#!/usr/bin/env python3
"""Phase-diagram baseline: the assessed-phase-diagram annotator (assessed_pd, deterministic layer only, no readers)
on a crop of the figure from each rendering.

Inputs given to the tool, and where they come from (recorded per run in the status block):
  * crop of the figure region           - truth figure bbox (oracle; a panel segmenter would need OpenCV)
  * printed labels with boxes           - vector: PDF text layer; rasters: Tesseract word boxes, words of one
                                          OCR line joined when the gap is under one word height
  * phase table (names, ranges)         - truth (oracle); the synthetic papers print no phase table
  * axis calibration                    - automatic tick/label detection first; on failure, manual frame + two
                                          ticks per axis from the truth axes (oracle, recorded)
"""

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.ocr_cache import WORK as OCR  # noqa: E402
from baselines.ocr_cache import read_tsv
from common import ASSESSED_PD, bench, docs_dir, truth_dir, load_json, write_json  # noqa: E402
from score import aabb  # noqa: E402

sys.path.insert(0, str(ASSESSED_PD))
from assessed_pd.calibration import CalibrationError  # noqa: E402
from assessed_pd.config import AnnotatorConfig  # noqa: E402
from assessed_pd.export import build_document  # noqa: E402
from assessed_pd.model import (  # noqa: E402
    ContainerSource,
    DiagramInput,
    ManualCalibration,
    PhaseRow,
    PrintedSource,
    TextLabel,
    TickPair,
)
from assessed_pd.pipeline import annotate  # noqa: E402

OUT = bench / "outputs" / "assessed_pd"
WORK = bench / "work" / "assessed_pd"


def merge_words(words):
    out = []
    for w in sorted(words, key=lambda w: (w["line"], w["bbox"][0])):
        if out and out[-1]["line"] == w["line"]:
            p = out[-1]
            h = max(p["bbox"][3] - p["bbox"][1], w["bbox"][3] - w["bbox"][1])
            if w["bbox"][0] - p["bbox"][2] < 1.0 * h:
                p["text"] += " " + w["text"]
                p["bbox"] = [
                    min(p["bbox"][0], w["bbox"][0]),
                    min(p["bbox"][1], w["bbox"][1]),
                    max(p["bbox"][2], w["bbox"][2]),
                    max(p["bbox"][3], w["bbox"][3]),
                ]
                continue
        out.append({**w, "bbox": list(w["bbox"])})
    return out


def text_layer_words(doc, page, scale):
    import pdfplumber

    with pdfplumber.open(docs_dir / doc / "vector.pdf") as pdf:
        ws = pdf.pages[page].extract_words(keep_blank_chars=False, use_text_flow=True)
    return [
        {
            "text": w["text"],
            "bbox": [w["x0"] * scale, w["top"] * scale, w["x1"] * scale, w["bottom"] * scale],
            "line": (round(w["top"], 0),),
        }
        for w in ws
    ]


def phase_table(t, xs_):
    rng = {}
    for f in t["phase_diagram"]["fields"]:
        for lab, x in zip(f["labels"], f["compositions"]):
            lo, hi = rng.get(lab, (x, x))
            rng[lab] = (min(lo, x), max(hi, x))
    return [PhaseRow(name=k, lo=round(v[0] * xs_, 4), hi=round(v[1] * xs_, 4), printed=None) for k, v in sorted(rng.items())]


def page_image(doc, rendering, page):
    if rendering in ("vector", "clean_raster"):
        return docs_dir / doc / "clean" / f"page-{page + 1}.png"
    if rendering == "degraded_raster":
        return docs_dir / doc / "degraded" / f"page-{page + 1}.jpg"
    return next((OCR / doc / "image_only_pdf").glob(f"page-{page + 1}.*[gG]"))


def run(t, rendering):
    from PIL import Image

    doc = t["doc_id"]
    fig = next(f for f in t["figures"] if f["kind"] == "phase_diagram")
    main = next(a for a in fig["axes"] if a["role"] == "main")
    page = fig["page"]
    img_path = page_image(doc, rendering, page)
    im = Image.open(img_path).convert("L")
    rkey = {"vector": "clean_raster", "image_only_pdf": "degraded_raster"}.get(rendering, rendering)
    fb = aabb(fig["corners"][rkey])
    pad = 4
    cx0, cy0 = max(0, int(fb[0]) - pad), max(0, int(fb[1]) - pad)
    cx1, cy1 = min(im.width, int(fb[2]) + pad), min(im.height, int(fb[3]) + pad)
    crop = im.crop((cx0, cy0, cx1, cy1))
    w = WORK / doc / rendering
    w.mkdir(parents=True, exist_ok=True)
    crop_path = w / "figure.png"
    crop.save(crop_path)
    if rendering == "vector":
        sc = t["renderings"]["clean_raster"]["transforms"][page][0][0]
        words = text_layer_words(doc, page, sc)
        label_src = "pdf text layer"
    else:
        words = read_tsv(OCR / doc / rendering / f"page-{page + 1}.tsv")
        label_src = "tesseract word boxes"
    words = merge_words(words)
    labels = [
        TextLabel(text=x["text"], bbox=(x["bbox"][0] - cx0, x["bbox"][1] - cy0, x["bbox"][2] - cx0, x["bbox"][3] - cy0))
        for x in words
        if x["bbox"][0] >= cx0 and x["bbox"][2] <= cx1 and x["bbox"][1] >= cy0 and x["bbox"][3] <= cy1
    ]
    xs_ = 100.0 if main["x"]["unit"] == "at.%" else 1.0
    A, B = t["elements"]
    tunit = "degC" if main["y"]["unit"] == "°C" else "K"
    s_px = 2.0 if rendering in ("vector", "clean_raster") else 1.0  # pixel thresholds scaled with render dpi (300 vs 150)
    cfg = AnnotatorConfig.from_dict(
        {
            "axes": {
                "composition_basis": "atomic_percent" if xs_ == 100 else "mole_fraction",
                "temperature_unit": tunit,
                "x_fit_tolerance": 0.5 if xs_ == 100 else 0.005,
                "y_fit_tolerance": 3.0,
                "tick_max_len_px": int(14 * s_px),
                "tick_min_len_px": int(3 * s_px),
                "tick_label_max_gap_px": int(30 * s_px),
                "tick_label_max_offset_px": 6.0 * s_px,
            },
            "naming": {
                "temperature_label_pattern": r"(-?\d+(?:\.\d+)?)\s*°C" if tunit == "degC" else r"(-?\d+(?:\.\d+)?)\s*K",
                "range_tolerance": 3.0 * (xs_ / 100),
                "table_fit_tolerance": 1.5 * (xs_ / 100),
                "tie_line_range_tolerance": 2.0 * (xs_ / 100),
                "line_compound_tolerance": 0.6 * (xs_ / 100),
                "tie_line_step": 10.0,
            },
            "tracing": {"ink_threshold": 160, "min_field_px": 60 if rendering in ("vector", "clean_raster") else 15},
        }
    )
    inp = DiagramInput(
        diagram_id=f"{doc}-{rendering}",
        system=f"{A}-{B}",
        elements=[A, B],
        axis_element=B,
        printed_source=PrintedSource(
            citation=f"ORCHID synthetic benchmark document {doc} (calculated, SGTE {t['source_tdb']['file']})"
        ),
        container=ContainerSource(title=doc, rights="CC BY 4.0 (synthetic benchmark document)", locator="Fig. 1"),
        image_path=str(crop_path),
        labels=labels,
        phase_table=phase_table(t, xs_),
    )
    status = {"labels": f"{len(labels)} from {label_src}", "crop": "truth figure bbox (oracle)", "phase_table": "truth (oracle)"}
    t0 = time.time()
    try:
        ann = annotate(inp, cfg)
        status["calibration"] = "automatic (ticks + printed labels)"
    except CalibrationError as e:
        q = main["corners"][rkey]
        bx = aabb(q)
        fr = (bx[0] - cx0, bx[1] - cy0, bx[2] - cx0, bx[3] - cy0)
        inp.calibration = ManualCalibration(
            frame_px=fr,
            x_ticks=[TickPair(fr[0], main["x"]["min"]), TickPair(fr[2], main["x"]["max"])],
            y_ticks=[TickPair(fr[3], main["y"]["min"]), TickPair(fr[1], main["y"]["max"])],
        )
        ann = annotate(inp, cfg)
        status["calibration"] = f"manual from truth axes (oracle) after automatic failure: {str(e)[:120]}"
    d = build_document(ann)
    status["seconds"] = round(time.time() - t0, 1)
    (w / "assessed_pd.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    full = d["diagram"]["composition_full_scale"]
    pdo = {
        "page": page,
        "bbox": [cx0, cy0, cx1, cy1] if rendering != "vector" else None,
        "x_unit": "mole_fraction",
        "T_unit": "°C" if tunit == "degC" else "K",
        "fields": [
            {"point": [f["label_point"]["x"] / full, f["label_point"]["T"]], "phases": f["phases"]}
            for f in d["fields"]
            if f["phases"]
        ],
        "boundaries": [{"points": [[x / full, T] for x, T in b["points"]]} for b in d["boundaries"]],
        "invariants": [{"T": i["T"], "type": i.get("type"), "points": []} for i in d["invariants"]],
    }
    if rendering == "vector":
        pdo["bbox"] = aabb(fig["corners"]["vector"])
    status["fields"] = f"{len(d['fields'])} traced, {sum(1 for f in d['fields'] if f['phases'])} named"
    return pdo, status


def one(tf, r):
    t = load_json(tf)
    if True:
        if True:
            try:
                pdo, st = run(t, r)
                res = {"phase_diagrams": [pdo], "status": {"phase_diagram": st}}
            except Exception as e:  # noqa: BLE001
                res = {
                    "phase_diagrams": [],
                    "status": {
                        "phase_diagram": {"error": f"{type(e).__name__}: {str(e)[:200]}", "trace": traceback.format_exc()[-600:]}
                    },
                }
            write_json(
                OUT / r / f"{t['doc_id']}.json", {"doc_id": t["doc_id"], "rendering": r, "extractor": "assessed_pd", **res}
            )
            print(t["doc_id"], r, json.dumps(res["status"]["phase_diagram"], ensure_ascii=False)[:220], flush=True)


def main(argv=None):
    from multiprocessing import Pool

    rends = (argv or sys.argv[1:]) or ["vector", "clean_raster", "degraded_raster", "image_only_pdf"]
    with Pool(8) as pool:
        pool.starmap(one, [(tf, r) for tf in sorted(truth_dir.glob("*.json")) for r in rends])


if __name__ == "__main__":
    main()
