#!/usr/bin/env python3
"""Raster phase-diagram routes now that OpenCV / scikit-image are available (venv .venv-vision), using the
thermo_extractor code in ~/projects/orchid-pipeline/src unchanged:

  1. `extract_pdf_figure_panels` (thermo_extractor.raster.figure_panels) on the PDF of each rendering:
       vector -> vector.pdf; clean / degraded -> a Tesseract searchable PDF of the page images (baselines/ocr_pdf.py);
       image-only -> image_only.pdf itself first (no text layer: the tool finds no caption), then its Tesseract
       searchable wrapper. The binary_candidate panel of Fig. 1 is the crop; if none is found the truth figure bbox
       is used (oracle, recorded).
  2. `detect_binary_axes` (thermo_extractor.raster.axes, legacy title mode) on that crop, with the caption. If it does
     not calibrate, declared axes written from the truth frame (oracle, recorded) are passed with --declared-axes.
  3. `extract_binary_topology` (thermo_extractor.raster.topology): invariant horizontals + unlabelled faces.
  4. `compare_binary_topology` against `build_binary_topology` of the source TDB (baselines/tdb_topology.py):
     outcome and invariant matches recorded (database side is the truth source, so its labels are not scored).

Glue (this benchmark): field labels = printed words inside the detected frame (vector: PDF text layer; rasters:
Tesseract words), split on '+', placed with the tool's calibration. The topology tool emits no boundary geometry,
so boundary metrics are not applicable to this route.

Second output `assessed_pd_auto`: the assessed_pd annotator as in assessed_pd_raster.py but with the figure crop from
step 1 instead of the truth bbox, and calibration: assessed_pd automatic -> thermo_extractor axis detector (ticks and
frame passed as ManualCalibration) -> truth axes (oracle). The phase table is still truth (oracle).
"""

import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.assessed_pd_raster import merge_words, phase_table, text_layer_words  # noqa: E402
from baselines.ocr_cache import WORK as OCR  # noqa: E402
from baselines.ocr_cache import read_tsv
from common import ASSESSED_PD, bench, docs_dir, load_json, sha256_file, truth_dir, write_json  # noqa: E402
from score import aabb  # noqa: E402

PIPE = Path.home() / "projects/orchid-pipeline"
PY = sys.executable
OUT = bench / "outputs" / "raster_topology"
OUT2 = bench / "outputs" / "assessed_pd_auto"
WORK = bench / "work" / "raster_pd"
import os

ENV = {
    **{k: v for k, v in os.environ.items() if "KEY" not in k.upper()},
    "PYTHONPATH": str(PIPE / "src"),
    "OMP_THREAD_LIMIT": "1",
}
RENDS = ["vector", "clean_raster", "degraded_raster", "image_only_pdf"]


def sh(args):
    """Run a subprocess command and return its exit code and combined output."""
    r = subprocess.run([str(a) for a in args], capture_output=True, env=ENV)
    return r.returncode, (r.stdout + r.stderr).decode("utf-8", "replace")


def pdf_for(doc, rendering):
    """Return the (name, path) PDF files for a document and rendering."""
    if rendering == "vector":
        return [("vector.pdf", docs_dir / doc / "vector.pdf")]
    if rendering == "image_only_pdf":
        return [
            ("image_only.pdf (no text layer)", docs_dir / doc / "image_only.pdf"),
            ("image_only.pdf + Tesseract text layer", bench / "work" / "ocrpdf" / doc / "image_only_pdf.pdf"),
        ]
    return [("Tesseract searchable PDF", bench / "work" / "ocrpdf" / doc / f"{rendering}.pdf")]


def page_image(doc, rendering, page):
    """Return the page image path for a document, rendering and page index."""
    if rendering in ("vector", "clean_raster"):
        return docs_dir / doc / "clean" / f"page-{page + 1}.png"
    if rendering == "degraded_raster":
        return docs_dir / doc / "degraded" / f"page-{page + 1}.jpg"
    return next((OCR / doc / "image_only_pdf").glob(f"page-{page + 1}.*[gG]"))


def words_for(t, rendering, page):
    """Printed words in rendering pixels (vector: text layer scaled to the 300 dpi clean raster)."""
    if rendering == "vector":
        sc = t["renderings"]["clean_raster"]["transforms"][page][0][0]
        return merge_words(text_layer_words(t["doc_id"], page, sc)), "pdf text layer"
    return merge_words(read_tsv(OCR / t["doc_id"] / rendering / f"page-{page + 1}.tsv")), "tesseract words"


def find_panel(t, rendering, w):
    """Locate the phase-diagram panel of a figure in one rendering."""
    doc = t["doc_id"]
    fig = next(f for f in t["figures"] if f["kind"] == "phase_diagram")
    tried = []
    for label, pdf in pdf_for(doc, rendering):
        out = w / f"panels-{len(tried)}.json"
        rc, log = sh(
            [PY, "-m", "thermo_extractor.raster.figure_panels", pdf, "--panel-dir", w / f"panels-{len(tried)}", "--output", out]
        )
        rep = load_json(out) if out.exists() else {}
        cands = [
            (f_, p)
            for f_ in rep.get("figures", [])
            for p in f_.get("panels", [])
            if p.get("status") == "binary_candidate" and p.get("panel_type") == "phase_diagram"
        ]
        cands.sort(key=lambda q: (q[0].get("pdf_page") != fig["page"] + 1, -q[1].get("segmentation_score", 0)))
        tried.append(
            f"{label}: {len(rep.get('figures', []))} figure(s), {len(cands)} binary candidate(s)"
            + ("" if rc == 0 else f" (exit {rc}: {log.strip()[-120:]})")
        )
        if cands:
            f_, p = cands[0]
            return {
                "source": label,
                "caption": f_.get("caption") or "",
                "png": Path(p["panel_png"]),
                "crop_box_px": p["crop_box_pixels"],
                "page_pixels": f_["page_pixels"],
                "pdf_page": f_["pdf_page"],
                "oracle": False,
            }, tried
    # oracle fallback: truth figure bbox on the rendering's page image (300 dpi clean image for vector)
    from PIL import Image

    rkey = {"vector": "clean_raster"}.get(rendering, rendering)
    if rkey == "image_only_pdf":
        rkey = "degraded_raster"
    im = Image.open(page_image(doc, rendering, fig["page"])).convert("L")
    fb = aabb(fig["corners"][rkey])
    box = [max(0, int(fb[0]) - 4), max(0, int(fb[1]) - 4), min(im.width, int(fb[2]) + 4), min(im.height, int(fb[3]) + 4)]
    png = w / "oracle-crop.png"
    im.crop(box).save(png)
    return {
        "source": "truth figure bbox (oracle)",
        "caption": "",
        "png": png,
        "crop_box_px": {"x": box[0], "y": box[1], "width": box[2] - box[0], "height": box[3] - box[1]},
        "page_pixels": {"width": im.width, "height": im.height},
        "pdf_page": fig["page"] + 1,
        "oracle": True,
    }, tried


def declared_axes(t, panel, rendering, png):
    """Oracle fallback: the truth frame corners mapped into the crop, two ticks per axis."""
    fig = next(f for f in t["figures"] if f["kind"] == "phase_diagram")
    main = next(a for a in fig["axes"] if a["role"] == "main")
    rkey = {"vector": "clean_raster", "image_only_pdf": "degraded_raster"}.get(rendering, rendering)
    bx = aabb(main["corners"][rkey])
    k = page_scale(t, rendering, panel)
    cb = panel["crop_box_px"]
    x0, y0, x1, y1 = (bx[0] * k - cb["x"], bx[1] * k - cb["y"], bx[2] * k - cb["x"], bx[3] * k - cb["y"])
    tu = "celsius" if main["y"]["unit"] == "°C" else "kelvin"
    basis = "atom_percent" if main["x"]["unit"] == "at.%" else "mole_fraction"
    return {
        "schema_version": 1,
        "source_image_sha256": sha256_file(png),
        "declared_by": "benchmark oracle (truth key frame)",
        "locator": "Fig. 1",
        "reasoning": "automatic calibration failed; frame corners and axis limits taken from the truth key",
        "temperature": {
            "unit": tu,
            "ticks": [{"pixel_y": y1, "value": main["y"]["min"]}, {"pixel_y": y0, "value": main["y"]["max"]}],
        },
        "composition": {
            "basis": basis,
            "ticks": [{"pixel_x": x0, "value": main["x"]["min"]}, {"pixel_x": x1, "value": main["x"]["max"]}],
        },
    }


def oracle_calibration(t, panel, rendering, png):
    """Calibrate the panel axes from the truth coordinates (oracle calibration)."""
    d = declared_axes(t, panel, rendering, png)
    (ty0, ty1), (tx0, tx1) = [q["pixel_y"] for q in d["temperature"]["ticks"]], [q["pixel_x"] for q in d["composition"]["ticks"]]
    fig = next(f for f in t["figures"] if f["kind"] == "phase_diagram")
    main = next(a for a in fig["axes"] if a["role"] == "main")
    sx = (main["x"]["max"] - main["x"]["min"]) / (tx1 - tx0)
    sy = (main["y"]["max"] - main["y"]["min"]) / (ty1 - ty0)
    off = 273.15 if main["y"]["unit"] == "°C" else 0.0
    xs_ = 100.0 if main["x"]["unit"] == "at.%" else 1.0
    return {
        "status": "calibrated",
        "calibration_source": "benchmark_oracle_truth_frame",
        "source_image_sha256": sha256_file(png),
        "frame_pixels": {
            "left": int(round(tx0)),
            "right": int(round(tx1)),
            "top": int(round(ty1)),
            "bottom": int(round(ty0)),
            "closure": "oracle",
        },
        "composition_axis": {
            "status": "fitted",
            "slope_value_per_pixel": sx,
            "intercept_value": main["x"]["min"] - sx * tx0,
            "labelled_ticks": [{"pixel": q["pixel_x"], "value": q["value"]} for q in d["composition"]["ticks"]],
        },
        "temperature_axis": {
            "status": "fitted",
            "slope_value_per_pixel": sy,
            "intercept_value": main["y"]["min"] - sy * ty0,
            "labelled_ticks": [{"pixel": q["pixel_y"], "value": q["value"]} for q in d["temperature"]["ticks"]],
        },
        "domain": {
            "composition_basis": d["composition"]["basis"],
            "temperature_unit": d["temperature"]["unit"],
            "mole_fraction_available": True,
            "mole_fraction_min": main["x"]["min"] / xs_,
            "mole_fraction_max": main["x"]["max"] / xs_,
            "temperature_min_kelvin": main["y"]["min"] + off,
            "temperature_max_kelvin": main["y"]["max"] + off,
        },
    }


def page_scale(t, rendering, panel):
    """Rendering pixels -> panel-render pixels (vector: clean-raster pixels)."""
    rkey = {"vector": "clean_raster"}.get(rendering, rendering)
    if rkey == "image_only_pdf":
        rkey = "degraded_raster"
    page = panel["pdf_page"] - 1
    wr = t["renderings"][rkey]["files"][page]["size_px"][0]
    return panel["page_pixels"]["width"] / wr


def fit(axis):
    """Fit a linear axis mapping for one axis."""
    return lambda p: axis["intercept_value"] + axis["slope_value_per_pixel"] * p


def run(t, rendering):
    """Run the raster phase-diagram baseline on one document and rendering."""
    doc = t["doc_id"]
    w = WORK / doc / rendering
    w.mkdir(parents=True, exist_ok=True)
    st = {}
    panel, tried = find_panel(t, rendering, w)
    st["panels"] = "; ".join(tried)
    st["crop"] = panel["source"] + ("" if panel["oracle"] else " -> extract_pdf_figure_panels binary_candidate")
    png = panel["png"]
    cal_p = w / "axes.json"
    cap = w / "caption.txt"
    cap.write_text(panel["caption"], encoding="utf-8")
    rc, log = sh([PY, "-m", "thermo_extractor.raster.axes", png, "--output", cal_p, "--caption-file", cap])
    cal = load_json(cal_p) if cal_p.exists() else {}
    auto_ok = cal.get("status") == "calibrated"
    st["calibration"] = (
        "automatic: detect_binary_axes calibrated"
        if auto_ok
        else f"detect_binary_axes: {cal.get('status', 'error')} ({log.strip().splitlines()[-1][:120] if log.strip() else ''})"
    )
    auto_cal = cal if auto_ok else None
    if not auto_ok:
        dp = w / "declared_axes.json"
        dp.write_text(json.dumps(declared_axes(t, panel, rendering, png)), encoding="utf-8")
        rc, log = sh(
            [PY, "-m", "thermo_extractor.raster.axes", png, "--output", cal_p, "--caption-file", cap, "--declared-axes", dp]
        )
        cal = load_json(cal_p) if cal_p.exists() else {}
        st["calibration"] += f"; then declared axes from truth (oracle): {cal.get('status', 'error')}"
        if cal.get("status") != "calibrated":
            # the detector refuses before using declared ticks (frame not found): write the truth frame as the
            # calibration itself (oracle; same fields the detector emits) so the topology stage can still be measured
            cal = oracle_calibration(t, panel, rendering, png)
            cal_p.write_text(json.dumps(cal), encoding="utf-8")
            st["calibration"] += (
                f" ({log.strip().splitlines()[-1][:100] if log.strip() else ''}); calibration written from truth frame (oracle)"
            )
    topo_p = w / "topology.json"
    rc, log = sh([PY, PIPE / "scripts/extract_binary_topology.py", png, cal_p, "--output", topo_p])
    if rc != 0:
        raise RuntimeError(f"extract_binary_topology failed: {log.strip()[-200:]}")
    topo = load_json(topo_p)
    st["topology"] = f"{len(topo['invariant_candidates'])} invariant candidates, {len(topo['face_hypotheses'])} unlabelled faces"
    comp = bench / "work" / "tdb_topology" / f"{doc}.json"
    if comp.exists():
        cp = w / "compare.json"
        rc, log = sh([PY, PIPE / "scripts/compare_binary_topology.py", comp, topo_p, "--output", cp, "--format", "json"])
        c = load_json(cp) if cp.exists() else {}
        st["compare"] = c.get("verdict") or f"exit {rc}: {log.strip()[-160:]}"
        inv = c.get("invariants") or {}
        st["compare_detail"] = {
            "mode": c.get("mode"),
            "discrepancies": len(c.get("discrepancies") or []),
            **{k: v for k, v in inv.items() if not isinstance(v, (list, dict))},
        }
    # calibration maps (crop pixels -> value)
    ca, ta = cal["composition_axis"], cal["temperature_axis"]
    fx, fT = fit(ca), fit(ta)
    xs_ = 100.0 if (cal.get("domain") or {}).get("composition_basis") == "atom_percent" else 1.0
    tunit = (cal.get("domain") or {}).get("temperature_unit")
    toK = (lambda v: v + 273.15) if tunit == "celsius" else (lambda v: v)
    fr = cal["frame_pixels"]
    k = page_scale(t, rendering, panel)
    cb = panel["crop_box_px"]
    # frame bbox in the rendering's coordinates (pt for vector)
    to_r = 1.0 / k
    bb = [
        (cb["x"] + fr["left"]) * to_r,
        (cb["y"] + fr["top"]) * to_r,
        (cb["x"] + fr["right"]) * to_r,
        (cb["y"] + fr["bottom"]) * to_r,
    ]
    if rendering == "vector":
        sc = t["renderings"]["clean_raster"]["transforms"][panel["pdf_page"] - 1][0][0]
        bb = [v / sc for v in bb]
    fields = []
    words, src = words_for(t, rendering, panel["pdf_page"] - 1)
    for wd in words:
        x0, y0, x1, y1 = [v * k for v in wd["bbox"]]
        cx, cy = (x0 + x1) / 2 - cb["x"], (y0 + y1) / 2 - cb["y"]
        if not (fr["left"] < cx < fr["right"] and fr["top"] < cy < fr["bottom"]):
            continue
        s = wd["text"]
        if re.fullmatch(r"[\d.\s°CK]+", s) or not re.search(r"[A-Za-zα-ωΑ-Ω()]", s):
            continue
        fields.append({"point": [fx(cx) / xs_, toK(fT(cy))], "phases": [q.strip() for q in s.split("+") if q.strip()], "text": s})
    invs = [
        {"T": i["temperature_kelvin"], "type": None, "points": [{"x": i["mole_fraction_start"]}, {"x": i["mole_fraction_end"]}]}
        for i in topo["invariant_candidates"]
        if i.get("temperature_kelvin") is not None
    ]
    st["fields"] = f"{len(fields)} printed labels from {src} placed with the tool calibration"
    pdo = {
        "page": panel["pdf_page"] - 1,
        "bbox": bb,
        "x_unit": "mole_fraction",
        "T_unit": "K",
        "fields": fields,
        "boundaries": [],
        "invariants": invs,
        "unlabelled_faces": [
            {"point": [f_["mole_fraction"], f_["temperature_kelvin"]], "area_px": f_["area_pixels"]}
            for f_ in topo["face_hypotheses"]
            if "mole_fraction" in f_
        ],
    }
    return pdo, st, panel, auto_cal


# ------------------------------------------------------------------------------------------------ assessed_pd, auto crop
def run_assessed(t, rendering, panel, auto_cal):
    """Run the assessed_pd extraction on one panel, optionally with automatic calibration."""
    from PIL import Image

    sys.path.insert(0, str(ASSESSED_PD))
    from assessed_pd.calibration import CalibrationError
    from assessed_pd.config import AnnotatorConfig
    from assessed_pd.export import build_document
    from assessed_pd.model import ContainerSource, DiagramInput, ManualCalibration, PrintedSource, TextLabel, TickPair
    from assessed_pd.pipeline import annotate

    doc = t["doc_id"]
    fig = next(f for f in t["figures"] if f["kind"] == "phase_diagram")
    main = next(a for a in fig["axes"] if a["role"] == "main")
    page = panel["pdf_page"] - 1
    k = page_scale(t, rendering, panel)  # rendering px -> panel px
    cb = panel["crop_box_px"]
    im = Image.open(page_image(doc, rendering, page)).convert("L")
    # crop on the native page image (vector: the 300 dpi clean raster), same box as the tool's panel
    box = (int(cb["x"] / k), int(cb["y"] / k), int((cb["x"] + cb["width"]) / k), int((cb["y"] + cb["height"]) / k))
    w = WORK / doc / rendering
    crop_path = w / "assessed_crop.png"
    im.crop(box).save(crop_path)
    cx0, cy0, cx1, cy1 = box
    words, label_src = words_for(t, rendering, page)
    labels = [
        TextLabel(text=x["text"], bbox=(x["bbox"][0] - cx0, x["bbox"][1] - cy0, x["bbox"][2] - cx0, x["bbox"][3] - cy0))
        for x in words
        if x["bbox"][0] >= cx0 and x["bbox"][2] <= cx1 and x["bbox"][1] >= cy0 and x["bbox"][3] <= cy1
    ]
    xs_ = 100.0 if main["x"]["unit"] == "at.%" else 1.0
    A, B = t["elements"]
    tunit = "degC" if main["y"]["unit"] == "°C" else "K"
    s_px = 2.0 if rendering in ("vector", "clean_raster") else 1.0
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

    def mk():
        return DiagramInput(
            diagram_id=f"{doc}-{rendering}",
            system=f"{A}-{B}",
            elements=[A, B],
            axis_element=B,
            printed_source=PrintedSource(citation=f"ORCHID synthetic benchmark document {doc}"),
            container=ContainerSource(title=doc, rights="CC BY 4.0 (synthetic benchmark document)", locator="Fig. 1"),
            image_path=str(crop_path),
            labels=labels,
            phase_table=phase_table(t, xs_),
        )

    st = {
        "crop": panel["source"] + (" (oracle)" if panel["oracle"] else " via extract_pdf_figure_panels"),
        "labels": f"{len(labels)} from {label_src}",
        "phase_table": "truth (oracle)",
    }
    inp = mk()
    t0 = time.time()
    try:
        ann = annotate(inp, cfg)
        st["calibration"] = "automatic (assessed_pd ticks + printed labels)"
    except CalibrationError as e:
        ann = None
        if auto_cal is not None:
            # thermo_extractor axis detector -> ManualCalibration (pixels scaled from the panel render to the crop)
            fr = auto_cal["frame_pixels"]
            sc = 1.0 / k
            frame = (fr["left"] * sc, fr["top"] * sc, fr["right"] * sc, fr["bottom"] * sc)
            xt = [TickPair(p["pixel"] * sc, p["value"]) for p in auto_cal["composition_axis"]["labelled_ticks"]]
            yt = [TickPair(p["pixel"] * sc, p["value"]) for p in auto_cal["temperature_axis"]["labelled_ticks"]]
            inp = mk()
            inp.calibration = ManualCalibration(frame_px=frame, x_ticks=xt, y_ticks=yt)
            try:
                ann = annotate(inp, cfg)
                st["calibration"] = f"thermo_extractor detect_binary_axes after assessed_pd failure ({str(e)[:80]})"
            except CalibrationError as e2:
                st["calibration"] = f"assessed_pd failed ({str(e)[:60]}); detect_binary_axes ticks refused ({str(e2)[:60]})"
        if ann is None:
            rkey = {"vector": "clean_raster", "image_only_pdf": "degraded_raster"}.get(rendering, rendering)
            bx = aabb(main["corners"][rkey])
            fr = (bx[0] - cx0, bx[1] - cy0, bx[2] - cx0, bx[3] - cy0)
            inp = mk()
            inp.calibration = ManualCalibration(
                frame_px=fr,
                x_ticks=[TickPair(fr[0], main["x"]["min"]), TickPair(fr[2], main["x"]["max"])],
                y_ticks=[TickPair(fr[3], main["y"]["min"]), TickPair(fr[1], main["y"]["max"])],
            )
            ann = annotate(inp, cfg)
            st["calibration"] = (
                st.get("calibration", f"assessed_pd failed ({str(e)[:60]}); no detector calibration") + " -> truth axes (oracle)"
            )
    import pickle

    with open(w / "assessed_input.pkl", "wb") as fh:  # exact inputs, so the AI-reader stage can rebuild the annotation
        pickle.dump({"inp": inp, "cfg": cfg}, fh)
    d = build_document(ann)
    st["seconds"] = round(time.time() - t0, 1)
    (w / "assessed_pd.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    full = d["diagram"]["composition_full_scale"]
    bbox = [cx0, cy0, cx1, cy1]
    if rendering == "vector":
        sc = t["renderings"]["clean_raster"]["transforms"][page][0][0]
        bbox = [v / sc for v in bbox]
    pdo = {
        "page": page,
        "bbox": bbox,
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
    st["fields"] = f"{len(d['fields'])} traced, {sum(1 for f in d['fields'] if f['phases'])} named"
    return pdo, st


def one(tf, r):
    """Process one truth document for one rendering."""
    t = load_json(tf)
    doc = t["doc_id"]
    panel = auto_cal = None
    try:
        pdo, st, panel, auto_cal = run(t, r)
        res = {"phase_diagrams": [pdo], "status": {"phase_diagram": st}}
    except Exception as e:  # noqa: BLE001
        res = {
            "phase_diagrams": [],
            "status": {"phase_diagram": {"error": f"{type(e).__name__}: {str(e)[:200]}", "trace": traceback.format_exc()[-500:]}},
        }
    write_json(OUT / r / f"{doc}.json", {"doc_id": doc, "rendering": r, "extractor": "raster_topology", **res})
    if panel is not None:
        try:
            pdo2, st2 = run_assessed(t, r, panel, auto_cal)
            res2 = {"phase_diagrams": [pdo2], "status": {"phase_diagram": st2}}
        except Exception as e:  # noqa: BLE001
            res2 = {
                "phase_diagrams": [],
                "status": {
                    "phase_diagram": {"error": f"{type(e).__name__}: {str(e)[:200]}", "trace": traceback.format_exc()[-500:]}
                },
            }
        write_json(OUT2 / r / f"{doc}.json", {"doc_id": doc, "rendering": r, "extractor": "assessed_pd_auto", **res2})
    print(doc, r, json.dumps(res["status"]["phase_diagram"], ensure_ascii=False)[:300], flush=True)


def main(argv=None):
    """Run the raster phase-diagram baseline over all documents (command-line entry point)."""
    from multiprocessing import Pool

    args = argv or sys.argv[1:]
    rends = [a for a in args if a in RENDS] or RENDS
    docs = [a for a in args if a not in RENDS]
    tfs = [tf for tf in sorted(truth_dir.glob("*.json")) if not docs or tf.stem in docs]
    with Pool(8) as pool:
        pool.starmap(one, [(tf, r) for tf in tfs for r in rends])


if __name__ == "__main__":
    main()
