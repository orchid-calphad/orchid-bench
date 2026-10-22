#!/usr/bin/env python3
"""Vector-figure baseline on the born-digital PDF, built from the open-tdb thermo_extractor PDF route:

  1. `inspect-pdf` object inventory (pdfplumber) and `propose-pdf-graphics` geometric role proposals;
  2. axis calibration from the page geometry with the frame/tick/label stage of
     `propose_multipanel_manifest` (thermo_extractor.proposals.vector_manifest), given the figure region and axis
     quantity/unit as hints (the tool's contract: a human supplies them; here they come from the truth key);
  3. the full `propose_multipanel_manifest` + `propose_vector_markers` route, attempted for every xy panel;
  4. glue written for this benchmark: paths whose top role is `data_curve` -> calibrated polylines; paths whose top
     role is `marker` -> calibrated points grouped by paint style; for phase diagrams, curve/line paths ->
     boundaries, horizontal lines -> invariant temperatures, text-layer words inside the frame -> field labels.
"""
import json
import re
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BENCH, DOCS, OPEN_TDB, TRUTH, load_json, write_json  # noqa: E402

sys.path.insert(0, str(OPEN_TDB / "src"))
from thermo_extractor.proposals import vector_manifest as VM  # noqa: E402
from thermo_extractor.proposals.vector_markers import page_transform, propose_vector_markers  # noqa: E402

OUT = BENCH / "outputs" / "vector_route"
WORK = BENCH / "work" / "vector"
PY = sys.executable


def run_inventory(doc):
    w = WORK / doc
    w.mkdir(parents=True, exist_ok=True)
    inv, gr = w / "inventory.json", w / "graphics.json"
    env = {"PYTHONPATH": str(OPEN_TDB / "src"), "PATH": "/usr/bin:/bin"}
    if not inv.exists():
        subprocess.run([PY, "-m", "thermo_extractor.cli", "inspect-pdf", str(DOCS / doc / "vector.pdf"), "--output", str(inv)],
                       check=True, capture_output=True, env=env)
    if not gr.exists():
        subprocess.run([PY, "-m", "thermo_extractor.cli", "propose-pdf-graphics", str(inv), "--output", str(gr)],
                       check=True, capture_output=True, env=env)
    return inv, gr


def vertices(path):
    pts = []
    for c in path.get("commands", []):
        for op in c.get("operands", []):
            if isinstance(op, list) and len(op) == 2 and all(isinstance(v, (int, float)) for v in op):
                pts.append((float(op[0]), float(op[1])))
    return pts


def calibrate(page, region, hx, hy, img_w, img_h):
    tr = page_transform(page, img_w, img_h)
    frame = VM._select_frame(VM._frame_candidates(page, region), region)
    ax_x, rep_x = VM._calibrate_axis(page, frame["bbox"], "x", hx, tr)
    ax_y, rep_y = VM._calibrate_axis(page, frame["bbox"], "y", hy, tr)
    return frame, rep_x["fit"], rep_y["fit"]


def mapper(fit):
    if fit["scale"] == "log10":
        return lambda p: 10 ** (fit["intercept"] + fit["slope"] * p)
    return lambda p: fit["intercept"] + fit["slope"] * p


def roles(graphics, pno):
    out = {}
    for pg in graphics["pages"]:
        if pg.get("page_number") != pno:
            continue
        for o in pg["objects"]:
            if o.get("candidates"):
                out[o.get("object_id") or o.get("id")] = o["candidates"][0]["role"]
    return out


def style_key(p):
    return json.dumps([p.get("stroke_color"), p.get("fill"), p.get("fill_color"), round(p.get("line_width") or 0, 2)])


def figure_objects(page, frame_bb, H, role_of):
    fx0, fy0, fx1, fy1 = frame_bb               # user space (y up)
    curves, markers, lines = [], [], []
    for p in page["paths"]:
        x0, y0, x1, y1 = p["bbox"]
        if x0 < fx0 - 1 or x1 > fx1 + 1 or y0 < fy0 - 1 or y1 > fy1 + 1:
            continue
        role = role_of.get(p["id"], "unknown")
        if role in ("plot_frame",):
            continue
        w, h = x1 - x0, y1 - y0
        if role == "marker":
            markers.append(p)
        elif role in ("data_curve", "axis_or_gridline", "unknown") and max(w, h) > 0.02 * (fx1 - fx0):
            # exclude the frame edges themselves
            if (h < 0.5 and (abs(y0 - fy0) < 0.6 or abs(y0 - fy1) < 0.6)) or (w < 0.5 and (abs(x0 - fx0) < 0.6 or abs(x0 - fx1) < 0.6)):
                continue
            (lines if (h < 0.5 or w < 0.5) else curves).append(p)
    return curves, markers, lines


def to_data(pts_tl, H, mx, my):
    return [[mx(x), my(H - y)] for x, y in pts_tl]


def main():
    for tf in sorted(TRUTH.glob("*.json")):
        t = load_json(tf)
        doc = t["doc_id"]
        inv_p, gr_p = run_inventory(doc)
        inv, gr = load_json(inv_p), load_json(gr_p)
        H = t["renderings"]["vector"]["page_size_pt"][1]
        status = {"xy": [], "phase_diagram": []}
        xy_out, pd_out, tool_panels = [], [], []
        for f in t["figures"]:
            pno = f["page"] + 1
            page = next(p for p in inv["pages"] if p["page_number"] == pno)
            role_of = roles(gr, pno)
            img = DOCS / doc / "clean" / f"page-{pno}.png"
            iw, ih = t["renderings"]["clean_raster"]["files"][f["page"]]["size_px"]
            panels = []
            for ax in f["axes"]:
                if ax["role"] == "inset":
                    continue
                x0, y0, x1, y1 = ax["bbox_pt"]
                region = [x0 - 45, H - (y1 + 32), x1 + 8, H - (y0 - 8)]
                hx = {"quantity": "x", "unit": ax["x"]["unit"]}
                hy = {"quantity": "y", "unit": ax["y"]["unit"], **({"scale": "log10"} if ax["y"]["scale"] == "log" else {})}
                try:
                    frame, fx, fy = calibrate(page, region, hx, hy, iw, ih)
                except Exception as e:  # noqa: BLE001
                    status["phase_diagram" if f["kind"] == "phase_diagram" else "xy"].append(
                        {"figure": f["id"], "axes": ax["id"], "calibration": f"failed: {e}"})
                    continue
                mx, my = mapper(fx), mapper(fy)
                fb = frame["bbox"]
                bbox_tl = [fb[0], H - fb[3], fb[2], H - fb[1]]
                curves, markers, lines = figure_objects(page, fb, H, role_of)
                rec = {"figure": f["id"], "axes": ax["id"], "calibration": "ok",
                       "frame_bbox_pt": bbox_tl, "curves": len(curves), "markers": len(markers), "lines": len(lines)}
                if f["kind"] == "phase_diagram":
                    xs_ = 100.0 if ax["x"]["unit"] == "at.%" else 1.0
                    bnds = [{"polyline": True, "points": [[x / xs_, T] for x, T in to_data(vertices(p), H, mx, my)]}
                            for p in curves + lines]
                    invs = []
                    for p in lines:
                        v = vertices(p)
                        if len(v) >= 2 and abs(v[0][1] - v[-1][1]) < 0.3 and abs(v[0][0] - v[-1][0]) > 0.02 * (fb[2] - fb[0]):
                            T = my(H - v[0][1])
                            invs.append({"T": T, "type": None, "points": [{"x": mx(min(q[0] for q in v)) / xs_},
                                                                          {"x": mx(max(q[0] for q in v)) / xs_}]})
                    fields = []
                    for tx in page["text"]:
                        bx0, by0, bx1, by1 = tx["bbox"]
                        if not (fb[0] < bx0 and bx1 < fb[2] and fb[1] < by0 and by1 < fb[3]):
                            continue
                        s = tx.get("text") or "".join(g["text"] for g in tx.get("glyphs", []))
                        if re.fullmatch(r"[\d.\s°CK]+", s) or not re.search(r"[A-Za-zα-ωΑ-Ω()]", s):
                            continue
                        cxp, cyp = (bx0 + bx1) / 2, (by0 + by1) / 2
                        fields.append({"point": [mx(cxp) / xs_, my(cyp)], "phases": [q.strip() for q in s.split("+") if q.strip()],
                                       "text": s})
                    pd_out.append({"page": f["page"], "bbox": bbox_tl, "x_unit": "mole_fraction",
                                   "T_unit": "°C" if ax["y"]["unit"] == "°C" else "K",
                                   "fields": fields, "boundaries": bnds, "invariants": invs})
                    status["phase_diagram"].append(rec)
                    continue
                series = []
                for i, p in enumerate(curves + lines):
                    series.append({"name": f"path-{p['id']}", "points": to_data(vertices(p), H, mx, my)})
                groups = {}
                for p in markers:
                    b = p["bbox"]
                    groups.setdefault(style_key(p), []).append([mx((b[0] + b[2]) / 2), my((b[1] + b[3]) / 2)])
                for k, pts in groups.items():
                    series.append({"name": f"markers-{len(series)}", "points": pts})
                panels.append({"page": f["page"], "bbox": bbox_tl, "series": series})
                # the full tool route: manifest proposal + vector marker proposals
                hints = {"axes": {"x": {"quantity": "x", "unit": ax["x"]["unit"]}, "y": {"quantity": "y", "unit": ax["y"]["unit"]}},
                         "license": {"status": "CC BY 4.0", "redistribution_allowed": True, "notes": "synthetic"},
                         "reference": {"id": doc, "title": "synthetic", "authors": ["ORCHID synthetic benchmark"], "year": 2026,
                                       "doi": "10.0000/orchid-synthetic-benchmark", "url": "https://orchid-project.org/synthetic-benchmark"},
                         "source": {"figure_label": f"{f['id']}-{ax['id']}", "locator": f"page {pno}"}, "figure_region": region}
                if ax["y"]["scale"] == "log":
                    hints["axes"]["y"]["scale"] = "log10"
                w = WORK / doc / f"{f['id']}-{ax['id']}"
                w.mkdir(parents=True, exist_ok=True)
                (w / "hints.json").write_text(json.dumps(hints), encoding="utf-8")
                try:
                    man, rep = VM.propose_multipanel_manifest(inv_p, img, w / "hints.json", page_number=pno)
                    (w / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
                    rec["manifest"] = f"ok ({len(man['panels'][0]['legend'].get('entries', []))} legend entries)"
                    try:
                        data = propose_vector_markers(w / "manifest.json", DOCS / doc / "vector.pdf", img, inv_p, page_number=pno)
                        rec["vector_markers"] = f"ok ({len(data['points'])} points)"
                        grp = {}
                        for q in data["points"]:
                            grp.setdefault(str(q.get("legend_entry_id") or q.get("series_id") or "all"), []).append(
                                [q["coordinates"]["x"], q["coordinates"]["y"]])
                        tool_panels.append({"page": f["page"], "bbox": bbox_tl,
                                            "series": [{"name": k, "points": v} for k, v in grp.items()]})
                        (w / "markers.json").write_text(json.dumps(data), encoding="utf-8")
                    except Exception as e:  # noqa: BLE001
                        rec["vector_markers"] = f"failed: {str(e)[:160]}"
                except Exception as e:  # noqa: BLE001
                    rec["manifest"] = f"failed: {str(e)[:160]}"
                status["xy"].append(rec)
            if panels:
                xy_out.append({"page": f["page"], "panels": panels})
        for r in ("vector",):
            write_json(OUT / r / f"{doc}.json", {"doc_id": doc, "rendering": r, "extractor": "vector_route",
                                                  "status": status, "xy_figures": xy_out, "phase_diagrams": pd_out})
        write_json(BENCH / "outputs" / "vector_markers_tool" / "vector" / f"{doc}.json",
                   {"doc_id": doc, "rendering": "vector", "extractor": "propose_vector_markers (official route only)",
                    "status": {"xy": [{k: v for k, v in r_.items() if k in ("figure", "axes", "calibration", "manifest", "vector_markers")}
                                      for r_ in status["xy"]]},
                    "xy_figures": [{"panels": tool_panels}] if tool_panels else []})
        print(doc, "xy panels", sum(len(x["panels"]) for x in xy_out), "pd", len(pd_out), flush=True)


if __name__ == "__main__":
    main()
