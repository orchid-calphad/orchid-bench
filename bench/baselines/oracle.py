#!/usr/bin/env python3
"""Self-test: writes the truth itself in extractor-output format; scoring it must give perfect scores."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import bench, truth_dir, load_json, write_json  # noqa: E402
from score import renderings, aabb  # noqa: E402

OUT = bench / "outputs" / "oracle"
for tf in sorted(truth_dir.glob("*.json")):
    t = load_json(tf)
    for r in renderings:
        xy = []
        for f in t["figures"]:
            if not f["kind"].startswith("property"):
                continue
            xy.append({"page": f["page"], "panels": [{"page": f["page"], "bbox": aabb(a["corners"][r]),
                                                       "series": [{"name": s["name"], "points": s.get("markers") or s.get("line")}
                                                                  for s in a["series"]]} for a in f["axes"]]})
        pdf_ = next(f for f in t["figures"] if f["kind"] == "phase_diagram")
        pd = t["phase_diagram"]
        pdo = {"page": pdf_["page"], "bbox": aabb(pdf_["axes"][0]["corners"][r]), "x_unit": "mole_fraction", "T_unit": "K",
               "fields": [{"point": f["point"], "phases": f["labels"]} for f in pd["fields"]],
               "boundaries": [{"polyline": True, "points": [[x, T] for x, T in zip(b["x"], b["T_K"])]} for b in pdf_["boundaries"]]
               + [{"polyline": True, "points": [[il["x_min"], il["T_K"]], [il["x_max"], il["T_K"]]]} for il in pdf_["invariant_lines"]],
               "invariants": [{"T": i["T"], "type": i["type"], "points": i["points"]} for i in pd["invariants"]]}
        sem = t.get("semantic") or {}
        tv = [{"value": v["value"], "unit": v["unit"]} for v in t["text"]["values"]]
        tv += [{"value": x["value"], "unit": x["unit"]} for f in t["figures"] for x in f.get("text_in_figure", [])]
        import re
        for f in t["figures"]:
            for a in f["axes"]:
                for s in a.get("series", []):
                    tv += [{"value": float(m.group(1)), "unit": "K"} for m in re.finditer(r"(\d+(?:\.\d+)?) K\b", s["name"])]
        for tb in t["tables"]:
            for c in tb["cells"]:
                if c["header"]:
                    tv += [{"value": float(m.group(1)), "unit": "K"} for m in re.finditer(r"(\d+(?:\.\d+)?) K\b", c["text"])]
        write_json(OUT / r / f"{t['doc_id']}.json", {
            "doc_id": t["doc_id"], "rendering": r, "extractor": "oracle",
            "tables": [{"page": sg["page"], "cells": [{"row": c.get("seg_row", c["row"]), "col": c["col"], "text": c["text"]}
                                                      for c in tb["cells"] if c.get("segment", 0) == si]}
                       for tb in t["tables"] for si, sg in enumerate(tb.get("segments") or [{"page": tb["page"]}])],
            "xy_figures": xy, "phase_diagrams": [pdo], "text_values": tv,
            "invariant_rows": [{"reaction": r["reaction"], "type": r["type"], "T": r["T_value"], "T_unit": r["T_unit"],
                                "x_unit": r["x_unit"], "compositions": [{"phase": p["label"], "x": p["value"]} for p in r["phases"]],
                                "evidence": [{"class": "experimental", "T": e["value"], "citation": e["citation"]}
                                             for e in r["evidence"] if e["class"] == "experimental_cited"]}
                               for r in sem.get("invariant_rows", [])],
            "phase_rows": [{"phase": r["phase"], "pearson": r["pearson"], "strukturbericht": r["strukturbericht_printed"],
                            "prototype": r["prototype"], "T_low": None if r["T_low_K"] is None else float(r["T_low_printed"]),
                            "T_high": float(r["T_high_printed"].rstrip("ᵃᵇᶜ")) if r["T_high_K"] is not None else None,
                            "T_unit": r["T_unit"],
                            "homogeneity": {"x_min": float(r["homogeneity"]["printed"].split("–")[0]),
                                            "x_max": float(r["homogeneity"]["printed"].split("–")[-1]),
                                            "x_unit": r["homogeneity"]["x_unit"], "T": float(r["homogeneity"]["T_printed"]),
                                            "T_unit": r["T_unit"]}} for r in sem.get("phase_rows", [])],
            "property_records": [{"table": r["table"], "phase": r.get("phase"),
                                  "x": (float(r["x_printed"]) if r["table"] == "mixing" else None),
                                  "x_unit": r.get("x_unit"), "quantity": r["quantity"], "value": r["value"], "unit": r["unit"],
                                  "basis": r["basis"]} for r in sem.get("property_records", [])]})
