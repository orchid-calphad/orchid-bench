#!/usr/bin/env python3
"""Score extractor outputs against the truth keys.

    python bench/score.py --outputs bench/outputs/<extractor> [--out bench/results/<extractor>.json]

Output layout: <outputs>/<rendering>/<doc_id>.json with rendering in
{vector, clean_raster, degraded_raster, image_only_pdf}; the format is documented in bench/README.md.
Coordinates in "bbox" fields use the rendering's own units (pt for PDFs, px for rasters, origin top-left)."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TRUTH, load_json, write_json  # noqa: E402

RENDERINGS = ["vector", "clean_raster", "degraded_raster", "image_only_pdf"]
TOL = 0.01          # normalised axis distance counted as a hit
SERIES_TOL = 0.02   # median distance for a predicted series to match a truth series
MIN_FIELD_CELLS = 6
INV_T_TOL = 15.0    # K


# ------------------------------------------------------------------------------------------------ helpers
def norm_text(s):
    s = (s or "").replace("−", "-").replace("–", "-").replace(" ", " ")
    return re.sub(r"\s+", " ", s).strip()


def num_of(s):
    m = re.search(r"[-+]?\d+(?:\.\d+)?", norm_text(s).replace(" ", ""))
    return float(m.group(0)) if m else None


def aabb(corners):
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return [min(xs), min(ys), max(xs), max(ys)]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def densify(poly, step=0.002):
    pts = []
    for (x0, y0), (x1, y1) in zip(poly[:-1], poly[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        for k in range(n):
            pts.append((x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n))
    if poly:
        pts.append(tuple(poly[-1]))
    return np.array(pts, float).reshape(-1, 2)


def nn_dist(a, b):
    """For every row of a, distance to the nearest row of b."""
    if len(a) == 0:
        return np.zeros(0)
    if len(b) == 0:
        return np.full(len(a), np.inf)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(b).query(a)
    return d


def stats(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return {"n": len(v), "mean": float(np.mean(v)) if v else None, "median": float(np.median(v)) if v else None}


def prf(tp_pred, n_pred, tp_true, n_true):
    p = tp_pred / n_pred if n_pred else None
    r = tp_true / n_true if n_true else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if (p is not None and r is not None) else None)
    return {"precision": p, "recall": r, "f1": f, "n_pred": n_pred, "n_true": n_true}


# ------------------------------------------------------------------------------------------------ tables
def score_tables(truth, pred):
    out = {"tables_true": len(truth["tables"]), "tables_pred": len(pred), "per_table": []}
    tot = {"cells": 0, "exact": 0, "exact_norm": 0, "num_cells": 0, "num_exact": 0, "num_abs_err": [], "num_rel_err": []}
    for t in truth["tables"]:
        tc = {(c["row"], c["col"]): c for c in t["cells"] if c["text"]}
        best = None
        for p in pred:
            pc = {(c["row"], c["col"]): c.get("text", "") for c in p.get("cells", [])}
            for dr in range(-4, 5):
                for dc in range(-3, 4):
                    hits = sum(1 for (r_, c_), c in tc.items() if norm_text(pc.get((r_ + dr, c_ + dc))) == norm_text(c["text"]))
                    if best is None or hits > best[0]:
                        best = (hits, pc, dr, dc)
        rec = {"id": t["id"], "cells": len(tc), "exact": 0, "exact_norm": 0, "num_cells": 0, "num_exact": 0}
        nae, nre = [], []
        for (r_, c_), c in tc.items():
            got = best[1].get((r_ + best[2], c_ + best[3])) if best else None
            if got is not None and got.strip() == c["text"].strip():
                rec["exact"] += 1
            if got is not None and norm_text(got) == norm_text(c["text"]):
                rec["exact_norm"] += 1
            if c.get("value") is not None:
                rec["num_cells"] += 1
                v = num_of(got) if got else None
                if v is not None:
                    err = abs(v - c["value"])
                    nae.append(err)
                    nre.append(err / max(abs(c["value"]), 1e-9))
                    if err < 1e-9:
                        rec["num_exact"] += 1
        rec["numeric_abs_err"] = stats(nae)
        out["per_table"].append(rec)
        for k in ("cells", "exact", "exact_norm", "num_cells", "num_exact"):
            tot[k] += rec[k]
        tot["num_abs_err"] += nae
        tot["num_rel_err"] += nre
    out["cell_exact"] = tot["exact"] / tot["cells"] if tot["cells"] else None
    out["cell_exact_normalised"] = tot["exact_norm"] / tot["cells"] if tot["cells"] else None
    out["numeric_exact"] = tot["num_exact"] / tot["num_cells"] if tot["num_cells"] else None
    out["numeric_rel_err"] = stats(tot["num_rel_err"])
    out["_tot"] = {k: v for k, v in tot.items()}
    return out


# ------------------------------------------------------------------------------------------------ xy figures
def _norm(ax, pts):
    a = np.asarray(pts, float).reshape(-1, 2)
    def n1(v, d):
        if d["scale"] == "log":
            return (np.log10(np.clip(v, 1e-300, None)) - math.log10(d["min"])) / (math.log10(d["max"]) - math.log10(d["min"]))
        return (v - d["min"]) / (d["max"] - d["min"])
    return np.stack([n1(a[:, 0], ax["x"]), n1(a[:, 1], ax["y"])], axis=1) if len(a) else a


def match_panel(ax, rendering, pred_panels):
    tb = aabb(ax["corners"][rendering])
    best, bi = None, 0.3
    for p in pred_panels:
        if p.get("page") is not None and p["page"] != ax["page"]:
            continue
        if p.get("axes_id") and p.get("figure_id"):
            if (p["figure_id"], p["axes_id"]) == (ax["_fig"], ax["id"]):
                return p
            continue
        if p.get("bbox"):
            v = iou(tb, p["bbox"])
            if v > bi:
                best, bi = p, v
    return best


def score_xy(truth, rendering, pred_figs):
    panels = [p for f in pred_figs for p in f.get("panels", [])]
    for f in pred_figs:
        for p in f.get("panels", []):
            p.setdefault("page", f.get("page"))
            p.setdefault("figure_id", f.get("figure_id"))
    res = {"panels_true": 0, "panels_found": 0, "point_err": [], "hits": 0, "points": 0, "marker_points": 0, "marker_hits": 0,
           "series_true": 0, "series_true_matched": 0, "series_pred": 0, "series_pred_matched": 0}
    for f in truth["figures"]:
        if not f["kind"].startswith("property"):
            continue
        for ax in f["axes"]:
            ax = {**ax, "_fig": f["id"]}
            res["panels_true"] += 1
            p = match_panel(ax, rendering, panels)
            tser = []
            for s in ax["series"]:
                pts = s.get("markers") or s.get("line") or []
                line = s.get("line") or s.get("markers") or []
                tser.append((_norm(ax, pts), densify([tuple(q) for q in _norm(ax, line)]), bool(s.get("markers"))))
                res["series_true"] += 1
                res["points"] += len(pts)
                res["marker_points"] += len(pts) if s.get("markers") else 0
            if p is None:
                continue
            res["panels_found"] += 1
            pser = [_norm(ax, s["points"]) for s in p.get("series", []) if s.get("points")]
            allp = np.concatenate(pser) if pser else np.zeros((0, 2))
            for tp, _, ism in tser:
                d = nn_dist(tp, allp)
                res["point_err"] += [float(x) for x in d if np.isfinite(x)]
                res["hits"] += int((d <= TOL).sum())
                if ism:
                    res["marker_hits"] += int((d <= TOL).sum())
            matched_t = set()
            for ps in pser:
                res["series_pred"] += 1
                meds = [float(np.median(nn_dist(ps, tl))) for _, tl, _ in tser]
                if meds and min(meds) <= SERIES_TOL:
                    res["series_pred_matched"] += 1
                    matched_t.add(int(np.argmin(meds)))
            res["series_true_matched"] += len(matched_t)
    return res


# ------------------------------------------------------------------------------------------------ phase diagrams
def _canon(lbl):
    s = norm_text(lbl).replace(" ", "").upper()
    s = re.sub(r"#\d+$", "", s)
    return {"LIQUID": "L"}.get(s, s)


def _label_sets(field):
    conv = sorted(_canon(x) for x in field["labels"])
    tdb = sorted(_canon(x.split("#")[0]) for x in field["assemblage"])
    loose = sorted(re.sub(r"\d+$", "", x) for x in conv)
    return conv, tdb, loose


def score_pd(truth, rendering, pred_pds):
    pdt = truth["phase_diagram"]
    fig = next(f for f in truth["figures"] if f["kind"] == "phase_diagram")
    main = next(a for a in fig["axes"] if a["role"] == "main")
    out = {"found": False}
    if not pred_pds:
        return out
    tb = aabb(main["corners"][rendering])
    pred = max(pred_pds, key=lambda p: iou(tb, p["bbox"]) if p.get("bbox") else 0.5)
    out["found"] = True
    Tlo_K = main["y"]["min"] + (273.15 if main["y"]["unit"] == "°C" else 0)
    Thi_K = main["y"]["max"] + (273.15 if main["y"]["unit"] == "°C" else 0)
    TS = Thi_K - Tlo_K
    xu = pred.get("x_unit", "mole_fraction")
    tu = pred.get("T_unit", "K")
    cx = (lambda v: v / 100.0) if xu in ("at.%", "atomic_percent") else (lambda v: v)
    cT = (lambda v: v + 273.15) if tu in ("°C", "degC", "C") else (lambda v: v)
    occ = []
    for o in fig.get("occluders", []):
        sc = fig["units"]["x_scale_factor"]
        off = 273.15 if fig["units"]["T"] == "°C" else 0.0
        occ.append((o["x_plot"][0] / sc, o["x_plot"][1] / sc, o["T_plot"][0] + off, o["T_plot"][1] + off))

    def visible(x, T):
        if not (Tlo_K <= T <= Thi_K and -1e-9 <= x <= 1 + 1e-9):
            return False
        return not any(a <= x <= b and c <= T <= d for a, b, c, d in occ)

    # truth boundary samples (normalised x, T)
    polys = []
    for b in fig["boundaries"]:
        if b["on_frame"]:
            continue
        polys.append([(x, (T - Tlo_K) / TS) for x, T in zip(b["x"], b["T_K"])])
    for il in fig["invariant_lines"]:
        polys.append([(il["x_min"], (il["T_K"] - Tlo_K) / TS), (il["x_max"], (il["T_K"] - Tlo_K) / TS)])
    tpts = np.concatenate([densify(p) for p in polys]) if polys else np.zeros((0, 2))
    keep = np.array([visible(x, y * TS + Tlo_K) for x, y in tpts]) if len(tpts) else np.zeros(0, bool)
    tpts = tpts[keep] if len(tpts) else tpts
    ppts = []
    for b in pred.get("boundaries", []):
        q = [(cx(x), (cT(T) - Tlo_K) / TS) for x, T in b["points"]]
        ppts += [tuple(r_) for r_ in densify(q)] if b.get("polyline") and len(q) > 1 else q
    ppts = np.array(ppts, float).reshape(-1, 2)
    # the frame's vertical edges coincide with zero-solubility boundaries; points there are not false positives
    frame = np.concatenate([np.stack([np.full(501, e), np.linspace(0, 1, 501)], 1) for e in (0.0, 1.0)])
    if len(ppts):
        ppts = ppts[np.array([visible(x, y * TS + Tlo_K) for x, y in ppts])]
    d_pt = nn_dist(ppts, np.concatenate([tpts, frame]) if len(tpts) else frame)
    d_tp = nn_dist(tpts, ppts)
    out["boundary"] = {
        "pred_points": int(len(ppts)), "true_samples": int(len(tpts)),
        "precision@0.01": float((d_pt <= TOL).mean()) if len(d_pt) else None,
        "recall@0.01": float((d_tp <= TOL).mean()) if len(d_tp) else None,
        "mean_pred_to_true": float(np.mean(d_pt)) if len(d_pt) else None,
        "mean_true_to_pred": float(np.mean(d_tp[np.isfinite(d_tp)])) if len(d_tp) and np.isfinite(d_tp).any() else None,
        "hausdorff95": float(max(np.percentile(d_pt, 95), np.percentile(d_tp, 95))) if len(d_pt) and len(d_tp) else None,
    }
    if len(ppts) and len(tpts):
        from scipy.spatial import cKDTree
        _, idx = cKDTree(tpts).query(ppts)
        dx = np.abs(ppts[:, 0] - tpts[idx, 0])
        dT = np.abs(ppts[:, 1] - tpts[idx, 1]) * TS
        out["boundary"]["median_abs_dx"] = float(np.median(dx))
        out["boundary"]["median_abs_dT_K"] = float(np.median(dT))
    # fields
    fm = pdt["field_map"]
    grid = []
    for row in fm["rows"]:
        r_ = []
        for v, n in row:
            r_ += [v] * n
        grid.append(r_)
    grid = np.array(grid)
    fields = {f["id"]: f for f in pdt["fields"]}

    def field_at(x, T):
        j = int(round((x - fm["x0"]) / fm["dx"]))
        i = int(round((T - fm["T0"]) / fm["dT"]))
        if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
            return int(grid[i, j])
        return -1
    scorable = {f["id"] for f in pdt["fields"] if f["n_cells"] >= MIN_FIELD_CELLS and visible(*f["point"])}
    correct_fields, tp, npred, loose_tp = set(), 0, 0, 0
    for pf in pred.get("fields", []):
        if not pf.get("phases"):
            continue
        x, T = cx(pf["point"][0]), cT(pf["point"][1])
        if not visible(x, T):
            continue
        npred += 1
        fid = field_at(x, T)
        if fid < 0:
            continue
        conv, tdb, loose = _label_sets(fields[fid])
        p = sorted(_canon(q) for q in pf["phases"])
        if p == conv or p == tdb:
            tp += 1
            correct_fields.add(fid)
        if p == conv or p == tdb or sorted(re.sub(r"\d+$", "", q) for q in p) == loose:
            loose_tp += 1
    out["fields"] = {**prf(tp, npred, len(correct_fields & scorable), len(scorable)), "loose_precision": loose_tp / npred if npred else None}
    # invariants
    tinv = [i for i in pdt["invariants"] if visible(i["points"][1]["x"], i["T"])]
    pinv = [{"T": cT(i["T"]), "type": i.get("type"), "xs": sorted(cx(p["x"]) for p in i.get("points", []) if p.get("x") is not None)}
            for i in pred.get("invariants", []) if i.get("T") is not None and Tlo_K <= cT(i["T"]) <= Thi_K]
    pinv = [p for p in pinv if len(p["xs"]) != 3 or visible(p["xs"][1], p["T"])]
    used, terr, xerr, typ_ok, matched = set(), [], [], 0, 0
    for ti in sorted(tinv, key=lambda q: q["T"]):
        cands = [(abs(p["T"] - ti["T"]), k) for k, p in enumerate(pinv) if k not in used and abs(p["T"] - ti["T"]) <= INV_T_TOL]
        if not cands:
            continue
        _, k = min(cands)
        used.add(k)
        matched += 1
        p = pinv[k]
        terr.append(abs(p["T"] - ti["T"]))
        txs = sorted(q["x"] for q in ti["points"])
        if len(p["xs"]) == 3:
            xerr += [abs(a - b) for a, b in zip(p["xs"], txs)]
        if p["type"] and p["type"] == ti["type"]:
            typ_ok += 1
    out["invariants"] = {**prf(matched, len(pinv), matched, len(tinv)), "T_abs_err_K": stats(terr),
                         "x_abs_err": stats(xerr), "type_accuracy": typ_ok / matched if matched else None}
    return out


# ------------------------------------------------------------------------------------------------ text values
UNIT_ALIASES = {"k": "K", "°c": "°C", "c": "°C", "degc": "°C", "at.%": "at.%", "at%": "at.%", "mole_fraction": "mole_fraction",
                "x": "mole_fraction", "kj/mol": "kJ/mol", "j/mol": "J/mol", "atm": "atm"}


def score_text(truth, pred_vals):
    tv = [(v["value"], v["unit"]) for v in truth["text"]["values"]]
    for f in truth["figures"]:
        for t in f.get("text_in_figure", []):
            tv.append((t["value"], t["unit"]))
        for ax in f.get("axes", []):           # legend entries such as "T = 1450 K" (generated strings)
            for s_ in ax.get("series", []):
                for m in re.finditer(r"(\d+(?:\.\d+)?) K\b", s_["name"]):
                    tv.append((float(m.group(1)), "K"))
    for tb in truth["tables"]:                  # table headers such as "ΔfH(298 K)"
        for c in tb["cells"]:
            if c["header"]:
                for m in re.finditer(r"(\d+(?:\.\d+)?) K\b", c["text"]):
                    tv.append((float(m.group(1)), "K"))
    pv = [(float(v["value"]), UNIT_ALIASES.get(str(v.get("unit", "")).lower(), v.get("unit"))) for v in pred_vals
          if v.get("value") is not None]
    def mm(a, b, with_unit=True):
        rem = list(b)
        hit = 0
        for v, u in a:
            for k, (w, x) in enumerate(rem):
                if abs(v - w) < 1e-6 and (not with_unit or u == x):
                    hit += 1
                    rem.pop(k)
                    break
        return hit
    h = mm(tv, pv)
    hv = mm(tv, pv, with_unit=False)
    return {**prf(h, len(pv), h, len(tv)), "value_only_recall": hv / len(tv) if tv else None}


# ------------------------------------------------------------------------------------------------ driver
def score_doc(truth, rendering, pred):
    st = pred.get("status", {})
    out = {"status": st}
    if "tables" in pred:
        out["tables"] = score_tables(truth, pred["tables"])
    if "xy_figures" in pred:
        out["xy"] = score_xy(truth, rendering, pred["xy_figures"])
    if "phase_diagrams" in pred:
        out["phase_diagram"] = score_pd(truth, rendering, pred["phase_diagrams"])
    if "text_values" in pred:
        out["text"] = score_text(truth, pred["text_values"])
    return out


def aggregate(per_doc):
    agg = {}
    for rendering in RENDERINGS:
        docs = [d[rendering] for d in per_doc.values() if rendering in d]
        if not docs:
            continue
        a = {"documents": len(docs)}
        tabs = [d["tables"] for d in docs if "tables" in d]
        if tabs:
            tot = {k: sum(t["_tot"][k] for t in tabs) for k in ("cells", "exact", "exact_norm", "num_cells", "num_exact")}
            rel = [x for t in tabs for x in t["_tot"]["num_rel_err"]]
            a["tables"] = {"tables_true": sum(t["tables_true"] for t in tabs), "tables_pred": sum(t["tables_pred"] for t in tabs),
                           "cells": tot["cells"], "cell_exact": tot["exact"] / tot["cells"] if tot["cells"] else None,
                           "cell_exact_normalised": tot["exact_norm"] / tot["cells"] if tot["cells"] else None,
                           "numeric_exact": tot["num_exact"] / tot["num_cells"] if tot["num_cells"] else None,
                           "numeric_rel_err_median": float(np.median(rel)) if rel else None}
        xys = [d["xy"] for d in docs if "xy" in d]
        if xys:
            pe = [x for d in xys for x in d["point_err"]]
            S = lambda k: sum(d[k] for d in xys)  # noqa: E731
            a["xy"] = {"panels_true": S("panels_true"), "panels_found": S("panels_found"), "points": S("points"),
                       "point_recall@0.01": S("hits") / S("points") if S("points") else None,
                       "marker_points": S("marker_points"),
                       "marker_recall@0.01": S("marker_hits") / S("marker_points") if S("marker_points") else None,
                       "point_err_median": float(np.median(pe)) if pe else None,
                       "point_err_mean": float(np.mean(pe)) if pe else None,
                       "series_recall": S("series_true_matched") / S("series_true") if S("series_true") else None,
                       "series_precision": S("series_pred_matched") / S("series_pred") if S("series_pred") else None,
                       "series_pred": S("series_pred")}
        pds = [d["phase_diagram"] for d in docs if "phase_diagram" in d]
        if pds:
            found = [p for p in pds if p.get("found")]
            def mean(path):
                vals = []
                for p in found:
                    v = p
                    for k in path:
                        v = v.get(k) if isinstance(v, dict) else None
                    if v is not None:
                        vals.append(v)
                return float(np.mean(vals)) if vals else None
            ft = {k: sum((p.get("fields") or {}).get(k) or 0 for p in found) for k in ("n_pred", "n_true")}
            fp = sum(round(((p.get("fields") or {}).get("precision") or 0) * ((p.get("fields") or {}).get("n_pred") or 0)) for p in found)
            fr = sum(round(((p.get("fields") or {}).get("recall") or 0) * ((p.get("fields") or {}).get("n_true") or 0)) for p in found)
            it = {k: sum((p.get("invariants") or {}).get(k) or 0 for p in found) for k in ("n_pred", "n_true")}
            im = sum(round(((p.get("invariants") or {}).get("recall") or 0) * ((p.get("invariants") or {}).get("n_true") or 0)) for p in found)
            a["phase_diagram"] = {
                "diagrams": len(pds), "diagrams_with_output": len(found),
                "boundary_precision@0.01": mean(["boundary", "precision@0.01"]),
                "boundary_recall@0.01": mean(["boundary", "recall@0.01"]),
                "boundary_mean_dist": mean(["boundary", "mean_pred_to_true"]),
                "boundary_hausdorff95": mean(["boundary", "hausdorff95"]),
                "boundary_median_dx": mean(["boundary", "median_abs_dx"]),
                "boundary_median_dT_K": mean(["boundary", "median_abs_dT_K"]),
                "field_precision": fp / ft["n_pred"] if ft["n_pred"] else None,
                "field_recall": fr / ft["n_true"] if ft["n_true"] else None,
                "field_pred": ft["n_pred"], "field_true_scorable": ft["n_true"],
                "invariant_recall": im / it["n_true"] if it["n_true"] else None,
                "invariant_precision": im / it["n_pred"] if it["n_pred"] else None,
                "invariant_T_err_K": mean(["invariants", "T_abs_err_K", "mean"]),
                "invariant_x_err": mean(["invariants", "x_abs_err", "mean"]),
                "invariant_type_accuracy": mean(["invariants", "type_accuracy"]),
            }
        txs = [d["text"] for d in docs if "text" in d]
        if txs:
            npred = sum(t["n_pred"] for t in txs)
            ntrue = sum(t["n_true"] for t in txs)
            tp = sum(round((t["recall"] or 0) * t["n_true"]) for t in txs)
            a["text"] = {"values_true": ntrue, "values_pred": npred, "precision": tp / npred if npred else None,
                         "recall": tp / ntrue if ntrue else None,
                         "value_only_recall": float(np.mean([t["value_only_recall"] for t in txs if t["value_only_recall"] is not None]))}
        agg[rendering] = a
    return agg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    root = Path(a.outputs)
    per_doc = {}
    for rendering in RENDERINGS:
        for f in sorted((root / rendering).glob("*.json")) if (root / rendering).exists() else []:
            pred = load_json(f)
            truth = load_json(TRUTH / f"{pred['doc_id']}.json")
            per_doc.setdefault(pred["doc_id"], {})[rendering] = score_doc(truth, rendering, pred)
    for d in per_doc.values():
        for r_ in d.values():
            for k in ("tables",):
                if k in r_:
                    r_[k].pop("_tot_keep", None)
    agg = aggregate(per_doc)
    for d in per_doc.values():
        for r_ in d.values():
            if "tables" in r_:
                r_["tables"].pop("_tot", None)
            if "xy" in r_:
                r_["xy"]["point_err"] = stats(r_["xy"]["point_err"])
    res = {"extractor": root.name, "aggregate": agg, "per_document": per_doc}
    out = Path(a.out) if a.out else root / "scores.json"
    write_json(out, res, pretty=True)
    print(json.dumps(agg, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
