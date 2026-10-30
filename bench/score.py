#!/usr/bin/env python3
"""Score extractor outputs against the truth keys.

    python bench/score.py --outputs bench/outputs/<extractor> [--out bench/results/<extractor>.json]

Output layout: <outputs>/<rendering>/<doc_id>.json with rendering in
{vector, clean_raster, degraded_raster, image_only_pdf}; the format is documented in bench/README.md.
Coordinates in "bbox" fields use the rendering's own units (pt for PDFs, px for rasters, origin top-left).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import truth_dir, load_json, write_json  # noqa: E402

renderings = ["vector", "clean_raster", "degraded_raster", "image_only_pdf"]
hit_tol = 0.01  # normalised axis distance counted as a hit
series_tol = 0.02  # median distance for a predicted series to match a truth series
min_field_cells = 6
inv_t_tol = 15.0  # K


# ------------------------------------------------------------------------------------------------ helpers
def norm_text(s):
    """Normalise a text string for comparison: case and whitespace folded."""
    s = (s or "").replace("−", "-").replace("–", "-").replace(" ", " ")
    return re.sub(r"\s+", " ", s).strip()


def num_of(s):
    """Parse a number from a string, or return None when there is none."""
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", norm_text(s).replace(" ", ""))
    return float(m.group(0)) if m else None


def aabb(corners):
    """Return the axis-aligned bounding box (x0, y0, x1, y1) of a set of corners."""
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return [min(xs), min(ys), max(xs), max(ys)]


def iou(a, b):
    """Return the intersection over union of two axis-aligned boxes."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def densify(poly, step=0.002):
    """Resample the edges of a polygon so that consecutive points are at most step apart."""
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
    """Return summary statistics (median and count) of a list of values."""
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return {"n": len(v), "mean": float(np.mean(v)) if v else None, "median": float(np.median(v)) if v else None}


def prf(tp_pred, n_pred, tp_true, n_true):
    """Return precision, recall and F1 from matched and total counts."""
    p = tp_pred / n_pred if n_pred else None
    r = tp_true / n_true if n_true else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if (p is not None and r is not None) else None)
    return {"precision": p, "recall": r, "f1": f, "n_pred": n_pred, "n_true": n_true}


# ------------------------------------------------------------------------------------------------ tables
def _segments(t):
    """Truth table -> list of segments {(seg_row, col): cell}; v1 truth keys have one segment keyed by row."""
    segs = {}
    for c in t["cells"]:
        if not c["text"]:
            continue
        si = c.get("segment", 0)
        segs.setdefault(si, {})[(c.get("seg_row", c["row"]), c["col"])] = c
    return [segs[k] for k in sorted(segs)]


def align_segment(tc, pred):
    """Best predicted table and (row, col) offset for one truth segment: returns (hits, pred_cells, dr, dc, index)."""
    best = None
    for pi, p in enumerate(pred):
        pc = {(c["row"], c["col"]): c.get("text", "") for c in p.get("cells", [])}
        for dr in range(-4, 5):
            for dc in range(-3, 4):
                hits = sum(1 for (r_, c_), c in tc.items() if norm_text(pc.get((r_ + dr, c_ + dc))) == norm_text(c["text"]))
                if best is None or hits > best[0]:
                    best = (hits, pc, dr, dc, pi)
    return best


def score_tables(truth, pred):
    """Score predicted tables against the truth tables of a document."""
    out = {"tables_true": len(truth["tables"]), "tables_pred": len(pred), "per_table": []}
    tot = {"cells": 0, "exact": 0, "exact_norm": 0, "num_cells": 0, "num_exact": 0, "num_abs_err": [], "num_rel_err": []}
    by_id = {}
    for t in truth["tables"]:
        rec = {"id": t["id"], "segments": 0, "cells": 0, "exact": 0, "exact_norm": 0, "num_cells": 0, "num_exact": 0}
        nae, nre = [], []
        for tc in _segments(t):
            rec["segments"] += 1
            rec["cells"] += len(tc)
            best = align_segment(tc, pred)
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
        bi = by_id.setdefault(
            t["id"], {"cells": 0, "exact": 0, "exact_norm": 0, "num_cells": 0, "num_exact": 0, "num_rel_err": []}
        )
        for k in ("cells", "exact", "exact_norm", "num_cells", "num_exact"):
            tot[k] += rec[k]
            bi[k] += rec[k]
        bi["num_rel_err"] += nre
        tot["num_abs_err"] += nae
        tot["num_rel_err"] += nre
    out["cell_exact"] = tot["exact"] / tot["cells"] if tot["cells"] else None
    out["cell_exact_normalised"] = tot["exact_norm"] / tot["cells"] if tot["cells"] else None
    out["numeric_exact"] = tot["num_exact"] / tot["num_cells"] if tot["num_cells"] else None
    out["numeric_rel_err"] = stats(tot["num_rel_err"])
    out["_tot"] = {k: v for k, v in tot.items()}
    out["_by_id"] = by_id
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
    """Find the predicted panel that best matches a truth panel on one rendering."""
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
    """Score predicted x-y figure data against the truth figures."""
    panels = [p for f in pred_figs for p in f.get("panels", [])]
    for f in pred_figs:
        for p in f.get("panels", []):
            p.setdefault("page", f.get("page"))
            p.setdefault("figure_id", f.get("figure_id"))
    res = {
        "panels_true": 0,
        "panels_found": 0,
        "point_err": [],
        "hits": 0,
        "points": 0,
        "marker_points": 0,
        "marker_hits": 0,
        "series_true": 0,
        "series_true_matched": 0,
        "series_pred": 0,
        "series_pred_matched": 0,
        "by_quantity": {},
    }
    for f in truth["figures"]:
        if not f["kind"].startswith("property"):
            continue
        for ax in f["axes"]:
            ax = {**ax, "_fig": f["id"]}
            res["panels_true"] += 1
            p = match_panel(ax, rendering, panels)
            tser = []
            q = ax.get("quantity", "?")
            bq = res["by_quantity"].setdefault(q, {"panels": 0, "found": 0, "points": 0, "hits": 0, "err": []})
            bq["panels"] += 1
            for s in ax["series"]:
                pts = s.get("markers") or s.get("line") or []
                line = s.get("line") or s.get("markers") or []
                nl = _norm(ax, line)
                if s.get("segments") and not s.get("markers"):
                    dl = (
                        np.concatenate([densify([tuple(q_) for q_ in nl[a_:b_]]) for a_, b_ in s["segments"]])
                        if s["segments"]
                        else np.zeros((0, 2))
                    )
                else:
                    dl = densify([tuple(q_) for q_ in nl])
                tser.append((_norm(ax, pts), dl, bool(s.get("markers"))))
                bq["points"] += len(pts)
                res["series_true"] += 1
                res["points"] += len(pts)
                res["marker_points"] += len(pts) if s.get("markers") else 0
            if p is None:
                continue
            res["panels_found"] += 1
            bq["found"] += 1
            pser = [_norm(ax, s["points"]) for s in p.get("series", []) if s.get("points")]
            allp = np.concatenate(pser) if pser else np.zeros((0, 2))
            for tp, _, ism in tser:
                d = nn_dist(tp, allp)
                res["point_err"] += [float(x) for x in d if np.isfinite(x)]
                res["hits"] += int((d <= hit_tol).sum())
                bq["hits"] += int((d <= hit_tol).sum())
                bq["err"] += [float(x) for x in d if np.isfinite(x)]
                if ism:
                    res["marker_hits"] += int((d <= hit_tol).sum())
            matched_t = set()
            for ps in pser:
                res["series_pred"] += 1
                meds = [float(np.median(nn_dist(ps, tl))) for _, tl, _ in tser]
                if meds and min(meds) <= series_tol:
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
    """Score predicted phase diagrams against the truth boundaries."""
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
        "pred_points": int(len(ppts)),
        "true_samples": int(len(tpts)),
        "precision@0.01": float((d_pt <= hit_tol).mean()) if len(d_pt) else None,
        "recall@0.01": float((d_tp <= hit_tol).mean()) if len(d_tp) else None,
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

    scorable = {f["id"] for f in pdt["fields"] if f["n_cells"] >= min_field_cells and visible(*f["point"])}
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
    out["fields"] = {
        **prf(tp, npred, len(correct_fields & scorable), len(scorable)),
        "loose_precision": loose_tp / npred if npred else None,
    }
    # invariants
    tinv = [i for i in pdt["invariants"] if visible(i["points"][1]["x"], i["T"])]
    pinv = [
        {"T": cT(i["T"]), "type": i.get("type"), "xs": sorted(cx(p["x"]) for p in i.get("points", []) if p.get("x") is not None)}
        for i in pred.get("invariants", [])
        if i.get("T") is not None and Tlo_K <= cT(i["T"]) <= Thi_K
    ]
    pinv = [p for p in pinv if len(p["xs"]) != 3 or visible(p["xs"][1], p["T"])]
    used, terr, xerr, typ_ok, matched = set(), [], [], 0, 0
    for ti in sorted(tinv, key=lambda q: q["T"]):
        cands = [(abs(p["T"] - ti["T"]), k) for k, p in enumerate(pinv) if k not in used and abs(p["T"] - ti["T"]) <= inv_t_tol]
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
    out["invariants"] = {
        **prf(matched, len(pinv), matched, len(tinv)),
        "T_abs_err_K": stats(terr),
        "x_abs_err": stats(xerr),
        "type_accuracy": typ_ok / matched if matched else None,
    }
    return out


# ------------------------------------------------------------------------------------------------ text values
unit_aliases = {
    "k": "K",
    "°c": "°C",
    "c": "°C",
    "degc": "°C",
    "at.%": "at.%",
    "at%": "at.%",
    "mole_fraction": "mole_fraction",
    "x": "mole_fraction",
    "kj/mol": "kJ/mol",
    "j/mol": "J/mol",
    "atm": "atm",
    "cal/mol": "cal/mol",
    "j/(mol k)": "J/(mol K)",
    "j/mol/k": "J/(mol K)",
    "j": "J",
}


def score_text(truth, pred_vals):
    """Score predicted text values against the truth values."""
    tv = [(v["value"], v["unit"]) for v in truth["text"]["values"]]
    for f in truth["figures"]:
        for t in f.get("text_in_figure", []):
            tv.append((t["value"], t["unit"]))
        for ax in f.get("axes", []):  # legend entries such as "T = 1450 K" (generated strings)
            for s_ in ax.get("series", []):
                for m in re.finditer(r"(\d+(?:\.\d+)?) K\b", s_["name"]):
                    tv.append((float(m.group(1)), "K"))
    for tb in truth["tables"]:  # table headers such as "ΔfH(298 K)"
        for c in tb["cells"]:
            if c["header"]:
                for m in re.finditer(r"(\d+(?:\.\d+)?) K\b", c["text"]):
                    tv.append((float(m.group(1)), "K"))
    pv = [
        (float(v["value"]), unit_aliases.get(str(v.get("unit", "")).lower(), v.get("unit")))
        for v in pred_vals
        if v.get("value") is not None
    ]

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


# ------------------------------------------------------------------------------------------------ semantic tables (v2)
inv_t_row_tol = 15.0
e_factor = {
    "J/mol": 1.0,
    "kJ/mol": 1000.0,
    "cal/mol": 4.184,
    "kcal/mol": 4184.0,
    "J/(mol K)": 1.0,
    "cal/(mol K)": 4.184,
    "1": 1.0,
}
unit_norm = {
    "j/mol": "J/mol",
    "kj/mol": "kJ/mol",
    "cal/mol": "cal/mol",
    "kcal/mol": "kcal/mol",
    "j/(mol k)": "J/(mol K)",
    "j/mol/k": "J/(mol K)",
    "cal/(mol k)": "cal/(mol K)",
    "1": "1",
    "": "1",
    "none": "1",
}
basis_norm = {
    "mol_atoms": "mol_atoms",
    "atom": "mol_atoms",
    "mol-atom": "mol_atoms",
    "per mole of atoms": "mol_atoms",
    "formula_unit": "formula_unit",
    "f.u.": "formula_unit",
    "per formula unit": "formula_unit",
}


def _loose(lbl):
    c = _canon(lbl)
    c = re.sub(r"^L\d$", "L", c)
    return re.sub(r"\)\d$", ")", c)


def _toK(v, unit):
    return None if v is None else (float(v) + 273.15 if unit in ("°C", "degC", "C") else float(v))


def _tox(v, unit):
    return None if v is None else (float(v) / 100.0 if unit in ("at.%", "atomic_percent") else float(v))


def _close(a, b, tol=1e-6):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def score_invariant_rows(truth, pred):
    """Score predicted invariant-reaction rows against the truth rows."""
    rows = (truth.get("semantic") or {}).get("invariant_rows", [])
    P = []
    for r in pred:
        TK = _toK(r.get("T"), r.get("T_unit", "K"))
        if TK is None:
            continue
        comps = r.get("compositions") or []
        P.append(
            {
                **r,
                "_TK": TK,
                "_set": sorted(_loose(c.get("phase", "")) for c in comps) or sorted(_loose(x) for x in r.get("phases", [])),
            }
        )
    used, res = (
        set(),
        {
            "rows_true": len(rows),
            "rows_pred": len(P),
            "matched": 0,
            "T_err": [],
            "T_printed_exact": 0,
            "x_err": [],
            "x_printed_exact": 0,
            "x_n": 0,
            "type_ok": 0,
            "ev_n": 0,
            "ev_ok": 0,
            "cite_n": 0,
            "cite_ok": 0,
        },
    )
    for t in sorted(rows, key=lambda q: q["T_K"]):
        tset = sorted(_loose(p["label"]) for p in t["phases"])
        tset2 = sorted(_loose(p["tdb_phase"] or "") for p in t["phases"])
        cands = [
            (abs(p["_TK"] - t["T_K"]), k)
            for k, p in enumerate(P)
            if k not in used and p["_set"] in (tset, tset2) and abs(p["_TK"] - t["T_K"]) <= inv_t_row_tol
        ]
        if not cands:
            continue
        _, k = min(cands)
        used.add(k)
        p = P[k]
        res["matched"] += 1
        res["T_err"].append(abs(p["_TK"] - _toK(t["T_value"], t["T_unit"])))
        if p.get("T_unit", "K") == t["T_unit"] and _close(p.get("T"), t["T_value"]):
            res["T_printed_exact"] += 1
        pc = {}
        for c in p.get("compositions") or []:
            pc.setdefault(_loose(c.get("phase", "")), []).append(c)
        for tp in t["phases"]:
            res["x_n"] += 1
            lst = pc.get(_loose(tp["label"])) or pc.get(_loose(tp["tdb_phase"] or "")) or []
            if not lst:
                continue
            xu = p.get("x_unit", "mole_fraction")
            c = min(lst, key=lambda c_: abs(_tox(c_.get("x"), xu) - tp["x"]) if c_.get("x") is not None else 9)
            if c.get("x") is None:
                continue
            res["x_err"].append(abs(_tox(c["x"], xu) - _tox(tp["value"], tp["x_unit"])))
            if xu == tp["x_unit"] and _close(c["x"], tp["value"]):
                res["x_printed_exact"] += 1
        if (p.get("type") or "").strip().lower() == t["type"]:
            res["type_ok"] += 1
        if t.get("has_exp_column"):
            res["ev_n"] += 1
            te = [e for e in t["evidence"] if e["class"] == "experimental_cited"]
            pe = [e for e in p.get("evidence") or [] if str(e.get("class", "")).startswith("exp")]
            if not te and not pe:
                res["ev_ok"] += 1
            elif te and pe:
                e0, q0 = te[0], pe[0]
                res["cite_n"] += 1
                cite_ok = str(q0.get("citation")).strip("[] ") == str(e0["citation"])
                res["cite_ok"] += int(cite_ok)
                if cite_ok and _close(q0.get("T"), e0["value"]):
                    res["ev_ok"] += 1
            elif te:
                res["cite_n"] += 1
    return res


def _norm_struct(v):
    if v is None:
        return ""
    s = str(v).translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")).replace("_", "").replace(" ", "").replace("-", "")
    s = s.replace("–", "").lower()
    return s


def score_phase_rows(truth, pred):
    """Score predicted phase rows against the truth phase rows."""
    rows = (truth.get("semantic") or {}).get("phase_rows", [])
    res = {
        "rows_true": len(rows),
        "rows_pred": len(pred),
        "matched": 0,
        "struct_n": 0,
        "struct_ok": 0,
        "pearson_ok": 0,
        "sb_ok": 0,
        "proto_ok": 0,
        "T_low_err": [],
        "T_high_err": [],
        "hom_err": [],
        "hom_T_err": [],
    }
    used = set()
    for t in rows:
        keys = {_loose(t["phase"])} | {_loose(q) for q in t["tdb_phases"]}
        k = next((i for i, p in enumerate(pred) if i not in used and _loose(p.get("phase", "")) in keys), None)
        if k is None:
            continue
        used.add(k)
        p = pred[k]
        res["matched"] += 1
        for f_, key in (("pearson", "pearson_ok"), ("strukturbericht", "sb_ok"), ("prototype", "proto_ok")):
            ok = _norm_struct(p.get(f_)) == _norm_struct(t[f_])
            res[key] += int(ok)
            res["struct_ok"] += int(ok)
            res["struct_n"] += 1
        tu = p.get("T_unit", "K")
        if t["T_low_K"] is not None and p.get("T_low") is not None:
            res["T_low_err"].append(abs(_toK(p["T_low"], tu) - _toK(t["T_low_value"], t["T_unit"])))
        if t["T_high_K"] is not None and p.get("T_high") is not None:
            res["T_high_err"].append(abs(_toK(p["T_high"], tu) - _toK(t["T_high_value"], t["T_unit"])))
        h, ph = t["homogeneity"], p.get("homogeneity") or {}
        xu = ph.get("x_unit", "mole_fraction")
        for a_ in ("x_min", "x_max"):
            if ph.get(a_) is not None:
                res["hom_err"].append(abs(_tox(ph[a_], xu) - _tox(h[a_ + "_value"], h["x_unit"])))
        if ph.get("T") is not None:
            res["hom_T_err"].append(abs(_toK(ph["T"], ph.get("T_unit", tu)) - _toK(h["T_value"], t["T_unit"])))
    return res


def score_property_records(truth, pred):
    """Score predicted property records against the truth records."""
    recs = (truth.get("semantic") or {}).get("property_records", [])
    res = {
        "records_true": len(recs),
        "records_pred": len(pred),
        "matched": 0,
        "value_exact": 0,
        "rel_err": [],
        "within_1pct": 0,
        "unit_ok": 0,
        "basis_ok": 0,
    }

    def key_t(r):
        return (r["table"], _loose(r["phase"]) if r["table"] == "formation" else round(r["x"], 4), r["quantity"])

    def key_p(r):
        if r.get("table") == "formation":
            return ("formation", _loose(r.get("phase", "")), r.get("quantity"))
        x = _tox(r.get("x"), r.get("x_unit", "mole_fraction"))
        return (r.get("table"), round(x, 4) if x is not None else None, r.get("quantity"))

    idx = {}
    for i, r in enumerate(pred):
        idx.setdefault(key_p(r), []).append(i)
    for t in recs:
        lst = idx.get(key_t(t))
        if not lst:
            continue
        p = pred[lst.pop(0)]
        res["matched"] += 1
        if _close(p.get("value"), t["value"], 1e-9):
            res["value_exact"] += 1
        u = unit_norm.get(str(p.get("unit", "")).lower(), p.get("unit"))
        bas = basis_norm.get(str(p.get("basis", "mol_atoms")).lower(), p.get("basis"))
        res["unit_ok"] += int(u == t["unit"])
        res["basis_ok"] += int(bas == t["basis"])
        if p.get("value") is None or u not in e_factor:
            continue
        v = float(p["value"]) * e_factor[u]
        if bas == "formula_unit":
            v /= t.get("atoms_per_formula_unit") or 1
        e = (
            t["value"]
            * e_factor.get(t["unit"], 1.0)
            / ((t.get("atoms_per_formula_unit") or 1) if t["basis"] == "formula_unit" else 1)
        )
        re_ = abs(v - e) / max(abs(e), 1e-12)
        res["rel_err"].append(re_)
        res["within_1pct"] += int(re_ <= 0.01)
    return res


# ------------------------------------------------------------------------------------------------ driver
def score_doc(truth, rendering, pred):
    """Score the predictions of one document for one rendering."""
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
    if "invariant_rows" in pred:
        out["invariant_rows"] = score_invariant_rows(truth, pred["invariant_rows"])
    if "phase_rows" in pred:
        out["phase_rows"] = score_phase_rows(truth, pred["phase_rows"])
    if "property_records" in pred:
        out["property_records"] = score_property_records(truth, pred["property_records"])
    return out


def aggregate(per_doc):
    """Aggregate per-document scores into one summary per rendering."""
    agg = {}
    for rendering in renderings:
        docs = [d[rendering] for d in per_doc.values() if rendering in d]
        if not docs:
            continue
        a = {"documents": len(docs)}
        tabs = [d["tables"] for d in docs if "tables" in d]
        if tabs:
            tot = {k: sum(t["_tot"][k] for t in tabs) for k in ("cells", "exact", "exact_norm", "num_cells", "num_exact")}
            rel = [x for t in tabs for x in t["_tot"]["num_rel_err"]]
            a["tables"] = {
                "tables_true": sum(t["tables_true"] for t in tabs),
                "tables_pred": sum(t["tables_pred"] for t in tabs),
                "cells": tot["cells"],
                "cell_exact": tot["exact"] / tot["cells"] if tot["cells"] else None,
                "cell_exact_normalised": tot["exact_norm"] / tot["cells"] if tot["cells"] else None,
                "numeric_exact": tot["num_exact"] / tot["num_cells"] if tot["num_cells"] else None,
                "numeric_rel_err_median": float(np.median(rel)) if rel else None,
            }
        xys = [d["xy"] for d in docs if "xy" in d]
        if xys:
            pe = [x for d in xys for x in d["point_err"]]
            S = lambda k: sum(d[k] for d in xys)  # noqa: E731
            a["xy"] = {
                "panels_true": S("panels_true"),
                "panels_found": S("panels_found"),
                "points": S("points"),
                "point_recall@0.01": S("hits") / S("points") if S("points") else None,
                "marker_points": S("marker_points"),
                "marker_recall@0.01": S("marker_hits") / S("marker_points") if S("marker_points") else None,
                "point_err_median": float(np.median(pe)) if pe else None,
                "point_err_mean": float(np.mean(pe)) if pe else None,
                "series_recall": S("series_true_matched") / S("series_true") if S("series_true") else None,
                "series_precision": S("series_pred_matched") / S("series_pred") if S("series_pred") else None,
                "series_pred": S("series_pred"),
            }
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
            fp = sum(
                round(((p.get("fields") or {}).get("precision") or 0) * ((p.get("fields") or {}).get("n_pred") or 0))
                for p in found
            )
            fr = sum(
                round(((p.get("fields") or {}).get("recall") or 0) * ((p.get("fields") or {}).get("n_true") or 0)) for p in found
            )
            it = {k: sum((p.get("invariants") or {}).get(k) or 0 for p in found) for k in ("n_pred", "n_true")}
            im = sum(
                round(((p.get("invariants") or {}).get("recall") or 0) * ((p.get("invariants") or {}).get("n_true") or 0))
                for p in found
            )
            a["phase_diagram"] = {
                "diagrams": len(pds),
                "diagrams_with_output": len(found),
                "boundary_precision@0.01": mean(["boundary", "precision@0.01"]),
                "boundary_recall@0.01": mean(["boundary", "recall@0.01"]),
                "boundary_mean_dist": mean(["boundary", "mean_pred_to_true"]),
                "boundary_hausdorff95": mean(["boundary", "hausdorff95"]),
                "boundary_median_dx": mean(["boundary", "median_abs_dx"]),
                "boundary_median_dT_K": mean(["boundary", "median_abs_dT_K"]),
                "field_precision": fp / ft["n_pred"] if ft["n_pred"] else None,
                "field_recall": fr / ft["n_true"] if ft["n_true"] else None,
                "field_pred": ft["n_pred"],
                "field_true_scorable": ft["n_true"],
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
            a["text"] = {
                "values_true": ntrue,
                "values_pred": npred,
                "precision": tp / npred if npred else None,
                "recall": tp / ntrue if ntrue else None,
                "value_only_recall": float(np.mean([t["value_only_recall"] for t in txs if t["value_only_recall"] is not None])),
            }

        # ---- v2 tasks: invariant table, phase stability, thermodynamic property tables
        def cells(ids):
            if not tabs:
                return {}
            t_ = dict.fromkeys(("cells", "exact", "exact_norm", "num_cells", "num_exact"), 0)
            rel = []
            for tb in tabs:
                for i_, v_ in tb["_by_id"].items():
                    if i_ in ids:
                        for k in t_:
                            t_[k] += v_[k]
                        rel += v_["num_rel_err"]
            if not t_["cells"]:
                return {}
            return {
                "cells": t_["cells"],
                "cell_exact": t_["exact"] / t_["cells"],
                "cell_exact_normalised": t_["exact_norm"] / t_["cells"],
                "numeric_exact": t_["num_exact"] / t_["num_cells"] if t_["num_cells"] else None,
                "numeric_rel_err_median": float(np.median(rel)) if rel else None,
            }

        def med(lst):
            return float(np.median(lst)) if lst else None

        def frac(n, d_):
            return n / d_ if d_ else None

        inv = cells({"invariants"})
        ir = [d["invariant_rows"] for d in docs if "invariant_rows" in d]
        if ir:
            S = lambda k: sum(r[k] for r in ir)  # noqa: E731
            L = lambda k: [x for r in ir for x in r[k]]  # noqa: E731
            inv.update(
                {
                    "rows_true": S("rows_true"),
                    "rows_pred": S("rows_pred"),
                    "row_recall": frac(S("matched"), S("rows_true")),
                    "row_precision": frac(S("matched"), S("rows_pred")),
                    "T_err_K_median": med(L("T_err")),
                    "T_printed_exact": frac(S("T_printed_exact"), S("matched")),
                    "x_err_median": med(L("x_err")),
                    "x_recall": frac(len(L("x_err")), S("x_n")),
                    "x_printed_exact": frac(S("x_printed_exact"), S("x_n")),
                    "type_accuracy": frac(S("type_ok"), S("matched")),
                    "evidence_accuracy": frac(S("ev_ok"), S("ev_n")),
                    "citation_accuracy": frac(S("cite_ok"), S("cite_n")),
                }
            )
        if inv:
            a["invariant_table"] = inv
        ps = cells({"phases"})
        pr = [d["phase_rows"] for d in docs if "phase_rows" in d]
        if pr:
            S = lambda k: sum(r[k] for r in pr)  # noqa: E731
            L = lambda k: [x for r in pr for x in r[k]]  # noqa: E731
            ps.update(
                {
                    "rows_true": S("rows_true"),
                    "rows_pred": S("rows_pred"),
                    "row_recall": frac(S("matched"), S("rows_true")),
                    "row_precision": frac(S("matched"), S("rows_pred")),
                    "structure_accuracy": frac(S("struct_ok"), S("struct_n")),
                    "pearson_accuracy": frac(S("pearson_ok"), S("matched")),
                    "strukturbericht_accuracy": frac(S("sb_ok"), S("matched")),
                    "prototype_accuracy": frac(S("proto_ok"), S("matched")),
                    "T_low_err_K_median": med(L("T_low_err")),
                    "T_high_err_K_median": med(L("T_high_err")),
                    "homogeneity_x_err_median": med(L("hom_err")),
                    "homogeneity_T_err_K_median": med(L("hom_T_err")),
                }
            )
        npq = [d["xy"]["by_quantity"]["NP"] for d in docs if "xy" in d and "NP" in d["xy"].get("by_quantity", {})]
        if npq:
            ps.update(
                {
                    "phase_fraction_panels": sum(q["panels"] for q in npq),
                    "phase_fraction_panels_found": sum(q["found"] for q in npq),
                    "phase_fraction_point_recall@0.01": frac(sum(q["hits"] for q in npq), sum(q["points"] for q in npq)),
                    "phase_fraction_point_err_median": med([x for q in npq for x in q["err"]]),
                }
            )
        if ps:
            a["phase_stability"] = ps
        th = cells({"formation", "mixing", "rk_liquid"})
        prr = [d["property_records"] for d in docs if "property_records" in d]
        if prr:
            S = lambda k: sum(r[k] for r in prr)  # noqa: E731
            L = lambda k: [x for r in prr for x in r[k]]  # noqa: E731
            th.update(
                {
                    "records_true": S("records_true"),
                    "records_pred": S("records_pred"),
                    "record_recall": frac(S("matched"), S("records_true")),
                    "record_precision": frac(S("matched"), S("records_pred")),
                    "value_exact": frac(S("value_exact"), S("matched")),
                    "rel_err_median": med(L("rel_err")),
                    "within_1pct": frac(S("within_1pct"), S("matched")),
                    "unit_accuracy": frac(S("unit_ok"), S("matched")),
                    "basis_accuracy": frac(S("basis_ok"), S("matched")),
                }
            )
        if th:
            a["thermo_tables"] = th
        agg[rendering] = a
    return agg


def main(argv=None):
    """Command-line entry point: score one extractor's outputs against the truth keys."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    root = Path(a.outputs)
    per_doc = {}
    for rendering in renderings:
        for f in sorted((root / rendering).glob("*.json")) if (root / rendering).exists() else []:
            pred = load_json(f)
            truth = load_json(truth_dir / f"{pred['doc_id']}.json")
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
                r_["tables"].pop("_by_id", None)
            if "xy" in r_:
                for q in r_["xy"].get("by_quantity", {}).values():
                    q["err"] = stats(q["err"])
            for k in ("invariant_rows", "phase_rows", "property_records"):
                for kk, vv in list((r_.get(k) or {}).items()):
                    if isinstance(vv, list):
                        r_[k][kk] = stats(vv)
            if "xy" in r_:
                r_["xy"]["point_err"] = stats(r_["xy"]["point_err"])
    res = {"extractor": root.name, "aggregate": agg, "per_document": per_doc}
    out = Path(a.out) if a.out else root / "scores.json"
    write_json(out, res, pretty=True)
    print(json.dumps(agg, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
