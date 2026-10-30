#!/usr/bin/env python3
"""Table baseline on the vector PDF: pdfplumber table finder (Camelot, the thermo_extractor table route, is not
installed). Strategy 1: ruling lines both ways; strategy 2 (regions bounded by long horizontal rules that strategy 1
did not cover, e.g. booktabs or zebra tables): horizontal rules + text-aligned columns inside that region only.
Raster renderings: not run (thermo_extractor.tables.ocr needs OpenCV).

Glue written for this benchmark (no truth used): each extracted table is classified by its header words and parsed into
the semantic records scored by score.py (invariant_rows, phase_rows, property_records): units and basis from the header
text, reaction phases split on '→' and '+', experimental column and citation from 'exp.' / 'Ref.' / '[n]'.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import bench, docs_dir, truth_dir, load_json, write_json  # noqa: E402

OUT = bench / "outputs" / "tables_pdfplumber"
MARKS = "ᵃᵇᶜᵈ"
TYPES = [
    "congruent melting",
    "congruent transformation",
    "melting (pure component)",
    "eutectic",
    "eutectoid",
    "peritectic",
    "peritectoid",
    "monotectic",
    "monotectoid",
    "syntectic",
    "metatectic",
]


def rule_regions(page):
    """Bounding boxes of groups of long horizontal rules (same x-extent, vertically close)."""
    hs = []
    for e in page.edges:
        if e["orientation"] == "h" and e["x1"] - e["x0"] > 0.25 * page.width:
            hs.append((round(e["x0"]), round(e["x1"]), e["top"]))
    groups = []
    for x0, x1, y in sorted(hs, key=lambda q: q[2]):
        for g in groups:
            if abs(g["x0"] - x0) <= 2 and abs(g["x1"] - x1) <= 2 and y - g["y1"] < 350:
                g["y1"] = max(g["y1"], y)
                g["n"] += 1
                break
        else:
            groups.append({"x0": x0, "x1": x1, "y0": y, "y1": y, "n": 1})
    return [(g["x0"] - 1, g["y0"] - 1, g["x1"] + 1, g["y1"] + 1) for g in groups if g["n"] >= 2 and g["y1"] - g["y0"] > 15]


def _overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def tables_of(page):
    out = []
    for tb in page.find_tables():
        rows = tb.extract()
        if rows and len(rows) >= 2:
            out.append({"page": page.page_number - 1, "bbox": list(tb.bbox), "route": "lines/lines", "rows": rows})
    for reg in rule_regions(page):
        if any(_overlap(reg, t["bbox"]) for t in out):
            continue
        try:
            found = page.crop(reg).find_tables({"vertical_strategy": "text", "horizontal_strategy": "text"})
        except Exception:  # noqa: BLE001
            continue
        for tb in found:
            rows = [r for r in tb.extract() if any((c or "").strip() for c in r)]
            if rows and len(rows) >= 2:
                out.append(
                    {"page": page.page_number - 1, "bbox": list(tb.bbox), "route": "text/text in ruled region", "rows": rows}
                )
    res = []
    for t in out:
        cells = [
            {"row": i, "col": j, "text": (c or "").replace("\n", " ")} for i, r in enumerate(t["rows"]) for j, c in enumerate(r)
        ]
        res.append({"page": t["page"], "bbox": t["bbox"], "route": t["route"], "cells": cells})
    return res


# ------------------------------------------------------------------------------------------------ semantic glue
def num(s):
    if s is None:
        return None
    t = s.replace("−", "-").replace("–", "-").replace(" ", "")
    t = t.strip(MARKS).lstrip("<>")
    m = re.match(r"^\(?([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", t)
    return float(m.group(1)) if m else None


def grid(tb):
    rows = {}
    for c in tb["cells"]:
        rows.setdefault(c["row"], {})[c["col"]] = c["text"] or ""
    n = max((max(r) for r in rows.values() if r), default=-1) + 1
    return [[rows[i].get(j, "") for j in range(n)] for i in sorted(rows)]


def split_header(g):
    """Header rows: leading rows until the first row with at least two numeric cells."""
    for i, r in enumerate(g):
        if sum(num(c) is not None for c in r) >= 2:
            return g[:i], g[i:]
    return g, []


def col_headers(head, n):
    return [" ".join(r[j] for r in head if j < len(r) and r[j]).strip() for j in range(n)]


def units_T(text):
    return "°C" if "°C" in text else "K"


def units_x(text):
    return "at.%" if "at.%" in text else "mole_fraction"


def classify(head_text, body):
    h = head_text
    if "Reaction" in h or any("→" in c for r in body for c in r):
        return "invariants"
    if "Pearson" in h or "Prototype" in h:
        return "phases"
    if "ΔfH" in h:
        return "formation"
    if "ΔmixH" in h or "a(" in h:
        return "mixing"
    return None


def parse_invariants(head, body, H):
    allh = " ".join(H)
    tu, xu = units_T(allh), units_x(allh)
    n = len(H)
    tc = next((j for j in range(n) if any(r[j].strip(MARKS).strip() in TYPES for r in body)), None)
    rc = next((j for j in range(n) if any("→" in r[j] for r in body)), 0)
    if tc is not None and tc > 1 and rc < tc:  # reaction split over several text columns: join them
        body = [[" ".join(c for c in r[:tc] if c.strip())] + r[1:] for r in body]
        rc = 0
    expc = next((j for j in range(n) if "exp" in H[j]), None)
    refc = next((j for j in range(n) if "Ref" in H[j]), None)
    Tc = next(
        (
            j
            for j in range(n)
            if j not in (rc, tc, expc, refc) and j > (tc or 0) and (re.search(r"\bT\b", H[j]) or "this work" in H[j])
        ),
        None,
    )
    xcs = [
        j
        for j in range(n)
        if j not in (rc, tc, expc, refc, Tc) and ("phase" in H[j] or "x(" in H[j] or "at.%" in H[j] or re.search(r", ?\d$", H[j]))
    ]
    out = []
    for r in body:
        rx = r[rc]
        if "→" not in rx:
            continue
        lhs, rhs = rx.split("→", 1)
        phases = [p.strip() for p in lhs.split("+")] + [p.strip() for p in rhs.split("+")]
        comps = []
        for k, j in enumerate(xcs[: len(phases)]):
            v = num(r[j])
            if v is not None:
                comps.append({"phase": phases[k], "x": v})
        ev = []
        if expc is not None:
            e = r[expc]
            v = num(e)
            m = re.search(r"\[(\d+)\]", e + " " + (r[refc] if refc is not None else ""))
            if v is not None:
                ev.append({"class": "experimental", "T": v, "citation": int(m.group(1)) if m else None})
        out.append(
            {
                "reaction": rx,
                "phases": phases,
                "type": r[tc].strip(MARKS).strip() if tc is not None else None,
                "T": num(r[Tc]) if Tc is not None else None,
                "T_unit": tu,
                "x_unit": xu,
                "compositions": comps,
                "evidence": ev,
            }
        )
    return out


def parse_phases(head, body, H):
    allh = " ".join(H)
    tu = units_T(allh)
    n = len(H)

    def find(*keys):
        return next((j for j in range(n) for k in keys if k in H[j]), None)

    jp, jpe, jsb, jpr = find("Phase"), find("Pearson"), find("Strukturb", "SB"), find("Prototype")
    jlo = next((j for j in range(n) if re.search(r"(\bfrom|T min.*)$", H[j])), None)
    jhi = next((j for j in range(n) if re.search(r"(\bto|T max.*)$", H[j])), None)
    jr = find("Range", "x(", "at.%")
    jt = next(
        (j for j in range(n) if j not in (jlo, jhi) and ("at T" in H[j] or re.fullmatch(r"T \((K|°C)\)", H[j].strip()))), None
    )
    xu = units_x(H[jr]) if jr is not None else "mole_fraction"
    out = []
    for r in body:

        def g(j):
            return r[j].strip() if j is not None and j < len(r) else ""

        if not g(jp):
            continue
        blank = lambda s: None if s in ("", "–", "-") else s  # noqa: E731
        lo = g(jlo)
        rng = g(jr).split("–")
        out.append(
            {
                "phase": g(jp),
                "pearson": blank(g(jpe)),
                "strukturbericht": blank(g(jsb)),
                "prototype": blank(g(jpr)),
                "T_low": None if lo.startswith("<") else num(lo),
                "T_high": num(g(jhi)),
                "T_unit": tu,
                "homogeneity": {"x_min": num(rng[0]), "x_max": num(rng[-1]), "x_unit": xu, "T": num(g(jt)), "T_unit": tu},
            }
        )
    return out


def eunit(h):
    if "kJ" in h:
        return "kJ/mol"
    if "cal" in h:
        return "cal/(mol K)" if "K⁻¹" in h else "cal/mol"
    if "K⁻¹" in h:
        return "J/(mol K)"
    if "J" in h:
        return "J/mol"
    return "1"


def parse_formation(head, body, H):
    allh = " ".join(H) + " " + " ".join(c for r in head for c in r)
    basis = "formula_unit" if ("f.u." in allh or "formula unit" in allh) else "mol_atoms"
    n = len(H)
    jp = next((j for j in range(n) if "Phase" in H[j]), 0)
    qs = [(j, q) for j in range(n) for q, k in (("dfH", "ΔfH"), ("dfS", "ΔfS"), ("Cp", "Cp")) if k in H[j]]
    out = []
    for r in body:
        ph = r[jp].strip().strip(MARKS)
        for j, q in qs:
            v = num(r[j])
            if v is not None:
                out.append({"table": "formation", "phase": ph, "quantity": q, "value": v, "unit": eunit(H[j]), "basis": basis})
    return out


def parse_mixing(head, body, H):
    n = len(H)
    jx = next((j for j in range(n) if "x(" in H[j] or "at.%" in H[j]), 0)
    xu = units_x(H[jx])
    acts = [j for j in range(n) if re.search(r"(^|\s)a\(", H[j])]
    parts = [j for j in range(n) if "H̄" in H[j]]
    out = []
    for r in body:
        x = num(r[jx])
        if x is None:
            continue
        for j in range(n):
            if j == jx:
                continue
            h = H[j]
            q = None
            if "ΔmixH" in h:
                q = "HM_MIX"
            elif "ΔmixG" in h:
                q = "GM_MIX"
            elif j in acts:
                q = "a_A" if j == acts[0] else "a_B"
            elif j in parts:
                q = "HM_PART_A" if j == parts[0] else "HM_PART_B"
            v = num(r[j])
            if q and v is not None:
                out.append(
                    {
                        "table": "mixing",
                        "x": x,
                        "x_unit": xu,
                        "quantity": q,
                        "value": v,
                        "unit": "1" if q.startswith("a_") else eunit(h),
                        "basis": "mol_atoms",
                    }
                )
    return out


def semantic(tables):
    inv, ph, recs = [], [], []
    for tb in tables:
        g = grid(tb)
        if not g:
            continue
        head, body = split_header(g)
        n = len(g[0])
        H = col_headers(head, n) if head else [""] * n
        kind = classify(" ".join(H), body)
        try:
            if kind == "invariants":
                inv += parse_invariants(head, body, H)
            elif kind == "phases":
                ph += parse_phases(head, body, H)
            elif kind == "formation":
                recs += parse_formation(head, body, H)
            elif kind == "mixing":
                recs += parse_mixing(head, body, H)
        except Exception:  # noqa: BLE001
            continue
        tb["glue_kind"] = kind
    return inv, ph, recs


def main():
    import pdfplumber

    for tf in sorted(truth_dir.glob("*.json")):
        t = load_json(tf)
        doc = t["doc_id"]
        with pdfplumber.open(docs_dir / doc / "vector.pdf") as pdf:
            tabs = [x for p in pdf.pages for x in tables_of(p)]
        inv, ph, recs = semantic(tabs)
        write_json(
            OUT / "vector" / f"{doc}.json",
            {
                "doc_id": doc,
                "rendering": "vector",
                "extractor": "tables_pdfplumber",
                "status": {"tables": "ran: pdfplumber 0.11.9", "semantic": "ran: header-keyword glue over pdfplumber cells"},
                "tables": tabs,
                "invariant_rows": inv,
                "phase_rows": ph,
                "property_records": recs,
            },
        )


if __name__ == "__main__":
    main()
