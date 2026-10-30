#!/usr/bin/env python3
"""AI-reader stage 3: reader answers -> benchmark extractor outputs (no model calls, no truth used).

Extractors written (outputs/<name>/<rendering>/<doc>.json):
  ai_gemini         gemini-3.8-flash, single reader, every document and rendering it answered
  ai_gemini_subset  the same answers restricted to the Claude subset (5 documents x vector, degraded)
  ai_claude         Claude (claude-opus-5.5 blind reader, file broker), the subset only
  ai_dual           production consensus on the subset: a record is accepted only when both readers give it
                    (same key and the same printed value); phase-diagram fields go through the assessed_pd ladder
                    with the two readers as a pair tier (both give the same choice -> accepted, then the rule gate)

Mappings (glue): inv-table rows -> invariant_rows (one row per reaction; rows whose source is experimental /
cited become that reaction's evidence); thermo-table v7.1 rows -> phase_rows (crystallographic tables) and
property_records (formation / mixing tables); running-text rows -> text_values after the production linker rule
(quote verbatim in the supplied passages and the printed number inside the quote; others refused); pd_fields answers ->
phase-diagram fields on top of the assessed_pd_auto annotation (single reader: its answer must be one of the choices
and pass the rule gate). Run with ~/orchid-py313/bin/python."""
import json
import pickle
import re
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from common import ASSESSED_PD, ORCHID_PIPELINE, TRUTH, load_json, write_json  # noqa: E402

FIX = ORCHID_PIPELINE / "pipelines/sgte-corpus/release-fixup-20261022/bin-context"
sys.path.insert(0, str(FIX))
import text_reader_v6 as R6  # noqa: E402

APD = str(ASSESSED_PD)
sys.path.insert(0, APD)

AI = BENCH / "work" / "ai"
SUBSET = ["syn-AgCu-06He-v0", "syn-AlNi-14Zhu-v0", "syn-CuZn-08Gie-v0", "syn-PbZn-98Jan-v0", "syn-BiSn-08Li-v0"]
SUB_R = ["vector", "degraded_raster"]
RENDS = ["vector", "clean_raster", "degraded_raster", "image_only_pdf"]


def jobs():
    out = {}
    for f in sorted((AI / "jobs").glob("*.json")):
        j = json.loads(f.read_text())
        out.setdefault((j["doc"], j["rendering"]), []).append(j)
    return out


def gemini_answer(key):
    p = AI / "gemini" / f"{key}.json"
    return json.loads(p.read_text())["result"] if p.exists() else None


def claude_answer(key):
    p = AI / "claude" / "answers" / f"{key}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def tunit(u):
    return {"C": "°C", "°C": "°C", "K": "K"}.get(u or "", "K")


def xunit(b):
    return "at.%" if b in ("atomic_percent", "atom_percent") else "mole_fraction"


def cit(c):
    for x in c or []:
        m = re.search(r"\d+", str(x))
        if m:
            return int(m.group(0))
    return None


# ------------------------------------------------------------------------------------------------ tables
def inv_rows(res):
    if not res:
        return []
    by = {}
    order = []
    xu = xunit(res.get("composition_basis"))
    for r in res.get("rows", []):
        key = re.sub(r"\s+", "", (r.get("reaction") or "").replace("→", "->"))
        if key not in by:
            by[key] = {"main": None, "ev": []}
            order.append(key)
        src = r.get("source")
        if src in ("experimental", "cited_unspecified") or (src == "other_calculation" and r.get("citations")):
            if r.get("temperature") is not None:
                by[key]["ev"].append({"class": "experimental", "T": r["temperature"], "citation": cit(r.get("citations"))})
        elif by[key]["main"] is None:
            by[key]["main"] = r
    out = []
    for k in order:
        r = by[k]["main"]
        if r is None or r.get("temperature") is None:
            continue
        out.append({"reaction": r.get("reaction"), "phases": r.get("phases") or [], "type": (r.get("reaction_type") or "").strip() or None,
                    "T": r["temperature"], "T_unit": tunit(r.get("temperature_unit") or res.get("temperature_unit")), "x_unit": xu,
                    "compositions": [{"phase": c.get("phase", ""), "x": c.get("value")} for c in r.get("compositions") or []
                                     if c.get("value") is not None],
                    "evidence": by[k]["ev"]})
    return out


def is_phase_table(res, job):
    cap = job["prompt"]
    return res.get("table_kind") == "crystallographic" or re.search(r"Caption: [^\n]*(Solid phases|structure)", cap)


def is_mixing(job):
    return bool(re.search(r"Caption: [^\n]*mixing", job["prompt"], re.I))


def basis_of(text):
    return "formula_unit" if re.search(r"(?i)f\.?\s?u\.?|formula unit", text or "") else "mol_atoms"


def phase_rows(res):
    out = []
    xu = xunit(res.get("composition_basis"))
    for r in res.get("rows", []):
        st = r.get("structure") or {}
        ph = st.get("phase") or r.get("row_label")
        if not ph:
            continue
        tu = tunit(r.get("temperature_unit") or res.get("temperature_unit"))
        pcs = [c for c in r.get("phase_compositions") or [] if c.get("value") is not None]
        hom = {"x_min": pcs[0]["value"] if pcs else None, "x_max": (pcs[0].get("value_max") if pcs else None), "x_unit": xu,
               "T": None, "T_unit": tu}
        for p in r.get("properties") or []:
            if re.search(r"(?i)^T\b|temperature", p.get("quantity", "")) and p.get("value") is not None:
                hom["T"] = p["value"]
        out.append({"phase": ph, "pearson": st.get("pearson_symbol"), "strukturbericht": st.get("strukturbericht"),
                    "prototype": st.get("prototype"),
                    "T_low": None if r.get("temperature_bound") == "<" else r.get("temperature"),
                    "T_high": r.get("temperature_max"), "T_unit": tu, "homogeneity": hom})
    return out


QK = {"formation_enthalpy": "dfH", "entropy": "dfS", "heat_capacity": "Cp", "mixing_enthalpy": "HM_MIX", "gibbs_energy": "GM_MIX",
      "partial_enthalpy": "HM_PART", "activity": "a"}


def prop_records(res, job, elements):
    out = []
    A, B = elements
    cu = " ".join(f"{c.get('column_header', '')} {c.get('unit', '')}" for c in res.get("column_units") or [])
    mixing = is_mixing(job)
    xu = xunit(res.get("composition_basis"))
    for r in res.get("rows", []):
        x = None
        for c in r.get("overall_composition") or []:
            if str(c.get("element", "")).lower() == B.lower() and c.get("value") is not None:
                x = c["value"]
        for p in r.get("properties") or []:
            if p.get("value") is None:
                continue
            k = QK.get(p.get("quantity_kind"))
            q = p.get("quantity", "")
            if k is None:
                if re.search(r"ΔfH|ΔHf|formation", q):
                    k = "dfH"
                elif re.search(r"ΔfS|S\b", q):
                    k = "dfS"
                elif "Cp" in q:
                    k = "Cp"
                else:
                    continue
            comp = (p.get("component") or "") + " " + q
            if k == "a":
                k = "a_A" if re.search(rf"\b{A}\b", comp) else ("a_B" if re.search(rf"\b{B}\b", comp) else None)
            elif k == "HM_PART":
                k = "HM_PART_A" if re.search(rf"\b{A}\b", comp) else ("HM_PART_B" if re.search(rf"\b{B}\b", comp) else None)
            if k is None:
                continue
            unit = (p.get("unit") or "").replace("−", "-").replace("⁻¹", "-1")
            u = unit.lower().replace(" ", "")
            un = "1" if k.startswith("a_") else ("kJ/mol" if "kj" in u else "cal/(mol K)" if "cal" in u and "k" in u.replace("kcal", "") else
                                                  "cal/mol" if "cal" in u else "J/(mol K)" if re.search(r"j/?\(?mol.?k|jmol-1k-1|j/molk|jk", u) else
                                                  "J/mol" if "j" in u else unit)
            if mixing or k in ("HM_MIX", "GM_MIX", "a_A", "a_B", "HM_PART_A", "HM_PART_B"):
                if x is None:
                    continue
                out.append({"table": "mixing", "x": x, "x_unit": xu, "quantity": k, "value": p["value"], "unit": un, "basis": "mol_atoms"})
            else:
                out.append({"table": "formation", "phase": p.get("phase") or r.get("row_label") or "", "quantity": k, "value": p["value"],
                            "unit": un, "basis": basis_of(unit + " " + cu + " " + job["prompt"][:400])})
    return out


# ------------------------------------------------------------------------------------------------ text
def text_values(res, job):
    if not res:
        return [], 0
    passages = " ".join(R6.norm(q["text"]) for q in job["meta"]["passages"])
    out, refused = [], 0
    xu = xunit(res.get("composition_basis"))
    for r in res.get("rows", []):
        quote = R6.norm(r.get("quote") or "")
        if not quote or quote not in passages:
            refused += 1
            continue
        vals = []
        if r.get("temperature") is not None:
            vals.append((r["temperature"], tunit(r.get("temperature_unit") or res.get("temperature_unit")), r.get("temperature_printed")))
        for c in (r.get("phase_compositions") or []) + (r.get("overall_composition") or []):
            if c.get("value") is not None:
                vals.append((c["value"], xu, None))
        for p in r.get("properties") or []:
            if p.get("value") is not None:
                u = (p.get("unit") or "").replace(" ", "")
                vals.append((p["value"], "kJ/mol" if "kJ" in u else "cal/mol" if "cal" in u else "J/(mol K)" if "K" in u and "J" in u
                             else "J/mol" if "J" in u else u, None))
        for v, u, printed in vals:
            num = printed if printed else (f"{v:g}" if isinstance(v, (int, float)) else str(v))
            n0 = re.search(r"-?\d+(?:\.\d+)?", str(num).replace("−", "-"))
            if not n0 or n0.group(0).lstrip("-") not in quote:
                refused += 1
                continue
            out.append({"value": v, "unit": u})
    return out, refused


# ------------------------------------------------------------------------------------------------ phase-diagram fields
class FileReader:
    def __init__(self, name, answers, markers):
        self.name = name
        self.by_field = {}
        for a in (answers or {}).get("answers", []):
            m = next((m for m in markers if m["marker"] == a.get("marker")), None)
            if m:
                self.by_field[m["field_id"]] = a.get("assemblage")

    def read(self, questions):
        from assessed_pd.readers import ReaderAnswer
        return [ReaderAnswer(field_id=q.field_id, answer=self.by_field[q.field_id]) for q in questions if q.field_id in self.by_field]


def pd_output(doc, r, job, ans_a, ans_b):
    from assessed_pd.export import build_document
    from assessed_pd.pipeline import annotate
    from assessed_pd.readers import Ladder, Tier
    pk = BENCH / "work" / "raster_pd" / doc / r / "assessed_input.pkl"
    base = load_json(BENCH / "outputs" / "assessed_pd_auto" / r / f"{doc}.json")
    if not pk.exists() or not base.get("phase_diagrams"):
        return None, {}
    d = pickle.load(open(pk, "rb"))
    ann = annotate(d["inp"], d["cfg"])
    run = {}
    if job is not None and ans_a is not None and (ans_b is not None):
        mk = job["meta"]["markers"]
        lad = Ladder([Tier("pair", (FileReader("a", ans_a, mk), FileReader("b", ans_b, mk)))])
        run = lad.run(ann)
    doc_ = build_document(ann)
    full = doc_["diagram"]["composition_full_scale"]
    pdo = dict(base["phase_diagrams"][0])
    pdo["fields"] = [{"point": [f["label_point"]["x"] / full, f["label_point"]["T"]], "phases": f["phases"]} for f in doc_["fields"] if f["phases"]]
    return pdo, {"accepted": len(run.get("accepted", {})), "human_queue": len(run.get("human_queue", []))}


# ------------------------------------------------------------------------------------------------ assemble
def build(doc, r, js, reader):
    t = load_json(TRUTH / f"{doc}.json")
    inv, ph, recs, tv = [], [], [], []
    st = {"answered": 0, "missing": 0, "text_refused": 0}
    pdjob = None
    for j in js:
        if j["kind"] == "context":
            continue
        res = reader(j["key"])
        if j["kind"] == "pd_fields":
            pdjob = j
            continue
        if res is None:
            st["missing"] += 1
            continue
        st["answered"] += 1
        if j["kind"] == "inv_table":
            inv += inv_rows(res)
        elif j["kind"] == "thermo_table":
            if is_phase_table(res, j):
                ph += phase_rows(res)
            else:
                recs += prop_records(res, j, t["elements"])
        elif j["kind"] == "text":
            v, nref = text_values(res, j)
            tv += v
            st["text_refused"] += nref
    return {"invariant_rows": inv, "phase_rows": ph, "property_records": recs, "text_values": tv}, st, pdjob


def agree(a, b):
    """Consensus of two readers' records: kept when both give the same key and the same printed value."""
    out = {}
    # invariant rows: same phase set and same printed T; compositions / evidence kept where both agree
    ib = list(b["invariant_rows"])
    inv = []
    for r in a["invariant_rows"]:
        m = next((q for q in ib if sorted(map(str, q["phases"])) == sorted(map(str, r["phases"])) and q["T"] == r["T"] and q["T_unit"] == r["T_unit"]), None)
        if m is None:
            continue
        ib.remove(m)
        comps = [c for c in r["compositions"] if any(c2["phase"] == c["phase"] and c2["x"] == c["x"] for c2 in m["compositions"])]
        ev = [e for e in r["evidence"] if any(e2["T"] == e["T"] and e2["citation"] == e["citation"] for e2 in m["evidence"])]
        inv.append({**r, "type": r["type"] if r["type"] == m["type"] else None, "compositions": comps,
                    "evidence": ev if (ev or (not r["evidence"] and not m["evidence"])) else [], "x_unit": r["x_unit"] if r["x_unit"] == m["x_unit"] else None})
    out["invariant_rows"] = [x for x in inv if x["x_unit"] is not None] + [{**x, "x_unit": "mole_fraction", "compositions": []} for x in inv if x["x_unit"] is None]
    pb = {str(p["phase"]): p for p in b["phase_rows"]}
    ph = []
    for p in a["phase_rows"]:
        q = pb.get(str(p["phase"]))
        if q is None:
            continue
        rec = {"phase": p["phase"], "T_unit": p["T_unit"] if p["T_unit"] == q["T_unit"] else None}
        for k in ("pearson", "strukturbericht", "prototype"):
            rec[k] = p[k] if p[k] == q[k] else None
        for k in ("T_low", "T_high"):
            rec[k] = p[k] if (p[k] == q[k] and rec["T_unit"]) else None
        ha, hb = p["homogeneity"], q["homogeneity"]
        rec["homogeneity"] = {k: (ha[k] if ha.get(k) == hb.get(k) else None) for k in ("x_min", "x_max", "T")}
        rec["homogeneity"].update(x_unit=ha["x_unit"] if ha["x_unit"] == hb["x_unit"] else "mole_fraction",
                                  T_unit=ha["T_unit"])
        if ha["x_unit"] != hb["x_unit"]:
            rec["homogeneity"].update(x_min=None, x_max=None)
        ph.append(rec)
    out["phase_rows"] = ph

    def pk(r):
        return (r["table"], r.get("phase") if r["table"] == "formation" else r.get("x"), r["quantity"], r["value"], r["unit"], r.get("basis"))
    pbk = [pk(r) for r in b["property_records"]]
    recs = []
    for r in a["property_records"]:
        k = pk(r)
        if k in pbk:
            pbk.remove(k)
            recs.append(r)
    out["property_records"] = recs
    tb = [(v["value"], v["unit"]) for v in b["text_values"]]
    tv = []
    for v in a["text_values"]:
        k = (v["value"], v["unit"])
        if k in tb:
            tb.remove(k)
            tv.append(v)
    out["text_values"] = tv
    return out


def ablation():
    """Pilot documents x (vector, degraded): thermo tables re-read with prompts v5 / v6 (same crops) vs v7.1."""
    abl = {}
    for f in sorted((AI / "jobs_ablation").glob("*.json")):
        j = json.loads(f.read_text())
        abl[(j["meta"]["ablation_of"], j["kind"][-2:])] = j
    J = jobs()
    for (doc, r), js in sorted(J.items()):
        if doc not in ("syn-AgCu-06He-v0", "syn-AlNi-14Zhu-v0") or r not in SUB_R:
            continue
        t = load_json(TRUTH / f"{doc}.json")
        for ver in ("v71", "v5", "v6"):
            ph, recs = [], []
            for j in js:
                if j["kind"] != "thermo_table":
                    continue
                jj = j if ver == "v71" else abl.get((j["key"], ver))
                res = gemini_answer(jj["key"]) if jj else None
                if not res:
                    continue
                if is_phase_table(res, j):
                    ph += phase_rows(res)
                else:
                    recs += prop_records(res, j, t["elements"])
            write_json(BENCH / "outputs" / f"ai_gemini_thermo_{ver}_pilot" / r / f"{doc}.json",
                       {"doc_id": doc, "rendering": r, "extractor": f"ai_gemini_thermo_{ver}_pilot",
                        "status": {"ablation": f"thermo tables read with prompt {ver}"}, "phase_rows": ph, "property_records": recs})


def main():
    ablation()
    J = jobs()
    for (doc, r), js in sorted(J.items()):
        g, gst, pdjob = build(doc, r, js, gemini_answer)
        ga = gemini_answer(pdjob["key"]) if pdjob else None
        pdo, pst = pd_output(doc, r, pdjob, ga, ga) if pdjob else pd_output(doc, r, None, None, None)
        status = {"reader": "gemini-3.8-flash (single reader)", **gst, "pd_fields": pst or "no open fields / no answer"}
        out = {"doc_id": doc, "rendering": r, "extractor": "ai_gemini", "status": status, **g,
               "phase_diagrams": [pdo] if pdo else []}
        write_json(BENCH / "outputs" / "ai_gemini" / r / f"{doc}.json", out)
        if doc in SUBSET and r in SUB_R:
            write_json(BENCH / "outputs" / "ai_gemini_subset" / r / f"{doc}.json", {**out, "extractor": "ai_gemini_subset"})
            c, cst, _ = build(doc, r, js, claude_answer)
            ca = claude_answer(pdjob["key"]) if pdjob else None
            cpdo, cpst = pd_output(doc, r, pdjob, ca, ca) if pdjob else (None, {})
            write_json(BENCH / "outputs" / "ai_claude" / r / f"{doc}.json",
                       {"doc_id": doc, "rendering": r, "extractor": "ai_claude",
                        "status": {"reader": "claude-opus-5.5 (blind file-broker reader)", **cst, "pd_fields": cpst}, **c,
                        "phase_diagrams": [cpdo] if cpdo else []})
            dual = agree(g, c)
            dpdo, dpst = pd_output(doc, r, pdjob, ga, ca) if pdjob else (None, {})
            write_json(BENCH / "outputs" / "ai_dual" / r / f"{doc}.json",
                       {"doc_id": doc, "rendering": r, "extractor": "ai_dual",
                        "status": {"rule": "accepted only when both readers agree (production consensus)", "pd_fields": dpst}, **dual,
                        "phase_diagrams": [dpdo] if dpdo else []})
        print(doc, r, json.dumps(status)[:200], flush=True)


if __name__ == "__main__":
    main()
