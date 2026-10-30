"""Truth-schema v2 tables: invariant reactions (with optional cited 'experimental' columns), phase stability and
structure, formation properties at 298.15 K, liquid mixing properties. Every builder returns a table spec for
document.Doc.table (header, rows, align, footnotes, column_units) plus `meta` (per body cell: semantic field, exact
SI value, printed unit and basis, evidence class, citation, footnote marker) and the semantic records of the table.
"""
from __future__ import annotations

from common import fmt
from systems import base

cal = 4.184
mark = ["ᵃ", "ᵇ", "ᶜ", "ᵈ"]
inv_order = ["eutectic", "peritectic", "monotectic", "syntectic", "metatectic", "eutectoid", "peritectoid",
             "monotectoid", "congruent melting", "congruent transformation", "melting (pure component)"]


def tval(T_K, unit):
    return T_K - 273.15 if unit == "°C" else T_K


def xprint(x, xunit):
    return fmt(100 * x, 1) if xunit == "at.%" else fmt(x, 3)


def xu_label(xunit, B):
    return f"at.% {B}" if xunit == "at.%" else f"x({B})"


def _m(field, **kw):
    return {"field": field, **kw}


# ------------------------------------------------------------------------------------------------ invariant table
def invariant_rows(data, ext, labeller):
    """Logical invariant rows: three-phase invariants, congruent points of intermediate phases, and (systems without
    either) the melting of the pure components.
    """
    rows = []
    for k, inv in enumerate(data["invariants"]):
        pmap = {p["phase"]: p for p in inv["points"]}
        order = inv["reactants"] + inv["products"]
        rows.append({"source": f"invariant:{k}", "T_K": inv["T"], "type": inv["type"], "reaction": inv["reaction"],
                     "phases": [{"label": pmap[p]["label"], "tdb_phase": base(p), "x": pmap[p]["x"]} for p in order]})
    for p in ext["phase_stability"]:
        cp = p.get("congruent_point")
        if not cp or cp["kind"] not in ("congruent melting", "congruent transformation"):
            continue
        if cp["kind"] == "congruent melting":
            parent = {"label": "L", "tdb_phase": "LIQUID", "x": cp["x"]}
        else:
            ph, xx = cp["neighbour_state"][0]
            parent = {"label": labeller.label(ph, xx), "tdb_phase": base(ph), "x": cp["x"]}
        me = {"label": p["label"], "tdb_phase": p["tdb_phases"][0], "x": cp["x"]}
        rows.append({"source": f"congruent:{p['label']}", "T_K": cp["T_K"], "type": cp["kind"],
                     "reaction": f"{parent['label']} → {me['label']}", "phases": [parent, me]})
    if not rows:
        A, B = data["elements"]
        for e, Tm, x in ((A, data["Tm_A"], 0.0), (B, data["Tm_B"], 1.0)):
            if Tm is None:
                continue
            sol = next((p for p in ext["phase_stability"] if abs(x - p["homogeneity"]["x_min" if x == 0 else "x_max"]) < 1e-3), None)
            lab = sol["label"] if sol else f"({e})"
            rows.append({"source": f"melting:{e}", "T_K": Tm, "type": "melting (pure component)",
                         "reaction": f"L → {lab}", "phases": [{"label": "L", "tdb_phase": "LIQUID", "x": x},
                                                              {"label": lab, "tdb_phase": sol["tdb_phases"][0] if sol else None, "x": x}]})
    rows.sort(key=lambda r: -r["T_K"])
    return rows


def invariant_table(data, ext, labeller, st, rng, refs):
    B = data["elements"][1]
    tunit, xunit = st["T_unit"], st["x_unit"]
    exp = st["exp_column"]
    inline = st["exp_ref_style"] == "inline"
    logical = invariant_rows(data, ext, labeller)
    rows, meta, sem = [], [], []
    marks = st["markers"]
    cong_mark = mark[0] if marks else ""
    exp_mark = (mark[1] if marks else "") if exp else ""
    for i, r in enumerate(logical):
        Tv = tval(r["T_K"], tunit)
        Ts = fmt(Tv, 1)
        typ = r["type"]
        typ_txt = typ + (cong_mark if typ.startswith("congruent") or typ.startswith("melting") else "")
        row = [r["reaction"], typ_txt, Ts]
        mrow = [_m("reaction", value=r["reaction"]), _m("type", value=typ, footnote_marker=(cong_mark if typ_txt != typ else None)),
                _m("T", exact_SI=r["T_K"], unit=tunit, evidence_class="calculated_this_work", citation=None)]
        ev = [{"class": "calculated_this_work", "value": float(Ts), "printed": Ts, "unit": tunit, "citation": None}]
        if exp:
            if rng.random() < 0.75:
                off = float([-7, -5, -4, -3, -2, 2, 3, 4, 6, 9][int(rng.integers(10))])
                cite = int(rng.integers(3, 3 + len(refs["fictitious"])))
                es = fmt(float(Ts) + off, 1)
                ev.append({"class": "experimental_cited", "value": float(es), "printed": es, "unit": tunit, "citation": cite,
                           "offset_K": off, "fictitious": True})
                if inline:
                    row.append(f"{es} [{cite}]")
                    mrow.append(_m("T_exp", exact_SI=r["T_K"] + off, unit=tunit, evidence_class="experimental_cited",
                                   citation=cite, offset_K=off))
                else:
                    row += [es, f"[{cite}]"]
                    mrow += [_m("T_exp", exact_SI=r["T_K"] + off, unit=tunit, evidence_class="experimental_cited",
                                citation=cite, offset_K=off), _m("citation", value=cite, evidence_class="experimental_cited")]
            else:
                row += ["–"] if inline else ["–", "–"]
                mrow += [_m("T_exp", evidence_class="none")] if inline else [_m("T_exp", evidence_class="none"),
                                                                              _m("citation", evidence_class="none")]
        phs = []
        for k in range(3):
            if k < len(r["phases"]):
                p = r["phases"][k]
                s = xprint(p["x"], xunit)
                row.append(s)
                mrow.append(_m(f"x:{k}", exact_SI=p["x"], unit=xunit, phase=p["label"], tdb_phase=p["tdb_phase"]))
                phs.append({**p, "printed": s, "value": float(s), "x_unit": xunit})
            else:
                row.append("–")
                mrow.append(_m(f"x:{k}", phase=None))
        rows.append(row)
        meta.append(mrow)
        sem.append({"row": i, "source": r["source"], "reaction": r["reaction"], "type": typ, "T_K": r["T_K"],
                    "T_printed": Ts, "T_value": float(Ts), "T_unit": tunit, "phases": phs, "x_unit": xunit, "evidence": ev,
                    "has_exp_column": exp})
    xu = xu_label(xunit, B)
    tcol = f"T ({tunit})" if not st["merged_header"] else f"T / {tunit}"
    expcols = []
    if exp:
        expcols = [{"text": "T exp." if inline else "T exp."}] + ([] if inline else [{"text": "Ref."}])
    if st["merged_header"]:
        tw = [{"text": "this work"}] + ([{"text": "exp."}] + ([] if inline else [{"text": "Ref."}]) if exp else [])
        header = [[{"text": "Reaction"}, {"text": "Type"}, {"text": tcol, "colspan": len(tw)},
                   {"text": f"Compositions ({xu})", "colspan": 3}],
                  [{"text": ""}, {"text": ""}] + tw + [{"text": "phase 1"}, {"text": "phase 2"}, {"text": "phase 3"}]]
    else:
        header = [[{"text": "Reaction"}, {"text": "Type"}, {"text": tcol + (" this work" if exp else "")}] + expcols +
                  [{"text": f"{xu}, 1"}, {"text": f"{xu}, 2"}, {"text": f"{xu}, 3"}]]
    notes = []
    if st["footnotes"]:
        notes.append("Phases 1–3 in the order written in the reaction. Calculated with pycalphad 0.11.2 from the SGTE "
                     "dataset; synthetic document.")
    if marks and any(r["type"].startswith(("congruent", "melting")) for r in logical):
        notes.append(f"{mark[0]} Two-phase equilibrium at a temperature extremum; both phases have the same composition.")
    if exp:
        notes.append((f"{mark[1]} " if marks else "") + "Experimental temperatures are fictitious values attributed to "
                     "invented references [3–5] for testing evidence-class extraction; they are not data.")
        if marks:
            for hrow in header:
                for h in hrow:
                    if h["text"] in ("T exp.", "exp."):
                        h["text"] += mark[1]
    ncol = len(rows[0]) if rows else 6
    units = [None, None, tunit] + ([tunit] if exp else []) + ([None] if exp and not inline else []) + [xunit] * 3
    align = ["l", "l"] + ["r"] * (ncol - 2)
    return {"id": "invariants", "header": header, "rows": rows, "align": align, "footnotes": notes,
            "column_units": units, "meta": meta}, sem


# ------------------------------------------------------------------------------------------------ phase table
def phase_table(data, ext, st):
    B = data["elements"][1]
    tunit, xunit = st["T_unit"], st["x_unit"]
    marks = st["markers"]
    stab = sorted(ext["phase_stability"], key=lambda p: (0.5 * (p["homogeneity"]["x_min"] + p["homogeneity"]["x_max"]), p["label"]))
    rows, meta, sem = [], [], []
    low_mark = mark[0] if marks else ""
    cong_mark = mark[1] if marks else ""
    used = set()
    for i, p in enumerate(stab):
        lo = "< " + fmt(tval(300.0, tunit), 0) + low_mark if p["T_low_K"] is None else fmt(tval(p["T_low_K"], tunit), 1)
        if p["T_low_K"] is None and marks:
            used.add("low")
        hi = fmt(tval(p["T_high_K"], tunit), 1) if p["T_high_K"] is not None else "–"
        if p["T_high_kind"] and p["T_high_kind"].startswith("congruent") and marks:
            hi += cong_mark
            used.add("cong")
        h = p["homogeneity"]
        a, b = xprint(h["x_min"], xunit), xprint(h["x_max"], xunit)
        hr = a if a == b else f"{a}–{b}"
        Th = fmt(tval(h["T_K"], tunit), 0)
        sb = p["strukturbericht"] or "–"
        row = [p["label"], p["pearson"] or "–", sb, p["prototype"] or "–", lo, hi, hr, Th]
        rows.append(row)
        meta.append([_m("phase", value=p["label"], tdb_phase=p["tdb_phases"]),
                     _m("pearson", value=p["pearson"]), _m("strukturbericht", value=p["strukturbericht_ascii"]),
                     _m("prototype", value=p["prototype"]),
                     _m("T_low", exact_SI=p["T_low_K"], unit=tunit, bound="below 300 K" if p["T_low_K"] is None else None),
                     _m("T_high", exact_SI=p["T_high_K"], unit=tunit, kind=p["T_high_kind"]),
                     _m("homogeneity", exact_SI=[h["x_min"], h["x_max"]], unit=xunit),
                     _m("homogeneity_T", exact_SI=h["T_K"], unit=tunit)])
        sem.append({"row": i, "phase": p["label"], "tdb_phases": p["tdb_phases"], "pearson": p["pearson"],
                    "strukturbericht": p["strukturbericht_ascii"], "strukturbericht_printed": p["strukturbericht"],
                    "prototype": p["prototype"], "structure_derivation": p["derivation"],
                    "T_low_K": p["T_low_K"], "T_low_printed": lo, "T_low_kind": p["T_low_kind"],
                    "T_low_value": None if p["T_low_K"] is None else float(lo),
                    "T_high_K": p["T_high_K"], "T_high_printed": hi, "T_high_kind": p["T_high_kind"], "T_unit": tunit,
                    "T_high_value": None if p["T_high_K"] is None else float(hi.rstrip("".join(mark))),
                    "homogeneity": {"T_K": h["T_K"], "T_printed": Th, "T_value": float(Th), "x_min": h["x_min"], "x_max": h["x_max"],
                                    "x_min_value": float(a), "x_max_value": float(b), "printed": hr, "x_unit": xunit}})
    xu = xu_label(xunit, B)
    if st["merged_header"]:
        header = [[{"text": ""}, {"text": "Structure", "colspan": 3}, {"text": f"Stability range / {tunit}", "colspan": 2},
                   {"text": "Homogeneity range", "colspan": 2}],
                  [{"text": "Phase"}, {"text": "Pearson"}, {"text": "Strukturb."}, {"text": "Prototype"}, {"text": "from"},
                   {"text": "to"}, {"text": xu}, {"text": f"at T / {tunit}"}]]
    else:
        header = [[{"text": "Phase"}, {"text": "Pearson"}, {"text": "SB"}, {"text": "Prototype"}, {"text": f"T min ({tunit})"},
                   {"text": f"T max ({tunit})"}, {"text": f"Range ({xu})"}, {"text": f"T ({tunit})"}]]
    notes = []
    if st["footnotes"]:
        notes.append("Structure labels follow the TDB phase names (– where the name does not identify a structure). "
                     "Stability range: temperatures between which the phase is stable in some part of the diagram.")
    if "low" in used:
        notes.append(f"{mark[0]} Stable at the lowest calculated temperature.")
    if "cong" in used:
        notes.append(f"{mark[1]} Congruent point (melting or transformation).")
    return {"id": "phases", "header": header, "rows": rows, "align": ["l", "l", "l", "l", "r", "r", "r", "r"],
            "footnotes": notes, "column_units": [None, None, None, None, tunit, tunit, xunit, tunit], "meta": meta}, sem


# ------------------------------------------------------------------------------------------------ formation table
def e_conv(J, unit):
    return {"J/mol": J, "kJ/mol": J / 1000.0, "cal/mol": J / cal, "kcal/mol": J / cal / 1000.0}[unit]


def formation_table(data, ext, st, rng, minus):
    B = data["elements"][1]
    eu, basis = st["energy_unit"], st["basis"]
    su = "cal/(mol K)" if eu.startswith("cal") or eu.startswith("kcal") else "J/(mol K)"
    nd = 2 if eu == "kJ/mol" else 0
    rows, meta, recs = [], [], []
    marks = st["markers"]
    used = False
    for i, c in enumerate(ext["formation"]["rows"]):
        n = c["formula_atoms"] if basis == "formula_unit" else 1
        H = c["dfH_J_per_mol_atoms"] * n
        S = c["dfS_J_per_mol_atoms_K"] * n
        Cp = c["Cp_J_per_mol_atoms_K"] * n if c["Cp_J_per_mol_atoms_K"] is not None else None
        hs = fmt(e_conv(H, eu), nd)
        unc = None
        if st["uncertainty"]:
            unc = [0.5, 1.0, 1.5, 2.0][int(rng.integers(4))] * (1000.0 if eu == "J/mol" else (1000 / cal if eu == "cal/mol" else 1.0))
            hs = f"{hs} ± {fmt(unc, 1 if eu == 'kJ/mol' else 0)}"
        sconv = (lambda v: v / cal) if su.startswith("cal") else (lambda v: v)
        ss = fmt(sconv(S), 2)
        cs = fmt(sconv(Cp), 2) if Cp is not None else "–"
        lab = c["label"]
        if marks and not c["stable_at_298K"]:
            lab += mark[0]
            used = True
        row = [lab, fmt(100 * c["x"], 1), hs.replace("-", minus), ss.replace("-", minus), cs.replace("-", minus)]
        rows.append(row)
        meta.append([_m("phase", value=c["label"], tdb_phase=c["phase"]), _m("x", exact_SI=c["x"], unit="at.%"),
                     _m("dfH", exact_SI=c["dfH_J_per_mol_atoms"], unit=eu, basis=basis, atoms_per_formula_unit=n),
                     _m("dfS", exact_SI=c["dfS_J_per_mol_atoms_K"], unit=su, basis=basis, atoms_per_formula_unit=n),
                     _m("Cp", exact_SI=c["Cp_J_per_mol_atoms_K"], unit=su, basis=basis, atoms_per_formula_unit=n,
                        note=c["Cp_note"])])
        for q, exact, s_, u_ in (("dfH", c["dfH_J_per_mol_atoms"], hs, eu), ("dfS", c["dfS_J_per_mol_atoms_K"], ss, su),
                                 ("Cp", c["Cp_J_per_mol_atoms_K"], cs, su)):
            if exact is None:
                continue
            from document import parse_number
            v, u = parse_number(s_)
            recs.append({"table": "formation", "phase": c["label"], "tdb_phase": c["phase"], "x": c["x"], "quantity": q,
                         "printed": s_.replace("-", minus), "value": v, "uncertainty": u, "unit": u_, "basis": basis,
                         "atoms_per_formula_unit": n, "exact_SI_per_mol_atoms": exact, "T_K": 298.15,
                         "stable_at_298K": c["stable_at_298K"]})
    per = "per mole of atoms" if basis == "mol_atoms" else "per mole of formula units"
    ue = {"kJ/mol": "kJ mol⁻¹", "J/mol": "J mol⁻¹", "cal/mol": "cal mol⁻¹"}[eu]
    us = "cal mol⁻¹ K⁻¹" if su.startswith("cal") else "J mol⁻¹ K⁻¹"
    if st["merged_header"]:
        header = [[{"text": ""}, {"text": ""}, {"text": f"298.15 K, {per}", "colspan": 3}],
                  [{"text": "Phase"}, {"text": f"at.% {B}"}, {"text": f"ΔfH / {ue}"}, {"text": f"ΔfS / {us}"}, {"text": f"Cp / {us}"}]]
    else:
        fu = " (f.u.)" if basis == "formula_unit" else ""
        header = [[{"text": "Phase"}, {"text": f"at.% {B}"}, {"text": f"ΔfH(298.15 K) / {ue}{fu}"},
                   {"text": f"ΔfS / {us}{fu}"}, {"text": f"Cp(298.15 K) / {us}{fu}"}]]
    notes = []
    if st["footnotes"] or basis == "formula_unit" or eu == "cal/mol":
        n_ = f"Reference: stable elements (SER) at 298.15 K; values {per}"
        if basis == "formula_unit":
            n_ += " (formula unit as written in the Phase column)"
        if eu == "cal/mol":
            n_ += "; 1 cal = 4.184 J (thermochemical calorie)"
        n_ += "."
        if st["uncertainty"]:
            n_ += " ± values are synthetic illustration uncertainties, not data."
        notes.append(n_)
    if used:
        notes.append(f"{mark[0]} Not stable at 298.15 K in this description; properties of the metastable phase.")
    return {"id": "formation", "header": header, "rows": rows, "align": ["l", "r", "r", "r", "r"], "footnotes": notes,
            "column_units": [None, "at.%", eu, su, su], "meta": meta, "basis": basis}, recs


# ------------------------------------------------------------------------------------------------ mixing table
def mixing_table(data, ext, st, minus):
    A, B = data["elements"]
    M = ext["mixing_table"]
    eu = st["energy_unit"]
    nd = 2 if eu == "kJ/mol" else 0
    xunit = st["x_unit"]
    xs = [r for r in M["rows"] if (st["n_rows"] == 9 or round(r["x"] * 10) % 2 == 1)]
    part = st["partials"]
    rows, meta, recs = [], [], []

    def act(a):
        return f"{a:.4f}" if a >= 0.001 else f"{a:.3e}".replace("e-0", "E-").replace("e-", "E-")
    for i, r in enumerate(xs):
        xs_ = xprint(r["x"], xunit) if xunit == "mole_fraction" else fmt(100 * r["x"], 0)
        vals = [("HM_MIX", r["HM_MIX"], fmt(e_conv(r["HM_MIX"], eu), nd), eu), ("GM_MIX", r["GM_MIX"], fmt(e_conv(r["GM_MIX"], eu), nd), eu),
                ("a_A", r["a_A"], act(r["a_A"]), "1"), ("a_B", r["a_B"], act(r["a_B"]), "1")]
        if part:
            vals += [("HM_PART_A", r["HM_PART_A"], fmt(e_conv(r["HM_PART_A"], eu), nd), eu),
                     ("HM_PART_B", r["HM_PART_B"], fmt(e_conv(r["HM_PART_B"], eu), nd), eu)]
        row = [xs_] + [s.replace("-", minus) for _, _, s, _ in vals]
        rows.append(row)
        meta.append([_m("x", exact_SI=r["x"], unit=xunit)] + [_m(q, exact_SI=e, unit=u, basis="mol_atoms") for q, e, _, u in vals])
        from document import parse_number
        for q, e, s, u in vals:
            v, _ = parse_number(s)
            recs.append({"table": "mixing", "x": r["x"], "x_printed": xs_, "x_unit": xunit, "quantity": q, "printed": s.replace("-", minus),
                         "value": v, "unit": u, "basis": "mol_atoms", "exact_SI_per_mol_atoms": e, "T_K": M["T"]})
    ue = {"kJ/mol": "kJ mol⁻¹", "J/mol": "J mol⁻¹"}[eu]
    xh = f"at.% {B}" if xunit == "at.%" else f"x({B})"
    q = [f"ΔmixH / {ue}", f"ΔmixG / {ue}", f"a({A})", f"a({B})"] + ([f"ΔH̄({A}) / {ue}", f"ΔH̄({B}) / {ue}"] if part else [])
    if st["merged_header"]:
        header = [[{"text": ""}, {"text": f"Liquid at {M['T']:.0f} K", "colspan": len(q)}],
                  [{"text": xh}] + [{"text": t} for t in q]]
    else:
        header = [[{"text": xh}] + [{"text": t} for t in q]]
    notes = []
    if st["footnotes"] or part:
        notes.append(f"Reference states: pure liquid {A} and {B} at {M['T']:.0f} K; per mole of atoms."
                     + (" ΔH̄: partial molar enthalpies of mixing." if part else ""))
    return {"id": "mixing", "header": header, "rows": rows, "align": ["r"] * (1 + len(q)), "footnotes": notes,
            "column_units": [xunit] + [u for _, _, _, u in vals], "meta": meta, "T_K": M["T"]}, recs
