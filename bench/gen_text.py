"""Running-text sentences and table rows for the synthetic documents, split from generate.py."""

import numpy as np
from common import docs_dir, truth_dir, fmt, sha256_file, write_json  # noqa: E402
from systems import system_specs, base  # noqa: E402


type_article = {
    "eutectic": "a",
    "eutectoid": "a",
    "peritectic": "a",
    "peritectoid": "a",
    "monotectic": "a",
    "syntectic": "a",
    "metatectic": "a",
    "monotectoid": "a",
    "unknown": "an",
}


def invariant_sentences(rng, data, vals, tunit, xunit, ndT):
    """Build running-text sentences that describe the invariant reactions."""
    out = []
    for i, inv in enumerate(data["invariants"]):
        T, ref = inv["T"], f"invariant:{i}"
        pts = inv["points"]
        mid = pts[1]
        liq = [p for p in pts if base(p["phase"]) == "LIQUID"]
        key = liq[0] if (liq and inv["type"] in ("eutectic", "peritectic", "monotectic", "metatectic")) else mid
        form = int(rng.integers(5))
        ts, ti = vals.T(T, tunit, ndT, "invariant_temperature", ref)
        xs, xi = vals.X(key["x"], xunit, f"invariant_composition:{key['label']}", ref, style=int(rng.integers(2)))
        r_ = inv["reaction"]
        typ = inv["type"]
        if form == 0:
            s = f"The {typ} reaction {r_} is calculated at {ts}, with {key['label']} at {xs}."
        elif form == 1:
            s = f"At {ts} the {typ} {r_} takes place; the {key['label']} phase then contains {xs}."
        elif form == 2:
            s = f"We obtain {ts} for the {typ} reaction ({r_}), where {key['label']} has {xs}."
        elif form == 3:
            s = f"{typ.capitalize()} equilibrium {r_} is found at {ts} and {xs} ({key['label']})."
        else:
            s = f"The dataset places {type_article.get(typ, 'a')} {typ} ({r_}) at {ts}; the composition of {key['label']} is {xs}."
        out.append((s, [ti, xi]))
    return out


def tm_sentences(rng, data, vals, tunit, ndT):
    """Build running-text sentences that give the melting temperatures of the pure components."""
    out = []
    for side, Tm in (("A", data["Tm_A"]), ("B", data["Tm_B"])):
        if Tm is None:
            continue
        e = data["elements"][0 if side == "A" else 1]
        ts, ti = vals.T(Tm, tunit, ndT, f"melting_point:{e}", f"melting:{e}")
        s = [
            f"Pure {e} melts at {ts}.",
            f"The melting point of {e} reproduced by the dataset is {ts}.",
            f"For {e}, the liquidus meets the axis at {ts}.",
        ][int(rng.integers(3))]
        out.append((s, [ti]))
    return out


def invariant_table(data, style, xunit, tunit):
    """Build the invariant-reaction table for a synthetic document."""
    B = data["elements"][1]
    rows = []
    cvals = []
    for inv in data["invariants"]:
        T = inv["T"] - 273.15 if tunit == "°C" else inv["T"]
        order = inv["reactants"] + inv["products"]
        pmap = {p["phase"]: p for p in inv["points"]}
        comps = [pmap[p]["x"] for p in order]
        cfmt = [(fmt(100 * x, 1) if xunit == "at.%" else fmt(x, 3)) for x in comps]
        rows.append([inv["reaction"], inv["type"], fmt(T, 1)] + cfmt)
        cvals.append(comps)
    xu = f"at.% {B}" if xunit == "at.%" else f"x({B})"
    if style["merged_header"]:
        header = [
            [{"text": "Reaction"}, {"text": "Type"}, {"text": f"T / {tunit}"}, {"text": f"Compositions ({xu})", "colspan": 3}],
            [{"text": ""}, {"text": ""}, {"text": ""}, {"text": "phase 1"}, {"text": "phase 2"}, {"text": "phase 3"}],
        ]
    else:
        header = [
            [
                {"text": "Reaction"},
                {"text": "Type"},
                {"text": f"T ({tunit})"},
                {"text": f"{xu}, 1"},
                {"text": f"{xu}, 2"},
                {"text": f"{xu}, 3"},
            ]
        ]
    notes = []
    if style["footnotes"]:
        notes.append(
            "Phases 1–3 in the order written in the reaction. Calculated with pycalphad 0.11.2 from the "
            "SGTE dataset; synthetic document."
        )
    units = [None, None, tunit, xunit, xunit, xunit]
    return {
        "id": "invariants",
        "header": header,
        "rows": rows,
        "align": ["l", "l", "r", "r", "r", "r"],
        "footnotes": notes,
        "column_units": units,
    }


def liquidus_table(data, style, xunit, tunit):
    """For systems without invariants: liquidus/solidus temperatures at selected compositions."""
    B = data["elements"][1]
    rows = []
    for x0 in (0.2, 0.4, 0.6, 0.8):
        Tl = Ts = None
        for rg in data["boundaries"]:
            phs = [base(c["phase"]) for c in rg["curves"]]
            if "LIQUID" not in phs or len(set(phs)) != 2:
                continue
            for c in rg["curves"]:
                x, T = np.asarray(c["x"]), np.asarray(c["T"])
                for k in range(len(x) - 1):
                    if (x[k] - x0) * (x[k + 1] - x0) <= 0 and x[k] != x[k + 1]:
                        Tv = T[k] + (T[k + 1] - T[k]) * (x0 - x[k]) / (x[k + 1] - x[k])
                        if base(c["phase"]) == "LIQUID":
                            Tl = Tv if Tl is None else max(Tl, Tv)
                        else:
                            Ts = Tv if Ts is None else max(Ts, Tv)
        if Tl is None or Ts is None:
            continue
        cv = lambda T: T - 273.15 if tunit == "°C" else T  # noqa: E731
        rows.append([fmt(100 * x0, 0) if xunit == "at.%" else fmt(x0, 2), fmt(cv(Tl), 1), fmt(cv(Ts), 1)])
    xu = f"at.% {B}" if xunit == "at.%" else f"x({B})"
    return {
        "id": "liquidus",
        "header": [[{"text": xu}, {"text": f"T liquidus / {tunit}"}, {"text": f"T solidus / {tunit}"}]],
        "rows": rows,
        "align": ["r", "r", "r"],
        "footnotes": ["Linear interpolation of the calculated boundaries."] if style["footnotes"] else [],
        "column_units": [xunit, tunit, tunit],
    }


def thermo_table(data, style, rng, minus):
    """Build the thermodynamic-property table for a synthetic document."""
    P = data["properties"]
    B = data["elements"][1]
    if P.get("formation"):
        rows = []
        for c in P["formation"]:
            H = c["HM_298"] / 1000
            S = c["SM_298"]
            hs = fmt(H, 2)
            if style["uncertainty"]:
                u = [0.5, 1.0, 1.5, 2.0][int(rng.integers(4))]
                hs = f"{hs} ± {u:.1f}"
            rows.append([c["label"], fmt(100 * c["x"], 1), hs.replace("-", minus), fmt(S, 2).replace("-", minus)])
        header = [
            [{"text": "Phase"}, {"text": f"at.% {B}"}, {"text": "ΔfH(298 K) / kJ mol⁻¹"}, {"text": "S(298 K) / J mol⁻¹ K⁻¹"}]
        ]
        if style["merged_header"]:
            header = [
                [{"text": ""}, {"text": ""}, {"text": "Calculated at 298.15 K, per mole of atoms", "colspan": 2}],
                [{"text": "Phase"}, {"text": f"at.% {B}"}, {"text": "ΔfH / kJ mol⁻¹"}, {"text": "S / J mol⁻¹ K⁻¹"}],
            ]
        notes = []
        if style["footnotes"]:
            notes.append(
                "Reference: stable elements (SER) at 298.15 K."
                + (" ± values are synthetic illustration uncertainties, not data." if style["uncertainty"] else "")
            )
        return {
            "id": "formation",
            "header": header,
            "rows": rows,
            "align": ["l", "r", "r", "r"],
            "footnotes": notes,
            "column_units": [None, "at.%", "kJ/mol", "J/(mol K)"],
        }
    rk = P.get("rk_liquid") or []
    if rk:
        rows = []
        for q in rk:
            rows.append(
                [
                    f"L{q['order']}",
                    repr(float(q["a"])).replace("-", minus),
                    repr(float(q["b"])).replace("-", minus),
                    repr(float(q["c"])).replace("-", minus),
                ]
            )
        header = [[{"text": "ν"}, {"text": "a / J mol⁻¹"}, {"text": "b / J mol⁻¹ K⁻¹"}, {"text": "c / J mol⁻¹ K⁻¹"}]]
        if style["merged_header"]:
            header = [
                [{"text": ""}, {"text": "Liquid: L(ν) = a + b·T + c·T·ln T", "colspan": 3}],
                [{"text": "ν"}, {"text": "a"}, {"text": "b"}, {"text": "c"}],
            ]
        return {
            "id": "rk_liquid",
            "header": header,
            "rows": rows,
            "align": ["l", "r", "r", "r"],
            "footnotes": ["Values as printed in the TDB file."] if style["footnotes"] else [],
            "column_units": [None, "J/mol", "J/(mol K)", "J/(mol K)"],
        }
    return None
