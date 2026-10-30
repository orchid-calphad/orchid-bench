"""Random presentation styles (layout, fonts, colours) for the synthetic documents, split from generate.py."""


def choose_style(rng, data):
    """Draw a random presentation style for a document from rng."""
    ch = lambda seq: seq[int(rng.integers(len(seq)))]  # noqa: E731
    pd = {
        "T_unit": ch(["K", "°C"]),
        "x_unit": ch(["at.%", "mole_fraction"]),
        "linewidth": float(ch([0.6, 0.8, 1.0, 1.3, 1.6])),
        "color": bool(rng.random() < 0.45),
        "field_labels": bool(rng.random() < 0.7),
        "label_names": ch(["conventional"] * 4 + ["tdb"]),
        "label_fontsize": float(ch([6.5, 7.0, 7.5])),
        "label_bg": bool(rng.random() < 0.5),
        "invariant_T_labels": bool(rng.random() < 0.5),
        "inset": bool(rng.random() < 0.22),
        "markers": bool(rng.random() < 0.45),
        "overlay": bool(rng.random() < 0.35),
        "fig_size": ch([(4.6, 3.6), (5.0, 3.8), (4.2, 3.6)]),
    }
    pd["overlay_offset"] = (float(ch([0.01, -0.01, 0.015])), float(ch([15.0, -12.0, 20.0])))
    pd["legend"] = pd["markers"] or pd["overlay"]
    pd["legend_loc"] = ch(["upper center", "lower center", "upper left", "upper right"])
    if pd["inset"]:
        pd["legend_loc"] = "lower center"
    prop = {
        "x_unit": pd["x_unit"] if rng.random() < 0.7 else ch(["at.%", "mole_fraction"]),
        "color": bool(rng.random() < 0.5),
        "linewidth": float(ch([0.8, 1.0, 1.4])),
        "representation": ch(["line+markers", "line", "markers"]),
        "energy_unit": ch(["kJ/mol", "J/mol"]),
        "two_series": bool(rng.random() < 0.5),
        "log_y": bool(rng.random() < 0.35),
        "raoult": bool(rng.random() < 0.5),
        "fig_size": ch([(3.6, 2.8), (4.0, 3.0)]),
    }
    tab = {
        "rules": ch(["booktabs", "grid", "plain"]),
        "merged_header": bool(rng.random() < 0.5),
        "footnotes": bool(rng.random() < 0.6),
        "uncertainty": bool(rng.random() < 0.5),
        "T_unit": pd["T_unit"] if rng.random() < 0.6 else ch(["K", "°C"]),
        "x_unit": pd["x_unit"] if rng.random() < 0.6 else ch(["at.%", "mole_fraction"]),
    }
    text = {"T_unit": ch(["K", "°C"]), "x_unit": ch(["at.%", "mole_fraction"]), "ndT": int(ch([0, 1, 1]))}
    page = {
        "size": ch([(595.28, 841.89), (612.0, 792.0)]),
        "margins": ch([(62, 60, 62, 62), (72, 66, 72, 70)]),
        "body_size": float(ch([9.5, 10.0, 10.5])),
    }
    kinds = []
    P = data["properties"]
    for k in ("mixing", "activity", "cp", "gibbs", "two_panel"):
        need = {"two_panel": "mixing"}.get(k, k)
        if P.get(need):
            kinds.append(k)
    rng.shuffle(kinds)
    # prefer one of mixing/two_panel at most
    sel = []
    for k in kinds:
        if k in ("mixing", "two_panel") and any(s in ("mixing", "two_panel") for s in sel):
            continue
        sel.append(k)
        if len(sel) == 2:
            break
    return {"phase_diagram": pd, "property": prop, "table": tab, "text": text, "page": page, "property_kinds": sel}


def choose_style_v2(rng, data, ext, style):
    """Draw a presentation style for a v2 document, extending the style with ext."""
    ch = lambda seq: seq[int(rng.integers(len(seq)))]  # noqa: E731
    tb = style["table"]

    def tstyle(**kw):
        d = {
            "rules": ch(["booktabs", "grid", "plain"]),
            "merged_header": bool(rng.random() < 0.5),
            "footnotes": bool(rng.random() < 0.6),
            "markers": bool(rng.random() < 0.5),
        }
        d.update(kw)
        return d

    inv = tstyle(
        T_unit=tb["T_unit"], x_unit=tb["x_unit"], exp_column=bool(rng.random() < 0.55), exp_ref_style=ch(["inline", "column"])
    )
    inv["rules"] = tb["rules"]
    inv["merged_header"] = tb["merged_header"]
    ph = tstyle(T_unit=ch(["K", "K", "°C"]), x_unit=ch(["at.%", "mole_fraction"]))
    rows = ext["formation"]["rows"]
    fu_ok = bool(rows) and all(r["formula_atoms"] for r in rows)
    form = tstyle(
        energy_unit=ch(["kJ/mol"] * 6 + ["J/mol"] * 3 + ["cal/mol"]),
        basis="formula_unit" if fu_ok and rng.random() < 0.7 else "mol_atoms",
        uncertainty=tb["uncertainty"],
    )
    mix = tstyle(
        energy_unit=ch(["kJ/mol", "J/mol"]),
        x_unit=ch(["at.%", "mole_fraction"]),
        partials=bool(rng.random() < 0.45),
        n_rows=ch([9, 9, 5]),
    )
    split = None
    if rng.random() < 0.2:
        split = ch(["phases", "mixing"]) if len(ext["phase_stability"]) >= 5 else "mixing"
    for d_, k in ((ph, "phases"), (mix, "mixing")):
        d_["split"] = split == k
    pf = {
        "T_unit": ch(["K", "°C"]),
        "x_unit": style["property"]["x_unit"],
        "color": style["property"]["color"],
        "linewidth": style["property"]["linewidth"],
        "fig_size": (4.0, 0),
        "n_alloys": ch([1, 2]),
        "alloy_choice": [int(v) for v in rng.permutation(3)],
    }
    text = {
        "cal_unit": form["energy_unit"] == "cal/mol" or rng.random() < 0.06,
        "energy_units": [ch(["kJ/mol", "J/mol", "kJ/mol-atom"]) for _ in range(3)],
    }
    return {
        "invariant_table": inv,
        "phase_table": ph,
        "formation_table": form,
        "mixing_table": mix,
        "phase_fraction": pf,
        "text_v2": text,
    }
