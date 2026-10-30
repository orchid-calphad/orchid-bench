"""Truth-schema v2 additions, computed with pycalphad from the recorded TDB and cached in bench/cache/<stem>.ext.json:

* phase stability: per printed phase label, the temperature range over which it is stable anywhere in the
  diagram (limits snapped to the exact invariant / congruent / melting temperature when one lies within the grid
  step, otherwise bisected to 0.01 K at a fixed composition), and the homogeneity range at a stated grid T;
* phase fractions vs temperature (equilibrium step) for three candidate alloys;
* formation properties of the intermediate phases at 298.15 K (enthalpy, entropy of formation, Cp);
* liquid mixing properties at one temperature (integral H and G, activities, partial molar enthalpies);
* crystal-structure labels derivable from the TDB phase name (small explicit mapping).
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cache, p_atm, tdb_dir, load_json, write_json  # noqa: E402
from systems import Labeller, base  # noqa: E402

warnings.filterwarnings("ignore")
ext_version = 6
cal = 4.184  # J per thermochemical calorie

# Strukturbericht token -> (Pearson symbol, Strukturbericht as printed, ascii form, prototype)
sb_table = {
    "A1": ("cF4", "A1", "A1", "Cu"), "A2": ("cI2", "A2", "A2", "W"), "A3": ("hP2", "A3", "A3", "Mg"),
    "A4": ("cF8", "A4", "A4", "C (diamond)"), "A5": ("tI4", "A5", "A5", "β-Sn"), "A7": ("hR2", "A7", "A7", "α-As"),
    "A12": ("cI58", "A12", "A12", "α-Mn"), "B2": ("cP2", "B2", "B2", "CsCl"), "L12": ("cP4", "L1₂", "L1_2", "Cu3Au"),
    "C1": ("cF12", "C1", "C1", "CaF2"), "C14": ("hP12", "C14", "C14", "MgZn2"), "C15": ("cF24", "C15", "C15", "MgCu2"),
    "C36": ("hP24", "C36", "C36", "MgNi2"), "CB": ("oF48", "Cb", "Cb", "Mg2Cu"), "D0A": ("oP8", "D0a", "D0_a", "β-Cu3Ti"),
    "D011": ("oP16", "D0₁₁", "D0_11", "Fe3C"), "D019": ("hP8", "D0₁₉", "D0_19", "Ni3Sn"),
    "D024": ("hP16", "D0₂₄", "D0_24", "Ni3Ti"), "D513": ("hP5", "D5₁₃", "D5_13", "Ni2Al3"),
    "D82": ("cI52", "D8₂", "D8_2", "Cu5Zn8"), "D8B": ("tP30", "D8b", "D8_b", "σ-CrFe"),
}
lattice_word = {"FCC": "A1", "BCC": "A2", "HCP": "A3", "DIAMOND": "A4", "BCT": "A5", "RHOMBOHEDRAL": "A7"}


def structure_of(phase: str) -> dict:
    """Structure labels derivable from the TDB phase name: an explicit Strukturbericht token (e.g. *_D513, L12_FCC,
    C15_LAVES) wins; otherwise a lattice word (FCC/BCC/HCP/...) when it is the only token besides an element tag.
    """
    toks = base(phase).upper().split("_")
    for t in toks:
        if t in sb_table:
            p, sb, sba, proto = sb_table[t]
            return {"pearson": p, "strukturbericht": sb, "strukturbericht_ascii": sba, "prototype": proto,
                    "derivation": f"Strukturbericht token '{t}' in the TDB phase name"}
    for t in toks:
        if t in lattice_word:
            p, sb, sba, proto = sb_table[lattice_word[t]]
            return {"pearson": p, "strukturbericht": sb, "strukturbericht_ascii": sba, "prototype": proto,
                    "derivation": f"lattice word '{t}' in the TDB phase name"}
    return {"pearson": None, "strukturbericht": None, "strukturbericht_ascii": None, "prototype": None,
            "derivation": "not derivable from the TDB phase name (left blank in the document)"}


def formula_atoms(label: str, elements) -> tuple[int, float] | None:
    """'Al3Ni2' -> (5, x_B = 2/5) using the system's element order; None when the label is not a formula."""
    A, B = elements
    m = re.fullmatch(r"((?:[A-Z][a-z]?\d*)+)", label or "")
    if not m:
        return None
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", label)
    cnt = {}
    for s, n in parts:
        if s not in (A, B):
            return None
        cnt[s] = cnt.get(s, 0) + (int(n) if n else 1)
    if len(cnt) != 2:
        return None
    tot = cnt[A] + cnt[B]
    return tot, cnt[B] / tot


def _pyc():
    from pycalphad import Database, equilibrium
    from pycalphad import variables as v
    return Database, equilibrium, v


def _sets(ph, xb, npf, Y, ctx):
    from pd_data import disorder_name
    cs = []
    for k in range(len(ph)):
        n = str(ph[k])
        if n and np.isfinite(npf[k]) and npf[k] > 1e-6:
            cs.append({"phase": disorder_name(*ctx, n, Y[k]), "x": float(xb[k]), "np": float(npf[k])})
    cs.sort(key=lambda c: c["x"])
    seen = {}
    for c in cs:
        seen[c["phase"]] = seen.get(c["phase"], 0) + 1
        if seen[c["phase"]] > 1:
            c["phase"] = f"{c['phase']}#{seen[c['phase']]}"
    return cs


def grid_sets(dbf, comps, phases, B, fm):
    _, equilibrium, v = _pyc()
    xgrid = np.round(np.arange(fm["nx"]) * fm["dx"], 6)
    xcalc = xgrid.copy()
    xcalc[0], xcalc[-1] = 1e-5, 1 - 1e-5
    Tgrid = fm["T0"] + fm["dT"] * np.arange(fm["nT"])
    out = [[None] * fm["nx"] for _ in range(fm["nT"])]
    for i0 in range(0, fm["nT"], 40):
        Ts = Tgrid[i0:i0 + 40]
        res = equilibrium(dbf, comps, phases, {v.X(B): xcalc.tolist(), v.T: Ts.tolist(), v.P: p_atm, v.N: 1})
        ph = res.Phase.values[0, 0]
        xb = res.X.sel(component=B).values[0, 0]
        npf = res.NP.values[0, 0]
        Y = res.Y.values[0, 0]
        for a in range(len(Ts)):
            for j in range(fm["nx"]):
                out[i0 + a][j] = _sets(ph[a, j], xb[a, j], npf[a, j], Y[a, j], (dbf, comps, phases))
    return xcalc, Tgrid, out


def _sets_at(dbf, comps, phases, B, x, T):
    _, equilibrium, v = _pyc()
    res = equilibrium(dbf, comps, phases, {v.X(B): float(x), v.T: float(T), v.P: p_atm, v.N: 1})
    ph = np.atleast_1d(res.Phase.values.squeeze())
    return _sets(ph, np.atleast_1d(res.X.sel(component=B).values.squeeze()), np.atleast_1d(res.NP.values.squeeze()),
                 res.Y.values.reshape(len(ph), -1), (dbf, comps, phases))


def present(dbf, comps, phases, B, x, T, lab, labeller):
    _, equilibrium, v = _pyc()
    res = equilibrium(dbf, comps, phases, {v.X(B): float(x), v.T: float(T), v.P: p_atm, v.N: 1})
    ph = np.atleast_1d(res.Phase.values.squeeze())
    cs = _sets(ph, np.atleast_1d(res.X.sel(component=B).values.squeeze()), np.atleast_1d(res.NP.values.squeeze()),
               res.Y.values.reshape(len(ph), -1), (dbf, comps, phases))
    return any(base(c["phase"]) != "LIQUID" and labeller.label(c["phase"], c["x"]) == lab for c in cs)


def bisect_T(dbf, comps, phases, B, x, T_in, T_out, lab, labeller, tol=0.01):
    """T_in: phase present at x; T_out: absent. Returns the transition T (to tol)."""
    a, b = float(T_in), float(T_out)
    while abs(b - a) > tol:
        m = 0.5 * (a + b)
        if present(dbf, comps, phases, B, x, m, lab, labeller):
            a = m
        else:
            b = m
    return round(0.5 * (a + b), 3)


def stability(data, dbf, comps, phases, labeller, xcalc, Tgrid, sets):
    A, B = data["components"]
    Ael, Bel = data["elements"]
    dT = float(Tgrid[1] - Tgrid[0])
    rows = {}
    for i in range(len(Tgrid)):
        for j in range(len(xcalc)):
            for c in sets[i][j]:
                if base(c["phase"]) == "LIQUID":
                    continue
                lab = labeller.label(c["phase"], c["x"])
                r_ = rows.setdefault(lab, {"tdb": set(), "rows": {}})
                r_["tdb"].add(base(c["phase"]))
                lo, hi, xs_at = r_["rows"].get(i, (1.0, 0.0, []))
                r_["rows"][i] = (min(lo, c["x"]), max(hi, c["x"]), xs_at + [float(xcalc[j])])
    out = []
    for lab, r_ in sorted(rows.items()):
        idx = sorted(r_["rows"])
        i_lo, i_hi = idx[0], idx[-1]
        # candidates: invariants with this label, congruent points, element melting points
        cand = []
        for k, inv in enumerate(data["invariants"]):
            if any(labeller.label(p["phase"], p["x"]) == lab for p in inv["points"]):
                cand.append((inv["T"], f"invariant:{k}", inv["type"]))
        term_A = any(any(xx < 1e-3 for xx in r_["rows"][i][2]) for i in idx)
        term_B = any(any(xx > 1 - 1e-3 for xx in r_["rows"][i][2]) for i in idx)
        if term_A and data["Tm_A"]:
            cand.append((data["Tm_A"], f"melting:{Ael}", "melting of the pure element"))
        if term_B and data["Tm_B"]:
            cand.append((data["Tm_B"], f"melting:{Bel}", "melting of the pure element"))

        def limit(i_edge, upward):
            """Extremum over composition of the temperature at which the phase disappears (bisection to 0.01 K,
            golden-section in x), snapped to an invariant / element melting point within 0.05 K.
            """
            Tg = float(Tgrid[i_edge])
            if upward and i_edge == len(Tgrid) - 1:
                return None, "above_calculated_range", None, None
            if not upward and i_edge == 0:
                return None, "below_calculated_range", None, None
            xs_at = sorted(r_["rows"][i_edge][2])
            Tn = Tg + dT if upward else Tg - dT
            sgn = 1.0 if upward else -1.0

            def Tx(x):
                if not present(dbf, comps, phases, B, x, Tg, lab, labeller):
                    return Tg
                return bisect_T(dbf, comps, phases, B, x, Tg, Tn, lab, labeller)
            probe = xs_at if len(xs_at) <= 7 else [xs_at[int(round(k * (len(xs_at) - 1) / 6))] for k in range(7)]
            vals = [(sgn * Tx(x), x) for x in probe]
            Fb, xb = max(vals)
            k = probe.index(xb)
            lo_ = probe[k - 1] if k > 0 else max(xb - 0.01, 1e-5)
            hi_ = probe[k + 1] if k < len(probe) - 1 else min(xb + 0.01, 1 - 1e-5)
            g = (5 ** 0.5 - 1) / 2
            c, d = hi_ - g * (hi_ - lo_), lo_ + g * (hi_ - lo_)
            fc, fd = sgn * Tx(c), sgn * Tx(d)
            while hi_ - lo_ > 5e-4:
                if fc >= fd:
                    hi_, d, fd = d, c, fc
                    c = hi_ - g * (hi_ - lo_)
                    fc = sgn * Tx(c)
                else:
                    lo_, c, fc = c, d, fd
                    d = lo_ + g * (hi_ - lo_)
                    fd = sgn * Tx(d)
            for F_, x_ in ((fc, c), (fd, d)):
                if Fb < F_:
                    Fb, xb = F_, x_
            Tb = sgn * Fb
            near = [(abs(T - Tb), T, src, how) for T, src, how in cand if abs(T - Tb) <= 0.05]
            if near:
                _, T, src, how = min(near)
                return float(T), src, how, None
            nb = _sets_at(dbf, comps, phases, B, xb, Tb + 0.2 * sgn)
            edge = xb < 1e-3 or xb > 1 - 1e-3
            if upward and len(nb) == 1 and base(nb[0]["phase"]) == "LIQUID":
                how = "melting of the pure element" if edge else "congruent melting"
            elif len(nb) == 1:
                how = "transformation of the pure element" if edge else "congruent transformation"
            else:
                # not congruent: the limit is an invariant; ORCHID invariant temperatures carry ~0.5 K mapping error
                near = [(abs(T - Tb), T, src, how) for T, src, how in cand if abs(T - Tb) <= 2.0]
                if near:
                    _, T, src, how = min(near)
                    return float(T), f"{src} (composition search gave {Tb} K)", how, None
                how = "maximum of the stability range" if upward else "minimum of the stability range"
            cp = {"x": round(xb, 5), "T_K": Tb, "kind": how, "neighbour_state": [[q["phase"], round(q["x"], 5)] for q in nb]}
            return Tb, f"{'maximum' if upward else 'minimum'} over composition at x({Bel}) = {round(xb, 5)} " \
                       f"(golden-section in x, bisection 0.01 K)", how, cp
        T_hi, src_hi, how_hi, cp_hi = limit(i_hi, True)
        T_lo, src_lo, how_lo, cp_lo = limit(i_lo, False)
        if cp_hi and cp_hi["kind"] in ("congruent melting", "congruent transformation"):
            r_["congruent_point"] = cp_hi
        widths = [(r_["rows"][i][1] - r_["rows"][i][0], -i) for i in idx]
        w, mi = max(widths)
        ih = -mi
        x_lo, x_hi = r_["rows"][ih][0], r_["rows"][ih][1]
        tdb = sorted(r_["tdb"])
        st = structure_of(tdb[0]) if len(tdb) == 1 else {**structure_of(tdb[0]), "derivation": "several TDB phases share this label; first used"}
        # maximum solubility (terminal phases): largest (A) x or smallest (B) x over all rows
        if lab == f"({Ael})":
            im = max(idx, key=lambda i: (r_["rows"][i][1], -i))
            sol = {"side": "B in A", "x": r_["rows"][im][1], "T_K": float(Tgrid[im]), "source": "grid row"}
        elif lab == f"({Bel})":
            im = min(idx, key=lambda i: (r_["rows"][i][0], i))
            sol = {"side": "A in B", "x": r_["rows"][im][0], "T_K": float(Tgrid[im]), "source": "grid row"}
        else:
            sol = None
        if sol:  # snap to an invariant point of this phase when the maximum lies at an invariant
            for k, inv in enumerate(data["invariants"]):
                for p in inv["points"]:
                    if labeller.label(p["phase"], p["x"]) == lab and abs(inv["T"] - sol["T_K"]) <= dT + 0.5 and \
                            ((sol["side"] == "B in A" and p["x"] >= sol["x"] - 1e-9) or (sol["side"] == "A in B" and p["x"] <= sol["x"] + 1e-9)):
                        sol = {**sol, "x": p["x"], "T_K": inv["T"], "source": f"invariant:{k}"}
        out.append({"label": lab, "tdb_phases": tdb, **{k: st[k] for k in st},
                    "T_low_K": T_lo, "T_low_source": src_lo, "T_low_kind": how_lo,
                    "T_high_K": T_hi, "T_high_source": src_hi, "T_high_kind": how_hi,
                    "grid_rows": [float(Tgrid[i_lo]), float(Tgrid[i_hi])],
                    "homogeneity": {"T_K": float(Tgrid[ih]), "x_min": x_lo, "x_max": x_hi,
                                    "note": "min/max equilibrium composition of this phase over the grid row (two-phase cells give exact boundary compositions)"},
                    "max_solubility": sol, "stable_at_300K": i_lo == 0,
                    "congruent_point": r_.get("congruent_point")})
    return out


def phase_fraction(data, dbf, comps, phases, labeller, fm):
    """Candidate alloys: compositions whose vertical path crosses the most fields; step T in 5 K."""
    _, equilibrium, v = _pyc()
    B = data["components"][1]
    grid = []
    for row in fm["rows"]:
        r_ = []
        for val, n in row:
            r_ += [val] * n
        grid.append(r_)
    grid = np.array(grid)
    cands = []
    for j in range(5, 96, 5):
        col = grid[:, j]
        ch = int(np.sum(col[1:] != col[:-1]))
        cands.append((-len(set(col.tolist()) - {-1}), -ch, j))
    cands.sort()
    out = []
    Ts = np.arange(300.0, data["T_axis_max"] + 1e-9, 5.0)
    for _, _, j in cands[:3]:
        x = round(j * fm["dx"], 2)
        res = equilibrium(dbf, comps, phases, {v.X(B): x, v.T: Ts.tolist(), v.P: p_atm, v.N: 1})
        ph = res.Phase.values[0, 0, :, 0]
        xb = res.X.sel(component=B).values[0, 0, :, 0]
        npf = res.NP.values[0, 0, :, 0]
        Y = res.Y.values[0, 0, :, 0]
        series = {}
        for a in range(len(Ts)):
            cs = _sets(ph[a], xb[a], npf[a], Y[a], (dbf, comps, phases))
            labs = labeller.assemblage(cs)
            for c, lab in zip(cs, labs):
                series.setdefault(lab, {"tdb": base(c["phase"]), "NP": [0.0] * len(Ts)})["NP"][a] = c["np"]
        out.append({"x": x, "T": Ts.tolist(), "series": {k: series[k] for k in sorted(series)}})
    return out


def formation_ext(data, dbf, comps, phases, stab_labels):
    _, equilibrium, v = _pyc()
    A, B = data["components"]
    sref = {}
    for el_ in (A, B):
        res = equilibrium(dbf, [el_, "VA"], phases, {v.T: 298.15, v.P: p_atm, v.N: 1}, output="SM")
        sref[el_] = float(np.asarray(res.SM.values).squeeze())
        res2 = equilibrium(dbf, [el_, "VA"], phases, {v.T: 298.15, v.P: p_atm, v.N: 1}, output="HM")
        sref[el_ + "_H"] = float(np.asarray(res2.HM.values).squeeze())
    rows = []
    for c in data["properties"].get("formation") or []:
        res = equilibrium(dbf, comps, [c["phase"]], {v.X(B): float(c["x"]), v.T: 298.15, v.P: p_atm, v.N: 1}, output="CPM")
        cp = float(np.asarray(res.CPM.values).squeeze())
        dS = c["SM_298"] - (1 - c["x"]) * sref[A] - c["x"] * sref[B]
        fa = formula_atoms(c["label"], data["elements"])
        rows.append({"phase": c["phase"], "label": c["label"], "x": c["x"], "dfH_J_per_mol_atoms": c["HM_298"],
                     "S_J_per_mol_atoms_K": c["SM_298"], "dfS_J_per_mol_atoms_K": dS,
                     "Cp_J_per_mol_atoms_K": cp if np.isfinite(cp) else None,
                     "Cp_note": None if np.isfinite(cp) else "CPM not finite from pycalphad at 298.15 K",
                     "formula_atoms": fa[0] if fa else None, "formula_x": fa[1] if fa else None,
                     "stable_at_298K": c["label"] in stab_labels})
    return {"S_ref_298": {A: sref[A], B: sref[B]}, "H_ref_298": {A: sref[A + "_H"], B: sref[B + "_H"]}, "rows": rows,
            "method": "single-phase equilibrium at the phase's low-T composition, 298.15 K; dfS = S - sum x_i S_i(SER); "
                      "SER entropies from unary equilibria at 298.15 K"}


def mixing_table(data, dbf, comps):
    from pd_data import liquid_activity, liquid_mixing
    A, B = data["components"]
    T = data["properties"]["T_liquid"]
    xs = [round(0.1 * k, 1) for k in range(1, 10)]
    h = 1e-4
    grid = [0.0]
    for x in xs:
        grid += [x - h, x, x + h]
    grid += [1.0]
    H, G = liquid_mixing(dbf, comps, A, B, T, np.array(grid))
    aA, aB = liquid_activity(dbf, comps, A, B, T, xs)
    rows = []
    for k, x in enumerate(xs):
        i = 1 + 3 * k
        Hx, dH = H[i + 1], (H[i + 2] - H[i]) / (2 * h)
        rows.append({"x": x, "HM_MIX": float(Hx), "GM_MIX": float(G[i + 1]), "a_A": float(aA[k]), "a_B": float(aB[k]),
                     "HM_PART_A": float(Hx - x * dH), "HM_PART_B": float(Hx + (1 - x) * dH)})
    return {"T": T, "rows": rows, "reference": "pure liquid A and B at T",
            "partials": "H_i - H_i(pure liquid) from central differences of the integral HM_MIX (h = 1e-4)"}


def prepare_ext(stem: str, data: dict, force: bool = False) -> dict:
    path = cache / f"{stem}.ext.json"
    if path.exists() and not force:
        d = load_json(path)
        if d.get("ext_version") == ext_version:
            return d
    Database, _, _ = _pyc()
    dbf = Database(str(tdb_dir / data["tdb"]["file"]))
    A, B = data["components"]
    comps = [A, B, "VA"]
    phases = data["phases"]
    labeller = Labeller(stem, data["elements"], data["stable_A"], data["stable_B"], data["continuous"])
    xcalc, Tgrid, sets = grid_sets(dbf, comps, phases, B, data["field_map"])
    stab = stability(data, dbf, comps, phases, labeller, xcalc, Tgrid, sets)
    labs300 = {s["label"] for s in stab if s["stable_at_300K"]}
    out = {"ext_version": ext_version, "stem": stem, "phase_stability": stab,
           "phase_fraction": phase_fraction(data, dbf, comps, phases, labeller, data["field_map"]),
           "formation": formation_ext(data, dbf, comps, phases, labs300),
           "mixing_table": mixing_table(data, dbf, comps),
           "method": {"grid": "same composition-set-aware grid as the field map", "phase_fraction": "equilibrium step, dT 5 K",
                      "stability_limits": "grid limit snapped to an invariant/congruent/melting T within one grid step, "
                                          "otherwise bisection to 0.01 K at a fixed composition"}}
    from pd_data import _finite
    out = _finite(out)
    write_json(path, out)
    return out


if __name__ == "__main__":
    import time

    from pd_data import prepare
    for s in sys.argv[1:]:
        t = time.time()
        d = prepare_ext(s, prepare(s), force=True)
        print(s, f"{time.time()-t:.1f}s", [(p["label"], p["T_low_K"], p["T_high_K"], p["strukturbericht"]) for p in d["phase_stability"]],
              [f["x"] for f in d["phase_fraction"]], [(r["label"], r["Cp_J_per_mol_atoms_K"]) for r in d["formation"]["rows"]], flush=True)
