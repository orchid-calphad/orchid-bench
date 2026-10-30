"""Per-system data preparation: reuse ORCHID calculated diagrams, add field assemblages, invariant types and
property curves computed with pycalphad. Results are cached in bench/cache/<stem>.json (deterministic).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cache, p_atm, pd_dir, r_gas, tdb_dir, el, load_json, sha256_file, write_json  # noqa: E402
from systems import Labeller, base  # noqa: E402

warnings.filterwarnings("ignore")
cache_version = 12


# ----------------------------------------------------------------------------------------------- order/disorder naming
_LAYOUT = {}
DISORDERED_OF = (("FCC", "FCC_A1"), ("BCC", "BCC_A2"), ("HCP", "HCP_A3"))


def _layout(dbf, comps, phase):
    key = (id(dbf), phase)
    if key not in _LAYOUT:
        from pycalphad import Model

        _LAYOUT[key] = [sorted(str(s.name) for s in sub) for sub in Model(dbf, comps, phase).constituents]
    return _LAYOUT[key]


def disorder_name(dbf, comps, phases, name, y):
    """A phase modelled with equivalent sublattices (e.g. L12_FCC with an A1_FCC disordered part) that pycalphad
    reports in its disordered state is renamed to the disordered lattice phase (FCC_A1, ...) when that phase exists.
    """
    cons = _layout(dbf, comps, name)
    groups, off = {}, 0
    for sub in cons:
        if len(sub) > 1:
            groups.setdefault(tuple(sub), []).append(off)
        off += len(sub)
    groups = [g for g in groups.values() if len(g) > 1]
    if not groups:
        return name
    spread = max(max(float(y[o]) for o in g) - min(float(y[o]) for o in g) for g in groups)
    if spread < 1e-4:
        hint = str(dbf.phases[name].model_hints.get("disordered_phase", ""))
        for k, dis in DISORDERED_OF:
            if (k in name or k in hint) and dis in phases and dis != name:
                return dis
    return name


# ----------------------------------------------------------------------------------------------- pycalphad helpers
def _pyc():
    from pycalphad import Database, calculate, equilibrium
    from pycalphad import variables as v

    return Database, equilibrium, calculate, v


def eq_point(dbf, comps, phases, B, x, T):
    """Stable composition sets at (x_B, T): list of {phase, x, np} sorted by x, '#2' for repeated phases."""
    _, equilibrium, _, v = _pyc()
    res = equilibrium(dbf, comps, phases, {v.X(B): float(x), v.T: float(T), v.P: p_atm, v.N: 1})
    names = res.Phase.values.squeeze()
    xb = res.X.sel(component=B).values.squeeze()
    npf = res.NP.values.squeeze()
    Y = res.Y.values.reshape(len(np.atleast_1d(names)), -1)
    out = []
    for n, xx, f, y in zip(np.atleast_1d(names), np.atleast_1d(xb), np.atleast_1d(npf), Y):
        if n and str(n) != "" and np.isfinite(f) and f > 1e-6:
            out.append({"phase": disorder_name(dbf, comps, phases, str(n), y), "x": float(xx), "np": float(f)})
    out.sort(key=lambda c: c["x"])
    seen = {}
    for c in out:
        k = c["phase"]
        seen[k] = seen.get(k, 0) + 1
        if seen[k] > 1:
            c["phase"] = f"{k}#{seen[k]}"
    return out


# ----------------------------------------------------------------------------------------------- geometry cleaning
def clean_regions(regions):
    """Drop duplicate and degenerate (frame-edge) two-phase regions; return boundary polylines."""
    seen, out = set(), []
    for reg in regions:
        if len(reg) != 2:
            continue
        a, b = reg
        key = tuple(
            sorted(
                (base(s["phase"]), tuple(round(v, 3) for v in s["x"][::4]), tuple(round(v, 1) for v in s["T"][::4])) for s in reg
            )
        )
        if key in seen:
            continue
        seen.add(key)

        def edge(s):
            xs = np.asarray(s["x"])
            if xs.max() - xs.min() > 0.002:
                return None
            m = xs.mean()
            return 0 if m < 0.002 else (1 if m > 0.998 else None)

        ea, eb = edge(a), edge(b)
        if ea is not None and ea == eb:
            continue
        Ts = np.concatenate([a["T"], b["T"]])
        if Ts.max() - Ts.min() < 1.5 and max(np.ptp(a["x"]), np.ptp(b["x"])) < 0.003:
            continue
        out.append(
            {
                "phases": [a["phase"], b["phase"]],
                "curves": [{"phase": s["phase"], "x": [float(v) for v in s["x"]], "T": [float(v) for v in s["T"]]} for s in reg],
            }
        )
    return out


def decode_grid(grid):
    """Decode a stored grid encoding into a list of values."""
    labels = grid["labels"]
    rows = []
    for row in grid["rows"]:
        cells = []
        for idx, cnt in row:
            cells.extend([labels[idx]] * cnt)
        rows.append(cells)
    arr = np.array(rows, dtype=object)
    return arr  # shape (nT, nx)


# ----------------------------------------------------------------------------------------------- invariant typing
def _match(pts, probe):
    """Map probe composition sets to invariant point indices (same base phase, nearest composition)."""
    idx = set()
    for c in probe:
        cand = [(abs(p["x"] - c["x"]), i) for i, p in enumerate(pts) if base(p["phase"]) == base(c["phase"])]
        if cand:
            idx.add(min(cand)[1])
    return idx


def classify_invariant(pts, above, below):
    """Pts sorted by x (3 composition sets); above/below: probe composition sets near x_mid at T +/- dT."""
    left, mid, right = (p["phase"] for p in pts)
    ia, ib = _match(pts, above), _match(pts, below)
    is_liq = lambda p: base(p) == "LIQUID"  # noqa: E731
    if ia == {0, 2} and ib != {0, 2}:
        kind = "formation"
    elif ib == {0, 2} and ia != {0, 2} or 1 in ia and 1 not in ib:
        kind = "decomposition"
    elif 1 in ib and 1 not in ia:
        kind = "formation"
    else:
        kind = "unknown"
    if kind == "decomposition":
        if is_liq(mid):
            typ = "monotectic" if (is_liq(left) or is_liq(right)) else "eutectic"
        else:
            typ = "metatectic" if (is_liq(left) or is_liq(right)) else "eutectoid"
        reactants, products = [mid], [left, right]
    elif kind == "formation":
        if is_liq(left) and is_liq(right):
            typ = "syntectic"
        elif is_liq(left) or is_liq(right):
            typ = "peritectic"
        else:
            typ = "peritectoid"
        reactants, products = [left, right], [mid]
    else:
        typ, reactants, products = "unknown", [left, mid, right], []
    return typ, reactants, products


# ----------------------------------------------------------------------------------------------- property curves
def liquid_mixing(dbf, comps, A, B, T, xs):
    """Integral mixing enthalpy/Gibbs energy of the liquid relative to pure liquid A and B (J/mol of atoms),
    from LIQUID-only equilibria (works for substitutional and associate liquids).
    """
    _, equilibrium, _, v = _pyc()
    xs = np.asarray(xs, float)
    xc = np.clip(xs, 1e-7, 1 - 1e-7)
    res = equilibrium(dbf, comps, ["LIQUID"], {v.X(B): xc.tolist(), v.T: float(T), v.P: p_atm, v.N: 1}, output="HM")
    H = res.HM.values.ravel()
    G = res.GM.values.ravel()
    Hm = H - ((1 - xc) * H[0] + xc * H[-1]) if xs[0] == 0 and xs[-1] == 1 else H
    Gm = G - ((1 - xc) * G[0] + xc * G[-1]) if xs[0] == 0 and xs[-1] == 1 else G
    Hm[0] = Hm[-1] = 0.0
    Gm[0] = Gm[-1] = 0.0
    return Hm, Gm


def liquid_activity(dbf, comps, A, B, T, xs):
    """Activities of A and B in the liquid (reference: pure liquid at T) from LIQUID-only equilibria."""
    _, equilibrium, _, v = _pyc()
    xs = [float(x) for x in xs]
    res = equilibrium(dbf, comps, ["LIQUID"], {v.X(B): [1e-7] + xs + [1 - 1e-7], v.T: float(T), v.P: p_atm, v.N: 1})
    mu = res.MU.squeeze()
    muA = mu.sel(component=A).values.ravel()
    muB = mu.sel(component=B).values.ravel()
    gA = res.GM.values.ravel()[0]
    gB = res.GM.values.ravel()[-1]
    aA = np.exp((muA[1:-1] - gA) / (r_gas * T))
    aB = np.exp((muB[1:-1] - gB) / (r_gas * T))
    return aA, aB


def phase_min_curve(dbf, comps, phase, B, T, nbins=101):
    """Lower envelope of GM(x) for one phase at T from sampled internal configurations."""
    _, _, calculate, v = _pyc()
    res = calculate(dbf, comps, phase, T=T, P=p_atm, N=1, pdens=2000, output="GM")
    x = res.X.sel(component=B).values.ravel()
    g = res.GM.values.ravel()
    ok = np.isfinite(x) & np.isfinite(g)
    x, g = x[ok], g[ok]
    edges = np.linspace(0, 1, nbins + 1)
    xs, gs = [], []
    for i in range(nbins):
        m = (x >= edges[i]) & (x < edges[i + 1] if i < nbins - 1 else x <= edges[i + 1])
        if m.any():
            j = np.argmin(np.where(m, g, np.inf))
            xs.append(float(x[j]))
            gs.append(float(g[j]))
    return np.array(xs), np.array(gs)


def compound_at(dbf, comps, phase, B, x0, T, prop):
    """Return the stable compound data of phase at composition x0 and temperature T in K."""
    _, equilibrium, _, v = _pyc()
    res = equilibrium(dbf, comps, [phase], {v.X(B): float(x0), v.T: T, v.P: p_atm, v.N: 1}, output=prop)
    return np.asarray(getattr(res, prop).values).squeeze(), res


def rk_parameters(dbf, phase, A, B):
    """Liquid Redlich-Kister L parameters printed as a + b*T (+ c*T*ln T); None if not of that form."""
    import sympy
    from tinydb import where

    rows = []
    q = dbf._parameters.search((where("phase_name") == phase) & (where("parameter_type").one_of(["G", "L"])))
    for p in q:
        cons = p["constituent_array"]
        flat = [sorted(str(c) for c in sub) for sub in cons]
        if len(flat) != 1 or sorted(flat[0]) != sorted([A, B]):
            continue
        expr = p["parameter"]
        for _ in range(6):
            expr = expr.subs(dbf.symbols)
        Tsym = sympy.Symbol("T")
        expr = sympy.sympify(str(expr), locals={"T": Tsym})
        if isinstance(expr, sympy.Piecewise):
            expr = expr.args[0][0]
        expr = sympy.expand(expr)
        coeffs = {"a": 0.0, "b": 0.0, "c": 0.0}
        ok = True
        for term in sympy.Add.make_args(expr):
            if term.free_symbols == set():
                coeffs["a"] += float(term)
            elif term.has(sympy.log):
                c = sympy.simplify(term / (Tsym * sympy.log(Tsym)))
                if c.free_symbols:
                    ok = False
                else:
                    coeffs["c"] += float(c)
            else:
                c = sympy.simplify(term / Tsym)
                if c.free_symbols:
                    ok = False
                else:
                    coeffs["b"] += float(c)
        if ok:
            rows.append({"order": int(p["parameter_order"]), **coeffs})
    rows.sort(key=lambda r_: r_["order"])
    return rows


# ----------------------------------------------------------------------------------------------- main preparation
def prepare(stem: str, force: bool = False) -> dict:
    """Prepare, and cache, the phase-diagram data for a TDB system stem."""
    out_path = cache / f"{stem}.json"
    if out_path.exists() and not force:
        d = load_json(out_path)
        if d.get("cache_version") == cache_version:
            return d
    Database, equilibrium, calculate, v = _pyc()
    from scipy import ndimage

    pd_path = pd_dir / f"{stem}.json"
    pdj = load_json(pd_path)
    tdb = tdb_dir / pdj["tdb"]
    dbf = Database(str(tdb))
    A, B = [c.upper() for c in pdj["components"]]
    assert pdj["x_component"].upper() == B
    comps = [A, B, "VA"]
    phases = list(pdj["phases"])
    Ael, Bel = el(A), el(B)

    # boundaries (reused from the ORCHID calculated diagram)
    regs = clean_regions(pdj["regions"])
    allT = [t for rg in regs for c in rg["curves"] for t in c["T"]]
    T_top = max(allT) if allT else 1500.0
    T_axis_max = float(np.ceil((T_top + 0.06 * (T_top - 300.0)) / 100.0) * 100.0)

    def tm(edge):
        best = None
        for rg in regs:
            for c in rg["curves"]:
                if base(c["phase"]) != "LIQUID":
                    continue
                for xx, TT in zip(c["x"], c["T"]):
                    if xx < 1e-4 if edge == 0 else xx > 1 - 1e-4:
                        best = TT if best is None else max(best, TT)
        return best

    Tm_A, Tm_B = tm(0), tm(1)

    # field map: our own composition-set-aware equilibrium grid (the ORCHID grid merges same-phase gaps)
    dx, dT = 0.01, 10.0
    xgrid = np.round(np.arange(101) * dx, 6)
    xcalc = xgrid.copy()
    xcalc[0] = 1e-5
    xcalc[-1] = 1 - 1e-5  # noqa: E702
    Tgrid = np.arange(300.0, T_axis_max + 1e-9, dT)
    nT, nx = len(Tgrid), len(xgrid)
    cellsets = [[None] * nx for _ in range(nT)]
    for i0 in range(0, nT, 40):
        Ts = Tgrid[i0 : i0 + 40]
        res = equilibrium(dbf, comps, phases, {v.X(B): xcalc.tolist(), v.T: Ts.tolist(), v.P: p_atm, v.N: 1})
        ph = res.Phase.values[0, 0]
        xb = res.X.sel(component=B).values[0, 0]
        npf = res.NP.values[0, 0]
        Yv = res.Y.values[0, 0]
        for a in range(len(Ts)):
            for j in range(nx):
                cs = []
                for k in range(ph.shape[-1]):
                    n = str(ph[a, j, k])
                    if n and np.isfinite(npf[a, j, k]) and npf[a, j, k] > 1e-6:
                        cs.append({"phase": disorder_name(dbf, comps, phases, n, Yv[a, j, k]), "x": float(xb[a, j, k])})
                cs.sort(key=lambda c: c["x"])
                seen = {}
                for c in cs:
                    seen[c["phase"]] = seen.get(c["phase"], 0) + 1
                    if seen[c["phase"]] > 1:
                        c["phase"] = f"{c['phase']}#{seen[c['phase']]}"
                cellsets[i0 + a][j] = cs

    # stable phases of the pure elements and continuous single-phase bands
    def single(cs):
        return cs[0]["phase"] if len(cs) == 1 else None

    stable_A = sorted(
        {base(c["phase"]) for i in range(nT) for j in (0, 1) for c in cellsets[i][j] if c["x"] < 0.005} - {"LIQUID"}
    )
    stable_B = sorted(
        {base(c["phase"]) for i in range(nT) for j in (-1, -2) for c in cellsets[i][j] if c["x"] > 0.995} - {"LIQUID"}
    )
    continuous = sorted(
        {
            single(cellsets[i][0])
            for i in range(nT)
            if single(cellsets[i][0]) not in (None, "LIQUID")
            and all(single(cellsets[i][j]) == single(cellsets[i][0]) for j in range(nx))
        }
    )
    labeller = Labeller(stem, [Ael, Bel], stable_A, stable_B, continuous)
    keys = np.empty((nT, nx), dtype=object)
    for i in range(nT):
        for j in range(nx):
            cs = cellsets[i][j]
            keys[i, j] = " + ".join(sorted(labeller.assemblage(cs))) if cs else ""
    fid = -np.ones((nT, nx), dtype=int)
    fields = []
    for key in sorted(set(keys.ravel())):
        if not key:
            continue
        comp, n = ndimage.label(keys == key)
        for k in range(1, n + 1):
            m = comp == k
            cnt = int(m.sum())
            if cnt < 2:
                continue
            dist = ndimage.distance_transform_edt(np.pad(m, 1), sampling=[dT / (T_axis_max - 300.0), dx])[1:-1, 1:-1]
            i, j = np.unravel_index(np.argmax(dist), dist.shape)
            fields.append({"key": key, "mask": m, "cell": (int(i), int(j)), "n_cells": cnt, "inradius": float(dist[i, j])})
    fields.sort(key=lambda f: (f["cell"][0], f["cell"][1]))
    out_fields = []
    for idx, f in enumerate(fields):
        fid[f["mask"]] = idx
        i, j = f["cell"]
        cs = cellsets[i][j]
        out_fields.append(
            {
                "id": idx,
                "key": f["key"],
                "point": [float(xgrid[j]), float(Tgrid[i])],
                "n_cells": f["n_cells"],
                "inradius": f["inradius"],
                "assemblage": [c["phase"] for c in cs],
                "compositions": [c["x"] for c in cs],
                "labels": labeller.assemblage(cs),
            }
        )
    fmap_rows = []
    for i in range(nT):
        row, prev, cnt = [], None, 0
        for j in range(nx):
            val = int(fid[i, j])
            if val == prev:
                cnt += 1
            else:
                if prev is not None:
                    row.append([prev, cnt])
                prev, cnt = val, 1
        row.append([prev, cnt])
        fmap_rows.append(row)

    # invariants: dedupe and classify
    invs = []
    for inv in sorted(pdj["invariants"], key=lambda q: -q["T"]):
        pts = sorted(inv["points"], key=lambda p: p["x"])
        if len(pts) != 3:
            continue
        sig = tuple(sorted(base(p["phase"]) for p in pts))
        if any(abs(o["T"] - inv["T"]) < 0.5 and o["_sig"] == sig for o in invs):
            continue
        gl, gr = pts[1]["x"] - pts[0]["x"], pts[2]["x"] - pts[1]["x"]
        gaps = [abs(o["T"] - inv["T"]) for o in pdj["invariants"] if abs(o["T"] - inv["T"]) > 0.05]
        dTp = min([1.5] + [0.4 * gp for gp in gaps])
        sides = [pts[1]["x"] + 0.3 * gr, pts[1]["x"] - 0.3 * gl]
        if gl > gr:
            sides.reverse()
        typ = "unknown"
        for xm in sides + [pts[1]["x"]]:
            xm = min(max(xm, 1e-4), 1 - 1e-4)
            above = eq_point(dbf, comps, phases, B, xm, inv["T"] + dTp)
            below = eq_point(dbf, comps, phases, B, xm, inv["T"] - dTp)
            typ, reac, prod = classify_invariant(pts, above, below)
            above = [[c["phase"], round(c["x"], 5)] for c in above]
            below = [[c["phase"], round(c["x"], 5)] for c in below]
            if typ != "unknown":
                break
        if typ == "unknown":
            # geometric fallback: do the two-phase regions holding the middle phase leave the line downwards?
            votes = []
            for rg in regs:
                for c in rg["curves"]:
                    if base(c["phase"]) != base(pts[1]["phase"]):
                        continue
                    for k_ in (0, -1):
                        if abs(c["T"][k_] - inv["T"]) < 0.05 and abs(c["x"][k_] - pts[1]["x"]) < 2e-3:
                            other_T = c["T"][-1 - k_] if k_ == 0 else c["T"][0]
                            votes.append(other_T < inv["T"])
            if votes:
                fake_above = [pts[0], pts[2]] if sum(votes) > len(votes) / 2 else [pts[1]]
                fake_below = [pts[1]] if sum(votes) > len(votes) / 2 else [pts[0], pts[2]]
                typ, reac, prod = classify_invariant(pts, fake_above, fake_below)
                above, below = "geometry", "geometry"
        lab_pts = labeller.assemblage(pts)
        lmap = {p["phase"]: l for p, l in zip(pts, lab_pts)}
        invs.append(
            {
                "_sig": sig,
                "T": float(inv["T"]),
                "type": typ,
                "points": [{"phase": p["phase"], "x": float(p["x"]), "label": lmap[p["phase"]]} for p in pts],
                "reactants": reac,
                "products": prod,
                "reaction": " + ".join(lmap[p] for p in reac) + " → " + " + ".join(lmap[p] for p in prod),
                "check_above": above,
                "check_below": below,
            }
        )
    for i_ in invs:
        i_.pop("_sig")
    invs.sort(key=lambda q: -q["T"])

    # congruent melting of intermediate phases (liquidus maximum where both curves meet)
    congruent = []
    for rg in regs:
        ph = [base(c["phase"]) for c in rg["curves"]]
        if "LIQUID" not in ph:
            continue
        others = [c for c in rg["curves"] if base(c["phase"]) != "LIQUID"]
        if not others:
            continue
        other = others[0]
        liq = [c for c in rg["curves"] if base(c["phase"]) == "LIQUID"][0]
        k = int(np.argmax(other["T"]))
        if 0.02 < other["x"][k] < 0.98 and abs(other["x"][k] - liq["x"][k]) < 0.004 and abs(other["T"][k] - max(liq["T"])) < 0.5:
            lab_c = labeller.label(other["phase"], other["x"][k])
            chk = eq_point(dbf, comps, phases, B, other["x"][k], other["T"][k] - 1.0)
            ok_c = len(chk) == 1 and labeller.label(chk[0]["phase"], chk[0]["x"]) == lab_c
            if ok_c and not any(abs(c_["T"] - other["T"][k]) < 1 for c_ in congruent):
                congruent.append(
                    {
                        "phase": other["phase"],
                        "x": float(other["x"][k]),
                        "T": float(other["T"][k]),
                        "label": labeller.label(other["phase"], other["x"][k]),
                    }
                )

    # property curves
    xs = np.round(np.linspace(0.0, 1.0, 51), 4)
    T_liq = float(np.ceil((T_top + 60) / 50.0) * 50.0)
    props = {"T_liquid": T_liq}
    try:
        H1, G1 = liquid_mixing(dbf, comps, A, B, T_liq, xs)
        H2, G2 = liquid_mixing(dbf, comps, A, B, T_liq + 200, xs)
        props["mixing"] = {
            "x": xs.tolist(),
            "T": [T_liq, T_liq + 200],
            "HM_MIX": [H1.tolist(), H2.tolist()],
            "GM_MIX": [G1.tolist(), G2.tolist()],
        }
    except Exception as e:  # noqa: BLE001
        props["mixing_error"] = repr(e)
    try:
        xa = np.round(np.linspace(0.02, 0.98, 49), 4)
        aA, aB = liquid_activity(dbf, comps, A, B, T_liq, xa)
        props["activity"] = {"x": xa.tolist(), "T": T_liq, "a_A": aA.tolist(), "a_B": aB.tolist()}
    except Exception as e:  # noqa: BLE001
        props["activity_error"] = repr(e)

    # intermediate phases (not terminal, not liquid) with a composition at low T
    inter = []
    for ph in phases:
        if ph == "LIQUID" or ph in stable_A or ph in stable_B or ph in continuous:
            continue
        xs_ph = [x for f in out_fields for p, x in zip(f["assemblage"], f["compositions"]) if base(p) == ph]
        if not xs_ph:
            continue
        lowT = [(f["point"][1], x) for f in out_fields for p, x in zip(f["assemblage"], f["compositions"]) if base(p) == ph]
        lowT.sort()
        x0 = float(np.median([x for T_, x in lowT[:6]]))
        inter.append({"phase": ph, "x": x0, "label": labeller.label(ph, x0)})
    # several ordered compounds of one phase (e.g. FCC_4SL): keep distinct labels
    comp_rows = []
    for c in inter:
        groups = {}
        for f in out_fields:
            for p, x in zip(f["assemblage"], f["compositions"]):
                if base(p) == c["phase"]:
                    groups.setdefault(labeller.label(p, x), []).append((f["point"][1], x))
        for labname, vals in sorted(groups.items()):
            vals.sort()
            x0 = float(np.median([x for _, x in vals[:6]]))
            comp_rows.append({"phase": c["phase"], "label": labname, "x": x0})
    hf = []
    for c in comp_rows:
        try:
            H, _ = compound_at(dbf, comps, c["phase"], B, c["x"], 298.15, "HM")
            S, _ = compound_at(dbf, comps, c["phase"], B, c["x"], 298.15, "SM")
            if np.isfinite(H):
                hf.append({**c, "HM_298": float(H), "SM_298": float(S)})
        except Exception:  # noqa: BLE001
            pass
    props["formation"] = hf
    cp = None
    for c in hf:
        Ts = np.arange(300.0, min(T_top, 1600.0) + 1e-9, 25.0)
        try:
            vals, _ = compound_at(dbf, comps, c["phase"], B, c["x"], list(map(float, Ts)), "CPM")
            vals = np.atleast_1d(vals)
            if vals.size == Ts.size and np.all(np.isfinite(vals)):
                cp = {"phase": c["phase"], "label": c["label"], "x": c["x"], "T": Ts.tolist(), "CPM": vals.tolist()}
                break
        except Exception:  # noqa: BLE001
            continue
    if cp is not None:
        props["cp"] = cp
    # Gibbs energy curves at a temperature inside a two-phase region involving the liquid
    try:
        cand = [
            f
            for f in out_fields
            if len(f["assemblage"]) == 2 and "LIQUID" in [base(p) for p in f["assemblage"]] and f["n_cells"] >= 6
        ]
        cand.sort(key=lambda f: -f["n_cells"])
        if cand:
            f = cand[0]
            Tg = float(f["point"][1])
            curves = []
            ref = None
            for ph in sorted({base(p) for p in f["assemblage"]}) + [p for p in ["FCC_A1", "BCC_A2", "HCP_A3"] if p in phases]:
                if any(c_["phase"] == ph for c_ in curves):
                    continue
                xg, gg = phase_min_curve(dbf, comps, ph, B, Tg)
                if len(xg) >= 5 and np.ptp(xg) > 0.2:
                    curves.append({"phase": ph, "x": xg.tolist(), "GM": gg.tolist()})
                if len(curves) >= 3:
                    break
            # reference: pure-liquid endpoints (LIQUID-only equilibria at x -> 0, 1)
            res0 = equilibrium(dbf, comps, ["LIQUID"], {v.X(B): [1e-7, 1 - 1e-7], v.T: Tg, v.P: p_atm, v.N: 1})
            ref = res0.GM.values.ravel()
            tang = eq_point(dbf, comps, phases, B, f["point"][0], Tg)
            props["gibbs"] = {
                "T": Tg,
                "ref_A": float(ref[0]),
                "ref_B": float(ref[1]),
                "curves": curves,
                "tangent": [{"phase": c_["phase"], "x": c_["x"]} for c_ in tang],
                "field_point": f["point"],
            }
            # G values at tangent points from the curves' phases
            from pycalphad import variables as vv  # noqa: F401

            _, equilibrium_, _, v_ = _pyc()
            res = equilibrium_(dbf, comps, phases, {v_.X(B): f["point"][0], v_.T: Tg, v_.P: p_atm, v_.N: 1})
            mu = res.MU.squeeze()
            props["gibbs"]["mu"] = [float(mu.sel(component=A)), float(mu.sel(component=B))]
    except Exception as e:  # noqa: BLE001
        props["gibbs_error"] = repr(e)
    try:
        props["rk_liquid"] = rk_parameters(dbf, "LIQUID", A, B)
    except Exception as e:  # noqa: BLE001
        props["rk_error"] = repr(e)

    data = {
        "cache_version": cache_version,
        "stem": stem,
        "tdb": {"file": tdb.name, "sha256": sha256_file(tdb)},
        "diagram_source": {
            "file": f"orchid-data/phase-diagrams/sgte/{pd_path.name}",
            "sha256": sha256_file(pd_path),
            "engine": pdj.get("engine"),
        },
        "original_source": pdj.get("original_source"),
        "components": [A, B],
        "elements": [Ael, Bel],
        "phases": phases,
        "stable_A": stable_A,
        "stable_B": stable_B,
        "continuous": continuous,
        "Tm_A": Tm_A,
        "Tm_B": Tm_B,
        "T_top": T_top,
        "boundaries": regs,
        "fields": out_fields,
        "T_axis_max": T_axis_max,
        "field_map": {
            "x0": 0.0,
            "dx": dx,
            "nx": nx,
            "T0": 300.0,
            "dT": dT,
            "nT": nT,
            "rows": fmap_rows,
            "note": "field id per cell (row = T0 + i*dT, column = i*dx); -1 = not a field",
        },
        "invariants": invs,
        "congruent": congruent,
        "properties": props,
        "calculation": {
            "engine": "pycalphad 0.11.2",
            "P_Pa": p_atm,
            "N": 1,
            "phases": phases,
            "field_map": "own pycalphad equilibrium grid, dx 0.01 (ends 1e-5), dT 10 K; fields = connected cells with equal labelled assemblage",
            "order_disorder": "composition sets of phases with equivalent sublattices reported in the disordered state are renamed to the disordered lattice phase (e.g. L12_FCC -> FCC_A1)",
            "invariant_type": "equilibria at x_mid, T +/- 1.5 K",
            "liquid_mixing": "calculate HM/GM of LIQUID minus linear pure-liquid reference",
            "activity": "LIQUID-only equilibria; reference pure liquid at T",
            "formation": "single-phase equilibrium at 298.15 K; HM relative to SER",
        },
    }
    data = _finite(data)
    write_json(out_path, data)
    return data


def _finite(o):
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v_) for k, v_ in o.items()}
    if isinstance(o, list):
        return [_finite(v_) for v_ in o]
    return o


if __name__ == "__main__":
    import time

    for s in sys.argv[1:]:
        t = time.time()
        d = prepare(s, force=True)
        print(
            s,
            f"{time.time() - t:.1f}s",
            len(d["fields"]),
            "fields",
            len(d["invariants"]),
            "inv",
            [(i["type"], round(i["T"], 1), i["reaction"]) for i in d["invariants"]],
            sorted(k for k in d["properties"]),
        )
