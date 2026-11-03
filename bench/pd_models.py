"""Pycalphad-backed property helpers for phase-diagram data, split from pd_data.py."""

import numpy as np
from common import p_atm, r_gas


def _pyc():
    from pycalphad import Database, calculate, equilibrium
    from pycalphad import variables as v

    return Database, equilibrium, calculate, v


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
    """Return the property prop of phase at composition x0 and temperature T in K, with the equilibrium result."""
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
