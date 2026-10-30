"""Region and invariant classification of phase-diagram grids, split from pd_data.py."""

import numpy as np
from systems import base  # noqa: E402


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
