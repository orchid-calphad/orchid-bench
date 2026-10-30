"""Numeric, text and unit helpers for the scoring harness, split from score.py."""

from __future__ import annotations

import math
import re
import numpy as np


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


# ------------------------------------------------------------------------------------------------ phase diagrams
def _canon(lbl):
    s = norm_text(lbl).replace(" ", "").upper()
    s = re.sub(r"#\d+$", "", s)
    return {"LIQUID": "L"}.get(s, s)


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
