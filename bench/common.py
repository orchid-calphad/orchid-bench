"""Shared helpers: paths, canonical JSON, hashing, element/phase label rules."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
TDB_DIR = Path(os.environ.get("ORCHID_TDB_DIR", Path.home() / "projects/open-tdb/databases/sgte-binary"))
PD_DIR = Path(os.environ.get("ORCHID_PD_DIR", Path.home() / "projects/orchid-data/phase-diagrams/sgte"))
CACHE = BENCH / "cache"
DOCS = BENCH / "docs"
TRUTH = BENCH / "truth"
RESULTS = BENCH / "results"
OPEN_TDB = Path(os.environ.get("OPEN_TDB", Path.home() / "projects/open-tdb"))
# The private orchid-pipeline checkout: set ORCHID_PIPELINE, or keep it next to this repository.
ORCHID_PIPELINE = Path(os.environ.get("ORCHID_PIPELINE", ROOT.parent / "orchid-pipeline"))
ASSESSED_PD = Path(os.environ.get("ASSESSED_PD_SRC", ORCHID_PIPELINE / "pipelines/assessed-phase-diagrams/src"))
FONT_DIR = None
try:
    import matplotlib
    FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
except Exception:  # pragma: no cover
    pass

R_GAS = 8.314462618
P_ATM = 101325.0


def _clean(o):
    if isinstance(o, float):
        if not math.isfinite(o):
            raise ValueError("non-finite float in canonical JSON")
        return o
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):  # numpy scalar
        return _clean(o.item())
    return o


def canonical_json(obj) -> str:
    return json.dumps(_clean(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def write_json(path, obj, pretty=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        path.write_text(json.dumps(_clean(obj), sort_keys=True, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    else:
        path.write_text(canonical_json(obj), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def el(sym: str) -> str:
    """'AL' -> 'Al'."""
    return sym[:1].upper() + sym[1:].lower()


def r(v, nd):
    return float(round(float(v), nd))


def fmt(v, nd):
    s = f"{float(v):.{nd}f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    return s
