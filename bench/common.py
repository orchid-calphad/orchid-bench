"""Shared helpers: paths, canonical JSON, hashing, element/phase label rules."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

bench = Path(__file__).resolve().parent
root = bench.parent
tdb_dir = Path(os.environ.get("ORCHID_TDB_DIR", Path.home() / "projects/open-tdb/databases/sgte-binary"))
pd_dir = Path(os.environ.get("ORCHID_PD_DIR", Path.home() / "projects/orchid-data/phase-diagrams/sgte"))
cache = bench / "cache"
docs_dir = bench / "docs"
truth_dir = bench / "truth"
results = bench / "results"
open_tdb = Path(os.environ.get("OPEN_TDB", Path.home() / "projects/open-tdb"))
ASSESSED_PD = Path(
    os.environ.get(
        "ASSESSED_PD_SRC",
        "/private/tmp/claude-504/-Users-raymundo-arroyave-projects-open-tdb/5d97acc6-0c9e-4398-87a5-e3422e273f18/scratchpad/assessed-pd-annotator/src",
    )
)
font_dir = None
try:
    import matplotlib

    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
except Exception:  # pragma: no cover
    pass

r_gas = 8.314462618
p_atm = 101325.0


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


def canonical_json(obj: Any) -> str:
    """Serialise obj to canonical JSON text: sorted keys, compact separators, trailing newline."""
    return json.dumps(_clean(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def write_json(path: str | Path, obj: Any, pretty: bool = False) -> None:
    """Write obj as UTF-8 JSON, creating parent directories; indented when pretty is True."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        path.write_text(json.dumps(_clean(obj), sort_keys=True, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    else:
        path.write_text(canonical_json(obj), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    """Read and parse a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path) -> str:
    """Return the hex SHA-256 digest of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def el(sym: str) -> str:
    """'AL' -> 'Al'."""
    return sym[:1].upper() + sym[1:].lower()


def r(v: float, nd: int) -> float:
    """Round v to nd decimal places and return a float."""
    return float(round(float(v), nd))


def fmt(v: float, nd: int) -> str:
    """Format v with exactly nd decimals, dropping the sign of negative zero."""
    s = f"{float(v):.{nd}f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    return s
