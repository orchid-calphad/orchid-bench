#!/usr/bin/env python3
"""Database side of the raster comparison: thermo_extractor/ORCHID `build_binary_topology.py` on each document's
source TDB over the figure's axis window (60 x 60 cells). Output work/tdb_topology/<doc>.json (input of
compare_binary_topology). Model-side reference only; never scored as an extractor.
"""

import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import bench, load_json, truth_dir  # noqa: E402

PIPE = Path.home() / "projects/orchid-pipeline"
PY = sys.executable
OUT = bench / "work" / "tdb_topology"


def one(tf):
    t = load_json(tf)
    doc = t["doc_id"]
    fig = next(f for f in t["figures"] if f["kind"] == "phase_diagram")
    main = next(a for a in fig["axes"] if a["role"] == "main")
    off = 273.15 if main["y"]["unit"] == "°C" else 0.0
    A, B = [e.upper() for e in t["elements"]]
    m = {
        "schema_version": 1,
        "id": doc,
        "database": {"path": f"databases/sgte-binary/{t['source_tdb']['file']}", "sha256": t["source_tdb"]["sha256"]},
        "components": [A, B],
        "composition_component": B,
        "conditions": {
            "temperature_min_kelvin": f"{main['y']['min'] + off:.2f}",
            "temperature_max_kelvin": f"{main['y']['max'] + off:.2f}",
            "temperature_cells": 60,
            "composition_min": "0.005",
            "composition_max": "0.995",
            "composition_cells": 60,
            "pressure_pa": "101325",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    mp, op = OUT / f"{doc}.manifest.json", OUT / f"{doc}.json"
    mp.write_text(json.dumps(m, indent=1), encoding="utf-8")
    if op.exists():
        return doc, "cached"
    r = subprocess.run(
        [
            PY,
            str(PIPE / "scripts/build_binary_topology.py"),
            str(mp),
            "--calculation-python",
            PY,
            "--output",
            str(op),
            "--timeout",
            "3000",
        ],
        capture_output=True,
        text=True,
    )
    return doc, (r.stdout + r.stderr)[-300:]


if __name__ == "__main__":
    with Pool(8) as p:
        for d, s in p.imap_unordered(one, sorted(truth_dir.glob("*.json"))):
            print(d, s.replace("\n", " | "), flush=True)
