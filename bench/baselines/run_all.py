#!/usr/bin/env python3
"""Run every deterministic baseline, then score every extractor (outputs/<extractor>/scores.json) and rebuild results.

Run with the vision venv and the orchid-pipeline package on PYTHONPATH:
    PYTHONPATH=~/projects/orchid-pipeline/src .venv-vision/bin/python bench/baselines/run_all.py
The AI-reader stages (bench/ai/prepare.py, run_gemini.py, map_outputs.py) spend money and are run separately with
~/orchid-py313/bin/python; their outputs are scored here when present."""
import subprocess
import sys
from pathlib import Path

B = Path(__file__).resolve().parents[1]
PY = sys.executable
steps = ["baselines/oracle.py", "baselines/ocr_cache.py", "baselines/text_regex.py", "baselines/tables_pdfplumber.py",
         "baselines/vector_figures.py", "baselines/assessed_pd_raster.py",
         # OpenCV / Camelot routes (.venv-vision)
         "baselines/ocr_pdf.py", "baselines/tdb_topology.py", "baselines/raster_pd.py", "baselines/tables_camelot.py",
         "baselines/tables_ocr.py", "baselines/raster_xy.py"]
for s in steps:
    print("==", s, flush=True)
    subprocess.run([PY, str(B / s)], check=True, cwd=B, stdout=subprocess.DEVNULL)
EXTRACTORS = ["oracle", "text_regex", "tables_pdfplumber", "vector_route", "vector_markers_tool", "assessed_pd",
              "assessed_pd_auto", "raster_topology", "camelot_stream", "camelot_lattice", "ocr_table_cells", "raster_markers",
              "ai_gemini", "ai_gemini_subset", "ai_claude", "ai_dual"]
for e in EXTRACTORS:
    if (B / "outputs" / e).exists():
        subprocess.run([PY, str(B / "score.py"), "--outputs", str(B / "outputs" / e)], check=True, cwd=B, stdout=subprocess.DEVNULL)
subprocess.run([PY, str(B / "results" / "build_results.py")], check=True, cwd=B)
