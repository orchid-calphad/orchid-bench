#!/usr/bin/env python3
"""Run every baseline, then score each one (outputs/<extractor>/scores.json)."""
import subprocess
import sys
from pathlib import Path

B = Path(__file__).resolve().parents[1]
PY = sys.executable
steps = ["baselines/oracle.py", "baselines/ocr_cache.py", "baselines/text_regex.py", "baselines/tables_pdfplumber.py",
         "baselines/vector_figures.py", "baselines/assessed_pd_raster.py"]
for s in steps:
    print("==", s, flush=True)
    subprocess.run([PY, str(B / s)], check=True, cwd=B, stdout=subprocess.DEVNULL)
for e in ["oracle", "text_regex", "tables_pdfplumber", "vector_route", "vector_markers_tool", "assessed_pd"]:
    subprocess.run([PY, str(B / "score.py"), "--outputs", str(B / "outputs" / e)], check=True, cwd=B, stdout=subprocess.DEVNULL)
subprocess.run([PY, str(B / "results" / "build_results.py")], check=True, cwd=B)
