#!/usr/bin/env python3
"""Diagnostic ablation (small spend): the same table crops of the two pilot documents (vector, degraded) re-asked with
the earlier production thermo-table prompts v5 (thermo-table-v5) and v6 (thermo-table-v6-units), to test whether the
near-empty v7.1 answers of gemini-3.8-flash come from the v7.1 schema. Writes work/ai/jobs_ablation/."""
import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from common import ORCHID_PIPELINE  # noqa: E402
sys.path.insert(0, str(ORCHID_PIPELINE / "pipelines/sgte-corpus/tools-release-v2.1.18"))
import hashlib  # noqa: E402
import thermo_table_reader as T  # noqa: E402

AI = BENCH / "work" / "ai"
OUT = AI / "jobs_ablation"
OUT.mkdir(parents=True, exist_ok=True)
for f in sorted((AI / "jobs").glob("*__thermo_table__*.json")):
    j = json.loads(f.read_text())
    if j["doc"] not in ("syn-AgCu-06He-v0", "syn-AlNi-14Zhu-v0") or j["rendering"] not in ("vector", "degraded_raster"):
        continue
    head = j["prompt"].split("Classify the table")[0]
    for ver, tmpl, sch in (("v5", T.PROMPT, T.SCHEMA), ("v6", T.PROMPT_V6, T.SCHEMA_V6)):
        prompt = head + tmpl[tmpl.index("Classify the table"):]
        key = hashlib.sha256(json.dumps([ver, j["key"], prompt], sort_keys=True).encode()).hexdigest()[:24]
        k = {**j, "key": key, "kind": f"thermo_table_{ver}", "prompt": prompt, "schema": sch,
             "meta": {**j["meta"], "prompt_version": {"v5": T.PROMPT_VERSION, "v6": T.PROMPT_VERSION_V6}[ver], "ablation_of": j["key"]}}
        (OUT / f"{f.stem}__{ver}.json").write_text(json.dumps(k, ensure_ascii=False, indent=1))
print(len(list(OUT.glob("*.json"))), "ablation jobs")
