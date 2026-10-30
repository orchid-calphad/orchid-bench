#!/usr/bin/env python3
"""Collect outputs/<extractor>/scores.json into results/results.json and results/RESULTS.md (task x rendering):
headline comparison (previous no-OpenCV baselines vs OpenCV/Camelot routes vs AI readers vs dual-reader agreement),
AI spend from the Gemini ledger, then the full per-extractor tables and tool status notes.
"""

import json
import sys
from pathlib import Path

B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B))
from common import load_json, write_json  # noqa: E402

R = ["vector", "clean_raster", "degraded_raster", "image_only_pdf"]
RN = {
    "vector": "vector PDF",
    "clean_raster": "clean raster 300 dpi",
    "degraded_raster": "degraded raster 150 dpi",
    "image_only_pdf": "image-only PDF",
}
PD = [
    "diagrams_with_output",
    "boundary_precision@0.01",
    "boundary_recall@0.01",
    "boundary_hausdorff95",
    "boundary_median_dx",
    "boundary_median_dT_K",
    "field_pred",
    "field_precision",
    "field_recall",
    "invariant_recall",
    "invariant_precision",
    "invariant_T_err_K",
    "invariant_type_accuracy",
]
TAB = ["tables_true", "tables_pred", "cells", "cell_exact", "cell_exact_normalised", "numeric_exact"]
INV = [
    "rows_true",
    "rows_pred",
    "row_recall",
    "row_precision",
    "T_err_K_median",
    "T_printed_exact",
    "x_err_median",
    "x_printed_exact",
    "type_accuracy",
    "evidence_accuracy",
    "citation_accuracy",
]
PHS = [
    "rows_true",
    "rows_pred",
    "row_recall",
    "row_precision",
    "structure_accuracy",
    "pearson_accuracy",
    "strukturbericht_accuracy",
    "prototype_accuracy",
    "T_low_err_K_median",
    "T_high_err_K_median",
    "homogeneity_x_err_median",
    "homogeneity_T_err_K_median",
]
THT = [
    "records_true",
    "records_pred",
    "record_recall",
    "record_precision",
    "value_exact",
    "rel_err_median",
    "within_1pct",
    "unit_accuracy",
    "basis_accuracy",
]
XY = [
    "panels_true",
    "panels_found",
    "point_recall@0.01",
    "marker_recall@0.01",
    "point_err_median",
    "series_recall",
    "series_precision",
]
TXT = ["values_true", "values_pred", "precision", "recall", "value_only_recall"]
PF = [
    "phase_fraction_panels",
    "phase_fraction_panels_found",
    "phase_fraction_point_recall@0.01",
    "phase_fraction_point_err_median",
]

# (task, extractor, family, metric keys); family: prev = no-OpenCV baselines (earlier run), det = OpenCV/Camelot routes
# (this run), ai = single AI reader
DETAIL = [
    ("tables", "tables_pdfplumber", "prev", TAB),
    ("tables", "camelot_stream", "det", TAB),
    ("tables", "camelot_lattice", "det", TAB),
    ("tables", "ocr_table_cells", "det", TAB),
    ("invariant_table", "tables_pdfplumber", "prev", ["cells", "cell_exact", "numeric_exact"] + INV),
    ("invariant_table", "camelot_stream", "det", INV),
    ("invariant_table", "camelot_lattice", "det", INV),
    ("invariant_table", "ocr_table_cells", "det", INV),
    ("invariant_table", "ai_gemini", "ai", INV),
    ("phase_stability", "tables_pdfplumber", "prev", PHS),
    ("phase_stability", "camelot_stream", "det", PHS),
    ("phase_stability", "camelot_lattice", "det", PHS),
    ("phase_stability", "ocr_table_cells", "det", PHS),
    ("phase_stability", "ai_gemini", "ai", PHS),
    ("phase_stability", "vector_route", "prev", PF),
    ("thermo_tables", "tables_pdfplumber", "prev", THT),
    ("thermo_tables", "camelot_stream", "det", THT),
    ("thermo_tables", "camelot_lattice", "det", THT),
    ("thermo_tables", "ocr_table_cells", "det", THT),
    ("thermo_tables", "ai_gemini", "ai", THT),
    ("xy", "vector_route", "prev", XY),
    ("xy", "vector_markers_tool", "prev", XY),
    ("xy", "raster_markers", "det", XY),
    ("phase_diagram", "vector_route", "prev", PD),
    ("phase_diagram", "assessed_pd", "prev", PD),
    ("phase_diagram", "assessed_pd_auto", "det", PD),
    ("phase_diagram", "raster_topology", "det", PD),
    ("phase_diagram", "ai_gemini", "ai", PD),
    ("text", "text_regex", "prev", TXT),
    ("text", "ai_gemini", "ai", TXT),
]
SUBSET = ["ai_gemini_subset", "ai_claude", "ai_dual"]
SUB_TASKS = [
    (
        "invariant_table",
        ["row_recall", "row_precision", "T_printed_exact", "x_printed_exact", "type_accuracy", "evidence_accuracy"],
    ),
    ("phase_stability", ["row_recall", "row_precision", "structure_accuracy"]),
    ("thermo_tables", ["record_recall", "record_precision", "value_exact", "unit_accuracy"]),
    ("phase_diagram", ["field_pred", "field_precision", "field_recall"]),
    ("text", ["values_pred", "precision", "recall"]),
]
HEAD = [
    (
        "tables",
        "table cells exact",
        "cell_exact",
        [("prev", "tables_pdfplumber"), ("det", "camelot_stream"), ("det", "camelot_lattice"), ("det", "ocr_table_cells")],
    ),
    (
        "invariant_table",
        "invariant rows recall",
        "row_recall",
        [("prev", "tables_pdfplumber"), ("det", "camelot_stream"), ("det", "ocr_table_cells"), ("ai", "ai_gemini")],
    ),
    (
        "phase_stability",
        "phase rows recall",
        "row_recall",
        [("prev", "tables_pdfplumber"), ("det", "camelot_stream"), ("det", "ocr_table_cells"), ("ai", "ai_gemini")],
    ),
    (
        "phase_stability",
        "structure labels correct",
        "structure_accuracy",
        [("prev", "tables_pdfplumber"), ("det", "camelot_stream"), ("det", "ocr_table_cells"), ("ai", "ai_gemini")],
    ),
    (
        "thermo_tables",
        "property records recall",
        "record_recall",
        [("prev", "tables_pdfplumber"), ("det", "camelot_stream"), ("det", "ocr_table_cells"), ("ai", "ai_gemini")],
    ),
    (
        "xy",
        "marker recall@0.01",
        "marker_recall@0.01",
        [("prev", "vector_route"), ("prev", "vector_markers_tool"), ("det", "raster_markers")],
    ),
    (
        "phase_diagram",
        "boundary recall@0.01",
        "boundary_recall@0.01",
        [("prev", "vector_route"), ("prev", "assessed_pd"), ("det", "assessed_pd_auto")],
    ),
    (
        "phase_diagram",
        "field precision",
        "field_precision",
        [
            ("prev", "vector_route"),
            ("prev", "assessed_pd"),
            ("det", "assessed_pd_auto"),
            ("det", "raster_topology"),
            ("ai", "ai_gemini"),
        ],
    ),
    (
        "phase_diagram",
        "field recall",
        "field_recall",
        [
            ("prev", "vector_route"),
            ("prev", "assessed_pd"),
            ("det", "assessed_pd_auto"),
            ("det", "raster_topology"),
            ("ai", "ai_gemini"),
        ],
    ),
    (
        "phase_diagram",
        "invariant recall",
        "invariant_recall",
        [("prev", "vector_route"), ("prev", "assessed_pd"), ("det", "assessed_pd_auto"), ("det", "raster_topology")],
    ),
    ("text", "text values recall", "recall", [("prev", "text_regex"), ("ai", "ai_gemini")]),
    ("text", "text values precision", "precision", [("prev", "text_regex"), ("ai", "ai_gemini")]),
]


FINDINGS = """## Findings (run of 2026-10-22, Mac Studio)

- **Tables, born-digital.** Camelot stream through the thermo_extractor contract (`extract_assessment_table`, whole-page
  profile) recovers the property tables best (records recall 0.94, all values exact) but splits the phase table (rows
  0.18); pdfplumber stays best on cells (0.76). Camelot lattice only finds fully ruled tables (cells 0.33). The tool's
  `network` flavor is not available in camelot-py 0.11.0 (every page raises NotImplementedError).
- **Tables, rasters.** `ocr_table_cells` with an oracle-declared layout reads 45 % of cells exactly on clean pages and
  16–17 % on degraded / image-only pages. Invariant rows are never recovered (Tesseract does not return the reaction
  arrow, so the row parser finds no reaction).
- **Phase diagrams.** `extract_pdf_figure_panels` finds Fig. 1 in 20/20 vector PDFs, 18/20 clean and 16/20 degraded /
  image-only pages once a Tesseract text layer is added (image_only.pdf alone: 0 captions, 0 panels). With that crop and
  `detect_binary_axes`, assessed_pd needs the truth calibration in 1/20 vector, 3/20 clean and 11/20 degraded diagrams
  (earlier: 2, 14 and 16 with the truth crop) and scores the same as before. `detect_binary_axes` calibrates 16/20
  vector and clean crops but only 9/20 degraded ones. The blind topology extractor finds invariants (recall 0.35–0.53)
  but no boundaries. `compare_binary_topology` returns `proposal_conflicts_with_database` for 19/20 diagrams on every
  rendering, so it cannot separate good extractions from bad ones on this set.
- **XY markers on rasters.** The OpenCV template detector, given the vector-twin manifests, matches the vector route on
  clean pages (marker recall 0.74) and drops to 0.62 on degraded pages. 32 of 73 panels have no manifest (calibration
  or legend failed on the vector route), so they are never attempted.
- **AI reader, Gemini 3.8 Flash, production prompts.** The invariant-table reader (inv-table-v5) is the strongest
  extractor on every rendering: rows recall 0.99 on vector and clean, 0.85 on degraded, with precision, printed T and
  evidence 1.00, type 0.99–1.00 and printed compositions 0.83–0.84. Running text (thermo-text-v6.1-quote with the quote linker) is precise (0.96–1.00) but
  reads only the paragraphs the production filter selects (recall 0.27–0.53, against 0.74–0.97 for the regex).
  Phase-field naming (open-field reader on the assessed_pd open fields) raises field recall from 0.07–0.26 to
  0.36–0.45 at precision 0.64–0.79.
- **Failure: thermo-table-v7.1 with Flash.** With the v7.1 schema, Flash returns one stub row per table (no
  properties, no structure fields), so phase-table structure accuracy is 0 and property records recall is about 0.
  A 24-call ablation on the two pilot documents with the same crops confirms the schema is the cause: prompt v5 and v6
  give property records recall 1.00 and phase rows recall 1.00 on vector (degraded: records 0.15 / 0.46, phase rows 0.33 / 1.00). Claude, given
  the same v7.1 requests, transcribes these tables fully (phase rows 1.00, structures 1.00, records 1.00 on vector,
  0.86 on degraded).
- **Dual reader (5 documents).** Agreement keeps every Gemini invariant row on vector (precision 1.00) and lifts
  degraded phase-field precision from 0.60 (Gemini) and 0.75 (Claude) to 0.82. Because agreement requires Gemini, it
  inherits Gemini's empty v7.1 tables.
- **Raster crop artefact.** On some degraded pages the caption finder (Tesseract lines starting with "Table N")
  probably took a sentence that mentions "Table 1" for a caption; the resulting crop shows no table (Claude flagged 4 such requests). This lowers degraded invariant recall for both
  readers on the subset.

"""
FINDINGS = FINDINGS.splitlines()


def f(v):
    """Format one metric value for a Markdown table cell."""
    if v is None:
        return "–"
    if isinstance(v, float) and 0 < abs(v) < 0.001:
        return f"{v:.1e}"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def agg(ext):
    """Return the aggregate scores of one extractor, or an empty dict when it has none."""
    p = B / "outputs" / ext / "scores.json"
    return load_json(p)["aggregate"] if p.exists() else {}


def spend():
    """Summarise the AI spend recorded in the Gemini ledger."""
    led = B / "work" / "ai" / "gemini_ledger.jsonl"
    rows = [json.loads(x) for x in led.read_text(encoding="utf-8").splitlines()] if led.exists() else []
    by = {}
    for r in rows:
        u = r.get("usage") or {}
        k = by.setdefault(r["kind"], {"calls": 0, "errors": 0, "input": 0, "output": 0, "thinking": 0})
        k["calls"] += 1
        k["errors"] += int(bool(r.get("error")))
        k["input"] += u.get("prompt_token_count") or 0
        k["output"] += u.get("candidates_token_count") or 0
        k["thinking"] += u.get("thoughts_token_count") or 0
    tot = {key: sum(v[key] for v in by.values()) for key in ("calls", "errors", "input", "output", "thinking")}
    for v in list(by.values()) + [tot]:
        v["usd_list"] = round((v["input"] * 0.75 + (v["output"] + v["thinking"]) * 3.75) / 1e6, 4)
        v["usd_conservative"] = round((v["input"] * 1.50 + (v["output"] + v["thinking"]) * 7.50) / 1e6, 4)
    return by, tot


def main():
    """Write results/results.json and results/RESULTS.md from the per-extractor scores."""
    res = {"headline": [], "tasks": [], "ai_subset": [], "ai_spend": {}, "extractor_status": {}}
    md = [
        "# Benchmark results",
        "",
        "Generated by `bench/results/build_results.py` from `bench/outputs/*/scores.json` (truth schema v2). Distances are "
        "in axis-normalised units (axis range = 1); `@0.01` = within 1 % of the axis range; `–` = not applicable or nothing "
        "to score; `n/r` = not run for that rendering (reasons below). Semantic table errors are measured against the printed "
        "value. Families: **prev** = the earlier no-OpenCV baselines; **det** = thermo_extractor routes that needed "
        "OpenCV / Camelot, run in `.venv-vision` (this run); **ai** = production AI readers (prompts and schemas imported "
        "from the sgte-corpus release), gemini-3.8-flash, single reader; dual-reader results are on a 5-document subset "
        "(next section). Findings and failures: next section.",
        "",
    ]
    md += FINDINGS
    md += [
        "## Headline: task × rendering",
        "",
        "| task / metric | extractor | family | " + " | ".join(RN[r] for r in R) + " |",
        "|---|---|---|" + "---|" * len(R),
    ]
    for task, label, metric, exts in HEAD:
        for fam, ext in exts:
            a = agg(ext)
            vals = [(a.get(r) or {}).get(task, {}).get(metric) if (a.get(r) or {}).get(task) else None for r in R]
            present = [bool((a.get(r) or {}).get(task)) for r in R]
            md.append(f"| {label} | `{ext}` | {fam} | " + " | ".join(f(v) if p else "n/r" for v, p in zip(vals, present)) + " |")
            res["headline"].append(
                {
                    "task": task,
                    "metric": metric,
                    "extractor": ext,
                    "family": fam,
                    "by_rendering": {r: (v if p else "n/r") for r, v, p in zip(R, vals, present)},
                }
            )
    md += [
        "",
        "Oracle inputs still used by the det routes (recorded per document in each output's `status`): "
        "`ocr_table_cells` region and column boundaries (the tool requires them declared); `raster_markers` manifests "
        "(calibration, legend marker) from the vector twin, mapped into degraded pages with the truth page transform; "
        "`assessed_pd_auto` phase table; truth-frame calibration in `raster_topology` / `assessed_pd_auto` only where every "
        "detector failed (counts under Tool status notes). Their figure crop comes from `extract_pdf_figure_panels` "
        "(truth bbox only where no panel was found).",
        "",
    ]
    md += [
        "## AI readers: single reader vs dual-reader agreement (subset)",
        "",
        "Subset: syn-AgCu-06He, syn-AlNi-14Zhu, syn-CuZn-08Gie, syn-PbZn-98Jan, syn-BiSn-08Li; vector PDF and degraded "
        "raster. `ai_gemini_subset` = the Gemini answers on the subset; `ai_claude` = Claude (claude-opus-5.5) blind reader "
        "answering the same request files (prompt, schema, crop) through a file broker (2 sub-agents); `ai_dual` = "
        "production consensus (a record is accepted only when both readers give it; phase-diagram fields through the "
        "assessed_pd ladder pair tier and its rule gate).",
        "",
        "| task | metric | reader | vector PDF | degraded raster 150 dpi |",
        "|---|---|---|---|---|",
    ]
    for task, keys in SUB_TASKS:
        for k in keys:
            for ext in SUBSET:
                a = agg(ext)
                vals = [(a.get(r) or {}).get(task, {}).get(k) for r in ("vector", "degraded_raster")]
                md.append(f"| {task} | {k} | `{ext}` | " + " | ".join(f(v) for v in vals) + " |")
                res["ai_subset"].append({"task": task, "metric": k, "reader": ext, "vector": vals[0], "degraded_raster": vals[1]})
    by, tot = spend()
    res["ai_spend"] = {
        "model": "gemini-3.8-flash",
        "by_kind": by,
        "total": tot,
        "price_usd_per_million": {
            "list": {"input": 0.75, "output_incl_thinking": 3.75},
            "conservative": {"input": 1.50, "output_incl_thinking": 7.50},
        },
        "price_source": "third-party price trackers, Oct 2026 (list); conservative = 2x list. Not confirmed on Google's page.",
        "claude_subset": "answered by Claude Code sub-agents; no separate API billing recorded",
    }
    md += [
        "",
        "## AI spend (Gemini 3.8 Flash)",
        "",
        "Token counts from each response's `usage_metadata` (`work/ai/gemini_ledger.jsonl`, copied to "
        "`results/gemini_ledger.jsonl`). Prices per million tokens: list $0.75 input / $3.75 output incl. thinking "
        "(third-party price trackers, Oct 2026; not confirmed on Google's page); conservative = 2× list. Hard cap US$25; "
        "the runner's guard stops new calls at a conservative $22.",
        "",
        "| reader request | calls | errors | input tok | output tok | thinking tok | US$ list | US$ conservative |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, v in sorted(by.items()):
        md.append(
            f"| {k} | {v['calls']} | {v['errors']} | {v['input']:,} | {v['output']:,} | {v['thinking']:,} | {v['usd_list']:.2f} | {v['usd_conservative']:.2f} |"
        )
    md.append(
        f"| **total** | {tot['calls']} | {tot['errors']} | {tot['input']:,} | {tot['output']:,} | {tot['thinking']:,} | "
        f"**{tot['usd_list']:.2f}** | {tot['usd_conservative']:.2f} |"
    )
    md.append("")
    md += ["## Per-extractor detail", ""]
    for task, ext, fam, keys in DETAIL:
        a = agg(ext)
        row = {"task": task, "extractor": ext, "family": fam, "by_rendering": {}}
        md += [
            f"### {task} — `{ext}` ({fam})",
            "",
            "| metric | " + " | ".join(RN[r] for r in R) + " |",
            "|---|" + "---|" * len(R),
        ]
        for r in R:
            v = (a.get(r) or {}).get(task)
            row["by_rendering"][r] = v if v else "n/r"
        for k in keys:
            md.append(
                f"| {k} | "
                + " | ".join(f(row["by_rendering"][r].get(k)) if isinstance(row["by_rendering"][r], dict) else "n/r" for r in R)
                + " |"
            )
        md.append("")
        res["tasks"].append(row)
    md += [
        "### n/r reasons",
        "",
        "- `camelot_*`: Camelot reads PDF text objects; the raster renderings have none. The tool contract "
        "(`thermo_extractor.tables.extraction`) allows `stream` and `network`; camelot-py 0.11.0 has no `network` flavor "
        "(NotImplementedError on every page, so no `camelot_network` result); `lattice` was called directly with a "
        "pdftoppm conversion backend (Ghostscript and the pdftopng wheel are not installed).",
        "- `tables_pdfplumber`, `text_regex`, `vector_route`, `vector_markers_tool`: as in the earlier run (text layer / "
        "PDF vector objects).",
        "- `raster_markers` vector: covered by the vector route. `raster_topology` emits no boundary geometry (faces and "
        "invariant horizontals only).",
        "- phase-fraction curves on rasters: thermo_extractor has no raster curve tracer.",
        "- `ai_*` cell metrics: the readers return records, not cell grids.",
        "",
    ]
    md += ["## Tool status notes", ""]
    for ext in ("vector_route", "assessed_pd", "assessed_pd_auto", "raster_topology", "raster_markers"):
        counts = {}
        for fp in sorted((B / "outputs" / ext).glob("*/*.json")):
            st = load_json(fp).get("status", {})
            for k, v in st.items():
                for it in v if isinstance(v, list) else [v]:
                    if not isinstance(it, dict):
                        continue
                    for key in ("calibration", "crop", "manifest", "vector_markers", "markers", "compare", "error"):
                        if key in it:
                            msg = str(it[key])
                            msg = msg.split(" (", maxsplit=1)[0] if key in ("vector_markers", "markers") else msg[:110]
                            counts[(fp.parent.name, k, key, msg)] = counts.get((fp.parent.name, k, key, msg), 0) + 1
        res["extractor_status"][ext] = [
            {"rendering": a_, "task": b_, "stage": c_, "outcome": d_, "count": n}
            for (a_, b_, c_, d_), n in sorted(counts.items())
        ]
        if counts:
            md += (
                [f"### {ext}", ""]
                + [f"- {RN.get(a_, a_)} / {b_} / {c_}: {d_} — {n}" for (a_, b_, c_, d_), n in sorted(counts.items())]
                + [""]
            )
    oracle = agg("oracle")
    md += [
        "## Oracle self-test",
        "",
        "The truth written back in extractor format scores 1.0 on every precision/recall metric:",
        "",
    ]
    for r in R:
        o = oracle.get(r, {})
        md.append(
            f"- {RN[r]}: tables {f(o.get('tables', {}).get('cell_exact'))}, xy {f(o.get('xy', {}).get('point_recall@0.01'))}, "
            f"boundaries {f(o.get('phase_diagram', {}).get('boundary_recall@0.01'))}, fields {f(o.get('phase_diagram', {}).get('field_recall'))}, "
            f"text {f(o.get('text', {}).get('recall'))}, invariant rows {f(o.get('invariant_table', {}).get('row_recall'))}, "
            f"phase rows {f(o.get('phase_stability', {}).get('row_recall'))}, property records {f(o.get('thermo_tables', {}).get('record_recall'))}"
        )
    write_json(B / "results" / "results.json", res, pretty=True)
    (B / "results" / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:70]))


if __name__ == "__main__":
    main()
