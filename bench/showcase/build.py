#!/usr/bin/env python3
"""Self-contained showcase: three example documents with page image, phase-diagram overlay (truth vs extracted, SVG)
and per-rendering metrics. Images are embedded as data URIs; all dynamic content is built with DOM APIs.
"""

import base64
import io
import json
import os
import sys
from pathlib import Path

B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B))
from common import docs_dir, load_json, truth_dir  # noqa: E402
from score import _segments, aabb, align_segment, norm_text  # noqa: E402

EXAMPLES = ["syn-PbSn-17Gie-v0", "syn-AlCu-02Mie-v0", "syn-AgNi-04Din-v0"]
NAMING = [("syn-AgCu-06He-v0", "vector"), ("syn-PbZn-98Jan-v0", "degraded_raster")]
R = ["vector", "clean_raster", "degraded_raster", "image_only_pdf"]


def img_uri(im, fmt="JPEG", q=80):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format=fmt, quality=q)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def doc_payload(doc):
    from PIL import Image

    t = load_json(truth_dir / f"{doc}.json")
    fig = next(f for f in t["figures"] if f["kind"] == "phase_diagram")
    main = next(a for a in fig["axes"] if a["role"] == "main")
    page = fig["page"]
    full = Image.open(docs_dir / doc / "clean" / f"page-{page + 1}.png")
    fb = aabb(fig["corners"]["clean_raster"])
    crop = full.crop(tuple(int(v) for v in fb))
    scale = 900 / crop.width
    crop = crop.resize((900, int(crop.height * scale)))
    thumb = full.copy()
    thumb.thumbnail((620, 800))
    deg = Image.open(docs_dir / doc / "degraded" / f"page-{page + 1}.jpg")
    deg.thumbnail((620, 800))
    ab = aabb(main["corners"]["clean_raster"])
    X0, Y0, X1, Y1 = [(ab[0] - fb[0]) * scale, (ab[1] - fb[1]) * scale, (ab[2] - fb[0]) * scale, (ab[3] - fb[1]) * scale]
    xs_ = fig["units"]["x_scale_factor"]
    off = 273.15 if main["y"]["unit"] == "°C" else 0.0

    def to_px(x, T_K):
        u = (x * xs_ - main["x"]["min"]) / (main["x"]["max"] - main["x"]["min"])
        v = (T_K - off - main["y"]["min"]) / (main["y"]["max"] - main["y"]["min"])
        return [round(X0 + u * (X1 - X0), 1), round(Y1 - v * (Y1 - Y0), 1)]

    def clip(pts):
        return [p for p in pts if X0 - 2 <= p[0] <= X1 + 2 and Y0 - 2 <= p[1] <= Y1 + 2]

    truth_lines = [clip([to_px(x, T) for x, T in zip(b["x"], b["T_K"])]) for b in fig["boundaries"] if not b["on_frame"]]
    truth_lines += [clip([to_px(il["x_min"], il["T_K"]), to_px(il["x_max"], il["T_K"])]) for il in fig["invariant_lines"]]
    truth_fields = [
        {"p": to_px(*f["point"]), "label": " + ".join(f["labels"])} for f in t["phase_diagram"]["fields"] if f["n_cells"] >= 6
    ]
    ext = {}
    for name, rend in (
        ("assessed_pd · clean raster", ("assessed_pd", "clean_raster")),
        ("assessed_pd · degraded raster", ("assessed_pd", "degraded_raster")),
        ("vector route · vector PDF", ("vector_route", "vector")),
        ("assessed_pd_auto (OpenCV crop/axes) · degraded", ("assessed_pd_auto", "degraded_raster")),
        ("AI reader (Gemini) fields · degraded", ("ai_gemini", "degraded_raster")),
    ):
        p = B / "outputs" / rend[0] / rend[1] / f"{doc}.json"
        if not p.exists():
            continue
        o = load_json(p)
        pds = o.get("phase_diagrams") or []
        if not pds:
            continue
        pd = pds[0]
        cx = (lambda v: v / 100) if pd["x_unit"] in ("at.%",) else (lambda v: v)
        cT = (lambda v: v + 273.15) if pd["T_unit"] == "°C" else (lambda v: v)
        pts = []
        for b in pd["boundaries"]:
            q = [to_px(cx(x), cT(T)) for x, T in b["points"]]
            pts.append(clip(q) if b.get("polyline") else clip(q))
        ext[name] = {
            "polyline": any(b.get("polyline") for b in pd["boundaries"]),
            "boundaries": pts,
            "fields": [
                {"p": to_px(cx(f["point"][0]), cT(f["point"][1])), "label": " + ".join(f["phases"])}
                for f in pd["fields"]
                if f.get("phases")
            ],
            "status": o.get("status", {}).get("phase_diagram"),
        }
    metrics = {}
    for e in (
        "assessed_pd",
        "assessed_pd_auto",
        "raster_topology",
        "vector_route",
        "text_regex",
        "tables_pdfplumber",
        "camelot_stream",
        "ocr_table_cells",
        "raster_markers",
        "ai_gemini",
    ):
        sp = B / "outputs" / e / "scores.json"
        if not sp.exists():
            continue
        per = load_json(sp)["per_document"].get(doc, {})
        for r in R:
            m = per.get(r)
            if not m:
                continue
            row = metrics.setdefault(r, {})
            if "phase_diagram" in m and m["phase_diagram"].get("found"):
                pdm = m["phase_diagram"]
                row[f"{e}: boundary recall@0.01"] = (pdm.get("boundary") or {}).get("recall@0.01")
                row[f"{e}: boundary precision@0.01"] = (pdm.get("boundary") or {}).get("precision@0.01")
                row[f"{e}: field precision"] = (pdm.get("fields") or {}).get("precision")
                row[f"{e}: invariant recall"] = (pdm.get("invariants") or {}).get("recall")
            for key, lab in (
                ("invariant_table", "invariant rows recall"),
                ("phase_stability", "phase rows recall"),
                ("thermo_tables", "property records recall"),
            ):
                sub = {"invariant_table": "invariant_rows", "phase_stability": "phase_rows", "thermo_tables": "property_records"}[
                    key
                ]
                if sub in m and m[sub].get("rows_true", m[sub].get("records_true")):
                    n_t = m[sub].get("rows_true", m[sub].get("records_true"))
                    row[f"{e}: {lab}"] = m[sub]["matched"] / n_t if n_t else None
            if "text" in m:
                row[f"{e}: text value recall"] = m["text"]["recall"]
                row[f"{e}: text value precision"] = m["text"]["precision"]
            if "tables" in m:
                row[f"{e}: table cell exact"] = m["tables"].get("cell_exact")
            if "xy" in m:
                row[f"{e}: xy panels found"] = f"{m['xy']['panels_found']}/{m['xy']['panels_true']}"
    st = t["style"]["phase_diagram"]
    return {
        "doc": doc,
        "system": t["system"],
        "tdb": t["source_tdb"]["file"],
        "tdb_sha256": t["source_tdb"]["sha256"],
        "types": t["diagram_types"],
        "pages": t["renderings"]["vector"]["pages"],
        "style": {k: st[k] for k in ("T_unit", "x_unit", "color", "field_labels", "markers", "overlay", "inset", "linewidth")},
        "page_img": img_uri(thumb),
        "deg_img": img_uri(deg, q=70),
        "fig_img": img_uri(crop, q=85),
        "fig_size": [crop.width, crop.height],
        "truth": {"lines": truth_lines, "fields": truth_fields},
        "extracted": ext,
        "metrics": metrics,
        "invariants": [{"T": i["T"], "type": i["type"], "reaction": i["reaction"]} for i in t["phase_diagram"]["invariants"]],
    }


def table_payload(doc, table_id):
    """Truth cells vs pdfplumber cells (best-aligned per segment) for one table of one document."""
    from PIL import Image

    t = load_json(truth_dir / f"{doc}.json")
    tb = next(x for x in t["tables"] if x["id"] == table_id)
    o = load_json(B / "outputs" / "tables_pdfplumber" / "vector" / f"{doc}.json")
    segs = []
    for si, tc in enumerate(_segments(tb)):
        best = align_segment(tc, o.get("tables", []))
        nrow = max(r for r, _ in tc) + 1
        ncol = tb["n_cols"]
        grid = []
        for r in range(nrow):
            row = []
            c = 0
            while c < ncol:
                cell = tc.get((r, c))
                span = (cell or {}).get("colspan", 1)
                got = best[1].get((r + best[2], c + best[3])) if best else None
                txt = cell["text"] if cell else ""
                row.append(
                    {
                        "truth": txt,
                        "got": got,
                        "span": span,
                        "header": bool(cell and cell["header"]),
                        "ok": (norm_text(got) == norm_text(txt)) if txt else None,
                    }
                )
                c += span
            grid.append(row)
        sg = tb["segments"][si]
        page = Image.open(docs_dir / doc / "clean" / f"page-{sg['page'] + 1}.png")
        bb = aabb(sg["corners"]["clean_raster"])
        crop = page.crop((int(bb[0]) - 20, int(bb[1]) - 60, int(bb[2]) + 20, int(bb[3]) + 20))
        crop.thumbnail((1100, 900))
        route = o["tables"][best[4]]["route"] if best else None
        segs.append({"grid": grid, "img": img_uri(crop, q=85), "route": route, "page": sg["page"] + 1})
    n_ok = sum(1 for sgm in segs for r in sgm["grid"] for c in r if c["ok"])
    n = sum(1 for sgm in segs for r in sgm["grid"] for c in r if c["ok"] is not None)
    return {
        "doc": doc,
        "system": t["system"],
        "table": table_id,
        "caption": tb["caption"],
        "style": tb["style"],
        "footnotes": tb["footnotes"],
        "segments": segs,
        "cell_exact": n_ok / n if n else None,
    }


HEADLINE = [
    ("invariant_table", "row_recall", "invariant-table rows recall"),
    ("phase_stability", "row_recall", "phase-table rows recall"),
    ("phase_stability", "structure_accuracy", "structure labels correct"),
    ("thermo_tables", "record_recall", "property records recall"),
    ("phase_diagram", "field_precision", "phase-field names precision"),
    ("phase_diagram", "field_recall", "phase-field names recall"),
    ("phase_diagram", "invariant_recall", "invariant lines recall"),
    ("xy", "marker_recall@0.01", "XY markers recall@0.01"),
    ("text", "recall", "running-text values recall"),
    ("tables", "cell_exact", "table cells exact"),
]
FAMS = {
    "prev": ["tables_pdfplumber", "vector_route", "vector_markers_tool", "assessed_pd", "text_regex"],
    "det": ["camelot_stream", "camelot_lattice", "ocr_table_cells", "assessed_pd_auto", "raster_topology", "raster_markers"],
    "ai": ["ai_gemini"],
    "dual": ["ai_dual"],
}


def comparison_payload():
    aggs = {
        e: load_json(B / "outputs" / e / "scores.json")["aggregate"]
        for es in FAMS.values()
        for e in es
        if (B / "outputs" / e / "scores.json").exists()
    }
    rows = []
    for task, metric, label in HEADLINE:
        for r in R:
            cell = {"label": label, "rendering": r}
            for fam, es in FAMS.items():
                best = None
                for e in es:
                    v = ((aggs.get(e, {}).get(r) or {}).get(task) or {}).get(metric)
                    if v is not None and (best is None or v > best[1]):
                        best = (e, v)
                cell[fam] = best
            rows.append(cell)
    return rows


def naming_payload(doc, r):
    """AI naming of the open phase fields of one diagram vs the truth field at each marker."""
    import pickle

    import numpy as np
    from score import _canon, _label_sets

    sys.path.insert(
        0,
        os.environ.get(
            "ASSESSED_PD_SRC",
            "/private/tmp/claude-504/-Users-raymundo-arroyave-projects-open-tdb/"
            "5d97acc6-0c9e-4398-87a5-e3422e273f18/scratchpad/assessed-pd-annotator/src",
        ),
    )
    from assessed_pd.pipeline import annotate
    from PIL import Image

    t = load_json(truth_dir / f"{doc}.json")
    job = next((load_json(p) for p in (B / "work" / "ai" / "jobs").glob(f"{doc}__{r}__pd_fields__*.json")), None)
    if job is None:
        return None
    ga = B / "work" / "ai" / "gemini" / f"{job['key']}.json"
    ca = B / "work" / "ai" / "claude" / "answers" / f"{job['key']}.json"
    g = {a["marker"]: a for a in (load_json(ga)["result"]["answers"] if ga.exists() else [])}
    c = {a["marker"]: a for a in (load_json(ca).get("answers", []) if ca.exists() else [])}
    d = pickle.load(open(B / "work" / "raster_pd" / doc / r / "assessed_input.pkl", "rb"))
    ann = annotate(d["inp"], d["cfg"])
    pdt = t["phase_diagram"]
    fm = pdt["field_map"]
    grid = np.array([sum(([v] * n for v, n in row), []) for row in fm["rows"]])
    fields = {f_["id"]: f_ for f_ in pdt["fields"]}
    xs_ = 100.0 if ann.cal.composition_basis in ("atomic_percent", "at.%") else 1.0
    off = 273.15 if str(ann.cal.temperature_unit).lower() in ("degc", "°c", "c", "celsius") else 0.0
    rows = []
    for m in job["meta"]["markers"]:
        reg = ann.fields.regions[m["field_id"]]
        x, T = reg.label_point[0] / xs_, reg.label_point[1] + off
        i, j = int(round((T - fm["T0"]) / fm["dT"])), int(round((x - fm["x0"]) / fm["dx"]))
        fid = int(grid[i, j]) if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1] else -1
        tf = fields.get(fid)
        truth = " + ".join(tf["labels"]) if tf else "(outside / unknown)"

        def ok(ans):
            if not ans or ans in ("none of these", "cannot tell") or not tf:
                return None
            p = sorted(_canon(q) for q in ans.split("+"))
            conv, tdb, _ = _label_sets(tf)
            return p == conv or p == tdb

        ga_, ca_ = (g.get(m["marker"]) or {}).get("assemblage"), (c.get(m["marker"]) or {}).get("assemblage")
        rows.append(
            {
                "marker": m["marker"],
                "choices": [q for q in m["choices"] if q not in ("none of these", "cannot tell")],
                "gemini": ga_,
                "gemini_reason": (g.get(m["marker"]) or {}).get("reason"),
                "claude": ca_,
                "agree": bool(ga_ and ga_ == ca_ and ga_ not in ("none of these", "cannot tell")),
                "truth": truth,
                "ok_gemini": ok(ga_),
                "ok_claude": ok(ca_),
            }
        )
    im = Image.open(B / job["images"][0])
    im.thumbnail((1000, 900))
    return {
        "doc": doc,
        "system": t["system"],
        "rendering": r,
        "img": img_uri(im, q=85),
        "rows": rows,
        "printed_labels": t["style"]["phase_diagram"]["field_labels"],
        "has_claude": ca.exists(),
    }


def pick_tables():
    out = []
    for tf in sorted(truth_dir.glob("*.json")):
        t = load_json(tf)
        inv = next((x for x in t["tables"] if x["id"] == "invariants"), None)
        if inv and t["style"]["invariant_table"]["exp_column"] and inv["n_body_rows"] >= 4 and not out:
            out.append((t["doc_id"], "invariants"))
    for tf in sorted(truth_dir.glob("*.json")):
        t = load_json(tf)
        sp = [x for x in t["tables"] if len(x.get("segments") or []) > 1]
        if sp and t["doc_id"] != out[0][0]:
            out.append((t["doc_id"], sp[0]["id"]))
            break
    return out


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Synthetic Papers Showcase</title>
<style>
:root{--bg:#f7f7f5;--panel:#ffffff;--ink:#1d1f23;--muted:#5d636b;--line:#d9dbde;--truth:#1f6feb;--ext:#d1242f;--ext2:#8250df;--ext3:#1a7f37;--chip:#eef1f4;--ext4:#bc4c00;--ext5:#0a7d87}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15171a;--panel:#1d2024;--ink:#e6e8eb;--muted:#9aa1a9;--line:#33383e;--truth:#58a6ff;--ext:#ff7b72;--ext2:#d2a8ff;--ext3:#56d364;--chip:#262a30;--ext4:#ffa657;--ext5:#39c5cf}}
:root[data-theme="dark"]{--bg:#15171a;--panel:#1d2024;--ink:#e6e8eb;--muted:#9aa1a9;--line:#33383e;--truth:#58a6ff;--ext:#ff7b72;--ext2:#d2a8ff;--ext3:#56d364;--chip:#262a30;--ext4:#ffa657;--ext5:#39c5cf}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}h1{font-size:1.6rem;margin:0 0 4px}h2{font-size:1.25rem;margin:0 0 6px}
p.lead{color:var(--muted);margin:0 0 20px;max-width:75ch}.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin:0 0 24px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 12px}.chip{background:var(--chip);border-radius:999px;padding:2px 10px;font-size:.8rem;color:var(--muted)}
.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,2fr);gap:16px}@media (max-width:820px){.grid{grid-template-columns:minmax(0,1fr)}}
img.page{width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff}
.figwrap{position:relative;width:100%}.figwrap svg{width:100%;height:auto;display:block;border:1px solid var(--line);border-radius:6px;background:#fff}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:.85rem;margin:8px 0}.legend label{display:flex;align-items:center;gap:6px;cursor:pointer}
.sw{display:inline-block;width:18px;height:3px;border-radius:2px}
table{border-collapse:collapse;width:100%;font-size:.82rem}th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}td.n{text-align:right;font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto}details{margin-top:8px}summary{cursor:pointer;color:var(--muted)}small{color:var(--muted)}
.tabs{display:flex;gap:6px;margin:8px 0}.tabs button{background:var(--chip);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:3px 10px;cursor:pointer;font:inherit;font-size:.85rem}
.tabs button[aria-pressed="true"]{border-color:var(--truth)}
table.cmp td,table.cmp th{border:1px solid var(--line);vertical-align:top;font-size:.78rem;padding:3px 5px}
table.cmp .t{display:block}table.cmp .g{display:block;font-family:ui-monospace,Menlo,monospace;font-size:.72rem}
table.cmp td.ok .g{color:var(--ext3)}table.cmp td.bad .g{color:var(--ext)}table.cmp th{background:var(--chip)}
img.tab{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff;margin:6px 0}
</style></head><body><main>
<h1>ORCHID synthetic-papers benchmark</h1>
<p class="lead">Three of the twenty generated documents (4–5 pages each). Each document is a short synthetic paper calculated with pycalphad from a CC BY 4.0 SGTE dataset (Hallstedt 2025). The overlay draws the truth phase boundaries (blue) over the figure, with what each deterministic baseline extracted. Not real publications; generated documents are CC BY 4.0.</p>
<h2>Task × rendering: best result per family</h2>
<p class="lead">Previous baselines (no OpenCV), the thermo_extractor routes that need OpenCV / Camelot, the production AI reader (Gemini 3.8 Flash, single reader) and dual-reader agreement (Gemini + Claude; 5-document subset, vector and degraded only). Each cell gives the best extractor of that family and its score; full tables in <code>results/RESULTS.md</code>.</p>
<div class="card scroll" id="cmp"></div>
<h2>AI naming of phase fields vs truth</h2>
<p class="lead">The handbook open-field reader prompt: red numbered markers sit in the fields the deterministic annotator (assessed_pd) could not name; grey names are fields it named. Each reader picks one of the offered assemblages, "none of these" or "cannot tell"; the production rule accepts a name only when both readers agree (and the adjacency rules allow it).</p>
<div id="naming"></div>
<div id="app"></div>
<h2 style="margin-top:32px">Table extraction: truth vs extracted cells</h2>
<p class="lead">Each cell shows the printed truth (top) and what the pdfplumber baseline returned at the best-aligned position (bottom, green = exact match, red = different or missing). Split tables are scored segment by segment ("Table N (continued)").</p>
<div id="tabs"></div>
<script id="data" type="application/json">__DATA__</script>
<script id="tdata" type="application/json">__TDATA__</script>
<script id="cdata" type="application/json">__CDATA__</script>
<script id="ndata" type="application/json">__NDATA__</script>
<script>
(function(){
const data=JSON.parse(document.getElementById('data').textContent);
const app=document.getElementById('app');const NS='http://www.w3.org/2000/svg';
function el(t,a,txt){const e=document.createElement(t);if(a)for(const k in a)e.setAttribute(k,a[k]);if(txt!=null)e.textContent=txt;return e}
function sv(t,a){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e}
const colors=['var(--ext)','var(--ext2)','var(--ext3)','var(--ext4)','var(--ext5)'];
function fmt(v){if(v==null)return '–';if(typeof v==='number')return v.toFixed(3);return String(v)}
for(const d of data){
  const card=el('section',{class:'card'});app.appendChild(card);
  card.appendChild(el('h2',null,d.system+' — '+d.doc));
  const chips=el('div',{class:'chips'});card.appendChild(chips);
  [d.tdb, ...d.types.map(s=>s.replace(/_/g,' ')), d.pages+' pages', 'T in '+d.style.T_unit, 'x in '+d.style.x_unit,
   d.style.field_labels?'field labels':'no field labels', d.style.markers?'markers':null, d.style.overlay?'shifted overlay':null,
   d.style.inset?'inset':null].filter(Boolean).forEach(s=>chips.appendChild(el('span',{class:'chip'},s)));
  const grid=el('div',{class:'grid'});card.appendChild(grid);
  const left=el('div');grid.appendChild(left);
  const tabs=el('div',{class:'tabs'});left.appendChild(tabs);
  const img=el('img',{class:'page',alt:'Page with the phase diagram, '+d.doc,src:d.page_img});
  [['clean page',d.page_img],['degraded scan',d.deg_img]].forEach(([n,src],i)=>{const b=el('button',{type:'button','aria-pressed':i===0?'true':'false'},n);
    b.addEventListener('click',()=>{img.src=src;tabs.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed','false'));b.setAttribute('aria-pressed','true')});tabs.appendChild(b)});
  left.appendChild(img);
  const right=el('div');grid.appendChild(right);
  const W=d.fig_size[0],H=d.fig_size[1];
  const svg=sv('svg',{viewBox:'0 0 '+W+' '+H,role:'img','aria-label':'Phase diagram with truth and extracted boundaries'});
  const im=sv('image',{href:d.fig_img,x:0,y:0,width:W,height:H,opacity:'0.45'});svg.appendChild(im);
  const layers={};
  function layer(name){const g=sv('g',{});svg.appendChild(g);layers[name]=g;return g}
  const gt=layer('truth');
  for(const l of d.truth.lines){if(l.length<2)continue;gt.appendChild(sv('polyline',{points:l.map(p=>p.join(',')).join(' '),fill:'none',stroke:'var(--truth)','stroke-width':2.2,opacity:.85}))}
  for(const f of d.truth.fields){gt.appendChild(sv('circle',{cx:f.p[0],cy:f.p[1],r:4,fill:'var(--truth)'}));const tt=sv('title',{});tt.textContent='truth: '+f.label;gt.lastChild.appendChild(tt)}
  Object.keys(d.extracted).forEach((name,i)=>{const e=d.extracted[name];const g=layer(name);const c=colors[i%5];
    for(const l of e.boundaries){if(e.polyline&&l.length>1){g.appendChild(sv('polyline',{points:l.map(p=>p.join(',')).join(' '),fill:'none',stroke:c,'stroke-width':1.4,'stroke-dasharray':'5 3'}))}
      else{for(const p of l){g.appendChild(sv('rect',{x:p[0]-1,y:p[1]-1,width:2,height:2,fill:c}))}}}
    for(const f of e.fields){const s=sv('path',{d:'M'+(f.p[0]-5)+' '+(f.p[1]-5)+'l10 10m0 -10l-10 10',stroke:c,'stroke-width':2});const tt=sv('title',{});tt.textContent=name+': '+f.label;s.appendChild(tt);g.appendChild(s)}
    if(i>0)g.setAttribute('display','none')});
  const wrap=el('div',{class:'figwrap'});wrap.appendChild(svg);right.appendChild(wrap);
  const lg=el('div',{class:'legend'});right.insertBefore(lg,wrap);
  Object.keys(layers).forEach((name,i)=>{const lab=el('label');const cb=el('input',{type:'checkbox'});cb.checked=layers[name].getAttribute('display')!=='none';
    cb.addEventListener('change',()=>layers[name].setAttribute('display',cb.checked?'inline':'none'));
    const sw=el('span',{class:'sw'});sw.style.background=name==='truth'?'var(--truth)':colors[(i-1)%5];lab.append(cb,sw,document.createTextNode(name==='truth'?'truth (lines, field points)':name));lg.appendChild(lab)});
  right.appendChild(el('small',null,'Dots: truth field label points; crosses: named fields from the extractor (hover for names). Point samples are drawn as dots, polylines dashed.'));
  const rends=['vector','clean_raster','degraded_raster','image_only_pdf'];
  const keys=[...new Set(rends.flatMap(r=>Object.keys(d.metrics[r]||{})))].sort();
  const sc=el('div',{class:'scroll'});const tb=el('table');sc.appendChild(tb);card.appendChild(el('h3',null,'Metrics for this document'));card.appendChild(sc);
  const hr=el('tr');hr.appendChild(el('th',null,'metric'));rends.forEach(r=>hr.appendChild(el('th',null,r.replace(/_/g,' '))));tb.appendChild(hr);
  for(const k of keys){const tr=el('tr');tr.appendChild(el('td',null,k));rends.forEach(r=>tr.appendChild(el('td',{class:'n'},fmt((d.metrics[r]||{})[k]))));tb.appendChild(tr)}
  const det=el('details');det.appendChild(el('summary',null,'Invariant reactions in the truth key ('+d.invariants.length+')'));
  const ul=el('ul');d.invariants.forEach(i=>ul.appendChild(el('li',null,i.type+': '+i.reaction+' at '+i.T.toFixed(2)+' K')));det.appendChild(ul);card.appendChild(det);
}
const cdata=JSON.parse(document.getElementById('cdata').textContent);
{const host=document.getElementById('cmp');const tb=el('table');host.appendChild(tb);
 const hr=el('tr');['task / metric','rendering','previous baseline','deterministic (OpenCV / Camelot)','AI readers (Gemini)','dual-reader agreement'].forEach(h=>hr.appendChild(el('th',null,h)));tb.appendChild(hr);
 for(const c of cdata){const tr=el('tr');tr.appendChild(el('td',null,c.label));tr.appendChild(el('td',null,c.rendering.replace(/_/g,' ')));
  for(const fam of ['prev','det','ai','dual']){const v=c[fam];const td=el('td',{class:'n'});
   if(v){td.appendChild(el('strong',null,v[1].toFixed(3)));td.appendChild(el('small',null,' '+v[0]))}else td.textContent='–';tr.appendChild(td)}
  tb.appendChild(tr)}}
const ndata=JSON.parse(document.getElementById('ndata').textContent);
{const host=document.getElementById('naming');
 for(const N of ndata){const card=el('section',{class:'card'});host.appendChild(card);
  card.appendChild(el('h2',null,N.system+' — '+N.doc+' — '+N.rendering.replace(/_/g,' ')));
  const chips=el('div',{class:'chips'});card.appendChild(chips);
  [N.printed_labels?'field labels printed':'no field labels printed', N.rows.length+' open fields', N.has_claude?'second reader: Claude':'single reader only']
   .forEach(x=>chips.appendChild(el('span',{class:'chip'},x)));
  const grid=el('div',{class:'grid'});card.appendChild(grid);
  const l=el('div');grid.appendChild(l);l.appendChild(el('img',{class:'page',src:N.img,alt:'Diagram packet with numbered markers, '+N.doc}));
  const r=el('div',{class:'scroll'});grid.appendChild(r);const tb=el('table');r.appendChild(tb);
  const hr=el('tr');['#','truth field','Gemini','Claude','accepted'].forEach(h=>hr.appendChild(el('th',null,h)));tb.appendChild(hr);
  for(const q of N.rows){const tr=el('tr');tr.appendChild(el('td',null,String(q.marker)));tr.appendChild(el('td',null,q.truth));
   for(const [a,ok,why] of [[q.gemini,q.ok_gemini,q.gemini_reason],[q.claude,q.ok_claude,null]]){const td=el('td');
    td.appendChild(el('span',null,a==null?'–':a));if(ok===true)td.style.color='var(--ext3)';else if(ok===false)td.style.color='var(--ext)';
    if(why){const t_=el('title');td.title=why}tr.appendChild(td)}
   tr.appendChild(el('td',null,q.agree?(q.ok_gemini?'yes (correct)':q.ok_gemini===false?'yes (wrong)':'yes'):'no'));tb.appendChild(tr)}
  r.appendChild(el('small',null,'Green = matches the truth field; red = a different assemblage; plain = abstained ("cannot tell" / "none of these"). Hover a Gemini answer for its reason.'));
 }}
const tdata=JSON.parse(document.getElementById('tdata').textContent);const tapp=document.getElementById('tabs');
for(const T of tdata){
  const card=el('section',{class:'card'});tapp.appendChild(card);
  card.appendChild(el('h2',null,T.system+' — '+T.doc+' — table "'+T.table+'"'));
  const chips=el('div',{class:'chips'});card.appendChild(chips);
  ['rules: '+T.style.rules, T.style.merged_header?'merged header':'single header', T.style.markers?'footnote markers':null,
   T.segments.length>1?'continued over '+T.segments.length+' pages':null, 'cell exact '+(T.cell_exact==null?'–':T.cell_exact.toFixed(3))]
   .filter(Boolean).forEach(x=>chips.appendChild(el('span',{class:'chip'},x)));
  card.appendChild(el('p',null,T.caption||''));
  T.segments.forEach((sg,si)=>{
    card.appendChild(el('h3',null,'Segment '+(si+1)+' (page '+sg.page+'; extracted by: '+(sg.route||'nothing')+')'));
    card.appendChild(el('img',{class:'tab',src:sg.img,alt:'Rendered table segment '+(si+1)}));
    const sc=el('div',{class:'scroll'});const tb=el('table',{class:'cmp'});sc.appendChild(tb);card.appendChild(sc);
    for(const row of sg.grid){const tr=el('tr');for(const c of row){const td=el(c.header?'th':'td',{colspan:String(c.span)});
      if(c.ok===true)td.className='ok';else if(c.ok===false)td.className='bad';
      td.appendChild(el('span',{class:'t'},c.truth||'\u00a0'));td.appendChild(el('span',{class:'g'},c.truth?(c.got==null?'(missing)':(c.got||'(empty)')):'\u00a0'));tr.appendChild(td)}tb.appendChild(tr)}
  });
  if(T.footnotes.length){const ul=el('ul');T.footnotes.forEach(x=>ul.appendChild(el('li',null,x)));card.appendChild(ul)}
}
})();
</script></main></body></html>"""


def main():
    data = [doc_payload(d) for d in EXAMPLES]
    js = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    tj = json.dumps([table_payload(d, tid) for d, tid in pick_tables()], ensure_ascii=False).replace("</", "<\\/")
    cj = json.dumps(comparison_payload(), ensure_ascii=False).replace("</", "<\\/")
    nj = json.dumps([x for x in (naming_payload(d, r) for d, r in NAMING) if x], ensure_ascii=False).replace("</", "<\\/")
    out = B / "showcase" / "index.html"
    out.write_text(
        HTML.replace("__DATA__", js).replace("__TDATA__", tj).replace("__CDATA__", cj).replace("__NDATA__", nj), encoding="utf-8"
    )
    print(out, out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
