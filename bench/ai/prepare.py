#!/usr/bin/env python3
"""AI-reader stage 1 (no model calls): build every reader request ("job") exactly as the production readers would,
from the production prompt/schema objects themselves (imported read-only):

  * inv_table      sgte-corpus tools-release-v2.1.18 gemini_table_reader  PROMPT / SCHEMA   (inv-table-v5)
  * thermo_table   thermo_table_reader PROMPT_V7_1 / SCHEMA_V7_1                           (thermo-table-v7.1-sites)
  * text           release-fixup-20261022 text_reader_v6 PROMPT / schema(T)                (thermo-text-v6.1-quote)
  * context        release-fixup-20261022 context_reader_v2 PROMPT / SCHEMA                (exp-context-v2.1-quote)
  * pd_fields      handbook-pilot gemini_reader PROMPT + marker choices (open-field reader of the assessed_pd ladder)

Inputs per rendering (no truth used, except where the assessed_pd stage already used it, see raster_pd.py):
  vector     production crop_png (PDFium, 160 dpi, caption-to-next-caption, full width) + text-layer lines of the crop;
             running text = pdftotext reading order (text_reader_v6.pages_for).
  rasters    the same crop rule applied to the page image using Tesseract line boxes for the captions (clean pages
             resampled to 160 dpi; degraded pages at their native 150 dpi); no text-layer hint ('(none)', as for a scan);
             running text = Tesseract paragraphs.
  Continued tables ('Table N (continued).') are sent as a two-page stack, as table_continuation does.
Run with ~/orchid-py313/bin/python (google-genai present; nothing is installed). Writes work/ai/jobs/*.json and
work/ai/img/*.png.
"""

import ast
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from baselines.ocr_cache import WORK as OCR  # noqa: E402
from common import docs_dir, load_json, truth_dir  # noqa: E402

SGTE = Path.home() / "projects/orchid-pipeline/pipelines/sgte-corpus"
REL = SGTE / "tools-release-v2.1.18"
FIX = SGTE / "release-fixup-20261022/bin-context"
sys.path.insert(0, str(REL))
sys.path.insert(0, str(FIX))
import gemini_table_reader as G  # noqa: E402
import text_reader as TX  # noqa: E402
import text_reader_v6 as R6  # noqa: E402
import thermo_table_reader as T  # noqa: E402

APD = os.environ.get(
    "ASSESSED_PD_SRC",
    "/private/tmp/claude-504/-Users-raymundo-arroyave-projects-open-tdb/"
    "5d97acc6-0c9e-4398-87a5-e3422e273f18/scratchpad/assessed-pd-annotator/src",
)
sys.path.insert(0, APD)

AI = BENCH / "work" / "ai"
JOBS, IMG = AI / "jobs", AI / "img"
RENDS = ["vector", "clean_raster", "degraded_raster", "image_only_pdf"]
CAP = re.compile(r"^(Table)\s+(\d+)\s*(\(continued\))?[.:]?")
NEXT = re.compile(r"^(Table|TABLE|Fig\.?|Figure)\s*\d+")


def context_reader_objects():
    """PROMPT_VERSION / PROMPT / SCHEMA / methods_passages / cue_passages of context_reader_v2, without importing it
    (its imports open the production database).
    """
    src = (FIX / "context_reader_v2.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    keep = {
        "PROMPT_VERSION",
        "HEAD",
        "NOT_HEAD",
        "INLINE_HEAD",
        "END",
        "ITEM_Q",
        "SCHEMA",
        "PROMPT",
        "METHOD_CUE",
        "OWN_CUE",
        "NOT_OWN",
        "methods_passages",
        "cue_passages",
    }
    body = [
        n
        for n in tree.body
        if (isinstance(n, ast.Assign) and any(getattr(t_, "id", None) in keep for t_ in n.targets))
        or (isinstance(n, ast.FunctionDef) and n.name in keep)
    ]
    ns = {"re": re}
    exec(compile(ast.Module(body=body, type_ignores=[]), "context_reader_v2", "exec"), ns)  # noqa: S102
    return ns


CTX = context_reader_objects()
HANDBOOK = Path.home() / "projects/open-tdb/build/okamoto-pilot/tools/gemini_reader.py"
HB_PROMPT = ast.literal_eval(
    next(
        n.value
        for n in ast.parse(HANDBOOK.read_text()).body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "PROMPT"
    )
)
HB_SCHEMA = {
    "type": "OBJECT",
    "required": ["answers"],
    "properties": {
        "answers": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["marker", "assemblage", "reason"],
                "properties": {"marker": {"type": "INTEGER"}, "assemblage": {"type": "STRING"}, "reason": {"type": "STRING"}},
            },
        }
    },
}


# ------------------------------------------------------------------------------------------------ page text
def ocr_lines(doc, r, page):
    """Tesseract lines (rendering pixels) with block/paragraph ids."""
    path = OCR / doc / r / f"page-{page}.tsv"
    lines = {}
    for ln in path.read_text(encoding="utf-8").splitlines()[1:]:
        f = ln.split("\t")
        if len(f) < 12 or f[0] != "5" or not f[11].strip():
            continue
        k = (int(f[2]), int(f[3]), int(f[4]))
        x, y, w, h = int(f[6]), int(f[7]), int(f[8]), int(f[9])
        L = lines.setdefault(k, {"words": [], "box": [x, y, x + w, y + h]})
        L["words"].append(f[11])
        b = L["box"]
        L["box"] = [min(b[0], x), min(b[1], y), max(b[2], x + w), max(b[3], y + h)]
    out = [{"text": " ".join(v["words"]), "box": v["box"], "block": k[0], "par": k[1]} for k, v in lines.items()]
    return sorted(out, key=lambda q: (q["box"][1], q["box"][0]))


def ocr_pages_text(doc, r, n):
    """Tesseract paragraphs, blank line between paragraphs (the input pages_for gives for a text layer)."""
    pages = []
    for p in range(1, n + 1):
        paras, cur, key = [], [], None
        for L in sorted(ocr_lines(doc, r, p), key=lambda q: (q["block"], q["par"], q["box"][1])):
            k = (L["block"], L["par"])
            if key is not None and k != key and cur:
                paras.append("\n".join(cur))
                cur = []
            key = k
            cur.append(L["text"])
        if cur:
            paras.append("\n".join(cur))
        pages.append("\n\n".join(paras))
    return pages


def vector_lines(pdf, page):
    """Group the words of one PDF page into text lines."""
    with G.pdfplumber.open(str(pdf)) as d:
        pg = d.pages[page - 1]
        ws = pg.extract_words(keep_blank_chars=False, use_text_flow=True)
    lines = []
    for w in sorted(ws, key=lambda w: (round(w["top"]), w["x0"])):
        if lines and abs(lines[-1]["top"] - w["top"]) < 2.5:
            L = lines[-1]
            L["text"] += " " + w["text"]
            L["box"] = [
                min(L["box"][0], w["x0"]),
                min(L["box"][1], w["top"]),
                max(L["box"][2], w["x1"]),
                max(L["box"][3], w["bottom"]),
            ]
        else:
            lines.append({"text": w["text"], "top": w["top"], "box": [w["x0"], w["top"], w["x1"], w["bottom"]]})
    return lines


def captions(lines):
    """Find the caption lines (figure and table headings) in a list of text lines."""
    out = []
    for i, L in enumerate(lines):
        m = CAP.match(L["text"].strip())
        if m:
            cap = L["text"].strip()
            if (
                i + 1 < len(lines)
                and not cap.endswith(".")
                and lines[i + 1]["box"][1] - L["box"][3] < 0.8 * (L["box"][3] - L["box"][1]) + 2
            ):
                cap += " " + lines[i + 1]["text"].strip()
            out.append({"number": int(m.group(2)), "continued": bool(m.group(3)), "caption": cap, "box": L["box"], "index": i})
    return out


# ------------------------------------------------------------------------------------------------ crops
def png_bytes(im):
    """Encode a PIL image as PNG bytes."""
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def page_image(doc, r, page):
    """Return the page image of one document, rendering and page as a PIL image."""
    from PIL import Image

    if r == "clean_raster":
        im = Image.open(docs_dir / doc / "clean" / f"page-{page}.png").convert("RGB")
        return im.resize((int(round(im.width * 160 / 300)), int(round(im.height * 160 / 300))), Image.LANCZOS), 160 / 300
    if r == "degraded_raster":
        return Image.open(docs_dir / doc / "degraded" / f"page-{page}.jpg").convert("RGB"), 1.0
    return Image.open(next((OCR / doc / "image_only_pdf").glob(f"page-{page}.*[gG]"))).convert("RGB"), 1.0


def raster_crop(doc, r, page, cap, lines):
    """table_extent rule on a raster (caption boxes in the rendering's own pixels): from the caption top - 6 pt to the
    next Table/Fig caption at least 40 pt below it on the same side of the page, else the page bottom; full width.
    """
    im, s = page_image(doc, r, page)
    native = 300 if r == "clean_raster" else 150
    pt = native / 72.0
    W = im.width / s
    left = cap["box"][0] < 0.5 * W
    y0 = max(0.0, cap["box"][1] - 6 * pt)
    below = [
        L["box"][1]
        for L in lines
        if L["box"][1] > cap["box"][3] + 40 * pt and NEXT.match(L["text"].strip()) and (L["box"][0] < 0.5 * W) == left
    ]
    y1 = (min(below) - 2 * pt) if below else (im.height / s - 4 * pt)
    return im.crop((0, int(y0 * s), im.width, int(min(y1 * s, im.height))))


def stack(images):
    """Stack PIL images vertically on a white background."""
    from PIL import Image

    w = max(i.width for i in images)
    out = Image.new("RGB", (w, sum(i.height for i in images)), "white")
    y = 0
    for i in images:
        out.paste(i, (0, y))
        y += i.height
    return out


def save_img(im_bytes):
    """Store image bytes under their content hash and return the file path."""
    h = hashlib.sha256(im_bytes).hexdigest()
    p = IMG / f"{h[:24]}.png"
    if not p.exists():
        p.write_bytes(im_bytes)
    return str(p.relative_to(BENCH)), h


def write_job(kind, doc, r, prompt, schema, images, meta):
    """Write one AI job file named by a hash of its inputs."""
    hs = [h for _, h in images]
    key = hashlib.sha256(json.dumps([kind, doc, r, prompt, schema, hs], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[
        :24
    ]
    job = {
        "key": key,
        "kind": kind,
        "doc": doc,
        "rendering": r,
        "prompt": prompt,
        "schema": schema,
        "images": [p for p, _ in images],
        "image_sha256": hs,
        "meta": meta,
    }
    (JOBS / f"{doc}__{r}__{kind}__{meta.get('label', '')}.json".replace(" ", "_")).write_text(
        json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return job


# ------------------------------------------------------------------------------------------------ jobs
def table_jobs(t, r):
    """Create the table-extraction jobs for one document and rendering."""
    doc = t["doc_id"]
    n = t["renderings"]["vector"]["pages"]
    pdf = docs_dir / doc / "vector.pdf"
    per_page = {}
    for p in range(1, n + 1):
        lines = vector_lines(pdf, p) if r == "vector" else ocr_lines(doc, r, p)
        per_page[p] = (lines, captions(lines))
    if r == "vector":
        full = R6.pages_for(str(pdf))
    else:
        full = ocr_pages_text(doc, r, n)
    sentences = [(i + 1, s) for i, pg in enumerate(full) for s in re.split(r"(?<=[.!?])\s+", R6.norm(pg))]
    jobs = []
    for p in range(1, n + 1):
        lines, caps = per_page[p]
        for c in caps:
            if c["continued"]:
                continue
            imgs, recipe = [], {}
            if r == "vector":
                box = {"x_min": c["box"][0], "y_min": c["box"][1], "x_max": c["box"][2], "y_max": c["box"][3]}
                imgs.append(G.crop_png(str(pdf), p, box, recipe=recipe))
                ext, _ = G.table_extent(G.pdfplumber.open(str(pdf)).pages[p - 1], box)
                tl = [L["text"] for L in lines if ext[1] <= L["box"][1] <= ext[3]]
            else:
                from PIL import Image  # noqa: F401

                imgs.append(png_bytes(raster_crop(doc, r, p, c, lines)))
                tl = []
            cont = None
            if p + 1 <= n:
                nl, nc = per_page[p + 1]
                cont = next((x for x in nc if x["continued"] and x["number"] == c["number"]), None)
                if cont is not None:
                    if r == "vector":
                        box2 = {
                            "x_min": cont["box"][0],
                            "y_min": cont["box"][1],
                            "x_max": cont["box"][2],
                            "y_max": cont["box"][3],
                        }
                        imgs.append(G.crop_png(str(pdf), p + 1, box2))
                        ext2, _ = G.table_extent(G.pdfplumber.open(str(pdf)).pages[p], box2)
                        tl += [L["text"] for L in nl if ext2[1] <= L["box"][1] <= ext2[3]]
                    else:
                        imgs.append(png_bytes(raster_crop(doc, r, p + 1, cont, nl)))
            from PIL import Image

            im = png_bytes(stack([Image.open(io.BytesIO(b)).convert("RGB") for b in imgs])) if len(imgs) > 1 else imgs[0]
            ctx = [
                {"pdf_page": pg, "snippet": s[:300]}
                for pg, s in sentences
                if re.search(rf"\bTable\s+{c['number']}\b", s) and not CAP.match(s)
            ][:8]
            ctxs = "\n".join(f"- (p{m['pdf_page']}) {m['snippet']}" for m in ctx) or "(none found)"
            inv = bool(re.search(r"(?i)invariant|reaction", c["caption"]))
            tmpl, sch, ver = (
                (G.PROMPT, G.SCHEMA, G.PROMPT_VERSION) if inv else (T.PROMPT_V7_1, T.SCHEMA_V7_1, T.PROMPT_VERSION_V7_1)
            )
            prompt = tmpl.format(caption=c["caption"], text="\n".join(tl[:80]) or "(none)", context=ctxs)
            jobs.append(
                write_job(
                    "inv_table" if inv else "thermo_table",
                    doc,
                    r,
                    prompt,
                    sch,
                    [save_img(im)],
                    {
                        "label": f"table{c['number']}",
                        "table_number": c["number"],
                        "page": p,
                        "continued_page": p + 1 if cont else None,
                        "prompt_version": ver,
                        "crop": recipe or {"rule": "caption-to-next-caption, full width", "rendering": r},
                    },
                )
            )
    return jobs, full


def text_jobs(t, r, pages):
    """Create the text-value jobs for one document and rendering from the selected paragraphs."""
    doc = t["doc_id"]
    passages = R6.selected_paragraphs(pages, TX.UNIT, TX.TERMS)
    sch = R6.schema(T)
    jobs, chunk, size = [], [], 0

    def flush(i):
        blocks, last = [], None
        for q in chunk:
            if q["page"] != last:
                blocks.append(f"[page {q['page']}]")
                last = q["page"]
            blocks.append(q["text"])
        text = "\n\n".join(blocks)
        jobs.append(
            write_job(
                "text",
                doc,
                r,
                R6.PROMPT.format(text=text),
                sch,
                [],
                {
                    "label": f"chunk{i}",
                    "prompt_version": R6.PROMPT_VERSION,
                    "passages": list(chunk),
                    "text_layer": R6.TEXT_LAYER if r == "vector" else "Tesseract paragraphs (no text layer)",
                },
            )
        )

    for q in passages:
        if size + len(q["text"]) > 6000 and chunk:
            flush(len(jobs))
            chunk, size = [], 0
        chunk.append(q)
        size += len(q["text"])
    if chunk:
        flush(len(jobs))
    pp = [R6.page_passages(pg) for pg in pages]
    section = CTX["methods_passages"](pp) or CTX["cue_passages"](pp)
    ctx_jobs = []
    if section:
        blocks, last = [], None
        for pg, _, txt in section:
            if pg != last:
                blocks.append(f"[page {pg}]")
                last = pg
            blocks.append(txt)
        ctx_jobs.append(
            write_job(
                "context",
                doc,
                r,
                CTX["PROMPT"].format(text="\n\n".join(blocks)),
                CTX["SCHEMA"],
                [],
                {"label": "methods", "prompt_version": CTX["PROMPT_VERSION"]},
            )
        )
    return jobs, ctx_jobs, len(passages)


def pd_job(t, r):
    """Open fields of the assessed_pd_auto annotation -> one handbook-reader packet (red numbered markers)."""
    import pickle

    from assessed_pd.pipeline import annotate
    from assessed_pd.readers import Ladder
    from PIL import Image, ImageDraw, ImageFont

    doc = t["doc_id"]
    pk = BENCH / "work" / "raster_pd" / doc / r / "assessed_input.pkl"
    if not pk.exists():
        return None, 0
    d = pickle.load(open(pk, "rb"))
    ann = annotate(d["inp"], d["cfg"])
    pending = sorted(rid for rid, g in ann.names.items() if not g.get("phases") and g.get("status") is None)
    qs = Ladder([]).questions(ann, pending)
    if not qs:
        return None, 0
    im = Image.open(d["inp"].image_path).convert("RGB")
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", max(12, im.width // 70))
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", max(10, im.width // 90))
    except OSError:
        font = small = ImageFont.load_default()
    for rid, g in ann.names.items():
        if g.get("phases"):
            x, y = ann.fields.regions[rid].label_point_px
            dr.text((x, y), "+".join(g["phases"]), fill=(150, 150, 150), font=small, anchor="mm")
    rad = max(9, im.width // 90)
    markers = []
    for i, q in enumerate(qs, 1):
        x, y = ann.fields.regions[q.field_id].label_point_px
        dr.ellipse((x - rad, y - rad, x + rad, y + rad), fill=(220, 20, 20))
        dr.text((x, y), str(i), fill="white", font=font, anchor="mm")
        markers.append({"marker": i, "field_id": q.field_id, "choices": q.choices})
    lines = "\n".join(f"- marker {m['marker']}: choices {m['choices']}" for m in markers)
    job = write_job(
        "pd_fields",
        doc,
        r,
        HB_PROMPT + lines,
        HB_SCHEMA,
        [save_img(png_bytes(im))],
        {"label": "fig1", "markers": markers, "prompt_version": "handbook-open-field-reader (gemini_reader.py)"},
    )
    return job, len(qs)


def main():
    """Prepare the AI jobs for the given documents (default: every truth document)."""
    JOBS.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    docs = sys.argv[1:] or [tf.stem for tf in sorted(truth_dir.glob("*.json"))]
    summary = []
    for doc in docs:
        t = load_json(truth_dir / f"{doc}.json")
        for r in RENDS:
            tj, pages = table_jobs(t, r)
            xj, cj, npass = text_jobs(t, r, pages)
            pj, nq = pd_job(t, r)
            summary.append(
                {
                    "doc": doc,
                    "rendering": r,
                    "tables": [j["meta"]["label"] for j in tj],
                    "text_chunks": len(xj),
                    "passages": npass,
                    "context_jobs": len(cj),
                    "pd_questions": nq,
                }
            )
            print(json.dumps(summary[-1]), flush=True)
    (AI / "prepare_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
