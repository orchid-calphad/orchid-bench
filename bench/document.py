"""Born-digital document composition with PyMuPDF: real text objects, vector figures, ruled tables.
Every placed element's page coordinates (PDF points, origin top-left, y down) are recorded for the truth key.
"""

from __future__ import annotations

import re

import pymupdf
from common import font_dir

fonts = {
    "serif": font_dir / "DejaVuSerif.ttf",
    "serif-b": font_dir / "DejaVuSerif-Bold.ttf",
    "serif-i": font_dir / "DejaVuSerif-Italic.ttf",
    "sans": font_dir / "DejaVuSans.ttf",
    "sans-b": font_dir / "DejaVuSans-Bold.ttf",
}
_FONT_OBJ = {}


def font(name):
    """Return the cached pymupdf Font for a registered font name."""
    if name not in _FONT_OBJ:
        _FONT_OBJ[name] = pymupdf.Font(fontfile=str(fonts[name]))
    return _FONT_OBJ[name]


def tlen(text, name, size):
    """Return the rendered width in points of text set in font name at size pt."""
    return font(name).text_length(text, fontsize=size)


num_re = re.compile(r"[−\-–]?\d+(?:\.\d+)?(?:[eE][−\-]?\d+)?")


def parse_number(text):
    """First number in a cell; '−12.3 ± 0.5' -> (-12.3, 0.5)."""
    t = text.replace("−", "-").replace("–", "-").replace(" ", "").replace(" ", "")
    t = t.lstrip("<>")
    m = re.match(r"^\(?([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:±([0-9.]+))?", t)
    if not m:
        return None, None
    try:
        return float(m.group(1)), (float(m.group(2)) if m.group(2) else None)
    except ValueError:
        return None, None


class Doc:
    """Incremental PDF writer that keeps a vertical cursor and flows text and figures."""

    def __init__(self, page_size, margins, body_font="serif", body_size=10.0, leading=1.32, footer=""):
        """Open an empty PDF with a page size, margins, body font, body size and leading factor."""
        self.pdf = pymupdf.open()
        self.W, self.H = page_size
        self.ml, self.mt, self.mr, self.mb = margins
        self.body_font, self.body_size, self.leading = body_font, body_size, leading
        self.footer = footer
        self.page = None
        self.pno = -1
        self.y = 0.0
        self.new_page()

    @property
    def text_width(self):
        """Return the text block width in points between the left and right margins."""
        return self.W - self.ml - self.mr

    def new_page(self):
        """Append a new page of the document size and reset the cursor."""
        self.page = self.pdf.new_page(width=self.W, height=self.H)
        self.pno += 1
        for k, f in fonts.items():
            self.page.insert_font(fontname=k.replace("-", ""), fontfile=str(f))
        self.y = self.mt
        if self.footer:
            s = f"{self.footer} — page {self.pno + 1}"
            w = tlen(s, "sans", 7)
            self.page.insert_text(
                ((self.W - w) / 2, self.H - self.mb / 2), s, fontname="sans", fontsize=7, color=(0.35, 0.35, 0.35)
            )

    def ensure(self, h):
        """Start a new page when h points do not fit above the bottom margin."""
        if self.y + h > self.H - self.mb:
            self.new_page()

    def space(self, h):
        """Advance the vertical cursor by h points."""
        self.y += h

    def _put(self, x, baseline, text, fname, size, color=(0, 0, 0)):
        self.page.insert_text((x, baseline), text, fontname=fname.replace("-", ""), fontsize=size, color=color)

    def _bbox(self, x, baseline, text, fname, size):
        f = font(fname)
        return [x, baseline - f.ascender * size, x + tlen(text, fname, size), baseline - f.descender * size]

    def paragraph(self, pieces, fname=None, size=None, indent=0.0, align="justify", gap_after=4.0, x0=None, width=None):
        """pieces: list of (text, tag). Lays out wrapped lines; returns {tag: [ {page, bbox}, ... ]} and span lookups."""
        fname = fname or self.body_font
        size = size or self.body_size
        x0 = self.ml if x0 is None else x0
        width = self.text_width if width is None else width
        text, spans, pos = "", [], 0
        for t, tag in pieces:
            if text:
                text += " "
                pos += 1
            spans.append((pos, pos + len(t), tag))
            text += t
            pos += len(t)
        words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        lines, cur = [], []
        for w in words:
            cand = cur + [w]
            s = text[cand[0][0] : cand[-1][1]]
            ind = indent if not lines else 0.0
            if cur and tlen(s, fname, size) > width - ind:
                lines.append(cur)
                cur = [w]
            else:
                cur = cand
        if cur:
            lines.append(cur)
        lh = size * self.leading
        f = font(fname)
        placed = []
        for li, ln in enumerate(lines):
            self.ensure(lh)
            baseline = self.y + f.ascender * size
            ind = indent if li == 0 else 0.0
            a, b = ln[0][0], ln[-1][1]
            seg = text[a:b]
            last = li == len(lines) - 1
            if align == "justify" and not last and len(ln) > 1:
                natural = sum(tlen(text[s_:e_], fname, size) for s_, e_ in ln)
                gap = (width - ind - natural) / (len(ln) - 1)
                xx = x0 + ind
                xs = []
                for s_, e_ in ln:
                    self._put(xx, baseline, text[s_:e_], fname, size)
                    xs.append((s_, e_, xx))
                    xx += tlen(text[s_:e_], fname, size) + gap
            else:
                xx = x0 + ind if align != "center" else x0 + (width - tlen(seg, fname, size)) / 2
                self._put(xx, baseline, seg, fname, size)
                xs, cx = [], xx
                for s_, e_ in ln:
                    off = tlen(text[a:s_], fname, size)
                    xs.append((s_, e_, cx + off))
            placed.append({"page": self.pno, "a": a, "b": b, "baseline": baseline, "words": xs})
            self.y += lh
        self.y += gap_after

        def locate(sa, sb):
            rects = []
            for pl in placed:
                lo, hi = max(sa, pl["a"]), min(sb, pl["b"])
                if lo >= hi:
                    continue
                x_lo = x_hi = None
                for s_, e_, wx in pl["words"]:
                    if e_ <= lo or s_ >= hi:
                        continue
                    w_lo = wx + tlen(text[s_ : max(s_, lo)], fname, size)
                    w_hi = wx + tlen(text[s_ : min(e_, hi)], fname, size)
                    x_lo = w_lo if x_lo is None else min(x_lo, w_lo)
                    x_hi = w_hi if x_hi is None else max(x_hi, w_hi)
                if x_lo is None:
                    continue
                bl = pl["baseline"]
                rects.append(
                    {
                        "page": pl["page"],
                        "bbox": [
                            round(x_lo, 2),
                            round(bl - f.ascender * size, 2),
                            round(x_hi, 2),
                            round(bl - f.descender * size, 2),
                        ],
                    }
                )
            return rects

        out = {}
        for sa, sb, tag in spans:
            out[tag] = {"text": text[sa:sb], "rects": locate(sa, sb), "start": sa}
        out["_locate"] = locate
        out["_text"] = text
        return out

    def heading(self, text, size=11.5):
        """Set a bold heading of the given size in points and return its anchor."""
        self.ensure(size * 3)
        self.y += size * 0.6
        r = self.paragraph([(text, "h")], fname="serif-b", size=size, align="left", gap_after=3)
        return r["h"]

    def figure(self, fig_pdf, fig_size_in, width_frac, caption_pieces, cap_size=8.5):
        """Place a figure at width_frac of the text width together with its caption pieces."""
        w = self.text_width * width_frac
        h = w * fig_size_in[1] / fig_size_in[0]
        cap_h = cap_size * 1.3 * 3
        self.ensure(h + cap_h + 8)
        x0 = self.ml + (self.text_width - w) / 2
        rect = pymupdf.Rect(x0, self.y, x0 + w, self.y + h)
        src = pymupdf.open(fig_pdf)
        self.page.show_pdf_page(rect, src, 0)
        src.close()
        page = self.pno
        self.y += h + 4
        cap = self.paragraph(
            caption_pieces,
            fname="sans",
            size=cap_size,
            align="left",
            gap_after=10,
            x0=self.ml + 0.04 * self.text_width,
            width=0.92 * self.text_width,
        )
        return {"page": page, "rect": [rect.x0, rect.y0, rect.x1, rect.y1], "caption": cap}

    def table(self, spec, style, size=8.5, cap_pieces=None, cont_pieces=None, split_at=None):
        """spec: {'header': [[{text, colspan}]], 'rows': [[text]], 'align': ['l','r',...], 'footnotes': [str]}.
        split_at: body-row index at which the table is continued on a new page under cont_pieces
        ("Table N (continued)") with the header repeated.
        """
        ncol = len(spec["rows"][0]) if spec["rows"] else sum(c.get("colspan", 1) for c in spec["header"][-1])
        fn, fnb = "sans", "sans-b"
        pad = 5.0
        while True:
            colw = [0.0] * ncol
            for row in spec["rows"]:
                for j, t in enumerate(row):
                    colw[j] = max(colw[j], tlen(t, fn, size) + 2 * pad)
            for hrow in spec["header"]:
                j = 0
                for c in hrow:
                    span = c.get("colspan", 1)
                    need = tlen(c["text"], fnb, size) + 2 * pad
                    have = sum(colw[j : j + span])
                    if need > have:
                        for k in range(j, j + span):
                            colw[k] += (need - have) / span
                    j += span
            total = sum(colw)
            if total <= self.text_width or size < 6.0:
                break
            size -= 0.5
            pad = max(3.0, pad - 0.5)
        rh = size * 1.75
        segs = [(0, len(spec["rows"]))]
        if split_at and 0 < split_at < len(spec["rows"]):
            segs = [(0, split_at), (split_at, len(spec["rows"]))]
        x0 = self.ml + (self.text_width - total) / 2
        xs = [x0]
        for w in colw:
            xs.append(xs[-1] + w)
        f = font(fn)
        lw_heavy, lw_light = 0.9, 0.45
        black = (0, 0, 0)
        cells, caps, pages, bboxes = [], [], [], []
        nh = len(spec["header"])
        for si, (r0, r1) in enumerate(segs):
            pieces = cap_pieces if si == 0 else cont_pieces
            nrows = nh + (r1 - r0)
            cap_h = (size + 1) * 1.3 * 3 if pieces else 0
            if si > 0:
                self.new_page()
            self.ensure(cap_h + rh * nrows + 30 + (12 * len(spec.get("footnotes", [])) if si == len(segs) - 1 else 0))
            cap = self.paragraph(pieces, fname="sans", size=size + 0.5, align="left", gap_after=3) if pieces else None
            caps.append(cap)
            page = self.page
            pages.append(self.pno)
            y = self.y
            top = y
            seg_cells = []
            if style["rules"] == "booktabs":
                page.draw_line((xs[0], y), (xs[-1], y), color=black, width=lw_heavy)
            for hi, hrow in enumerate(spec["header"]):
                j = 0
                for c in hrow:
                    span = c.get("colspan", 1)
                    cx0, cx1 = xs[j], xs[j + span]
                    tw = tlen(c["text"], fnb, size)
                    tx = (cx0 + cx1 - tw) / 2
                    bl = y + (rh + f.ascender * size + f.descender * size) / 2
                    if c["text"]:
                        self._put(tx, bl, c["text"], fnb, size)
                    seg_cells.append(
                        {
                            "row": hi,
                            "col": j,
                            "colspan": span,
                            "rowspan": 1,
                            "header": True,
                            "text": c["text"],
                            "page": self.pno,
                            "segment": si,
                            "seg_row": hi,
                            "repeated_header": si > 0,
                            "bbox": [round(cx0, 2), round(y, 2), round(cx1, 2), round(y + rh, 2)],
                        }
                    )
                    if span > 1 and style["rules"] == "booktabs":
                        page.draw_line((cx0 + 3, y + rh - 1), (cx1 - 3, y + rh - 1), color=black, width=lw_light)
                    j += span
                y += rh
            if style["rules"] == "booktabs":
                page.draw_line((xs[0], y), (xs[-1], y), color=black, width=lw_light)
            elif style["rules"] == "plain":
                page.draw_line((xs[0], top), (xs[-1], top), color=black, width=lw_light)
                page.draw_line((xs[0], y), (xs[-1], y), color=black, width=lw_light)
            for bi in range(r0, r1):
                row = spec["rows"][bi]
                if style["rules"] == "plain" and (bi - r0) % 2 == 1:
                    page.draw_rect(pymupdf.Rect(xs[0], y, xs[-1], y + rh), color=None, fill=(0.92, 0.92, 0.92), width=0)
                for j, t in enumerate(row):
                    al = spec["align"][j]
                    tw = tlen(t, fn, size)
                    tx = xs[j] + pad if al == "l" else (xs[j + 1] - pad - tw if al == "r" else (xs[j] + xs[j + 1] - tw) / 2)
                    bl = y + (rh + f.ascender * size + f.descender * size) / 2
                    self._put(tx, bl, t, fn, size)
                    seg_cells.append(
                        {
                            "row": nh + bi,
                            "col": j,
                            "colspan": 1,
                            "rowspan": 1,
                            "header": False,
                            "text": t,
                            "page": self.pno,
                            "segment": si,
                            "seg_row": nh + bi - r0,
                            "repeated_header": False,
                            "bbox": [round(xs[j], 2), round(y, 2), round(xs[j + 1], 2), round(y + rh, 2)],
                        }
                    )
                y += rh
            if style["rules"] in ("booktabs", "plain"):
                page.draw_line((xs[0], y), (xs[-1], y), color=black, width=lw_heavy if style["rules"] == "booktabs" else lw_light)
            if style["rules"] == "grid":
                yy = top
                for i in range(nrows + 1):
                    page.draw_line((xs[0], yy), (xs[-1], yy), color=black, width=lw_light if 0 < i < nrows else lw_heavy * 0.8)
                    yy += rh
                for c in seg_cells:
                    bx = c["bbox"]
                    page.draw_line((bx[0], bx[1]), (bx[0], bx[3]), color=black, width=lw_light)
                    page.draw_line((bx[2], bx[1]), (bx[2], bx[3]), color=black, width=lw_light)
            cells += seg_cells
            bboxes.append({"page": self.pno, "bbox": [round(xs[0], 2), round(top, 2), round(xs[-1], 2), round(y, 2)]})
            self.y = y + 3
        notes = []
        for k, note in enumerate(spec.get("footnotes", [])):
            r = self.paragraph(
                [(note, f"fn{k}")],
                fname="sans",
                size=size - 1,
                align="left",
                gap_after=0,
                x0=xs[0],
                width=max(total, 0.6 * self.text_width),
            )
            notes.append(r[f"fn{k}"])
        self.y += 10
        return {
            "page": pages[0],
            "bbox": bboxes[0]["bbox"],
            "segments": bboxes,
            "font_size": size,
            "cells": cells,
            "caption": caps[0],
            "continued_captions": caps[1:],
            "footnotes": notes,
        }

    def save(self, path):
        """Write the PDF to path."""
        self.pdf.set_metadata(
            {
                "title": "",
                "author": "ORCHID synthetic benchmark",
                "producer": "PyMuPDF",
                "creator": "orchid-pipeline-bench",
                "creationDate": "",
                "modDate": "",
            }
        )
        try:
            self.pdf.subset_fonts()
        except Exception:  # noqa: BLE001
            pass
        self.pdf.save(path, garbage=3, deflate=True, no_new_id=True)
        self.pdf.close()
