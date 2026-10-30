"""Page-to-rendering geometry and per-rendering page images for the synthetic documents, split from generate.py."""

import io
import math
import numpy as np
from common import sha256_file  # noqa: E402


def page_affine(scale_x, scale_y, angle_deg=0.0, size=(0, 0)):
    """2x3 matrix M mapping vector-PDF points (x, y down) to rendering coordinates: [u, v] = M @ [x, y, 1]."""
    if not angle_deg:
        return [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0]]
    t = math.radians(angle_deg)
    cx, cy = size[0] / 2, size[1] / 2
    c, s = math.cos(t), math.sin(t)
    # p' = C + R (S p - C) with R = [[c, s], [-s, c]]
    return [[c * scale_x, s * scale_y, cx - c * cx - s * cy], [-s * scale_x, c * scale_y, cy + s * cx - c * cy]]


def render_all(vector_pdf, outdir, rng):
    """Render the page images of every rendering of a document into outdir."""
    import pymupdf
    from PIL import Image, ImageEnhance, ImageFilter

    doc = pymupdf.open(vector_pdf)
    rend = {
        "vector": {
            "file": "vector.pdf",
            "sha256": sha256_file(vector_pdf),
            "pages": doc.page_count,
            "units": "pt",
            "page_size_pt": [doc[0].rect.width, doc[0].rect.height],
            "transforms": [page_affine(1, 1) for _ in range(doc.page_count)],
            "text_layer": True,
        }
    }
    clean_files, deg_files, deg_tr, angles = [], [], [], []
    (outdir / "clean").mkdir(exist_ok=True)
    (outdir / "degraded").mkdir(exist_ok=True)
    clean_tr = []
    for i, page in enumerate(doc):
        pm = page.get_pixmap(dpi=300, alpha=False)
        fn = outdir / "clean" / f"page-{i + 1}.png"
        pm.save(str(fn))
        clean_files.append({"file": f"clean/page-{i + 1}.png", "sha256": sha256_file(fn), "size_px": [pm.width, pm.height]})
        clean_tr.append(page_affine(pm.width / page.rect.width, pm.height / page.rect.height))
        pm2 = page.get_pixmap(dpi=150, alpha=False)
        im = Image.frombytes("RGB", (pm2.width, pm2.height), pm2.samples)
        ang = float(np.round(rng.uniform(-1.0, 1.0), 3))
        gray = bool(rng.random() < 0.5)
        im = ImageEnhance.Contrast(im).enhance(float(rng.uniform(0.72, 0.88)))
        im = ImageEnhance.Brightness(im).enhance(float(rng.uniform(0.95, 1.03)))
        im = im.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.5, 0.9))))
        im = im.rotate(ang, resample=Image.BICUBIC, expand=False, fillcolor=(246, 245, 240))
        arr = np.asarray(im).copy()
        dens = float(rng.uniform(0.0008, 0.0025))
        m = rng.random(arr.shape[:2])
        arr[m < dens / 2] = 0
        arr[(m >= dens / 2) & (m < dens)] = 255
        im = Image.fromarray(arr)
        if gray:
            im = im.convert("L")
        q = int(rng.integers(30, 55))
        fn2 = outdir / "degraded" / f"page-{i + 1}.jpg"
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q, optimize=False)
        fn2.write_bytes(buf.getvalue())
        deg_files.append(
            {
                "file": f"degraded/page-{i + 1}.jpg",
                "sha256": sha256_file(fn2),
                "size_px": [im.width, im.height],
                "rotation_deg": ang,
                "jpeg_quality": q,
                "grayscale": gray,
                "salt_pepper_density": round(dens, 5),
            }
        )
        deg_tr.append(page_affine(pm2.width / page.rect.width, pm2.height / page.rect.height, ang, (pm2.width, pm2.height)))
        angles.append(ang)
    rend["clean_raster"] = {"files": clean_files, "dpi": 300, "units": "px", "transforms": clean_tr, "text_layer": False}
    rend["degraded_raster"] = {
        "files": deg_files,
        "dpi": 150,
        "units": "px",
        "transforms": deg_tr,
        "text_layer": False,
        "degradation": "contrast loss, brightness shift, Gaussian blur, rotation about the page "
        "centre (PIL, counter-clockwise positive), salt-and-pepper noise, JPEG",
    }
    # image-only PDF wrapping the degraded raster
    out = pymupdf.open()
    io_tr = []
    for i, page in enumerate(doc):
        p2 = out.new_page(width=page.rect.width, height=page.rect.height)
        p2.insert_image(p2.rect, filename=str(outdir / "degraded" / f"page-{i + 1}.jpg"))
        w, h = deg_files[i]["size_px"]
        sx, sy = page.rect.width / w, page.rect.height / h
        M = deg_tr[i]
        io_tr.append([[M[0][0] * sx, M[0][1] * sx, M[0][2] * sx], [M[1][0] * sy, M[1][1] * sy, M[1][2] * sy]])
    out.set_metadata({"producer": "PyMuPDF", "creator": "orchid-pipeline-bench", "creationDate": "", "modDate": ""})
    out.save(str(outdir / "image_only.pdf"), garbage=3, deflate=True, no_new_id=True)
    out.close()
    rend["image_only_pdf"] = {
        "file": "image_only.pdf",
        "sha256": sha256_file(outdir / "image_only.pdf"),
        "units": "pt",
        "transforms": io_tr,
        "text_layer": False,
        "note": "each page is the degraded raster placed full-page; no text layer",
    }
    doc.close()
    return rend


def apply(M, x, y):
    """Map the point (x, y) through the transform M."""
    return [round(M[0][0] * x + M[0][1] * y + M[0][2], 3), round(M[1][0] * x + M[1][1] * y + M[1][2], 3)]


def rect_in_renderings(page, bbox, rend):
    """Express a page bounding box in the coordinate frame of a rendering."""
    out = {}
    for key, rr in rend.items():
        M = rr["transforms"][page]
        x0, y0, x1, y1 = bbox
        out[key] = [apply(M, x0, y0), apply(M, x1, y0), apply(M, x1, y1), apply(M, x0, y1)]
    return out
