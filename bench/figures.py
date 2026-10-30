"""Matplotlib figure generators. Each returns the truth for the figure (axes in figure fractions, data series,
boundaries, labels) and writes a vector PDF; layout is explicit (fig.add_axes) so truth geometry is exact.
"""
from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from systems import base  # noqa: E402

plt.rcParams.update({
    "pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.8,
    "svg.hashsalt": "orchid-bench", "path.simplify": False,
})
pdf_meta = {"CreationDate": None, "Producer": "ORCHID synthetic benchmark (matplotlib)", "Creator": None}
colors = ["#1f4e9c", "#c0392b", "#1e8449", "#7d3c98", "#b9770e", "#117a65", "#566573"]
markers = ["o", "s", "^", "v", "D", "x", "+"]


def _save(fig, path):
    fig.savefig(path, format="pdf", metadata=pdf_meta)
    plt.close(fig)


def _axes_record(ax, fig, name, xlim, ylim, xunit, yunit, xlabel, ylabel, xscale="linear", yscale="linear", role="main"):
    pos = ax.get_position()
    return {"id": name, "role": role,
            "bbox_fig": [float(pos.x0), float(pos.y0), float(pos.x1), float(pos.y1)],  # figure fraction, origin bottom-left
            "x": {"min": float(xlim[0]), "max": float(xlim[1]), "unit": xunit, "label": xlabel, "scale": xscale},
            "y": {"min": float(ylim[0]), "max": float(ylim[1]), "unit": yunit, "label": ylabel, "scale": yscale}}


# --------------------------------------------------------------------------------------------- phase diagram
def phase_diagram(data, style, path, rng):
    """Draw a phase diagram from data and style and save it to path as a vector PDF."""
    A, B = data["elements"]
    tunit, xunit = style["T_unit"], style["x_unit"]
    xs_ = 100.0 if xunit == "at.%" else 1.0
    tof = (lambda T: T - 273.15) if tunit == "°C" else (lambda T: T)
    Tmax_K = data["T_axis_max"]
    if tunit == "°C":
        ylim = (50.0, float(np.ceil((Tmax_K - 273.15) / 100.0) * 100.0))
    else:
        ylim = (300.0, Tmax_K)
    xlim = (0.0, xs_)
    W, H = style["fig_size"]
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.13, 0.12, 0.83, 0.80] if not style["legend"] else [0.13, 0.12, 0.83, 0.80])
    lw = style["linewidth"]
    colored = style["color"]
    color_of = {"LIQUID": colors[0]}
    truth_bounds = []
    for k, rg in enumerate(data["boundaries"]):
        for c in rg["curves"]:
            x = np.asarray(c["x"]) * xs_
            T = np.array([tof(t) for t in c["T"]])
            on_frame = bool(np.all(np.asarray(c["x"]) < 1e-3) or np.all(np.asarray(c["x"]) > 1 - 1e-3))
            col = "black"
            if colored:
                col = color_of.setdefault(base(c["phase"]), colors[len(color_of) % len(colors)])
            ax.plot(x, T, color=col, lw=lw, solid_capstyle="round", zorder=3)
            truth_bounds.append({"region": k, "phase": c["phase"], "two_phase": rg["phases"], "on_frame": on_frame,
                                 "x": [float(v) for v in c["x"]], "T_K": [float(v) for v in c["T"]],
                                 "color": col, "linestyle": "solid"})
    inv_lines = []
    for i, inv in enumerate(data["invariants"]):
        xs = [p["x"] for p in inv["points"]]
        ax.plot([min(xs) * xs_, max(xs) * xs_], [tof(inv["T"])] * 2, color="black", lw=lw, zorder=3)
        inv_lines.append({"invariant": i, "T_K": inv["T"], "x_min": min(xs), "x_max": max(xs)})
    texts = []
    # temperature annotations on invariant lines
    if style["invariant_T_labels"]:
        for i, inv in enumerate(data["invariants"]):
            xs = [p["x"] for p in inv["points"]]
            Tv = tof(inv["T"])
            s = f"{Tv:.0f}" + ("°C" if tunit == "°C" else " K")
            xr = max(xs) * xs_ + 0.01 * xs_
            ha = "left"
            if max(xs) > 0.85:
                xr, ha = min(xs) * xs_ - 0.01 * xs_, "right"
            ax.text(xr, Tv, s, fontsize=6.5, ha=ha, va="center", zorder=5,
                    bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none") if style["label_bg"] else None)
            texts.append({"kind": "invariant_T", "invariant": i, "text": s, "value": float(f"{Tv:.0f}"),
                          "unit": "°C" if tunit == "°C" else "K", "x": xr / xs_, "T_K": inv["T"]})
    # field labels
    labels_out = []
    ax_w_pt = W * 72 * 0.83
    ax_h_pt = H * 72 * 0.80
    fs = style["label_fontsize"]
    for f in data["fields"]:
        lab = " + ".join(l_ for _, l_ in sorted(zip(f["compositions"], f["labels"])))
        printed = None
        if style["field_labels"] and f["n_cells"] >= 6:
            if style["label_names"] == "tdb":
                lab_t = " + ".join(base(p_) for _, p_ in sorted(zip(f["compositions"], f["assemblage"])))
            else:
                lab_t = lab
            half_w = (len(lab_t) * fs * 0.56 / 2) / ax_w_pt
            half_h = (fs * 0.75) / ax_h_pt
            x0, T0 = f["point"]
            yfrac = (tof(T0) - ylim[0]) / (ylim[1] - ylim[0])
            if f["inradius"] > max(half_w, half_h) * 1.05 and 0.03 < yfrac < 0.97:
                ax.text(x0 * xs_, tof(T0), lab_t, fontsize=fs, ha="center", va="center", zorder=4)
                printed = lab_t
        labels_out.append({"field": f["id"], "printed_label": printed, "conventional_label": lab, "point_x": f["point"][0], "point_T_K": f["point"][1]})
    # experimental-style markers near boundaries
    marker_series = []
    if style["markers"]:
        names = ["DTA, synthetic", "EPMA, synthetic"]
        cand = [b for b in truth_bounds if not b["on_frame"]]
        for si, nm in enumerate(names):
            pts = []
            liq = [b for b in cand if (base(b["phase"]) == "LIQUID") == (si == 0)] or cand
            for _ in range(int(rng.integers(6, 12))):
                b = liq[int(rng.integers(len(liq)))]
                j = int(rng.integers(len(b["x"])))
                xx = float(np.clip(b["x"][j] + rng.normal(0, 0.006), 0.0, 1.0))
                TT = float(b["T_K"][j] + rng.normal(0, 6.0))
                if ylim[0] < tof(TT) < ylim[1]:
                    pts.append([round(xx, 5), round(TT, 2)])
            mk = markers[si]
            ax.scatter([p[0] * xs_ for p in pts], [tof(p[1]) for p in pts], marker=mk, s=16,
                       facecolors="none" if si == 0 else "black", edgecolors="black", linewidths=0.7, zorder=6, label=nm)
            marker_series.append({"name": nm, "marker": mk, "points": [{"x": p[0], "T_K": p[1]} for p in pts]})
    overlay = None
    if style["overlay"]:
        dx_, dT_ = style["overlay_offset"]
        for b in truth_bounds:
            if b["on_frame"]:
                continue
            x = np.clip(np.asarray(b["x"]) + dx_, 0, 1) * xs_
            T = np.array([tof(t + dT_) for t in b["T_K"]])
            ax.plot(x, T, color="#7f7f7f" if not colored else "#d35400", lw=lw * 0.9, ls=(0, (4, 2.5)), zorder=2)
        overlay = {"name": "Previous assessment (synthetic, shifted)", "dx": dx_, "dT_K": dT_, "linestyle": "dashed",
                   "applies_to": "every boundary curve not on the frame; x clipped to [0, 1]"}
    if style["legend"]:
        handles = [Line2D([], [], color="black", lw=lw, label="This work (calculated)")]
        if overlay:
            handles.append(Line2D([], [], color="#7f7f7f" if not colored else "#d35400", lw=lw * 0.9, ls=(0, (4, 2.5)),
                                  label=overlay["name"]))
        for ms in marker_series:
            handles.append(Line2D([], [], ls="none", marker=ms["marker"], mec="black",
                                  mfc="none" if ms["marker"] == "o" else "black", ms=4, label=ms["name"]))
        ax.legend(handles=handles, loc=style["legend_loc"], fontsize=6.5, frameon=True, framealpha=1.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    xl = f"{B} / at.%" if xunit == "at.%" else f"Mole fraction {B}"
    yl = "Temperature / °C" if tunit == "°C" else "Temperature / K"
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.text(0.0, 1.01, A, transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
    ax.text(1.0, 1.01, B, transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
    axes = [_axes_record(ax, fig, "main", xlim, ylim, xunit, tunit, xl, yl)]
    inset = None
    if style["inset"] and data["invariants"]:
        inv = min(data["invariants"], key=lambda q: q["T"])
        xm = inv["points"][1]["x"]
        ix = (max(0.0, xm - 0.08) * xs_, min(1.0, xm + 0.08) * xs_)
        iy = (tof(inv["T"] - 50), tof(inv["T"] + 50))
        corner = [0.62, 0.60, 0.30, 0.28] if xm < 0.5 else [0.20, 0.60, 0.30, 0.28]
        axi = fig.add_axes(corner)
        for b in truth_bounds:
            axi.plot(np.asarray(b["x"]) * xs_, [tof(t) for t in b["T_K"]], color=b["color"], lw=lw)
        for inv_ in data["invariants"]:
            xs = [p["x"] for p in inv_["points"]]
            axi.plot([min(xs) * xs_, max(xs) * xs_], [tof(inv_["T"])] * 2, color="black", lw=lw)
        axi.set_xlim(*ix)
        axi.set_ylim(*iy)
        axi.tick_params(labelsize=6)
        axi.set_facecolor("white")
        axi.set_zorder(10)
        ax.add_patch(plt.Rectangle((ix[0], iy[0]), ix[1] - ix[0], iy[1] - iy[0], fill=False, lw=0.6, ls=":", zorder=7))
        axes.append(_axes_record(axi, fig, "inset", ix, iy, xunit, tunit, "", "", role="inset"))
        inset = {"zoom_of_invariant": data["invariants"].index(inv)}
    occluders = []
    fig.canvas.draw()
    inv_t = ax.transData.inverted()
    boxes = []
    if ax.get_legend() is not None:
        boxes.append(("legend", ax.get_legend().get_window_extent()))
    if len(axes) > 1:
        boxes.append(("inset", axi.get_tightbbox()))
    for name, bb in boxes:
        (u0, v0), (u1, v1) = inv_t.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        occluders.append({"kind": name, "x_plot": [float(min(u0, u1)), float(max(u0, u1))],
                          "T_plot": [float(min(v0, v1)), float(max(v0, v1))]})
    _save(fig, path)
    return {"kind": "phase_diagram", "occluders": occluders, "fig_size_in": [W, H], "axes": axes, "style": style,
            "boundaries": truth_bounds, "invariant_lines": inv_lines, "field_labels": labels_out,
            "text_in_figure": texts, "marker_series": marker_series, "overlay": overlay, "inset": inset,
            "units": {"x": xunit, "T": tunit, "x_scale_factor": xs_,
                      "T_conversion": "T[°C] = T[K] - 273.15" if tunit == "°C" else "none"}}


# --------------------------------------------------------------------------------------------- property plots
def _series_plot(ax, series, style):
    out = []
    for i, s in enumerate(series):
        col = colors[i % len(colors)] if style["color"] else "black"
        ls = ["-", "--", "-.", ":"][i % 4] if not style["color"] else "-"
        mk = markers[i % len(markers)]
        x, y = np.asarray(s["x"], float), np.asarray(s["y"], float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        rep = s.get("representation", style["representation"])
        rec = {"name": s["name"], "unit_x": s["unit_x"], "unit_y": s["unit_y"], "color": col, "representation": rep}
        if rep in ("line", "line+markers"):
            ax.plot(x, y, color=col, ls=s.get("ls", ls), lw=style["linewidth"], label=s["name"] if rep == "line" else None)
            rec["line"] = [[float(a), float(b)] for a, b in zip(x, y)]
            rec["linestyle"] = s.get("ls", ls)
        if rep in ("markers", "line+markers"):
            step = s.get("marker_step", 4)
            xm, ym = x[::step], y[::step]
            ax.plot(xm, ym, ls="none", marker=mk, ms=3.5, mfc="none" if i % 2 == 0 else col, mec=col, mew=0.8,
                    label=s["name"] if rep != "line" else None)
            if rep == "line+markers":
                ax.plot([], [], color=col, ls=ls, marker=mk, ms=3.5, label=None)
            rec["markers"] = [[float(a), float(b)] for a, b in zip(xm, ym)]
            rec["marker"] = mk
        out.append(rec)
    return out


def _limits(series, pad=0.06, log=False):
    ys = np.concatenate([np.asarray(s["y"], float) for s in series])
    ys = ys[np.isfinite(ys)]
    if log:
        lo, hi = np.log10(max(ys.min(), 1e-6)), np.log10(ys.max())
        return 10 ** np.floor(lo), 10 ** np.ceil(hi)
    lo, hi = ys.min(), ys.max()
    span = hi - lo if hi > lo else 1.0
    lo, hi = lo - pad * span, hi + pad * span
    step = 10 ** np.floor(np.log10(span / 2))
    return float(np.floor(lo / step) * step), float(np.ceil(hi / step) * step)


def property_figure(data, kind, style, path):
    """Draw a property figure of the given kind and save it to path as a vector PDF."""
    A, B = data["elements"]
    P = data["properties"]
    W, H = style["fig_size"]
    fig = plt.figure(figsize=(W, H))
    xs_ = 100.0 if style["x_unit"] == "at.%" else 1.0
    xlab = f"{B} / at.%" if xs_ == 100 else f"x({B})"
    panels = []
    title = ""
    if kind == "mixing":
        m = P["mixing"]
        kj = style["energy_unit"] == "kJ/mol"
        f = 1e-3 if kj else 1.0
        ser = [{"name": f"T = {T:.0f} K", "x": [x * xs_ for x in m["x"]], "y": [h * f for h in H_],
                "unit_x": style["x_unit"], "unit_y": style["energy_unit"], "marker_step": 5}
               for T, H_ in zip(m["T"], m["HM_MIX"])]
        if not style["two_series"]:
            ser = ser[:1]
        ax = fig.add_axes([0.17, 0.14, 0.78, 0.80])
        recs = _series_plot(ax, ser, style)
        ylim = _limits(ser)
        ylab = f"ΔmixH / {style['energy_unit']}"
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set(xlim=(0, xs_), ylim=ylim, xlabel=xlab, ylabel=ylab)
        ax.legend(fontsize=6.5, loc="best")
        panels.append({**_axes_record(ax, fig, "a", (0, xs_), ylim, style["x_unit"], style["energy_unit"], xlab, ylab),
                       "quantity": "HM_MIX", "phase": "LIQUID", "reference_state": "pure liquid A and B at T",
                       "series": recs})
        title = f"mixing enthalpy of the liquid at {m['T'][0]:.0f} K" + (f" and {m['T'][1]:.0f} K" if style["two_series"] else "")
    elif kind == "activity":
        a = P["activity"]
        log = style["log_y"]
        ser = [{"name": f"a({A})", "x": [x * xs_ for x in a["x"]], "y": a["a_A"], "unit_x": style["x_unit"], "unit_y": "1"},
               {"name": f"a({B})", "x": [x * xs_ for x in a["x"]], "y": a["a_B"], "unit_x": style["x_unit"], "unit_y": "1"}]
        if style["raoult"] and not log:
            ser.append({"name": "Raoult", "x": [0, xs_], "y": [1, 0], "unit_x": style["x_unit"], "unit_y": "1",
                        "representation": "line", "ls": ":"})
            ser.append({"name": "Raoult ", "x": [0, xs_], "y": [0, 1], "unit_x": style["x_unit"], "unit_y": "1",
                        "representation": "line", "ls": ":"})
        ax = fig.add_axes([0.17, 0.14, 0.78, 0.80])
        if log:
            ax.set_yscale("log")
        recs = _series_plot(ax, ser, style)
        ylim = _limits(ser[:2], log=True) if log else (0.0, 1.0)
        ax.set(xlim=(0, xs_), ylim=ylim, xlabel=xlab, ylabel="Activity")
        ax.legend(fontsize=6.5, loc="best")
        panels.append({**_axes_record(ax, fig, "a", (0, xs_), ylim, style["x_unit"], "1", xlab, "Activity",
                                      yscale="log" if log else "linear"),
                       "quantity": "ACR", "phase": "LIQUID", "reference_state": "pure liquid at T", "T_K": a["T"],
                       "series": recs})
        title = f"activities in the liquid at {a['T']:.0f} K (reference: pure liquids)"
    elif kind == "cp":
        c = P["cp"]
        ser = [{"name": c["label"], "x": c["T"], "y": c["CPM"], "unit_x": "K", "unit_y": "J/(mol K)", "marker_step": 3}]
        ax = fig.add_axes([0.17, 0.14, 0.78, 0.80])
        recs = _series_plot(ax, ser, style)
        ylim = _limits(ser)
        xlim = (float(c["T"][0]), float(c["T"][-1]))
        ax.set(xlim=xlim, ylim=ylim, xlabel="Temperature / K", ylabel="Cp / J mol⁻¹ K⁻¹")
        ax.legend(fontsize=6.5, loc="best")
        panels.append({**_axes_record(ax, fig, "a", xlim, ylim, "K", "J/(mol K)", "Temperature / K", "Cp / J mol⁻¹ K⁻¹"),
                       "quantity": "CPM", "phase": c["phase"], "x_B": c["x"], "series": recs,
                       "basis": "per mole of atoms"})
        title = f"heat capacity of {c['label']}"
    elif kind == "gibbs":
        g = P["gibbs"]
        T = g["T"]
        ser = []
        for cv in g["curves"]:
            x = np.asarray(cv["x"])
            ref = (1 - x) * g["ref_A"] + x * g["ref_B"]
            y = (np.asarray(cv["GM"]) - ref) / 1000.0
            keep = y < (np.nanmin(y) + 0.6 * (np.nanmax(y) - np.nanmin(y)) + 2)
            ser.append({"name": cv["phase"], "x": (x[keep] * xs_).tolist(), "y": y[keep].tolist(),
                        "unit_x": style["x_unit"], "unit_y": "kJ/mol", "representation": "line"})
        tp = g["tangent"]
        if len(tp) == 2:
            xa, xb = tp[0]["x"], tp[1]["x"]
            muA, muB = g["mu"]
            ya = ((1 - xa) * muA + xa * muB - ((1 - xa) * g["ref_A"] + xa * g["ref_B"])) / 1000
            yb = ((1 - xb) * muA + xb * muB - ((1 - xb) * g["ref_A"] + xb * g["ref_B"])) / 1000
            ext = 0.12
            xt = [max(0, xa - ext), min(1, xb + ext)]
            yt = [ya + (yb - ya) * (q - xa) / (xb - xa) for q in xt]
            ser.append({"name": "common tangent", "x": [q * xs_ for q in xt], "y": yt, "unit_x": style["x_unit"],
                        "unit_y": "kJ/mol", "representation": "line", "ls": (0, (1, 1.2))})
        ax = fig.add_axes([0.17, 0.14, 0.78, 0.80])
        recs = _series_plot(ax, ser, {**style, "representation": "line"})
        ylim = _limits(ser)
        ax.set(xlim=(0, xs_), ylim=ylim, xlabel=xlab, ylabel="G − G_ref / kJ mol⁻¹")
        ax.legend(fontsize=6.5, loc="best")
        panels.append({**_axes_record(ax, fig, "a", (0, xs_), ylim, style["x_unit"], "kJ/mol", xlab, "G − G_ref / kJ mol⁻¹"),
                       "quantity": "GM", "T_K": T, "reference_state": "pure liquid A and B at T",
                       "tangent_points": tp, "series": recs})
        title = f"Gibbs energy curves at {T:.0f} K"
    elif kind == "two_panel":
        m = P["mixing"]
        top = fig.add_axes([0.17, 0.585, 0.78, 0.37])
        bot = fig.add_axes([0.17, 0.155, 0.78, 0.37])
        sH = [{"name": f"ΔmixH, {m['T'][0]:.0f} K", "x": [x * xs_ for x in m["x"]], "y": [h / 1000 for h in m["HM_MIX"][0]],
               "unit_x": style["x_unit"], "unit_y": "kJ/mol", "marker_step": 5}]
        sG = [{"name": f"ΔmixG, {m['T'][0]:.0f} K", "x": [x * xs_ for x in m["x"]], "y": [h / 1000 for h in m["GM_MIX"][0]],
               "unit_x": style["x_unit"], "unit_y": "kJ/mol", "marker_step": 5}]
        r1 = _series_plot(top, sH, style)
        r2 = _series_plot(bot, sG, style)
        yl1, yl2 = _limits(sH), _limits(sG)
        top.set(xlim=(0, xs_), ylim=yl1, ylabel="ΔmixH / kJ mol⁻¹")
        top.tick_params(labelbottom=False)
        bot.set(xlim=(0, xs_), ylim=yl2, xlabel=xlab, ylabel="ΔmixG / kJ mol⁻¹")
        top.text(0.02, 0.9, "(a)", transform=top.transAxes, fontsize=8)
        bot.text(0.02, 0.9, "(b)", transform=bot.transAxes, fontsize=8)
        panels.append({**_axes_record(top, fig, "a", (0, xs_), yl1, style["x_unit"], "kJ/mol", "", "ΔmixH / kJ mol⁻¹"),
                       "quantity": "HM_MIX", "phase": "LIQUID", "T_K": m["T"][0], "series": r1})
        panels.append({**_axes_record(bot, fig, "b", (0, xs_), yl2, style["x_unit"], "kJ/mol", xlab, "ΔmixG / kJ mol⁻¹"),
                       "quantity": "GM_MIX", "phase": "LIQUID", "T_K": m["T"][0], "series": r2})
        title = f"(a) mixing enthalpy and (b) mixing Gibbs energy of the liquid at {m['T'][0]:.0f} K"
    else:
        raise ValueError(kind)
    _save(fig, path)
    return {"kind": f"property_{kind}", "fig_size_in": [W, H], "axes": panels, "style": style, "title": title}


# --------------------------------------------------------------------------------------------- phase fractions
def phase_fraction_figure(data, alloys, style, path):
    """Equilibrium phase fractions vs temperature for one or two alloys (one panel each, stacked).
    Each phase is drawn only where it is present (NP > 0); segments are recorded in the truth.
    """
    A, B = data["elements"]
    tunit = style["T_unit"]
    tof = (lambda T: T - 273.15) if tunit == "°C" else (lambda T: T)
    W = style["fig_size"][0]
    H = 2.6 if len(alloys) == 1 else 4.4
    fig = plt.figure(figsize=(W, H))
    n = len(alloys)
    panels = []
    xlab = "Temperature / °C" if tunit == "°C" else "Temperature / K"
    for k, al in enumerate(alloys):
        if n == 1:
            rect = [0.15, 0.17, 0.80, 0.76]
        else:
            rect = [0.15, 0.585 if k == 0 else 0.105, 0.80, 0.37]
        ax = fig.add_axes(rect)
        T = np.asarray(al["T"], float)
        Tp = np.array([tof(t) for t in T])
        recs = []
        for i, (lab, s) in enumerate(al["series"].items()):
            y = np.asarray(s["NP"], float)
            on = y > 0
            col = colors[i % len(colors)] if style["color"] else "black"
            ls = "-" if style["color"] else ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))][i % 6]
            segs, line = [], []
            j = 0
            while j < len(y):
                if on[j]:
                    j0 = j
                    while j < len(y) and on[j]:
                        j += 1
                    xs_, ys_ = Tp[j0:j], y[j0:j]
                    ax.plot(xs_, ys_, color=col, ls=ls, lw=style["linewidth"], label=lab if not segs else None)
                    segs.append([len(line), len(line) + (j - j0)])
                    line += [[float(a), float(b)] for a, b in zip(xs_, ys_)]
                else:
                    j += 1
            recs.append({"name": lab, "tdb_phase": s["tdb"], "unit_x": tunit, "unit_y": "1", "color": col,
                         "linestyle": ls if isinstance(ls, str) else "custom", "representation": "line",
                         "line": line, "segments": segs})
        xlim = (float(tof(T[0])), float(tof(T[-1])))
        ylim = (0.0, 1.05)
        ax.set(xlim=xlim, ylim=ylim, ylabel="Phase fraction")
        if k == n - 1:
            ax.set_xlabel(xlab)
        else:
            ax.tick_params(labelbottom=False)
        xl = f"{100 * al['x']:.0f} at.% {B}" if style["x_unit"] == "at.%" else f"x({B}) = {al['x']:.2f}"
        ax.text(0.02, 0.92, ("(a) " if n > 1 and k == 0 else "(b) " if n > 1 else "") + f"{A}–{xl}", transform=ax.transAxes,
                fontsize=7, va="top")
        ax.legend(fontsize=6, loc="center right", ncol=1)
        panels.append({**_axes_record(ax, fig, "ab"[k], xlim, ylim, tunit, "1", xlab if k == n - 1 else "", "Phase fraction"),
                       "quantity": "NP", "x_B": al["x"], "basis": "mole fraction of atoms in each phase", "series": recs})
    _save(fig, path)
    comp = " and ".join((f"{100 * a['x']:.0f} at.% {B}" if style["x_unit"] == "at.%" else f"x({B}) = {a['x']:.2f}") for a in alloys)
    return {"kind": "property_phase_fraction", "fig_size_in": [W, H], "axes": panels, "style": style,
            "title": f"equilibrium phase fractions versus temperature for {A}–{comp}"}
