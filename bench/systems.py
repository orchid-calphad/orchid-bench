"""The benchmark's system list: one SGTE assessment per binary, chosen to span diagram types."""
from __future__ import annotations

import re

from common import el

# (tdb stem, diagram type tags, display-label overrides {TDB phase: label or [(x_lo, x_hi, label), ...]})
system_specs = [
    ("AgCu-06He", ["simple_eutectic"], {}),
    ("BiSn-08Li", ["simple_eutectic"], {}),
    ("PbSn-17Gie", ["simple_eutectic"], {}),
    ("AlSi-12Tan", ["simple_eutectic"], {}),
    ("CuNi-87Jan", ["isomorphous", "solid_miscibility_gap"], {}),
    ("CuPb-86Hay-LB", ["miscibility_gap", "monotectic"], {}),
    ("AlBi-06Kim", ["miscibility_gap", "monotectic"], {}),
    ("PbZn-98Jan", ["miscibility_gap", "monotectic"], {}),
    ("AgNi-04Din", ["miscibility_gap", "monotectic"], {}),
    ("CuZn-08Gie", ["peritectic", "intermetallic_rich", "order_disorder"],
     {"BCC_A2": [(0.0, 0.65, "β"), (0.65, 1.0, "δ")], "CUZN_B2": "β′", "CU5ZN8_D82": "γ", "HCP_IM_A3": "ε"}),
    ("AgSn-12Gie", ["peritectic"], {"HCP_A3": "ζ", "AG3SN_D0A": "Ag3Sn"}),
    ("AlNi-14Zhu", ["intermetallic_rich", "order_disorder", "congruent_compound"],
     {"AL3NI_D011": "Al3Ni", "AL3NI2_D513": "Al3Ni2", "AL3NI5": "Al3Ni5", "B2_BCC": "NiAl", "L12_FCC": "Ni3Al"}),
    ("NiTi-04Tok", ["intermetallic_rich", "congruent_compound"],
     {"NI3TI_D024": "Ni3Ti", "NITI2": "NiTi2", "NITI_B2": "NiTi", "HCP_A3": "(αTi)", "BCC_A2": "(βTi)"}),
    ("AlCu-02Mie", ["intermetallic_rich", "peritectic"], {"AL4CU9_GAMMA": "γ", "BCC_A2": "β", "HCP_A3": "ζ"}),
    ("MgSi-12Sch", ["congruent_compound", "simple_eutectic"], {"MG2SI_C1": "Mg2Si"}),
    ("MgSn-04Fri", ["congruent_compound"], {"MG2SN_C1": "Mg2Sn"}),
    ("CuMg-16Hal", ["congruent_compound", "intermetallic_rich"], {"C15_LAVES": "Cu2Mg", "CUMG2_CB": "CuMg2"}),
    ("AlZn-93Mey-LB", ["simple_eutectic", "solid_miscibility_gap", "monotectoid"], {}),
    ("AlMg-12Tan", ["intermetallic_rich", "peritectic"], {"AL12MG17_A12": "Al12Mg17", "ALMG_BETA": "β", "ALMG_EPS": "ε"}),
    ("CrFe-11Xio", ["intermetallic", "gamma_loop", "solid_miscibility_gap"], {"SIGMA_D8B": "σ", "FCC_A1": "(γFe)"}),
]
system_index = {s[0]: s for s in system_specs}


def base(phase: str) -> str:
    """Return the phase name without its sublattice suffix (the text before the hash sign)."""
    return phase.split("#", maxsplit=1)[0]


def pretty_formula(name: str, elements: list[str]) -> str | None:
    """'AL3NI2_D513' -> 'Al3Ni2' when the prefix is a formula in the system's elements."""
    head = name.split("_", maxsplit=1)[0]
    syms = sorted({e.upper() for e in elements}, key=len, reverse=True)
    pat = re.compile("(" + "|".join(syms) + r")(\d*)")
    pos, out = 0, []
    while pos < len(head):
        m = pat.match(head, pos)
        if not m:
            return None
        out.append(el(m.group(1)) + m.group(2))
        pos = m.end()
    return "".join(out) if len(out) >= 2 else None


class Labeller:
    """Map (TDB phase, composition) to the conventional label printed in the synthetic paper."""

    def __init__(self, stem, elements, stable_A, stable_B, continuous):
        """Store the system description: stem, display elements, stable phases A and B, and continuous phases."""
        self.elements = elements                  # display symbols, e.g. ['Ag', 'Cu']
        self.stable_A, self.stable_B = stable_A, stable_B
        self.continuous = continuous              # phases forming a single-phase band across all x
        self.overrides = system_index[stem][2] if stem in system_index else {}

    def label(self, phase: str, x: float) -> str:
        """Return the display label of phase at composition x."""
        b = base(phase)
        A, B = self.elements
        if b == "LIQUID":
            return "L"
        ov = self.overrides.get(b)
        if isinstance(ov, str):
            return ov
        if isinstance(ov, list):
            for lo, hi, lab in ov:
                if lo <= x <= hi:
                    return lab
        if b in self.continuous:
            return f"({A},{B})"
        inA, inB = b in self.stable_A, b in self.stable_B
        if inA and inB:
            return f"({A})" if x < 0.5 else f"({B})"
        if inA:
            return f"({A})"
        if inB:
            return f"({B})"
        return pretty_formula(b, self.elements) or b

    def assemblage(self, comps: list[dict]) -> list[str]:
        """Labels for a list of {phase, x} composition sets; two liquids become L1 (lower x) and L2."""
        labs = [self.label(c["phase"], c["x"]) for c in comps]
        order = sorted(range(len(comps)), key=lambda i: comps[i]["x"])
        for lab in set(labs):
            idx = [i for i in order if labs[i] == lab]
            if len(idx) > 1:
                for n, i in enumerate(idx, 1):
                    labs[i] = f"{lab}{n}"
        return labs
