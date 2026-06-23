"""Build Figure 1 (conceptual overview) as Excalidraw JSON.

Visual argument (left -> right):
  Zone A  The finite-T blind spot  : double-well E(Q) harmonic (imaginary at Q=0)
          overlaid with finite-T free energy F(Q;T) that becomes a single well (stable).
  Zone B  The benchmark pipeline   : 20 systems + 5 MLIPs -> harmonic baseline / soft-mode
          screen (hero) / SSCHA cross-check.
  Zone C  Findings                 : H1 / H2 / H3 + the SSCHA trap.

Palette from .claude/skills/excalidraw-diagram-skill/references/color-palette.md.
"""
import json, itertools

_seed = itertools.count(1000)
def s(): return next(_seed)

# ---- palette ----
PRIMARY_F, PRIMARY_S = "#3b82f6", "#1e3a5f"
TERT_F             = "#93c5fd"
ORANGE_F, ORANGE_S = "#fed7aa", "#c2410c"
PURPLE_F, PURPLE_S = "#ddd6fe", "#6d28d9"
DEC_F, DEC_S       = "#fef3c7", "#b45309"
RED_F, RED_S       = "#fee2e2", "#b91c1c"
T_TITLE, T_SUB, T_BODY = "#1e40af", "#3b82f6", "#64748b"
ON_LIGHT, ON_DARK  = "#374151", "#ffffff"
GREEN, RED, BLUE   = "#047857", "#b91c1c", "#1e40af"
SLATE              = "#94a3b8"

els = []

def base(**kw):
    d = dict(angle=0, seed=s(), version=1, versionNonce=s(), isDeleted=False,
             groupIds=[], boundElements=None, link=None, locked=False,
             strokeWidth=2, strokeStyle="solid", roughness=0, opacity=100,
             fillStyle="solid", backgroundColor="transparent")
    d.update(kw); return d

def rect(x, y, w, h, fill, stroke, dashed=False, sw=2):
    els.append(base(type="rectangle", id=f"r{s()}", x=x, y=y, width=w, height=h,
                    strokeColor=stroke, backgroundColor=fill, strokeWidth=sw,
                    strokeStyle="dashed" if dashed else "solid",
                    roundness={"type": 3}))

def diamond(x, y, w, h, fill, stroke):
    els.append(base(type="diamond", id=f"d{s()}", x=x, y=y, width=w, height=h,
                    strokeColor=stroke, backgroundColor=fill))

def text(x, y, txt, color, size=16, align="left", w=None, font=2):
    lines = txt.split("\n")
    nch = max(len(l) for l in lines)
    if w is None:
        w = int(nch * size * 0.62) + 6
    h = int(len(lines) * size * 1.27) + 4
    els.append(base(type="text", id=f"t{s()}", x=x, y=y, width=w, height=h,
                    text=txt, originalText=txt, fontSize=size, fontFamily=font,
                    textAlign=align, verticalAlign="top", strokeColor=color,
                    strokeWidth=1, lineHeight=1.27))
    return w, h

def ctext(cx_box, box_w, y, txt, color, size=16, font=2):
    """Center a (multi-line) text block horizontally within [cx_box, cx_box+box_w]."""
    lines = txt.split("\n")
    nch = max(len(l) for l in lines)
    w = int(nch * size * 0.62) + 6
    x = cx_box + (box_w - w) / 2
    text(x, y, txt, color, size, align="center", w=w, font=font)

def box_label(x, y, w, h, txt, color, size=15, font=2):
    """Vertically+horizontally center a label inside a box at (x,y,w,h)."""
    lines = txt.split("\n")
    th = len(lines) * size * 1.27
    ctext(x, w, y + (h - th) / 2, txt, color, size, font)

def line(x, y, pts, color, width=2, dashed=False):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    els.append(base(type="line", id=f"l{s()}", x=x, y=y,
                    width=max(xs)-min(xs), height=max(ys)-min(ys),
                    strokeColor=color, strokeWidth=width,
                    strokeStyle="dashed" if dashed else "solid", points=pts))

def arrow(x, y, pts, color, width=2, dashed=False, endhead="arrow", starthead=None):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    els.append(base(type="arrow", id=f"a{s()}", x=x, y=y,
                    width=max(xs)-min(xs), height=max(ys)-min(ys),
                    strokeColor=color, strokeWidth=width,
                    strokeStyle="dashed" if dashed else "solid", points=pts,
                    startBinding=None, endBinding=None,
                    startArrowhead=starthead, endArrowhead=endhead))

def dot(cx, cy, color, d=14):
    els.append(base(type="ellipse", id=f"e{s()}", x=cx-d/2, y=cy-d/2, width=d, height=d,
                    strokeColor=color, backgroundColor=color, strokeWidth=1))

# ======================================================================
# HEADER
# ======================================================================
text(60, 28, "Finite-temperature dynamic stability is a blind spot of foundation MLIPs",
     T_TITLE, 26)
text(60, 64, "A finite-T benchmark of five foundation MLIPs, a cheap quantum soft-mode screen, and a cautionary SSCHA cross-check",
     T_BODY, 15)

# zone dividers
line(745, 120, [[0, 0], [0, 660]], SLATE, 1, dashed=True)
line(1420, 120, [[0, 0], [0, 660]], SLATE, 1, dashed=True)

# ======================================================================
# ZONE A  --  the blind spot
# ======================================================================
text(70, 130, "1 · The finite-temperature blind spot", T_TITLE, 20)

# axes
AX, AY = 130, 200          # E-axis top
line(AX, AY, [[0, 0], [0, 250]], ON_LIGHT, 2)              # E axis (vertical)
line(AX, AY+250, [[0, 0], [360, 0]], ON_LIGHT, 2)          # Q axis (horizontal)
text(AX-38, AY-26, "E , F", ON_LIGHT, 14)
text(AX+250, AY+258, "Q  (soft-mode amplitude)", T_BODY, 13)

# harmonic double-well E(Q): minima at +/-Q0, local-max barrier at Q=0 (imaginary mode)
cx0 = AX + 8
WY = AY                      # curve element origin y
wpts = [[0,40],[55,165],[100,225],[150,160],[185,128],[220,160],[270,225],[315,165],[360,40]]
line(cx0, WY, wpts, RED, 3)
# finite-T free energy F(Q;T): single well, minimum at Q=0 (stable)
upts = [[15,70],[95,178],[185,212],[275,178],[355,70]]
line(cx0, WY, upts, GREEN, 2, dashed=True)
# inline curve legends
text(cx0+8, WY+18, "E(Q)  harmonic 0 K", RED, 13)
text(cx0+232, WY+150, "F(Q;T)  finite-T", GREEN, 13)

# Q=0 marker (center of barrier ~ px 185)
zc = cx0 + 185
line(zc, AY, [[0,0],[0,250]], SLATE, 1, dashed=True)
text(zc-44, AY-2, "Q = 0\n(cubic)", T_BODY, 12, align="center", w=88)
# minima dots
dot(cx0+100, WY+225, RED, 12); dot(cx0+270, WY+225, RED, 12)
text(cx0+150, WY+232, "±Q₀  distorted (low-T) phase", T_BODY, 12, align="center", w=240)
# blind-spot pointer at barrier
arrow(cx0+185, WY+90, [[0,0],[60,-42]], RED, 2)
text(cx0+250, WY+40, "harmonic misreads\nthis barrier", RED, 12)

# callouts
text(70, 512, "HARMONIC (0 K):  ∂²E/∂Q² < 0 at Q=0", RED, 14)
text(86, 534, "→ imaginary soft mode → “UNSTABLE”", RED, 13)
text(86, 556, "the regime MLIP phonon benchmarks stop at", T_BODY, 12)
text(70, 588, "FINITE-T  F(Q;T): anharmonic entropy → single well", GREEN, 14)
text(86, 610, "→ cubic phase STABLE above T_c", GREEN, 13)
text(70, 644, "Regime: cubic perovskites · bcc Ti/Zr/Hf · fluorites · α-AgI", T_BODY, 13)
text(70, 666, "Harmonically unstable, yet the equilibrium phase above T_c.", T_BODY, 12)

# ======================================================================
# ZONE B  --  the benchmark pipeline
# ======================================================================
text(770, 130, "2 · The benchmark", T_TITLE, 20)

# inputs
rect(770, 180, 280, 92, ORANGE_F, ORANGE_S)
box_label(770, 180, 280, 92,
          "20 curated reference systems\nliterature ground truth:\nharmonic-unstable → finite-T-stable", ON_LIGHT, 13)
rect(1090, 180, 290, 92, ORANGE_F, ORANGE_S)
box_label(1090, 180, 290, 92,
          "5 foundation MLIPs\nMACE-MP-0 · CHGNet · ORB-v2\nSevenNet-0 · MatterSim", ON_LIGHT, 13)

# inputs -> layers (converge)
arrow(910, 272, [[0,0],[0,26]], ORANGE_S, 2)
arrow(1235, 272, [[0,0],[0,26]], ORANGE_S, 2)

# layer 1 (harmonic baseline)
rect(770, 300, 280, 110, TERT_F, PRIMARY_S)
box_label(770, 300, 280, 110,
          "Layer 1 · Harmonic baseline\nphonopy 2×2×2 + MLIP\n→ min phonon frequency\n(validation anchor)", ON_LIGHT, 13)

# layer 2 (HERO: soft-mode screen)
rect(1090, 300, 300, 150, PRIMARY_F, PRIMARY_S, sw=3)
box_label(1090, 300, 300, 150,
          "Layer 2 · Soft-mode free-energy\nscreen  (this work)\nrelax → softest commensurate mode\n→ static E(Q) double well\n→ single-mode quantum SCHA  F(Q;T)", ON_DARK, 13)

# decision diamond
diamond(1120, 480, 240, 120, DEC_F, DEC_S)
box_label(1120, 480, 240, 120, "cubic stable\n⇔  Q₀ ≈ 0 is the\nglobal min of F", ON_LIGHT, 14)
arrow(1240, 450, [[0,0],[0,28]], PRIMARY_S, 3)

# SSCHA cross-check
rect(770, 470, 280, 110, PURPLE_F, PURPLE_S)
box_label(770, 470, 280, 110,
          "Multi-mode SSCHA\ngold-standard cross-check\npython-sscha + MLIP forces", ON_LIGHT, 13)
# cross-check double arrow between SSCHA and screen/decision
arrow(1050, 525, [[0,0],[70,0]], PURPLE_S, 2, endhead="arrow", starthead="arrow")

# layer1 -> findings (H1) feed, and decision -> findings
arrow(1360, 540, [[0,0],[60,0]], DEC_S, 3)

# ======================================================================
# ZONE C  --  findings
# ======================================================================
text(1445, 130, "3 · What we find", T_TITLE, 20)

dot(1455, 196, RED, 15)
text(1478, 184, "H1 · PES softening → false-stable", RED, 16)
text(1478, 208, "bcc Zr/Hf soft modes pushed toward 0;\nMACE, CHGNet, ORB-v2 carry 15% false-stable", T_BODY, 13)

dot(1455, 290, BLUE, 15)
text(1478, 278, "H2 · Harmonic accuracy ≠ finite-T accuracy", BLUE, 16)
text(1478, 302, "harmonic leaders (MatterSim, SevenNet-0) are\nNOT the finite-T leaders (MACE, CHGNet)", T_BODY, 13)

dot(1455, 384, GREEN, 15)
text(1478, 372, "H3 · Ensemble disagreement flags errors", GREEN, 16)
text(1478, 396, "inter-model vote-split: AUC 0.75\n(5.5× consensus-error enrichment)", T_BODY, 13)

# the SSCHA trap (hero finding)
rect(1445, 470, 470, 150, RED_F, RED_S, sw=3)
box_label(1445, 470, 470, 150,
          "The SSCHA trap\nthe gold standard false-stabilises deep\ndisplacive (ferroelectric) instabilities:\nrecall 0.23  vs  soft-mode screen 0.77\n→ the cheap screen is the more reliable finite-T probe",
          RED, 14)

# ======================================================================
doc = {"type": "excalidraw", "version": 2, "source": "fig1-builder",
       "elements": els, "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None},
       "files": {}}
import sys
out = sys.argv[1] if len(sys.argv) > 1 else "fig1_overview.excalidraw"
with open(out, "w", encoding="utf-8") as f:
    json.dump(doc, f, indent=1)
print(f"wrote {out}  ({len(els)} elements)")
