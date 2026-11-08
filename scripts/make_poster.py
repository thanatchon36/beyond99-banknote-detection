#!/usr/bin/env python3
"""LinkedIn launch poster -> docs/assets/linkedin_poster.png (1440 x 1800, 4:5).

Every number comes from code/results/ through the same summary as the paper's Table 5 (K = 3 counterfeit
families, 1% genuine-rejection budget, 10 seeds). Fira Sans (SIL OFL) is downloaded into .cache/fonts/ on
first use; the text-collision check of make_figures.py runs before saving.

    python scripts/make_poster.py        # (make poster)
"""
import importlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / ".cache" / "fonts"
WEIGHTS = ["Regular", "Medium", "SemiBold", "Bold", "ExtraBold"]
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/firasans/FiraSans-{}.ttf"
DOI = "10.5281/zenodo.24049790"

FONTS.mkdir(parents=True, exist_ok=True)
for w in WEIGHTS:
    path = FONTS / f"FiraSans-{w}.ttf"
    if not path.exists():
        urllib.request.urlretrieve(FONT_URL.format(w), path)
    font_manager.fontManager.addfont(str(path))

sys.path.insert(0, str(ROOT / "code"))
os.chdir(ROOT / "code")
argv, sys.argv = sys.argv, ["make_figures.py", tempfile.mkdtemp()]
mf = importlib.import_module("make_figures")
sys.argv = argv
import banknote_experiments as be  # noqa: E402
from paper_analysis import families_k  # noqa: E402

BG, PANEL, RULE = "#05070d", "#0b1220", "#1c2740"
INK, MUTED, FAINT = "#f2f4f8", "#9aa3b5", "#6b7489"
AMBER, CREAM, GREEN = "#f5a623", "#ffe8a3", "#7dcfac"
FAMCOL = ["#f5a623", "#56b4e9", "#e07fb8"]

plt.rcParams.update({"font.family": "Fira Sans", "savefig.bbox": None, "axes.titlelocation": "center"})

# ------------------------------------------------------------------ numbers
X, y = be.load_data()
fam = families_k(X, y, 3)
s = mf.lofo_summary(3).set_index("model")
d = pd.read_csv("results/e5x_K3.csv")
d = d[np.isclose(d.alpha, 0.01) & d.setting.str.startswith("family")]
per_family = d.groupby(["model", "setting"]).cdr.mean()  # mean over seeds, each family withheld
q = pd.read_csv("results/quadrant.csv")
xgb = q[(q.model == "XGBoost") & (q.group == "held-out")]
missed = xgb[xgb.sup <= xgb.sup_thr]
outside = (missed.maha / missed.maha_thr > 1).mean()
p = lambda v: f"{v * 100:.2f}%" if 0.9995 <= v < 1 else f"{v * 100:.1f}%"  # Table 5 rounding
xg, ae = s.loc["XGBoost"], s.loc["Autoencoder"]

# ------------------------------------------------------------------ canvas (pixel coordinates)
W, H = 1440, 1800
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)


def T(x, yy, text, size, color=INK, weight="regular", ha="left", va="baseline", **kw):
    return fig.text(x / W, 1 - yy / H, text, fontsize=size * 0.72, color=color, fontweight=weight,
                    ha=ha, va=va, **kw)


def axes(x, yy, w, h, **kw):
    return fig.add_axes([x / W, 1 - (yy + h) / H, w / W, h / H], **kw)


# header
T(94, 92, "BEYOND 99%", 30, AMBER, "bold")
T(368, 92, "open-set evaluation of counterfeit banknote detectors", 25, MUTED)

# hero: the same model, two exams
T(86, 290, f"{xg.random * 100:.1f}%", 168, INK, "heavy")
T(100, 384, "XGBoost on the standard", 34, INK)
T(100, 432, "banknote benchmark.", 34, INK)
T(86, 620, f"{xg.lofo * 100:.0f}%", 168, AMBER, "heavy")
T(100, 714, "The same model on a counterfeit", 34, INK)
T(100, 762, "family it has never seen.", 34, INK)
T(100, 850, "A detector that learns only", 32, GREEN, "semibold")
T(100, 898, f"genuine notes catches {p(ae.lofo)}.", 32, GREEN, "semibold")

# data art: every note in the data, counterfeit families coloured
ax = axes(820, 150, 560, 760, facecolor=BG)
g = y == 0
ax.scatter(X[g, 0], X[g, 1], s=16, color="#c9cfdb", alpha=0.35, lw=0)
for f in range(3):
    m = (y == 1) & (fam == f)
    ax.scatter(X[m, 0], X[m, 1], s=16, color=FAMCOL[f], alpha=0.85, lw=0)
for side in ax.spines.values():
    side.set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(-8.5, 7.5); ax.set_ylim(-17, 14.5)
T(1100, 940, "1,372 banknotes · grey = genuine · colour = counterfeit family", 17, FAINT, ha="center")

# family call-outs: what XGBoost catches when that family is new (leader lines to the cluster centres)
callouts = [(0, 1030, 160, "right"), (1, 905, 525, "right"), (2, 905, 742, "right")]
for f, tx, ty, ha in callouts:
    rate = per_family.loc[("XGBoost", f"family {f}")]
    lab = T(tx, ty, f"Family {f + 1}", 22, FAMCOL[f], "semibold", ha=ha)
    T(tx, ty + 30, f"XGBoost catches {p(rate).replace('.0%', '%')}", 18, INK, ha=ha)
    lab._allow_data = True

# ------------------------------------------------------------------ dumbbell: two exams
T(94, 1040, "Same models, two exams", 30, INK, "semibold")
T(94, 1078, "Counterfeits caught at a 1% genuine-rejection budget:", 19, MUTED)
for x0, label, filled in ((585, "random split", False), (755, "unseen family", True)):
    fig.add_artist(plt.Line2D([x0 / W], [1 - 1071 / H], marker="o", ms=10, mew=2, ls="",
                              mfc=AMBER if filled else BG, mec=AMBER if filled else MUTED))
    T(x0 + 16, 1078, label, 19, MUTED)
rows = [("XGBoost", "XGBoost"), ("Random Forest", "Random forest"), ("LightGBM", "LightGBM"),
        ("MLP (3x64)", "Neural net (MLP)"), ("SVM (RBF)", "SVM"), ("Mahalanobis", "Mahalanobis*"),
        ("Autoencoder", "Autoencoder*")]
bx = axes(330, 1110, 1000, 300, facecolor=BG)
bx.set_xlim(30, 102); bx.set_ylim(-0.6, len(rows) - 0.4)
for i, (m, label) in enumerate(rows):
    r = s.loc[m]
    yy = len(rows) - 1 - i
    col = GREEN if r.kind == "one-class" else AMBER
    bx.plot([r.random * 100, r.lofo * 100], [yy, yy], color=col, lw=3, alpha=0.55, solid_capstyle="round")
    bx.scatter([r.random * 100], [yy], s=110, facecolor=BG, edgecolor=MUTED, lw=2, zorder=3)
    bx.scatter([r.lofo * 100], [yy], s=130, color=col, zorder=4)
    T(310, 1110 + (i + 0.5) * 300 / len(rows) + 8, label, 21, INK if r.kind != "one-class" else GREEN, ha="right")
    vx = bx.transData.transform((r.lofo * 100, yy))[0] * W / fig.bbox.width
    T(vx - 18, 1110 + (i + 0.5) * 300 / len(rows) + 8, p(r.lofo).replace(".0%", "%"), 19, col, "semibold",
      ha="right")
for side in bx.spines.values():
    side.set_visible(False)
bx.set_yticks([]); bx.set_xticks([40, 60, 80, 100], ["40%", "60%", "80%", "100%"])
bx.tick_params(colors=FAINT, labelsize=14, length=0)
bx.grid(axis="x", color=RULE, lw=1)
bx.set_axisbelow(True)
T(1346, 1040, "* one-class: trained on genuine notes only", 16, FAINT, ha="right")

# ------------------------------------------------------------------ stats row
fig.add_artist(plt.Line2D([94 / W, 1346 / W], [1 - 1478 / H] * 2, color=RULE, lw=1.5))
stats = [("18", "detectors:", "supervised, one-class, hybrid"),
         (f"{outside * 100:.1f}%", "of XGBoost's misses lie", "outside the genuine region"),
         ("K = 2–6", "the result holds at every", "family granularity"),
         ("1", "command regenerates", "every number: make all")]
for i, (big, l1, l2) in enumerate(stats):
    x0 = 94 + i * 324
    T(x0, 1552, big, 50, AMBER, "bold")
    T(x0, 1590, l1, 18, MUTED)
    T(x0, 1616, l2, 18, MUTED)

# ------------------------------------------------------------------ links + footer
fig.add_artist(FancyBboxPatch((86 / W, 1 - 1752 / H), 1268 / W, 104 / H, boxstyle="round,pad=0,rounding_size=0.008",
                              transform=fig.transFigure, facecolor=PANEL, edgecolor=RULE, lw=1.5))
T(126, 1688, "Code + paper", 16, MUTED)
T(126, 1724, "github.com/thanatchon36/beyond99-banknote-detection", 22, INK)
T(790, 1688, "Dataset", 16, MUTED)
T(790, 1724, "hf.co/datasets/thanatchon36/beyond99-banknote-detection", 19, INK)
T(94, 1784, f"Thanatchon Chongmankhong  ·  views are my own  ·  DOI {DOI}", 16, MUTED)
T(1346, 1784, "UCI Banknote Authentication (CC BY 4.0) · K = 3 families · 10 seeds", 14, FAINT, ha="right")

issues = mf.check(fig, "poster")
if issues:
    raise SystemExit("poster text collides:\n  " + "\n  ".join(issues))
out = ROOT / "docs" / "assets" / "linkedin_poster.png"
fig.savefig(out, facecolor=BG, dpi=100)
print(f"wrote {out.relative_to(ROOT)}")
