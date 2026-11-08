#!/usr/bin/env python3
"""README banner -> docs/assets/banner.png (1680 x 624 px), drawn from the UCI data.

Genuine notes and the K = 3 counterfeit families in the variance-skewness plane, beside the headline
numbers from Table 5. The text-collision check of make_figures.py runs before saving.

    python scripts/make_banner.py        # (make banner)
"""
import importlib
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
os.chdir(ROOT / "code")
argv, sys.argv = sys.argv, ["make_figures.py", tempfile.mkdtemp()]
mf = importlib.import_module("make_figures")
sys.argv = argv
import banknote_experiments as be  # noqa: E402
from paper_analysis import families_k  # noqa: E402

BG, INK, MUTED, ACCENT, GOOD = "#0b1020", "#f2f4f8", "#9aa3b5", "#f5a623", "#7dcfac"

X, y = be.load_data()
fam = families_k(X, y, 3)
s = mf.lofo_summary(3).set_index("model")
trees = s.loc[["XGBoost", "Random Forest", "LightGBM"]]
floor = s[s.kind == "supervised"].random.min()

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.bbox": None})
fig = plt.figure(figsize=(16.8, 6.24), dpi=100, facecolor=BG)
ax = fig.add_axes([0.60, 0.10, 0.37, 0.80], facecolor=BG)
g = y == 0
ax.scatter(X[g, 0], X[g, 1], s=9, color="#c9cfdb", alpha=0.45, lw=0, label="genuine notes")
for f in range(3):
    m = (y == 1) & (fam == f)
    ax.scatter(X[m, 0], X[m, 1], s=9, color=mf.CFCOL[f], alpha=0.8, lw=0, label=f"counterfeit family {f + 1}")
for side in ax.spines.values():
    side.set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("variance of the wavelet image", color=MUTED, fontsize=12)
ax.set_ylabel("skewness", color=MUTED, fontsize=12)
leg = ax.legend(loc="lower right", bbox_to_anchor=(1.02, -0.02), frameon=False, fontsize=11, labelcolor=MUTED,
                handletextpad=0.3, markerscale=2.2)

fig.text(0.05, 0.80, "Beyond 99%", color=INK, fontsize=54, fontweight="bold", va="center")
fig.text(0.05, 0.655, "Open-set evaluation of counterfeit banknote detectors", color=MUTED, fontsize=19,
         va="center")
rows = [
    (f"{floor * 100:.1f}%+", INK, "caught by every supervised model on a random split"),
    (f"{trees.lofo.min() * 100:.0f}–{trees.lofo.max() * 100:.0f}%", ACCENT,
     "caught by tree ensembles on unseen counterfeit families"),
    (f"{s.loc['Autoencoder', 'lofo'] * 100:.2f}%", GOOD, "caught by a one-class autoencoder trained on genuine notes"),
]
for i, (num, col, text) in enumerate(rows):
    yy = 0.46 - i * 0.115
    fig.text(0.05, yy, num, color=col, fontsize=25, fontweight="bold", va="center")
    fig.text(0.175, yy, text, color=INK, fontsize=14, va="center")
fig.text(0.05, 0.08, "K = 3 families · 1% genuine-rejection budget · UCI Banknote Authentication", color=MUTED,
         fontsize=11, va="center")

issues = mf.check(fig, "banner")
if issues:
    raise SystemExit("banner text collides:\n  " + "\n  ".join(issues))
out = ROOT / "docs" / "assets" / "banner.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor=BG)
print(f"wrote {out.relative_to(ROOT)}")
