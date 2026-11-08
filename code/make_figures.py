#!/usr/bin/env python3
"""Builds every figure (vector PDF + PNG preview) and LaTeX table used in the paper.

Each figure is checked automatically before saving:
  * text-text collisions: bounding boxes of every drawn label, tick label, title,
    legend entry and annotation are compared pairwise;
  * text-data collisions: annotations placed inside a data panel must not cover
    scatter points or line vertices.
Usage: python make_figures_v2.py [outdir]     (default ../paper)
"""
import os
import sys
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import banknote_experiments as be
from paper_analysis import AE1, families_k

warnings.filterwarnings("ignore")
OUT = sys.argv[1] if len(sys.argv) > 1 else "../paper"
FIGDIR, TABDIR, RES = f"{OUT}/figures", f"{OUT}/tables", "results"
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABDIR, exist_ok=True)

mpl.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5,
    "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
    "grid.color": "#ebebeb", "grid.linewidth": 0.5, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "legend.frameon": False,
    "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlepad": 4,
})
FAMCOL = {"tree": "#D55E00", "deep": "#0072B2", "classical": "#009E73", "hybrid": "#7B3294"}
CFCOL = ["#E69F00", "#56B4E9", "#CC79A7"]
GENCOL = "#6e6e6e"
GREY = "#8c8c8c"
DISPLAY = {"MLP (3x64)": "MLP (3×64)", "Logistic Regression": "Logistic regression",
           "Random Forest": "Random forest", "Isolation Forest": "Isolation forest",
           "Mahalanobis": "Mahalanobis (Gaussian)", "Decision Tree": "Decision tree",
           "Extra Trees": "Extra trees", "kNN (k=5)": "k-NN (k = 5)",
           "XGBoost + Isolation Forest": "XGBoost + Isolation forest",
           "MLP + Isolation Forest": "MLP + Isolation forest"}
LINESTYLE = {"XGBoost": "-", "Random Forest": "--", "LightGBM": ":", "MLP (3x64)": "-",
             "SVM (RBF)": "-", "Logistic Regression": "--"}
MARKER = {"XGBoost": "o", "Random Forest": "s", "LightGBM": "^", "MLP (3x64)": "o",
          "SVM (RBF)": "D", "Logistic Regression": "v", "Autoencoder": "o",
          "Mahalanobis": "s", "Isolation Forest": "D"}
REPORT = []


def disp(m):
    return DISPLAY.get(m, m)


def color(kind, family):
    return FAMCOL["hybrid"] if kind == "hybrid" else FAMCOL[family]


def boot_ci(v, B=10000, seed=0):
    v = np.asarray(v, float)
    m = np.random.default_rng(seed).choice(v, (B, len(v)), replace=True).mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def minus(s):
    return s.replace("-", "\u2212")


def sectioned_handles(sections):
    """sections: [(header, [(handle, label), ...]), ...] -> handles, labels, header indices"""
    hs, ls, heads = [], [], []
    for head, items in sections:
        heads.append(len(hs))
        hs.append(Line2D([], [], ls="", marker=""))
        ls.append(head)
        for h_, l_ in items:
            hs.append(h_); ls.append(l_)
    return hs, ls, heads


def bold_headers(leg, heads):
    for i, t in enumerate(leg.get_texts()):
        if i in heads:
            t.set_fontweight("bold")
            t.set_x(-18)


def panel_label(ax, letter, dx=0.0, dy=0.0):
    """Bold (a)/(b) label just above the top-left corner of the axes."""
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(-14 + dx, 6 + dy),
                textcoords="offset points", fontsize=9, fontweight="bold", ha="left", va="bottom")


# ----------------------------------------------------------------- quality checks
def drawn_ticklabels(ax):
    out = []
    for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
        lo, hi = sorted(lim)
        tol = 1e-9 * (hi - lo)
        for loc, lab in zip(axis.get_majorticklocs(), axis.get_majorticklabels()):
            if lo - tol <= loc <= hi + tol:
                out.append(lab)
    return out


def _texts(fig):
    out = []
    for ax in fig.axes:
        cand = [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label]
        cand += drawn_ticklabels(ax)
        cand += list(ax.texts)
        if ax.get_legend() is not None:
            cand += list(ax.get_legend().get_texts()) + [ax.get_legend().get_title()]
        out += [(t, ax) for t in cand]
    out += [(t, None) for t in fig.texts]
    for lg in fig.legends:
        out += [(t, None) for t in list(lg.get_texts()) + [lg.get_title()]]
    return out


def check(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for t, ax in _texts(fig):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(renderer=r)
        if bb.width > 0.5 and bb.height > 0.5:
            boxes.append((t, ax, bb))
    issues = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, _, A = boxes[i]
            b, _, B = boxes[j]
            if A.padded(0.6).overlaps(B.padded(0.6)):
                issues.append(f"text/text: {a.get_text()!r} x {b.get_text()!r}")
    # annotations inside data panels must not sit on data points
    for t, ax, bb in boxes:
        if ax is None or t not in ax.texts or any(isinstance(c, AxesImage) for c in ax.get_children()):
            continue
        if getattr(t, "_allow_data", False):
            continue
        pts = []
        for c in ax.collections:
            if isinstance(c, PathCollection) and len(c.get_offsets()):
                pts.append(c.get_offset_transform().transform(c.get_offsets()))
        for ln in ax.lines:
            if ln.get_visible() and len(ln.get_xydata()) and ln.get_transform() == ax.transData:
                pts.append(ax.transData.transform(ln.get_xydata()))
        if pts:
            P = np.vstack(pts)
            inside = ((P[:, 0] >= bb.x0 - 1) & (P[:, 0] <= bb.x1 + 1) &
                      (P[:, 1] >= bb.y0 - 1) & (P[:, 1] <= bb.y1 + 1)).sum()
            if inside:
                issues.append(f"text/data: {t.get_text()!r} covers {inside} data points")
    REPORT.append((name, len(boxes), issues))
    return issues


def save(fig, name):
    check(fig, name)
    fig.savefig(f"{FIGDIR}/{name}.pdf")
    fig.savefig(f"{FIGDIR}/{name}.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------- summaries
def e5x(K):
    return pd.read_csv(f"{RES}/e5x_K{K}.csv")


def lofo_summary(K=3, alpha=0.01):
    d = e5x(K)
    d = d[np.isclose(d.alpha, alpha)]
    fam = d[d.setting.str.startswith("family")]
    per_seed = fam.groupby(["model", "seed"]).cdr.agg(["mean", "min"]).reset_index()
    rnd = d[d.setting == "random split"].groupby("model").cdr.mean()
    meta = d.groupby("model")[["kind", "family"]].first()
    out = []
    for m, g in per_seed.groupby("model"):
        lo, hi = boot_ci(g["mean"])
        fl = meta.loc[m, "family"]
        out.append(dict(model=m, kind=meta.loc[m, "kind"], family=fl if "+" not in fl else "hybrid",
                        lofo=g["mean"].mean(), lo=lo, hi=hi, worst=g["min"].mean(), random=rnd[m],
                        grr=fam[fam.model == m].grr.mean(), auc=fam[fam.model == m].auc.mean()))
    return pd.DataFrame(out)


KINDS = [("supervised", "Supervised"), ("one-class", "One-class"), ("hybrid", "Hybrid")]


def ordered(s):
    return pd.concat([s[s.kind == k].sort_values("lofo", ascending=False) for k, _ in KINDS])


# ----------------------------------------------------------------------- figure 1
def fig1(s):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    fig.subplots_adjust(left=0.30, right=0.83, top=0.85, bottom=0.12)
    y, ticks, labels, tcols, groups = 0.0, [], [], [], []
    for kind, gname in KINDS:
        y0 = y
        for _, r in s[s.kind == kind].sort_values("lofo", ascending=False).iterrows():
            c = color(r.kind, r.family)
            ax.plot([r.lofo * 100, r.random * 100], [y, y], color="#d6d6d6", lw=1.3, zorder=1)
            ax.plot([r.lo * 100, r.hi * 100], [y, y], color=c, lw=3.4, alpha=0.30,
                    solid_capstyle="round", zorder=2)
            ax.scatter(r.random * 100, y, s=27, facecolor="white", edgecolor=GREY, lw=0.9, zorder=3)
            ax.scatter(r.worst * 100, y, marker="D", s=15, facecolor="white", edgecolor=c, lw=0.9, zorder=3)
            ax.scatter(r.lofo * 100, y, s=36, color=c, edgecolor="white", lw=0.5, zorder=4)
            val, gap = r.lofo * 100, (r.lofo - r.random) * 100
            ax.text(104, y, f"{val:.2f}" if 99.95 <= val < 100 else f"{val:.1f}", va="center", ha="left",
                    fontsize=7.3, color=c, fontweight="bold", clip_on=False)
            ax.text(119, y, "0.0" if abs(gap) < 0.05 else minus(f"{gap:+.1f}"), va="center", ha="left",
                    fontsize=7.3, color="#555555", clip_on=False)
            ticks.append(y); labels.append(disp(r.model)); tcols.append(c)
            y += 1
        groups.append((gname, y0, y - 1))
        y += 0.7
    ax.text(104, -1.05, "Unseen\nfamily (%)", fontsize=6.8, color="#444", ha="left", va="bottom", clip_on=False)
    ax.text(119, -1.05, "Δ vs.\nrandom (pp)", fontsize=6.8, color="#444", ha="left", va="bottom", clip_on=False)
    ax.set_yticks(ticks, labels)
    for t, c in zip(ax.get_yticklabels(), tcols):
        t.set_color(c)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(y - 0.2, -2.0)
    ax.set_xlim(0, 101)
    ax.set_xticks(range(0, 101, 20))
    ax.grid(axis="x")
    ax.set_xlabel("Counterfeits detected at a 1% genuine-rejection budget (%)")
    # group brackets placed left of the longest tick label, measured after drawing
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    xmin = min(t.get_window_extent(r).x0 for t in ax.get_yticklabels())
    xb = ax.transAxes.inverted().transform((xmin, 0))[0] - 0.018
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for gname, ya, yb in groups:
        ax.plot([xb, xb], [ya - 0.38, yb + 0.38], transform=trans, color="#a0a0a0", lw=0.9,
                clip_on=False, solid_capstyle="butt")
        for yy in (ya - 0.38, yb + 0.38):
            ax.plot([xb, xb + 0.008], [yy, yy], transform=trans, color="#a0a0a0", lw=0.9, clip_on=False)
        ax.text(xb - 0.012, (ya + yb) / 2, gname, transform=trans, rotation=90, ha="right", va="center",
                fontsize=7.6, fontweight="bold", color="#444444")
    h = [Line2D([], [], marker="o", ls="", mfc="white", mec=GREY, ms=5.5, label="Random split (standard protocol)"),
         Line2D([], [], marker="o", ls="", color="#444", ms=6, label="Unseen family: mean"),
         Line2D([], [], lw=4.5, color="#444", alpha=0.3, label="95% bootstrap CI"),
         Line2D([], [], marker="D", ls="", mfc="white", mec="#444", ms=4.5, label="Unseen family: worst")]
    h2 = [Patch(color=FAMCOL[k], label=lab) for k, lab in
          [("tree", "Tree-based"), ("deep", "Neural (deep)"), ("classical", "Classical"), ("hybrid", "Hybrid")]]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.55, 0.925), handletextpad=0.7,
               columnspacing=2.0, handlelength=1.8)
    fig.legend(handles=h2, loc="lower center", ncol=4, bbox_to_anchor=(0.55, 0.875), handletextpad=0.5,
               columnspacing=2.4, handlelength=1.2)
    save(fig, "fig1_teaser")


# ----------------------------------------------------------------------- figure 2
def fig2(X, y):
    fam = families_k(X, y, 3)
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.35), gridspec_kw=dict(width_ratios=[1.12, 1.18, 0.78]))
    fig.subplots_adjust(wspace=0.42, top=0.80, bottom=0.17)
    ax = axs[0]
    g = X[y == 0][:, :2]
    kde = gaussian_kde(g.T)
    xx, yy = np.meshgrid(np.linspace(-8, 8, 220), np.linspace(-15, 14, 220))
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    lv = np.quantile(kde(g.T), [0.01, 0.10, 0.50])
    ax.contourf(xx, yy, zz, levels=[lv[0], lv[1], lv[2], zz.max()], colors=["#f2f2f2", "#e0e0e0", "#c9c9c9"], zorder=0)
    ax.contour(xx, yy, zz, levels=lv, colors="#9a9a9a", linewidths=0.5, zorder=1)
    ax.scatter(g[:, 0], g[:, 1], s=2, color=GENCOL, alpha=0.45, lw=0, zorder=2, rasterized=True)
    mk = ["^", "s", "o"]
    for f in range(3):
        p = X[fam == f][:, :2]
        ax.scatter(p[:, 0], p[:, 1], s=6, marker=mk[f], color=CFCOL[f], alpha=0.85, lw=0, zorder=3, rasterized=True)
    ax.set_xlim(-8, 8); ax.set_ylim(-15, 14)
    ax.set_xlabel("Wavelet variance"); ax.set_ylabel("Wavelet skewness")
    panel_label(ax, "a")
    ax = axs[1]
    mu, sd = X[y == 0].mean(0), X[y == 0].std(0)
    Z = (X - mu) / sd
    groups = [("g", y == 0, GENCOL)] + [(f, fam == f, CFCOL[f]) for f in range(3)]
    xs = np.arange(4)
    for key, msk, c in groups:
        q = np.percentile(Z[msk], [25, 50, 75], axis=0)
        ax.fill_between(xs, q[0], q[2], color=c, alpha=0.18, lw=0)
        ax.plot(xs, q[1], color=c, lw=1.4, marker="o" if key == "g" else mk[key], ms=3.2)
    ax.axhline(0, color="#bdbdbd", lw=0.5, zorder=0)
    ax.set_xticks(xs, ["Variance", "Skewness", "Kurtosis", "Entropy"])
    ax.set_xlim(-0.25, 3.25)
    ax.set_ylabel("z-score vs. genuine (median, IQR)")
    ax.grid(axis="y")
    panel_label(ax, "b")
    ax = axs[2]
    Xc = StandardScaler().fit_transform(X[y == 1])
    ks = list(range(2, 7))
    sil = [silhouette_score(Xc, KMeans(k, n_init=20, random_state=0).fit_predict(Xc)) for k in ks]
    best = ks[int(np.argmax(sil))]
    ax.barh(ks, np.array(sil) - 0.35, left=0.35, color=["#0072B2" if k == best else "#cdcdcd" for k in ks],
            height=0.62)
    for k, v in zip(ks, sil):
        ax.text(v + 0.002, k, f"{v:.3f}", ha="left", va="center", fontsize=6.8,
                fontweight="bold" if k == best else "normal")
    ax.set_xlim(0.35, 0.462); ax.set_xticks([0.36, 0.40, 0.44], ["0.36", "0.40", "0.44"])
    ax.set_yticks(ks); ax.set_ylim(6.6, 1.4)
    ax.set_ylabel("Number of families K"); ax.set_xlabel("Silhouette (counterfeits)")
    panel_label(ax, "c")
    sizes = np.bincount(fam[fam >= 0])
    h = [Line2D([], [], color=GENCOL, marker="o", ms=3.5, lw=1.2, label=f"Genuine (n = {(y == 0).sum()})")] + \
        [Line2D([], [], color=CFCOL[f], marker=mk[f], ms=3.5, lw=1.2, label=f"Counterfeit family {f} (n = {sizes[f]})")
         for f in range(3)]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.905), columnspacing=1.6,
               handlelength=1.8)
    save(fig, "fig2_data")
    return sil


# ----------------------------------------------------------------------- figure 3
def ci_frame(df, by, val):
    out = df.groupby(by)[val].agg(["mean", "std", "count"]).reset_index()
    out["half"] = 1.96 * out["std"] / np.sqrt(out["count"])
    return out


def fig3():
    models = ["MLP (3x64)", "SVM (RBF)", "Logistic Regression", "XGBoost", "Random Forest", "LightGBM"]
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.55))
    fig.subplots_adjust(wspace=0.26, bottom=0.285, top=0.90)
    e2 = pd.read_csv(f"{RES}/e2_learning_curve.csv")
    e2["acc"] = e2.accuracy * 100
    c2 = ci_frame(e2, ["model", "n_train"], "acc")
    ax = axs[0]
    ax.axvline(80, color="#9a9a9a", lw=0.7, ls=(0, (3, 2)), zorder=0)
    for m in models:
        d = c2[c2.model == m]
        c = FAMCOL[be.FAMILY[m]]
        ax.fill_between(d.n_train, d["mean"] - d.half, d["mean"] + d.half, color=c, alpha=0.12, lw=0)
        ax.plot(d.n_train, d["mean"], color=c, ls=LINESTYLE[m], lw=1.3, marker=MARKER[m], ms=2.6, label=disp(m))
    ax.set_xscale("log", base=2)
    ax.set_xticks([10, 20, 40, 80, 160, 320, 640], ["10", "20", "40", "80", "160", "320", "640"])
    ax.minorticks_off()
    ax.set_xlim(8.5, 760); ax.set_ylim(58, 101)
    ax.grid(True)
    ax.set_xlabel("Training notes (log scale)"); ax.set_ylabel("Test accuracy (%)")
    ax.text(90, 69.5, "at 80 training notes:\nMLP 98.4%, SVM 98.5%\ntree ensembles 92.7–93.6%", fontsize=6.8,
            color="#333", ha="left", va="center", linespacing=1.2)
    panel_label(ax, "a")
    e4 = pd.read_csv(f"{RES}/e4_noise.csv")
    e4 = e4.groupby(["model", "alpha", "rep"]).accuracy.mean().reset_index()
    e4["acc"] = e4.accuracy * 100
    c4 = ci_frame(e4, ["model", "alpha"], "acc")
    ax = axs[1]
    for m in models:
        d = c4[c4.model == m]
        c = FAMCOL[be.FAMILY[m]]
        ax.fill_between(d.alpha, d["mean"] - d.half, d["mean"] + d.half, color=c, alpha=0.12, lw=0)
        ax.plot(d.alpha, d["mean"], color=c, ls=LINESTYLE[m], lw=1.3, marker=MARKER[m], ms=2.6)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(68, 101)
    ax.grid(True)
    ax.set_xlabel("Test-time noise σ (multiples of the feature s.d.)"); ax.set_ylabel("Test accuracy (%)")
    panel_label(ax, "b")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0),
               handlelength=2.3, columnspacing=1.3, handletextpad=0.5)
    save(fig, "fig3_regimes")


# ----------------------------------------------------------------------- figure 4
def fig4(s):
    d = e5x(3)
    d1 = d[np.isclose(d.alpha, 0.01) & d.setting.str.startswith("family")]
    o = ordered(s)
    cols = sorted(d1.setting.unique())
    M = d1.pivot_table(index="model", columns="setting", values="cdr", aggfunc="mean").loc[o.model, cols] * 100
    M["Mean"] = o.set_index("model").lofo * 100
    M["Worst"] = o.set_index("model").worst * 100
    fam = families_k(*be.load_data(), 3)
    sizes = np.bincount(fam[fam >= 0])
    fig = plt.figure(figsize=(6.6, 3.55))
    gsl = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.07, left=0.255, right=0.585,
                           top=0.87, bottom=0.03)
    ax = fig.add_subplot(gsl[0])
    cax = fig.add_subplot(gsl[1])
    im = ax.imshow(M.values, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.6,
                    color="white" if v < 62 else "black", fontweight="bold" if j >= 3 else "normal")
    ax.set_xticks(range(M.shape[1]), [f"F{c.split()[-1]}\nn={sizes[int(c.split()[-1])]}" for c in cols] + ["Mean", "Worst"])
    ax.set_yticks(range(M.shape[0]), [disp(m) for m in o.model])
    for t, (_, r) in zip(ax.get_yticklabels(), o.iterrows()):
        t.set_color(color(r.kind, r.family))
    ax.tick_params(length=0, pad=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    nsup, noc = (o.kind == "supervised").sum(), (o.kind == "one-class").sum()
    for b in (nsup, nsup + noc):
        ax.axhline(b - 0.5, color="white", lw=2.4)
    ax.axvline(len(cols) - 0.5, color="white", lw=2.4)
    ax.xaxis.tick_top()
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6.6, length=2)
    cb.outline.set_linewidth(0.4)
    cax.set_title("%", fontsize=7, fontweight="normal", loc="center", pad=3)
    ax.annotate("(a)", xy=(0, 1), xycoords="axes fraction", xytext=(-122, 26), textcoords="offset points",
                fontsize=9, fontweight="bold", ha="left", va="bottom")
    ax2 = fig.add_subplot(fig.add_gridspec(1, 1, left=0.695, right=0.99, top=0.87, bottom=0.375)[0])
    da = d[d.setting.str.startswith("family")]
    curves = da.groupby(["model", "alpha"])[["grr", "cdr"]].mean().reset_index()
    handles = {}
    for m in ["Autoencoder", "MLP (3x64)", "Mahalanobis", "SVM (RBF)", "Isolation Forest", "XGBoost"]:
        c_ = curves[curves.model == m].sort_values("alpha")
        oc = m in be.ONE_CLASS
        fam_ = {"Isolation Forest": "tree", "Autoencoder": "deep", "Mahalanobis": "classical"}.get(m, be.FAMILY.get(m))
        c = FAMCOL[fam_]
        handles[m], = ax2.plot(c_.grr * 100, c_.cdr * 100, color=c, lw=1.4, ls="--" if oc else "-",
                               marker=MARKER[m], ms=2.6)
        p = c_[np.isclose(c_.alpha, 0.01)]
        ax2.scatter(p.grr * 100, p.cdr * 100, s=30, facecolor="white", edgecolor=c, lw=1.1, zorder=4)
    ax2.set_xscale("log")
    ax2.set_xticks([0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
    ax2.minorticks_off()
    ax2.set_xlim(0.45, 30); ax2.set_ylim(28, 102)
    ax2.set_xlabel("Genuine notes rejected (%, log scale)")
    ax2.set_ylabel("Unseen-family counterfeits detected (%)")
    ax2.grid(True)
    ax2.annotate("(b)", xy=(0, 1), xycoords="axes fraction", xytext=(-30, 26), textcoords="offset points",
                 fontsize=9, fontweight="bold", ha="left", va="bottom")
    hs, ls, heads = sectioned_handles([
        ("Supervised", [(handles[m], disp(m)) for m in ["SVM (RBF)", "MLP (3x64)", "XGBoost"]]),
        ("One-class", [(handles[m], disp(m)) for m in ["Autoencoder", "Mahalanobis", "Isolation Forest"]])])
    pos = ax2.get_position()
    leg = fig.legend(hs, ls, loc="upper center", ncol=2, bbox_to_anchor=(0.5 * (pos.x0 + pos.x1) - 0.03, 0.235),
                     handlelength=2.2, columnspacing=1.2, fontsize=6.8)
    bold_headers(leg, heads)
    save(fig, "fig4_lofo")


# ----------------------------------------------------------------------- figure 5
def toy_data(rng):
    cov = np.array([[1.0, 0.78], [0.78, 1.0]]) * 0.55
    gen = rng.multivariate_normal([0, 0], cov, 700)
    A = rng.normal([2.6, -0.4], 0.38, (160, 2))
    B = rng.normal([1.0, -2.4], 0.38, (160, 2))
    C1 = rng.normal([-1.9, 2.0], 0.33, (120, 2))
    C2 = rng.normal([-2.9, -2.9], 0.33, (120, 2))
    return gen, A, B, C1, C2


def fig5():
    rng = np.random.RandomState(3)
    gen, A, B, C1, C2 = toy_data(rng)
    g_tr, g_val = gen[:450], gen[450:]
    Xs = np.r_[g_tr, A, B]
    ys = np.r_[np.zeros(len(g_tr)), np.ones(len(A) + len(B))].astype(int)
    sup = be.supervised_models(0, names=["XGBoost", "MLP (3x64)"])
    dets = [("XGBoost", "supervised, tree-based", sup["XGBoost"].fit(Xs, ys), True),
            ("MLP (3×64)", "supervised, neural", sup["MLP (3x64)"].fit(Xs, ys), True),
            ("Isolation forest", "one-class, tree-based", be.IForestScorer(0).fit(g_tr), False),
            ("Autoencoder", "one-class, neural", AE1(0).fit(g_tr), False)]
    xx, yy = np.meshgrid(np.linspace(-4.6, 4.2, 320), np.linspace(-4.6, 4.2, 320))
    grid = np.c_[xx.ravel(), yy.ravel()]
    fig, axs = plt.subplots(1, 4, figsize=(6.6, 2.45))
    fig.subplots_adjust(wspace=0.10, bottom=0.25, top=0.80, left=0.01, right=0.99)
    for k, (ax, (name, sub, m, is_sup)) in enumerate(zip(axs, dets)):
        sc = (lambda Z: be.sup_score(m, Z)) if is_sup else (lambda Z: m.score(Z))
        thr = np.quantile(sc(g_val), 0.99)
        flag = (sc(grid) > thr).reshape(xx.shape)
        ax.contourf(xx, yy, flag, levels=[0.5, 1.5], colors=["#f6d5c3"], zorder=0)
        ax.contour(xx, yy, flag.astype(float), levels=[0.5], colors="#b5562a", linewidths=0.7, zorder=1)
        ax.scatter(*g_tr.T, s=1.6, color=GENCOL, alpha=0.5, lw=0, rasterized=True, zorder=2)
        if is_sup:
            ax.scatter(*np.r_[A, B].T, s=3, marker="x", color="#3b3b3b", lw=0.4, alpha=0.6, rasterized=True, zorder=2)
        ax.scatter(*C1.T, s=4.5, marker="^", color=CFCOL[2], lw=0, alpha=0.9, rasterized=True, zorder=3)
        ax.scatter(*C2.T, s=4.5, marker="s", color=CFCOL[0], lw=0, alpha=0.9, rasterized=True, zorder=3)
        r1, r2 = (sc(C1) > thr).mean() * 100, (sc(C2) > thr).mean() * 100
        ax.text(0.0, 1.13, f"({'abcd'[k]}) {name}", transform=ax.transAxes, fontsize=8, fontweight="bold",
                ha="left", va="bottom")
        ax.text(0.0, 1.03, sub, transform=ax.transAxes, fontsize=6.9, color="#555", ha="left", va="bottom")
        ax.text(0.5, -0.045, f"new family 1 flagged: {r1:.0f}%\nnew family 2 flagged: {r2:.0f}%",
                transform=ax.transAxes, fontsize=6.8, ha="center", va="top", color="#222", linespacing=1.25)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_xlim(-4.6, 4.2); ax.set_ylim(-4.6, 4.2)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color("#bbbbbb")
    h = [Patch(facecolor="#f6d5c3", edgecolor="#b5562a", lw=0.6, label="flagged (1% genuine-rejection budget)"),
         Line2D([], [], marker="o", ls="", color=GENCOL, ms=3, label="genuine"),
         Line2D([], [], marker="x", ls="", color="#3b3b3b", ms=3.5, label="known families (supervised only)"),
         Line2D([], [], marker="^", ls="", color=CFCOL[2], ms=4, label="new family 1"),
         Line2D([], [], marker="s", ls="", color=CFCOL[0], ms=3.6, label="new family 2")]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0), handletextpad=0.4,
               columnspacing=1.4)
    save(fig, "fig5_toy")


# ----------------------------------------------------------------------- figure 6
def fig6():
    q = pd.read_csv(f"{RES}/quadrant.csv")
    q["ym"] = q.sup - q.sup_thr
    q["xm"] = q.maha / q.maha_thr
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.05))
    fig.subplots_adjust(wspace=0.22, bottom=0.27, top=0.90)
    stats = {}
    for k, (ax, m) in enumerate(zip(axs, ["XGBoost", "MLP (3x64)"])):
        d = q[q.model == m]
        g, h = d[d.group == "genuine"], d[d.group == "held-out"]
        lo, hi = d.ym.min(), d.ym.max()
        span = hi - lo
        ylo, yhi = lo - 0.30 * span, hi + 0.13 * span
        ax.axvspan(1, 25, ymin=0, ymax=(0 - ylo) / (yhi - ylo), color="#fbe3d6", zorder=0, lw=0)
        ax.scatter(g.xm, g.ym, s=1.3, color="#7a7a7a", alpha=0.28, lw=0, rasterized=True, zorder=1)
        for f in range(3):
            hf = h[h.family == f]
            ax.scatter(hf.xm, hf.ym, s=1.7, color=CFCOL[f], alpha=0.40, lw=0, rasterized=True, zorder=2)
        ax.axvline(1, color="#555", lw=0.7, ls="--", zorder=3)
        ax.axhline(0, color="#555", lw=0.7, ls="--", zorder=3)
        ax.set_xscale("log"); ax.set_xlim(0.01, 25); ax.set_ylim(ylo, yhi)
        ax.set_xticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"]); ax.minorticks_off()
        share = ((h.ym <= 0) & (h.xm > 1)).mean()
        miss = h[h.ym <= 0]
        stats[m] = (len(miss), len(h), (miss.xm > 1).mean(), share)
        yb = lo - 0.15 * span            # centre of the empty band below the data
        t1 = ax.text(1.18, yb + 0.055 * span, f"{share * 100:.1f}%", fontsize=9, fontweight="bold",
                     color="#a8410f", ha="left", va="center")
        t2 = ax.text(1.18, yb - 0.075 * span, "accepted, non-genuine", fontsize=6.9, color="#a8410f",
                     ha="left", va="center")
        t3 = ax.text(0.8, yb, "accepted, genuine-like", fontsize=6.9, color="#444", ha="right", va="center")
        t4 = ax.text(1.18, hi + 0.065 * span, "flagged, non-genuine", fontsize=6.9, color="#444", ha="left",
                     va="center")
        ax.set_title(f"({'ab'[k]}) {disp(m)}, supervised", fontsize=8)
        ax.grid(True)
        ax.set_ylabel("Margin over threshold (logit units)" if k == 0 else "")
    fig.text(0.5, 0.135, "Squared Mahalanobis distance from the genuine class ÷ its 99% threshold (log scale)",
             ha="center", va="center", fontsize=8)
    h = [Line2D([], [], marker="o", ls="", color="#7a7a7a", ms=3.2, label="genuine (test)")] + \
        [Line2D([], [], marker="o", ls="", color=CFCOL[f], ms=3.2, label=f"unseen family {f}") for f in range(3)] + \
        [Patch(color="#fbe3d6", label="accepted although outside the genuine region")]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0), handletextpad=0.35,
               columnspacing=1.1, handlelength=1.2)
    save(fig, "fig6_mechanism")
    return stats


# ----------------------------------------------------------------------- figure 7
def fig7():
    s = pd.concat([lofo_summary(K).assign(K=K) for K in range(2, 7)])
    models = ["Autoencoder", "Mahalanobis", "SVM (RBF)", "MLP (3x64)", "XGBoost", "Random Forest",
              "LightGBM", "Isolation Forest"]
    handles = {}
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.45), sharey=True)
    fig.subplots_adjust(wspace=0.10, right=0.76, top=0.88, bottom=0.17)
    for k, (ax, col, lab) in enumerate(zip(axs, ["lofo", "worst"], ["Mean over held-out families",
                                                                     "Worst held-out family"])):
        for m in models:
            d = s[s.model == m].sort_values("K")
            r0 = d.iloc[0]
            c = color(r0.kind, r0.family)
            oc = r0.kind == "one-class"
            hd, = ax.plot(d.K, d[col] * 100, color=c, ls="--" if oc else "-", marker=MARKER[m], ms=3.3, lw=1.3)
            handles[m] = hd
            if col == "lofo":
                ax.fill_between(d.K, d.lo * 100, d.hi * 100, color=c, alpha=0.10, lw=0)
        ax.set_xticks(range(2, 7)); ax.set_xlabel("Number of counterfeit families K")
        ax.set_title(f"({'ab'[k]}) {lab}", fontsize=8)
        ax.grid(True); ax.set_ylim(-2, 102); ax.set_xlim(1.8, 6.2)
    axs[0].set_ylabel("Counterfeits detected (%)")
    hs, ls, heads = sectioned_handles([
        ("One-class", [(handles[m], disp(m)) for m in ["Autoencoder", "Mahalanobis", "Isolation Forest"]]),
        ("Supervised", [(handles[m], disp(m)) for m in ["SVM (RBF)", "MLP (3x64)", "XGBoost", "Random Forest", "LightGBM"]])])
    leg = axs[1].legend(hs, ls, loc="center left", bbox_to_anchor=(1.03, 0.5), handlelength=2.2)
    bold_headers(leg, heads)
    save(fig, "fig7_k")
    return s


# ----------------------------------------------------------------------- figure 8
def fig8():
    N, GDP = 6e9, 18e12          # illustrative Thailand-scale inputs
    ppm = np.logspace(-1, 3, 300)
    dv = np.logspace(np.log10(20), 3, 300)
    P, D = np.meshgrid(ppm, dv)
    V = P * N / 1e6 * D / GDP * 100
    fig, ax = plt.subplots(figsize=(5.3, 3.15))
    im = ax.pcolormesh(P, D, np.log10(V), cmap="magma_r", shading="auto", vmin=-8, vmax=-1.5, rasterized=True)
    cs = ax.contour(P, D, np.log10(V), levels=[-6, -5, -4, -3, -2], colors="white", linewidths=0.6, linestyles="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(20, 1000); ax.set_xlim(0.1, 1000)
    refs = [(1, "Thailand\n(order of magnitude)", "left"), (15, "euro area,\nrecent years", "left"),
            (470, "Canada,\n2004 peak", "right")]
    for x, lab, side in refs:
        ax.axvline(x, color="#1b1b1b", lw=0.8, ls=(0, (3, 2)))
        ax.text(x * (1.14 if side == "left" else 1 / 1.14), 900, lab, fontsize=6.8, va="top",
                ha=side, color="#1b1b1b", linespacing=1.15,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2, boxstyle="round,pad=0.25"))
    ax.scatter([1], [500], marker="*", s=80, color="white", edgecolor="black", lw=0.6, zorder=5)
    ax.text(1.32, 420, "assumed\n~1 ppm, THB 500", fontsize=6.6, color="#1b1b1b", va="top", ha="left",
            linespacing=1.1, bbox=dict(fc="white", ec="none", alpha=0.85, boxstyle="round,pad=0.25"))
    v_peak = 470 * N / 1e6 * 500 / GDP * 100
    ax.annotate(f"at Canada's 2004 peak:\n≈{v_peak:.3f}% of GDP", xy=(470, 500), xytext=(25, 40),
                fontsize=6.8, color="#1b1b1b", ha="left", va="center", linespacing=1.15,
                arrowprops=dict(arrowstyle="->", lw=0.6, color="#1b1b1b", shrinkB=3),
                bbox=dict(fc="white", ec="#999999", lw=0.4, boxstyle="round,pad=0.3"))
    ax.set_xlabel("Counterfeit incidence (ppm of notes in circulation, log scale)")
    ax.set_ylabel("Mean face value of counterfeits (THB, log scale)")
    ax.set_yticks([50, 100, 200, 500, 1000], ["50", "100", "200", "500", "1,000"])
    ax.set_xticks([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1,000"])
    ax.minorticks_off()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)
    cb = fig.colorbar(im, ax=ax, pad=0.025, fraction=0.05)
    cb.set_ticks([-8, -6, -4, -2]); cb.set_ticklabels([f"$10^{{{v}}}$" for v in (-8, -6, -4, -2)])
    cb.add_lines(cs)
    cb.set_label("Detected counterfeits (% of GDP per year)", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.8)
    cb.outline.set_linewidth(0.4)
    save(fig, "fig8_econ")
    return v_peak


# ----------------------------------------------------------------------- figure 9
def fig9():
    tasks = [("Governance: data-sharing agreement, ethics, legal review", 0, 3, "#8a8a8a"),
             ("Economics: Thai counterfeit statistics, merchant survey", 1, 8, "#009E73"),
             ("Data: multispectral Thai banknote corpus, expert family labels", 3, 12, "#E69F00"),
             ("Modelling: LOFO, cross-device and adversarial evaluation", 9, 18, "#0072B2"),
             ("Policy: cost–benefit and redesign-trigger model", 15, 22, "#CC79A7"),
             ("Dissemination: papers, open code, policy brief", 19, 24, "#555555")]
    fig, ax = plt.subplots(figsize=(6.5, 2.35))
    fig.subplots_adjust(left=0.42, right=0.94, top=0.97, bottom=0.17)
    n = len(tasks)
    ax.axvline(3, color="#0072B2", lw=0.8, ls=(0, (3, 2)), zorder=1)
    for i, (lab, s0, e0, c) in enumerate(tasks):
        ax.barh(i, e0 - s0, left=s0, height=0.5, color=c, alpha=0.88, edgecolor="none", zorder=2)
        ax.text(e0 + 0.45, i, f"M{s0 + 1}–M{e0}", va="center", ha="left", fontsize=6.8, color="#444",
                clip_on=False)
    ax.set_yticks(range(n), [t[0] for t in tasks])
    ax.tick_params(axis="y", length=0, pad=6)
    gates = [("Gate A: access to seized counterfeits", 3, n - 0.05, "#0072B2", "bold"),
             ("Gate B: corpus passes quality audit", 12, n + 0.75, "#444444", "normal")]
    for lab, mth, yv, c, w in gates:
        ax.scatter(mth, yv, marker="D", s=30, color=c, zorder=4)
        ax.text(mth + 0.45, yv, lab, va="center", ha="left", fontsize=6.9, color=c, fontweight=w)
    ax.set_xlim(0, 24); ax.set_ylim(n + 1.35, -0.6)
    ax.set_xticks(range(0, 25, 3)); ax.set_xlabel("Month")
    ax.spines["left"].set_visible(False)
    ax.set_axisbelow(True); ax.grid(axis="x")
    save(fig, "fig9_roadmap")


# ------------------------------------------------------------------------- tables
def tables(s):
    e1 = pd.read_csv(f"{RES}/e1_summary.csv")
    best = e1.accuracy.max()
    lines = []
    for _, r in e1.iterrows():
        acc = f"{r.accuracy * 100:.2f} $\\pm$ {r.accuracy_sd * 100:.2f}"
        if np.isclose(r.accuracy, best):
            acc = f"\\textbf{{{r.accuracy * 100:.2f}}} $\\pm$ {r.accuracy_sd * 100:.2f}"
        lines.append(f"{disp(r.model).replace('×', '$\\times$')} & {r.family.capitalize()} & {acc} & {r.roc_auc:.4f} & "
                     f"{r.miss_rate * 100:.2f} & {r.false_reject * 100:.2f} & {r.fit_ms:.0f} & {r.predict_us_per_note:.1f} \\\\")
    open(f"{TABDIR}/e1.tex", "w").write("\n".join(lines))
    e3 = pd.read_csv(f"{RES}/e3_ablation.csv")
    p = e3.pivot_table(index=["model", "family"], columns="subset", values="accuracy", aggfunc="mean") * 100
    p = p.reset_index().sort_values("Without variance", ascending=False)
    cols = ["All 4 features", "Without variance", "Kurtosis + entropy only"]
    mx = {c: p[c].max() for c in cols}
    lines = []
    for _, r in p.iterrows():
        cells = [f"\\textbf{{{r[c]:.2f}}}" if np.isclose(r[c], mx[c]) else f"{r[c]:.2f}" for c in cols]
        lines.append(f"{disp(r.model).replace('×', '$\\times$')} & {r.family.capitalize()} & " + " & ".join(cells) + " \\\\")
    open(f"{TABDIR}/e3.tex", "w").write("\n".join(lines))
    o = ordered(s)
    lines, prev = [], None
    for _, r in o.iterrows():
        if prev is not None and r.kind != prev:
            lines.append("\\midrule")
        prev = r.kind
        fam = r.family.capitalize() if r.kind != "hybrid" else "--"
        auc = "--" if np.isnan(r.auc) else (f"{r.auc:.4f}" if r.auc >= 0.9995 else f"{r.auc:.3f}")
        gap = (r.lofo - r.random) * 100
        gap_s = "0.0" if abs(gap) < 0.05 else f"{gap:+.1f}".replace("-", "$-$")
        f1 = lambda v: f"{v * 100:.2f}" if 0.9995 <= v < 1 else f"{v * 100:.1f}"
        lines.append(f"{disp(r.model).replace('×', '$\\times$')} & {r.kind.capitalize()} & {fam} & {f1(r.random)} & "
                     f"{f1(r.lofo)} [{f1(r.lo)}, {f1(r.hi)}] & {f1(r.worst)} & {gap_s} & {r.grr * 100:.1f} & {auc} \\\\")
    open(f"{TABDIR}/lofo.tex", "w").write("\n".join(lines))


if __name__ == "__main__":
    X, y = be.load_data()
    s = lofo_summary(3)
    only = set(sys.argv[2:])
    jobs = [("fig1", lambda: fig1(s)), ("fig2", lambda: fig2(X, y)), ("fig3", fig3), ("fig4", lambda: fig4(s)),
            ("fig5", fig5), ("fig6", fig6), ("fig7", fig7), ("fig8", fig8), ("fig9", fig9),
            ("tables", lambda: tables(s))]
    for name, job in jobs:
        if not only or name in only:
            res = job()
            if name == "fig6":
                print("mechanism:", {k: (v[0], v[1], round(v[2], 4), round(v[3], 4)) for k, v in res.items()})
    print("\nTEXT-COLLISION REPORT")
    for name, n, issues in REPORT:
        print(f"  {name:16s} {n:3d} text elements  ->  {'OK' if not issues else str(len(issues)) + ' issue(s)'}")
        for i in issues:
            print("      -", i)
