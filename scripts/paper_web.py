#!/usr/bin/env python3
"""Build the web reader docs/paper/ from paper/paper.tex (same text, tables and figures as the PDF).

pandoc renders the LaTeX source with paper/reader.html: masthead, abstract, a results strip whose
numbers are computed from code/results/ the same way as the paper's tables, contents, numbered
sections and the appendices. The PDF and the PNG previews of the figures are copied next to the
page; GitHub Pages serves docs/ at https://thanatchon36.github.io/beyond99-banknote-detection/.

    python scripts/paper_web.py            # (make web) requires pandoc on PATH

Before pandoc sees the source, the generated table rows (\\inputrows) are inlined, the arXiv style
is dropped (pandoc cannot read arxiv.sty), figures point at their PNG previews and the algorithm
becomes a numbered list.
"""
from __future__ import annotations

import html
import importlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER, CODE, OUT = ROOT / "paper", ROOT / "code", ROOT / "docs" / "paper"
REPO_URL = "https://github.com/thanatchon36/beyond99-banknote-detection"
PAPER_URL = "https://thanatchon36.github.io/beyond99-banknote-detection/paper/"
DATA_URL = "https://doi.org/10.24432/C55P57"  # UCI Banknote Authentication


# ------------------------------------------------------------------ results strip
def results() -> list[tuple[str, str, str, str]]:
    """(label, style, finding, detail); style "pos" = one-class result, "ref" = failure, "null" = pending."""
    sys.path.insert(0, str(CODE))
    argv, sys.argv = sys.argv, ["make_figures.py", tempfile.mkdtemp()]  # make_figures reads its outdir at import
    try:
        import matplotlib
        matplotlib.use("Agg")
        mf = importlib.import_module("make_figures")
    finally:
        sys.argv = argv
    with _cwd(CODE):
        s = mf.lofo_summary(3).set_index("model")
        by_k = {k: mf.lofo_summary(k).set_index("model") for k in range(2, 7)}
        q = pd.read_csv("results/quadrant.csv")
    p = lambda v: f"{v * 100:.2f}" if 0.9995 <= v < 1 else f"{v * 100:.1f}"  # same rounding as Table 5
    trees = s.loc[["XGBoost", "Random Forest", "LightGBM"]]
    ae, ma = s.loc["Autoencoder"], s.loc["Mahalanobis"]
    floor = s[s.kind == "supervised"].random.min()
    xgb = q[(q.model == "XGBoost") & (q.group == "held-out")]
    accepted = xgb[xgb.sup <= xgb.sup_thr]
    outside = (accepted.maha / accepted.maha_thr > 1).mean()
    ae_k = min(min(t.loc["Autoencoder", "lofo"], t.loc["Autoencoder", "worst"]) for t in by_k.values())
    tree_k = max(t.loc[["XGBoost", "Random Forest", "LightGBM"], "lofo"].max() for t in by_k.values())
    return [
        ("Random split", "ref",
         f"Every supervised detector catches at least {p(floor)}% of counterfeits",
         "1% genuine-rejection budget; the UCI benchmark is saturated and cannot rank models"),
        ("Unseen family · trees", "ref",
         f"XGBoost, random forest and LightGBM catch {trees.lofo.min() * 100:.0f}–{trees.lofo.max() * 100:.0f}%",
         f"only {trees.worst.min() * 100:.0f}–{trees.worst.max() * 100:.0f}% of the hardest family; "
         "leave-one-family-out, K = 3, 10 seeds"),
        ("Unseen family · one-class", "pos",
         f"An autoencoder trained on genuine notes alone catches {p(ae.lofo)}%",
         f"95% CI {p(ae.lo)}–{p(ae.hi)}; {ae.grr * 100:.1f}% of genuine notes rejected; "
         f"Mahalanobis {p(ma.lofo)}%"),
        ("Mechanism", "ref",
         f"{outside * 100:.1f}% of the counterfeits XGBoost accepted lie outside the 99% genuine region",
         "the failure is extrapolation, not class overlap"),
        ("Robustness", "pos",
         f"Autoencoder at or above {p(ae_k)}% for every K from 2 to 6",
         f"mean and worst family; the tree ensembles never exceed {tree_k * 100:.1f}% on average"),
        ("Thai banknotes", "null",
         "Multispectral Thai data pending",
         "a pre-specified follow-up study on infrared, ultraviolet and magnetic features (Section 8)"),
    ]


class _cwd:
    def __init__(self, path: Path):
        self.path, self.old = path, None

    def __enter__(self):
        import os
        self.old = os.getcwd()
        os.chdir(self.path)

    def __exit__(self, *exc):
        import os
        os.chdir(self.old)


# ------------------------------------------------------------------ LaTeX -> pandoc
def _algorithm(m: re.Match) -> str:
    """algorithm/algorithmic -> a marked block of nested enumerate lists that pandoc can read."""
    block = m.group(0)
    caption = re.search(r"\\caption\{(.*?)\}\n", block).group(1)
    label = re.search(r"\\label\{(.*?)\}", block).group(1)
    body = re.search(r"\\begin\{algorithmic\}(?:\[\d\])?\n(.*?)\\end\{algorithmic\}", block, re.DOTALL).group(1)
    out, require = [], ""
    for line in body.splitlines():
        line = line.strip()
        line = re.sub(r"\\Comment\{(.*)\}", r" \\emph{(\1)}", line)
        if line.startswith("\\Require"):
            require = r"\textbf{Require:} " + line[len("\\Require"):].strip() + "\n\n"
        elif line.startswith("\\For{"):
            out.append(r"\item \textbf{for} " + line[len("\\For{"):-1] + r" \textbf{do}" + "\n\\begin{enumerate}")
        elif line.startswith("\\EndFor"):
            out.append("\\end{enumerate}")
        elif line.startswith("\\State \\Return"):
            out.append(r"\item \textbf{return} " + line[len("\\State \\Return"):].strip())
        elif line.startswith("\\State"):
            out.append(r"\item " + line[len("\\State"):].strip())
    return (f"\n\nALGORITHM-START {label}|{caption}\n\n{require}\\begin{{enumerate}}\n" + "\n".join(out)
            + "\n\\end{enumerate}\n\nALGORITHM-END\n\n")


def preprocess(tex: str) -> tuple[str, list[str]]:
    keywords = re.search(r"\\keywords\{(.*?)\}\n", tex).group(1)
    tex = tex.replace(re.search(r"\\keywords\{.*?\}\n", tex).group(0), "")
    tex = re.sub(r"\\newcommand\{\\inputrows\}.*\n", "", tex)
    tex = re.sub(r"\\inputrows\{([^}]*)\}", lambda m: (PAPER / m.group(1)).read_text(encoding="utf-8"), tex)
    tex = re.sub(r"\\usepackage\{arxiv\}\n", "", tex)
    tex = re.sub(r"\\renewcommand\{\\(?:shorttitle|headeright|undertitle)\}.*\n", "", tex)
    tex = re.sub(r"(figures/[\w.-]+)\.pdf", r"\1.png", tex)
    tex = re.sub(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", _algorithm, tex, flags=re.DOTALL)
    return tex, [k.strip() for k in keywords.split("\\and")]


# ------------------------------------------------------------------ HTML fixes
def _number_captions(page: str) -> str:
    """Table and figure numbers in captions, in document order (the PDF numbers them the same way)."""
    count = {"caption": 0, "figcaption": 0}

    def caption(m: re.Match) -> str:
        tag, attrs = m.group(1), m.group(2)
        count[tag] += 1
        word = "Table" if tag == "caption" else "Figure"
        return f"<{tag}{attrs}><b>{word} {count[tag]}.</b> "

    return re.sub(r"<(caption|figcaption)(\b[^>]*)>", caption, page)


def _appendix_letters(page: str) -> str:
    """pandoc numbers the appendices on from the last section; the PDF letters them A, B, ..."""
    first = re.search(r'<h1 data-number="(\d+)" id="app:', page)
    if not first:
        return page
    n0 = int(first.group(1))
    top = max(int(n) for n in re.findall(r'<h1 data-number="(\d+)"', page))
    for n in range(n0, top + 1):
        letter = chr(ord("A") + n - n0)
        page = re.sub(rf'(data-number="|section-number">){n}(?=[".<])', rf"\g<1>{letter}", page)
        page = re.sub(rf'(data-reference="app:[^"]*">){n}(?=[.<])', rf"\g<1>{letter}", page)
    return page


def _algorithms(page: str) -> str:
    n = 0

    def block(m: re.Match) -> str:
        nonlocal n
        n += 1
        label, caption = m.group(1), m.group(2)
        page_refs[label] = str(n)
        return (f'<div class="algo" id="{label}"><p class="tcap"><b>Algorithm {n}.</b> {caption}</p>'
                f"{m.group(3)}</div>")

    page_refs: dict[str, str] = {}
    page = re.sub(r"<p>ALGORITHM-START ([^|]+)\|(.*?)</p>(.*?)<p>ALGORITHM-END</p>", block, page, flags=re.DOTALL)
    for label, num in page_refs.items():
        page = page.replace(f'data-reference="{label}">[{label}]</a>', f'data-reference="{label}">{num}</a>')
    return page


def _web_images(page: str) -> str:
    """Drop the PDF sizing and load lazily."""
    def img(m: re.Match) -> str:
        tag = re.sub(r'\s(?:style|width|height)="[^"]*"', "", m.group(0))
        return tag.replace("<img ", '<img loading="lazy" ', 1)

    return re.sub(r"<img\b[^>]*>", img, page)


def build(out_dir: Path = OUT) -> Path:
    if shutil.which("pandoc") is None:
        raise RuntimeError("pandoc not found; install pandoc to build the web reader")
    tex, keywords = preprocess((PAPER / "paper.tex").read_text(encoding="utf-8"))
    args = [
        "pandoc", "--from=latex", "--to=html5", "--mathml", "--wrap=none", "--standalone",
        "--template=reader.html", "--number-sections", "--toc", "--toc-depth=1",
        "--citeproc", "--bibliography=references.bib",
        "--metadata=pagetitle:Beyond 99%",
        "--metadata=description:Preprint: open-set evaluation of counterfeit banknote detectors "
        "on counterfeit families they have never seen.",
        "--metadata=link-citations:true",
    ] + [f"--metadata=keywords:{k}" for k in keywords]
    page = subprocess.run(args, input=tex, capture_output=True, text=True, check=True, cwd=str(PAPER),
                          timeout=120).stdout

    page = _number_captions(page)
    page = _appendix_letters(page)
    page = _algorithms(page)
    page = _web_images(page)
    # The caption sits above the table, outside its sideways scroll, so it wraps at the text width.
    page = re.sub(
        r"<table\b[^>]*>\s*<caption>(.*?)</caption>(.*?)</table>",
        r'<div class="tbl"><p class="tcap">\1</p><div class="tscroll"><table>\2</table></div></div>',
        page, flags=re.DOTALL,
    )
    page = re.sub(r"<colgroup>.*?</colgroup>\s*", "", page, flags=re.DOTALL)
    page = re.sub(r' style="text-align: (?:left|center|right);"', "", page)

    items = "\n".join(
        f'<li class="r r-{style}"><span class="r-id">{html.escape(label)}</span>'
        f'<span class="r-text">{html.escape(finding)}<small>{html.escape(detail)}</small></span></li>'
        for label, style, finding, detail in results()
    )
    if re.search(r"\bnan\b", items):
        raise RuntimeError("results strip has unresolved numbers")
    actions = (
        '<li><a href="paper.pdf">Download PDF</a></li>'
        f'<li><a href="{REPO_URL}">Code and replication</a></li>'
        f'<li><a href="{DATA_URL}">Data (UCI)</a></li>'
    )
    title = html.unescape(re.sub(r"<[^>]+>", "", re.search(r"<h1>(.*?)</h1>", page, re.DOTALL).group(1)))
    bib = (
        "@misc{chongmankhong2026beyond,\n"
        "  author = {Chongmankhong, Thanatchon},\n"
        f"  title  = {{{title.replace('%', chr(92) + '%')}}},\n"
        "  year   = {2026},\n"
        "  note   = {Preprint},\n"
        f"  url    = {{{PAPER_URL}}}\n"
        "}"
    )
    cite = f"<h2>How to cite</h2><pre><code>{html.escape(bib)}</code></pre>"
    colophon = (
        "<p>Rendered from <code>paper/paper.tex</code> by <code>scripts/paper_web.py</code> "
        "(<code>make web</code>). Tables are the same generated rows as the PDF, and the results strip "
        "is computed from <code>code/results/</code>. The PDF is authoritative.</p>"
    )
    for marker, value in (
        ("<!--ACTIONS-->", actions),
        ("<!--RESULTS-NOTE-->", (
            "Headline numbers, computed from the result files: counterfeits detected at a 1% "
            "genuine-rejection budget. Amber: where supervised detectors fail; green: one-class screening; "
            "grey: pending."
        )),
        ("<!--RESULTS-->", items),
        ("<!--CITE-->", cite),
        ("<!--COLOPHON-->", colophon),
    ):
        if marker not in page:
            raise RuntimeError(f"reader.html lacks {marker}")
        page = page.replace(marker, value)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    shutil.copyfile(PAPER / "paper.pdf", out_dir / "paper.pdf")
    figs = out_dir / "figures"
    shutil.rmtree(figs, ignore_errors=True)
    figs.mkdir()
    for name in sorted(set(re.findall(r'src="figures/([\w.-]+)"', page))):
        shutil.copyfile(PAPER / "figures" / name, figs / name)
    return out_dir / "index.html"


if __name__ == "__main__":
    print(f"wrote {build().relative_to(ROOT)}")
