#!/usr/bin/env python3
"""Assemble the Hugging Face model repo thanatchon36/beyond99-banknote-detection in hf_models/ (git-ignored).

    python scripts/hf_models.py                  # (make hf-models) trains code/models/ first if missing
    hf upload thanatchon36/beyond99-banknote-detection hf_models . --repo-type model

Contents: the 18 calibrated detectors and manifest.json from `train_all.py train`, the two modules the
pickles need (banknote_experiments.py, train_all.py), pinned requirements, the model card with each
detector's results from the paper (scripts/hf_model_README.md), the MIT licence and SHA256SUMS.
"""
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE, OUT = ROOT / "code", ROOT / "hf_models"
PIP = {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "xgboost": "xgboost",
       "lightgbm": "lightgbm", "joblib": "joblib"}


def results_table(manifest: dict) -> str:
    sys.path.insert(0, str(CODE))
    import matplotlib
    matplotlib.use("Agg")
    argv, sys.argv = sys.argv, ["make_figures.py", tempfile.mkdtemp()]
    try:
        mf = importlib.import_module("make_figures")
    finally:
        sys.argv = argv
    import os
    old = os.getcwd()
    os.chdir(CODE)
    try:
        lofo = mf.lofo_summary(3).set_index("model")
        e1 = pd.read_csv("results/e1_summary.csv").set_index("model")
    finally:
        os.chdir(old)
    pct = lambda v: f"{v * 100:.2f}%" if 0.9995 <= v < 1 else f"{v * 100:.1f}%"
    rows = ["| file | detector | kind | calibration rejection | E1 accuracy, random split | "
            "Table 5: unseen family, mean (worst) |", "|---|---|---|---:|---:|---:|"]
    for d in manifest["detectors"]:
        name = d["name"]
        acc = pct(e1.loc[name, "accuracy"]) if name in e1.index else "—"
        lo = f"{pct(lofo.loc[name, 'lofo'])} ({pct(lofo.loc[name, 'worst'])})" if name in lofo.index else "—"
        label = mf.DISPLAY.get(name, name)
        if name in ("Autoencoder", "Mahalanobis"):
            label = f"**{label}**"
        rows.append(f"| `{d['file'][:-7]}` | {label} | {d['kind']} | {d['calibration_genuine_rejected'] * 100:.1f}% | "
                    f"{acc} | {lo} |")
    return "\n".join(rows) + "\n\n" + degenerate_note(manifest)


def degenerate_note(manifest: dict) -> str:
    """Name any saved detector that flags no counterfeit at all (a coarse score whose calibrated
    threshold equals its maximum, so `score > threshold` never fires)."""
    import train_all
    import banknote_experiments as be
    X, y = be.load_data(str(CODE / "banknote.csv"))
    dead = []
    for d in manifest["detectors"]:
        det = train_all.load(d["file"][:-7], str(CODE / "models"))
        if train_all.predict(det, X[y == 1]).sum() == 0:
            dead.append((d["file"][:-7], det.get("threshold")))
    if not dead:
        return ""
    names = ", ".join(f"`{n}` (threshold {thr:g})" for n, thr in dead)
    return ("> **Known issue.** " + names + " flags no note at all, not even the counterfeits it was trained on: its "
            "score takes only a few values, the 99th percentile of the calibration scores equals the maximum, and "
            "the rule `score > threshold` never fires. Use it only for its raw `score`, or pick another detector.")


def main() -> None:
    models = CODE / "models"
    if not (models / "manifest.json").exists():
        subprocess.run([sys.executable, "train_all.py", "train"], cwd=CODE, check=True)
    manifest = json.loads((models / "manifest.json").read_text())
    if len(manifest["detectors"]) != 18:
        raise SystemExit("expected 18 detectors in code/models/manifest.json")

    shutil.rmtree(OUT, ignore_errors=True)
    shutil.copytree(models, OUT / "models")
    for f in ("banknote_experiments.py", "train_all.py"):
        shutil.copyfile(CODE / f, OUT / f)
    v = manifest["versions"]
    (OUT / "requirements.txt").write_text(
        "".join(f"{PIP[k]}=={v[k]}\n" for k in PIP) + "scipy\nmatplotlib\n")
    card = (ROOT / "scripts" / "hf_model_README.md").read_text()
    card = card.replace("{{TABLE}}", results_table(manifest))
    card = card.replace("{{VERSIONS}}", ", ".join(f"{PIP.get(k, k)} {v[k]}" for k in ["python", *PIP]))
    if "{{" in card:
        raise SystemExit("model card has unresolved placeholders")
    (OUT / "README.md").write_text(card)
    shutil.copyfile(ROOT / "LICENSE", OUT / "LICENSE")

    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in {"README.md", "SHA256SUMS"})
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT).as_posix()}" for p in files]
    (OUT / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(f"wrote {len(files) + 2} files to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
