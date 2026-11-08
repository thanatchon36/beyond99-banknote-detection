#!/usr/bin/env python3
"""
Train every detector from the paper on the full UCI Banknote Authentication data,
calibrate its decision threshold at a genuine-rejection budget, save it as a
ready-to-use bundle, and score new notes.

  python train_all.py train   [--alpha 0.01] [--out models] [--seed 0]
  python train_all.py list    [--dir models]
  python train_all.py predict --model autoencoder --input notes.csv [--out scores.csv] [--dir models]

Input for `predict`: a CSV whose first four numeric columns are the features
(variance, skewness, kurtosis, entropy), with or without a header row; extra
columns such as a label are ignored.

Protocol
  Genuine notes are split 80/20 into a training part and a calibration part.
  Supervised detectors are trained on the genuine training part plus all
  counterfeits; one-class detectors on the genuine training part only.
  Each threshold is the (1 - alpha) quantile of the detector's scores on the
  calibration notes, so about alpha of genuine notes are rejected. Hybrids flag
  a note if either component exceeds its own alpha/2 threshold.

These bundles are for demonstration and downstream use; the evaluation of the
detectors is in banknote_experiments.py and paper_analysis.py.

Python API (run from this directory):
  import train_all
  det = train_all.load("autoencoder")
  flags = train_all.predict(det, [[-2.1, -3.4, 4.6, -0.9]])   # 1 = counterfeit
"""
import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time

import joblib
import numpy as np
import pandas as pd

import banknote_experiments as be

HYBRIDS = [("XGBoost", "Isolation Forest"), ("XGBoost", "Autoencoder"),
           ("MLP (3x64)", "Isolation Forest"), ("MLP (3x64)", "Autoencoder")]
ONE_CLASS_FAMILY = {"Isolation Forest": "tree", "Autoencoder": "deep",
                    "One-class SVM": "classical", "Mahalanobis": "classical"}


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ----------------------------------------------------------------- scoring API
def score(bundle, X):
    """Continuous score of a single detector; higher = more likely counterfeit."""
    X = np.asarray(X, dtype=float).reshape(-1, len(be.FEATURES))
    if bundle["kind"] == "supervised":
        return be.sup_score(bundle["model"], X)
    if bundle["kind"] == "one-class":
        return bundle["model"].score(X)
    raise ValueError("hybrids have no single score; use component_flags()")


def component_flags(bundle, X):
    """For a hybrid: {component name: 0/1 flags}."""
    return {c["name"]: (score(c, X) > c["threshold"]).astype(int) for c in bundle["components"]}


def predict(bundle, X):
    """1 = flagged as counterfeit, 0 = accepted as genuine."""
    if bundle["kind"] == "hybrid":
        return np.max(np.vstack(list(component_flags(bundle, X).values())), axis=0).astype(int)
    return (score(bundle, X) > bundle["threshold"]).astype(int)


def load(name, model_dir="models"):
    path = name if name.endswith(".joblib") else os.path.join(model_dir, f"{slug(name)}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run `python train_all.py train` first "
                                f"or see `python train_all.py list`")
    return joblib.load(path)


# ------------------------------------------------------------------- training
def train(alpha=0.01, out="models", seed=0):
    X, y = be.load_data()
    rng = np.random.RandomState(seed)
    gen = rng.permutation(np.where(y == 0)[0])
    cut = int(0.8 * len(gen))
    g_tr, g_cal = gen[:cut], gen[cut:]
    cf = np.where(y == 1)[0]
    tr = np.concatenate([g_tr, cf])
    os.makedirs(out, exist_ok=True)

    singles = {}
    for name, model in be.supervised_models(seed).items():
        t0 = time.time()
        model.fit(X[tr], y[tr])
        singles[name] = dict(name=name, kind="supervised", family=be.FAMILY[name], model=model,
                             fit_seconds=round(time.time() - t0, 3))
    for name, cls in be.ONE_CLASS.items():
        t0 = time.time()
        det = cls(seed).fit(X[g_tr])
        singles[name] = dict(name=name, kind="one-class", family=ONE_CLASS_FAMILY[name], model=det,
                             fit_seconds=round(time.time() - t0, 3))

    def threshold(bundle, a):
        return float(np.quantile(score(bundle, X[g_cal]), 1 - a))

    files = []
    for name, b in singles.items():
        b.update(threshold=threshold(b, alpha), alpha=alpha, features=list(be.FEATURES))
        fn = f"{slug(name)}.joblib"
        joblib.dump(b, os.path.join(out, fn), compress=3)
        files.append(dict(file=fn, name=name, kind=b["kind"], family=b["family"],
                          threshold=b["threshold"], fit_seconds=b["fit_seconds"],
                          calibration_genuine_rejected=float(predict(b, X[g_cal]).mean())))
    for a_name, b_name in HYBRIDS:
        comps = []
        for c in (a_name, b_name):
            cb = dict(singles[c])
            cb.update(threshold=threshold(cb, alpha / 2), alpha=alpha / 2, features=list(be.FEATURES))
            comps.append(cb)
        short = a_name.split(" (")[0]
        hb = dict(name=f"{short} + {b_name}", kind="hybrid", family="hybrid", components=comps,
                  alpha=alpha, features=list(be.FEATURES))
        fn = f"hybrid_{slug(short)}_{slug(b_name)}.joblib"
        joblib.dump(hb, os.path.join(out, fn), compress=3)
        files.append(dict(file=fn, name=hb["name"], kind="hybrid", family="hybrid",
                          threshold=None, fit_seconds=None,
                          calibration_genuine_rejected=float(predict(hb, X[g_cal]).mean())))

    raw = open("banknote.csv", "rb").read() if os.path.exists("banknote.csv") else b""
    manifest = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"), alpha=alpha, seed=seed,
        data=dict(source="UCI Banknote Authentication", n_genuine_train=len(g_tr),
                  n_genuine_calibration=len(g_cal), n_counterfeit=len(cf),
                  sha256=hashlib.sha256(raw).hexdigest() if raw else None),
        features=list(be.FEATURES), label_convention="0 = genuine, 1 = counterfeit",
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      **{m: __import__(m).__version__ for m in ("sklearn", "xgboost", "lightgbm", "joblib")}),
        detectors=files)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"saved {len(files)} detectors to {out}/ (genuine-rejection budget {alpha:.1%})")
    for d in files:
        print(f"  {d['file']:44s} {d['kind']:10s} calibration rejection {d['calibration_genuine_rejected']:.1%}")


# ------------------------------------------------------------------- CLI helpers
def read_features(path):
    df = pd.read_csv(path, header=None)
    if not all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes[:4]):
        df = pd.read_csv(path)                      # the file has a header row
        cols = {c.lower(): c for c in df.columns.astype(str)}
        if all(f in cols for f in be.FEATURES):
            return df[[cols[f] for f in be.FEATURES]].to_numpy(float)
    num = df.select_dtypes("number")
    if num.shape[1] < 4:
        raise ValueError("expected at least four numeric feature columns")
    return num.iloc[:, :4].to_numpy(float)


def cmd_predict(args):
    det = load(args.model, args.dir)
    X = read_features(args.input)
    out = pd.DataFrame({"row": np.arange(len(X))})
    if det["kind"] == "hybrid":
        for name, fl in component_flags(det, X).items():
            out[f"flag_{slug(name)}"] = fl
    else:
        out["score"] = score(det, X)
        out["threshold"] = det["threshold"]
    out["flagged_counterfeit"] = predict(det, X)
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"wrote {len(out)} rows to {args.out}")
    else:
        print(out.to_string(index=False))
    print(f"{det['name']}: flagged {out.flagged_counterfeit.sum()} of {len(out)} notes "
          f"(budget {det['alpha']:.1%} genuine rejection)", file=sys.stderr)


def cmd_list(args):
    path = os.path.join(args.dir, "manifest.json")
    if not os.path.exists(path):
        print(f"no manifest in {args.dir}/; run `python train_all.py train` first")
        return
    m = json.load(open(path))
    print(f"{len(m['detectors'])} detectors, trained {m['created']}, budget {m['alpha']:.1%}")
    for d in m["detectors"]:
        print(f"  {d['file'][:-7]:40s} {d['kind']:10s} {d['family']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train and save every detector")
    t.add_argument("--alpha", type=float, default=0.01, help="genuine-rejection budget (default 0.01)")
    t.add_argument("--out", default="models")
    t.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("predict", help="score notes with a saved detector")
    p.add_argument("--model", required=True, help="detector name, e.g. autoencoder, xgboost, "
                                                  "hybrid_xgboost_autoencoder")
    p.add_argument("--input", required=True)
    p.add_argument("--out")
    p.add_argument("--dir", default="models")
    li = sub.add_parser("list", help="list saved detectors")
    li.add_argument("--dir", default="models")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.alpha, a.out, a.seed)
    elif a.cmd == "predict":
        cmd_predict(a)
    else:
        cmd_list(a)
