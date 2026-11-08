#!/usr/bin/env python3
"""
Counterfeit banknote detection: tree-based vs deep learning, a reproducible pilot.

Data    UCI Banknote Authentication (Lohweg et al., 2013): 1,372 notes,
        4 statistics of a wavelet-transformed image (variance, skewness,
        kurtosis, entropy). Label convention used here, as in most of the
        literature: 0 = genuine, 1 = counterfeit (the positive class).
        The original documentation does not state the mapping explicitly.
Source  UCI Machine Learning Repository (dataset 267); a GitHub mirror is used as fallback.

Experiments
  e1  Standard benchmark: repeated stratified 10-fold CV (5 x 10), 10 models
  e2  Learning curve: accuracy and miss rate vs. number of training notes
  e3  Feature ablation: a weaker sensor (drop the strongest features)
  e4  Noise robustness: Gaussian noise on test features (worn notes, sensor drift)
  e5  Unseen counterfeit families: leave-one-family-out (LOFO), supervised vs
      one-class vs hybrid, at a fixed 1% genuine-rejection operating point

Usage
  python banknote_experiments.py            # run everything (about 10 min on 1 CPU)
  python banknote_experiments.py e1 e5      # run selected experiments
Outputs: ./results/*.csv and ./figures/*.png
"""
import os
import sys
import time
import urllib.request
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              IsolationForest, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedShuffleSplit)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

URLS = [  # official UCI file first, then a widely used GitHub mirror of the same file
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt",
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv",
]
FEATURES = ["variance", "skewness", "kurtosis", "entropy"]
RES, FIG = "results", "figures"
TARGET_FPR = 0.01  # operating point: at most ~1% of genuine notes rejected

FAMILY = {  # model family used in the paper's comparison
    "Logistic Regression": "classical", "kNN (k=5)": "classical", "SVM (RBF)": "classical",
    "Decision Tree": "tree", "Random Forest": "tree", "Extra Trees": "tree",
    "HistGradientBoosting": "tree", "XGBoost": "tree", "LightGBM": "tree",
    "MLP (3x64)": "deep",
}


# ----------------------------------------------------------------------------- data
def load_data(path="banknote.csv"):
    """Download the data on first use (UCI, then mirror) and validate it."""
    if not os.path.exists(path):
        errors = []
        for url in URLS:
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    payload = r.read()
                with open(path, "wb") as f:
                    f.write(payload)
                break
            except Exception as e:  # network error, blocked host, ...
                errors.append(f"{url}: {e}")
        else:
            raise RuntimeError("could not download the dataset:\n" + "\n".join(errors))
    df = pd.read_csv(path, header=None, names=FEATURES + ["label"])
    counts = df["label"].value_counts().to_dict()
    if df.shape != (1372, 5) or counts != {0: 762, 1: 610}:
        raise ValueError(f"unexpected data in {path}: shape {df.shape}, labels {counts}")
    return df[FEATURES].to_numpy(float), df["label"].to_numpy(int)


# --------------------------------------------------------------------------- models
def supervised_models(seed, names=None):
    m = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "kNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=seed),
        "MLP (3x64)": make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 64, 64), alpha=1e-4, max_iter=1000,
            n_iter_no_change=20, tol=1e-5, random_state=seed)),
    }
    if XGBClassifier is not None:
        m["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                                     subsample=0.9, random_state=seed, n_jobs=1,
                                     verbosity=0, eval_metric="logloss")
    if LGBMClassifier is not None:
        m["LightGBM"] = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                       min_child_samples=5, random_state=seed, n_jobs=1,
                                       verbose=-1)
    return {k: v for k, v in m.items() if names is None or k in names}


class AutoencoderScorer:
    """Deep one-class detector: 4-32-2-32-4 autoencoder, score = reconstruction error."""

    def __init__(self, seed):
        self.sc = StandardScaler()
        self.ae = MLPRegressor(hidden_layer_sizes=(32, 2, 32), activation="tanh",
                               alpha=1e-4, max_iter=3000, tol=1e-6,
                               n_iter_no_change=30, random_state=seed)

    def fit(self, X):
        Z = self.sc.fit_transform(X)
        self.ae.fit(Z, Z)
        return self

    def score(self, X):
        Z = self.sc.transform(X)
        return ((self.ae.predict(Z) - Z) ** 2).mean(axis=1)


class IForestScorer:
    """Tree-based one-class detector."""

    def __init__(self, seed):
        self.m = IsolationForest(n_estimators=300, random_state=seed)

    def fit(self, X):
        self.m.fit(X)
        return self

    def score(self, X):
        return -self.m.score_samples(X)


class OCSVMScorer:
    def __init__(self, seed):
        self.m = make_pipeline(StandardScaler(), OneClassSVM(nu=0.05, gamma="scale"))

    def fit(self, X):
        self.m.fit(X)
        return self

    def score(self, X):
        return -self.m.decision_function(X)


class MahalanobisScorer:
    """Statistical baseline: distance from the genuine-note distribution."""

    def __init__(self, seed=None):
        self.m = EmpiricalCovariance()

    def fit(self, X):
        self.m.fit(X)
        return self

    def score(self, X):
        return self.m.mahalanobis(X)


ONE_CLASS = {
    "Isolation Forest": IForestScorer,
    "One-class SVM": OCSVMScorer,
    "Autoencoder": AutoencoderScorer,
    "Mahalanobis": MahalanobisScorer,
}


def sup_score(model, X):
    """Continuous score, higher = more likely counterfeit."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


# -------------------------------------------------------------------------- metrics
def metrics(y, pred, score=None):
    tp = int(((y == 1) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    out = {"accuracy": (tp + tn) / len(y),
           "miss_rate": fn / max(tp + fn, 1),      # counterfeit accepted as genuine
           "false_reject": fp / max(tn + fp, 1),   # genuine rejected
           "f1": 2 * tp / max(2 * tp + fp + fn, 1)}
    if score is not None and len(np.unique(y)) == 2:
        out["roc_auc"] = roc_auc_score(y, score)
    return out


def corrected_ttest(d, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test on paired fold differences."""
    d = np.asarray(d, float)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), df=k - 1))


# ---------------------------------------------------------------------- experiments
def e1_benchmark(X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        for name, model in supervised_models(seed=fold).items():
            t0 = time.perf_counter(); model.fit(X[tr], y[tr]); fit_s = time.perf_counter() - t0
            t0 = time.perf_counter(); pred = model.predict(X[te])
            us_per_note = (time.perf_counter() - t0) / len(te) * 1e6
            r = metrics(y[te], pred, sup_score(model, X[te]))
            rows.append({"fold": fold, "model": name, "family": FAMILY[name],
                         "fit_seconds": fit_s, "predict_us_per_note": us_per_note, **r})
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/e1_folds.csv", index=False)
    summ = (df.groupby(["model", "family"])
              .agg(accuracy=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
                   roc_auc=("roc_auc", "mean"), miss_rate=("miss_rate", "mean"),
                   false_reject=("false_reject", "mean"), fit_ms=("fit_seconds", "mean"),
                   predict_us_per_note=("predict_us_per_note", "mean"))
              .reset_index().sort_values("accuracy", ascending=False))
    summ["fit_ms"] *= 1000
    summ.to_csv(f"{RES}/e1_summary.csv", index=False)
    # paired test: best tree-based model vs MLP
    best_tree = summ[summ.family == "tree"].iloc[0]["model"]
    a = df[df.model == best_tree].sort_values("fold")["accuracy"].to_numpy()
    b = df[df.model == "MLP (3x64)"].sort_values("fold")["accuracy"].to_numpy()
    t, p = corrected_ttest(a - b, n_train=0.9 * len(y), n_test=0.1 * len(y))
    try:
        w_p = stats.wilcoxon(a, b, zero_method="zsplit").pvalue
    except ValueError:
        w_p = 1.0
    pd.DataFrame([{"comparison": f"{best_tree} vs MLP (3x64)", "mean_diff": (a - b).mean(),
                   "corrected_t": t, "corrected_p": p, "wilcoxon_p": w_p}]
                 ).to_csv(f"{RES}/e1_test.csv", index=False)
    print(summ.round(4).to_string(index=False))
    print(f"{best_tree} vs MLP: diff={(a - b).mean():.4f}, corrected p={p:.3f}, Wilcoxon p={w_p:.3f}")


def e2_learning_curve(X, y, sizes=(10, 20, 40, 80, 160, 320, 640), reps=30):
    names = ["Logistic Regression", "SVM (RBF)", "Random Forest", "XGBoost", "LightGBM", "MLP (3x64)"]
    outer = StratifiedShuffleSplit(n_splits=reps, test_size=0.3, random_state=7)
    rows = []
    for rep, (pool, te) in enumerate(outer.split(X, y)):
        rng = np.random.RandomState(rep)
        for n in sizes:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=rng.randint(1e6))
            sub, _ = next(sss.split(X[pool], y[pool]))
            tr = pool[sub]
            for name, model in supervised_models(seed=rep, names=names).items():
                model.fit(X[tr], y[tr])
                r = metrics(y[te], model.predict(X[te]))
                rows.append({"rep": rep, "n_train": n, "model": name, "family": FAMILY[name], **r})
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/e2_learning_curve.csv", index=False)
    piv = df.pivot_table(index="n_train", columns="model", values="accuracy", aggfunc="mean")
    print((piv * 100).round(2).to_string())


def e3_ablation(X, y):
    subsets = {"All 4 features": [0, 1, 2, 3],
               "Without variance": [1, 2, 3],
               "Kurtosis + entropy only": [2, 3]}
    names = ["Logistic Regression", "kNN (k=5)", "SVM (RBF)", "Decision Tree",
             "Random Forest", "XGBoost", "LightGBM", "MLP (3x64)"]
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=11)
    rows = []
    for sname, cols in subsets.items():
        Xs = X[:, cols]
        for fold, (tr, te) in enumerate(cv.split(Xs, y)):
            for name, model in supervised_models(seed=fold, names=names).items():
                model.fit(Xs[tr], y[tr])
                r = metrics(y[te], model.predict(Xs[te]), sup_score(model, Xs[te]))
                rows.append({"subset": sname, "fold": fold, "model": name, "family": FAMILY[name], **r})
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/e3_ablation.csv", index=False)
    piv = df.pivot_table(index="model", columns="subset", values="accuracy", aggfunc="mean")
    print((piv[list(subsets)] * 100).round(2).sort_values("Without variance", ascending=False).to_string())


def e4_noise(X, y, alphas=(0, 0.1, 0.25, 0.5, 0.75, 1.0), reps=10, draws=5):
    names = ["Logistic Regression", "kNN (k=5)", "SVM (RBF)", "Decision Tree",
             "Random Forest", "XGBoost", "LightGBM", "MLP (3x64)"]
    sss = StratifiedShuffleSplit(n_splits=reps, test_size=0.3, random_state=21)
    rows = []
    for rep, (tr, te) in enumerate(sss.split(X, y)):
        sd = X[tr].std(axis=0)
        for name, model in supervised_models(seed=rep, names=names).items():
            model.fit(X[tr], y[tr])
            for a in alphas:
                for d in range(draws):
                    rng = np.random.RandomState(1000 * rep + d)
                    Xn = X[te] + rng.normal(0, 1, X[te].shape) * sd * a
                    r = metrics(y[te], model.predict(Xn))
                    rows.append({"rep": rep, "alpha": a, "draw": d, "model": name,
                                 "family": FAMILY[name], **r})
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/e4_noise.csv", index=False)
    piv = df.pivot_table(index="model", columns="alpha", values="accuracy", aggfunc="mean")
    print((piv * 100).round(2).sort_values(0.5, ascending=False).to_string())


def counterfeit_families(X, y, kmax=6):
    """Proxy 'counterfeit families' = clusters of the counterfeit class.
    K is chosen by the best silhouette score in 2..kmax (rule fixed before testing)."""
    Xc = StandardScaler().fit_transform(X[y == 1])
    best = max(range(2, kmax + 1), key=lambda k: silhouette_score(
        Xc, KMeans(k, n_init=20, random_state=0).fit_predict(Xc)))
    lab = KMeans(best, n_init=20, random_state=0).fit_predict(Xc)
    fam = np.full(len(y), -1)
    fam[y == 1] = lab
    return fam, best


def e5_unseen_families(X, y, seeds=10):
    fam, K = counterfeit_families(X, y)
    print(f"counterfeit families: K={K}, sizes={np.bincount(fam[fam >= 0]).tolist()}")
    sup_names = ["Logistic Regression", "SVM (RBF)", "Random Forest", "XGBoost",
                 "LightGBM", "MLP (3x64)"]
    gen_idx, cf_idx = np.where(y == 0)[0], np.where(y == 1)[0]
    rows = []
    for seed in range(seeds):
        rng = np.random.RandomState(seed)
        g = rng.permutation(gen_idx)
        n = len(g)
        g_tr, g_val, g_te = g[:int(.5 * n)], g[int(.5 * n):int(.7 * n)], g[int(.7 * n):]
        # one-class detectors are fitted on genuine notes only, once per seed
        oc = {k: cls(seed).fit(X[g_tr]) for k, cls in ONE_CLASS.items()}
        splits = [(f"family {f}", cf_idx[fam[cf_idx] != f], cf_idx[fam[cf_idx] == f])
                  for f in range(K)]
        c_perm = rng.permutation(cf_idx)  # reference: the usual random split
        splits.append(("random split", c_perm[:int(.7 * len(c_perm))], c_perm[int(.7 * len(c_perm)):]))
        for setting, c_tr, c_te in splits:
            tr = np.concatenate([g_tr, c_tr])
            te = np.concatenate([g_te, c_te])
            scores = {}
            for name, model in supervised_models(seed, names=sup_names).items():
                model.fit(X[tr], y[tr])
                s_val, s_te = sup_score(model, X[g_val]), sup_score(model, X[te])
                scores[name] = (s_val, s_te)
                rows.append({"seed": seed, "setting": setting, "model": name, "kind": "supervised",
                             "family": FAMILY[name], "threshold": "default 0.5",
                             **metrics(y[te], model.predict(X[te]))})
            for name, det in oc.items():
                scores[name] = (det.score(X[g_val]), det.score(X[te]))
            for name, (s_val, s_te) in scores.items():
                thr = np.quantile(s_val, 1 - TARGET_FPR)
                kind = "one-class" if name in ONE_CLASS else "supervised"
                fam_lbl = {"Isolation Forest": "tree", "Autoencoder": "deep"}.get(
                    name, FAMILY.get(name, "classical"))
                rows.append({"seed": seed, "setting": setting, "model": name, "kind": kind,
                             "family": fam_lbl, "threshold": "1% genuine reject",
                             **metrics(y[te], (s_te > thr).astype(int), s_te)})
            # hybrids: supervised OR one-class, each at half the rejection budget
            for hname, a, b, fl in [("Hybrid: XGBoost + Isolation Forest", "XGBoost", "Isolation Forest", "tree"),
                                    ("Hybrid: MLP + Autoencoder", "MLP (3x64)", "Autoencoder", "deep")]:
                flag = np.zeros(len(te), bool)
                for comp in (a, b):
                    s_val, s_te = scores[comp]
                    flag |= s_te > np.quantile(s_val, 1 - TARGET_FPR / 2)
                rows.append({"seed": seed, "setting": setting, "model": hname, "kind": "hybrid",
                             "family": fl, "threshold": "1% genuine reject",
                             **metrics(y[te], flag.astype(int))})
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/e5_unseen_families.csv", index=False)
    op = df[df.threshold == "1% genuine reject"].copy()
    op["detection"] = 1 - op["miss_rate"]
    lofo = op[op.setting.str.startswith("family")]
    summ = (lofo.groupby(["model", "kind", "family"])
                .agg(detection_unseen=("detection", "mean"),
                     false_reject=("false_reject", "mean")).reset_index())
    worst = (lofo.groupby(["model", "seed", "setting"])["detection"].mean()
                 .groupby(["model", "seed"]).min().groupby("model").mean()
                 .rename("detection_worst_family"))
    rand = (op[op.setting == "random split"].groupby("model")["detection"].mean()
              .rename("detection_random_split"))
    summ = summ.join(worst, on="model").join(rand, on="model")
    per_fam = lofo.pivot_table(index="model", columns="setting", values="detection", aggfunc="mean")
    summ = summ.join(per_fam, on="model").sort_values("detection_unseen", ascending=False)
    summ.to_csv(f"{RES}/e5_summary.csv", index=False)
    default = df[(df.threshold == "default 0.5") & df.setting.str.startswith("family")]
    print(summ.round(3).to_string(index=False))
    print("\nSupervised at default threshold 0.5 (unseen family detection):")
    print((1 - default.groupby("model")["miss_rate"].mean()).round(3).to_string())


# -------------------------------------------------------------------------- figures
def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if os.path.exists(f"{RES}/e2_learning_curve.csv"):
        df = pd.read_csv(f"{RES}/e2_learning_curve.csv")
        piv = df.pivot_table(index="n_train", columns="model", values="accuracy", aggfunc="mean")
        ax = (piv * 100).plot(marker="o", logx=True, figsize=(7, 4.2))
        ax.set_xlabel("training notes (log scale)"); ax.set_ylabel("test accuracy (%)")
        ax.set_title("E2 Learning curve"); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f"{FIG}/e2_learning_curve.png", dpi=160); plt.close()
    if os.path.exists(f"{RES}/e4_noise.csv"):
        df = pd.read_csv(f"{RES}/e4_noise.csv")
        piv = df.pivot_table(index="alpha", columns="model", values="accuracy", aggfunc="mean")
        ax = (piv * 100).plot(marker="o", figsize=(7, 4.2))
        ax.set_xlabel("test noise (x feature std)"); ax.set_ylabel("test accuracy (%)")
        ax.set_title("E4 Noise robustness"); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f"{FIG}/e4_noise.png", dpi=160); plt.close()
    if os.path.exists(f"{RES}/e5_summary.csv"):
        s = pd.read_csv(f"{RES}/e5_summary.csv").sort_values("detection_unseen")
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.barh(s.model, s.detection_unseen * 100, color="#9aa3ad")
        ax.scatter(s.detection_random_split * 100, s.model, color="#1f5fbf", zorder=3,
                   label="random split (usual evaluation)")
        ax.set_xlabel("counterfeits detected at 1% genuine rejection (%)")
        ax.set_title("E5 Unseen counterfeit family vs random split"); ax.legend(loc="lower right")
        ax.set_xlim(0, 100); plt.tight_layout()
        plt.savefig(f"{FIG}/e5_unseen_families.png", dpi=160); plt.close()


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    X, y = load_data()
    todo = sys.argv[1:] or ["e1", "e2", "e3", "e4", "e5"]
    runs = {"e1": e1_benchmark, "e2": e2_learning_curve, "e3": e3_ablation,
            "e4": e4_noise, "e5": e5_unseen_families}
    for e in todo:
        print(f"\n===== {e} =====")
        t0 = time.time(); runs[e](X, y); print(f"[{e} done in {time.time() - t0:.0f}s]")
    figures()
