#!/usr/bin/env python3
"""
Extended analyses for the paper (reuses models and data from banknote_experiments.py).

  lofo K      Leave-one-family-out with K counterfeit families (K = 2..6), 10 seeds,
              14 detectors + a 2x2 grid of hybrids, at 7 genuine-rejection budgets.
              -> results/e5x_K{K}.csv   (K = 3 reproduces Experiment E5 exactly)
  quadrant    4-D mechanism check: supervised score vs distance from the genuine
              class for every unseen-family note.  -> results/quadrant.csv

Usage:  python paper_analysis.py lofo 2 3 4 5 6      python paper_analysis.py quadrant
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import banknote_experiments as be

warnings.filterwarnings("ignore")

ALPHAS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
SUP = ["Logistic Regression", "SVM (RBF)", "Random Forest", "XGBoost", "LightGBM", "MLP (3x64)"]
HYBRIDS = [("XGBoost", "Isolation Forest"), ("XGBoost", "Autoencoder"),
           ("MLP (3x64)", "Isolation Forest"), ("MLP (3x64)", "Autoencoder")]
FAM = {**be.FAMILY, "Isolation Forest": "tree", "Autoencoder": "deep",
       "One-class SVM": "classical", "Mahalanobis": "classical"}


def families_k(X, y, K):
    """Counterfeit 'families' = KMeans clusters of the standardized counterfeit class."""
    Xc = StandardScaler().fit_transform(X[y == 1])
    lab = KMeans(K, n_init=20, random_state=0).fit_predict(Xc)
    fam = np.full(len(y), -1)
    fam[y == 1] = lab
    return fam


def genuine_split(gen_idx, rng):
    g = rng.permutation(gen_idx)
    n = len(g)
    return g[:int(.5 * n)], g[int(.5 * n):int(.7 * n)], g[int(.7 * n):]


def lofo(X, y, K, seeds=10):
    fam = families_k(X, y, K)
    gen_idx, cf_idx = np.where(y == 0)[0], np.where(y == 1)[0]
    rows = []
    for seed in range(seeds):
        rng = np.random.RandomState(seed)
        g_tr, g_val, g_te = genuine_split(gen_idx, rng)
        oc = {k: cls(seed).fit(X[g_tr]) for k, cls in be.ONE_CLASS.items()}
        splits = [(f"family {f}", cf_idx[fam[cf_idx] != f], cf_idx[fam[cf_idx] == f])
                  for f in range(K)]
        c_perm = rng.permutation(cf_idx)
        cut = int(.7 * len(c_perm))
        splits.append(("random split", c_perm[:cut], c_perm[cut:]))
        for setting, c_tr, c_te in splits:
            tr = np.concatenate([g_tr, c_tr])
            s = {}
            for name, model in be.supervised_models(seed, names=SUP).items():
                model.fit(X[tr], y[tr])
                s[name] = tuple(be.sup_score(model, X[i]) for i in (g_val, g_te, c_te))
            for name, det in oc.items():
                s[name] = tuple(det.score(X[i]) for i in (g_val, g_te, c_te))
            for name, (sv, sg, sc) in s.items():
                auc = roc_auc_score(np.r_[np.zeros(len(sg)), np.ones(len(sc))], np.r_[sg, sc])
                for a in ALPHAS:
                    thr = np.quantile(sv, 1 - a)
                    rows.append(dict(K=K, seed=seed, setting=setting, model=name,
                                     kind="one-class" if name in be.ONE_CLASS else "supervised",
                                     family=FAM[name], alpha=a, cdr=float((sc > thr).mean()),
                                     grr=float((sg > thr).mean()), auc=auc))
            for a_name, b_name in HYBRIDS:   # flag if either component flags, alpha/2 each
                for a in ALPHAS:
                    fg = np.zeros(len(g_te), bool)
                    fc = np.zeros(len(c_te), bool)
                    for comp in (a_name, b_name):
                        sv, sg, sc = s[comp]
                        thr = np.quantile(sv, 1 - a / 2)
                        fg |= sg > thr
                        fc |= sc > thr
                    rows.append(dict(K=K, seed=seed, setting=setting,
                                     model=f"{a_name.split(' (')[0]} + {b_name}", kind="hybrid",
                                     family=f"{FAM[a_name]}+{FAM[b_name]}", alpha=a,
                                     cdr=float(fc.mean()), grr=float(fg.mean()), auc=np.nan))
    return pd.DataFrame(rows)


class AE1:
    """2-D autoencoder with a one-unit bottleneck (a 2-unit code could learn the identity)."""

    def __init__(self, seed):
        self.sc = StandardScaler()
        self.m = MLPRegressor(hidden_layer_sizes=(32, 1, 32), activation="tanh", alpha=1e-4,
                              max_iter=4000, tol=1e-7, n_iter_no_change=50, random_state=seed)

    def fit(self, X):
        Z = self.sc.fit_transform(X)
        self.m.fit(Z, Z)
        return self

    def score(self, X):
        Z = self.sc.transform(X)
        return ((self.m.predict(Z) - Z) ** 2).mean(axis=1)


def sup_margin(model, X):
    """Exact pre-sigmoid margin (logit) of a fitted supervised model, with no
    probability clipping: XGBoost raw margin, or a manual forward pass of the MLP."""
    from xgboost import XGBClassifier
    from sklearn.neural_network import MLPClassifier
    if isinstance(model, XGBClassifier):
        return model.predict(X, output_margin=True).astype(float)
    est = model[-1] if hasattr(model, "steps") else model
    Z = model[:-1].transform(X) if hasattr(model, "steps") else X
    if isinstance(est, MLPClassifier) and est.activation == "relu":
        h = Z
        for W, b in zip(est.coefs_[:-1], est.intercepts_[:-1]):
            h = np.maximum(h @ W + b, 0.0)
        return (h @ est.coefs_[-1] + est.intercepts_[-1]).ravel()
    raise TypeError(type(est))


def quadrant(X, y, K=3, seeds=10, alpha=0.01):
    """4-D mechanism check: for each held-out family, the supervised margin (XGBoost, MLP)
    against distance from the genuine class (Mahalanobis, autoencoder error), both with
    thresholds at the same genuine-rejection budget. -> results/quadrant.csv"""
    fam = families_k(X, y, K)
    gen_idx, cf_idx = np.where(y == 0)[0], np.where(y == 1)[0]
    rows = []
    for seed in range(seeds):
        g_tr, g_val, g_te = genuine_split(gen_idx, np.random.RandomState(seed))
        maha = be.MahalanobisScorer().fit(X[g_tr])
        ae = be.AutoencoderScorer(seed).fit(X[g_tr])
        t_m = np.quantile(maha.score(X[g_val]), 1 - alpha)
        t_a = np.quantile(ae.score(X[g_val]), 1 - alpha)
        for f in range(K):
            c_tr, c_ho = cf_idx[fam[cf_idx] != f], cf_idx[fam[cf_idx] == f]
            tr = np.concatenate([g_tr, c_tr])
            for name, m in be.supervised_models(seed, names=["XGBoost", "MLP (3x64)"]).items():
                m.fit(X[tr], y[tr])
                # threshold set exactly as in E5 (on probabilities), then mapped to logit space
                t_p = float(np.quantile(be.sup_score(m, X[g_val]).astype(float), 1 - alpha))
                t = np.log(t_p) - np.log1p(-t_p)
                for grp, idx in (("genuine", g_te), ("held-out", c_ho)):
                    rows.append(pd.DataFrame(dict(
                        seed=seed, family=f, model=name, group=grp, idx=idx,
                        sup=sup_margin(m, X[idx]), sup_thr=t,
                        maha=maha.score(X[idx]), maha_thr=t_m,
                        ae=ae.score(X[idx]), ae_thr=t_a)))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv("results/quadrant.csv", index=False)
    h = df[df.group == "held-out"]
    for name, d in h.groupby("model"):
        miss = d[d.sup <= d.sup_thr]
        print(f"{name}: missed {len(miss)}/{len(d)} held-out notes over seeds; of these "
              f"{(miss.maha > miss.maha_thr).mean():.3f} lie outside the 99% Mahalanobis "
              f"region and {(miss.ae > miss.ae_thr).mean():.3f} exceed the autoencoder threshold")


if __name__ == "__main__":
    X, y = be.load_data()
    t0 = time.time()
    if sys.argv[1] == "lofo":
        for K in map(int, sys.argv[2:]):
            df = lofo(X, y, K)
            df.to_csv(f"results/e5x_K{K}.csv", index=False)
            v = df[(df.alpha == 0.01) & df.setting.str.startswith("family")]
            print(f"K={K}:", (v.groupby("model").cdr.mean() * 100).round(1).sort_values().to_dict())
    elif sys.argv[1] == "quadrant":
        quadrant(X, y)
    print(f"[done in {time.time() - t0:.0f}s]")
