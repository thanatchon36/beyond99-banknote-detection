---
license: cc-by-4.0
pretty_name: "Beyond 99%: open-set evaluation of counterfeit banknote detectors"
language:
  - en
  - th
size_categories:
  - 1K<n<10K
task_categories:
  - tabular-classification
tags:
  - counterfeit-detection
  - banknote-authentication
  - open-set-recognition
  - one-class-classification
  - anomaly-detection
  - central-banking
configs:
  - config_name: banknote
    data_files: data/banknote.csv
    default: true
  - config_name: counterfeit_families
    data_files: data/counterfeit_families.csv
  - config_name: lofo_k3
    data_files: results/e5x_K3.csv
  - config_name: lofo_all_k
    data_files: results/e5x_K*.csv
  - config_name: benchmark
    data_files: results/e1_summary.csv
  - config_name: mechanism
    data_files: results/quadrant.csv
---

# Beyond 99%: open-set evaluation of counterfeit banknote detectors

Data and results for the preprint *Beyond 99%: Open-Set Evaluation Reveals Silent Failures of Counterfeit
Banknote Detectors, and the Case for One-Class Screening* by Thanatchon Chongmankhong
([read online](https://thanatchon36.github.io/beyond99-banknote-detection/paper/) ·
[PDF](https://github.com/thanatchon36/beyond99-banknote-detection/blob/main/paper/paper.pdf)).
The paper evaluates counterfeit detectors on **counterfeit families they have never seen**, at a **fixed 1%
genuine-rejection budget**. Under that protocol, tree ensembles that score above 99% on random splits catch
51–61% of unseen-family counterfeits, while a one-class autoencoder trained on genuine notes alone catches
99.98%. Code: <https://github.com/thanatchon36/beyond99-banknote-detection>.

*Independent personal research. Does not represent the views of any employer or institution.*

## Files

| path | content |
|---|---|
| `data/banknote.csv` | UCI Banknote Authentication with a header: `note_id` (0–1371, row order of the UCI file), the four features `variance`, `skewness`, `kurtosis`, `entropy` of a wavelet-transformed image, and `label` (0 = genuine, 762 notes; 1 = counterfeit, 610 notes) |
| `data/data_banknote_authentication.txt` | the original UCI file, unchanged (no header; SHA-256 in `SHA256SUMS`) |
| `data/counterfeit_families.csv` | the counterfeit "families" used by the paper: `family_k2` … `family_k6` give each counterfeit's K-means cluster (standardised features, `n_init=20`, `random_state=0`) and −1 for genuine notes. Family sizes at K = 3: 71, 360, 179 |
| `results/e1_folds.csv`, `e1_summary.csv`, `e1_test.csv` | E1, random-split benchmark: 10 supervised models, repeated stratified 10-fold cross-validation (5 × 10); per fold, summary, and the corrected resampled t-test of the best tree ensemble against the MLP |
| `results/e2_learning_curve.csv` | E2, accuracy and miss rate against 10–640 training notes (30 repetitions) |
| `results/e3_ablation.csv` | E3, a weaker sensor: all four features, without variance, kurtosis and entropy only |
| `results/e4_noise.csv` | E4, Gaussian noise on test features (worn notes, sensor drift) |
| `results/e5_unseen_families.csv`, `e5_summary.csv` | E5, unseen counterfeit families at K = 3: per seed and setting (`family 0..2` withheld, or `random split`), and the summary |
| `results/e5x_K2.csv` … `e5x_K6.csv` | E6, leave-one-family-out for K = 2..6: 10 seeds × (K families + random split) × 18 detectors × 7 budgets `alpha` ∈ {0.2%, 0.5%, 1%, 2%, 5%, 10%, 20%}. `cdr` = counterfeits detected in the held-out family, `grr` = genuine test notes rejected, `auc` = ROC AUC (empty for hybrids). `e5x_K3.csv` at `alpha = 0.01` is Table 5 of the paper |
| `results/quadrant.csv` | mechanism analysis, K = 3, 1% budget: for every genuine test note and every held-out counterfeit (`idx` = `note_id`), the exact margin of XGBoost or the MLP (`sup`, threshold `sup_thr`, logit units), the squared Mahalanobis distance from the genuine class (`maha`, `maha_thr`) and the autoencoder reconstruction error (`ae`, `ae_thr`) |
| `SHA256SUMS` | checksums of every data and result file |

Detectors: supervised (logistic regression, k-NN, SVM with RBF kernel, decision tree, random forest, extra trees,
HistGradientBoosting, XGBoost, LightGBM, MLP 3 × 64), one-class (autoencoder, Mahalanobis, one-class SVM,
isolation forest) and hybrids that flag a note if either component exceeds its own α/2 threshold. Thresholds are
the (1 − α) quantile of each detector's scores on held-out genuine notes. Every number regenerates with
`make all` in the code repository.

## Usage

```python
from datasets import load_dataset

notes = load_dataset("thanatchon36/beyond99-banknote-detection", "banknote", split="train").to_pandas()
lofo = load_dataset("thanatchon36/beyond99-banknote-detection", "lofo_k3", split="train").to_pandas()
t5 = lofo[(lofo.alpha == 0.01) & lofo.setting.str.startswith("family")]
print(t5.groupby("model").cdr.mean().sort_values())   # mean unseen-family detection, Table 5
```

## Source and licence

The banknote data are the UCI Banknote Authentication dataset (Lohweg, 2013,
[doi:10.24432/C55P57](https://doi.org/10.24432/C55P57)), released by the UCI Machine Learning Repository under
CC BY 4.0 and redistributed here unchanged apart from the added header and `note_id`. Images of genuine and
forged banknote-like specimens were taken with an industrial print-inspection camera (400 × 400 pixels, about
660 dpi) and reduced to four wavelet statistics. The family assignments and all results are by Thanatchon
Chongmankhong, also CC BY 4.0. Counterfeit "families" are K-means clusters, a statistical proxy for real
production processes, not labels from the data source.

## Citation

```bibtex
@misc{chongmankhong2026beyond,
  author = {Chongmankhong, Thanatchon},
  title  = {Beyond 99\%: Open-Set Evaluation Reveals Silent Failures of Counterfeit Banknote Detectors,
            and the Case for One-Class Screening},
  year   = {2026},
  note   = {Preprint},
  url    = {https://thanatchon36.github.io/beyond99-banknote-detection/paper/}
}

@misc{lohweg2013banknote,
  author       = {Lohweg, Volker},
  title        = {Banknote Authentication},
  howpublished = {UCI Machine Learning Repository},
  year         = {2013},
  doi          = {10.24432/C55P57}
}
```
