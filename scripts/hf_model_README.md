---
license: mit
library_name: scikit-learn
pipeline_tag: tabular-classification
datasets:
  - thanatchon36/beyond99-banknote-detection
tags:
  - counterfeit-detection
  - banknote-authentication
  - anomaly-detection
  - one-class-classification
  - open-set-recognition
  - xgboost
  - lightgbm
  - joblib
---

# Beyond 99%: 18 counterfeit banknote detectors

Ready-to-use detectors from the preprint *Beyond 99%: Open-Set Evaluation Reveals Silent Failures of Counterfeit
Banknote Detectors, and the Case for One-Class Screening* by Thanatchon Chongmankhong
([read online](https://thanatchon36.github.io/beyond99-banknote-detection/paper/) ·
[code](https://github.com/thanatchon36/beyond99-banknote-detection) ·
[dataset](https://huggingface.co/datasets/thanatchon36/beyond99-banknote-detection) ·
DOI [10.5281/zenodo.24049790](https://doi.org/10.5281/zenodo.24049790)).

Ten supervised classifiers, four one-class detectors trained on genuine notes only, and four hybrids, each
trained on the full UCI Banknote Authentication data and calibrated to reject about 1% of genuine notes.
**The paper's recommendation is the one-class `autoencoder` (or `mahalanobis`)**: on counterfeit families
withheld from training, supervised tree ensembles catch 51–61% while the autoencoder catches 99.98%.

*Independent personal research. Does not represent the views of any employer or institution. A demonstration on
four wavelet features from one imaging set-up, not a production banknote authenticator.*

## Usage

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("thanatchon36/beyond99-banknote-detection")
sys.path.insert(0, path)          # the bundles reference classes in banknote_experiments.py
import train_all

det = train_all.load("autoencoder", f"{path}/models")
train_all.predict(det, [[3.6216, 8.6661, -2.8073, -0.4470],    # a genuine note from the data
                        [-1.3971, 3.3191, -1.3927, -1.9948]])  # a counterfeit from the data
# array([0, 1])   0 = accepted as genuine, 1 = flagged as counterfeit
train_all.score(det, [[3.6216, 8.6661, -2.8073, -0.4470]])    # continuous score; > det["threshold"] is flagged
```

Input: the four features `variance, skewness, kurtosis, entropy` of a wavelet-transformed note image, in that
order. From the command line, after `pip install -r requirements.txt`:

```bash
python train_all.py predict --dir models --model hybrid_xgboost_autoencoder --input notes.csv --out scores.csv
```

> **Security.** `.joblib` files are Python pickles: loading one runs code. Load them only if you trust this
> repository, check them against `SHA256SUMS`, or retrain them yourself in seconds with `make models` in the
> code repository (same seed, same data).

## Detectors

Each bundle is a dict with the fitted model (`model`), `kind`, `threshold`, `alpha` and `features`; a hybrid holds
two `components` and flags a note if either exceeds its own α/2 threshold. *Calibration rejection* is the share of
held-out genuine notes each saved detector rejects. The last two columns are the paper's evaluation of the same
architecture under its protocols (E1: repeated 10-fold cross-validation; Table 5: leave-one-family-out, K = 3,
1% budget, 10 seeds), not of these exact files, which were trained on all the data.

{{TABLE}}

— : not part of that experiment.

## Training

Genuine notes (762) are split 80/20 with seed 0 into a training part (609) and a calibration part (153).
Supervised detectors train on the genuine training part plus all 610 counterfeits; one-class detectors on the
genuine training part only. Each threshold is the 99th percentile of the detector's scores on the calibration
notes. `models/manifest.json` records the data checksum, every threshold and the library versions:
{{VERSIONS}}. Other versions of scikit-learn, XGBoost or LightGBM may refuse to unpickle the bundles;
`requirements.txt` pins the tested ones.

## Licence and citation

Code and models: MIT. Training data: UCI Banknote Authentication (Lohweg, 2013,
[doi:10.24432/C55P57](https://doi.org/10.24432/C55P57)), CC BY 4.0.

```bibtex
@misc{chongmankhong2026beyond,
  author = {Chongmankhong, Thanatchon},
  title  = {Beyond 99\%: Open-Set Evaluation Reveals Silent Failures of Counterfeit Banknote Detectors,
            and the Case for One-Class Screening},
  year   = {2026},
  note   = {Preprint},
  doi    = {10.5281/zenodo.24049790},
  url    = {https://thanatchon36.github.io/beyond99-banknote-detection/paper/}
}
```
