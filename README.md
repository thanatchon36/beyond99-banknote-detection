# Beyond 99%: Open-Set Evaluation of Counterfeit Banknote Detectors

**Thanatchon Chongmankhong** · Preprint, November 2026 · [Paper (PDF)](paper/paper.pdf)

Machine-learning studies of banknote authentication routinely report 98–100% accuracy. Those numbers come from random train–test splits in which every counterfeit production process ("family") appears on both sides of the split. This repository evaluates detectors the way they fail in practice: on **counterfeit families they have never seen**, at a **fixed 1% genuine-rejection budget**. Under that protocol, tree ensembles that score above 99% catch only about half of unseen-family counterfeits, while a one-class autoencoder trained on genuine notes alone catches 99.98%.

![Random splits hide silent failures on unseen counterfeit families](paper/figures/fig1_teaser.png)

## Key findings

- **The standard benchmark is saturated.** Under random splits every supervised model catches at least 98.8% of counterfeits, and the best tree ensemble is statistically indistinguishable from an MLP (corrected resampled t-test, p = 0.24).
- **Unseen families break supervised detectors.** XGBoost, random forest and LightGBM detect 51–61% of unseen-family counterfeits on average and only 29–39% of the hardest family.
- **One-class screening closes the gap.** An autoencoder trained on genuine notes only detects 99.98% (95% CI 99.94–100) at about 1.7% genuine rejection.
- **The failure is extrapolation, not overlap.** Of the counterfeits XGBoost accepted, 98.9% lie outside the region holding 99% of genuine notes.
- **The result is robust.** It holds for every family granularity from K = 2 to 6 and at every genuine-rejection budget tested (0.2–20%).

## Results at a glance

Counterfeits detected (%) at a 1% genuine-rejection budget, K = 3 counterfeit families, 10 seeds.

| Detector | Paradigm | Random split | Unseen family, mean | Unseen family, worst |
|---|---|---:|---:|---:|
| SVM (RBF) | Supervised, classical | 100.0 | 96.1 | 88.9 |
| MLP (3×64) | Supervised, neural | 100.0 | 84.4 | 67.8 |
| XGBoost | Supervised, tree | 99.6 | 61.0 | 35.7 |
| Random forest | Supervised, tree | 99.8 | 57.6 | 38.8 |
| LightGBM | Supervised, tree | 99.2 | 51.0 | 29.3 |
| Autoencoder | One-class, neural | 100.0 | 99.98 | 99.9 |
| Mahalanobis | One-class, classical | 98.3 | 98.8 | 97.3 |
| Isolation forest | One-class, tree | 34.3 | 45.7 | 4.7 |
| XGBoost + Autoencoder | Hybrid | 100.0 | 99.8 | 99.3 |

All 18 detectors, with confidence intervals, are in Table 5 of the paper and in `code/results/`.

## Repository structure

```
beyond99-banknote-detection/
├── code/
│   ├── banknote_experiments.py   # data, 14 detectors, experiments E1–E5
│   ├── paper_analysis.py         # E6: leave-one-family-out for K = 2..6; mechanism analysis
│   ├── make_figures.py           # every figure and LaTeX table, with text-collision checks
│   ├── train_all.py              # train, save and use all 18 detectors
│   └── results/                  # outputs of the runs reported in the paper (CSV)
├── paper/
│   ├── paper.pdf                 # compiled preprint
│   ├── paper.tex, references.bib, paper.bbl, arxiv.sty
│   ├── figures/                  # vector PDFs and PNG previews
│   └── tables/                   # generated LaTeX tables
├── reproduce.sh                  # regenerate everything end to end
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Quick start

```bash
git clone https://github.com/<your-username>/beyond99-banknote-detection.git
cd beyond99-banknote-detection
pip install -r requirements.txt
./reproduce.sh            # about 10–15 minutes on one CPU core
```

`reproduce.sh` runs the four steps below and rebuilds the PDF if `latexmk` is installed.

```bash
cd code
python banknote_experiments.py              # E1–E5: benchmark, scarce data, weak features, noise, unseen families
python paper_analysis.py lofo 2 3 4 5 6     # E6: leave-one-family-out for K = 2..6 and 7 budgets
python paper_analysis.py quadrant           # mechanism analysis (exact logits)
python make_figures.py ../paper             # figures/ and tables/
```

The data are downloaded automatically on first run (UCI repository, with a GitHub mirror as fallback) and validated against the expected 762 genuine and 610 counterfeit notes.

## Train and use the detectors

`train_all.py` trains all 18 detectors (10 supervised, 4 one-class, 4 hybrids) on the full data, calibrates each threshold on held-out genuine notes, and saves ready-to-use bundles.

```bash
cd code
python train_all.py train                   # -> models/*.joblib and models/manifest.json (seconds)
python train_all.py list
python train_all.py predict --model autoencoder --input notes.csv --out scores.csv
python train_all.py predict --model hybrid_xgboost_autoencoder --input notes.csv
```

The input CSV needs the four features `variance, skewness, kurtosis, entropy`, with or without a header row; extra columns are ignored. From Python:

```python
import train_all
det = train_all.load("autoencoder")
train_all.predict(det, [[3.6216, 8.6661, -2.8073, -0.4470],    # a genuine note from the data
                        [-1.3971, 3.3191, -1.3927, -1.9948]])  # a counterfeit from the data
# array([0, 1])   0 = accepted as genuine, 1 = flagged as counterfeit
```

Notes on the saved detectors:

- Genuine notes are split 80/20 into training and calibration parts. Supervised detectors train on the genuine training part plus all counterfeits; one-class detectors on genuine notes only.
- Each threshold is the 99th percentile of calibration scores (`--alpha 0.01`). Hybrids flag a note if either component exceeds its own α/2 threshold.
- k-NN and the single decision tree produce coarse scores, so their realised rejection rate cannot match α exactly.
- These bundles are for demonstration and downstream use. Detector evaluation is in `banknote_experiments.py` and `paper_analysis.py`.

## Reproducibility

- Every random choice is seeded. In the tested environment, rerunning an experiment reproduces every reported metric in `code/results/` exactly; only the timing columns of E1 differ between machines.
- Tested with Python 3.12.3, NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1, scikit-learn 1.9.1, XGBoost 3.4.1, LightGBM 4.7.0, Matplotlib 3.11.2 and joblib 1.6.0. Other versions may change results slightly.
- `make_figures.py` checks each figure before saving it: no two text elements may overlap, and no annotation may cover data points.

## Data

UCI Banknote Authentication (Lohweg, 2013): 1,372 specimens described by four statistics of a wavelet-transformed image. The dataset is not redistributed here; please cite the original source. Label convention: 0 = genuine, 1 = counterfeit.

## Limitations

- The UCI specimens come from one imaging set-up and four features, without infrared, ultraviolet or magnetic information.
- Counterfeit "families" are K-means clusters, a statistical proxy for real production processes; the analysis is repeated for K = 2–6 to reduce this concern.
- "Deep" models here are small multilayer networks. Convolutional or transformer models on multispectral images are part of the proposed follow-up study.

## Citation

```bibtex
@misc{chongmankhong2026beyond,
  title  = {Beyond 99\%: Open-Set Evaluation Reveals Silent Failures of Counterfeit
            Banknote Detectors, and the Case for One-Class Screening},
  author = {Chongmankhong, Thanatchon},
  year   = {2026},
  note   = {Preprint}
}
```

## License

Code is released under the [MIT License](LICENSE). The paper text and figures in `paper/` are © 2026 Thanatchon Chongmankhong.

## สรุปภาษาไทย

งานวิจัยนี้ทดสอบโมเดล AI ตรวจจับธนบัตรปลอมกับ “ของปลอมตระกูลใหม่” ที่โมเดลไม่เคยเห็นตอนเทรน แทนการสุ่มแบ่งข้อมูลแบบที่งานส่วนใหญ่ใช้ ผลคือโมเดลต้นไม้ (XGBoost, Random Forest, LightGBM) ที่ได้คะแนนเกิน 99% ในการทดสอบปกติ จับของปลอมตระกูลใหม่ได้เพียง 51–61% ขณะที่ autoencoder ซึ่งเรียนจากธนบัตรจริงอย่างเดียวจับได้ 99.98% ทุกตัวเลข ตาราง และรูปในบทความรันซ้ำได้ด้วย `./reproduce.sh`
