# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Zenodo DOI [10.5281/zenodo.24049790](https://doi.org/10.5281/zenodo.24049790) (all versions; v0.1.0 is 10.5281/zenodo.24049791) in the README
  badge, `CITATION.cff`, the web reader and the Hugging Face dataset card.
- LinkedIn launch poster, `docs/assets/linkedin_poster.png` (1440 × 1800), drawn from `code/results/` by
  `make poster`.

## [0.1.0] - 2026-11-08

### Added
- Experiments E1–E5 (`code/banknote_experiments.py`): random-split benchmark, scarce data, weaker
  features, noise, and unseen counterfeit families at a fixed 1% genuine-rejection budget.
- E6 (`code/paper_analysis.py`): leave-one-family-out for K = 2–6 and seven budgets, and the mechanism
  analysis with exact logits.
- Every figure and LaTeX table of the paper (`code/make_figures.py`), with text-collision checks.
- `code/train_all.py`: train, calibrate, save and use all 18 detectors (10 supervised, 4 one-class,
  4 hybrids).
- The preprint (`paper/paper.pdf`) with its LaTeX source in arXiv layout, and a web reader, `docs/paper/`,
  served by GitHub Pages at <https://thanatchon36.github.io/beyond99-banknote-detection/paper/> and built
  from `paper/paper.tex` by `make web` (pandoc). Its results strip is computed from `code/results/`.
- Dataset on Hugging Face, [`thanatchon36/beyond99-banknote-detection`](https://huggingface.co/datasets/thanatchon36/beyond99-banknote-detection):
  the UCI data (CC BY 4.0) with a header, the counterfeit family assignments for K = 2–6 and every result
  file, with checksums; built by `make hf`.
- `Makefile` with one target per step, `pyproject.toml`, `reproduce.sh`, and a test suite: data validation,
  the 18 saved detectors, the README numbers against Table 5, the figure collision check, and the paper's
  tables regenerating byte for byte from `code/results/`. CI on GitHub Actions.
- README banner drawn from the data (`make banner`), `CITATION.cff` and `.zenodo.json`.
