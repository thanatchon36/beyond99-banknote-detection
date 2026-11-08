PY ?= $(CURDIR)/.venv/bin/python
K ?= 2 3 4 5 6

.PHONY: setup data experiments lofo quadrant figures models paper web banner arxiv hf test all clean

setup:          ## create venv and install the dependencies (needs uv)
	uv venv .venv --python 3.12
	uv pip install --python .venv/bin/python -r pyproject.toml --extra dev

data:           ## download the UCI data into code/banknote.csv and validate it (762 genuine, 610 counterfeit)
	cd code && $(PY) -c "import banknote_experiments as be; be.load_data()"

experiments:    ## E1-E5: benchmark, scarce data, weak features, noise, unseen families -> code/results/
	cd code && $(PY) banknote_experiments.py

lofo:           ## E6: leave-one-family-out for K = $(K) and 7 budgets -> code/results/e5x_K*.csv
	cd code && $(PY) paper_analysis.py lofo $(K)

quadrant:       ## mechanism analysis (exact logits) -> code/results/quadrant.csv
	cd code && $(PY) paper_analysis.py quadrant

figures:        ## every figure and LaTeX table, with text-collision checks -> paper/figures, paper/tables
	cd code && $(PY) make_figures.py ../paper

models:         ## train, calibrate and save all 18 detectors -> code/models/
	cd code && $(PY) train_all.py train

paper:          ## build paper/paper.pdf (tectonic) and the web reader docs/paper/
	cd paper && tectonic -X compile paper.tex
	$(MAKE) web

web:            ## web reader only -> docs/paper/index.html (pandoc; copies the current paper.pdf)
	$(PY) scripts/paper_web.py

banner:         ## README banner -> docs/assets/banner.png
	$(PY) scripts/make_banner.py

arxiv:          ## arXiv source bundle -> paper/arxiv.tar.gz
	cd paper && tar czf arxiv.tar.gz paper.tex paper.bbl references.bib arxiv.sty figures/*.pdf tables/*.tex

hf:             ## Hugging Face dataset bundle -> hf/ (upload: hf upload thanatchon36/beyond99-banknote-detection hf . --repo-type dataset)
	$(PY) scripts/hf_dataset.py

test:
	$(PY) -m pytest -q

all: data experiments lofo quadrant figures paper

clean:
	rm -rf code/models hf paper/arxiv.tar.gz paper/*.aux paper/*.log paper/*.out paper/*.blg paper/*.fls paper/*.fdb_latexmk
