"""The headline numbers in the README are the ones in the paper's Table 5, computed from code/results/."""
from pathlib import Path

import pytest

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
NAME = {"MLP (3×64)": "MLP (3x64)", "Random forest": "Random Forest", "Isolation forest": "Isolation Forest"}


def readme_rows():
    table = README.split("## Results at a glance", 1)[1].split("\n\n")[2]
    for line in table.splitlines()[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        yield NAME.get(cells[0], cells[0]), cells[2:5]


ROWS = list(readme_rows())


def pct(v, shown):
    digits = len(shown.split(".")[1]) if "." in shown else 0
    return round(100 * v, digits)


def test_the_readme_table_lists_nine_detectors():
    assert len(ROWS) == 9


@pytest.mark.parametrize("model,cells", ROWS, ids=[r[0] for r in ROWS])
def test_readme_table_matches_table_5(lofo, model, cells):
    row = lofo.loc[model]
    for shown, col in zip(cells, ["random", "lofo", "worst"]):
        assert float(shown) == pct(row[col], shown), (model, col)


def test_headline_claims_match_table_5(lofo):
    trees = lofo.loc[["XGBoost", "Random Forest", "LightGBM"]]
    assert f"{100 * trees.lofo.min():.0f}–{100 * trees.lofo.max():.0f}%" in README
    assert f"{100 * trees.worst.min():.0f}–{100 * trees.worst.max():.0f}%" in README
    assert f"{100 * lofo.loc['Autoencoder', 'lofo']:.2f}%" in README
    floor = 100 * lofo[lofo.kind == "supervised"].random.min()
    assert f"at least {floor:.1f}%" in README
