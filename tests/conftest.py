"""The experiment scripts live in code/ and read code/banknote.csv and code/results/ by relative path,
so the test session imports them from there and runs inside that directory."""
import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))


@pytest.fixture(scope="session", autouse=True)
def in_code_dir():
    old = os.getcwd()
    os.chdir(CODE)
    yield
    os.chdir(old)


@pytest.fixture(scope="session")
def data():
    import banknote_experiments as be
    return be.load_data()


@pytest.fixture(scope="session")
def mf(tmp_path_factory):
    """make_figures, writing into a temporary paper/ (it reads its output directory from argv at import)."""
    import matplotlib
    matplotlib.use("Agg")
    out = tmp_path_factory.mktemp("paper")
    argv, sys.argv = sys.argv, ["make_figures.py", str(out)]
    try:
        return importlib.import_module("make_figures")
    finally:
        sys.argv = argv


@pytest.fixture(scope="session")
def lofo(mf):
    """Table 5 of the paper: leave-one-family-out summary, K = 3, 1% budget, 10 seeds."""
    return mf.lofo_summary(3).set_index("model")
