import numpy as np
import pandas as pd
import pytest

import banknote_experiments as be


def test_uci_data_has_the_expected_notes(data):
    X, y = data
    assert X.shape == (1372, 4)
    assert np.bincount(y).tolist() == [762, 610]  # 0 = genuine, 1 = counterfeit
    assert np.isfinite(X).all()


def test_load_data_rejects_a_wrong_file(tmp_path):
    bad = tmp_path / "banknote.csv"
    pd.DataFrame([[0.0, 0.0, 0.0, 0.0, 0]] * 10).to_csv(bad, header=False, index=False)
    with pytest.raises(ValueError, match="unexpected data"):
        be.load_data(str(bad))


def test_counterfeit_families_partition_the_counterfeits(data):
    from paper_analysis import families_k
    X, y = data
    fam = families_k(X, y, 3)
    cf = fam[y == 1]
    assert set(np.unique(cf)) == {0, 1, 2}
