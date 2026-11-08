import json

import numpy as np
import pytest

import train_all

GENUINE = [3.6216, 8.6661, -2.8073, -0.4470]       # first genuine note in the UCI file
COUNTERFEIT = [-1.3971, 3.3191, -1.3927, -1.9948]  # first counterfeit note in the UCI file


@pytest.fixture(scope="module")
def models(tmp_path_factory, data):
    out = tmp_path_factory.mktemp("models")
    train_all.train(alpha=0.01, out=str(out), seed=0)
    return out


def test_all_eighteen_detectors_are_saved(models):
    manifest = json.loads((models / "manifest.json").read_text())
    kinds = [d["kind"] for d in manifest["detectors"]]
    assert len(kinds) == 18
    assert kinds.count("supervised") == 10 and kinds.count("one-class") == 4 and kinds.count("hybrid") == 4
    assert manifest["label_convention"] == "0 = genuine, 1 = counterfeit"


@pytest.mark.parametrize("name", ["autoencoder", "mahalanobis", "hybrid_xgboost_autoencoder"])
def test_readme_example_is_classified_correctly(models, name):
    det = train_all.load(name, str(models))
    assert train_all.predict(det, [GENUINE, COUNTERFEIT]).tolist() == [0, 1]


def test_calibrated_rejection_respects_the_budget(models):
    manifest = json.loads((models / "manifest.json").read_text())
    for d in manifest["detectors"]:
        # k-NN and the single tree have coarse scores, so their realised rate cannot match alpha exactly
        limit = 0.05 if d["name"] in {"kNN (k=5)", "Decision Tree"} else 0.025
        assert d["calibration_genuine_rejected"] <= limit, d["name"]


def test_predict_reads_csv_with_and_without_header(models, tmp_path):
    rows = np.array([GENUINE, COUNTERFEIT])
    plain = tmp_path / "plain.csv"
    np.savetxt(plain, np.c_[rows, [0, 1]], delimiter=",")
    header = tmp_path / "header.csv"
    header.write_text("label,entropy,kurtosis,skewness,variance\n" + "\n".join(
        f"{lab},{r[3]},{r[2]},{r[1]},{r[0]}" for lab, r in zip([0, 1], rows)))
    for path in (plain, header):
        assert np.allclose(train_all.read_features(str(path)), rows)


def test_missing_detector_explains_how_to_train(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_all.py train"):
        train_all.load("autoencoder", str(tmp_path))
