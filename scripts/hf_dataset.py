#!/usr/bin/env python3
"""Assemble the Hugging Face dataset thanatchon36/beyond99-banknote-detection in hf/ (git-ignored).

    python scripts/hf_dataset.py                 # (make hf) build hf/
    hf upload thanatchon36/beyond99-banknote-detection hf . --repo-type dataset

Contents: the UCI Banknote Authentication data (CC BY 4.0, Lohweg 2013) with a header, the counterfeit
family assignments for K = 2..6 used by the paper, every result file in code/results/, the dataset
card (scripts/hf_README.md), the licence and SHA256SUMS.
"""
import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE, OUT = ROOT / "code", ROOT / "hf"
sys.path.insert(0, str(CODE))

import banknote_experiments as be  # noqa: E402
from paper_analysis import families_k  # noqa: E402

UCI_SHA256 = "d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9"


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "data").mkdir(parents=True)
    raw = CODE / "banknote.csv"
    X, y = be.load_data(str(raw))  # downloads and validates on first use
    if hashlib.sha256(raw.read_bytes()).hexdigest() != UCI_SHA256:
        raise SystemExit(f"{raw} differs from the UCI file the paper used")
    shutil.copyfile(raw, OUT / "data" / "data_banknote_authentication.txt")

    df = pd.DataFrame(X, columns=be.FEATURES)
    df.insert(0, "note_id", range(len(df)))
    df["label"] = y
    df.to_csv(OUT / "data" / "banknote.csv", index=False)

    fam = df[["note_id", "label"]].copy()
    for k in range(2, 7):
        fam[f"family_k{k}"] = families_k(X, y, k)
    fam.to_csv(OUT / "data" / "counterfeit_families.csv", index=False)

    shutil.copytree(CODE / "results", OUT / "results")
    shutil.copyfile(ROOT / "scripts" / "hf_README.md", OUT / "README.md")
    shutil.copyfile(ROOT / "scripts" / "hf_LICENSE", OUT / "LICENSE")

    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in {"README.md", "SHA256SUMS"})
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT).as_posix()}" for p in files]
    (OUT / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(f"wrote {len(files) + 2} files to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
