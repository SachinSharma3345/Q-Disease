"""
Setup script: Download or generate the Breast Cancer Wisconsin dataset.
Source: sklearn.datasets (mirrors UCI ML Repository dataset #17)
License: CC BY 4.0
"""
import sys
import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = DATASETS_DIR / "breast_cancer_wisconsin.csv"


def setup():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    feature_names = [n.replace(" ", "_") for n in data.feature_names]

    rows = []
    for i, (x, y) in enumerate(zip(data.data, data.target)):
        row = {name: round(float(val), 6) for name, val in zip(feature_names, x)}
        row["diagnosis"] = "B" if y == 1 else "M"  # 1=benign, 0=malignant (sklearn convention)
        rows.append(row)

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset saved: {OUTPUT_PATH}")
    print(f"Rows: {len(rows)}, Features: {len(feature_names)}, Target: diagnosis (B/M)")
    print("Provenance: sklearn.datasets.load_breast_cancer (UCI ML Repository #17)")
    print("License: CC BY 4.0 — for research and educational use only")


if __name__ == "__main__":
    setup()
