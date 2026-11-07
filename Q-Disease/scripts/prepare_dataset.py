#!/usr/bin/env python3
"""
Prepare the Breast Cancer Wisconsin (Diagnostic) dataset for Q-Disease.
Source: scikit-learn (originally from UCI ML Repository)
License: CC BY 4.0
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

def prepare_breast_cancer():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['diagnosis'] = pd.Categorical.from_codes(data.target, ['malignant', 'benign'])

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'datasets', 'breast_cancer_wisconsin.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Dataset saved to {out_path}")
    print(f"Shape: {df.shape}")
    print(f"Target distribution: {df['diagnosis'].value_counts().to_dict()}")
    print("Provenance: scikit-learn load_breast_cancer(), originally UCI ML Repository")
    print("License: CC BY 4.0")
    return out_path

if __name__ == '__main__':
    prepare_breast_cancer()
