import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.preprocessing.pipeline import PreprocessingPipeline, prepare_data

@pytest.fixture
def breast_cancer_df():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['diagnosis'] = pd.Categorical.from_codes(data.target, ['malignant', 'benign'])
    return df

def test_no_leakage(breast_cancer_df):
    config = {'imputation_strategy': 'mean', 'scaling_method': 'standard'}
    result = prepare_data(breast_cancer_df, 'diagnosis', None, [], 0.2, 42, config)
    assert result['X_train'].shape[0] == result['train_size']
    assert result['X_test'].shape[0] == result['test_size']
    assert result['train_size'] + result['test_size'] == len(breast_cancer_df)

def test_stratified_split(breast_cancer_df):
    config = {'imputation_strategy': 'mean', 'scaling_method': 'standard'}
    result = prepare_data(breast_cancer_df, 'diagnosis', None, [], 0.2, 42, config)
    assert len(np.unique(result['y_train'])) == 2
    assert len(np.unique(result['y_test'])) == 2

def test_feature_selection(breast_cancer_df):
    config = {'imputation_strategy': 'mean', 'scaling_method': 'standard',
              'feature_selection': 'kbest', 'n_features': 10}
    result = prepare_data(breast_cancer_df, 'diagnosis', None, [], 0.2, 42, config)
    assert result['X_train'].shape[1] == 10
    assert result['X_test'].shape[1] == 10

def test_pca(breast_cancer_df):
    config = {'imputation_strategy': 'mean', 'scaling_method': 'standard',
              'feature_selection': 'pca', 'n_features': 5}
    result = prepare_data(breast_cancer_df, 'diagnosis', None, [], 0.2, 42, config)
    assert result['X_train'].shape[1] == 5
