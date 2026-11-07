"""
Preprocessing pipeline with strict train-only fitting.
Prevents data leakage: all transforms fitted only on training data.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from typing import Any, Dict, List, Optional, Tuple
import joblib
import os
from app.core.config import settings


class PreprocessingPipeline:
    """
    Fit on train, transform train+test.
    Stores all fitted objects for reproducible inference.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.imputer = None
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.label_encoder = None
        self.selected_features: List[str] = []
        self.feature_names_out: List[str] = []
        self.target_mapping: Dict = {}
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame, target_col: str, exclude_cols: List[str] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split, then fit on train and transform both sets.
        Returns X_train, X_test, y_train, y_test
        """
        cfg = self.config
        exclude = set(exclude_cols or [])

        # Separate features and target
        y_raw = df[target_col].values
        feature_cols = [c for c in df.columns if c != target_col and c not in exclude]
        X_raw = df[feature_cols].copy()

        # Encode target
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y_raw)
        self.target_mapping = {str(cls): int(i) for i, cls in enumerate(self.label_encoder.classes_)}

        # Drop purely non-numeric columns that can't be imputed numerically
        non_numeric = [c for c in X_raw.columns if not pd.api.types.is_numeric_dtype(X_raw[c])]
        X_raw = X_raw.drop(columns=non_numeric)
        self.selected_features = list(X_raw.columns)

        # Stratified train/test split
        test_size = float(cfg.get("test_size", 0.2))
        random_state = int(cfg.get("random_state", settings.RANDOM_SEED))
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw.values, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if len(np.unique(y)) > 1 else None,
        )

        # 1. Impute (fit on train only)
        strategy = cfg.get("imputation", "mean")
        self.imputer = SimpleImputer(strategy=strategy)
        X_train = self.imputer.fit_transform(X_train)
        X_test = self.imputer.transform(X_test)

        # 2. Scale (fit on train only)
        scaler_type = cfg.get("scaler", "standard")
        if scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.feature_names_out = list(self.selected_features)

        # 3. Feature selection (fit on train only)
        n_features = cfg.get("n_features")
        method = cfg.get("feature_selection", "none")
        use_pca = bool(cfg.get("use_pca", False))
        # When n_features is set but no explicit method is given, default to
        # SelectKBest(f_classif) so the "Max Features" slider in the UI has
        # an observable effect.  When use_pca is also True we defer to step 4.
        if method == "none" and n_features and int(n_features) < X_train.shape[1] and not use_pca:
            method = "kbest"
        if method != "none" and n_features and int(n_features) < X_train.shape[1]:
            k = min(int(n_features), X_train.shape[1])
            score_func = mutual_info_classif if method == "mutual_info" else f_classif
            self.feature_selector = SelectKBest(score_func=score_func, k=k)
            X_train = self.feature_selector.fit_transform(X_train, y_train)
            X_test = self.feature_selector.transform(X_test)
            mask = self.feature_selector.get_support()
            self.feature_names_out = [f for f, m in zip(self.selected_features, mask) if m]

        # 4. PCA (fit on train only)
        # The frontend sends use_pca=True/False; honour it here.
        # n_pca_components is the lower-level override used by tests / API.
        n_components = cfg.get("n_pca_components")
        if use_pca and not n_components:
            # Use n_features as the component count, capped sensibly.
            n_components = n_features if n_features else X_train.shape[1]
        if n_components and int(n_components) < X_train.shape[1]:
            n_comp = min(int(n_components), X_train.shape[1], X_train.shape[0] - 1)
            self.pca = PCA(n_components=n_comp, random_state=random_state)
            X_train = self.pca.fit_transform(X_train)
            X_test = self.pca.transform(X_test)
            self.feature_names_out = [f"PC{i+1}" for i in range(n_comp)]

        self.is_fitted = True
        return X_train, X_test, y_train, y_test

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted pipeline."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted.")
        # Verify all required features are present — fail loudly rather than
        # silently producing a shape mismatch inside the downstream model.
        missing = [c for c in self.selected_features if c not in X_df.columns]
        if missing:
            raise ValueError(
                f"Input is missing {len(missing)} feature(s) required by the "
                f"fitted pipeline: {missing}.  "
                f"Expected features: {self.selected_features}"
            )
        cols = [c for c in self.selected_features if c in X_df.columns]
        X = X_df[cols].values.astype(float)
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        if self.feature_selector:
            X = self.feature_selector.transform(X)
        if self.pca:
            X = self.pca.transform(X)
        return X

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "PreprocessingPipeline":
        return joblib.load(path)


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: Optional[List[str]],
    drop_cols: List[str],
    test_size: float,
    random_state: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convenience wrapper around PreprocessingPipeline for tests and scripts.

    Maps test-friendly kwargs onto the pipeline config dict and returns a
    plain dict with X_train, X_test, y_train, y_test plus size metadata.

    ``config`` keys recognised: imputation_strategy, scaling_method,
    feature_selection, n_features.
    """
    # Translate config keys used by the test suite to pipeline config keys
    pipe_config: Dict[str, Any] = {
        "test_size": test_size,
        "random_state": random_state,
        "imputation": config.get("imputation_strategy", "mean"),
        "scaler": "standard" if config.get("scaling_method", "standard") == "standard" else "minmax",
    }
    fs = config.get("feature_selection")
    if fs == "kbest":
        pipe_config["feature_selection"] = "kbest"
        pipe_config["n_features"] = config.get("n_features")
    elif fs == "pca":
        pipe_config["feature_selection"] = "none"
        pipe_config["n_pca_components"] = config.get("n_features")

    exclude = list(exclude_cols or []) + list(drop_cols or [])
    pipeline = PreprocessingPipeline(pipe_config)
    X_train, X_test, y_train, y_test = pipeline.fit_transform(df, target_col, exclude)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_names": pipeline.feature_names_out,
        "pipeline": pipeline,
    }