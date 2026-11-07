from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class ModelType(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    RANDOM_FOREST = "random_forest"
    QUANTUM_KERNEL_SVM = "quantum_kernel_svm"
    VQC = "vqc"

class DatasetUploadResponse(BaseModel):
    dataset_id: str
    name: str
    rows: int
    columns: int
    column_names: List[str]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    duplicates: int
    message: str

class DatasetSummary(BaseModel):
    dataset_id: str
    name: str
    rows: int
    columns: int
    target_column: Optional[str]
    feature_columns: List[str]
    class_distribution: Optional[Dict[str, int]]
    missing_values: Dict[str, int]
    descriptive_stats: Optional[Dict[str, Any]]
    created_at: datetime

class TrainingConfig(BaseModel):
    dataset_id: str
    target_column: str
    feature_columns: Optional[List[str]] = None
    exclude_columns: Optional[List[str]] = []
    model_type: ModelType
    test_size: float = 0.2
    random_seed: int = 42
    imputation_strategy: str = "mean"
    scaling_method: str = "standard"
    feature_selection: Optional[str] = None
    n_features: Optional[int] = None
    model_params: Optional[Dict[str, Any]] = {}
    n_qubits: int = 4
    circuit_depth: int = 2
    feature_map: str = "ZZFeatureMap"
    ansatz: str = "RealAmplitudes"
    optimizer: str = "COBYLA"
    max_iterations: int = 50
    n_training_samples: Optional[int] = 100

class TrainingResponse(BaseModel):
    experiment_id: str
    model_id: str
    model_type: str
    status: str
    metrics: Optional[Dict[str, Any]]
    training_time: Optional[float]
    message: str

class PredictionRequest(BaseModel):
    model_id: str
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction_id: str
    model_id: str
    predicted_class: str
    probability: Optional[float]
    risk_category: Optional[str]
    timestamp: datetime
    disclaimer: str = "This is an experimental research model. Not for clinical use."

class MetricsResponse(BaseModel):
    model_id: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]
    sensitivity: float
    specificity: float
    confusion_matrix: List[List[int]]
    training_time: float
    inference_time: float
    test_size: int
    positive_class: str
    evaluation_source: str = "held-out test set"
