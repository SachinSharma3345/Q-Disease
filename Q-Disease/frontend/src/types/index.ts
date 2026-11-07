// ─── Dataset ─────────────────────────────────────────────────────────────────
// Matches the actual JSON document stored in storage/db/datasets/<id>.json
// and returned by GET /api/datasets and GET /api/datasets/:id

export interface DatasetColumn {
  name: string
  dtype: 'float' | 'integer' | 'boolean' | 'string'
  missing: number
  unique: number
  min?: number
  max?: number
  mean?: number
  std?: number
}

export interface DatasetRecord {
  _id: string
  name: string
  filepath?: string           // internal server path — present but not displayed
  n_rows: number
  n_cols: number
  columns: DatasetColumn[]
  missing_total: number
  duplicates: number
  target_col: string
  exclude_cols?: string[]
  target_distribution?: Record<string, number>
  created_at: string
}

// Alias kept for pages that import Dataset
export type Dataset = DatasetRecord

// ─── Distribution (from GET /api/datasets/:id/distributions) ─────────────────

export interface FeatureDistributions {
  [feature: string]: {
    bins: number[]
    counts: number[]
  }
}

// ─── Model ───────────────────────────────────────────────────────────────────
// Matches storage/db/models/<id>.json

export interface Metrics {
  accuracy: number
  precision: number
  recall: number
  f1_score: number
  roc_auc?: number
  sensitivity?: number
  specificity?: number
  confusion_matrix?: number[][]
  training_time_s?: number
  inference_time_s?: number
  test_size?: number
  positive_class?: string
  evaluation_source?: string
  class_report?: Record<string, Record<string, number>>
}

export interface Explanations {
  method: string
  feature_names: string[]
  importance_values: number[]
  is_quantum: boolean
  shap_available?: boolean
  note?: string
}

export interface ModelRecord {
  _id: string
  model_type: string
  dataset_id: string
  status: string            // 'completed' | 'failed' | 'running' | 'queued'
  metrics?: Metrics
  explanations?: Explanations
  feature_names?: string[]
  label_names?: string[]
  train_samples?: number
  test_samples?: number
  training_time_s?: number
  inference_time_s?: number
  model_params?: Record<string, unknown>
  preproc_config?: Record<string, unknown>
  version?: string
  created_at: string
}

// Alias
export type Model = ModelRecord

// ─── Experiment ──────────────────────────────────────────────────────────────

export interface ExperimentRecord {
  _id: string
  model_id: string
  model_type: string
  dataset_id: string
  status: string
  metrics?: Metrics
  feature_names?: string[]
  training_time_s?: number
  inference_time_s?: number
  model_params?: Record<string, unknown>
  preproc_config?: Record<string, unknown>
  completed_at?: string
  created_at?: string
}

export type Experiment = ExperimentRecord

// ─── Prediction ──────────────────────────────────────────────────────────────

export interface PredictionRecord {
  _id?: string
  model_id: string
  predicted_class: string
  predicted_index?: number
  probabilities?: number[][]    // shape [[p0, p1, ...]]
  timestamp: string
  disclaimer: string
}

export type Prediction = PredictionRecord

// ─── Model types ─────────────────────────────────────────────────────────────

export type ModelType =
  | 'logistic_regression'
  | 'svm'
  | 'random_forest'
  | 'quantum_kernel_svm'
  | 'vqc'
