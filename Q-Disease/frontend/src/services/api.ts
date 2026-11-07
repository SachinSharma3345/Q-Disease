import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 300000,
})

api.interceptors.response.use(
  res => res,
  err => {
    // Flask backend returns {success:false, error:"..."} — FastAPI uses "detail"
    const msg =
      err.response?.data?.error ||
      err.response?.data?.detail ||
      err.message ||
      'Network error'
    return Promise.reject(new Error(msg))
  }
)

export const healthApi = {
  check: () => api.get('/health'),
  stats: () => api.get('/stats'),
}

export const datasetsApi = {
  list: () => api.get('/datasets'),
  upload: (file: File, targetCol?: string, name?: string) => {
    const form = new FormData()
    form.append('file', file)
    if (targetCol) form.append('target_col', targetCol)
    if (name) form.append('name', name)
    return api.post('/datasets/upload', form)
  },
  /** Step 2 of upload: register an already-saved file with its target column */
  register: (data: { filepath: string; name: string; target_col: string; exclude_cols?: string[] }) =>
    api.post('/datasets/register', data),
  loadDemo: () => api.post('/demo/load'),
  get: (id: string) => api.get(`/datasets/${id}`),
  delete: (id: string) => api.delete(`/datasets/${id}`),
  summary: (id: string) => api.get(`/datasets/${id}/summary`),
  distributions: (id: string) => api.get(`/datasets/${id}/distributions`),
  correlation: (id: string) => api.get(`/datasets/${id}/correlation`),
}

export const modelsApi = {
  list: () => api.get('/models'),
  get: (id: string) => api.get(`/models/${id}`),
  train: (config: Record<string, unknown>) => api.post('/models/train', config),
  trainQuantum: (config: Record<string, unknown>) => api.post('/models/quantum/train', config),
  metrics: (id: string) => api.get(`/models/${id}/metrics`),
  explanations: (id: string) => api.get(`/models/${id}/explanations`),
  jobStatus: (jobId: string) => api.get(`/models/jobs/${jobId}`),
}

export const experimentsApi = {
  list: () => api.get('/experiments'),
  get: (id: string) => api.get(`/experiments/${id}`),
  comparison: (id: string) => api.get(`/experiments/${id}/comparison`),
}

export const predictionsApi = {
  predict: (data: Record<string, unknown>) => api.post('/predict', data),
  batch: (data: Record<string, unknown>) => api.post('/predict/batch', data),
  history: () => api.get('/predictions'),
}

export const reportsApi = {
  json: (id: string) => api.get(`/reports/${id}/json`, { responseType: 'blob' }),
  csv: (id: string) => api.get(`/reports/${id}/csv`, { responseType: 'blob' }),
}

export default api
