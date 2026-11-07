
import { useEffect, useState } from 'react'
import { modelsApi, predictionsApi } from '../services/api'
import { Zap, AlertCircle, CheckCircle, Info, Download } from 'lucide-react'

interface ModelOption {
  _id: string
  model_type: string
  dataset_id: string
  feature_names?: string[]
  metrics?: { accuracy?: number }
  created_at: string
}

interface PredResult {
  predicted_class: string
  predicted_index?: number
  probabilities?: number[][] | number[]
  risk_category?: string
  disclaimer?: string
}

function getProbability(result: PredResult): number | undefined {
  const probabilities = result.probabilities

  if (!probabilities || probabilities.length === 0) {
    return undefined
  }

  const index = result.predicted_index ?? 1

  let probability: number | undefined

  if (Array.isArray(probabilities[0])) {
    probability = (probabilities[0] as number[])[index]
  } else {
    probability = (probabilities as number[])[index]
  }

  if (
    probability === undefined ||
    !Number.isFinite(probability) ||
    probability < 0 ||
    probability > 1
  ) {
    return undefined
  }

  return probability
}

function formatPredictedClass(value: string): string {
  const label = String(value ?? '').trim()
  const normalized = label.toLowerCase()

  const knownLabels: Record<string, string> = {
    pos: 'Positive',
    neg: 'Negative',
    positive: 'Positive',
    negative: 'Negative',
  }

  return knownLabels[normalized] ?? (label || 'Unknown')
}

const RISK_COLOR: Record<string, string> = {
  Low: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  Moderate: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
  High: 'text-red-400 bg-red-500/10 border-red-500/20',
  Unknown: 'text-slate-400 bg-slate-500/10 border-slate-500/20',
}

function getRiskColor(risk?: string): string {
  return RISK_COLOR[risk || 'Unknown'] || RISK_COLOR.Unknown
}

const BREAST_CANCER_FEATURES = [
  'mean radius',
  'mean texture',
  'mean perimeter',
  'mean area',
  'mean smoothness',
  'mean compactness',
  'mean concavity',
  'mean concave points',
  'mean symmetry',
  'mean fractal dimension',
  'radius error',
  'texture error',
  'perimeter error',
  'area error',
  'smoothness error',
  'compactness error',
  'concavity error',
  'concave points error',
  'symmetry error',
  'fractal dimension error',
  'worst radius',
  'worst texture',
  'worst perimeter',
  'worst area',
  'worst smoothness',
  'worst compactness',
  'worst concavity',
  'worst concave points',
  'worst symmetry',
  'worst fractal dimension',
]

export default function Predictions() {
  const [models, setModels] = useState<ModelOption[]>([])
  const [selectedModel, setSelectedModel] = useState<ModelOption | null>(null)
  const [features, setFeatures] = useState<Record<string, string>>({})
  const [result, setResult] = useState<PredResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [batchText, setBatchText] = useState('')
  const [batchResults, setBatchResults] = useState<PredResult[]>([])
  const [mode, setMode] = useState<'single' | 'batch'>('single')

  useEffect(() => {
    modelsApi.list()
      .then(r => {
        const ms = r.data?.data || r.data?.models || []
        const done = ms.filter(
          (m: any) =>
            !m.status ||
            m.status === 'completed' ||
            m.status === 'done'
        )

        setModels(done)

        if (done.length > 0) {
          setSelectedModel(done[0])
          initFeatures(done[0])
        }
      })
      .catch(() => setError('Failed to load models'))
  }, [])

  function initFeatures(model: ModelOption) {
  const modelFeatures = model.feature_names

  // Use the complete feature list for this dataset
  // when the saved model metadata is incomplete.
  const names =
    modelFeatures && modelFeatures.length === 30
      ? modelFeatures
      : BREAST_CANCER_FEATURES

  setFeatures(
    Object.fromEntries(names.map(name => [name, '0']))
  )

  setResult(null)
  setBatchResults([])
  setError('')
}

  function handleModelChange(id: string) {
    const model = models.find(item => item._id === id)

    if (model) {
      setSelectedModel(model)
      initFeatures(model)
    }
  }

  async function handlePredict() {
    if (!selectedModel) return

    setLoading(true)
    setError('')
    setResult(null)

    const numFeatures: Record<string, number> = {}

    for (const [key, value] of Object.entries(features)) {
      const parsed = Number.parseFloat(value)
      numFeatures[key] = Number.isFinite(parsed) ? parsed : 0
    }

    try {
      const response = await predictionsApi.predict({
        model_id: selectedModel._id,
        features: numFeatures,
      })

      const payload = response.data?.data || response.data

      if (!payload || !payload.predicted_class) {
        throw new Error('The backend returned an invalid prediction response.')
      }

      setResult(payload)
    } catch (e: any) {
      setError(
        e.response?.data?.error ||
        e.response?.data?.message ||
        e.message ||
        'Prediction failed. Please try again.'
      )
    } finally {
      setLoading(false)
    }
  }

  async function handleBatchPredict() {
    if (!selectedModel || !batchText.trim()) return

    setLoading(true)
    setError('')
    setBatchResults([])

    try {
      const lines = batchText
        .trim()
        .split(/\r?\n/)
        .filter(Boolean)

      if (lines.length < 2) {
        throw new Error(
          'Enter a CSV header followed by at least one data row.'
        )
      }

      const header = lines[0].split(',').map(value => value.trim())

      const rows = lines.slice(1).map(line => {
        const values = line.split(',')
        const row: Record<string, number> = {}

        header.forEach((name, index) => {
          const parsed = Number.parseFloat(values[index] ?? '')
          row[name] = Number.isFinite(parsed) ? parsed : 0
        })

        return row
      })

      const response = await predictionsApi.batch({
        model_id: selectedModel._id,
        rows,
      })

      const payload = response.data?.data || response.data?.results || response.data

      const results = Array.isArray(payload)
        ? payload
        : payload?.results

      if (!Array.isArray(results)) {
        throw new Error('The backend returned an invalid batch response.')
      }

      setBatchResults(results)
    } catch (e: any) {
      setError(
        e.response?.data?.error ||
        e.response?.data?.message ||
        e.message ||
        'Batch prediction failed.'
      )
    } finally {
      setLoading(false)
    }
  }

  function downloadBatch() {
    if (!batchResults.length) return

    const rows = batchResults.map((item, index) => {
      const probability = getProbability(item)

      return [
        index + 1,
        item.predicted_class,
        probability?.toFixed(4) ?? '',
        item.risk_category ?? '',
      ].map(value => `"${String(value).replace(/"/g, '""')}"`).join(',')
    })

    const csv = [
      'row,predicted_class,predicted_class_probability,risk_category',
      ...rows,
    ].join('\n')

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')

    anchor.href = url
    anchor.download = 'predictions.csv'
    anchor.click()

    URL.revokeObjectURL(url)
  }

  // The current Breast Cancer Wisconsin pipeline expects all 30 features.
  // Some saved model records expose only 10 feature names, so use the
  // dataset's complete feature list for this model's prediction form.
  const displayFeatures =
    selectedModel?.feature_names?.length === 30
      ? selectedModel.feature_names
      : selectedModel
        ? BREAST_CANCER_FEATURES
        : []

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white">
          Prediction Studio
        </h1>
        <p className="text-slate-400 mt-1">
          Generate predictions using trained models
        </p>
      </div>

      {error && (
        <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-red-400 text-sm">
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      )}

      {models.length === 0 ? (
        <div className="card text-center py-12">
          <Zap size={32} className="text-slate-600 mx-auto mb-3" />
          <div className="text-slate-400">
            No trained models yet.{' '}
            <a href="/training" className="text-blue-400 underline">
              Train one first
            </a>.
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            {/* Model Selector */}
            <div className="card">
              <h2 className="font-semibold text-white mb-3">
                Select Model
              </h2>

              <select
                className="select"
                value={selectedModel?._id || ''}
                onChange={event => handleModelChange(event.target.value)}
              >
                {models.map(model => (
                  <option key={model._id} value={model._id}>
                    {model.model_type
                      .replace(/_/g, ' ')
                      .replace(/\b\w/g, character => character.toUpperCase())}
                    {' — '}
                    {model._id.slice(0, 8)}
                    {model.metrics?.accuracy != null
                      ? ` (acc: ${(model.metrics.accuracy * 100).toFixed(1)}%)`
                      : ''}
                  </option>
                ))}
              </select>
            </div>

            {/* Prediction Mode */}
            <div className="card">
              <div className="flex gap-2 mb-5">
                <button
                  onClick={() => setMode('single')}
                  className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
                    mode === 'single'
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-700 text-slate-300'
                  }`}
                >
                  Single
                </button>

                <button
                  onClick={() => setMode('batch')}
                  className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
                    mode === 'batch'
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-700 text-slate-300'
                  }`}
                >
                  Batch CSV
                </button>
              </div>

              {mode === 'single' ? (
                <>
                  {displayFeatures.length > 0 ? (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-h-96 overflow-y-auto pr-1">
                      {displayFeatures.map(featureName => (
                        <div key={featureName}>
                          <label
                            className="label text-xs truncate"
                            title={featureName}
                          >
                            {featureName}
                          </label>

                          <input
                            type="number"
                            step="any"
                            className="input text-sm"
                            value={features[featureName] ?? '0'}
                            onChange={event =>
                              setFeatures(previous => ({
                                ...previous,
                                [featureName]: event.target.value,
                              }))
                            }
                          />
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-slate-400 text-sm">
                      No feature information available for this model.
                    </div>
                  )}

                  <button
                    onClick={handlePredict}
                    disabled={loading || !selectedModel}
                    className="btn-primary mt-4 w-full flex items-center justify-center gap-2 disabled:opacity-50"
                  >
                    <Zap size={16} />
                    {loading ? 'Predicting...' : 'Predict'}
                  </button>
                </>
              ) : (
                <>
                  <p className="text-xs text-slate-400 mb-2">
                    Paste CSV data with a header row. Column names must match
                    the model features.
                  </p>

                  <textarea
                    rows={8}
                    className="input font-mono text-xs resize-none"
                    placeholder={'age,glucose,bmi\n45,130,28.5'}
                    value={batchText}
                    onChange={event => setBatchText(event.target.value)}
                  />

                  <button
                    onClick={handleBatchPredict}
                    disabled={loading || !batchText.trim()}
                    className="btn-primary mt-3 w-full flex items-center justify-center gap-2 disabled:opacity-50"
                  >
                    <Zap size={16} />
                    {loading ? 'Predicting...' : 'Run Batch Predict'}
                  </button>
                </>
              )}
            </div>

            {/* Batch Results */}
            {batchResults.length > 0 && (
              <div className="card">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="font-semibold text-white">
                    Batch Results ({batchResults.length})
                  </h2>

                  <button
                    onClick={downloadBatch}
                    className="btn-secondary flex items-center gap-2 text-sm py-1 px-3"
                  >
                    <Download size={14} />
                    CSV
                  </button>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-700/50">
                        <th className="text-left pb-2 text-slate-400">#</th>
                        <th className="text-left pb-2 text-slate-400">
                          Prediction
                        </th>
                        <th className="text-left pb-2 text-slate-400">
                          Probability
                        </th>
                        <th className="text-left pb-2 text-slate-400">Risk</th>
                      </tr>
                    </thead>

                    <tbody className="divide-y divide-slate-700/30">
                      {batchResults.map((item, index) => {
                        const probability = getProbability(item)

                        return (
                          <tr key={index}>
                            <td className="py-2 text-slate-500">
                              {index + 1}
                            </td>

                            <td className="py-2 text-white font-medium">
                              {formatPredictedClass(item.predicted_class)}
                            </td>

                            <td className="py-2 font-mono text-blue-400">
                              {probability !== undefined
                                ? `${(probability * 100).toFixed(1)}%`
                                : '—'}
                            </td>

                            <td className="py-2">
                              <span
                                className={`text-xs px-2 py-0.5 rounded-full border ${getRiskColor(item.risk_category)}`}
                              >
                                {item.risk_category || 'Unknown'}
                              </span>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>

          {/* Result Panel */}
          <div>
            <div className="card sticky top-6">
              <h2 className="font-semibold text-white mb-4 flex items-center gap-2">
                <Zap size={16} className="text-blue-400" />
                Result
              </h2>

              {result ? (
                <div className="space-y-4">
                  <div className="text-center py-4">
                    <div className="text-3xl font-bold text-white mb-1">
                      {formatPredictedClass(result.predicted_class)}
                    </div>

                    <div className="text-slate-400 text-sm">
                      Predicted Class
                    </div>

                    <div className="text-xs text-slate-500 mt-1">
                      Model label: {result.predicted_class}
                    </div>
                  </div>

                  {(() => {
                    const probability = getProbability(result)

                    if (probability === undefined) return null

                    return (
                      <div>
                        <div className="flex justify-between gap-3 text-sm mb-1">
                          <span className="text-slate-400">
                            Predicted-class probability
                          </span>

                          <span className="text-white font-mono shrink-0">
                            {(probability * 100).toFixed(1)}%
                          </span>
                        </div>

                        <div
                          className="h-2 bg-slate-700 rounded-full overflow-hidden"
                          role="progressbar"
                          aria-label="Predicted-class probability"
                          aria-valuenow={probability * 100}
                          aria-valuemin={0}
                          aria-valuemax={100}
                        >
                          <div
                            className="h-full bg-gradient-to-r from-blue-500 to-violet-500 rounded-full transition-all"
                            style={{ width: `${probability * 100}%` }}
                          />
                        </div>
                      </div>
                    )
                  })()}

                  {result.risk_category && (
                    <div
                      className={`text-center px-3 py-2 rounded-lg border ${getRiskColor(result.risk_category)}`}
                    >
                      <div className="text-sm font-semibold">
                        {result.risk_category}
                      </div>

                      <div className="text-xs opacity-80 mt-1">
                        Application-defined category
                      </div>
                    </div>
                  )}

                  <div className="flex items-start gap-2 bg-amber-500/10 border border-amber-500/20 rounded-lg p-3 text-xs text-amber-300">
                    <Info size={14} className="mt-0.5 shrink-0" />

                    <span>
                      {result.disclaimer ||
                        'Experimental research output only. Not a clinical diagnosis.'}
                    </span>
                  </div>

                  <CheckCircle
                    size={16}
                    className="text-emerald-400 mx-auto"
                  />
                </div>
              ) : (
                <div className="text-center py-8 text-slate-500">
                  <Zap size={32} className="mx-auto mb-2 opacity-30" />

                  <div className="text-sm">
                    Enter feature values and click Predict
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
