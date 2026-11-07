import { useEffect, useState } from 'react'
import { experimentsApi, modelsApi } from '../services/api'
import { GitCompare, Brain, Zap, AlertCircle, TrendingUp } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, Radar } from 'recharts'

interface ModelSummary {
  model_id: string
  model_type: string
  metrics?: Record<string, number>
  training_time_s?: number
  inference_time_s?: number
  train_samples?: number
  test_samples?: number
  created_at?: string
}

const METRIC_LABELS: Record<string, string> = {
  accuracy: 'Accuracy',
  precision: 'Precision',
  recall: 'Recall',
  f1_score: 'F1 Score',
  roc_auc: 'ROC-AUC',
  sensitivity: 'Sensitivity',
  specificity: 'Specificity',
}

const TYPE_COLOR: Record<string, string> = {
  logistic_regression: '#3b82f6',
  svm: '#06b6d4',
  random_forest: '#10b981',
  quantum_kernel_svm: '#8b5cf6',
  vqc: '#a855f7',
}

function fmt(v: number | undefined, pct = true) {
  if (v === undefined || v === null) return '—'
  return pct ? `${(v * 100).toFixed(1)}%` : v.toFixed(3)
}

export default function Comparison() {
  const [models, setModels] = useState<ModelSummary[]>([])
  const [selected, setSelected] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    modelsApi.list()
      .then(r => {
        const ms: ModelSummary[] = (r.data?.data || r.data?.models || [])
          .filter((m: any) => !m.status || m.status === 'completed' || m.status === 'done')
          .map((m: any) => ({
            model_id: m._id,
            model_type: m.model_type,
            metrics: m.metrics,
            training_time_s: m.training_time_s,
            inference_time_s: m.inference_time_s,
            train_samples: m.train_samples,
            test_samples: m.test_samples,
            created_at: m.created_at,
          }))
        setModels(ms)
        setSelected(ms.slice(0, 4).map(m => m.model_id))
      })
      .catch(() => setError('Failed to load models'))
      .finally(() => setLoading(false))
  }, [])

  const displayed = models.filter(m => selected.includes(m.model_id))

  const barData = Object.keys(METRIC_LABELS)
    .filter(k => displayed.some(m => m.metrics?.[k] !== undefined))
    .map(k => {
      const obj: Record<string, unknown> = { metric: METRIC_LABELS[k] }
      displayed.forEach(m => { obj[m.model_type + '_' + m.model_id.slice(0, 4)] = m.metrics?.[k] ? +(m.metrics[k] * 100).toFixed(2) : undefined })
      return obj
    })

  const radarData = Object.keys(METRIC_LABELS).map(k => {
    const obj: Record<string, unknown> = { subject: METRIC_LABELS[k], fullMark: 100 }
    displayed.forEach(m => { obj[m.model_id.slice(0, 8)] = m.metrics?.[k] ? +(m.metrics[k] * 100).toFixed(2) : 0 })
    return obj
  })

  const labelForModel = (m: ModelSummary) =>
    m.model_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) + ' ' + m.model_id.slice(0, 4)

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white">Model Comparison</h1>
        <p className="text-slate-400 mt-1">Compare classical and quantum models on the same held-out test split</p>
      </div>

      {error && (
        <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-red-400 text-sm">
          <AlertCircle size={16} /> {error}
        </div>
      )}

      {loading ? (
        <div className="card text-center py-12 text-slate-400">Loading models...</div>
      ) : models.length === 0 ? (
        <div className="card text-center py-12">
          <GitCompare size={32} className="text-slate-600 mx-auto mb-3" />
          <div className="text-slate-400">No trained models yet. <a href="/training" className="text-blue-400 underline">Train some models</a> to compare.</div>
        </div>
      ) : (
        <>
          {/* Model Selector */}
          <div className="card">
            <h2 className="font-semibold text-white mb-4">Select Models to Compare</h2>
            <div className="flex flex-wrap gap-2">
              {models.map(m => {
                const isQ = m.model_type.includes('quantum') || m.model_type === 'vqc'
                const isSel = selected.includes(m.model_id)
                return (
                  <button
                    key={m.model_id}
                    onClick={() => setSelected(prev => isSel ? prev.filter(x => x !== m.model_id) : [...prev, m.model_id])}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm transition-colors ${
                      isSel ? 'border-blue-500/50 bg-blue-500/10 text-white' : 'border-slate-700/50 text-slate-400 hover:border-slate-600'
                    }`}
                  >
                    {isQ ? <Zap size={12} className="text-violet-400" /> : <Brain size={12} className="text-blue-400" />}
                    {labelForModel(m)}
                    {m.metrics?.accuracy && <span className="text-xs text-emerald-400">{(m.metrics.accuracy * 100).toFixed(1)}%</span>}
                  </button>
                )
              })}
            </div>
          </div>

          {displayed.length === 0 ? (
            <div className="card text-center py-8 text-slate-400 text-sm">Select at least one model to compare.</div>
          ) : (
            <>
              {/* Metrics Table */}
              <div className="card overflow-x-auto">
                <h2 className="font-semibold text-white mb-4 flex items-center gap-2"><TrendingUp size={16} className="text-blue-400" /> Metrics (held-out test set)</h2>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-700/50">
                      <th className="text-left pb-3 text-slate-400">Metric</th>
                      {displayed.map(m => (
                        <th key={m.model_id} className="text-right pb-3 text-slate-300 min-w-[120px]">
                          <div>{m.model_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</div>
                          <div className="text-xs text-slate-500 font-normal">{m.model_id.slice(0, 8)}</div>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-700/30">
                    {Object.entries(METRIC_LABELS).map(([k, label]) => {
                      const vals = displayed.map(m => m.metrics?.[k])
                      const maxVal = Math.max(...vals.filter(v => v !== undefined) as number[])
                      return (
                        <tr key={k}>
                          <td className="py-2 text-slate-400">{label}</td>
                          {displayed.map(m => {
                            const v = m.metrics?.[k]
                            const isBest = v !== undefined && v === maxVal && vals.filter(x => x !== undefined).length > 1
                            return (
                              <td key={m.model_id} className={`py-2 text-right font-mono ${isBest ? 'text-emerald-400 font-bold' : 'text-white'}`}>
                                {fmt(v)}
                              </td>
                            )
                          })}
                        </tr>
                      )
                    })}
                    <tr className="border-t border-slate-600/50">
                      <td className="pt-3 text-slate-400">Training Time</td>
                      {displayed.map(m => (
                        <td key={m.model_id} className="pt-3 text-right font-mono text-slate-300">
                          {m.training_time_s !== undefined ? `${m.training_time_s.toFixed(2)}s` : '—'}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="py-2 text-slate-400">Test Samples</td>
                      {displayed.map(m => (
                        <td key={m.model_id} className="py-2 text-right font-mono text-slate-300">
                          {m.test_samples ?? m.metrics?.test_size ?? '—'}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
                <p className="text-xs text-slate-500 mt-3">
                  ⬆ Best value highlighted. All metrics computed on the same held-out test set with random_state=42.
                  Simulator runtime ≠ real quantum hardware runtime.
                </p>
              </div>

              {/* Bar Chart */}
              <div className="card">
                <h2 className="font-semibold text-white mb-4">Performance Comparison</h2>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={barData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                    <YAxis domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 12 }} tickFormatter={v => `${v}%`} />
                    <Tooltip
                      contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                      labelStyle={{ color: '#f1f5f9' }}
                      formatter={(v: number) => [`${v.toFixed(2)}%`]}
                    />
                    <Legend wrapperStyle={{ color: '#94a3b8' }} />
                    {displayed.map((m, i) => (
                      <Bar
                        key={m.model_id}
                        dataKey={m.model_type + '_' + m.model_id.slice(0, 4)}
                        name={labelForModel(m)}
                        fill={Object.values(TYPE_COLOR)[i % Object.values(TYPE_COLOR).length]}
                        radius={[4, 4, 0, 0]}
                      />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Radar Chart */}
              {displayed.length >= 2 && (
                <div className="card">
                  <h2 className="font-semibold text-white mb-4">Radar Comparison</h2>
                  <ResponsiveContainer width="100%" height={320}>
                    <RadarChart data={radarData}>
                      <PolarGrid stroke="#334155" />
                      <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                      {displayed.map((m, i) => (
                        <Radar
                          key={m.model_id}
                          name={labelForModel(m)}
                          dataKey={m.model_id.slice(0, 8)}
                          stroke={Object.values(TYPE_COLOR)[i % Object.values(TYPE_COLOR).length]}
                          fill={Object.values(TYPE_COLOR)[i % Object.values(TYPE_COLOR).length]}
                          fillOpacity={0.15}
                        />
                      ))}
                      <Legend wrapperStyle={{ color: '#94a3b8' }} />
                      <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Confusion Matrices */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {displayed.map(m => {
                  const cm = m.metrics?.confusion_matrix as unknown as number[][]
                  if (!cm || !Array.isArray(cm)) return null
                  const labels = ['Negative', 'Positive']
                  return (
                    <div key={m.model_id} className="card">
                      <h3 className="font-medium text-white mb-3 text-sm">{labelForModel(m)} — Confusion Matrix</h3>
                      <div className="inline-grid grid-cols-3 gap-1 text-xs">
                        <div />
                        {labels.map(l => <div key={l} className="text-center text-slate-400 font-medium px-2">Pred {l}</div>)}
                        {cm.map((row, ri) => [
                          <div key={`r${ri}`} className="text-slate-400 font-medium flex items-center pr-2">Act {labels[ri]}</div>,
                          ...row.map((val, ci) => (
                            <div key={`${ri}${ci}`} className={`text-center py-3 rounded font-mono font-bold ${
                              ri === ci ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/15 text-red-400'
                            }`}>{val}</div>
                          ))
                        ])}
                      </div>
                    </div>
                  )
                })}
              </div>
            </>
          )}
        </>
      )}
    </div>
  )
}
