import { useEffect, useState } from 'react'
import { experimentsApi } from '../services/api'
import { Clock, Brain, Zap, CheckCircle, AlertCircle, Loader, Filter, ChevronDown, ChevronUp } from 'lucide-react'

interface Experiment {
  _id: string
  experiment_id?: string
  model_id?: string
  model_type: string
  dataset_id: string
  status: string
  metrics?: Record<string, number>
  created_at: string
  completed_at?: string
  training_time_s?: number
  train_samples?: number
  test_samples?: number
  model_params?: Record<string, unknown>
  preproc_config?: Record<string, unknown>
}

const STATUS_STYLE: Record<string, string> = {
  completed: 'bg-emerald-500/20 text-emerald-400',
  done: 'bg-emerald-500/20 text-emerald-400',
  running: 'bg-blue-500/20 text-blue-400',
  error: 'bg-red-500/20 text-red-400',
  queued: 'bg-amber-500/20 text-amber-400',
}

function StatusIcon({ status }: { status: string }) {
  if (status === 'completed' || status === 'done') return <CheckCircle size={14} className="text-emerald-400" />
  if (status === 'error') return <AlertCircle size={14} className="text-red-400" />
  if (status === 'running') return <Loader size={14} className="text-blue-400 animate-spin" />
  return <Clock size={14} className="text-amber-400" />
}

function fmt(v: number | undefined) {
  if (v === undefined || v === null) return '—'
  return (v * 100).toFixed(1) + '%'
}

export default function History() {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<'all' | 'classical' | 'quantum'>('all')
  const [expanded, setExpanded] = useState<string | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    experimentsApi.list()
      .then(r => {
        const ms: Experiment[] = (r.data?.data || []).map((m: any) => ({
          _id: m._id,
          model_type: m.model_type,
          dataset_id: m.dataset_id,
          status: m.status === 'done' ? 'completed' : m.status,
          metrics: m.metrics,
          created_at: m.created_at,
          completed_at: m.completed_at,
          training_time_s: m.training_time_s,
          train_samples: m.train_samples,
          test_samples: m.test_samples,
          model_params: m.model_params,
          preproc_config: m.preproc_config,
        }))
        setExperiments(ms)
      })
      .catch(() => setError('Failed to load experiment history'))
      .finally(() => setLoading(false))
  }, [])

  const filtered = experiments.filter(e => {
    if (filter === 'classical') return !e.model_type.includes('quantum') && e.model_type !== 'vqc'
    if (filter === 'quantum') return e.model_type.includes('quantum') || e.model_type === 'vqc'
    return true
  })

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white">Experiment History</h1>
        <p className="text-slate-400 mt-1">All training runs, metrics, and configurations</p>
      </div>

      {error && (
        <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-red-400 text-sm">
          <AlertCircle size={16} /> {error}
        </div>
      )}

      {/* Filter bar */}
      <div className="flex items-center gap-2">
        <Filter size={14} className="text-slate-400" />
        {(['all', 'classical', 'quantum'] as const).map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1 rounded-full text-sm transition-colors ${
              filter === f ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
        <span className="ml-auto text-slate-500 text-sm">{filtered.length} experiments</span>
      </div>

      {loading ? (
        <div className="card text-center py-12 text-slate-400">Loading history...</div>
      ) : filtered.length === 0 ? (
        <div className="card text-center py-12">
          <Clock size={32} className="text-slate-600 mx-auto mb-3" />
          <div className="text-slate-400">No experiments found. <a href="/training" className="text-blue-400 underline">Train a model</a> to get started.</div>
        </div>
      ) : (
        <div className="space-y-3">
          {filtered.map(exp => {
            const isQ = exp.model_type.includes('quantum') || exp.model_type === 'vqc'
            const isExpanded = expanded === exp._id
            return (
              <div key={exp._id} className="card overflow-hidden transition-all">
                {/* Header row */}
                <div
                  className="flex items-center gap-4 cursor-pointer"
                  onClick={() => setExpanded(isExpanded ? null : exp._id)}
                >
                  <div className={`w-9 h-9 rounded-lg flex items-center justify-center ${isQ ? 'bg-violet-500/20' : 'bg-blue-500/20'}`}>
                    {isQ ? <Zap size={16} className="text-violet-400" /> : <Brain size={16} className="text-blue-400" />}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-white font-medium text-sm">
                      {exp.model_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                    </div>
                    <div className="text-xs text-slate-500 mt-0.5 flex items-center gap-2">
                      <span>ID: {exp._id.slice(0, 8)}</span>
                      <span>•</span>
                      <span>{new Date(exp.created_at).toLocaleString()}</span>
                    </div>
                  </div>

                  {/* Quick metrics */}
                  <div className="hidden md:flex items-center gap-6 text-sm">
                    {exp.metrics?.accuracy !== undefined && (
                      <div className="text-right">
                        <div className="text-xs text-slate-500">Accuracy</div>
                        <div className="text-emerald-400 font-mono">{fmt(exp.metrics.accuracy)}</div>
                      </div>
                    )}
                    {exp.metrics?.f1_score !== undefined && (
                      <div className="text-right">
                        <div className="text-xs text-slate-500">F1</div>
                        <div className="text-blue-400 font-mono">{fmt(exp.metrics.f1_score)}</div>
                      </div>
                    )}
                    {exp.training_time_s !== undefined && (
                      <div className="text-right">
                        <div className="text-xs text-slate-500">Time</div>
                        <div className="text-slate-300 font-mono">{exp.training_time_s.toFixed(2)}s</div>
                      </div>
                    )}
                  </div>

                  <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${STATUS_STYLE[exp.status] || 'bg-slate-500/20 text-slate-400'}`}>
                    {exp.status}
                  </span>
                  <StatusIcon status={exp.status} />
                  {isExpanded ? <ChevronUp size={14} className="text-slate-400" /> : <ChevronDown size={14} className="text-slate-400" />}
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="mt-4 pt-4 border-t border-slate-700/50 grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div>
                      <h4 className="text-xs text-slate-500 uppercase tracking-wider mb-2">Metrics</h4>
                      <div className="space-y-1 text-sm">
                        {exp.metrics && Object.entries(exp.metrics)
                          .filter(([k]) => !['confusion_matrix', 'test_size'].includes(k) && typeof exp.metrics![k] === 'number')
                          .map(([k, v]) => (
                            <div key={k} className="flex justify-between">
                              <span className="text-slate-400">{k.replace(/_/g, ' ')}</span>
                              <span className="text-white font-mono">{(v as number) <= 1 && (v as number) >= 0 ? fmt(v as number) : (v as number).toFixed(4)}</span>
                            </div>
                          ))}
                      </div>
                    </div>
                    <div>
                      <h4 className="text-xs text-slate-500 uppercase tracking-wider mb-2">Configuration</h4>
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span className="text-slate-400">Dataset ID</span>
                          <span className="text-slate-300 font-mono text-xs">{exp.dataset_id?.slice(0, 8)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-400">Train samples</span>
                          <span className="text-slate-300">{exp.train_samples ?? '—'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-400">Test samples</span>
                          <span className="text-slate-300">{exp.test_samples ?? exp.metrics?.test_size ?? '—'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-400">Training time</span>
                          <span className="text-slate-300">{exp.training_time_s !== undefined ? `${exp.training_time_s.toFixed(3)}s` : '—'}</span>
                        </div>
                      </div>
                    </div>
                    <div>
                      <h4 className="text-xs text-slate-500 uppercase tracking-wider mb-2">Actions</h4>
                      <div className="space-y-2">
                        <a href="/comparison" className="btn-secondary text-xs py-1.5 px-3 w-full text-center block">View in Comparison</a>
                        <a href="/explainability" className="btn-secondary text-xs py-1.5 px-3 w-full text-center block">View Explanations</a>
                        <a href="/predictions" className="btn-secondary text-xs py-1.5 px-3 w-full text-center block">Make Predictions</a>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
