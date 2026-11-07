import { useEffect, useState } from 'react'
import { modelsApi, reportsApi } from '../services/api'
import { FileDown, Brain, Zap, AlertCircle, CheckCircle, Download } from 'lucide-react'

interface ModelItem {
  _id: string
  model_type: string
  dataset_id: string
  status: string
  metrics?: { accuracy?: number; f1_score?: number }
  created_at: string
  training_time_s?: number
}

export default function Reports() {
  const [models, setModels] = useState<ModelItem[]>([])
  const [loading, setLoading] = useState(true)
  const [downloading, setDownloading] = useState<Record<string, boolean>>({})
  const [success, setSuccess] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    modelsApi.list()
      .then(r => {
        const ms = (r.data?.data || r.data?.models || []).filter((m: any) => !m.status || m.status === 'completed' || m.status === 'done')
        setModels(ms)
      })
      .catch(() => setError('Failed to load models'))
      .finally(() => setLoading(false))
  }, [])

  async function download(modelId: string, format: 'json' | 'csv') {
    const key = `${modelId}-${format}`
    setDownloading(d => ({ ...d, [key]: true }))
    setError(''); setSuccess('')
    try {
      const r = await reportsApi[format](modelId)
      const blob = new Blob([r.data], { type: format === 'json' ? 'application/json' : 'text/csv' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `qdisease_report_${modelId.slice(0, 8)}.${format}`
      a.click()
      URL.revokeObjectURL(url)
      setSuccess(`Report downloaded successfully`)
    } catch (e: any) {
      setError(e.message || 'Download failed')
    } finally {
      setDownloading(d => ({ ...d, [key]: false }))
    }
  }

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white">Reports</h1>
        <p className="text-slate-400 mt-1">Download experiment reports with metrics, configurations, and disclaimers</p>
      </div>

      {error && (
        <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-red-400 text-sm">
          <AlertCircle size={16} /> {error}
        </div>
      )}
      {success && (
        <div className="flex items-center gap-2 bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-4 text-emerald-400 text-sm">
          <CheckCircle size={16} /> {success}
        </div>
      )}

      {/* Report Format Info */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card border-blue-500/20">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
              <FileDown size={16} className="text-blue-400" />
            </div>
            <h3 className="font-semibold text-white">JSON Report</h3>
          </div>
          <p className="text-sm text-slate-400">Full experiment data including metrics, preprocessing config, model configuration, and research limitations. Structured for programmatic use.</p>
        </div>
        <div className="card border-emerald-500/20">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-8 h-8 bg-emerald-500/20 rounded-lg flex items-center justify-center">
              <FileDown size={16} className="text-emerald-400" />
            </div>
            <h3 className="font-semibold text-white">CSV Report</h3>
          </div>
          <p className="text-sm text-slate-400">Tabular summary of metrics, configuration, and disclaimers. Suitable for spreadsheet analysis and academic reporting.</p>
        </div>
      </div>

      {loading ? (
        <div className="card text-center py-12 text-slate-400">Loading experiments...</div>
      ) : models.length === 0 ? (
        <div className="card text-center py-12">
          <FileDown size={32} className="text-slate-600 mx-auto mb-3" />
          <div className="text-slate-400">No completed experiments yet. <a href="/training" className="text-blue-400 underline">Train a model</a> to generate reports.</div>
        </div>
      ) : (
        <div className="card overflow-hidden">
          <h2 className="font-semibold text-white mb-4">Available Reports</h2>
          <div className="divide-y divide-slate-700/50">
            {models.map(m => {
              const isQ = m.model_type.includes('quantum') || m.model_type === 'vqc'
              return (
                <div key={m._id} className="py-4 flex items-center gap-4">
                  <div className={`w-9 h-9 rounded-lg flex items-center justify-center ${isQ ? 'bg-violet-500/20' : 'bg-blue-500/20'}`}>
                    {isQ ? <Zap size={16} className="text-violet-400" /> : <Brain size={16} className="text-blue-400" />}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-white font-medium text-sm">
                      {m.model_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                    </div>
                    <div className="text-xs text-slate-500 mt-0.5 flex items-center gap-2">
                      <span>ID: {m._id.slice(0, 8)}</span>
                      <span>•</span>
                      <span>{new Date(m.created_at).toLocaleDateString()}</span>
                      {m.metrics?.accuracy && (
                        <>
                          <span>•</span>
                          <span className="text-emerald-400">Acc: {(m.metrics.accuracy * 100).toFixed(1)}%</span>
                        </>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => download(m._id, 'json')}
                      disabled={downloading[`${m._id}-json`]}
                      className="flex items-center gap-1.5 bg-blue-600/20 hover:bg-blue-600/40 text-blue-400 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                    >
                      <Download size={14} />
                      {downloading[`${m._id}-json`] ? 'Downloading...' : 'JSON'}
                    </button>
                    <button
                      onClick={() => download(m._id, 'csv')}
                      disabled={downloading[`${m._id}-csv`]}
                      className="flex items-center gap-1.5 bg-emerald-600/20 hover:bg-emerald-600/40 text-emerald-400 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                    >
                      <Download size={14} />
                      {downloading[`${m._id}-csv`] ? 'Downloading...' : 'CSV'}
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <div className="card border-amber-500/20 bg-amber-500/5">
        <h3 className="font-medium text-amber-300 mb-2">Report Disclaimer</h3>
        <p className="text-sm text-amber-300/80">
          All reports include the Q-Disease research disclaimer. Metrics are computed on held-out test data using random_state=42.
          No personally identifiable patient data is included. Quantum model metrics use simulator-based evaluation —
          actual quantum hardware performance may differ.
        </p>
      </div>
    </div>
  )
}
