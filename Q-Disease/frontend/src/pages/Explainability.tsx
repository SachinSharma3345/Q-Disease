import { useEffect, useState } from 'react'
import { modelsApi } from '../services/api'
import { Lightbulb, AlertCircle, Info, Brain, Zap } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

interface ModelOption {
  _id: string
  model_type: string
  metrics?: { accuracy?: number }
}

interface Explanation {
  method: string
  feature_names: string[]
  importance_values: number[]
  is_quantum?: boolean
  note?: string
  method_note?: string
  shap_available?: boolean
}

const COLORS = ['#3b82f6', '#6366f1', '#8b5cf6', '#a855f7', '#06b6d4', '#10b981', '#f59e0b', '#ef4444']

export default function Explainability() {
  const [models, setModels] = useState<ModelOption[]>([])
  const [selected, setSelected] = useState('')
  const [explanation, setExplanation] = useState<Explanation | null>(null)
  const [loading, setLoading] = useState(false)
  const [modelsLoading, setModelsLoading] = useState(true)
  const [error, setError] = useState('')
  const [topN, setTopN] = useState(15)

  useEffect(() => {
    modelsApi.list()
      .then(r => {
        const ms = (r.data?.data || r.data?.models || []).filter((m: any) => !m.status || m.status === 'completed' || m.status === 'done')
        setModels(ms)
        if (ms.length > 0) {
          setSelected(ms[0]._id)
          fetchExplanations(ms[0]._id)
        }
      })
      .catch(() => setError('Failed to load models'))
      .finally(() => setModelsLoading(false))
  }, [])

  async function fetchExplanations(modelId: string) {
    setLoading(true); setError(''); setExplanation(null)
    try {
      const r = await modelsApi.explanations(modelId)
      const data = r.data?.data || r.data
      // Handle nested 'explanations' key
      setExplanation(data?.explanations || data)
    } catch (e: any) {
      setError(e.message || 'Failed to load explanations')
    } finally {
      setLoading(false)
    }
  }

  function handleModelChange(id: string) {
    setSelected(id)
    fetchExplanations(id)
  }

  const selectedModel = models.find(m => m._id === selected)
  const isQuantum = selectedModel?.model_type?.includes('quantum') || selectedModel?.model_type === 'vqc'

  // Prepare sorted chart data
  const chartData = explanation
    ? explanation.feature_names.map((name, i) => ({
        name: name.length > 22 ? name.slice(0, 20) + '…' : name,
        fullName: name,
        importance: Math.abs(explanation.importance_values[i]),
      }))
        .sort((a, b) => b.importance - a.importance)
        .slice(0, topN)
    : []

  const maxImp = chartData[0]?.importance || 1

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white">Explainability</h1>
        <p className="text-slate-400 mt-1">Understand which features drive model predictions</p>
      </div>

      <div className="flex items-start gap-3 bg-blue-500/10 border border-blue-500/20 rounded-xl p-4">
        <Info size={16} className="text-blue-400 mt-0.5 shrink-0" />
        <p className="text-sm text-blue-300">
          Feature importance values indicate the model's reliance on each feature and <strong>do not establish medical causation</strong>.
          For quantum models, permutation-based importance is used over the full quantum pipeline.
        </p>
      </div>

      {error && (
        <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-red-400 text-sm">
          <AlertCircle size={16} /> {error}
        </div>
      )}

      {modelsLoading ? (
        <div className="card text-center py-12 text-slate-400">Loading models...</div>
      ) : models.length === 0 ? (
        <div className="card text-center py-12">
          <Lightbulb size={32} className="text-slate-600 mx-auto mb-3" />
          <div className="text-slate-400">No trained models yet. <a href="/training" className="text-blue-400 underline">Train one first</a>.</div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar */}
          <div className="space-y-4">
            <div className="card">
              <h2 className="font-semibold text-white mb-3 text-sm">Select Model</h2>
              <div className="space-y-2">
                {models.map(m => {
                  const isQ = m.model_type?.includes('quantum') || m.model_type === 'vqc'
                  return (
                    <button
                      key={m._id}
                      onClick={() => handleModelChange(m._id)}
                      className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg border text-left text-sm transition-colors ${
                        selected === m._id
                          ? 'border-blue-500/50 bg-blue-500/10 text-white'
                          : 'border-slate-700/50 text-slate-400 hover:border-slate-600'
                      }`}
                    >
                      {isQ ? <Zap size={12} className="text-violet-400 shrink-0" /> : <Brain size={12} className="text-blue-400 shrink-0" />}
                      <div className="min-w-0">
                        <div className="truncate text-xs font-medium">{m.model_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</div>
                        <div className="text-xs text-slate-500">{m._id.slice(0, 8)}</div>
                      </div>
                    </button>
                  )
                })}
              </div>
            </div>

            {explanation && (
              <div className="card text-sm space-y-2">
                <div className="text-slate-400 font-medium mb-2 text-xs uppercase tracking-wider">Method</div>
                <div className="text-white font-mono text-xs">{explanation.method}</div>
                {explanation.is_quantum && (
                  <span className="inline-block bg-violet-500/20 text-violet-400 px-2 py-0.5 rounded-full text-xs">Quantum Model</span>
                )}
                {explanation.shap_available && (
                  <span className="inline-block bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded-full text-xs">SHAP</span>
                )}
                {(explanation.note || explanation.method_note) && (
                  <p className="text-slate-500 text-xs mt-2 leading-relaxed">{explanation.note || explanation.method_note}</p>
                )}
                <div className="mt-3">
                  <label className="label text-xs">Show Top</label>
                  <select className="select text-xs" value={topN} onChange={e => setTopN(Number(e.target.value))}>
                    <option value={5}>5 features</option>
                    <option value={10}>10 features</option>
                    <option value={15}>15 features</option>
                    <option value={20}>20 features</option>
                    <option value={999}>All</option>
                  </select>
                </div>
              </div>
            )}
          </div>

          {/* Main Chart Area */}
          <div className="lg:col-span-3 space-y-6">
            {loading ? (
              <div className="card text-center py-12 text-slate-400">Computing explanations...</div>
            ) : explanation && chartData.length > 0 ? (
              <>
                <div className="card">
                  <h2 className="font-semibold text-white mb-6">
                    Feature Importance
                    <span className="ml-2 text-xs text-slate-500 font-normal">({explanation.method})</span>
                  </h2>
                  <ResponsiveContainer width="100%" height={Math.max(300, chartData.length * 28)}>
                    <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 30, top: 5, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                      <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} tickFormatter={v => v.toFixed(3)} />
                      <YAxis type="category" dataKey="name" width={160} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                        labelStyle={{ color: '#f1f5f9' }}
                        formatter={(v: number, _n: string, props: any) => [v.toFixed(4), props.payload.fullName]}
                      />
                      <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                        {chartData.map((_, i) => (
                          <Cell key={i} fill={COLORS[i % COLORS.length]} fillOpacity={0.8} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Feature Importance Table */}
                <div className="card">
                  <h2 className="font-semibold text-white mb-4">Feature Ranking</h2>
                  <div className="space-y-2">
                    {chartData.map((item, i) => (
                      <div key={item.fullName} className="flex items-center gap-3">
                        <span className="text-xs text-slate-500 w-5 text-right">{i + 1}</span>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-sm text-slate-300 truncate" title={item.fullName}>{item.fullName}</span>
                            <span className="text-xs font-mono text-blue-400 ml-2 shrink-0">{item.importance.toFixed(4)}</span>
                          </div>
                          <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all"
                              style={{
                                width: `${(item.importance / maxImp) * 100}%`,
                                background: COLORS[i % COLORS.length],
                              }}
                            />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              !loading && !error && (
                <div className="card text-center py-12">
                  <Lightbulb size={32} className="text-slate-600 mx-auto mb-3" />
                  <div className="text-slate-400 text-sm">Select a model to view feature importance</div>
                </div>
              )
            )}
          </div>
        </div>
      )}
    </div>
  )
}
