import { useEffect, useState } from 'react'
import { datasetsApi } from '../services/api'
import { DatasetRecord, FeatureDistributions } from '../types'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { BarChart2, Database } from 'lucide-react'

export default function DataExploration() {
  const [datasets, setDatasets] = useState<DatasetRecord[]>([])
  const [selected, setSelected] = useState('')
  const [dataset, setDataset] = useState<DatasetRecord | null>(null)
  const [distributions, setDistributions] = useState<FeatureDistributions | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Load dataset list
  useEffect(() => {
    datasetsApi.list()
      .then(r => {
        const list: DatasetRecord[] = r.data.data || []
        setDatasets(list)
        if (list.length > 0) setSelected(list[0]._id)
      })
      .catch(() => setError('Failed to load datasets'))
  }, [])

  // Load dataset details + distributions when selection changes
  useEffect(() => {
    if (!selected) { setDataset(null); setDistributions(null); return }
    setLoading(true)
    setError('')
    Promise.all([
      datasetsApi.get(selected),
      datasetsApi.distributions(selected),
    ])
      .then(([d, dist]) => {
        // GET /api/datasets/:id  → {success, data: <DatasetRecord>}
        setDataset(d.data.data ?? d.data)
        // GET /api/datasets/:id/distributions → {success, data: {feature: {bins, counts}}}
        setDistributions(dist.data.data ?? dist.data)
      })
      .catch(() => setError('Failed to load dataset details'))
      .finally(() => setLoading(false))
  }, [selected])

  const distChartData = (col: string) => {
    if (!distributions?.[col]) return []
    const { counts, bins } = distributions[col]
    return counts.map((c: number, i: number) => ({
      bin: bins[i]?.toFixed(1),
      count: c,
    }))
  }

  return (
    <div className="p-8 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Data Exploration</h1>
        <p className="text-slate-400 text-sm mt-1">Visualize distributions, correlations, and column statistics</p>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 text-red-400 text-sm">{error}</div>
      )}

      <div>
        <label className="label">Select Dataset</label>
        <select
          className="select max-w-xs"
          value={selected}
          onChange={e => setSelected(e.target.value)}
        >
          <option value="">-- choose --</option>
          {datasets.map(d => (
            <option key={d._id} value={d._id}>{d.name}</option>
          ))}
        </select>
      </div>

      {!selected && (
        <div className="card text-center py-12">
          <Database size={32} className="text-slate-600 mx-auto mb-3" />
          <div className="text-slate-500">Select a dataset above to explore it.</div>
        </div>
      )}

      {loading && <div className="text-slate-400 text-sm">Loading...</div>}

      {dataset && !loading && (
        <div className="space-y-6">
          {/* Overview */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Rows', value: dataset.n_rows?.toLocaleString() },
              { label: 'Columns', value: dataset.n_cols },
              { label: 'Missing Values', value: dataset.missing_total },
              { label: 'Duplicates', value: dataset.duplicates },
            ].map(({ label, value }) => (
              <div key={label} className="card text-center py-4">
                <div className="text-2xl font-bold text-white">{value ?? '—'}</div>
                <div className="text-slate-400 text-sm mt-1">{label}</div>
              </div>
            ))}
          </div>

          {/* Target distribution */}
          {dataset.target_distribution && Object.keys(dataset.target_distribution).length > 0 && (
            <div className="card">
              <h2 className="font-semibold text-white mb-3">
                Target Distribution
                <span className="text-slate-400 font-normal text-sm ml-2 font-mono">({dataset.target_col})</span>
              </h2>
              <div className="flex gap-3 flex-wrap">
                {Object.entries(dataset.target_distribution).map(([cls, count]) => {
                  const pct = dataset.n_rows ? Math.round((count / dataset.n_rows) * 100) : 0
                  return (
                    <div key={cls} className="bg-slate-700/60 rounded-lg px-4 py-3 min-w-[120px]">
                      <div className="font-mono font-semibold text-emerald-400">{cls}</div>
                      <div className="text-white text-lg font-bold">{count}</div>
                      <div className="text-slate-400 text-xs">{pct}%</div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Feature distributions */}
          {distributions && Object.keys(distributions).length > 0 && (
            <div className="card">
              <div className="flex items-center gap-2 mb-4">
                <BarChart2 size={18} className="text-blue-400" />
                <h2 className="font-semibold text-white">Feature Distributions</h2>
                <span className="text-slate-400 text-xs">(first 6 numeric features)</span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {Object.keys(distributions).slice(0, 6).map(col => (
                  <div key={col}>
                    <div className="text-slate-400 text-xs font-mono mb-2">{col}</div>
                    <ResponsiveContainer width="100%" height={120}>
                      <BarChart
                        data={distChartData(col)}
                        margin={{ top: 0, right: 0, bottom: 0, left: -20 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                        <XAxis dataKey="bin" tick={{ fontSize: 9, fill: '#64748b' }} />
                        <YAxis tick={{ fontSize: 9, fill: '#64748b' }} />
                        <Tooltip
                          contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }}
                        />
                        <Bar dataKey="count" fill="#3b82f6" radius={[2, 2, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Column details table */}
          <div className="card overflow-auto">
            <h2 className="font-semibold text-white mb-4">Column Details ({dataset.n_cols} columns)</h2>
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-slate-700">
                  {['Column', 'Type', 'Unique', 'Missing', 'Min', 'Max', 'Mean'].map(h => (
                    <th key={h} className="text-left py-2 px-2 text-slate-400 font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {dataset.columns?.map(col => (
                  <tr
                    key={col.name}
                    className={`border-b border-slate-800 hover:bg-slate-800/30 ${col.name === dataset.target_col ? 'bg-emerald-500/5' : ''}`}
                  >
                    <td className="py-1.5 px-2 font-mono text-slate-300">
                      {col.name}
                      {col.name === dataset.target_col && (
                        <span className="ml-1 text-emerald-400 text-xs">★</span>
                      )}
                    </td>
                    <td className="py-1.5 px-2 text-slate-400">{col.dtype}</td>
                    <td className="py-1.5 px-2 text-slate-400">{col.unique}</td>
                    <td className={`py-1.5 px-2 ${col.missing > 0 ? 'text-amber-400' : 'text-slate-400'}`}>
                      {col.missing}
                    </td>
                    <td className="py-1.5 px-2 text-slate-400">{col.min?.toFixed(3) ?? '—'}</td>
                    <td className="py-1.5 px-2 text-slate-400">{col.max?.toFixed(3) ?? '—'}</td>
                    <td className="py-1.5 px-2 text-slate-400">{col.mean?.toFixed(3) ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
