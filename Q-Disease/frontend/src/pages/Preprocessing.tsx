import { useEffect, useState } from 'react'
import { datasetsApi } from '../services/api'
import { DatasetRecord } from '../types'
import { Settings2, Info } from 'lucide-react'

const IMPUTATION = ['mean', 'median', 'most_frequent', 'constant']
const SCALING = ['standard', 'minmax', 'none']
const FEATURE_SEL = [
  { value: '', label: 'None — keep all features' },
  { value: 'kbest', label: 'SelectKBest (χ² / mutual info)' },
  { value: 'pca', label: 'PCA (dimensionality reduction)' },
]

export default function Preprocessing() {
  const [datasets, setDatasets] = useState<DatasetRecord[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [dataset, setDataset] = useState<DatasetRecord | null>(null)
  const [config, setConfig] = useState({
    imputation_strategy: 'mean',
    scaling_method: 'standard',
    feature_selection: '',
    n_features: '',
    test_size: '0.2',
  })

  // Load dataset list
  useEffect(() => {
    datasetsApi.list()
      .then(r => {
        const list: DatasetRecord[] = r.data.data || []
        setDatasets(list)
        if (list.length > 0) setSelectedId(list[0]._id)
      })
      .catch(() => {})
  }, [])

  // Load individual dataset when selection changes
  useEffect(() => {
    if (!selectedId) { setDataset(null); return }
    datasetsApi.get(selectedId)
      .then(r => {
        // GET /api/datasets/:id → {success, data: <DatasetRecord>}
        setDataset(r.data.data ?? r.data)
      })
      .catch(() => setDataset(null))
  }, [selectedId])

  const set = (k: string, v: string) => setConfig(c => ({ ...c, [k]: v }))

  const testPct = Math.round(parseFloat(config.test_size || '0.2') * 100)
  const trainPct = 100 - testPct
  const columnNames = dataset?.columns?.map(c => c.name) ?? []
  const missingCols = dataset?.columns?.filter(c => c.missing > 0) ?? []

  return (
    <div className="p-8 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Preprocessing</h1>
        <p className="text-slate-400 text-sm mt-1">Configure the data preprocessing pipeline</p>
      </div>

      <div className="bg-blue-500/5 border border-blue-500/20 rounded-xl p-4 flex gap-3 text-sm">
        <Info size={16} className="text-blue-400 shrink-0 mt-0.5" />
        <div className="text-slate-400">
          All transformers (scaler, imputer, feature selector) are <strong className="text-slate-300">fit on the
          training split only</strong> and then applied to the test split. This prevents data leakage.
          These settings are saved and used when you train a model on the Training page.
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Config Form */}
        <div className="card space-y-4">
          <div className="flex items-center gap-2 mb-2">
            <Settings2 size={18} className="text-violet-400" />
            <h2 className="font-semibold text-white">Pipeline Configuration</h2>
          </div>

          {/* Dataset selector */}
          <div>
            <label className="label">Dataset</label>
            <select className="select" value={selectedId} onChange={e => setSelectedId(e.target.value)}>
              <option value="">-- select dataset --</option>
              {datasets.map(d => (
                <option key={d._id} value={d._id}>{d.name}</option>
              ))}
            </select>
            {datasets.length === 0 && (
              <p className="text-slate-500 text-xs mt-1">No datasets — upload one on the Datasets page.</p>
            )}
          </div>

          {/* Test split */}
          <div>
            <label className="label">Test Split Size</label>
            <input
              type="number"
              min="0.1"
              max="0.5"
              step="0.05"
              className="input"
              value={config.test_size}
              onChange={e => set('test_size', e.target.value)}
            />
            <p className="text-slate-500 text-xs mt-1">
              Fraction held out as test set (0.2 = 20%). Stratified by class.
            </p>
          </div>

          {/* Imputation */}
          <div>
            <label className="label">Missing Value Strategy</label>
            <select className="select" value={config.imputation_strategy} onChange={e => set('imputation_strategy', e.target.value)}>
              {IMPUTATION.map(s => (
                <option key={s} value={s}>{s.replace(/_/g, ' ')}</option>
              ))}
            </select>
          </div>

          {/* Scaling */}
          <div>
            <label className="label">Feature Scaling</label>
            <select className="select" value={config.scaling_method} onChange={e => set('scaling_method', e.target.value)}>
              {SCALING.map(s => (
                <option key={s} value={s}>
                  {s === 'none' ? 'None (raw values)' : s === 'standard' ? 'StandardScaler — zero mean, unit variance' : 'MinMaxScaler — [0, 1] range'}
                </option>
              ))}
            </select>
          </div>

          {/* Feature selection */}
          <div>
            <label className="label">Feature Selection</label>
            <select className="select" value={config.feature_selection} onChange={e => set('feature_selection', e.target.value)}>
              {FEATURE_SEL.map(s => (
                <option key={s.value} value={s.value}>{s.label}</option>
              ))}
            </select>
          </div>

          {config.feature_selection && (
            <div>
              <label className="label">Number of Features (k or components)</label>
              <input
                type="number"
                min="1"
                className="input"
                value={config.n_features}
                onChange={e => set('n_features', e.target.value)}
                placeholder={columnNames.length > 0 ? `max ${columnNames.length - 1}` : 'e.g. 10'}
              />
            </div>
          )}
        </div>

        {/* Right panel: pipeline preview + dataset info */}
        <div className="space-y-4">
          <div className="card">
            <h2 className="font-semibold text-white mb-4">Pipeline Steps</h2>
            <div className="space-y-3">
              {[
                {
                  step: 1,
                  name: `Stratified Train/Test Split (${testPct}% test / ${trainPct}% train)`,
                  detail: 'Class proportions preserved in both splits',
                  color: 'blue',
                },
                {
                  step: 2,
                  name: `Missing Value Imputation (${config.imputation_strategy.replace(/_/g, ' ')})`,
                  detail: 'Fit on train only, transform both',
                  color: 'violet',
                },
                {
                  step: 3,
                  name: `Feature Scaling (${config.scaling_method === 'none' ? 'disabled' : config.scaling_method === 'standard' ? 'StandardScaler' : 'MinMaxScaler'})`,
                  detail: config.scaling_method === 'none' ? 'Skipped' : 'Fit on train only, transform both',
                  color: 'cyan',
                },
                {
                  step: 4,
                  name: `Feature Selection (${config.feature_selection ? (config.feature_selection === 'kbest' ? `SelectKBest, k=${config.n_features || '?'}` : `PCA, n=${config.n_features || '?'}`) : 'all features kept'})`,
                  detail: config.feature_selection ? `Reduce dimensionality to ${config.n_features || '?'} features` : 'No reduction',
                  color: 'emerald',
                },
              ].map(({ step, name, detail, color }) => (
                <div key={step} className="flex gap-3 items-start">
                  <div className={`w-6 h-6 rounded-full bg-${color}-500/20 text-${color}-400 flex items-center justify-center text-xs font-bold shrink-0 mt-0.5`}>
                    {step}
                  </div>
                  <div>
                    <div className="text-white text-sm font-medium">{name}</div>
                    <div className="text-slate-500 text-xs">{detail}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Dataset summary */}
          {dataset && (
            <div className="card bg-slate-800/60 space-y-2 text-sm">
              <div className="font-medium text-white">{dataset.name}</div>
              <div className="text-slate-400">
                {dataset.n_rows?.toLocaleString()} rows × {dataset.n_cols} columns
              </div>
              {dataset.target_col && (
                <div className="text-emerald-400 text-xs">
                  Target: <span className="font-mono">{dataset.target_col}</span>
                </div>
              )}
              {missingCols.length > 0 && (
                <div className="text-amber-400 text-xs">
                  {missingCols.length} column{missingCols.length > 1 ? 's' : ''} with missing values
                  — will be imputed using <span className="font-mono">{config.imputation_strategy}</span>
                </div>
              )}
              {config.feature_selection && config.n_features && (
                <div className="text-blue-400 text-xs">
                  Features will be reduced from {dataset.n_cols - 1} → {config.n_features}
                </div>
              )}
            </div>
          )}

          <div className="text-xs text-slate-500 px-1">
            These settings are passed to the training page. The pipeline is re-fit for every training run.
          </div>
        </div>
      </div>
    </div>
  )
}
