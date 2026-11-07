import { useEffect, useState } from 'react'
import { datasetsApi, modelsApi } from '../services/api'
import { Dataset, ModelType } from '../types'
import { Brain, Cpu, Zap, CheckCircle, AlertCircle, Clock, ChevronRight, Info } from 'lucide-react'

const MODEL_OPTIONS = [
  { value: 'logistic_regression', label: 'Logistic Regression', category: 'classical', desc: 'Fast linear classifier, interpretable baseline' },
  { value: 'svm', label: 'Support Vector Machine', category: 'classical', desc: 'Kernel SVM with RBF kernel' },
  { value: 'random_forest', label: 'Random Forest', category: 'classical', desc: 'Ensemble of decision trees' },
  { value: 'quantum_kernel_svm', label: 'Quantum Kernel SVM', category: 'quantum', desc: 'SVM with ZZFeatureMap quantum kernel' },
  { value: 'vqc', label: 'Variational Quantum Classifier', category: 'quantum', desc: 'Parameterized quantum circuit + ADAM optimizer' },
]

interface Job {
  job_id: string
  status: string
  progress?: string
  error?: string
  model_id?: string
  metrics?: Record<string, number>
}

export default function Training() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [selectedDataset, setSelectedDataset] = useState('')
  const [modelType, setModelType] = useState<ModelType>('logistic_regression')
  const [preproc, setPreproc] = useState({ test_size: 0.2, n_features: 10, use_pca: false })
  const [qParams, setQParams] = useState({ n_qubits: 4, max_iter: 30, max_train_samples: 80, C: 1.0 })
  const [job, setJob] = useState<Job | null>(null)
  const [polling, setPolling] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    datasetsApi.list()
      .then(r => {
        const ds = r.data?.data || r.data?.datasets || []
        setDatasets(ds)
        if (ds.length > 0) setSelectedDataset(ds[0]._id)
      })
      .catch(() => setError('Could not load datasets'))
  }, [])

  useEffect(() => {
    if (!job?.job_id || job.status === 'completed' || job.status === 'failed') return
    setPolling(true)
    const interval = setInterval(() => {
      modelsApi.jobStatus(job.job_id)
        .then(r => {
          const j = r.data?.data || r.data
          setJob(j)
          // Backend terminal statuses: 'completed' (success) | 'failed' (error)
          if (j.status === 'completed' || j.status === 'failed') {
            clearInterval(interval)
            setPolling(false)
          }
        })
        .catch(() => { clearInterval(interval); setPolling(false) })
    }, 2000)
    return () => clearInterval(interval)
  }, [job?.job_id, job?.status])

  const handleTrain = async () => {
    if (!selectedDataset) { setError('Select a dataset first'); return }
    setError('')
    setJob(null)
    const isQuantum = modelType === 'quantum_kernel_svm' || modelType === 'vqc'
    const config = {
      dataset_id: selectedDataset,
      model_type: modelType,
      model_params: isQuantum ? qParams : {},
      preproc_config: { test_size: preproc.test_size, n_features: preproc.n_features, use_pca: preproc.use_pca },
    }
    try {
      const fn = isQuantum ? modelsApi.trainQuantum : modelsApi.train
      const r = await fn(config)
      const data = r.data?.data || r.data
      setJob({ job_id: data.job_id, status: 'queued', progress: 'Queued...' })
    } catch (e: any) {
      setError(e.message || 'Training failed')
    }
  }

  const selectedModel = MODEL_OPTIONS.find(m => m.value === modelType)
  const isQuantum = modelType === 'quantum_kernel_svm' || modelType === 'vqc'

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white">Train Models</h1>
        <p className="text-slate-400 mt-1">Configure and launch classical or quantum ML training jobs</p>
      </div>

      {error && (
        <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-red-400 text-sm">
          <AlertCircle size={16} /> {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Config Panel */}
        <div className="lg:col-span-2 space-y-6">
          {/* Dataset Selection */}
          <div className="card">
            <h2 className="font-semibold text-white mb-4 flex items-center gap-2"><Brain size={16} className="text-blue-400" /> Dataset</h2>
            {datasets.length === 0 ? (
              <div className="text-slate-400 text-sm">No datasets found. Upload one in <a href="/datasets" className="text-blue-400 underline">Datasets</a>.</div>
            ) : (
              <select className="select" value={selectedDataset} onChange={e => setSelectedDataset(e.target.value)}>
                {datasets.map((d: any) => (
                  <option key={d._id} value={d._id}>
                    {d.name} ({d.n_rows?.toLocaleString() ?? '?'} rows)
                  </option>
                ))}
              </select>
            )}
          </div>

          {/* Model Selection */}
          <div className="card">
            <h2 className="font-semibold text-white mb-4">Model Type</h2>
            <div className="space-y-2">
              {MODEL_OPTIONS.map(m => (
                <button
                  key={m.value}
                  onClick={() => setModelType(m.value as ModelType)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg border text-left transition-all ${
                    modelType === m.value
                      ? 'border-blue-500/50 bg-blue-500/10 text-white'
                      : 'border-slate-700/50 bg-slate-800/40 text-slate-300 hover:border-slate-600'
                  }`}
                >
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                    m.category === 'quantum' ? 'bg-violet-500/20' : 'bg-blue-500/20'
                  }`}>
                    {m.category === 'quantum' ? <Zap size={14} className="text-violet-400" /> : <Cpu size={14} className="text-blue-400" />}
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-sm">{m.label}</div>
                    <div className="text-xs text-slate-500 mt-0.5">{m.desc}</div>
                  </div>
                  {m.category === 'quantum' && (
                    <span className="text-xs bg-violet-500/20 text-violet-400 px-2 py-0.5 rounded-full">Quantum</span>
                  )}
                  {modelType === m.value && <CheckCircle size={14} className="text-blue-400 shrink-0" />}
                </button>
              ))}
            </div>
          </div>

          {/* Preprocessing */}
          <div className="card">
            <h2 className="font-semibold text-white mb-4">Preprocessing</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="label">Test Split ({Math.round(preproc.test_size * 100)}%)</label>
                <input type="range" min="10" max="40" value={preproc.test_size * 100}
                  onChange={e => setPreproc(p => ({ ...p, test_size: Number(e.target.value) / 100 }))}
                  className="w-full accent-blue-500" />
              </div>
              <div>
                <label className="label">Max Features</label>
                <input type="number" className="input" value={preproc.n_features} min={2} max={30}
                  onChange={e => setPreproc(p => ({ ...p, n_features: Number(e.target.value) }))} />
              </div>
              <div className="flex items-center gap-3">
                <input type="checkbox" id="use_pca" checked={preproc.use_pca}
                  onChange={e => setPreproc(p => ({ ...p, use_pca: e.target.checked }))}
                  className="accent-blue-500" />
                <label htmlFor="use_pca" className="text-slate-300 text-sm">Use PCA</label>
              </div>
            </div>
          </div>

          {/* Quantum Params */}
          {isQuantum && (
            <div className="card border-violet-500/20">
              <h2 className="font-semibold text-white mb-1 flex items-center gap-2"><Zap size={16} className="text-violet-400" /> Quantum Parameters</h2>
              <p className="text-xs text-slate-500 mb-4">Input features are PCA-reduced to n_qubits dimensions before quantum encoding.</p>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Qubits</label>
                  <select className="select" value={qParams.n_qubits}
                    onChange={e => setQParams(p => ({ ...p, n_qubits: Number(e.target.value) }))}>
                    <option value={2}>2</option>
                    <option value={3}>3</option>
                    <option value={4}>4</option>
                  </select>
                </div>
                <div>
                  <label className="label">Max Train Samples</label>
                  <input type="number" className="input" value={qParams.max_train_samples} min={20} max={200}
                    onChange={e => setQParams(p => ({ ...p, max_train_samples: Number(e.target.value) }))} />
                </div>
                {modelType === 'vqc' && (
                  <div>
                    <label className="label">Max Iterations</label>
                    <input type="number" className="input" value={qParams.max_iter} min={5} max={100}
                      onChange={e => setQParams(p => ({ ...p, max_iter: Number(e.target.value) }))} />
                  </div>
                )}
                {modelType === 'quantum_kernel_svm' && (
                  <div>
                    <label className="label">C (SVM)</label>
                    <input type="number" className="input" value={qParams.C} min={0.01} step={0.1}
                      onChange={e => setQParams(p => ({ ...p, C: Number(e.target.value) }))} />
                  </div>
                )}
              </div>
              <div className="mt-3 flex items-start gap-2 bg-amber-500/10 border border-amber-500/20 rounded-lg p-3 text-xs text-amber-300">
                <Info size={12} className="mt-0.5 shrink-0" />
                Quantum simulation runtime scales with samples². Start with ≤80 samples.
              </div>
            </div>
          )}
        </div>

        {/* Status Panel */}
        <div className="space-y-6">
          <div className="card sticky top-6">
            <h2 className="font-semibold text-white mb-4 flex items-center gap-2"><Brain size={16} className="text-blue-400" /> Training</h2>

            <div className="space-y-3 text-sm mb-6">
              <div className="flex justify-between">
                <span className="text-slate-400">Model</span>
                <span className="text-white font-medium">{selectedModel?.label}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Category</span>
                <span className={isQuantum ? 'text-violet-400' : 'text-blue-400'}>
                  {isQuantum ? 'Quantum' : 'Classical'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Test Split</span>
                <span className="text-white">{Math.round(preproc.test_size * 100)}%</span>
              </div>
              {isQuantum && (
                <div className="flex justify-between">
                  <span className="text-slate-400">Qubits</span>
                  <span className="text-violet-400">{qParams.n_qubits}</span>
                </div>
              )}
            </div>

            <button
              onClick={handleTrain}
              disabled={!selectedDataset || job?.status === 'running' || job?.status === 'queued'}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              <Brain size={16} />
              {job?.status === 'running' || job?.status === 'queued' ? 'Training…' : 'Start Training'}
            </button>

            {/* Job Status */}
            {job && (
              <div className="mt-4 p-4 rounded-lg bg-slate-700/40 border border-slate-600/30">
                <div className="flex items-center gap-2 mb-2">
                  {job.status === 'completed' ? (
                    <CheckCircle size={14} className="text-emerald-400" />
                  ) : job.status === 'failed' ? (
                    <AlertCircle size={14} className="text-red-400" />
                  ) : (
                    <Clock size={14} className="text-blue-400 animate-spin" />
                  )}
                  <span className={`text-sm font-medium ${
                    job.status === 'completed' ? 'text-emerald-400' :
                    job.status === 'failed' ? 'text-red-400' : 'text-blue-400'
                  }`}>{job.status.charAt(0).toUpperCase() + job.status.slice(1)}</span>
                </div>
                {job.progress && <div className="text-xs text-slate-400">{job.progress}</div>}
                {job.error && <div className="text-xs text-red-400 mt-1">{job.error}</div>}
                {job.status === 'completed' && job.metrics && (
                  <div className="mt-3 space-y-1">
                    {Object.entries(job.metrics)
                      .filter(([k]) => ['accuracy', 'f1_score', 'roc_auc'].includes(k))
                      .map(([k, v]) => (
                        <div key={k} className="flex justify-between text-xs">
                          <span className="text-slate-400">{k.replace('_', ' ')}</span>
                          <span className="text-emerald-400 font-mono">{(v as number).toFixed(4)}</span>
                        </div>
                      ))}
                    <a href="/history" className="mt-2 flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300">
                      View in History <ChevronRight size={12} />
                    </a>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
