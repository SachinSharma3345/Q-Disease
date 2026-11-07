import { useEffect, useState, useRef } from 'react'
import { datasetsApi } from '../services/api'
import { DatasetRecord } from '../types'
import { Upload, Trash2, Eye, RefreshCw, Database, FileText, FlaskConical, X } from 'lucide-react'

// ─── Pending upload state ─────────────────────────────────────────────────────
// When a CSV is uploaded without a target column the backend responds with
// {registered: false, filepath, name, column_names, analysis}.
// We store this and show a column-picker before calling /api/datasets/register.
interface PendingUpload {
  filepath: string
  name: string
  columnNames: string[]
}

export default function Datasets() {
  const [datasets, setDatasets] = useState<DatasetRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [demoLoading, setDemoLoading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [preview, setPreview] = useState<DatasetRecord | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  // Two-step upload state
  const [pending, setPending] = useState<PendingUpload | null>(null)
  const [targetCol, setTargetCol] = useState('')
  const [registering, setRegistering] = useState(false)

  const load = () => {
    setLoading(true)
    datasetsApi.list()
      .then(r => setDatasets(r.data.data || []))
      .catch(() => setError('Failed to load datasets'))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  // ─── Step 1: upload file, get analysis back ───────────────────────────────
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    setError('')
    setSuccess('')
    setPending(null)
    setTargetCol('')
    try {
      const r = await datasetsApi.upload(file)
      const data = r.data.data ?? r.data

      if (data.registered === true) {
        // Backend accepted a target_col in the upload (shouldn't happen here but handle it)
        setSuccess(`"${file.name}" uploaded and registered.`)
        load()
      } else {
        // registered: false — need to pick a target column
        setPending({
          filepath: data.filepath,
          name: data.name || file.name.replace(/\.csv$/i, ''),
          columnNames: data.column_names || [],
        })
      }
    } catch (err: any) {
      setError(err.message)
    } finally {
      setUploading(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  // ─── Step 2: register with target column ─────────────────────────────────
  const handleRegister = async () => {
    if (!pending || !targetCol) return
    setRegistering(true)
    setError('')
    try {
      await datasetsApi.register({
        filepath: pending.filepath,
        name: pending.name,
        target_col: targetCol,
      })
      setSuccess(`"${pending.name}" registered with target column "${targetCol}".`)
      setPending(null)
      setTargetCol('')
      load()
    } catch (err: any) {
      setError(err.message)
    } finally {
      setRegistering(false)
    }
  }

  // ─── Load demo dataset ────────────────────────────────────────────────────
  const handleLoadDemo = async () => {
    setDemoLoading(true)
    setError('')
    setSuccess('')
    try {
      const r = await datasetsApi.loadDemo()
      const data = r.data.data ?? r.data
      if (data.message === 'Already loaded') {
        setSuccess('Breast Cancer Wisconsin dataset is already loaded.')
      } else {
        setSuccess(`Demo dataset loaded: ${data.message} (${data.n_rows} rows, ${data.n_features} features).`)
      }
      load()
    } catch (err: any) {
      setError(err.message)
    } finally {
      setDemoLoading(false)
    }
  }

  const handleDelete = async (id: string, name: string) => {
    if (!confirm(`Delete dataset "${name}"?`)) return
    setError('')
    setSuccess('')
    try {
      await datasetsApi.delete(id)
      setSuccess(`"${name}" deleted.`)
      load()
    } catch (err: any) {
      setError(err.message)
    }
  }

  const handlePreview = async (id: string) => {
    try {
      const r = await datasetsApi.get(id)
      setPreview(r.data.data ?? r.data)
    } catch {}
  }

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Datasets</h1>
          <p className="text-slate-400 text-sm mt-1">Upload and manage CSV datasets</p>
        </div>
        <div className="flex gap-2">
          <button onClick={load} className="btn-secondary flex items-center gap-2">
            <RefreshCw size={16} /> Refresh
          </button>
          <button
            onClick={handleLoadDemo}
            disabled={demoLoading}
            className="btn-secondary flex items-center gap-2"
          >
            <FlaskConical size={16} />
            {demoLoading ? 'Loading...' : 'Load Demo'}
          </button>
          <button
            onClick={() => fileRef.current?.click()}
            disabled={uploading}
            className="btn-primary flex items-center gap-2"
          >
            <Upload size={16} /> {uploading ? 'Uploading...' : 'Upload CSV'}
          </button>
          <input ref={fileRef} type="file" accept=".csv" className="hidden" onChange={handleUpload} />
        </div>
      </div>

      {/* Messages */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 text-red-400 text-sm">
          {error}
        </div>
      )}
      {success && (
        <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-3 text-emerald-400 text-sm">
          {success}
        </div>
      )}

      {/* ── Target column picker (step 2 of upload) ─────────────────────── */}
      {pending && (
        <div className="card border-blue-500/30 bg-blue-500/5 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold text-white">Select Target Column for "{pending.name}"</h2>
            <button onClick={() => { setPending(null); setTargetCol('') }} className="text-slate-400 hover:text-white">
              <X size={18} />
            </button>
          </div>
          <p className="text-slate-400 text-sm">
            The file was uploaded successfully. Choose which column the model should predict.
          </p>
          <div>
            <label className="label">Target Column</label>
            <select
              className="select max-w-xs"
              value={targetCol}
              onChange={e => setTargetCol(e.target.value)}
            >
              <option value="">-- select target --</option>
              {pending.columnNames.map(col => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
          </div>
          <button
            onClick={handleRegister}
            disabled={!targetCol || registering}
            className="btn-primary flex items-center gap-2 w-fit"
          >
            {registering ? 'Registering...' : 'Register Dataset'}
          </button>
        </div>
      )}

      {/* Dataset list */}
      {loading ? (
        <div className="text-slate-400 text-sm">Loading...</div>
      ) : datasets.length === 0 ? (
        <div className="card text-center py-16">
          <Database size={48} className="text-slate-600 mx-auto mb-4" />
          <div className="text-slate-400 font-medium">No datasets yet</div>
          <div className="text-slate-500 text-sm mt-2">
            Upload a CSV file, or click <strong className="text-slate-400">Load Demo</strong> to use the
            built-in Breast Cancer Wisconsin dataset.
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {datasets.map(ds => (
            <div key={ds._id} className="card flex items-center gap-4">
              <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center shrink-0">
                <FileText size={20} className="text-blue-400" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="font-medium text-white truncate">{ds.name}</div>
                <div className="text-slate-400 text-xs mt-0.5">
                  {ds.n_rows?.toLocaleString()} rows · {ds.n_cols} cols ·{' '}
                  {ds.target_col && (
                    <span className="text-emerald-400">target: <span className="font-mono">{ds.target_col}</span> · </span>
                  )}
                  {ds.duplicates > 0 && (
                    <span className="text-amber-400">{ds.duplicates} duplicates · </span>
                  )}
                  {ds.missing_total > 0 && (
                    <span className="text-amber-400">{ds.missing_total} missing values · </span>
                  )}
                  {ds.created_at ? new Date(ds.created_at).toLocaleDateString() : ''}
                </div>
              </div>
              <div className="flex gap-2 shrink-0">
                <button
                  onClick={() => handlePreview(ds._id)}
                  className="btn-secondary flex items-center gap-1 py-1.5 px-3 text-sm"
                >
                  <Eye size={14} /> Preview
                </button>
                <button
                  onClick={() => handleDelete(ds._id, ds.name)}
                  className="btn-danger flex items-center gap-1 py-1.5 px-3 text-sm"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Preview Modal */}
      {preview && (
        <div
          className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4"
          onClick={() => setPreview(null)}
        >
          <div
            className="bg-slate-800 border border-slate-700 rounded-xl max-w-2xl w-full max-h-[80vh] overflow-auto p-6"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-bold text-white">{preview.name}</h2>
              <button onClick={() => setPreview(null)} className="text-slate-400 hover:text-white">
                <X size={18} />
              </button>
            </div>

            {/* Stats row */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mb-5">
              <div className="bg-slate-700/50 rounded-lg p-3">
                <div className="text-slate-400">Rows</div>
                <div className="text-white font-semibold">{preview.n_rows?.toLocaleString()}</div>
              </div>
              <div className="bg-slate-700/50 rounded-lg p-3">
                <div className="text-slate-400">Columns</div>
                <div className="text-white font-semibold">{preview.n_cols}</div>
              </div>
              <div className="bg-slate-700/50 rounded-lg p-3">
                <div className="text-slate-400">Target</div>
                <div className="text-emerald-400 font-mono text-xs font-semibold">{preview.target_col || '—'}</div>
              </div>
              <div className="bg-slate-700/50 rounded-lg p-3">
                <div className="text-slate-400">Missing</div>
                <div className={preview.missing_total > 0 ? 'text-amber-400 font-semibold' : 'text-white font-semibold'}>
                  {preview.missing_total}
                </div>
              </div>
            </div>

            {/* Target distribution */}
            {preview.target_distribution && Object.keys(preview.target_distribution).length > 0 && (
              <div className="mb-5">
                <div className="text-slate-400 text-xs font-medium mb-2">Class Distribution</div>
                <div className="flex gap-3">
                  {Object.entries(preview.target_distribution).map(([cls, count]) => (
                    <div key={cls} className="bg-slate-700/50 rounded px-3 py-2 text-sm">
                      <span className="font-mono text-emerald-400">{cls}</span>
                      <span className="text-slate-400 ml-2">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Column list */}
            <div className="text-slate-400 text-xs font-medium mb-2">All Columns ({preview.n_cols})</div>
            <div className="flex flex-wrap gap-1.5 mb-4">
              {preview.columns?.map(col => (
                <span
                  key={col.name}
                  className={`px-2 py-0.5 rounded text-xs font-mono ${
                    col.name === preview.target_col
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                      : 'bg-slate-700 text-slate-300'
                  }`}
                >
                  {col.name}
                  <span className="text-slate-500 ml-1">{col.dtype[0]}</span>
                </span>
              ))}
            </div>

            {/* Missing values detail */}
            {preview.columns?.some(c => c.missing > 0) && (
              <>
                <div className="text-amber-400 text-xs font-medium mb-2">
                  Columns with Missing Values
                </div>
                <div className="space-y-1">
                  {preview.columns.filter(c => c.missing > 0).map(c => (
                    <div key={c.name} className="flex justify-between text-xs">
                      <span className="text-slate-400 font-mono">{c.name}</span>
                      <span className="text-amber-400">{c.missing} missing</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
