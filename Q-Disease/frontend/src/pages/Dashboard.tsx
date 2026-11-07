import { useEffect, useState } from 'react'
import { healthApi } from '../services/api'
import { Database, Brain, Zap, FlaskConical, Activity, AlertTriangle } from 'lucide-react'
import { Link } from 'react-router-dom'

interface Stats {
  total_datasets: number
  total_experiments: number
  total_models: number
  total_predictions: number
}

interface Health {
  app: string
  version: string
  storage: string   // "file-based" or "mongodb"
  datasets: number
  models: number
  experiments: number
}

export default function Dashboard() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([healthApi.check(), healthApi.stats()])
      .then(([h, s]) => {
        // Both endpoints return {success, data:{...}} — unwrap one level
        setHealth(h.data.data ?? h.data)
        setStats(s.data.data ?? s.data)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const statCards = [
    { label: 'Datasets', value: stats?.total_datasets ?? 0, icon: Database, color: 'text-blue-400', bg: 'bg-blue-500/10', link: '/datasets' },
    { label: 'Experiments', value: stats?.total_experiments ?? 0, icon: FlaskConical, color: 'text-violet-400', bg: 'bg-violet-500/10', link: '/history' },
    { label: 'Trained Models', value: stats?.total_models ?? 0, icon: Brain, color: 'text-emerald-400', bg: 'bg-emerald-500/10', link: '/training' },
    { label: 'Predictions', value: stats?.total_predictions ?? 0, icon: Zap, color: 'text-amber-400', bg: 'bg-amber-500/10', link: '/predictions' },
  ]

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white">Dashboard</h1>
        <p className="text-slate-400 mt-1">Quantum-Classical ML Platform for Disease Detection</p>
      </div>

      {/* Disclaimer Banner */}
      <div className="flex items-start gap-3 bg-amber-500/10 border border-amber-500/20 rounded-xl p-4">
        <AlertTriangle size={18} className="text-amber-400 mt-0.5 shrink-0" />
        <div className="text-sm text-amber-300">
          <span className="font-semibold">Research Platform Only.</span> Q-Disease is an experimental research tool. Models trained here are not validated for clinical use and should never be used for medical diagnosis or treatment decisions.
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {statCards.map(({ label, value, icon: Icon, color, bg, link }) => (
          <Link key={label} to={link} className="card hover:border-slate-600 transition-colors">
            <div className={`w-10 h-10 rounded-lg ${bg} flex items-center justify-center mb-3`}>
              <Icon size={20} className={color} />
            </div>
            <div className="text-2xl font-bold text-white">{loading ? '—' : value}</div>
            <div className="text-slate-400 text-sm mt-0.5">{label}</div>
          </Link>
        ))}
      </div>

      {/* Health */}
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <Activity size={18} className="text-blue-400" />
          <h2 className="font-semibold text-white">System Status</h2>
        </div>
        {health ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <div className="text-slate-400 mb-1">API Status</div>
              <span className="badge-success">Online</span>
            </div>
            <div>
              <div className="text-slate-400 mb-1">Version</div>
              <span className="text-white font-mono">{health.version}</span>
            </div>
            <div>
              <div className="text-slate-400 mb-1">Storage</div>
              <span className={health.storage === 'mongodb' ? 'badge-success' : 'badge-warning'}>
                {health.storage === 'mongodb' ? 'MongoDB' : 'File-based'}
              </span>
            </div>
            <div>
              <div className="text-slate-400 mb-1">App</div>
              <span className="text-white">{health.app}</span>
            </div>
          </div>
        ) : (
          <div className="text-slate-500 text-sm">
            {loading ? 'Checking backend...' : 'Backend unavailable. Start with: uvicorn app.main:app --port 8000'}
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h2 className="font-semibold text-white mb-4">Quick Start</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <Link to="/datasets" className="flex items-center gap-3 bg-blue-600/10 border border-blue-500/20 rounded-lg p-4 hover:bg-blue-600/20 transition-colors">
            <Database size={20} className="text-blue-400" />
            <div>
              <div className="font-medium text-white text-sm">Upload Dataset</div>
              <div className="text-slate-400 text-xs">Add a CSV for analysis</div>
            </div>
          </Link>
          <Link to="/training" className="flex items-center gap-3 bg-violet-600/10 border border-violet-500/20 rounded-lg p-4 hover:bg-violet-600/20 transition-colors">
            <Brain size={20} className="text-violet-400" />
            <div>
              <div className="font-medium text-white text-sm">Train Model</div>
              <div className="text-slate-400 text-xs">Classical or quantum ML</div>
            </div>
          </Link>
          <Link to="/predictions" className="flex items-center gap-3 bg-emerald-600/10 border border-emerald-500/20 rounded-lg p-4 hover:bg-emerald-600/20 transition-colors">
            <Zap size={20} className="text-emerald-400" />
            <div>
              <div className="font-medium text-white text-sm">Run Prediction</div>
              <div className="text-slate-400 text-xs">Use a trained model</div>
            </div>
          </Link>
        </div>
      </div>
    </div>
  )
}
