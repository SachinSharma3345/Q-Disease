import { useEffect, useState } from 'react'
import { healthApi } from '../services/api'
import { Cpu, CheckCircle, XCircle, Database, AlertTriangle, Info, Shield } from 'lucide-react'

interface HealthData {
  app: string
  version: string
  storage: string
  disclaimer: string
  datasets: number
  models: number
  experiments: number
}

export default function Settings() {
  const [health, setHealth] = useState<HealthData | null>(null)
  const [loading, setLoading] = useState(true)
  const [backendOk, setBackendOk] = useState(false)

  useEffect(() => {
    healthApi.check()
      .then(r => {
        setHealth(r.data?.data || r.data)
        setBackendOk(true)
      })
      .catch(() => setBackendOk(false))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="p-8 space-y-8 max-w-3xl">
      <div>
        <h1 className="text-3xl font-bold text-white">Settings</h1>
        <p className="text-slate-400 mt-1">System configuration and platform information</p>
      </div>

      {/* Research Disclaimer */}
      <div className="flex items-start gap-4 bg-amber-500/10 border border-amber-500/30 rounded-xl p-5">
        <AlertTriangle size={20} className="text-amber-400 mt-0.5 shrink-0" />
        <div>
          <h3 className="text-amber-300 font-semibold mb-1">Research Platform Disclaimer</h3>
          <p className="text-amber-300/80 text-sm leading-relaxed">
            Q-Disease is an experimental research and educational platform. It is <strong>not a clinically validated medical device</strong> and must not be used to diagnose, treat, or make medical decisions for patients. All outputs are for research and software testing purposes only.
          </p>
        </div>
      </div>

      {/* Connection Status */}
      <div className="card">
        <h2 className="font-semibold text-white mb-4 flex items-center gap-2"><Cpu size={16} className="text-blue-400" /> System Status</h2>
        <div className="space-y-3">
          <div className="flex items-center justify-between py-2 border-b border-slate-700/50">
            <div className="flex items-center gap-2 text-sm text-slate-300">
              <Cpu size={14} className="text-blue-400" />
              Backend API
            </div>
            {loading ? (
              <span className="text-slate-500 text-sm">Checking...</span>
            ) : backendOk ? (
              <span className="flex items-center gap-1 text-emerald-400 text-sm"><CheckCircle size={14} /> Connected</span>
            ) : (
              <span className="flex items-center gap-1 text-red-400 text-sm"><XCircle size={14} /> Disconnected</span>
            )}
          </div>
          <div className="flex items-center justify-between py-2 border-b border-slate-700/50">
            <div className="flex items-center gap-2 text-sm text-slate-300">
              <Database size={14} className="text-violet-400" />
              Database
            </div>
            <span className="text-sm">
              {health?.storage === 'mongodb'
                ? <span className="flex items-center gap-1 text-emerald-400"><CheckCircle size={14} /> MongoDB</span>
                : <span className="flex items-center gap-1 text-amber-400"><Info size={14} /> File-based (MongoDB optional)</span>
              }
            </span>
          </div>
          <div className="flex items-center justify-between py-2 border-b border-slate-700/50">
            <div className="flex items-center gap-2 text-sm text-slate-300">
              <Shield size={14} className="text-blue-400" />
              Quantum Backend
            </div>
            <span className="flex items-center gap-1 text-blue-400 text-sm"><CheckCircle size={14} /> NumPy Simulator</span>
          </div>
        </div>
      </div>

      {/* Platform Info */}
      {health && (
        <div className="card">
          <h2 className="font-semibold text-white mb-4">Platform Information</h2>
          <div className="grid grid-cols-2 gap-x-8 gap-y-3 text-sm">
            {[
              ['Application', health.app],
              ['Version', health.version],
              ['Storage Backend', health.storage],
              ['Datasets Stored', String(health.datasets)],
              ['Models Trained', String(health.models)],
              ['Experiments Run', String(health.experiments)],
            ].map(([k, v]) => (
              <div key={k} className="flex items-center justify-between py-1 border-b border-slate-700/30">
                <span className="text-slate-400">{k}</span>
                <span className="text-white font-mono">{v}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Dataset Limits */}
      <div className="card">
        <h2 className="font-semibold text-white mb-4">Dataset Limits</h2>
        <div className="space-y-2 text-sm">
          {[
            ['Max upload size', '50 MB'],
            ['Accepted formats', 'CSV only'],
            ['Max quantum train samples', '200 (recommended ≤80)'],
            ['Default random seed', '42'],
          ].map(([k, v]) => (
            <div key={k} className="flex items-center justify-between py-1 border-b border-slate-700/30">
              <span className="text-slate-400">{k}</span>
              <span className="text-white">{v}</span>
            </div>
          ))}
        </div>
      </div>

      {/* MongoDB Setup */}
      <div className="card border-violet-500/20">
        <h2 className="font-semibold text-white mb-3 flex items-center gap-2"><Database size={16} className="text-violet-400" /> MongoDB Setup (Optional)</h2>
        <p className="text-sm text-slate-400 mb-4">
          Q-Disease uses file-based storage by default. To enable MongoDB persistence, set the <code className="bg-slate-700 px-1.5 py-0.5 rounded text-xs text-blue-300">MONGODB_URI</code> environment variable.
        </p>
        <div className="bg-slate-900 rounded-lg p-4 font-mono text-sm">
          <div className="text-slate-500"># Local MongoDB</div>
          <div className="text-green-400">MONGODB_URI=mongodb://localhost:27017</div>
          <div className="text-slate-500 mt-2"># MongoDB Atlas</div>
          <div className="text-green-400">MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net</div>
        </div>
        <p className="text-xs text-slate-500 mt-3">Restart the backend after changing environment variables.</p>
      </div>

      {/* Technical Notes */}
      <div className="card border-blue-500/20">
        <h2 className="font-semibold text-white mb-3 flex items-center gap-2"><Info size={16} className="text-blue-400" /> Technical Notes</h2>
        <ul className="text-sm text-slate-400 space-y-2 list-disc list-inside">
          <li>Quantum circuits are simulated using NumPy statevector simulation (no Qiskit installation required)</li>
          <li>ZZFeatureMap encodes features into qubit rotation angles</li>
          <li>Simulator runtime scales quadratically with training samples — keep ≤100 for Quantum Kernel SVM</li>
          <li>VQC uses finite-difference gradient estimation with ADAM optimization</li>
          <li>All model artifacts are stored locally in the <code className="bg-slate-700 px-1 py-0.5 rounded text-xs">artifacts/</code> directory</li>
          <li>Preprocessing transformations are fitted on training data only to prevent leakage</li>
          <li>Simulator performance metrics are not comparable to real quantum hardware</li>
        </ul>
      </div>
    </div>
  )
}
