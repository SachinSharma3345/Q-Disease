import { Routes, Route } from 'react-router-dom'
import Sidebar from './layouts/Sidebar'
import Dashboard from './pages/Dashboard'
import Datasets from './pages/Datasets'
import DataExploration from './pages/DataExploration'
import Preprocessing from './pages/Preprocessing'
import Training from './pages/Training'
import Predictions from './pages/Predictions'
import Comparison from './pages/Comparison'
import Explainability from './pages/Explainability'
import History from './pages/History'
import Reports from './pages/Reports'
import Settings from './pages/Settings'

export default function App() {
  return (
    <div className="flex min-h-screen" style={{ background: '#0a0f1e' }}>
      <Sidebar />
      <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/exploration" element={<DataExploration />} />
          <Route path="/preprocessing" element={<Preprocessing />} />
          <Route path="/training" element={<Training />} />
          <Route path="/predictions" element={<Predictions />} />
          <Route path="/comparison" element={<Comparison />} />
          <Route path="/explainability" element={<Explainability />} />
          <Route path="/history" element={<History />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>
    </div>
  )
}
