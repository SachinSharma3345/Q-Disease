import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, Database, BarChart2, Settings2, FlaskConical,
  Brain, Zap, GitCompare, Lightbulb, Clock, FileDown, Cpu, ChevronRight
} from 'lucide-react'

const nav = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/datasets', label: 'Datasets', icon: Database },
  { to: '/exploration', label: 'Data Exploration', icon: BarChart2 },
  { to: '/preprocessing', label: 'Preprocessing', icon: Settings2 },
  { to: '/training', label: 'Train Models', icon: Brain },
  { to: '/predictions', label: 'Predictions', icon: Zap },
  { to: '/comparison', label: 'Comparison', icon: GitCompare },
  { to: '/explainability', label: 'Explainability', icon: Lightbulb },
  { to: '/history', label: 'History', icon: Clock },
  { to: '/reports', label: 'Reports', icon: FileDown },
  { to: '/settings', label: 'Settings', icon: Cpu },
]

export default function Sidebar() {
  return (
    <aside className="w-64 min-h-screen bg-slate-900/80 border-r border-slate-700/50 flex flex-col">
      {/* Logo */}
      <div className="px-6 py-5 border-b border-slate-700/50">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center">
            <FlaskConical size={16} className="text-white" />
          </div>
          <div>
            <div className="font-bold text-white text-sm">Q-Disease</div>
            <div className="text-xs text-slate-500">Quantum ML Platform</div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-0.5">
        {nav.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors group ${
                isActive
                  ? 'bg-blue-600/20 text-blue-400 border border-blue-500/20'
                  : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/60'
              }`
            }
          >
            <Icon size={16} />
            <span className="flex-1">{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-4 py-4 border-t border-slate-700/50">
        <div className="text-xs text-slate-600 text-center">
          v1.0.0 · Research Only
        </div>
      </div>
    </aside>
  )
}
