import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter, Route, Routes } from 'react-router-dom'
import './index.css'
import './decision.css'
import './live.css'
import Layout from './components/Layout'
import ScoreView from './views/ScoreView'
import DecisionView from './views/DecisionView'
import QueueView from './views/QueueView'
import DashboardView from './views/DashboardView'
import AuditView from './views/AuditView'
import LearningView from './views/LearningView'
import RedTeamView from './views/RedTeamView'
import LiveView from './views/LiveView'

// Hash routing: refresh works when web/dist is served as static files inside the desktop window.
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <HashRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<ScoreView />} />
          <Route path="decisions/:id" element={<DecisionView />} />
          <Route path="live" element={<LiveView />} />
          <Route path="queue" element={<QueueView />} />
          <Route path="dashboard" element={<DashboardView />} />
          <Route path="learning" element={<LearningView />} />
          <Route path="audit" element={<AuditView />} />
          <Route path="redteam" element={<RedTeamView />} />
          <Route path="*" element={<ScoreView />} />
        </Route>
      </Routes>
    </HashRouter>
  </StrictMode>,
)
