import { StrictMode, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter, Route, Routes } from 'react-router-dom'
import './index.css'
import './shell.css'
import './pages.css'
import './decision.css'
import './live.css'
import './login.css'
import Layout from './components/Layout'
import ScoreView from './views/ScoreView'
import DecisionView from './views/DecisionView'
import QueueView from './views/QueueView'
import DashboardView from './views/DashboardView'
import AuditView from './views/AuditView'
import LearningView from './views/LearningView'
import LiveView from './views/LiveView'
import LoginView from './views/LoginView'
import RedTeamView from './views/RedTeamView'
import { SessionContext, applySavedTheme, readAnalyst, writeAnalyst } from './lib/session'

applySavedTheme()

function App() {
  const [analyst, setAnalyst] = useState<string | null>(readAnalyst)
  const session = useMemo(
    () => ({
      analyst: analyst ?? '',
      signOut: () => {
        writeAnalyst(null)
        setAnalyst(null)
      },
    }),
    [analyst],
  )

  if (!analyst)
    return (
      <LoginView
        onSignIn={(name) => {
          writeAnalyst(name)
          setAnalyst(name)
        }}
      />
    )

  return (
    <SessionContext.Provider value={session}>
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
    </SessionContext.Provider>
  )
}

// Hash routing: refresh works when web/dist is served as static files inside the desktop window.
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
