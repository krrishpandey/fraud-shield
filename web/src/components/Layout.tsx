import { useEffect } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { API_LABEL, MOCK, api } from '../api/client'
import { useAsync } from '../lib/useAsync'

const NAV = [
  { to: '/', label: 'Score a booking', id: 'score', end: true, icon: 'M3 4h14v12H3zM6 7v6M9 7v6M11 7v6M14 7v6' },
  { to: '/live', label: 'Live stream', id: 'live', icon: 'M2 10h4l2-5 4 10 2-5h4' },
  { to: '/queue', label: 'Review queue', id: 'queue', icon: 'M3 5h14M3 10h14M3 15h9' },
  { to: '/dashboard', label: 'Dashboard', id: 'dashboard', icon: 'M4 16V9M8 16V5M12 16v-4M16 16V7' },
  { to: '/learning', label: 'Learning', id: 'learning', icon: 'M3 15l4-4 3 3 7-7M13 7h4v4' },
  { to: '/audit', label: 'Audit', id: 'audit', icon: 'M10 3l6 3v4c0 4-3 6-6 7-3-1-6-3-6-7V6zM7.5 10l2 2 3-4' },
]

function HealthStatus() {
  const { data, error, reload } = useAsync(() => api.health(), [])
  useEffect(() => {
    const t = setInterval(reload, 15000)
    return () => clearInterval(t)
  }, [reload])

  let tone = 'var(--muted)'
  let text = 'Checking API...'
  let state = 'unknown'
  if (error) {
    tone = 'var(--danger)'
    text = 'API unreachable'
    state = 'down'
  } else if (data) {
    const degraded = Boolean(data.degraded) || !data.ok
    tone = degraded ? 'var(--act-hold)' : 'var(--ok)'
    state = degraded ? 'degraded' : 'ok'
    const mode = data.laya_mode === 'cached' ? 'Laya: cached answers' : `Laya: ${data.laya_mode}`
    text = `${degraded ? 'Degraded' : 'Online'}  |  ${mode}  |  ${data.gpu ? 'GPU' : 'CPU only'}`
  }
  return (
    <div
      data-testid="health-status"
      data-state={state}
      data-laya-mode={data?.laya_mode ?? ''}
      data-gpu={data ? String(data.gpu) : ''}
      className="flex items-center gap-2 text-[0.8rem] whitespace-pre text-ink-2"
      role="status"
      aria-live="polite"
      title={error ? `${API_LABEL}: ${error}` : `API: ${API_LABEL}`}
    >
      <span aria-hidden="true" className="inline-block h-2 w-2 rounded-full" style={{ background: tone }} />
      <span>{text}</span>
      {data?.laya_mode === 'cached' && (
        <span className="rounded-sm bg-tape px-1.5 text-[0.7rem] font-bold text-[#14212e]">Cached</span>
      )}
    </div>
  )
}

export default function Layout() {
  return (
    <div className="flex h-screen flex-col overflow-hidden">
      {MOCK && (
        <div data-testid="mock-banner" role="note" className="shrink-0 bg-tape px-4 py-1 text-center text-[0.8rem] font-semibold text-[#14212e]">
          Mock data: running on built-in fixtures, not the FraudShield API. All numbers are invented.
        </div>
      )}
      <div className="flex min-h-0 flex-1">
        <aside className="flex w-[196px] shrink-0 flex-col border-r border-rule bg-surface" aria-label="Sidebar">
          <div className="flex items-center gap-2 px-4 pt-4 pb-5">
            <svg width="26" height="19" viewBox="0 0 30 22" aria-hidden="true">
              <rect x="0.5" y="0.5" width="29" height="21" rx="2" fill="var(--brand)" />
              <rect x="4" y="5" width="2" height="12" fill="var(--tape)" />
              <rect x="8" y="5" width="1" height="12" fill="var(--brand-ink)" />
              <rect x="11" y="5" width="3" height="12" fill="var(--brand-ink)" />
              <rect x="16" y="5" width="1" height="12" fill="var(--brand-ink)" />
              <rect x="19" y="5" width="2" height="12" fill="var(--brand-ink)" />
              <rect x="23" y="5" width="1" height="12" fill="var(--brand-ink)" />
            </svg>
            <span className="condensed text-lg font-extrabold tracking-tight">FraudShield</span>
          </div>
          <nav aria-label="Views">
            <ul className="flex flex-col gap-0.5 px-2">
              {NAV.map((n) => (
                <li key={n.to}>
                  <NavLink
                    to={n.to}
                    end={n.end}
                    data-testid={`nav-${n.id}`}
                    className={({ isActive }) =>
                      `flex items-center gap-2.5 rounded-sm px-2.5 py-1.5 text-[0.85rem] font-semibold ${
                        isActive ? 'bg-brand text-brand-ink' : 'text-ink-2 hover:bg-surface-2 hover:text-ink'
                      }`
                    }
                  >
                    <svg width="16" height="16" viewBox="0 0 20 20" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.6">
                      <path d={n.icon} />
                    </svg>
                    {n.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
          <div className="mt-auto border-t border-rule px-4 py-3 text-[0.72rem] leading-snug text-muted">
            Booking-time fraud scoring for parcel accounts.
            <br />
            API: {MOCK ? 'mock fixtures' : API_LABEL}
          </div>
        </aside>
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="flex h-9 shrink-0 items-center justify-end border-b border-rule bg-surface px-4">
            <HealthStatus />
          </div>
          <main id="main" className="min-h-0 flex-1 overflow-auto px-5 py-4">
            <Outlet />
          </main>
        </div>
      </div>
    </div>
  )
}
