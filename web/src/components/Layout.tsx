import { useEffect, useMemo, useRef, useState } from 'react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { MOCK } from '../api/client'
import { useSession } from '../lib/session'
import { StatusContext, lastDecision, useConsoleStatus, useStatus } from '../lib/status'
import { Icon, type IconName } from './Icon'

type Item = { to: string; label: string; id: string; icon: IconName; end?: boolean; pill?: string; count?: number }

/** Wordmark glyph: the bars of a shipping-label barcode, the one object every booking produces. */
function LogoMark() {
  // Five even bars of a shipping-label barcode: wide, thin, wide, thin, wide.
  const bars = [0, 5, 8, 13, 16]
  const widths = [3, 1.5, 3, 1.5, 3]
  return (
    <svg className="sb-mark" width="20" height="24" viewBox="0 0 19 24" aria-hidden="true" shapeRendering="crispEdges">
      {bars.map((x, i) => (
        <rect key={x} x={x} y={0} width={widths[i]} height={24} fill="currentColor" />
      ))}
    </svg>
  )
}

function Brand({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  return (
    <div className="sb-brand">
      <NavLink to="/" className="sb-logo" aria-label="tracd home">
        <LogoMark />
        <b>TRACD</b>
      </NavLink>
      <button type="button" className="sb-icon" onClick={onToggle} aria-label={collapsed ? 'Expand the sidebar' : 'Collapse the sidebar'}>
        <Icon name="panel" size={17} />
      </button>
    </div>
  )
}

const PAGES: { label: string; to: string; icon: IconName }[] = [
  { label: 'Score a booking', to: '/', icon: 'box' },
  { label: 'Live stream', to: '/live', icon: 'pulse' },
  { label: 'Review queue', to: '/queue', icon: 'list' },
  { label: 'Dashboard', to: '/dashboard', icon: 'chart' },
  { label: 'Learning', to: '/learning', icon: 'refresh' },
  { label: 'Audit log', to: '/audit', icon: 'shield' },
]

function QuickFind({ onClose }: { onClose: () => void }) {
  const nav = useNavigate()
  const { recent } = useStatus()
  const [q, setQ] = useState('')
  const [i, setI] = useState(0)
  const ref = useRef<HTMLInputElement>(null)
  useEffect(() => {
    ref.current?.focus()
  }, [])
  const term = q.trim().toLowerCase()
  const results = [
    ...PAGES.filter((p) => !term || p.label.toLowerCase().includes(term)),
    ...recent
      .filter((d) => term && (d.decision_id.includes(term) || d.account_id.includes(term) || d.booking_id.includes(term)))
      .slice(0, 6)
      .map((d) => ({ label: `${d.decision_id} · acct ${d.account_id.slice(0, 8)}`, to: `/decisions/${d.decision_id}`, icon: 'decision' as IconName })),
  ]
  const go = (to: string) => {
    nav(to)
    onClose()
  }
  return (
    <div className="qf-scrim" onMouseDown={onClose}>
      <div className="qf" role="dialog" aria-label="Quick find" onMouseDown={(e) => e.stopPropagation()}>
        <div className="search">
          <Icon name="search" size={17} />
          <input
            ref={ref}
            className="field qf-input"
            placeholder="Jump to a page, or type a decision or account id"
            value={q}
            onChange={(e) => {
              setQ(e.target.value)
              setI(0)
            }}
            onKeyDown={(e) => {
              if (e.key === 'Escape') onClose()
              if (e.key === 'ArrowDown') setI((x) => Math.min(x + 1, results.length - 1))
              if (e.key === 'ArrowUp') setI((x) => Math.max(x - 1, 0))
              if (e.key === 'Enter' && results[i]) go(results[i].to)
            }}
          />
        </div>
        <ul className="qf-list">
          {results.length === 0 && <li className="qf-empty">Nothing matches “{q}”.</li>}
          {results.map((r, k) => (
            <li key={r.to + r.label}>
              <button type="button" className={k === i ? 'is-on' : ''} onMouseEnter={() => setI(k)} onClick={() => go(r.to)}>
                <Icon name={r.icon} size={16} />
                <span className={r.icon === 'decision' ? 'mono' : ''}>{r.label}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}

function Group({ title, items, collapsed }: { title: string; items: Item[]; collapsed: boolean }) {
  const [open, setOpen] = useState(true)
  return (
    <div className="sb-group">
      <button type="button" className="sb-group-h" aria-expanded={open} tabIndex={collapsed ? -1 : 0} onClick={() => setOpen((o) => !o)}>
        {title}
        <Icon name="chevronDown" size={15} style={{ transform: open ? 'none' : 'rotate(-90deg)', transition: 'transform .25s ease' }} />
      </button>
      <div className="sb-group-body" data-open={open || collapsed}>
        <ul>
          {items.map((n) => (
            <li key={n.id}>
              <NavLink
                to={n.to}
                end={n.end}
                data-testid={`nav-${n.id}`}
                className={({ isActive }) => `sb-item${isActive ? ' is-on' : ''}`}
                title={collapsed ? n.label : undefined}
              >
                {({ isActive }) => (
                  <>
                    <span className="sb-ico">
                      <Icon name={n.icon} size={18} stroke={isActive ? 2.1 : 1.8} />
                    </span>
                    <span className="sb-label">{n.label}</span>
                    {n.pill && <span className="sb-pill">{n.pill}</span>}
                    {n.count != null && n.count > 0 && <span className="sb-count">{n.count > 99 ? '99+' : n.count}</span>}
                  </>
                )}
              </NavLink>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}

export default function Layout() {
  const { analyst, signOut } = useSession()
  const status = useConsoleStatus()
  const loc = useLocation()
  const [collapsed, setCollapsed] = useState(false)
  const [finding, setFinding] = useState(false)
  const [menu, setMenu] = useState(false)

  useEffect(() => {
    document.getElementById('main')?.scrollTo(0, 0)
    setMenu(false)
  }, [loc.pathname])
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setFinding((f) => !f)
      }
    }
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [])

  const decisionTo = useMemo(() => {
    if (/^\/decisions\/.+$/.test(loc.pathname)) return loc.pathname
    const id = lastDecision() ?? status.recent[0]?.decision_id
    return id ? `/decisions/${id}` : null
  }, [loc.pathname, status.recent])

  const gate: Item[] = [
    { to: '/', label: 'Score a booking', id: 'score', icon: 'box', end: true },
    ...(decisionTo ? [{ to: decisionTo, label: 'Decision', id: 'decision', icon: 'decision' as IconName }] : []),
  ]
  const operate: Item[] = [
    { to: '/live', label: 'Live stream', id: 'live', icon: 'pulse', pill: 'Replay' },
    { to: '/queue', label: 'Review queue', id: 'queue', icon: 'list', count: status.openCases.length },
  ]
  const monitor: Item[] = [
    { to: '/dashboard', label: 'Dashboard', id: 'dashboard', icon: 'chart' },
    { to: '/learning', label: 'Learning', id: 'learning', icon: 'refresh' },
    { to: '/audit', label: 'Audit log', id: 'audit', icon: 'shield' },
  ]
  const initial = (analyst || 'A').trim().slice(0, 1).toUpperCase()

  return (
    <StatusContext.Provider value={status}>
      <div className="app" data-collapsed={collapsed}>
        <aside className="sb" aria-label="Sidebar">
          <Brand collapsed={collapsed} onToggle={() => setCollapsed((c) => !c)} />
          <button type="button" className="sb-item sb-find" onClick={() => setFinding(true)} title="Quick find (Ctrl K)">
            <span className="sb-ico">
              <Icon name="search" size={18} />
            </span>
            <span className="sb-label">Quick find</span>
            <kbd>Ctrl K</kbd>
          </button>
          <nav aria-label="Views" className="sb-nav">
            <Group title="Gate" items={gate} collapsed={collapsed} />
            <Group title="Operate" items={operate} collapsed={collapsed} />
            <Group title="Monitor" items={monitor} collapsed={collapsed} />
          </nav>
          <div className="sb-foot">
            <div className="sb-settings-wrap">
              {menu && (
                <div className="sb-menu" role="menu">
                  <div className="sb-menu-h">Signed in as {analyst}</div>
                  <button type="button" role="menuitem" onClick={signOut} data-testid="sign-out">
                    <Icon name="logout" size={16} /> Sign out
                  </button>
                </div>
              )}
              <button type="button" className="sb-user" aria-expanded={menu} aria-haspopup="menu" onClick={() => setMenu((m) => !m)} title="Account">
                <span className="sb-avatar">{initial}</span>
                <span className="sb-user-t min-w-0">
                  <b title={analyst}>{analyst}</b>
                  <span>Analyst</span>
                </span>
                <Icon name="updown" size={15} className="sb-user-chev" />
              </button>
            </div>
          </div>
        </aside>

        <div className="main-col">
          {MOCK && (
            <div data-testid="mock-banner" role="note" className="fs-tape-banner">
              Mock data: running on built-in fixtures, not the tracd API. All numbers are invented.
            </div>
          )}
          <main id="main" className="main">
            <div className="route-in" key={loc.pathname.replace(/^\/decisions\/.*/, '/decisions')}>
              <Outlet />
            </div>
          </main>
        </div>
        {finding && <QuickFind onClose={() => setFinding(false)} />}
      </div>
    </StatusContext.Provider>
  )
}
