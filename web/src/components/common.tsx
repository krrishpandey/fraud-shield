import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { API_LABEL } from '../api/client'
import { useStatus } from '../lib/status'
import { Icon, type IconName } from './Icon'

export function ErrorBox({ message, onRetry, testId }: { message: string; onRetry?: () => void; testId?: string }) {
  return (
    <div role="alert" data-testid={testId ?? 'error-box'} className="panel errbox">
      <Icon name="x" size={18} style={{ color: 'var(--danger)', marginTop: 2 }} />
      <div className="min-w-0 flex-1">
        <p className="font-semibold text-danger">Request failed</p>
        <p className="mt-0.5 text-ink-2">{message}</p>
      </div>
      {onRetry && (
        <button type="button" className="btn" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  )
}

export function Loading({ what }: { what: string }) {
  return (
    <p className="flex items-center gap-2.5 py-2 text-muted" role="status" aria-live="polite">
      <span aria-hidden="true" className="fs-loading-bar" />
      Loading {what}
    </p>
  )
}

export function Section({
  title,
  aside,
  children,
  id,
  testId,
  icon,
  className,
}: {
  title: string
  aside?: ReactNode
  children: ReactNode
  id?: string
  testId?: string
  icon?: IconName
  className?: string
}) {
  const hid = id ? `${id}-h` : undefined
  return (
    <section className={`panel min-w-0 ${className ?? ''}`} aria-labelledby={hid} id={id} data-testid={testId}>
      <div className="card-h">
        <h2 id={hid}>
          {icon && <Icon name={icon} size={18} />}
          {title}
        </h2>
        {aside && <div className="ml-auto text-[13.5px] text-muted">{aside}</div>}
      </div>
      <div className="card-b">{children}</div>
    </section>
  )
}

/** The CPU · latency · open-cases cluster from the design, plus history and help buttons. */
function StatusCluster() {
  const { health, healthError, latencyMs, openCases } = useStatus()
  let state = 'unknown'
  let word = 'Checking'
  let tone = 'var(--muted)'
  if (healthError) {
    state = 'down'
    word = 'API unreachable'
    tone = 'var(--danger)'
  } else if (health) {
    const degraded = Boolean(health.degraded) || !health.ok
    state = degraded ? 'degraded' : 'ok'
    word = degraded ? 'Degraded' : 'Online'
    tone = degraded ? 'var(--indigo)' : 'var(--ok)'
  }
  const layaWord = health ? (health.laya_mode === 'cached' ? 'Laya: cached answers' : `Laya: ${health.laya_mode}`) : ''
  return (
    <div className="ph-right">
      <div className="cluster">
        <span
          className="cluster-seg"
          data-testid="health-status"
          data-state={state}
          data-laya-mode={health?.laya_mode ?? ''}
          data-gpu={health ? String(health.gpu) : ''}
          role="status"
          aria-live="polite"
          title={healthError ? `${API_LABEL}: ${healthError}` : `${word} · ${layaWord} · API ${API_LABEL}`}
        >
          <span className="dot" style={{ background: tone }} aria-hidden="true" />
          <span className="sr-only">{word}. {layaWord}. </span>
          <b>{health ? (health.gpu ? 'GPU' : 'CPU') : '—'}</b>
          <span className="sr-only">{health && !health.gpu ? ' only' : ''}</span>
          {health?.laya_mode === 'cached' && (
            <span className="cluster-tag" title="Laya answers come from a cache">
              Cached
            </span>
          )}
        </span>
        <span className="cluster-seg" title="Median decision time">
          <Icon name="pulse" size={16} />
          <b className="tnum">{latencyMs == null ? '—' : `${Math.round(latencyMs)} ms`}</b>
        </span>
        <Link to="/queue" className="cluster-seg" title="Open cases waiting for an analyst">
          <Icon name="list" size={16} />
          <b className="tnum">{openCases.length}</b>
        </Link>
      </div>
      <Link to="/audit" className="ph-icon" aria-label="Audit log" title="Audit log">
        <Icon name="history" size={18} />
      </Link>
      <a className="ph-icon" href="#/" aria-label="Help: start from Score a booking" title="Start from Score a booking">
        <Icon name="help" size={18} />
      </a>
    </div>
  )
}

/** Page header bar: title, optional mono subtitle, page controls, then the status cluster. */
export function PageTitle({ title, sub, children }: { title: string; sub?: ReactNode; kicker?: ReactNode; children?: ReactNode }) {
  return (
    <header className="ph">
      <div className="ph-left">
        <h1>{title}</h1>
        {sub && <span className="ph-sub">{sub}</span>}
        {children && <div className="ph-controls">{children}</div>}
      </div>
      <StatusCluster />
    </header>
  )
}

/** Page body wrapper with the design's 28px page padding. */
export function Page({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={`page ${className ?? ''}`}>{children}</div>
}
