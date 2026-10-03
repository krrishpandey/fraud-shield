import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { Handover, ModelInUse } from '../api/types'
import { shortHash } from '../lib/format'
import { fmtSince, handoverHeadline } from '../lib/handover'

const TONE: Record<Handover['verdict'], { color: string; icon: string; label: string }> = {
  new_model_in_use: { color: 'var(--ok)', icon: '✓', label: 'New model in use' },
  previous_model_kept: { color: 'var(--act-hold)', icon: '!', label: 'Previous model kept' },
  rolled_back: { color: 'var(--act-owner_confirm)', icon: '↺', label: 'Rolled back' },
}

/** Prominent banner after a retrain or rollback: which model scores new bookings from now on, and why. */
export function HandoverBanner({ h, testId = 'handover-banner' }: { h: Handover; testId?: string }) {
  const t = TONE[h.verdict]
  return (
    <div
      role="status"
      data-testid={testId}
      data-verdict={h.verdict}
      data-active-version={h.active_version}
      className="rounded-sm border-2 px-3.5 py-3"
      style={{ borderColor: t.color, background: `color-mix(in srgb, ${t.color} 9%, transparent)` }}
    >
      <div className="text-[0.72rem] font-bold tracking-wide uppercase" style={{ color: t.color }}>
        <span aria-hidden="true">{t.icon} </span>
        {t.label}
        {h.run_id ? ` · run ${h.run_id}` : ''}
      </div>
      <p className="mt-1 text-[1rem] leading-snug font-semibold" data-testid={`${testId}-text`}>
        {handoverHeadline(h)}
      </p>
      {h.verdict === 'previous_model_kept' && h.failed_checks.length > 0 && (
        <ul className="mt-1 list-disc pl-5 text-[0.86rem]" data-testid={`${testId}-reasons`}>
          {h.failed_checks.map((c) => (
            <li key={c.name} data-name={c.name}>
              {c.plain}
            </li>
          ))}
        </ul>
      )}
      <p className="mt-1.5 text-[0.75rem] text-ink-2">
        Rollback target: <span className="font-mono" data-testid={`${testId}-rollback-target`}>{h.rollback_target ?? 'none (no other version has passed the gate)'}</span>
        {h.audit_hash ? (
          <>
            {' '}
            · audit hash <code className="font-mono">{shortHash(h.audit_hash, 12)}</code>
          </>
        ) : null}
      </p>
    </div>
  )
}

/** One line: the GBM version scoring new bookings now, and since when. */
export function ModelInUseLine({ m, testId = 'model-in-use' }: { m: ModelInUse; testId?: string }) {
  return (
    <p className="text-[0.82rem]" data-testid={testId} data-version={m.version}>
      Model in use: <span className="font-mono font-semibold">{m.version}</span> since {fmtSince(m.since)}
      {!m.matches_registry && m.registry_active_version && (
        <span className="ml-1 text-[0.75rem]" style={{ color: 'var(--danger)' }}>
          (the registry's active version is {m.registry_active_version}; this service was started with another scorer)
        </span>
      )}
    </p>
  )
}

/** Self-loading "Model in use" line (polls /learning/status), for views outside Learning. Renders nothing if learning is off. */
export function ModelInUseBadge({ pollMs = 10_000, testId = 'model-in-use' }: { pollMs?: number; testId?: string }) {
  const [m, setM] = useState<ModelInUse | null>(null)
  useEffect(() => {
    let alive = true
    const load = () =>
      api
        .learningStatus()
        .then((s) => alive && setM(s.model_in_use ?? null))
        .catch(() => alive && setM(null))
    load()
    const t = setInterval(load, pollMs)
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [pollMs])
  return m ? <ModelInUseLine m={m} testId={testId} /> : null
}
