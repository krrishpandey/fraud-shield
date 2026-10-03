import type { Action } from '../api/types'
import { ACTION_META } from '../lib/domain'

const actVar = (a: string) => `var(--act-${a}, var(--muted))`

/** Large routing-label stamp for the decision detail page. */
export function ActionStamp({ action, degraded, explored }: { action: Action; degraded?: boolean; explored?: boolean }) {
  const meta = ACTION_META[action]
  return (
    <div
      className="route-label"
      style={{ ['--act' as string]: actVar(action) }}
      data-testid="action-badge"
      data-action={action}
      role="status"
      aria-label={`Decision: ${meta?.label ?? action}`}
    >
      <div className="band" aria-hidden="true" />
      <div className="px-4 pt-2.5 pb-3">
        <div className="flex items-baseline gap-3">
          <span aria-hidden="true" className="text-3xl leading-none" style={{ color: actVar(action) }}>
            {meta?.glyph}
          </span>
          <span className="condensed text-[2.2rem] leading-none font-extrabold tracking-tight">{meta?.label ?? action}</span>
        </div>
        <p className="mt-2 max-w-[60ch] text-ink-2">{meta?.meaning}</p>
        <div className="mt-3 flex flex-wrap gap-2 text-sm">
          <span className="rounded-sm border border-rule px-2 py-0.5 font-semibold" style={{ borderColor: actVar(action) }}>
            {meta?.label_status}
          </span>
          {degraded && (
            <span className="rounded-sm border border-danger px-2 py-0.5 font-semibold text-danger" data-testid="degraded-flag">
              Degraded: Laya unavailable, decided by the backup model with stricter thresholds
            </span>
          )}
          {explored && (
            <span className="rounded-sm border border-rule px-2 py-0.5" data-testid="explored-flag">
              Exploration sample: routed to scan-gated allow to measure the policy
            </span>
          )}
        </div>
      </div>
    </div>
  )
}

/** Small inline pill for tables and lists. Text always present. */
export function ActionPill({ action, testId }: { action: string; testId?: string }) {
  const meta = (ACTION_META as Record<string, (typeof ACTION_META)[Action]>)[action]
  return (
    <span
      data-testid={testId}
      data-action={action}
      className="inline-flex items-center gap-1.5 rounded-sm border px-1.5 py-0.5 text-[0.8rem] font-semibold whitespace-nowrap"
      style={{ borderColor: actVar(action), color: 'var(--ink)' }}
    >
      <span aria-hidden="true" style={{ color: actVar(action) }}>
        {meta?.glyph ?? '•'}
      </span>
      {meta?.short ?? action}
    </span>
  )
}
