import type { Action } from '../api/types'
import { ACTION_META } from '../lib/domain'

const actVar = (a: string) => `var(--act-${a}, var(--muted))`
const actBg = (a: string) => `var(--act-${a}-bg, var(--surface-2))`

/** One glyph per action, drawn so it reads without colour. */
export function ActionIcon({ action, size = 14 }: { action: string; size?: number }) {
  const p = { fill: 'none', stroke: 'currentColor', strokeWidth: 2.4 } as const
  let g
  switch (action) {
    case 'allow':
      g = <path d="M4.5 12.5l5 5L19.5 7" {...p} strokeWidth={2.8} strokeLinecap="square" />
      break
    case 'allow_scan_gated':
      g = <path d="M3.5 8V3.5H8M16 3.5h4.5V8M20.5 16v4.5H16M8 20.5H3.5V16M3 12h18" {...p} />
      break
    case 'owner_confirm':
      g = (
        <>
          <circle cx="12" cy="8" r="4" {...p} />
          <path d="M4 21c1-4.2 4.2-6.2 8-6.2s7 2 8 6.2" {...p} />
        </>
      )
      break
    case 'review':
      g = (
        <>
          <circle cx="10.5" cy="10.5" r="6" {...p} />
          <path d="M15 15l6 6" {...p} strokeWidth={2.8} />
        </>
      )
      break
    case 'hold':
      g = (
        <>
          <rect x="5.5" y="4" width="4.5" height="16" fill="currentColor" />
          <rect x="14" y="4" width="4.5" height="16" fill="currentColor" />
        </>
      )
      break
    case 'block':
      g = (
        <>
          <circle cx="12" cy="12" r="8.5" {...p} strokeWidth={2.8} />
          <path d="M6 18L18 6" {...p} strokeWidth={2.8} />
        </>
      )
      break
    default:
      g = <circle cx="12" cy="12" r="4" fill="currentColor" />
  }
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" style={{ flex: 'none' }}>
      {g}
    </svg>
  )
}

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
        <div className="flex items-center gap-3" style={{ color: actVar(action) }}>
          <ActionIcon action={action} size={30} />
          <span className="disp text-[2.6rem]">{meta?.label ?? action}</span>
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

/** Small inline chip for tables and lists. Text always present. */
export function ActionPill({ action, testId }: { action: string; testId?: string }) {
  const meta = (ACTION_META as Record<string, (typeof ACTION_META)[Action]>)[action]
  return (
    <span
      data-testid={testId}
      data-action={action}
      className="act-chip"
      style={{ ['--act' as string]: actVar(action), ['--act-bg' as string]: actBg(action) }}
    >
      {meta?.short ?? action}
    </span>
  )
}
