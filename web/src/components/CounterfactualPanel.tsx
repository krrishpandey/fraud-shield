import { useState } from 'react'
import { api } from '../api/client'
import type { Counterfactual, CounterfactualItem } from '../api/types'
import { fmtMs, fmtPct, shortHash } from '../lib/format'
import { ActionPill } from './ActionBadge'

/*
  "What would change this decision?" ANALYST-ONLY. The smallest changes to fields the booker controls (declared value,
  weight, size, service, a sender the account already used, booking time) that the real model and cost rule would
  treat more softly. These are evasion hints: loaded only on click, every view is written to the audit log, and they
  must never be shown to the booker.
*/

function Item({ it, testId }: { it: CounterfactualItem; testId: string }) {
  return (
    <li className="cf-item" data-testid={testId} data-action={it.action} data-n-changes={it.n_changes}>
      <div className="flex flex-wrap items-center gap-2">
        <ActionPill action={it.action} />
        <span className="tnum text-[0.8rem] text-muted">risk {fmtPct(it.probability, 1)}</span>
        <span className="text-[0.8rem] text-muted">
          {it.n_changes} {it.n_changes === 1 ? 'change' : 'changes'}
        </span>
      </div>
      <p className="mt-1 text-[0.88rem]">
        if {it.changes.map((c) => c.text).join(' and ')}
      </p>
    </li>
  )
}

export function CounterfactualPanel({ decisionId }: { decisionId: string }) {
  const [cf, setCf] = useState<Counterfactual | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  const load = async () => {
    setBusy(true)
    setErr(null)
    try {
      setCf(await api.counterfactual(decisionId))
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="mt-6 max-w-[60ch]" data-testid="counterfactual-panel">
      <h2 className="verdict-h2">What would change this decision?</h2>
      <p className="cf-warning" data-testid="counterfactual-warning">
        Analyst only. These are evasion hints: never show them to the booker. Each view is written to the audit log.
      </p>
      {!cf && (
        <button type="button" className="btn mt-2" onClick={load} disabled={busy} data-testid="counterfactual-load">
          {busy ? 'Trying small changes...' : 'Show what would change it'}
        </button>
      )}
      {err && (
        <p className="mt-2 text-[0.85rem] text-danger" role="alert" data-testid="counterfactual-error">
          {err}
        </p>
      )}
      {cf && (
        <div data-testid="counterfactual-result" data-found={cf.found}>
          {cf.found ? (
            <ul className="cf-list" data-testid="counterfactual-list">
              {cf.counterfactuals.map((it, i) => (
                <Item key={i} it={it} testId="counterfactual-item" />
              ))}
            </ul>
          ) : (
            <>
              <p className="mt-2 text-[0.88rem]" data-testid="counterfactual-none" role="status">
                {cf.message}
              </p>
              {cf.closest && (
                <>
                  <p className="mt-2 text-[0.8rem] text-muted">Closest it came:</p>
                  <ul className="cf-list">
                    <Item it={cf.closest} testId="counterfactual-closest" />
                  </ul>
                </>
              )}
            </>
          )}
          <p className="mt-2 text-[0.75rem] text-muted" data-testid="counterfactual-meta" data-evaluations={cf.evaluations}>
            {cf.evaluations} of {cf.budget} what-if bookings scored by the real model and cost rule, at most{' '}
            {cf.max_fields} fields changed, {fmtMs(cf.latency_ms)}. Audit {shortHash(cf.audit_hash, 12)}.
          </p>
        </div>
      )}
    </div>
  )
}
