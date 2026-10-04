import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { CheckedClaim, ExplanationClaims } from '../api/types'
import '../claims.css'

/*
  "Every sentence checked": the explanation as a list of claims, each with its reason code and a tick or a cross
  from the claim checker (fraudshield/explain/claims.py), plus how the cited reasons agree with what the LightGBM
  model itself leaned on (pred_contrib, summed per reason code). If the language model's claims failed, the template
  is shown and the rejected claims are listed under it, crossed out where they failed.
*/

function plainClaimProblem(p: string): string {
  if (p.startsWith('number not in record: ')) return `${p.slice(22)} is not in the decision record`
  if (p.includes("is not from this claim's cited fields")) return `${p.split(' ')[1]} belongs to a different reason`
  if (p.includes("is not one of this decision's reasons")) return 'This decision does not have that reason'
  if (p.startsWith('the sentence does not talk about')) return 'The sentence is not about the reason it cites'
  if (p.startsWith('field not in record: ')) return `${p.slice(21)} is not a field of this decision`
  if (p.startsWith('id or code not in record: ')) return `${p.slice(26)} is not an id from this booking`
  if (p.startsWith('banned phrase: ')) return `It claims too much (${p.slice(15)})`
  return p
}

const MODE_TEXT: Record<string, string> = {
  json_schema: 'strict JSON schema',
  json_object: 'JSON mode (the strict schema was refused)',
  prose: 'free text (JSON was refused)',
  template: 'no language model',
}

function ClaimRow({ c, crossed }: { c: CheckedClaim; crossed?: boolean }) {
  return (
    <li className="claim" data-testid="claim" data-ok={c.ok} data-reason={c.reason_code}>
      <span className={c.ok ? 'claim-mark ok' : 'claim-mark bad'} role="img" aria-label={c.ok ? 'Passes the check' : 'Fails the check'}>
        {c.ok ? '✓' : '✗'}
      </span>
      <div className="min-w-0">
        <p className={crossed && !c.ok ? 'claim-text is-crossed' : 'claim-text'}>{c.text}</p>
        <p className="claim-reason">
          Cites <span className="claim-code">{c.reason_code || 'no reason'}</span>
          {c.reason_label ? ` (${c.reason_label})` : ''}
          {c.cited_fields.length > 0 && <> from {c.cited_fields.join(', ')}</>}
        </p>
        {!c.ok && (
          <ul className="claim-problems" data-testid="claim-problems">
            {c.problems.map((p) => (
              <li key={p}>{plainClaimProblem(p)}</li>
            ))}
          </ul>
        )}
      </div>
    </li>
  )
}

export function ClaimsChecked({ decisionId, text }: { decisionId: string; text: string }) {
  const key = `${decisionId}\n${text}`
  const [got, setGot] = useState<{ key: string; v: ExplanationClaims | null } | null>(null)
  useEffect(() => {
    let alive = true
    api
      .claims(decisionId)
      .then((x) => alive && setGot({ key, v: x }))
      .catch(() => alive && setGot({ key, v: null }))
    return () => {
      alive = false
    }
  }, [decisionId, key])

  if (!got || got.key !== key) return <p className="claims-sub">Checking each sentence...</p>
  const v = got.v
  if (!v) return null // mock mode or an old backend: the prose check above still stands
  const a = v.attribution
  const ag = a.available ? a.agreement : undefined
  return (
    <div className="claims" data-testid="claims" data-source={v.claims_source} data-mode={v.mode}>
      <h3 className="claims-h">Every sentence checked</h3>
      <p className="claims-sub">
        {v.claims_source === 'llm'
          ? `Each sentence the language model wrote (${MODE_TEXT[v.mode] ?? v.mode}), checked on its own against the decision record.`
          : v.mode === 'prose'
            ? 'The language model wrote free text, so these are the template\'s claims, each checked against the decision record.'
            : 'The template\'s claims, one per reason, each checked against the decision record.'}
      </p>
      {v.claims.length === 0 ? (
        <p className="claims-sub" data-testid="claims-empty">
          No single reason stands out for this decision, so there is no claim to check beyond the action.
        </p>
      ) : (
        <ol className="claims-list" data-testid="claims-list">
          {v.claims.map((c, i) => (
            <ClaimRow key={i} c={c} />
          ))}
        </ol>
      )}
      {v.rejected_claims.length > 0 && (
        <div className="claims-rejected" data-testid="claims-rejected">
          <p className="claims-sub">
            The language model's claims were rejected, so the template above is shown. What it wrote:
          </p>
          <ol className="claims-list">
            {v.rejected_claims.map((c, i) => (
              <ClaimRow key={i} c={c} crossed />
            ))}
          </ol>
        </div>
      )}
      {ag && ag.hits !== null ? (
        <p className="claims-agree" data-testid="claims-agreement" data-hits={ag.hits} data-of={ag.of}>
          <strong>
            Agrees with the model's top reasons: {ag.hits} of {ag.of}.
          </strong>{' '}
          <span className="claims-sub">
            The model leaned most on {ag.model.join(', ')} (LightGBM per-feature contributions, summed per reason).
            {ag.reachable !== null && ag.reachable < ag.of
              ? ` Only ${ag.reachable} of these fired as a rule for this booking, so ${ag.reachable} is the most any checked explanation can cite.`
              : ''}
          </span>
        </p>
      ) : (
        <p className="claims-sub" data-testid="claims-agreement" data-available="false">
          Agreement with the model's own reasons is not available: {a.why ?? 'no positive contributions'}.
        </p>
      )}
    </div>
  )
}
