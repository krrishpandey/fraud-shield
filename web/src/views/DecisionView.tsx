import { useEffect, useState, type FormEvent } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api/client'
import { ACTIONS, SERVED_QUESTIONS, type AskResponse, type DecisionDetail } from '../api/types'
import { ActionPill, ActionStamp } from '../components/ActionBadge'
import { ErrorBox, Loading, Section } from '../components/common'
import { ACTION_META, QUESTION_META, actionLabel, featureLabel, reasonText } from '../lib/domain'
import { fmtBRL, fmtMs, fmtNum, fmtPct, shortHash } from '../lib/format'

const POLL_MS = 1000
const POLL_MAX = 90

function useDecision(id: string) {
  const [data, setData] = useState<DecisionDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [polling, setPolling] = useState(false)
  const [tick, setTick] = useState(0)

  useEffect(() => {
    let alive = true
    let timer: ReturnType<typeof setTimeout> | undefined
    let n = 0
    const load = async () => {
      try {
        const d = await api.decision(id)
        if (!alive) return
        setError(null)
        setData(d)
        const pending = d.explanation == null && (d.explanation_status === 'pending' || d.explanation_status == null)
        if (pending && n++ < POLL_MAX) {
          setPolling(true)
          timer = setTimeout(load, POLL_MS)
        } else {
          setPolling(false)
        }
      } catch (e) {
        if (alive) {
          setError(e instanceof Error ? e.message : String(e))
          setPolling(false)
        }
      }
    }
    load()
    return () => {
      alive = false
      if (timer) clearTimeout(timer)
    }
  }, [id, tick])

  return { data, error, polling, reload: () => setTick((t) => t + 1), setData }
}

/* ---------- Pieces ---------- */

const LATENCY_STAGES: { key: string; label: string; slot: number }[] = [
  { key: 'features', label: 'Features', slot: 1 },
  { key: 'laya', label: 'Laya (4 questions)', slot: 2 },
  { key: 'decide', label: 'Decision rule', slot: 3 },
  { key: 'audit', label: 'Audit write', slot: 7 },
]

function LatencyBreakdown({ lat }: { lat: DecisionDetail['latency_ms'] }) {
  const total = lat.total ?? 0
  const known = LATENCY_STAGES.filter((s) => typeof lat[s.key] === 'number')
  const extra = Object.keys(lat).filter((k) => k !== 'total' && !LATENCY_STAGES.some((s) => s.key === k))
  const parts = [
    ...known.map((s) => ({ label: s.label, ms: lat[s.key], color: `var(--series-${s.slot})` })),
    ...extra.map((k) => ({ label: featureLabel(k), ms: lat[k], color: 'var(--series-5)' })),
  ]
  const sum = parts.reduce((a, p) => a + p.ms, 0)
  if (total - sum > 0.05) parts.push({ label: 'Network, framework', ms: total - sum, color: 'var(--axis)' })
  const denom = Math.max(total, sum) || 1
  return (
    <div data-testid="latency-breakdown" data-total={total}>
      <div className="mb-1.5 flex items-baseline gap-2">
        <span className="tnum text-xl font-bold">{fmtMs(total)}</span>
        <span className="text-[0.8rem] text-muted">total, synchronous, before the label is issued</span>
      </div>
      <div className="flex h-3.5 gap-[2px] overflow-hidden rounded-sm" role="img" aria-label={parts.map((p) => `${p.label} ${fmtMs(p.ms)}`).join(', ')}>
        {parts.map((p) => (
          <div key={p.label} title={`${p.label}: ${fmtMs(p.ms)}`} style={{ width: `${(p.ms / denom) * 100}%`, background: p.color, minWidth: p.ms > 0 ? 2 : 0 }} />
        ))}
      </div>
      <ul className="mt-2 grid grid-cols-2 gap-x-4 gap-y-0.5 text-[0.8rem]">
        {parts.map((p) => (
          <li key={p.label} className="flex items-center gap-1.5">
            <span aria-hidden="true" className="inline-block h-2.5 w-2.5 rounded-[2px]" style={{ background: p.color }} />
            <span className="text-ink-2">{p.label}</span>
            <span className="tnum ml-auto whitespace-nowrap">{fmtMs(p.ms)}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function ProbBars({ d }: { d: DecisionDetail }) {
  const extra = Object.keys(d.probabilities).filter((k) => !(SERVED_QUESTIONS as readonly string[]).includes(k))
  const qids = [...SERVED_QUESTIONS.filter((q) => q in d.probabilities || q in d.raw_probabilities), ...extra]
  return (
    <div>
      <ul className="flex flex-col gap-3">
        {qids.map((q) => {
          const meta = QUESTION_META[q as keyof typeof QUESTION_META]
          const cal = d.probabilities[q]
          const raw = d.raw_probabilities[q]
          return (
            <li key={q} data-testid={`prob-${q}`} data-calibrated={cal ?? ''} data-raw={raw ?? ''}>
              <div className="flex flex-wrap items-baseline justify-between gap-x-3">
                <span className="font-semibold">
                  {meta?.title ?? featureLabel(q)}
                  {meta?.zeroShot && (
                    <span className="ml-2 rounded-sm border border-rule px-1 text-[0.7rem] font-semibold text-ink-2">zero-shot, not trained</span>
                  )}
                </span>
                <span className="tnum text-[0.85rem]">
                  <strong>{fmtPct(cal)}</strong>
                  <span className="text-muted"> calibrated, {fmtPct(raw)} raw</span>
                </span>
              </div>
              {meta && <div className="text-[0.78rem] text-muted">{meta.question}</div>}
              <div className="prob-track mt-1" role="img" aria-label={`${meta?.title ?? q}: calibrated ${fmtPct(cal)}, raw ${fmtPct(raw)}`}>
                {cal != null && (
                  <div className="absolute inset-y-0 left-0 rounded-l-[2px]" style={{ width: `${cal * 100}%`, background: 'var(--series-1)' }} />
                )}
                {raw != null && (
                  <div
                    className="absolute -top-[3px] -bottom-[3px] w-[3px] -translate-x-1/2 rounded-[1px]"
                    style={{ left: `${raw * 100}%`, background: 'var(--ink)', boxShadow: '0 0 0 2px var(--surface)' }}
                  />
                )}
              </div>
            </li>
          )
        })}
      </ul>
      <div className="mt-3 flex flex-wrap gap-4 text-[0.75rem] text-muted">
        <span className="flex items-center gap-1.5">
          <span aria-hidden="true" className="inline-block h-2.5 w-4 rounded-[2px]" style={{ background: 'var(--series-1)' }} />
          Calibrated probability (used for the decision)
        </span>
        <span className="flex items-center gap-1.5">
          <span aria-hidden="true" className="inline-block h-3 w-[3px]" style={{ background: 'var(--ink)' }} />
          Raw Laya output before calibration
        </span>
        {d.gbm_score != null && <span>Backup GBM score: {fmtPct(d.gbm_score)}</span>}
      </div>
    </div>
  )
}

function CostTable({ d }: { d: DecisionDetail }) {
  const costs = d.expected_costs
  const vals = ACTIONS.map((a) => costs[a]).filter((v): v is number => typeof v === 'number')
  const max = Math.max(...vals, 1)
  return (
    <div>
      <table className="w-full text-[0.85rem]" data-testid="cost-table">
        <caption className="sr-only">Expected cost per action, in Brazilian reais</caption>
        <thead>
          <tr className="border-b border-rule text-left text-[0.75rem] text-muted">
            <th scope="col" className="py-1 font-semibold">Action</th>
            <th scope="col" className="py-1 text-right font-semibold whitespace-nowrap">Expected cost (R$)</th>
            <th scope="col" className="w-[38%] py-1 pl-3 font-semibold">
              <span className="sr-only">Relative cost</span>
            </th>
            <th scope="col" className="py-1 pl-2 font-semibold">
              <span className="sr-only">Marks</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {ACTIONS.map((a) => {
            const v = costs[a]
            const chosen = a === d.action
            const greedy = a === d.greedy_action
            return (
              <tr
                key={a}
                data-testid={`cost-row-${a}`}
                data-chosen={chosen}
                data-greedy={greedy}
                className="border-b border-rule last:border-0"
                style={chosen ? { background: 'var(--surface-2)', boxShadow: `inset 3px 0 0 var(--act-${a})` } : undefined}
              >
                <th scope="row" className="py-1.5 pl-2 text-left font-normal">
                  <ActionPill action={a} />
                </th>
                <td className="tnum py-1.5 text-right">{typeof v === 'number' ? fmtBRL(v) : 'n/a'}</td>
                <td className="py-1.5 pl-3">
                  {typeof v === 'number' && (
                    <div className="h-2 rounded-r-[2px]" style={{ width: `${Math.max((v / max) * 100, 1)}%`, background: chosen ? `var(--act-${a})` : 'var(--axis)' }} />
                  )}
                </td>
                <td className="py-1.5 pl-2 text-[0.75rem] whitespace-nowrap">
                  {chosen && <span className="mr-1 rounded-sm bg-brand px-1.5 py-0.5 font-semibold text-brand-ink">Chosen</span>}
                  {greedy && <span className="rounded-sm border border-rule px-1.5 py-0.5 text-ink-2">Lowest cost</span>}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="mt-2 text-[0.75rem] text-muted" data-testid="cost-note">
        Costs are assumptions: estimated loss if fraud gets through plus friction to a legitimate shipper, weighted by the
        calibrated probabilities. The rule picks the lowest expected cost unless a guardrail or an exploration sample applies
        {d.explored ? ' (this booking was an exploration sample)' : ''}. Logged propensity {fmtNum(d.propensity)}.
      </p>
    </div>
  )
}

function Explanation({ d, polling }: { d: DecisionDetail; polling: boolean }) {
  const e = d.explanation
  if (!e) {
    const failed = d.explanation_status === 'failed'
    return (
      <p className="text-ink-2" data-testid="explanation-status" data-status={d.explanation_status} role="status" aria-live="polite">
        {failed ? 'No explanation was produced for this decision.' : polling ? 'Writing the explanation. It is checked against the decision record before it is shown...' : 'Explanation still pending. Reload to check again.'}
      </p>
    )
  }
  return (
    <div data-testid="explanation" aria-live="polite">
      <p className="max-w-[75ch] text-[0.95rem] leading-relaxed" data-testid="explanation-text">
        {e.text}
      </p>
      <div className="mt-2 flex flex-wrap gap-2 text-[0.75rem]">
        <span className="rounded-sm border border-rule px-1.5 py-0.5" data-testid="explanation-source" data-source={e.source}>
          {e.source === 'llm' ? `Written by a language model${e.model_id ? ` (${e.model_id})` : ''}` : 'Fixed template'}
        </span>
        <span
          className="rounded-sm border px-1.5 py-0.5"
          data-testid="explanation-validator"
          data-valid={e.valid}
          style={{ borderColor: e.valid ? 'var(--ok)' : 'var(--danger)' }}
        >
          {e.valid
            ? '✓ Validator passed: every number and id matches the decision record'
            : '✕ Validator failed: showing the template instead'}
        </span>
      </div>
    </div>
  )
}

const EXAMPLE_Q = {
  instructions: 'Does the consignee look like a reshipping drop: a recent address receiving parcels from many unrelated senders?',
  yes: 'likely a drop address',
  no: 'an ordinary consignee',
}

function AskPanel({ id }: { id: string }) {
  const [q, setQ] = useState({ instructions: '', yes: '', no: '' })
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [history, setHistory] = useState<(AskResponse & { instructions: string })[]>([])
  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setErr(null)
    try {
      const r = await api.ask(id, {
        instructions: q.instructions.trim(),
        yes: q.yes.trim() || 'yes',
        no: q.no.trim() || 'no',
      })
      setHistory((h) => [{ ...r, instructions: q.instructions.trim() }, ...h])
    } catch (e2) {
      setErr(e2 instanceof Error ? e2.message : String(e2))
    } finally {
      setBusy(false)
    }
  }
  const latest = history[0]
  return (
    <form onSubmit={submit} aria-label="Ask a new question">
      <label htmlFor="ask-input" className="block text-[0.78rem] text-muted">
        Yes or no question about this booking
      </label>
      <textarea
        id="ask-input"
        data-testid="ask-input"
        className="field min-h-[3.6rem] resize-y"
        required
        value={q.instructions}
        placeholder="Does the consignee look like a reshipping drop?"
        onChange={(e) => setQ({ ...q, instructions: e.target.value })}
      />
      <div className="mt-2 grid grid-cols-2 gap-2">
        <div>
          <label htmlFor="ask-yes" className="block text-[0.78rem] text-muted">
            What yes means
          </label>
          <input id="ask-yes" data-testid="ask-yes" className="field" value={q.yes} placeholder="likely a drop address" onChange={(e) => setQ({ ...q, yes: e.target.value })} />
        </div>
        <div>
          <label htmlFor="ask-no" className="block text-[0.78rem] text-muted">
            What no means
          </label>
          <input id="ask-no" data-testid="ask-no" className="field" value={q.no} placeholder="an ordinary consignee" onChange={(e) => setQ({ ...q, no: e.target.value })} />
        </div>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <button type="submit" className="btn btn-primary" disabled={busy || !q.instructions.trim()} data-testid="ask-submit">
          {busy ? 'Asking...' : 'Ask Laya'}
        </button>
        <button type="button" className="btn" onClick={() => setQ(EXAMPLE_Q)} data-testid="ask-example">
          Use the drop example
        </button>
      </div>
      {err && <p className="mt-2 text-[0.85rem] text-danger" role="alert">{err}</p>}
      <div aria-live="polite">
        {latest && (
          <div className="mt-3 rounded-sm border border-rule bg-surface-2 p-3" data-testid="ask-result" data-probability={latest.probability_yes} data-qid={latest.qid}>
            <div className="flex items-baseline gap-2">
              <span className="tnum text-2xl font-bold">{fmtPct(latest.probability_yes)}</span>
              <span className="text-[0.85rem] text-ink-2">probability of yes</span>
              <span className="tnum ml-auto text-[0.75rem] text-muted">{fmtMs(latest.latency_ms)}</span>
            </div>
            <div className="prob-track mt-1.5">
              <div className="absolute inset-y-0 left-0 rounded-l-[2px]" style={{ width: `${latest.probability_yes * 100}%`, background: 'var(--series-1)' }} />
            </div>
            <p className="mt-1.5 text-[0.75rem] text-muted">
              Answered zero-shot by Laya from the same state text. No retraining. {latest.calibrated ? 'Calibrated.' : 'Not calibrated, so read it as a ranking signal, not an exact rate.'}
            </p>
          </div>
        )}
        {history.length > 1 && (
          <ul className="mt-2 text-[0.78rem] text-ink-2">
            {history.slice(1).map((h) => (
              <li key={h.qid} className="tnum">
                {fmtPct(h.probability_yes)} {h.instructions}
              </li>
            ))}
          </ul>
        )}
      </div>
    </form>
  )
}

function AnalystPanel({ d, onDone }: { d: DecisionDetail; onDone: () => void }) {
  const [note, setNote] = useState('')
  const [busy, setBusy] = useState<null | 'fraud' | 'legit'>(null)
  const [err, setErr] = useState<string | null>(null)
  const [result, setResult] = useState<{ label: string; hash: string } | null>(null)
  const send = async (label: 'fraud' | 'legit') => {
    setBusy(label)
    setErr(null)
    try {
      const r = await api.analyst(d.decision_id, { label, note: note.trim() })
      setResult({ label, hash: r.audit_hash })
      onDone()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }
  return (
    <div>
      {d.analyst && !result && (
        <p className="mb-2 text-[0.85rem]" data-testid="analyst-existing" data-label={d.analyst.label}>
          Labelled <strong>{d.analyst.label === 'fraud' ? 'fraud' : 'legitimate'}</strong> on {d.analyst.at.replace('T', ' ').slice(0, 16)}
          {d.analyst.note ? `: ${d.analyst.note}` : ''}
        </p>
      )}
      <label htmlFor="analyst-note" className="block text-[0.78rem] text-muted">
        Note for the audit record
      </label>
      <textarea
        id="analyst-note"
        data-testid="analyst-note"
        className="field min-h-[2.8rem] resize-y"
        value={note}
        placeholder="What you checked and why"
        onChange={(e) => setNote(e.target.value)}
      />
      <div className="mt-2 flex flex-wrap gap-2">
        <button type="button" className="btn" style={{ borderColor: 'var(--act-block)' }} disabled={busy !== null} onClick={() => send('fraud')} data-testid="analyst-confirm-fraud">
          {busy === 'fraud' ? 'Saving...' : 'Confirm fraud'}
        </button>
        <button type="button" className="btn" style={{ borderColor: 'var(--act-allow)' }} disabled={busy !== null} onClick={() => send('legit')} data-testid="analyst-mark-legit">
          {busy === 'legit' ? 'Saving...' : 'Mark legitimate'}
        </button>
      </div>
      {err && <p className="mt-2 text-[0.85rem] text-danger" role="alert">{err}</p>}
      {result && (
        <div className="mt-3 rounded-sm border border-rule bg-surface-2 p-2.5 text-[0.82rem]" role="status" data-testid="analyst-result" data-label={result.label}>
          {result.label === 'fraud' ? 'Fraud confirmed' : 'Marked legitimate'} and added to the audit log and the label store.
          <div className="mt-1">
            Audit hash{' '}
            <code className="font-mono text-[0.78rem] break-all" data-testid="analyst-result-hash">
              {result.hash}
            </code>
          </div>
        </div>
      )}
    </div>
  )
}

/* ---------- Page ---------- */

export default function DecisionView() {
  const { id = '' } = useParams()
  const { data: d, error, polling, reload } = useDecision(id)
  const [stateOpen, setStateOpen] = useState(false)

  if (error && !d) return <ErrorBox message={error} onRetry={reload} testId="decision-error" />
  if (!d) return <Loading what="decision" />

  const b = d.booking
  return (
    <div data-testid="decision-detail" data-decision-id={d.decision_id}>
      <div className="mb-3 flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <Link to="/" className="text-[0.8rem] text-ink-2 underline underline-offset-2">
          Score another booking
        </Link>
        <h1 className="condensed text-[1.45rem] font-extrabold tracking-tight">Decision {d.decision_id}</h1>
        <span className="text-[0.85rem] text-ink-2">
          Booking {d.booking_id}
          {b ? `, account ${b.account_id}, ${b.origin_uf}-${b.origin_zip3} to ${b.dest_uf}-${b.dest_zip3}, ${fmtBRL(b.carrier_cost)}, booked ${b.booked_at.replace('T', ' ').slice(0, 16)}` : ''}
        </span>
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <div className="flex min-w-0 flex-col gap-4">
          <ActionStamp action={d.action} degraded={d.degraded} explored={d.explored} />

          <Section title="Why" aside={polling ? 'checking for explanation...' : undefined} testId="explanation-panel">
            <Explanation d={d} polling={polling} />
            {d.reasons.length > 0 && (
              <div className="mt-3">
                <h3 className="text-[0.78rem] font-semibold text-muted">Reason codes</h3>
                <ul className="mt-1 flex flex-col gap-0.5 text-[0.85rem]" data-testid="reason-codes">
                  {d.reasons.map((r) => (
                    <li key={r} data-code={r}>
                      {reasonText(r)} <code className="font-mono text-[0.72rem] text-muted">{r}</code>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </Section>

          <Section title="What Laya answered" aside="four named questions, one batched call">
            <ProbBars d={d} />
          </Section>

          <Section title="Top features against the account's own baseline">
            <table className="w-full text-[0.85rem]" data-testid="top-features">
              <thead>
                <tr className="border-b border-rule text-left text-[0.75rem] text-muted">
                  <th scope="col" className="py-1 font-semibold">Feature</th>
                  <th scope="col" className="py-1 text-right font-semibold">This booking</th>
                  <th scope="col" className="py-1 text-right font-semibold">Account baseline</th>
                </tr>
              </thead>
              <tbody>
                {d.top_features.map((f) => (
                  <tr key={f.name} className="border-b border-rule last:border-0">
                    <th scope="row" className="py-1 text-left font-normal">
                      {featureLabel(f.name)}
                    </th>
                    <td className="tnum py-1 text-right font-semibold">{fmtNum(f.value)}</td>
                    <td className="tnum py-1 text-right text-ink-2">{fmtNum(f.baseline)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Section>

          <section className="panel p-4" aria-labelledby="state-h">
            <button
              type="button"
              className="flex w-full items-center justify-between text-left"
              aria-expanded={stateOpen}
              aria-controls="state-body"
              onClick={() => setStateOpen((o) => !o)}
              data-testid="state-text-toggle"
            >
              <h2 id="state-h" className="text-[0.95rem] font-bold">What the model read</h2>
              <span className="text-[0.8rem] text-ink-2">{stateOpen ? 'Hide' : 'Show'} state text</span>
            </button>
            {stateOpen && (
              <div id="state-body">
                <p className="mt-1 text-[0.78rem] text-muted">The exact fixed-order text Laya scored. No free text from the shipper goes in.</p>
                <pre className="mt-2 overflow-x-auto rounded-sm border border-rule bg-surface-2 p-3 font-mono text-[0.78rem] leading-relaxed whitespace-pre-wrap" data-testid="state-text">
                  {d.state_text}
                </pre>
              </div>
            )}
          </section>
        </div>

        <div className="flex min-w-0 flex-col gap-4">
          <Section title="Latency">
            <LatencyBreakdown lat={d.latency_ms} />
          </Section>

          <Section title="Expected cost per action" aside={`chosen: ${actionLabel(d.action)}`}>
            <CostTable d={d} />
          </Section>

          <Section title="Ask a new question" testId="ask-panel">
            <AskPanel id={d.decision_id} />
          </Section>

          <Section title="Analyst decision" testId="analyst-panel">
            <AnalystPanel d={d} onDone={reload} />
          </Section>

          <Section title="Record">
            <dl className="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-0.5 text-[0.78rem]">
              <dt className="text-muted">Audit hash</dt>
              <dd className="font-mono break-all" data-testid="decision-audit-hash" title={d.audit_hash}>
                {shortHash(d.audit_hash, 24)}
              </dd>
              <dt className="text-muted">Action meaning</dt>
              <dd>{ACTION_META[d.action]?.label_status}</dd>
              {Object.entries(d.model_versions ?? {}).map(([k, v]) => (
                <div key={k} className="contents">
                  <dt className="text-muted">{featureLabel(k)} version</dt>
                  <dd className="font-mono break-all">{v}</dd>
                </div>
              ))}
            </dl>
          </Section>
        </div>
      </div>
    </div>
  )
}
