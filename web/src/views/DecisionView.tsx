import { useEffect, useState, type FormEvent } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api/client'
import { ACTIONS, SERVED_QUESTIONS, type AskResponse, type DecisionDetail } from '../api/types'
import { AccountStory } from '../components/AccountStory'
import { ActionPill } from '../components/ActionBadge'
import { ErrorBox, Loading, Page, PageTitle, Section } from '../components/common'
import { SlideInd } from '../components/SlideInd'
import { Icon, type IconName } from '../components/Icon'
import { rememberDecision, useStatus } from '../lib/status'
import { CounterfactualPanel } from '../components/CounterfactualPanel'
import { FactCheckedExplanation } from '../components/FactCheckedExplanation'
import { OwnerPasskey } from '../components/OwnerPasskey'
import { ShippingLabel } from '../components/ShippingLabel'
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
                  <strong>{fmtPct(cal, 1)}</strong>
                  <span className="text-muted">
                    {d.degraded ? ' backup model score' : ` calibrated, ${fmtPct(raw, 1)} raw`}
                  </span>
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
  const priced = ACTIONS.filter((a) => typeof costs[a] === 'number')
  const cheapest = priced.length ? priced.reduce((best, a) => ((costs[a] as number) < (costs[best] as number) ? a : best)) : null
  return (
    <div>
      <table className="tbl cost-tbl" data-testid="cost-table">
        <caption className="sr-only">Expected cost per action, in Brazilian reais</caption>
        <thead>
          <tr>
            <th scope="col">Action</th>
            <th scope="col" className="w-[46%]">Expected cost</th>
            <th scope="col" className="r">R$</th>
            <th scope="col">
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
              <tr key={a} data-testid={`cost-row-${a}`} data-chosen={chosen} data-greedy={greedy} className={chosen ? 'is-chosen' : ''}>
                <th scope="row" className="text-left font-normal">
                  <ActionPill action={a} />
                </th>
                <td>
                  {typeof v === 'number' && (
                    <div className="cost-track">
                      <span style={{ width: `${Math.max((v / max) * 100, 2)}%`, background: chosen ? `var(--act-${a})` : '#4a4a52' }} />
                    </div>
                  )}
                </td>
                <td className="r tnum mono">{typeof v === 'number' ? fmtBRL(v) : 'n/a'}</td>
                <td className="whitespace-nowrap text-[13.5px]">
                  {chosen && <b style={{ color: `var(--act-${a})` }}>Chosen</b>}
                  {a === cheapest && !chosen && <span className="text-muted">Lowest cost</span>}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      {cheapest && cheapest !== d.action && (
        <p className="note mt-3" data-testid="cost-override">
          <Icon name="info" size={16} />
          <span>
            {d.policy_trace?.block_downgraded && cheapest === 'block'
              ? 'Block had the lowest expected cost, but a block needs a hard signal (a link to confirmed fraud) and is limited to one per account a day, so the booking was held instead.'
              : `${actionLabel(cheapest)} had the lowest expected cost, but a guardrail${d.explored ? ' or an exploration sample' : ''} chose ${actionLabel(d.action)}.`}
          </span>
        </p>
      )}
      <p className="mt-3 text-[13px] text-muted" data-testid="cost-note">
        Costs are assumptions: estimated loss if fraud gets through plus friction to a legitimate shipper, weighted by the calibrated
        probabilities. The rule picks the lowest expected cost unless a guardrail or an exploration sample applies
        {d.explored ? ' (this booking was an exploration sample)' : ''}. Logged propensity {fmtNum(d.propensity)}.
      </p>
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
              Answered by {latest.answered_by ?? 'Laya'} from the same input text, with no retraining.{' '}
              {latest.calibrated ? 'Calibrated.' : 'Not calibrated, so read it as a ranking signal, not an exact rate.'}
              {latest.answered_by?.includes('not fine-tuned') && ' The base model was not trained on parcel fraud, so treat this as a lead, not evidence.'}
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
    <div className="analyst">
      {(result || d.analyst || err) && (
        <div className="analyst-msg">
          {d.analyst && !result && (
            <p data-testid="analyst-existing" data-label={d.analyst.label}>
              <span className={d.analyst.label === 'fraud' ? 'text-danger' : 'text-ok'}>
                {d.analyst.label === 'fraud' ? '✕ Labelled fraud' : '✓ Labelled legitimate'}
              </span>{' '}
              on {d.analyst.at.replace('T', ' ').slice(0, 16)}
              {d.analyst.note ? `: ${d.analyst.note}` : ''}
            </p>
          )}
          {err && (
            <p className="text-danger" role="alert">
              {err}
            </p>
          )}
          {result && (
            <p role="status" data-testid="analyst-result" data-label={result.label}>
              <span className={result.label === 'fraud' ? 'text-danger' : 'text-ok'}>{result.label === 'fraud' ? '✕ Fraud confirmed' : '✓ Marked legitimate'}</span>{' '}
              and added to the audit log and the training labels. Audit hash{' '}
              <code className="text-[12.5px] break-all text-ink-2" data-testid="analyst-result-hash">
                {result.hash}
              </code>
            </p>
          )}
        </div>
      )}
      <div className="analyst-row">
        <label className="analyst-note">
          <Icon name="lines" size={17} />
          <span className="sr-only">Note for the audit record</span>
          <textarea
            id="analyst-note"
            data-testid="analyst-note"
            rows={1}
            value={note}
            placeholder="Note for the audit record — what you checked and why"
            onChange={(e) => setNote(e.target.value)}
          />
        </label>
        <button type="button" className="btn" disabled={busy !== null} onClick={() => send('legit')} data-testid="analyst-mark-legit">
          <Icon name="check" size={17} />
          {busy === 'legit' ? 'Saving...' : 'Mark legitimate'}
        </button>
        <button type="button" className="btn btn-danger" disabled={busy !== null} onClick={() => send('fraud')} data-testid="analyst-confirm-fraud">
          <Icon name="flag" size={17} />
          {busy === 'fraud' ? 'Saving...' : 'Confirm fraud'}
        </button>
      </div>
    </div>
  )
}

/* ---------- Depot scan (first-scan check) ---------- */

function DepotScan({ d, onDone }: { d: DecisionDetail; onDone: () => void }) {
  const declared = d.booking?.weight_kg ?? 0
  const [w, setW] = useState(String(declared))
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const scan = d.first_scan
  const send = async () => {
    setBusy(true)
    setErr(null)
    try {
      await api.firstScan(d.decision_id, Number(w))
      onDone()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }
  return (
    <Section title="Depot scan" icon="scale" testId="depot-scan">
      {scan ? (
        <p className={`scan-result${scan.mismatch ? ' is-bad' : ''}`} data-testid="depot-scan-result" data-mismatch={scan.mismatch} role="status">
          {scan.mismatch
            ? `Failed: declared ${scan.declared_weight_kg} kg, the scale read ${scan.measured_weight_kg} kg. Held at the depot; this account's next parcels are weighed too.`
            : `Passed: declared ${scan.declared_weight_kg} kg, the scale read ${scan.measured_weight_kg} kg (within 0.5 lb or 3%).`}
          {scan.source !== 'depot' && <span className="muted"> ({scan.source})</span>}
        </p>
      ) : (
        <>
          <p className="text-[0.85rem] text-ink-2">
            The label was issued with a check at first scan. Enter what the depot scale reads (declared {declared} kg).
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <label htmlFor="scan-w" className="sr-only">Measured weight in kg</label>
            <input id="scan-w" className="field" style={{ width: '8rem' }} type="number" min="0.01" step="0.01" value={w}
              onChange={(e) => setW(e.target.value)} data-testid="depot-scan-input" />
            <span className="text-[0.85rem] text-ink-2">kg</span>
            <button type="button" className="btn btn-primary" disabled={busy || !(Number(w) > 0)} onClick={send}
              data-testid="depot-scan-record">
              {busy ? 'Recording...' : 'Record first scan'}
            </button>
          </div>
          {err && <p className="mt-2 text-[0.85rem] text-danger" role="alert">{err}</p>}
        </>
      )}
    </Section>
  )
}

/* ---------- Page ---------- */

const short = (id: string, n = 8) => (id.length > n ? id.slice(0, n) : id)

const LOOK: Record<string, { tone: string; icon: IconName }> = {
  allow: { tone: 'green', icon: 'check' },
  allow_scan_gated: { tone: 'teal', icon: 'scale' },
  owner_confirm: { tone: 'orange', icon: 'user' },
  review: { tone: 'indigo', icon: 'eye' },
  hold: { tone: 'indigo', icon: 'pause' },
  block: { tone: 'rose', icon: 'x' },
}

/** Fraud probability on a scale split into the six action bands, with a marker at this booking. */
function ProbScale({ p }: { p: number | undefined }) {
  const v = p ?? 0
  return (
    <div className="pscale">
      <div className="pscale-top">
        <span className="text-ink-2">Fraud probability</span>
        <b className="tnum">{p == null ? 'n/a' : v.toFixed(2)}</b>
      </div>
      <div className="pscale-bar" role="img" aria-label={`Fraud probability ${p == null ? 'not available' : v.toFixed(2)}`}>
        {ACTIONS.map((a) => (
          <span key={a} style={{ background: `var(--act-${a})` }} />
        ))}
        {p != null && <i style={{ left: `${Math.min(Math.max(v * 100, 1), 99)}%` }} />}
      </div>
      <div className="pscale-legend" aria-hidden="true">
        {ACTIONS.map((a) => (
          <span key={a}>
            <span className="dot" style={{ background: `var(--act-${a})` }} />
            {ACTION_META[a].short}
          </span>
        ))}
      </div>
    </div>
  )
}

/** The owner-passkey panel renders nothing for actions it cannot release; only wrap it in a card when it shows. */
function OwnerCard({ d, onDone }: { d: DecisionDetail; onDone: () => void }) {
  const can = d.owner_confirmation || ['owner_confirm', 'hold', 'review'].includes(d.action)
  if (!can || !d.booking) return null
  return (
    <section className="panel min-w-0">
      <div className="card-b">
        <OwnerPasskey key={`pk-${d.decision_id}`} d={d} onDone={onDone} />
      </div>
    </section>
  )
}

const SECTIONS: [string, string, IconName][] = [
  ['sum', 'Summary', 'lines'],
  ['why', 'Explanation', 'info'],
  ['act', 'Owner & what-if', 'user'],
  ['cost', 'Cost', 'dollar'],
  ['story', 'Account history', 'user'],
  ['model', 'Model & audit', 'shieldCheck'],
]

export default function DecisionView() {
  const { id = '' } = useParams()
  const { data: d, error, polling, reload } = useDecision(id)
  const { refresh } = useStatus()
  const [howOpen, setHowOpen] = useState(false)
  const [stateOpen, setStateOpen] = useState(false)
  const [sec, setSec] = useState('sum')

  useEffect(() => {
    if (id) rememberDecision(id)
  }, [id])

  if (error && !d)
    return (
      <div>
        <PageTitle title="Decision" />
        <Page>
          <ErrorBox message={error} onRetry={reload} testId="decision-error" />
        </Page>
      </div>
    )
  if (!d)
    return (
      <div>
        <PageTitle title="Decision" />
        <Page>
          <Loading what="decision" />
        </Page>
      </div>
    )

  const b = d.booking
  const meta = ACTION_META[d.action]
  const look = LOOK[d.action] ?? LOOK.review
  // Tabs switch what is shown in place; Summary shows everything. Nothing scrolls.
  const jump = (k: string) => {
    setSec(k)
    if (k === 'model') setHowOpen(true)
  }
  const show = (k: string) => sec === 'sum' || sec === k
  return (
    <div data-testid="decision-detail" data-decision-id={d.decision_id} className="decision">
      <PageTitle title="Decision" sub={`${d.booking_id}${b ? ` · acct ${short(b.account_id)}` : ''}`}>
        <Link to="/" className="btn">
          <Icon name="box" size={16} />
          Score another
        </Link>
      </PageTitle>

      <div className="page dec-page" style={{ ['--act' as string]: `var(--act-${d.action})` }}>
        <section className="panel verdict-card" id="dec-sum">
          <div className="verdict-left">
            <Icon name={look.icon} size={44} stroke={2.2} className="verdict-ico" />
            <div className="min-w-0">
              <p className="verdict-id mono">
                {d.decision_id} · decided in {fmtMs(d.latency_ms?.total)}
              </p>
              <h1 className="verdict-word" data-testid="action-badge" data-action={d.action}>
                {meta?.label ?? d.action}
              </h1>
              <p className="verdict-mean">{meta?.meaning}</p>
            </div>
          </div>
          <ProbScale p={d.probabilities.misuse ?? d.gbm_score ?? undefined} />
          {(d.degraded || d.decider || d.explored || d.model_versions?.gbm || (d.policy_trace?.rule_hits ?? []).includes('LINK_TO_CONFIRMED_FRAUD')) && (
            <div className="verdict-flags">
              {d.degraded && (
                <p data-testid="degraded-flag">
                  <Icon name="cpu" size={15} /> Decided by the backup model with stricter thresholds, because the Laya model is not loaded.
                </p>
              )}
              {!d.degraded && d.decider && (
                <p data-testid="decider-flag" data-decider={d.decider}>
                  <Icon name="sparkle" size={15} />
                  {d.laya_action
                    ? d.laya_action.accepted
                      ? `Decided by Laya: it weighed the LightGBM score${d.gbm_score != null ? ` (${fmtPct(d.gbm_score)})` : ''} and the rules as evidence and chose this action; the cost check agreed.`
                      : `Laya proposed "${actionLabel(d.laya_action.proposed)}"; the cost check overruled it: ${d.laya_action.overrule_reason}.`
                    : `Decided from Laya's answers: it weighed the LightGBM score${d.gbm_score != null ? ` (${fmtPct(d.gbm_score)})` : ''} and the rules as evidence, and the cost rule picked the cheapest action.`}
                </p>
              )}
              {d.model_versions?.gbm && (
                <p data-testid="decision-gbm-version" data-version={d.model_versions.gbm}>
                  <Icon name="stack" size={15} /> LightGBM model that scored this booking: <span className="mono">{d.model_versions.gbm}</span>
                </p>
              )}
              {d.explored && (
                <p data-testid="explored-flag">
                  <Icon name="info" size={15} /> Exploration sample: sent to a first-scan check to measure the policy.
                </p>
              )}
              {(d.policy_trace?.rule_hits ?? []).includes('LINK_TO_CONFIRMED_FRAUD') && (
                <p className="is-hard" data-testid="hard-signal">
                  <Icon name="link" size={15} /> The sender or receiver on this booking appears in a confirmed fraud case.
                </p>
              )}
            </div>
          )}
        </section>

        <div className="itabs dec-tabs" role="tablist" aria-label="Decision sections">

          <SlideInd />
          {SECTIONS.map(([k, label, icon]) => (
            <button key={k} type="button" role="tab" aria-selected={sec === k} onClick={() => jump(k)}>
              <Icon name={icon} size={18} />
              {label}
            </button>
          ))}
        </div>

        <div className="tab-body" key={sec}>
        {show('why') && (
        <div className="dec-two" id="dec-why">
          <Section title="What stood out" icon="flag">
            {d.reasons.length > 0 ? (
              <ul className="stood" data-testid="reason-codes">
                {d.reasons.map((r, i) => (
                  <li key={r} data-code={r}>
                    <span className="stood-n tnum">{String(i + 1).padStart(2, '0')}</span>
                    <span>{reasonText(r)}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-muted">No single signal stood out; the booking matches the account's usual pattern.</p>
            )}
          </Section>
          <section className="panel min-w-0" aria-labelledby="why-h" data-testid="explanation-panel">
            <div className="card-h">
              <h2 id="why-h">
                <Icon name="lines" size={18} />
                Why, in plain words
              </h2>
            </div>
            <div className="card-b">
              <FactCheckedExplanation d={d} polling={polling} />
            </div>
          </section>
        </div>

        )}

        {show('why') && (d.action === 'allow_scan_gated' || d.first_scan) && <DepotScan d={d} onDone={reload} />}

        {show('act') && (
          <div className="dec-two dec-act">
            {/* keyed by decision: their click-loaded results belong to one decision and must not carry over */}
            {d.action !== 'allow' && b && (
              <section className="panel min-w-0">
                <div className="card-b">
                  <CounterfactualPanel key={`cf-${d.decision_id}`} decisionId={d.decision_id} />
                  <p className="mt-3 text-[14px]">
                    <Link className="underline underline-offset-2" data-testid="redteam-link" to={`/redteam?decision=${encodeURIComponent(d.decision_id)}`}>
                      Red-team this decision: watch an attacker try to get it through
                    </Link>
                  </p>
                </div>
              </section>
            )}
            <OwnerCard d={d} onDone={reload} />
          </div>
        )}

        {show('cost') && (
        <div id="dec-cost">
          <Section title="Expected cost of each action" icon="dollar" aside="lowest wins">
            <CostTable d={d} />
          </Section>
        </div>

        )}

        {show('story') && (
        <section className="panel" id="dec-story" aria-labelledby="story-h">
          <div className="card-h">
            <h2 id="story-h">
              <Icon name="user" size={18} />
              Who is this account paying for?
            </h2>
            <span className="ml-auto text-[13.5px] text-muted">last 10 bookings vs the account's previous 90 days</span>
          </div>
          <div className="card-b story-grid">
            {b && <ShippingLabel booking={b} action={d.action} />}
            <div className="min-w-0">
              <AccountStory decisionId={d.decision_id} action={d.action} />
            </div>
          </div>
        </section>

        )}

        {show('model') && (
        <section className="panel how" id="dec-model">
          <button
            type="button"
            className="how-toggle"
            aria-expanded={howOpen}
            aria-controls="how-body"
            onClick={() => setHowOpen((o) => !o)}
            data-testid="how-toggle"
          >
            <span className="flex items-center gap-2.5">
              <Icon name="shieldCheck" size={18} />
              How this was decided
            </span>
            <span className="how-hint">
              
              <Icon name="chevronDown" size={16} style={{ transform: howOpen ? 'rotate(180deg)' : 'none' }} />
            </span>
          </button>
          {howOpen && (
            <div id="how-body" className="how-body">
              <div className="flex min-w-0 flex-col gap-4">
                <Section
                  title={d.degraded ? 'What the backup model answered' : 'What Laya answered'}
                  aside={d.degraded ? 'Laya is not loaded, so only the misuse score exists' : 'four named questions, one batched call'}
                >
                  <ProbBars d={d} />
                </Section>
                <Section title="This booking against the account's own baseline">
                  <table className="tbl" data-testid="top-features">
                    <thead>
                      <tr>
                        <th scope="col">Signal</th>
                        <th scope="col" className="r">This booking</th>
                        <th scope="col" className="r">Account baseline</th>
                      </tr>
                    </thead>
                    <tbody>
                      {d.top_features.map((f) => (
                        <tr key={f.name}>
                          <th scope="row" className="text-left font-normal">
                            {featureLabel(f.name)}
                          </th>
                          <td className="tnum r font-semibold">{fmtNum(f.value)}</td>
                          <td className="tnum r text-ink-2">{fmtNum(f.baseline)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Section>
              </div>
              <div className="flex min-w-0 flex-col gap-4">
                <Section title="Speed" icon="pulse">
                  <LatencyBreakdown lat={d.latency_ms} />
                </Section>
                <Section title="Ask a new question" icon="sparkle" testId="ask-panel">
                  <AskPanel id={d.decision_id} />
                </Section>
                <section className="panel" aria-labelledby="state-h">
                  <button
                    type="button"
                    className="card-h w-full text-left"
                    aria-expanded={stateOpen}
                    aria-controls="state-body"
                    onClick={() => setStateOpen((o) => !o)}
                    data-testid="state-text-toggle"
                    style={{ paddingBottom: 18, background: 'none', border: 0, color: 'inherit', cursor: 'pointer' }}
                  >
                    <h2 id="state-h">
                      <Icon name="eye" size={18} />
                      What the model read
                    </h2>
                    <span className="ml-auto text-[13.5px] text-muted">{stateOpen ? 'Hide' : 'Show'} the input text</span>
                  </button>
                  {stateOpen && (
                    <div id="state-body" className="card-b pt-0">
                      <p className="text-[13px] text-muted">The exact fixed-order text the model scored. No free text from the shipper goes in.</p>
                      <pre className="state-pre" data-testid="state-text">
                        {d.state_text}
                      </pre>
                    </div>
                  )}
                </section>
                <Section title="Audit record" icon="lock">
                  <dl className="kv text-[13.5px]">
                    <dt>Audit hash</dt>
                    <dd className="mono break-all" data-testid="decision-audit-hash" title={d.audit_hash}>
                      {shortHash(d.audit_hash, 24)}
                    </dd>
                    {Object.entries(d.model_versions ?? {}).map(([k, v]) => (
                      <div key={k} className="contents">
                        <dt>{featureLabel(k)} version</dt>
                        <dd className="mono break-all">{v}</dd>
                      </div>
                    ))}
                  </dl>
                </Section>
              </div>
            </div>
          )}
        </section>
        )}
        </div>
      </div>

      <div className="dec-dock" data-testid="analyst-panel">
        <AnalystPanel
          d={d}
          onDone={() => {
            reload()
            refresh()
          }}
        />
      </div>
    </div>
  )
}
