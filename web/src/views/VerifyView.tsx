import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import type { Verification, VerifyMetrics, VerifyStatus } from '../api/verify'
import { ErrorBox, Loading, PageTitle } from '../components/common'
import { Icon } from '../components/Icon'
import { PackageJourney } from '../components/PackageJourney'
import { actionLabel, reasonText } from '../lib/domain'
import { fmtBRL, shortHash } from '../lib/format'
import '../verify.css'

const ACTIONS = ['allow', 'allow_scan_gated', 'owner_confirm', 'review', 'hold', 'block']
const DONE: VerifyStatus[] = ['released', 'blocked', 'analyst_review']
const DEMO = ['verify-fraud', 'verify-honest', 'verify-contact-changed']
type Show = 'held' | 'person' | 'demo'

// One plain word per state, and its colour (the console's action colours)
const STATE: Record<VerifyStatus, { word: string; tone: string }> = {
  pending_verification: { word: 'Calling the owner', tone: 'owner_confirm' },
  calling: { word: 'Calling the owner', tone: 'owner_confirm' },
  ringing: { word: 'Phone ringing', tone: 'owner_confirm' },
  in_call: { word: 'On the phone', tone: 'owner_confirm' },
  retry_scheduled: { word: 'No answer, will call again', tone: 'hold' },
  awaiting_simulated_answer: { word: 'Waiting', tone: 'hold' },
  released: { word: 'Released', tone: 'allow_scan_gated' },
  blocked: { word: 'Stopped', tone: 'block' },
  analyst_review: { word: 'Needs a person', tone: 'review' },
}

const CASE_REASON: Record<string, string> = {
  owner_denied: 'The owner said it was not them.',
  unsure: 'The owner was not sure.',
  detail_mismatch: 'The owner said yes, but gave the wrong pickup address.',
  hard_signal: 'There is a hard fraud signal. A "yes" on the phone is never enough for that.',
  still_risky: 'Even after the "yes", it still looks too risky.',
  no_answer: 'Nobody answered, twice. An SMS was logged.',
  contact_changed_recently: 'The phone number changed less than 30 days ago, so Tracd did not call (a fraudster may have changed it).',
  no_verified_contact: 'There is no phone number on file, so Tracd could not call.',
}

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e))
const clock = (iso: string | null | undefined) => {
  if (!iso) return ''
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleTimeString([], { hour12: false })
}
const pct = (p: number | null | undefined) => (p == null ? 'n/a' : `${(p * 100).toFixed(1)}%`)
const riskWord = (p: number | null | undefined) =>
  p == null ? '' : p >= 0.9 ? 'very high' : p >= 0.5 ? 'high' : p >= 0.1 ? 'medium' : 'low'

/** Polls fn every ms; restarts when key changes. */
function usePoll<T>(fn: () => Promise<T>, ms: number, key: string) {
  const ref = useRef(fn)
  const [state, setState] = useState<{ key: string; data: T | null; error: string | null }>({ key, data: null, error: null })
  useEffect(() => {
    ref.current = fn
  })
  useEffect(() => {
    let alive = true
    let t: number | undefined
    const tick = () => {
      ref.current().then(
        (data) => alive && setState({ key, data, error: null }),
        (e) => alive && setState((s) => ({ key, data: s.key === key ? s.data : null, error: errText(e) })),
      ).finally(() => {
        if (alive) t = window.setTimeout(tick, ms)
      })
    }
    tick()
    return () => {
      alive = false
      window.clearTimeout(t)
    }
  }, [key, ms])
  return state.key === key ? state : { key, data: null, error: null }
}

function StateTag({ status }: { status: VerifyStatus }) {
  const s = STATE[status]
  const live = !DONE.includes(status)
  return (
    <span className={`vf-state${live ? ' is-live' : ''}`} style={{ ['--act' as string]: `var(--act-${s.tone})` }} data-status={status}>
      {live && <span className="vf-dot" aria-hidden="true" />}
      {s.word}
    </span>
  )
}

/** One compact line of call numbers, like Bolna's balance chip. All counted from this run. */
function Summary({ m }: { m: VerifyMetrics | null }) {
  if (!m) return null
  const items: [string, string][] = [
    ['Calls', String(m.calls_made)],
    ['Minutes', m.call_minutes.toFixed(1)],
    ['Cost', `$${m.call_cost_usd.toFixed(2)}`],
    ['Said "no"', String(m.owner_said.no)],
    ['Said "yes"', String(m.owner_said.yes)],
    ['Waiting for a person', String(m.open_cases)],
  ]
  return (
    <div className="cv-sum" data-testid="verify-kpis">
      {items.map(([k, v]) => (
        <span key={k}>
          {k} <b className="tnum">{v}</b>
        </span>
      ))}
      {m.simulated_calls > 0 && <span className="vf-sim">{m.simulated_calls} demo calls (free)</span>}
    </div>
  )
}

const CALL_STATUS: Record<string, string> = {
  queued: 'Dialling', initiated: 'Dialling', ringing: 'Ringing', 'in-progress': 'On the phone', completed: 'Finished',
  'call-disconnected': 'Finished', 'no-answer': 'No answer', busy: 'Busy', failed: 'Failed', canceled: 'Cancelled', error: 'Failed',
}

function CallTable({ rows, sel, onSel }: { rows: Verification[]; sel: string | null; onSel: (id: string) => void }) {
  return (
    <div className="overflow-x-auto">
      <table className="tbl cd-tbl" data-testid="verify-feed">
        <thead>
          <tr>
            <th scope="col">Call ID</th>
            <th scope="col">Customer</th>
            <th scope="col">Status</th>
            <th scope="col">Owner said</th>
            <th scope="col" className="r">Duration</th>
            <th scope="col" className="r">Cost</th>
            <th scope="col">Result</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((v) => {
            const att = v.attempts[v.attempts.length - 1]
            const secs = v.attempts.reduce((s, a) => s + (a.duration_s ?? 0), 0)
            const cost = v.attempts.reduce((s, a) => s + (a.cost_usd ?? 0), 0)
            const said = v.outcome?.owner_confirmed
            return (
              <tr key={v.decision_id} className={`click${sel === v.decision_id ? ' is-sel' : ''}`} onClick={() => onSel(v.decision_id)}
                data-testid="verify-row" data-status={v.status}>
                <td>
                  <button type="button" className="vf-rowbtn mono" onClick={() => onSel(v.decision_id)} title={att?.execution_id}>
                    {att ? shortHash(att.execution_id.replace(/^sim_\d+_/, 'demo-'), 14) : v.verification_id}
                  </button>
                  <span className="vf-sub">{clock(v.held_at)}</span>
                </td>
                <td>
                  {v.owner.company}
                  {v.mode === 'simulated' && <span className="vf-sub vf-sim">demo call</span>}
                </td>
                <td>{att ? CALL_STATUS[att.status] ?? att.status : <span className="text-muted">Not called</span>}{v.attempts.length > 1 && <span className="vf-sub">{v.attempts.length} tries</span>}</td>
                <td className={said === 'no' ? 'text-danger font-semibold' : said === 'yes' ? 'font-semibold' : 'text-muted'}>
                  {said ? (said === 'unsure' ? 'not sure' : `"${said}"`) : '-'}
                </td>
                <td className="r tnum">{secs ? `${Math.floor(secs / 60)}:${String(Math.round(secs % 60)).padStart(2, '0')}` : '-'}</td>
                <td className="r tnum">{att ? `$${cost.toFixed(2)}` : '-'}</td>
                <td><StateTag status={v.status} /></td>
              </tr>
            )
          })}
          {rows.length === 0 && (
            <tr>
              <td colSpan={7} className="py-8 text-center text-muted">No records found</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}

function Transcript({ text, live }: { text: string; live: boolean }) {
  const lines = text
    .split('\n')
    .map((l) => l.match(/^\s*(assistant|agent|user|owner|customer)\s*:\s*(.*)$/i))
    .filter((m): m is RegExpMatchArray => m !== null)
  if (!lines.length) return <p className="vf-note">{live ? 'The conversation appears here as the call goes on.' : 'There was no conversation.'}</p>
  return (
    <ol className="vf-transcript" data-testid="verify-transcript" aria-live="polite">
      {lines.map((m, i) => {
        const agent = /assistant|agent/i.test(m[1])
        return (
          <li key={i} className={agent ? 'is-agent' : 'is-owner'}>
            <span className="vf-who">{agent ? 'Tracd' : 'Owner'}</span>
            <span>{m[2]}</span>
          </li>
        )
      })}
    </ol>
  )
}

function AnalystButtons({ v }: { v: Verification }) {
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  if (!v.case || v.case.status !== 'open') return null
  const act = async (label: 'fraud' | 'legit') => {
    setBusy(true)
    setErr(null)
    try {
      await api.verifyResolve(v.decision_id, label)
    } catch (e) {
      setErr(errText(e))
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="vf-decide" data-testid="verify-case" data-reason={v.case.reason}>
      <p className="m-0 font-semibold">You decide: is this fraud?</p>
      <div className="mt-2 flex flex-wrap gap-2">
        <button type="button" className="btn btn-danger" disabled={busy} onClick={() => act('fraud')} data-testid="verify-case-fraud">
          Yes, fraud: stop it
        </button>
        <button type="button" className="btn" disabled={busy} onClick={() => act('legit')} data-testid="verify-case-fine">
          No, it is fine: release it
        </button>
      </div>
      {err && <p className="mt-2 text-danger" role="alert">{err}</p>}
    </div>
  )
}

function SimAnswerBox({ v }: { v: Verification }) {
  const [busy, setBusy] = useState(false)
  if (v.status !== 'awaiting_simulated_answer') return null
  const go = async (owner_confirmed: 'yes' | 'no' | 'unsure' | 'no_answer') => {
    setBusy(true)
    try {
      await api.verifyStart(v.decision_id, { owner_confirmed, detail_answer: owner_confirmed === 'yes' ? `from ${v.booking.origin_label}` : null, language: 'en' })
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="vf-decide">
      <p className="m-0">There is no real phone number for this account. Pick what the owner says (demo call):</p>
      <div className="mt-2 flex flex-wrap gap-2">
        {(['yes', 'no', 'unsure', 'no_answer'] as const).map((a) => (
          <button key={a} type="button" className="btn" disabled={busy} onClick={() => go(a)}>
            {a === 'no_answer' ? 'No answer' : `"${a}"`}
          </button>
        ))}
      </div>
    </div>
  )
}

function resultSentence(v: Verification): { text: string; tone: 'good' | 'bad' | 'wait' } {
  if (v.final_action === 'block') return { text: `Stopped. No label was printed${v.account_locked ? ' and the account is locked' : ''}.`, tone: 'bad' }
  if (v.final_action) {
    return {
      text: `Released${v.minutes_to_final != null ? ` after ${v.minutes_to_final.toFixed(1)} min` : ''}. The depot will weigh the parcel when it arrives.`,
      tone: 'good',
    }
  }
  if (v.case?.status === 'open') return { text: `On hold until a person decides. ${CASE_REASON[v.case.reason] ?? ''}`, tone: 'wait' }
  return { text: 'Waiting for the phone call to finish...', tone: 'wait' }
}

function Detail({ id }: { id: string }) {
  const { data: v, error } = usePoll(() => api.verification(id), 1200, id)
  if (error && !v) return <p className="vf-note" data-testid="verify-detail-empty">{error}</p>
  if (!v) return <Loading what="the booking" />
  const live = !DONE.includes(v.status)
  const b = v.booking
  const att = v.attempts[v.attempts.length - 1]
  const said = v.outcome?.owner_confirmed
  const result = resultSentence(v)
  const laya = v.laya
  return (
    <div className="vf-steps-v" data-testid="verify-detail" data-status={v.status}>
      <PackageJourney id={v.decision_id} action={v.original_action} risk={v.p_fraud} />
      <section className="vf-step">
        <h3><span className="vf-n">1</span> Transcript</h3>
        {!att ? (
          <p className="vf-big">No call was made. {v.case ? CASE_REASON[v.case.reason] : ''}</p>
        ) : (
          <>
            <p className="vf-note">
              {live ? 'Calling ' : 'Called '}
              {v.owner.company}
              {said && <>: the owner said <b>{said === 'yes' ? '"Yes"' : said === 'no' ? '"No"' : '"Not sure"'}</b></>}
              {!said && !live && ': nobody answered'}.
            </p>
            {v.mode === 'simulated' && (
              <p className="vf-demo-tag">Demo call: no real phone call was made ({v.simulated_reason}). With a Bolna key this is a real call.</p>
            )}
            <SimAnswerBox v={v} />
            {att.recording_url && (
              <div className="cv-rec">
                <span className="vf-sub">Recording of the real call (from Bolna)</span>
                <audio controls preload="none" src={att.recording_url} data-testid="verify-recording" />
              </div>
            )}
            <Transcript text={v.transcript} live={live} />
          </>
        )}
      </section>

      <section className={`vf-step vf-result is-${result.tone}`} data-testid="verify-final" data-final={v.final_action ?? ''}>
        <h3><span className="vf-n">2</span> Result</h3>
        <p className="vf-big">{result.text}</p>
        <AnalystButtons v={v} />
      </section>

      <section className="vf-step" data-testid="verify-laya">
        <h3><span className="vf-n">3</span> Why it was held {laya && laya.source !== 'demo' && <span className="vf-by">Laya</span>}</h3>
        {laya ? (
          <>
            <ul className="vf-wrong">
              {laya.items.filter((i) => i.yes).map((i) => (
                <li key={i.id}>
                  {i.text}
                  {i.probability != null && <span className="vf-sub">Laya: {pct(i.probability)} yes</span>}
                  {i.fact && <span className="vf-sub">{i.fact}</span>}
                </li>
              ))}
              {laya.items.every((i) => !i.yes) && <li>Nothing unusual was found.</li>}
            </ul>
            {laya.source !== 'demo' && <p className="vf-note">Answered by Laya.</p>}
          </>
        ) : (
          <Loading what="Laya's answer" />
        )}
      </section>

      <section className="vf-step">
        <h3><span className="vf-n">4</span> The booking</h3>
        <p className="vf-big">
          {b.weight_kg.toFixed(1)} kg {b.service} from {b.origin_label} to {b.dest_label}, at {b.time_label}
          {b.device_is_new && ', from a new device'}.
        </p>
        <p className="vf-note">
          {v.owner.company}. LightGBM gave it a <b>{riskWord(v.p_fraud)} risk ({pct(v.p_fraud)})</b>, so no label was printed and the owner was called.
        </p>
      </section>

      <details className="vf-more">
        <summary>Technical details (for analysts)</summary>
        {b.display_labels && <p className="vf-note">City and company names are demo labels. Data lane {b.lane_data}, cost {fmtBRL(b.carrier_cost)}.</p>}
        {!!v.decision?.reasons?.length && (
          <>
            <h4 className="vf-h3">LightGBM's reason codes</h4>
            <ul className="vf-reasons">
              {v.decision.reasons.slice(0, 5).map((r) => (
                <li key={r}>{reasonText(r)}</li>
              ))}
            </ul>
          </>
        )}
        {v.rescore && (
          <>
            <h4 className="vf-h3">Risk again after the "yes"</h4>
            <p className="vf-rescore tnum" data-testid="verify-rescore">
              {v.rescore.p_before.toFixed(3)} <Icon name="chevronRight" size={14} /> {v.rescore.p_model_after_change.toFixed(3)}{' '}
              <Icon name="chevronRight" size={14} /> <b>{v.rescore.p_after.toFixed(3)}</b>
            </p>
            <p className="vf-note">
              Original score, then LightGBM again treating the change the owner explained as normal, then counting the "yes" (x
              {v.rescore.likelihood_ratio.toFixed(3)}).
            </p>
          </>
        )}
        <h4 className="vf-h3">Cost of each action (R$, * = chosen)</h4>
        <table className="tbl vf-costs" data-testid="verify-costs">
          <tbody>
            {ACTIONS.filter((a) => a in (v.decision?.expected_costs ?? {})).map((a) => (
              <tr key={a}>
                <td>{actionLabel(a)}</td>
                <td className={`r tnum${a === v.original_action ? ' font-semibold' : ''}`}>
                  {fmtBRL((v.decision?.expected_costs ?? {})[a])}
                  {a === v.original_action && ' *'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {laya && (
          <>
            <h4 className="vf-h3">The four fraud questions</h4>
            <ul className="vf-reasons">
              {laya.items.map((i) => (
                <li key={i.id}>
                  {i.question} <b>{i.yes ? 'yes' : 'no'}</b>
                </li>
              ))}
            </ul>
          </>
        )}
        <h4 className="vf-h3">Timeline</h4>
        <ol className="vf-timeline" data-testid="verify-timeline">
          {v.timeline.map((t, i) => (
            <li key={i}>
              <span className="mono tnum">{clock(t.at)}</span>
              <b>{t.step.replace(/_/g, ' ')}</b>
              <span className="text-ink-2">{t.detail}</span>
            </li>
          ))}
        </ol>
        <h4 className="vf-h3">Audit records</h4>
        <ul className="vf-audit">
          {v.audit_hashes.map((h) => (
            <li key={h.hash}>
              <span>{h.event.replace(/_/g, ' ')}</span>
              <span className="mono">{shortHash(h.hash, 16)}</span>
            </li>
          ))}
        </ul>
        {att && <p className="vf-note mono">call {att.execution_id}</p>}
      </details>
    </div>
  )
}

function DemoBar({ onPick }: { onPick: (id: string) => void }) {
  const [running, setRunning] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const stop = useRef(false)
  const run = async () => {
    stop.current = false
    setErr(null)
    setRunning(true)
    try {
      for (const s of DEMO) {
        if (stop.current) break
        const r = await api.verifyDemo(s)
        onPick(r.decision_id)
        for (let i = 0; i < 120 && !stop.current; i++) {
          const v = await api.verification(r.decision_id).catch(() => null)
          if (!v || DONE.includes(v.status)) break
          await new Promise((res) => setTimeout(res, 1000))
        }
        await new Promise((res) => setTimeout(res, 2500))
      }
    } catch (e) {
      setErr(errText(e))
    } finally {
      setRunning(false)
    }
  }
  return (
    <div className="vf-demobar">
      <p className="m-0">Three made-up bookings for the demo video: a fraud, an honest shipper with a new warehouse, and a changed phone number.</p>
      {running ? (
        <button type="button" className="btn" onClick={() => (stop.current = true)} data-testid="verify-demo-stop">
          <Icon name="pause" size={16} /> Stop
        </button>
      ) : (
        <button type="button" className="btn btn-primary" onClick={run} data-testid="verify-demo">
          <Icon name="play" size={16} /> Play the demo
        </button>
      )}
      {err && <ErrorBox message={err} />}
    </div>
  )
}

export default function VerifyView() {
  const [params, setParams] = useSearchParams()
  const show = (params.get('show') as Show) || 'held'
  const calls = usePoll(() => api.verifications(200), 1500, 'calls')
  const metrics = usePoll(() => api.verifyMetrics(), 2500, 'metrics')
  const status = usePoll(() => api.verifyStatus(), 15000, 'status')
  const [q, setQ] = useState('')
  const [page, setPage] = useState(0)
  const [perPage, setPerPage] = useState(20)
  const all = calls.data ?? []
  const isDemo = (v: Verification) => String(v.booking.scenario ?? '').startsWith('verify-')
  const filtered = (show === 'demo' ? all.filter(isDemo) : show === 'person' ? all.filter((v) => v.case?.status === 'open') : all).filter((v) => {
    const t = q.trim().toLowerCase()
    if (!t) return true
    const ex = v.attempts.map((a) => a.execution_id).join(' ')
    return `${ex} ${v.owner.company} ${v.decision_id}`.toLowerCase().includes(t)
  })
  const pages = Math.max(1, Math.ceil(filtered.length / perPage))
  const cur = Math.min(page, pages - 1)
  const rows = filtered.slice(cur * perPage, cur * perPage + perPage)
  const sel = params.get('d')
  const go = (next: Record<string, string>) => setParams(next, { replace: true })
  const close = () => go({ show })

  return (
    <div>
      <PageTitle title="Owner calls">
        <span className={`vf-mode${status.data?.real_calls_ready ? ' is-real' : ''}`} data-testid="verify-mode">
          {status.data ? (status.data.real_calls_ready ? 'Real phone calls' : 'Demo calls (no Bolna key)') : ''}
        </span>
      </PageTitle>
      <div className="page">
        <Summary m={metrics.data} />

        <div className="cv-bar">
          <label className="sr-only" htmlFor="cv-filter">Which calls</label>
          <select id="cv-filter" className="field cv-select" value={show} onChange={(e) => { setPage(0); go({ show: e.target.value }) }}>
            <option value="held">All calls</option>
            <option value="person">Needs a person</option>
            <option value="demo">Demo calls</option>
          </select>
          <label className="cv-search">
            <Icon name="search" size={16} />
            <span className="sr-only">Search calls</span>
            <input className="field" placeholder="Search call ID or customer" value={q} onChange={(e) => { setQ(e.target.value); setPage(0) }} />
          </label>
          <span className="cv-grow" />
          {show === 'demo' && <DemoBar onPick={(id) => go({ show: 'demo', d: id })} />}
          <button type="button" className="btn" onClick={() => window.location.reload()}>
            <Icon name="refresh" size={16} /> Refresh
          </button>
        </div>

        <section className="panel min-w-0 cv-table" aria-label="Calls">
          {calls.error && !calls.data ? (
            <ErrorBox message={calls.error} />
          ) : !calls.data ? (
            <Loading what="calls" />
          ) : (
            <CallTable rows={rows} sel={sel} onSel={(id) => go({ show, d: id })} />
          )}
          <div className="cv-pager">
            <span>Rows per page</span>
            <select className="field cv-select" value={perPage} onChange={(e) => { setPerPage(Number(e.target.value)); setPage(0) }}>
              {[10, 20, 50].map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
            <span className="tnum">Page {cur + 1} of {pages}</span>
            <button type="button" className="btn" disabled={cur === 0} onClick={() => setPage(cur - 1)} aria-label="Previous page">‹</button>
            <button type="button" className="btn" disabled={cur >= pages - 1} onClick={() => setPage(cur + 1)} aria-label="Next page">›</button>
          </div>
        </section>
      </div>

      {sel && (
        <div className="cv-scrim" onMouseDown={close}>
          <aside className="cv-drawer" aria-label="Call details" onMouseDown={(e) => e.stopPropagation()}>
            <div className="cv-drawer-h">
              <b>Call details</b>
              <button type="button" className="ph-icon" aria-label="Close" onClick={close}>
                <Icon name="x" size={18} />
              </button>
            </div>
            <Detail id={sel} />
          </aside>
        </div>
      )}
    </div>
  )
}
