import { useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import type { Booking, RedTeamAttack, RedTeamRate, RedTeamResults } from '../api/types'
import { ActionPill } from '../components/ActionBadge'
import { ErrorBox, Loading, PageTitle, Section } from '../components/common'
import { actionLabel } from '../lib/domain'
import { fmtMs, fmtPct, shortHash } from '../lib/format'
import { useAsync } from '../lib/useAsync'
import '../redteam.css'

/*
  Red team you can watch. The attacker of scripts/redteam.py, run live on one stopped booking: it sees only the action
  we return (never the risk score), changes at most 2 fields the booker controls, and gets 50 tries. Every try is
  replayed row by row. Next to it: the measured run on the test window, and what our own gate did with the
  hardened model.
*/

const STEP_MS = 110
const reducedMotion = () => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

type Target = { key: string; label: string; booking?: Booking; decisionId?: string }

function rate(r: RedTeamRate, k: 'allow' | 'softer') {
  const m = k === 'allow' ? r.flip_allow_mean : r.flip_softer_mean
  const sd = k === 'allow' ? r.flip_allow_sd : r.flip_softer_sd
  return `${fmtPct(m, 1)} ± ${fmtPct(sd, 1)}`
}

function Verdict({ a }: { a: RedTeamAttack }) {
  const tone = a.evaded === 'allow' ? 'evaded' : a.still_stopped ? 'held' : 'partial'
  const head = tone === 'evaded' ? 'The attacker got through' : tone === 'held' ? 'Our model held' : 'The attacker got part of the way'
  return (
    <div className="rt-verdict" data-testid="redteam-verdict" data-tone={tone} data-evaded={a.evaded ?? 'none'}>
      <p className="rt-verdict-h">{head}</p>
      <p className="mt-1 text-[0.88rem]">{a.message}</p>
      {a.evasion && (
        <p className="mt-1 text-[0.85rem]" data-testid="redteam-evasion">
          Try {a.evasion.n}: <ActionPill action={a.evasion.action} /> if {a.evasion.changes.map((c) => c.text).join(' and ')}
        </p>
      )}
      <p className="mt-1.5 text-[0.75rem] text-muted">
        Original decision <strong>{actionLabel(a.original_action)}</strong> · {a.queries} of {a.budget} tries · {fmtMs(a.latency_ms)} ·{' '}
        <Link className="underline underline-offset-2" to={`/decisions/${encodeURIComponent(a.decision_id)}`}>
          open the decision
        </Link>{' '}
        · logged to the audit trail {shortHash(a.audit_hash)}
      </p>
    </div>
  )
}

function LiveAttack() {
  const [params] = useSearchParams()
  const fromDecision = params.get('decision')
  const demos = useAsync(() => api.demoBookings(), [])
  const targets = useAsync(() => api.redteamTargets(), [])
  const [pick, setPick] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [attack, setAttack] = useState<RedTeamAttack | null>(null)
  const [shown, setShown] = useState(0)
  const logRef = useRef<HTMLOListElement>(null)

  const options: Target[] = [
    ...(fromDecision ? [{ key: `d:${fromDecision}`, label: `Decision ${fromDecision} (opened from the decision page)`, decisionId: fromDecision }] : []),
    ...(demos.data ?? [])
      .filter((d) => d.scenario === 'takeover' || d.scenario === 'reshipping-drop')
      .map((d) => ({ key: `demo:${d.scenario}`, label: `Demo: ${d.title}`, booking: d.booking })),
    ...(targets.data ?? []).map((t) => ({ key: `t:${t.booking.booking_id}`, label: t.title, booking: t.booking })),
  ]
  const selected = options.find((o) => o.key === pick) ?? options[0]

  useEffect(() => {
    if (!attack || shown >= attack.attempts.length) return
    const t = window.setTimeout(() => setShown((s) => s + 1), STEP_MS)
    return () => window.clearTimeout(t)
  }, [attack, shown])

  useEffect(() => {
    // keep the newest try in view while the attack replays (the audience watches the bottom row)
    const el = logRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [shown])

  const launch = async () => {
    if (!selected) return
    setBusy(true)
    setErr(null)
    setAttack(null)
    setShown(0)
    try {
      const id = selected.decisionId ?? (await api.score(selected.booking as Booking)).decision_id
      const a = await api.redteamAttack(id)
      setAttack(a)
      setShown(reducedMotion() ? a.attempts.length : 0)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const done = attack != null && shown >= attack.attempts.length
  const rows = attack ? attack.attempts.slice(0, shown) : []
  return (
    <Section title="Watch an attack" testId="redteam-live" aside="the attacker sees only our decision, never the risk score">
      <p className="text-[0.85rem] text-ink-2">
        Pick a booking our system stopped. The attacker changes at most 2 things a fraudster controls (declared value, weight, size,
        service, a sender the account already used, booking time) and gets 50 tries.
      </p>
      <div className="mt-3 flex flex-wrap items-end gap-2">
        <label className="flex min-w-[18rem] flex-1 flex-col gap-1 text-[0.8rem] text-muted">
          Target
          <select className="field" data-testid="redteam-target" value={selected?.key ?? ''} onChange={(e) => setPick(e.target.value)}>
            {options.map((o) => (
              <option key={o.key} value={o.key}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
        <button type="button" className="btn btn-primary" data-testid="redteam-launch" disabled={busy || !selected} onClick={launch}>
          {busy ? 'Attacking…' : 'Launch attack'}
        </button>
        {attack && !done && (
          <button type="button" className="btn" data-testid="redteam-skip" onClick={() => setShown(attack.attempts.length)}>
            Skip to the result
          </button>
        )}
      </div>
      {(demos.loading || targets.loading) && <Loading what="targets" />}
      {err && <ErrorBox message={err} testId="redteam-error" />}
      {attack && (
        <div className="mt-3">
          <p className="mb-1.5 text-[0.8rem] text-muted" data-testid="redteam-progress" data-shown={shown}>
            Try {shown} of {attack.queries} (budget {attack.budget}) · each row is one booking the attacker submitted, and the decision it got back
          </p>
          <ol className="rt-log" data-testid="redteam-log" ref={logRef}>
            {rows.map((t) => (
              <li key={t.n} className="rt-row" data-testid="redteam-attempt" data-result={t.allow ? 'allow' : t.softer ? 'softer' : 'same'}>
                <span className="rt-n">#{t.n}</span>
                <span>{t.changes.join(' and ')}</span>
                <ActionPill action={t.action} />
              </li>
            ))}
          </ol>
          {done && <Verdict a={attack} />}
        </div>
      )}
    </Section>
  )
}

function Measured({ r }: { r: RedTeamResults }) {
  const fields = Object.values(r.fields_used ?? {}).reduce((a, b) => a + b, 0)
  const sender = r.fields_used?.sender_id ?? 0
  const rows: [string, RedTeamRate, string][] = [
    ['All trained fraud types', r.trained.all, 'redteam-row-all'],
    ...r.by_type.map((t): [string, RedTeamRate, string] => [t.type, t, `redteam-row-${t.type}`]),
    ['Never trained (T3, T5)', r.held_out.all, 'redteam-row-held-out'],
  ]
  return (
    <Section title="Measured on the test window" testId="redteam-results" aside={`run once · seeds ${r.seeds[0]}-${r.seeds[r.seeds.length - 1]}`}>
      <table className="w-full text-[0.85rem]">
        <thead>
          <tr className="border-b border-rule text-left text-[0.75rem] text-muted">
            <th scope="col" className="py-1.5 font-semibold">Stopped fraud</th>
            <th scope="col" className="py-1.5 text-right font-semibold">n</th>
            <th scope="col" className="py-1.5 text-right font-semibold">Got a plain allow</th>
            <th scope="col" className="py-1.5 text-right font-semibold">Got any softer action</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, v, id]) => (
            <tr key={id} className="border-b border-rule last:border-0" data-testid={id}>
              <td className="py-1.5">{label}</td>
              <td className="tnum py-1.5 text-right">{v.n}</td>
              <td className="tnum py-1.5 text-right">{rate(v, 'allow')}</td>
              <td className="tnum py-1.5 text-right">{rate(v, 'softer')}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-[0.78rem] text-muted">
        Mean ± sd over seeds; up to {r.per_seed} stopped fraud bookings per seed. {r.synthetic}. Most evasions reuse a sender the
        account already used ({sender} of {fields} changed fields). "Softer" includes allow-with-depot-check, where the parcel is still
        weighed. Evidence: {r.evidence}.
      </p>
    </Section>
  )
}

function Hardening({ r }: { r: RedTeamResults }) {
  const h = r.hardening
  return (
    <Section title="Then we tried to fix it, and our own gate said no" testId="redteam-hardening">
      <div className="rt-gate" data-testid="redteam-gate" data-passed={String(h.passed)}>
        <p className="text-[0.88rem]">
          We retrained on {h.n_labels} evasions made from training-window fraud only ({h.n_evaded} of {h.n_attacked} attacked bookings got
          softer). The pre-registered deployment gate, unchanged:{' '}
          <strong>{h.passed ? 'passed' : 'REJECTED the hardened model'}</strong>.
        </p>
        <ul className="mt-1.5 list-disc pl-5 text-[0.85rem]">
          {h.failed_checks.map((c) => (
            <li key={c.name}>{c.detail}</li>
          ))}
        </ul>
      </div>
      <p className="mt-2 text-[0.78rem] text-muted">
        Catching the evasions would have stopped more honest shippers, so the current model keeps scoring. See the Learning tab for the
        gate and docs/LEARNING_GATE.md for its rules.
      </p>
    </Section>
  )
}

export default function RedTeamView() {
  const res = useAsync(() => api.redteamResults(), [])
  return (
    <div data-testid="redteam-view">
      <PageTitle
        title="Red team: we attack our own model"
        sub="We play the fraudster against our own system and publish how often we lose, instead of only how often we are right."
      />
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        <LiveAttack />
        <div className="flex flex-col gap-4">
          {res.loading && <Loading what="red-team results" />}
          {res.error && <ErrorBox message={res.error} onRetry={res.reload} />}
          {res.data && <Measured r={res.data} />}
          {res.data && <Hardening r={res.data} />}
        </div>
      </div>
    </div>
  )
}
