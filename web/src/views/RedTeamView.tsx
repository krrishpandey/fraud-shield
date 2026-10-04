import { useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import type { Booking, RedTeamAttack, RedTeamHardening, RedTeamRate, RedTeamResults, RetrainResponse } from '../api/types'
import { ActionPill } from '../components/ActionBadge'
import { ErrorBox, Loading, PageTitle, Section } from '../components/common'
import { Icon } from '../components/Icon'
import { HandoverBanner, ModelInUseLine } from '../components/ModelHandover'
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
      {a.harvest && a.harvest.evasions > 0 && (
        <p className="mt-1.5 text-[0.82rem]" data-testid="redteam-harvest" data-labelled={a.harvest.labelled}>
          Kept {a.harvest.evasions} evading {a.harvest.evasions === 1 ? 'booking' : 'bookings'} for retraining
          {a.harvest.labelled > 0
            ? `: ${a.harvest.labelled} added as fraud labels${a.harvest.simulated ? ' (simulated: from injected ground truth)' : ''}.`
            : '.'}{' '}
          {a.harvest.labelled === 0 && a.harvest.note}
        </p>
      )}
    </div>
  )
}

function LiveAttack({ onAttacked }: { onAttacked: () => void }) {
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
      onAttacked()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const done = attack != null && shown >= attack.attempts.length
  const rows = attack ? attack.attempts.slice(0, shown) : []
  return (
    <Section title="Watch an attack" icon="target" testId="redteam-live" aside="the attacker sees only our decision, never the risk score">
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
    <Section title="Measured on the test window" icon="chart" testId="redteam-results" aside={`run once · seeds ${r.seeds[0]}-${r.seeds[r.seeds.length - 1]}`}>
      <table className="tbl">
        <thead>
          <tr>
            <th scope="col">Stopped fraud</th>
            <th scope="col" className="r">n</th>
            <th scope="col" className="r">Got a plain allow</th>
            <th scope="col" className="r">Got any softer action</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, v, id]) => (
            <tr key={id} data-testid={id}>
              <td>{label}</td>
              <td className="tnum r">{v.n}</td>
              <td className="tnum r">{rate(v, 'allow')}</td>
              <td className="tnum r">{rate(v, 'softer')}</td>
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

function Hardening({ r, h, reload }: { r: RedTeamResults | null; h: RedTeamHardening | null; reload: () => void }) {
  const [busy, setBusy] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [run, setRun] = useState<(RetrainResponse & { redteam_labels: number }) | null>(null)
  const act = async (key: string, fn: () => Promise<unknown>) => {
    setBusy(key)
    setErr(null)
    try {
      await fn()
      reload()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }
  const confirm = (id: string) =>
    act(`confirm-${id}`, () => api.analyst(id, { label: 'fraud', note: 'Red team: analyst confirmed the attacked booking is fraud' }))
  const retrain = () => act('retrain', async () => setRun(await api.redteamRetrain()))
  const fresh = h?.labelled_since_last_retrain ?? 0
  const handover = run?.handover ?? h?.last_retrain?.handover
  const off = r?.hardening
  const tiles: [string, number, string][] = h
    ? [
        ['Attacks run', h.attacks, 'redteam-attacks'],
        ['Evading bookings kept', h.evasions, 'redteam-evasions'],
        ['Fraud labels for retraining', h.labelled, 'redteam-labels'],
      ]
    : []
  return (
    <Section title="Harden the model with these attacks" icon="refresh" testId="redteam-hardening">
      {h ? (
        <>
          <dl className="grid grid-cols-3 gap-2 text-center" data-testid="redteam-session" data-attacks={h.attacks}>
            {tiles.map(([label, v, id]) => (
              <div key={id} className="rounded-sm bg-surface-2 p-2">
                <dd className="tnum text-[1.3rem] font-extrabold" data-testid={id}>
                  {v}
                </dd>
                <dt className="text-[0.72rem] text-muted">{label}</dt>
              </div>
            ))}
          </dl>
          <p className="mt-2 text-[0.8rem] text-muted">
            This session, live. An evading booking becomes a fraud label only once the original booking is known to be fraud: an
            analyst confirms it, or it carries injected ground truth (labelled simulated; {h.simulated_labels} so far).
          </p>
          {h.pending_decisions.length > 0 && (
            <div className="mt-2" data-testid="redteam-pending">
              <p className="text-[0.82rem] font-semibold">Waiting for an analyst to confirm the original booking:</p>
              <ul className="mt-1 flex flex-col gap-1">
                {h.pending_decisions.map((id) => (
                  <li key={id} className="flex flex-wrap items-center gap-2 text-[0.82rem]">
                    <Link className="font-mono underline underline-offset-2" to={`/decisions/${encodeURIComponent(id)}`}>
                      {id}
                    </Link>
                    <button type="button" className="btn" data-testid="redteam-confirm" disabled={busy != null} onClick={() => confirm(id)}>
                      {busy === `confirm-${id}` ? 'Saving…' : 'Confirm original as fraud (analyst)'}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <button type="button" className="btn btn-primary" data-testid="redteam-retrain" disabled={busy != null || fresh === 0} onClick={retrain}>
              {busy === 'retrain' ? 'Retraining… (about 10-20 s)' : `Retrain with ${fresh} new evasion ${fresh === 1 ? 'label' : 'labels'}`}
            </button>
            <span className="text-[0.78rem] text-muted">same retrain and pre-registered gate as the Learning tab, unchanged</span>
          </div>
          {err && <ErrorBox message={err} testId="redteam-hardening-error" />}
          {handover && (
            <div className="mt-3" data-testid="redteam-retrain-result">
              <HandoverBanner h={handover} testId="redteam-handover" />
            </div>
          )}
          <div className="mt-2">
            <ModelInUseLine m={h.model_in_use} testId="redteam-model-in-use" />
          </div>
        </>
      ) : (
        <p className="text-[0.85rem] text-muted">Continuous learning is off in this configuration, so live attacks are not kept for retraining.</p>
      )}
      {off && (
        <div className="rt-gate mt-3" data-testid="redteam-gate" data-passed={String(off.passed)}>
          <p className="text-[0.82rem] font-semibold">Measured offline, for reference (test window, run once, {r?.run_at})</p>
          <p className="mt-0.5 text-[0.82rem]">
            {off.n_labels} evasions from training-window fraud ({off.n_evaded} of {off.n_attacked} attacked bookings got softer): the gate{' '}
            <strong>{off.passed ? 'passed' : 'REJECTED the hardened model'}</strong>.
          </p>
          <ul className="mt-1 list-disc pl-5 text-[0.8rem]">
            {off.failed_checks.map((c) => (
              <li key={c.name}>{c.detail}</li>
            ))}
          </ul>
        </div>
      )}
    </Section>
  )
}

export default function RedTeamView() {
  const res = useAsync(() => api.redteamResults(), [])
  const hard = useAsync(() => api.redteamHardening().catch(() => null), [])
  return (
    <div data-testid="redteam-view">
      <PageTitle title="Red team" />
      <div className="page">
      <p className="note mb-4">
        <Icon name="info" size={16} />
        <span>We attack our own model: we play the fraudster against our own system and publish how often we lose, instead of only how often we are right.</span>
      </p>
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        <LiveAttack onAttacked={hard.reload} />
        <div className="flex flex-col gap-4">
          {res.loading && <Loading what="red-team results" />}
          {res.error && <ErrorBox message={res.error} onRetry={res.reload} />}
          {res.data && <Measured r={res.data} />}
          {!hard.loading && <Hardening r={res.data ?? null} h={hard.data ?? null} reload={hard.reload} />}
        </div>
      </div>
      </div>
    </div>
  )
}
