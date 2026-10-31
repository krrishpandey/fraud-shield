import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { Verification } from '../api/verify'
import '../verify.css'

type State = 'done' | 'now' | 'todo' | 'bad' | 'good'
type Step = { label: string; sub?: string; state: State }

const HELD = ['hold', 'owner_confirm']
const FINAL = ['released', 'blocked', 'analyst_review']

/** Builds the package's path so far from the decision and its owner call (if any). */
function stepsFor(action: string, risk: number | null | undefined, v: Verification | null): Step[] {
  const pct = risk == null ? '' : `risk ${(risk * 100).toFixed(1)}%`
  const s: Step[] = [
    { label: 'Booked', state: 'done' },
    { label: 'Checked', sub: pct, state: 'done' },
  ]
  if (action === 'allow' || action === 'allow_scan_gated') {
    s.push({ label: 'Allowed', state: 'good' }, { label: 'Label printed', sub: action === 'allow_scan_gated' ? 'depot weighs it' : '', state: 'good' })
    return s
  }
  if (action === 'block') {
    s.push({ label: 'Blocked', sub: 'hard fraud signal', state: 'bad' })
    return s
  }
  if (!HELD.includes(action)) {
    s.push({ label: 'Held', sub: 'sent to a person', state: 'now' })
    return s
  }
  s.push({ label: 'Held', sub: 'no label yet', state: 'done' })
  const att = v?.attempts[v.attempts.length - 1]
  const seen = new Set((att?.statuses ?? []).map((x) => x.status))
  const final = v ? FINAL.includes(v.status) : false
  if (v && !att && v.case) {
    s.push({ label: 'Not called', sub: 'number too new', state: 'bad' }, { label: 'Needs a person', state: 'now' })
    return s
  }
  const rang = seen.has('ringing') || seen.has('in-progress') || seen.has('completed') || seen.has('call-disconnected')
  const talked = seen.has('in-progress') || seen.has('completed') || seen.has('call-disconnected')
  const hung = seen.has('call-disconnected') || seen.has('completed') || !!v?.outcome
  const noAnswer = ['no-answer', 'busy', 'failed'].includes(att?.status ?? '')
  s.push({ label: 'Calling owner', sub: v?.mode === 'real' ? 'real phone call' : v?.mode === 'simulated' ? 'demo call' : '', state: att ? 'done' : 'now' })
  s.push({ label: noAnswer ? 'No answer' : 'Ringing', sub: v && v.attempts.length > 1 ? `try ${v.attempts.length}` : '', state: noAnswer ? 'bad' : rang ? 'done' : att ? 'now' : 'todo' })
  if (noAnswer && final) {
    s.push({ label: 'Needs a person', sub: 'nobody answered', state: 'now' })
    return s
  }
  s.push({ label: 'Talking', state: talked && hung ? 'done' : talked ? 'now' : 'todo' })
  const said = v?.outcome?.owner_confirmed
  s.push({
    label: said ? `Owner: "${said === 'unsure' ? 'not sure' : said}"` : 'Owner answers',
    sub: hung && !said ? 'reading the answer' : '',
    state: said === 'no' ? 'bad' : said === 'yes' ? 'good' : hung ? 'now' : 'todo',
  })
  if (v?.final_action === 'block') s.push({ label: 'Stopped', sub: v.account_locked ? 'account locked' : '', state: 'bad' })
  else if (v?.final_action) s.push({ label: 'Released', sub: 'label printed', state: 'good' })
  else if (v?.status === 'analyst_review') s.push({ label: 'Needs a person', state: 'now' })
  else s.push({ label: 'Result', state: 'todo' })
  return s
}

/** The package's journey, updated live while its owner call runs. */
export function PackageJourney({ id, action, risk }: { id: string; action: string; risk?: number | null }) {
  const [v, setV] = useState<Verification | null>(null)
  useEffect(() => {
    if (!HELD.includes(action)) return
    let alive = true
    let t: number | undefined
    const tick = () =>
      api.verification(id).then(
        (x) => {
          if (!alive) return
          setV(x)
          if (!FINAL.includes(x.status)) t = window.setTimeout(tick, 1200)
        },
        () => {
          if (alive) t = window.setTimeout(tick, 1500) // the call starts a moment after the score
        },
      )
    tick()
    return () => {
      alive = false
      window.clearTimeout(t)
    }
  }, [id, action])
  const steps = stepsFor(action, risk, v)
  return (
    <ol className="pj" aria-label="Package journey" data-testid="package-journey">
      {steps.map((st, i) => (
        <li key={i} className={`pj-step is-${st.state}`}>
          <span className="pj-dot" aria-hidden="true" />
          <b>{st.label}</b>
          {st.sub && <span className="pj-sub">{st.sub}</span>}
        </li>
      ))}
    </ol>
  )
}
