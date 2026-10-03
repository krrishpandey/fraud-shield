import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { StreamAccuracy, StreamFeedRow, StreamFlagged, StreamMetricsResponse, StreamStatus } from '../api/types'
import { ActionPill } from '../components/ActionBadge'
import { ACTION_META, reasonText } from '../lib/domain'
import { fmtBRL, fmtInt } from '../lib/format'

/*
  Live stream: the seed-0 test window replayed into its own decision service at a steady rate.
  The point of the screen is twofold: it keeps up with continuous bookings, and the accuracy measured live
  equals the offline test on the same bookings.
*/

const RATES = [1, 5, 20, 50]
const POLL_MS = 1000

const num = (v: number | null | undefined, d = 3) => (v == null || Number.isNaN(v) ? 'n/a' : v.toFixed(d))
const pct = (v: number | null | undefined, d = 1) => (v == null || Number.isNaN(v) ? 'n/a' : `${(v * 100).toFixed(d)}%`)

function useLive() {
  const [status, setStatus] = useState<StreamStatus | null>(null)
  const [metrics, setMetrics] = useState<StreamMetricsResponse | null>(null)
  const [feed, setFeed] = useState<StreamFeedRow[]>([])
  const [flagged, setFlagged] = useState<StreamFlagged | null>(null)
  const [freshIds, setFreshIds] = useState<Set<string>>(new Set())
  const seenRef = useRef<Set<string> | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)
  useEffect(() => {
    let alive = true
    let timer: ReturnType<typeof setTimeout> | undefined
    const load = async () => {
      try {
        const [m, f, fl] = await Promise.all([api.streamMetrics(), api.streamFeed(40), api.streamFlagged()])
        if (!alive) return
        setMetrics(m)
        setStatus(m.status)
        setFeed(f)
        setFlagged(fl)
        // rows first seen in this refresh flash once; nothing flashes on the first load
        const prev = seenRef.current
        setFreshIds(new Set(prev ? fl.rows.filter((r) => !prev.has(r.decision_id)).map((r) => r.decision_id) : []))
        seenRef.current = new Set(fl.rows.map((r) => r.decision_id))
        setError(null)
        if (m.status.state === 'running' || m.status.state === 'paused') timer = setTimeout(load, POLL_MS)
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e))
      }
    }
    load()
    return () => {
      alive = false
      if (timer) clearTimeout(timer)
    }
  }, [tick])
  return { status, metrics, feed, flagged, freshIds, error, refresh: () => setTick((t) => t + 1) }
}

function Spark({ values, unit, label, testId }: { values: (number | null)[]; unit: string; label: string; testId: string }) {
  const W = 360
  const H = 90
  const pad = { l: 34, r: 8, t: 8, b: 18 }
  const vs = values.map((v) => (v == null ? null : v))
  const max = Math.max(1, ...vs.filter((v): v is number => v != null)) * 1.15
  const x = (i: number) => pad.l + (i / Math.max(values.length - 1, 1)) * (W - pad.l - pad.r)
  const y = (v: number) => pad.t + (1 - v / max) * (H - pad.t - pad.b)
  let d = ''
  vs.forEach((v, i) => {
    if (v == null) return
    d += `${d && vs[i - 1] != null ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`
  })
  const last = [...vs].reverse().find((v) => v != null)
  return (
    <figure className="live-spark" data-testid={testId}>
      <figcaption>
        <span>{label}</span>
        <strong className="tnum">{last == null ? 'n/a' : `${last.toFixed(unit === 'ms' ? 0 : 1)} ${unit}`}</strong>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${label}, last two minutes, latest ${last ?? 'n/a'} ${unit}`}>
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line x1={pad.l} x2={W - pad.r} y1={y(max * f)} y2={y(max * f)} className={f === 0 ? 'spark-base' : 'spark-grid'} />
            <text x={pad.l - 6} y={y(max * f) + 4} textAnchor="end" className="spark-lab">
              {(max * f).toFixed(0)}
            </text>
          </g>
        ))}
        <text x={pad.l} y={H - 3} className="spark-lab">2 min ago</text>
        <text x={W - pad.r} y={H - 3} textAnchor="end" className="spark-lab">now</text>
        <path d={d} className="spark-line" />
        {vs.map((v, i) =>
          v == null ? null : (
            <rect key={i} x={x(i) - 2} y={pad.t} width={4} height={H - pad.t - pad.b} className="spark-hit">
              <title>{`${values.length - i} s ago: ${v.toFixed(unit === 'ms' ? 0 : 1)} ${unit}`}</title>
            </rect>
          ),
        )}
      </svg>
    </figure>
  )
}

function AccuracyTable({ m }: { m: StreamMetricsResponse }) {
  const live = m.accuracy
  const same = m.offline_same_rows
  const full = m.offline_full
  if (!live || !same) return null
  if (live.n_fraud === 0) {
    return (
      <p className="live-note" data-testid="live-accuracy" role="status">
        {fmtInt(live.n)} bookings scored, none of them fraud yet. Fraud is about 1 in 80 bookings, so the accuracy
        measures appear once the first fraud bookings arrive. Whole window, offline: PR-AUC {num(full?.pr_auc)}, ROC-AUC{' '}
        {num(full?.roc_auc)}.
      </p>
    )
  }
  const rows: [string, keyof StreamAccuracy][] = [
    ['PR-AUC', 'pr_auc'],
    ['ROC-AUC', 'roc_auc'],
    ['Precision, top 1% per day', 'precision_top1pct_day'],
    ['Recall, top 1% per day', 'recall_top1pct_day'],
  ]
  return (
    <div data-testid="live-accuracy">
      <table className="live-acc">
        <caption className="sr-only">Accuracy measured on the live stream compared with the offline test</caption>
        <thead>
          <tr>
            <th scope="col">Measure</th>
            <th scope="col">Live stream</th>
            <th scope="col">Offline, same bookings</th>
            <th scope="col">Offline, whole window</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, k]) => (
            <tr key={k} data-testid={`live-acc-${k}`} data-live={String(live[k] ?? '')} data-offline={String(same[k] ?? '')}>
              <th scope="row">{label}</th>
              <td className="tnum live-acc-live">{num(live[k] as number | null)}</td>
              <td className="tnum">{num(same[k] as number | null)}</td>
              <td className="tnum muted">{num(full?.[k] as number | null)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="live-note">
        {fmtInt(live.n)} bookings scored so far, {fmtInt(live.n_fraud)} of them fraud. Largest difference between a live and
        an offline score: {m.consistency ? m.consistency.max_abs_score_diff.toFixed(5) : 'n/a'}. Ground truth comes from the
        replayed dataset and is used only for these counters, never by the model.
      </p>
      <dl className="live-outcomes">
        <div>
          <dt>Fraud stopped</dt>
          <dd className="tnum">{fmtInt(live.fraud_caught)}</dd>
        </div>
        <div>
          <dt>Fraud sent to a first-scan check</dt>
          <dd className="tnum">{fmtInt(live.fraud_scan_checked)}</dd>
        </div>
        <div>
          <dt>Fraud allowed</dt>
          <dd className="tnum">{fmtInt(live.fraud_missed)}</dd>
        </div>
        <div>
          <dt>Honest bookings stopped</dt>
          <dd className="tnum">
            {fmtInt(live.honest_stopped)} <span className="muted">({pct(live.honest_stopped_rate, 2)})</span>
          </dd>
        </div>
      </dl>
    </div>
  )
}

type FlagFilter = 'all' | 'hold' | 'block' | 'unreviewed'

function FlaggedPanel({ data, fresh }: { data: StreamFlagged | null; fresh: Set<string> }) {
  const [filter, setFilter] = useState<FlagFilter>('all')
  const rows = data?.rows ?? []
  const counts = data?.counts ?? { hold: 0, block: 0, unreviewed: 0 }
  const shown = rows.filter((r) =>
    filter === 'all' ? true : filter === 'unreviewed' ? r.reviewed == null : r.action === filter,
  )
  const chips: [FlagFilter, string, number][] = [
    ['all', 'All', counts.hold + counts.block],
    ['hold', 'Held', counts.hold],
    ['block', 'Blocked', counts.block],
    ['unreviewed', 'Not reviewed', counts.unreviewed],
  ]
  return (
    <section
      className="flagged"
      aria-labelledby="flagged-h"
      data-testid="live-flagged"
      data-count={counts.hold + counts.block}
      data-unreviewed={counts.unreviewed}
    >
      <div className="flagged-head">
        <h2 id="flagged-h" className="live-h2">
          Held and blocked
        </h2>
        <p className="flagged-sub">Every streamed booking that did not get a label, newest first. Click one to review it.</p>
        <div className="flagged-chips" role="radiogroup" aria-label="Show">
          {chips.map(([k, label, n]) => (
            <button
              key={k}
              type="button"
              role="radio"
              aria-checked={filter === k}
              className={`flagged-chip${filter === k ? ' is-on' : ''}`}
              onClick={() => setFilter(k)}
              data-testid={`live-flagged-filter-${k}`}
            >
              {label} <span className="tnum">{fmtInt(n)}</span>
            </button>
          ))}
        </div>
      </div>
      {shown.length === 0 ? (
        <p className="flagged-empty">
          {counts.hold + counts.block === 0
            ? 'Nothing held or blocked yet. Most bookings are allowed; held ones appear here as they arrive.'
            : 'No bookings match this filter.'}
        </p>
      ) : (
        <ol className="flagged-list" data-testid="live-flagged-list">
          {shown.map((r) => (
            <li
              key={r.decision_id}
              className={`flagged-row is-${r.action}${fresh.has(r.decision_id) ? ' is-new' : ''}`}
              style={{ ['--act' as string]: `var(--act-${r.action})` }}
              data-testid="live-flagged-row"
              data-decision-id={r.decision_id}
              data-action={r.action}
            >
              <Link to={`/decisions/${r.decision_id}`} className="flagged-link">
                <span className="flagged-act">
                  <ActionPill action={r.action} />
                </span>
                <span className="flagged-main">
                  <span className="flagged-reason">{r.reasons.length ? reasonText(r.reasons[0]) : 'High risk score'}</span>
                  <span className="flagged-meta tnum">
                    {r.booked_at?.replace('T', ' ').slice(5, 16)}, {r.route}, account {r.account_id?.slice(0, 8)}
                  </span>
                  <span className="flagged-why" data-testid="live-flagged-why"
                    data-source={r.explanation?.source ?? ''} data-status={r.explanation_status ?? ''}>
                    {r.explanation ? (
                      <>
                        <span className="flagged-why-text">{r.explanation.text}</span>
                        <span className="flagged-why-src">
                          {r.explanation.source === 'llm'
                            ? `${r.explanation.model_id ?? 'Language model'}, fact-checked against the decision record`
                            : 'Fixed template (the model text was not used)'}
                        </span>
                      </>
                    ) : r.explanation_status === 'failed' ? (
                      'No explanation was produced.'
                    ) : (
                      'Writing the explanation...'
                    )}
                  </span>
                </span>
                <span className="flagged-cost tnum">{fmtBRL(r.carrier_cost)}</span>
                <span className="flagged-score tnum">{r.score == null ? 'n/a' : `${(r.score * 100).toFixed(1)}%`}</span>
                <span className={`flagged-status${r.reviewed ? ' is-done' : ''}`} data-testid="live-flagged-status">
                  {r.reviewed === 'fraud' ? 'Fraud confirmed' : r.reviewed === 'legit' ? 'Marked legitimate' : 'Not reviewed'}
                </span>
              </Link>
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}

export default function LiveView() {
  const { status, metrics, feed, flagged, freshIds, error, refresh } = useLive()
  const [rate, setRate] = useState(20)
  const [busy, setBusy] = useState(false)
  const [actionErr, setActionErr] = useState<string | null>(null)
  const feedRef = useRef<HTMLOListElement>(null)

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true)
    setActionErr(null)
    try {
      await fn()
      refresh()
    } catch (e) {
      setActionErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const state = status?.state ?? 'loading'
  const running = state === 'running' || state === 'paused'
  const load = metrics?.load
  const timeline = metrics?.timeline ?? []
  const total = status?.total ?? 0

  if (error && !metrics) {
    return (
      <div className="live">
        <h1 className="live-h1">Live stream</h1>
        <p className="live-unavailable" role="alert" data-testid="live-unavailable">
          The live stream is not available: {error}
        </p>
      </div>
    )
  }

  return (
    <div className="live" data-testid="live-view" data-state={state}>
      <header className="live-head">
        <div>
          <h1 className="live-h1">Live stream</h1>
          <p className="live-sub">
            Replays the seed-0 test window (real Olist bookings plus the injected fraud, in booking-time order) into its own
            decision service, the way a booking system sends bookings. A simulation of continuous traffic, not live carrier data.
          </p>
        </div>
        <p className="live-state" data-testid="live-state" data-state={state}>
          <span className={`live-dot is-${state}`} aria-hidden="true" />
          {state === 'loading' ? 'Loading' : state === 'idle' ? 'Not started' : state[0].toUpperCase() + state.slice(1)}
          {total > 0 && (
            <span className="tnum" data-testid="live-sent">
              {' '}
              {fmtInt(status?.sent)} of {fmtInt(total)} sent
            </span>
          )}
        </p>
      </header>

      <div className="live-controls" role="group" aria-label="Stream controls">
        <span className="live-ctl-label" id="rate-label">Bookings per second</span>
        <div className="live-rates" role="radiogroup" aria-labelledby="rate-label">
          {RATES.map((r) => (
            <button
              key={r}
              type="button"
              role="radio"
              aria-checked={rate === r}
              className={`live-rate${rate === r ? ' is-on' : ''}`}
              disabled={running}
              onClick={() => setRate(r)}
              data-testid={`live-rate-${r}`}
            >
              {r}
            </button>
          ))}
        </div>
        {!running ? (
          <button type="button" className="btn btn-primary" disabled={busy || state === 'loading' || status?.available === false}
            onClick={() => act(() => api.streamStart({ rate, concurrency: 4 }))} data-testid="live-start">
            {state === 'idle' ? 'Start the stream' : 'Start again'}
          </button>
        ) : (
          <>
            {state === 'running' ? (
              <button type="button" className="btn" disabled={busy} onClick={() => act(api.streamPause)} data-testid="live-pause">
                Pause
              </button>
            ) : (
              <button type="button" className="btn" disabled={busy} onClick={() => act(api.streamResume)} data-testid="live-resume">
                Resume
              </button>
            )}
            <button type="button" className="btn" disabled={busy} onClick={() => act(api.streamStop)} data-testid="live-stop">
              Stop
            </button>
          </>
        )}
        {actionErr && <p className="text-[0.85rem] text-danger" role="alert">{actionErr}</p>}
      </div>

      {state === 'loading' ? (
        <p className="live-empty" role="status">Loading the stream status...</p>
      ) : state === 'idle' ? (
        <p className="live-empty">
          Start the stream to watch every booking get a decision as it arrives. At 20 bookings per second the whole window of
          about 22,900 bookings takes about 19 minutes.
        </p>
      ) : (
        <>
          <FlaggedPanel data={flagged} fresh={freshIds} />
          <section className="live-main">
            <div className="live-feed-wrap">
              <h2 className="live-h2">Newest decisions</h2>
              <ol className="live-feed" ref={feedRef} data-testid="live-feed" aria-live="off">
                {feed.map((r) => (
                  <li key={r.decision_id} className="live-row" style={{ ['--act' as string]: `var(--act-${r.action})` }}
                    data-testid="live-feed-row" data-decision-id={r.decision_id}>
                    <Link to={`/decisions/${r.decision_id}`} className="live-row-link"
                      aria-label={`${ACTION_META[r.action]?.label ?? r.action}, ${r.route}, open decision`}>
                      <span className="tnum live-time">{r.booked_at.replace('T', ' ').slice(5, 16)}</span>
                      <span className="live-route">{r.route}</span>
                      <span className="tnum live-cost">{fmtBRL(r.carrier_cost)}</span>
                      <span className="tnum live-score">{(r.score * 100).toFixed(1)}%</span>
                      <ActionPill action={r.action} />
                    </Link>
                  </li>
                ))}
              </ol>
            </div>
            <div className="live-acc-wrap">
              <h2 className="live-h2">Accuracy, live against the offline test</h2>
              {metrics && <AccuracyTable m={metrics} />}
            </div>
          </section>

          <section className="live-load">
            <Spark testId="live-throughput" label="Bookings scored per second" unit="/s"
              values={timeline.map((b) => b.scored)} />
            <Spark testId="live-latency" label="Scoring time, p95 per second" unit="ms"
              values={timeline.map((b) => b.latency_p95_ms)} />
            <div className="live-facts">
              <dl>
                <div><dt>Scored</dt><dd className="tnum">{fmtInt(load?.scored)}</dd></div>
                <div><dt>Waiting in flight</dt><dd className="tnum">{fmtInt(load?.in_flight)}</dd></div>
                <div><dt>Scoring time p50 / p99</dt><dd className="tnum">{num(load?.latency_p50_ms, 0)} / {num(load?.latency_p99_ms, 0)} ms</dd></div>
                <div><dt>Errors / retries</dt><dd className="tnum">{fmtInt(load?.errors)} / {fmtInt(load?.retries)}</dd></div>
                <div><dt>Decided by the backup model</dt><dd className="tnum">{pct(metrics?.degraded_share, 0)}</dd></div>
              </dl>
              {metrics?.explanations && (
                <p className="live-note" data-testid="live-explanations">
                  Explanations: every held or blocked booking gets one from the language model
                  ({fmtInt(metrics.explanations.llm_held_blocked ?? 0)} so far). Other decisions share{' '}
                  {metrics.explanations.llm_cap_per_min} per minute; {fmtInt(metrics.explanations.template)} used the fixed template.
                </p>
              )}
            </div>
            <div className="live-mix" data-testid="live-actions">
              <h2 className="live-h2">Decisions so far</h2>
              <ul>
                {Object.entries(metrics?.actions ?? {}).map(([a, n]) => {
                  const share = load?.scored ? n / load.scored : 0
                  return (
                    <li key={a} style={{ ['--act' as string]: `var(--act-${a})` }} data-action={a} data-count={n}>
                      <span className="live-mix-name">{ACTION_META[a as keyof typeof ACTION_META]?.label ?? a}</span>
                      <span className="live-mix-bar" aria-hidden="true"><span style={{ width: `${Math.max(share * 100, n ? 0.6 : 0)}%` }} /></span>
                      <span className="tnum live-mix-n">{fmtInt(n)}</span>
                    </li>
                  )
                })}
              </ul>
            </div>
          </section>
        </>
      )}
    </div>
  )
}
