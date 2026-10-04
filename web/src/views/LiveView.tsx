import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { StreamAccuracy, StreamFeedRow, StreamFlagged, StreamMetricsResponse, StreamStatus } from '../api/types'
import { ActionPill } from '../components/ActionBadge'
import { ErrorBox, Loading, Page, PageTitle } from '../components/common'
import { SlideInd } from '../components/SlideInd'
import { Icon, type IconName } from '../components/Icon'
import LabelFreeMonitorCard from '../components/LabelFreeMonitorCard'
import { ModelInUseBadge } from '../components/ModelHandover'
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
          <SlideInd />
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

/** Bookings scored per second over the last two minutes, as bars; the newest bar is lit. */
function RateBars({ values }: { values: (number | null)[] }) {
  const vs = values.slice(-40)
  const max = Math.max(1, ...vs.map((v) => v ?? 0))
  const last = [...vs].reverse().find((v) => v != null) ?? null
  const prev = vs.length > 10 ? vs.slice(-11, -1).reduce<number>((a, v) => a + (v ?? 0), 0) / 10 : null
  const delta = last != null && prev != null ? last - prev : null
  return (
    <figure className="rate-card" data-testid="live-throughput">
      <figcaption>
        <span className="rate-h">Bookings scored per second</span>
        <span className="rate-big tnum">{last == null ? '—' : last.toFixed(0)}</span>
        {delta != null && (
          <span className={`rate-delta tnum${delta < 0 ? ' is-down' : ''}`}>
            {delta >= 0 ? '+' : ''}
            {delta.toFixed(1)} vs last 10 s
          </span>
        )}
      </figcaption>
      <div className="rate-bars" role="img" aria-label={`Bookings scored per second, latest ${last ?? 'n/a'}`}>
        {vs.length === 0 ? (
          <span className="rate-empty">Waiting for the first bookings</span>
        ) : (
          vs.map((v, i) => <span key={i} className={i === vs.length - 1 ? 'is-now' : ''} style={{ height: `${Math.max(((v ?? 0) / max) * 100, 4)}%` }} />)
        )}
      </div>
    </figure>
  )
}

function Stat({ icon, label, value, sub }: { tone?: string; icon: IconName; label: string; value: string; sub: string }) {
  return (
    <div className="panel stat">
      <div className="stat-h">
        <Icon name={icon} size={18} />
        <span>{label}</span>
      </div>
      <div className="stat-v tnum">{value}</div>
      <div className="stat-s">{sub}</div>
    </div>
  )
}

type FeedFilter = 'all' | 'flagged' | 'allowed'

export default function LiveView() {
  const { status, metrics, feed, flagged, freshIds, error, refresh } = useLive()
  const [rate, setRate] = useState(20)
  const [busy, setBusy] = useState(false)
  const [actionErr, setActionErr] = useState<string | null>(null)
  const [ff, setFf] = useState<FeedFilter>('all')
  const [q, setQ] = useState('')
  const nav = useNavigate()

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
  const acts = metrics?.actions ?? {}
  const pulled = (acts.hold ?? 0) + (acts.block ?? 0) + (acts.review ?? 0) + (acts.owner_confirm ?? 0)
  const term = q.trim().toLowerCase()
  const rows = feed.filter(
    (r) =>
      (ff === 'all' || (ff === 'allowed' ? r.action === 'allow' : r.action !== 'allow')) &&
      (!term || r.route.toLowerCase().includes(term) || r.booking_id.toLowerCase().includes(term) || r.decision_id.toLowerCase().includes(term)),
  )
  const stateWord = state === 'loading' ? 'Loading' : state === 'idle' ? 'Not started' : state[0].toUpperCase() + state.slice(1)

  const controls = (
    <>
      <div className="seg" role="radiogroup" aria-label="Bookings per second">
        <SlideInd />
        {RATES.map((r) => (
          <button key={r} type="button" role="radio" aria-checked={rate === r} disabled={running} onClick={() => setRate(r)} data-testid={`live-rate-${r}`} title={`${r} bookings per second`}>
            ×{r}
          </button>
        ))}
      </div>
      {!running ? (
        <button
          type="button"
          className="btn btn-primary"
          disabled={busy || state === 'loading' || status?.available === false}
          onClick={() => act(() => api.streamStart({ rate, concurrency: 4 }))}
          data-testid="live-start"
        >
          <Icon name="play" size={15} />
          {state === 'idle' ? 'Start the stream' : 'Start again'}
        </button>
      ) : (
        <>
          {state === 'running' ? (
            <button type="button" className="btn" disabled={busy} onClick={() => act(api.streamPause)} data-testid="live-pause">
              <Icon name="pause" size={15} />
              Pause
            </button>
          ) : (
            <button type="button" className="btn" disabled={busy} onClick={() => act(api.streamResume)} data-testid="live-resume">
              <Icon name="play" size={15} />
              Resume
            </button>
          )}
          <button type="button" className="btn" disabled={busy} onClick={() => act(api.streamStop)} data-testid="live-stop">
            <Icon name="x" size={15} />
            Stop
          </button>
        </>
      )}
      <span className={`live-state is-${state}`} data-testid="live-state" data-state={state}>
        <span className="live-dot" aria-hidden="true" />
        {stateWord}
        {total > 0 && (
          <span className="tnum text-muted" data-testid="live-sent">
            {' '}
            · {fmtInt(status?.sent)} of {fmtInt(total)} sent
          </span>
        )}
      </span>
    </>
  )

  if (error && !metrics) {
    return (
      <div className="live">
        <PageTitle title="Live stream" />
        <Page>
          <p className="note" role="alert" data-testid="live-unavailable">
            <Icon name="info" size={16} />
            <span>The live stream is not available: {error}</span>
          </p>
        </Page>
      </div>
    )
  }

  return (
    <div className="live" data-testid="live-view" data-state={state}>
      <PageTitle title="Live stream">{controls}</PageTitle>
      <Page>
        {actionErr && <ErrorBox message={actionErr} />}
        <div className="model-line">
          <ModelInUseBadge testId="live-model-in-use" />
        </div>
        <div className="stats">
          <Stat tone="indigo" icon="box" label="Scored" value={fmtInt(load?.scored ?? 0)} sub={total ? `of ${fmtInt(total)} held-out bookings` : 'of the test-window replay'} />
          <Stat tone="teal" icon="list" label="Waiting in flight" value={fmtInt(load?.in_flight ?? 0)} sub={`${fmtInt(load?.errors ?? 0)} errors · ${fmtInt(load?.retries ?? 0)} retries`} />
          <Stat tone="rose" icon="lock" label="Labels pulled" value={fmtInt(pulled)} sub="stopped before printing" />
          <Stat
            tone="green"
            icon="pulse"
            label="Per decision"
            value={load?.latency_p50_ms == null ? '—' : `${Math.round(load.latency_p50_ms)} ms`}
            sub={load?.latency_p95_ms == null ? 'median scoring time' : `p95 ${Math.round(load.latency_p95_ms)} ms`}
          />
        </div>

        {state === 'loading' ? (
          <Loading what="the stream status" />
        ) : state === 'idle' ? (
          <div className="panel live-empty">
            <div>
              <h2>Replay the test window</h2>
              <p>About 22,900 real and injected bookings, sent in booking-time order.</p>
            </div>
          </div>
        ) : (
          <>
            <div className="live-grid">
              <section className="panel min-w-0" aria-labelledby="feed-h">
                <div className="card-h">
                  <h2 id="feed-h">
                    <Icon name="pulse" size={18} />
                    Newest decisions
                  </h2>
                  <div className="ml-auto flex flex-wrap items-center gap-2">
                    <div className="seg" role="radiogroup" aria-label="Show">
                      <SlideInd />
                      {(
                        [
                          ['all', 'All'],
                          ['flagged', 'Flagged'],
                          ['allowed', 'Allowed'],
                        ] as [FeedFilter, string][]
                      ).map(([k, l]) => (
                        <button key={k} type="button" role="radio" aria-checked={ff === k} onClick={() => setFf(k)}>
                          {l}
                        </button>
                      ))}
                    </div>
                    <label className="search w-[210px]">
                      <Icon name="search" size={16} />
                      <span className="sr-only">Filter by lane or id</span>
                      <input className="field" placeholder="Lane or id" value={q} onChange={(e) => setQ(e.target.value)} />
                    </label>
                  </div>
                </div>
                <div className="feed-scroll">
                  <table className="tbl feed-tbl" data-testid="live-feed" aria-live="off">
                    <thead>
                      <tr>
                        <th scope="col">Time</th>
                        <th scope="col">Booking</th>
                        <th scope="col">Lane</th>
                        <th scope="col" className="r">R$</th>
                        <th scope="col" className="r">Fraud P</th>
                        <th scope="col">Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((r) => (
                        <tr
                          key={r.decision_id}
                          className="click"
                          data-testid="live-feed-row"
                          data-decision-id={r.decision_id}
                          onClick={() => nav(`/decisions/${r.decision_id}`)}
                        >
                          <td className="mono text-muted">
                            <Link
                              to={`/decisions/${r.decision_id}`}
                              className="row-link"
                              onClick={(e) => e.stopPropagation()}
                              aria-label={`${ACTION_META[r.action]?.label ?? r.action}, ${r.route}, open decision`}
                            >
                              {r.booked_at.replace('T', ' ').slice(11, 19) || r.booked_at.slice(5, 16)}
                            </Link>
                          </td>
                          <td className="mono">{r.booking_id.replace(/^bk_/, '').slice(0, 8)}</td>
                          <td>{r.route.replace(' to ', ' → ')}</td>
                          <td className="r tnum mono">{r.carrier_cost.toFixed(2).replace('.', ',')}</td>
                          <td className="r tnum mono">{r.score.toFixed(2)}</td>
                          <td>
                            <ActionPill action={r.action} />
                          </td>
                        </tr>
                      ))}
                      {rows.length === 0 && (
                        <tr>
                          <td colSpan={6} className="py-8 text-center text-muted">
                            {feed.length === 0 ? 'Waiting for the first decisions…' : 'No decisions match this filter.'}
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </section>

              <div className="flex min-w-0 flex-col gap-[18px]">
                <section className="panel rate-panel">
                  <RateBars values={timeline.map((b) => b.scored)} />
                </section>
                <section className="panel">
                  <div className="card-h">
                    <h2>
                      <Icon name="chart" size={18} />
                      Outcomes so far
                    </h2>
                    <span className="ml-auto pill-mute">Measured live</span>
                  </div>
                  <div className="card-b">{metrics && <AccuracyTable m={metrics} />}</div>
                </section>
              </div>
            </div>

            <FlaggedPanel data={flagged} fresh={freshIds} />

            <LabelFreeMonitorCard scored={load?.scored ?? 0} />

            <div className="live-load">
              <section className="panel min-w-0">
                <div className="card-b">
                  <Spark testId="live-latency" label="Scoring time, p95 per second" unit="ms" values={timeline.map((b) => b.latency_p95_ms)} />
                </div>
              </section>
              <section className="panel min-w-0">
                <div className="card-b live-facts">
                  <dl className="kv text-[14px]">
                    <dt>Scored</dt>
                    <dd className="tnum">{fmtInt(load?.scored)}</dd>
                    <dt>Scoring time p50 / p99</dt>
                    <dd className="tnum">
                      {num(load?.latency_p50_ms, 0)} / {num(load?.latency_p99_ms, 0)} ms
                    </dd>
                    <dt>Errors / retries</dt>
                    <dd className="tnum">
                      {fmtInt(load?.errors)} / {fmtInt(load?.retries)}
                    </dd>
                    <dt>Decided by the backup model</dt>
                    <dd className="tnum">{pct(metrics?.degraded_share, 0)}</dd>
                  </dl>
                  {metrics?.scans && (
                    <p className="mt-3 text-[13px] text-muted" data-testid="live-scans">
                      Depot scans (simulated from the dataset's true weights): {fmtInt(metrics.scans.checked)} parcels weighed,{' '}
                      {fmtInt(metrics.scans.mismatch)} heavier than declared. Their accounts' next parcels are weighed too.
                    </p>
                  )}
                  {metrics?.explanations && (
                    <p className="mt-2 text-[13px] text-muted" data-testid="live-explanations">
                      Explanations: every held or blocked booking gets one from the language model ({fmtInt(metrics.explanations.llm_held_blocked ?? 0)}{' '}
                      so far). Other decisions share {metrics.explanations.llm_cap_per_min} per minute; {fmtInt(metrics.explanations.template)} used
                      the fixed template.
                    </p>
                  )}
                </div>
              </section>
              <section className="panel min-w-0" data-testid="live-actions">
                <div className="card-h">
                  <h2>Decisions so far</h2>
                </div>
                <div className="card-b">
                  <ul className="mix-rows">
                    {Object.entries(acts).map(([a, n]) => {
                      const share = load?.scored ? n / load.scored : 0
                      return (
                        <li key={a} data-action={a} data-count={n}>
                          <span className="dot" style={{ background: `var(--act-${a})` }} aria-hidden="true" />
                          <span className="flex-1">{ACTION_META[a as keyof typeof ACTION_META]?.short ?? a}</span>
                          <b className="tnum">{fmtInt(n)}</b>
                          <span className="tnum mix-pct">{pct(share, share < 0.1 ? 1 : 0)}</span>
                        </li>
                      )
                    })}
                  </ul>
                </div>
              </section>
            </div>
          </>
        )}
      </Page>
    </div>
  )
}
