import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, type TooltipContentProps } from 'recharts'
import { api } from '../api/client'
import { ACTIONS, type DashboardMetrics, type FirstScanDial } from '../api/types'
import { ErrorBox, Loading, Page, PageTitle, Section } from '../components/common'
import { SlideInd } from '../components/SlideInd'
import { Icon } from '../components/Icon'
import { ACTION_META, TYPOLOGY_META, typologyName } from '../lib/domain'
import { fmtBRL, fmtInt, fmtMs, fmtPct } from '../lib/format'
import { useCssVars } from '../lib/useCssVars'

function Kpi({ name, label, value, sub }: { name: string; label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="panel kpi" data-testid={`dashboard-kpi-${name}`}>
      <div className="min-w-0">
        <div className="kpi-label">{label}</div>
        <div className="kpi-value">{value}</div>
        {sub && <div className="kpi-sub">{sub}</div>}
      </div>
    </div>
  )
}

const SLOT_VARS = Array.from({ length: 8 }, (_, i) => `--series-${i + 1}`)
const CHROME_VARS = ['--grid', '--axis', '--muted', '--surface', '--ink', '--series-1']

type Mode = 'typology' | 'held'

function TrendChart({ m }: { m: DashboardMetrics }) {
  const [mode, setMode] = useState<Mode>('typology')
  const [asTable, setAsTable] = useState(false)
  const c = useCssVars([...SLOT_VARS, ...CHROME_VARS])

  const typologies = useMemo(() => {
    const s = new Set<string>()
    m.trend.forEach((t) => Object.keys(t.by_typology ?? {}).forEach((k) => s.add(k)))
    return [...s].sort()
  }, [m])
  const rows = useMemo(
    () => m.trend.map((t) => ({ date: t.date, held: t.held, fraud_stopped: t.fraud_stopped, ...Object.fromEntries(typologies.map((k) => [k, t.by_typology?.[k] ?? 0])) })),
    [m, typologies],
  )
  const colorOf = (t: string) => c[`--series-${TYPOLOGY_META[t]?.slot ?? 8}`]

  const TooltipBody = ({ active, payload, label }: TooltipContentProps) => {
    if (!active || !payload?.length) return null
    const row = payload[0].payload as Record<string, number | string>
    return (
      <div className="chart-tip">
        <div className="font-semibold">{label}</div>
        <div className="tnum">Fraud stopped: {row.fraud_stopped}</div>
        <div className="tnum">Held: {row.held}</div>
        {mode === 'typology' &&
          typologies
            .filter((t) => Number(row[t]) > 0)
            .map((t) => (
              <div key={t} className="tnum flex items-center gap-1.5">
                <span aria-hidden="true" className="dot" style={{ background: colorOf(t) }} />
                {typologyName(t)}: {row[t]}
              </div>
            ))}
      </div>
    )
  }

  return (
    <div data-testid="trend-chart" data-mode={mode}>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <div role="group" aria-label="Series" className="seg">
          <SlideInd />
          {(
            [
              ['typology', 'By fraud type'],
              ['held', 'Held shipments'],
            ] as [Mode, string][]
          ).map(([k, l]) => (
            <button key={k} type="button" aria-pressed={mode === k} data-testid={`trend-mode-${k}`} onClick={() => setMode(k)}>
              {l}
            </button>
          ))}
        </div>
        <button type="button" className="btn ml-auto" aria-pressed={asTable} onClick={() => setAsTable((v) => !v)} data-testid="trend-table-toggle">
          <Icon name={asTable ? 'chart' : 'lines'} size={16} />
          {asTable ? 'Show chart' : 'Show as table'}
        </button>
      </div>
      {mode === 'typology' && !asTable && typologies.length > 0 && (
        <ul className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-ink-2" aria-label="Legend">
          {typologies.map((t) => (
            <li key={t} className="flex items-center gap-1.5">
              <span aria-hidden="true" className="dot" style={{ background: colorOf(t) }} />
              {typologyName(t)}
            </li>
          ))}
        </ul>
      )}
      {asTable ? (
        <div className="max-h-[280px] overflow-auto">
          <table className="tbl" data-testid="trend-table">
            <thead className="sticky top-0 bg-surface">
              <tr>
                <th scope="col">Date</th>
                <th scope="col" className="r">Fraud stopped</th>
                <th scope="col" className="r">Held</th>
                {typologies.map((t) => (
                  <th key={t} scope="col" className="r" title={typologyName(t)}>
                    {t}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.date} className="tnum">
                  <td>{r.date}</td>
                  <td className="r">{r.fraud_stopped}</td>
                  <td className="r">{r.held}</td>
                  {typologies.map((t) => (
                    <td key={t} className="r">
                      {(r as Record<string, number | string>)[t]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : rows.length === 0 ? (
        <div className="empty-chart">
          <Icon name="chart" size={22} />
          <p>No decisions in this window yet. Score a booking or start the live stream and the days fill in here.</p>
        </div>
      ) : (
        <div className="h-[250px]" role="img" aria-label={mode === 'typology' ? 'Fraud bookings stopped per day, stacked by typology' : 'Held shipments per day'}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows} margin={{ top: 4, right: 8, bottom: 0, left: -12 }} barCategoryGap="22%">
              <CartesianGrid vertical={false} stroke={c['--grid']} />
              <XAxis dataKey="date" tickFormatter={(d: string) => d.slice(5)} tick={{ fill: c['--muted'], fontSize: 12 }} tickLine={false} axisLine={{ stroke: c['--axis'] }} minTickGap={12} />
              <YAxis allowDecimals={false} tick={{ fill: c['--muted'], fontSize: 12 }} tickLine={false} axisLine={false} width={44} />
              <Tooltip content={TooltipBody} cursor={{ fill: c['--grid'], opacity: 0.6 }} isAnimationActive={false} />
              {mode === 'typology' ? (
                typologies.map((t, i) => (
                  <Bar key={t} dataKey={t} name={typologyName(t)} stackId="t" fill={colorOf(t)} radius={i === typologies.length - 1 ? [4, 4, 0, 0] : 0} isAnimationActive={false} />
                ))
              ) : (
                <Bar dataKey="held" name="Held shipments" fill={c['--series-1']} radius={[4, 4, 0, 0]} isAnimationActive={false} />
              )}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      <p className="mt-2 text-[13px] text-muted">
        Bookings per day{m.window.from ? `, ${m.window.from} to ${m.window.to}` : ''}. Fraud type comes from the evaluation labels.
      </p>
    </div>
  )
}

/** "What the gate did": share let through with no friction, then a stacked bar and a row per stopping action. */
function ActionMix({ m }: { m: DashboardMetrics }) {
  const by = m.totals.by_action
  const total = m.totals.bookings || 0
  const others = ACTIONS.filter((a) => a !== 'allow')
  const otherTotal = others.reduce((s, a) => s + (by[a] ?? 0), 0)
  const allowShare = total ? (by.allow ?? 0) / total : 0
  return (
    <div data-testid="action-mix">
      <p className="gate-hero">
        <span className="gate-big tnum">{total ? fmtPct(allowShare, 1) : '—'}</span>
        <span className="text-ink-2">let through, no friction</span>
      </p>
      <p className="mt-3 text-[14px] text-muted">
        <span className="tnum">{fmtInt(by.allow ?? 0)}</span> of <span className="tnum">{fmtInt(total)}</span> bookings allowed normally. The other{' '}
        {total ? fmtPct(otherTotal / total, 1) : '0%'} · <span className="tnum">{fmtInt(otherTotal)}</span> bookings:
      </p>
      <div className="mix-bar" aria-hidden="true">
        {otherTotal > 0 ? (
          others.map((a) => {
            const v = by[a] ?? 0
            return v ? <span key={a} style={{ flexGrow: v, background: `var(--act-${a})` }} /> : null
          })
        ) : (
          <span style={{ flexGrow: 1, background: 'var(--surface-2)' }} />
        )}
      </div>
      <ul className="mix-rows">
        {others.map((a) => {
          const v = by[a] ?? 0
          return (
            <li key={a} data-testid={`action-mix-${a}`} data-count={v}>
              <span className="dot" style={{ background: `var(--act-${a})` }} aria-hidden="true" />
              <span className="flex-1">{ACTION_META[a].short}</span>
              <b className="tnum">{fmtInt(v)}</b>
              <span className="tnum mix-pct">{otherTotal ? `${Math.round((v / otherTotal) * 100)}%` : '—'}</span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

/* Offline test results (artifacts/results_gbm.md): LightGBM on real data columns only, 10 seeds, later time window. */
const OFFLINE = {
  rules: 0.17,
  types: [
    { k: 'T1', name: 'Label resale on a taken-over account', v: 0.96 },
    { k: 'T7', name: 'Third-party billing on a leaked account number', v: 0.85 },
    { k: 'T5', name: 'Test then burst · held out', v: 0.76 },
    { k: 'T4', name: 'Cost-maximizing takeover', v: 0.63 },
    { k: 'T2', name: 'Bust-out new account', v: 0.46 },
    { k: 'T3', name: 'Reshipping drop address · held out', v: 0.23 },
    { k: 'T6', name: 'Weight declared below real', v: 0.02 },
  ],
}

function ByFraudType() {
  return (
    <Section
      title="PR-AUC by fraud type"
      icon="chart"
      aside={
        <span className="flex items-center gap-2">
          <span className="rules-mark" aria-hidden="true" /> Rules {OFFLINE.rules}
        </span>
      }
      testId="dashboard-by-type"
    >
      <ul className="types">
        {OFFLINE.types.map((t) => (
          <li key={t.k}>
            <span className="types-k">{t.k}</span>
            <div className="min-w-0 flex-1">
              <div className="types-name">{t.name}</div>
              <div className="types-track" role="img" aria-label={`${t.name}: PR-AUC ${t.v}, rules ${OFFLINE.rules}`}>
                <span className="types-fill" style={{ width: `${Math.max(t.v * 100, 1.5)}%`, background: t.v >= 0.5 ? 'var(--ink-2)' : '#4a4a52' }} />
                <span className="types-rules" style={{ left: `${OFFLINE.rules * 100}%` }} />
              </div>
            </div>
            <b className="types-v tnum">{t.v.toFixed(2)}</b>
          </li>
        ))}
      </ul>
      <p className="note mt-4">
        <Icon name="info" size={16} />
        <span>
          T3 and T6 are the weak spots. T3 was held out of training on purpose, to test fraud the model has never seen. T6 only shows on a
          scale, which is why the depot weighs those parcels.
        </span>
      </p>
    </Section>
  )
}

/** Depot weighing dial: how many parcels to weigh at the first scan, with each level's measured result. */
function DepotDial() {
  const [dial, setDial] = useState<FirstScanDial | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    api.firstScanDial().then(setDial).catch((e) => setErr(e instanceof Error ? e.message : String(e)))
  }, [])
  const choose = async (level: string) => {
    setBusy(true)
    setErr(null)
    try {
      setDial(await api.setFirstScanDial(level))
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }
  if (!dial || dial.levels.length === 0) return null
  return (
    <Section title="Depot weighing" icon="scale" aside="weight fraud only shows on a scale" testId="depot-dial">
      <p className="mb-4 max-w-[86ch] text-[14px] text-ink-2">
        Parcels declared far lighter than an account usually ships are weighed at the first depot scan.
      </p>
      <div className="dial-row" role="radiogroup" aria-label="How many parcels to weigh">
        {dial.levels.map((lv) => {
          const on = dial.current === lv.name
          return (
            <button
              key={lv.name}
              type="button"
              role="radio"
              aria-checked={on}
              disabled={busy}
              className={`dial-card${on ? ' is-on' : ''}`}
              onClick={() => choose(lv.name)}
              data-testid={`dial-${lv.name}`}
              data-current={on}
            >
              <span className="dial-card-name">
                {lv.name === 'standard' ? 'Standard' : `Target ${lv.name}`}
                {on && <span className="dial-card-in">In use</span>}
              </span>
              <span className="dial-card-big tnum">{Math.round(lv.t6_caught * 100)}%</span>
              <span className="text-[13px] text-muted">of weight fraud caught</span>
              <span className="dial-card-meter" aria-hidden="true">
                <span style={{ width: `${Math.min(lv.honest_weighed * 100 * 4, 100)}%` }} />
              </span>
              <span className="tnum text-[13px] text-ink-2">{(lv.honest_weighed * 100).toFixed(1)}% of honest parcels weighed</span>
            </button>
          )
        })}
      </div>
      {err && (
        <p className="mt-2 text-[14px] text-danger" role="alert">
          {err}
        </p>
      )}
      <p className="mt-3 text-[13px] text-muted" data-testid="dial-current">
        In use: {dial.current}. Changes apply to new bookings and are written to the audit log.
      </p>
    </Section>
  )
}

type Source = 'all' | 'app' | 'stream'
const SOURCES: [Source, string][] = [
  ['all', 'All bookings'],
  ['app', 'Scored in the app'],
  ['stream', 'Live stream'],
]
const REFRESH_MS = 3000

/** Dashboard metrics that refresh every few seconds, so new decisions (and a running stream) show up. */
function useDashboard(source: Source) {
  const [m, setM] = useState<DashboardMetrics | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [updated, setUpdated] = useState<Date | null>(null)
  const [tick, setTick] = useState(0)
  useEffect(() => {
    let alive = true
    let timer: ReturnType<typeof setTimeout> | undefined
    const load = async () => {
      try {
        const d = await api.dashboard(source)
        if (!alive) return
        setM(d)
        setError(null)
        setUpdated(new Date())
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e))
      }
      if (alive) timer = setTimeout(load, REFRESH_MS)
    }
    load()
    return () => {
      alive = false
      if (timer) clearTimeout(timer)
    }
  }, [source, tick])
  return { m, error, updated, reload: () => setTick((t) => t + 1) }
}

export default function DashboardView() {
  const [source, setSource] = useState<Source>('all')
  const { m, error, updated, reload } = useDashboard(source)
  const loading = !m && !error
  return (
    <div>
      <PageTitle title="Dashboard">
        <span className="chip-btn">
          <Icon name="calendar" size={16} />
          {m && m.window.from ? `${m.window.from} – ${m.window.to}` : 'No decisions yet'}
        </span>
        <div className="seg" role="radiogroup" aria-label="Which bookings">
          <SlideInd />
          {SOURCES.map(([k, label]) => (
            <button key={k} type="button" role="radio" aria-checked={source === k} onClick={() => setSource(k)} data-testid={`dashboard-source-${k}`}>
              {label}
            </button>
          ))}
        </div>
      </PageTitle>
      <Page>
        <div className="offline-row">
          <Kpi name="offline-prauc" label="Main score · PR-AUC" value={<span className="tnum">0.74</span>} sub="Rules alone: 0.17 — about 4× better" />
          <Kpi name="offline-campaign" label="Campaign recall" value={<span className="tnum">87%</span>} sub="of coordinated fraud waves caught" />
          <Kpi name="offline-precision" label="Precision" value={<span className="tnum">55%</span>} sub="Recall 45% at the chosen threshold" />
          <Kpi name="offline-fpr" label="Honest shippers stopped" value={<span className="tnum">0.5%</span>} sub="False-positive rate · limit 1%" />
        </div>
        <p className="-mt-1.5 text-[12.5px] text-muted">Offline test · real data columns only · 10 seeds</p>

        {error && <ErrorBox message={error} onRetry={reload} />}
        {loading && <Loading what="metrics" />}

        <div className="dash-grid">
          <ByFraudType />
          {m && (
            <Section title="What the gate did" icon="box">
              <ActionMix m={m} />
            </Section>
          )}
        </div>

        {m && (
          <>
            <div className="sect-h">
              <h2>Decisions in this console</h2>
              <span className="text-[13px] text-muted" data-testid="dashboard-updated" role="status">
                {updated ? `Updated ${updated.toLocaleTimeString()}` : 'Loading...'}
              </span>
            </div>
            <div className="live-kpis">
              <Kpi name="bookings" label="Bookings scored" value={<span className="tnum">{fmtInt(m.totals.bookings)}</span>} />
              <Kpi name="held" label="Shipments held" value={<span className="tnum">{fmtInt(m.held_shipments)}</span>} sub="no label until verified" />
              <Kpi
                name="net-prevented"
                label="Net loss prevented"
                value={<span className="tnum">{fmtBRL(m.net_prevented_brl, true)}</span>}
                sub={
                  <span className="tnum">
                    {fmtBRL(m.revenue_loss_prevented_brl, true)} stopped − {fmtBRL(m.friction_cost_brl, true)} friction
                  </span>
                }
              />
              <Kpi name="fpr-legit" label="Honest flagged" value={<span className="tnum">{fmtPct(m.fpr_legit, 1)}</span>} sub="legitimate bookings stopped" />
              <Kpi name="fpr-hard-negative" label="Hard cases flagged" value={<span className="tnum">{fmtPct(m.fpr_hard_negative, 1)}</span>} sub="honest but unusual, like a first new state" />
              <Kpi name="latency" label="Decision time" value={<span className="tnum">{fmtMs(m.latency.p50_ms)}</span>} sub={<span className="tnum">p50 · p99 {fmtMs(m.latency.p99_ms)}</span>} />
            </div>

            <Section title="Fraud stopped and held, per day" icon="chart">
              <TrendChart m={m} />
            </Section>

            <DepotDial />

            <p className="note" data-testid="dashboard-assumptions" role="note">
              <Icon name="info" size={16} />
              <span>
                <b>Assumptions.</b> Money figures use cost matrix <code className="text-[12.5px]">{m.assumptions.cost_matrix_version}</code>:{' '}
                {m.assumptions.note}. Friction is the estimated cost to legitimate shippers of checks, delays and holds, shown separately from
                the loss prevented.
              </span>
            </p>
          </>
        )}
      </Page>
    </div>
  )
}
