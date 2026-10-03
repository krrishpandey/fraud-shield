import { useMemo, useState, type ReactNode } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, type TooltipContentProps } from 'recharts'
import { api } from '../api/client'
import { ACTIONS, type DashboardMetrics } from '../api/types'
import { ErrorBox, Loading, PageTitle, Section } from '../components/common'
import { ACTION_META, TYPOLOGY_META, typologyName } from '../lib/domain'
import { fmtBRL, fmtInt, fmtMs, fmtPct } from '../lib/format'
import { useAsync } from '../lib/useAsync'
import { useCssVars } from '../lib/useCssVars'

function Kpi({ name, label, value, sub }: { name: string; label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="panel px-3.5 py-3" data-testid={`dashboard-kpi-${name}`}>
      <div className="text-[0.78rem] text-ink-2">{label}</div>
      <div className="mt-0.5 text-[1.5rem] leading-tight font-bold">{value}</div>
      {sub && <div className="mt-0.5 text-[0.75rem] text-muted">{sub}</div>}
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
      <div className="panel px-3 py-2 text-[0.78rem] shadow-sm">
        <div className="font-semibold">{label}</div>
        <div className="tnum">Fraud stopped: {row.fraud_stopped}</div>
        <div className="tnum">Held: {row.held}</div>
        {mode === 'typology' &&
          typologies
            .filter((t) => Number(row[t]) > 0)
            .map((t) => (
              <div key={t} className="tnum flex items-center gap-1.5">
                <span aria-hidden="true" className="inline-block h-2 w-2 rounded-[2px]" style={{ background: colorOf(t) }} />
                {typologyName(t)}: {row[t]}
              </div>
            ))}
      </div>
    )
  }

  return (
    <div data-testid="trend-chart" data-mode={mode}>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <div role="group" aria-label="Series" className="flex gap-1">
          {(
            [
              ['typology', 'Fraud stopped, by typology'],
              ['held', 'Held shipments'],
            ] as [Mode, string][]
          ).map(([k, l]) => (
            <button
              key={k}
              type="button"
              aria-pressed={mode === k}
              data-testid={`trend-mode-${k}`}
              onClick={() => setMode(k)}
              className={`rounded-sm border px-2.5 py-1 text-[0.78rem] font-semibold ${mode === k ? 'border-brand bg-brand text-brand-ink' : 'border-rule text-ink-2'}`}
            >
              {l}
            </button>
          ))}
        </div>
        <button type="button" className="btn ml-auto py-1 text-[0.78rem]" aria-pressed={asTable} onClick={() => setAsTable((v) => !v)} data-testid="trend-table-toggle">
          {asTable ? 'Show chart' : 'Show as table'}
        </button>
      </div>
      {mode === 'typology' && !asTable && (
        <ul className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-[0.75rem] text-ink-2" aria-label="Legend">
          {typologies.map((t) => (
            <li key={t} className="flex items-center gap-1.5">
              <span aria-hidden="true" className="inline-block h-2.5 w-2.5 rounded-[2px]" style={{ background: colorOf(t) }} />
              {typologyName(t)}
            </li>
          ))}
        </ul>
      )}
      {asTable ? (
        <div className="max-h-[260px] overflow-auto">
          <table className="w-full text-[0.78rem]" data-testid="trend-table">
            <thead className="sticky top-0 bg-surface">
              <tr className="border-b border-rule text-left text-muted">
                <th scope="col" className="py-1 pr-3">Date</th>
                <th scope="col" className="py-1 pr-3 text-right">Fraud stopped</th>
                <th scope="col" className="py-1 pr-3 text-right">Held</th>
                {typologies.map((t) => (
                  <th key={t} scope="col" className="py-1 pr-3 text-right" title={typologyName(t)}>
                    {t}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.date} className="tnum border-b border-rule">
                  <td className="py-0.5 pr-3">{r.date}</td>
                  <td className="py-0.5 pr-3 text-right">{r.fraud_stopped}</td>
                  <td className="py-0.5 pr-3 text-right">{r.held}</td>
                  {typologies.map((t) => (
                    <td key={t} className="py-0.5 pr-3 text-right">
                      {(r as Record<string, number | string>)[t]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="h-[240px]" role="img" aria-label={mode === 'typology' ? 'Fraud bookings stopped per day, stacked by typology' : 'Held shipments per day'}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows} margin={{ top: 4, right: 8, bottom: 0, left: -12 }} barCategoryGap="18%">
              <CartesianGrid vertical={false} stroke={c['--grid']} />
              <XAxis dataKey="date" tickFormatter={(d: string) => d.slice(5)} tick={{ fill: c['--muted'], fontSize: 11 }} tickLine={false} axisLine={{ stroke: c['--axis'] }} minTickGap={12} />
              <YAxis allowDecimals={false} tick={{ fill: c['--muted'], fontSize: 11 }} tickLine={false} axisLine={false} width={44} label={{ value: 'bookings', angle: -90, position: 'insideLeft', offset: 18, fill: c['--muted'], fontSize: 11 }} />
              <Tooltip content={TooltipBody} cursor={{ fill: c['--grid'], opacity: 0.5 }} isAnimationActive={false} />
              {mode === 'typology' ? (
                typologies.map((t, i) => (
                  <Bar
                    key={t}
                    dataKey={t}
                    name={typologyName(t)}
                    stackId="t"
                    fill={colorOf(t)}
                    stroke={c['--surface']}
                    strokeWidth={1}
                    radius={i === typologies.length - 1 ? [3, 3, 0, 0] : 0}
                    isAnimationActive={false}
                  />
                ))
              ) : (
                <Bar dataKey="held" name="Held shipments" fill={c['--series-1']} radius={[3, 3, 0, 0]} isAnimationActive={false} />
              )}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      <p className="mt-1 text-[0.75rem] text-muted">Bookings per day, {m.window.from} to {m.window.to}. Typology comes from the evaluation labels.</p>
    </div>
  )
}

function ActionMix({ m }: { m: DashboardMetrics }) {
  const by = m.totals.by_action
  const total = m.totals.bookings || 1
  const others = ACTIONS.filter((a) => a !== 'allow')
  const otherTotal = others.reduce((s, a) => s + (by[a] ?? 0), 0)
  const max = Math.max(...others.map((a) => by[a] ?? 0), 1)
  return (
    <div data-testid="action-mix">
      <p className="mb-2 text-[0.85rem]">
        <strong className="tnum">{fmtInt(by.allow ?? 0)}</strong> of {fmtInt(m.totals.bookings)} bookings ({fmtPct((by.allow ?? 0) / total, 1)}) were allowed
        normally. The other <strong className="tnum">{fmtInt(otherTotal)}</strong>:
      </p>
      <ul className="flex flex-col gap-1.5">
        {others.map((a) => {
          const v = by[a] ?? 0
          return (
            <li key={a} className="grid grid-cols-[9.5rem_1fr_6.5rem] items-center gap-2 text-[0.8rem]" data-testid={`action-mix-${a}`} data-count={v}>
              <span className="flex items-center gap-1.5">
                <span aria-hidden="true" style={{ color: `var(--act-${a})` }}>
                  {ACTION_META[a].glyph}
                </span>
                {ACTION_META[a].short}
              </span>
              <span className="h-2.5">
                <span className="block h-full rounded-r-[2px]" style={{ width: `${Math.max((v / max) * 100, v ? 1 : 0)}%`, background: `var(--act-${a})` }} />
              </span>
              <span className="tnum text-right">
                {fmtInt(v)} <span className="text-muted">({fmtPct(v / total, 2)})</span>
              </span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

export default function DashboardView() {
  const { data: m, error, loading, reload } = useAsync(() => api.dashboard(), [])
  return (
    <div>
      <PageTitle title="Dashboard" sub={m ? `Window ${m.window.from} to ${m.window.to}.` : undefined} />
      {error && <ErrorBox message={error} onRetry={reload} />}
      {loading && !m && <Loading what="metrics" />}
      {m && (
        <div className="flex flex-col gap-4">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
            <Kpi name="bookings" label="Bookings scored" value={<span className="tnum">{fmtInt(m.totals.bookings)}</span>} />
            <Kpi name="held" label="Shipments held" value={<span className="tnum">{fmtInt(m.held_shipments)}</span>} sub="no label until verified" />
            <Kpi
              name="net-prevented"
              label="Net revenue loss prevented"
              value={<span className="tnum">{fmtBRL(m.net_prevented_brl, true)}</span>}
              sub={
                <span className="tnum">
                  {fmtBRL(m.revenue_loss_prevented_brl, true)} stopped, minus {fmtBRL(m.friction_cost_brl, true)} friction
                </span>
              }
            />
            <Kpi name="fpr-legit" label="False positives, legit shippers" value={<span className="tnum">{fmtPct(m.fpr_legit, 1)}</span>} sub="legitimate bookings flagged" />
            <Kpi name="fpr-hard-negative" label="False positives, hard negatives" value={<span className="tnum">{fmtPct(m.fpr_hard_negative, 1)}</span>} sub="legit but unusual, like a first new state" />
            <Kpi
              name="latency"
              label="Decision latency"
              value={<span className="tnum">{fmtMs(m.latency.p50_ms)}</span>}
              sub={<span className="tnum">p50, with p99 {fmtMs(m.latency.p99_ms)}</span>}
            />
          </div>

          <div className="grid gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
            <Section title="Fraud stopped and held, per day">
              <TrendChart m={m} />
            </Section>
            <Section title="Action mix">
              <ActionMix m={m} />
            </Section>
          </div>

          <div className="panel border-dashed p-3 text-[0.8rem] text-ink-2" data-testid="dashboard-assumptions" role="note">
            <strong>Assumptions.</strong> Money figures use cost matrix <code className="font-mono">{m.assumptions.cost_matrix_version}</code>: {m.assumptions.note}.
            Friction is the estimated cost to legitimate shippers of checks, delays and holds, and is shown separately from the loss
            prevented.
          </div>
        </div>
      )}
    </div>
  )
}
