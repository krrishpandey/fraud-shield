import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { MonitorEstimate } from '../api/types'
import { fmtInt } from '../lib/format'

/*
  Label-free monitor (GET /monitor/estimate): fraud labels arrive days to weeks after a booking, so the calibrated
  scores estimate how the stops are doing in the meantime (CBPE). The realized numbers beside it exist only because
  the stream replays a labelled dataset. The blind-spot line is the gap measured offline (artifacts/results_monitor.md).
*/

const pct = (v: number | null | undefined) => (v == null || Number.isNaN(v) ? 'n/a' : `${(v * 100).toFixed(1)}%`)
const cnt = (v: number | null | undefined) => (v == null || Number.isNaN(v) ? 'n/a' : v.toFixed(1))

export default function LabelFreeMonitorCard({ scored }: { scored: number }) {
  const [m, setM] = useState<MonitorEstimate | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let alive = true
    api
      .monitorEstimate()
      .then((r) => {
        if (!alive) return
        setM(r)
        setError(null)
      })
      .catch((e) => {
        if (alive) setError(e instanceof Error ? e.message : String(e))
      })
    return () => {
      alive = false
    }
  }, [scored])

  if (error && !m) {
    return (
      <section className="monitor-card" data-testid="monitor-card" data-state="error">
        <h2 className="live-h2">Label-free monitor</h2>
        <p className="live-note" role="status">The monitor is not available: {error}</p>
      </section>
    )
  }
  if (!m) return null
  const e = m.estimated
  const r = m.realized
  const bs = m.blind_spot
  return (
    <section className="monitor-card" aria-labelledby="monitor-h" data-testid="monitor-card" data-n={m.n}>
      <h2 id="monitor-h" className="live-h2">
        Label-free monitor
      </h2>
      <p className="live-note">
        Fraud labels arrive days to weeks after a booking. Until then the calibrated scores of the {fmtInt(m.n)} decisions so
        far estimate how the stops are doing.
      </p>
      <dl className="live-outcomes">
        <div data-testid="monitor-est-precision" data-value={e.precision_stopped ?? ''}>
          <dt>Estimated precision of stops</dt>
          <dd className="tnum">
            {e.n_stopped ? pct(e.precision_stopped) : 'n/a'} <span className="monitor-tag">(no labels needed)</span>
          </dd>
        </div>
        <div data-testid="monitor-est-missed" data-value={e.fraud_missed}>
          <dt>Estimated fraud missed</dt>
          <dd className="tnum">
            {cnt(e.fraud_missed)} <span className="monitor-tag">of {fmtInt(e.n_let_through)} let through</span>
          </dd>
        </div>
      </dl>
      <div className="monitor-truth" data-testid="monitor-realized">
        <p className="monitor-truth-label">{r.label[0].toUpperCase() + r.label.slice(1)}:</p>
        <p className="tnum">
          precision of stops {r.n_stopped ? pct(r.precision_stopped) : 'n/a'}, fraud missed {fmtInt(r.fraud_missed)}
          {r.missed_held_out > 0 && <> ({fmtInt(r.missed_held_out)} of them from types the model never learned)</>}.
        </p>
      </div>
      {bs && (
        <p className="monitor-blind" data-testid="monitor-blind-spot" data-gap={bs.missed_gap_per_period}>
          {bs.ui_note}
        </p>
      )}
      <p className="live-note" data-testid="monitor-drift">
        Score drift against the calibration window (PSI):{' '}
        {m.drift.psi == null ? `shown after ${fmtInt(m.drift.min_n)} decisions` : m.drift.psi.toFixed(3)}. It flags a shift
        in the scores, not new fraud the model scores low.
      </p>
    </section>
  )
}
