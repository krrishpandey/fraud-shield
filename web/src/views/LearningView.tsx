import { useEffect, useState, type ReactNode } from 'react'
import { api } from '../api/client'
import type { LearningMetrics, ModelVersion, RetrainResponse, SimulateFeedbackResponse } from '../api/types'
import { ErrorBox, Loading, PageTitle, Section } from '../components/common'
import { Icon } from '../components/Icon'
import { HandoverBanner, ModelInUseLine } from '../components/ModelHandover'
import { fmtBRL, fmtInt, fmtPct, shortHash } from '../lib/format'
import { useStatus } from '../lib/status'
import { useAsync } from '../lib/useAsync'

type MetricKey = keyof LearningMetrics
const METRICS: { key: MetricKey; label: string; higherBetter: boolean; fmt: (v: number) => string; eps: number }[] = [
  { key: 'pr_auc', label: 'PR-AUC', higherBetter: true, fmt: (v) => v.toFixed(3), eps: 0.0005 },
  { key: 'ece', label: 'Calibration error (ECE)', higherBetter: false, fmt: (v) => v.toFixed(3), eps: 0.0005 },
  { key: 'cost_per_1k_brl', label: 'Cost per 1,000 bookings (R$)', higherBetter: false, fmt: (v) => fmtBRL(v, true), eps: 0.05 },
  { key: 'fpr_hard_negative', label: 'Honest-but-unusual bookings stopped', higherBetter: false, fmt: (v) => fmtPct(v, 2), eps: 0.0005 },
  { key: 'recall_new_pattern', label: 'Unseen fraud caught', higherBetter: true, fmt: (v) => fmtPct(v, 0), eps: 0.0005 },
]

function SimulatedChip() {
  return <span className="sim-chip">Simulated</span>
}

function sourceLabel(src: string): ReactNode {
  if (src === 'analyst') return 'Analyst confirmations'
  if (src === 'simulated_analyst')
    return (
      <span className="flex items-center gap-2">
        Simulated analyst labels <SimulatedChip />
      </span>
    )
  return src.replace(/_/g, ' ')
}

function Delta({ cur, cand, higherBetter, eps }: { cur: number; cand: number; higherBetter: boolean; eps: number }) {
  const d = cand - cur
  if (Math.abs(d) <= eps)
    return (
      <span className="text-ink-2" data-direction="same">
        = Same
      </span>
    )
  const better = higherBetter ? d > 0 : d < 0
  return (
    <span data-direction={better ? 'better' : 'worse'} className="font-semibold" style={{ color: better ? 'var(--ok)' : 'var(--danger)' }}>
      <span aria-hidden="true">{better ? '▲' : '▼'}</span> {better ? 'Better' : 'Worse'}
    </span>
  )
}

/** One before → after box, coloured by whether the change helps. */
function Shift({ label, from, to, better, note }: { label: string; from: string; to: string; better: boolean; note: string }) {
  return (
    <div className="shift">
      <div className="shift-l">{label}</div>
      <div className="shift-row">
        <span className="shift-from tnum">{from}</span>
        <span className="text-muted">→</span>
        <span className="shift-to tnum" style={{ color: better ? 'var(--green)' : 'var(--act-block)' }}>
          {to}
        </span>
        <span className="shift-note" style={{ color: better ? 'var(--green)' : 'var(--act-block)' }}>
          {note}
        </span>
      </div>
    </div>
  )
}

function RetrainResult({ r }: { r: RetrainResponse }) {
  const deployed = !!r.deployed_version && r.deployed_version === r.candidate.version
  const passed = r.gate.checks.filter((c) => c.passed).length
  const t3Better = r.candidate.recall_new_pattern >= r.current.recall_new_pattern
  const fprBetter = r.candidate.fpr_hard_negative <= r.current.fpr_hard_negative
  return (
    <div className="flex flex-col gap-[18px]" data-testid="learning-result" data-run-id={r.run_id}>
      <section className="panel run-card">
        <div className="run-top">
          <div className="min-w-0 flex-1">
            <h2 className="mono run-v">{r.candidate.version}</h2>
            <p className="text-muted">
              Run {r.run_id} · trained with {fmtInt(r.n_new_labels)} new labels ({fmtInt(r.n_fraud)} fraud, {fmtInt(r.n_legit)} legitimate)
            </p>
          </div>
          <span
            className={`run-status ${deployed ? 'is-ok' : 'is-bad'}`}
            data-testid="learning-deployed-status"
            data-deployed={deployed}
            role="status"
          >
            <Icon name={deployed ? 'check' : 'x'} size={16} />
            {deployed ? `Deployed: ${r.candidate.version} is now the active model` : `Refused by the safety gate: ${r.current.version} stays active`}
          </span>
        </div>
        <div className="run-shifts">
          <Shift
            label="Unseen fraud caught"
            from={fmtPct(r.current.recall_new_pattern, 0)}
            to={fmtPct(r.candidate.recall_new_pattern, 0)}
            better={t3Better}
            note={t3Better ? 'Better' : 'Worse'}
          />
          <Shift
            label="Honest-but-unusual bookings stopped"
            from={fmtPct(r.current.fpr_hard_negative, 2)}
            to={fmtPct(r.candidate.fpr_hard_negative, 2)}
            better={fprBetter}
            note={fprBetter ? 'Better' : 'Worse'}
          />
        </div>
      </section>

      <Section title="Gate checks" icon="shield" aside={`${passed} of ${r.gate.checks.length} passed · one failure is enough to refuse`}>
        <ul className="gate-list">
          {r.gate.checks.map((c) => (
            <li key={c.name} data-testid="learning-gate-check" data-name={c.name} data-passed={c.passed}>
              <Icon name={c.passed ? 'check' : 'x'} size={17} style={{ color: c.passed ? 'var(--green)' : 'var(--danger)' }} />
              <span className="min-w-0 flex-1">
                <span className="font-semibold">{c.name.replace(/_/g, ' ')}</span>
              </span>
              <span className={`mono text-[13px] ${c.passed ? 'text-muted' : 'text-danger'}`}>{c.detail}</span>
            </li>
          ))}
        </ul>
      </Section>

      <Section title="Active vs candidate" icon="chart" aside={`same ${fmtInt(r.eval_set.n)} bookings`}>
        <table className="tbl" data-testid="learning-result-table">
          <caption className="mb-2 text-left text-[13px] text-muted">Both models scored on the same bookings: {r.eval_set.description}.</caption>
          <thead>
            <tr>
              <th scope="col">Metric</th>
              <th scope="col" className="r">
                Active <span className="mono block text-[12px] font-normal">{r.current.version}</span>
              </th>
              <th scope="col" className="r">
                Candidate <span className="mono block text-[12px] font-normal">{r.candidate.version}</span>
              </th>
              <th scope="col">Change</th>
            </tr>
          </thead>
          <tbody>
            {METRICS.map((m) => (
              <tr key={m.key} data-metric={m.key}>
                <th scope="row" className="text-left font-normal">
                  {m.label}
                  <span className="ml-1 text-[12px] text-muted">({m.higherBetter ? 'higher is better' : 'lower is better'})</span>
                </th>
                <td className="tnum r">{m.fmt(r.current[m.key])}</td>
                <td className="tnum r font-semibold">{m.fmt(r.candidate[m.key])}</td>
                <td className="whitespace-nowrap text-[13.5px]">
                  <Delta cur={r.current[m.key]} cand={r.candidate[m.key]} higherBetter={m.higherBetter} eps={m.eps} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-2 text-[12.5px] text-muted">
          Money figures depend on the cost assumptions. Audit hash <code>{shortHash(r.audit_hash, 16)}</code>
        </p>
      </Section>
    </div>
  )
}

function versionState(v: ModelVersion, active: string): { word: string; tone: string } {
  if (v.version === active) return { word: 'Active', tone: 'var(--green)' }
  if (v.ever_deployed === false || v.gate_passed === false) return { word: 'Refused by gate', tone: 'var(--danger)' }
  return { word: 'Retired', tone: 'var(--muted)' }
}

export default function LearningView() {
  const status = useAsync(() => api.learningStatus(), [])
  const { refresh } = useStatus()
  const s = status.data
  const [n, setN] = useState(200)
  const [simBusy, setSimBusy] = useState(false)
  const [simRes, setSimRes] = useState<SimulateFeedbackResponse | null>(null)
  const [retrainBusy, setRetrainBusy] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [result, setResult] = useState<RetrainResponse | null>(null)
  const [rollbackBusy, setRollbackBusy] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)

  // A retrain keeps running on the server when this page is left; on return the status says so.
  const serverRun = s?.retrain_running ?? null
  const running = retrainBusy || serverRun != null
  const shown = result ?? s?.last_result ?? null
  const reloadStatus = status.reload
  useEffect(() => {
    if (!serverRun || retrainBusy) return // our own request is still open: its response brings the result
    const t = setInterval(reloadStatus, 2000)
    return () => clearInterval(t)
  }, [serverRun, retrainBusy, reloadStatus])
  useEffect(() => {
    if (!running) return
    const start = serverRun?.started_at ? Date.parse(serverRun.started_at) : Date.now()
    const tick = () => setElapsed(Math.max(0, Math.floor((Date.now() - start) / 1000)))
    tick()
    const t = setInterval(tick, 500)
    return () => clearInterval(t)
  }, [running, serverRun?.started_at])

  const run = async (fn: () => Promise<void>) => {
    setErr(null)
    try {
      await fn()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  const simulate = () =>
    run(async () => {
      setSimBusy(true)
      try {
        setSimRes(await api.simulateFeedback(n))
        status.reload()
      } finally {
        setSimBusy(false)
      }
    })

  const retrain = () =>
    run(async () => {
      setElapsed(0)
      setRetrainBusy(true)
      try {
        setResult(await api.retrain(20))
        status.reload()
        refresh()
      } catch (e) {
        // already running (started before this page was opened): follow that run instead of failing
        if (e instanceof Error && /already running/i.test(e.message)) status.reload()
        else throw e
      } finally {
        setRetrainBusy(false)
      }
    })

  const rollback = (version: string) =>
    run(async () => {
      setRollbackBusy(version)
      try {
        await api.rollback(version)
        status.reload()
      } finally {
        setRollbackBusy(null)
      }
    })

  const sources = s ? Object.entries(s.label_sources) : []
  const versions = s ? [...s.versions].reverse() : []

  return (
    <div>
      <PageTitle title="Learning" />
      {status.error && (
        <div className="page">
          <ErrorBox message={status.error} onRetry={status.reload} />
        </div>
      )}
      {status.loading && !s && (
        <div className="page">
          <Loading what="learning status" />
        </div>
      )}
      {s && (
        <div className="split">
          <aside className="split-list" aria-label="Model versions">
            <h2 className="split-h">Model versions</h2>
            <ul className="vlist" data-testid="learning-version-table">
              {versions.map((v) => {
                const active = v.version === s.active_version
                const st = versionState(v, s.active_version)
                return (
                  <li key={v.version} className={`vcard${active ? ' is-active' : ''}`} data-testid="learning-version-row" data-version={v.version} data-active={active}>
                    <div className="vcard-top">
                      <span className="mono vcard-name">{v.version}</span>
                      {!active && v.ever_deployed !== false && v.gate_passed !== false && (
                        <button
                          type="button"
                          className="vcard-btn"
                          data-testid="learning-rollback-button"
                          data-version={v.version}
                          disabled={rollbackBusy !== null || running}
                          onClick={() => rollback(v.version)}
                          title={`Make ${v.version} the active model again`}
                        >
                          <Icon name="history" size={14} />
                          {rollbackBusy === v.version ? 'Rolling back...' : 'Roll back'}
                        </button>
                      )}
                    </div>
                    <div className="vcard-meta">
                      <span className="dot" style={{ background: st.tone }} aria-hidden="true" />
                      {st.word === 'Refused by gate' ? <span data-testid="learning-version-rejected">Rejected by gate</span> : st.word}
                      <span>· {fmtInt(v.n_train)} rows</span>
                      <span>· {v.metrics ? v.metrics.pr_auc.toFixed(2) : 'PR-AUC n/a'}</span>
                    </div>
                    <div className="vcard-sub">
                      {v.created_at ? v.created_at.replace('T', ' ').slice(0, 16) : 'pipeline-trained'} · {fmtInt(v.n_feedback_labels)} feedback labels
                      {v.parent ? ` · from ${v.parent}` : ''}
                    </div>
                  </li>
                )
              })}
            </ul>
            <div className="panel laya-card">
              <div className="flex items-center gap-2.5">
                <Icon name="cpu" size={18} style={{ color: 'var(--indigo)' }} />
                <b>Laya</b>
                <span className="ml-auto pill-mute">Fine-tuned offline</span>
              </div>
              <p className="mt-2 text-[13.5px] text-muted" data-testid="learning-laya-note">
                Laya's weights are not retrained here. They are fine-tuned offline, and the feedback labels are exported for that run
                {s.laya_export ? (
                  <>
                    {' '}
                    to <code className="break-all text-[12px]">{s.laya_export.path}</code> ({fmtInt(s.laya_export.rows)} rows so far).
                  </>
                ) : (
                  '. No export has been written yet.'
                )}
              </p>
            </div>
          </aside>

          <div className="split-main learn-main">
            {err && <ErrorBox message={err} testId="learning-error" />}
            {s.last_handover && <HandoverBanner h={s.last_handover} />}

            <div className="learn-top">
              <section className="panel learn-active">
                <div className="min-w-0">
                  <div className="kpi-label">Active model</div>
                  <div className="mono learn-active-v" data-testid="learning-active-version">
                    {s.active_version}
                  </div>
                  <div className="text-[13px] text-muted">
                    Calibration <span className="mono">{s.calibration_version}</span>
                  </div>
                  {s.model_in_use && (
                    <div className="mt-2 border-t border-rule pt-2">
                      <ModelInUseLine m={s.model_in_use} testId="learning-model-in-use" />
                    </div>
                  )}
                </div>
              </section>
              <section className="panel learn-labels">
                <div className="kpi-label">New labels since the last retrain</div>
                <div className="kpi-value tnum" data-testid="learning-label-count" data-count={s.labels_since_last_retrain}>
                  {fmtInt(s.labels_since_last_retrain)}
                </div>
                <ul className="mt-2 flex flex-col gap-1 text-[13.5px] text-ink-2">
                  {sources.map(([src, count]) => (
                    <li key={src} className="flex items-center gap-3" data-testid={`learning-source-${src}`} data-count={count}>
                      <span className="tnum w-12 text-right font-semibold text-ink">{fmtInt(count)}</span>
                      {sourceLabel(src)}
                    </li>
                  ))}
                  {sources.length === 0 && <li className="text-muted">None yet. Confirm or clear cases in the review queue.</li>}
                </ul>
              </section>
            </div>

            <Section title="Simulated analyst feedback" icon="user" aside={<SimulatedChip />}>
              <p className="text-[14px] text-ink-2">
                Labels past bookings from the known answers, as if an analyst had reviewed them.
              </p>
              <div className="mt-3 flex flex-wrap items-end gap-2.5">
                <div>
                  <label htmlFor="learning-n" className="fbox-k">
                    Number of labels
                  </label>
                  <input
                    id="learning-n"
                    data-testid="learning-simulate-n"
                    className="field tnum w-32"
                    type="number"
                    min={1}
                    max={5000}
                    value={n}
                    onChange={(e) => setN(Math.max(1, Math.floor(Number(e.target.value) || 1)))}
                  />
                </div>
                <button type="button" className="btn" data-testid="learning-simulate-button" disabled={simBusy || running} onClick={simulate}>
                  <Icon name="plus" size={16} />
                  {simBusy ? 'Adding labels...' : 'Simulate analyst feedback (demo)'}
                </button>
              </div>
              {simRes && (
                <p className="mt-3 text-[14px]" role="status" data-testid="learning-simulate-result">
                  Added {fmtInt(simRes.added)} simulated labels: {fmtInt(simRes.fraud)} fraud, {fmtInt(simRes.legit)} legitimate.
                </p>
              )}
            </Section>

            {running && (
              <section className="panel p-6" data-testid="learning-retrain-progress" role="status" aria-live="polite">
                <div className="flex items-center gap-3">
                  <div>
                    <b>Training the candidate and testing both models</b>
                    <p className="text-[13.5px] text-muted">
                      <span className="tnum">{elapsed} s</span> · usually under a minute
                    </p>
                  </div>
                </div>
                <div className="prob-track mt-4 overflow-hidden" role="progressbar" aria-label="Retraining" aria-valuemin={0} aria-valuemax={60} aria-valuenow={Math.min(elapsed, 60)}>
                  <div className="absolute inset-y-0 left-0 rounded-[5px]" style={{ width: `${Math.min((elapsed / 60) * 100, 95)}%`, background: 'var(--indigo)' }} />
                </div>
              </section>
            )}
            {shown && !running && <RetrainResult r={shown} />}
            {!shown && !running && (
              <div className="panel learn-empty">
                <Icon name="shieldCheck" size={22} />
                <p>
                  No retrain yet in this session.
                </p>
              </div>
            )}

            <div className="learn-dock">
              <span className="text-[14px] text-muted">
                <Icon name="lock" size={15} style={{ display: 'inline', verticalAlign: '-2px' }} /> Active: <span className="mono">{s.active_version}</span>
              </span>
              <span className="text-[13px] text-faint">Needs at least 20 new labels</span>
              <button type="button" className="btn btn-primary ml-auto" data-testid="learning-retrain-button" disabled={running || simBusy} onClick={retrain}>
                <Icon name="refresh" size={16} />
                {running ? 'Retraining...' : 'Retrain with new labels'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
