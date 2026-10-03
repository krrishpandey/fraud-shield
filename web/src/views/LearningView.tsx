import { useEffect, useState, type ReactNode } from 'react'
import { api } from '../api/client'
import type { LearningMetrics, RetrainResponse, SimulateFeedbackResponse } from '../api/types'
import { ErrorBox, Loading, PageTitle, Section } from '../components/common'
import { fmtBRL, fmtInt, fmtPct, shortHash } from '../lib/format'
import { useAsync } from '../lib/useAsync'

type MetricKey = keyof LearningMetrics
const METRICS: { key: MetricKey; label: string; higherBetter: boolean; fmt: (v: number) => string; eps: number }[] = [
  { key: 'pr_auc', label: 'PR-AUC', higherBetter: true, fmt: (v) => v.toFixed(3), eps: 0.0005 },
  { key: 'ece', label: 'Calibration error (ECE)', higherBetter: false, fmt: (v) => v.toFixed(3), eps: 0.0005 },
  { key: 'cost_per_1k_brl', label: 'Cost per 1,000 bookings (R$)', higherBetter: false, fmt: (v) => fmtBRL(v, true), eps: 0.05 },
  { key: 'fpr_hard_negative', label: 'False positives on hard negatives', higherBetter: false, fmt: (v) => fmtPct(v, 1), eps: 0.0005 },
  { key: 'recall_new_pattern', label: 'Recall on the new pattern', higherBetter: true, fmt: (v) => fmtPct(v, 0), eps: 0.0005 },
]

function SimulatedChip() {
  return <span className="rounded-sm bg-tape px-1.5 py-0.5 text-[0.7rem] font-bold text-[#14212e]">Simulated</span>
}

function sourceLabel(src: string): ReactNode {
  if (src === 'analyst') return 'Analyst confirmations'
  if (src === 'simulated_analyst')
    return (
      <span className="flex items-center gap-1.5">
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

function RetrainResult({ r }: { r: RetrainResponse }) {
  const deployed = !!r.deployed_version && r.deployed_version === r.candidate.version
  return (
    <div className="mt-3 flex flex-col gap-3" data-testid="learning-result" data-run-id={r.run_id}>
      <div
        data-testid="learning-deployed-status"
        data-deployed={deployed}
        role="status"
        className="rounded-sm border p-2.5"
        style={{ borderColor: deployed ? 'var(--ok)' : 'var(--danger)' }}
      >
        <p className="font-bold" style={{ color: deployed ? 'var(--ok)' : 'var(--danger)' }}>
          <span aria-hidden="true">{deployed ? '✓ ' : '✕ '}</span>
          {deployed
            ? `Deployed: ${r.candidate.version} is now the active model`
            : `Rejected: the gate failed, ${r.current.version} stays active`}
        </p>
        <p className="mt-0.5 text-[0.78rem] text-ink-2">
          Run {r.run_id}, trained with {fmtInt(r.n_new_labels)} new labels ({fmtInt(r.n_fraud)} fraud, {fmtInt(r.n_legit)} legitimate).
          Audit hash <code className="font-mono">{shortHash(r.audit_hash, 16)}</code>
        </p>
      </div>

      <div>
        <table className="w-full text-[0.85rem]" data-testid="learning-result-table">
          <caption className="mb-1 text-left text-[0.75rem] text-muted">
            Both models scored on the same {fmtInt(r.eval_set.n)} bookings: {r.eval_set.description}.
          </caption>
          <thead>
            <tr className="border-b border-rule text-left text-[0.75rem] text-muted">
              <th scope="col" className="py-1 font-semibold">Metric</th>
              <th scope="col" className="py-1 text-right font-semibold">
                Current <span className="block font-mono font-normal whitespace-nowrap">{r.current.version}</span>
              </th>
              <th scope="col" className="py-1 text-right font-semibold">
                Candidate <span className="block font-mono font-normal whitespace-nowrap">{r.candidate.version}</span>
              </th>
              <th scope="col" className="py-1 pl-3 font-semibold">Change</th>
            </tr>
          </thead>
          <tbody>
            {METRICS.map((m) => (
              <tr key={m.key} className="border-b border-rule last:border-0" data-metric={m.key}>
                <th scope="row" className="py-1.5 text-left font-normal">
                  {m.label}
                  <span className="ml-1 text-[0.72rem] text-muted">({m.higherBetter ? 'higher is better' : 'lower is better'})</span>
                </th>
                <td className="tnum py-1.5 text-right">{m.fmt(r.current[m.key])}</td>
                <td className="tnum py-1.5 text-right font-semibold">{m.fmt(r.candidate[m.key])}</td>
                <td className="py-1.5 pl-3 text-[0.8rem] whitespace-nowrap">
                  <Delta cur={r.current[m.key]} cand={r.candidate[m.key]} higherBetter={m.higherBetter} eps={m.eps} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-1 text-[0.72rem] text-muted">Money figures depend on the cost assumptions.</p>
      </div>

      <div>
        <h3 className="text-[0.8rem] font-semibold text-ink-2">Deployment gate</h3>
        <ul className="mt-1 flex flex-col gap-1">
          {r.gate.checks.map((c) => (
            <li key={c.name} data-testid="learning-gate-check" data-name={c.name} data-passed={c.passed} className="flex gap-2 text-[0.82rem]">
              <span
                className="w-12 shrink-0 rounded-sm border px-1 text-center text-[0.72rem] font-bold"
                style={{ borderColor: c.passed ? 'var(--ok)' : 'var(--danger)', color: c.passed ? 'var(--ok)' : 'var(--danger)' }}
              >
                {c.passed ? 'Pass' : 'Fail'}
              </span>
              <span>
                <code className="font-mono text-[0.75rem]">{c.name}</code> <span className="text-ink-2">{c.detail}</span>
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}

export default function LearningView() {
  const status = useAsync(() => api.learningStatus(), [])
  const s = status.data
  const [n, setN] = useState(200)
  const [simBusy, setSimBusy] = useState(false)
  const [simRes, setSimRes] = useState<SimulateFeedbackResponse | null>(null)
  const [retrainBusy, setRetrainBusy] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [result, setResult] = useState<RetrainResponse | null>(null)
  const [rollbackBusy, setRollbackBusy] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    if (!retrainBusy) return
    const start = Date.now()
    const t = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 500)
    return () => clearInterval(t)
  }, [retrainBusy])

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
      <PageTitle
        title="Learning"
        sub="Analyst labels become training data. A retrain builds a candidate model, compares it with the active one on bookings neither has seen, and deploys it only if the gate passes."
      />
      {status.error && <ErrorBox message={status.error} onRetry={status.reload} />}
      {status.loading && !s && <Loading what="learning status" />}
      {err && (
        <div className="mb-3">
          <ErrorBox message={err} testId="learning-error" />
        </div>
      )}
      {s && (
        <div className="flex flex-col gap-4">
          <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
            <div className="panel px-3.5 py-3">
              <div className="text-[0.78rem] text-ink-2">Active model</div>
              <div className="mt-0.5 font-mono text-[1.15rem] font-medium" data-testid="learning-active-version">
                {s.active_version}
              </div>
              <div className="mt-0.5 text-[0.75rem] text-muted">
                Calibration <span className="font-mono">{s.calibration_version}</span>
              </div>
            </div>
            <div className="panel px-3.5 py-3">
              <div className="flex flex-wrap items-baseline gap-x-3">
                <span className="text-[0.78rem] text-ink-2">Labels collected since the last retrain</span>
                <span className="tnum text-[1.5rem] leading-tight font-bold" data-testid="learning-label-count" data-count={s.labels_since_last_retrain}>
                  {fmtInt(s.labels_since_last_retrain)}
                </span>
              </div>
              <ul className="mt-1 flex flex-col gap-0.5 text-[0.82rem]">
                {sources.map(([src, count]) => (
                  <li key={src} className="flex items-center gap-3" data-testid={`learning-source-${src}`} data-count={count}>
                    <span className="tnum w-12 text-right font-semibold">{fmtInt(count)}</span>
                    {sourceLabel(src)}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
            <div className="flex flex-col gap-4">
              <Section title="Simulated analyst feedback" aside={<SimulatedChip />}>
                <p className="text-[0.82rem] text-ink-2">
                  Demo only. Replays past bookings and labels them from the injected ground truth, as if an analyst had reviewed them.
                  These labels are flagged simulated everywhere they are stored.
                </p>
                <div className="mt-2.5 flex flex-wrap items-end gap-2">
                  <div>
                    <label htmlFor="learning-n" className="block text-[0.75rem] text-muted">
                      Number of labels
                    </label>
                    <input
                      id="learning-n"
                      data-testid="learning-simulate-n"
                      className="field tnum w-28"
                      type="number"
                      min={1}
                      max={5000}
                      value={n}
                      onChange={(e) => setN(Math.max(1, Math.floor(Number(e.target.value) || 1)))}
                    />
                  </div>
                  <button type="button" className="btn" data-testid="learning-simulate-button" disabled={simBusy || retrainBusy} onClick={simulate}>
                    {simBusy ? 'Adding labels...' : 'Simulate analyst feedback (demo)'}
                  </button>
                </div>
                {simRes && (
                  <p className="mt-2 text-[0.8rem]" role="status" data-testid="learning-simulate-result">
                    Added {fmtInt(simRes.added)} simulated labels: {fmtInt(simRes.fraud)} fraud, {fmtInt(simRes.legit)} legitimate.
                  </p>
                )}
              </Section>

              <Section title="Laya">
                <p className="text-[0.82rem] text-ink-2" data-testid="learning-laya-note">
                  Laya's weights are not retrained here. They are fine-tuned offline, and the feedback labels are exported for that run
                  {s.laya_export ? (
                    <>
                      {' '}
                      to <code className="font-mono text-[0.75rem] break-all">{s.laya_export.path}</code> ({fmtInt(s.laya_export.rows)} rows so far).
                    </>
                  ) : (
                    '. No export has been written yet.'
                  )}
                </p>
              </Section>
            </div>

            <Section title="Retrain">
              <div className="flex flex-wrap items-center gap-3">
                <button type="button" className="btn btn-primary" data-testid="learning-retrain-button" disabled={retrainBusy || simBusy} onClick={retrain}>
                  {retrainBusy ? 'Retraining...' : 'Retrain from analyst labels'}
                </button>
                <span className="text-[0.78rem] text-muted">Needs at least 20 new labels. Can take up to about a minute.</span>
              </div>
              {retrainBusy && (
                <div className="mt-3" data-testid="learning-retrain-progress" role="status" aria-live="polite">
                  <div className="text-[0.82rem]">
                    Training the candidate and evaluating both models... <span className="tnum">{elapsed} s</span>
                  </div>
                  <div
                    className="prob-track mt-1.5 overflow-hidden"
                    role="progressbar"
                    aria-label="Retraining"
                    aria-valuemin={0}
                    aria-valuemax={60}
                    aria-valuenow={Math.min(elapsed, 60)}
                  >
                    <div className="absolute inset-y-0 left-0 rounded-l-[2px]" style={{ width: `${Math.min((elapsed / 60) * 100, 95)}%`, background: 'var(--series-1)' }} />
                  </div>
                </div>
              )}
              {result && !retrainBusy && <RetrainResult r={result} />}
              {!result && !retrainBusy && <p className="mt-3 text-[0.82rem] text-muted">No retrain has run in this session.</p>}
            </Section>
          </div>

          <Section title="Version history" aside="newest first">
            <div className="overflow-x-auto">
              <table className="w-full text-[0.82rem]" data-testid="learning-version-table">
                <thead>
                  <tr className="border-b border-rule text-left text-[0.75rem] text-muted">
                    <th scope="col" className="py-1 pr-3 font-semibold">Version</th>
                    <th scope="col" className="py-1 pr-3 font-semibold">Created</th>
                    <th scope="col" className="py-1 pr-3 font-semibold">Parent</th>
                    <th scope="col" className="py-1 pr-3 text-right font-semibold">Training rows</th>
                    <th scope="col" className="py-1 pr-3 text-right font-semibold">Feedback labels</th>
                    <th scope="col" className="py-1 pr-3 text-right font-semibold">PR-AUC</th>
                    <th scope="col" className="py-1 pr-3 text-right font-semibold">Cost per 1,000</th>
                    <th scope="col" className="py-1 pr-3 text-right font-semibold">FPR hard neg.</th>
                    <th scope="col" className="py-1 font-semibold">
                      <span className="sr-only">Status</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {versions.map((v) => {
                    const active = v.version === s.active_version
                    return (
                      <tr key={v.version} className="border-b border-rule last:border-0" data-testid="learning-version-row" data-version={v.version} data-active={active}>
                        <td className="py-1.5 pr-3 font-mono">{v.version}</td>
                        <td className="tnum py-1.5 pr-3 whitespace-nowrap">{v.created_at ? v.created_at.replace('T', ' ').slice(0, 16) : 'n/a'}</td>
                        <td className="py-1.5 pr-3 font-mono text-ink-2">{v.parent ?? 'none'}</td>
                        <td className="tnum py-1.5 pr-3 text-right">{fmtInt(v.n_train)}</td>
                        <td className="tnum py-1.5 pr-3 text-right">{fmtInt(v.n_feedback_labels)}</td>
                        <td className="tnum py-1.5 pr-3 text-right">{v.metrics ? v.metrics.pr_auc.toFixed(3) : 'n/a'}</td>
                        <td className="tnum py-1.5 pr-3 text-right">{v.metrics ? fmtBRL(v.metrics.cost_per_1k_brl, true) : 'n/a'}</td>
                        <td className="tnum py-1.5 pr-3 text-right">{v.metrics ? fmtPct(v.metrics.fpr_hard_negative, 1) : 'n/a'}</td>
                        <td className="py-1.5 text-right">
                          {active ? (
                            <span className="rounded-sm bg-brand px-1.5 py-0.5 text-[0.72rem] font-semibold text-brand-ink">Active</span>
                          ) : v.ever_deployed === false || v.gate_passed === false ? (
                            <span className="text-[0.72rem] text-ink-2" data-testid="learning-version-rejected">Rejected by gate</span>
                          ) : (
                            <button
                              type="button"
                              className="btn px-2 py-0.5 text-[0.75rem]"
                              data-testid="learning-rollback-button"
                              data-version={v.version}
                              disabled={rollbackBusy !== null || retrainBusy}
                              onClick={() => rollback(v.version)}
                            >
                              {rollbackBusy === v.version ? 'Rolling back...' : 'Roll back to this'}
                            </button>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}
    </div>
  )
}
