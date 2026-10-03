// MOCK MODE ONLY. Invented continuous-learning state for offline UI work and E2E tests.
// Retrain runs alternate: odd runs (1st, 3rd, ...) pass the gate and deploy, even runs are rejected,
// so both outcomes can be shown. Shapes follow docs/API.md "Continuous learning (v1.2)".
import type { Handover, LearningMetrics, LearningStatus, ModelVersion, RetrainResponse } from '../api/types'

const RETRAIN_DELAY_MS = 2500

const v1: ModelVersion = {
  version: 'gbm-B2-F-v1',
  created_at: '2026-09-28T09:00:00',
  parent: null,
  n_train: 59000,
  n_feedback_labels: 0,
  metrics: { pr_auc: 0.61, ece: 0.012, cost_per_1k_brl: 410.0, fpr_hard_negative: 0.03, recall_new_pattern: 0.2 },
  deployed: true,
}

const state = {
  versions: [v1] as ModelVersion[],
  active: v1.version,
  sources: { analyst: 0, simulated_analyst: 0 } as Record<string, number>,
  exportRows: 0,
  runs: 0,
  fraudSince: 0,
  activeSince: new Date().toISOString().slice(0, 19),
  lastHandover: null as Handover | null,
}

const now = () => new Date().toISOString().slice(0, 19)
const rollbackTarget = () =>
  [...state.versions].reverse().find((v) => v.version !== state.active && v.deployed !== false)?.version ?? null

export function recordAnalystLabel(label: 'fraud' | 'legit') {
  state.sources.analyst += 1
  state.exportRows += 1
  if (label === 'fraud') state.fraudSince += 1
}

const since = () => Object.values(state.sources).reduce((a, b) => a + b, 0)

export function learningStatus(): LearningStatus {
  return {
    active_version: state.active,
    labels_since_last_retrain: since(),
    label_sources: { ...state.sources },
    versions: state.versions.map((v) => ({ ...v, deployed: v.version === state.active })),
    calibration_version: 'cal-mock-00000000',
    laya_export: { path: 'artifacts/feedback/laya_feedback.jsonl', rows: state.exportRows },
    model_in_use: {
      version: state.active,
      since: state.activeSince,
      since_reason: 'activated',
      registry_active_version: state.active,
      matches_registry: true,
      rollback_target: rollbackTarget(),
    },
    last_handover: state.lastHandover,
  }
}

export function simulateFeedback(n: number, seed: number) {
  const fraud = Math.round(n * (0.15 + ((seed * 7) % 5) / 100))
  state.sources.simulated_analyst += n
  state.exportRows += n
  state.fraudSince += fraud
  return { added: n, fraud, legit: n - fraud, simulated: true }
}

const wait = (ms: number) => new Promise((r) => setTimeout(r, ms))

export async function retrain(minNew: number): Promise<{ status: number; body: unknown }> {
  const n = since()
  if (n < minNew) {
    return { status: 409, body: { detail: `Only ${n} new labels since the last retrain; at least ${minNew} are needed.` } }
  }
  await wait(RETRAIN_DELAY_MS)
  state.runs += 1
  const pass = state.runs % 2 === 1
  const cur = state.versions.find((v) => v.version === state.active)!
  const c = cur.metrics! // mock versions always carry metrics
  const cand: LearningMetrics = pass
    ? {
        pr_auc: +(c.pr_auc + 0.05).toFixed(3),
        ece: +(c.ece + 0.001).toFixed(3),
        cost_per_1k_brl: +(c.cost_per_1k_brl * 0.91).toFixed(1),
        fpr_hard_negative: c.fpr_hard_negative,
        recall_new_pattern: +Math.min(c.recall_new_pattern + 0.35, 0.95).toFixed(2),
      }
    : {
        pr_auc: +(c.pr_auc + 0.01).toFixed(3),
        ece: +(c.ece + 0.009).toFixed(3),
        cost_per_1k_brl: +(c.cost_per_1k_brl * 1.04).toFixed(1),
        fpr_hard_negative: +(c.fpr_hard_negative + 0.02).toFixed(3),
        recall_new_pattern: c.recall_new_pattern,
      }
  const nextNum = state.versions.length + 1
  const candVersion = `gbm-B2-F-v${nextNum}`
  const checks = [
    {
      name: 'cost_not_worse',
      passed: cand.cost_per_1k_brl <= c.cost_per_1k_brl,
      detail: `Cost per 1,000 bookings R$${cand.cost_per_1k_brl.toFixed(1)} vs R$${c.cost_per_1k_brl.toFixed(1)} (must not increase)`,
    },
    {
      name: 'hard_negative_fpr',
      passed: cand.fpr_hard_negative <= c.fpr_hard_negative + 0.005,
      detail: `FPR on hard negatives ${(cand.fpr_hard_negative * 100).toFixed(1)}% vs ${(c.fpr_hard_negative * 100).toFixed(1)}% (max +0.5 points)`,
    },
    {
      name: 'calibration',
      passed: cand.ece <= 0.02,
      detail: `ECE ${cand.ece.toFixed(3)} (must stay at or below 0.020)`,
    },
    {
      name: 'min_labels',
      passed: true,
      detail: `${n} new labels (minimum ${minNew})`,
    },
  ]
  const passed = checks.every((x) => x.passed)
  const nFraud = Math.min(state.fraudSince, n)
  const res: RetrainResponse = {
    run_id: `rt_${String(state.runs).padStart(4, '0')}`,
    n_new_labels: n,
    n_fraud: nFraud,
    n_legit: n - nFraud,
    eval_set: { n: 5000, description: 'validation window + latest 30% of feedback labels by time, never trained on' },
    current: { version: cur.version, ...c },
    candidate: { version: candVersion, ...cand },
    gate: { passed, checks },
    deployed_version: passed ? candVersion : cur.version,
    audit_hash: '',
  }
  if (passed) {
    state.versions.push({
      version: candVersion,
      created_at: new Date().toISOString().slice(0, 19),
      parent: cur.version,
      n_train: cur.n_train + n,
      n_feedback_labels: cur.n_feedback_labels + n,
      metrics: cand,
      deployed: true,
    })
    state.active = candVersion
    state.sources = { analyst: 0, simulated_analyst: 0 }
    state.fraudSince = 0
    state.activeSince = now()
  } else {
    state.versions.push({ version: candVersion, created_at: now(), parent: cur.version, n_train: cur.n_train + n,
      n_feedback_labels: cur.n_feedback_labels + n, metrics: cand, deployed: false, ever_deployed: false, gate_passed: false })
  }
  const failed = checks.filter((x) => !x.passed).map((x) => ({ name: x.name, plain: x.detail, detail: x.detail }))
  res.handover = {
    event: 'retrain', run_id: res.run_id, at: now(), verdict: passed ? 'new_model_in_use' : 'previous_model_kept',
    active_version: state.active, previous_version: cur.version, candidate_version: candVersion,
    active_since: state.activeSince, rollback_target: rollbackTarget(), failed_checks: failed,
    new_pattern: { current: c.recall_new_pattern, candidate: cand.recall_new_pattern, ci95: null, typologies: ['T5'], source: 'mock' },
    message: passed ? `From now on, new bookings are scored by ${candVersion}.` : `Still using ${cur.version}.`,
  }
  state.lastHandover = res.handover
  return { status: 200, body: res }
}

export function rollback(version: string): { status: number; body: unknown } {
  if (!state.versions.some((v) => v.version === version)) return { status: 404, body: { detail: `unknown version ${version}` } }
  const prev = state.active
  state.active = version
  state.activeSince = now()
  const handover: Handover = {
    event: 'rollback', run_id: null, at: state.activeSince, verdict: 'rolled_back', active_version: version,
    previous_version: prev, candidate_version: null, active_since: state.activeSince, rollback_target: rollbackTarget(),
    failed_checks: [], new_pattern: null, message: `Rolled back. From now on, new bookings are scored by ${version}.`,
  }
  state.lastHandover = handover
  return { status: 200, body: { active_version: version, previous_version: prev, audit_hash: '', handover } }
}
