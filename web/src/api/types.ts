// Mirrors docs/API.md and fraudshield/contracts.py. Keep in sync with the backend contract.

export const ACTIONS = ['allow', 'allow_scan_gated', 'owner_confirm', 'review', 'hold', 'block'] as const
export type Action = (typeof ACTIONS)[number]

export const SERVED_QUESTIONS = ['misuse', 'foreign_senders', 'payoff_max', 'drop_consignee'] as const
export type QuestionId = (typeof SERVED_QUESTIONS)[number]

export interface Booking {
  booking_id: string
  account_id: string
  booked_at: string
  channel: 'web' | 'api' | 'counter'
  login_device_age_days: number
  payment_method: 'account_billing' | 'card' | 'ach'
  sender_id: string
  origin_uf: string
  origin_zip3: string
  dest_uf: string
  dest_zip3: string
  consignee_id: string
  weight_kg: number
  length_cm: number
  width_cm: number
  height_cm: number
  service: 'standard' | 'express'
  category: string
  declared_value: number
  carrier_cost: number
  owner_contact_age_days?: number | null
  meta?: Record<string, unknown>
}

export interface TopFeature {
  name: string
  value: number | string
  baseline: number | string | null
}

export interface ScoreResponse {
  decision_id: string
  booking_id: string
  action: Action
  greedy_action: Action
  propensity: number
  explored: boolean
  degraded: boolean
  probabilities: Partial<Record<QuestionId, number>> & Record<string, number>
  raw_probabilities: Partial<Record<QuestionId, number>> & Record<string, number>
  gbm_score: number | null
  expected_costs: Partial<Record<Action, number>>
  reasons: string[]
  top_features: TopFeature[]
  state_text: string
  model_versions: Record<string, string>
  latency_ms: Record<string, number> & { total: number }
  explanation_status: 'pending' | 'ready' | 'failed' | string
  audit_hash: string
}

export interface Explanation {
  text: string
  source: 'llm' | 'template'
  valid: boolean
  model_id: string | null
}

export interface AnalystRecord {
  label: 'fraud' | 'legit'
  note: string
  at: string
}

export interface DecisionDetail extends ScoreResponse {
  booking: Booking
  explanation: Explanation | null
  analyst: AnalystRecord | null
}

export interface DecisionSummary {
  decision_id: string
  booking_id: string
  account_id: string
  booked_at: string
  action: Action
  misuse: number
  carrier_cost: number
  analyst_label: 'fraud' | 'legit' | null
  scenario: string | null
}

export interface AnalystRequest {
  label: 'fraud' | 'legit'
  note: string
}
export interface AnalystResponse {
  ok: boolean
  audit_hash: string
}

export interface AskRequest {
  instructions: string
  yes: string
  no: string
}
export interface AskResponse {
  qid: string
  probability_yes: number
  raw_probabilities: Record<string, number>
  latency_ms: number
  calibrated: boolean
}

export interface DashboardMetrics {
  window: { from: string; to: string }
  totals: { bookings: number; by_action: Partial<Record<Action, number>> }
  held_shipments: number
  revenue_loss_prevented_brl: number
  friction_cost_brl: number
  net_prevented_brl: number
  fpr_legit: number
  fpr_hard_negative: number
  trend: { date: string; fraud_stopped: number; held: number; by_typology: Record<string, number> }[]
  latency: { p50_ms: number; p99_ms: number }
  assumptions: { cost_matrix_version: string; note: string }
}

export interface AuditVerify {
  ok: boolean
  records: number
  head_hash: string
  first_bad_index: number | null
}

export interface Health {
  ok: boolean
  laya_mode: 'local' | 'http' | 'cached' | string
  gpu: boolean
  degraded?: boolean
  versions?: Record<string, string>
}

/** Proposed addition: GET /demo/bookings. Falls back to src/fixtures/demo_bookings.json. */
export interface DemoBooking {
  scenario: string
  title: string
  description: string
  expected?: string
  booking: Booking
}

/* ---------- Continuous learning (docs/API.md v1.2) ---------- */

export interface LearningMetrics {
  pr_auc: number
  ece: number
  cost_per_1k_brl: number
  fpr_hard_negative: number
  recall_new_pattern: number
}

export interface ModelVersion {
  version: string
  created_at: string | null
  parent: string | null
  n_train: number
  n_feedback_labels: number
  metrics: LearningMetrics | null
  deployed: boolean
  /** true if this version ever passed the gate; only such versions can be rolled back to */
  ever_deployed?: boolean
  gate_passed?: boolean | null
}

export interface LearningStatus {
  active_version: string
  labels_since_last_retrain: number
  label_sources: Record<string, number>
  versions: ModelVersion[]
  calibration_version: string
  laya_export: { path: string; rows: number } | null
}

export interface SimulateFeedbackResponse {
  added: number
  fraud: number
  legit: number
  simulated: boolean
}

export interface GateCheck {
  name: string
  passed: boolean
  detail: string
}

export interface RetrainResponse {
  run_id: string
  n_new_labels: number
  n_fraud: number
  n_legit: number
  eval_set: { n: number; description: string }
  current: LearningMetrics & { version: string }
  candidate: LearningMetrics & { version: string }
  gate: { passed: boolean; checks: GateCheck[] }
  deployed_version: string | null
  audit_hash: string
}

export interface RollbackResponse {
  active_version: string
  audit_hash: string
}
