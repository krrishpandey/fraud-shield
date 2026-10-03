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
  /** Laya v2 (docs/LAYA_V2.md): who made the call, and Laya's own action proposal when its action head is on */
  decider?: string
  laya_action?: LayaAction | null
}

export interface LayaAction {
  proposed: Action
  accepted: boolean
  overrule_reason: string | null
  margin_brl: number
  probabilities: Partial<Record<Action, number>>
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

export interface PolicyTrace {
  p_fraud?: number
  allowed?: Action[]
  block_downgraded?: boolean
  allow_guard_passed?: boolean
  rule_hits?: string[]
}

export interface DecisionDetail extends ScoreResponse {
  booking: Booking
  explanation: Explanation | null
  analyst: AnalystRecord | null
  policy_trace?: PolicyTrace | null
  first_scan?: FirstScan | null
  owner_confirmation?: OwnerConfirmation | null
}

export interface DialLevel {
  name: string
  target_honest_share?: number
  threshold: number | null
  t6_caught: number
  t6_caught_sd?: number
  honest_weighed: number
  weighs_per_1k_bookings?: number
}
export interface FirstScanDial {
  current: string
  levels: DialLevel[]
  version?: string
  audit_hash?: string
}

export interface FirstScan {
  declared_weight_kg: number
  measured_weight_kg: number
  mismatch: boolean
  source: string
  audit_hash?: string
}

/** GET /decisions/{id}/account-story: the account's own bookings as of this booking. */
export interface StoryBooking {
  booking_id: string
  booked_at: string
  own_goods: boolean
  new_sender: boolean
  new_receiver: boolean
  origin: string
  dest: string
  carrier_cost: number
  current: boolean
}
export interface AccountStory {
  account_id: string
  n_prior: number
  bookings: StoryBooking[]
  last10: { size: number; new_senders: number; new_receivers: number }
  usual: { new_senders_per10: number; new_receivers_per10: number; based_on: number } | null
}

/** POST /decisions/{id}/explanation/check: the explanation validator, run on any text. */
export interface NumberSpan {
  text: string
  start: number
  end: number
  ok: boolean
}
export interface ExplanationCheck {
  ok: boolean
  problems: string[]
  numbers: NumberSpan[]
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
  /** Which model answered, e.g. "stock Laya (not fine-tuned)" while the fine-tuned model is not live. */
  answered_by?: string
}

export interface DashboardMetrics {
  source?: 'app' | 'stream' | 'all'
  stream_bookings?: number
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

/* ---------- live booking stream (GET /stream/...) ---------- */
export interface StreamStatus {
  state: 'idle' | 'ready' | 'running' | 'paused' | 'stopped' | 'done'
  available: boolean
  sent?: number
  total?: number
  rate?: number
  concurrency?: number
  in_flight?: number
  seed?: number
}
export interface StreamAccuracy {
  n: number
  n_fraud: number
  pr_auc: number | null
  roc_auc: number | null
  precision_top1pct_day: number | null
  recall_top1pct_day: number | null
  fraud_caught?: number
  fraud_scan_checked?: number
  fraud_missed?: number
  honest_stopped?: number
  honest_stopped_rate?: number | null
  stopped_by_type?: Record<string, { n: number; stopped: number }>
}
export interface StreamMetricsResponse {
  status: StreamStatus
  load?: {
    scored: number
    errors: number
    retries: number
    in_flight: number
    elapsed_s: number
    throughput_total: number
    throughput_60s: number
    latency_p50_ms: number | null
    latency_p95_ms: number | null
    latency_p99_ms: number | null
    latency_p95_60s_ms: number | null
  }
  timeline?: { s_ago: number; scored: number; latency_p95_ms: number | null }[]
  actions?: Record<string, number>
  degraded_share?: number | null
  accuracy?: StreamAccuracy
  offline_same_rows?: StreamAccuracy
  offline_full?: StreamAccuracy | null
  consistency?: { n_compared: number; max_abs_score_diff: number }
  explanations?: { llm: number; template: number; llm_cap_per_min: number | null; llm_held_blocked?: number }
  scans?: { checked: number; mismatch: number }
  errors?: { booking_id: string; error: string }[]
}
export interface StreamFeedRow {
  decision_id: string
  booking_id: string
  booked_at: string
  route: string
  carrier_cost: number
  action: Action
  score: number
}
export interface StreamFlaggedRow {
  decision_id: string
  booking_id: string
  booked_at: string
  account_id: string
  route: string
  carrier_cost: number
  action: 'hold' | 'block'
  score: number | null
  reasons: string[]
  reviewed: 'fraud' | 'legit' | null
  explanation_status?: string | null
  explanation?: Explanation | null
}
export interface StreamFlagged {
  rows: StreamFlaggedRow[]
  counts: { hold: number; block: number; unreviewed: number }
}

/* ---------- Owner passkey "was this you?" (docs/PASSKEY.md) ---------- */
export interface PasskeyStatus {
  account_id: string
  enrolled: boolean
  rp_id: string
  credential_id_hash: string | null
  enrolled_at: string | null
  simulated_owner_device: boolean
}
/** WebAuthn creation options with binary fields as base64url strings. */
export interface PasskeyEnrollOptions {
  nonce_id: string
  expires_in_s: number
  publicKey: {
    challenge: string
    rp: { id: string; name: string }
    user: { id: string; name: string; displayName: string }
    pubKeyCredParams: { type: 'public-key'; alg: number }[]
    timeout: number
    attestation: 'none'
    authenticatorSelection: { residentKey: string; userVerification: string }
  }
}
export interface PasskeyEnrollResult {
  enrolled: boolean
  account_id?: string
  credential_id_hash?: string
  replaced?: boolean
  audit_hash?: string
  code?: string
  reason?: string
}
export type BoundFields = {
  booking_id: string
  account_id: string
  carrier_cost: number
  declared_value: number
  dest_zip3: string
  consignee_id: string
}
export interface OwnerConfirmOptions {
  nonce_id: string
  challenge: string
  rp_id: string
  allow_credentials: { type: 'public-key'; id: string }[]
  timeout: number
  user_verification: string
  expires_in_s: number
  bound_fields: BoundFields
  bound_fields_hash: string
  release_action: Action
}
/** Browser credential JSON (base64url fields) as sent to the verify endpoints. */
export type CredentialJSON = {
  id: string
  rawId: string
  type: string
  response: Record<string, string | null>
}
export interface OwnerConfirmation {
  verified: boolean
  released_action?: Action
  original_action?: Action
  credential_id_hash?: string
  bound_fields_hash?: string
  sign_count?: number
  verify_ms?: number
  server_ms?: number
  code?: string
  reason?: string
  audit_hash: string
  at?: string
}
export interface TamperTestResult {
  verified: boolean
  code: string | null
  reason: string
  tampered_field: string
  original_fields: BoundFields
  tampered_fields: BoundFields
  original_bound_fields_hash: string
  tampered_bound_fields_hash: string
  verify_ms: number | null
  audit_hash: string
}
