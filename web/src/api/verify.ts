// Tracd Verify API (fraudshield/voice/router.py). Shapes mirror the server's dicts.

export type VerifyStatus =
  | 'pending_verification'
  | 'calling'
  | 'ringing'
  | 'in_call'
  | 'retry_scheduled'
  | 'awaiting_simulated_answer'
  | 'released'
  | 'blocked'
  | 'analyst_review'

export interface VerifyFeedRow {
  decision_id: string
  booking_id: string
  account_id: string
  company: string | null
  booked_at: string
  origin: string
  dest: string
  display_labels: boolean
  weight_kg: number | null
  service: string | null
  carrier_cost: number | null
  risk: number | null
  action: string
  latency_ms: number | null
  verify_status: VerifyStatus | null
  verify_mode: 'real' | 'simulated' | null
  final_action: string | null
  scenario: string | null
  source: string | null
  kind: 'real' | 'demo'
  laya_summary: string | null
  label: 'fraud' | 'honest' | null // the dataset's own label (real test-window bookings only)
}

export interface CallAttempt {
  n: number
  execution_id: string
  mode: 'real' | 'simulated'
  started_at: string
  status: string
  statuses: { status: string; at: string; source?: string }[]
  final: boolean
  ended_at: string | null
  duration_s: number | null
  cost_usd?: number
  recording_url?: string | null
}

export interface CallOutcome {
  owner_confirmed: 'yes' | 'no' | 'unsure'
  detail_answer: string | null
  change_reason: string | null
  language: string | null
  extraction_source: string
  detail_match: boolean | null
  branch: 'deny' | 'confirm' | 'unsure' | 'detail_mismatch'
}

export interface Rescore {
  p_before: number
  p_model_after_change: number
  likelihood_ratio: number
  p_after: number
  damped_groups: string[]
  damped_features: Record<string, [number, number]>
  action: string
  expected_costs: Record<string, number>
  allowed: string[]
  floor: string
  hard_signal: boolean
}

export interface VerifyCase {
  opened_at: string
  reason: string
  todo: string
  status: 'open' | 'resolved'
  expected_loss: number
  review_min: number
  loss_per_analyst_min: number
  resolution: { label: 'fraud' | 'legit'; note: string; by: string; at: string } | null
}

export interface Verification {
  verification_id: string
  decision_id: string
  booking_id: string
  account_id: string
  original_action: string
  p_fraud: number
  carrier_cost: number | null
  booking: {
    weight_kg: number
    service: string
    origin_label: string
    dest_label: string
    lane_data: string
    display_labels: boolean
    time_label: string
    booked_at: string
    device_is_new: boolean
    carrier_cost: number | null
    scenario: string | null
    reasons: string[]
  }
  owner: {
    company: string
    masked_phone: string | null
    has_number: boolean
    language: string
    language_name: string
    contact_age_days: number | null
    source: string
  }
  status: VerifyStatus
  mode: 'real' | 'simulated' | null
  simulated_reason: string | null
  held_at: string
  attempts: CallAttempt[]
  transcript: string
  outcome: CallOutcome | null
  rescore: Rescore | null
  final_action: string | null
  final_at: string | null
  minutes_to_final?: number
  case: VerifyCase | null
  sms: { at: string; to: string | null; text: string; sent: boolean; stub: boolean } | null
  timeline: { at: string; step: string; detail: string; execution_id?: string; latency_ms?: number }[]
  audit_hashes: { event: string; hash: string; at: string }[]
  account_locked: boolean
  confirmed_change?: { groups: string[]; reason: string; until: string }
  laya?: LayaExplanation | null
  decision?: {
    expected_costs: Record<string, number> | null
    top_features: { name: string; value: number; baseline: number | null; code: string }[] | null
    reasons: string[] | null
    probabilities: Record<string, number> | null
    latency_ms: Record<string, number> | null
    audit_hash: string | null
    hard_signal: boolean | null
    explanation: { text: string; source: string; valid: boolean } | null
  }
}

export interface VerifyStatusInfo {
  enabled: boolean
  mode: string
  real_calls_ready: boolean
  agent_id: string | null
  api_key_present: boolean
  webhook_secret_set: boolean
  webhook_url: string | null
  min_contact_days: number
  max_attempts: number
  retry_delay_s: number
  yes_likelihood_ratio: number
  release_floor: string
}

export interface VerifyMetrics {
  calls_made: number
  calls_live: number
  real_calls: number
  simulated_calls: number
  call_minutes: number
  call_cost_usd: number
  owner_said: { no: number; yes: number; unsure: number }
  no_answer: number
  not_called: number
  source: string
  bookings_scored: number
  verifications: number
  simulated_verifications: number
  real_verifications: number
  calls_finished: number
  calls_answered: number
  call_answer_rate: number | null
  median_minutes_to_release_honest: number | null
  released_by_call: number
  blocked_by_call: number
  analyst_minutes_saved_per_1000: number | null
  review_min_assumption: number
  honest_held: number
  false_positive_rate: number | null
  fraud_caught_at_booking: number
  open_cases: number
  labels_note: string
}

export interface VerifyScenario {
  scenario: string
  title: string
  description: string
  expected: string
  say: string
}

export interface SimAnswer {
  owner_confirmed: 'yes' | 'no' | 'unsure' | 'no_answer'
  detail_answer?: string | null
  change_reason?: string | null
  language?: string | null
}

export interface AuditRecord {
  seq: number
  ts: string
  event_type: string
  payload: Record<string, unknown>
  prev_hash: string
  record_hash: string
}

export interface LayaExplanation {
  source: 'live' | 'cached' | 'demo'
  note: string
  summary: string
  decides: string
  items: { id: string; question: string; yes: boolean; probability: number | null; fact: string | null; text: string }[]
}

export interface LayaStatus {
  connected: boolean
  why_not: string
  decides: boolean
  cache_entries: number
  model: string
  runs: { name: string; where: string; data: string; questions: string[] | null; trainable_params: number | null; result_file: string }[]
  pr_auc: Record<string, number> | null
  verdict: string | null
  win_condition_met: boolean | null
  files: string[]
}
