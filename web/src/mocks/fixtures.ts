// MOCK DATA ONLY. Every number in this file is invented for offline UI development and E2E tests.
// Shapes follow docs/API.md exactly. The console shows a "mock data" banner whenever these are in use.
import type { Action, DashboardMetrics, Explanation, Health, ScoreResponse } from '../api/types'

export type ScoreTemplate = Omit<ScoreResponse, 'decision_id' | 'booking_id' | 'audit_hash' | 'explanation_status'> & {
  explanation: Explanation
}

const versions = {
  laya: 'mock-laya-ft-0000000',
  calibration: 'cal-mock-00000000',
  gbm: 'gbm-mock',
  policy: 'policy-mock',
  serializer: 'ser-mock',
}

export const templates: Record<string, ScoreTemplate> = {
  legit: {
    action: 'allow', greedy_action: 'allow', propensity: 1.0, explored: false, degraded: false,
    probabilities: { misuse: 0.012, foreign_senders: 0.03, payoff_max: 0.05, drop_consignee: 0.04 },
    raw_probabilities: { misuse: 0.06, foreign_senders: 0.08, payoff_max: 0.11, drop_consignee: 0.09 },
    gbm_score: 0.02,
    expected_costs: { allow: 0.2, allow_scan_gated: 1.2, owner_confirm: 6.1, review: 12.0, hold: 18.3, block: 41.0 },
    reasons: [],
    top_features: [
      { name: 'new_senders_last10', value: 0, baseline: 0.0 },
      { name: 'cost_vs_median', value: 1.1, baseline: 1.0 },
      { name: 'login_device_age_days', value: 640, baseline: 640 },
    ],
    state_text:
      'BOOKING 2018-06-14 Thu 10:12 | channel web | device age 640d | pay account_billing\n' +
      'ACCOUNT tenure 731d | bookings 90d 412 | senders 90d 1 (self) | origins 90d SP-130\n' +
      'THIS sender self | origin SP-130 seen 98% | dest RJ-220 seen 31%\n' +
      'PARCEL 1.4kg 30x20x12 standard housewares | cost R$18.40 = 1.1x median\n' +
      'CHANGE sender entropy 7d 0.00 vs 90d 0.00 | new senders last10 0',
    model_versions: versions,
    latency_ms: { total: 118.4, features: 15.2, laya: 92.7, decide: 0.3, audit: 2.9 },
    explanation: {
      text: 'Allowed. The booking matches this account: same sender, same origin, a device used for 640 days, and a cost of R$18.40, 1.1 times its usual parcel.',
      source: 'template', valid: true, model_id: null,
    },
  },
  hard_negative_new_state: {
    action: 'allow_scan_gated', greedy_action: 'allow_scan_gated', propensity: 0.95, explored: false, degraded: false,
    probabilities: { misuse: 0.061, foreign_senders: 0.09, payoff_max: 0.07, drop_consignee: 0.05 },
    raw_probabilities: { misuse: 0.21, foreign_senders: 0.24, payoff_max: 0.14, drop_consignee: 0.11 },
    gbm_score: 0.08,
    expected_costs: { allow: 1.9, allow_scan_gated: 1.4, owner_confirm: 6.6, review: 12.4, hold: 18.0, block: 40.2 },
    reasons: ['NEW_DESTINATION_STATE'],
    top_features: [
      { name: 'dest_uf_seen_before', value: 0, baseline: 1 },
      { name: 'new_senders_last10', value: 0, baseline: 0.0 },
      { name: 'cost_vs_median', value: 1.3, baseline: 1.0 },
    ],
    state_text:
      'BOOKING 2018-06-14 Thu 14:05 | channel web | device age 410d | pay account_billing\n' +
      'ACCOUNT tenure 520d | bookings 90d 188 | senders 90d 1 (self) | origins 90d MG-301\n' +
      'THIS sender self | origin MG-301 seen 100% | dest BA-402 seen 0% (first time in BA)\n' +
      'PARCEL 2.1kg 35x25x15 standard health_beauty | cost R$31.70 = 1.3x median\n' +
      'CHANGE sender entropy 7d 0.00 vs 90d 0.00 | new senders last10 0',
    model_versions: versions,
    latency_ms: { total: 124.0, features: 16.8, laya: 97.1, decide: 0.4, audit: 3.0 },
    explanation: {
      text: 'Label issued, checked at first scan. This is the first parcel from this account to BA, but the sender, origin, device and cost (R$31.70, 1.3 times usual) all match the account. A new destination alone is not a reason to stop it.',
      source: 'llm', valid: true, model_id: 'mock-explainer',
    },
  },
  T1: {
    action: 'hold', greedy_action: 'hold', propensity: 0.95, explored: false, degraded: false,
    probabilities: { misuse: 0.91, foreign_senders: 0.94, payoff_max: 0.88, drop_consignee: 0.07 },
    raw_probabilities: { misuse: 0.97, foreign_senders: 0.99, payoff_max: 0.93, drop_consignee: 0.12 },
    gbm_score: 0.71,
    expected_costs: { allow: 205.1, allow_scan_gated: 131.0, owner_confirm: 40.2, review: 66.0, hold: 23.0, block: 30.1 },
    reasons: ['NEW_SENDERS_UNDER_PAYER', 'COST_FAR_ABOVE_ACCOUNT_NORM', 'NEW_LOGIN_DEVICE'],
    top_features: [
      { name: 'new_senders_last10', value: 8, baseline: 0.0 },
      { name: 'cost_vs_median', value: 11.6, baseline: 1.0 },
      { name: 'login_device_age_days', value: 0, baseline: 412 },
      { name: 'origin_seen_share', value: 0.0, baseline: 0.97 },
    ],
    state_text:
      'BOOKING 2018-06-14 Thu 02:41 | channel api | device age 0d | pay account_billing\n' +
      'ACCOUNT tenure 412d | bookings 90d 236 | senders 90d 1 (self) | origins 90d SP-013\n' +
      'THIS sender snd_91aa new | origin PR-806 seen 0% | dest AM-690 seen 0%\n' +
      'PARCEL 9.8kg 50x40x35 express electronics | cost R$212.60 = 11.6x median\n' +
      'CHANGE sender entropy 7d 2.81 vs 90d 0.00 | new senders last10 8',
    model_versions: versions,
    latency_ms: { total: 142.3, features: 18.1, laya: 101.0, decide: 0.4, audit: 3.2 },
    explanation: {
      text: 'Held: no label until verified. This account has paid only for its own parcels from SP-013, but 8 of its last 10 bookings came from new senders. This one ships from PR-806, costs R$212.60 (11.6 times its usual parcel), and was booked at 02:41 from a device first seen today.',
      source: 'llm', valid: true, model_id: 'mock-explainer',
    },
  },
  T3: {
    action: 'review', greedy_action: 'review', propensity: 0.95, explored: false, degraded: false,
    probabilities: { misuse: 0.38, foreign_senders: 0.12, payoff_max: 0.31, drop_consignee: 0.82 },
    raw_probabilities: { misuse: 0.55, foreign_senders: 0.2, payoff_max: 0.42, drop_consignee: 0.82 },
    gbm_score: 0.29,
    expected_costs: { allow: 58.4, allow_scan_gated: 37.9, owner_confirm: 22.5, review: 14.1, hold: 19.8, block: 36.0 },
    reasons: ['CONSIGNEE_MANY_UNRELATED_PAYERS', 'CONSIGNEE_ADDRESS_RECENT', 'NEW_LOGIN_DEVICE'],
    top_features: [
      { name: 'consignee_payers_30d', value: 11, baseline: 1.0 },
      { name: 'consignee_age_days', value: 21, baseline: 400 },
      { name: 'login_device_age_days', value: 3, baseline: 95 },
    ],
    state_text:
      'BOOKING 2018-06-14 Thu 19:22 | channel web | device age 3d | pay card\n' +
      'ACCOUNT tenure 95d | bookings 90d 41 | senders 90d 1 (self) | origins 90d SP-041\n' +
      'THIS sender self | origin SP-041 seen 100% | dest SP-087 seen 0%\n' +
      'CONSIGNEE cns_d7f1 age 21d | paying accounts 30d 11\n' +
      'PARCEL 3.2kg 40x30x20 express computers_accessories | cost R$64.30 = 2.4x median',
    model_versions: versions,
    latency_ms: { total: 131.9, features: 17.4, laya: 99.2, decide: 0.4, audit: 3.1 },
    explanation: {
      text: 'Sent to an analyst before accepting. The consignee cns_d7f1 is 21 days old and has received parcels paid for by 11 unrelated accounts in 30 days, a reshipping drop pattern.',
      source: 'template', valid: true, model_id: null,
    },
  },
  T6: {
    action: 'allow_scan_gated', greedy_action: 'allow_scan_gated', propensity: 0.95, explored: false, degraded: false,
    probabilities: { misuse: 0.09, foreign_senders: 0.04, payoff_max: 0.33, drop_consignee: 0.06 },
    raw_probabilities: { misuse: 0.18, foreign_senders: 0.07, payoff_max: 0.47, drop_consignee: 0.1 },
    gbm_score: 0.14,
    expected_costs: { allow: 6.8, allow_scan_gated: 2.1, owner_confirm: 9.0, review: 13.5, hold: 19.6, block: 42.7 },
    reasons: ['DENSITY_IMPLAUSIBLE'],
    top_features: [
      { name: 'declared_density_kg_m3', value: 5.9, baseline: 120.0 },
      { name: 'cost_vs_median', value: 0.9, baseline: 1.0 },
    ],
    state_text:
      'BOOKING 2018-06-14 Thu 11:48 | channel web | device age 220d | pay account_billing\n' +
      'ACCOUNT tenure 300d | bookings 90d 97 | senders 90d 1 (self) | origins 90d SC-890\n' +
      'THIS sender self | origin SC-890 seen 100% | dest PE-500 seen 12%\n' +
      'PARCEL 0.8kg 60x50x45 standard furniture_decor | density 5.9 kg/m3 | cost R$47.90 = 0.9x median',
    model_versions: versions,
    latency_ms: { total: 120.6, features: 16.0, laya: 95.5, decide: 0.3, audit: 2.8 },
    explanation: {
      text: 'Label issued, checked at first scan. A 60x50x45 cm box declared at 0.8 kg is far lighter than similar parcels. The account itself looks normal, so the parcel is weighed at first scan instead of being stopped now.',
      source: 'template', valid: true, model_id: null,
    },
  },
}

export function templateFor(
  scenario: string | undefined,
  b: { sender_id: string; account_id: string; login_device_age_days: number },
): string {
  if (scenario && templates[scenario]) return scenario
  if (b.sender_id !== b.account_id && b.login_device_age_days < 2) return 'T1'
  return 'legit'
}

export const health: Health = {
  ok: true,
  laya_mode: 'local',
  gpu: true,
  degraded: false,
  versions: { laya: versions.laya, calibration: versions.calibration, policy: versions.policy },
}

// Seed history for the review queue (mock only).
export const seedHistory: { booking_id: string; account_id: string; booked_at: string; scenario: string; carrier_cost: number }[] = [
  { booking_id: 'mock-hist-01', account_id: 'acc_a101', booked_at: '2018-06-13T03:10:00', scenario: 'T1', carrier_cost: 188.2 },
  { booking_id: 'mock-hist-02', account_id: 'acc_b202', booked_at: '2018-06-13T09:44:00', scenario: 'legit', carrier_cost: 22.9 },
  { booking_id: 'mock-hist-03', account_id: 'acc_c303', booked_at: '2018-06-13T12:30:00', scenario: 'hard_negative_new_state', carrier_cost: 35.1 },
  { booking_id: 'mock-hist-04', account_id: 'acc_d404', booked_at: '2018-06-13T16:02:00', scenario: 'T3', carrier_cost: 71.0 },
  { booking_id: 'mock-hist-05', account_id: 'acc_e505', booked_at: '2018-06-13T20:15:00', scenario: 'T6', carrier_cost: 52.4 },
  { booking_id: 'mock-hist-06', account_id: 'acc_a101', booked_at: '2018-06-13T03:22:00', scenario: 'T1', carrier_cost: 240.8 },
  { booking_id: 'mock-hist-07', account_id: 'acc_f606', booked_at: '2018-06-13T22:51:00', scenario: 'legit', carrier_cost: 15.6 },
]

const typologies = ['T1', 'T2', 'T3', 'T4', 'T6', 'T7']
function buildTrend(): DashboardMetrics['trend'] {
  // Deterministic invented series for mock mode only.
  const out: DashboardMetrics['trend'] = []
  const start = Date.UTC(2018, 5, 1)
  for (let i = 0; i < 21; i++) {
    const d = new Date(start + i * 86400000).toISOString().slice(0, 10)
    const by: Record<string, number> = {}
    let total = 0
    typologies.forEach((t, k) => {
      const v = (i * 7 + k * 5 + (i % 3) * k) % (t === 'T1' ? 5 : 3)
      if (v > 0) {
        by[t] = v
        total += v
      }
    })
    out.push({ date: d, fraud_stopped: total, held: Math.max(0, Math.round(total * 0.6) - (i % 2)), by_typology: by })
  }
  return out
}

const byAction: Partial<Record<Action, number>> = {
  allow: 18800, allow_scan_gated: 300, owner_confirm: 60, review: 40, hold: 50, block: 13,
}

export const dashboard: DashboardMetrics = {
  window: { from: '2018-06-01', to: '2018-06-21' },
  totals: { bookings: Object.values(byAction).reduce((a, b) => a + (b ?? 0), 0), by_action: byAction },
  held_shipments: 63,
  revenue_loss_prevented_brl: 41250.0,
  friction_cost_brl: 3100.0,
  net_prevented_brl: 38150.0,
  fpr_legit: 0.012,
  fpr_hard_negative: 0.031,
  trend: buildTrend(),
  latency: { p50_ms: 120.0, p99_ms: 190.0 },
  assumptions: { cost_matrix_version: 'costs-v1', note: 'friction and loss costs are assumptions' },
}
