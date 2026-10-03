import type { Action, QuestionId } from '../api/types'

export interface ActionMeta {
  label: string
  short: string
  meaning: string
  label_status: string
  glyph: string
  severity: number
}

// Plain-language action semantics from DESIGN.md section 6.
export const ACTION_META: Record<Action, ActionMeta> = {
  allow: {
    label: 'Allow',
    short: 'Allow',
    meaning: 'Accepted normally. The label is issued with no extra checks.',
    label_status: 'Label issued',
    glyph: '✓',
    severity: 0,
  },
  allow_scan_gated: {
    label: 'Allow, check at first scan',
    short: 'Scan-gated',
    meaning:
      'The label is issued, but the parcel is held at first scan if its weight, size or drop-off point differ from the booking.',
    label_status: 'Label issued, gated at first scan',
    glyph: '◎',
    severity: 1,
  },
  owner_confirm: {
    label: 'Ask the owner',
    short: 'Owner confirm',
    meaning:
      'The account owner gets a one-tap confirmation on a contact verified more than 30 days ago. No answer in 30 minutes sends it to review.',
    label_status: 'Label waits for the owner',
    glyph: '?',
    severity: 2,
  },
  review: {
    label: 'Send to review',
    short: 'Review',
    meaning: 'A fraud analyst checks the booking before it is accepted. Target response time is 15 minutes.',
    label_status: 'Label waits for an analyst',
    glyph: '⚑',
    severity: 3,
  },
  hold: {
    label: 'Hold',
    short: 'Hold',
    meaning: 'No label is issued until the booking is verified.',
    label_status: 'Label not issued',
    glyph: '⏸',
    severity: 4,
  },
  block: {
    label: 'Block',
    short: 'Block',
    meaning:
      'The booking is refused and label creation is locked. Needs a hard signal and an analyst confirmation within 24 hours, otherwise it drops to hold.',
    label_status: 'Booking refused',
    glyph: '✕',
    severity: 5,
  },
}

export const actionLabel = (a: string) => (ACTION_META as Record<string, ActionMeta>)[a]?.label ?? a
export const actionShort = (a: string) => (ACTION_META as Record<string, ActionMeta>)[a]?.short ?? a

export const QUESTION_META: Record<QuestionId, { title: string; question: string; zeroShot: boolean }> = {
  misuse: {
    title: 'Account misuse',
    question: 'Is someone other than the legitimate holder using this account?',
    zeroShot: false,
  },
  foreign_senders: {
    title: 'Senders outside the account base',
    question: 'Is the account now paying for senders or origins it never served before?',
    zeroShot: false,
  },
  payoff_max: {
    title: 'Cost-maximizing parcel',
    question: 'Was this parcel chosen to push the shipping cost far above the account norm?',
    zeroShot: false,
  },
  drop_consignee: {
    title: 'Reshipping drop',
    question: 'Does the consignee look like a reshipping drop address?',
    zeroShot: true,
  },
}

// Same wording as fraudshield/policy/reasons.py REASON_PLAIN (served at GET /reason-codes).
const REASONS: Record<string, string> = {
  NEW_SENDERS_UNDER_PAYER: 'The account is paying for senders it never shipped for before',
  COST_FAR_ABOVE_ACCOUNT_NORM: 'Shipping cost is far above what this account usually pays',
  NEW_LOGIN_DEVICE: 'Booked from a login device first seen recently',
  NEW_ORIGINS: 'Parcels from origins new to this account',
  SENDER_DIFFERS_FROM_ACCOUNT: 'The sender is not the account holder',
  PAYER_MANY_SENDERS: 'The account paid for many different senders in 30 days',
  CONSIGNEE_MANY_SENDERS: 'The receiver gets parcels from many unrelated senders',
  WEIGHT_UNUSUAL: "Weight far above the account's usual parcels",
  DIMS_UNUSUAL: "Size far above the account's usual parcels",
  BURST_LAST_24H: 'A burst of bookings in the last 24 hours',
  NEW_ACCOUNT: 'The account is less than 30 days old',
  UNUSUAL_HOUR: 'Booked at an hour this account rarely uses',
  LINK_TO_CONFIRMED_FRAUD: 'Linked to a confirmed fraud case',
  DROP_ADDRESS_PATTERN: 'The receiver looks like a reshipping drop',
  UNDER_DECLARED_PARCEL: "Declared weight and size far below the account's usual parcels",
  ACCOUNT_FAILED_DEPOT_SCAN: 'A parcel from this account failed a depot weight check',
  NEW_DESTINATION_STATE: 'First parcel from this account to the destination state',
  CONSIGNEE_MANY_UNRELATED_PAYERS: 'The consignee receives parcels paid for by many unrelated accounts',
  CONSIGNEE_ADDRESS_RECENT: 'The consignee address is recent',
  DENSITY_IMPLAUSIBLE: 'Declared weight is implausibly low for the package size',
}
export function reasonText(code: string): string {
  if (REASONS[code]) return REASONS[code]
  const s = code.toLowerCase().replace(/_/g, ' ')
  return s.charAt(0).toUpperCase() + s.slice(1)
}

export function featureLabel(name: string): string {
  const s = name.replace(/_/g, ' ')
  return s.charAt(0).toUpperCase() + s.slice(1)
}

// Fixed categorical order (validated reference palette, dataviz skill). Color follows the typology.
export const TYPOLOGY_META: Record<string, { name: string; slot: number }> = {
  T1: { name: 'T1 Label-resale takeover', slot: 1 },
  T2: { name: 'T2 Bust-out new account', slot: 2 },
  T3: { name: 'T3 Reshipping drops', slot: 3 },
  T4: { name: 'T4 Cost-maximizing takeover', slot: 4 },
  T5: { name: 'T5 Test then burst', slot: 5 },
  T6: { name: 'T6 Weight or size manipulation', slot: 6 },
  T7: { name: 'T7 Third-party billing misuse', slot: 7 },
}
export const typologyName = (t: string) => TYPOLOGY_META[t]?.name ?? t
export const typologyColor = (t: string) => `var(--series-${TYPOLOGY_META[t]?.slot ?? 8})`
