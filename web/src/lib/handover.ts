import type { Handover } from '../api/types'

/** "14:02" today, "3 Oct 14:02" on another day; the raw string if it is not a timestamp. */
export function fmtSince(iso: string | null | undefined): string {
  if (!iso) return 'n/a'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  const hm = iso.slice(11, 16)
  const today = new Date()
  if (d.toDateString() === today.toDateString()) return hm
  return `${d.getDate()} ${d.toLocaleString('en-GB', { month: 'short' })} ${hm}`
}

const pts = (a: number, b: number) => `${(b - a) * 100 >= 0 ? '+' : ''}${((b - a) * 100).toFixed(1)} points`

/** The plain-words headline for a handover; the same sentence the banner shows. */
export function handoverHeadline(h: Handover): string {
  const since = fmtSince(h.active_since)
  if (h.verdict === 'new_model_in_use')
    return `From now on, new bookings are scored by ${h.active_version} (since ${since}). Previous model ${h.previous_version} kept for rollback.`
  if (h.verdict === 'rolled_back')
    return `Rolled back. From now on, new bookings are scored by ${h.active_version} (since ${since}). ${h.previous_version} is no longer used for new bookings.`
  const np = h.new_pattern
  const learned = np && np.candidate > np.current ? ` learned the new pattern (${pts(np.current, np.candidate)} recall on new patterns) but` : ''
  return `Still using ${h.active_version} (since ${since}). Candidate ${h.candidate_version}${learned} was not deployed because:`
}
