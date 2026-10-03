import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { Booking, DemoBooking } from '../api/types'
import { ActionPill } from '../components/ActionBadge'
import { ErrorBox, Loading, PageTitle, Section } from '../components/common'
import { fmtBRL } from '../lib/format'
import { useAsync } from '../lib/useAsync'

function Facts({ b }: { b: Booking }) {
  const own = b.sender_id === b.account_id
  const rows: [string, string][] = [
    ['Account', b.account_id],
    ['Sender', own ? `${b.sender_id} (the account itself)` : b.sender_id],
    ['Route', `${b.origin_uf}-${b.origin_zip3} to ${b.dest_uf}-${b.dest_zip3}`],
    ['Booked', `${b.booked_at.replace('T', ' ').slice(0, 16)} via ${b.channel}`],
    ['Device age', `${b.login_device_age_days} days`],
    ['Parcel', `${b.weight_kg} kg, ${b.length_cm}x${b.width_cm}x${b.height_cm} cm, ${b.service}`],
    ['Category', b.category],
    ['Carrier cost', fmtBRL(b.carrier_cost)],
    ['Declared value', fmtBRL(b.declared_value)],
    ['Owner contact', b.owner_contact_age_days == null ? 'none on file' : `verified ${b.owner_contact_age_days} days ago`],
  ]
  return (
    <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-0.5 text-[0.85rem]">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-muted">{k}</dt>
          <dd className="tnum">{v}</dd>
        </div>
      ))}
    </dl>
  )
}

const EMPTY: Booking = {
  booking_id: '',
  account_id: 'acc_7c3e',
  booked_at: '2018-06-14T02:41',
  channel: 'web',
  login_device_age_days: 0,
  payment_method: 'account_billing',
  sender_id: 'acc_7c3e',
  origin_uf: 'SP',
  origin_zip3: '013',
  dest_uf: 'RJ',
  dest_zip3: '220',
  consignee_id: 'cns_0001',
  weight_kg: 1,
  length_cm: 30,
  width_cm: 20,
  height_cm: 10,
  service: 'standard',
  category: 'housewares',
  declared_value: 100,
  carrier_cost: 20,
  owner_contact_age_days: 365,
  meta: {},
}

type FieldDef = { k: keyof Booking; label: string; type?: 'number' | 'text' | 'datetime-local'; options?: string[] }
const FIELDS: FieldDef[] = [
  { k: 'account_id', label: 'Paying account' },
  { k: 'sender_id', label: 'Sender' },
  { k: 'booked_at', label: 'Booked at', type: 'datetime-local' },
  { k: 'channel', label: 'Channel', options: ['web', 'api', 'counter'] },
  { k: 'login_device_age_days', label: 'Device age (days)', type: 'number' },
  { k: 'payment_method', label: 'Payment', options: ['account_billing', 'card', 'ach'] },
  { k: 'origin_uf', label: 'Origin state' },
  { k: 'origin_zip3', label: 'Origin zip3' },
  { k: 'dest_uf', label: 'Destination state' },
  { k: 'dest_zip3', label: 'Destination zip3' },
  { k: 'consignee_id', label: 'Consignee' },
  { k: 'category', label: 'Category' },
  { k: 'weight_kg', label: 'Weight (kg)', type: 'number' },
  { k: 'length_cm', label: 'Length (cm)', type: 'number' },
  { k: 'width_cm', label: 'Width (cm)', type: 'number' },
  { k: 'height_cm', label: 'Height (cm)', type: 'number' },
  { k: 'service', label: 'Service', options: ['standard', 'express'] },
  { k: 'declared_value', label: 'Declared value (R$)', type: 'number' },
  { k: 'carrier_cost', label: 'Carrier cost (R$)', type: 'number' },
  { k: 'owner_contact_age_days', label: 'Owner contact age (days)', type: 'number' },
]

function ManualForm({ onScore, busy }: { onScore: (b: Booking) => void; busy: boolean }) {
  const [b, setB] = useState<Booking>(EMPTY)
  const set = (k: keyof Booking, v: string, numeric: boolean) =>
    setB((prev) => ({ ...prev, [k]: numeric ? (v === '' ? null : Number(v)) : v }))
  const submit = (e: FormEvent) => {
    e.preventDefault()
    const booked = b.booked_at.length === 16 ? `${b.booked_at}:00` : b.booked_at
    onScore({ ...b, booked_at: booked, booking_id: `manual-${Date.now().toString(36)}`, meta: { scenario: 'manual' } })
  }
  return (
    <form onSubmit={submit} data-testid="manual-form" aria-label="Manual booking">
      <div className="grid grid-cols-2 gap-x-3 gap-y-2 lg:grid-cols-4">
        {FIELDS.map((f) => {
          const id = `mf-${f.k}`
          const val = b[f.k]
          return (
            <div key={f.k}>
              <label htmlFor={id} className="block text-[0.75rem] text-muted">
                {f.label}
              </label>
              {f.options ? (
                <select id={id} className="field" value={String(val)} onChange={(e) => set(f.k, e.target.value, false)}>
                  {f.options.map((o) => (
                    <option key={o}>{o}</option>
                  ))}
                </select>
              ) : (
                <input
                  id={id}
                  data-testid={`manual-${f.k}`}
                  className="field tnum"
                  type={f.type ?? 'text'}
                  step={f.type === 'number' ? 'any' : undefined}
                  required={f.k !== 'owner_contact_age_days'}
                  value={val == null ? '' : String(val)}
                  onChange={(e) => set(f.k, e.target.value, f.type === 'number')}
                />
              )}
            </div>
          )
        })}
      </div>
      <div className="mt-3 flex items-center gap-3">
        <button type="submit" className="btn btn-primary" disabled={busy} data-testid="manual-score-button">
          {busy ? 'Scoring...' : 'Score this booking'}
        </button>
        <span className="text-[0.8rem] text-muted">Leave owner contact empty if no verified contact is on file.</span>
      </div>
    </form>
  )
}

export default function ScoreView() {
  const nav = useNavigate()
  const demos = useAsync(() => api.demoBookings(), [])
  const [selected, setSelected] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [manual, setManual] = useState(false)

  const items: DemoBooking[] = demos.data ?? []
  const current = items.find((d) => d.scenario === selected) ?? items[0]

  async function score(b: Booking) {
    setBusy(true)
    setErr(null)
    try {
      const res = await api.score(b)
      nav(`/decisions/${encodeURIComponent(res.decision_id)}`, { state: { justScored: true } })
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <PageTitle
        title="Score a booking"
        sub="Pick a scripted booking and score it the way the booking system would, before a label is issued."
      />
      {err && (
        <div className="mb-3">
          <ErrorBox message={err} testId="score-error" />
        </div>
      )}
      <div className="grid gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
        <Section title="Demo bookings" aside={`${items.length} scenarios`}>
          {demos.loading && <Loading what="demo bookings" />}
          {demos.error && <ErrorBox message={demos.error} onRetry={demos.reload} />}
          <ul className="flex flex-col gap-1.5" role="listbox" aria-label="Demo bookings">
            {items.map((d) => {
              const active = current?.scenario === d.scenario
              return (
                <li key={d.scenario}>
                  <button
                    type="button"
                    role="option"
                    aria-selected={active}
                    data-testid={`demo-booking-${d.scenario}`}
                    onClick={() => setSelected(d.scenario)}
                    onDoubleClick={() => score(d.booking)}
                    className={`w-full rounded-sm border px-3 py-2 text-left ${
                      active ? 'border-rule-strong bg-surface-2' : 'border-rule hover:bg-surface-2'
                    }`}
                    style={active ? { boxShadow: 'inset 3px 0 0 var(--brand)' } : undefined}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-semibold">{d.title}</span>
                      <span className="tnum text-[0.8rem] text-muted">{fmtBRL(d.booking.carrier_cost)}</span>
                    </div>
                    <div className="mt-0.5 text-[0.8rem] text-ink-2">
                      {d.booking.account_id}, {d.booking.origin_uf} to {d.booking.dest_uf}, {d.booking.booking_id}
                    </div>
                  </button>
                </li>
              )
            })}
          </ul>
          <p className="mt-2 text-[0.75rem] text-muted">Double-click a booking to score it straight away.</p>
        </Section>

        <Section
          title={current ? current.title : 'Booking'}
          aside={current?.expected ? <span className="flex items-center gap-1.5">Script expects <ActionPill action={current.expected} /></span> : undefined}
          testId="selected-booking"
        >
          {current ? (
            <>
              {current.description && <p className="mb-3 max-w-[70ch] text-[0.9rem]">{current.description}</p>}
              <Facts b={current.booking} />
              <div className="mt-4 flex items-center gap-3">
                <button
                  type="button"
                  className="btn btn-primary"
                  data-testid="score-button"
                  disabled={busy}
                  onClick={() => score(current.booking)}
                >
                  {busy ? 'Scoring...' : 'Score'}
                </button>
                <span className="text-[0.8rem] text-muted">Sends POST /score. The same booking id returns the stored decision.</span>
              </div>
            </>
          ) : (
            !demos.loading && <p className="text-muted">No demo bookings available. Use the manual form below.</p>
          )}
        </Section>
      </div>

      <div className="mt-4">
        <Section
          title="Manual booking"
          aside={
            <button type="button" className="btn" aria-expanded={manual} onClick={() => setManual((m) => !m)} data-testid="manual-toggle">
              {manual ? 'Hide form' : 'Show form'}
            </button>
          }
        >
          {manual ? (
            <ManualForm onScore={score} busy={busy} />
          ) : (
            <p className="text-[0.85rem] text-muted">Enter any booking by hand and score it.</p>
          )}
        </Section>
      </div>
    </div>
  )
}
