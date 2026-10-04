import { useMemo, useState, type FormEvent, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { Booking, DemoBooking } from '../api/types'
import { ErrorBox, Loading, PageTitle } from '../components/common'
import { SlideInd } from '../components/SlideInd'
import { Icon, type IconName } from '../components/Icon'
import { fmtBRL } from '../lib/format'
import { useStatus } from '../lib/status'
import { useAsync } from '../lib/useAsync'

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

function ManualForm({ onScore, busy, onCancel }: { onScore: (b: Booking) => void; busy: boolean; onCancel: () => void }) {
  const [b, setB] = useState<Booking>(EMPTY)
  const set = (k: keyof Booking, v: string, numeric: boolean) => setB((prev) => ({ ...prev, [k]: numeric ? (v === '' ? null : Number(v)) : v }))
  const submit = (e: FormEvent) => {
    e.preventDefault()
    const booked = b.booked_at.length === 16 ? `${b.booked_at}:00` : b.booked_at
    onScore({ ...b, booked_at: booked, booking_id: `manual-${Date.now().toString(36)}`, meta: { scenario: 'manual' } })
  }
  return (
    <form onSubmit={submit} data-testid="manual-form" aria-label="New booking" className="panel">
      <div className="card-h">
        <h2>
          <Icon name="plus" size={18} />
          New booking
        </h2>
        <span className="ml-auto text-[13.5px] text-muted">Leave owner contact empty if no verified contact is on file.</span>
      </div>
      <div className="card-b">
        <div className="grid grid-cols-2 gap-x-4 gap-y-3 xl:grid-cols-4">
          {FIELDS.map((f) => {
            const id = `mf-${f.k}`
            const val = b[f.k]
            return (
              <div key={f.k} className="min-w-0">
                <label htmlFor={id} className="fbox-k">
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
      </div>
      <div className="dock-bar">
        <button type="button" className="btn" onClick={onCancel}>
          Cancel
        </button>
        <button type="submit" className="btn btn-primary ml-auto" disabled={busy} data-testid="manual-score-button">
          <Icon name="decision" size={17} />
          {busy ? 'Scoring...' : 'Send through the gate'}
        </button>
      </div>
    </form>
  )
}

/** Splits "Label-resale account takeover (injected T1)" into a name and its tags. */
function parseTitle(title: string) {
  const m = title.match(/^(.*?)\s*\(([^)]*)\)\s*$/)
  return { name: m ? m[1] : title, tags: m ? m[2].split(',').map((t) => t.trim()) : [] }
}
/** A glyph for what each scripted booking is about. */
function glyphFor(d: DemoBooking): IconName {
  const k = `${d.scenario} ${d.title}`.toLowerCase()
  if (k.includes('takeover')) return 'takeover'
  if (k.includes('reshipping') || k.includes('drop')) return 'inbox'
  if (k.includes('weight')) return 'scale'
  if (k.includes('new state') || k.includes('new-state')) return 'pin'
  if (k.includes('tenured') || k.includes('usual')) return 'store'
  return 'box'
}

function kindOf(d: DemoBooking) {
  const { tags } = parseTitle(d.title)
  if (d.expected === 'legit') return d.scenario.includes('hard') ? 'hard negative' : 'real'
  return tags.join(' · ') || 'injected'
}

/** A read-only value laid out like the design's input boxes. */
function FBox({ k, children, warn, hint }: { k: string; children: ReactNode; warn?: boolean; hint?: string }) {
  return (
    <div className="min-w-0">
      <div className="fbox-k">{k}</div>
      <div className={`fbox${warn ? ' is-warn' : ''}`}>{children}</div>
      {hint && <div className={`fbox-hint${warn ? ' is-warn' : ''}`}>{hint}</div>}
    </div>
  )
}

type Tab = 'booking' | 'account' | 'parcel' | 'route'
const TABS: [Tab, string, IconName][] = [
  ['booking', 'Booking', 'box'],
  ['account', 'Account', 'user'],
  ['parcel', 'Parcel', 'stack'],
  ['route', 'Route', 'route'],
]

function Details({ b, tab }: { b: Booking; tab: Tab }) {
  const own = b.sender_id === b.account_id
  const dev = b.login_device_age_days
  if (tab === 'booking')
    return (
      <div className="fgrid">
        <FBox k="Booking ID">
          <span className="mono">{b.booking_id}</span>
        </FBox>
        <FBox k="Booked at">{b.booked_at.replace('T', ' · ').slice(0, 18)}</FBox>
        <FBox k="Channel">{b.channel}</FBox>
        <FBox k="Payment">{b.payment_method.replace('_', ' ')}</FBox>
        <FBox k="Declared value">{fmtBRL(b.declared_value)}</FBox>
        <FBox k="Carrier cost">{fmtBRL(b.carrier_cost)}</FBox>
      </div>
    )
  if (tab === 'account')
    return (
      <div className="fgrid">
        <FBox k="Paying account">
          <span className="mono">{b.account_id}</span>
        </FBox>
        <FBox k="Sender" warn={!own} hint={own ? 'The account ships its own goods' : 'Someone else’s goods'}>
          <span className="mono">{b.sender_id}</span>
        </FBox>
        <FBox k="Device age" warn={dev === 0} hint={dev === 0 ? 'First login from this device' : undefined}>
          {Math.round(dev)} days
        </FBox>
        <FBox k="Owner contact" warn={b.owner_contact_age_days == null || b.owner_contact_age_days < 30}>
          {b.owner_contact_age_days == null ? 'None on file' : `Verified ${Math.round(b.owner_contact_age_days)} days ago`}
        </FBox>
      </div>
    )
  if (tab === 'parcel')
    return (
      <div className="fgrid">
        <FBox k="Weight">{b.weight_kg} kg</FBox>
        <FBox k="Dimensions">
          {b.length_cm} × {b.width_cm} × {b.height_cm} cm
        </FBox>
        <FBox k="Service">{b.service}</FBox>
        <FBox k="Category">{b.category}</FBox>
      </div>
    )
  return (
    <div className="fgrid">
      <FBox k="Origin">
        {b.origin_uf} {b.origin_zip3}
      </FBox>
      <FBox k="Destination">
        {b.dest_uf} {b.dest_zip3}
      </FBox>
      <FBox k="Consignee">
        <span className="mono">{b.consignee_id}</span>
      </FBox>
    </div>
  )
}

export default function ScoreView() {
  const nav = useNavigate()
  const { refresh } = useStatus()
  const demos = useAsync(() => api.demoBookings(), [])
  const [selected, setSelected] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [manual, setManual] = useState(false)
  const [q, setQ] = useState('')
  const [tab, setTab] = useState<Tab>('booking')
  const [allDone, setAllDone] = useState<number | null>(null)

  const items: DemoBooking[] = useMemo(() => demos.data ?? [], [demos.data])
  const shown = items.filter((d) => {
    const t = q.trim().toLowerCase()
    return !t || d.title.toLowerCase().includes(t) || d.booking.account_id.includes(t) || `${d.booking.origin_uf} ${d.booking.dest_uf}`.toLowerCase().includes(t)
  })
  const current = items.find((d) => d.scenario === selected) ?? items[0]

  async function score(b: Booking) {
    setBusy(true)
    setErr(null)
    try {
      const res = await api.score(b)
      refresh()
      nav(`/decisions/${encodeURIComponent(res.decision_id)}`, { state: { justScored: true } })
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  async function scoreAll() {
    setBusy(true)
    setErr(null)
    setAllDone(0)
    try {
      for (const [i, d] of items.entries()) {
        await api.score(d.booking)
        setAllDone(i + 1)
      }
      refresh()
      nav('/queue')
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
      setAllDone(null)
    }
  }

  const cur = current ? parseTitle(current.title) : null
  const b = current?.booking
  const fraudExpected = current?.expected && current.expected !== 'legit'

  return (
    <div>
      <PageTitle title="Score a booking" />
      <div className="split">
        <aside className="split-list" aria-label="Demo bookings">
          <h2 className="split-h">Demo bookings</h2>
          <div className="flex gap-2">
            <button
              type="button"
              className="btn flex-1"
              aria-expanded={manual}
              onClick={() => setManual((m) => !m)}
              data-testid="manual-toggle"
            >
              <Icon name="plus" size={16} />
              {manual ? 'Back to demos' : 'New booking'}
            </button>
          </div>
          <label className="search">
            <Icon name="search" size={16} />
            <span className="sr-only">Search bookings</span>
            <input className="field" placeholder="Search bookings..." value={q} onChange={(e) => setQ(e.target.value)} />
          </label>
          {demos.loading && <Loading what="demo bookings" />}
          {demos.error && <ErrorBox message={demos.error} onRetry={demos.reload} />}
          <ul className="blist" role="listbox" aria-label="Demo bookings">
            {shown.map((d) => {
              const active = !manual && current?.scenario === d.scenario
              return (
                <li key={d.scenario}>
                  <button
                    type="button"
                    role="option"
                    aria-selected={active}
                    data-testid={`demo-booking-${d.scenario}`}
                    onClick={() => {
                      setSelected(d.scenario)
                      setManual(false)
                    }}
                    onDoubleClick={() => score(d.booking)}
                    className="bcard"
                  >
                    <span className="glyph" aria-hidden="true">
                      <Icon name={glyphFor(d)} size={17} />
                    </span>
                    <span className="min-w-0">
                      <span className="bcard-t">{parseTitle(d.title).name}</span>
                      <span className="bcard-s tnum">
                        {d.booking.origin_uf} {d.booking.origin_zip3} → {d.booking.dest_uf} {d.booking.dest_zip3} · {kindOf(d)}
                      </span>
                    </span>
                  </button>
                </li>
              )
            })}
            {!demos.loading && shown.length === 0 && <li className="px-1 py-3 text-[14px] text-muted">No booking matches “{q}”.</li>}
          </ul>
        </aside>

        <div className="split-main">
          {err && <ErrorBox message={err} testId="score-error" />}
          {manual ? (
            <ManualForm onScore={score} busy={busy} onCancel={() => setManual(false)} />
          ) : current && b && cur ? (
            <>
              <section className="panel hero" data-testid="selected-booking" aria-labelledby="hero-t">
                <div className="hero-top">
                  <span className="glyph glyph-lg" aria-hidden="true">
                    <Icon name={glyphFor(current)} size={30} stroke={1.6} />
                  </span>
                  <div className="min-w-0 flex-1">
                    <h2 id="hero-t" className="hero-t">
                      {cur.name}
                    </h2>
                    <p className="hero-ids tnum">
                      {b.booking_id} · acct {b.account_id.slice(0, 8)} · via {b.channel}
                    </p>
                  </div>
                </div>
                <div className="hero-meta">
                  <span>
                    <Icon name="dollar" size={16} /> Carrier cost: <b>{fmtBRL(b.carrier_cost)}</b>
                  </span>
                  {current.expected && (
                    <span className={`hero-exp${fraudExpected ? ' is-fraud' : ''}`}>
                      Script expects {fraudExpected ? `fraud (${current.expected})` : 'legit'}
                    </span>
                  )}
                  {cur.tags.map((t) => (
                    <span key={t} className={`tag${t.startsWith('injected') ? ' tag-inj' : ''}`}>
                      {t}
                    </span>
                  ))}
                  <span className="ml-auto tnum text-ink-2">
                    <Icon name="route" size={15} /> {b.origin_uf} {b.origin_zip3} → {b.dest_uf} {b.dest_zip3}
                  </span>
                </div>
                {current.description && <p className="hero-desc">{current.description}</p>}
              </section>

              <div className="itabs" role="tablist" aria-label="Booking details">

                <SlideInd />
                {TABS.map(([k, label, icon]) => (
                  <button key={k} type="button" role="tab" aria-selected={tab === k} onClick={() => setTab(k)}>
                    <Icon name={icon} size={18} />
                    {label}
                  </button>
                ))}
              </div>

              <section className="panel" aria-label={`${TABS.find((t) => t[0] === tab)?.[1]} details`}>
                <div className="card-h">
                  <h2>
                    <Icon name="lines" size={18} />
                    {TABS.find((t) => t[0] === tab)?.[1]} details
                  </h2>
                </div>
                <div className="card-b">
                  <Details b={b} tab={tab} />
                </div>
                <div className="dock-bar">
                  <button type="button" className="btn ml-auto" disabled={busy || items.length === 0} onClick={scoreAll}>
                    <Icon name="pulse" size={16} />
                    {allDone != null ? `Scoring ${allDone} of ${items.length}...` : `Score all ${items.length} demos`}
                  </button>
                  <button type="button" className="btn btn-primary" data-testid="score-button" disabled={busy} onClick={() => score(b)}>
                    <Icon name="decision" size={17} />
                    {busy && allDone == null ? 'Scoring...' : 'Send through the gate'}
                  </button>
                </div>
              </section>
            </>
          ) : (
            !demos.loading && <p className="text-muted">No demo bookings available. Use New booking.</p>
          )}
        </div>
      </div>
    </div>
  )
}
