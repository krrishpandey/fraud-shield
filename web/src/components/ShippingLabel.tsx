import type { Action, Booking } from '../api/types'
import { fmtBRL } from '../lib/format'

/*
  The decision is about one thing: does this label print? So the page leads with the label itself,
  drawn from the booking. Allow prints it, scan-gated prints it with a depot check, every other
  action leaves it unprinted and says who it is waiting for.
*/

const WAITING: Partial<Record<Action, string>> = {
  owner_confirm: 'Waiting for the account owner to confirm',
  review: 'Waiting for an analyst',
  hold: 'Waiting for verification',
  block: 'Label creation locked for this booking',
}

const short = (id: string, n = 8) => (id.length > n ? id.slice(0, n) : id)

/** Deterministic bar pattern from the booking id. It identifies the label visually; it is not a scannable code. */
function bars(id: string): { x: number; w: number }[] {
  const out: { x: number; w: number }[] = []
  let x = 0
  for (const ch of id) {
    const c = ch.charCodeAt(0)
    for (let k = 0; k < 3; k++) {
      const w = 1 + ((c >> (k * 2)) & 3)
      out.push({ x, w })
      x += w + 1 + ((c >> (k + 1)) & 1)
    }
  }
  return out
}

function Barcode({ id }: { id: string }) {
  const b = bars(id)
  const width = b.length ? b[b.length - 1].x + b[b.length - 1].w : 1
  return (
    <svg className="label-barcode" viewBox={`0 0 ${width} 40`} preserveAspectRatio="none" aria-hidden="true">
      {b.map((r, i) => (
        <rect key={i} x={r.x} y={0} width={r.w} height={40} />
      ))}
    </svg>
  )
}

export function ShippingLabel({ booking, action }: { booking: Booking; action: Action }) {
  const printed = action === 'allow' || action === 'allow_scan_gated'
  const ownGoods = booking.sender_id === booking.account_id
  const waiting = WAITING[action]
  return (
    <figure className="ship-label-wrap" data-testid="shipping-label" data-printed={printed} data-action={action}>
      <div className={`ship-label${printed ? '' : ' is-unprinted'}`}>
        <div className="label-print" aria-hidden={!printed}>
          <div className="label-row label-from">
            <div>
              <div className="label-key">From</div>
              <div className="label-val">
                {ownGoods ? `Account ${short(booking.account_id)}` : `Sender ${short(booking.sender_id)}`}
              </div>
              <div className="label-val">{`${booking.origin_uf} ${booking.origin_zip3}`}</div>
              {!ownGoods && <div className="label-paid">Paid by account {short(booking.account_id)}</div>}
            </div>
            <div className="label-service" title={booking.service}>
              {booking.service === 'express' ? 'E' : 'S'}
            </div>
          </div>
          <div className="label-row label-to">
            <div className="label-key">Ship to</div>
            <div className="label-val">Receiver {short(booking.consignee_id)}</div>
            <div className="label-route">{`${booking.dest_uf} ${booking.dest_zip3}`}</div>
          </div>
          <div className="label-row label-facts">
            <span>{booking.weight_kg} kg</span>
            <span>{`${booking.length_cm}x${booking.width_cm}x${booking.height_cm} cm`}</span>
            <span>{booking.booked_at.replace('T', ' ').slice(0, 16)}</span>
            <span className="label-cost">{fmtBRL(booking.carrier_cost)}</span>
          </div>
          <div className="label-row label-code">
            <Barcode id={booking.booking_id} />
            <div className="label-id">{booking.booking_id}</div>
          </div>
          {action === 'allow_scan_gated' && (
            <div className="label-row label-gate" data-testid="label-scan-gate">
              <strong>Check at first scan.</strong> Weigh and measure at the depot before it travels.
            </div>
          )}
        </div>
        {!printed && (
          <div className="label-stamp" role="status">
            <span className="label-stamp-main">{action === 'block' ? 'Refused' : 'Not printed'}</span>
            <span className="label-stamp-sub">{waiting}</span>
          </div>
        )}
      </div>
      <figcaption className="label-caption">
        {printed
          ? action === 'allow'
            ? 'Label printed. The parcel ships normally.'
            : 'Label printed, with a weight and size check at the first depot scan.'
          : 'This is the label the booking asked for. It has not been printed.'}
      </figcaption>
    </figure>
  )
}
