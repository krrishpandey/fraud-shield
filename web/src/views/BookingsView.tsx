import { useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import type { VerifyFeedRow } from '../api/verify'
import { ErrorBox, Loading, PageTitle } from '../components/common'
import { Icon } from '../components/Icon'
import '../verify.css'

type Kind = 'allowed' | 'blocked' | 'call' | 'person'

/** What happened to a booking at booking time, in one of four plain outcomes. */
function kindOf(r: VerifyFeedRow): Kind {
  if (r.action === 'block') return 'blocked'
  if (r.action === 'hold' || r.action === 'owner_confirm') return 'call'
  if (r.action === 'review') return 'person'
  return 'allowed'
}

const KIND: Record<Kind, { word: string; say: string; tone: string }> = {
  allowed: { word: 'Allowed', say: 'Label printed. Looks normal.', tone: 'allow' },
  blocked: { word: 'Blocked', say: 'Stopped at once: a hard fraud signal.', tone: 'block' },
  call: { word: 'Held: owner call', say: 'No label yet. Tracd calls the owner.', tone: 'owner_confirm' },
  person: { word: 'Held: to a person', say: 'No label yet. An analyst checks it.', tone: 'review' },
}

const CALL_RESULT: Record<string, string> = {
  released: 'owner said yes, released',
  blocked: 'owner said no, stopped',
  analyst_review: 'waiting for a person',
}

function useFeed() {
  const [state, setState] = useState<{ data: VerifyFeedRow[] | null; error: string | null }>({ data: null, error: null })
  const alive = useRef(true)
  useEffect(() => {
    alive.current = true
    let t: number | undefined
    const tick = () =>
      api.verifyFeed(300).then(
        (data) => alive.current && setState({ data, error: null }),
        (e) => alive.current && setState((s) => ({ data: s.data, error: e instanceof Error ? e.message : String(e) })),
      ).finally(() => {
        if (alive.current) t = window.setTimeout(tick, 2000)
      })
    tick()
    return () => {
      alive.current = false
      window.clearTimeout(t)
    }
  }, [])
  return state
}

export default function BookingsView() {
  const { data, error } = useFeed()
  const [params, setParams] = useSearchParams()
  const [q, setQ] = useState('')
  const [page, setPage] = useState(0)
  const [perPage, setPerPage] = useState(20)
  const filter = (params.get('show') as Kind | null) ?? null
  const all = data ?? []
  const count = (k: Kind) => all.filter((r) => kindOf(r) === k).length
  const filtered = (filter ? all.filter((r) => kindOf(r) === filter) : all).filter((r) => {
    const t = q.trim().toLowerCase()
    return !t || `${r.decision_id} ${r.booking_id} ${r.company ?? ''} ${r.origin} ${r.dest}`.toLowerCase().includes(t)
  })
  const pages = Math.max(1, Math.ceil(filtered.length / perPage))
  const cur = Math.min(page, pages - 1)
  const shown = filtered.slice(cur * perPage, cur * perPage + perPage)

  return (
    <div>
      <PageTitle title="Live bookings" />
      <div className="page">
        <div className="cv-sum">
          <span>Bookings <b className="tnum">{all.length}</b></span>
          {(['allowed', 'blocked', 'call', 'person'] as Kind[]).map((k) => (
            <span key={k}>
              {KIND[k].word} <b className="tnum">{count(k)}</b>
            </span>
          ))}
        </div>

        <div className="cv-bar">
          <label className="sr-only" htmlFor="bk-filter">Which bookings</label>
          <select id="bk-filter" className="field cv-select" value={filter ?? ''} onChange={(e) => { setPage(0); setParams(e.target.value ? { show: e.target.value } : {}) }}>
            <option value="">All bookings</option>
            {(['allowed', 'blocked', 'call', 'person'] as Kind[]).map((k) => (
              <option key={k} value={k}>{KIND[k].word}</option>
            ))}
          </select>
          <label className="cv-search">
            <Icon name="search" size={16} />
            <span className="sr-only">Search bookings</span>
            <input className="field" placeholder="Search booking ID, customer or city" value={q} onChange={(e) => { setQ(e.target.value); setPage(0) }} />
          </label>
          <span className="cv-grow" />
          <button type="button" className="btn" onClick={() => window.location.reload()}>
            <Icon name="refresh" size={16} /> Refresh
          </button>
        </div>

        {error && !data && <ErrorBox message={error} />}
        {!data && !error && <Loading what="bookings" />}
        {data && (
          <section className="panel min-w-0 cv-table">
            <div className="overflow-x-auto">
              <table className="tbl bk-tbl" data-testid="bookings-table">
                <thead>
                  <tr>
                    <th scope="col">Booking</th>
                    <th scope="col">From - to</th>
                    <th scope="col" className="r">Risk</th>
                    <th scope="col">What happened</th>
                  </tr>
                </thead>
                <tbody>
                  {shown.map((r) => {
                    const k = kindOf(r)
                    const to = r.verify_status ? `/calls?d=${encodeURIComponent(r.decision_id)}` : `/decisions/${encodeURIComponent(r.decision_id)}`
                    return (
                      <tr key={r.decision_id} data-kind={k}>
                        <td>
                          <Link to={to} className="row-link mono">{r.decision_id}</Link>
                          {r.company && <span className="vf-sub">{r.company}</span>}
                        </td>
                        <td>
                          {r.origin} to {r.dest}
                          <span className="vf-sub">{r.weight_kg == null ? '' : `${r.weight_kg.toFixed(1)} kg ${r.service ?? ''}`}</span>
                        </td>
                        <td className="r tnum">{r.risk == null ? '' : `${(r.risk * 100).toFixed(1)}%`}</td>
                        <td>
                          <span className="vf-state" style={{ ['--act' as string]: `var(--act-${KIND[k].tone})` }}>
                            {KIND[k].word}
                          </span>
                          <span className="vf-sub">
                            {r.verify_status && CALL_RESULT[r.verify_status] ? `Call: ${CALL_RESULT[r.verify_status]}` : KIND[k].say}
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                  {shown.length === 0 && (
                    <tr>
                      <td colSpan={4} className="py-8 text-center text-muted">
                        No records found. <Link className="row-link" to="/">Try a booking</Link>
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            <div className="cv-pager">
              <span>Rows per page</span>
              <select className="field cv-select" value={perPage} onChange={(e) => { setPerPage(Number(e.target.value)); setPage(0) }}>
                {[10, 20, 50].map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
              <span className="tnum">Page {cur + 1} of {pages}</span>
              <button type="button" className="btn" disabled={cur === 0} onClick={() => setPage(cur - 1)} aria-label="Previous page">‹</button>
              <button type="button" className="btn" disabled={cur >= pages - 1} onClick={() => setPage(cur + 1)} aria-label="Next page">›</button>
            </div>
          </section>
        )}
      </div>
    </div>
  )
}
