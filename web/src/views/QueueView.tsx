import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import { ACTIONS, type DecisionSummary } from '../api/types'
import { ActionPill } from '../components/ActionBadge'
import { ErrorBox, Loading, PageTitle } from '../components/common'
import { actionShort } from '../lib/domain'
import { fmtBRL, fmtDateTime, fmtPct } from '../lib/format'
import { useAsync } from '../lib/useAsync'

type SortKey = 'newest' | 'misuse' | 'carrier_cost'

export default function QueueView() {
  const nav = useNavigate()
  const [action, setAction] = useState('')
  const [sort, setSort] = useState<SortKey>('misuse')
  const { data, error, loading, reload } = useAsync(() => api.decisions({ limit: 200, action: action || undefined }), [action])

  const rows = useMemo(() => {
    const r: DecisionSummary[] = [...(data ?? [])]
    if (sort === 'misuse') r.sort((a, b) => b.misuse - a.misuse)
    else if (sort === 'carrier_cost') r.sort((a, b) => b.carrier_cost - a.carrier_cost)
    return r
  }, [data, sort])

  return (
    <div>
      <PageTitle title="Review queue" sub="Scored bookings, newest first from the API. Filter by action and sort by risk or by what the carrier stands to lose." />
      <div className="mb-3 flex flex-wrap items-end gap-3">
        <div role="group" aria-label="Filter by action" className="flex flex-wrap gap-1" data-testid="queue-filter">
          {['', ...ACTIONS].map((a) => (
            <button
              key={a || 'all'}
              type="button"
              aria-pressed={action === a}
              data-testid={`queue-filter-${a || 'all'}`}
              onClick={() => setAction(a)}
              className={`rounded-sm border px-2.5 py-1 text-[0.8rem] font-semibold ${
                action === a ? 'border-brand bg-brand text-brand-ink' : 'border-rule bg-surface text-ink-2 hover:text-ink'
              }`}
            >
              {a ? actionShort(a) : 'All actions'}
            </button>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-2">
          <label htmlFor="queue-sort" className="text-[0.8rem] text-muted">
            Sort by
          </label>
          <select id="queue-sort" data-testid="queue-sort" className="field w-auto" value={sort} onChange={(e) => setSort(e.target.value as SortKey)}>
            <option value="misuse">Misuse probability, highest first</option>
            <option value="carrier_cost">Carrier cost, highest first</option>
            <option value="newest">Newest first</option>
          </select>
          <button type="button" className="btn" onClick={reload} data-testid="queue-refresh">
            Refresh
          </button>
        </div>
      </div>

      {error && <ErrorBox message={error} onRetry={reload} />}
      {loading && !data && <Loading what="decisions" />}
      {data && (
        <div className="panel overflow-x-auto">
          <table className="w-full text-[0.85rem]" data-testid="queue-table">
            <caption className="sr-only">Scored decisions</caption>
            <thead className="sticky top-0 bg-surface">
              <tr className="border-b border-rule text-left text-[0.75rem] text-muted">
                <th scope="col" className="px-3 py-2 font-semibold">Decision</th>
                <th scope="col" className="px-3 py-2 font-semibold">Booked</th>
                <th scope="col" className="px-3 py-2 font-semibold">Account</th>
                <th scope="col" className="px-3 py-2 font-semibold">Action</th>
                <th scope="col" className="px-3 py-2 text-right font-semibold" aria-sort={sort === 'misuse' ? 'descending' : 'none'}>
                  Misuse
                </th>
                <th scope="col" className="px-3 py-2 text-right font-semibold" aria-sort={sort === 'carrier_cost' ? 'descending' : 'none'}>
                  Carrier cost
                </th>
                <th scope="col" className="px-3 py-2 font-semibold">Analyst</th>
                <th scope="col" className="px-3 py-2 font-semibold">Scenario</th>
              </tr>
            </thead>
            <tbody>
              {rows.length === 0 && (
                <tr>
                  <td colSpan={8} className="px-3 py-6 text-center text-muted">
                    No decisions{action ? ` with action ${actionShort(action)}` : ''} yet. Score a booking to fill the queue.
                  </td>
                </tr>
              )}
              {rows.map((r) => (
                <tr
                  key={r.decision_id}
                  data-testid="queue-row"
                  data-decision-id={r.decision_id}
                  data-action={r.action}
                  data-misuse={r.misuse}
                  data-carrier-cost={r.carrier_cost}
                  className="cursor-pointer border-b border-rule last:border-0 hover:bg-surface-2"
                  onClick={() => nav(`/decisions/${encodeURIComponent(r.decision_id)}`)}
                >
                  <td className="px-3 py-1.5">
                    <Link to={`/decisions/${encodeURIComponent(r.decision_id)}`} className="font-mono text-[0.8rem] underline underline-offset-2" onClick={(e) => e.stopPropagation()}>
                      {r.decision_id}
                    </Link>
                    <div className="text-[0.72rem] text-muted">{r.booking_id}</div>
                  </td>
                  <td className="tnum px-3 py-1.5 whitespace-nowrap">{fmtDateTime(r.booked_at)}</td>
                  <td className="px-3 py-1.5">{r.account_id}</td>
                  <td className="px-3 py-1.5">
                    <ActionPill action={r.action} />
                  </td>
                  <td className="tnum px-3 py-1.5 text-right">{fmtPct(r.misuse, 1)}</td>
                  <td className="tnum px-3 py-1.5 text-right">{fmtBRL(r.carrier_cost)}</td>
                  <td className="px-3 py-1.5">{r.analyst_label === 'fraud' ? 'Fraud' : r.analyst_label === 'legit' ? 'Legitimate' : <span className="text-muted">Open</span>}</td>
                  <td className="px-3 py-1.5 text-ink-2">{r.scenario ?? ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
