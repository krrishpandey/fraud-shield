import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import { ACTIONS, type DecisionDetail, type DecisionSummary } from '../api/types'
import { ActionPill } from '../components/ActionBadge'
import { ErrorBox, Loading, PageTitle } from '../components/common'
import { SlideInd } from '../components/SlideInd'
import { Icon, type IconName } from '../components/Icon'
import { ACTION_META, actionShort, reasonText } from '../lib/domain'
import { fmtDateTime } from '../lib/format'
import { useStatus } from '../lib/status'
import { useAsync } from '../lib/useAsync'

type SortKey = 'newest' | 'misuse' | 'carrier_cost'
type Tab = 'open' | 'labelled' | 'all'

const LOOK: Record<string, { tone: string; icon: IconName }> = {
  allow: { tone: 'green', icon: 'check' },
  allow_scan_gated: { tone: 'teal', icon: 'scale' },
  owner_confirm: { tone: 'orange', icon: 'user' },
  review: { tone: 'indigo', icon: 'eye' },
  hold: { tone: 'indigo', icon: 'pause' },
  block: { tone: 'rose', icon: 'x' },
}
const isOpen = (r: DecisionSummary) => r.action !== 'allow' && r.analyst_label == null
const brl = (v: number) => v.toFixed(2).replace('.', ',')

function Drawer({ id, onClose, onLabelled }: { id: string; onClose: () => void; onLabelled: () => void }) {
  const nav = useNavigate()
  const { data: d, error, reload } = useAsync<DecisionDetail>(() => api.decision(id), [id])
  const [busy, setBusy] = useState<null | 'fraud' | 'legit'>(null)
  const [err, setErr] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const label = async (l: 'fraud' | 'legit') => {
    setBusy(l)
    setErr(null)
    try {
      await api.analyst(id, { label: l, note: 'Labelled from the review queue' })
      reload()
      onLabelled()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }
  const b = d?.booking
  const look = d ? (LOOK[d.action] ?? LOOK.review) : LOOK.review
  return (
    <aside className="drawer" aria-label={`Case ${id}`} data-testid="queue-drawer">
      <div className="drawer-h">
        <span className="mono">{id}</span>
        <button
          type="button"
          className="ph-icon"
          aria-label="Copy the decision id"
          title={copied ? 'Copied' : 'Copy the decision id'}
          onClick={() => {
            navigator.clipboard?.writeText(id).then(
              () => setCopied(true),
              () => setCopied(false),
            )
          }}
        >
          <Icon name={copied ? 'check' : 'copy'} size={17} />
        </button>
        <button type="button" className="ph-icon" aria-label="Close the case" onClick={onClose}>
          <Icon name="x" size={17} />
        </button>
      </div>
      <div className="drawer-b">
        {error && <ErrorBox message={error} onRetry={reload} />}
        {!d && !error && <Loading what="the case" />}
        {d && b && (
          <>
            <div className="drawer-verdict" style={{ ['--act' as string]: `var(--act-${d.action})` }}>
              <Icon name={look.icon} size={34} stroke={2.2} className="verdict-ico" />
              <div className="min-w-0">
                <h2>{ACTION_META[d.action]?.label ?? d.action}</h2>
                <p className="text-muted">
                  Fraud P {(d.probabilities.misuse ?? d.gbm_score ?? 0).toFixed(2)} · booked {fmtDateTime(b.booked_at)}
                </p>
              </div>
            </div>
            <div className="drawer-route">
              <Icon name="route" size={17} />
              <b>
                {b.origin_uf} {b.origin_zip3}
              </b>
              <span className="text-muted">→</span>
              <b>
                {b.dest_uf} {b.dest_zip3}
              </b>
              <span className="ml-auto mono text-muted">R$ {brl(b.carrier_cost)}</span>
            </div>
            <dl className="kv drawer-kv">
              <dt>Whose goods</dt>
              <dd className={b.sender_id !== b.account_id ? 'is-bad' : ''}>{b.sender_id === b.account_id ? 'Its own' : 'Someone else’s'}</dd>
              <dt>Device age</dt>
              <dd className={b.login_device_age_days < 1 ? 'is-bad' : ''}>{Math.round(b.login_device_age_days)} days</dd>
              <dt>Owner contact</dt>
              <dd className={b.owner_contact_age_days == null || b.owner_contact_age_days < 30 ? 'is-bad' : ''}>
                {b.owner_contact_age_days == null ? 'None on file' : `${Math.round(b.owner_contact_age_days)} days old`}
              </dd>
              <dt>Channel</dt>
              <dd>{b.channel}</dd>
              <dt>Model</dt>
              <dd>{d.degraded ? 'LightGBM · backup' : d.decider ?? 'Laya'}</dd>
            </dl>
            {d.reasons.length > 0 && (
              <ul className="drawer-reasons">
                {d.reasons.slice(0, 3).map((r) => (
                  <li key={r}>{reasonText(r)}</li>
                ))}
              </ul>
            )}
            {d.analyst && (
              <p className={`drawer-done ${d.analyst.label === 'fraud' ? 'text-danger' : 'text-ok'}`}>
                {d.analyst.label === 'fraud' ? '✕ Labelled fraud' : '✓ Labelled legitimate'} · {d.analyst.at.replace('T', ' ').slice(0, 16)}
              </p>
            )}
            {err && (
              <p className="text-danger" role="alert">
                {err}
              </p>
            )}
            <button type="button" className="btn w-full" onClick={() => nav(`/decisions/${encodeURIComponent(id)}`)}>
              Open full decision
            </button>
          </>
        )}
      </div>
      <div className="drawer-foot">
        <button type="button" className="btn flex-1" disabled={!d || busy !== null} onClick={() => label('legit')} data-testid="queue-legit">
          <Icon name="check" size={17} />
          {busy === 'legit' ? 'Saving...' : 'Legit'}
        </button>
        <button type="button" className="btn btn-danger flex-1" disabled={!d || busy !== null} onClick={() => label('fraud')} data-testid="queue-fraud">
          <Icon name="flag" size={17} />
          {busy === 'fraud' ? 'Saving...' : 'Fraud'}
        </button>
      </div>
    </aside>
  )
}

export default function QueueView() {
  const nav = useNavigate()
  const { refresh } = useStatus()
  const [action, setAction] = useState('')
  const [sort, setSort] = useState<SortKey>('misuse')
  const [tab, setTab] = useState<Tab>('all')
  const [q, setQ] = useState('')
  const [picked, setSel] = useState<string | null>(null)
  const { data, error, loading, reload } = useAsync(() => api.decisions({ limit: 200, action: action || undefined }), [action])
  const learning = useAsync(() => api.learningStatus(), [])

  const all = useMemo(() => data ?? [], [data])
  const counts = { open: all.filter(isOpen).length, labelled: all.filter((r) => r.analyst_label != null).length, all: all.length }
  const rows = useMemo(() => {
    const t = q.trim().toLowerCase()
    const r = all.filter(
      (x) =>
        (tab === 'all' || (tab === 'open' ? isOpen(x) : x.analyst_label != null)) &&
        (!t || x.decision_id.toLowerCase().includes(t) || x.account_id.toLowerCase().includes(t) || x.booking_id.toLowerCase().includes(t)),
    )
    if (sort === 'misuse') r.sort((a, b) => b.misuse - a.misuse)
    else if (sort === 'carrier_cost') r.sort((a, b) => b.carrier_cost - a.carrier_cost)
    return r
  }, [all, sort, tab, q])

  // the selected case, if it is still in the filtered list (a filtered-out selection is not shown)
  const sel = picked && rows.some((r) => r.decision_id === picked) ? picked : null

  // J / K move through the list; Enter opens the full decision.
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT')) return
      if (e.ctrlKey || e.metaKey || e.altKey || rows.length === 0) return
      const i = rows.findIndex((r) => r.decision_id === sel)
      if (e.key === 'j') setSel(rows[Math.min(i + 1, rows.length - 1)].decision_id)
      if (e.key === 'k') setSel(rows[Math.max(i - 1, 0)].decision_id)
      if (e.key === 'Enter' && sel) nav(`/decisions/${encodeURIComponent(sel)}`)
    }
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [rows, sel, nav])

  return (
    <div>
      <PageTitle title="Review queue" />
      <div className={`queue${sel ? ' has-drawer' : ''}`}>
        <div className="queue-main">
          <div className="queue-tools">
            <div className="seg" role="radiogroup" aria-label="Which cases">
              <SlideInd />
              {(
                [
                  ['open', 'Open'],
                  ['labelled', 'Labelled'],
                  ['all', 'All'],
                ] as [Tab, string][]
              ).map(([k, l]) => (
                <button key={k} type="button" role="radio" aria-checked={tab === k} onClick={() => setTab(k)} data-testid={`queue-tab-${k}`}>
                  {l} <span className="n">{counts[k]}</span>
                </button>
              ))}
            </div>
            <label className="search min-w-[200px] flex-1">
              <Icon name="search" size={16} />
              <span className="sr-only">Search cases</span>
              <input className="field" placeholder="Search cases" value={q} onChange={(e) => setQ(e.target.value)} />
            </label>
            <label className="sr-only" htmlFor="queue-filter">
              Action
            </label>
            <select id="queue-filter" data-testid="queue-filter" className="field w-auto" value={action} onChange={(e) => setAction(e.target.value)}>
              <option value="">Every action</option>
              {ACTIONS.map((a) => (
                <option key={a} value={a}>
                  {actionShort(a)}
                </option>
              ))}
            </select>
            <label className="sr-only" htmlFor="queue-sort">
              Sort by
            </label>
            <select id="queue-sort" data-testid="queue-sort" className="field w-auto" value={sort} onChange={(e) => setSort(e.target.value as SortKey)}>
              <option value="misuse">Riskiest first</option>
              <option value="carrier_cost">Most money at risk</option>
              <option value="newest">Newest first</option>
            </select>
            <button type="button" className="btn" onClick={reload} data-testid="queue-refresh">
              <Icon name="refresh" size={16} />
              Refresh
            </button>
          </div>

          {error && <ErrorBox message={error} onRetry={reload} />}
          {loading && !data && <Loading what="decisions" />}
          {data && (
            <div className="panel overflow-x-auto">
              <table className="tbl queue-tbl" data-testid="queue-table">
                <caption className="sr-only">Scored decisions</caption>
                <thead>
                  <tr>
                    <th scope="col">Decision</th>
                    <th scope="col">Account</th>
                    <th scope="col">Booked</th>
                    <th scope="col">Action</th>
                    <th scope="col" className="r" aria-sort={sort === 'misuse' ? 'descending' : 'none'}>
                      Fraud P
                    </th>
                    <th scope="col" className="r" aria-sort={sort === 'carrier_cost' ? 'descending' : 'none'}>
                      R$
                    </th>
                    <th scope="col">Label</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.length === 0 && (
                    <tr>
                      <td colSpan={7} className="py-10 text-center text-muted">
                        {all.length === 0 ? 'No decisions yet. Score a booking to fill the queue.' : 'No cases match these filters.'}
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
                      className={`click${sel === r.decision_id ? ' is-sel' : ''}`}
                      aria-selected={sel === r.decision_id}
                      onClick={() => setSel(r.decision_id)}
                      onDoubleClick={() => nav(`/decisions/${encodeURIComponent(r.decision_id)}`)}
                    >
                      <td>
                        <Link to={`/decisions/${encodeURIComponent(r.decision_id)}`} className="row-link mono font-semibold" onClick={(e) => e.stopPropagation()}>
                          {r.decision_id}
                        </Link>
                        {r.scenario && <div className="text-[12px] text-faint">{r.scenario}</div>}
                      </td>
                      <td className="mono text-muted">{r.account_id.slice(0, 8)}</td>
                      <td className="tnum whitespace-nowrap text-ink-2">{fmtDateTime(r.booked_at)}</td>
                      <td>
                        <ActionPill action={r.action} />
                      </td>
                      <td className="r tnum mono">{r.misuse.toFixed(2)}</td>
                      <td className="r tnum mono">{brl(r.carrier_cost)}</td>
                      <td>
                        {r.analyst_label === 'fraud' ? (
                          <b className="text-danger">Fraud</b>
                        ) : r.analyst_label === 'legit' ? (
                          <b className="text-ok">Legitimate</b>
                        ) : (
                          <span className="text-muted">{r.action === 'allow' ? '—' : 'Unlabelled'}</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="queue-foot">
                <span>
                  Showing {rows.length} of {counts[tab]} {tab === 'all' ? 'cases' : tab === 'open' ? 'open cases' : 'labelled cases'}
                </span>
              </div>
            </div>
          )}

          <div className="panel train-card">
            <span className="train-ico">
              <Icon name="refresh" size={20} />
            </span>
            <div className="min-w-0 flex-1">
              <b>Your labels train the next model</b>
              <p>
                {learning.data ? `${learning.data.labels_since_last_retrain} new labels since the last retrain. ` : ''}Every retrain must pass the
                safety gate first.
              </p>
            </div>
            <Link to="/learning" className="btn">
              Open Learning
            </Link>
          </div>
        </div>

        {sel && (
          <Drawer
            id={sel}
            onClose={() => setSel(null)}
            onLabelled={() => {
              reload()
              learning.reload()
              refresh()
            }}
          />
        )}
      </div>
    </div>
  )
}
