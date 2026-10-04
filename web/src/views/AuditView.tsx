import { useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { ActionPill } from '../components/ActionBadge'
import { ErrorBox, PageTitle, Section } from '../components/common'
import { Icon } from '../components/Icon'
import { fmtDateTime, fmtInt } from '../lib/format'
import { useStatus } from '../lib/status'
import { useAsync } from '../lib/useAsync'

export default function AuditView() {
  const { data, error, loading, reload } = useAsync(() => api.auditVerify(), [])
  const { recent } = useStatus()
  const [q, setQ] = useState('')
  const state = loading ? 'checking' : error ? 'error' : data?.ok ? 'ok' : 'broken'
  const term = q.trim().toLowerCase()
  const rows = recent.filter((r) => !term || r.decision_id.toLowerCase().includes(term) || r.account_id.toLowerCase().includes(term)).slice(0, 12)

  return (
    <div>
      <PageTitle title="Audit log">
        <button type="button" className="btn btn-primary" onClick={reload} disabled={loading} data-testid="audit-verify-button">
          <Icon name="shieldCheck" size={17} />
          {loading ? 'Verifying...' : 'Verify chain'}
        </button>
      </PageTitle>
      <div className="page">
        {error && <ErrorBox message={error} onRetry={reload} />}

        <section className={`panel audit-hero is-${state}`} data-testid="audit-status" data-state={state} data-ok={data ? String(data.ok) : ''} role="status" aria-live="polite">
          <Icon name={state === 'broken' || state === 'error' ? 'link' : 'shieldCheck'} size={40} stroke={2} className="audit-ico" />
          <div className="min-w-0 flex-1">
            {loading ? (
              <>
                <h2>Recomputing the chain…</h2>
                <p>Every record's hash is recomputed from the first record onwards.</p>
              </>
            ) : data ? (
              data.ok ? (
                <>
                  <h2>Chain intact</h2>
                  <p>
                    All <b>{fmtInt(data.records)}</b> records verify. No decision, explanation or analyst label has been changed after it was
                    written.
                  </p>
                </>
              ) : (
                <>
                  <h2>Chain broken at record {data.first_bad_index ?? 'unknown'}</h2>
                  <p>That record, or one before it, was changed after it was written. Every record after it can't be trusted until it is fixed.</p>
                </>
              )
            ) : (
              <h2>Verification failed</h2>
            )}
          </div>
        </section>

        <div className="audit-grid">
          <section className="panel min-w-0" aria-labelledby="rec-h">
            <div className="card-h">
              <h2 id="rec-h">
                <Icon name="lines" size={18} />
                Latest decisions in the log
              </h2>
              <label className="search ml-auto w-[260px]">
                <Icon name="search" size={16} />
                <span className="sr-only">Find a decision</span>
                <input className="field" placeholder="Decision or account id" value={q} onChange={(e) => setQ(e.target.value)} />
              </label>
            </div>
            <div className="mt-2 overflow-x-auto">
              <table className="tbl">
                <thead>
                  <tr>
                    <th scope="col">ID</th>
                    <th scope="col">Type</th>
                    <th scope="col">Value</th>
                    <th scope="col">Analyst label</th>
                    <th scope="col">Booked</th>
                    <th scope="col">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.decision_id}>
                      <td>
                        <Link to={`/decisions/${encodeURIComponent(r.decision_id)}`} className="row-link mono font-semibold">
                          {r.decision_id}
                        </Link>
                      </td>
                      <td className="text-ink-2">Decision</td>
                      <td>
                        <ActionPill action={r.action} />
                      </td>
                      <td>
                        {r.analyst_label ? (
                          <span className={r.analyst_label === 'fraud' ? 'text-danger font-semibold' : 'text-ok'}>{r.analyst_label === 'fraud' ? 'fraud' : 'legitimate'}</span>
                        ) : (
                          <span className="text-muted">—</span>
                        )}
                      </td>
                      <td className="tnum text-ink-2">{fmtDateTime(r.booked_at)}</td>
                      <td>
                        {state === 'ok' ? (
                          <span className="text-ok">✓ Verified</span>
                        ) : state === 'broken' ? (
                          <span className="font-semibold text-danger">! Check the chain</span>
                        ) : (
                          <span className="text-muted">…</span>
                        )}
                      </td>
                    </tr>
                  ))}
                  {rows.length === 0 && (
                    <tr>
                      <td colSpan={6} className="py-8 text-center text-muted">
                        {recent.length === 0 ? 'No decisions yet. Every booking you score is appended here.' : 'No decision matches that id.'}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>

          <div className="flex min-w-0 flex-col gap-[18px]">
            <Section title="Chain head" icon="lock">
              {data && (
                <>
                  <div className="head-hash mono" data-testid="audit-head-hash">
                    {data.head_hash || 'none (empty log)'}
                  </div>
                  <dl className="kv mt-2">
                    <dt>Records checked</dt>
                    <dd className="tnum" data-testid="audit-records">
                      {fmtInt(data.records)}
                    </dd>
                    {data.first_bad_index != null && (
                      <>
                        <dt>First bad record</dt>
                        <dd className="tnum text-danger" data-testid="audit-first-bad">
                          {data.first_bad_index}
                        </dd>
                      </>
                    )}
                    <dt>Anchored</dt>
                    <dd>Daily, off-system</dd>
                    <dt>Stores</dt>
                    <dd>Decision · reason · label</dd>
                  </dl>
                </>
              )}
              {!data && !error && <p className="text-muted">Verifying…</p>}
            </Section>
            <Section title="How it works" icon="info">
              <p className="text-[14.5px] leading-relaxed text-ink-2">
                Each record's hash includes the one before it. Change any old record and every hash after it stops matching. The head hash is
                anchored outside the system once a day, so a matching head plus a clean chain means nothing was edited after the fact.
              </p>
            </Section>
          </div>
        </div>
      </div>
    </div>
  )
}
