import { api } from '../api/client'
import { ErrorBox, PageTitle, Section } from '../components/common'
import { fmtInt } from '../lib/format'
import { useAsync } from '../lib/useAsync'

export default function AuditView() {
  const { data, error, loading, reload } = useAsync(() => api.auditVerify(), [])
  const state = loading ? 'checking' : error ? 'error' : data?.ok ? 'ok' : 'broken'
  return (
    <div>
      <PageTitle
        title="Audit log"
        sub="Every decision, explanation and analyst label is appended to a hash chain. Verification recomputes every hash from the first record."
      />
      <Section
        title="Chain verification"
        aside={
          <button type="button" className="btn btn-primary" onClick={reload} disabled={loading} data-testid="audit-verify-button">
            {loading ? 'Verifying...' : 'Verify again'}
          </button>
        }
      >
        {error && <ErrorBox message={error} onRetry={reload} />}
        <div data-testid="audit-status" data-state={state} data-ok={data ? String(data.ok) : ''} role="status" aria-live="polite">
          {loading && <p className="text-muted">Recomputing the chain...</p>}
          {!loading && data && (
            <div>
              <p className="flex items-center gap-2 text-lg font-bold" style={{ color: data.ok ? 'var(--ok)' : 'var(--danger)' }}>
                <span aria-hidden="true">{data.ok ? '✓' : '✕'}</span>
                <span>
                  {data.ok
                    ? 'Chain intact: no record has been changed'
                    : `Chain broken at record ${data.first_bad_index ?? 'unknown'}: that record or one before it was changed`}
                </span>
              </p>
              <dl className="mt-3 grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-[0.85rem]">
                <dt className="text-muted">Records checked</dt>
                <dd className="tnum" data-testid="audit-records">
                  {fmtInt(data.records)}
                </dd>
                <dt className="text-muted">Head hash</dt>
                <dd className="font-mono text-[0.8rem] break-all" data-testid="audit-head-hash">
                  {data.head_hash || 'none (empty log)'}
                </dd>
                {data.first_bad_index != null && (
                  <>
                    <dt className="text-muted">First bad record</dt>
                    <dd className="tnum" data-testid="audit-first-bad">
                      {data.first_bad_index}
                    </dd>
                  </>
                )}
              </dl>
              <p className="mt-3 max-w-[80ch] text-[0.78rem] text-muted">
                The head hash is anchored outside the system once a day. If it matches the anchored value and the chain verifies,
                nothing in the log was edited after the fact.
              </p>
            </div>
          )}
        </div>
      </Section>
    </div>
  )
}
