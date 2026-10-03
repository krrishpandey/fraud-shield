import { useCallback, useEffect, useState } from 'react'
import { api } from '../api/client'
import type { CredentialJSON, DecisionDetail, OwnerConfirmation, PasskeyStatus, TamperTestResult } from '../api/types'
import { actionLabel } from '../lib/domain'
import { fmtBRL, fmtMs, shortHash } from '../lib/format'
import { createPasskey, getAssertion, passkeyBlocker, passkeyErrorText } from '../lib/webauthn'

// Stopped actions the owner's passkey can release, and to what (fraudshield/api/passkey.py RELEASE).
const RELEASE: Record<string, string> = { owner_confirm: 'allow', hold: 'allow_scan_gated', review: 'allow_scan_gated' }
const TAMPER_DELTA = 100

/** "Was this you?": the owner's passkey signs a challenge bound to this booking's details (docs/PASSKEY.md). */
export function OwnerPasskey({ d, onDone }: { d: DecisionDetail; onDone: () => void }) {
  const b = d.booking
  const [status, setStatus] = useState<PasskeyStatus | null>(null)
  const [busy, setBusy] = useState<null | 'enroll' | 'confirm' | 'tamper'>(null)
  const [err, setErr] = useState<string | null>(null)
  const [result, setResult] = useState<OwnerConfirmation | null>(null)
  const [signed, setSigned] = useState<{ nonce_id: string; credential: CredentialJSON } | null>(null)
  const [tamper, setTamper] = useState<TamperTestResult | null>(null)
  const blocker = passkeyBlocker()
  const release = RELEASE[d.action]
  const done = d.owner_confirmation?.verified ? d.owner_confirmation : result?.verified ? result : null

  const loadStatus = useCallback(() => {
    if (!b) return
    api.passkeyStatus(b.account_id).then(setStatus, (e) => setErr(e instanceof Error ? e.message : String(e)))
  }, [b])
  useEffect(loadStatus, [loadStatus])

  if (!b || (!release && !d.owner_confirmation)) return null

  const run = async (what: 'enroll' | 'confirm' | 'tamper', fn: () => Promise<void>) => {
    setBusy(what)
    setErr(null)
    try {
      await fn()
    } catch (e) {
      setErr(passkeyErrorText(e))
    } finally {
      setBusy(null)
    }
  }

  const enroll = () =>
    run('enroll', async () => {
      const o = await api.passkeyEnrollOptions(b.account_id)
      const credential = await createPasskey(o.publicKey)
      const r = await api.passkeyEnrollVerify({ nonce_id: o.nonce_id, account_id: b.account_id, credential })
      if (!r.enrolled) throw new Error(`Enrollment rejected by the server: ${r.reason}`)
      loadStatus()
    })

  const confirm = () =>
    run('confirm', async () => {
      const o = await api.ownerConfirmOptions(d.decision_id)
      const credential = await getAssertion(o)
      const r = await api.ownerConfirmVerify(d.decision_id, { nonce_id: o.nonce_id, credential })
      setResult(r)
      setTamper(null)
      if (r.verified) setSigned({ nonce_id: o.nonce_id, credential })
      onDone()
    })

  const tamperTest = () =>
    run('tamper', async () => {
      if (!signed) return
      setTamper(await api.ownerConfirmTamper(d.decision_id, { ...signed, field: 'carrier_cost', delta: TAMPER_DELTA }))
    })

  return (
    <div className="mt-6 max-w-[60ch]" data-testid="owner-passkey">
      <h2 className="verdict-h2">Was this you? The owner's passkey</h2>
      <p className="text-[0.8rem] text-muted" data-testid="owner-passkey-simulated">
        Simulated owner device: in this demo, this laptop plays the account owner's phone. In production the request goes
        to the passkey on the owner's own device.
      </p>
      {blocker && (
        <p className="scan-result is-bad mt-2" role="alert" data-testid="owner-passkey-blocker">
          {blocker}
        </p>
      )}

      <dl className="mt-2 grid grid-cols-[max-content_1fr] gap-x-3 gap-y-0.5 text-[0.82rem]" data-testid="owner-passkey-bound">
        <dt className="text-muted">The signature covers</dt>
        <dd>
          booking {b.booking_id}, account {b.account_id}, carrier cost {fmtBRL(b.carrier_cost)}, declared value{' '}
          {fmtBRL(b.declared_value)}, destination ZIP {b.dest_zip3}, consignee {b.consignee_id}
        </dd>
        {release && (
          <>
            <dt className="text-muted">If confirmed</dt>
            <dd>
              released as <strong>{actionLabel(release)}</strong>
              {d.action !== 'owner_confirm' && ' (demo: a held booking is only released with a depot weight check)'}
            </dd>
          </>
        )}
      </dl>

      <div className="mt-2 text-[0.85rem]" data-testid="owner-passkey-status" data-enrolled={status?.enrolled ?? ''}>
        {status?.enrolled ? (
          <span>
            Owner passkey enrolled for this account (credential {shortHash(status.credential_id_hash, 12)}).
          </span>
        ) : (
          status && <span className="text-ink-2">No owner passkey enrolled for this account yet.</span>
        )}
      </div>

      <div className="mt-2 flex flex-wrap gap-2">
        {!status?.enrolled && (
          <button type="button" className="btn" disabled={busy !== null || !!blocker || !status} onClick={enroll}
            data-testid="owner-passkey-enroll">
            {busy === 'enroll' ? 'Waiting for the passkey...' : "Enroll owner passkey (demo: this laptop plays the owner's phone)"}
          </button>
        )}
        {!done && (
          <button type="button" className="btn btn-primary" disabled={busy !== null || !!blocker || !status?.enrolled}
            onClick={confirm} data-testid="owner-passkey-confirm">
            {busy === 'confirm' ? 'Waiting for the passkey...' : 'Confirm as owner'}
          </button>
        )}
        <button type="button" className="btn" disabled={busy !== null || !signed} onClick={tamperTest}
          title={signed ? undefined : 'Confirm as owner first: the test re-uses that signature'} data-testid="owner-passkey-tamper">
          {busy === 'tamper' ? 'Checking...' : `Tamper test: same signature, carrier cost + ${fmtBRL(TAMPER_DELTA, true)}`}
        </button>
      </div>
      {err && <p className="mt-2 text-[0.85rem] text-danger" role="alert" data-testid="owner-passkey-error">{err}</p>}

      <div aria-live="polite">
        {done ? (
          <p className="scan-result mt-3" role="status" data-testid="owner-passkey-result" data-verified="true"
            data-released={done.released_action}>
            Owner confirmed with their passkey{done.verify_ms != null ? `, signature checked in ${fmtMs(done.verify_ms)}` : ''}.
            Booking released: <strong>{actionLabel(done.released_action ?? '')}</strong>. Audit{' '}
            <code className="font-mono text-[0.78rem]">{shortHash(done.audit_hash, 16)}</code>
          </p>
        ) : (
          result && (
            <p className="scan-result is-bad mt-3" role="status" data-testid="owner-passkey-result" data-verified="false"
              data-code={result.code}>
              Not confirmed: {result.reason}. The booking stays stopped. Audit{' '}
              <code className="font-mono text-[0.78rem]">{shortHash(result.audit_hash, 16)}</code>
            </p>
          )
        )}
        {tamper && (
          <div className={`scan-result mt-2${tamper.verified ? '' : ' is-bad'}`} role="status" data-testid="owner-passkey-tamper-result"
            data-verified={tamper.verified} data-code={tamper.code ?? ''}>
            {tamper.verified
              ? 'The signature still verified: the booking details did not change.'
              : `Rejected: the same signature was checked against the booking with carrier cost ${fmtBRL(tamper.tampered_fields.carrier_cost)} instead of ${fmtBRL(tamper.original_fields.carrier_cost)}. The server recomputed the challenge from the changed booking and the signature no longer matches.`}
            <div className="mt-1 text-[0.78rem] font-normal text-ink-2">
              Bound-fields hash {shortHash(tamper.original_bound_fields_hash, 12)} became{' '}
              {shortHash(tamper.tampered_bound_fields_hash, 12)}. Audit{' '}
              <code className="font-mono">{shortHash(tamper.audit_hash, 16)}</code>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
