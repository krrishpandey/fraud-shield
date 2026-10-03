// Browser side of the owner passkey step: base64url <-> bytes, and navigator.credentials.create/get.
// The server (fraudshield/identity/webauthn.py) does every check; this file only runs the ceremony.
import type { CredentialJSON, OwnerConfirmOptions, PasskeyEnrollOptions } from '../api/types'

export function b64urlToBytes(s: string): Uint8Array<ArrayBuffer> {
  const b64 = s.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - (s.length % 4)) % 4)
  const bin = atob(b64)
  const out = new Uint8Array(new ArrayBuffer(bin.length))
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

export function bytesToB64url(buf: ArrayBuffer | ArrayBufferView): string {
  const u8 = buf instanceof ArrayBuffer ? new Uint8Array(buf) : new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength)
  let bin = ''
  for (const b of u8) bin += String.fromCharCode(b)
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

/** Why passkeys cannot run in this page, or null if they can. */
export function passkeyBlocker(rpId = 'localhost'): string | null {
  if (typeof window === 'undefined' || !window.PublicKeyCredential || !navigator.credentials)
    return `This window has no passkey support. Open http://${rpId}:${window.location.port} in Edge or Chrome.`
  if (window.location.hostname !== rpId)
    return `Passkeys only work at http://${rpId}:${window.location.port} (WebAuthn refuses IP addresses). Open the console there.`
  return null
}

/** Plain-language text for a failed browser ceremony. */
export function passkeyErrorText(e: unknown): string {
  const name = e instanceof DOMException ? e.name : ''
  if (name === 'NotAllowedError')
    return 'The passkey prompt was cancelled or timed out. If Windows Hello has no PIN on this PC, set one up (Settings, Accounts, Sign-in options) or pick a phone or security key in the prompt.'
  if (name === 'SecurityError')
    return `The browser refused the passkey for this address. Open http://localhost:${window.location.port} instead.`
  if (name === 'InvalidStateError') return 'This device already holds a passkey for this account.'
  return e instanceof Error ? e.message : String(e)
}

export async function createPasskey(o: PasskeyEnrollOptions['publicKey']): Promise<CredentialJSON> {
  const cred = (await navigator.credentials.create({
    publicKey: {
      ...o,
      challenge: b64urlToBytes(o.challenge),
      user: { ...o.user, id: b64urlToBytes(o.user.id) },
      authenticatorSelection: o.authenticatorSelection as AuthenticatorSelectionCriteria,
    },
  })) as PublicKeyCredential | null
  if (!cred) throw new Error('The browser returned no passkey.')
  const r = cred.response as AuthenticatorAttestationResponse
  return {
    id: cred.id,
    rawId: bytesToB64url(cred.rawId),
    type: cred.type,
    response: { clientDataJSON: bytesToB64url(r.clientDataJSON), attestationObject: bytesToB64url(r.attestationObject) },
  }
}

export async function getAssertion(o: OwnerConfirmOptions): Promise<CredentialJSON> {
  const cred = (await navigator.credentials.get({
    publicKey: {
      challenge: b64urlToBytes(o.challenge),
      rpId: o.rp_id,
      allowCredentials: o.allow_credentials.map((c) => ({ type: c.type, id: b64urlToBytes(c.id) })),
      timeout: o.timeout,
      userVerification: o.user_verification as UserVerificationRequirement,
    },
  })) as PublicKeyCredential | null
  if (!cred) throw new Error('The browser returned no signature.')
  const r = cred.response as AuthenticatorAssertionResponse
  return {
    id: cred.id,
    rawId: bytesToB64url(cred.rawId),
    type: cred.type,
    response: {
      clientDataJSON: bytesToB64url(r.clientDataJSON),
      authenticatorData: bytesToB64url(r.authenticatorData),
      signature: bytesToB64url(r.signature),
      userHandle: r.userHandle ? bytesToB64url(r.userHandle) : null,
    },
  }
}
