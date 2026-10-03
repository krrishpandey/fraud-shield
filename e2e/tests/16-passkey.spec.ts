import { readFileSync } from 'node:fs'
import { expect, test, type Page } from '@playwright/test'
import { PASS_ACTIONS, STOP_ACTIONS, actionOf, auditCli } from './helpers'

// WebAuthn refuses IP addresses as the relying party, so this spec opens the console at localhost (the backend still
// binds 127.0.0.1 only: this also checks the browser falls back from ::1 to 127.0.0.1).
const PORT = process.env.E2E_PORT ?? '8090'
const LOCAL = `http://localhost:${PORT}`

async function virtualAuthenticator(page: Page) {
  const cdp = await page.context().newCDPSession(page)
  await cdp.send('WebAuthn.enable')
  const { authenticatorId } = await cdp.send('WebAuthn.addVirtualAuthenticator', {
    options: {
      protocol: 'ctap2', transport: 'internal', hasResidentKey: true, hasUserVerification: true,
      isUserVerified: true, automaticPresenceSimulation: true,
    },
  })
  return { cdp, authenticatorId }
}

async function scoreTakeoverAt(page: Page, base: string): Promise<string> {
  await page.goto(`${base}/#/`)
  await page.getByTestId('demo-booking-takeover').click()
  await page.getByTestId('score-button').click()
  await expect(page).toHaveURL(/#\/decisions\/[^/]+$/)
  const id = await page.getByTestId('decision-detail').getAttribute('data-decision-id')
  expect(id).toBeTruthy()
  return id!
}

test('at 127.0.0.1 the passkey step explains it needs localhost', async ({ page }) => {
  await scoreTakeoverAt(page, '')
  await expect(page.getByTestId('owner-passkey')).toBeVisible()
  await expect(page.getByTestId('owner-passkey-blocker')).toContainText(`localhost:${PORT}`)
  await expect(page.getByTestId('owner-passkey-confirm')).toBeDisabled()
})

test('owner passkey: enroll, confirm releases the booking, the same signature fails on a tampered booking', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  const { cdp, authenticatorId } = await virtualAuthenticator(page)

  const id = await scoreTakeoverAt(page, LOCAL)
  const action = await actionOf(page)
  expect(STOP_ACTIONS).toContain(action)
  const box = page.getByTestId('owner-passkey')
  await expect(box).toBeVisible()
  await expect(page.getByTestId('owner-passkey-simulated')).toContainText('Simulated owner device')
  await expect(page.getByTestId('owner-passkey-blocker')).toHaveCount(0)

  // enroll (the virtual authenticator plays the owner's phone)
  await expect(page.getByTestId('owner-passkey-status')).toHaveAttribute('data-enrolled', 'false')
  await page.getByTestId('owner-passkey-enroll').click()
  await expect(page.getByTestId('owner-passkey-status')).toHaveAttribute('data-enrolled', 'true')
  const creds = await cdp.send('WebAuthn.getCredentials', { authenticatorId })
  expect(creds.credentials.length).toBe(1)
  expect(creds.credentials[0].rpId).toBe('localhost')

  // confirm as owner: the booking is released
  const t0 = Date.now()
  const verifyResp = page.waitForResponse((r) => r.url().endsWith(`/decisions/${id}/owner_confirm/verify`))
  await page.getByTestId('owner-passkey-confirm').click()
  const verified = await (await verifyResp).json()
  const result = page.getByTestId('owner-passkey-result')
  await expect(result).toHaveAttribute('data-verified', 'true')
  const clickToReleasedMs = Date.now() - t0
  const released = await result.getAttribute('data-released')
  expect(PASS_ACTIONS).toContain(released)
  expect(released).toBe(action === 'owner_confirm' ? 'allow' : 'allow_scan_gated')
  expect(verified.verified).toBe(true)
  console.log(`passkey verify: server signature check ${verified.verify_ms} ms, endpoint ${verified.server_ms} ms, ` +
              `click to released ${clickToReleasedMs} ms (virtual authenticator, includes the browser ceremony)`)

  // the decision record carries the release
  const detail = await (await page.request.get(`${LOCAL}/decisions/${id}`)).json()
  expect(detail.owner_confirmation.verified).toBe(true)
  expect(detail.action).toBe(action) // the audited decision itself is not rewritten

  // tamper test: the SAME assertion against carrier_cost + R$100 is rejected by the challenge binding
  await page.getByTestId('owner-passkey-tamper').click()
  const tamper = page.getByTestId('owner-passkey-tamper-result')
  await expect(tamper).toHaveAttribute('data-verified', 'false')
  await expect(tamper).toHaveAttribute('data-code', 'challenge')
  await expect(tamper).toContainText('Rejected')

  // audit: enrollment, confirmation and the failed tamper check are all in the hash chain
  const auditPath = (await (await page.request.get(`${LOCAL}/health`)).json()).audit_path as string
  // only our event lines: other records may hold NaN (valid for Python's json, not for JSON.parse)
  const events = readFileSync(auditPath, 'utf-8').split('\n')
    .filter((l) => /"event_type":"(passkey_enrolled|owner_confirmed|owner_confirm_failed)"/.test(l))
    .map((l) => JSON.parse(l))
  const enrolled = events.filter((e) => e.event_type === 'passkey_enrolled')
  const confirmed = events.filter((e) => e.event_type === 'owner_confirmed' && e.payload.decision_id === id)
  const failed = events.filter((e) => e.event_type === 'owner_confirm_failed' && e.payload.decision_id === id)
  expect(enrolled.length).toBeGreaterThanOrEqual(1)
  expect(confirmed.length).toBe(1)
  expect(confirmed[0].payload.credential_id_hash).toMatch(/^[0-9a-f]{64}$/)
  expect(confirmed[0].payload.bound_fields_hash).toBe(verified.bound_fields_hash)
  expect(failed.at(-1).payload.tamper_test).toBe(true)
  expect(failed.at(-1).payload.code).toBe('challenge')
  expect(failed.at(-1).payload.bound_fields_hash).not.toBe(verified.bound_fields_hash)
  const cli = auditCli(auditPath)
  expect(cli.code).toBe(0)
  expect(cli.out.ok).toBe(true)

  // after a reload the release is still shown (from the decision record)
  await page.reload()
  await expect(page.getByTestId('owner-passkey-result')).toHaveAttribute('data-verified', 'true')
  await expect(page.getByTestId('owner-passkey-confirm')).toHaveCount(0)
  expect(errors).toEqual([])
})
