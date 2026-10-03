import { expect, test } from '@playwright/test'
import { demoBookings } from './helpers'

test('a failed depot scan sends the account’s next parcel to a scan check, and the console records scans', async ({ page, request }) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  const wm = (await demoBookings(request)).find((d) => d.scenario === 'weight-manipulation')!.booking
  const { meta: _meta, ...base } = wm as typeof wm & { meta?: unknown }
  void _meta
  const run = Date.now().toString(36)

  // a parcel from this account, then the depot scale shows it was far heavier than declared
  const first = await (await request.post('/score', { data: { ...base, booking_id: `scan-a-${run}`, weight_kg: 0.1 } })).json()
  const scan = await (await request.post(`/decisions/${first.decision_id}/first-scan`, { data: { measured_weight_kg: 4.4 } })).json()
  expect(scan.mismatch).toBe(true)

  // the account's next, ordinary-looking parcel is checked at first scan automatically
  const next = await (await request.post('/score', { data: { ...base, booking_id: `scan-b-${run}`, booked_at: '2018-06-11T09:00:00' } })).json()
  expect(next.action).toBe('allow_scan_gated')
  expect(next.reasons).toContain('ACCOUNT_FAILED_DEPOT_SCAN')

  // the decision page offers the depot scan and shows the result
  await page.goto(`/#/decisions/${next.decision_id}`)
  await expect(page.getByTestId('depot-scan')).toBeVisible()
  await page.getByTestId('depot-scan-input').fill('1.33')
  await page.getByTestId('depot-scan-record').click()
  await expect(page.getByTestId('depot-scan-result')).toHaveAttribute('data-mismatch', 'false')
  await expect(page.getByTestId('reason-codes')).toContainText('failed a depot weight check')
  expect(errors).toEqual([])
})
