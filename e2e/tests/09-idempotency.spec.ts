import { expect, test } from '@playwright/test'
import { demoBookings, kpiInt, scoreDemoInUi } from './helpers'

test('scoring the same booking twice returns the same decision id (API)', async ({ request }) => {
  const demo = (await demoBookings(request)).find((d) => d.scenario === 'weight-manipulation')!
  const a = await (await request.post('/score', { data: demo.booking })).json()
  const b = await (await request.post('/score', { data: demo.booking })).json()
  expect(a.decision_id).toBeTruthy()
  expect(b.decision_id).toBe(a.decision_id)
  expect(b.audit_hash).toBe(a.audit_hash)
})

test('scoring the same booking twice in the console opens the same decision and adds no booking', async ({ page }) => {
  const first = await scoreDemoInUi(page, 'legit-tenured')
  await page.goto('/#/dashboard')
  await expect(page.getByTestId('dashboard-kpi-bookings')).toBeVisible()
  const n = await kpiInt(page, 'bookings')
  const second = await scoreDemoInUi(page, 'legit-tenured')
  expect(second).toBe(first)
  await page.goto('/#/dashboard')
  await expect(page.getByTestId('dashboard-kpi-bookings')).toBeVisible()
  await expect.poll(() => kpiInt(page, 'bookings')).toBe(n)
})
