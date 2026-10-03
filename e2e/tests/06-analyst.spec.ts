import { expect, test } from '@playwright/test'
import { kpiInt, scoreDemoInUi } from './helpers'

test('analyst confirms fraud: audit hash, queue label, dashboard bookings count goes up', async ({ page }) => {
  await page.goto('/#/dashboard')
  await expect(page.getByTestId('dashboard-kpi-bookings')).toBeVisible()
  const before = await kpiInt(page, 'bookings')

  const id = await scoreDemoInUi(page, 'takeover') // already scored in spec 04: idempotent, no new booking
  await page.getByTestId('analyst-note').fill('e2e: confirmed label resale takeover')
  await page.getByTestId('analyst-confirm-fraud').click()
  await expect(page.getByTestId('analyst-result')).toHaveAttribute('data-label', 'fraud')
  await expect(page.getByTestId('analyst-result-hash')).toHaveText(/^[0-9a-f]{64}$/)

  await page.getByTestId('nav-queue').click()
  const row = page.locator(`[data-testid="queue-row"][data-decision-id="${id}"]`)
  await expect(row).toBeVisible()
  await expect(row).toContainText('Fraud')

  // a booking not scored before raises the bookings KPI by exactly one
  await scoreDemoInUi(page, 'reshipping-drop')
  await page.getByTestId('nav-dashboard').click()
  await expect(page.getByTestId('dashboard-kpi-bookings')).toBeVisible()
  await expect.poll(() => kpiInt(page, 'bookings')).toBe(before + 1)
  await expect(page.getByTestId('action-mix')).toBeVisible()
})
