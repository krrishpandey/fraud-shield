import { expect, test, type APIRequestContext } from '@playwright/test'
import { demoBookings } from './helpers'

/** Scores clones of the demo bookings (fresh ids) and labels them as an analyst would. */
async function seedAnalystLabels(request: APIRequestContext, n: number): Promise<void> {
  const demo = await demoBookings(request)
  for (let i = 0; i < n; i++) {
    const d = demo[i % demo.length]
    const booking = { ...d.booking, booking_id: `e2e-learn-${i}-${d.booking.booking_id}` }
    const s = await request.post('/score', { data: booking })
    expect(s.ok()).toBeTruthy()
    const { decision_id } = await s.json()
    const label = d.expected === 'legit' || d.expected == null ? 'legit' : 'fraud'
    const a = await request.post(`/decisions/${decision_id}/analyst`, { data: { label, note: 'e2e seed label' } })
    expect(a.ok()).toBeTruthy()
  }
}

test('learning view renders against the real API (regression: v1 has null created_at/metrics)', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  await page.goto('/#/learning')
  await expect(page.getByTestId('learning-version-table')).toBeVisible({ timeout: 5_000 })
  expect(errors).toEqual([])
})

test('simulate feedback: the console shows a result or a clear error, never a crash', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  await page.goto('/#/learning')
  await expect(page.getByTestId('learning-active-version')).toBeVisible()
  await page.getByTestId('learning-simulate-n').fill('200')
  const resp = page.waitForResponse((r) => r.url().endsWith('/learning/simulate_feedback'))
  await page.getByTestId('learning-simulate-button').click()
  const r = await resp
  if (r.ok()) {
    await expect(page.getByTestId('learning-simulate-result')).toBeVisible()
  } else {
    await expect(page.getByTestId('learning-error')).toBeVisible()
  }
  await expect(page.getByTestId('learning-retrain-button')).toBeEnabled()
  expect(errors).toEqual([])
})

test('simulate feedback n=200 succeeds (regression: NaN features used to crash the serializer)', async ({ request }) => {
  const r = await request.post('/learning/simulate_feedback', { data: { n: 200, seed: 1, advance_days: 90 } })
  expect(r.status()).toBe(200)
  const body = await r.json()
  expect(body.simulated).toBe(true)
  expect(body.added).toBeGreaterThan(0)
})

test('retrain: result table, gate checks, deployed banner consistent with the gate, rollback control', async ({ page, request }) => {
  test.setTimeout(150_000)
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))

  // the 20+ new labels come from analyst confirmations made through the API (the real analyst path)
  await seedAnalystLabels(request, 24)

  await page.goto('/#/learning')
  const count = page.getByTestId('learning-label-count')
  await expect(count).toBeVisible()
  expect(Number(await count.getAttribute('data-count'))).toBeGreaterThanOrEqual(20)
  await expect(page.getByTestId('learning-source-analyst')).toBeVisible()
  await expect(page.getByTestId('learning-laya-note')).toBeVisible()
  const activeBefore = (await page.getByTestId('learning-active-version').innerText()).trim()

  const resp = page.waitForResponse((r) => r.url().endsWith('/learning/retrain'), { timeout: 90_000 })
  await page.getByTestId('learning-retrain-button').click()
  await expect(page.getByTestId('learning-retrain-progress')).toBeVisible()
  const r = await resp
  expect(r.status()).toBe(200)
  const run = await r.json()

  const result = page.getByTestId('learning-result')
  await expect(result).toBeVisible()
  await expect(result).toHaveAttribute('data-run-id', run.run_id)
  await expect(page.getByTestId('learning-retrain-progress')).toHaveCount(0)
  expect(await page.getByTestId('learning-result-table').locator('[data-metric]').count()).toBeGreaterThan(0)

  const checks = page.getByTestId('learning-gate-check')
  await expect(checks).toHaveCount(run.gate.checks.length)
  for (const c of run.gate.checks as { name: string; passed: boolean }[]) {
    await expect(page.locator(`[data-testid="learning-gate-check"][data-name="${c.name}"]`)).toHaveAttribute(
      'data-passed', String(c.passed))
  }

  // deployed or rejected are both valid; the banner must agree with the gate
  const deployed = run.gate.passed === true
  expect(run.deployed_version === run.candidate.version).toBe(deployed)
  await expect(page.getByTestId('learning-deployed-status')).toHaveAttribute('data-deployed', String(deployed))
  const active = page.getByTestId('learning-active-version')
  await expect(active).toHaveText(deployed ? run.candidate.version : activeBefore)

  // version history: candidate listed; every non-active row offers rollback, or says the gate rejected it
  const rows = page.getByTestId('learning-version-row')
  const candRow = page.locator(`[data-testid="learning-version-row"][data-version="${run.candidate.version}"]`)
  await expect(candRow).toBeVisible()
  const nRows = await rows.count()
  expect(nRows).toBeGreaterThan(1)
  const nButtons = await page.getByTestId('learning-rollback-button').count()
  const nRejected = await page.getByTestId('learning-version-rejected').count()
  expect(nButtons + nRejected).toBe(nRows - 1)
  if (!deployed) await expect(candRow.getByTestId('learning-version-rejected')).toBeVisible()
  await expect(page.locator('[data-testid="learning-version-row"][data-active="true"]')).toHaveCount(1)

  if (deployed) {
    // roll back to the previous (deployed before) version
    await page.locator(`[data-testid="learning-rollback-button"][data-version="${activeBefore}"]`).click()
    await expect(active).toHaveText(activeBefore)
  }
  expect(errors).toEqual([])
})
