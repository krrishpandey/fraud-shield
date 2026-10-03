import { expect, test } from '@playwright/test'

test('label-free monitor: estimated precision and missed fraud on the live stream, with the blind spot', async ({ page, request }) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  await page.goto('/#/live')
  await expect(page.getByTestId('live-view')).toBeVisible()
  await expect(page.getByTestId('live-state')).not.toHaveAttribute('data-state', 'loading')

  // an earlier spec may have left a stream running: stop it, then start a fresh one
  if ((await page.getByTestId('live-state').getAttribute('data-state')) === 'running') {
    await page.getByTestId('live-stop').click()
    await expect(page.getByTestId('live-state')).toHaveAttribute('data-state', /stopped|done/, { timeout: 15_000 })
  }
  await page.getByTestId('live-rate-50').click()
  await page.getByTestId('live-start').click()
  await expect(page.getByTestId('live-state')).toHaveAttribute('data-state', /running|done/)

  // the card fills in with numbers once some bookings have been decided and at least one was stopped
  const card = page.getByTestId('monitor-card')
  await expect(card).toBeVisible()
  await expect.poll(async () => Number(await card.getAttribute('data-n')), { timeout: 30_000 }).toBeGreaterThanOrEqual(50)
  await expect(page.getByTestId('monitor-est-precision')).toContainText('Estimated precision of stops')
  await expect(page.getByTestId('monitor-est-precision')).toContainText('(no labels needed)')
  await expect
    .poll(async () => (await page.getByTestId('monitor-est-precision').locator('dd').innerText()).trim(), { timeout: 30_000 })
    .toMatch(/^\d+\.\d%/)
  await expect(page.getByTestId('monitor-est-missed')).toContainText(/Estimated fraud missed\s*\d+\.\d/)
  await expect(page.getByTestId('monitor-realized')).toContainText(
    'Simulation ground truth, not available in production until labels arrive',
  )
  await expect(page.getByTestId('monitor-blind-spot')).toContainText('never learned')
  await expect(page.getByTestId('monitor-blind-spot')).toContainText(/\d+\.\d+ per week/)

  // the API says the same thing: estimate without labels, realized labelled as simulation ground truth
  const m = await (await request.get('/monitor/estimate')).json()
  expect(m.n).toBeGreaterThan(0)
  expect(m.estimated.n_stopped + m.estimated.n_let_through).toBe(m.n)
  expect(m.estimated.fraud_missed).toBeGreaterThanOrEqual(0)
  expect(m.realized.label).toBe('simulation ground truth, not available in production until labels arrive')
  expect(m.blind_spot.period).toBe('week')
  expect(m.blind_spot.missed_gap_per_period).toBeLessThan(0)

  await page.getByTestId('live-stop').click()
  await expect(page.getByTestId('live-state')).toHaveAttribute('data-state', /stopped|done/, { timeout: 15_000 })
  expect(errors).toEqual([])
})
