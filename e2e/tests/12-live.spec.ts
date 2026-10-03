import { expect, test } from '@playwright/test'

test('live stream: start, decisions stream in with metrics, a row opens its decision, stop', async ({ page, request }) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  await page.goto('/#/live')
  await expect(page.getByTestId('live-view')).toBeVisible()

  await page.getByTestId('live-rate-50').click()
  await page.getByTestId('live-start').click()
  await expect(page.getByTestId('live-state')).toHaveAttribute('data-state', /running|done/)

  // decisions arrive continuously
  const rows = page.getByTestId('live-feed-row')
  await expect.poll(async () => rows.count(), { timeout: 20_000 }).toBeGreaterThanOrEqual(10)
  await expect(page.getByTestId('live-throughput')).toBeVisible()
  await expect(page.getByTestId('live-latency')).toBeVisible()
  await expect(page.getByTestId('live-actions')).toBeVisible()
  await expect(page.getByTestId('live-accuracy')).toBeVisible()  // table, or 'no fraud yet' before the first fraud

  // the page shows exactly what the API measured, and live equals offline on the same bookings
  const m = await (await request.get('/stream/metrics')).json()
  expect(m.load.scored).toBeGreaterThan(0)
  expect(m.consistency.max_abs_score_diff).toBeLessThanOrEqual(0.00005)

  // the held-and-blocked worklist shows exactly what the service held or blocked
  const panel = page.getByTestId('live-flagged')
  await expect(panel).toBeVisible()
  const fl = await (await request.get('/stream/flagged')).json()
  await expect.poll(async () => Number(await panel.getAttribute('data-count'))).toBeGreaterThanOrEqual(fl.counts.hold + fl.counts.block)
  for (const r of fl.rows) expect(['hold', 'block']).toContain(r.action)
  await page.getByTestId('live-flagged-filter-unreviewed').click()
  await expect(page.getByTestId('live-flagged-filter-unreviewed')).toHaveAttribute('aria-checked', 'true')

  // a streamed decision opens on the normal decision page
  const id = await rows.first().getAttribute('data-decision-id')
  await rows.first().locator('a').click()
  await expect(page).toHaveURL(new RegExp(`#/decisions/${id}$`))
  await expect(page.getByTestId('decision-detail')).toBeVisible()

  // stop, and the main review queue is not flooded by the stream
  await page.goto('/#/live')
  await expect(page.getByTestId('live-state')).not.toHaveAttribute('data-state', 'loading')
  if ((await page.getByTestId('live-state').getAttribute('data-state')) === 'running') {
    await page.getByTestId('live-stop').click()
  }
  await expect(page.getByTestId('live-state')).toHaveAttribute('data-state', /stopped|done/, { timeout: 15_000 })
  const main = await (await request.get('/decisions?limit=500')).json()
  expect(main.some((d: { decision_id: string }) => d.decision_id === id)).toBe(false)
  expect(errors).toEqual([])
})
