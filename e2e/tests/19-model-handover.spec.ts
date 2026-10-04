import { expect, test, type APIRequestContext } from '@playwright/test'
import { demoBookings } from './helpers'

/** Scores clones of the demo bookings (fresh ids) and labels them as an analyst would. */
async function seedAnalystLabels(request: APIRequestContext, n: number, tag: string): Promise<void> {
  const demo = await demoBookings(request)
  for (let i = 0; i < n; i++) {
    const d = demo[i % demo.length]
    const booking = { ...d.booking, booking_id: `e2e-${tag}-${i}-${d.booking.booking_id}` }
    const s = await request.post('/score', { data: booking })
    expect(s.ok()).toBeTruthy()
    const { decision_id } = await s.json()
    const label = d.expected === 'legit' || d.expected == null ? 'legit' : 'fraud'
    const a = await request.post(`/decisions/${decision_id}/analyst`, { data: { label, note: 'e2e seed label' } })
    expect(a.ok()).toBeTruthy()
  }
}

/** Scores one fresh booking through the API and returns its decision id and logged GBM version. */
async function scoreFresh(request: APIRequestContext, id: string): Promise<{ decisionId: string; gbm: string }> {
  const demo = await demoBookings(request)
  const s = await request.post('/score', { data: { ...demo[0].booking, booking_id: id } })
  expect(s.ok()).toBeTruthy()
  const body = await s.json()
  return { decisionId: body.decision_id, gbm: body.model_versions.gbm }
}

test('retrain: the banner says which model scores new bookings from now on; decisions show the serving version', async ({ page, request }) => {
  test.setTimeout(180_000)
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))

  await seedAnalystLabels(request, 24, 'handover')
  const before = await (await request.get('/learning/status')).json()
  expect(before.model_in_use.version).toBe(before.active_version) // the service logs the exact registry version

  await page.goto('/#/learning')
  const inUse = page.getByTestId('learning-model-in-use')
  await expect(inUse).toHaveAttribute('data-version', before.active_version)
  await expect(inUse).toContainText(`Model in use: ${before.active_version} since`)

  const resp = page.waitForResponse((r) => r.url().endsWith('/learning/retrain'), { timeout: 120_000 })
  await page.getByTestId('learning-retrain-button').click()
  const r = await resp
  expect(r.status()).toBe(200)
  const run = await r.json()
  const h = run.handover
  const deployed = run.gate.passed === true
  expect(h.verdict).toBe(deployed ? 'new_model_in_use' : 'previous_model_kept')
  expect(h.candidate_version).toBe(run.candidate.version)
  expect(h.previous_version).toBe(before.active_version)
  expect(h.active_version).toBe(deployed ? run.candidate.version : before.active_version)
  expect(h.active_since).toBeTruthy()

  // the banner states the verdict, the version in use and, when rejected, every failed check in plain words
  const banner = page.getByTestId('handover-banner')
  await expect(banner).toHaveAttribute('data-verdict', h.verdict)
  await expect(banner).toHaveAttribute('data-active-version', h.active_version)
  const text = page.getByTestId('handover-banner-text')
  if (deployed) {
    await expect(text).toContainText(`From now on, new bookings are scored by ${h.active_version}`)
    await expect(text).toContainText(`Previous model ${before.active_version} kept for rollback`)
  } else {
    await expect(text).toContainText(`Still using ${before.active_version}`)
    await expect(text).toContainText(`Candidate ${run.candidate.version}`)
    await expect(text).toContainText('was not deployed because')
    const reasons = page.getByTestId('handover-banner-reasons').locator('li')
    await expect(reasons).toHaveCount(h.failed_checks.length)
    expect(h.failed_checks.length).toBeGreaterThan(0)
  }
  await expect(inUse).toHaveAttribute('data-version', h.active_version)

  // the swap really applies to new bookings: API record and decision page both name the serving version
  const fresh = await scoreFresh(request, `e2e-handover-after-${run.run_id}`)
  expect(fresh.gbm).toBe(h.active_version)
  await page.goto(`/#/decisions/${fresh.decisionId}`)
  await expect(page.getByTestId('decision-gbm-version')).toHaveAttribute('data-version', h.active_version)

  // status persists the handover, so the banner is still there after a reload
  await page.goto('/#/learning')
  await expect(page.getByTestId('handover-banner')).toHaveAttribute('data-verdict', h.verdict)

  if (deployed) {
    // rollback shows its own banner, and new bookings go back to the previous version
    await page.locator(`[data-testid="learning-rollback-button"][data-version="${before.active_version}"]`).click()
    await expect(page.getByTestId('handover-banner')).toHaveAttribute('data-verdict', 'rolled_back')
    await expect(page.getByTestId('handover-banner-text')).toContainText(
      `Rolled back. From now on, new bookings are scored by ${before.active_version}`)
    const back = await scoreFresh(request, `e2e-handover-rollback-${run.run_id}`)
    expect(back.gbm).toBe(before.active_version)
  }
  expect(errors).toEqual([])
})

test('leaving the Learning tab during a retrain: coming back shows the run, then its result', async ({ page, request }) => {
  test.setTimeout(180_000)
  await seedAnalystLabels(request, 24, 'leave')
  await page.goto('/#/learning')
  const resp = page.waitForResponse((r) => r.url().endsWith('/learning/retrain'), { timeout: 120_000 })
  await page.getByTestId('learning-retrain-button').click()
  await expect(page.getByTestId('learning-retrain-progress')).toBeVisible()
  await page.getByTestId('nav-score').click() // leave while the server is still training
  await expect(page.getByTestId('learning-retrain-progress')).toHaveCount(0)
  await page.getByTestId('nav-learning').click()
  const run = await (await resp).json() // the original request finishes on the server regardless
  // the reopened page finds the run (progress) or its outcome, and ends on the result of that very run
  await expect(page.getByTestId('learning-result')).toHaveAttribute('data-run-id', run.run_id, { timeout: 60_000 })
  await expect(page.getByTestId('learning-retrain-button')).toBeEnabled()
})

test('clicking Retrain while one is already running follows that run instead of erroring', async ({ page, request }) => {
  test.setTimeout(180_000)
  await seedAnalystLabels(request, 24, 'twice')
  const first = request.post('/learning/retrain', { data: { min_new_labels: 20 } }) // e.g. started from another page
  await page.goto('/#/learning')
  await expect(page.getByTestId('learning-retrain-progress')).toBeVisible({ timeout: 15_000 })
  const run = await (await first).json()
  await expect(page.getByTestId('learning-result')).toHaveAttribute('data-run-id', run.run_id, { timeout: 60_000 })
})

test('live view names the model in use', async ({ page, request }) => {
  const st = await (await request.get('/learning/status')).json()
  await page.goto('/#/live')
  await expect(page.getByTestId('live-model-in-use')).toHaveAttribute('data-version', st.model_in_use.version)
})
