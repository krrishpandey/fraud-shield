import { expect, test } from '@playwright/test'
import { STOP_ACTIONS, actionOf, scoreDemoInUi } from './helpers'

test('analyst sees what would change a stopped decision, and the view is audited', async ({ page, request }) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  const id = await scoreDemoInUi(page, 'takeover')
  expect(STOP_ACTIONS).toContain(await actionOf(page))
  const before = (await (await request.get('/audit/verify')).json()).records as number

  // loaded on click only, marked analyst-only
  const panel = page.getByTestId('counterfactual-panel')
  await expect(panel).toBeVisible()
  await expect(page.getByTestId('counterfactual-warning')).toContainText('never show them to the booker')
  await expect(page.getByTestId('counterfactual-result')).toHaveCount(0)
  await page.getByTestId('counterfactual-load').click()

  // either at least one counterfactual, or an explicit "no change within budget" message
  const result = page.getByTestId('counterfactual-result')
  await expect(result).toBeVisible({ timeout: 30_000 })
  if ((await result.getAttribute('data-found')) === 'true') {
    const items = page.getByTestId('counterfactual-item')
    await expect(items.first()).toBeVisible()
    await expect(items.first()).toContainText(' if ')
  } else {
    await expect(page.getByTestId('counterfactual-none')).toContainText('within 50 evaluations')
  }
  const evals = Number(await page.getByTestId('counterfactual-meta').getAttribute('data-evaluations'))
  expect(evals).toBeGreaterThan(0)
  expect(evals).toBeLessThanOrEqual(50)

  // the API agrees, and every view is written to the audit log (chain still valid)
  const r = await request.get(`/decisions/${id}/counterfactual`)
  expect(r.ok()).toBeTruthy()
  const body = await r.json()
  expect(body.analyst_only).toBe(true)
  expect(body.current.action).toBe(await actionOf(page))
  const audit = await (await request.get('/audit/verify')).json()
  expect(audit.ok).toBe(true)
  expect(audit.records).toBeGreaterThanOrEqual(before + 2) // the UI view and the API view

  // opening another stopped decision must not carry this decision's result over (panel is per decision)
  const demo = await (await request.get('/demo/bookings')).json()
  const drop = demo.find((d: { scenario: string }) => d.scenario === 'reshipping-drop')
  const s = await (await request.post('/score', { data: { ...drop.booking, booking_id: `e2e-cf-other-${drop.booking.booking_id}` } })).json()
  expect(STOP_ACTIONS).toContain(s.action)
  await page.goto(`/#/decisions/${s.decision_id}`)
  await expect(page.getByTestId('decision-detail')).toHaveAttribute('data-decision-id', s.decision_id)
  await expect(page.getByTestId('counterfactual-load')).toBeVisible()
  await expect(page.getByTestId('counterfactual-result')).toHaveCount(0)
  expect(errors).toEqual([])
})
