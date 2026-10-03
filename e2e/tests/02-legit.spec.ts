import { expect, test } from '@playwright/test'
import { actionOf, scoreDemoInUi } from './helpers'

test('legit tenured booking is allowed, with probabilities and latency shown', async ({ page, request }) => {
  const id = await scoreDemoInUi(page, 'legit-tenured')
  expect(await actionOf(page)).toBe('allow')

  const misuse = page.getByTestId('prob-misuse')
  await expect(misuse).toBeVisible()
  const p = Number(await misuse.getAttribute('data-calibrated'))
  expect(p).toBeGreaterThanOrEqual(0)
  expect(p).toBeLessThanOrEqual(1)

  const lat = page.getByTestId('latency-breakdown')
  await expect(lat).toBeVisible()
  expect(Number(await lat.getAttribute('data-total'))).toBeGreaterThan(0)

  // degraded flag in the UI matches the stored decision
  const d = await (await request.get(`/decisions/${id}`)).json()
  await expect(page.getByTestId('degraded-flag')).toHaveCount(d.degraded ? 1 : 0)
  await expect(page.getByTestId('cost-row-allow')).toHaveAttribute('data-chosen', 'true')
})
