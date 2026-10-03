import { expect, test } from '@playwright/test'
import { openDetails, scoreDemoInUi } from './helpers'

test('asking a new question shows a probability or a clear error, never a crash', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  const id = await scoreDemoInUi(page, 'takeover') // idempotent: same decision as spec 04

  await openDetails(page)
  const panel = page.getByTestId('ask-panel')
  await page.getByTestId('ask-example').click()
  await expect(page.getByTestId('ask-input')).not.toHaveValue('')
  const resp = page.waitForResponse((r) => r.url().endsWith(`/decisions/${id}/ask`))
  await page.getByTestId('ask-submit').click()
  const r = await resp

  const result = page.getByTestId('ask-result')
  const alert = panel.getByRole('alert')
  await expect(result.or(alert)).toBeVisible()
  if (r.ok()) {
    const p = Number(await result.getAttribute('data-probability'))
    expect(p).toBeGreaterThanOrEqual(0)
    expect(p).toBeLessThanOrEqual(1)
    expect(await result.getAttribute('data-qid')).toBeTruthy()
  } else {
    // cached/degraded Laya has no answer for a new question: API 503, panel shows its detail
    expect(r.status()).toBe(503)
    const detail = (await r.json()).detail as string
    await expect(alert).toContainText(detail)
    await expect(result).toHaveCount(0)
  }
  // page still alive and usable
  await expect(page.getByTestId('decision-detail')).toBeVisible()
  await expect(page.getByTestId('ask-submit')).toBeEnabled()
  expect(errors).toEqual([])
})
