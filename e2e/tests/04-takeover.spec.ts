import { expect, test } from '@playwright/test'
import { STOP_ACTIONS, actionOf, openDetails, scoreDemoInUi } from './helpers'

test('label-resale takeover is stopped, cost table marks the action, reasons and explanation shown', async ({ page }) => {
  await scoreDemoInUi(page, 'takeover')
  const action = await actionOf(page)
  expect(STOP_ACTIONS).toContain(action)

  // exactly one chosen cost row, and it is the action on the badge
  await openDetails(page)
  await expect(page.getByTestId('cost-table')).toBeVisible()
  await expect(page.getByTestId(`cost-row-${action}`)).toHaveAttribute('data-chosen', 'true')
  await expect(page.locator('[data-testid^="cost-row-"][data-chosen="true"]')).toHaveCount(1)
  await expect(page.getByTestId('cost-note')).toBeVisible()

  const reasons = page.getByTestId('reason-codes').locator('li[data-code]')
  await expect(reasons.first()).toBeVisible()
  expect(await reasons.count()).toBeGreaterThan(0)

  // explanation arrives asynchronously (template in e2e: LLM keys are blanked)
  const text = page.getByTestId('explanation-text')
  await expect(text).toBeVisible({ timeout: 30_000 })
  await expect(text).not.toBeEmpty()
  await expect(page.getByTestId('explanation-source')).toHaveAttribute('data-source', 'template')
  await expect(page.getByTestId('explanation-validator')).toHaveAttribute('data-valid', /^(true|false)$/)
  await expect(page.getByTestId('explanation-status')).toHaveCount(0)
})
