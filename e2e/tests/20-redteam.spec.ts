import { expect, test } from '@playwright/test'
import { scoreDemoInUi } from './helpers'

test('red team tab: measured results, the gate rejecting the hardened model, and a live attack that gets through', async ({ page, request }) => {
  test.setTimeout(120_000)
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))

  await page.goto('/#/')
  await page.getByTestId('nav-redteam').click()
  await expect(page.getByTestId('redteam-view')).toBeVisible()

  // measured run (artifacts/results_redteam.json)
  await expect(page.getByTestId('redteam-row-all')).toContainText('320')
  await expect(page.getByTestId('redteam-row-all')).toContainText('18.1%')
  await expect(page.getByTestId('redteam-gate')).toHaveAttribute('data-passed', 'false')
  await expect(page.getByTestId('redteam-gate')).toContainText('REJECTED')

  // live: blk-06 is a held booking the attacker gets through (declared weight and value)
  await page.getByTestId('redteam-target').selectOption('t:blk-06')
  await page.getByTestId('redteam-launch').click()
  await expect(page.getByTestId('redteam-attempt').first()).toBeVisible({ timeout: 60_000 })
  await page.getByTestId('redteam-skip').click()
  const verdict = page.getByTestId('redteam-verdict')
  await expect(verdict).toHaveAttribute('data-evaded', 'allow')
  await expect(verdict).toContainText('The attacker got through')
  await expect(page.getByTestId('redteam-evasion')).toContainText('declared')
  const n = await page.getByTestId('redteam-attempt').count()
  expect(n).toBeGreaterThan(1)
  expect(n).toBeLessThanOrEqual(50)
  await expect(page.locator('[data-testid="redteam-attempt"][data-result="allow"]').first()).toBeVisible()

  // the attack is in the audit trail and the chain still verifies
  const audit = await (await request.get('/audit/verify')).json()
  expect(audit.ok).toBe(true)
  expect(errors).toEqual([])
})

test('red team from a decision page: the takeover booking holds against the attacker', async ({ page }) => {
  test.setTimeout(120_000)
  const id = await scoreDemoInUi(page, 'takeover')
  await page.getByTestId('redteam-link').click()
  await expect(page.getByTestId('redteam-target')).toHaveValue(`d:${id}`)
  await page.getByTestId('redteam-launch').click()
  await expect(page.getByTestId('redteam-attempt').first()).toBeVisible({ timeout: 60_000 })
  await page.getByTestId('redteam-skip').click()
  const verdict = page.getByTestId('redteam-verdict')
  await expect(verdict).toHaveAttribute('data-tone', 'held')
  await expect(verdict).toContainText('Our model held')
  await expect(page.getByTestId('redteam-attempt')).toHaveCount(50)
})
