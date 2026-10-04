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

  // the attack's evasions are kept for retraining, waiting for the original's truth (blk-06 has no injected label)
  await expect(page.getByTestId('redteam-harvest')).toHaveAttribute('data-labelled', '0')
  await expect(page.getByTestId('redteam-attacks')).toHaveText('1')
  const kept = Number(await page.getByTestId('redteam-evasions').innerText())
  expect(kept).toBeGreaterThan(0)
  await expect(page.getByTestId('redteam-labels')).toHaveText('0')
  await expect(page.getByTestId('redteam-retrain')).toBeDisabled()

  // an analyst confirms the original as fraud: the kept evasions become fraud labels
  await page.getByTestId('redteam-confirm').first().click()
  await expect(page.getByTestId('redteam-labels')).toHaveText(String(kept))
  await expect(page.getByTestId('redteam-retrain')).toBeEnabled()

  // retrain with them through the unchanged gate; the banner says which model scores new bookings from now on
  const resp = page.waitForResponse((r) => r.url().endsWith('/redteam/retrain'), { timeout: 90_000 })
  await page.getByTestId('redteam-retrain').click()
  const run = await (await resp).json()
  expect(run.redteam_labels).toBe(kept)
  const banner = page.getByTestId('redteam-handover')
  await expect(banner).toBeVisible()
  await expect(banner).toHaveAttribute('data-verdict', run.handover.verdict)
  await expect(page.getByTestId('redteam-model-in-use')).toHaveAttribute('data-version', run.deployed_version)
  await expect(page.getByTestId('redteam-retrain')).toBeDisabled() // no new evasion labels since this retrain

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
