import { expect, test } from '@playwright/test'
import { PASS_ACTIONS, actionOf, openDetails, scoreDemoInUi } from './helpers'

test('real hard negative (first parcel to a new state) is not blocked or held', async ({ page }) => {
  await scoreDemoInUi(page, 'hard-negative-new-state')
  expect(PASS_ACTIONS).toContain(await actionOf(page))
  await openDetails(page)
  await expect(page.getByTestId('prob-misuse')).toBeVisible()
})
