import { expect, test } from '@playwright/test'
import { scoreDemoInUi } from './helpers'

test('account story: the takeover pays for new senders, the tenured seller only ships its own goods', async ({ page, request }) => {
  const id = await scoreDemoInUi(page, 'takeover')
  const story = await (await request.get(`/decisions/${id}/account-story`)).json()
  const senders = page.getByTestId('story-senders')
  await expect(senders).toBeVisible()
  // the page shows exactly what the API computed, and it matches the model feature
  await expect(senders).toHaveAttribute('data-count', String(story.last10.new_senders))
  expect(story.last10.new_senders).toBeGreaterThanOrEqual(5)
  const d = await (await request.get(`/decisions/${id}`)).json()
  const f = d.top_features.find((x: { name: string }) => x.name === 'new_senders_l10')
  if (f) expect(story.last10.new_senders).toBe(f.value)
  await expect(page.getByTestId('story-insight')).toBeVisible()
  await expect(page.getByTestId('shipping-label')).toHaveAttribute('data-printed', 'false')

  await scoreDemoInUi(page, 'legit-tenured')
  await expect(page.getByTestId('story-senders')).toHaveAttribute('data-count', '0')
  await expect(page.getByTestId('shipping-label')).toHaveAttribute('data-printed', 'true')
})

test('fact check: an invented number in the explanation is rejected by the live validator', async ({ page }) => {
  await scoreDemoInUi(page, 'takeover')
  await expect(page.getByTestId('explanation-text')).toBeVisible({ timeout: 30_000 })
  await page.getByTestId('explanation-try-toggle').click()
  const box = page.getByTestId('explanation-try-input')
  const original = await box.inputValue()
  expect(original).toMatch(/\d/)

  await page.getByTestId('explanation-try-check').click()
  await expect(page.getByTestId('explanation-try-result')).toHaveAttribute('data-ok', 'true')

  await box.fill(original.replace(/\d+(\.\d+)?/, '98765'))
  await page.getByTestId('explanation-try-check').click()
  const result = page.getByTestId('explanation-try-result')
  await expect(result).toHaveAttribute('data-ok', 'false')
  await expect(result.getByTestId('num-bad')).toHaveText('98765')
})
