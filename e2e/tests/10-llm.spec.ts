import { expect, test } from '@playwright/test'
import { scoreDemoInUi } from './helpers'

// Optional: needs GROQ_API_KEY or ANTHROPIC_API_KEY (from .env). Run with `npm run e2e:llm`.
// Excluded from the default run (grepInvert /@llm/ in playwright.config.ts).
test('takeover explanation comes from the LLM and passes the validator @llm', async ({ page, request }) => {
  const h = await (await request.get('/health')).json()
  test.skip(h.components?.explainer === 'template', 'no LLM key available to the server')
  await scoreDemoInUi(page, 'takeover')
  await expect(page.getByTestId('explanation-text')).toBeVisible({ timeout: 45_000 })
  await expect(page.getByTestId('explanation-source')).toHaveAttribute('data-source', 'llm')
  await expect(page.getByTestId('explanation-validator')).toHaveAttribute('data-valid', 'true')
})
