import { expect, test } from '@playwright/test'
import { STOP_ACTIONS, actionOf, scoreDemoInUi } from './helpers'

type Claim = { text: string; reason_code: string; ok: boolean }
type Claims = {
  mode: string
  claims_source: string
  ok: boolean
  claims: Claim[]
  attribution: { available: boolean; agreement?: { hits: number; of: number; model: string[]; cited: string[] } }
}

// Template path (LLM keys blanked in e2e): every claim is the template's, one per reason, each with a tick.
test('every sentence checked: a stopped decision shows ticked claims and agreement with the model', async ({ page, request }) => {
  const id = await scoreDemoInUi(page, 'takeover')
  expect(STOP_ACTIONS).toContain(await actionOf(page))
  await expect(page.getByTestId('explanation-text')).toBeVisible({ timeout: 30_000 })

  const d = await (await request.get(`/decisions/${id}`)).json()
  const v = (await (await request.get(`/decisions/${id}/claims`)).json()) as Claims
  expect(v.claims_source).toBe('template')
  expect(v.ok).toBe(true)
  expect(v.claims.map((c) => c.reason_code)).toEqual(d.reasons.slice(0, 3))

  const list = page.getByTestId('claims-list')
  await expect(list).toBeVisible()
  const rows = list.getByTestId('claim')
  await expect(rows).toHaveCount(v.claims.length)
  expect(v.claims.length).toBeGreaterThanOrEqual(2)
  for (let i = 0; i < v.claims.length; i++) {
    await expect(rows.nth(i)).toHaveAttribute('data-ok', 'true')
    await expect(rows.nth(i)).toHaveAttribute('data-reason', v.claims[i].reason_code)
    await expect(rows.nth(i)).toContainText(v.claims[i].text)
  }

  // agreement with LightGBM's own attributions (the real B2_F model runs in e2e)
  expect(v.attribution.available).toBe(true)
  const ag = v.attribution.agreement!
  expect(ag.cited).toEqual(v.claims.map((c) => c.reason_code))
  const line = page.getByTestId('claims-agreement')
  await expect(line).toHaveAttribute('data-hits', String(ag.hits))
  await expect(line).toHaveAttribute('data-of', String(ag.of))
  await expect(line).toContainText(`Agrees with the model's top reasons: ${ag.hits} of ${ag.of}`)
})

// Optional (npm run e2e:llm): with a Groq key the explanation is written as strict-JSON claims.
test('claims come from the language model in strict JSON mode @llm', async ({ page, request }) => {
  const h = await (await request.get('/health')).json()
  test.skip(h.components?.explainer === 'template', 'no LLM key available to the server')
  const id = await scoreDemoInUi(page, 'takeover')
  await expect(page.getByTestId('explanation-text')).toBeVisible({ timeout: 45_000 })
  const v = (await (await request.get(`/decisions/${id}/claims`)).json()) as Claims
  expect(['json_schema', 'json_object', 'prose']).toContain(v.mode)
  if (v.claims_source === 'llm') expect(v.claims.every((c) => c.ok)).toBe(true)
  await expect(page.getByTestId('claims')).toHaveAttribute('data-mode', v.mode)
})
