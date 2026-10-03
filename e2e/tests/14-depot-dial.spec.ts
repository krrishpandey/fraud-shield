import { readFileSync } from 'node:fs'
import { expect, test } from '@playwright/test'

test('the depot weighing dial shows each level and switching it is audited', async ({ page, request }) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  const before = await (await request.get('/first-scan/dial')).json()
  expect(before.current).toBe('standard')
  expect(before.levels.map((l: { name: string }) => l.name)).toEqual(['standard', '3%', '5%', '10%'])

  // an unknown level is refused
  expect((await request.post('/first-scan/dial', { data: { level: '50%' } })).status()).toBe(422)

  try {
    await page.goto('/#/dashboard')
    const dial = page.getByTestId('depot-dial')
    await expect(dial).toBeVisible()
    await expect(page.getByTestId('dial-standard')).toHaveAttribute('data-current', 'true')
    await page.getByTestId('dial-10%').click()
    await expect(page.getByTestId('dial-10%')).toHaveAttribute('data-current', 'true')
    await expect(page.getByTestId('dial-standard')).toHaveAttribute('data-current', 'false')
    await expect(page.getByTestId('dial-current')).toContainText('10%')

    // the server uses the new level, and the change is in the audit log
    expect((await (await request.get('/first-scan/dial')).json()).current).toBe('10%')
    const auditPath = (await (await request.get('/health')).json()).audit_path as string
    const last = readFileSync(auditPath, 'utf-8').split('\n').filter((l) => l.includes('"config_change"')).at(-1)
    expect(last).toContain('first_scan.level')
    expect(last).toContain('10%')
  } finally {
    await request.post('/first-scan/dial', { data: { level: 'standard' } })
  }
  expect(errors).toEqual([])
})
