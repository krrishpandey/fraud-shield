import { expect, test } from '@playwright/test'
import { copyFileSync, readFileSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { TMP_DIR, auditCli } from './helpers'

test('audit chain verifies in the console', async ({ page }) => {
  await page.goto('/#/audit')
  await page.getByTestId('audit-verify-button').click()
  const status = page.getByTestId('audit-status')
  await expect(status).toHaveAttribute('data-state', 'ok')
  await expect(status).toHaveAttribute('data-ok', 'true')
  expect(Number((await page.getByTestId('audit-records').innerText()).replace(/[^0-9]/g, ''))).toBeGreaterThan(0)
  await expect(page.getByTestId('audit-head-hash')).toHaveText(/[0-9a-f]{16,}/)
  await expect(page.getByTestId('audit-first-bad')).toHaveCount(0)
})

test('flipping one byte in a copy of the audit log makes the CLI verify fail', async ({ request }) => {
  const live = (await (await request.get('/health')).json()).audit_path as string
  expect(path.resolve(live).startsWith(path.resolve(TMP_DIR))).toBeTruthy() // isolated e2e log

  const good = path.join(TMP_DIR, 'audit-copy-good.jsonl')
  const bad = path.join(TMP_DIR, 'audit-copy-tampered.jsonl')
  copyFileSync(live, good)
  const ok = auditCli(good)
  expect(ok.code).toBe(0)
  expect(ok.out.ok).toBe(true)

  const bytes = readFileSync(good)
  const firstLineEnd = bytes.indexOf(0x0a)
  const i = Math.floor((firstLineEnd > 0 ? firstLineEnd : bytes.length) / 2) // inside record 0
  bytes[i] = bytes[i] ^ 0x01
  writeFileSync(bad, bytes)
  const res = auditCli(bad)
  expect(res.code).toBe(1)
  expect(res.out.ok).toBe(false)
  expect(res.out.first_bad_index).toBe(0)

  // the live file is untouched
  const live2 = await (await request.get('/audit/verify')).json()
  expect(live2.ok).toBe(true)
})
