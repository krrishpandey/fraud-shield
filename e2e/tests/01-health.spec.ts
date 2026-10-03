import { expect, test } from '@playwright/test'

test('status bar shows Laya cached mode, CPU only and the degraded state from /health', async ({ page, request }) => {
  const h = await (await request.get('/health')).json()
  expect(h.laya_mode).toBe('cached')
  expect(h.gpu).toBe(false) // e2e never touches the GPU

  await page.goto('/#/')
  const status = page.getByTestId('health-status')
  await expect(status).toHaveAttribute('data-laya-mode', 'cached')
  await expect(status).toHaveAttribute('data-gpu', 'false')
  await expect(status).toHaveAttribute('data-state', h.degraded ? 'degraded' : 'ok')
  await expect(status).toContainText(h.degraded ? 'Degraded' : 'Online')
  await expect(status).toContainText('Cached')
  await expect(status).toContainText('CPU only')
  await expect(page.getByTestId('mock-banner')).toHaveCount(0) // real API, not fixtures
})
