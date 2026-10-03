import { expect, type APIRequestContext, type Page } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

export const E2E_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
export const REPO_DIR = path.resolve(E2E_DIR, '..')
export const TMP_DIR = path.join(E2E_DIR, '.tmp')

export const STOP_ACTIONS = ['hold', 'review', 'block', 'owner_confirm']
export const PASS_ACTIONS = ['allow', 'allow_scan_gated']
export const ALL_ACTIONS = ['allow', 'allow_scan_gated', 'owner_confirm', 'review', 'hold', 'block']

export type Booking = Record<string, unknown> & { booking_id: string }
export type DemoBooking = { scenario: string; title: string; expected: string | null; booking: Booking }

export async function demoBookings(request: APIRequestContext): Promise<DemoBooking[]> {
  const r = await request.get('/demo/bookings')
  expect(r.ok()).toBeTruthy()
  return (await r.json()) as DemoBooking[]
}

/** Scores a demo booking through the console (select row, click Score) and returns the decision id. */
export async function scoreDemoInUi(page: Page, scenario: string): Promise<string> {
  await page.goto('/#/')
  const row = page.getByTestId(`demo-booking-${scenario}`)
  await expect(row).toBeVisible()
  await row.click()
  await expect(page.getByTestId('selected-booking')).toBeVisible()
  await page.getByTestId('score-button').click()
  await expect(page).toHaveURL(/#\/decisions\/[^/]+$/)
  const detail = page.getByTestId('decision-detail')
  await expect(detail).toBeVisible()
  const id = await detail.getAttribute('data-decision-id')
  expect(id).toBeTruthy()
  return id!
}

export async function actionOf(page: Page): Promise<string> {
  const badge = page.getByTestId('action-badge')
  await expect(badge).toBeVisible()
  const a = await badge.getAttribute('data-action')
  expect(ALL_ACTIONS).toContain(a)
  return a!
}

/** Opens the "How this was decided" section of the decision page (model answers, costs, speed, ask). */
export async function openDetails(page: Page): Promise<void> {
  const t = page.getByTestId('how-toggle')
  if ((await t.getAttribute('aria-expanded')) !== 'true') await t.click()
  await expect(t).toHaveAttribute('aria-expanded', 'true')
}

/** Integer shown in a KPI tile, e.g. "1,234" -> 1234. */
export async function kpiInt(page: Page, name: string): Promise<number> {
  const txt = (await page.getByTestId(`dashboard-kpi-${name}`).locator('.tnum').first().innerText()).trim()
  return Number(txt.replace(/[^0-9]/g, ''))
}

/** Runs the audit CLI on a file. Returns exit code and parsed JSON. Never touches the GPU. */
export function auditCli(file: string): { code: number; out: { ok: boolean; records: number; first_bad_index: number | null } } {
  const env = {
    ...process.env,
    UV_PROJECT_ENVIRONMENT: process.env.UV_PROJECT_ENVIRONMENT ?? 'C:/Users/05nik/.venvs/fraudshield',
    PYTHONPATH: path.join(E2E_DIR, 'pyshim'),
    CUDA_VISIBLE_DEVICES: '',
  }
  let stdout: string
  let code = 0
  try {
    stdout = execFileSync('uv', ['run', '--no-sync', 'python', '-m', 'fraudshield.audit', 'verify', '--path', file], {
      cwd: REPO_DIR, env, encoding: 'utf-8', timeout: 60_000,
    })
  } catch (e) {
    const err = e as { status?: number; stdout?: string }
    code = err.status ?? -1
    stdout = err.stdout ?? ''
  }
  const line = stdout.trim().split(/\r?\n/).pop() ?? '{}'
  return { code, out: JSON.parse(line) }
}
