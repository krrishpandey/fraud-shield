import { defineConfig, devices } from '@playwright/test'

// One backend per run, state shared across specs, so specs run serially in file order (01 -> 09).
const PORT = process.env.E2E_PORT ?? '8090'
const BASE = `http://127.0.0.1:${PORT}`
const LLM = process.env.E2E_LLM === '1'
const CHANNEL = process.env.E2E_CHANNEL // e.g. "msedge" if the bundled Chromium is unavailable

export default defineConfig({
  testDir: './tests',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 60_000,
  expect: { timeout: 15_000 },
  reporter: [['list'], ['html', { open: 'never', outputFolder: '.tmp-report' }]],
  outputDir: '.tmp-results',
  grepInvert: LLM ? undefined : /@llm/,
  use: {
    baseURL: BASE,
    // The console opens on a sign-in screen; tests start already signed in.
    storageState: {
      cookies: [],
      origins: [BASE, `http://localhost:${PORT}`].map((origin) => ({ origin, localStorage: [{ name: 'fs-analyst', value: 'e2e' }] })),
    },
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    ...(CHANNEL ? { channel: CHANNEL } : {}),
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'], ...(CHANNEL ? { channel: CHANNEL } : {}) } }],
  webServer: {
    command: 'node start-server.mjs',
    url: `${BASE}/health`,
    env: { E2E_PORT: PORT, ...(LLM ? { E2E_LLM: '1' } : {}) },
    reuseExistingServer: false, // always a fresh audit log, label store and registry
    timeout: 120_000,
    stdout: 'ignore',
    stderr: 'pipe', // backend output goes to e2e/.tmp/server.log
  },
})
