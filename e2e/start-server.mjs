// Starts the FraudShield backend for e2e runs (used by playwright.config.ts webServer).
// - wipes e2e/.tmp so audit log, label store, registry and model versions are fresh per run
// - Laya in cached mode (app.e2e.yaml), torch blocked by pyshim/, CUDA hidden: no GPU use at all
// - LLM keys blanked so explanations come from the deterministic template (E2E_LLM=1 keeps them)
import { spawn } from 'node:child_process'
import { rmSync, mkdirSync, openSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const repo = path.resolve(here, '..')
const tmp = path.join(here, '.tmp')
const port = process.env.E2E_PORT ?? '8080'

rmSync(tmp, { recursive: true, force: true })
mkdirSync(path.join(tmp, 'audit'), { recursive: true })
const log = openSync(path.join(tmp, 'server.log'), 'w') // backend stdout+stderr for debugging

const env = {
  ...process.env,
  UV_PROJECT_ENVIRONMENT: process.env.UV_PROJECT_ENVIRONMENT ?? 'C:/Users/05nik/.venvs/fraudshield',
  FS_CONFIG: path.join(here, 'app.e2e.yaml'),
  PYTHONPATH: [path.join(here, 'pyshim'), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
  CUDA_VISIBLE_DEVICES: '',
  PYTHONUNBUFFERED: '1',
}
if (process.env.E2E_LLM !== '1') {
  // Empty values win over .env (the app's .env loader never overrides existing variables).
  env.GROQ_API_KEY = ''
  env.ANTHROPIC_API_KEY = ''
}

const child = spawn(
  'uv',
  ['run', '--no-sync', 'uvicorn', 'fraudshield.api.app:create_app', '--factory', '--host', '127.0.0.1', '--port', port],
  { cwd: repo, env, stdio: ['ignore', log, log] },
)
const stop = () => child.kill()
process.on('SIGINT', stop)
process.on('SIGTERM', stop)
child.on('exit', (code) => process.exit(code ?? 0))
