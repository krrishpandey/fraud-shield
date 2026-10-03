// MOCK MODE ONLY (VITE_MOCK=1). An in-browser stand-in for the FastAPI backend that follows docs/API.md.
// State lives in memory and resets on reload. All values come from ./fixtures.ts and are invented.
import type {
  AnalystRequest, AskRequest, AskResponse, Booking, DecisionDetail, DecisionSummary, ScoreResponse,
} from '../api/types'
import demo from '../fixtures/demo_bookings.json'
import { dashboard, health, seedHistory, templateFor, templates } from './fixtures'
import * as learning from './learning'

const EXPLANATION_DELAY_MS = 1500
type Stored = DecisionDetail & { _createdAt: number; _scenario: string }
const decisions = new Map<string, Stored>()
const byBooking = new Map<string, string>()
const auditChain: string[] = []
let seq = 100
let customQ = 0
let seedPromise: Promise<void> | null = null

async function sha256(s: string): Promise<string> {
  if (globalThis.crypto?.subtle) {
    const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(s))
    return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('')
  }
  // Fallback for non-secure contexts (FNV-1a stretched to 64 hex chars). Mock only.
  let h = 0x811c9dc5
  let out = ''
  for (let r = 0; r < 8; r++) {
    for (const ch of s + r) h = Math.imul(h ^ ch.charCodeAt(0), 0x01000193) >>> 0
    out += h.toString(16).padStart(8, '0')
  }
  return out
}

async function appendAudit(record: unknown): Promise<string> {
  const prev = auditChain[auditChain.length - 1] ?? '0'.repeat(64)
  const h = await sha256(prev + JSON.stringify(record))
  auditChain.push(h)
  return h
}

async function createDecision(booking: Booking, createdAt: number): Promise<Stored> {
  const existing = byBooking.get(booking.booking_id)
  if (existing) return decisions.get(existing)!
  const key = templateFor(String(booking.meta?.scenario ?? ''), booking)
  const { explanation: _unused, ...rest } = templates[key]
  void _unused
  const decision_id = `dec_${String(++seq).padStart(6, '0')}`
  const base: ScoreResponse = {
    ...structuredClone(rest),
    decision_id,
    booking_id: booking.booking_id,
    explanation_status: 'pending',
    audit_hash: '',
  }
  base.audit_hash = await appendAudit({ kind: 'decision', decision_id, action: base.action })
  const d: Stored = { ...base, booking, explanation: null, analyst: null, _createdAt: createdAt, _scenario: key }
  decisions.set(decision_id, d)
  byBooking.set(booking.booking_id, decision_id)
  return d
}

function ensureSeed(): Promise<void> {
  if (!seedPromise) {
    seedPromise = (async () => {
      const tmpl = (demo as { booking: Booking }[])[0].booking
      for (const h of seedHistory) {
        const sender = h.scenario === 'T1' ? 'snd_x' + h.booking_id.slice(-2) : h.account_id
        await createDecision(
          {
            ...tmpl,
            booking_id: h.booking_id,
            account_id: h.account_id,
            sender_id: sender,
            booked_at: h.booked_at,
            carrier_cost: h.carrier_cost,
            meta: { scenario: h.scenario },
          },
          0,
        )
      }
    })()
  }
  return seedPromise
}

function view(d: Stored): DecisionDetail {
  const ready = Date.now() - d._createdAt >= EXPLANATION_DELAY_MS
  const { _createdAt: _c, _scenario, ...pub } = d
  void _c
  return {
    ...pub,
    explanation_status: ready ? 'ready' : 'pending',
    explanation: ready ? templates[_scenario].explanation : null,
  }
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms))

function mockAsk(d: Stored, body: AskRequest): AskResponse {
  const text = `${body.instructions} ${body.yes}`.toLowerCase()
  let p = 0.12
  if (/drop|reship|forward/.test(text)) p = d._scenario === 'T3' ? 0.82 : 0.07
  else if (/sender|origin|resale|takeover|stolen|misuse/.test(text)) p = d._scenario === 'T1' ? 0.89 : 0.06
  else if (/weight|size|dimension|light/.test(text)) p = d._scenario === 'T6' ? 0.77 : 0.1
  return {
    qid: `custom_${++customQ}`,
    probability_yes: p,
    raw_probabilities: { a: p, b: Math.round((1 - p) * 1000) / 1000 },
    latency_ms: 40.2,
    calibrated: false,
  }
}

export async function mockFetch(path: string, init?: RequestInit): Promise<Response> {
  await ensureSeed()
  await wait(120)
  const url = new URL(path, 'http://mock.local')
  const method = (init?.method ?? 'GET').toUpperCase()
  const body = init?.body ? JSON.parse(String(init.body)) : undefined
  const p = url.pathname

  if (method === 'GET' && p === '/health') return json(health)
  if (method === 'GET' && p === '/demo/bookings') return json(demo)
  if (method === 'POST' && p === '/score') {
    const d = await createDecision(body as Booking, Date.now())
    const { booking: _b, explanation: _e, analyst: _a, ...score } = view(d)
    void _b
    void _e
    void _a
    return json(score)
  }
  if (method === 'GET' && p === '/decisions') {
    const limit = Number(url.searchParams.get('limit') ?? 50)
    const action = url.searchParams.get('action')
    const rows: DecisionSummary[] = [...decisions.values()]
      .filter((d) => !action || d.action === action)
      .sort((a, b) =>
        a._createdAt === b._createdAt ? b.booking.booked_at.localeCompare(a.booking.booked_at) : b._createdAt - a._createdAt,
      )
      .slice(0, limit)
      .map((d) => ({
        decision_id: d.decision_id,
        booking_id: d.booking_id,
        account_id: d.booking.account_id,
        booked_at: d.booking.booked_at,
        action: d.action,
        misuse: d.probabilities.misuse ?? 0,
        carrier_cost: d.booking.carrier_cost,
        analyst_label: d.analyst?.label ?? null,
        scenario: (d.booking.meta?.scenario as string | undefined) ?? null,
      }))
    return json(rows)
  }
  // The live stream replays the real dataset in the Python backend; the mock never invents a stream.
  if (p.startsWith('/stream/'))
    return json({ detail: 'not available in mock mode: run the backend to replay the dataset as a live stream' }, 503)
  // Account history and the explanation validator live in the Python backend only; the mock never invents them.
  if (/^\/decisions\/[^/]+\/(account-story|explanation\/check)$/.test(p))
    return json({ detail: 'not available in mock mode: run the backend to see account history and the live fact check' }, 503)
  const m = p.match(/^\/decisions\/([^/]+)(\/(analyst|ask))?$/)
  if (m) {
    const d = decisions.get(decodeURIComponent(m[1]))
    if (!d) return json({ detail: 'decision not found' }, 404)
    if (method === 'GET' && !m[3]) return json(view(d))
    if (method === 'POST' && m[3] === 'analyst') {
      const req = body as AnalystRequest
      if (req.label !== 'fraud' && req.label !== 'legit') return json({ detail: 'label must be fraud or legit' }, 422)
      d.analyst = { label: req.label, note: req.note ?? '', at: new Date().toISOString() }
      const h = await appendAudit({ kind: 'analyst', decision_id: d.decision_id, ...req })
      learning.recordAnalystLabel(req.label)
      return json({ ok: true, audit_hash: h })
    }
    if (method === 'POST' && m[3] === 'ask') {
      const req = body as AskRequest
      if (!req?.instructions?.trim()) return json({ detail: 'instructions are required' }, 422)
      return json(mockAsk(d, req))
    }
  }
  if (method === 'GET' && p === '/dashboard/metrics') return json(dashboard)
  if (method === 'GET' && p === '/learning/status') return json(learning.learningStatus())
  if (method === 'POST' && p === '/learning/simulate_feedback') {
    const n = Math.floor(Number(body?.n ?? 0))
    if (!(n > 0 && n <= 5000)) return json({ detail: 'n must be between 1 and 5000' }, 422)
    return json(learning.simulateFeedback(n, Number(body?.seed ?? 1)))
  }
  if (method === 'POST' && p === '/learning/retrain') {
    const r = await learning.retrain(Number(body?.min_new_labels ?? 20))
    if (r.status === 200) {
      const res = r.body as { run_id: string; deployed_version: string | null; audit_hash: string }
      res.audit_hash = await appendAudit({ kind: 'retrain', run_id: res.run_id, deployed: res.deployed_version })
    }
    return json(r.body, r.status)
  }
  if (method === 'POST' && p === '/learning/rollback') {
    const r = learning.rollback(String(body?.version ?? ''))
    if (r.status === 200) {
      const res = r.body as { active_version: string; audit_hash: string }
      res.audit_hash = await appendAudit({ kind: 'config_change', rollback_to: res.active_version })
    }
    return json(r.body, r.status)
  }
  if (method === 'GET' && p === '/audit/verify')
    return json({
      ok: true,
      records: auditChain.length,
      head_hash: auditChain[auditChain.length - 1] ?? '',
      first_bad_index: null,
    })
  return json({ detail: `mock: no route for ${method} ${p}` }, 404)
}
